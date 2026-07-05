from neo4j import GraphDatabase
from settings import settings
import uuid
from typing import List, Dict, Any

class Graph:
    def __init__(self, uri: str = None, user: str = None, password: str = None):
        """Initializes the Neo4j driver using the configured settings."""
        self.uri = uri or settings.NEO4J_URI
        self.user = user or settings.NEO4J_USER
        self.password = password or settings.NEO4J_PASSWORD
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        """Closes the driver connection."""
        self.driver.close()

    def write_entity(self, entity_id: uuid.UUID, entity_type: str, name: str, normalized_name: str, attributes: Dict[str, Any], source_chunk_ids: List[uuid.UUID]):
        """Creates or updates an Entity node in Neo4j (deduped by normalized_name)."""
        import json
        query = """
        MERGE (e:Entity {normalized_name: $normalized_name})
        ON CREATE SET e.id = $id, e.type = $type, e.name = $name, e.attributes = $attributes, e.source_chunk_ids = $source_chunk_ids
        ON MATCH SET e.type = $type, e.name = $name, e.attributes = $attributes, e.source_chunk_ids = $source_chunk_ids
        RETURN e
        """
        chunk_ids_str = [str(cid) for cid in source_chunk_ids]
        with self.driver.session() as session:
            session.run(
                query,
                id=str(entity_id),
                type=entity_type,
                name=name,
                normalized_name=normalized_name,
                attributes=json.dumps(attributes or {}),
                source_chunk_ids=chunk_ids_str
            ).consume()

    def write_relationship(self, rel_id: uuid.UUID, rel_type: str, from_id: uuid.UUID, to_id: uuid.UUID, evidence_chunk_ids: List[uuid.UUID], confidence: float):
        """Creates or updates a Relationship edge in Neo4j between two Entities."""
        # Enforce SPEC.md evidence-chunk requirement: "Every edge must carry at least one evidence chunk — no unsupported edges."
        if not evidence_chunk_ids:
            raise ValueError("Relationship must carry at least one evidence chunk ID")

        # Sanitize rel_type to be alphanumeric + underscore for Cypher syntax safety
        safe_rel_type = "".join([c for c in rel_type if c.isalnum() or c == "_"])
        if not safe_rel_type or not safe_rel_type[0].isalpha():
            raise ValueError(
                f"Invalid relationship type '{rel_type}': after sanitization "
                f"('{safe_rel_type}') it must be non-empty and start with a letter"
            )
        
        query = f"""
        MATCH (from:Entity {{id: $from_id}})
        MATCH (to:Entity {{id: $to_id}})
        MERGE (from)-[r:{safe_rel_type}]->(to)
        ON CREATE SET r.id = $id, r.evidence_chunk_ids = $evidence_chunk_ids, r.confidence = $confidence, r.created_at = timestamp()
        ON MATCH SET r.evidence_chunk_ids = $evidence_chunk_ids, r.confidence = $confidence
        RETURN r
        """
        evidence_ids_str = [str(ecid) for ecid in evidence_chunk_ids]
        with self.driver.session() as session:
            session.run(
                query,
                id=str(rel_id),
                from_id=str(from_id),
                to_id=str(to_id),
                evidence_chunk_ids=evidence_ids_str,
                confidence=confidence
            ).consume()

    def get_entity_graph(self, entity_id: uuid.UUID) -> Dict[str, Any]:
        """Traverses from an Entity, returning the entity, its neighbors, relationships, and evidence chunks."""
        query = """
        MATCH (e:Entity {id: $entity_id})
        OPTIONAL MATCH (e)-[r]-(neighbor:Entity)
        RETURN e {.*} as entity,
               collect(DISTINCT r {.*, type: type(r), start_id: startNode(r).id, end_id: endNode(r).id}) as relationships,
               collect(DISTINCT neighbor {.*}) as neighbors
        """
        with self.driver.session() as session:
            result = session.run(query, entity_id=str(entity_id)).single()
            if not result or not result.get("entity"):
                return None
            return {
                "entity": result["entity"],
                "neighbors": result["neighbors"] if result.get("neighbors") else [],
                "relationships": result["relationships"] if result.get("relationships") else []
            }

    def get_evidence_chunks_for_text(self, text: str) -> List[uuid.UUID]:
        """Finds entities whose normalized_name is a substring of the text,
        then traverses 1-hop relationships to return all associated evidence chunk IDs."""
        query = """
        MATCH (e:Entity)
        WHERE toLower($text) CONTAINS toLower(e.normalized_name)
        MATCH (e)-[r]-(neighbor:Entity)
        RETURN DISTINCT r.evidence_chunk_ids as chunk_ids
        """
        chunk_uuids = set()
        with self.driver.session() as session:
            result = session.run(query, text=text)
            for record in result:
                for cid_str in record["chunk_ids"]:
                    try:
                        chunk_uuids.add(uuid.UUID(cid_str))
                    except ValueError:
                        pass
        return list(chunk_uuids)
