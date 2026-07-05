"""End-to-end integration tests for Phase 3 Knowledge Graph.

Tests the deterministic fake extraction, Neo4j persistence, Entity normalization,
and the `/graph/{entity_id}` endpoint using local offline defaults.
"""
import uuid
import datetime
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from api.main import app
from db.session import get_session, init_db
from models.database import IngestJob
from worker.main import process_job
from db.graph import Graph
from ingest.interfaces import FakeLLMClient


@pytest.fixture(scope="module")
def setup_db_and_graph():
    """Initializes DB schemas for tests."""
    init_db()

@pytest.fixture
def clean_graph():
    """Cleans up the Neo4j graph before and after the test."""
    graph = Graph()
    try:
        with graph.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        yield graph
    finally:
        with graph.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        graph.close()

def test_fake_llm_client_extracts_deterministically():
    """Verifies that FakeLLMClient extract_entities and extract_relationships return deterministic output."""
    llm = FakeLLMClient()
    text = "Apple Inc. designs iPhone and Tim Cook is the CEO."
    
    # Extract entities
    entities = llm.extract_entities(text)
    assert len(entities) == 3
    entity_names = [e["name"] for e in entities]
    assert "Apple Inc." in entity_names
    assert "Tim Cook" in entity_names
    assert "SEC" in entity_names
    
    # Extract relationships
    rels = llm.extract_relationships(text, entities)
    assert len(rels) >= 2
    rel_types = [r["type"] for r in rels]
    assert "HAS_CEO" in rel_types
    assert "REGULATED_BY" in rel_types

def test_worker_ingest_to_graph(setup_db_and_graph, clean_graph):
    """End-to-end test verifying full ingestion cycle populates the graph correctly."""
    db = next(get_session())
    graph = clean_graph
    
    # Create a job using 'edgar' mock source
    job = IngestJob(
        id=uuid.uuid4(),
        source="edgar",
        params={"tickers": ["AAPL"]},
        status="Queued"
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    # Run pipeline synchronously
    process_job(job, db)
    
    # Job should be Completed
    assert job.status == "Completed"
    assert job.entities_count > 0
    assert job.relationships_count > 0
    
    # Query Neo4j directly to verify
    with graph.driver.session() as session:
        # Check entities
        result = session.run("MATCH (e:Entity) RETURN count(e) as count, collect(e.normalized_name) as names").single()
        assert result["count"] > 0
        names = result["names"]
        assert "apple inc." in names
        
        # Check relationships
        rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()
        assert rel_result["count"] > 0
        
        # Verify deduplication (apple inc. should only have 1 node)
        apple_count = session.run("MATCH (e:Entity {normalized_name: 'apple inc.'}) RETURN count(e) as c").single()["c"]
        assert apple_count == 1
        
        # Verify provenance on relationships
        rel_prov = session.run("MATCH ()-[r:HAS_CEO]->() RETURN r.evidence_chunk_ids as chunks").single()
        assert rel_prov is not None
        assert len(rel_prov["chunks"]) > 0

def test_api_graph_endpoint(setup_db_and_graph, clean_graph, client):
    """Verifies that the GET /graph/{entity_id} endpoint returns the correct structure."""
    db = next(get_session())
    graph = clean_graph
    
    # Inject a known entity and relationship directly into Neo4j
    apple_id = uuid.uuid4()
    cook_id = uuid.uuid4()
    chunk_id = uuid.uuid4()
    
    graph.write_entity(apple_id, "Company", "Apple Inc.", "apple inc.", {"ticker": "AAPL"}, [chunk_id])
    graph.write_entity(cook_id, "Person", "Tim Cook", "tim cook", {"role": "CEO"}, [chunk_id])
    graph.write_relationship(uuid.uuid4(), "HAS_CEO", apple_id, cook_id, [chunk_id], 0.9)
    
    # Hit the endpoint
    response = client.get(f"/graph/{apple_id}")
    assert response.status_code == 200
    
    data = response.json()["data"]
    assert data["entity"]["name"] == "Apple Inc."
    
    neighbors = data["neighbors"]
    assert len(neighbors) == 1
    assert neighbors[0]["name"] == "Tim Cook"
    
    rels = data["relationships"]
    assert len(rels) == 1
    assert rels[0]["type"] == "HAS_CEO"
    assert str(chunk_id) in rels[0]["evidence_chunk_ids"]

def test_api_graph_endpoint_not_found(clean_graph, client):
    """Verifies 404 response for unknown entity."""
    response = client.get(f"/graph/{uuid.uuid4()}")
    assert response.status_code == 404
