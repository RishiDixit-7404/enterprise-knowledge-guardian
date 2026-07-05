import asyncio
import logging
import uuid
import datetime
from typing import Optional
from sqlalchemy import select, func
from sqlalchemy.orm import Session
from db.session import SessionLocal, init_db
from models.database import IngestJob, Document, Chunk
from db.graph import Graph
from ingest.interfaces import FakeFilingSource, FakeNewsSource, FakeLLMClient, get_embedding_model
from settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("worker")

# Initialize external sources (mock-first)
fake_filing_source = FakeFilingSource()
fake_news_source = FakeNewsSource()
embedding_model = get_embedding_model()
fake_llm_client = FakeLLMClient()

def normalize_entity_name(name: str) -> str:
    """Helper to consistently normalize entity names."""
    return name.strip().lower()

def claim_next_job(session: Session) -> Optional[IngestJob]:
    """Atomically claims the next Queued job in the database queue using SKIP LOCKED."""
    # Find next queued job
    subq = (
        select(IngestJob.id)
        .where(IngestJob.status == "Queued")
        .order_by(IngestJob.created_at.asc())
        .limit(1)
        .with_for_update(skip_locked=True)
    ).scalar_subquery()
    
    job = session.execute(
        select(IngestJob).where(IngestJob.id == subq)
    ).scalar_one_or_none()
    
    if job:
        job.status = "Fetching"
        job.updated_at = datetime.datetime.utcnow()
        session.commit()
        session.refresh(job)
        return job
    return None

def process_job(job: IngestJob, session: Session):
    """Executes the ingestion pipeline steps sequentially."""
    try:
        logger.info(f"Processing job {job.id} (source: {job.source})...")
        
        # 1. Fetch
        logger.info(f"Job {job.id}: Fetching documents...")
        tickers = job.params.get("tickers") if job.params else None
        ticker = tickers[0] if tickers else "AAPL"
        
        if job.source == "edgar":
            raw_docs = fake_filing_source.fetch_filings(ticker, None, None)
        else:
            raw_docs = fake_news_source.fetch_news(ticker, None, None)
            
        # 2. Parse
        logger.info(f"Job {job.id}: Parsing documents...")
        job.status = "Parsing"
        session.commit()
        
        parsed_docs = []
        for rd in raw_docs:
            parsed_docs.append({
                "external_id": rd["external_id"],
                "title": rd["title"],
                "company": rd["company"],
                "doc_type": rd["doc_type"],
                "published_at": rd["published_at"],
                "url": rd["url"],
                "raw_path": rd["raw_path"],
                "text": rd["text_content"]
            })
            
        # Save documents to database
        db_docs = []
        for pd in parsed_docs:
            existing_doc = session.execute(
                select(Document).where(Document.external_id == pd["external_id"])
            ).scalar_one_or_none()
            
            if existing_doc:
                doc = existing_doc
                doc.title = pd["title"]
                doc.company = pd["company"]
                doc.doc_type = pd["doc_type"]
                doc.published_at = pd["published_at"]
                doc.url = pd["url"]
                doc.raw_path = pd["raw_path"]
                doc.status = "Parsed"
            else:
                doc_id = uuid.uuid5(uuid.NAMESPACE_DNS, pd["external_id"])
                doc = Document(
                    id=doc_id,
                    source=job.source,
                    external_id=pd["external_id"],
                    title=pd["title"],
                    company=pd["company"],
                    doc_type=pd["doc_type"],
                    published_at=pd["published_at"],
                    url=pd["url"],
                    raw_path=pd["raw_path"],
                    status="Parsed"
                )
                session.add(doc)
            db_docs.append(doc)
        session.commit()
        for doc in db_docs:
            session.refresh(doc)
            
        # 3. Chunking
        logger.info(f"Job {job.id}: Chunking documents...")
        job.status = "Chunking"
        session.commit()
        
        db_chunks = []
        for doc_idx, doc in enumerate(db_docs):
            raw_text = parsed_docs[doc_idx]["text"]
            # Simple sentence chunking
            sentences = [s.strip() for s in raw_text.split(".") if s.strip()]
            chunk_size = 3
            chunk_groups = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]
            
            for idx, group in enumerate(chunk_groups):
                chunk_text = ". ".join(group) + "."
                chunk_id = uuid.uuid5(doc.id, str(idx))
                chunk = Chunk(
                    id=chunk_id,
                    document_id=doc.id,
                    ordinal=idx,
                    text=chunk_text,
                    token_count=len(chunk_text.split()),
                    page=f"Section {idx+1}",
                    chunk_metadata={
                        "company": doc.company,
                        "doc_type": doc.doc_type,
                        "date": doc.published_at.isoformat() if doc.published_at else None,
                        "section": f"Section {idx+1}"
                    }
                )
                session.add(chunk)
                db_chunks.append(chunk)
        session.commit()
        for chunk in db_chunks:
            session.refresh(chunk)
            
        # 4. Embedding
        logger.info(f"Job {job.id}: Embedding chunks...")
        job.status = "Embedding"
        session.commit()
        
        for chunk in db_chunks:
            chunk.embedding = embedding_model.embed_text(chunk.text)
            # Populate tsv for full-text search
            session.execute(
                Chunk.__table__.update()
                .where(Chunk.id == chunk.id)
                .values(tsv=func.to_tsvector('english', chunk.text))
            )
        session.commit()
        
        # 5. Extracting Graph
        logger.info(f"Job {job.id}: Extracting Knowledge Graph...")
        job.status = "Extracting Graph"
        session.commit()
        
        graph = Graph()
        written_rels_count = 0
        all_entities = {}
        
        try:
            all_relationships = []
            for chunk in db_chunks:
                # 1. Extract entities
                extracted_entities = fake_llm_client.extract_entities(chunk.text)
                
                # Normalise and accumulate entities
                for ent in extracted_entities:
                    name = ent["name"]
                    etype = ent["type"]
                    attrs = ent.get("attributes", {})
                    norm_name = normalize_entity_name(name)
                    
                    if norm_name not in all_entities:
                        all_entities[norm_name] = {
                            "id": uuid.uuid4(),
                            "type": etype,
                            "name": name,
                            "normalized_name": norm_name,
                            "attributes": attrs,
                            "source_chunk_ids": [chunk.id]
                        }
                    else:
                        if chunk.id not in all_entities[norm_name]["source_chunk_ids"]:
                            all_entities[norm_name]["source_chunk_ids"].append(chunk.id)
                
                # 2. Extract relationships based on extracted entities
                extracted_rels = fake_llm_client.extract_relationships(chunk.text, extracted_entities)
                
                # Accumulate relationships (only if they have evidence)
                for rel in extracted_rels:
                    all_relationships.append({
                        "type": rel["type"],
                        "from_name": normalize_entity_name(rel["from"]),
                        "to_name": normalize_entity_name(rel["to"]),
                        "evidence_chunk_ids": [chunk.id],
                        "confidence": rel.get("confidence", 1.0)
                    })
            
            # Store entities to Neo4j
            for norm_name, ent_data in all_entities.items():
                graph.write_entity(
                    entity_id=ent_data["id"],
                    entity_type=ent_data["type"],
                    name=ent_data["name"],
                    normalized_name=norm_name,
                    attributes=ent_data["attributes"],
                    source_chunk_ids=ent_data["source_chunk_ids"]
                )
            
            # Store relationships to Neo4j
            for rel in all_relationships:
                from_ent = all_entities.get(rel["from_name"])
                to_ent = all_entities.get(rel["to_name"])
                if from_ent and to_ent:
                    graph.write_relationship(
                        rel_id=uuid.uuid4(),
                        rel_type=rel["type"],
                        from_id=from_ent["id"],
                        to_id=to_ent["id"],
                        evidence_chunk_ids=rel["evidence_chunk_ids"],
                        confidence=rel["confidence"]
                    )
                    written_rels_count += 1
        finally:
            graph.close()
            
        # 6. Indexed
        logger.info(f"Job {job.id}: Indexing documents...")
        job.status = "Indexed"
        session.commit()
        
        # 7. Completed
        logger.info(f"Job {job.id}: Completing job...")
        job.status = "Completed"
        job.documents_count = len(db_docs)
        job.chunks_count = len(db_chunks)
        job.entities_count = len(all_entities)
        job.relationships_count = written_rels_count
        session.commit()
        logger.info(f"Job {job.id} completed successfully.")
        
    except Exception as e:
        logger.exception(f"Error processing job {job.id}: {e}")
        job.status = "Failed"
        job.error = str(e)
        session.commit()

async def poll_loop():
    """Continuously polls the table-backed job queue."""
    logger.info("Worker starting poll loop...")
    while True:
        try:
            with SessionLocal() as session:
                job = claim_next_job(session)
                if job:
                    # Run the pipeline synchronously for this job
                    process_job(job, session)
                    continue
            
            logger.info("polling")
            await asyncio.sleep(settings.WORKER_POLL_INTERVAL)
        except Exception as e:
            logger.error(f"Error in poll loop: {e}")
            await asyncio.sleep(settings.WORKER_POLL_INTERVAL)

def main():
    logger.info("Initializing database...")
    init_db()
    try:
        asyncio.run(poll_loop())
    except KeyboardInterrupt:
        logger.info("Worker stopped via KeyboardInterrupt.")

if __name__ == "__main__":
    main()
