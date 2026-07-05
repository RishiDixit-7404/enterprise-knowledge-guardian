import uuid
from sqlalchemy.orm import Session
from db.session import get_session, init_db
from models.database import IngestJob, Document, Chunk
from worker.main import process_job
from unittest.mock import MagicMock, patch

def test_ingestion_flow(client):
    """
    Integration test for the ingestion pipeline:
    1. POST /ingest creates an IngestJob in the database.
    2. Worker process_job is called (with patched Neo4j Graph).
    3. Verify Document and Chunks are written to PostgreSQL.
    4. Verify Entity and Relationship extraction counts and state.
    """
    # Initialize database tables
    init_db()
    
    # Obtain a session for verification
    db = next(get_session())
    
    # Cleanup state from previous runs
    db.query(Chunk).delete()
    db.query(Document).delete()
    db.query(IngestJob).delete()
    db.commit()
    
    # 1. Trigger POST /ingest to queue a job
    response = client.post("/ingest", json={"source": "edgar", "tickers": ["AAPL"]})
    assert response.status_code == 200
    
    body = response.json()
    assert body["success"] is True
    job_id_str = body["data"]["job_id"]
    job_id = uuid.UUID(job_id_str)
    
    # Check that job is successfully queued in PostgreSQL
    job = db.query(IngestJob).filter(IngestJob.id == job_id).first()
    assert job is not None
    assert job.status == "Queued"
    
    # 2. Run the ingestion process with Graph mocked to keep the test offline
    mock_graph = MagicMock()
    with patch("worker.main.Graph", return_value=mock_graph):
        process_job(job, db)
        
    # Refresh job from database to check outcome
    db.refresh(job)
    assert job.status == "Completed"
    assert job.error is None
    assert job.documents_count == 1
    assert job.chunks_count > 0
    assert job.entities_count > 0
    assert job.relationships_count > 0
    
    # 3. Assert PostgreSQL Document records
    doc = db.query(Document).filter(Document.source == "edgar").first()
    assert doc is not None
    assert doc.company == "Apple Inc. / AAPL"
    assert doc.doc_type == "10-K"
    assert doc.status == "Parsed"
    
    # 4. Assert PostgreSQL Chunk records and check pgvector dimensions
    chunks = db.query(Chunk).filter(Chunk.document_id == doc.id).all()
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.embedding is not None
        assert len(chunk.embedding) == 384
        assert chunk.chunk_metadata["company"] == doc.company
        
    # 5. Assert Graph node/edge write calls were triggered
    assert mock_graph.write_entity.called
    assert mock_graph.write_relationship.called
    
    # Inspect arguments on write_entity to ensure correctness
    call_args = mock_graph.write_entity.call_args[1]
    assert "entity_id" in call_args
    assert "entity_type" in call_args
    assert "name" in call_args
    assert "normalized_name" in call_args
    
    db.close()
