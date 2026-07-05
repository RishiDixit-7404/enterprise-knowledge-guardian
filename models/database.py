import uuid
import datetime
from sqlalchemy import Column, String, DateTime, Integer, Float, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB, TSVECTOR
from sqlalchemy.orm import relationship
from db.session import Base
from pgvector.sqlalchemy import Vector

class IngestJob(Base):
    __tablename__ = "ingest_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source = Column(String, nullable=False)  # 'edgar' or 'news'
    params = Column(JSONB, nullable=True)     # jsonb: tickers, date range
    status = Column(String, nullable=False, default="Queued")
    
    # Counts
    documents_count = Column(Integer, default=0)
    chunks_count = Column(Integer, default=0)
    entities_count = Column(Integer, default=0)
    relationships_count = Column(Integer, default=0)
    
    error = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source = Column(String, nullable=False)  # 'edgar' or 'news'
    external_id = Column(String, nullable=False, unique=True)
    title = Column(String, nullable=True)
    company = Column(String, nullable=True)  # Name + CIK/ticker
    doc_type = Column(String, nullable=True)  # '10-K', '10-Q', '8-K', 'news'
    published_at = Column(DateTime, nullable=True)
    url = Column(String, nullable=True)
    raw_path = Column(String, nullable=True)  # Path to local binary/text blob
    ingested_at = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String, nullable=True)

class Chunk(Base):
    __tablename__ = "chunks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    ordinal = Column(Integer, nullable=False)
    text = Column(String, nullable=False)
    token_count = Column(Integer, nullable=False)
    page = Column(String, nullable=True)     # Page or section header identifier
    
    # Search fields
    tsv = Column(TSVECTOR, nullable=True)     # Postgres full-text vector
    embedding = Column(Vector(384), nullable=True)  # pgvector column (dimension 384)
    chunk_metadata = Column("metadata", JSONB, nullable=True)  # jsonb: company, doc_type, date, section
    
    document = relationship("Document", back_populates="chunks")

Document.chunks = relationship("Chunk", order_by=Chunk.ordinal, back_populates="document", cascade="all, delete-orphan")

class DBEntity(Base):
    __tablename__ = "entities"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    type = Column(String, nullable=False)     # Company|Person|Regulator|Filing|Risk|Topic|Location
    name = Column(String, nullable=False)
    normalized_name = Column(String, nullable=False, unique=True)
    attributes = Column(JSONB, nullable=True)
    source_chunk_ids = Column(JSONB, nullable=True)  # List of chunk UUIDs

class DBRelationship(Base):
    __tablename__ = "relationships"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    type = Column(String, nullable=False)     # MENTIONS|FILED_BY|SUBSIDIARY_OF etc.
    from_id = Column(UUID(as_uuid=True), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False)
    to_id = Column(UUID(as_uuid=True), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False)
    evidence_chunk_ids = Column(JSONB, nullable=False) # Must carry at least one evidence chunk
    confidence = Column(Float, default=1.0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    from_entity = relationship("DBEntity", foreign_keys=[from_id])
    to_entity = relationship("DBEntity", foreign_keys=[to_id])

class QueryRecord(Base):
    __tablename__ = "query_records"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question = Column(String, nullable=False)
    answer = Column(String, nullable=False)
    citations = Column(JSONB, nullable=True)  # List of chunk and doc ids
    retrieval_trace = Column(JSONB, nullable=True)
    agent_trace = Column(JSONB, nullable=True)
    verification_results = Column(JSONB, nullable=True)
    retry_count = Column(Integer, default=0)
    run_id = Column(UUID(as_uuid=True), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class RunLog(Base):
    __tablename__ = "run_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), nullable=False)
    operation = Column(String, nullable=False) # entity_extraction|answer|verify|eval
    prompt_version = Column(String, nullable=True)
    model_name = Column(String, nullable=True)
    embedding_model = Column(String, nullable=True)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    latency_ms = Column(Integer, default=0)
    cost_estimate = Column(Float, default=0.0)
    retrieved_chunk_ids = Column(JSONB, nullable=True)
    tool_calls = Column(JSONB, nullable=True)
    eval_score = Column(Float, nullable=True)
    failure_reason = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class EvalRecord(Base):
    __tablename__ = "eval_records"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    eval_run_id = Column(UUID(as_uuid=True), nullable=False)  # Groups records from the same eval run
    query_record_id = Column(UUID(as_uuid=True), ForeignKey("query_records.id", ondelete="CASCADE"), nullable=False)
    metric = Column(String, nullable=False)   # faithfulness|answer_relevancy|etc.
    score = Column(Float, nullable=True)      # nullable: tokens_used/cost are null in Phase 6
    error_reason = Column(String, nullable=True)
    dataset_version = Column(String, nullable=True)
    model_versions = Column(JSONB, nullable=True)  # {embedding_model, reranker, llm_client}
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    query_record = relationship("QueryRecord")
