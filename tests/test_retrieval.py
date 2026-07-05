"""End-to-end retrieval tests for Phase 2.

Tests dense retrieval, BM25 retrieval, metadata-filtered retrieval,
and reciprocal rank fusion over deterministic fixture documents.

Requires a running PostgreSQL instance with pgvector extension
(provided by docker-compose).
"""
import uuid
import datetime
import pytest
from sqlalchemy import func
from sqlalchemy.orm import Session
from db.session import get_session, init_db
from models.database import IngestJob, Document, Chunk
from db.repositories import ChunkRepository
from retrieval.fusion import reciprocal_rank_fusion
from ingest.interfaces import FakeEmbeddingModel


# Shared fixture data
FIXTURE_DOCS = [
    {
        "source": "edgar",
        "external_id": f"retrieval-test-doc-001-{uuid.uuid4().hex[:8]}",
        "title": "Apple Inc 10-K 2023",
        "company": "Apple Inc.",
        "doc_type": "10-K",
        "published_at": datetime.datetime(2023, 10, 31, 16, 0, 0),
        "chunks": [
            "Apple Inc designs manufactures and markets consumer electronics software and services.",
            "Apple Inc reported strong revenue growth driven by iPhone and services segments.",
            "Tim Cook serves as Chief Executive Officer of Apple Inc since August 2011.",
        ]
    },
    {
        "source": "news",
        "external_id": f"retrieval-test-doc-002-{uuid.uuid4().hex[:8]}",
        "title": "Tesla Regulatory Review",
        "company": "Tesla Inc.",
        "doc_type": "news",
        "published_at": datetime.datetime(2023, 11, 15, 9, 30, 0),
        "chunks": [
            "Tesla Inc is under regulatory scrutiny by the SEC for disclosure practices.",
            "Elon Musk serves as CEO of Tesla and has faced regulatory challenges.",
        ]
    },
]


@pytest.fixture()
def retrieval_db():
    """Provides a clean database session with fixture data seeded, cleaned up after test."""
    init_db()
    db = next(get_session())
    embedding_model = FakeEmbeddingModel()
    doc_ids = []
    chunk_ids = []

    for doc_data in FIXTURE_DOCS:
        doc = Document(
            id=uuid.uuid4(),
            source=doc_data["source"],
            external_id=doc_data["external_id"],
            title=doc_data["title"],
            company=doc_data["company"],
            doc_type=doc_data["doc_type"],
            published_at=doc_data["published_at"],
            status="Parsed",
        )
        db.add(doc)
        db.flush()
        doc_ids.append(doc.id)

        for idx, text in enumerate(doc_data["chunks"]):
            chunk = Chunk(
                id=uuid.uuid4(),
                document_id=doc.id,
                ordinal=idx,
                text=text,
                token_count=len(text.split()),
                page=f"Section {idx + 1}",
                embedding=embedding_model.embed_text(text),
                chunk_metadata={
                    "company": doc_data["company"],
                    "doc_type": doc_data["doc_type"],
                    "date": doc_data["published_at"].isoformat(),
                },
            )
            db.add(chunk)
            chunk_ids.append(chunk.id)

    db.commit()

    # Populate tsv columns for full-text search
    for cid in chunk_ids:
        db.execute(
            Chunk.__table__.update()
            .where(Chunk.__table__.c.id == cid)
            .values(tsv=func.to_tsvector("english", Chunk.__table__.c.text))
        )
    db.commit()

    yield db, chunk_ids, doc_ids

    # Cleanup only our fixture data
    for cid in chunk_ids:
        db.query(Chunk).filter(Chunk.id == cid).delete()
    for did in doc_ids:
        db.query(Document).filter(Document.id == did).delete()
    db.commit()
    db.close()


def test_embeddings_stored(retrieval_db):
    """Verifies that fixture chunks have 384-dim embeddings stored."""
    db, chunk_ids, _ = retrieval_db
    for cid in chunk_ids:
        chunk = db.query(Chunk).filter(Chunk.id == cid).one()
        assert chunk.embedding is not None
        assert len(chunk.embedding) == 384
    assert len(chunk_ids) == 5


def test_dense_retrieval(retrieval_db):
    """Verifies dense retrieval returns expected chunks ordered by cosine similarity."""
    db, chunk_ids, _ = retrieval_db
    repo = ChunkRepository(db)
    embedding_model = FakeEmbeddingModel()

    query_embedding = embedding_model.embed_text("Apple consumer electronics revenue")
    results = repo.search(query_embedding, top_k=5)

    assert len(results) > 0
    for chunk, score in results:
        assert isinstance(chunk, Chunk)
        assert isinstance(score, float)
    # Similarity scores should be between 0 and ~1 for cosine
    for _, score in results:
        assert -0.01 <= score <= 1.01


def test_bm25_retrieval(retrieval_db):
    """Verifies BM25 full-text retrieval returns matching chunks."""
    db, chunk_ids, _ = retrieval_db
    repo = ChunkRepository(db)

    # BM25 search for "Apple revenue"
    results = repo.search_bm25("Apple revenue", top_k=5)
    assert len(results) > 0
    for chunk, rank in results:
        assert isinstance(chunk, Chunk)
        assert isinstance(rank, float)
        assert rank > 0

    # BM25 search for "Tesla regulatory" should find Tesla chunks
    tesla_results = repo.search_bm25("Tesla regulatory", top_k=5)
    assert len(tesla_results) > 0
    for chunk, _ in tesla_results:
        assert "Tesla" in chunk.text or "regulatory" in chunk.text.lower()


def test_metadata_filter_company(retrieval_db):
    """Verifies metadata filters restrict results to matching company."""
    db, chunk_ids, _ = retrieval_db
    repo = ChunkRepository(db)
    embedding_model = FakeEmbeddingModel()
    query_embedding = embedding_model.embed_text("general finance query")

    # Dense search filtered to Apple
    apple_results = repo.search(query_embedding, top_k=10, company="Apple Inc.")
    assert len(apple_results) > 0
    for chunk, _ in apple_results:
        assert chunk.chunk_metadata["company"] == "Apple Inc."

    # Dense search filtered to Tesla
    tesla_results = repo.search(query_embedding, top_k=10, company="Tesla Inc.")
    assert len(tesla_results) > 0
    for chunk, _ in tesla_results:
        assert chunk.chunk_metadata["company"] == "Tesla Inc."


def test_metadata_filter_doc_type(retrieval_db):
    """Verifies doc_type metadata filter restricts results."""
    db, chunk_ids, _ = retrieval_db
    repo = ChunkRepository(db)
    embedding_model = FakeEmbeddingModel()
    query_embedding = embedding_model.embed_text("general query")

    # Filter to 10-K only — should get Apple's 3 chunks
    results_10k = repo.search(query_embedding, top_k=10, doc_type="10-K")
    for chunk, _ in results_10k:
        assert chunk.chunk_metadata["doc_type"] == "10-K"
    apple_fixture_count = sum(1 for cid in chunk_ids
                               for d in FIXTURE_DOCS if d["doc_type"] == "10-K"
                               for _ in d["chunks"])
    # At minimum our fixtures are present
    assert len(results_10k) >= 3

    # Filter to news only — should get Tesla's 2 chunks
    results_news = repo.search(query_embedding, top_k=10, doc_type="news")
    for chunk, _ in results_news:
        assert chunk.chunk_metadata["doc_type"] == "news"
    assert len(results_news) >= 2


def test_reciprocal_rank_fusion(retrieval_db):
    """Verifies RRF correctly combines dense and BM25 result lists."""
    db, chunk_ids, _ = retrieval_db
    repo = ChunkRepository(db)
    embedding_model = FakeEmbeddingModel()

    query_embedding = embedding_model.embed_text("Apple revenue growth")
    dense_results = repo.search(query_embedding, top_k=5)
    bm25_results = repo.search_bm25("Apple revenue growth", top_k=5)

    # Run RRF
    fused = reciprocal_rank_fusion([dense_results, bm25_results], k=60)

    # Basic structural checks
    assert len(fused) > 0
    for chunk, score in fused:
        assert isinstance(chunk, Chunk)
        assert score > 0

    # Scores should be in descending order
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)

    # Items appearing in both lists should have higher RRF scores
    dense_ids = {c.id for c, _ in dense_results}
    bm25_ids = {c.id for c, _ in bm25_results}
    overlap_ids = dense_ids & bm25_ids
    if overlap_ids:
        fused_dict = {c.id: s for c, s in fused}
        only_one_ids = (dense_ids | bm25_ids) - overlap_ids
        if only_one_ids:
            max_single = max(fused_dict[iid] for iid in only_one_ids)
            min_overlap = min(fused_dict[iid] for iid in overlap_ids)
            assert min_overlap >= max_single
