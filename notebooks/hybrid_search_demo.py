#!/usr/bin/env python3
"""
Hybrid Search Demo — Phase 2 Exploratory Notebook

Demonstrates dense retrieval, BM25 retrieval, metadata-filtered search,
and reciprocal rank fusion over ingested fixture data.

Usage:
    1. Ensure the Docker stack is running: docker compose up -d
    2. Ingest data:  curl -X POST http://localhost:8000/ingest \
                       -H "Content-Type: application/json" \
                       -d '{"source": "edgar", "tickers": ["AAPL"]}'
    3. Wait for completion, then run:
       python notebooks/hybrid_search_demo.py
"""
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db.session import get_session, init_db
from db.repositories import ChunkRepository
from ingest.interfaces import FakeEmbeddingModel
from retrieval.fusion import reciprocal_rank_fusion
from models.database import Chunk


def print_results(title: str, results):
    """Pretty-prints search results."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    if not results:
        print("  (no results)")
        return
    for i, (chunk, score) in enumerate(results, 1):
        text_preview = chunk.text[:80] + "..." if len(chunk.text) > 80 else chunk.text
        company = chunk.chunk_metadata.get("company", "N/A") if chunk.chunk_metadata else "N/A"
        doc_type = chunk.chunk_metadata.get("doc_type", "N/A") if chunk.chunk_metadata else "N/A"
        print(f"  [{i}] score={score:.6f}  company={company}  doc_type={doc_type}")
        print(f"      {text_preview}")
    print()


def main():
    init_db()
    db = next(get_session())
    repo = ChunkRepository(db)
    embedding_model = FakeEmbeddingModel()

    # Check that we have chunks
    total_chunks = db.query(Chunk).count()
    print(f"\nTotal chunks in database: {total_chunks}")
    if total_chunks == 0:
        print("No chunks found. Please ingest data first (see Usage above).")
        db.close()
        return

    query = "Apple revenue growth consumer electronics"
    print(f"\nQuery: \"{query}\"")

    # -------------------------------------------------------------------
    # 1. Dense Retrieval
    # -------------------------------------------------------------------
    query_embedding = embedding_model.embed_text(query)
    dense_results = repo.search(query_embedding, top_k=5)
    print_results("Dense Retrieval (pgvector cosine distance)", dense_results)

    # -------------------------------------------------------------------
    # 2. BM25 Retrieval (PostgreSQL full-text search)
    # -------------------------------------------------------------------
    bm25_results = repo.search_bm25(query, top_k=5)
    print_results("BM25 Retrieval (PostgreSQL ts_rank)", bm25_results)

    # -------------------------------------------------------------------
    # 3. Reciprocal Rank Fusion (hybrid)
    # -------------------------------------------------------------------
    fused_results = reciprocal_rank_fusion([dense_results, bm25_results], k=60)
    print_results("Hybrid (RRF) — Dense + BM25", fused_results)

    # -------------------------------------------------------------------
    # 4. Metadata Filters — Company
    # -------------------------------------------------------------------
    dense_apple = repo.search(query_embedding, top_k=5, company="Apple Inc. / AAPL")
    print_results("Dense Retrieval — Filtered: company='Apple Inc. / AAPL'", dense_apple)

    # -------------------------------------------------------------------
    # 5. Metadata Filters — Doc Type
    # -------------------------------------------------------------------
    bm25_10k = repo.search_bm25(query, top_k=5, doc_type="10-K")
    print_results("BM25 Retrieval — Filtered: doc_type='10-K'", bm25_10k)

    # -------------------------------------------------------------------
    # 6. Hybrid with metadata filter
    # -------------------------------------------------------------------
    dense_filtered = repo.search(query_embedding, top_k=5, doc_type="10-K")
    bm25_filtered = repo.search_bm25(query, top_k=5, doc_type="10-K")
    fused_filtered = reciprocal_rank_fusion([dense_filtered, bm25_filtered], k=60)
    print_results("Hybrid (RRF) — Filtered: doc_type='10-K'", fused_filtered)

    print("Demo complete.\n")
    db.close()


if __name__ == "__main__":
    main()
