"""Integration tests for Phase 4: Hybrid Retrieval and /query endpoint.

Tests the RetrievalAgent, the 4-arm retrieval fusion (Dense, BM25, Graph, Metadata),
CrossEncoder reranking, section expansion, and trace collection.
"""
import uuid
import datetime
import pytest
from fastapi.testclient import TestClient

from api.main import app
from db.session import get_session, init_db
from models.database import IngestJob
from worker.main import process_job
from db.graph import Graph

client = TestClient(app)



def test_query_endpoint_hybrid_retrieval(populated_db_and_graph):
    """Verifies that RetrievalAgent executes successfully and utilizes all retrieval arms."""
    from agents.retrieval import RetrievalAgent
    db, graph = populated_db_and_graph
    agent = RetrievalAgent(db, graph)
    
    result = agent.retrieve(
        question="Apple Inc. consumer electronics",
        top_k=3
    )
    
    expanded_chunks = result["expanded_chunks"]
    trace = result["retrieval_trace"]
    
    # Verify trace structure and arms
    stages = trace["stages"]
    assert "dense" in stages
    assert "bm25" in stages
    assert "graph" in stages
    assert "fusion" in stages
    assert "reranker" in stages
    
    # Dense and BM25 should find candidates
    assert len(stages["dense"]) > 0
    assert len(stages["bm25"]) > 0
    
    # Graph should find candidates since "apple inc." is in the question
    assert len(stages["graph"]) > 0
    
    # Fusion should merge them
    assert len(stages["fusion"]) >= max(len(stages["dense"]), len(stages["graph"]))
    
    # Reranker should have output scores
    assert len(stages["reranker"]) <= 3  # top_k
    
    # Expanded chunks should contain text from the chunks
    assert len(expanded_chunks) > 0
    for chunk in expanded_chunks:
        assert "text" in chunk
        assert "metadata" in chunk
        assert "document_id" in chunk
        assert "reranker_score" in chunk
        assert isinstance(chunk["constituent_chunk_ids"], list)
        assert len(chunk["constituent_chunk_ids"]) > 0

def test_query_metadata_filters(populated_db_and_graph):
    """Verifies that metadata filters successfully limit the hybrid retrieval."""
    from agents.retrieval import RetrievalAgent
    db, graph = populated_db_and_graph
    agent = RetrievalAgent(db, graph)
    
    # Tesla query without filter
    result1 = agent.retrieve(
        question="What is going on with Tesla?",
        top_k=5
    )
    assert any("tsla" in c["text"].lower() for c in result1["expanded_chunks"])
    
    # Tesla query filtered exclusively to Apple's document type (10-K)
    result2 = agent.retrieve(
        question="What is going on with Tesla?",
        doc_type="10-K",
        top_k=5
    )
    # We should get no Tesla chunks because they are filtered out
    # Actually, we might get Apple chunks because the query embedding might match loosely,
    # but their doc_type must be 10-K.
    for c in result2["expanded_chunks"]:
        assert c["metadata"].get("doc_type") == "10-K"
        assert "tsla" not in c["text"].lower()

def test_section_expansion(populated_db_and_graph):
    """Verifies that parent sections are expanded and merged correctly."""
    from agents.retrieval import RetrievalAgent
    db, graph = populated_db_and_graph
    agent = RetrievalAgent(db, graph)
    
    result = agent.retrieve(
        question="Apple",
        top_k=5
    )
    chunks = result["expanded_chunks"]
    
    # In the mock, the text is split into sentences and grouped by 3.
    # The edgar mock has ~4 sentences, so it might span 2 sections or 1.
    # Just verify that the output structure correctly exposes constituent_chunk_ids
    for chunk in chunks:
        # Number of constituent chunks should be at least 1
        assert len(chunk["constituent_chunk_ids"]) >= 1
        # If it's more than 1, text should be longer
        if len(chunk["constituent_chunk_ids"]) > 1:
            assert len(chunk["text"].split(".")) > 2
