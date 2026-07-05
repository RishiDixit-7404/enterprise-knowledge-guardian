"""Integration tests for Phase 6: Evaluation Harness.

Tests golden dataset loading, /eval/run execution, all metrics computation,
EvalRecord persistence, and /metrics aggregation.
All tests use deterministic fixture data with no internet dependency.
"""

import json
import uuid
import datetime
import os
import pytest
from fastapi.testclient import TestClient

from api.main import app
from db.session import get_session, init_db
from models.database import EvalRecord, QueryRecord

# Disable DeepEval telemetry in tests
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"



def test_golden_dataset_loads():
    """Verifies that the golden evaluation dataset loads correctly."""
    from eval.runner import load_golden_set

    records = load_golden_set("v1")
    assert len(records) > 0, "Golden set must have at least one record"

    for record in records:
        assert "id" in record
        assert "question" in record
        assert "expected_answer_components" in record
        assert "expected_chunk_ids" in record
        assert "dataset_version" in record
        assert record["dataset_version"] == "v1"


def test_context_recall_at_k():
    """Verifies Context Recall@K computation with known data."""
    from eval.metrics import context_recall_at_k

    chunks = [
        {"id": "chunk-1", "text": "Apple Inc. designs consumer electronics."},
        {"id": "chunk-2", "text": "Tim Cook is the CEO."},
    ]

    # All keywords found
    score = context_recall_at_k(chunks, ["chunk-1", "chunk-2"])
    assert score == 1.0

    # Only one keyword found
    score = context_recall_at_k(chunks, ["chunk-1", "chunk-3"])
    assert score == 0.5

    # No keywords found
    score = context_recall_at_k(chunks, ["chunk-4"])
    assert score == 0.0

    # Empty expected keywords
    score = context_recall_at_k(chunks, [])
    assert score == 1.0


def test_citation_correctness():
    """Verifies Citation Correctness computation."""
    from eval.metrics import citation_correctness

    expanded_chunks = [
        {"text": "Apple designs electronics.", "constituent_chunk_ids": ["id-1", "id-2"]},
        {"text": "Tesla under review.", "constituent_chunk_ids": ["id-3"]},
    ]

    # All citations valid
    score = citation_correctness("answer", ["id-1", "id-3"], expanded_chunks)
    assert score == 1.0

    # One invalid citation
    score = citation_correctness("answer", ["id-1", "id-999"], expanded_chunks)
    assert score == 0.5

    # No citations
    score = citation_correctness("answer", [], expanded_chunks)
    assert score == 1.0


def test_hallucination_rate():
    """Verifies Hallucination Rate computation: 1 - (supported / total)."""
    from eval.metrics import hallucination_rate

    # All supported
    results = [{"supported": True}, {"supported": True}]
    assert hallucination_rate(results) == 0.0

    # Half unsupported
    results = [{"supported": True}, {"supported": False}]
    assert hallucination_rate(results) == 0.5

    # All unsupported
    results = [{"supported": False}, {"supported": False}]
    assert hallucination_rate(results) == 1.0

    # Empty
    assert hallucination_rate([]) == 0.0


def test_ragas_wrapper_executes():
    """Verifies that the Ragas wrapper returns unavailable error in mock mode."""
    from eval.ragas_wrapper import compute_ragas_metrics

    scores = compute_ragas_metrics(
        question="What does Apple design?",
        answer="Apple designs consumer electronics.",
        contexts=["Apple Inc. designs consumer electronics."],
        ground_truth="consumer electronics",
    )

    assert isinstance(scores, dict)
    assert scores["faithfulness"] is None
    assert scores["answer_relevancy"] is None
    assert "error_reason" in scores
    assert "not computed under FakeLLMClient" in scores["error_reason"]


def test_deepeval_wrapper_executes():
    """Verifies that the DeepEval wrapper returns unavailable error in mock mode."""
    from eval.deepeval_wrapper import compute_deepeval_hallucination

    score = compute_deepeval_hallucination(
        question="What does Apple design?",
        answer="Apple designs consumer electronics.",
        contexts=["Apple Inc. designs consumer electronics."],
    )

    assert score is None


def test_eval_run_endpoint(populated_db_and_graph, client):
    """Verifies that POST /eval/run executes the evaluation pipeline end-to-end."""
    response = client.post("/eval/run", json={"dataset_version": "v1"})
    assert response.status_code == 200

    data = response.json()["data"]
    assert "eval_run_id" in data
    assert data["dataset_version"] == "v1"
    assert data["golden_set_size"] > 0
    assert data["questions_evaluated"] > 0

    # Verify aggregates contain expected metrics
    aggregates = data["aggregates"]
    assert "context_recall" in aggregates
    assert "citation_correctness" in aggregates
    assert "hallucination_rate" in aggregates
    assert "latency_ms" in aggregates

    # tokens_used and cost should NOT have aggregates (they're null in Phase 6)
    assert "tokens_used" not in aggregates
    assert "cost" not in aggregates

    # Verify per-question results
    per_question = data["per_question"]
    assert len(per_question) > 0
    for pq in per_question:
        assert "golden_id" in pq
        if "error" not in pq:
            assert "metrics" in pq
            assert "context_recall" in pq["metrics"]
            assert "citation_correctness" in pq["metrics"]
            assert "hallucination_rate" in pq["metrics"]
            assert "latency_ms" in pq["metrics"]


def test_eval_records_persisted(populated_db_and_graph, client):
    """Verifies that EvalRecords are persisted in the database after /eval/run."""
    response = client.post("/eval/run", json={"dataset_version": "v1"})
    assert response.status_code == 200

    eval_run_id = response.json()["data"]["eval_run_id"]

    db = next(get_session())
    records = db.query(EvalRecord).filter(
        EvalRecord.eval_run_id == uuid.UUID(eval_run_id)
    ).all()

    assert len(records) > 0

    # Check that each record has the right structure
    for record in records:
        assert record.eval_run_id == uuid.UUID(eval_run_id)
        assert record.query_record_id is not None
        assert record.metric is not None
        assert record.dataset_version == "v1"
        assert record.model_versions is not None
        assert "embedding_model" in record.model_versions
        
        if record.score is None:
            assert record.error_reason is not None
            assert "requires real LLM" in record.error_reason


def test_metrics_endpoint(populated_db_and_graph, client):
    """Verifies that GET /metrics aggregates evaluation results correctly."""
    # First run an evaluation to generate data
    client.post("/eval/run", json={"dataset_version": "v1"})

    # Now query metrics
    response = client.get("/metrics")
    assert response.status_code == 200

    data = response.json()["data"]
    assert "aggregates" in data
    assert "recent_runs" in data

    # Verify aggregates contain expected metrics
    aggregates = data["aggregates"]
    assert len(aggregates) > 0

    # Each aggregate should have avg, count, min, max
    for metric_name, stats in aggregates.items():
        assert "avg" in stats
        assert "count" in stats
        assert "min" in stats
        assert "max" in stats

    # Verify recent runs exist
    recent_runs = data["recent_runs"]
    assert len(recent_runs) > 0
    for run in recent_runs:
        assert "eval_run_id" in run
        assert "metrics" in run
