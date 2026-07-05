"""Evaluation runner for Phase 6.

Orchestrates the full evaluation pipeline: load golden set, execute queries,
compute all metrics, persist EvalRecords, and return aggregate summary.
"""

import json
import time
import uuid
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from sqlalchemy.orm import Session

from models.database import QueryRecord, EvalRecord
from db.repositories import EvalRecordRepository
from eval.metrics import context_recall_at_k, citation_correctness, hallucination_rate

logger = logging.getLogger("eval.runner")

GOLDEN_SET_PATH = Path(__file__).parent / "golden.jsonl"

MODEL_VERSIONS = {
    "embedding_model": "FakeEmbeddingModel",
    "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "llm_client": "FakeLLMClient",
}


def load_golden_set(dataset_version: str = "v1") -> List[Dict[str, Any]]:
    """Loads the golden evaluation dataset from eval/golden.jsonl.

    Filters by dataset_version if specified.
    """
    records = []
    with open(GOLDEN_SET_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("dataset_version") == dataset_version:
                records.append(record)
    return records


def execute_query_pipeline(
    question: str,
    db: Session,
    graph: Any,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Executes the full LangGraph pipeline for a single question.

    Returns the final state dict with answer, retrieval_trace, etc.
    """
    from langchain_agents.flow import app as flow_app

    initial_state = {
        "question": question,
        "filters": {},
        "top_k": top_k,
        "db": db,
        "graph": graph,
        "retry_count": 0,
        "agent_trace": [],
    }

    final_state = flow_app.invoke(initial_state)
    return final_state


def run_evaluation(
    db: Session,
    graph: Any,
    dataset_version: str = "v1",
) -> Dict[str, Any]:
    """Runs the full evaluation harness over the golden dataset.

    Pipeline per golden record:
      1. Execute /query pipeline
      2. Collect answer, retrieval_trace, citations, verification_results
      3. Time the execution (wall-clock latency)
      4. Run Ragas metrics (faithfulness, answer_relevancy)
      5. Run custom metrics (context_recall, citation_correctness, hallucination_rate)
      6. Run DeepEval hallucination cross-check
      7. Create QueryRecord and EvalRecord entries

    Returns aggregate summary.
    """
    golden_set = load_golden_set(dataset_version)
    if not golden_set:
        return {"error": "No golden set records found for version: " + dataset_version}

    eval_run_id = uuid.uuid4()
    eval_repo = EvalRecordRepository(db)
    all_eval_records: List[EvalRecord] = []
    per_question_results: List[Dict[str, Any]] = []

    for record in golden_set:
        question = record["question"]
        expected_chunk_ids = record.get("expected_chunk_ids", [])
        expected_answer_components = record.get("expected_answer_components", [])
        golden_id = record["id"]

        logger.info(f"Evaluating golden record {golden_id}: {question}")

        # Time the query execution
        start_time = time.time()
        try:
            final_state = execute_query_pipeline(question, db, graph)
        except Exception as e:
            logger.error(f"Query pipeline failed for {golden_id}: {e}")
            per_question_results.append({
                "golden_id": golden_id,
                "error": str(e),
            })
            continue
        latency_ms = (time.time() - start_time) * 1000

        answer = final_state.get("final_answer", "")
        citations = final_state.get("citations", [])
        expanded_chunks = final_state.get("expanded_chunks", [])
        retrieval_trace = final_state.get("retrieval_trace", {})
        agent_trace = final_state.get("agent_trace", [])
        claims = final_state.get("claims", [])

        # Build verification_results from claims
        verification_results = [
            {
                "claim": c.get("claim"),
                "supported": c.get("supported"),
                "evidence_chunk_ids": [str(cid) for cid in c.get("evidence_chunk_ids", [])],
            }
            for c in claims
        ]

        # Persist QueryRecord
        query_record = QueryRecord(
            question=question,
            answer=answer,
            citations=citations,
            retrieval_trace=retrieval_trace,
            agent_trace=agent_trace,
            verification_results=verification_results,
            retry_count=final_state.get("retry_count", 0),
        )
        db.add(query_record)
        db.commit()
        db.refresh(query_record)

        # Compute custom metrics
        ctx_recall = context_recall_at_k(expanded_chunks, expected_chunk_ids)
        cite_correctness = citation_correctness(answer, citations, expanded_chunks)
        halluc_rate = hallucination_rate(verification_results)

        # Compute Ragas metrics
        contexts = [c.get("text", "") for c in expanded_chunks]
        ground_truth = " ".join(expected_answer_components)
        try:
            from eval.ragas_wrapper import compute_ragas_metrics
            ragas_scores = compute_ragas_metrics(question, answer, contexts, ground_truth)
        except Exception as e:
            logger.warning(f"Ragas metrics failed for {golden_id}: {e}")
            ragas_scores = {"faithfulness": None, "answer_relevancy": None}

        # Compute DeepEval hallucination cross-check
        try:
            from eval.deepeval_wrapper import compute_deepeval_hallucination
            deepeval_halluc = compute_deepeval_hallucination(question, answer, contexts)
        except Exception as e:
            logger.warning(f"DeepEval metric failed for {golden_id}: {e}")
            deepeval_halluc = None

        # Build metrics dict
        metrics = {
            "context_recall": ctx_recall,
            "citation_correctness": cite_correctness,
            "hallucination_rate": halluc_rate,
            "latency_ms": latency_ms,
            "tokens_used": None,  # Phase 7: RunLog instrumentation
            "cost": None,  # Phase 7: RunLog instrumentation
        }

        # Add Ragas scores
        metrics["faithfulness"] = ragas_scores.get("faithfulness")
        metrics["answer_relevancy"] = ragas_scores.get("answer_relevancy")

        # Add DeepEval score
        metrics["deepeval_hallucination"] = deepeval_halluc

        # Get the reason from Ragas or DeepEval wrapper if None
        error_reason_ragas = ragas_scores.get("error_reason")
        
        # Create EvalRecords for each metric
        for metric_name, score in metrics.items():
            reason = None
            if score is None:
                if metric_name in ["faithfulness", "answer_relevancy"]:
                    reason = error_reason_ragas
                elif metric_name == "deepeval_hallucination":
                    reason = "requires real LLM — not computed under FakeLLMClient"
            
            eval_record = EvalRecord(
                eval_run_id=eval_run_id,
                query_record_id=query_record.id,
                metric=metric_name,
                score=score,
                error_reason=reason,
                dataset_version=dataset_version,
                model_versions=MODEL_VERSIONS,
            )
            all_eval_records.append(eval_record)

        per_question_results.append({
            "golden_id": golden_id,
            "question": question,
            "answer": answer,
            "query_record_id": str(query_record.id),
            "metrics": {k: v for k, v in metrics.items() if v is not None},
        })

    # Persist all EvalRecords in batch
    if all_eval_records:
        eval_repo.add_batch(all_eval_records)

    # Compute aggregates
    aggregates = {}
    for metric_name in ["context_recall", "citation_correctness", "hallucination_rate",
                        "latency_ms", "faithfulness", "answer_relevancy", "deepeval_hallucination"]:
        scores = [r.score for r in all_eval_records if r.metric == metric_name and r.score is not None]
        if scores:
            aggregates[metric_name] = {
                "avg": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "count": len(scores),
            }

    return {
        "eval_run_id": str(eval_run_id),
        "dataset_version": dataset_version,
        "golden_set_size": len(golden_set),
        "questions_evaluated": len(per_question_results),
        "aggregates": aggregates,
        "per_question": per_question_results,
    }
