import pytest
from fastapi.testclient import TestClient
from api.main import app
from db.session import get_session
from models.database import RunLog


def test_e2e_pipeline_generates_runlog(populated_db_and_graph, monkeypatch, client):
    db_session, graph_db = populated_db_and_graph
    
    # Enable MLflow explicitly for this test
    from settings import settings
    monkeypatch.setattr(settings, "MLFLOW_ENABLED", True)
    
    # Overwrite get_session and get_graph
    app.dependency_overrides[get_session] = lambda: db_session
    from api.main import get_graph
    app.dependency_overrides[get_graph] = lambda: graph_db
    
    # 1. Execute /query
    response = client.post("/query", json={
        "question": "Who is the CEO of Apple Inc.?",
        "top_k": 5
    })
    assert response.status_code == 200
    data = response.json()["data"]
    assert "query_record_id" in data
    
    # 2. Verify RunLog created for query
    run_logs = db_session.query(RunLog).filter(RunLog.operation == "answer").all()
    assert len(run_logs) >= 1
    runlog = run_logs[-1]
    
    # Verify exact phase 7 instrumentation
    assert runlog.prompt_version == "v1"
    assert "prompts/response.py" in runlog.prompt_source or "prompts/verification.py" in runlog.prompt_source
    assert runlog.model_name == "fake"
    assert runlog.embedding_model == "fake"
    
    # Latency and Tokens must be > 0 because we wrapped it in time.time() and tiktoken
    assert runlog.latency_ms > 0
    assert runlog.input_tokens > 0
    assert runlog.output_tokens > 0
    
    # Cost is null for mock
    assert runlog.cost_estimate is None
    
    assert runlog.retrieval_trace is not None
    assert len(runlog.agent_execution) > 0
    
    # 3. Execute /eval/run
    response = client.post("/eval/run?dataset_version=v1")
    assert response.status_code == 200
    eval_data = response.json()["data"]
    
    # 4. Verify RunLog created for eval
    eval_run_logs = db_session.query(RunLog).filter(RunLog.operation == "eval").all()
    assert len(eval_run_logs) > 0
    
    # Check that EvalRecords correctly copied the latency/tokens
    # We can fetch /metrics
    response = client.get("/metrics")
    assert response.status_code == 200
    metrics_data = response.json()["data"]
    
    assert "runlog_stats" in metrics_data
    runlog_stats = metrics_data["runlog_stats"]
    assert runlog_stats["total_runs"] > 0
    assert runlog_stats["avg_latency_ms"] > 0
    assert runlog_stats["avg_tokens"] > 0
    
    app.dependency_overrides.clear()

    # 5. Verify MLflow tracking explicitly
    from settings import settings
    if settings.MLFLOW_ENABLED:
        import mlflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        experiment = mlflow.get_experiment_by_name("EKG_Pipeline")
        assert experiment is not None, "MLflow experiment 'EKG_Pipeline' was not created."
        
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        assert not runs.empty, "No runs were logged to MLflow."
        
        # Verify the most recent run has expected params
        latest_run = runs.iloc[0]
        assert latest_run["params.operation"] in ("answer", "eval")
        assert latest_run["params.model_name"] == "fake"
        assert latest_run["metrics.input_tokens"] > 0
        assert latest_run["metrics.latency_ms"] > 0
