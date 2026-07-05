import mlflow
from typing import Any
import os

def log_run(run_log: Any):
    """
    Logs a RunLog entry to a local MLflow SQLite tracking server.
    Only active if MLFLOW_ENABLED=true.
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("EKG_Pipeline")
    
    with mlflow.start_run():
        mlflow.log_param("run_id", str(run_log.run_id))
        mlflow.log_param("operation", run_log.operation)
        mlflow.log_param("prompt_version", run_log.prompt_version)
        mlflow.log_param("model_name", run_log.model_name)
        mlflow.log_param("embedding_model", run_log.embedding_model)
        mlflow.log_param("reranker_model", run_log.reranker_model)
        
        mlflow.log_metric("input_tokens", run_log.input_tokens or 0)
        mlflow.log_metric("output_tokens", run_log.output_tokens or 0)
        mlflow.log_metric("latency_ms", run_log.latency_ms or 0)
        if run_log.cost_estimate is not None:
            mlflow.log_metric("cost_estimate", run_log.cost_estimate)
