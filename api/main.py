from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Any, Optional, List
import uuid
import datetime
from sqlalchemy.orm import Session
from db.session import get_session, init_db
from models.database import IngestJob
from settings import settings

bearer_scheme = HTTPBearer(auto_error=False)

def verify_token(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)):
    if not settings.API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized - API Key unconfigured")
    if not credentials or credentials.credentials != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing authentication token")
    return credentials

app = FastAPI(
    title="Enterprise Knowledge Guardian (EKG) API",
    description="Compliance & risk intelligence platform over public company data.",
    version="0.1.0",
    dependencies=[Depends(verify_token)]
)

# Initialize database schema on API startup
@app.on_event("startup")
def startup_event():
    init_db()

# Standard Response Envelope per SPEC.md
class ResponseEnvelope(BaseModel):
    success: bool = True
    status_code: int = 200
    error_message: str = ""
    data: Optional[Any] = None

class IngestRequest(BaseModel):
    source: str = Field(..., description="Source of data to ingest, e.g., 'edgar' or 'news'")
    tickers: Optional[List[str]] = Field(default=None, description="Tickers or search queries")
    date_from: Optional[datetime.date] = Field(default=None, description="Start date for filtering")
    date_to: Optional[datetime.date] = Field(default=None, description="End date for filtering")

@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirect root to Swagger UI /docs."""
    return RedirectResponse(url="/docs")

@app.get("/health", response_model=ResponseEnvelope)
async def health_check():
    """Health check endpoint returning 200 and standard envelope status."""
    return ResponseEnvelope(
        success=True,
        status_code=200,
        error_message="",
        data={"status": "ok"}
    )

@app.post("/ingest", response_model=ResponseEnvelope)
def queue_ingest(req: IngestRequest, db: Session = Depends(get_session)):
    """Creates a new IngestJob and writes it to the database with status 'Queued'."""
    if req.source not in ("edgar", "news"):
        raise HTTPException(status_code=400, detail="Invalid source. Must be 'edgar' or 'news'.")
    
    # Store request parameters in parameters dict
    params = {
        "tickers": req.tickers,
        "date_from": req.date_from.isoformat() if req.date_from else None,
        "date_to": req.date_to.isoformat() if req.date_to else None,
    }
    
    job = IngestJob(
        source=req.source,
        params=params,
        status="Queued"
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    return ResponseEnvelope(
        success=True,
        status_code=200,
        data={"job_id": str(job.id)}
    )

@app.get("/jobs/{id}", response_model=ResponseEnvelope)
def get_job(id: uuid.UUID, db: Session = Depends(get_session)):
    """Retrieves an IngestJob status and extraction counts by ID."""
    job = db.query(IngestJob).filter(IngestJob.id == id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    return ResponseEnvelope(
        success=True,
        status_code=200,
        data={
            "id": str(job.id),
            "source": job.source,
            "params": job.params,
            "status": job.status,
            "counts": {
                "documents": job.documents_count,
                "chunks": job.chunks_count,
                "entities": job.entities_count,
                "relationships": job.relationships_count
            },
            "error": job.error,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "updated_at": job.updated_at.isoformat() if job.updated_at else None
        }
    )

@app.get("/graph/{entity_id}", response_model=ResponseEnvelope)
def get_entity_graph(entity_id: uuid.UUID):
    """Retrieves an entity, its neighbors, relationships, and evidence chunks."""
    from db.graph import Graph
    
    graph = Graph()
    try:
        data = graph.get_entity_graph(entity_id)
        if not data:
            raise HTTPException(status_code=404, detail="Entity not found")
            
        return ResponseEnvelope(
            success=True,
            status_code=200,
            data=data
        )
    finally:
        graph.close()

class QueryRequest(BaseModel):
    question: str = Field(..., description="The query string to search for.")
    top_k: int = Field(default=5, description="Number of results to return.")
    company: Optional[str] = Field(default=None, description="Filter by company name.")
    doc_type: Optional[str] = Field(default=None, description="Filter by document type.")
    date_from: Optional[datetime.date] = Field(default=None, description="Filter from date.")
    date_to: Optional[datetime.date] = Field(default=None, description="Filter to date.")

def get_graph():
    from db.graph import Graph
    graph = Graph()
    try:
        yield graph
    finally:
        graph.close()

@app.get("/entities", response_model=ResponseEnvelope)
def get_entities(name: Optional[str] = None, type: Optional[str] = None):
    """Query nodes in the knowledge graph matching optional name and type filters."""
    from db.graph import Graph
    graph = Graph()
    query = "MATCH (n)"
    conditions = []
    params = {}
    if name:
        conditions.append("n.name = $name")
        params["name"] = name
    if type:
        conditions.append("n.type = $type")
        params["type"] = type
        
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " RETURN n LIMIT 100"
    
    with graph.driver.session() as session:
        result = session.run(query, params)
        nodes = [dict(record["n"]) for record in result]
            
    return ResponseEnvelope(
        success=True,
        data={"nodes": nodes}
    )

@app.post("/query", response_model=ResponseEnvelope)
def execute_query(req: QueryRequest, db: Session = Depends(get_session)):
    """Executes the Agentic RAG pipeline to answer a user question."""
    from db.graph import Graph
    from models.database import QueryRecord
    
    graph = Graph()
    # Prepare initial state for LangGraph
    initial_state = {
        "question": req.question,
        "filters": {
            "company": req.company,
            "doc_type": req.doc_type,
            "date_from": req.date_from,
            "date_to": req.date_to
        },
        "top_k": req.top_k,
        "db": db,
        "graph": graph,
        "retry_count": 0,
        "agent_trace": []
    }
    
    import time
    from settings import settings
    
    start_time = time.time()
    # Execute the LangGraph workflow
    from langchain_agents.flow import app as flow_app
    final_state = flow_app.invoke(initial_state)
    latency_ms = int((time.time() - start_time) * 1000)
    
    # Construct verification results from the claims
    verification_results = [
        {"claim": c.get("claim"), "supported": c.get("supported"), "evidence_chunk_ids": [str(cid) for cid in c.get("evidence_chunk_ids", [])]}
        for c in final_state.get("claims", [])
    ]
    
    # Store complete execution trace in QueryRecord
    query_record = QueryRecord(
        question=req.question,
        answer=final_state["final_answer"],
        citations=final_state["citations"],
        retrieval_trace=final_state.get("retrieval_trace", {}),
        agent_trace=final_state.get("agent_trace", []),
        verification_results=verification_results,
        retry_count=final_state.get("retry_count", 0)
    )
    db.add(query_record)
    db.commit()
    db.refresh(query_record)
    
    # Extract total usage from agent trace
    total_input_tokens = 0
    total_output_tokens = 0
    prompt_sources = []
    agent_executions = []
    for event in final_state.get("agent_trace", []):
        agent_executions.append(event)
        if "usage" in event:
            u = event["usage"]
            total_input_tokens += u.get("input_tokens", 0)
            total_output_tokens += u.get("output_tokens", 0)
            prompt_sources.append(u.get("prompt_source", "unknown"))
            
    # Cost tracking requires provider-specific rate sheets. Null in mock mode.
    cost_estimate = None
        
    from models.database import RunLog
    run_log = RunLog(
        run_id=query_record.id,
        operation="answer",
        prompt_version="v1",
        prompt_source=",".join(set(prompt_sources)),
        model_name=settings.LLM_MODEL,
        embedding_model=settings.EMBEDDING_MODEL,
        reranker_model=settings.RERANKER_MODEL,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        latency_ms=latency_ms,
        cost_estimate=cost_estimate,
        retrieval_trace=final_state.get("retrieval_trace", {}),
        retrieved_chunk_ids=[c.get("chunk_id") for c in final_state.get("expanded_chunks", [])],
        agent_execution=agent_executions,
        request_ids=None # Not populated for mock
    )
    db.add(run_log)
    db.commit()
    
    if settings.MLFLOW_ENABLED:
        from llmops.mlflow_logger import log_run
        log_run(run_log)
    
    return ResponseEnvelope(data={
        "query_record_id": str(query_record.id),
        "answer": final_state["final_answer"],
        "citations": final_state["citations"],
        "retrieval_trace": final_state.get("retrieval_trace", {}),
        "agent_trace": final_state.get("agent_trace", []),
        "retry_count": final_state.get("retry_count", 0),
        "expanded_chunks": final_state.get("expanded_chunks", [])
    })


class EvalRunRequest(BaseModel):
    dataset_version: str = Field(default="v1", description="Version of the golden evaluation dataset to use.")


@app.post("/eval/run", response_model=ResponseEnvelope)
def run_evaluation_endpoint(req: EvalRunRequest, db: Session = Depends(get_session), graph: Any = Depends(get_graph)):
    """Runs the evaluation harness over the golden dataset and returns aggregate scores."""
    from eval.runner import run_evaluation

    result = run_evaluation(db, graph, dataset_version=req.dataset_version)

    return ResponseEnvelope(data=result)


@app.get("/metrics", response_model=ResponseEnvelope)
def get_metrics(dataset_version: Optional[str] = None, db: Session = Depends(get_session)):
    """Returns evaluation metrics, recent runs, and operational RunLog statistics."""
    from db.repositories import EvalRecordRepository, RunLogRepository
    
    eval_repo = EvalRecordRepository(db)
    aggregated = eval_repo.get_aggregated_metrics(dataset_version=dataset_version)
    recent_runs = eval_repo.get_recent_eval_runs(limit=10)
    
    runlog_repo = RunLogRepository(db)
    runlog_stats = runlog_repo.get_aggregated_stats()
    
    return ResponseEnvelope(
        success=True,
        status_code=200,
        data={
            "aggregated": aggregated,
            "recent_runs": recent_runs,
            "runlog_stats": runlog_stats
        }
    )
