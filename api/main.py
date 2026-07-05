from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from typing import Any, Optional, List
import uuid
import datetime
from sqlalchemy.orm import Session
from db.session import get_session, init_db
from models.database import IngestJob

app = FastAPI(
    title="Enterprise Knowledge Guardian (EKG) API",
    description="Compliance & risk intelligence platform over public company data.",
    version="0.1.0"
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

@app.post("/query", response_model=ResponseEnvelope)
def execute_query(req: QueryRequest, db: Session = Depends(get_session), graph: Any = Depends(get_graph)):
    """Executes the full Retrieval + LangGraph Verification + Response pipeline."""
    from langchain_agents.flow import app as flow_app
    from models.database import QueryRecord
    
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
    
    # Execute the LangGraph workflow
    final_state = flow_app.invoke(initial_state)
    
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
def get_metrics(db: Session = Depends(get_session)):
    """Aggregates recent EvalRecords and returns evaluation history."""
    from db.repositories import EvalRecordRepository

    eval_repo = EvalRecordRepository(db)
    aggregates = eval_repo.get_aggregated_metrics()
    recent_runs = eval_repo.get_recent_eval_runs(limit=10)

    return ResponseEnvelope(data={
        "aggregates": aggregates,
        "recent_runs": recent_runs,
    })

