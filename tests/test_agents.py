import pytest
from fastapi.testclient import TestClient
from api.main import app
from db.session import get_session
from models.database import QueryRecord
from tests.test_query import setup_db_and_graph, populated_db_and_graph

client = TestClient(app)

def test_query_endpoint_langgraph_flow(populated_db_and_graph):
    """
    Tests the end-to-end /query endpoint powered by LangGraph.
    Verifies that the VerificationAgent flags unsupported claims, triggers the retry loop exactly once,
    and ResponseAgent generates an answer with valid citations.
    """
    # Use a query that triggers both a supported claim and an unsupported one in FakeLLMClient.
    # The word "apple" triggers an unsupported claim about "electric cars", causing 1 retry.
    response = client.post("/query", json={
        "question": "Does Apple make electric cars and consumer electronics?",
        "top_k": 3
    })
    
    assert response.status_code == 200
    data = response.json()["data"]

    
    # Verify retry loop triggered once
    assert data["retry_count"] == 1
    
    # Verify agent trace contains verify -> retry -> retrieve -> verify -> response
    agent_trace = data["agent_trace"]
    node_names = [event["node"] for event in agent_trace]
    
    assert node_names == [
        "retrieve_node", 
        "verify_node", 
        "retry_increment_node", 
        "retrieve_node", 
        "verify_node", 
        "response_node"
    ]
    
    # Ensure final answer does not contain the unsupported claim
    answer = data["answer"]
    assert "consumer electronics" in answer
    assert "electric cars" not in answer
    
    # Ensure citations are present and map to actual chunks
    citations = data["citations"]
    assert len(citations) > 0
    expanded_chunk_ids = []
    for c in data["expanded_chunks"]:
        expanded_chunk_ids.extend([str(cid) for cid in c.get("constituent_chunk_ids", [])])
        
    for cite in citations:
        assert cite in expanded_chunk_ids
        
    # Verify database persistence
    db = next(get_session())
    record = db.query(QueryRecord).filter(QueryRecord.id == data["query_record_id"]).first()
    assert record is not None
    assert record.retry_count == 1
    assert len(record.verification_results) > 0
    assert record.agent_trace is not None
    assert record.retrieval_trace is not None
    assert record.question == "Does Apple make electric cars and consumer electronics?"
    assert record.answer == answer
