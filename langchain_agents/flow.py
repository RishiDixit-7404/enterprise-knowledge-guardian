"""LangGraph orchestration for the Retrieval -> Verification -> Response pipeline."""

from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from agents.retrieval import RetrievalAgent
from agents.verification import VerificationAgent
from agents.response import ResponseAgent
from ingest.interfaces import FakeLLMClient

class GraphState(TypedDict):
    question: str
    filters: Dict[str, Any]
    top_k: int
    db: Any
    graph: Any
    
    retrieval_trace: Dict[str, Any]
    expanded_chunks: List[Dict[str, Any]]
    
    claims: List[Dict[str, Any]]
    retry_count: int
    
    final_answer: str
    citations: List[str]
    agent_trace: List[Dict[str, Any]]

def retrieve_node(state: GraphState):
    db = state["db"]
    graph = state["graph"]
    retrieval_agent = RetrievalAgent(db, graph)
    
    # Increase top_k slightly on retry to simulate fetching more evidence
    current_top_k = state["top_k"]
    if state.get("retry_count", 0) > 0:
        current_top_k += 2
        
    filters = state.get("filters", {})
    result = retrieval_agent.retrieve(
        question=state["question"],
        top_k=current_top_k,
        company=filters.get("company"),
        doc_type=filters.get("doc_type"),
        date_from=filters.get("date_from"),
        date_to=filters.get("date_to")
    )
    
    trace_event = {
        "node": "retrieve_node",
        "retry_count": state.get("retry_count", 0),
        "retrieved_chunk_count": len(result["expanded_chunks"])
    }
    
    return {
        "retrieval_trace": result["retrieval_trace"],
        "expanded_chunks": result["expanded_chunks"],
        "agent_trace": state.get("agent_trace", []) + [trace_event]
    }

def verify_node(state: GraphState):
    llm_client = FakeLLMClient()
    verification_agent = VerificationAgent(llm_client)
    
    claims, usage = verification_agent.verify(
        question=state["question"],
        expanded_chunks=state["expanded_chunks"]
    )
    
    trace_event = {
        "node": "verify_node",
        "retry_count": state.get("retry_count", 0),
        "claims_generated": len(claims),
        "usage": usage
    }
    
    return {
        "claims": claims,
        "agent_trace": state.get("agent_trace", []) + [trace_event]
    }

def response_node(state: GraphState):
    llm_client = FakeLLMClient()
    responder = ResponseAgent(llm_client)
    
    final_answer, citations, usage = responder.generate(state["question"], state["claims"])
    
    trace_event = {
        "node": "response_node",
        "citations_count": len(citations),
        "usage": usage
    }
    
    return {
        "final_answer": final_answer,
        "citations": citations,
        "agent_trace": state.get("agent_trace", []) + [trace_event]
    }

def check_verification(state: GraphState):
    """
    Edge condition to determine routing after verification.
    """
    claims = state.get("claims", [])
    unsupported_count = sum(1 for c in claims if not c.get("supported"))
    retry_count = state.get("retry_count", 0)
    
    if unsupported_count > 0 and retry_count < 1:
        # Increment retry_count directly via StateGraph's update logic
        # But edges can't modify state directly in newer LangGraph easily without a node.
        # Wait, the edge just returns the next node name. 
        # We need a small node to increment retry count. 
        # Actually, we can just return "retry_increment_node" which then goes to "retrieve_node"
        return "retry_increment_node"
    return "response_node"

def retry_increment_node(state: GraphState):
    return {
        "retry_count": state.get("retry_count", 0) + 1,
        "agent_trace": state.get("agent_trace", []) + [{"node": "retry_increment_node"}]
    }

# Build the workflow
workflow = StateGraph(GraphState)

workflow.add_node("retrieve_node", retrieve_node)
workflow.add_node("verify_node", verify_node)
workflow.add_node("response_node", response_node)
workflow.add_node("retry_increment_node", retry_increment_node)

workflow.set_entry_point("retrieve_node")

workflow.add_edge("retrieve_node", "verify_node")

workflow.add_conditional_edges(
    "verify_node",
    check_verification,
    {
        "retry_increment_node": "retry_increment_node",
        "response_node": "response_node"
    }
)

workflow.add_edge("retry_increment_node", "retrieve_node")
workflow.add_edge("response_node", END)

app = workflow.compile()
