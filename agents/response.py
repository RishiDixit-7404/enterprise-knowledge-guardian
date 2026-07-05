"""Response Agent for generating the final answer based on verified claims."""

from typing import Dict, Any, List, Tuple

class ResponseAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        
    def generate(self, question: str, claims: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        """
        Filters out unsupported claims and generates the final answer with inline citations.
        Returns a tuple of (final_answer, list_of_cited_chunk_ids).
        """
        # Strictly filter out unsupported claims
        supported_claims = [c for c in claims if c.get("supported")]
        
        # Call LLM to generate narrative response
        final_answer = self.llm_client.generate_response(question, supported_claims)
        
        # Aggregate all valid citations used in the response
        citations_set = set()
        for claim in supported_claims:
            for cid in claim.get("evidence_chunk_ids", []):
                citations_set.add(str(cid))
                
        return final_answer, list(citations_set)
