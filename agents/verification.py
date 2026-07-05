"""Verification Agent for generating and verifying claims against retrieved evidence."""

from typing import Dict, Any, List
from ingest.interfaces import get_embedding_model

class VerificationAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        
    def verify(self, question: str, expanded_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Drafts candidate claims and verifies them against the evidence chunks.
        Returns a list of claims containing text, supported boolean, and evidence_chunk_ids.
        """
        # Call LLM (fake or real) to generate and verify claims
        claims = self.llm_client.generate_and_verify_claims(question, expanded_chunks)
        
        # Ensure that citations only refer to chunk IDs actually present in expanded_chunks.
        # This prevents the LLM from hallucinating chunk IDs.
        valid_chunk_ids = set()
        for c in expanded_chunks:
            for cid in c.get("constituent_chunk_ids", []):
                valid_chunk_ids.add(str(cid))
                
        for claim in claims:
            filtered_cites = [cid for cid in claim.get("evidence_chunk_ids", []) if str(cid) in valid_chunk_ids]
            claim["evidence_chunk_ids"] = filtered_cites
            # If a claim was marked supported but has no valid citations after filtering, flag as unsupported
            if claim.get("supported") and not claim["evidence_chunk_ids"]:
                claim["supported"] = False
                
        return claims
