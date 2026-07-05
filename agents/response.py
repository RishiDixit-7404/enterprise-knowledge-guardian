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
        import time
        from llmops.tokens import count_tokens
        from prompts.response import RESPONSE_PROMPT_TEMPLATE, PROMPT_ID, PROMPT_VERSION, PROMPT_SOURCE
        
        # Strictly filter out unsupported claims
        supported_claims = [c for c in claims if c.get("supported")]
        
        start_time = time.time()
        # Call LLM to generate narrative response
        final_answer = self.llm_client.generate_response(question, supported_claims)
        latency_ms = int((time.time() - start_time) * 1000)
        
        prompt_text = RESPONSE_PROMPT_TEMPLATE.format(question=question)
        input_tokens = count_tokens(prompt_text)
        output_tokens = count_tokens(final_answer)
        
        usage = {
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "prompt_id": PROMPT_ID,
            "prompt_version": PROMPT_VERSION,
            "prompt_source": PROMPT_SOURCE,
            "model_name": "fake" if self.llm_client.__class__.__name__ == "FakeLLMClient" else "real"
        }
        
        # Aggregate all valid citations used in the response
        citations_set = set()
        for claim in supported_claims:
            for cid in claim.get("evidence_chunk_ids", []):
                citations_set.add(str(cid))
                
        return final_answer, list(citations_set), usage
