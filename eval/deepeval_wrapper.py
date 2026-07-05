"""DeepEval integration wrapper for Phase 6 evaluation.

Provides a deterministic mock LLM wrapper so DeepEval hallucination 
metrics can run offline. Real scores require a real LLM.
"""

from typing import List, Any
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

class FakeDeepEvalLLM(DeepEvalBaseLLM):
    """Deterministic mock LLM for DeepEval evaluation."""
    def get_model_name(self) -> str:
        return "FakeDeepEvalLLM"
    def load_model(self) -> Any:
        return self
    def generate(self, prompt: str, **kwargs) -> str:
        return '{"verdicts": [{"verdict": "no", "reason": "The statement is supported by the context."}]}'
    async def a_generate(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)

def compute_deepeval_hallucination(question: str, answer: str, contexts: List[str]) -> Any:
    """Computes hallucination using DeepEval.
    
    If running under the FakeLLMClient environment, metric is skipped
    and marked as unavailable.
    """
    from settings import settings

    if settings.EMBEDDING_MODEL == "fake":
        return None

    # If real LLM was configured, logic would go here.
    return None
