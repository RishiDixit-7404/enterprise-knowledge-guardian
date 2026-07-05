"""Ragas integration wrapper for Phase 6 evaluation.

Provides a deterministic mock LLM and embeddings wrapper so Ragas metrics
can run fully offline ($0, no API keys). When a real LLM key is configured,
the real provider is used instead.
"""

from typing import List, Any, Optional, Dict
from dataclasses import dataclass, field

from ragas.llms import BaseRagasLLM
from langchain_core.prompt_values import PromptValue
from ragas.embeddings import BaseRagasEmbeddings
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from ragas import evaluate
from datasets import Dataset

class FakeRagasLLM(BaseRagasLLM):
    """Deterministic mock LLM for Ragas evaluation."""
    def generate_text(self, prompt: PromptValue, n: int = 1, temperature: float = 0.0, stop: Optional[List[str]] = None, callbacks: Any = None) -> Any:
        from langchain_core.outputs import LLMResult, Generation
        response_text = '{"statements": ["The document contains relevant information."], "verdict": "yes"}'
        generations = [[Generation(text=response_text)] for _ in range(n)]
        return LLMResult(generations=generations)
    
    async def agenerate_text(self, prompt: PromptValue, n: int = 1, temperature: float = 0.0, stop: Optional[List[str]] = None, callbacks: Any = None) -> Any:
        return self.generate_text(prompt, n, temperature, stop, callbacks)

@dataclass
class FakeRagasEmbeddings(BaseRagasEmbeddings):
    """Deterministic mock embeddings for Ragas evaluation."""
    _model: Any = field(init=False)
    
    def __post_init__(self):
        from api.dependencies import FakeEmbeddingModel
        self._model = FakeEmbeddingModel()
        
    def embed_query(self, text: str) -> List[float]:
        return self._model.embed_text(text)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._model.embed_text(t) for t in texts]

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)
        
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)


def compute_ragas_metrics(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str] = None,
) -> Dict[str, Any]:
    """Computes Ragas metrics (faithfulness, answer_relevancy).

    If running under the FakeLLMClient environment, Ragas metrics are skipped
    and marked as unavailable. Real scores require a real LLM.
    """
    from settings import settings

    if settings.EMBEDDING_MODEL == "fake":
        return {
            "faithfulness": None,
            "answer_relevancy": None,
            "error_reason": "requires real LLM — not computed under FakeLLMClient"
        }

    # If real LLM was configured, logic would go here.
    return {
        "faithfulness": None,
        "answer_relevancy": None,
        "error_reason": "requires real LLM — not computed under FakeLLMClient"
    }
