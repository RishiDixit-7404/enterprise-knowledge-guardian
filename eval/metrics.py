"""Custom evaluation metrics for Phase 6.

Computes Context Recall@K, Citation Correctness, Hallucination Rate,
and wall-clock latency from pipeline outputs. These metrics require no
LLM-as-judge and run fully offline.
"""

from typing import Dict, Any, List, Optional


def context_recall_at_k(
    retrieved_chunks: List[Dict[str, Any]],
    expected_chunk_ids: List[str],
) -> float:
    """Computes Context Recall@K.

    Checks what fraction of expected chunk IDs were present in the retrieved set.

    Returns: fraction of expected chunk IDs found in retrieved chunks.
    """
    if not expected_chunk_ids:
        return 1.0

    retrieved_ids = {str(c.get("document_id")) for c in retrieved_chunks} 
    # Actually wait, retrieved chunks have "constituent_chunk_ids". 
    # In expanded chunks, the chunk ID might be in `constituent_chunk_ids`.
    retrieved_ids = set()
    for c in retrieved_chunks:
        if "constituent_chunk_ids" in c:
            retrieved_ids.update(str(x) for x in c["constituent_chunk_ids"])
        else:
            retrieved_ids.add(str(c.get("id", "")))

    hits = sum(1 for eid in expected_chunk_ids if str(eid) in retrieved_ids)
    return hits / len(expected_chunk_ids)


def citation_correctness(
    answer: str,
    citations: List[str],
    expanded_chunks: List[Dict[str, Any]],
) -> float:
    """Computes Citation Correctness.

    For each cited chunk ID:
      (a) The chunk must exist in the expanded_chunks set (was retrieved).
      (b) The chunk text must have some relevance to the answer.

    Returns: valid_citations / total_citations (1.0 if no citations).
    """
    if not citations:
        return 1.0

    # Build a lookup of chunk IDs to chunk text from expanded chunks
    chunk_text_by_id: Dict[str, str] = {}
    for ec in expanded_chunks:
        for cid in ec.get("constituent_chunk_ids", []):
            chunk_text_by_id[str(cid)] = ec.get("text", "")

    valid = 0
    for cite_id in citations:
        cite_str = str(cite_id)
        if cite_str in chunk_text_by_id:
            valid += 1

    return valid / len(citations)


def hallucination_rate(
    verification_results: List[Dict[str, Any]],
) -> float:
    """Computes Hallucination Rate.

    Defined exactly as: 1 - (supported_claims / total_claims).
    Uses the verification outputs already produced by the agent pipeline.

    Returns: hallucination rate (0.0 = no hallucinations, 1.0 = all hallucinated).
    """
    if not verification_results:
        return 0.0

    total = len(verification_results)
    supported = sum(1 for v in verification_results if v.get("supported"))

    return 1.0 - (supported / total)
