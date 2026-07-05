"""Reciprocal Rank Fusion (RRF) for combining multiple ranked result lists."""

from typing import Any, Dict, List, Tuple


def reciprocal_rank_fusion(
    result_lists: List[List[Tuple[Any, float]]],
    k: int = 60,
) -> List[Tuple[Any, float]]:
    """Combines multiple ranked lists into a single fused list using RRF.

    Each input list is a list of (item, score) tuples, where items are ranked
    by their position in the list (index 0 = rank 1). The score from each
    input list is NOT used for fusion — only the ordinal rank matters.

    RRF score for an item = sum over all lists of 1 / (k + rank_i)

    Args:
        result_lists: List of ranked result lists. Each result list contains
            (item, original_score) tuples. Items are identified by their `id`
            attribute for deduplication.
        k: Smoothing constant (default 60, standard RRF parameter).

    Returns:
        List of (item, rrf_score) tuples, sorted by rrf_score descending.
    """
    fused_scores: Dict[Any, float] = {}
    item_lookup: Dict[Any, Any] = {}

    for result_list in result_lists:
        for rank, (item, _score) in enumerate(result_list, start=1):
            item_id = item.id
            fused_scores[item_id] = fused_scores.get(item_id, 0.0) + 1.0 / (k + rank)
            # Keep the item reference (last seen wins, all should be the same object)
            item_lookup[item_id] = item

    # Sort by fused score descending
    sorted_ids = sorted(fused_scores.keys(), key=lambda iid: fused_scores[iid], reverse=True)
    return [(item_lookup[iid], fused_scores[iid]) for iid in sorted_ids]
