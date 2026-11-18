"""Fuse two rankings when their scores do not share a unit.

BM25 is a sum of idf-weighted saturations. Cosine lives in ``[-1, 1]``.
Adding them is a unit error. Reciprocal rank fusion ignores the raw numbers
and trusts the order:

    score(d) = sum over lists of  1 / (k + rank(d))

``k`` (default 60) keeps a rank-1 hit from erasing a document that only one
list found. A scroll both lists like still floats above a scroll only one
list likes.
"""

from __future__ import annotations


def reciprocal_rank_fusion(rankings: list[list[str]], *, k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked id lists. Ranks are 1-based. Ties break on document id.

    Args:
        rankings: Each inner list is best-first. Duplicates inside one list
            are ignored after the first occurrence.
        k: The constant in ``1 / (k + rank)``. Larger k flattens the top.

    Returns:
        ``(doc_id, score)`` pairs, best first.
    """
    if k < 0:
        raise ValueError("rrf k must be non-negative")
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen: set[str] = set()
        rank = 0
        for doc_id in ranking:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            rank += 1
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
    return [(doc_id, scores[doc_id]) for doc_id in ordered]


def minmax(scores: dict[str, float]) -> dict[str, float]:
    """Scale scores to ``[0, 1]``. A constant list becomes zeros."""
    if not scores:
        return {}
    low = min(scores.values())
    high = max(scores.values())
    if high - low < 1e-12:
        return {doc_id: 0.0 for doc_id in scores}
    return {doc_id: (score - low) / (high - low) for doc_id, score in scores.items()}


def linear_fuse(
    left: dict[str, float],
    right: dict[str, float],
    *,
    alpha: float = 0.5,
) -> list[tuple[str, float]]:
    """Weighted sum of min-max normalized scores.

    ``alpha`` is the weight on ``left``. Documents missing from a list
    contribute 0 on that side. This is the fusion you use when you insist on
    mixing raw scores, after you have put them on a shared scale.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    left_n = minmax(left)
    right_n = minmax(right)
    keys = set(left_n) | set(right_n)
    fused = {
        doc_id: alpha * left_n.get(doc_id, 0.0) + (1.0 - alpha) * right_n.get(doc_id, 0.0)
        for doc_id in keys
    }
    ordered = sorted(fused, key=lambda doc_id: (-fused[doc_id], doc_id))
    return [(doc_id, fused[doc_id]) for doc_id in ordered]
