"""Graded ranking metrics. The boss fight is just these three numbers."""

from __future__ import annotations

import math
from dataclasses import dataclass


def dcg(grades: list[float], k: int | None = None) -> float:
    """Discounted cumulative gain.

    Uses the exponential gain ``2**grade - 1`` so a grade of 2 is worth more
    than two grade-1 hits, and divides by ``log2(rank + 1)`` so rank 1 is
    undiscounted (``log2(2) == 1``).
    """
    window = grades if k is None else grades[:k]
    total = 0.0
    for index, grade in enumerate(window):
        total += (2.0**grade - 1.0) / math.log2(index + 2)
    return total


def ndcg_at_k(ranked_ids: list[str], qrels: dict[str, int], k: int) -> float:
    """Normalized DCG. A perfect ordering scores 1.

    Args:
        ranked_ids: Document ids, best first.
        qrels: Document id to graded relevance. Missing ids are grade 0.
        k: Cutoff.

    Returns:
        NDCG in ``[0, 1]``. Zero when nothing in ``qrels`` is relevant.
    """
    gains = [float(qrels.get(doc_id, 0)) for doc_id in ranked_ids[:k]]
    ideal = sorted((float(grade) for grade in qrels.values() if grade > 0), reverse=True)
    idcg = dcg(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg(gains, k) / idcg


def mrr(ranked_ids: list[str], qrels: dict[str, int]) -> float:
    """Mean reciprocal rank for a single query (the reciprocal of the first hit)."""
    for index, doc_id in enumerate(ranked_ids, start=1):
        if qrels.get(doc_id, 0) > 0:
            return 1.0 / index
    return 0.0


def recall_at_k(ranked_ids: list[str], qrels: dict[str, int], k: int) -> float:
    """Fraction of relevant documents that landed in the top ``k``."""
    relevant = {doc_id for doc_id, grade in qrels.items() if grade > 0}
    if not relevant:
        return 0.0
    found = sum(1 for doc_id in ranked_ids[:k] if doc_id in relevant)
    return found / len(relevant)


@dataclass(frozen=True, slots=True)
class QueryScore:
    """One query on the boss scoreboard."""

    query_id: str
    ndcg: float
    mrr: float
    recall: float


@dataclass(frozen=True, slots=True)
class RetrieverReport:
    """Mean metrics for one retriever across the challenge set."""

    name: str
    queries: tuple[QueryScore, ...]

    @property
    def mean_ndcg(self) -> float:
        return _mean([row.ndcg for row in self.queries])

    @property
    def mean_mrr(self) -> float:
        return _mean([row.mrr for row in self.queries])

    @property
    def mean_recall(self) -> float:
        return _mean([row.recall for row in self.queries])


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def score_run(
    name: str,
    rankings: dict[str, list[str]],
    qrels_by_query: dict[str, dict[str, int]],
    k: int,
) -> RetrieverReport:
    """Score a retriever that has already produced a ranking per query id."""
    rows: list[QueryScore] = []
    for query_id, qrels in qrels_by_query.items():
        ranked = rankings.get(query_id, [])
        rows.append(
            QueryScore(
                query_id=query_id,
                ndcg=ndcg_at_k(ranked, qrels, k),
                mrr=mrr(ranked, qrels),
                recall=recall_at_k(ranked, qrels, k),
            )
        )
    return RetrieverReport(name, tuple(rows))
