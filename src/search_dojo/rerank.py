"""A feature reranker you can ablate with one zero.

Production second stages are often cross-encoders: one network reads the
query and the scroll together. They are sharp and expensive, so they only
see the top of a fast first stage. This module is the same slot with the
weights written out. If a document never reached ``candidates``, no weight
can bring it back.
"""

from __future__ import annotations

import numpy as np

from search_dojo.index import InvertedIndex
from search_dojo.tokenize import unique_terms
from search_dojo.types import Hit


def feature_rows(
    query: str,
    candidates: list[Hit],
    *,
    index: InvertedIndex,
    doc_vectors: dict[str, np.ndarray],
    query_vector: np.ndarray,
    titles: dict[str, set[str]],
) -> list[dict[str, float]]:
    """Raw features for each candidate, in candidate order.

    Keys: ``bm25``, ``dense``, ``coverage``, ``title``, ``phrase``,
    ``unique_ratio``.
    """
    terms = unique_terms(query)
    phrase = tuple(terms)
    rows: list[dict[str, float]] = []
    for hit in candidates:
        tokens = index.doc_tokens.get(hit.doc_id, [])
        unique = set(tokens)
        title_terms = titles.get(hit.doc_id, set())
        title_hit = 0.0
        if terms and any(term in title_terms for term in terms):
            title_hit = 1.0
        phrase_hit = 1.0 if phrase and index.contains_phrase(hit.doc_id, phrase) else 0.0
        bm25, _ = index.explain_bm25(query, hit.doc_id)
        dense = 0.0
        vector = doc_vectors.get(hit.doc_id)
        if vector is not None and float(np.linalg.norm(query_vector)) > 0:
            dense = float(query_vector @ vector)
        rows.append(
            {
                "bm25": bm25,
                "dense": dense,
                "coverage": index.coverage(query, hit.doc_id),
                "title": title_hit,
                "phrase": phrase_hit,
                "unique_ratio": (len(unique) / len(tokens)) if tokens else 0.0,
            }
        )
    return rows


def normalize_rows(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    """Min-max each feature across the candidate list. Constants become 0."""
    if not rows:
        return []
    keys = list(rows[0])
    scaled: list[dict[str, float]] = [{key: 0.0 for key in keys} for _ in rows]
    for key in keys:
        values = [row[key] for row in rows]
        low = min(values)
        high = max(values)
        for index, value in enumerate(values):
            if high - low < 1e-12:
                scaled[index][key] = 0.0
            else:
                scaled[index][key] = (value - low) / (high - low)
    return scaled


def combine(rows: list[dict[str, float]], weights: dict[str, float]) -> list[float]:
    """Dot normalized feature rows with ``weights``. Missing keys count as 0."""
    scores: list[float] = []
    for row in rows:
        scores.append(sum(weights.get(key, 0.0) * value for key, value in row.items()))
    return scores


def rerank(
    query: str,
    candidates: list[Hit],
    *,
    index: InvertedIndex,
    doc_vectors: dict[str, np.ndarray],
    query_vector: np.ndarray,
    titles: dict[str, set[str]],
    weights: dict[str, float],
) -> list[Hit]:
    """Reorder ``candidates``. The output never contains a new document id."""
    if not candidates:
        return []
    raw = feature_rows(
        query,
        candidates,
        index=index,
        doc_vectors=doc_vectors,
        query_vector=query_vector,
        titles=titles,
    )
    scaled = normalize_rows(raw)
    scores = combine(scaled, weights)
    hits: list[Hit] = []
    for hit, score, row in zip(candidates, scores, scaled, strict=True):
        hits.append(Hit(hit.doc_id, score, _reason(row, weights), dict(row)))
    hits.sort(key=lambda hit: (-hit.score, hit.doc_id))
    return hits


def _reason(row: dict[str, float], weights: dict[str, float]) -> str:
    parts = sorted(
        ((weights.get(key, 0.0) * value, key, value) for key, value in row.items()),
        reverse=True,
    )
    shown = [f"{key}:{value:.2f}" for _contrib, key, value in parts if _contrib > 0][:3]
    if not shown:
        return "rerank 0"
    return "rerank " + " ".join(shown)
