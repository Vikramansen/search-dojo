"""Inverted index, sparse vectors, and Okapi BM25.

This is the yellow belt and the white belt in one file. A posting list maps
a term to the documents that contain it. BM25 walks those lists and adds a
contribution per query term:

    idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |D| / avgdl))

``k1`` bends term frequency so the 20th repeat barely matters. ``b`` makes a
long document work harder for the same single mention.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from search_dojo.tokenize import tokenize, unique_terms
from search_dojo.types import Hit


@dataclass(frozen=True, slots=True)
class TermContribution:
    """One query term's piece of a BM25 score."""

    term: str
    tf: int
    idf: float
    saturation: float
    contribution: float


class InvertedIndex:
    """Posting lists plus the scores we can compute from them."""

    def __init__(self, docs: list[tuple[str, str]]) -> None:
        """Index ``(doc_id, text)`` pairs.

        Args:
            docs: Documents in archive order. Ids must be unique.
        """
        self.doc_ids = [doc_id for doc_id, _ in docs]
        self.n_docs = len(self.doc_ids)
        if len(self.doc_ids) != len(set(self.doc_ids)):
            raise ValueError("duplicate document id")

        self.doc_tokens: dict[str, list[str]] = {}
        self.doc_token_set: dict[str, set[str]] = {}
        self.doc_len: dict[str, int] = {}
        positions: dict[str, dict[str, list[int]]] = {}

        for doc_id, text in docs:
            tokens = tokenize(text)
            self.doc_tokens[doc_id] = tokens
            self.doc_token_set[doc_id] = set(tokens)
            self.doc_len[doc_id] = len(tokens)
            for pos, term in enumerate(tokens):
                positions.setdefault(term, {}).setdefault(doc_id, []).append(pos)

        self.positions: dict[str, dict[str, tuple[int, ...]]] = {
            term: {doc_id: tuple(pos) for doc_id, pos in per_doc.items()}
            for term, per_doc in positions.items()
        }
        self.tf: dict[str, dict[str, int]] = {
            term: {doc_id: len(pos) for doc_id, pos in per_doc.items()}
            for term, per_doc in self.positions.items()
        }
        self.df: dict[str, int] = {term: len(per_doc) for term, per_doc in self.tf.items()}
        total = sum(self.doc_len.values())
        self.avgdl = (total / self.n_docs) if self.n_docs else 1.0
        self.tf_vectors = {doc_id: self._tf_vector(doc_id) for doc_id in self.doc_ids}
        self.tfidf_vectors = {doc_id: self._tfidf_vector(doc_id) for doc_id in self.doc_ids}

    def bm25_idf(self, term: str) -> float:
        """Smooth Okapi idf. Rare terms score higher, and the value stays positive."""
        df = self.df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def smooth_idf(self, term: str) -> float:
        """TF-IDF idf used by the white belt: ``log((1+N)/(1+df)) + 1``."""
        df = self.df.get(term, 0)
        return math.log((1.0 + self.n_docs) / (1.0 + df)) + 1.0

    def term_tf(self, doc_id: str, term: str) -> int:
        """Raw count of ``term`` in ``doc_id``."""
        return self.tf.get(term, {}).get(doc_id, 0)

    def coverage(self, query: str, doc_id: str) -> float:
        """Fraction of unique query terms that appear in the document."""
        terms = unique_terms(query)
        if not terms:
            return 0.0
        have = self.doc_token_set.get(doc_id, set())
        return sum(1 for term in terms if term in have) / len(terms)

    def contains_phrase(self, doc_id: str, terms: tuple[str, ...]) -> bool:
        """True when ``terms`` occur on adjacent token positions."""
        if not terms:
            return False
        if any(doc_id not in self.positions.get(term, {}) for term in terms):
            return False
        first = self.positions[terms[0]][doc_id]
        rest = [set(self.positions[term][doc_id]) for term in terms[1:]]
        for start in first:
            if all((start + offset + 1) in positions for offset, positions in enumerate(rest)):
                return True
        return False

    def phrase_docs(self, terms: tuple[str, ...]) -> set[str]:
        """Document ids whose token stream contains the phrase."""
        if not terms:
            return set()
        candidates: set[str] | None = None
        for term in terms:
            docs = set(self.positions.get(term, {}))
            candidates = docs if candidates is None else candidates & docs
        if not candidates:
            return set()
        return {doc_id for doc_id in candidates if self.contains_phrase(doc_id, terms)}

    def explain_bm25(
        self,
        query: str,
        doc_id: str,
        *,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> tuple[float, list[TermContribution]]:
        """Break a BM25 score into per-term contributions.

        Each unique query term is scored once. Repeating a word in the
        question does not double it. Repeating it in the scroll does, with
        saturation.
        """
        if doc_id not in self.doc_len:
            raise KeyError(doc_id)
        dl = self.doc_len[doc_id]
        parts: list[TermContribution] = []
        for term in unique_terms(query):
            tf = self.term_tf(doc_id, term)
            idf = self.bm25_idf(term)
            saturation = _tf_saturation(tf, dl, self.avgdl, k1, b)
            parts.append(
                TermContribution(
                    term=term,
                    tf=tf,
                    idf=idf,
                    saturation=saturation,
                    contribution=idf * saturation,
                )
            )
        return sum(part.contribution for part in parts), parts

    def bm25(
        self, query: str, *, k: int | None = None, k1: float = 1.2, b: float = 0.75
    ) -> list[Hit]:
        """Rank scrolls by Okapi BM25. Zero-score scrolls are omitted."""
        return self._rank(query, k, lambda doc_id: self._bm25_hit(query, doc_id, k1, b))

    def tf_cosine(self, query: str, *, k: int | None = None) -> list[Hit]:
        """Cosine of raw term counts. The stall sign shouts here."""
        return self._cosine(query, self.tf_vectors, "tf", k)

    def tfidf_cosine(self, query: str, *, k: int | None = None) -> list[Hit]:
        """Cosine of log-tf * smooth idf. Rare words get a louder coordinate."""
        return self._cosine(query, self.tfidf_vectors, "tfidf", k)

    def tfidf_score(self, query: str, doc_id: str) -> float:
        """TF-IDF cosine between one query and one document."""
        return _sparse_cosine(self._query_tfidf_vector(query), self.tfidf_vectors[doc_id])

    def _bm25_hit(self, query: str, doc_id: str, k1: float, b: float) -> Hit | None:
        score, parts = self.explain_bm25(query, doc_id, k1=k1, b=b)
        if score <= 0.0:
            return None
        return Hit(doc_id, score, _bm25_reason(parts))

    def _tf_vector(self, doc_id: str) -> dict[str, float]:
        counts: dict[str, float] = {}
        for token in self.doc_tokens[doc_id]:
            counts[token] = counts.get(token, 0.0) + 1.0
        return counts

    def _tfidf_vector(self, doc_id: str) -> dict[str, float]:
        return {
            term: (1.0 + math.log(tf)) * self.smooth_idf(term)
            for term, tf in self.tf_vectors[doc_id].items()
            if tf > 0
        }

    def _query_tf_vector(self, query: str) -> dict[str, float]:
        counts: dict[str, float] = {}
        for token in tokenize(query):
            counts[token] = counts.get(token, 0.0) + 1.0
        return counts

    def _query_tfidf_vector(self, query: str) -> dict[str, float]:
        return {
            term: (1.0 + math.log(tf)) * self.smooth_idf(term)
            for term, tf in self._query_tf_vector(query).items()
        }

    def _cosine(
        self,
        query: str,
        vectors: dict[str, dict[str, float]],
        label: str,
        k: int | None,
    ) -> list[Hit]:
        query_vector = (
            self._query_tf_vector(query) if label == "tf" else self._query_tfidf_vector(query)
        )
        return self._rank(
            query,
            k,
            lambda doc_id: _cosine_hit(doc_id, query_vector, vectors[doc_id], label),
        )

    def _rank(
        self,
        query: str,
        k: int | None,
        score_one: Callable[[str], Hit | None],
    ) -> list[Hit]:
        terms = unique_terms(query)
        if not terms:
            return []
        docs: set[str] = set()
        for term in terms:
            docs.update(self.tf.get(term, {}))
        hits: list[Hit] = []
        for doc_id in docs:
            hit = score_one(doc_id)
            if hit is not None and hit.score > 0:
                hits.append(hit)
        hits.sort(key=lambda hit: (-hit.score, hit.doc_id))
        if k is None:
            return hits
        return hits[:k]


def _tf_saturation(tf: int, dl: int, avgdl: float, k1: float, b: float) -> float:
    if tf <= 0 or k1 < 0:
        return 0.0
    length_norm = 1.0 - b + b * (dl / avgdl if avgdl else 1.0)
    return tf * (k1 + 1.0) / (tf + k1 * length_norm)


def _bm25_reason(parts: list[TermContribution]) -> str:
    ranked = sorted(parts, key=lambda part: -part.contribution)
    shown = [f"{part.term}:{part.contribution:.2f}" for part in ranked if part.contribution > 0][:3]
    if not shown:
        return "bm25 0"
    return "bm25 " + " ".join(shown)


def _cosine_hit(
    doc_id: str,
    query_vector: dict[str, float],
    doc_vector: dict[str, float],
    label: str,
) -> Hit | None:
    score = _sparse_cosine(query_vector, doc_vector)
    if score <= 0.0:
        return None
    overlap = sorted(set(query_vector) & set(doc_vector))
    preview = " ".join(overlap[:4]) if overlap else "none"
    return Hit(doc_id, score, f"{label} {score:.2f} overlap {preview}")


def _sparse_cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    small, large = (left, right) if len(left) <= len(right) else (right, left)
    dot = 0.0
    for term, weight in small.items():
        other = large.get(term)
        if other is not None:
            dot += weight * other
    left_norm = math.sqrt(sum(weight * weight for weight in left.values()))
    right_norm = math.sqrt(sum(weight * weight for weight in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
