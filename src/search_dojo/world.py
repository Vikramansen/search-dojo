"""The expanded Dojo DB, graded queries, and a simulated query log.

The white belt uses the 9 starter documents. Every later walkthrough uses
all 35 documents plus 18 queries with graded relevance judgments:

    3 = perfect answer   2 = highly relevant   1 = somewhat relevant   0 = not relevant

A document missing from a query's judgments counts as 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from search_dojo.corpus import DATA_DIR, _rows
from search_dojo.index import InvertedIndex
from search_dojo.minidb import MiniDoc, load_mini

SEED = 7

# Position-based click model used by every click simulation:
# P(click at rank r) = P(examine rank r) × P(click | examined, grade).
EXAMINE = (1.0, 0.62, 0.42, 0.28, 0.18)
ATTRACT = {0: 0.04, 1: 0.25, 2: 0.55, 3: 0.85}

# Unjudged queries that also show up in the log: typos and things the database lacks.
TAIL_QUERIES = {
    "haland goals": 9,
    "thalapthy": 7,
    "meesseeks": 5,
    "lebron james": 6,
    "messi": 8,
    "dhoni retirement": 4,
}


@dataclass(frozen=True, slots=True)
class GradedQuery:
    """A query with relevance judgments (qrels)."""

    id: str
    text: str
    qrels: dict[str, int]

    def grade(self, doc_id: str) -> int:
        return self.qrels.get(doc_id, 0)


@dataclass
class World:
    """Everything the later lessons share."""

    docs: list[MiniDoc]
    index: InvertedIndex
    queries: list[GradedQuery]

    def __post_init__(self) -> None:
        self.by_id = {doc.id: doc for doc in self.docs}
        self.by_query = {query.id: query for query in self.queries}

    def bm25_ids(self, text: str, k: int = 10) -> list[str]:
        return [hit.doc_id for hit in self.index.bm25(text, k=k)]

    def tfidf_ids(self, text: str, k: int = 10) -> list[str]:
        return [hit.doc_id for hit in self.index.tfidf_cosine(text, k=k)]

    def query_counts(self) -> dict[str, int]:
        """How often each query appears in the simulated log. Roughly Zipf: a few queries dominate."""
        rng = np.random.default_rng(SEED)
        order = rng.permutation(len(self.queries))
        counts: dict[str, int] = {}
        for rank, index in enumerate(order, start=1):
            counts[self.queries[int(index)].text] = int(round(900 / rank))
        counts.update(TAIL_QUERIES)
        return counts


def load_more_docs() -> list[MiniDoc]:
    return [
        MiniDoc(id=str(row["id"]), topic=str(row["topic"]), text=str(row["text"]))
        for row in _rows(DATA_DIR / "dojo_db_more.jsonl")
    ]


def load_queries() -> list[GradedQuery]:
    queries: list[GradedQuery] = []
    for row in _rows(DATA_DIR / "dojo_queries.jsonl"):
        raw = row["qrels"]
        if not isinstance(raw, dict):
            raise ValueError(f"query {row.get('id')} qrels must be an object")
        queries.append(
            GradedQuery(
                id=str(row["id"]),
                text=str(row["text"]),
                qrels={str(doc_id): int(grade) for doc_id, grade in raw.items()},
            )
        )
    return queries


@lru_cache(maxsize=1)
def build_world() -> World:
    """Load all 35 documents and the graded queries, and check every reference."""
    docs = load_mini() + load_more_docs()
    ids = [doc.id for doc in docs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate document id across dojo_db files")
    known = set(ids)
    queries = load_queries()
    for query in queries:
        for doc_id, grade in query.qrels.items():
            if doc_id not in known:
                raise ValueError(f"{query.id} judges unknown document {doc_id}")
            if grade not in (1, 2, 3):
                raise ValueError(f"{query.id} grade for {doc_id} must be 1, 2 or 3")
    index = InvertedIndex([(doc.id, doc.text) for doc in docs])
    return World(docs=docs, index=index, queries=queries)
