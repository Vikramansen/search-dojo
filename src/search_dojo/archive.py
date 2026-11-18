"""The whole search system, in one room.

    query
      -> expand, only when the top lexical hit covers the question poorly
      -> BM25 list and bi-encoder list
      -> reciprocal rank fusion
      -> feature rerank
      -> top scrolls

``build_archive`` trains the tower once and caches it. Read this file when
the belts start to feel like separate tricks. They are one pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from loguru import logger

from search_dojo.biencoder import TinyBiEncoder
from search_dojo.config import DojoConfig
from search_dojo.corpus import (
    PROMPT_PATH,
    Challenge,
    Drill,
    LogEvent,
    Scroll,
    assert_references,
    load_challenges,
    load_drills,
    load_logs,
    load_scrolls,
)
from search_dojo.expand import expand_query, llm_rewrite, prf_terms
from search_dojo.hybrid import reciprocal_rank_fusion
from search_dojo.index import InvertedIndex
from search_dojo.logs import Reformulation, lexicon_from_logs
from search_dojo.queryexpr import content_terms, match_expr, parse_query
from search_dojo.rerank import rerank
from search_dojo.tokenize import tokenize
from search_dojo.types import Hit

RETRIEVERS = ("tf", "tfidf", "bm25", "dense", "hybrid", "rerank", "expand", "pipeline", "expr")


@dataclass
class Archive:
    """Scrolls, index, trained tower, and the logs that rewrite questions."""

    config: DojoConfig
    scrolls: list[Scroll]
    drills: list[Drill]
    challenges: list[Challenge]
    logs: list[LogEvent]
    index: InvertedIndex
    encoder: TinyBiEncoder
    doc_vectors: dict[str, np.ndarray]
    loss_curve: list[float]
    lexicon: dict[str, list[str]]
    kept_reforms: list[Reformulation]
    suspect_reforms: list[Reformulation]

    def __post_init__(self) -> None:
        self.by_id = {scroll.id: scroll for scroll in self.scrolls}
        self.titles = {scroll.id: set(tokenize(scroll.title)) for scroll in self.scrolls}

    def search(
        self,
        query: str,
        retriever: str,
        k: int | None = None,
        *,
        k1: float | None = None,
        b: float | None = None,
        weights: dict[str, float] | None = None,
    ) -> list[Hit]:
        """Run one retriever. ``k`` defaults to ``result_k``."""
        limit = self.config.result_k if k is None else k
        if retriever == "tf":
            return self.index.tf_cosine(query, k=limit)
        if retriever == "tfidf":
            return self.index.tfidf_cosine(query, k=limit)
        if retriever == "bm25":
            return self.index.bm25(query, k=limit, k1=self._k1(k1), b=self._b(b))
        if retriever == "dense":
            return self.search_dense(query, k=limit)
        if retriever == "hybrid":
            return self.search_hybrid(query, k=limit, k1=k1, b=b)
        if retriever == "rerank":
            return self.search_rerank(query, k=limit, weights=weights)
        if retriever == "expand":
            return self.search_expanded_bm25(query, k=limit)
        if retriever == "pipeline":
            return self.search_pipeline(query, k=limit, weights=weights)
        if retriever == "expr":
            return self.search_expr(query, k=limit)
        known = ", ".join(RETRIEVERS)
        raise ValueError(f"unknown retriever {retriever!r}. choose one of: {known}")

    def search_dense(self, query: str, k: int | None = None) -> list[Hit]:
        """Cosine against precomputed scroll vectors."""
        query_vector = self.encoder.encode(query)
        hits: list[Hit] = []
        if float(np.linalg.norm(query_vector)) == 0.0:
            return []
        for doc_id, vector in self.doc_vectors.items():
            score = float(query_vector @ vector)
            if score <= 0.0:
                continue
            hits.append(Hit(doc_id, score, f"cosine {score:.2f}"))
        hits.sort(key=lambda hit: (-hit.score, hit.doc_id))
        return _cut(hits, k)

    def search_hybrid(
        self,
        query: str,
        k: int | None = None,
        *,
        k1: float | None = None,
        b: float | None = None,
    ) -> list[Hit]:
        """RRF of the BM25 list and the dense list."""
        pool = max(self.config.first_stage_k, k or 0)
        lexical = self.index.bm25(query, k=pool, k1=self._k1(k1), b=self._b(b))
        dense = self.search_dense(query, k=pool)
        lexical_ids = [hit.doc_id for hit in lexical]
        dense_ids = [hit.doc_id for hit in dense]
        fused = reciprocal_rank_fusion([lexical_ids, dense_ids], k=self.config.rrf_k)
        hits = [
            Hit(doc_id, score, _fusion_reason(doc_id, score, lexical_ids, dense_ids))
            for doc_id, score in fused
        ]
        return _cut(hits, k)

    def search_rerank(
        self,
        query: str,
        k: int | None = None,
        *,
        weights: dict[str, float] | None = None,
        source: str = "bm25",
    ) -> list[Hit]:
        """Rerank a first-stage list. Default first stage is BM25."""
        pool = self.config.first_stage_k
        if source == "bm25":
            candidates = self.index.bm25(query, k=pool)
        elif source == "hybrid":
            candidates = self.search_hybrid(query, k=pool)
        else:
            raise ValueError(f"unknown rerank source {source!r}")
        hits = rerank(
            query,
            candidates,
            index=self.index,
            doc_vectors=self.doc_vectors,
            query_vector=self.encoder.encode(query),
            titles=self.titles,
            weights=weights or self.config.weight_map(),
        )
        return _cut(hits, k)

    def expand(self, query: str) -> str:
        """Apply the default expander (logs + neighbors, no feedback)."""
        cfg = self.config
        return expand_query(
            query,
            index=self.index,
            encoder=self.encoder,
            lexicon=self.lexicon,
            gate=cfg.expand_coverage_gate,
            min_sim=cfg.neighbor_min_sim,
            pool=cfg.neighbor_pool,
            limit=cfg.max_expansion_terms,
        )

    def search_expanded_bm25(self, query: str, k: int | None = None) -> list[Hit]:
        """BM25 on the expanded query. The reason shows the rewrite."""
        rewritten = self.expand(query)
        hits = self.index.bm25(rewritten, k=k)
        if rewritten == query:
            prefix = "expand kept"
        else:
            prefix = f"expand -> {rewritten}"
        return [hit.with_reason(f"{prefix} | {hit.reason}") for hit in hits]

    def search_pipeline(
        self,
        query: str,
        k: int | None = None,
        *,
        weights: dict[str, float] | None = None,
    ) -> list[Hit]:
        """Expand, fuse BM25 with the tower, then rerank."""
        rewritten = self.expand(query)
        candidates = self.search_hybrid(rewritten, k=self.config.first_stage_k)
        hits = rerank(
            rewritten,
            candidates,
            index=self.index,
            doc_vectors=self.doc_vectors,
            query_vector=self.encoder.encode(rewritten),
            titles=self.titles,
            weights=weights or self.config.weight_map(),
        )
        prefix = "pipeline" if rewritten == query else f"pipeline -> {rewritten}"
        tagged = [hit.with_reason(f"{prefix} | {hit.reason}") for hit in hits]
        return _cut(tagged, k)

    def search_expr(self, query: str, k: int | None = None) -> list[Hit]:
        """Boolean match, then BM25 among the survivors."""
        expr = parse_query(query)
        matched = match_expr(self.index, expr)
        lexical = " ".join(content_terms(expr))
        ranked = {hit.doc_id: hit for hit in self.index.bm25(lexical)}
        hits: list[Hit] = []
        for doc_id in matched:
            base = ranked.get(doc_id)
            score = 0.0 if base is None else base.score
            hits.append(Hit(doc_id, score, "expr match"))
        hits.sort(key=lambda hit: (-hit.score, hit.doc_id))
        return _cut(hits, k)

    def prf(self, query: str) -> list[str]:
        """Feedback terms. Useful, and sometimes poisonous."""
        return prf_terms(
            self.index,
            query,
            k_docs=self.config.prf_docs,
            k_terms=self.config.prf_terms,
        )

    def rewrite_with(self, query: str, complete: object) -> str:
        """Send ``query`` through the prompt file and a completion function."""
        if not callable(complete):
            raise TypeError("complete must be a function from prompt to text")
        return llm_rewrite(query, complete, PROMPT_PATH)

    def challenge(self, challenge_id: str) -> Challenge:
        """Look up a graded question by id."""
        for item in self.challenges:
            if item.id == challenge_id:
                return item
        raise KeyError(challenge_id)

    def _k1(self, k1: float | None) -> float:
        return self.config.bm25_k1 if k1 is None else k1

    def _b(self, b: float | None) -> float:
        return self.config.bm25_b if b is None else b


def build_archive(config: DojoConfig | None = None) -> Archive:
    """Load the archive, train the tower, and mine the logs.

    Results are cached per config. The default config is built once per process.
    """
    cfg = config or DojoConfig()
    return _build_cached(cfg)


@lru_cache(maxsize=4)
def _build_cached(config: DojoConfig) -> Archive:
    scrolls = load_scrolls()
    drills = load_drills()
    challenges = load_challenges()
    logs = load_logs()
    assert_references(scrolls, drills, challenges, logs)
    by_id = {scroll.id: scroll for scroll in scrolls}
    index = InvertedIndex([(scroll.id, scroll.body) for scroll in scrolls])
    texts = [scroll.body for scroll in scrolls]
    texts.extend(drill.query for drill in drills)
    encoder = TinyBiEncoder.from_texts(texts, dim=config.embed_dim, seed=config.seed)
    triples: list[tuple[str, str, str]] = []
    for drill in drills:
        positive = by_id[drill.positive].body
        for negative_id in drill.negatives:
            triples.append((drill.query, positive, by_id[negative_id].body))
    curve = encoder.train(
        triples,
        epochs=config.epochs,
        lr=config.lr,
        aux_lr=config.aux_lr,
        aux_terms=config.aux_terms,
        df=index.df,
        anchor=config.anchor,
    )
    if curve:
        logger.debug("bi-encoder loss {:.4f} -> {:.4f}", curve[0], curve[-1])
    doc_vectors = {scroll.id: encoder.encode(scroll.body) for scroll in scrolls}
    lexicon, kept, suspect = lexicon_from_logs(
        logs,
        encoder,
        min_sim=config.neighbor_min_sim,
    )
    logger.debug("log lexicon kept {} vetoed {}", len(kept), len(suspect))
    return Archive(
        config=config,
        scrolls=scrolls,
        drills=drills,
        challenges=challenges,
        logs=logs,
        index=index,
        encoder=encoder,
        doc_vectors=doc_vectors,
        loss_curve=curve,
        lexicon=lexicon,
        kept_reforms=kept,
        suspect_reforms=suspect,
    )


def _cut(hits: list[Hit], k: int | None) -> list[Hit]:
    if k is None:
        return hits
    return hits[:k]


def _fusion_reason(doc_id: str, score: float, lexical: list[str], dense: list[str]) -> str:
    def _place(ids: list[str]) -> str:
        if doc_id not in ids:
            return "miss"
        return f"#{ids.index(doc_id) + 1}"

    return f"rrf {score:.3f} bm25 {_place(lexical)} dense {_place(dense)}"
