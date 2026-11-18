"""Knobs for the dojo. Change these, then re-run a belt."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DojoConfig:
    """Hyperparameters a relevance engineer would actually touch.

    Feature weights are applied after per-query min-max normalization, so a
    weight of 0.35 means "this feature may contribute up to 0.35 of the
    rerank score," not "multiply the raw BM25 number by 0.35."
    """

    seed: int = 17
    embed_dim: int = 32
    epochs: int = 60
    lr: float = 0.55
    aux_lr: float = 0.30
    aux_terms: int = 5
    anchor: float = 0.08
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    rrf_k: int = 60
    first_stage_k: int = 10
    result_k: int = 5
    expand_coverage_gate: float = 0.67
    neighbor_min_sim: float = 0.55
    neighbor_pool: int = 15
    max_expansion_terms: int = 4
    prf_docs: int = 3
    prf_terms: int = 4
    weights: tuple[tuple[str, float], ...] = (
        ("bm25", 0.15),
        ("dense", 0.25),
        ("coverage", 0.10),
        ("title", 0.10),
        ("phrase", 0.05),
        ("unique_ratio", 0.35),
    )

    def weight_map(self) -> dict[str, float]:
        """Return a mutable copy of the rerank weights."""
        return dict(self.weights)
