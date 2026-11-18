"""Simulated users clicking on ranked lists, with a position-based click model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from search_dojo.world import ATTRACT, EXAMINE, GradedQuery, World


@dataclass(frozen=True)
class ClickConfig:
    """Size and randomness of the simulated click log."""

    sessions: int = 5000
    random_share: float = 0.10
    k: int = len(EXAMINE)
    seed: int = 21


@dataclass(frozen=True)
class Session:
    """One search: the query, what was shown, and what got clicked."""

    query: GradedQuery
    shown: tuple[str, ...]
    clicks: tuple[bool, ...]
    randomized: bool


def click_probability(rank: int, grade: int) -> float:
    """P(click) = P(examine rank) × P(click | examined, grade). ``rank`` is 0-based."""
    return EXAMINE[rank] * ATTRACT[grade]


def simulate_clicks(
    query: GradedQuery, shown: list[str], rng: np.random.Generator
) -> tuple[bool, ...]:
    """Each position is examined and clicked independently under the position-based model."""
    return tuple(
        bool(rng.random() < click_probability(rank, query.grade(doc_id)))
        for rank, doc_id in enumerate(shown)
    )


def traffic(world: World) -> tuple[list[GradedQuery], np.ndarray]:
    """Judged queries and their share of the simulated traffic."""
    counts = world.query_counts()
    queries = world.queries
    weights = np.array([counts[query.text] for query in queries], dtype=float)
    return queries, weights / weights.sum()


def simulate_log(world: World, config: ClickConfig) -> list[Session]:
    """Production traffic ranks by BM25; a small bucket shuffles the top k."""
    rng = np.random.default_rng(config.seed)
    queries, probabilities = traffic(world)
    rankings = {query.id: world.bm25_ids(query.text, config.k) for query in queries}
    log: list[Session] = []
    for index in rng.choice(len(queries), size=config.sessions, p=probabilities):
        query = queries[int(index)]
        shown = list(rankings[query.id])
        randomized = bool(rng.random() < config.random_share)
        if randomized:
            rng.shuffle(shown)
        log.append(Session(query, tuple(shown), simulate_clicks(query, shown, rng), randomized))
    return log
