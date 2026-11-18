"""Query logs: free labels, with the poison still in the bottle.

A reformulation is two queries in one session where the first got no click
and the second did. Some of those are vocabulary gaps (``white pig soup``
becomes ``tonkotsu pork bone``). Some are a new intent (``milky soup``
becomes a search for the bleach spill). The encoder gets a veto: an added
term is kept only when it already lives near the original wording.
"""

from __future__ import annotations

from dataclasses import dataclass

from search_dojo.biencoder import TinyBiEncoder
from search_dojo.corpus import LogEvent
from search_dojo.tokenize import unique_terms


@dataclass(frozen=True, slots=True)
class Reformulation:
    """A miss followed by a click, inside one session."""

    session: str
    before: str
    after: str
    click: str


def reformulations(events: list[LogEvent]) -> list[Reformulation]:
    """Extract miss-then-click pairs. File order is session order."""
    grouped: dict[str, list[LogEvent]] = {}
    for event in events:
        grouped.setdefault(event.session, []).append(event)
    found: list[Reformulation] = []
    for session, rows in grouped.items():
        for previous, current in zip(rows, rows[1:], strict=False):
            if previous.click is not None or current.click is None:
                continue
            if previous.query == current.query:
                continue
            found.append(
                Reformulation(
                    session=session,
                    before=previous.query,
                    after=current.query,
                    click=current.click,
                )
            )
    return found


def zero_result_queries(events: list[LogEvent]) -> list[str]:
    """Queries the index showed nothing for."""
    return [event.query for event in events if event.shown == 0]


def mean_click_rank(events: list[LogEvent]) -> float:
    """Average rank of clicks. Lower means searchers found the scroll sooner."""
    ranks = [event.rank for event in events if event.rank is not None]
    if not ranks:
        return 0.0
    return sum(ranks) / len(ranks)


def lexicon_from_logs(
    events: list[LogEvent],
    encoder: TinyBiEncoder,
    *,
    min_sim: float,
    pool: int = 8,
) -> tuple[dict[str, list[str]], list[Reformulation], list[Reformulation]]:
    """Split reformulations into a lexicon and a suspect pile.

    Returns:
        A map of normalized query to extra terms, the reformulations that
        contributed, and the reformulations the encoder vetoed.
    """
    lexicon: dict[str, list[str]] = {}
    kept: list[Reformulation] = []
    suspect: list[Reformulation] = []
    for item in reformulations(events):
        accepted = _accepted_terms(item, encoder, min_sim=min_sim, pool=pool)
        if not accepted:
            suspect.append(item)
            continue
        kept.append(item)
        lexicon[" ".join(unique_terms(item.before))] = accepted
    return lexicon, kept, suspect


def _accepted_terms(
    item: Reformulation,
    encoder: TinyBiEncoder,
    *,
    min_sim: float,
    pool: int,
) -> list[str]:
    before = unique_terms(item.before)
    added = [term for term in unique_terms(item.after) if term not in set(before)]
    if not added:
        return []
    neighbors: set[str] = set()
    for term in before:
        for neighbor, sim in encoder.nearest(term, k=pool):
            if sim >= min_sim:
                neighbors.add(neighbor)
    return [term for term in added if term in neighbors]
