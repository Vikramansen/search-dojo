"""Three ways to say the query again, plus a socket for a language model.

Default expansion runs only when the top BM25 hit covers the question
poorly. A strong lexical hit is left alone. Pseudo-relevance feedback is
implemented and deliberately not on that default path: if the first pass is
wrong, feedback copies the mistake.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from pathlib import Path

from search_dojo.biencoder import TinyBiEncoder
from search_dojo.index import InvertedIndex
from search_dojo.tokenize import is_identifier, unique_terms


def should_expand(index: InvertedIndex, query: str, gate: float) -> bool:
    """True when the best BM25 hit contains only a slice of the query terms."""
    hits = index.bm25(query, k=1)
    if not hits:
        return True
    return index.coverage(query, hits[0].doc_id) < gate


def prf_terms(
    index: InvertedIndex,
    query: str,
    *,
    k_docs: int = 3,
    k_terms: int = 4,
) -> list[str]:
    """Distinctive terms from the top BM25 hits, scored by smooth idf.

    Query terms are not repeated. High-idf words from a bad top hit are how
    feedback poisons the second pass.
    """
    hits = index.bm25(query, k=k_docs)
    query_terms = set(unique_terms(query))
    scores: Counter[str] = Counter()
    for hit in hits:
        for term in index.doc_token_set.get(hit.doc_id, set()):
            if term in query_terms or is_identifier(term):
                continue
            scores[term] += index.term_tf(hit.doc_id, term) * index.smooth_idf(term)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [term for term, _score in ranked[:k_terms]]


def neighbor_terms(
    encoder: TinyBiEncoder,
    index: InvertedIndex,
    query: str,
    *,
    min_sim: float,
    pool: int,
    limit: int,
) -> list[str]:
    """Rare in-vocabulary neighbors of the query terms.

    Similarity times idf prefers ``tonkotsu`` over a common word that happens
    to sit nearby. Terms already in the query are skipped.
    """
    query_terms = unique_terms(query)
    query_set = set(query_terms)
    chosen: list[str] = []
    for term in query_terms:
        best: tuple[float, str] | None = None
        for neighbor, sim in encoder.nearest(term, k=pool):
            if neighbor in query_set or neighbor not in index.df or sim < min_sim:
                continue
            score = sim * (1.0 + index.bm25_idf(neighbor))
            if best is None or score > best[0]:
                best = (score, neighbor)
        if best is not None and best[1] not in chosen:
            chosen.append(best[1])
        if len(chosen) >= limit:
            break
    return chosen[:limit]


def expand_query(
    query: str,
    *,
    index: InvertedIndex,
    encoder: TinyBiEncoder,
    lexicon: dict[str, list[str]],
    gate: float,
    min_sim: float,
    pool: int,
    limit: int,
) -> str:
    """Return the query plus a few extra terms, or the query unchanged.

    ``lexicon`` is built from query logs. Neighbor terms come from the
    bi-encoder. Feedback terms are not added here.
    """
    if not should_expand(index, query, gate):
        return query
    extras: list[str] = []
    key = " ".join(unique_terms(query))
    extras.extend(lexicon.get(key, []))
    extras.extend(neighbor_terms(encoder, index, query, min_sim=min_sim, pool=pool, limit=limit))
    query_set = set(unique_terms(query))
    cleaned: list[str] = []
    for term in extras:
        if term in query_set or term in cleaned or is_identifier(term):
            continue
        cleaned.append(term)
        if len(cleaned) == limit:
            break
    if not cleaned:
        return query
    return f"{query} {' '.join(cleaned)}"


def llm_rewrite(query: str, complete: Callable[[str], str], prompt_path: Path) -> str:
    """Rewrite ``query`` with a completion function.

    ``complete`` receives the prompt in ``prompts/rewrite.txt`` and returns
    the rewritten line. The dojo does not ship a network client. Pass one in
    when you want a real model in this socket. Tests pass a lambda.
    """
    template = prompt_path.read_text(encoding="utf-8")
    prompt = template.replace("{query}", query)
    rewritten = complete(prompt).strip()
    if not rewritten:
        raise ValueError("rewriter returned an empty query")
    return rewritten
