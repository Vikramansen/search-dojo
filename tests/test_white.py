"""The white belt's hand arithmetic has to agree with the index."""

import io
import math

import pytest
from rich.console import Console

from search_dojo.index import InvertedIndex
from search_dojo.minidb import load_mini
from search_dojo.tokenize import tokenize, trace_tokenize
from search_dojo.white import _norm, _query_vector, run_white


def _index() -> InvertedIndex:
    return InvertedIndex([(doc.id, doc.text) for doc in load_mini()])


def test_trace_matches_tokenize_on_every_document() -> None:
    for doc in load_mini():
        assert trace_tokenize(doc.text).stemmed == tokenize(doc.text), doc.id


def test_story_numbers() -> None:
    index = _index()
    assert index.n_docs == 9
    assert index.term_tf("csk-whistle", "whistle") == 4
    assert index.df["whistle"] == 1
    assert index.df["vijay"] == 2
    assert "thalapathy" not in index.df


def test_hand_cosine_matches_index() -> None:
    index = _index()
    terms = tokenize("vijay mass")
    query_vector = _query_vector(index, terms)
    q_norm = _norm(query_vector)
    expected = {hit.doc_id: hit.score for hit in index.tfidf_cosine("vijay mass")}
    for doc_id, score in expected.items():
        doc_vector = index.tfidf_vectors[doc_id]
        dot = sum(w * doc_vector.get(t, 0.0) for t, w in query_vector.items())
        assert dot / (q_norm * _norm(doc_vector)) == pytest.approx(score)
    assert max(expected, key=expected.__getitem__) == "vijay-ghilli"


def test_whistle_weight() -> None:
    index = _index()
    idf = math.log(10 / 2) + 1
    assert index.tfidf_vectors["csk-whistle"]["whistle"] == pytest.approx((1 + math.log(4)) * idf)


@pytest.mark.parametrize("query", [None, "rick pickle", "thalapathy", "the a"])
def test_lesson_runs_without_pausing(query: str | None) -> None:
    buffer = io.StringIO()
    run_white(Console(file=buffer, width=120), query=query, pause=True)
    output = buffer.getvalue()
    assert "Step 8/8" in output
    assert "[whistle, podu," in output
    if query is None:
        assert "vijay → [vijay-ghilli, vijay-leo]" in output
