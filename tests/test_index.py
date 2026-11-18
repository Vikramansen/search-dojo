"""BM25 length, saturation, idf, and sparse cosine."""

import pytest

from search_dojo.index import InvertedIndex


def test_length_penalty_only_bites_when_b_is_on() -> None:
    index = InvertedIndex(
        [
            ("short", "dashi pot"),
            ("long", "dashi " + "water " * 30),
        ]
    )
    short_b0, _ = index.explain_bm25("dashi", "short", k1=1.2, b=0.0)
    long_b0, _ = index.explain_bm25("dashi", "long", k1=1.2, b=0.0)
    short_b1, _ = index.explain_bm25("dashi", "short", k1=1.2, b=1.0)
    long_b1, _ = index.explain_bm25("dashi", "long", k1=1.2, b=1.0)
    assert short_b0 == pytest.approx(long_b0)
    assert short_b1 > long_b1


def test_saturation_bends_repetition() -> None:
    index = InvertedIndex(
        [
            ("once", "dashi"),
            ("spam", " ".join(["dashi"] * 20)),
        ]
    )
    once, _ = index.explain_bm25("dashi", "once", k1=1.2, b=0.0)
    spam, _ = index.explain_bm25("dashi", "spam", k1=1.2, b=0.0)
    assert spam > once
    assert spam / once < 3


def test_rarer_term_contributes_more_at_the_same_tf() -> None:
    index = InvertedIndex(
        [
            ("rare", "katsuobushi"),
            ("soup-a", "soup"),
            ("soup-b", "soup"),
            ("soup-c", "soup"),
            ("both", "katsuobushi soup"),
        ]
    )
    _, parts = index.explain_bm25("katsuobushi soup", "both", k1=1.2, b=0.0)
    by_term = {part.term: part.contribution for part in parts}
    assert by_term["katsuobushi"] > by_term["soup"]


def test_explain_sums_to_the_score() -> None:
    index = InvertedIndex([("doc", "okapi ranking function")])
    score, parts = index.explain_bm25("okapi ranking", "doc")
    assert score == pytest.approx(sum(part.contribution for part in parts))


def test_identical_tfidf_vectors_have_cosine_one() -> None:
    index = InvertedIndex([("a", "opaque broth"), ("b", "opaque broth")])
    hits = index.tfidf_cosine("opaque broth")
    assert hits[0].score == pytest.approx(1.0)
    assert {hit.doc_id for hit in hits} == {"a", "b"}
