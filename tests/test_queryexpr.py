"""Boolean trees over a tiny index."""

import pytest

from search_dojo.index import InvertedIndex
from search_dojo.queryexpr import QuerySyntaxError, format_expr, match_expr, parse_query


def _index() -> InvertedIndex:
    return InvertedIndex(
        [
            ("broth", "opaque white broth from pork bones"),
            ("chant", "milky pork bone soup milky pork bone soup"),
            ("animal", "shy forest animal"),
            ("formula", "okapi ranking function"),
        ]
    )


def test_phrase_and_term() -> None:
    index = _index()
    expr = parse_query('"opaque white broth" AND pork')
    assert "AND" in format_expr(expr)
    assert match_expr(index, expr) == {"broth"}


def test_not_removes_the_forest() -> None:
    index = _index()
    matched = match_expr(index, parse_query("okapi AND NOT forest"))
    assert matched == {"formula"}
    matched = match_expr(index, parse_query("animal OR function"))
    assert matched == {"animal", "formula"}


def test_bad_syntax_raises() -> None:
    with pytest.raises(QuerySyntaxError):
        parse_query("okapi AND")
    with pytest.raises(QuerySyntaxError):
        parse_query("")
