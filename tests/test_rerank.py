"""Feature combination after min-max."""

from search_dojo.rerank import combine, normalize_rows


def test_minmax_and_weights() -> None:
    rows = [
        {"bm25": 10.0, "unique_ratio": 0.2},
        {"bm25": 0.0, "unique_ratio": 0.9},
    ]
    scaled = normalize_rows(rows)
    assert scaled[0]["bm25"] == 1.0
    assert scaled[1]["unique_ratio"] == 1.0
    scores = combine(scaled, {"bm25": 0.2, "unique_ratio": 0.8})
    assert scores[1] > scores[0]


def test_constant_feature_contributes_nothing() -> None:
    scaled = normalize_rows([{"bm25": 3.0}, {"bm25": 3.0}])
    assert scaled[0]["bm25"] == 0.0
    assert scaled[1]["bm25"] == 0.0
