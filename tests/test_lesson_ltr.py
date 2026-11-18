import numpy as np
from rich.console import Console

from search_dojo.lesson_ltr import (
    LtrConfig,
    _full_results,
    _ranks,
    build_dataset,
    fit_pairwise,
    proximity,
)
from search_dojo.lessons import run_lesson
from search_dojo.world import build_world


def test_proximity_window() -> None:
    positions = {"a": {"d": (0, 9)}, "b": {"d": (5,)}, "c": {"d": (6,)}}
    assert proximity(positions, ["b", "c"], "d") == 1.0
    assert proximity(positions, ["a", "b"], "d") == 2 / 5
    assert proximity(positions, ["a"], "d") == 0.0


def test_ranks_are_per_query() -> None:
    scores = np.array([0.1, 0.9, 0.5, 0.3, 0.2])
    qid = np.array([0, 0, 0, 1, 1])
    assert _ranks(scores, qid).tolist() == [2, 0, 1, 0, 1]


def test_ranknet_learns_a_separable_toy_problem() -> None:
    x = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]])
    y = np.array([2.0, 0.0, 1.0, 0.0])
    qid = np.array([0, 0, 1, 1])
    weights, history = fit_pairwise(x, y, qid, LtrConfig(), lambdas=True)
    assert weights[0] > weights[1]
    assert history[-1] < history[0]


def test_dataset_shapes() -> None:
    data = build_dataset(build_world(), LtrConfig())
    assert data.x.shape[0] == data.y.shape[0] == data.qid.shape[0]
    assert not np.isnan(data.x).any()


def test_lambdarank_beats_bm25_and_lesson_runs() -> None:
    results = _full_results()
    assert np.mean(results["LambdaRank"]) > np.mean(results["bm25 only"])
    console = Console(record=True, width=120)
    run_lesson(console, "ltr", pause=False)
    assert "leave-one-query-out" in console.export_text()
