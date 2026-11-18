import numpy as np
from rich.console import Console

from search_dojo.clicks import ClickConfig, click_probability, simulate_log
from search_dojo.lesson_clicks import _ctr_by_rank, _estimates, spearman
from search_dojo.lessons import run_lesson
from search_dojo.world import ATTRACT, EXAMINE, build_world


def test_click_probability() -> None:
    assert click_probability(0, 3) == EXAMINE[0] * ATTRACT[3]
    assert click_probability(4, 0) < click_probability(0, 0)


def test_log_is_deterministic_and_sized() -> None:
    config = ClickConfig(sessions=300)
    first = simulate_log(build_world(), config)
    second = simulate_log(build_world(), config)
    assert len(first) == 300
    assert first == second


def test_randomized_bucket_recovers_position_bias() -> None:
    log = simulate_log(build_world(), ClickConfig(sessions=20000, random_share=0.5))
    ctr = _ctr_by_rank([session for session in log if session.randomized])
    theta = ctr / ctr[0]
    assert np.allclose(theta, EXAMINE, atol=0.06)


def test_ipw_is_less_biased_than_naive() -> None:
    log = simulate_log(build_world(), ClickConfig())
    production = [session for session in log if not session.randomized]
    naive, ipw, grade = _estimates(production, np.array(EXAMINE))
    naive_err = np.mean([abs(naive[key] - ATTRACT[grade[key]]) for key in naive])
    ipw_err = np.mean([abs(ipw[key] - ATTRACT[grade[key]]) for key in ipw])
    assert ipw_err < naive_err


def test_spearman_handles_ties() -> None:
    assert spearman(np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])) == 1.0
    assert spearman(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])) == 0.0


def test_clicks_lesson_runs() -> None:
    console = Console(record=True, width=120)
    run_lesson(console, "clicks", pause=False)
    assert "skip-above" in console.export_text()
