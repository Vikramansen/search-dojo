import itertools
from collections import Counter

from rich.console import Console

from search_dojo.lesson_online import sample_size, srm_test, team_draft, z_test
from search_dojo.lessons import run_lesson


def test_z_test_symmetric_and_zero() -> None:
    diff, z, p = z_test(500, 1000, 500, 1000)
    assert diff == 0 and z == 0 and p == 1.0
    _, z_up, p_up = z_test(500, 1000, 560, 1000)
    assert z_up > 2 and p_up < 0.01


def test_sample_size_scales_with_inverse_square_mde() -> None:
    big = sample_size(0.5, 0.01, 0.05, 0.8)
    small = sample_size(0.5, 0.02, 0.05, 0.8)
    assert 3.9 < big / small < 4.1
    assert 39_000 < big < 40_000


def test_srm() -> None:
    chi2, p = srm_test((10_000, 10_000), 0.5)
    assert chi2 == 0 and p == 1.0
    assert srm_test((10_000, 9_700), 0.5)[1] < 0.05


def test_team_draft_is_fair_on_identical_lists() -> None:
    docs = ["a", "b", "c", "d", "e"]
    first_slot: Counter[str] = Counter()
    for flips in itertools.product((True, False), repeat=3):
        merged = team_draft(docs, docs, iter(flips).__next__, 5)
        assert [doc for doc, _ in merged] == docs
        first_slot[merged[0][1]] += 1
    assert first_slot["A"] == first_slot["B"]


def test_team_draft_credits_the_team_that_supplied_the_document() -> None:
    merged = team_draft(["a"], ["a", "b", "c"], iter([True, True]).__next__, 3)
    assert merged == [("a", "A"), ("b", "B"), ("c", "B")]


def test_online_lesson_runs() -> None:
    console = Console(record=True, width=120)
    run_lesson(console, "online", pause=False)
    text = console.export_text()
    assert "interleaving" in text and "χ²" in text
