from rich.console import Console

from search_dojo.lesson_llm import (
    QUESTION,
    cohen_kappa,
    extractive_answer,
    load_judgments,
    render,
    unsupported_facts,
)
from search_dojo.lessons import run_lesson
from search_dojo.world import build_world


def test_kappa() -> None:
    assert cohen_kappa([(0, 0), (1, 1), (2, 2), (3, 3)])[2] == 1.0
    observed, chance, kappa = cohen_kappa([(0, 0), (0, 1), (1, 0), (1, 1)])
    assert observed == 0.5 and chance == 0.5 and kappa == 0.0


def test_prompts_render() -> None:
    assert "csk" in render("rewrite", query="csk")
    assert "[1] doc" in render("rag_answer", sources="[1] doc", question="q")
    assert "Query: q" in render("judge", query="q", document="d")


def test_grounding_uses_the_cited_source_only() -> None:
    sources = ["They won in 2018.", "He hit a four."]
    assert unsupported_facts("They won four times [1].", sources) == ["four"]
    assert unsupported_facts("They won in 2018 [1].", sources) == []


def test_rag_answer_depends_on_retrieval() -> None:
    world = build_world()
    good = extractive_answer(world, QUESTION, ["csk-five-titles", "csk-comeback"])
    assert "five times" in good and good.endswith("[1].")
    assert extractive_answer(world, QUESTION, ["rick-pickle"]) == "The sources do not say."


def test_judgments_reference_known_pairs() -> None:
    judgments = load_judgments(build_world())
    assert len(judgments) > 50
    assert all(j.llm in (0, 1, 2, 3) for j in judgments)


def test_llm_lesson_runs() -> None:
    console = Console(record=True, width=120)
    run_lesson(console, "llm", pause=False)
    text = console.export_text()
    assert "κ =" in text and "four" in text
