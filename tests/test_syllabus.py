from rich.console import Console

from search_dojo.lessons import BELT_BLURB, BELT_ORDER, DOJO_LESSONS
from search_dojo.syllabus import load_syllabus, show_syllabus


def test_syllabus_only_names_real_lessons() -> None:
    syllabus = load_syllabus()
    known = set(BELT_ORDER)
    for topic in syllabus.role:
        assert set(topic.lessons) <= known
    assert {item.lesson for item in syllabus.reading} <= known
    for case in syllabus.cases:
        assert set(case.lessons) <= known


def test_every_lesson_is_listed_and_described() -> None:
    assert set(DOJO_LESSONS) <= set(BELT_ORDER)
    assert set(BELT_BLURB) == set(BELT_ORDER)


def test_syllabus_prints() -> None:
    console = Console(record=True, width=120)
    show_syllabus(console, BELT_ORDER)
    assert "LambdaMART" in console.export_text()
