"""The syllabus: role topics mapped to lessons, a reading list and case studies."""

from __future__ import annotations

import json
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from search_dojo.corpus import DATA_DIR
from search_dojo.interact import load_progress


@dataclass(frozen=True)
class RoleTopic:
    topic: str
    lessons: tuple[str, ...]


@dataclass(frozen=True)
class Reading:
    lesson: str
    title: str
    authors: str
    year: int


@dataclass(frozen=True)
class Case:
    name: str
    year: int
    lessons: tuple[str, ...]
    takeaway: str


@dataclass(frozen=True)
class Syllabus:
    role: tuple[RoleTopic, ...]
    reading: tuple[Reading, ...]
    cases: tuple[Case, ...]


def load_syllabus() -> Syllabus:
    """Read data/syllabus.json."""
    data = json.loads((DATA_DIR / "syllabus.json").read_text(encoding="utf-8"))
    return Syllabus(
        role=tuple(RoleTopic(row["topic"], tuple(row["lessons"])) for row in data["role"]),
        reading=tuple(
            Reading(row["lesson"], row["title"], row["authors"], int(row["year"]))
            for row in data["reading"]
        ),
        cases=tuple(
            Case(row["name"], int(row["year"]), tuple(row["lessons"]), row["takeaway"])
            for row in data["cases"]
        ),
    )


def show_syllabus(console: Console, order: tuple[str, ...]) -> None:
    """Print role coverage, the reading list and case studies."""
    syllabus = load_syllabus()
    done = load_progress()
    table = Table(
        header_style="bold", title="What the role asks for, and where the dojo teaches it"
    )
    table.add_column("topic")
    table.add_column("lessons")
    for row in syllabus.role:
        lessons = sorted(row.lessons, key=order.index)
        cells = [f"[green]{name}✓[/green]" if name in done else name for name in lessons]
        table.add_row(row.topic, ", ".join(cells))
    console.print(table)

    reading = Table(header_style="bold", title="Reading list, in lesson order")
    reading.add_column("lesson")
    reading.add_column("paper or book")
    reading.add_column("authors")
    reading.add_column("year", justify="right")
    for item in sorted(syllabus.reading, key=lambda entry: order.index(entry.lesson)):
        reading.add_row(item.lesson, item.title, item.authors, str(item.year))
    console.print(reading)

    cases = Table(header_style="bold", title="Case studies to know for interviews")
    cases.add_column("case")
    cases.add_column("year", justify="right")
    cases.add_column("lessons")
    cases.add_column("takeaway")
    for case in syllabus.cases:
        cases.add_row(case.name, str(case.year), ", ".join(case.lessons), case.takeaway)
    console.print(cases)
    console.print(
        "Search the titles to read them. For each case study, be ready to say: the problem, "
        "the metric, the model change, how they evaluated it, and what surprised them."
    )
