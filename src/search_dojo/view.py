"""Rich tables shared by the belts."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from search_dojo.corpus import Scroll
from search_dojo.types import Hit

BELT_STYLE = {
    "white": "white",
    "yellow": "yellow",
    "green": "green",
    "blue": "bright_blue",
    "brown": "rgb(205,133,63)",
    "black": "bright_white",
    "sensei": "gold1",
}


def belt_panel(console: Console, belt: str, title: str, body: str) -> None:
    """Open a lesson with a colored panel."""
    style = BELT_STYLE.get(belt, "white")
    console.print(
        Panel(Text(body, justify="left"), title=f"{belt} belt  ·  {title}", border_style=style)
    )


def hits_table(
    console: Console,
    hits: list[Hit],
    by_id: dict[str, Scroll],
    *,
    title: str,
    k: int = 5,
) -> None:
    """Print a ranking."""
    table = Table(title=title, header_style="bold", show_lines=False)
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("score", justify="right", width=8)
    table.add_column("document")
    table.add_column("why")
    for rank, hit in enumerate(hits[:k], start=1):
        document = by_id.get(hit.doc_id)
        name = document.title if document else hit.doc_id
        table.add_row(str(rank), f"{hit.score:.3f}", name, hit.reason)
    if not hits:
        table.add_row("-", "-", "(no hit)", "no document scored above zero")
    console.print(table)


def simple_table(console: Console, title: str, columns: list[str], rows: list[list[str]]) -> None:
    """Print a small table of strings."""
    table = Table(title=title, header_style="bold")
    for column in columns:
        table.add_column(column)
    for row in rows:
        table.add_row(*row)
    console.print(table)
