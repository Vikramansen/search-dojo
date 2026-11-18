"""Search Dojo command line."""

from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from search_dojo.archive import RETRIEVERS, build_archive
from search_dojo.interact import load_progress
from search_dojo.lessons import BELT_BLURB, BELT_ORDER, run_lesson, show_boss, show_logs
from search_dojo.logsetup import configure_logging
from search_dojo.syllabus import show_syllabus
from search_dojo.view import hits_table

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    help="Search Dojo: sparse encoding, BM25, bi-encoders, hybrid retrieval, rerank, query expressions.",
)
console = Console()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Open the dojo. With no subcommand, show the menu."""
    configure_logging()
    if ctx.invoked_subcommand is not None:
        return
    belts()
    if not (console.is_interactive and sys.stdin.isatty()):
        return
    while True:
        choice = (
            Prompt.ask(
                "\nLesson number or name, [bold]s[/bold] for syllabus, [bold]q[/bold] to quit",
                default=_next_lesson(),
                console=console,
            )
            .strip()
            .lower()
        )
        if choice in ("q", "quit", "exit"):
            return
        if choice in ("s", "syllabus"):
            show_syllabus(console, BELT_ORDER)
            continue
        name = (
            BELT_ORDER[int(choice) - 1]
            if choice.isdigit() and 0 < int(choice) <= len(BELT_ORDER)
            else choice
        )
        try:
            run_lesson(console, name)
        except KeyError as exc:
            console.print(f"[red]{exc}[/red]")
            continue
        belts()


def _next_lesson() -> str:
    done = load_progress()
    return next((name for name in BELT_ORDER if name not in done), BELT_ORDER[0])


@app.command()
def belts() -> None:
    """List the lessons in order, with your progress."""
    configure_logging()
    console.print("[bold]SEARCH DOJO[/bold]  relevance is a contact sport\n")
    done = load_progress()
    table = Table(header_style="bold", title="Lessons")
    table.add_column("#", justify="right")
    table.add_column("lesson")
    table.add_column("what you learn")
    table.add_column("quiz", justify="right")
    for number, name in enumerate(BELT_ORDER, start=1):
        score = done.get(name)
        mark = f"[green]✓ {score['correct']}/{score['asked']}[/green]" if score else ""
        table.add_row(str(number), name, BELT_BLURB[name], mark)
    console.print(table)
    console.print(f"\nNext:  [bold]uv run search-dojo lesson {_next_lesson()}[/bold]")


@app.command()
def syllabus() -> None:
    """Map the role's topics to lessons, plus a reading list and case studies."""
    configure_logging()
    show_syllabus(console, BELT_ORDER)


@app.command()
def lesson(
    belt: str = typer.Argument(help="A lesson name from `search-dojo belts`."),
    k1: float | None = typer.Option(None, help="BM25 k1. Used by the yellow belt."),
    b: float | None = typer.Option(None, help="BM25 length penalty b. Used by the yellow belt."),
    query: str | None = typer.Option(None, help="Your own query for the white belt's step 7."),
    pause: bool = typer.Option(True, help="Wait for Enter and ask the quizzes (Dojo DB lessons)."),
) -> None:
    """Run one belt."""
    configure_logging()
    try:
        run_lesson(console, belt.strip().lower(), k1=k1, b=b, query=query, pause=pause)
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def ask(
    query: str = typer.Argument(help="What a searcher typed."),
    retriever: str = typer.Option(
        "pipeline",
        "--retriever",
        "-r",
        help="tf, tfidf, bm25, dense, hybrid, rerank, expand, pipeline, or expr.",
    ),
    k: int = typer.Option(5, "--k", help="How many documents to show."),
) -> None:
    """Rank documents for one query."""
    configure_logging()
    name = retriever.strip().lower()
    if name not in RETRIEVERS:
        known = ", ".join(RETRIEVERS)
        raise typer.BadParameter(f"unknown retriever {name!r}. choose one of: {known}")
    archive = build_archive()
    try:
        hits = archive.search(query, name, k=k)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    hits_table(console, hits, archive.by_id, title=f"{name}  ·  {query}", k=k)


@app.command()
def boss(
    k: int = typer.Option(5, "--k", help="Cutoff for NDCG and recall."),
) -> None:
    """Score every challenge. This is the exam."""
    configure_logging()
    show_boss(console, build_archive(), k=k)


@app.command()
def logs() -> None:
    """Show reformulations the encoder kept, and the ones it vetoed."""
    configure_logging()
    show_logs(console, build_archive())


if __name__ == "__main__":
    app()
