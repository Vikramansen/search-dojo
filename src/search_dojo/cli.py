"""Search Dojo command line."""

from __future__ import annotations

import typer
from rich.console import Console

from search_dojo.archive import RETRIEVERS, build_archive
from search_dojo.lessons import BELT_BLURB, BELT_ORDER, run_lesson, show_boss, show_logs
from search_dojo.logsetup import configure_logging
from search_dojo.view import hits_table, simple_table

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    help="Search Dojo: sparse encoding, BM25, bi-encoders, hybrid retrieval, rerank, query expressions.",
)
console = Console()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Open the dojo. With no subcommand, list the belts."""
    configure_logging()
    if ctx.invoked_subcommand is None:
        belts()


@app.command()
def belts() -> None:
    """List the belts in order."""
    configure_logging()
    console.print("[bold]SEARCH DOJO[/bold]  relevance is a contact sport\n")
    simple_table(
        console,
        "Belts",
        ["belt", "lesson"],
        [[belt, BELT_BLURB[belt]] for belt in BELT_ORDER],
    )
    console.print("\nNext:  [bold]uv run search-dojo lesson white[/bold]")


@app.command()
def lesson(
    belt: str = typer.Argument(help="white, yellow, green, blue, brown, black, or sensei."),
    k1: float | None = typer.Option(None, help="BM25 k1. Used by the yellow belt."),
    b: float | None = typer.Option(None, help="BM25 length penalty b. Used by the yellow belt."),
) -> None:
    """Run one belt against the Midnight Archive."""
    configure_logging()
    archive = build_archive()
    try:
        run_lesson(console, archive, belt.strip().lower(), k1=k1, b=b)
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
    k: int = typer.Option(5, "--k", help="How many scrolls to show."),
) -> None:
    """Rank scrolls for one question."""
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
