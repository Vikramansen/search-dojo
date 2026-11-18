"""Evaluation: how to tell whether a ranking is good, worked by hand."""

from __future__ import annotations

import math

import numpy as np
from rich.console import Console
from rich.table import Table

from search_dojo.interact import Session
from search_dojo.metrics import (
    average_precision,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from search_dojo.view import belt_panel
from search_dojo.world import GradedQuery, World, build_world

QUERY_ID = "q-csk-titles"
K = 5
STEPS = 9
GRADE_MEANING = {3: "perfect", 2: "highly relevant", 1: "somewhat relevant", 0: "not relevant"}


def run_eval(console: Console, *, pause: bool = True) -> None:
    """Run the evaluation lesson."""
    world = build_world()
    query = world.by_query[QUERY_ID]
    ranking = world.bm25_ids(query.text, k=10)
    steps = Session.start(console, "eval", STEPS, pause)

    belt_panel(
        console,
        "eval",
        "offline evaluation",
        "Goal: put a number on ranking quality, so you can tell whether a change helped.\n\n"
        "You will compute precision, recall, MRR, average precision and NDCG by hand on one "
        "query, then compare two retrievers across 22 queries and check whether the difference "
        "is real or noise.",
    )
    steps.step(1, "Relevance judgments (qrels)")
    _step_judgments(console, world, query)
    steps.step(2, "The ranking to grade")
    _step_ranking(console, world, query, ranking)
    steps.step(3, "Precision@k and recall@k")
    _step_precision_recall(console, query, ranking, steps)
    steps.step(4, "Reciprocal rank and MRR")
    _step_mrr(console, world, query, ranking)
    steps.step(5, "Average precision and MAP")
    _step_ap(console, query, ranking)
    steps.step(6, "DCG and NDCG: graded relevance with position discount")
    _step_ndcg(console, world, query, ranking, steps)
    steps.step(7, "Comparing two retrievers across all queries")
    per_query = _step_compare(console, world)
    steps.step(8, "Is the difference real? Paired bootstrap")
    _step_bootstrap(console, per_query, steps)
    steps.step(9, "Offline metrics vs online metrics")
    _step_online(console)
    steps.finish()


def _step_judgments(console: Console, world: World, query: GradedQuery) -> None:
    console.print(
        "To score a ranking you need ground truth: for a query, which documents are relevant "
        "and how much. These labels are called [bold]relevance judgments[/bold], or "
        "[bold]qrels[/bold]. This database has 22 judged queries on a 0-3 scale:"
    )
    table = Table(header_style="bold")
    table.add_column("grade", justify="right")
    table.add_column("meaning")
    for grade, meaning in GRADE_MEANING.items():
        table.add_row(str(grade), meaning)
    console.print(table)
    console.print(f'Judgments for the query [bold]"{query.text}"[/bold]:')
    table = Table(header_style="bold")
    table.add_column("document", style="cyan")
    table.add_column("grade", justify="right")
    table.add_column("text")
    for doc_id, grade in sorted(query.qrels.items(), key=lambda item: -item[1]):
        table.add_row(doc_id, str(grade), world.by_id[doc_id].text)
    console.print(table)
    console.print(
        "Every document not listed is grade 0. Where judgments come from in practice:\n"
        "  • human raters following written guidelines (accurate, slow, expensive)\n"
        "  • LLM judges checked against a human-labeled sample (lesson llm)\n"
        "  • clicks and other behavior from query logs, after correcting for position bias "
        "(lesson clicks)"
    )


def _step_ranking(console: Console, world: World, query: GradedQuery, ranking: list[str]) -> None:
    console.print(
        f'BM25 returned this top 10 for "{query.text}". The grade column comes from the qrels. '
        "Every metric below is computed from this table only."
    )
    _ranking_table(console, world, query, ranking)
    console.print(
        "Note: csk is not a term in any document. BM25 matched on ipl and title only. "
        "The best document, csk-five-titles, landed at rank 5."
    )


def _step_precision_recall(
    console: Console, query: GradedQuery, ranking: list[str], steps: Session
) -> None:
    relevant_total = sum(1 for grade in query.qrels.values() if grade > 0)
    console.print(
        "For these two metrics a document is either relevant (grade ≥ 1) or not.\n"
        "[bold]precision@k[/bold] = (relevant documents in the top k) / k\n"
        "  Of what I showed, how much was useful?\n"
        "[bold]recall@k[/bold] = (relevant documents in the top k) / (all relevant documents)\n"
        "  Of everything useful, how much did I show?"
    )
    top3 = sum(1 for doc_id in ranking[:3] if query.grade(doc_id) > 0)
    steps.guess(
        "What is precision@3 for this ranking?",
        precision_at_k(ranking, query.qrels, 3),
        0.01,
        f"{top3} of the top 3 are relevant: {top3}/3.",
    )
    table = Table(header_style="bold")
    table.add_column("k", justify="right")
    table.add_column("relevant in top k", justify="right")
    table.add_column("precision@k", justify="right")
    table.add_column("recall@k", justify="right")
    for k in (1, 3, 5, 10):
        found = sum(1 for doc_id in ranking[:k] if query.grade(doc_id) > 0)
        table.add_row(
            str(k),
            str(found),
            f"{found}/{k} = {precision_at_k(ranking, query.qrels, k):.2f}",
            f"{found}/{relevant_total} = {recall_at_k(ranking, query.qrels, k):.2f}",
        )
    console.print(table)
    console.print(
        "As k grows, recall can only go up and precision usually goes down. Recall matters most "
        "for a first-stage retriever: whatever it misses, no reranker can bring back. "
        "Precision at small k matters most for what the user actually sees."
    )


def _step_mrr(console: Console, world: World, query: GradedQuery, ranking: list[str]) -> None:
    console.print(
        "[bold]reciprocal rank (RR)[/bold] = 1 / (rank of the first relevant document), "
        "or 0 if none is retrieved.\n"
        "[bold]MRR[/bold] = the mean of RR over all queries.\n"
        "It only cares about the first good result, which fits navigational queries and "
        "question answering, where one right answer is enough."
    )
    rows = []
    for query_id in (QUERY_ID, "q-city", "q-csk-ban"):
        item = world.by_query[query_id]
        ranked = world.bm25_ids(item.text, k=10)
        first = next((i for i, d in enumerate(ranked, 1) if item.grade(d) > 0), None)
        shown = ranked[0] if ranked else "(no results)"
        rr = mrr(ranked, item.qrels)
        rule = f"1/{first} = {rr:.2f}" if first else "0 (nothing relevant retrieved)"
        rows.append([item.text, shown, str(first or "-"), rule])
    table = Table(header_style="bold")
    for column in ("query", "rank 1", "first relevant at", "RR"):
        table.add_column(column)
    for row in rows:
        table.add_row(*row)
    console.print(table)
    console.print(
        'For "city", BM25 put the Citadel of Ricks first: it is a short document where city '
        "is a large share of the text. The Manchester City documents start at rank 2, so RR "
        "is 0.5. For the ban query BM25 returns nothing, so RR is 0."
    )


def _step_ap(console: Console, query: GradedQuery, ranking: list[str]) -> None:
    relevant_total = sum(1 for grade in query.qrels.values() if grade > 0)
    console.print(
        "[bold]average precision (AP)[/bold] = the mean of precision@i taken at every rank i "
        "that holds a relevant document, divided by the total number of relevant documents. "
        "It rewards putting all the relevant documents early, not just the first one.\n"
        "[bold]MAP[/bold] = the mean of AP over all queries."
    )
    table = Table(header_style="bold")
    table.add_column("rank i", justify="right")
    table.add_column("document", style="cyan")
    table.add_column("relevant?")
    table.add_column("precision@i at this rank", justify="right")
    hits = 0
    parts: list[str] = []
    for i, doc_id in enumerate(ranking, start=1):
        if query.grade(doc_id) > 0:
            hits += 1
            parts.append(f"{hits}/{i}")
            table.add_row(str(i), doc_id, "yes", f"{hits}/{i} = {hits / i:.2f}")
        else:
            table.add_row(str(i), doc_id, "[dim]no[/dim]", "[dim]skipped[/dim]")
    console.print(table)
    ap = average_precision(ranking, query.qrels)
    console.print(
        f"AP = ({' + '.join(parts)}) / {relevant_total} = [bold]{ap:.3f}[/bold]\n"
        "Relevant documents that never appear still count in the denominator, so missing them "
        "lowers AP."
    )


def _step_ndcg(
    console: Console, world: World, query: GradedQuery, ranking: list[str], steps: Session
) -> None:
    console.print(
        "Precision, recall, RR and AP treat grade 1 and grade 3 the same. NDCG uses the grades.\n"
        "[bold]gain(d) = 2^grade − 1[/bold]   grade 3 → 7, grade 2 → 3, grade 1 → 1, grade 0 → 0. "
        "A perfect document is worth much more than a somewhat relevant one.\n"
        "[bold]discount(i) = 1 / log2(i + 1)[/bold]   rank 1 → 1.00, rank 2 → 0.63, "
        "rank 5 → 0.39. Users look at the top, so lower ranks count less.\n"
        "[bold]DCG@k[/bold] = sum over the top k of gain × discount.\n"
        "[bold]IDCG@k[/bold] = DCG@k of the best possible ordering of the judged documents.\n"
        "[bold]NDCG@k = DCG@k / IDCG@k[/bold], so 1.0 means perfect at every rank."
    )
    dcg = _dcg_table(console, "BM25 ranking", ranking[:K], query)
    ideal = sorted(query.qrels, key=lambda doc_id: -query.qrels[doc_id])[:K]
    idcg = _dcg_table(console, "ideal ranking", ideal, query)
    console.print(f"NDCG@{K} = {dcg:.3f} / {idcg:.3f} = [bold]{dcg / idcg:.3f}[/bold]")

    options = {
        "move csk-five-titles from rank 5 to rank 1": _moved(ranking, "csk-five-titles", 0),
        "swap ranks 2 and 3": _swapped(ranking, 1, 2),
        "drop warriors-dynasty (rank 4) out of the list": [
            d for d in ranking if d != "warriors-dynasty"
        ],
    }
    scores = {label: ndcg_at_k(order, query.qrels, K) for label, order in options.items()}
    labels = list(options)
    best = max(range(len(labels)), key=lambda i: scores[labels[i]])
    steps.quiz(
        "Which single change raises NDCG@5 the most?",
        labels,
        best,
        "  ".join(f"{label}: {scores[label]:.3f}." for label in labels)
        + " Putting the grade-3 document at rank 1 multiplies its gain of 7 by a discount of "
        "1.0 instead of 0.39.",
    )


def _step_compare(console: Console, world: World) -> dict[str, tuple[float, float]]:
    console.print(
        "One query proves nothing. Real evaluation runs the whole judged set and averages. "
        "Here are BM25 and TF-IDF cosine (white belt) on all 22 queries, cutoff k = 5."
    )
    per_query: dict[str, tuple[float, float]] = {}
    sums = {name: [0.0] * 5 for name in ("bm25", "tfidf")}
    for item in world.queries:
        runs = {"bm25": world.bm25_ids(item.text, 10), "tfidf": world.tfidf_ids(item.text, 10)}
        for name, ranked in runs.items():
            values = (
                precision_at_k(ranked, item.qrels, K),
                recall_at_k(ranked, item.qrels, K),
                mrr(ranked, item.qrels),
                average_precision(ranked, item.qrels),
                ndcg_at_k(ranked, item.qrels, K),
            )
            sums[name] = [total + value for total, value in zip(sums[name], values, strict=True)]
        per_query[item.text] = (
            ndcg_at_k(runs["bm25"], item.qrels, K),
            ndcg_at_k(runs["tfidf"], item.qrels, K),
        )
    count = len(world.queries)
    table = Table(header_style="bold", title="Mean over 22 queries")
    for column in ("retriever", "P@5", "R@5", "MRR", "MAP", "NDCG@5"):
        table.add_column(column, justify="right")
    for name, totals in sums.items():
        table.add_row(name, *(f"{value / count:.3f}" for value in totals))
    console.print(table)
    table = Table(header_style="bold", title="Queries where NDCG@5 differs")
    for column in ("query", "bm25", "tfidf", "difference"):
        table.add_column(column, justify="right")
    for text, (bm25, tfidf) in per_query.items():
        if abs(bm25 - tfidf) > 1e-9:
            table.add_row(text, f"{bm25:.3f}", f"{tfidf:.3f}", f"{bm25 - tfidf:+.3f}")
    console.print(table)
    console.print(
        "The two retrievers agree on most queries. The mean hides that: always look at the "
        "per-query differences, and especially at the queries that got worse."
    )
    return per_query


def _step_bootstrap(
    console: Console, per_query: dict[str, tuple[float, float]], steps: Session
) -> None:
    diffs = np.array([bm25 - tfidf for bm25, tfidf in per_query.values()])
    wins = int((diffs > 1e-9).sum())
    losses = int((diffs < -1e-9).sum())
    ties = len(diffs) - wins - losses
    rng = np.random.default_rng(11)
    samples = rng.integers(0, len(diffs), size=(5000, len(diffs)))
    means = diffs[samples].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    console.print(
        "A mean difference on 22 queries could be luck: a different 22 queries might flip it. "
        "The [bold]paired bootstrap[/bold] estimates how much the mean would move:\n"
        "  1. Take the per-query differences (bm25 − tfidf). Paired: same query, both systems.\n"
        "  2. Resample 22 of them with replacement, 5,000 times, and record each mean.\n"
        "  3. The middle 95% of those means is a 95% confidence interval."
    )
    console.print(
        f"Wins / ties / losses for BM25: {wins} / {ties} / {losses}\n"
        f"Mean difference: {diffs.mean():+.3f}   95% interval: [{low:+.3f}, {high:+.3f}]"
    )
    crosses = low <= 0 <= high
    steps.quiz(
        "The interval is shown above. Can you claim BM25 beats TF-IDF on this query set?",
        ["Yes, the mean difference is positive", "No, the interval includes 0"],
        1 if crosses else 0,
        (
            "If the 95% interval contains 0, the data is consistent with no difference. "
            "With few queries and few disagreements, you need more judged queries before claiming a win."
            if crosses
            else "The whole interval is above 0, so the difference is unlikely to be noise."
        ),
    )


def _step_online(console: Console) -> None:
    table = Table(header_style="bold", title="Online metrics, measured on live traffic")
    table.add_column("metric")
    table.add_column("what it measures")
    table.add_column("watch out for")
    table.add_row(
        "click-through rate", "searches with at least one click", "position bias, clickbait"
    )
    table.add_row(
        "success rate / long clicks",
        "clicks with dwell time over a threshold (e.g. 30 s)",
        "threshold choice varies by content type",
    )
    table.add_row(
        "abandonment", "searches with no click", "good abandonment: answer shown on the page"
    )
    table.add_row(
        "reformulation rate",
        "searches followed by a rewritten query",
        "some rewrites are exploration",
    )
    table.add_row(
        "time to first click", "how fast users find something", "fast is not always satisfied"
    )
    table.add_row("zero-result rate", "searches returning nothing", "a guardrail, not a goal")
    console.print(table)
    console.print(
        "Offline metrics are cheap and repeatable, but they only know the judged documents and "
        "the queries you sampled. Online metrics see real users but are noisy, slow and biased by "
        "what you showed. The standard loop: iterate offline, ship the winners to an A/B test "
        "or interleaving experiment (lesson online), and check that offline gains predicted "
        "online gains.\n\n"
        "[bold]Next[/bold]  uv run search-dojo lesson green"
    )


def _ranking_table(console: Console, world: World, query: GradedQuery, ranking: list[str]) -> None:
    table = Table(header_style="bold")
    table.add_column("rank", justify="right")
    table.add_column("document", style="cyan")
    table.add_column("grade", justify="right")
    table.add_column("text")
    for rank, doc_id in enumerate(ranking, start=1):
        grade = query.grade(doc_id)
        style = "bold" if grade >= 2 else ("" if grade else "dim")
        table.add_row(
            str(rank),
            doc_id,
            f"[{style}]{grade}[/{style}]" if style else str(grade),
            world.by_id[doc_id].text,
        )
    console.print(table)


def _dcg_table(console: Console, title: str, order: list[str], query: GradedQuery) -> float:
    table = Table(header_style="bold", title=title)
    table.add_column("rank i", justify="right")
    table.add_column("document", style="cyan", no_wrap=True)
    for column in ("grade", "gain 2^g−1", "discount 1/log2(i+1)", "gain × discount"):
        table.add_column(column, justify="right")
    total = 0.0
    for i, doc_id in enumerate(order, start=1):
        grade = query.grade(doc_id)
        gain = 2**grade - 1
        discount = 1 / math.log2(i + 1)
        total += gain * discount
        table.add_row(
            str(i), doc_id, str(grade), str(gain), f"{discount:.3f}", f"{gain * discount:.3f}"
        )
    table.add_row("", "", "", "", "[bold]sum[/bold]", f"[bold]{total:.3f}[/bold]")
    console.print(table)
    return total


def _moved(ranking: list[str], doc_id: str, position: int) -> list[str]:
    rest = [d for d in ranking if d != doc_id]
    return rest[:position] + [doc_id] + rest[position:]


def _swapped(ranking: list[str], i: int, j: int) -> list[str]:
    order = list(ranking)
    order[i], order[j] = order[j], order[i]
    return order
