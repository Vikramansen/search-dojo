"""Query logs and clicks: traffic shape, position bias, IPW and click-derived pairs."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from search_dojo.clicks import ClickConfig, Session, simulate_log
from search_dojo.interact import Session as Steps
from search_dojo.interact import bar
from search_dojo.view import belt_panel
from search_dojo.world import ATTRACT, EXAMINE, GradedQuery, World, build_world

STEPS = 7
EXAMPLE_QUERY_ID = "q-city"
HEAD_SHARE = 0.2


def run_clicks(console: Console, *, pause: bool = True) -> None:
    """Run the clicks lesson."""
    config = ClickConfig()
    world = build_world()
    log = simulate_log(world, config)
    steps = Steps.start(console, "clicks", STEPS, pause)
    belt_panel(
        console,
        "clicks",
        "query logs and clicks",
        "Goal: learn what the query log and the click log can and cannot tell you.\n\n"
        "Human judgments are accurate but expensive: thousands of labels. Clicks are free and "
        "there are billions, but they are biased: people click what they see, and they see the "
        f"top. You will simulate {config.sessions:,} searches with a known click model, then try "
        "to recover the true relevance from the clicks alone.",
    )
    steps.step(1, "Query log analysis")
    _step_query_log(console, world, steps)
    steps.step(2, "The click log")
    _step_click_log(console, log)
    steps.step(3, "Click-through rate by position")
    _step_ctr(console, log, steps)
    steps.step(4, "Measuring position bias with randomization")
    theta = _step_randomization(console, log, config)
    steps.step(5, "Inverse propensity weighting")
    _step_ipw(console, world, log, theta, steps)
    steps.step(6, "Pairs from clicks: skip-above")
    _step_skip_above(console, log, steps)
    steps.step(7, "Click pitfalls")
    _step_pitfalls(console)
    steps.finish()


def _step_query_log(console: Console, world: World, steps: Steps) -> None:
    counts = world.query_counts()
    ordered = sorted(counts.items(), key=lambda item: -item[1])
    total = sum(counts.values())
    head_n = max(1, round(len(ordered) * HEAD_SHARE))
    head_share = sum(count for _, count in ordered[:head_n]) / total
    console.print(
        f"The query log has {total:,} searches over {len(ordered)} distinct queries. Search "
        "traffic is close to a [bold]Zipf distribution[/bold]: the r-th most common query gets "
        "about 1/r of the traffic of the first.\n"
        "  head = the few queries with most of the traffic: cache them, judge them, hand-tune them\n"
        "  tail = the many rare queries: typos, long questions, new names. Most failures live "
        "here, and each one is too rare to fix by hand."
    )
    table = Table(header_style="bold")
    table.add_column("rank", justify="right")
    table.add_column("query")
    table.add_column("searches", justify="right")
    table.add_column("")
    table.add_column("results", justify="right")
    top = ordered[0][1]
    zero_traffic = 0
    for rank, (query, count) in enumerate(ordered, start=1):
        results = len(world.index.bm25(query))
        if results == 0:
            zero_traffic += count
        style = "red" if results == 0 else ""
        cell = f"[{style}]{results}[/{style}]" if style else str(results)
        if rank <= 8 or results == 0:
            table.add_row(str(rank), escape(query), str(count), bar(count / top, 20), cell)
    console.print(table)
    console.print("(Rows 9 and below are shown only when they return zero results.)")
    steps.guess(
        f"What share of all searches do the top {head_n} queries ({HEAD_SHARE:.0%} of distinct "
        "queries) take? Answer as a fraction, e.g. 0.5.",
        head_share,
        0.1,
        f"{head_share:.0%}. A small head carries most of the traffic, so improving those queries "
        "moves the average a lot, while the tail needs general fixes (spelling, expansion, "
        "dense retrieval).",
    )
    console.print(
        f"[bold]Zero-result rate[/bold]: {zero_traffic / total:.1%} of searches return nothing. "
        "Each one is a user you failed for sure. Track it by query and fix the biggest ones: "
        "'csk comeback after ban' needs synonyms (lesson text), typos need spelling "
        "correction, and 'lebron james' or 'messi' is a content gap: no ranker can fix it."
    )
    console.print(
        "Other things to mine from the query log: reformulations (a query followed quickly by "
        "a different query means the first one failed), query length, new queries trending, "
        "and queries with high abandonment."
    )


def _step_click_log(console: Console, log: list[Session]) -> None:
    console.print(
        "Every session records the query, the documents shown in order, and which were "
        "clicked. Clicks here come from the [bold]position-based model[/bold] (PBM):\n"
        "  P(click on rank r) = P(examine rank r) × P(click | examined, relevance)\n"
        "The user looks at rank r with a probability that depends only on the position, and "
        "clicks an examined document with a probability that depends only on its relevance."
    )
    table = Table(header_style="bold", title="First 5 sessions (● = click)")
    table.add_column("query")
    for rank in range(1, len(EXAMINE) + 1):
        table.add_column(f"rank {rank}")
    for session in log[:5]:
        cells = [
            f"[bold green]● {doc_id}[/bold green]" if clicked else f"[dim]{doc_id}[/dim]"
            for doc_id, clicked in zip(session.shown, session.clicks, strict=True)
        ]
        cells += [""] * (len(EXAMINE) - len(cells))
        table.add_row(escape(session.query.text), *cells)
    console.print(table)


def _step_ctr(console: Console, log: list[Session], steps: Steps) -> None:
    production = [session for session in log if not session.randomized]
    ctr = _ctr_by_rank(production)
    table = Table(header_style="bold", title="Production traffic (BM25 order)")
    table.add_column("rank", justify="right")
    table.add_column("CTR", justify="right")
    table.add_column("")
    for rank, value in enumerate(ctr, start=1):
        table.add_row(str(rank), f"{value:.3f}", bar(value / max(ctr), 30))
    console.print(
        "CTR (click-through rate) at rank r = clicks at rank r / searches that showed rank r."
    )
    console.print(table)
    steps.quiz(
        "Rank 1 gets far more clicks than rank 5. What does that prove?",
        [
            "Rank-1 documents are more relevant",
            "Nothing alone: rank 1 is better and also seen more, and the two are mixed together",
            "Users dislike rank 5",
        ],
        1,
        "This is [bold]position bias[/bold]. In production traffic, position and relevance "
        "are confounded because the ranker already put better documents higher. Training on "
        "raw clicks teaches the new model to copy the old one.",
    )


def _step_randomization(console: Console, log: list[Session], config: ClickConfig) -> np.ndarray:
    randomized = [session for session in log if session.randomized]
    ctr = _ctr_by_rank(randomized)
    theta = ctr / ctr[0]
    console.print(
        f"{config.random_share:.0%} of traffic ({len(randomized):,} sessions) shuffles the top "
        f"{config.k} at random. In that bucket every document is equally likely at every rank, "
        "so relevance averages out and the CTR per rank measures the examination probability "
        "alone, up to a constant. Normalize by rank 1:\n"
        "  θ_r = CTR_r / CTR_1   (the propensity: how likely rank r is looked at, relative to rank 1)"
    )
    table = Table(header_style="bold", title="Estimated vs true examination probability")
    table.add_column("rank", justify="right")
    table.add_column("CTR in random bucket", justify="right")
    table.add_column("θ estimated", justify="right")
    table.add_column("θ true (simulator)", justify="right")
    for rank in range(config.k):
        table.add_row(
            str(rank + 1), f"{ctr[rank]:.3f}", f"{theta[rank]:.2f}", f"{EXAMINE[rank]:.2f}"
        )
    console.print(table)
    console.print(
        "Randomization costs some quality for those users, so production systems shuffle only "
        "a small share, only nearby positions (swap rank k and k+1), or estimate θ from "
        "natural variation, when the same document lands at different ranks over time, "
        "without explicit randomization."
    )
    return theta


def _step_ipw(
    console: Console, world: World, log: list[Session], theta: np.ndarray, steps: Steps
) -> None:
    production = [session for session in log if not session.randomized]
    naive, ipw, grade = _estimates(production, theta)
    console.print(
        "Goal: estimate how attractive (relevant) each (query, document) pair is from clicks.\n"
        "  naive estimate = clicks / impressions\n"
        "  IPW estimate   = Σ (click / θ_rank) / impressions\n"
        "[bold]Inverse propensity weighting[/bold] (IPW) counts a click at rank 4 as worth "
        "1/θ_4 clicks at rank 1, because only about θ_4 of users ever looked there. In "
        "expectation the IPW estimate equals P(click | examined), which depends only on "
        "relevance."
    )
    example = world.by_query[EXAMPLE_QUERY_ID]
    table = Table(header_style="bold", title=f"Query {example.text!r}")
    table.add_column("rank", justify="right")
    table.add_column("document", style="cyan")
    table.add_column("grade", justify="right")
    table.add_column("naive CTR", justify="right")
    table.add_column("IPW", justify="right")
    table.add_column("true P(click | examined)", justify="right")
    for rank, doc_id in enumerate(world.bm25_ids(example.text, len(EXAMINE)), start=1):
        key = (example.id, doc_id)
        table.add_row(
            str(rank),
            doc_id,
            str(example.grade(doc_id)),
            f"{naive[key]:.3f}",
            f"{ipw[key]:.3f}",
            f"{ATTRACT[example.grade(doc_id)]:.2f}",
        )
    console.print(table)
    rank_of = {
        (session.query.id, doc_id): rank
        for session in production
        for rank, doc_id in enumerate(session.shown)
    }
    errors = Table(
        header_style="bold",
        title="Mean absolute error vs the true P(click | examined), all queries",
    )
    errors.add_column("rank", justify="right")
    errors.add_column("pairs", justify="right")
    errors.add_column("naive error", justify="right")
    errors.add_column("IPW error", justify="right")
    for rank in range(len(EXAMINE)):
        keys = [key for key in naive if rank_of[key] == rank]
        truth = np.array([ATTRACT[grade[key]] for key in keys])
        naive_err = float(np.mean(np.abs(np.array([naive[key] for key in keys]) - truth)))
        ipw_err = float(np.mean(np.abs(np.array([ipw[key] for key in keys]) - truth)))
        errors.add_row(str(rank + 1), str(len(keys)), f"{naive_err:.3f}", f"{ipw_err:.3f}")
    console.print(errors)
    keys = sorted(naive)
    grades = np.array([grade[key] for key in keys], dtype=float)
    rho_naive = spearman(np.array([naive[key] for key in keys]), grades)
    rho_ipw = spearman(np.array([ipw[key] for key in keys]), grades)
    console.print(
        "Naive CTR underestimates every document below rank 1, more the lower it sits. IPW "
        "removes most of that bias; what is left is noise from few impressions and from the "
        "estimated θ.\n"
        f"Spearman rank correlation with the grades over all {len(keys)} pairs: naive "
        f"{rho_naive:.3f}, IPW {rho_ipw:.3f}. Close, because pooled ordering is dominated by "
        "the big gaps (grade 0 vs grade 3). The bias matters when the estimates become "
        "training labels: naive labels tell the new model that whatever sat at rank 1 is "
        "best, so it learns to copy the old ranker."
    )
    steps.quiz(
        "IPW divides by θ. What goes wrong when θ is tiny (rank 50)?",
        [
            "Nothing",
            "One lucky click gets a huge weight, so the estimate has high variance",
            "The estimate becomes biased toward 0",
        ],
        1,
        "Clip the weights (cap 1/θ, for example at 10) or only use the top ranks. Clipping "
        "trades a little bias for much lower variance.",
    )


def _step_skip_above(console: Console, log: list[Session], steps: Steps) -> None:
    production = [session for session in log if not session.randomized]
    skip_pairs: Counter[tuple[str, str, str]] = Counter()
    naive_pairs: Counter[tuple[str, str, str]] = Counter()
    for session in production:
        for rank, clicked in enumerate(session.clicks):
            if not clicked:
                continue
            for other, other_clicked in zip(session.shown, session.clicks, strict=True):
                if not other_clicked:
                    naive_pairs[(session.query.id, session.shown[rank], other)] += 1
            for above in range(rank):
                if not session.clicks[above]:
                    skip_pairs[(session.query.id, session.shown[rank], session.shown[above])] += 1
    grades = {session.query.id: session.query for session in production}
    rank_of = {
        (session.query.id, doc_id): rank
        for session in production
        for rank, doc_id in enumerate(session.shown)
    }
    restating = Counter(
        {
            key: count
            for key, count in naive_pairs.items()
            if rank_of[(key[0], key[1])] < rank_of[(key[0], key[2])]
        }
    )
    skip_acc = _pair_accuracy(skip_pairs, grades)
    naive_acc = _pair_accuracy(naive_pairs, grades)
    restating_share = sum(restating.values()) / max(sum(naive_pairs.values()), 1)
    console.print(
        "Pairwise LTR (lesson ltr) needs preferences, not absolute labels. Joachims' "
        "[bold]skip-above[/bold] rule: if the user clicked rank r and skipped a document above "
        "it, they probably looked at that document and preferred the clicked one.\n"
        "  click at rank r, no click at rank i < r   →   doc_r preferred over doc_i\n"
        "The naive rule, clicked beats every unclicked document, also uses documents below "
        "the click, which the user may never have seen."
    )
    table = Table(header_style="bold", title="Pairs that agree with the human grades")
    table.add_column("rule")
    table.add_column("pairs", justify="right")
    table.add_column("agree with grades", justify="right")
    table.add_column("")
    for name, pairs, accuracy in (
        ("skip-above", skip_pairs, skip_acc),
        ("clicked > all unclicked", naive_pairs, naive_acc),
    ):
        table.add_row(name, f"{sum(pairs.values()):,}", f"{accuracy:.1%}", bar(accuracy, 30))
    console.print(table)
    console.print(
        "Agreement counts only pairs whose grades differ.\n"
        f"The naive rule looks more accurate, but {restating_share:.0%} of its pairs say a "
        "higher-ranked document beats a lower-ranked one: they restate the current ranking, "
        "which the ranker already knows, and the unclicked lower documents were often never "
        "examined. Skip-above pairs are the ones that contradict the current order. They are "
        f"the only new information, and {skip_acc:.0%} of them are right; the rest are noise "
        "from users clicking a less relevant document."
    )
    steps.quiz(
        "Skip-above never produces the pair 'rank 1 beats rank 2'. Why is that a problem?",
        [
            "It is not a problem",
            "The training data only ever says 'move this up', so a model trained on it alone "
            "is biased toward reversing the old ranking",
            "It makes training slower",
        ],
        1,
        "Pair it with other signals (clicked-last, dwell time, randomized data) or with "
        "unbiased LTR methods that correct for position directly.",
    )


def _step_pitfalls(console: Console) -> None:
    table = Table(header_style="bold")
    table.add_column("pitfall")
    table.add_column("what happens")
    table.add_column("mitigation")
    table.add_row(
        "trust bias",
        "users click rank 1 even when rank 2 is better",
        "randomization, click models with trust",
    )
    table.add_row(
        "presentation bias",
        "images, snippets and bold text attract clicks",
        "features for presentation, consistent snippets",
    )
    table.add_row(
        "clicks ≠ satisfaction",
        "a click followed by a quick return is a failure",
        "dwell time, long clicks, pogo-sticking",
    )
    table.add_row(
        "good abandonment",
        "the answer was in the snippet, no click needed",
        "answer-box impressions as success",
    )
    table.add_row(
        "feedback loop",
        "unshown documents never get clicks, so never rise",
        "exploration, randomized buckets",
    )
    table.add_row(
        "bots and spam", "automated traffic inflates clicks", "filter by session behavior and IP"
    )
    console.print(table)
    console.print("[bold]Next[/bold]  uv run search-dojo lesson online")


def _ctr_by_rank(sessions: list[Session]) -> np.ndarray:
    clicks = np.zeros(len(EXAMINE))
    shown = np.zeros(len(EXAMINE))
    for session in sessions:
        for rank, clicked in enumerate(session.clicks):
            shown[rank] += 1
            clicks[rank] += clicked
    return clicks / np.maximum(shown, 1)


def _estimates(
    sessions: list[Session], theta: np.ndarray
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float], dict[tuple[str, str], int]]:
    impressions: defaultdict[tuple[str, str], int] = defaultdict(int)
    clicks: defaultdict[tuple[str, str], float] = defaultdict(float)
    weighted: defaultdict[tuple[str, str], float] = defaultdict(float)
    grade: dict[tuple[str, str], int] = {}
    for session in sessions:
        for rank, (doc_id, clicked) in enumerate(zip(session.shown, session.clicks, strict=True)):
            key = (session.query.id, doc_id)
            impressions[key] += 1
            clicks[key] += clicked
            weighted[key] += clicked / theta[rank]
            grade[key] = session.query.grade(doc_id)
    naive = {key: clicks[key] / impressions[key] for key in impressions}
    ipw = {key: weighted[key] / impressions[key] for key in impressions}
    return naive, ipw, grade


def _pair_accuracy(pairs: Counter[tuple[str, str, str]], queries: dict[str, GradedQuery]) -> float:
    agree = 0
    total = 0
    for (query_id, preferred, other), count in pairs.items():
        query = queries[query_id]
        if query.grade(preferred) == query.grade(other):
            continue
        total += count
        agree += count * (query.grade(preferred) > query.grade(other))
    return agree / total if total else 0.0


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2.0
        start = end + 1
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman rank correlation, with tied values sharing their average rank."""
    a = _average_ranks(left)
    b = _average_ranks(right)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
