"""Online evaluation: A/B tests, sample size, peeking, SRM and interleaving."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from rich.console import Console
from rich.table import Table

from search_dojo.clicks import click_probability, traffic
from search_dojo.interact import Session, bar
from search_dojo.lesson_ltr import trained_rankings
from search_dojo.view import belt_panel
from search_dojo.world import EXAMINE, GradedQuery, World, build_world

STEPS = 7
LONG_CLICK_GRADE = 2
EXAMPLE_QUERY_ID = "q-city"


@dataclass(frozen=True)
class OnlineConfig:
    """Sizes and seeds for the online simulations."""

    ab_sessions_per_arm: int = 20000
    alpha: float = 0.05
    power: float = 0.80
    aa_experiments: int = 400
    aa_sessions_per_arm: int = 4000
    peeks: int = 20
    srm_expected_share: float = 0.5
    srm_counts: tuple[int, int] = (10_000, 9_700)
    sensitivity_sizes: tuple[int, ...] = (500, 2000, 8000, 16000)
    sensitivity_trials: int = 300
    k: int = len(EXAMINE)
    seed: int = 31


@dataclass
class Ranker:
    """A name and a top-k list per query id."""

    name: str
    lists: dict[str, list[str]]


def run_online(console: Console, *, pause: bool = True) -> None:
    """Run the online evaluation lesson."""
    config = OnlineConfig()
    world = build_world()
    rng = np.random.default_rng(config.seed)
    control = Ranker("bm25", {q.id: world.bm25_ids(q.text, config.k) for q in world.queries})
    ltr = trained_rankings()
    treatment = Ranker("LambdaRank", {q.id: ltr[q.id][: config.k] for q in world.queries})
    steps = Session.start(console, "online", STEPS, pause)
    belt_panel(
        console,
        "online",
        "online evaluation",
        "Goal: decide whether a new ranker is better for real users, and how sure you are.\n\n"
        "Offline NDCG says LambdaRank beats BM25 (lesson ltr). Users decide. You will run a "
        "simulated A/B test, size it, see how peeking breaks it, check the traffic split, and "
        "compare it with interleaving, which needs far less traffic.",
    )
    steps.step(1, "A/B test setup")
    _step_setup(console, world, control, treatment)
    steps.step(2, "Run it: two-proportion z-test")
    baseline = _step_ab(console, world, control, treatment, config, rng, steps)
    steps.step(3, "How much traffic do you need?")
    _step_sample_size(console, baseline, config, steps)
    steps.step(4, "Peeking inflates false positives")
    _step_peeking(console, baseline, config, rng, steps)
    steps.step(5, "Sample ratio mismatch")
    _step_srm(console, config, steps)
    steps.step(6, "Interleaving")
    _step_interleaving(console, world, control, treatment, config, rng)
    steps.step(7, "Production practice")
    _step_production(console)
    steps.finish()


def _step_setup(console: Console, world: World, control: Ranker, treatment: Ranker) -> None:
    item = world.by_query[EXAMPLE_QUERY_ID]
    console.print(
        "An [bold]A/B test[/bold] randomly splits users: control (A) sees the current ranker, "
        "treatment (B) sees the new one. Randomization makes the groups alike in everything "
        "except the ranker, so a difference in the metric is caused by the ranker.\n"
        "  unit of randomization: the user (not the search), so one person sees one ranker\n"
        "  primary metric: long-click rate = searches with at least one long click (dwell over "
        "30 s). In the simulator a click on a grade 2 or 3 document counts as a long click.\n"
        "  guardrails: latency, zero-result rate, abandonment. These must not get worse."
    )
    table = Table(header_style="bold", title=f"What each arm shows for {item.text!r}")
    table.add_column("rank", justify="right")
    table.add_column(f"A: {control.name}", style="cyan")
    table.add_column("grade", justify="right")
    table.add_column(f"B: {treatment.name}", style="cyan")
    table.add_column("grade", justify="right")
    for rank, (a, b) in enumerate(
        zip(control.lists[item.id], treatment.lists[item.id], strict=False), start=1
    ):
        table.add_row(str(rank), a, str(item.grade(a)), b, str(item.grade(b)))
    console.print(table)


def _long_click_prob(query: GradedQuery, shown: list[str]) -> float:
    """P(at least one long click) under the position-based model."""
    miss = 1.0
    for rank, doc_id in enumerate(shown):
        if query.grade(doc_id) >= LONG_CLICK_GRADE:
            miss *= 1.0 - click_probability(rank, query.grade(doc_id))
    return 1.0 - miss


def _arm_rate(world: World, ranker: Ranker) -> tuple[np.ndarray, np.ndarray]:
    queries, weights = traffic(world)
    probs = np.array([_long_click_prob(q, ranker.lists[q.id]) for q in queries])
    return probs, weights


def _simulate_arm(
    world: World, ranker: Ranker, sessions: int, rng: np.random.Generator
) -> np.ndarray:
    probs, weights = _arm_rate(world, ranker)
    picks = rng.choice(len(probs), size=sessions, p=weights)
    return rng.random(sessions) < probs[picks]


def z_test(success_a: int, n_a: int, success_b: int, n_b: int) -> tuple[float, float, float]:
    """Two-proportion z-test. Returns (difference b − a, z, two-sided p-value)."""
    p_a, p_b = success_a / n_a, success_b / n_b
    pooled = (success_a + success_b) / (n_a + n_b)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se if se else 0.0
    return p_b - p_a, z, math.erfc(abs(z) / math.sqrt(2))


def _normal_quantile(p: float) -> float:
    """Inverse standard normal CDF by bisection on erf. Accurate to about 1e-9."""
    low, high = -10.0, 10.0
    for _ in range(100):
        mid = (low + high) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def sample_size(baseline: float, mde: float, alpha: float, power: float) -> int:
    """Users per arm to detect an absolute lift ``mde`` with a two-sided test."""
    z_alpha = _normal_quantile(1 - alpha / 2)
    z_beta = _normal_quantile(power)
    variance = 2 * baseline * (1 - baseline)
    return math.ceil((z_alpha + z_beta) ** 2 * variance / mde**2)


def _step_ab(
    console: Console,
    world: World,
    control: Ranker,
    treatment: Ranker,
    config: OnlineConfig,
    rng: np.random.Generator,
    steps: Session,
) -> float:
    n = config.ab_sessions_per_arm
    a = _simulate_arm(world, control, n, rng)
    b = _simulate_arm(world, treatment, n, rng)
    diff, z, p_value = z_test(int(a.sum()), n, int(b.sum()), n)
    p_a, p_b = a.mean(), b.mean()
    pooled = (a.sum() + b.sum()) / (2 * n)
    se = math.sqrt(pooled * (1 - pooled) * 2 / n)
    console.print(
        f"{n:,} searches per arm.\n"
        f"  control   long-click rate p_A = {int(a.sum()):,} / {n:,} = {p_a:.4f}\n"
        f"  treatment long-click rate p_B = {int(b.sum()):,} / {n:,} = {p_b:.4f}\n"
        f"  difference = {diff:+.4f}   relative lift = {diff / p_a:+.1%}\n\n"
        "[bold]Two-proportion z-test[/bold]: is the difference larger than chance would make it?\n"
        f"  pooled p = (successes A + successes B) / (n_A + n_B) = {pooled:.4f}\n"
        f"  standard error SE = √(p(1 − p)(1/n_A + 1/n_B)) = {se:.5f}\n"
        f"  z = difference / SE = {z:.2f}\n"
        f"  p-value = P(|Z| ≥ {abs(z):.2f}) under 'no difference' = {p_value:.2g}\n"
        f"  95% confidence interval = difference ± 1.96 × SE = [{diff - 1.96 * se:+.4f}, "
        f"{diff + 1.96 * se:+.4f}]"
    )
    steps.quiz(
        f"p-value = {p_value:.2g}. Which statement is correct?",
        [
            "It is the probability that the treatment is not better",
            "If there were no real difference, a difference this large would show up with "
            "this probability",
            "It is the size of the improvement",
        ],
        1,
        "The p-value assumes no difference and asks how surprising the data is. Report the "
        "effect size and its interval too: a tiny lift can be significant with enough traffic.",
    )
    return float(p_a)


def _step_sample_size(
    console: Console, baseline: float, config: OnlineConfig, steps: Session
) -> None:
    console.print(
        "Decide the sample size before starting. You need the baseline rate p, the minimum "
        "detectable effect (MDE: the smallest lift worth detecting), the significance level α "
        "and the power (chance of detecting a real effect of size MDE):\n"
        "  n per arm = (z_(1−α/2) + z_power)² × 2p(1 − p) / MDE²\n"
        f"Here p = {baseline:.3f}, α = {config.alpha}, power = {config.power:.0%}, so "
        f"z_(1−α/2) = {_normal_quantile(1 - config.alpha / 2):.2f} and "
        f"z_power = {_normal_quantile(config.power):.2f}."
    )
    table = Table(header_style="bold")
    table.add_column("MDE (absolute)", justify="right")
    table.add_column("relative", justify="right")
    table.add_column("users per arm", justify="right")
    for mde in (0.02, 0.01, 0.005, 0.0025):
        table.add_row(
            f"{mde:.4f}",
            f"{mde / baseline:.1%}",
            f"{sample_size(baseline, mde, config.alpha, config.power):,}",
        )
    console.print(table)
    steps.quiz(
        "You halve the MDE. What happens to the sample size?",
        ["It doubles", "It grows 4×", "It halves"],
        1,
        "n grows with 1 / MDE². Small wins in search (0.5% relative is typical for a ranking "
        "change) need a lot of traffic, which is why sensitivity matters.",
    )
    mde = steps.ask_float("Try your own MDE (absolute, e.g. 0.01)", default=0.01)
    if mde > 0:
        console.print(
            f"MDE {mde:g} needs {sample_size(baseline, mde, config.alpha, config.power):,} users per arm."
        )


def _step_peeking(
    console: Console,
    baseline: float,
    config: OnlineConfig,
    rng: np.random.Generator,
    steps: Session,
) -> None:
    n = config.aa_sessions_per_arm
    a = rng.random((config.aa_experiments, n)) < baseline
    b = rng.random((config.aa_experiments, n)) < baseline
    checkpoints = np.linspace(n / config.peeks, n, config.peeks).astype(int)
    peek_hits = np.zeros(config.aa_experiments, dtype=bool)
    final_hits = np.zeros(config.aa_experiments, dtype=bool)
    cum_a = np.cumsum(a, axis=1)
    cum_b = np.cumsum(b, axis=1)
    for index, checkpoint in enumerate(checkpoints):
        for experiment in range(config.aa_experiments):
            _, _, p_value = z_test(
                int(cum_a[experiment, checkpoint - 1]),
                int(checkpoint),
                int(cum_b[experiment, checkpoint - 1]),
                int(checkpoint),
            )
            significant = p_value < config.alpha
            peek_hits[experiment] |= significant
            if index == len(checkpoints) - 1:
                final_hits[experiment] = significant
    console.print(
        f"An [bold]A/A test[/bold] gives both arms the same ranker, so every 'significant' "
        f"result is a false positive. Simulated: {config.aa_experiments} A/A tests with "
        f"{n:,} users per arm. Expected false-positive rate at α = {config.alpha}: 5%."
    )
    steps.quiz(
        f"If you check the p-value {config.peeks} times during each test and stop at the "
        "first p < 0.05, the false-positive rate is about:",
        ["5%", "10%", "25% or more"],
        2,
        "Every look is another chance for noise to cross the line. Fix the horizon in advance, "
        "or use sequential tests built for continuous monitoring (alpha spending, mSPRT).",
    )
    table = Table(header_style="bold")
    table.add_column("procedure")
    table.add_column("false-positive rate", justify="right")
    table.add_column("")
    for name, hits in (
        ("look once at the end", final_hits),
        (f"peek {config.peeks} times, stop at first p < 0.05", peek_hits),
    ):
        rate = float(hits.mean())
        table.add_row(name, f"{rate:.1%}", bar(rate / 0.5, 30))
    console.print(table)


def srm_test(counts: tuple[int, int], expected_share: float) -> tuple[float, float]:
    """Chi-square goodness of fit for a two-way split. Returns (chi2, p-value), 1 degree of freedom."""
    total = sum(counts)
    expected = (total * expected_share, total * (1 - expected_share))
    chi2 = sum((observed - exp) ** 2 / exp for observed, exp in zip(counts, expected, strict=True))
    return chi2, math.erfc(math.sqrt(chi2 / 2))


def _step_srm(console: Console, config: OnlineConfig, steps: Session) -> None:
    counts = config.srm_counts
    chi2, p_value = srm_test(counts, config.srm_expected_share)
    total = sum(counts)
    console.print(
        "Before reading any metric, check that the split matches the design. A "
        "[bold]sample ratio mismatch[/bold] (SRM) means users were lost or duplicated in one "
        "arm, for example the new ranker times out and those sessions are not logged. The "
        "remaining users are no longer comparable.\n"
        "Chi-square test: χ² = Σ (observed − expected)² / expected.\n"
        f"  observed {counts[0]:,} / {counts[1]:,}, expected {total / 2:,.0f} / {total / 2:,.0f}\n"
        f"  χ² = {chi2:.2f}, p-value = {p_value:.2g} (1 degree of freedom)"
    )
    steps.quiz(
        f"The split is {counts[0] / total:.1%} / {counts[1] / total:.1%} and the SRM p-value is "
        f"{p_value:.2g}. The treatment shows +2% long clicks. What do you do?",
        [
            "Ship it: +2% is significant",
            "Don't trust any metric; find why treatment lost users first",
            "Drop the extra control users to balance the arms",
        ],
        1,
        "Users missing from one arm are usually not random (slow sessions, crashes, bots). "
        "Common practice is to flag any SRM with p < 0.001 and debug the assignment and "
        "logging before reading results.",
    )


def team_draft(
    a: list[str], b: list[str], coin: Callable[[], bool], k: int
) -> list[tuple[str, str]]:
    """Team-draft interleaving: each round a coin picks who drafts first; each team adds its best unpicked document."""
    result: list[tuple[str, str]] = []
    picked: set[str] = set()
    count = {"A": 0, "B": 0}
    lists = {"A": a, "B": b}
    while len(result) < k:
        if count["A"] < count["B"] or (count["A"] == count["B"] and coin()):
            team = "A"
        else:
            team = "B"
        choice = next((doc for doc in lists[team] if doc not in picked), None)
        if choice is None:
            team = "B" if team == "A" else "A"
            choice = next((doc for doc in lists[team] if doc not in picked), None)
            if choice is None:
                break
        result.append((choice, team))
        picked.add(choice)
        count[team] += 1
    return result


def _interleaving_prob(query: GradedQuery, merged: list[tuple[str, str]]) -> np.ndarray:
    probs = np.zeros(len(EXAMINE))
    for rank, (doc_id, _) in enumerate(merged):
        probs[rank] = click_probability(rank, query.grade(doc_id))
    return probs


def _step_interleaving(
    console: Console,
    world: World,
    control: Ranker,
    treatment: Ranker,
    config: OnlineConfig,
    rng: np.random.Generator,
) -> None:
    item = world.by_query[EXAMPLE_QUERY_ID]
    merged = team_draft(
        control.lists[item.id], treatment.lists[item.id], lambda: bool(rng.random() < 0.5), config.k
    )
    console.print(
        "[bold]Interleaving[/bold] shows each user one list that mixes both rankers, and "
        "asks which ranker's documents got the clicks. Every user compares both, so user "
        "differences cancel out.\n"
        "[bold]Team draft[/bold]: like picking teams in a playground. Each round, a coin "
        "decides which ranker picks first; each ranker adds its highest document not already "
        "picked. Each document remembers its team. A click credits that team; the team with "
        "more clicks wins the session."
    )
    table = Table(header_style="bold", title=f"One interleaved list for {item.text!r}")
    table.add_column("rank", justify="right")
    table.add_column("document", style="cyan")
    table.add_column("team")
    table.add_column("grade", justify="right")
    for rank, (doc_id, team) in enumerate(merged, start=1):
        name = control.name if team == "A" else treatment.name
        table.add_row(str(rank), doc_id, f"{team} ({name})", str(item.grade(doc_id)))
    console.print(table)

    queries, weights = traffic(world)
    query_probs_a, _ = _arm_rate(world, control)
    query_probs_b, _ = _arm_rate(world, treatment)
    coin_rounds = math.ceil(config.k / 2)
    outcomes = list(itertools.product((True, False), repeat=coin_rounds))
    shape = (len(queries), len(outcomes), len(EXAMINE))
    click_probs = np.zeros(shape)
    on_b = np.zeros(shape, dtype=bool)
    on_a = np.zeros(shape, dtype=bool)
    for qi, query in enumerate(queries):
        for mi, flips in enumerate(outcomes):
            merged_q = team_draft(
                control.lists[query.id], treatment.lists[query.id], iter(flips).__next__, config.k
            )
            click_probs[qi, mi] = _interleaving_prob(query, merged_q)
            for rank, (_, team) in enumerate(merged_q):
                on_b[qi, mi, rank] = team == "B"
                on_a[qi, mi, rank] = team == "A"

    table = Table(
        header_style="bold",
        title=f"Detected the better ranker (p < {config.alpha}), {config.sensitivity_trials} trials each",
    )
    table.add_column("users", justify="right")
    table.add_column("A/B test", justify="right")
    table.add_column("")
    table.add_column("interleaving", justify="right")
    table.add_column("")
    for users in config.sensitivity_sizes:
        ab_hits = 0
        il_hits = 0
        for _ in range(config.sensitivity_trials):
            half = users // 2
            qa = rng.choice(len(queries), size=half, p=weights)
            qb = rng.choice(len(queries), size=half, p=weights)
            success_a = int((rng.random(half) < query_probs_a[qa]).sum())
            success_b = int((rng.random(half) < query_probs_b[qb]).sum())
            diff, _, p_value = z_test(success_a, half, success_b, half)
            ab_hits += diff > 0 and p_value < config.alpha
            q = rng.choice(len(queries), size=users, p=weights)
            m = rng.integers(len(outcomes), size=users)
            clicks = rng.random((users, len(EXAMINE))) < click_probs[q, m]
            credit_b = (clicks & on_b[q, m]).sum(axis=1)
            credit_a = (clicks & on_a[q, m]).sum(axis=1)
            wins_b = int((credit_b > credit_a).sum())
            wins_a = int((credit_a > credit_b).sum())
            decided = wins_a + wins_b
            if decided:
                z = (wins_b - wins_a) / math.sqrt(decided)
                il_hits += z > 0 and math.erfc(abs(z) / math.sqrt(2)) < config.alpha
        ab_rate = ab_hits / config.sensitivity_trials
        il_rate = il_hits / config.sensitivity_trials
        table.add_row(
            str(users), f"{ab_rate:.0%}", bar(ab_rate, 15), f"{il_rate:.0%}", bar(il_rate, 15)
        )
    console.print(table)
    console.print(
        "The A/B test splits users and compares two noisy group means. Interleaving gets a "
        "paired comparison from every user who clicks, so it reaches the same power with "
        "about half the users here. Simulated users are all alike; real users differ a lot "
        "(heavy vs light clickers), which adds noise to A/B tests but cancels in interleaving, "
        "so published results report 10× to 100× less traffic. Most of the traffic here only "
        "gives noise: on most queries both rankers return the same list. The cost: it "
        "answers 'which ranker is preferred', not 'how much does the business metric move', "
        "so teams interleave to screen many candidates, then A/B test the winner."
    )


def _step_production(console: Console) -> None:
    table = Table(header_style="bold")
    table.add_column("practice")
    table.add_column("why")
    table.add_row(
        "pre-register metric, MDE, duration", "prevents picking the metric that happened to win"
    )
    table.add_row("run full weeks", "weekday and weekend users search differently")
    table.add_row(
        "novelty and learning effects", "users react to change itself; watch the effect over time"
    )
    table.add_row(
        "CUPED variance reduction",
        "subtract each user's pre-experiment behavior; same power with less traffic",
    )
    table.add_row("guardrail metrics", "latency, zero-result rate, revenue must not regress")
    table.add_row("ramp up 1% → 10% → 50%", "catch crashes and SRM before most users see them")
    table.add_row(
        "long-term holdout", "a small group keeps the old system to measure cumulative impact"
    )
    console.print(table)
    console.print("[bold]Next[/bold]  uv run search-dojo lesson black")
