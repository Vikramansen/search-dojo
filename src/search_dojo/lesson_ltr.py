"""Learning to rank: features, pointwise, pairwise (RankNet) and listwise-aware (LambdaRank)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from rich.console import Console
from rich.table import Table

from search_dojo.interact import Session, bar
from search_dojo.lesson_text import NaiveBayes
from search_dojo.metrics import ndcg_at_k
from search_dojo.tokenize import tokenize
from search_dojo.view import belt_panel
from search_dojo.world import GradedQuery, World, build_world

STEPS = 8
SHOW_QUERY_ID = "q-city"
FEATURES = ("bm25", "tfidf", "coverage", "proximity", "topic_match", "intent", "log_len")
FEATURE_HELP = {
    "bm25": "BM25 score of the document for the query (yellow belt)",
    "tfidf": "TF-IDF cosine (white belt)",
    "coverage": "fraction of unique query terms found in the document",
    "proximity": "matched terms / shortest window holding all of them; 0 if under 2 match",
    "topic_match": "fraction of query terms found in the document's topic field",
    "intent": "Naive Bayes P(document's topic | query) (lesson text)",
    "log_len": "ln(document length in tokens)",
}
SPARK = "▁▂▃▄▅▆▇█"


@dataclass(frozen=True)
class LtrConfig:
    """Hyperparameters for the learning-to-rank lesson."""

    candidates_per_retriever: int = 10
    ridge_lambda: float = 1.0
    epochs: int = 150
    learning_rate: float = 0.5
    k: int = 5
    seed: int = 13


@dataclass
class QueryData:
    """Candidates and features for one query."""

    query: GradedQuery
    doc_ids: list[str]
    features: np.ndarray
    grades: np.ndarray


@dataclass
class Dataset:
    """All queries stacked into one matrix, with a query id per row."""

    queries: list[QueryData]
    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    qid: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        width = len(FEATURES)
        blocks = [data.features for data in self.queries] or [np.zeros((0, width))]
        self.x = np.vstack(blocks)
        self.y = np.concatenate([data.grades for data in self.queries] or [np.zeros(0)])
        self.qid = np.concatenate(
            [np.full(len(data.doc_ids), i) for i, data in enumerate(self.queries)] or [np.zeros(0)]
        ).astype(int)


def run_ltr(console: Console, *, pause: bool = True) -> None:
    """Run the learning-to-rank lesson."""
    config = LtrConfig()
    world = build_world()
    data = build_dataset(world, config)
    steps = Session.start(console, "ltr", STEPS, pause)
    belt_panel(
        console,
        "ltr",
        "learning to rank",
        "Goal: stop hand-tuning ranking weights and learn them from relevance judgments.\n\n"
        "A ranking system has stages: candidate generation (cheap, high recall: BM25, dense "
        "retrieval) and then a ranker that scores each candidate with many features. "
        "Learning to rank (LTR) trains that ranker. You will build 7 features, train three "
        "models from scratch in numpy, and evaluate them honestly.",
    )
    steps.step(1, "Candidate generation")
    _step_candidates(console, world, data, config)
    steps.step(2, "Ranking features")
    _step_features(console, world, data, steps)
    steps.step(3, "Pointwise: regression on the grade")
    _step_pointwise(console, data, config)
    steps.step(4, "Pairwise: RankNet")
    _step_ranknet(console, data, config, steps)
    steps.step(5, "LambdaRank: weight each pair by how much NDCG it moves")
    _step_lambdarank(console, data, config)
    steps.step(6, "Honest evaluation: leave one query out")
    _step_evaluate(console, data, config)
    steps.step(7, "Feature ablation")
    _step_ablation(console, data, config, steps)
    steps.step(8, "What production uses")
    _step_production(console)
    steps.finish()


def build_dataset(world: World, config: LtrConfig) -> Dataset:
    """Generate candidates per query and compute every feature."""
    classifier = NaiveBayes.train([(doc.text, doc.topic) for doc in world.docs])
    queries: list[QueryData] = []
    for query in world.queries:
        doc_ids = candidates(world, query.text, config)
        intent = classifier.posterior(query.text)
        rows = [_features(world, query.text, doc_id, intent) for doc_id in doc_ids]
        queries.append(
            QueryData(
                query=query,
                doc_ids=doc_ids,
                features=np.array(rows, dtype=float).reshape(len(doc_ids), len(FEATURES)),
                grades=np.array([query.grade(doc_id) for doc_id in doc_ids], dtype=float),
            )
        )
    return Dataset(queries)


def candidates(world: World, text: str, config: LtrConfig) -> list[str]:
    """Union of the BM25 and TF-IDF top lists, BM25 order first."""
    merged = world.bm25_ids(text, config.candidates_per_retriever)
    for doc_id in world.tfidf_ids(text, config.candidates_per_retriever):
        if doc_id not in merged:
            merged.append(doc_id)
    return merged


def _features(world: World, text: str, doc_id: str, intent: dict[str, float]) -> list[float]:
    index = world.index
    terms = tokenize(text)
    bm25 = index.explain_bm25(text, doc_id)[0]
    tfidf = index.tfidf_score(text, doc_id)
    topic = world.by_id[doc_id].topic
    topic_terms = set(tokenize(topic))
    topic_match = sum(term in topic_terms for term in terms) / len(terms) if terms else 0.0
    return [
        bm25,
        tfidf,
        index.coverage(text, doc_id),
        proximity(index.positions, terms, doc_id),
        topic_match,
        intent[topic],
        math.log(max(index.doc_len[doc_id], 1)),
    ]


def proximity(
    positions: dict[str, dict[str, tuple[int, ...]]], terms: list[str], doc_id: str
) -> float:
    """Matched terms divided by the shortest window that contains all of them."""
    events = sorted(
        (pos, term) for term in set(terms) for pos in positions.get(term, {}).get(doc_id, ())
    )
    needed = {term for _, term in events}
    if len(needed) < 2:
        return 0.0
    best = math.inf
    counts: dict[str, int] = {}
    left = 0
    for pos, term in events:
        counts[term] = counts.get(term, 0) + 1
        while len(counts) == len(needed):
            best = min(best, pos - events[left][0] + 1)
            left_term = events[left][1]
            counts[left_term] -= 1
            if counts[left_term] == 0:
                del counts[left_term]
            left += 1
    return len(needed) / best


def standardize(train: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    """z-score with the training mean and standard deviation."""
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1.0
    return [(matrix - mean) / std for matrix in (train, *others)]


def fit_ridge(x: np.ndarray, y: np.ndarray, config: LtrConfig) -> np.ndarray:
    """Closed-form ridge regression. The last weight is the intercept."""
    design = np.hstack([x, np.ones((len(x), 1))])
    penalty = config.ridge_lambda * np.eye(design.shape[1])
    penalty[-1, -1] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y)


def _pairs(y: np.ndarray, qid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    better: list[int] = []
    worse: list[int] = []
    for q in np.unique(qid):
        rows = np.flatnonzero(qid == q)
        for i in rows:
            for j in rows:
                if y[i] > y[j]:
                    better.append(int(i))
                    worse.append(int(j))
    return np.array(better, dtype=int), np.array(worse, dtype=int)


def _ranks(scores: np.ndarray, qid: np.ndarray) -> np.ndarray:
    order = np.lexsort((-scores, qid))
    ranks = np.empty(len(scores), dtype=int)
    starts = {int(q): int(np.flatnonzero(qid[order] == q)[0]) for q in np.unique(qid)}
    for position, row in enumerate(order):
        ranks[row] = position - starts[int(qid[row])]
    return ranks


def _ideal_dcg(y: np.ndarray, qid: np.ndarray) -> dict[int, float]:
    ideal: dict[int, float] = {}
    for q in np.unique(qid):
        gains = np.sort(2.0 ** y[qid == q] - 1)[::-1]
        ideal[int(q)] = float(np.sum(gains / np.log2(np.arange(2, len(gains) + 2)))) or 1.0
    return ideal


def fit_pairwise(
    x: np.ndarray, y: np.ndarray, qid: np.ndarray, config: LtrConfig, *, lambdas: bool
) -> tuple[np.ndarray, list[float]]:
    """Linear RankNet, or LambdaRank when ``lambdas`` is true. Returns weights and loss per epoch."""
    rng = np.random.default_rng(config.seed)
    weights = rng.normal(0.0, 0.01, x.shape[1])
    better, worse = _pairs(y, qid)
    if len(better) == 0:
        return weights, []
    diff = x[better] - x[worse]
    gains = 2.0**y - 1
    ideal = _ideal_dcg(y, qid)
    idcg = np.array([ideal[int(q)] for q in qid[better]])
    history: list[float] = []
    for _ in range(config.epochs):
        margin = diff @ weights
        history.append(float(np.mean(np.logaddexp(0.0, -margin))))
        rho = 1.0 / (1.0 + np.exp(margin))
        if lambdas:
            ranks = _ranks(x @ weights, qid)
            discount = 1.0 / np.log2(ranks + 2.0)
            delta = (
                np.abs((gains[better] - gains[worse]) * (discount[better] - discount[worse])) / idcg
            )
            rho = rho * delta / max(float(delta.mean()), 1e-12)
        weights += config.learning_rate * (rho @ diff) / len(better)
    return weights, history


def _score(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if len(weights) == x.shape[1] + 1:
        return x @ weights[:-1] + weights[-1]
    return x @ weights


def rerank(data: QueryData, scores: np.ndarray) -> list[str]:
    order = np.argsort(-scores, kind="stable")
    return [data.doc_ids[int(i)] for i in order]


ModelName = str
MODELS: tuple[ModelName, ...] = ("bm25 only", "pointwise ridge", "RankNet", "LambdaRank")


def leave_one_out(
    data: Dataset, config: LtrConfig, keep: tuple[int, ...] | None = None
) -> dict[ModelName, list[float]]:
    """NDCG@k per held-out query for each model, training on all the other queries."""
    columns = list(range(len(FEATURES))) if keep is None else list(keep)
    results: dict[ModelName, list[float]] = {name: [] for name in MODELS}
    for held_out, test in enumerate(data.queries):
        train_rows = data.qid != held_out
        x_train = data.x[train_rows][:, columns]
        y_train = data.y[train_rows]
        q_train = data.qid[train_rows]
        if len(test.doc_ids) == 0:
            for name in MODELS:
                results[name].append(0.0)
            continue
        x_train_z, x_test_z = standardize(x_train, test.features[:, columns])
        fitted = {
            "pointwise ridge": fit_ridge(x_train_z, y_train, config),
            "RankNet": fit_pairwise(x_train_z, y_train, q_train, config, lambdas=False)[0],
            "LambdaRank": fit_pairwise(x_train_z, y_train, q_train, config, lambdas=True)[0],
        }
        results["bm25 only"].append(ndcg_at_k(test.doc_ids, test.query.qrels, config.k))
        for name, weights in fitted.items():
            ranked = rerank(test, _score(x_test_z, weights))
            results[name].append(ndcg_at_k(ranked, test.query.qrels, config.k))
    return results


@lru_cache(maxsize=1)
def trained_rankings() -> dict[str, list[str]]:
    """LambdaRank trained on every judged query; the reranked candidates per query id."""
    config = LtrConfig()
    data = build_dataset(build_world(), config)
    x_z = standardize(data.x)[0]
    weights, _ = fit_pairwise(x_z, data.y, data.qid, config, lambdas=True)
    return {
        item.query.id: rerank(item, x_z[data.qid == i] @ weights) if item.doc_ids else []
        for i, item in enumerate(data.queries)
    }


@lru_cache(maxsize=1)
def _full_results() -> dict[ModelName, list[float]]:
    config = LtrConfig()
    return leave_one_out(build_dataset(build_world(), config), config)


def _step_candidates(console: Console, world: World, data: Dataset, config: LtrConfig) -> None:
    table = Table(header_style="bold", title="Candidate pool per query")
    table.add_column("query")
    table.add_column("candidates", justify="right")
    table.add_column("judged relevant", justify="right")
    table.add_column("relevant in pool", justify="right")
    total_relevant = 0
    total_found = 0
    for item in data.queries:
        found = sum(1 for doc_id in item.doc_ids if item.query.grade(doc_id) > 0)
        total_relevant += len(item.query.qrels)
        total_found += found
        style = "red" if found < len(item.query.qrels) else ""
        table.add_row(
            item.query.text,
            str(len(item.doc_ids)),
            str(len(item.query.qrels)),
            f"[{style}]{found}[/{style}]" if style else str(found),
        )
    console.print(
        f"Candidates = BM25 top {config.candidates_per_retriever} ∪ TF-IDF top "
        f"{config.candidates_per_retriever}. The ranker only reorders this pool."
    )
    console.print(table)
    console.print(
        f"Candidate recall: {total_found}/{total_relevant} = {total_found / total_relevant:.0%} "
        "of judged relevant documents are in a pool. Red rows have relevant documents the "
        "ranker can never show, however good it is. Recall problems are fixed in candidate "
        "generation (expansion, dense retrieval), not in the ranker."
    )


def _step_features(console: Console, world: World, data: Dataset, steps: Session) -> None:
    table = Table(header_style="bold")
    table.add_column("feature")
    table.add_column("definition")
    for name in FEATURES:
        table.add_row(name, FEATURE_HELP[name])
    console.print(table)
    console.print(
        "Query-dependent features (bm25 through intent) change with the query. log_len is "
        "query-independent, like popularity or freshness would be in production."
    )
    item = next(entry for entry in data.queries if entry.query.id == SHOW_QUERY_ID)
    matrix = Table(header_style="bold", title=f"Feature rows for {item.query.text!r}")
    matrix.add_column("document", style="cyan", no_wrap=True)
    matrix.add_column("grade", justify="right")
    for name in FEATURES:
        matrix.add_column(name, justify="right")
    for doc_id, row, grade in zip(item.doc_ids, item.features, item.grades, strict=True):
        matrix.add_row(doc_id, str(int(grade)), *(f"{value:.2f}" for value in row))
    console.print(matrix)
    console.print("Rows are in BM25 order. Each row is one training example: features → grade.")
    steps.quiz(
        "BM25 puts citadel (Rick and Morty) above the Manchester City documents for 'city'. "
        "Which feature separates them best?",
        ["bm25", "topic_match", "log_len"],
        1,
        "'city' is in the topic 'Manchester City' but not in 'Rick and Morty'. A metadata "
        "field the text scorer ignores is exactly what a learned ranker can use.",
    )


def _step_pointwise(console: Console, data: Dataset, config: LtrConfig) -> None:
    x_z = standardize(data.x)[0]
    weights = fit_ridge(x_z, data.y, config)
    console.print(
        "First, [bold]standardize[/bold] every feature: z = (value − mean) / standard deviation, "
        "using the training rows. BM25 runs from 0 to about 10, coverage from 0 to 1; without "
        "this, the weights are not comparable and gradient descent is unstable.\n\n"
        "[bold]Pointwise[/bold] LTR treats each (query, document) row on its own and predicts "
        "the grade with regression. Ridge regression, closed form:\n"
        f"  w = (XᵀX + λI)⁻¹ Xᵀy     λ = {config.ridge_lambda} shrinks the weights so "
        "correlated features don't explode.\n"
        "Score = w · z. Sort candidates by score."
    )
    _weights_table(console, "ridge weights, trained on all queries", weights[:-1])
    console.print(
        "Weakness: pointwise loss cares about the exact grade, but ranking only needs the "
        "order. Predicting 2.1 vs 1.9 for two documents costs nothing in loss but flips the "
        "ranking; predicting 0.5 for every grade-0 document wastes effort."
    )


def _step_ranknet(console: Console, data: Dataset, config: LtrConfig, steps: Session) -> None:
    better, _ = _pairs(data.y, data.qid)
    x_z = standardize(data.x)[0]
    weights, history = fit_pairwise(x_z, data.y, data.qid, config, lambdas=False)
    console.print(
        "[bold]Pairwise[/bold] LTR learns from pairs: inside one query, document i has a higher "
        "grade than document j, so we want s_i > s_j.\n"
        "RankNet models P(i above j) = σ(s_i − s_j), where σ(z) = 1 / (1 + e^(−z)), and minimizes "
        "the cross-entropy\n"
        "  loss = ln(1 + e^(−(s_i − s_j)))   about 0 when s_i ≫ s_j, large when the order is wrong.\n"
        "With a linear scorer s = w · z, the gradient step for one pair is\n"
        "  w ← w + η · σ(−(s_i − s_j)) · (z_i − z_j)\n"
        "Pairs already ordered correctly by a wide margin barely move w.\n"
        f"Training pairs across all queries: {len(better):,}. η = {config.learning_rate}, "
        f"{config.epochs} epochs."
    )
    console.print(f"loss per epoch  {_sparkline(history)}  {history[0]:.3f} → {history[-1]:.3f}")
    _weights_table(console, "RankNet weights", weights)
    steps.quiz(
        "Pairs only compare documents of the same query. Why not compare across queries?",
        [
            "It would be too slow",
            "Grades are only comparable within a query: a grade 2 for 'city' says nothing "
            "about a grade 1 for 'dhoni'",
            "Cross-query pairs have no gradient",
        ],
        1,
        "Users choose between results of one query. The ranking problem lives inside a query.",
    )


def _step_lambdarank(console: Console, data: Dataset, config: LtrConfig) -> None:
    item = next(entry for entry in data.queries if entry.query.id == SHOW_QUERY_ID)
    gains = 2.0**item.grades - 1
    ideal = _ideal_dcg(item.grades, np.zeros(len(item.grades), dtype=int))[0]
    console.print(
        "RankNet treats every misordered pair the same. But NDCG cares mostly about the top: "
        "swapping ranks 1 and 2 matters far more than swapping 9 and 10.\n"
        "[bold]LambdaRank[/bold] keeps the RankNet gradient and multiplies each pair's push by\n"
        "  |ΔNDCG| = how much NDCG would change if i and j swapped places in the current ranking\n"
        "  |ΔNDCG| = |gain_i − gain_j| × |1/log2(rank_i + 1) − 1/log2(rank_j + 1)| / IDCG\n"
        "This is how a non-differentiable metric (sorting is not differentiable) gets into "
        "gradient descent: optimize a smooth loss, but weight it by the metric."
    )
    table = Table(
        header_style="bold", title=f"Two misordered pairs for {item.query.text!r}, BM25 order"
    )
    table.add_column("pair")
    table.add_column("ranks", justify="right")
    table.add_column("grades", justify="right")
    table.add_column("|ΔNDCG|", justify="right")
    misordered = [
        (i, j)
        for i in range(len(item.doc_ids))
        for j in range(i + 1, len(item.doc_ids))
        if item.grades[j] > item.grades[i]
    ]
    picks = [misordered[0], max(misordered, key=lambda pair: (pair[0], pair[1]))]
    for i, j in dict.fromkeys(picks):
        delta = abs((gains[i] - gains[j]) * (1 / math.log2(i + 2) - 1 / math.log2(j + 2))) / ideal
        table.add_row(
            f"{item.doc_ids[j]} should beat {item.doc_ids[i]}",
            f"{j + 1} vs {i + 1}",
            f"{int(item.grades[j])} vs {int(item.grades[i])}",
            f"{delta:.3f}",
        )
    console.print(table)
    console.print(
        "The pair near the top gets a much larger push. Swap the linear scorer for gradient "
        "boosted trees and you get [bold]LambdaMART[/bold], the standard production LTR model "
        "(LightGBM objective=lambdarank, XGBoost rank:ndcg)."
    )


def _step_evaluate(console: Console, data: Dataset, config: LtrConfig) -> None:
    results = _full_results()
    console.print(
        "Training and testing on the same queries rewards memorizing them. With only "
        f"{len(data.queries)} queries, use [bold]leave-one-query-out[/bold] cross-validation: "
        "train on all queries but one, score the held-out query, repeat for each query, "
        "average. Folds are split by query, never by row: rows of one query share the query, "
        "so splitting rows leaks it."
    )
    baseline = float(np.mean(results["bm25 only"]))
    table = Table(header_style="bold", title=f"Mean NDCG@{config.k}, leave-one-query-out")
    table.add_column("model")
    table.add_column(f"NDCG@{config.k}", justify="right")
    table.add_column("vs bm25", justify="right")
    table.add_column("queries better / worse", justify="right")
    table.add_column("")
    for name in MODELS:
        values = results[name]
        mean = float(np.mean(values))
        better = sum(a > b + 1e-9 for a, b in zip(values, results["bm25 only"], strict=True))
        worse = sum(a < b - 1e-9 for a, b in zip(values, results["bm25 only"], strict=True))
        table.add_row(
            name, f"{mean:.3f}", f"{mean - baseline:+.3f}", f"{better} / {worse}", bar(mean, 20)
        )
    console.print(table)
    lambda_scores = results["LambdaRank"]
    gains = sorted(
        zip(data.queries, lambda_scores, results["bm25 only"], strict=True),
        key=lambda item: item[2] - item[1],
    )
    detail = Table(header_style="bold", title="Biggest changes, LambdaRank vs bm25")
    detail.add_column("query")
    detail.add_column("bm25", justify="right")
    detail.add_column("LambdaRank", justify="right")
    for item, new, old in gains[:3] + gains[-2:]:
        detail.add_row(item.query.text, f"{old:.2f}", f"{new:.2f}")
    console.print(detail)
    console.print(
        "Check the losses as well as the mean. In an interview, say how you validated: query-"
        "level splits, a held-out test set, and a significance test (lesson eval)."
    )


def _step_ablation(console: Console, data: Dataset, config: LtrConfig, steps: Session) -> None:
    full = float(np.mean(_full_results()["LambdaRank"]))
    drops: list[tuple[str, float]] = []
    for removed in range(len(FEATURES)):
        keep = tuple(i for i in range(len(FEATURES)) if i != removed)
        score = float(np.mean(leave_one_out(data, config, keep)["LambdaRank"]))
        drops.append((FEATURES[removed], full - score))
    most = max(drops, key=lambda item: item[1])
    console.print(
        "Ablation: retrain without one feature and measure the loss. It tells you which "
        "features carry the model, and which you could drop to save serving cost. Correlated "
        "features hide each other: dropping one lets its twin take over."
    )
    steps.quiz(
        "Which feature costs the most NDCG when removed?",
        list(FEATURES),
        FEATURES.index(most[0]),
        f"Removing {most[0]} costs {most[1]:+.3f} NDCG@{config.k}.",
    )
    table = Table(header_style="bold", title=f"LambdaRank without each feature (full: {full:.3f})")
    table.add_column("removed")
    table.add_column("NDCG lost", justify="right")
    table.add_column("")
    top = max(max(drop for _, drop in drops), 1e-9)
    for name, drop in sorted(drops, key=lambda item: -item[1]):
        table.add_row(name, f"{drop:+.3f}", bar(max(drop, 0.0) / top, 20))
    console.print(table)
    console.print(
        "A negative value means the model did slightly better without the feature: on 22 "
        "queries that is noise or mild overfitting. Differences this small need the bootstrap "
        "from lesson eval before you trust them."
    )


def _step_production(console: Console) -> None:
    table = Table(header_style="bold")
    table.add_column("topic")
    table.add_column("in production")
    table.add_row(
        "model",
        "LambdaMART (LightGBM, XGBoost) on hundreds of features; neural rankers and cross-encoders on top",
    )
    table.add_row(
        "labels", "human judgments for evaluation, clicks for scale (lesson clicks: position bias)"
    )
    table.add_row(
        "features",
        "text match per field, embeddings similarity, popularity, freshness, personalization, quality",
    )
    table.add_row(
        "feature logging",
        "log features at serving time; recomputing them later causes train/serve skew",
    )
    table.add_row(
        "stages", "cheap model over 1000s of candidates, expensive model over the top 100"
    )
    table.add_row(
        "monitoring", "feature distributions, score drift, offline NDCG on fresh judgments"
    )
    console.print(table)
    console.print("[bold]Next[/bold]  uv run search-dojo lesson clicks")


def _weights_table(console: Console, title: str, weights: np.ndarray) -> None:
    table = Table(header_style="bold", title=title)
    table.add_column("feature")
    table.add_column("weight", justify="right")
    table.add_column("")
    top = max(float(np.max(np.abs(weights))), 1e-9)
    for name, weight in zip(FEATURES, weights, strict=True):
        sign = "[green]+[/green]" if weight >= 0 else "[red]−[/red]"
        table.add_row(name, f"{weight:+.3f}", f"{sign} {bar(abs(weight) / top, 20)}")
    console.print(table)


def _sparkline(values: list[float]) -> str:
    if not values:
        return ""
    step = max(1, len(values) // 40)
    sampled = values[::step]
    low, high = min(sampled), max(sampled)
    span = (high - low) or 1.0
    return "".join(SPARK[int((value - low) / span * (len(SPARK) - 1))] for value in sampled)
