"""Belt scripts. Each one runs the archive and talks about what just happened."""

from __future__ import annotations

from collections.abc import Callable

from rich.console import Console

from search_dojo.archive import Archive
from search_dojo.index import InvertedIndex
from search_dojo.metrics import recall_at_k, score_run
from search_dojo.types import Hit
from search_dojo.view import belt_panel, hits_table, simple_table

LessonFn = Callable[[Console, Archive, float | None, float | None], None]

BELT_ORDER = ("white", "yellow", "green", "blue", "brown", "black", "sensei")

BELT_BLURB = {
    "white": "Sparse encoding. A scroll is a bag of spikes.",
    "yellow": "BM25. Repetition bends. Length costs.",
    "green": "Bi-encoder. Two towers, one table, cosine at the end.",
    "blue": "Hybrid. Let each retriever cover the other's miss.",
    "brown": "Rerank. Features you can zero out and feel.",
    "black": "Query expressions, then expansion, then the model socket.",
    "sensei": "Logs, the full pipeline, and the boss fight.",
}


def run_lesson(
    console: Console,
    archive: Archive,
    belt: str,
    *,
    k1: float | None = None,
    b: float | None = None,
) -> None:
    """Run one belt. Unknown names raise ``KeyError``."""
    lessons: dict[str, LessonFn] = {
        "white": lesson_white,
        "yellow": lesson_yellow,
        "green": lesson_green,
        "blue": lesson_blue,
        "brown": lesson_brown,
        "black": lesson_black,
        "sensei": lesson_sensei,
    }
    try:
        fn = lessons[belt]
    except KeyError as exc:
        known = ", ".join(BELT_ORDER)
        raise KeyError(f"unknown belt {belt!r}. choose one of: {known}") from exc
    fn(console, archive, k1, b)


def lesson_white(console: Console, archive: Archive, k1: float | None, b: float | None) -> None:
    belt_panel(
        console,
        "white",
        "sparse encoding",
        "A scroll is not a paragraph in the index. It is a vector that is zero almost everywhere, "
        "with a weight on each term it actually used. Inverse document frequency is the volume knob: "
        "words the whole archive shares get quiet, words that show up once get loud.\n\n"
        "The tokenizer also drops the word after not, no, never, or without. "
        "Dashi says 'never milky', so milky is not in that scroll.",
    )
    dashi = archive.index.tfidf_vectors["dashi"]
    rows = []
    for term, weight in sorted(dashi.items(), key=lambda item: -item[1])[:8]:
        tf = archive.index.term_tf("dashi", term)
        rows.append([term, str(tf), f"{archive.index.smooth_idf(term):.2f}", f"{weight:.2f}"])
    simple_table(console, "Dashi, the spikes that survived", ["term", "tf", "idf", "tf-idf"], rows)

    idf_rows = sorted(
        (
            (term, archive.index.df[term], archive.index.smooth_idf(term))
            for term in archive.index.df
        ),
        key=lambda item: -item[2],
    )
    show = idf_rows[:5] + idf_rows[-4:]
    simple_table(
        console,
        "IDF, rare at the top, everywhere at the bottom",
        ["term", "df", "idf"],
        [[term, str(df), f"{idf:.2f}"] for term, df, idf in show],
    )

    query = "ramen"
    tf_hits = archive.search(query, "tf", k=4)
    tfidf_hits = archive.search(query, "tfidf", k=4)
    hits_table(console, tf_hits, archive.by_id, title="raw tf cosine  ·  ramen")
    hits_table(console, tfidf_hits, archive.by_id, title="tf-idf cosine  ·  ramen")
    _narrate_ramen(console, tf_hits, tfidf_hits)
    console.print(
        "\n[bold]Read[/bold]  src/search_dojo/tokenize.py  and  InvertedIndex.tfidf_cosine in index.py\n"
        "[bold]Hack[/bold]  Add a scroll to data/scrolls.jsonl and run this belt again. Watch df move."
    )
    _ = (k1, b)


def lesson_yellow(console: Console, archive: Archive, k1: float | None, b: float | None) -> None:
    k1_v = archive.config.bm25_k1 if k1 is None else k1
    b_v = archive.config.bm25_b if b is None else b
    belt_panel(
        console,
        "yellow",
        "BM25",
        "Term frequency is a loudmouth. BM25 saturates it with k1, so the 20th repeat of a word "
        "adds a sliver. It also charges rent for length with b: a long scroll that mentions a term "
        "once scores below a short scroll with the same single mention.\n\n"
        "idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |D| / avgdl))",
    )
    toy = InvertedIndex(
        [
            ("short", "dashi pot"),
            ("long", "dashi " + "water " * 30),
            ("once", "dashi"),
            ("spam", " ".join(["dashi"] * 20)),
        ]
    )
    rows = []
    for doc_id in ("short", "long"):
        for b_try in (0.0, 1.0):
            score, _ = toy.explain_bm25("dashi", doc_id, k1=1.2, b=b_try)
            rows.append([doc_id, "1.2", f"{b_try:.0f}", f"{score:.3f}"])
    simple_table(
        console, "Same single 'dashi'. Only length changes.", ["scroll", "k1", "b", "bm25"], rows
    )
    once, _ = toy.explain_bm25("dashi", "once", k1=1.2, b=0.0)
    spam, _ = toy.explain_bm25("dashi", "spam", k1=1.2, b=0.0)
    ratio = spam / once if once else 0.0
    console.print(
        f"Twenty repetitions score [bold]{ratio:.2f}x[/bold] a single mention when b is 0 and k1 is 1.2. "
        "That bend is k1."
    )

    challenge = archive.challenge("okapi-trap")
    console.print(f"\n[bold]Live archive[/bold]  {challenge.query}")
    console.print(challenge.note)
    for doc_id in ("bm25-scroll", "okapi-animal"):
        score, parts = archive.index.explain_bm25(challenge.query, doc_id, k1=k1_v, b=b_v)
        simple_table(
            console,
            f"{archive.by_id[doc_id].title}  ·  total {score:.2f}  ·  k1={k1_v} b={b_v}",
            ["term", "tf", "idf", "saturation", "contribution"],
            [
                [
                    part.term,
                    str(part.tf),
                    f"{part.idf:.2f}",
                    f"{part.saturation:.2f}",
                    f"{part.contribution:.2f}",
                ]
                for part in parts
            ],
        )
    console.print(
        "\n[bold]Read[/bold]  explain_bm25 in src/search_dojo/index.py\n"
        "[bold]Hack[/bold]  uv run search-dojo lesson yellow --k1 0.2 --b 1"
    )


def lesson_green(console: Console, archive: Archive, k1: float | None, b: float | None) -> None:
    curve = archive.loss_curve
    first = curve[0] if curve else 0.0
    last = curve[-1] if curve else 0.0
    belt_panel(
        console,
        "green",
        "bi-encoder",
        "A bi-encoder embeds the query in one tower and the scroll in another. Here the towers "
        "share one embedding table. A text is the average of its token vectors. Cosine is the score. "
        "Scroll vectors are computed once, when the archive boots, which is why this can sit in front "
        "of millions of documents later.\n\n"
        "Bag-of-words still loves a document that copies the query. The win is the vocabulary gap: "
        "words the scroll never used, sitting near words it did. Product codes like sku1844 are left "
        "out of the table on purpose. A real model would also miss them. BM25 will not.",
    )
    console.print(f"Training loss  {first:.3f}  →  {last:.3f}   seed {archive.config.seed}")
    neighbor_rows = []
    for token in ("pig", "cloudy", "soup", "white"):
        nearest = archive.encoder.nearest(token, k=4)
        rendered = ", ".join(f"{word} {sim:.2f}" for word, sim in nearest) or "(missing)"
        neighbor_rows.append([token, rendered])
    simple_table(console, "Who moved next to whom", ["token", "nearest"], neighbor_rows)
    console.print(
        "Those four drifted toward the broth's own words. milky already lived in the chant "
        "and in the bowl, so the tower had no new place to send it."
    )

    challenge = archive.challenge("vocab-gap")
    console.print(f"\n[bold]{challenge.query}[/bold]  {challenge.note}")
    hits_table(
        console,
        archive.search(challenge.query, "bm25", k=5),
        archive.by_id,
        title="BM25",
    )
    hits_table(
        console,
        archive.search(challenge.query, "dense", k=5),
        archive.by_id,
        title="bi-encoder",
    )
    console.print(
        "\n[bold]Read[/bold]  TinyBiEncoder.train in src/search_dojo/biencoder.py\n"
        "[bold]Hack[/bold]  epochs, lr, and aux_lr live in src/search_dojo/config.py. "
        "Drills live in data/drills.jsonl."
    )
    _ = (k1, b)


def lesson_blue(console: Console, archive: Archive, k1: float | None, b: float | None) -> None:
    belt_panel(
        console,
        "blue",
        "hybrid retrieval",
        "BM25 and cosine are not in the same unit, so adding the raw scores is a category error. "
        "Reciprocal rank fusion ignores the numbers and trusts the order: each list contributes "
        "1 / (k + rank). A scroll both lists rank near the top beats a scroll only one list found.\n\n"
        "The mixed query is a permit code plus a paraphrase of the broth. The code is a rare string "
        "the tower was told to forget. The paraphrase is a neighborhood the inverted index has never seen.",
    )
    challenge = archive.challenge("mixed-intent")
    console.print(f"[bold]{challenge.query}[/bold]  {challenge.note}")
    lexical = archive.search(challenge.query, "bm25", k=5)
    dense = archive.search(challenge.query, "dense", k=5)
    hybrid = archive.search(challenge.query, "hybrid", k=5)
    hits_table(console, lexical, archive.by_id, title="BM25")
    hits_table(console, dense, archive.by_id, title="bi-encoder")
    hits_table(console, hybrid, archive.by_id, title="RRF hybrid")
    qrels = challenge.qrels
    console.print(
        "recall@5   "
        f"bm25 {_recall(lexical, qrels):.2f}   "
        f"dense {_recall(dense, qrels):.2f}   "
        f"hybrid {_recall(hybrid, qrels):.2f}"
    )
    console.print(
        "\n[bold]Read[/bold]  reciprocal_rank_fusion in src/search_dojo/hybrid.py\n"
        "[bold]Hack[/bold]  rrf_k in config.py. Larger k flattens the difference between rank 1 and rank 5."
    )
    _ = (k1, b)


def lesson_brown(console: Console, archive: Archive, k1: float | None, b: float | None) -> None:
    belt_panel(
        console,
        "brown",
        "rerank",
        "A cross-encoder reads the query and the scroll together. Sharp, slow, and only worth it "
        "on the top of a fast list. This belt is that slot with the weights written down: BM25, "
        "cosine, query coverage, a title hit, an exact phrase, and unique_ratio (unique tokens "
        "divided by length). unique_ratio is the anti-spam feature. The chant is the query four times.\n\n"
        "Min-max happens inside the candidate list, so a weight is a vote, not a raw multiplier. "
        "Zero a feature and the top scroll moves, or it does not. That is feature development.",
    )
    challenge = archive.challenge("spam-chant")
    console.print(f"[bold]{challenge.query}[/bold]  {challenge.note}")
    first = archive.index.bm25(challenge.query, k=archive.config.first_stage_k)
    hits_table(console, first, archive.by_id, title="first stage  ·  BM25", k=5)
    weights = archive.config.weight_map()
    reranked = archive.search_rerank(challenge.query, k=5, weights=weights)
    hits_table(console, reranked, archive.by_id, title="after the feature rerank", k=5)
    if reranked and reranked[0].features:
        simple_table(
            console,
            "Normalized features of the new top scroll",
            ["feature", "value", "weight"],
            [
                [
                    name,
                    f"{reranked[0].features.get(name, 0.0):.2f}",
                    f"{weights.get(name, 0.0):.2f}",
                ]
                for name in weights
            ],
        )
    ablation_rows = []
    for name in weights:
        dropped = dict(weights)
        dropped[name] = 0.0
        top = archive.search_rerank(challenge.query, k=1, weights=dropped)
        title = archive.by_id[top[0].doc_id].title if top else "(none)"
        ablation_rows.append([name, title])
    simple_table(
        console, "Ablation  ·  feature zeroed, new #1", ["zeroed", "top scroll"], ablation_rows
    )
    console.print(
        "\nIf the broth scroll was outside the first-stage window, this table could not have saved it. "
        "Rerank reorders. It does not retrieve.\n"
        "[bold]Read[/bold]  src/search_dojo/rerank.py\n"
        "[bold]Hack[/bold]  the weights tuple in src/search_dojo/config.py"
    )
    _ = (k1, b)


def lesson_black(console: Console, archive: Archive, k1: float | None, b: float | None) -> None:
    belt_panel(
        console,
        "black",
        "query expressions and expansion",
        "A query expression is a tree: terms, quoted phrases, AND, OR, NOT, compiled onto posting "
        "lists. Phrases need token positions. Counts cannot see adjacency.\n\n"
        "Expansion is the other move. Add words the searcher left unsaid, then run the index again. "
        "The default expander stays quiet when BM25 already covers the question. Pseudo-relevance "
        "feedback is implemented beside it, and left off the default path, because a bad first pass "
        "teaches the second pass the wrong words.",
    )
    from search_dojo.queryexpr import format_expr, match_expr, parse_query

    for expression in ('"opaque white broth" AND pork', "okapi AND NOT forest"):
        parsed = parse_query(expression)
        matched = sorted(match_expr(archive.index, parsed))
        titles = ", ".join(archive.by_id[doc_id].title for doc_id in matched) or "(none)"
        console.print(f"\n[bold]{expression}[/bold]")
        console.print(f"tree   {format_expr(parsed)}")
        console.print(f"hits   {titles}")

    gap = archive.challenge("vocab-gap")
    rewritten = archive.expand(gap.query)
    console.print(f"\n[bold]Expansion[/bold]  {gap.query}")
    console.print(f"rewritten  {rewritten}")
    hits_table(
        console,
        archive.search(gap.query, "bm25", k=4),
        archive.by_id,
        title="BM25 before expansion",
    )
    hits_table(
        console,
        archive.search(gap.query, "expand", k=4),
        archive.by_id,
        title="BM25 after expansion",
    )

    poison = archive.challenge("poison-prf")
    feedback = archive.prf(poison.query)
    console.print(f"\n[bold]Feedback[/bold]  {poison.query}")
    console.print("PRF terms  " + (", ".join(feedback) if feedback else "(none)"))
    console.print(
        "If bleach showed up, the spill scroll poisoned the rewrite. "
        "That is why the pipeline leaves PRF off."
    )

    def _stand_in(prompt: str) -> str:
        _ = prompt
        return "cloudy pig soup tonkotsu pork broth"

    sample = archive.rewrite_with("cloudy pig soup", _stand_in)
    console.print("\n[bold]LLM socket[/bold]  prompts/rewrite.txt")
    console.print(f"stand-in returned  {sample}")
    console.print(
        "llm_rewrite takes a function. Pass it a model client and this line is a real rewriter. "
        "The archive ships the socket and stays offline."
    )
    console.print(
        "\n[bold]Read[/bold]  src/search_dojo/queryexpr.py  and  src/search_dojo/expand.py\n"
        "[bold]Hack[/bold]  the prompt file, or the coverage gate in config.py"
    )
    _ = (k1, b)


def lesson_sensei(console: Console, archive: Archive, k1: float | None, b: float | None) -> None:
    belt_panel(
        console,
        "sensei",
        "the system",
        "query\n"
        "  -> expand, only when the top lexical hit covers the question poorly\n"
        "  -> BM25 list and bi-encoder list\n"
        "  -> reciprocal rank fusion\n"
        "  -> feature rerank\n"
        "  -> top scrolls\n\n"
        "On a question-answering platform the generator answers from whatever you retrieved. "
        "A wrong top scroll becomes a wrong answer. The boss fight is that fact, measured.",
    )
    show_logs(console, archive)
    console.print()
    show_boss(console, archive)
    console.print(
        "\n[bold]Read[/bold]  src/search_dojo/archive.py  — the pipeline is search_pipeline\n"
        "[bold]Hack[/bold]  data/challenges.jsonl is the exam. Add a question you care about and run the boss."
    )
    _ = (k1, b)


def show_logs(console: Console, archive: Archive) -> None:
    """Print reformulations, vetoes, and click rank."""
    from search_dojo.logs import mean_click_rank, zero_result_queries

    zeros = zero_result_queries(archive.logs)
    console.print(
        f"[bold]Query logs[/bold]  mean click rank {mean_click_rank(archive.logs):.2f}  ·  "
        f"zero-result queries: {', '.join(zeros) if zeros else '(none)'}"
    )
    kept_rows = [
        [item.session, item.before, item.after, archive.by_id[item.click].title]
        for item in archive.kept_reforms
    ]
    suspect_rows = [
        [item.session, item.before, item.after, archive.by_id[item.click].title]
        for item in archive.suspect_reforms
    ]
    simple_table(
        console,
        "Kept rewrites  ·  the new words already sat near the old ones",
        ["session", "before", "after", "click"],
        kept_rows or [["-", "-", "-", "-"]],
    )
    simple_table(
        console,
        "Vetoed rewrites  ·  looks like a new intent, not a synonym",
        ["session", "before", "after", "click"],
        suspect_rows or [["-", "-", "-", "-"]],
    )


def show_boss(console: Console, archive: Archive, k: int | None = None) -> None:
    """Score every challenge with the retrievers that matter."""
    cutoff = archive.config.result_k if k is None else k
    names = ("bm25", "dense", "hybrid", "rerank", "expand", "pipeline")
    qrels = {item.id: item.qrels for item in archive.challenges}
    reports = []
    for name in names:
        rankings = {
            item.id: [hit.doc_id for hit in archive.search(item.query, name, k=cutoff)]
            for item in archive.challenges
        }
        reports.append(score_run(name, rankings, qrels, cutoff))

    header = ["query", *names]
    rows: list[list[str]] = []
    for challenge in archive.challenges:
        row = [challenge.id]
        for report in reports:
            found = next(score for score in report.queries if score.query_id == challenge.id)
            row.append(f"{found.ndcg:.2f}")
        rows.append(row)
    mean_row = ["mean"] + [f"{report.mean_ndcg:.2f}" for report in reports]
    rows.append(mean_row)
    simple_table(console, f"Boss fight  ·  NDCG@{cutoff}", header, rows)

    recall_bits = "   ".join(f"{report.name} {report.mean_recall:.2f}" for report in reports)
    console.print(f"mean recall@{cutoff}   {recall_bits}")
    best = max(reports, key=lambda report: (report.mean_ndcg, report.mean_recall, report.name))
    console.print(f"Highest mean NDCG: [bold]{best.name}[/bold] at {best.mean_ndcg:.2f}")


def _narrate_ramen(console: Console, tf_hits: list[Hit], tfidf_hits: list[Hit]) -> None:
    tf_rank = _rank_of(tf_hits, "repeat-ramen")
    real_rank = _rank_of(tf_hits, "shoyu")
    if tf_rank < real_rank:
        console.print(
            "Raw term frequency puts the stall sign above the shoyu bowl. "
            "The sign says ramen and means nothing else."
        )
    tfidf_sign = _rank_of(tfidf_hits, "repeat-ramen")
    tfidf_bowl = _rank_of(tfidf_hits, "shoyu")
    if tfidf_sign < tfidf_bowl:
        console.print(
            "TF-IDF still hears the sign. It turned the volume of common words down. "
            "It did not saturate repetition. That is the next belt."
        )


def _rank_of(hits: list[Hit], doc_id: str) -> int:
    for index, hit in enumerate(hits):
        if hit.doc_id == doc_id:
            return index
    return 999


def _recall(hits: list[Hit], qrels: dict[str, int]) -> float:
    return recall_at_k([hit.doc_id for hit in hits], qrels, k=max(len(hits), 1))
