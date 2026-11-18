"""LLMs in search: rewriting, HyDE, RAG with citations, grounding, LLM-as-judge and cost."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from search_dojo.corpus import DATA_DIR, ROOT, _rows
from search_dojo.interact import Session
from search_dojo.metrics import ndcg_at_k
from search_dojo.tokenize import tokenize
from search_dojo.view import belt_panel
from search_dojo.world import World, build_world

STEPS = 8
PROMPTS = ROOT / "prompts"
QUESTION = "How many times have Chennai Super Kings won the IPL?"
RAG_QUERY = "csk ipl titles"
GRADES = (0, 1, 2, 3)
NUMBER = re.compile(r"\b\d+\b")
NUMBER_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}


@dataclass(frozen=True)
class LlmCostConfig:
    """Example traffic and prices for the cost calculator. Prices are placeholders: check your provider."""

    queries_per_day: float = 1_000_000
    prompt_tokens: int = 1_500
    output_tokens: int = 150
    usd_per_million_input: float = 0.50
    usd_per_million_output: float = 1.50
    time_to_first_token_ms: float = 300
    ms_per_output_token: float = 15
    top_sources: int = 3
    k: int = 5


def run_llm(console: Console, *, pause: bool = True) -> None:
    """Run the LLM lesson."""
    config = LlmCostConfig()
    world = build_world()
    outputs = load_outputs()
    steps = Session.start(console, "llm", STEPS, pause)
    belt_panel(
        console,
        "llm",
        "LLMs in search",
        "Goal: know where a large language model fits in a search system, what it costs, and "
        "how to check it.\n\n"
        "Everything runs offline. Prompts live in prompts/*.txt. Model outputs are recorded "
        "stand-ins in data/llm_outputs.json and data/llm_judgments.jsonl, written to behave "
        "like typical models, so the numbers are reproducible. Swap in a real model by sending "
        "the same prompts to it.",
    )
    steps.step(1, "Where LLMs plug in")
    _step_map(console)
    steps.step(2, "Query rewriting")
    _step_rewrite(console, world, outputs, config, steps)
    steps.step(3, "HyDE: retrieve with a hypothetical answer")
    _step_hyde(console, world, outputs, config)
    steps.step(4, "Retrieval-augmented generation")
    _step_rag(console, world, outputs, config)
    steps.step(5, "When retrieval fails, and grounding checks")
    _step_grounding(console, world, outputs, config, steps)
    steps.step(6, "LLM as a relevance judge")
    _step_judge(console, world, steps)
    steps.step(7, "Cost and latency")
    _step_cost(console, world, config, steps)
    steps.step(8, "Evaluating and shipping LLM search")
    _step_production(console)
    steps.finish()


@lru_cache(maxsize=1)
def load_outputs() -> dict[str, object]:
    data = json.loads((DATA_DIR / "llm_outputs.json").read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("llm_outputs.json must be an object")
    return data


def render(name: str, **values: str) -> str:
    """Fill a prompt template from prompts/."""
    template = (PROMPTS / f"{name}.txt").read_text(encoding="utf-8")
    return template.format(**values)


def _section(outputs: dict[str, object], key: str) -> dict[str, str]:
    value = outputs[key]
    if not isinstance(value, dict):
        raise ValueError(f"llm_outputs.json {key} must be an object")
    return {str(k): str(v) for k, v in value.items()}


def _step_map(console: Console) -> None:
    table = Table(header_style="bold")
    table.add_column("stage")
    table.add_column("LLM job")
    table.add_column("runs")
    table.add_column("latency budget")
    table.add_row(
        "query understanding",
        "rewrite, expand, classify intent, spell-correct",
        "per query (cache head queries)",
        "tight: before retrieval",
    )
    table.add_row(
        "retrieval", "HyDE; LLM-based embedding models", "per query / offline per document", "tight"
    )
    table.add_row(
        "ranking",
        "cross-encoder or listwise LLM reranker (RankGPT) on the top 20-100",
        "per query × candidates",
        "moderate",
    )
    table.add_row(
        "answer",
        "RAG: write an answer with citations from the top documents",
        "per query",
        "streams to the user",
    )
    table.add_row(
        "offline: labels",
        "LLM-as-judge relevance grades for evaluation",
        "per (query, document)",
        "none",
    )
    table.add_row(
        "offline: training data",
        "generate queries for documents (InPars, Promptagator) to train retrievers",
        "per document",
        "none",
    )
    table.add_row(
        "offline: enrichment",
        "summaries, tags, entities stored in the index",
        "per document",
        "none",
    )
    console.print(table)
    console.print(
        "Rule of thumb: the further from the user's keystroke, the bigger the model you can "
        "afford. Offline uses scale with the corpus size; online uses scale with traffic."
    )


def _step_rewrite(
    console: Console,
    world: World,
    outputs: dict[str, object],
    config: LlmCostConfig,
    steps: Session,
) -> None:
    rewrites = _section(outputs, "rewrite")
    query = "csk comeback after ban"
    console.print("The prompt, from prompts/rewrite.txt:")
    console.print(Panel(escape(render("rewrite", query=query)), border_style="dim"))
    console.print(f"Recorded model output: [bold]{escape(rewrites[query])}[/bold]")
    table = Table(header_style="bold", title="BM25 with and without the LLM rewrite")
    table.add_column("query")
    table.add_column("rewrite adds")
    table.add_column(f"NDCG@{config.k} before", justify="right")
    table.add_column("after", justify="right")
    by_text = {item.text: item for item in world.queries}
    for original, rewritten in rewrites.items():
        item = by_text[original]
        before = ndcg_at_k(world.bm25_ids(original, config.k), item.qrels, config.k)
        after = ndcg_at_k(world.bm25_ids(rewritten, config.k), item.qrels, config.k)
        table.add_row(original, rewritten[len(original) :].strip(), f"{before:.2f}", f"{after:.2f}")
    console.print(table)
    console.print(
        "Same idea as the synonym table (lesson text), but the model knows that csk means "
        "Chennai Super Kings without anyone writing it down. The prompt says 'keep the "
        "original query' and 'do not answer', which limits drift and stops the model from "
        "inventing facts into the query."
    )
    steps.quiz(
        "An LLM call adds 300 ms. How do you rewrite queries without adding 300 ms to every search?",
        [
            "Use a bigger model",
            "Precompute rewrites for head queries offline, cache them, and only call the model "
            "online for tail queries that look weak (zero or few results)",
            "Rewrite every query twice",
        ],
        1,
        "The head is a small set of queries with most of the traffic (lesson clicks), so a "
        "cache covers most searches. A small distilled model handles the rest.",
    )


def _step_hyde(
    console: Console, world: World, outputs: dict[str, object], config: LlmCostConfig
) -> None:
    hyde = _section(outputs, "hyde")
    by_text = {item.text: item for item in world.queries}
    console.print(
        "[bold]HyDE[/bold] (hypothetical document embeddings): ask the model to write a fake "
        "answer, then search with that passage instead of the short query. The passage uses "
        "the vocabulary real documents use. Facts in it may be wrong; only the words matter. "
        "The original method embeds the passage with a dense encoder; here BM25 searches with it."
    )
    console.print(Panel(escape(render("hyde", query="vijay")), border_style="dim"))
    table = Table(header_style="bold", title="Searching with the hypothetical passage")
    table.add_column("query")
    table.add_column("hypothetical passage")
    table.add_column(f"NDCG@{config.k} query", justify="right")
    table.add_column("passage", justify="right")
    for original, passage in hyde.items():
        item = by_text[original]
        before = ndcg_at_k(world.bm25_ids(original, config.k), item.qrels, config.k)
        after = ndcg_at_k(world.bm25_ids(passage, config.k), item.qrels, config.k)
        table.add_row(original, escape(passage), f"{before:.2f}", f"{after:.2f}")
    console.print(table)
    console.print(
        "The risk: when the model's guess is wrong, the search follows the wrong guess. "
        "Production systems often search with both and fuse the results (blue belt)."
    )


def _sources(world: World, query: str, n: int) -> list[str]:
    return world.bm25_ids(query, n)


def _sources_block(world: World, doc_ids: list[str]) -> str:
    return "\n".join(
        f"[{i}] {world.by_id[doc_id].text}" for i, doc_id in enumerate(doc_ids, start=1)
    )


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def extractive_answer(world: World, question: str, doc_ids: list[str]) -> str:
    """Stand-in generator: the source sentence sharing the most terms with the question, cited."""
    terms = set(tokenize(question))
    best = ("The sources do not say.", 0.0, 0)
    for number, doc_id in enumerate(doc_ids, start=1):
        for sentence in _sentences(world.by_id[doc_id].text):
            score = float(len(terms & set(tokenize(sentence))))
            if score > best[1]:
                best = (sentence, score, number)
    sentence, score, number = best
    if score <= 0:
        return "The sources do not say."
    return f"{sentence.rstrip('.')} [{number}]."


CITATION = re.compile(r"\[(\d+)\]")


def unsupported_facts(answer: str, sources: list[str]) -> list[str]:
    """Numbers and number words in the answer that its cited sources do not contain."""
    cited = [sources[int(n) - 1] for n in CITATION.findall(answer) if 0 < int(n) <= len(sources)]
    source_tokens = set(re.findall(r"[a-z0-9]+", " ".join(cited).lower()))
    claims = CITATION.sub("", answer)
    claimed = set(NUMBER.findall(claims)) | {
        word for word in re.findall(r"[a-z]+", claims.lower()) if word in NUMBER_WORDS
    }
    return sorted(fact for fact in claimed if fact not in source_tokens)


def _step_rag(
    console: Console, world: World, outputs: dict[str, object], config: LlmCostConfig
) -> None:
    rewrites = _section(outputs, "rewrite")
    doc_ids = _sources(world, rewrites[RAG_QUERY], config.top_sources)
    prompt = render("rag_answer", sources=_sources_block(world, doc_ids), question=QUESTION)
    console.print(
        "[bold]RAG[/bold] (retrieval-augmented generation): retrieve the top documents, put "
        "them in the prompt as numbered sources, and ask the model to answer only from them, "
        "with citations. Search does the finding; the model does the reading and writing.\n"
        f"Question: {QUESTION}\n"
        f"Retrieval: BM25 top {config.top_sources} for the rewritten query "
        f"{escape(rewrites[RAG_QUERY])!r}. The filled prompt (prompts/rag_answer.txt):"
    )
    console.print(Panel(escape(prompt), border_style="dim"))
    answer = extractive_answer(world, QUESTION, doc_ids)
    console.print(f"Answer: [bold green]{escape(answer)}[/bold green]")
    console.print(
        "(The stand-in generator copies the source sentence that shares the most terms with "
        "the question. A real model writes fluent text, but the contract is the same: every "
        "claim cites a source.)\nCitations let the user check the answer, and let you check "
        "it automatically."
    )


def _step_grounding(
    console: Console,
    world: World,
    outputs: dict[str, object],
    config: LlmCostConfig,
    steps: Session,
) -> None:
    doc_ids = _sources(world, RAG_QUERY, config.top_sources)
    sources = [world.by_id[doc_id].text for doc_id in doc_ids]
    answer = extractive_answer(world, QUESTION, doc_ids)
    ungrounded = str(outputs["ungrounded_answer"])
    console.print(
        f"Now retrieve with the raw query {RAG_QUERY!r} and no rewrite. Top {config.top_sources}:"
    )
    for number, doc_id in enumerate(doc_ids, start=1):
        console.print(f"  [{number}] {doc_id}: {escape(world.by_id[doc_id].text)}")
    console.print(
        f"csk-five-titles, the document with the answer, is not in the top {config.top_sources} "
        "(lesson eval: it sits at rank 5). The extractive generator says:\n"
        f"  [yellow]{escape(answer)}[/yellow]\n"
        "True but not an answer. A fluent model does worse: it fills the gap from memory and "
        "cites a source anyway:\n"
        f"  [red]{escape(ungrounded)}[/red]"
    )
    steps.quiz(
        "Which component failed?",
        ["The language model", "Retrieval: the answer was never in the context", "The prompt"],
        1,
        "RAG quality is capped by retrieval recall at the context size. Evaluate retrieval "
        "(recall@k of the answer-bearing document) separately from generation.",
    )
    flags = unsupported_facts(ungrounded, sources)
    console.print(
        "[bold]Grounding check[/bold]: verify each claim against the source it cites. A cheap "
        "version: every number in the answer must appear in the cited source. (Checking "
        "against all sources is too weak: 'four' appears in the Jadeja document, about a "
        "boundary, not titles.)\n"
        f"  numbers in the answer missing from the cited source [1]: [red]{', '.join(flags) or 'none'}[/red]\n"
        "Production checks use an entailment (NLI) model or a second LLM call per claim: "
        "'does source [1] support this sentence?'. Unsupported answers are blocked, "
        "regenerated, or shown with a warning."
    )


@dataclass(frozen=True)
class Judgment:
    query_id: str
    doc_id: str
    human: int
    llm: int


def load_judgments(world: World) -> list[Judgment]:
    rows = _rows(DATA_DIR / "llm_judgments.jsonl")
    return [
        Judgment(
            query_id=str(row["query_id"]),
            doc_id=str(row["doc_id"]),
            human=world.by_query[str(row["query_id"])].grade(str(row["doc_id"])),
            llm=int(row["llm_grade"]),
        )
        for row in rows
    ]


def cohen_kappa(
    pairs: list[tuple[int, int]], labels: tuple[int, ...] = GRADES
) -> tuple[float, float, float]:
    """Returns (observed agreement, chance agreement, kappa)."""
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    chance = sum((left[label] / n) * (right[label] / n) for label in labels)
    kappa = (observed - chance) / (1 - chance) if chance < 1 else 1.0
    return observed, chance, kappa


def _step_judge(console: Console, world: World, steps: Session) -> None:
    judgments = load_judgments(world)
    example = judgments[0]
    console.print(
        "Human judgments (lesson eval) are slow and expensive. An LLM can grade (query, "
        "document) pairs with the same guidelines, at a fraction of the cost. The prompt "
        "(prompts/judge.txt) for the first pair:"
    )
    console.print(
        Panel(
            escape(
                render(
                    "judge",
                    query=world.by_query[example.query_id].text,
                    document=world.by_id[example.doc_id].text,
                )
            ),
            border_style="dim",
        )
    )
    console.print(
        f"Before trusting it, compare with human grades. data/llm_judgments.jsonl has "
        f"{len(judgments)} LLM grades for the BM25 top 4 of every query."
    )
    matrix = Table(header_style="bold", title="Confusion matrix (rows: human, columns: LLM)")
    matrix.add_column("human \\ LLM")
    for grade in GRADES:
        matrix.add_column(str(grade), justify="right")
    counts = Counter((j.human, j.llm) for j in judgments)
    for human in GRADES:
        cells = []
        for llm in GRADES:
            value = counts[(human, llm)]
            cells.append(
                f"[bold]{value}[/bold]"
                if human == llm
                else (str(value) if value else "[dim]0[/dim]")
            )
        matrix.add_row(str(human), *cells)
    console.print(matrix)
    observed, chance, kappa = cohen_kappa([(j.human, j.llm) for j in judgments])
    console.print(
        "Raw agreement is misleading: two raters who both say 0 most of the time agree a lot "
        "by luck. [bold]Cohen's kappa[/bold] subtracts the agreement expected by chance:\n"
        f"  observed agreement p_o = diagonal / total = {observed:.3f}\n"
        "  chance agreement p_e = Σ over grades of P(human gives g) × P(LLM gives g) = "
        f"{chance:.3f}\n"
        f"  κ = (p_o − p_e) / (1 − p_e) = {kappa:.3f}\n"
        "Rough reading: below 0.4 weak, 0.4 to 0.6 moderate, 0.6 to 0.8 substantial, above "
        "0.8 near perfect. Human raters often agree with each other only moderately on "
        "graded relevance, so compare the LLM with human-human agreement, not with 1.0."
    )
    lenient = sum(1 for j in judgments if j.llm > j.human)
    strict = sum(1 for j in judgments if j.llm < j.human)
    steps.quiz(
        f"The LLM grades higher than the human {lenient} times and lower {strict} times. "
        "What is the most common fix?",
        [
            "Ignore it",
            "Calibrate: refine the guidelines and few-shot examples on a human-labeled set until "
            "agreement is acceptable, then keep auditing a sample",
            "Use only the LLM grades of 3",
        ],
        1,
        "LLM judges tend to be lenient with on-topic documents that don't answer the query, "
        "prefer longer documents, and can be swayed by text in the document itself. Use them "
        "to scale judgments, and keep humans for calibration and audits.",
    )


def _step_cost(console: Console, world: World, config: LlmCostConfig, steps: Session) -> None:
    counts = world.query_counts()
    ordered = sorted(counts.values(), reverse=True)
    head = max(1, round(len(ordered) * 0.2))
    cache_hit = sum(ordered[:head]) / sum(ordered)
    qpd = steps.ask_float("Queries per day", default=config.queries_per_day)
    price_in = steps.ask_float(
        "USD per million input tokens (example price)", default=config.usd_per_million_input
    )
    price_out = steps.ask_float(
        "USD per million output tokens (example price)", default=config.usd_per_million_output
    )
    per_call = config.prompt_tokens * price_in / 1e6 + config.output_tokens * price_out / 1e6
    latency = config.time_to_first_token_ms + config.output_tokens * config.ms_per_output_token
    table = Table(header_style="bold", title="RAG answer cost")
    table.add_column("item")
    table.add_column("value", justify="right")
    table.add_row("tokens per call", f"{config.prompt_tokens:,} in + {config.output_tokens} out")
    table.add_row("cost per call", f"${per_call:.5f}")
    table.add_row("calls per day, no cache", f"{qpd:,.0f}")
    table.add_row("cost per day, no cache", f"${qpd * per_call:,.0f}")
    table.add_row("cache hit rate (top 20% of queries, from the log)", f"{cache_hit:.0%}")
    table.add_row("cost per day, with cache", f"${qpd * (1 - cache_hit) * per_call:,.0f}")
    table.add_row("cost per year, with cache", f"${qpd * (1 - cache_hit) * per_call * 365:,.0f}")
    table.add_row("latency: first token", f"{config.time_to_first_token_ms:.0f} ms")
    table.add_row("latency: full answer", f"{latency:,.0f} ms")
    console.print(table)
    console.print(
        "Levers: fewer and shorter sources in the prompt (better retrieval → shorter "
        "context), a smaller model for easy queries, caching, and streaming so the user sees "
        "the first token at once. Answers for fresh or personalized queries cannot be cached. "
        f"Prompt tokens dominate: {config.prompt_tokens:,} of {config.prompt_tokens + config.output_tokens:,} "
        "tokens per call are the retrieved sources."
    )


def _step_production(console: Console) -> None:
    table = Table(header_style="bold")
    table.add_column("concern")
    table.add_column("what to measure or do")
    table.add_row("retrieval", "recall@k of the answer-bearing document, before looking at answers")
    table.add_row("faithfulness", "share of answer claims supported by the cited sources")
    table.add_row("answer relevance", "does the answer address the question (human or LLM judge)")
    table.add_row("abstention", "says 'the sources do not say' when retrieval found nothing")
    table.add_row(
        "prompt injection", "documents can contain instructions; treat sources as data, filter them"
    )
    table.add_row(
        "freshness", "the index updates hourly; the model's memory does not, so ground everything"
    )
    table.add_row(
        "online",
        "A/B test with answer-level metrics: copy, follow-up query rate, thumbs, reformulations",
    )
    console.print(table)
    console.print("[bold]Next[/bold]  uv run search-dojo lesson sensei")
