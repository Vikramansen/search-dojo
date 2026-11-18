"""Query understanding: spelling, synonyms, intent and autocomplete, before retrieval runs."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from search_dojo.corpus import DATA_DIR
from search_dojo.interact import Session, bar
from search_dojo.metrics import ndcg_at_k
from search_dojo.tokenize import tokenize
from search_dojo.view import belt_panel
from search_dojo.world import World, build_world

STEPS = 7
TYPO = "haland"
INTENT_QUERIES = ("kings record", "city", "chennai mall")
PREFIXES = ("th", "c", "ri")


def run_text(console: Console, *, pause: bool = True) -> None:
    """Run the query understanding lesson."""
    world = build_world()
    steps = Session.start(console, "text", STEPS, pause)
    belt_panel(
        console,
        "text",
        "query understanding",
        "Goal: fix and enrich the query before it reaches the index. Real queries are short, "
        "misspelled, abbreviated and ambiguous. This layer is usually called query "
        "understanding (QU).\n\n"
        "You will build a spell corrector from edit distance, expand abbreviations, train a "
        "Naive Bayes intent classifier, and build an autocomplete trie from the query log.",
    )
    steps.step(1, "Where query understanding sits")
    _step_pipeline(console)
    steps.step(2, "Edit distance, cell by cell")
    _step_edit_distance(console, steps)
    steps.step(3, "Spell correction: candidates and the noisy channel")
    _step_spelling(console, world)
    steps.step(4, "Synonyms and abbreviations")
    _step_synonyms(console, world, steps)
    steps.step(5, "Intent classification with Naive Bayes")
    _step_intent(console, world, steps)
    steps.step(6, "Autocomplete from the query log")
    _step_autocomplete(console, world, steps)
    steps.step(7, "What production adds")
    _step_production(console)
    steps.finish()


def _step_pipeline(console: Console) -> None:
    console.print(
        "raw query\n"
        "  → normalize   lowercase, strip punctuation, Unicode folding\n"
        "  → spell-correct   haland → haaland\n"
        "  → expand   csk → chennai super kings\n"
        "  → classify intent   which topic or vertical: sports, movies, TV\n"
        "  → retrieval and ranking (the other lessons)\n\n"
        "Each step is cheap compared with retrieval and ranking, runs on every query, and fails "
        "quietly: a bad correction or expansion sends a correct query to the wrong documents. "
        "Measure every QU change with the evaluation set, like any ranking change."
    )


def _step_edit_distance(console: Console, steps: Session) -> None:
    target = "haaland"
    console.print(
        "[bold]Levenshtein edit distance[/bold] = the fewest single-character insertions, "
        "deletions or substitutions that turn one string into another.\n"
        "Dynamic programming: cell (i, j) is the distance between the first i letters of one "
        "word and the first j letters of the other:\n"
        "  D[i][j] = min( D[i−1][j] + 1   delete,\n"
        "               D[i][j−1] + 1   insert,\n"
        "               D[i−1][j−1] + (0 if the letters match else 1)   match or substitute )\n"
        f"The bottom-right cell is the answer. Here is {TYPO!r} against {target!r}:"
    )
    grid = _levenshtein_grid(TYPO, target)
    table = Table(header_style="bold", show_lines=False)
    table.add_column("")
    table.add_column("ε", justify="right")
    for letter in target:
        table.add_column(letter, justify="right")
    for i, row in enumerate(grid):
        label = "ε" if i == 0 else TYPO[i - 1]
        cells = [str(value) for value in row]
        if i == len(TYPO):
            cells[-1] = f"[bold reverse]{cells[-1]}[/bold reverse]"
        table.add_row(label, *cells)
    console.print(table)
    steps.quiz(
        "What is the edit distance between 'meesseeks' and 'meeseeks'?",
        ["0", "1", "2"],
        1,
        f"One deletion (the extra s). Computed: {_levenshtein('meesseeks', 'meeseeks')}.",
    )


def _step_spelling(console: Console, world: World) -> None:
    counts = _corpus_counts(world)
    console.print(
        "Spell correction for a query term that is not in the vocabulary:\n"
        "  1. Candidates: every vocabulary term within edit distance 2.\n"
        "  2. Score each candidate with the [bold]noisy channel[/bold] model: "
        "P(candidate) × P(typo | candidate).\n"
        "     P(candidate): how common the term is (corpus counts here, query-log counts in "
        "production).\n"
        "     P(typo | candidate): how likely this typo is. Approximated as e^(−2 × distance), "
        "so each extra edit cuts the probability by about 7×.\n"
        "  3. Take the best candidate, but only if it clearly beats leaving the word alone."
    )
    console.print(
        "The index stores stemmed terms (white belt), so the typo is stemmed first: "
        "meesseeks → meesseek, matched against meeseek."
    )
    for raw in (TYPO, "thalapthy", "meesseeks"):
        typo = tokenize(raw)[0]
        candidates = _candidates(typo, counts)
        table = Table(header_style="bold", title=f"query term {raw!r} (stemmed: {typo!r})")
        table.add_column("candidate")
        table.add_column("edit distance", justify="right")
        table.add_column("corpus count", justify="right")
        table.add_column("score = ln(count) − 2·distance", justify="right")
        for term, distance, score in candidates[:4]:
            table.add_row(term, str(distance), str(counts[term]), f"{score:.2f}")
        if not candidates:
            table.add_row("(none within distance 2)", "", "", "")
        console.print(table)
    console.print(
        "Short words are risky: within distance 2 of 'ipl' there are many unrelated terms. "
        "Production spellers skip very short tokens, never correct a term that is already in "
        "the vocabulary, and learn P(typo | candidate) from query-log rewrites (users typing "
        "haland and then haaland)."
    )


def _step_synonyms(console: Console, world: World, steps: Session) -> None:
    synonyms = _load_synonyms()
    console.print(
        "Some query terms never appear in documents: abbreviations (csk), slang, or a "
        "different word for the same idea (ban vs suspension). A synonym table maps them to "
        "terms the index knows. Here is data/synonyms.json:"
    )
    table = Table(header_style="bold")
    table.add_column("query term")
    table.add_column("expands to")
    for key, value in synonyms.items():
        table.add_row(key, value)
    console.print(table)
    steps.quiz(
        "'csk comeback after ban' returns zero documents with BM25. Why?",
        [
            "BM25 needs at least 5 query terms",
            "None of csk, comeback or ban appears in any document; after is a stop word",
            "The ban document is too long",
        ],
        1,
        "The document says Chennai Super Kings returned after a two year suspension. "
        "Same meaning, different words: this is the vocabulary gap.",
    )
    table = Table(header_style="bold", title="BM25 before and after expansion")
    table.add_column("query")
    table.add_column("added terms")
    table.add_column("top 3 before")
    table.add_column("top 3 after")
    table.add_column("NDCG@5", justify="right")
    for query_id in ("q-csk-ban", "q-csk-titles", "q-politics"):
        item = world.by_query[query_id]
        expanded = _expand(item.text, synonyms)
        before = world.bm25_ids(item.text, 5)
        after = world.bm25_ids(expanded, 5)
        table.add_row(
            item.text,
            expanded[len(item.text) :].strip(),
            ", ".join(before[:3]) or "(nothing)",
            ", ".join(after[:3]),
            f"{ndcg_at_k(before, item.qrels, 5):.2f} → {ndcg_at_k(after, item.qrels, 5):.2f}",
        )
    console.print(table)
    console.print(
        "Expansion appends terms and keeps the originals, so a query that already worked still "
        "matches its documents. The risk is drift: 'ban' → 'suspension' is right for cricket "
        "and wrong for a query about a ban on plastic bags. Scope synonyms by context, and "
        "measure. Learned alternatives: mine synonym pairs from query-log reformulations, or "
        "use embeddings (green belt) or an LLM rewriter (lesson llm)."
    )


def _step_intent(console: Console, world: World, steps: Session) -> None:
    classifier = NaiveBayes.train([(doc.text, doc.topic) for doc in world.docs])
    console.print(
        "Intent classification answers: which part of the catalog is this query about? Here "
        "the classes are the 5 topics. Training data: each document's text labeled with its "
        "topic. In production you would train on queries labeled by the category of the "
        "result users clicked.\n\n"
        "[bold]Multinomial Naive Bayes[/bold]\n"
        "  score(class) = ln P(class) + Σ over query terms of ln P(term | class)\n"
        "  P(class) = documents in the class / all documents\n"
        "  P(term | class) = (count of term in the class + 1) / (total terms in the class + V)\n"
        "The + 1 is Laplace smoothing: a term never seen in a class gets a small probability "
        "instead of 0, which would wipe out the whole score. 'Naive' because it treats the terms "
        "as independent given the class."
    )
    first = INTENT_QUERIES[0]
    terms = tokenize(first)
    table = Table(header_style="bold", title=f"Scoring {first!r} term by term")
    table.add_column("class")
    table.add_column("ln P(class)", justify="right")
    for term in terms:
        table.add_column(f"ln P({term} | class)", justify="right")
    table.add_column("total", justify="right")
    table.add_column("probability", justify="right")
    probabilities = classifier.posterior(first)
    for label in classifier.labels:
        parts = [classifier.log_prior[label]] + [
            classifier.log_likelihood(term, label) for term in terms
        ]
        table.add_row(
            label,
            *(f"{value:.2f}" for value in parts),
            f"{sum(parts):.2f}",
            f"{probabilities[label]:.2f}",
        )
    console.print(table)
    console.print(
        "Probabilities come from a softmax over the totals: e^total / Σ e^total. "
        "'kings' appears for both the Warriors (Klay's 37 points against the Sacramento Kings) "
        "and Chennai Super Kings, so 'record' decides it."
    )
    for query in INTENT_QUERIES[1:]:
        ranked = sorted(classifier.posterior(query).items(), key=lambda item: -item[1])
        console.print(f"\n{query!r}")
        for label, probability in ranked[:3]:
            console.print(f"  {label:24} {bar(probability, 24)} {probability:.2f}")
    city = classifier.posterior("city")
    top_label = max(city, key=city.__getitem__)
    steps.quiz(
        "For an ambiguous query like 'city', what should a search system do?",
        [
            "Always route to the single most likely class",
            "Act on a confident class; when unsure, blend results from the top classes",
            "Refuse to answer",
        ],
        1,
        f"Here 'city' goes {city[top_label]:.0%} to {top_label}, but the Citadel of Ricks is "
        "also a city. Confident queries can be routed or boosted. Ambiguous ones get results "
        "from several intents, so at least one is right.",
    )


def _step_autocomplete(console: Console, world: World, steps: Session) -> None:
    counts = world.query_counts()
    trie = Trie()
    for query, count in counts.items():
        trie.insert(query, count)
    console.print(
        "Autocomplete suggests full queries as the user types. The classic structure is a "
        "[bold]trie[/bold] (prefix tree): one node per character, with each complete query "
        "stored at the node where it ends. A prefix lookup walks down the trie, then ranks every "
        "query under that node by how often it was searched.\n"
        f"The trie here holds the {len(counts)} distinct queries of the simulated query log "
        f"({sum(counts.values()):,} searches)."
    )
    for prefix in PREFIXES:
        _print_suggestions(console, trie, prefix)
    if steps.interactive:
        for _ in range(3):
            prefix = steps.ask_text("Type a prefix (Enter to stop)", default="").strip().lower()
            if not prefix:
                break
            _print_suggestions(console, trie, prefix)
    steps.quiz(
        "The log contains 'thalapthy' (a typo) 7 times. Should autocomplete suggest it?",
        ["Yes, it was searched", "No: filter typos and low-count queries out of suggestions"],
        1,
        "Suggestions teach users what to type. Drop typos, unsafe queries and anything below "
        "a minimum count, or map typos to their corrected form.",
    )


def _step_production(console: Console) -> None:
    table = Table(header_style="bold")
    table.add_column("component")
    table.add_column("what changes at scale")
    table.add_row(
        "spelling",
        "learn edit probabilities from logs; context-aware (bigram) correction; never touch known names",
    )
    table.add_row(
        "synonyms",
        "mined from reformulations and click graphs; scoped by category; weighted lower than original terms",
    )
    table.add_row(
        "intent",
        "fine-tuned transformer classifiers on clicked-category labels; multi-label; confidence thresholds",
    )
    table.add_row(
        "autocomplete",
        "time-decayed counts, personalization, safety filters, under 50 ms per keystroke",
    )
    table.add_row(
        "LLM rewriting",
        "handles long natural-language queries; cache results for head queries (lesson llm)",
    )
    console.print(table)
    console.print("[bold]Next[/bold]  uv run search-dojo lesson yellow")


@dataclass
class NaiveBayes:
    """Multinomial Naive Bayes with Laplace smoothing."""

    labels: list[str]
    log_prior: dict[str, float]
    term_counts: dict[str, Counter[str]]
    totals: dict[str, int]
    vocab_size: int

    @classmethod
    def train(cls, examples: list[tuple[str, str]]) -> NaiveBayes:
        labels = sorted({label for _, label in examples})
        docs_per_label = Counter(label for _, label in examples)
        term_counts: dict[str, Counter[str]] = {label: Counter() for label in labels}
        vocab: set[str] = set()
        for text, label in examples:
            tokens = tokenize(text)
            term_counts[label].update(tokens)
            vocab.update(tokens)
        return cls(
            labels=labels,
            log_prior={label: math.log(docs_per_label[label] / len(examples)) for label in labels},
            term_counts=term_counts,
            totals={label: sum(term_counts[label].values()) for label in labels},
            vocab_size=len(vocab),
        )

    def log_likelihood(self, term: str, label: str) -> float:
        count = self.term_counts[label][term]
        return math.log((count + 1) / (self.totals[label] + self.vocab_size))

    def posterior(self, text: str) -> dict[str, float]:
        terms = tokenize(text)
        scores = {
            label: self.log_prior[label] + sum(self.log_likelihood(term, label) for term in terms)
            for label in self.labels
        }
        top = max(scores.values())
        exps = {label: math.exp(score - top) for label, score in scores.items()}
        total = sum(exps.values())
        return {label: value / total for label, value in exps.items()}


@dataclass
class _Node:
    children: dict[str, _Node] = field(default_factory=dict)
    count: int = 0


class Trie:
    """Prefix tree of queries with search counts."""

    def __init__(self) -> None:
        self.root = _Node()

    def insert(self, query: str, count: int) -> None:
        node = self.root
        for char in query:
            node = node.children.setdefault(char, _Node())
        node.count += count

    def complete(self, prefix: str, k: int = 5) -> list[tuple[str, int]]:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        found: list[tuple[str, int]] = []
        stack = [(node, prefix)]
        while stack:
            current, text = stack.pop()
            if current.count:
                found.append((text, current.count))
            for char, child in current.children.items():
                stack.append((child, text + char))
        found.sort(key=lambda item: (-item[1], item[0]))
        return found[:k]


def _print_suggestions(console: Console, trie: Trie, prefix: str) -> None:
    suggestions = trie.complete(prefix)
    console.print(f"\n[bold]{escape(prefix)}[/bold]▌")
    if not suggestions:
        console.print("  (no query in the log starts with this)")
    top = suggestions[0][1] if suggestions else 1
    for text, count in suggestions:
        console.print(f"  {escape(text):30} {bar(count / top, 20)} {count}")


def _levenshtein_grid(left: str, right: str) -> list[list[int]]:
    grid = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i in range(len(left) + 1):
        grid[i][0] = i
    for j in range(len(right) + 1):
        grid[0][j] = j
    for i in range(1, len(left) + 1):
        for j in range(1, len(right) + 1):
            cost = 0 if left[i - 1] == right[j - 1] else 1
            grid[i][j] = min(grid[i - 1][j] + 1, grid[i][j - 1] + 1, grid[i - 1][j - 1] + cost)
    return grid


def _levenshtein(left: str, right: str) -> int:
    return _levenshtein_grid(left, right)[-1][-1]


def _corpus_counts(world: World) -> dict[str, int]:
    return {term: sum(per_doc.values()) for term, per_doc in world.index.tf.items()}


def _candidates(typo: str, counts: dict[str, int]) -> list[tuple[str, int, float]]:
    found: list[tuple[str, int, float]] = []
    for term, count in counts.items():
        if abs(len(term) - len(typo)) > 2:
            continue
        distance = _levenshtein(typo, term)
        if distance <= 2:
            found.append((term, distance, math.log(count) - 2 * distance))
    found.sort(key=lambda item: (-item[2], item[0]))
    return found


def _load_synonyms() -> dict[str, str]:
    data = json.loads((DATA_DIR / "synonyms.json").read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("synonyms.json must be an object")
    return {str(key): str(value) for key, value in data.items()}


def _expand(query: str, synonyms: dict[str, str]) -> str:
    extra = [synonyms[word] for word in query.lower().split() if word in synonyms]
    return " ".join([query, *extra])
