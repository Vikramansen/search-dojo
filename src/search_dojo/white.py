"""White belt: sparse vectors and tf-idf, worked by hand on the Dojo DB."""

from __future__ import annotations

import math

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from search_dojo.index import InvertedIndex
from search_dojo.interact import Session
from search_dojo.minidb import MiniDoc, load_mini
from search_dojo.tokenize import tokenize, trace_tokenize
from search_dojo.view import belt_panel

SPOTLIGHT_DOC = "csk-whistle"
DEFAULT_QUERY = "vijay mass"
GAP_QUERY = "thalapathy"
MATRIX_TERMS = ("whistle", "chennai", "win", "vijay", "mass", "rick", "goal")
TOTAL_STEPS = 8


def run_white(console: Console, *, query: str | None = None, pause: bool = True) -> None:
    """Run the white belt.

    Args:
        console: Where to print.
        query: Query for step 7. Defaults to ``vijay mass``.
        pause: Wait for Enter between steps. Ignored when output is not a terminal.
    """
    docs = load_mini()
    index = InvertedIndex([(doc.id, doc.text) for doc in docs])
    steps = Session.start(console, "white", TOTAL_STEPS, pause)

    belt_panel(
        console,
        "white",
        "sparse vectors and tf-idf",
        "Goal: turn each document into a vector of numbers, turn the query into the same kind "
        "of vector, and rank documents by how similar the vectors are.\n\n"
        "You will compute every number by hand, on a 9-document database: tokens, tf, df, "
        "idf, tf-idf, and cosine similarity.",
    )

    steps.step(1, "The database")
    _step_database(console, docs)
    steps.step(2, "Tokenization: text to tokens")
    _step_tokens(console, docs, index)
    steps.step(3, "Term frequency (tf)")
    _step_tf(console, docs, index)
    steps.step(4, "Document frequency (df) and the inverted index")
    _step_df(console, index, steps)
    steps.step(5, "Inverse document frequency (idf)")
    _step_idf(console, index, steps)
    steps.step(6, "tf-idf: the weight of a term in a document")
    _step_tfidf(console, index, steps)
    steps.step(7, "Running a query")
    _step_query(console, docs, index, query or DEFAULT_QUERY, steps)
    steps.step(8, "Where tf-idf breaks")
    _step_limits(console, index)
    steps.finish()


def _step_database(console: Console, docs: list[MiniDoc]) -> None:
    console.print(
        f"This is the entire database: N = {len(docs)} documents. "
        "A document is an id plus a piece of text. The topic column is only for you; "
        "the search engine never reads it.\n"
        "Search means: given a query, return the documents that match it best, best first."
    )
    table = Table(header_style="bold")
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("topic")
    table.add_column("text")
    for doc in docs:
        table.add_row(doc.id, doc.topic, doc.text)
    console.print(table)
    console.print("N (the number of documents) shows up again in the idf formula in step 5.")


def _step_tokens(console: Console, docs: list[MiniDoc], index: InvertedIndex) -> None:
    doc = next(item for item in docs if item.id == SPOTLIGHT_DOC)
    trace = trace_tokenize(doc.text)
    console.print(
        "The index never stores sentences. It stores tokens: the units it can count and "
        f"match. Here is document [cyan]{doc.id}[/cyan] going through the four stages."
    )
    table = Table(header_style="bold", show_lines=True)
    table.add_column("stage", no_wrap=True)
    table.add_column("what it does")
    table.add_column("output")
    table.add_row("0. raw text", "", escape(doc.text))
    table.add_row("1. lowercase", "So Whistle and whistle are the same token.", trace.lowered)
    table.add_row(
        "2. split",
        "Cut on anything that is not a letter or digit. Punctuation disappears.",
        _show_list(trace.split),
    )
    table.add_row(
        "3. drop stop words",
        "Remove words like the, a, at, and. They occur in almost every text, so they "
        "say nothing about the topic.",
        _show_list(trace.without_stop),
    )
    table.add_row(
        "4. stem",
        "Strip plural endings, so chants and chant count as one term.",
        _show_list(trace.stemmed),
    )
    console.print(table)

    console.print("The same four stages, applied to every document:")
    table = Table(header_style="bold")
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("tokens")
    table.add_column("length", justify="right")
    for item in docs:
        tokens = index.doc_tokens[item.id]
        table.add_row(item.id, " ".join(tokens), str(len(tokens)))
    console.print(table)
    console.print(
        f"The vocabulary is the set of distinct tokens across all documents: V = {len(index.df)} "
        "terms. From here on, a token in the vocabulary is called a term."
    )


def _step_tf(console: Console, docs: list[MiniDoc], index: InvertedIndex) -> None:
    console.print(
        "[bold]tf(t, d)[/bold] = the number of times term t appears in document d.\n"
        f"Counting the tokens of [cyan]{SPOTLIGHT_DOC}[/cyan]:"
    )
    table = Table(header_style="bold")
    table.add_column("term t")
    table.add_column(f"tf(t, {SPOTLIGHT_DOC})", justify="right")
    for term, tf in _tf_rows(index, SPOTLIGHT_DOC):
        table.add_row(term, str(tf))
    console.print(table)

    console.print(
        "Do that for every document and stack the rows: you get the document-term matrix. "
        f"One row per document, one column per term, so {len(docs)} rows by {len(index.df)} "
        f"columns. Here are {len(MATRIX_TERMS)} of the columns (· means 0):"
    )
    table = Table(header_style="bold")
    table.add_column("id", style="cyan", no_wrap=True)
    for term in MATRIX_TERMS:
        table.add_column(term, justify="right")
    for doc in docs:
        cells = []
        for term in MATRIX_TERMS:
            tf = index.term_tf(doc.id, term)
            cells.append(f"[bold]{tf}[/bold]" if tf else "[dim]·[/dim]")
        table.add_row(doc.id, *cells)
    console.print(table)

    nonzero = sum(len(set(tokens)) for tokens in index.doc_tokens.values())
    cells = len(docs) * len(index.df)
    console.print(
        f"Across the full matrix, {nonzero} of {cells} cells are non-zero "
        f"({100 * nonzero / cells:.0f}%). That is what [bold]sparse[/bold] means: each document "
        "only has values on the few terms it uses, so the index stores those and skips the zeros. "
        "With a million documents and a vocabulary of a million terms, storing the zeros is impossible."
    )


def _step_df(console: Console, index: InvertedIndex, steps: Session) -> None:
    console.print(
        "[bold]df(t)[/bold] = the number of documents that contain term t at least once.\n"
        "An [bold]inverted index[/bold] stores, for each term, its [bold]posting list[/bold]: "
        "the ids of the documents that contain it. df is the length of the posting list. "
        "It is the matrix from step 3 read column by column instead of row by row."
    )
    table = Table(header_style="bold")
    table.add_column("term t")
    table.add_column("posting list")
    table.add_column("df(t)", justify="right")
    for term in MATRIX_TERMS:
        postings = sorted(index.tf.get(term, {}))
        table.add_row(term, ", ".join(postings), str(index.df.get(term, 0)))
    console.print(table)
    whistle_tf = index.term_tf(SPOTLIGHT_DOC, "whistle")
    steps.quiz(
        f"whistle appears {whistle_tf} times in csk-whistle and nowhere else. What is df(whistle)?",
        [str(whistle_tf), "1", str(index.n_docs)],
        1,
        "df counts documents, not occurrences.",
    )
    console.print(
        f"whistle appears {whistle_tf} times, but all in one document, so df(whistle) = "
        f"{index.df['whistle']}. tf counts occurrences inside one document. df counts documents."
    )


def _step_idf(console: Console, index: InvertedIndex, steps: Session) -> None:
    n_docs = index.n_docs
    console.print(
        "[bold]idf(t) = ln( (1 + N) / (1 + df(t)) ) + 1[/bold]\n\n"
        "idf measures how rare a term is across the database. A rare term is strong evidence "
        "about which document you want: whistle points at exactly one document. A term in "
        "every document points at nothing in particular.\n"
        f"Here N = {n_docs}. Plugging in each df that can occur:"
    )
    table = Table(header_style="bold")
    table.add_column("df(t)", justify="right")
    table.add_column("example")
    table.add_column("calculation")
    table.add_column("idf(t)", justify="right")
    for df, example in (
        (1, "whistle, goal, pickle"),
        (2, "vijay, mass, chennai, rick"),
        (n_docs, "a term in all 9 documents"),
    ):
        ratio = (1 + n_docs) / (1 + df)
        idf = math.log(ratio) + 1
        calc = f"ln({1 + n_docs} / {1 + df}) + 1 = ln({ratio:.3f}) + 1"
        table.add_row(str(df), example, calc, f"{idf:.3f}")
    console.print(table)
    console.print(
        "Why the 1s:\n"
        "  • The 1 + in the fraction pretends there is one extra document containing every "
        "term, so nothing ever divides by zero.\n"
        "  • The + 1 at the end keeps a term found in every document at weight 1 instead of 0, "
        "so it still counts a little.\n"
        "ln is the natural log. It compresses the ratio, so a term in 1 document is worth a bit "
        "more than a term in 2, not twice as much."
    )
    steps.quiz(
        "goal is in 1 document, rick is in 2. Which has the larger idf?",
        ["goal", "rick", "they are equal"],
        0,
        f"goal: ln(10/2)+1 = {index.smooth_idf('goal'):.3f}. "
        f"rick: ln(10/3)+1 = {index.smooth_idf('rick'):.3f}. Rarer term, larger idf.",
    )


def _step_tfidf(console: Console, index: InvertedIndex, steps: Session) -> None:
    console.print(
        "[bold]tf-idf(t, d) = (1 + ln tf(t, d)) × idf(t)[/bold]\n\n"
        "It multiplies the answers to two questions:\n"
        "  1. How much does this document use the term?  → the tf part, 1 + ln tf\n"
        "  2. How rare is the term across the database?  → idf\n"
        "A term gets a big weight in a document only when both answers are big: the document "
        "uses it a lot, and other documents do not."
    )
    console.print(
        "\nWhy 1 + ln tf instead of raw tf: the 4th whistle should add something, but not as "
        "much as the 1st. ln grows slowly:"
    )
    table = Table(header_style="bold")
    table.add_column("tf", justify="right")
    table.add_column("1 + ln tf", justify="right")
    for tf in (1, 2, 4, 10, 100):
        table.add_row(str(tf), f"{1 + math.log(tf):.2f}")
    console.print(table)

    vector = index.tfidf_vectors[SPOTLIGHT_DOC]
    steps.guess(
        "whistle has tf 4 and idf 2.609 in csk-whistle. What is its tf-idf? " "(1 + ln 4 = 2.386)",
        vector["whistle"],
        0.05,
        f"2.386 × 2.609 = {vector['whistle']:.3f}.",
    )
    console.print(f"Now every term of [cyan]{SPOTLIGHT_DOC}[/cyan], step by step:")
    table = Table(header_style="bold")
    table.add_column("term t")
    table.add_column("tf", justify="right")
    table.add_column("1 + ln tf", justify="right")
    table.add_column("df", justify="right")
    table.add_column("idf", justify="right")
    table.add_column("tf-idf = (1 + ln tf) × idf", justify="right")
    for term, tf in _tf_rows(index, SPOTLIGHT_DOC):
        damped = 1 + math.log(tf)
        idf = index.smooth_idf(term)
        table.add_row(
            term,
            str(tf),
            f"{damped:.3f}",
            str(index.df[term]),
            f"{idf:.3f}",
            f"{damped:.3f} × {idf:.3f} = [bold]{vector[term]:.3f}[/bold]",
        )
    console.print(table)
    console.print(
        f"That list is the sparse vector for {SPOTLIGHT_DOC}: {len(vector)} non-zero entries out of "
        f"V = {len(index.df)} dimensions. Every other entry is 0.\n"
        "whistle gets the biggest weight: used 4 times here, used nowhere else. "
        "chennai is lower: used once, and shared with csk-dhoni."
    )


def _step_query(
    console: Console, docs: list[MiniDoc], index: InvertedIndex, query: str, steps: Session
) -> None:
    terms = tokenize(query)
    console.print(f'Query: [bold]"{escape(query)}"[/bold]')
    if not terms:
        console.print("Every word in that query is a stop word, so there is nothing to search for.")
        return
    console.print(
        "A query is treated exactly like a tiny document: same tokenizer, same tf-idf formula, "
        "same idf values from the database."
    )
    query_vector = _query_vector(index, terms)
    table = Table(header_style="bold")
    table.add_column("query term")
    table.add_column("tf", justify="right")
    table.add_column("idf", justify="right")
    table.add_column("weight = (1 + ln tf) × idf", justify="right")
    for term in dict.fromkeys(terms):
        tf = terms.count(term)
        idf = index.smooth_idf(term)
        known = "" if term in index.df else "  [dim](not in vocabulary)[/dim]"
        table.add_row(term + known, str(tf), f"{idf:.3f}", f"{query_vector[term]:.3f}")
    console.print(table)

    candidates = sorted({doc_id for term in terms for doc_id in index.tf.get(term, {})})
    lookups = "\n  ".join(
        f"{term} → {_show_list(sorted(index.tf.get(term, {})))}" for term in dict.fromkeys(terms)
    )
    console.print(
        f"\n[bold]Candidate generation.[/bold] Look up each query term's posting list:\n  {lookups}\n"
        f"Union: {', '.join(candidates) or 'nothing'}. Only these {len(candidates)} documents get "
        f"scored. The other {index.n_docs - len(candidates)} are never touched. Skipping documents "
        "that share no term with the query is why sparse retrieval stays fast on millions of documents."
    )
    if not candidates:
        console.print("No document contains any query term, so the result list is empty.")
        return
    if query == DEFAULT_QUERY:
        steps.quiz(
            "Predict before the math: which document ranks first for 'vijay mass'?",
            ["vijay-leo", "vijay-ghilli", "a tie"],
            1,
            "Ghilli says vijay twice and has fewer other terms. The next tables show the numbers.",
        )

    console.print(
        "\n[bold]cosine(q, d) = (q · d) / (|q| × |d|)[/bold]\n"
        "  • q · d, the dot product: for each term both vectors have, multiply the two weights, "
        "then add. A term only one side has contributes 0.\n"
        "  • |v|, the norm (vector length): the square root of the sum of squared weights.\n"
        "  • Dividing by the norms keeps a long document from winning just because it has more "
        "terms. Cosine is 1 when the vectors point the same way, 0 when they share no term."
    )
    q_norm = _norm(query_vector)
    scored: list[tuple[str, float]] = []
    for doc_id in candidates:
        doc_vector = index.tfidf_vectors[doc_id]
        dot = sum(weight * doc_vector.get(term, 0.0) for term, weight in query_vector.items())
        d_norm = _norm(doc_vector)
        scored.append((doc_id, dot / (q_norm * d_norm)))
    scored.sort(key=lambda item: (-item[1], item[0]))

    for doc_id, score in scored:
        doc_vector = index.tfidf_vectors[doc_id]
        table = Table(title=f"[cyan]{doc_id}[/cyan]", header_style="bold")
        table.add_column("query term")
        table.add_column("q weight", justify="right")
        table.add_column(f"d weight (tf {doc_id})", justify="right")
        table.add_column("q × d", justify="right")
        dot = 0.0
        for term, q_weight in query_vector.items():
            d_weight = doc_vector.get(term, 0.0)
            dot += q_weight * d_weight
            tf = index.term_tf(doc_id, term)
            table.add_row(
                term, f"{q_weight:.3f}", f"{d_weight:.3f}  (tf {tf})", f"{q_weight * d_weight:.3f}"
            )
        console.print(table)
        d_norm = _norm(doc_vector)
        console.print(
            f"  q · d = {dot:.3f}   |q| = {q_norm:.3f}   "
            f"|d| = {d_norm:.3f} (over all {len(doc_vector)} of its terms)\n"
            f"  cosine = {dot:.3f} / ({q_norm:.3f} × {d_norm:.3f}) = [bold]{score:.3f}[/bold]"
        )

    texts = {doc.id: doc.text for doc in docs}
    table = Table(title="Result", header_style="bold")
    table.add_column("rank", justify="right")
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("cosine", justify="right")
    table.add_column("text")
    for rank, (doc_id, score) in enumerate(scored, start=1):
        table.add_row(str(rank), doc_id, f"{score:.3f}", texts[doc_id])
    console.print(table)
    if query == DEFAULT_QUERY:
        console.print(
            "vijay-ghilli wins. It says vijay twice, so its vijay weight is "
            f"{index.tfidf_vectors['vijay-ghilli']['vijay']:.3f} against "
            f"{index.tfidf_vectors['vijay-leo']['vijay']:.3f} in vijay-leo. vijay-leo also has more "
            "other terms (parthiban, cafe, violent, …), which makes its norm larger and its cosine smaller."
        )


def _step_limits(console: Console, index: InvertedIndex) -> None:
    postings = sorted(index.tf.get(GAP_QUERY, {}))
    console.print(
        f'[bold]1. It only matches exact terms.[/bold] Query: "{GAP_QUERY}". '
        f"Posting list: {_show_list(postings)} (empty). Result: nothing.\n"
        "Thalapathy is what fans call Vijay. Two documents are about him, but neither contains "
        "that exact term, so tf-idf scores both 0. This is the [bold]vocabulary gap[/bold]. "
        "The green belt (bi-encoder) closes it by comparing meanings instead of exact terms.\n"
    )
    console.print(
        "[bold]2. Repetition never stops paying.[/bold] 1 + ln tf keeps growing: 10 whistles → "
        f"{1 + math.log(10):.2f}, 100 whistles → {1 + math.log(100):.2f}. A document can repeat a "
        "term to climb the ranking. BM25 puts a ceiling on that. That is the yellow belt.\n"
    )
    console.print(
        "[bold]Try it[/bold]\n"
        '  uv run search-dojo lesson white --no-pause --query "rick pickle"\n'
        '  uv run search-dojo lesson white --no-pause --query "chennai whistle"\n'
        "  Add a line to data/dojo_db.jsonl and watch df and idf change.\n"
        "[bold]Read[/bold]  src/search_dojo/index.py  (smooth_idf, tfidf_cosine)\n"
        "[bold]Next[/bold]  uv run search-dojo lesson yellow"
    )


def _tf_rows(index: InvertedIndex, doc_id: str) -> list[tuple[str, int]]:
    """Terms of a document with their tf, highest tf first, then first appearance."""
    order = list(dict.fromkeys(index.doc_tokens[doc_id]))
    rows = [(term, index.term_tf(doc_id, term)) for term in order]
    return sorted(rows, key=lambda row: (-row[1], order.index(row[0])))


def _query_vector(index: InvertedIndex, terms: list[str]) -> dict[str, float]:
    return {
        term: (1 + math.log(terms.count(term))) * index.smooth_idf(term)
        for term in dict.fromkeys(terms)
    }


def _norm(vector: dict[str, float]) -> float:
    return math.sqrt(sum(weight * weight for weight in vector.values()))


def _show_list(items: list[str]) -> str:
    """Bracketed list, escaped so Rich does not read it as markup."""
    return escape("[" + ", ".join(items) + "]")
