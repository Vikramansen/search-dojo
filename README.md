# Search Dojo

A command-line gym for search relevance. One archive, seven belts, and a boss fight that scores the whole pipeline.

The Midnight Archive answers questions from scrolls. Retrieve the wrong scroll and the answer is wrong. That is the job.

## Run it

```bash
uv sync
uv run search-dojo
uv run search-dojo lesson white
uv run search-dojo ask "milky pork bone soup" --retriever bm25
uv run search-dojo ask "cloudy pig soup" --retriever dense
uv run search-dojo boss
uv run pytest
```

Belts, in order: `white` `yellow` `green` `blue` `brown` `black` `sensei`.

```bash
uv run search-dojo lesson yellow --k1 0.2 --b 1
uv run search-dojo logs
```

## What each belt trains

| Belt | You will feel |
| --- | --- |
| white | Tokens, postings, tf, idf. A scroll is a sparse vector. |
| yellow | Okapi BM25. `k1` bends repetition. `b` charges rent for length. Every term has a contribution you can read. |
| green | A bi-encoder small enough to train on a laptop: shared embedding table, mean pool, cosine. The win is a paraphrase, not a copied query. |
| blue | Hybrid retrieval. Reciprocal rank fusion, because BM25 and cosine do not share a unit. |
| brown | A feature reranker and an ablation. Zero a weight, watch the top scroll move. |
| black | Boolean query expressions (phrases need positions), query expansion, and why pseudo-relevance feedback can poison the second pass. |
| sensei | Query logs, the veto on bad reformulations, and the boss scoreboard. |

The full path a question takes:

```text
query
  -> expand, only when the top lexical hit covers the question poorly
  -> BM25 list and bi-encoder list
  -> reciprocal rank fusion
  -> feature rerank
  -> top scrolls
```

## Where to hack

- `data/scrolls.jsonl` — the archive. Tags are not indexed. Add them to the body if you want them searchable.
- `data/drills.jsonl` — practice triples for the tower.
- `data/challenges.jsonl` — the exam. Grades are 2 (the answer) or 1 (a cousin).
- `data/query_logs.jsonl` — sessions. Order inside a session is the reformulation.
- `src/search_dojo/config.py` — `k1`, `b`, `rrf_k`, training steps, feature weights.
- `prompts/rewrite.txt` — the prompt a model would see. `llm_rewrite` in `expand.py` takes the completion function. The dojo stays offline until you pass a client in.

`sku1844` is a product code. BM25 keeps it. The bi-encoder vocabulary refuses it, on purpose, so hybrid retrieval has something only the inverted index can catch.

## The prep list, and where it lives

- **Relevance on a QA platform.** The `qa-impact` scroll, and the sensei belt. A wrong top scroll becomes a wrong answer.
- **LLMs inside search.** `llm_rewrite` and `prompts/rewrite.txt`. The model rewrites. The index still retrieves. The stand-in in the black belt shows the socket without a network call.
- **IR and learned ranking.** BM25, the tiny bi-encoder, the linear feature reranker.
- **Text, NLP, retrieval, ranking.** `tokenize.py`, `index.py`, `rerank.py`.
- **Query log analysis.** `logs.py`, `search-dojo logs`. A miss followed by a click is a label. The encoder vetoes rewrites that look like a new intent.
- **Ranking features.** `rerank.py`. The brown belt zeros each weight.

Recent ideas show up as the moves you can feel: dense retrieval for vocabulary gap, RRF because scores are incomparable, selective expansion, and a generator that only sees what you retrieved.

## Tests and format

```bash
uv run pytest
uv run ruff check .
uv run black --check .
```

Line length is 100. Ruff lints, black formats. Worth wiring into pre-commit if you keep hacking this.

Training logs stay quiet unless you ask:

```bash
SEARCH_DOJO_LOG=DEBUG uv run search-dojo lesson green
```

`git log` is the syllabus. Each belt landed as its own commit.
