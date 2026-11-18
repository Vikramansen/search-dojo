# Search Dojo

A command-line gym for search relevance, from tf-idf by hand to LLM-as-judge. Fourteen lessons, each one interactive: it pauses between steps, quizzes you, and saves your score.

## Run it

```bash
uv sync
uv run search-dojo                 # menu with your progress; pick a lesson by number
uv run search-dojo lesson white
uv run search-dojo lesson eval --no-pause   # print straight through, answers shown inline
uv run search-dojo syllabus        # role topics → lessons, reading list, case studies
uv run pytest
```

## Lessons, in order

| # | Lesson | What you learn |
| --- | --- | --- |
| 1 | white | Tokenization, tf, df, the inverted index, idf, tf-idf and cosine, every number worked by hand on a 9-document database. |
| 2 | text | Query understanding: Levenshtein edit distance cell by cell, noisy-channel spelling correction, synonym expansion, Naive Bayes intent classification, trie autocomplete from the query log. |
| 3 | yellow | Okapi BM25: `k1` saturates repetition, `b` normalizes length. |
| 4 | eval | Graded judgments (qrels), P@k, recall@k, MRR, average precision, DCG and NDCG by hand, comparing retrievers over 22 queries, and a paired bootstrap confidence interval. |
| 5 | green | A tiny bi-encoder: shared embedding table, mean pooling, cosine. |
| 6 | ann | Approximate nearest neighbor search: brute force, IVF with k-means, graph beam search (the idea behind HNSW), product quantization with exact reranking. Recall vs cost. |
| 7 | blue | Hybrid retrieval with reciprocal rank fusion. |
| 8 | brown | A feature reranker and an ablation. |
| 9 | ltr | Learning to rank: 7 features, pointwise ridge, RankNet, LambdaRank (ΔNDCG-weighted pairs), leave-one-query-out evaluation, feature ablation. |
| 10 | clicks | Query log analysis (Zipf head and tail, zero-result rate), CTR by position, position bias measured with a randomized bucket, inverse propensity weighting, skip-above pairs. |
| 11 | online | A/B testing: two-proportion z-test, sample size and MDE, how peeking inflates false positives, sample ratio mismatch, team-draft interleaving and its sensitivity. |
| 12 | black | Boolean query expressions, query expansion, pseudo-relevance feedback poisoning. |
| 13 | llm | LLMs in search: query rewriting, HyDE, RAG with citations, retrieval failure and grounding checks, LLM-as-judge with Cohen's kappa, a cost and latency calculator. |
| 14 | sensei | Query logs, the full pipeline, and the boss scoreboard. |

## Data

- `data/dojo_db.jsonl` (9 documents) and `data/dojo_db_more.jsonl` (26 more): the Dojo DB. Golden State Warriors, Manchester City, Chennai Super Kings, Rick and Morty, Vijay films.
- `data/dojo_queries.jsonl`: 22 queries with graded relevance (3 perfect, 2 highly relevant, 1 somewhat, missing = 0), including broad queries ("city") and a zero-result query.
- `data/synonyms.json`: the synonym table for lesson text.
- `data/llm_outputs.json`, `data/llm_judgments.jsonl`: recorded stand-in model outputs, so lesson llm runs offline and reproducibly. Nothing in them came from a live model.
- `data/syllabus.json`: role topics, reading list, case studies.
- `prompts/*.txt`: every prompt (rewrite, HyDE, RAG answer, relevance judge). Send them to a real model to replace the stand-ins.

Lessons white, text, eval, ltr, clicks, online and llm use the Dojo DB. Lesson ann uses synthetic embeddings, since ANN only matters at scale. Lessons yellow, green, blue, brown, black and sensei still run on the original corpus in `data/scrolls.jsonl` (with `drills.jsonl`, `challenges.jsonl` and `query_logs.jsonl`).

Simulated users (lessons clicks and online) follow a position-based click model defined in `src/search_dojo/world.py`: P(click) = P(examine rank) × P(click | grade). Every simulation is seeded.

## Other commands

```bash
uv run search-dojo ask "milky pork bone soup" --retriever bm25
uv run search-dojo lesson yellow --k1 0.2 --b 1
uv run search-dojo logs
uv run search-dojo boss
```

## Tests and format

```bash
uv run pytest
uv run ruff check .
uv run black --check .
```

Line length is 100. Every lesson has a test that runs it non-interactively and checks the numbers its narration depends on. Quiz progress is saved to `.dojo_progress.json` (git-ignored).

`git log` shows each lesson landing as its own commit.
