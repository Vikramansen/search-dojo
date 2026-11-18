"""Approximate nearest neighbor search: brute force, IVF, graph search and product quantization."""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from rich.console import Console
from rich.table import Table

from search_dojo.interact import Session, bar
from search_dojo.view import belt_panel

STEPS = 7
FLOAT_BYTES = 4


@dataclass(frozen=True)
class AnnConfig:
    """Sizes for the synthetic embedding collection and each index."""

    n_vectors: int = 10_000
    dim: int = 32
    n_topics: int = 200
    topic_spread: float = 0.2
    n_queries: int = 100
    k: int = 10
    ivf_lists: int = 64
    kmeans_iters: int = 12
    nprobes: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    graph_neighbors: int = 12
    graph_random_links: int = 4
    graph_entry_samples: int = 32
    efs: tuple[int, ...] = (10, 20, 40, 80, 160)
    pq_subspaces: int = 8
    pq_centroids: int = 256
    pq_rerank: int = 100
    seed: int = 5


@dataclass
class Collection:
    """Unit vectors, queries, and the exact answers."""

    vectors: np.ndarray
    queries: np.ndarray
    exact: np.ndarray


def run_ann(console: Console, *, pause: bool = True) -> None:
    """Run the ANN lesson."""
    config = AnnConfig()
    steps = Session.start(console, "ann", STEPS, pause)
    belt_panel(
        console,
        "ann",
        "approximate nearest neighbor search",
        "Goal: find the top-k most similar vectors without comparing the query against every "
        "vector.\n\n"
        "The bi-encoder (green belt) turns every document into a vector. Dense retrieval = "
        "find the documents whose vectors have the highest dot product with the query vector. "
        "Exact search scores everything; ANN indexes give up a little recall for a large "
        "speedup. You will build three of them in numpy and measure the trade-off.",
    )
    collection = build_collection(config)
    steps.step(1, "Why exact search does not scale")
    _step_brute_force(console, collection, config, steps)
    steps.step(2, "Recall@k: the ANN quality metric")
    _step_recall(console, config)
    steps.step(3, "IVF: cluster first, search a few clusters")
    ivf_rows = _step_ivf(console, collection, config, steps)
    steps.step(4, "Graph search: the idea behind HNSW")
    graph_rows = _step_graph(console, collection, config)
    steps.step(5, "Product quantization: compress the vectors")
    pq_rows = _step_pq(console, collection, config, steps)
    steps.step(6, "The trade-off")
    _step_tradeoff(console, config, ivf_rows, graph_rows, pq_rows)
    steps.step(7, "ANN in production")
    _step_production(console)
    steps.finish()


@lru_cache(maxsize=4)
def build_collection(config: AnnConfig) -> Collection:
    """Clustered unit vectors, like embeddings of documents about a few dozen topics."""
    rng = np.random.default_rng(config.seed)
    centers = _normalize(rng.normal(size=(config.n_topics, config.dim)))
    topic = rng.integers(config.n_topics, size=config.n_vectors)
    vectors = _normalize(
        centers[topic] + config.topic_spread * rng.normal(size=(config.n_vectors, config.dim))
    )
    query_topic = rng.integers(config.n_topics, size=config.n_queries)
    queries = _normalize(
        centers[query_topic] + config.topic_spread * rng.normal(size=(config.n_queries, config.dim))
    )
    exact = np.argsort(-(queries @ vectors.T), axis=1)[:, : config.k]
    return Collection(vectors.astype(np.float32), queries.astype(np.float32), exact)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def recall_at_k(found: np.ndarray, exact: np.ndarray) -> float:
    """Mean fraction of the exact top-k that the index returned."""
    hits = [
        len(set(f.tolist()) & set(e.tolist())) / len(e) for f, e in zip(found, exact, strict=True)
    ]
    return float(np.mean(hits))


def kmeans(
    data: np.ndarray, k: int, iters: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd's k-means with squared Euclidean distance. Returns (centroids, assignment)."""
    centroids = data[rng.choice(len(data), size=k, replace=False)].copy()
    assign = np.zeros(len(data), dtype=int)
    for _ in range(iters):
        distances = (
            (data**2).sum(axis=1, keepdims=True)
            - 2 * data @ centroids.T
            + (centroids**2).sum(axis=1)
        )
        assign = distances.argmin(axis=1)
        for c in range(k):
            members = data[assign == c]
            if len(members):
                centroids[c] = members.mean(axis=0)
    return centroids, assign


def _step_brute_force(
    console: Console, collection: Collection, config: AnnConfig, steps: Session
) -> None:
    start = time.perf_counter()
    scores = collection.queries @ collection.vectors.T
    np.argpartition(-scores, config.k, axis=1)
    elapsed = (time.perf_counter() - start) / config.n_queries
    console.print(
        f"The collection: {config.n_vectors:,} unit vectors of dimension {config.dim}, drawn "
        f"around {config.n_topics} topic centers, so similar documents loosely cluster like real "
        f"embeddings do. {config.n_queries} queries come from the same topics. (The Dojo DB has "
        "35 documents; ANN only matters with many more, so these vectors are synthetic.)\n\n"
        "[bold]Brute force[/bold] (flat index): score = q · v for every v, keep the top k.\n"
        f"  cost per query = N × d = {config.n_vectors:,} × {config.dim} = "
        f"{config.n_vectors * config.dim:,} multiply-adds\n"
        f"  measured here: {elapsed * 1e3:.3f} ms per query (numpy, batched)"
    )
    steps.quiz(
        "A real system has 1 billion documents with 768-dimensional embeddings. Multiply-adds "
        "per query for brute force?",
        ["About 7.7 million", "About 768 billion", "About 1 billion"],
        1,
        "10⁹ × 768 ≈ 7.7 × 10¹¹ per query, and 3 TB of float32 vectors to read. At thousands "
        "of queries per second that is impossible; hence ANN.",
    )


def _step_recall(console: Console, config: AnnConfig) -> None:
    console.print(
        f"An ANN index returns k = {config.k} vectors per query. Compare with the exact top "
        f"{config.k} from brute force:\n"
        f"  recall@{config.k} = |ANN top {config.k} ∩ exact top {config.k}| / {config.k}, "
        "averaged over queries\n"
        "This is a different recall from the relevance recall of lesson eval: it measures how "
        "well the index imitates exact search, not whether the documents are relevant. A "
        "bi-encoder with weak embeddings and a perfect index still retrieves the wrong "
        "documents.\n"
        "Cost is counted as vectors scored per query: it does not depend on your laptop."
    )


def _step_ivf(
    console: Console, collection: Collection, config: AnnConfig, steps: Session
) -> list[tuple[str, float, float]]:
    rng = np.random.default_rng(config.seed)
    centroids, assign = kmeans(collection.vectors, config.ivf_lists, config.kmeans_iters, rng)
    lists = [np.flatnonzero(assign == c) for c in range(config.ivf_lists)]
    sizes = np.array([len(members) for members in lists])
    console.print(
        "[bold]IVF[/bold] (inverted file index), the same idea as the inverted index from the "
        "white belt, with clusters instead of words:\n"
        f"  build: k-means groups the vectors into {config.ivf_lists} clusters (cells); each "
        "cell stores the ids of its vectors (a posting list)\n"
        "  search: score the query against the centroids, open the nprobe nearest cells, "
        "score only the vectors inside them\n"
        "k-means (Lloyd's algorithm): pick random centroids; repeat: assign each vector to its "
        "nearest centroid, move each centroid to the mean of its vectors.\n"
        f"Cell sizes: min {sizes.min()}, median {int(np.median(sizes))}, max {sizes.max()}."
    )
    results = [
        _ivf_search(collection, centroids, lists, nprobe, config) for nprobe in config.nprobes
    ]
    middle = config.nprobes[len(config.nprobes) // 2]
    middle_recall = results[config.nprobes.index(middle)][0]
    steps.guess(
        f"Guess recall@{config.k} with nprobe = {middle} of {config.ivf_lists} cells "
        f"(scanning about {middle / config.ivf_lists:.0%} of the vectors). Answer 0 to 1.",
        middle_recall,
        0.1,
        f"Measured {middle_recall:.3f}. Most true neighbors sit in the cells nearest the "
        "query, so a small fraction of the scan gets much of the recall. The misses are "
        "neighbors just across a cell boundary, which is why recall climbs slowly at the end.",
    )
    table = Table(header_style="bold", title="IVF")
    table.add_column("nprobe", justify="right")
    table.add_column(f"recall@{config.k}", justify="right")
    table.add_column("")
    table.add_column("vectors scored", justify="right")
    table.add_column("of brute force", justify="right")
    rows: list[tuple[str, float, float]] = []
    for nprobe, (recall, scored) in zip(config.nprobes, results, strict=True):
        share = scored / config.n_vectors
        table.add_row(
            str(nprobe), f"{recall:.3f}", bar(recall, 20), f"{scored:,.0f}", f"{share:.1%}"
        )
        rows.append((f"IVF nprobe={nprobe}", recall, scored))
    console.print(table)
    if steps.interactive:
        value = int(
            steps.ask_float(f"Try your own nprobe (1 to {config.ivf_lists})", default=float(middle))
        )
        value = min(max(value, 1), config.ivf_lists)
        recall, scored = _ivf_search(collection, centroids, lists, value, config)
        console.print(f"nprobe {value}: recall {recall:.3f}, {scored:,.0f} vectors scored")
    return rows


def _ivf_search(
    collection: Collection,
    centroids: np.ndarray,
    lists: list[np.ndarray],
    nprobe: int,
    config: AnnConfig,
) -> tuple[float, float]:
    found = np.zeros((config.n_queries, config.k), dtype=int)
    scored = 0
    for qi, query in enumerate(collection.queries):
        cells = np.argsort(-(centroids @ query))[:nprobe]
        ids = np.concatenate([lists[c] for c in cells])
        scores = collection.vectors[ids] @ query
        top = ids[np.argsort(-scores)[: config.k]]
        found[qi, : len(top)] = top
        scored += len(ids) + len(centroids)
    return recall_at_k(found, collection.exact), scored / config.n_queries


def build_graph(vectors: np.ndarray, config: AnnConfig) -> np.ndarray:
    """Each node links to its nearest neighbors plus a few random long-range nodes."""
    rng = np.random.default_rng(config.seed)
    n = len(vectors)
    near = np.zeros((n, config.graph_neighbors), dtype=int)
    for start in range(0, n, 1000):
        block = vectors[start : start + 1000] @ vectors.T
        for row in range(len(block)):
            block[row, start + row] = -np.inf
        near[start : start + 1000] = np.argpartition(-block, config.graph_neighbors, axis=1)[
            :, : config.graph_neighbors
        ]
    far = rng.integers(n, size=(n, config.graph_random_links))
    return np.hstack([near, far])


def graph_search(
    vectors: np.ndarray, graph: np.ndarray, query: np.ndarray, entry: int, ef: int, k: int
) -> tuple[list[int], int]:
    """Best-first beam search. Returns the top-k ids and how many vectors were scored."""
    visited = {entry}
    score = float(vectors[entry] @ query)
    candidates = [(-score, entry)]
    best = [(score, entry)]
    scored = 1
    while candidates:
        neg, node = heapq.heappop(candidates)
        if len(best) >= ef and -neg < best[0][0]:
            break
        fresh = [int(n) for n in graph[node] if int(n) not in visited]
        if not fresh:
            continue
        visited.update(fresh)
        scores = vectors[fresh] @ query
        scored += len(fresh)
        for neighbor, s in zip(fresh, scores, strict=True):
            s = float(s)
            if len(best) < ef or s > best[0][0]:
                heapq.heappush(candidates, (-s, neighbor))
                heapq.heappush(best, (s, neighbor))
                if len(best) > ef:
                    heapq.heappop(best)
    top = sorted(best, reverse=True)[:k]
    return [node for _, node in top], scored


def _step_graph(
    console: Console, collection: Collection, config: AnnConfig
) -> list[tuple[str, float, float]]:
    graph = build_graph(collection.vectors, config)
    rng = np.random.default_rng(config.seed + 1)
    console.print(
        "[bold]Graph search[/bold]: connect every vector to its nearest neighbors. To search, "
        "start somewhere and keep walking to whichever neighbor is closer to the query.\n"
        f"  build: link each vector to its {config.graph_neighbors} nearest neighbors, plus "
        f"{config.graph_random_links} random far links so a walk can cross between topics "
        "quickly (a 'small world' graph)\n"
        "  search: best-first beam search. Keep the ef best vectors found so far; repeatedly "
        "expand the closest unexpanded one by scoring its neighbors; stop when no candidate "
        "can beat the worst of the ef best.\n"
        "  entry point: the best of a few random vectors. HNSW does this with a hierarchy: "
        "sparse upper layers of the same graph find a good entry point in a few hops.\n"
        "ef (beam width) is the knob: larger ef explores more and finds more true neighbors."
    )
    table = Table(header_style="bold", title="Graph beam search")
    table.add_column("ef", justify="right")
    table.add_column(f"recall@{config.k}", justify="right")
    table.add_column("")
    table.add_column("vectors scored", justify="right")
    table.add_column("of brute force", justify="right")
    rows: list[tuple[str, float, float]] = []
    entries = rng.integers(config.n_vectors, size=(config.n_queries, config.graph_entry_samples))
    for ef in config.efs:
        found = np.zeros((config.n_queries, config.k), dtype=int)
        total = 0
        for qi, query in enumerate(collection.queries):
            samples = entries[qi]
            entry = int(samples[np.argmax(collection.vectors[samples] @ query)])
            ids, scored = graph_search(
                collection.vectors, graph, query, entry, max(ef, config.k), config.k
            )
            found[qi, : len(ids)] = ids
            total += scored + config.graph_entry_samples
        recall = recall_at_k(found, collection.exact)
        scored_mean = total / config.n_queries
        table.add_row(
            str(ef),
            f"{recall:.3f}",
            bar(recall, 20),
            f"{scored_mean:,.0f}",
            f"{scored_mean / config.n_vectors:.1%}",
        )
        rows.append((f"graph ef={ef}", recall, scored_mean))
    console.print(table)
    console.print(
        "Graph indexes give the best recall per vector scored, which is why HNSW is the default "
        "in most vector databases. The cost is memory for the links and a slow build: here the "
        "build compared every pair; real HNSW builds incrementally by inserting with the same "
        "search."
    )
    return rows


def _step_pq(
    console: Console, collection: Collection, config: AnnConfig, steps: Session
) -> list[tuple[str, float, float]]:
    rng = np.random.default_rng(config.seed + 2)
    sub = config.dim // config.pq_subspaces
    codebooks = []
    codes = np.zeros((config.n_vectors, config.pq_subspaces), dtype=np.uint8)
    for s in range(config.pq_subspaces):
        part = collection.vectors[:, s * sub : (s + 1) * sub]
        centroids, assign = kmeans(part, config.pq_centroids, config.kmeans_iters, rng)
        codebooks.append(centroids)
        codes[:, s] = assign
    raw_bytes = config.dim * FLOAT_BYTES
    code_bytes = config.pq_subspaces
    console.print(
        "Memory is the other limit: 1 billion × 768 float32 = 3 TB. [bold]Product "
        "quantization[/bold] (PQ) compresses each vector to a few bytes:\n"
        f"  split each {config.dim}-dimensional vector into {config.pq_subspaces} sub-vectors "
        f"of {sub} dimensions\n"
        f"  run k-means with {config.pq_centroids} centroids in each subspace (a codebook)\n"
        f"  store each sub-vector as the id of its nearest centroid: 1 byte\n"
        f"  {raw_bytes} bytes → {code_bytes} bytes per vector, {raw_bytes // code_bytes}× smaller\n"
        "Search with [bold]asymmetric distance[/bold]: keep the query exact; precompute its "
        f"dot product with every centroid in every subspace ({config.pq_subspaces} × "
        f"{config.pq_centroids} table); a vector's approximate score is the sum of "
        f"{config.pq_subspaces} table lookups. No multiplications per vector."
    )
    approx = np.zeros((config.n_queries, config.n_vectors), dtype=np.float32)
    for qi, query in enumerate(collection.queries):
        table_q = np.stack(
            [codebooks[s] @ query[s * sub : (s + 1) * sub] for s in range(config.pq_subspaces)]
        )
        approx[qi] = table_q[np.arange(config.pq_subspaces), codes].sum(axis=1)
    pq_top = np.argsort(-approx, axis=1)[:, : config.k]
    shortlist = np.argsort(-approx, axis=1)[:, : config.pq_rerank]
    reranked = np.zeros_like(pq_top)
    for qi, query in enumerate(collection.queries):
        ids = shortlist[qi]
        reranked[qi] = ids[np.argsort(-(collection.vectors[ids] @ query))[: config.k]]
    pq_recall = recall_at_k(pq_top, collection.exact)
    rerank_recall = recall_at_k(reranked, collection.exact)
    steps.quiz(
        "PQ scores are approximate. How do production systems get exact top results back?",
        [
            "They don't; approximate is final",
            f"Take the PQ top {config.pq_rerank}, then rescore those with the full vectors",
            "Use more subspaces until it is exact",
        ],
        1,
        "Rescoring a short list is cheap and fixes most ordering errors. The full vectors can "
        "live on SSD, since only a few are read per query.",
    )
    table = Table(header_style="bold", title="Product quantization (scans every code)")
    table.add_column("method")
    table.add_column(f"recall@{config.k}", justify="right")
    table.add_column("")
    table.add_column("bytes per vector", justify="right")
    table.add_row("PQ only", f"{pq_recall:.3f}", bar(pq_recall, 20), str(code_bytes))
    table.add_row(
        f"PQ + exact rerank of top {config.pq_rerank}",
        f"{rerank_recall:.3f}",
        bar(rerank_recall, 20),
        f"{code_bytes} in RAM, {raw_bytes} on disk",
    )
    table.add_row("brute force", "1.000", bar(1.0, 20), str(raw_bytes))
    console.print(table)
    console.print(
        "IVF and PQ combine: IVF-PQ opens a few cells and scans PQ codes inside them. That is "
        "how billion-scale indexes fit in memory on one machine (FAISS IndexIVFPQ)."
    )
    lookups = config.n_vectors * config.pq_subspaces / config.dim
    return [
        ("PQ only", pq_recall, lookups),
        (f"PQ + rerank {config.pq_rerank}", rerank_recall, lookups + config.pq_rerank),
    ]


def _step_tradeoff(
    console: Console,
    config: AnnConfig,
    ivf_rows: list[tuple[str, float, float]],
    graph_rows: list[tuple[str, float, float]],
    pq_rows: list[tuple[str, float, float]],
) -> None:
    table = Table(
        header_style="bold",
        title="Recall vs cost (cost in full-vector equivalents scored per query)",
    )
    table.add_column("index")
    table.add_column(f"recall@{config.k}", justify="right")
    table.add_column("cost", justify="right")
    table.add_column("cost vs brute force", justify="right")
    table.add_row("brute force", "1.000", f"{config.n_vectors:,}", "100%")
    for name, recall, cost in ivf_rows + graph_rows + pq_rows:
        table.add_row(name, f"{recall:.3f}", f"{cost:,.0f}", f"{cost / config.n_vectors:.1%}")
    console.print(table)
    console.print(
        "PQ cost counts table lookups as 1/d of a full dot product. There is no free lunch: "
        "each knob (nprobe, ef, rerank depth) buys recall with latency. In an interview, pick "
        "the operating point from the latency budget, then report recall at that point.\n"
        "Recall@10 of 0.95 means 1 in 20 true neighbors is lost before ranking even starts. "
        "Hybrid retrieval (blue belt) adds BM25 candidates, which also covers some ANN misses."
    )


def _step_production(console: Console) -> None:
    table = Table(header_style="bold")
    table.add_column("concern")
    table.add_column("what to know")
    table.add_row("libraries", "FAISS (flat, IVF, IVF-PQ, HNSW), ScaNN, hnswlib, DiskANN")
    table.add_row(
        "databases", "Vespa, Elasticsearch / OpenSearch kNN, Milvus, Weaviate, Qdrant, pgvector"
    )
    table.add_row(
        "filters",
        "'only Vijay movies': filtering after ANN loses results; filter during the search instead",
    )
    table.add_row(
        "updates",
        "graphs handle inserts; deletes are tombstones until a rebuild; IVF centroids drift",
    )
    table.add_row("sharding", "split by document, search every shard, merge the top k")
    table.add_row(
        "choice",
        "under ~1M vectors flat is fine; HNSW for low latency in RAM; IVF-PQ when memory is the limit",
    )
    console.print(table)
    console.print("[bold]Next[/bold]  uv run search-dojo lesson blue")
