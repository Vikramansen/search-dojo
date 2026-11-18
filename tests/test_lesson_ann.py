from dataclasses import replace

import numpy as np
from rich.console import Console

from search_dojo.lesson_ann import (
    AnnConfig,
    _ivf_search,
    build_collection,
    build_graph,
    graph_search,
    kmeans,
    recall_at_k,
)
from search_dojo.lessons import run_lesson

SMALL = replace(AnnConfig(), n_vectors=2000, n_queries=20, ivf_lists=16, n_topics=40)


def test_recall_at_k() -> None:
    assert recall_at_k(np.array([[1, 2]]), np.array([[2, 3]])) == 0.5


def test_kmeans_assigns_every_point() -> None:
    data = np.vstack([np.zeros((10, 2)), np.ones((10, 2)) * 5])
    centroids, assign = kmeans(data, 2, 5, np.random.default_rng(0))
    assert len(set(assign[:10])) == 1 and len(set(assign[10:])) == 1
    assert assign[0] != assign[10]


def test_ivf_full_probe_is_exact() -> None:
    collection = build_collection(SMALL)
    centroids, assign = kmeans(collection.vectors, SMALL.ivf_lists, 5, np.random.default_rng(0))
    lists = [np.flatnonzero(assign == c) for c in range(SMALL.ivf_lists)]
    recall, scored = _ivf_search(collection, centroids, lists, SMALL.ivf_lists, SMALL)
    assert recall == 1.0
    assert scored == SMALL.n_vectors + SMALL.ivf_lists


def test_graph_search_recall_grows_with_ef() -> None:
    collection = build_collection(SMALL)
    graph = build_graph(collection.vectors, SMALL)
    recalls = []
    for ef in (10, 80):
        found = np.zeros((SMALL.n_queries, SMALL.k), dtype=int)
        for qi, query in enumerate(collection.queries):
            ids, _ = graph_search(collection.vectors, graph, query, 0, ef, SMALL.k)
            found[qi, : len(ids)] = ids
        recalls.append(recall_at_k(found, collection.exact))
    assert recalls[1] >= recalls[0]
    assert recalls[1] > 0.8


def test_ann_lesson_runs() -> None:
    console = Console(record=True, width=120)
    run_lesson(console, "ann", pause=False)
    assert "product quantization" in console.export_text().lower()
