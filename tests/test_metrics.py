"""NDCG, MRR, and recall."""

from search_dojo.metrics import mrr, ndcg_at_k, recall_at_k


def test_perfect_ndcg_is_one() -> None:
    qrels = {"a": 2, "b": 1}
    assert ndcg_at_k(["a", "b", "c"], qrels, k=5) == 1.0


def test_reversed_ndcg_is_worse() -> None:
    qrels = {"a": 2, "b": 1}
    perfect = ndcg_at_k(["a", "b"], qrels, k=5)
    flipped = ndcg_at_k(["b", "a"], qrels, k=5)
    assert flipped < perfect


def test_mrr_and_recall() -> None:
    qrels = {"a": 2, "b": 1}
    assert mrr(["z", "a"], qrels) == 0.5
    assert recall_at_k(["a"], qrels, k=1) == 0.5
    assert recall_at_k(["a", "b"], qrels, k=2) == 1.0
    assert mrr(["z"], qrels) == 0.0
