"""Reciprocal rank fusion and scaled linear fusion."""

from search_dojo.hybrid import linear_fuse, reciprocal_rank_fusion


def test_document_in_both_lists_beats_a_one_list_hit() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60))
    assert fused["b"] > fused["a"]
    assert fused["b"] > fused["c"]


def test_rrf_uses_one_based_ranks() -> None:
    fused = dict(reciprocal_rank_fusion([["only"]], k=60))
    assert fused["only"] == 1.0 / 61.0


def test_linear_fuse_needs_a_shared_scale() -> None:
    fused = dict(linear_fuse({"a": 100.0, "b": 0.0}, {"a": 0.0, "b": 1.0}, alpha=0.5))
    assert fused["a"] == fused["b"]
