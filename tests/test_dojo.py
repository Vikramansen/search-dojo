"""The belts have to do what they claim on the Midnight Archive."""

from __future__ import annotations

from search_dojo.archive import Archive
from search_dojo.metrics import recall_at_k
from search_dojo.queryexpr import match_expr, parse_query


def _ids(archive: Archive, query: str, retriever: str, k: int = 8) -> list[str]:
    return [hit.doc_id for hit in archive.search(query, retriever, k=k)]


def _place(ids: list[str], doc_id: str) -> int:
    return ids.index(doc_id) if doc_id in ids else 999


def test_okapi_trap_prefers_the_function(archive: Archive) -> None:
    ranking = _ids(archive, "okapi ranking function", "bm25")
    assert ranking[0] == "bm25-scroll"


def test_bm25_hears_the_chant(archive: Archive) -> None:
    query = "milky pork bone soup"
    ranking = _ids(archive, query, "bm25", k=15)
    assert _place(ranking, "chant") < _place(ranking, "tonkotsu")


def test_dense_crosses_the_vocabulary_gap(archive: Archive) -> None:
    query = "cloudy pig soup"
    dense = _ids(archive, query, "dense")
    lexical = _ids(archive, query, "bm25", k=15)
    assert dense[0] == "tonkotsu", dense
    assert "tonkotsu" not in lexical[:5]


def test_hybrid_covers_code_and_paraphrase(archive: Archive) -> None:
    query = "sku1844 cloudy pig soup"
    qrels = {"license-sku": 2, "tonkotsu": 2}
    bm25 = _ids(archive, query, "bm25")
    dense = _ids(archive, query, "dense")
    hybrid = _ids(archive, query, "hybrid")
    assert recall_at_k(hybrid, qrels, k=5) > recall_at_k(bm25, qrels, k=5)
    assert recall_at_k(hybrid, qrels, k=5) > recall_at_k(dense, qrels, k=5)
    assert "sku1844" not in archive.encoder.stoi


def test_rerank_puts_broth_above_the_chant(archive: Archive) -> None:
    query = "milky pork bone soup"
    ranking = _ids(archive, query, "rerank")
    assert _place(ranking, "tonkotsu") < _place(ranking, "chant"), ranking


def test_expansion_lifts_the_broth_into_bm25(archive: Archive) -> None:
    query = "cloudy pig soup"
    rewritten = archive.expand(query)
    assert rewritten != query
    ranking = [hit.doc_id for hit in archive.search_expanded_bm25(query, k=5)]
    assert "tonkotsu" in ranking, rewritten


def test_strong_lexical_query_is_not_rewritten(archive: Archive) -> None:
    query = "okapi ranking function"
    assert archive.expand(query) == query


def test_prf_can_poison_and_the_default_path_does_not(archive: Archive) -> None:
    feedback = archive.prf("milky soup")
    assert "bleach" in feedback
    assert "bleach" not in archive.expand("milky soup")


def test_log_lexicon_keeps_the_gap_and_vetoes_the_spill(archive: Archive) -> None:
    kept = {item.before for item in archive.kept_reforms}
    suspect = {item.before for item in archive.suspect_reforms}
    assert "white pig soup" in kept
    assert "milky soup" in suspect
    assert "bleach" not in " ".join(archive.lexicon.get("white pig soup", []))


def test_phrase_expression_hits_the_broth(archive: Archive) -> None:
    matched = match_expr(archive.index, parse_query('"opaque white broth" AND pork'))
    assert "tonkotsu" in matched
    assert "chant" not in matched


def test_pipeline_recall_beats_bm25(archive: Archive) -> None:
    from search_dojo.metrics import score_run

    qrels = {item.id: item.qrels for item in archive.challenges}
    k = archive.config.result_k

    def report(name: str):
        rankings = {
            item.id: [hit.doc_id for hit in archive.search(item.query, name, k=k)]
            for item in archive.challenges
        }
        return score_run(name, rankings, qrels, k)

    bm25 = report("bm25")
    pipeline = report("pipeline")
    assert pipeline.mean_recall >= bm25.mean_recall
    assert pipeline.mean_ndcg >= bm25.mean_ndcg


def test_loss_falls(archive: Archive) -> None:
    curve = archive.loss_curve
    assert len(curve) > 2
    assert curve[-1] < curve[0]
