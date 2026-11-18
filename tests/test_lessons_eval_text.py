from rich.console import Console

from search_dojo.lesson_eval import QUERY_ID
from search_dojo.lesson_text import NaiveBayes, Trie, _candidates, _corpus_counts, _levenshtein
from search_dojo.lessons import run_lesson
from search_dojo.metrics import average_precision, ndcg_at_k, precision_at_k
from search_dojo.world import build_world


def _run(belt: str) -> str:
    console = Console(record=True, width=120)
    run_lesson(console, belt, pause=False)
    return console.export_text()


def test_eval_story_matches_the_numbers() -> None:
    world = build_world()
    query = world.by_query[QUERY_ID]
    ranked = world.bm25_ids(query.text, 10)
    assert ranked.index("csk-five-titles") == 4
    assert ranked[3] == "warriors-dynasty"
    assert round(ndcg_at_k(ranked, query.qrels, 5), 3) == 0.699
    assert 0.0 < average_precision(ranked, query.qrels) <= 1.0
    assert precision_at_k(ranked, query.qrels, 3) == 1.0


def test_eval_lesson_runs() -> None:
    text = _run("eval")
    assert "NDCG@5 = 7.839 / 11.210 = 0.699" in text
    assert "95% interval" in text


def test_levenshtein() -> None:
    assert _levenshtein("haland", "haaland") == 1
    assert _levenshtein("meesseeks", "meeseeks") == 1
    assert _levenshtein("", "abc") == 3


def test_spelling_finds_the_fix() -> None:
    counts = _corpus_counts(build_world())
    assert _candidates("haland", counts)[0][0] == "haaland"
    assert _candidates("thalapthy", counts)[0][0] == "thalapathy"


def test_naive_bayes_prefers_record_for_warriors() -> None:
    world = build_world()
    classifier = NaiveBayes.train([(doc.text, doc.topic) for doc in world.docs])
    posterior = classifier.posterior("kings record")
    assert max(posterior, key=posterior.__getitem__) == "Golden State Warriors"
    assert abs(sum(posterior.values()) - 1.0) < 1e-9


def test_trie_ranks_by_count() -> None:
    trie = Trie()
    for query, count in (("thala", 3), ("three", 9), ("the", 1), ("cat", 5)):
        trie.insert(query, count)
    assert trie.complete("th") == [("three", 9), ("thala", 3), ("the", 1)]
    assert trie.complete("x") == []


def test_text_lesson_runs() -> None:
    text = _run("text")
    assert "haaland" in text
    assert "0.00 → 1.00" in text
