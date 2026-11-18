"""The archive text has to keep the stories the belts tell."""

from search_dojo.corpus import (
    assert_references,
    load_challenges,
    load_drills,
    load_logs,
    load_scrolls,
)
from search_dojo.tokenize import is_identifier, tokenize


def test_references_resolve() -> None:
    scrolls = load_scrolls()
    drills = load_drills()
    challenges = load_challenges()
    logs = load_logs()
    assert_references(scrolls, drills, challenges, logs)
    assert len({scroll.id for scroll in scrolls}) == len(scrolls)


def test_story_tokens() -> None:
    scrolls = {scroll.id: scroll for scroll in load_scrolls()}
    dashi = tokenize(scrolls["dashi"].body)
    animal = tokenize(scrolls["okapi-animal"].body)
    license_tokens = tokenize(scrolls["license-sku"].body)
    tonkotsu = tokenize(scrolls["tonkotsu"].body)
    chant = tokenize(scrolls["chant"].body)
    assert "milky" not in dashi
    assert "pork" not in dashi
    assert "ranking" not in animal
    assert "function" not in animal
    assert "sku1844" in license_tokens
    assert is_identifier("sku1844")
    assert chant.count("soup") >= 4
    for term in ("milky", "pork", "bone", "broth", "opaque"):
        assert term in tonkotsu
