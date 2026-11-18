"""Tokenizer contracts the belts rely on."""

from search_dojo.tokenize import is_identifier, stem, tokenize, unique_terms


def test_stem_strips_plural_s_and_keeps_ss() -> None:
    assert stem("bones") == "bone"
    assert stem("hours") == "hour"
    assert stem("class") == "class"
    assert stem("sku1844") == "sku1844"


def test_negation_drops_the_following_content_word() -> None:
    assert "milky" not in tokenize("The bowl stays clear, never milky.")
    assert "pork" not in tokenize("never pork")
    assert tokenize("not a ranking function") == ["function"]


def test_unique_terms_keep_first_order() -> None:
    assert unique_terms("Soup soup bone") == ["soup", "bone"]


def test_identifiers_are_product_codes() -> None:
    assert is_identifier("sku1844")
    assert is_identifier("1844")
    assert not is_identifier("tonkotsu")
