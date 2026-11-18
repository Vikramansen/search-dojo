"""PRF on a toy index, and the LLM socket."""

from search_dojo.corpus import PROMPT_PATH
from search_dojo.expand import llm_rewrite, prf_terms
from search_dojo.index import InvertedIndex


def test_prf_pulls_distinctive_terms() -> None:
    index = InvertedIndex(
        [
            ("soup", "soup soup soup basil"),
            ("spill", "soup bleach ammonia"),
        ]
    )
    terms = prf_terms(index, "soup", k_docs=2, k_terms=4)
    assert "bleach" in terms
    assert "soup" not in terms


def test_llm_rewrite_fills_the_prompt() -> None:
    seen: dict[str, str] = {}

    def complete(prompt: str) -> str:
        seen["prompt"] = prompt
        return "  cloudy pig soup tonkotsu broth  "

    rewritten = llm_rewrite("cloudy pig soup", complete, PROMPT_PATH)
    assert rewritten == "cloudy pig soup tonkotsu broth"
    assert "cloudy pig soup" in seen["prompt"]
    assert "Do not answer the question." in seen["prompt"]
