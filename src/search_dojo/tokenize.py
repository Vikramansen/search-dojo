"""Turn scroll text into the tokens every belt shares.

The list is intentionally small and a little stubborn: lowercase, drop a
short stop list, strip a trailing plural ``s``, and throw away the word
after ``not`` / ``no`` / ``never`` / ``without``. Edit it. The belts will
notice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN = re.compile(r"[a-z0-9]+")

STOP = frozenset(
    {
        "a",
        "about",
        "after",
        "also",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "before",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "down",
        "each",
        "for",
        "from",
        "had",
        "has",
        "have",
        "here",
        "how",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "just",
        "more",
        "most",
        "of",
        "on",
        "only",
        "or",
        "other",
        "out",
        "over",
        "own",
        "same",
        "should",
        "so",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "too",
        "under",
        "up",
        "very",
        "was",
        "were",
        "what",
        "when",
        "which",
        "while",
        "who",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)

NEGATIONS = frozenset({"not", "no", "never", "without"})


# Plural "es" after a sibilant: punishes -> punish, boxes -> box. "bones" does
# not match these endings, so it still loses only the final s.
_ES_PLURAL = ("shes", "ches", "sses", "xes", "zes")


def stem(token: str) -> str:
    """Strip a simple plural ending.

    ``bones`` becomes ``bone``. ``punishes`` becomes ``punish``. ``class``
    stays ``class``. This is a teaching stemmer, not a claim about English.
    """
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    for ending in _ES_PLURAL:
        if len(token) > len(ending) + 1 and token.endswith(ending):
            return token[:-2]
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def is_identifier(token: str) -> bool:
    """True for product codes the dense vocab is told to forget.

    BM25 keeps these. A word-averaging encoder would memorize ``sku1844`` as
    if it were a word, which is the opposite of what happens with a real
    model's vocabulary cutoff. We drop them on purpose so hybrid retrieval
    has a rare string only the inverted index can see.
    """
    return token.startswith("sku") or token.isdigit()


def tokenize(text: str, *, drop_negated: bool = True) -> list[str]:
    """Tokenize ``text`` the way the archive indexes it.

    Args:
        text: Raw scroll or query text.
        drop_negated: When true, ``never milky`` indexes neither word.

    Returns:
        Stemmed content tokens, in order, with repeats preserved.
    """
    raw = _TOKEN.findall(text.lower())
    kept: list[str] = []
    index = 0
    while index < len(raw):
        token = raw[index]
        if drop_negated and token in NEGATIONS:
            index += 1
            while index < len(raw) and raw[index] in STOP:
                index += 1
            index += 1
            continue
        if token in STOP:
            index += 1
            continue
        kept.append(stem(token))
        index += 1
    return kept


@dataclass(frozen=True, slots=True)
class TokenTrace:
    """Each stage of ``tokenize``, kept so a lesson can print them."""

    lowered: str
    split: list[str]
    without_stop: list[str]
    stemmed: list[str]


def trace_tokenize(text: str) -> TokenTrace:
    """Run the tokenizer one stage at a time.

    Matches ``tokenize`` for text without negation words. The negation rule
    is left out so the four stages stay readable.
    """
    lowered = text.lower()
    split = _TOKEN.findall(lowered)
    without_stop = [token for token in split if token not in STOP]
    return TokenTrace(
        lowered=lowered,
        split=split,
        without_stop=without_stop,
        stemmed=[stem(token) for token in without_stop],
    )


def unique_terms(text: str) -> list[str]:
    """Return stemmed tokens once each, in first-seen order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for token in tokenize(text):
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered
