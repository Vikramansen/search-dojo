"""Turn scroll text into the tokens every belt shares.

The list is intentionally small and a little stubborn: lowercase, drop a
short stop list, strip a trailing plural ``s``, and throw away the word
after ``not`` / ``no`` / ``never`` / ``without``. Edit it. The belts will
notice.
"""

from __future__ import annotations

import re

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


def stem(token: str) -> str:
    """Strip a trailing plural ``s`` from longer tokens.

    ``bones`` becomes ``bone``. ``class`` stays ``class``. This is a teaching
    stemmer, not a claim about English.
    """
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
