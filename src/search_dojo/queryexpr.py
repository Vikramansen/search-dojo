"""Boolean query expressions over posting lists.

``okapi ranking`` is an implicit AND. Quotes demand adjacent tokens.
``AND``, ``OR``, and ``NOT`` are the whole vocabulary. The tree is small on
purpose so you can read the parser in one sitting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from search_dojo.index import InvertedIndex
from search_dojo.tokenize import tokenize

_LEX = re.compile(r'"([^"]+)"|\b(AND|OR|NOT)\b|\(|\)|([A-Za-z0-9]+)', re.IGNORECASE)


class QuerySyntaxError(ValueError):
    """A query expression that does not parse."""


@dataclass(frozen=True, slots=True)
class Term:
    value: str


@dataclass(frozen=True, slots=True)
class Phrase:
    terms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Not:
    child: Expr


@dataclass(frozen=True, slots=True)
class And:
    children: tuple[Expr, ...]


@dataclass(frozen=True, slots=True)
class Or:
    children: tuple[Expr, ...]


Expr = Term | Phrase | Not | And | Or


def parse_query(text: str) -> Expr:
    """Parse a query expression.

    Raises:
        QuerySyntaxError: The string is empty or the tree does not close.
    """
    parser = _Parser(_lex(text))
    expr = parser.parse_or()
    if parser.peek() is not None:
        raise QuerySyntaxError(f"unexpected token {parser.peek()}")
    return expr


def format_expr(expr: Expr) -> str:
    """Render a parsed expression with explicit parentheses."""
    if isinstance(expr, Term):
        return expr.value
    if isinstance(expr, Phrase):
        return '"' + " ".join(expr.terms) + '"'
    if isinstance(expr, Not):
        return f"(NOT {format_expr(expr.child)})"
    if isinstance(expr, And):
        return "(" + " AND ".join(format_expr(child) for child in expr.children) + ")"
    return "(" + " OR ".join(format_expr(child) for child in expr.children) + ")"


def match_expr(index: InvertedIndex, expr: Expr) -> set[str]:
    """Evaluate ``expr`` against posting lists. ``NOT`` is the archive minus the child."""
    if isinstance(expr, Term):
        return set(index.tf.get(expr.value, {}))
    if isinstance(expr, Phrase):
        return index.phrase_docs(expr.terms)
    if isinstance(expr, Not):
        return set(index.doc_ids) - match_expr(index, expr.child)
    if isinstance(expr, And):
        sets = [match_expr(index, child) for child in expr.children]
        return set.intersection(*sets) if sets else set()
    sets = [match_expr(index, child) for child in expr.children]
    return set.union(*sets) if sets else set()


def content_terms(expr: Expr) -> list[str]:
    """Term and phrase tokens, for ranking the matched set with BM25."""
    if isinstance(expr, Term):
        return [expr.value] if expr.value else []
    if isinstance(expr, Phrase):
        return list(expr.terms)
    if isinstance(expr, Not):
        return content_terms(expr.child)
    return [term for child in expr.children for term in content_terms(child)]


def looks_like_expr(text: str) -> bool:
    """True when the searcher used quotes or a boolean operator."""
    if '"' in text:
        return True
    padded = f" {text.upper()} "
    return any(op in padded for op in (" AND ", " OR ", " NOT "))


class _Parser:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.index = 0

    def peek(self) -> str | None:
        if self.index >= len(self.tokens):
            return None
        return self.tokens[self.index]

    def eat(self, expected: str | None = None) -> str:
        token = self.peek()
        if token is None:
            raise QuerySyntaxError("unexpected end of query")
        if expected is not None and token != expected:
            raise QuerySyntaxError(f"expected {expected}, found {token}")
        self.index += 1
        return token

    def parse_or(self) -> Expr:
        node = self.parse_and()
        nodes = [node]
        while self.peek() == "OR":
            self.eat("OR")
            nodes.append(self.parse_and())
        if len(nodes) == 1:
            return node
        return Or(tuple(nodes))

    def parse_and(self) -> Expr:
        node = self.parse_not()
        nodes = [node]
        while True:
            if self.peek() == "AND":
                self.eat("AND")
                nodes.append(self.parse_not())
            elif self.peek() not in {None, ")", "OR"}:
                nodes.append(self.parse_not())
            else:
                break
        if len(nodes) == 1:
            return node
        return And(tuple(nodes))

    def parse_not(self) -> Expr:
        if self.peek() == "NOT":
            self.eat("NOT")
            return Not(self.parse_not())
        return self.parse_primary()

    def parse_primary(self) -> Expr:
        token = self.peek()
        if token is None:
            raise QuerySyntaxError("unexpected end of query")
        if token == "(":
            self.eat("(")
            node = self.parse_or()
            self.eat(")")
            return node
        if token.startswith('"'):
            self.eat()
            terms = tuple(tokenize(token[1:-1]))
            if not terms:
                raise QuerySyntaxError("empty phrase")
            return Phrase(terms)
        if token in {")", "AND", "OR", "NOT"}:
            raise QuerySyntaxError(f"unexpected token {token}")
        word = self.eat()
        stemmed = tokenize(word)
        if not stemmed:
            return Term("")
        return Term(stemmed[0])


def _lex(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _LEX.finditer(text):
        phrase, operator, word = match.group(1), match.group(2), match.group(3)
        if phrase is not None:
            tokens.append('"' + phrase + '"')
        elif operator is not None:
            tokens.append(operator.upper())
        elif word is not None:
            tokens.append(word)
    if not tokens:
        raise QuerySyntaxError("empty query")
    return tokens
