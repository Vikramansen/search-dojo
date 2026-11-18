"""The Dojo DB: nine short documents the belts walk through by hand."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from search_dojo.corpus import DATA_DIR, _rows


@dataclass(frozen=True, slots=True)
class MiniDoc:
    """One document: an id, a topic label for display, and the text that gets indexed."""

    id: str
    topic: str
    text: str


def load_mini(path: Path | None = None) -> list[MiniDoc]:
    """Load the Dojo DB and reject duplicate ids."""
    docs = [
        MiniDoc(id=str(row["id"]), topic=str(row["topic"]), text=str(row["text"]))
        for row in _rows(path or DATA_DIR / "dojo_db.jsonl")
    ]
    ids = [doc.id for doc in docs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate id in dojo_db.jsonl")
    return docs
