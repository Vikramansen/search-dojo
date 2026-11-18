"""Load the Midnight Archive. Scrolls, drills, challenges, logs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROMPT_PATH = ROOT / "prompts" / "rewrite.txt"


@dataclass(frozen=True, slots=True)
class Scroll:
    """One document in the archive."""

    id: str
    title: str
    text: str
    tags: tuple[str, ...]

    @property
    def body(self) -> str:
        """Title plus text. This is what the index actually reads."""
        return f"{self.title}. {self.text}"


@dataclass(frozen=True, slots=True)
class Drill:
    """A practice triple for the bi-encoder: query, relevant scroll, hard negatives."""

    query: str
    positive: str
    negatives: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Challenge:
    """A graded question. The boss fight averages these."""

    id: str
    belt: str
    query: str
    qrels: dict[str, int]
    note: str


@dataclass(frozen=True, slots=True)
class LogEvent:
    """One line of a searcher's session."""

    session: str
    query: str
    shown: int
    click: str | None
    rank: int | None


def _rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"missing archive file: {path}")
    loaded: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        item = json.loads(stripped)
        if not isinstance(item, dict):
            raise ValueError(f"{path.name}:{line_number} is not an object")
        loaded.append(item)
    return loaded


def load_scrolls(path: Path | None = None) -> list[Scroll]:
    """Load scrolls and reject duplicate ids."""
    scrolls = [
        Scroll(
            id=str(row["id"]),
            title=str(row["title"]),
            text=str(row["text"]),
            tags=tuple(str(tag) for tag in row.get("tags", [])),
        )
        for row in _rows(path or DATA_DIR / "scrolls.jsonl")
    ]
    ids = [scroll.id for scroll in scrolls]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate scroll id in scrolls.jsonl")
    return scrolls


def load_drills(path: Path | None = None) -> list[Drill]:
    """Load bi-encoder practice drills."""
    return [
        Drill(
            query=str(row["query"]),
            positive=str(row["positive"]),
            negatives=tuple(str(item) for item in row["negatives"]),
        )
        for row in _rows(path or DATA_DIR / "drills.jsonl")
    ]


def load_challenges(path: Path | None = None) -> list[Challenge]:
    """Load graded queries."""
    challenges: list[Challenge] = []
    for row in _rows(path or DATA_DIR / "challenges.jsonl"):
        raw_qrels = row["qrels"]
        if not isinstance(raw_qrels, dict):
            raise ValueError(f"challenge {row.get('id')} qrels must be an object")
        qrels = {str(doc_id): int(grade) for doc_id, grade in raw_qrels.items()}
        if any(grade <= 0 for grade in qrels.values()):
            raise ValueError(f"challenge {row.get('id')} grades must be positive")
        challenges.append(
            Challenge(
                id=str(row["id"]),
                belt=str(row["belt"]),
                query=str(row["query"]),
                qrels=qrels,
                note=str(row.get("note", "")),
            )
        )
    return challenges


def load_logs(path: Path | None = None) -> list[LogEvent]:
    """Load query-log events in file order. Order inside a session matters."""
    events: list[LogEvent] = []
    for row in _rows(path or DATA_DIR / "query_logs.jsonl"):
        click = row.get("click")
        rank = row.get("rank")
        click_id = None if click is None else str(click)
        rank_n = None if rank is None else int(rank)
        if (click_id is None) != (rank_n is None):
            raise ValueError(f"session {row.get('session')} click and rank must both be set")
        events.append(
            LogEvent(
                session=str(row["session"]),
                query=str(row["query"]),
                shown=int(row["shown"]),
                click=click_id,
                rank=rank_n,
            )
        )
    return events


def assert_references(
    scrolls: list[Scroll],
    drills: list[Drill],
    challenges: list[Challenge],
    logs: list[LogEvent],
) -> None:
    """Raise if a drill, qrel, or click points at a scroll that does not exist."""
    known = {scroll.id for scroll in scrolls}

    def _check(doc_id: str, where: str) -> None:
        if doc_id not in known:
            raise ValueError(f"{where} references unknown scroll {doc_id}")

    for drill in drills:
        _check(drill.positive, f"drill {drill.query!r}")
        for negative in drill.negatives:
            _check(negative, f"drill {drill.query!r}")
    for challenge in challenges:
        for doc_id in challenge.qrels:
            _check(doc_id, f"challenge {challenge.id}")
    for event in logs:
        if event.click is not None:
            _check(event.click, f"log {event.session}")
