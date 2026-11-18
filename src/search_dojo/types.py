"""Shared records passed between belts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Hit:
    """One ranked scroll."""

    doc_id: str
    score: float
    reason: str = ""
    features: dict[str, float] = field(default_factory=dict)

    def with_reason(self, reason: str) -> Hit:
        """Return a copy with a new explanation string."""
        return Hit(self.doc_id, self.score, reason, dict(self.features))
