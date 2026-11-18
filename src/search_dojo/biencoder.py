"""A tiny bi-encoder you can train between sips of coffee.

Both towers are the same embedding table. A text becomes the average of its
token vectors, then we L2-normalize and take a dot product, which is cosine.
Documents can be encoded once and stored. That is the whole trick: the
question and the scroll never meet until the last dot product.

Training uses two signals:

* a pairwise softplus loss that pulls the query mean toward the relevant
  scroll and away from a negative scroll
* an auxiliary pull that moves query-only tokens toward the rare tokens that
  appear in the relevant scroll and not the negative

The auxiliary pull is what makes ``pig`` a neighbor of ``tonkotsu``. Shared
words cannot do that job, because moving ``soup`` also moves every scroll
that says soup.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from search_dojo.tokenize import is_identifier, tokenize


@dataclass
class TinyBiEncoder:
    """Shared-weight bag-of-tokens encoder."""

    stoi: dict[str, int]
    itos: list[str]
    emb: np.ndarray
    initial: np.ndarray
    seed: int

    @classmethod
    def from_texts(cls, texts: list[str], *, dim: int, seed: int) -> TinyBiEncoder:
        """Build a random embedding row for every non-identifier token in ``texts``."""
        vocab = sorted(
            {token for text in texts for token in tokenize(text) if not is_identifier(token)}
        )
        if not vocab:
            raise ValueError("bi-encoder vocabulary is empty")
        rng = np.random.default_rng(seed)
        emb = rng.normal(0.0, 1.0, size=(len(vocab), dim)).astype(np.float64)
        encoder = cls(
            stoi={token: index for index, token in enumerate(vocab)},
            itos=vocab,
            emb=emb,
            initial=emb.copy(),
            seed=seed,
        )
        encoder._project()
        encoder.initial = encoder.emb.copy()
        return encoder

    @property
    def dim(self) -> int:
        return int(self.emb.shape[1])

    def encode(self, text: str) -> np.ndarray:
        """Mean-pool token vectors and L2-normalize.

        Identifiers and unknown tokens are skipped. An empty encoding is a
        zero vector, and its cosine with anything is zero.
        """
        vector = self._mean(text)
        return _normalize(vector)

    def score(self, left: str, right: str) -> float:
        """Cosine between two texts."""
        return float(self.encode(left) @ self.encode(right))

    def nearest(self, token: str, k: int = 5) -> list[tuple[str, float]]:
        """Tokens whose rows point the same way as ``token``, closest first."""
        index = self.stoi.get(token)
        if index is None:
            return []
        vector = _normalize(self.emb[index])
        if float(np.linalg.norm(vector)) == 0.0:
            return []
        matrix = np.vstack([_normalize(row) for row in self.emb])
        sims = matrix @ vector
        order = np.argsort(-sims)
        found: list[tuple[str, float]] = []
        for row in order:
            word = self.itos[int(row)]
            if word == token:
                continue
            found.append((word, float(sims[int(row)])))
            if len(found) == k:
                break
        return found

    def train(
        self,
        triples: list[tuple[str, str, str]],
        *,
        epochs: int,
        lr: float,
        aux_lr: float,
        aux_terms: int,
        df: dict[str, int],
        anchor: float,
    ) -> list[float]:
        """Train on ``(query, positive, negative)`` text triples.

        Returns:
            Mean pairwise loss per epoch. It should fall.
        """
        if not triples:
            raise ValueError("no training triples")
        rng = np.random.default_rng(self.seed)
        curve: list[float] = []
        for _ in range(epochs):
            total = 0.0
            seen = 0
            for index in rng.permutation(len(triples)):
                query, positive, negative = triples[int(index)]
                loss = self._step(query, positive, negative, lr, aux_lr, aux_terms, df)
                if loss is None:
                    continue
                total += loss
                seen += 1
                self._project()
            self.emb += anchor * (self.initial - self.emb)
            self._project()
            curve.append(total / seen if seen else 0.0)
        return curve

    def _step(
        self,
        query: str,
        positive: str,
        negative: str,
        lr: float,
        aux_lr: float,
        aux_terms: int,
        df: dict[str, int],
    ) -> float | None:
        q_ids = self._ids(query)
        p_ids = self._ids(positive)
        n_ids = self._ids(negative)
        if not q_ids or not p_ids or not n_ids:
            return None
        eq = self.emb[q_ids].mean(axis=0)
        ep = self.emb[p_ids].mean(axis=0)
        en = self.emb[n_ids].mean(axis=0)
        gq, gp, gn, loss = _pairwise_grads(eq, ep, en)
        self._scatter(q_ids, gq, lr)
        self._scatter(p_ids, gp, lr)
        self._scatter(n_ids, gn, lr)
        self._attract_query_tokens(q_ids, p_ids, n_ids, aux_lr, aux_terms, df)
        return loss

    def _attract_query_tokens(
        self,
        q_ids: list[int],
        p_ids: list[int],
        n_ids: list[int],
        lr: float,
        aux_terms: int,
        df: dict[str, int],
    ) -> None:
        """Move query tokens that are not in the negative toward rare positive tokens.

        Destination tokens stay put. Moving a shared word would drag every
        scroll that contains it, including the negative.
        """
        neg = set(n_ids)
        distinctive = [token_id for token_id in dict.fromkeys(p_ids) if token_id not in neg]
        distinctive.sort(key=lambda token_id: (df.get(self.itos[token_id], 0), self.itos[token_id]))
        targets = distinctive[:aux_terms]
        # Prefer query tokens the relevant scroll never used. Those are the paraphrase.
        sources = [token_id for token_id in dict.fromkeys(q_ids) if token_id not in set(p_ids)]
        if not sources:
            sources = [token_id for token_id in dict.fromkeys(q_ids) if token_id not in neg]
        if not sources or not targets:
            return
        dest = self.emb[targets].mean(axis=0)
        for token_id in sources:
            dot = float(self.emb[token_id] @ dest)
            # softplus(-dot): gradient pulls the query token toward dest.
            weight = 1.0 / (1.0 + math.exp(max(-20.0, min(20.0, dot))))
            self.emb[token_id] += lr * weight * dest

    def _ids(self, text: str) -> list[int]:
        return [self.stoi[token] for token in tokenize(text) if token in self.stoi]

    def _mean(self, text: str) -> np.ndarray:
        ids = self._ids(text)
        if not ids:
            return np.zeros(self.dim, dtype=np.float64)
        return self.emb[ids].mean(axis=0)

    def _scatter(self, ids: list[int], grad: np.ndarray, lr: float) -> None:
        if not ids:
            return
        step = (lr / len(ids)) * grad
        for token_id in ids:
            self.emb[token_id] -= step

    def _project(self) -> None:
        norms = np.linalg.norm(self.emb, axis=1, keepdims=True)
        safe = np.maximum(norms, 1e-8)
        self.emb = np.where(norms < 1e-8, self.emb, self.emb / safe)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def _pairwise_grads(
    eq: np.ndarray,
    ep: np.ndarray,
    en: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Gradients of ``softplus(-(s_pos - s_neg))`` for mean-pooled vectors.

    ``w`` is how badly the negative is winning. The query mean moves toward
    the positive and away from the negative.
    """
    diff = float(eq @ ep - eq @ en)
    diff = max(-20.0, min(20.0, diff))
    weight = 1.0 / (1.0 + math.exp(diff))
    gq = weight * (en - ep)
    gp = -weight * eq
    gn = weight * eq
    loss = math.log1p(math.exp(-diff))
    return gq, gp, gn, loss
