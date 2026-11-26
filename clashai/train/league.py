"""The opponent decks PPO trains against, drawn toward the matchups Piggies EQ loses.

The pool is every opponent deck Piggies EQ meets in the labelled replays, as many times as it meets
it, less the held-out decks (:func:`clashai.decks.is_held_out`), which training never plays. A game's
deck is drawn from a mix of two distributions:

- with probability ``uniform``, as often as Piggies EQ meets it in the replays;
- otherwise, also weighted by how often the learner loses to it, so its worst matchups come up more.

Results are sparse (about 1,500 distinct decks), so a deck's loss rate is read over its
neighbourhood, the decks sharing at least ``share`` of its 8 cards, and shrunk toward the learner's
overall loss rate by ``prior`` games.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import paths
from ..decks import PIGGIES_EQ, is_held_out

DEFAULT_INDEX = "dataset/piggies_eq-screen-team0-v0-t0-standard/index.jsonl"


@dataclass(frozen=True)
class PoolDeck:
    cards: tuple[str, ...]
    slots: tuple[int, ...]
    met: int
    """how many times Piggies EQ meets it in the replays"""

    @property
    def mirror(self) -> bool:
        return set(self.cards) == set(PIGGIES_EQ.cards)


def opponent_decks(index: Path | None = None) -> tuple[list[PoolDeck], int]:
    """The opponent decks of every Piggies EQ seat in a labelled dataset's index, and the number of
    games against held-out decks left out."""
    index = index or paths.data_root() / DEFAULT_INDEX
    met: Counter = Counter()
    held = 0
    with open(index, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    for r in rows:
        for seat in r["piggies_seats"]:
            other = 1 - seat
            cards, slots = tuple(r["decks"][other]), tuple(r["slots"][other])
            if is_held_out(cards):
                held += 1
            else:
                met[(cards, slots)] += 1
    return [PoolDeck(c, s, n) for (c, s), n in sorted(met.items())], held


class League:
    def __init__(self, decks: list[PoolDeck], *, uniform: float = 0.5, power: float = 1.0,
                 floor: float = 0.05, share: int = 6, prior: float = 4.0):
        self.decks, self.uniform, self.power, self.floor = decks, uniform, power, floor
        self.prior = prior
        cards = sorted({c for d in decks for c in d.cards})
        col = {c: i for i, c in enumerate(cards)}
        m = np.zeros((len(decks), len(cards)), np.int32)
        for i, d in enumerate(decks):
            m[i, [col[c] for c in d.cards]] = 1
        self.near = (m @ m.T) >= share  # (D, D): each deck with itself and its neighbours
        self.met = np.array([d.met for d in decks], np.float64)
        self.games = np.zeros(len(decks))
        self.lost = np.zeros(len(decks))  # losses, a draw counting half

    def loss_rates(self) -> np.ndarray:
        games, lost = self.near @ self.games, self.near @ self.lost
        overall = (self.lost.sum() + 0.5 * self.prior) / (self.games.sum() + self.prior)
        return (lost + self.prior * overall) / (games + self.prior)

    def probabilities(self) -> np.ndarray:
        freq = self.met / self.met.sum()
        hard = freq * (self.loss_rates() + self.floor) ** self.power
        return self.uniform * freq + (1 - self.uniform) * hard / hard.sum()

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.choice(len(self.decks), n, p=self.probabilities())

    def record(self, deck: int, score: float) -> None:
        """A finished game against deck ``deck``: the learner's score, 1, 0.5 or 0."""
        self.games[deck] += 1
        self.lost[deck] += 1 - score

    def state_dict(self) -> dict:
        return {"games": self.games.tolist(), "lost": self.lost.tolist()}

    def load_state_dict(self, state: dict) -> None:
        self.games, self.lost = np.array(state["games"]), np.array(state["lost"])

    def hardest(self, k: int = 8) -> list[tuple[PoolDeck, float, float]]:
        """The decks drawn most above their replay frequency: (deck, loss rate, draw share)."""
        p, freq, rates = self.probabilities(), self.met / self.met.sum(), self.loss_rates()
        order = np.argsort(-(p / freq))[:k]
        return [(self.decks[i], float(rates[i]), float(p[i])) for i in order]
