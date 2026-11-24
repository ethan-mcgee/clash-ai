"""Matches between two policies on paired seeds.

Every seed is played twice, once with each side on seat 0 and once on seat 1. Both games of a
pair start from the same battle seed, so luck in the battle's random draws (the shuffle, random
delays and spreads) is shared between the two sides instead of deciding the result.

Results are reproducible: the same sides, seeds, batch size and ``rng_seed`` give the same games.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..decks import Deck
from ..sim.env import make_vector


@dataclass(frozen=True)
class Side:
    policy: Any
    deck: Deck
    setup: Any = None
    """A crforge ``SideSetup`` (tower troop and levels), or None for the vector's defaults."""


@dataclass(frozen=True)
class Game:
    seed: int
    a_seat: int
    score: float | None
    """Side A's result: 1 win, 0.5 draw, 0 loss, None for no result (truncated or faulted)."""
    crowns_a: int
    crowns_b: int
    ticks: int
    fault: str


def play(a: Side, b: Side, seeds: Sequence[int], *, batch: int = 256, rng_seed: int = 0,
         threads: int | None = None, **vector_options) -> list[Game]:
    """Plays side A against side B on every seed from both seats; returns the games in order."""
    games = [(int(seed), seat) for seed in seeds for seat in (0, 1)]
    if not games:
        return []
    n = min(batch, len(games))
    out: list[Game] = []
    with make_vector(n, threads, **vector_options) as v:
        s = v.schema
        info = s.info
        a.policy.bind(v)
        b.policy.bind(v)
        for start in range(0, len(games), n):
            chunk = games[start:start + n]
            real = len(chunk)
            chunk = chunk + [chunk[0]] * (n - real)  # padding, played but not counted
            a_seat = np.array([seat for _, seat in chunk])
            on = [[a if seat_of_a == seat else b for seat_of_a in a_seat] for seat in (0, 1)]
            decks = [[list(side.deck.cards) for side in on[seat]] for seat in (0, 1)]
            slots = [[side.deck.slots(s.slot_evolution) for side in on[seat]] for seat in (0, 1)]
            setups = [None if a.setup is None and b.setup is None
                      else [side.setup for side in on[seat]] for seat in (0, 1)]
            v.reset([seed for seed, _ in chunk], decks[0], decks[1], slots[0], slots[1],
                    setups[0], setups[1])
            finished = np.arange(n) >= real
            rng_a = np.random.default_rng([rng_seed, start, 0])
            rng_b = np.random.default_rng([rng_seed, start, 1])
            results: dict[int, Game] = {}
            while not finished.all():
                live = np.flatnonzero(~finished)
                actions = np.zeros((n, 2), np.int32)
                for side, seats, rng in ((a, a_seat, rng_a), (b, 1 - a_seat, rng_b)):
                    seat = seats[live]
                    actions[live, seat] = side.policy.act(v.observations[live, seat],
                                                          v.masks[live, seat], rng)
                v.actions[:] = actions
                v.step()
                for i in np.flatnonzero(v.dones.astype(bool) & ~finished):
                    finished[i] = True
                    row = v.infos[i]
                    winner, seat = int(row[info["winner"]]), int(a_seat[i])
                    if winner == seat:
                        score = 1.0
                    elif winner == 1 - seat:
                        score = 0.0
                    elif winner == s.winner_draw:
                        score = 0.5
                    else:
                        score = None
                    crowns = (int(row[info["crowns_0"]]), int(row[info["crowns_1"]]))
                    results[i] = Game(chunk[i][0], seat, score, crowns[seat], crowns[1 - seat],
                                      int(row[info["tick"]]), s.fault_names[int(row[info["fault"]])])
            out += [results[i] for i in range(real)]
    return out
