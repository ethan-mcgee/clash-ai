"""Matches between two policies on paired seeds.

Every seed is played twice, once with each side on seat 0 and once on seat 1. Both games of a
pair start from the same battle seed, so luck in the battle's random draws (the shuffle, random
delays and spreads) is shared between the two sides instead of deciding the result.

Results are reproducible: the same sides, seeds, batch size and ``rng_seed`` give the same games.

:func:`play_matches` plays several matches (a scorecard's decks) side by side in one vector, so
the engine's threads and the GPU get every battle's work at each decision instead of one deck's.
A single match plays exactly as :func:`play` always has.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..decks import Deck
from ..sim.env import DECISION_TICKS, make_vector


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
    behaviour: dict | None = None
    """Side A's play in this game (:mod:`clashai.eval.behaviour`), when asked for."""


def play(a: Side, b: Side, seeds: Sequence[int], *, batch: int = 256, rng_seed: int = 0,
         threads: int | None = None, behaviour: bool = False, **vector_options) -> list[Game]:
    """Plays side A against side B on every seed from both seats; returns the games in order."""
    return play_matches([(a, b)], seeds, batch=batch, rng_seed=rng_seed, threads=threads,
                        behaviour=behaviour, **vector_options)[0]


def play_matches(matches: Sequence[tuple[Side, Side]], seeds: Sequence[int], *, batch: int = 1024,
                 rng_seed: int = 0, threads: int | None = None, behaviour: bool = False,
                 **vector_options) -> list[list[Game]]:
    """Plays every match (side A, side B) on every seed from both seats, all in one vector of up
    to ``batch`` battles; returns each match's games in order of seed, then seat.

    Each policy acts once per decision on all of its rows. Rows are grouped by role (A or B) and
    policy object, each group drawing from its own generator, so one match reproduces
    :func:`play`'s games exactly. With ``behaviour`` each game records side A's play.

    A ``stateful`` policy (a recurrent network) is reset with the vector's size before each chunk
    and told which seat each row is (``keys``: battle row * 2 + seat).
    """
    seeds = [int(seed) for seed in seeds]
    games = [(m, seed, seat) for m in range(len(matches)) for seed in seeds for seat in (0, 1)]
    out: list[list[Game]] = [[] for _ in matches]
    if not games:
        return out
    chunks = -(-len(games) // batch)
    n = -(-len(games) // chunks)  # even chunks: little padding
    groups: dict[tuple[int, int], int] = {}
    policies: dict[int, Any] = {}
    for role in (0, 1):
        for match in matches:
            groups.setdefault((role, id(match[role].policy)), len(groups))
            policies.setdefault(id(match[role].policy), match[role].policy)
    with make_vector(n, threads, **vector_options) as v:
        s = v.schema
        info = s.info
        for policy in policies.values():
            policy.bind(v)
        stateful = {key for key, policy in policies.items() if getattr(policy, "stateful", False)}
        tracker = None
        if behaviour:
            from .behaviour import Behaviour

            tracker = Behaviour(s.header, v.vocabulary, s.hand_slots, s.cols * s.rows,
                                s.first_ability, vector_options.get("decision_ticks", DECISION_TICKS), n)
        hi = s.header_ints
        for start in range(0, len(games), n):
            chunk = games[start:start + n]
            real = len(chunk)
            chunk = chunk + [chunk[0]] * (n - real)  # padding, played but not counted
            match_of = np.array([m for m, _, _ in chunk])
            a_seat = np.array([seat for _, _, seat in chunk])
            role_seat = np.stack([a_seat, 1 - a_seat])  # (2, n): each role's seat
            role_group = np.array([[groups[(r, id(matches[m][r].policy))] for m in match_of]
                                   for r in (0, 1)])
            on = [[matches[m][0] if a_seat[i] == seat else matches[m][1]
                   for i, m in enumerate(match_of)] for seat in (0, 1)]
            decks = [[list(side.deck.cards) for side in on[seat]] for seat in (0, 1)]
            slots = [[side.deck.slots(s.slot_evolution) for side in on[seat]] for seat in (0, 1)]
            setups = [None if all(side.setup is None for side in on[0] + on[1])
                      else [side.setup for side in on[seat]] for seat in (0, 1)]
            v.reset([seed for _, seed, _ in chunk], decks[0], decks[1], slots[0], slots[1],
                    setups[0], setups[1])
            for key in stateful:
                policies[key].reset(n)
            finished = np.arange(n) >= real
            rngs = [np.random.default_rng([rng_seed, start, g]) for g in range(len(groups))]
            if tracker is not None:
                tracker.reset(n)
            results: dict[int, Game] = {}
            while not finished.all():
                live = np.flatnonzero(~finished)
                actions = np.zeros((n, 2), np.int32)
                for (role, key), g in groups.items():
                    rows = live[role_group[role, live] == g]
                    if not len(rows):
                        continue
                    seat = role_seat[role, rows]
                    extra = {"keys": rows * 2 + seat} if key in stateful else {}
                    actions[rows, seat] = policies[key].act(v.observations[rows, seat],
                                                            v.masks[rows, seat], rngs[g], **extra)
                if tracker is not None:
                    seat = role_seat[0, live]
                    tracker.add(live, v.observations[live, seat, :hi], actions[live, seat])
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
                    results[i] = Game(chunk[i][1], seat, score, crowns[seat], crowns[1 - seat],
                                      int(row[info["tick"]]), s.fault_names[int(row[info["fault"]])],
                                      tracker.game(i) if tracker is not None else None)
            for i in range(real):
                out[match_of[i]].append(results[i])
    return out
