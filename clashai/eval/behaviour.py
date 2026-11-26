"""How a player plays, per game, measured the same way for the bot's games and the human replays.

Per game, from that seat's decisions (every 5 ticks) and the action taken at each:

- ``plays_per_min`` and ``abilities_per_min``: cards played and champion abilities used
- ``full_elixir_share``: the share of decisions taken at full elixir (10), elixir going to waste
- ``first_play_s``: when the first card was played, in seconds of game time
- ``cards``: how many times each card was played, by name

A game that is still being fed can be read at any time. :func:`summarize` gives each measure's
mean and its 10th, 50th and 90th percentiles over games, and each card's share of all plays.
:func:`human_reference` measures the human seats of one deck in a labelled dataset.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np

TICKS_PER_SECOND = 20
FULL_ELIXIR = 100_000
"""Elixir in the observation is counted in ten-thousandths: 10 elixir is 100,000."""
MEASURES = ("plays_per_min", "abilities_per_min", "full_elixir_share", "first_play_s")


class Behaviour:
    """Counters for ``n`` games at once; :meth:`add` feeds decisions, :meth:`game` reads one."""

    def __init__(self, header: dict, vocabulary: Sequence[str], hand_slots: int, tiles: int,
                 first_ability: int, decision_ticks: int, n: int):
        self.tick, self.elixir = header["tick"][0], header["elixir"][0]
        self.hand = np.asarray(header["hand_card"][:hand_slots])
        self.vocabulary = list(vocabulary)
        self.tiles, self.first_ability = tiles, first_ability
        self.decision_ticks = decision_ticks
        self.reset(n)

    def reset(self, n: int) -> None:
        self.decisions = np.zeros(n, np.int64)
        self.full = np.zeros(n, np.int64)
        self.plays = np.zeros(n, np.int64)
        self.abilities = np.zeros(n, np.int64)
        self.first = np.full(n, np.iinfo(np.int64).max, np.int64)
        self.last = np.zeros(n, np.int64)
        self.cards = np.zeros((n, len(self.vocabulary)), np.int64)

    def add(self, games: np.ndarray, header: np.ndarray, actions: np.ndarray) -> None:
        """Decisions of the given games (an index per decision, repeats allowed): their header
        ints ``(k, H)`` and the actions taken ``(k,)``."""
        games, actions = np.asarray(games), np.asarray(actions)
        tick = header[:, self.tick].astype(np.int64)
        np.add.at(self.decisions, games, 1)
        np.add.at(self.full, games, header[:, self.elixir] >= FULL_ELIXIR)
        np.maximum.at(self.last, games, tick)
        play = (actions > 0) & (actions < self.first_ability)
        np.add.at(self.abilities, games, actions >= self.first_ability)
        if play.any():
            rows = np.flatnonzero(play)
            slot = (actions[rows] - 1) // self.tiles
            card = header[rows, self.hand[slot]]
            known = (card >= 0) & (card < len(self.vocabulary))
            np.add.at(self.plays, games[rows], 1)
            np.minimum.at(self.first, games[rows], tick[rows])
            np.add.at(self.cards, (games[rows][known], card[known]), 1)

    def game(self, i: int) -> dict:
        minutes = (self.last[i] + self.decision_ticks) / (60 * TICKS_PER_SECOND)
        first = self.first[i]
        return {
            "minutes": round(float(minutes), 3),
            "plays_per_min": float(self.plays[i] / minutes) if minutes else 0.0,
            "abilities_per_min": float(self.abilities[i] / minutes) if minutes else 0.0,
            "full_elixir_share": float(self.full[i] / max(1, self.decisions[i])),
            "first_play_s": (None if first == np.iinfo(np.int64).max
                             else float(first / TICKS_PER_SECOND)),
            "cards": {self.vocabulary[c]: int(k) for c, k in enumerate(self.cards[i]) if k},
        }


def summarize(games: Sequence[dict]) -> dict:
    """Each measure's mean and 10th, 50th and 90th percentiles over games (a game with no play
    left out of ``first_play_s``), and each card's share of all plays."""
    out: dict = {"games": len(games)}
    for m in MEASURES:
        x = np.array([g[m] for g in games if g.get(m) is not None], np.float64)
        out[m] = ({"mean": float(x.mean()), "p10": float(np.percentile(x, 10)),
                   "p50": float(np.percentile(x, 50)), "p90": float(np.percentile(x, 90))}
                  if len(x) else None)
    cards: Counter = Counter()
    for g in games:
        cards.update(g.get("cards", {}))
    total = sum(cards.values())
    out["card_share"] = {c: k / total for c, k in cards.most_common()} if total else {}
    return out


def human_reference(dataset: Path, cards: Sequence[str], cache: Path | None = None) -> dict:
    """:func:`summarize` of every human seat playing exactly the deck ``cards`` in a labelled
    dataset (``<dataset>/index.jsonl`` and ``battles/<tag>.npz``). Cached in ``cache`` if given."""
    if cache is not None and cache.exists():
        with open(cache, encoding="utf-8") as f:
            return json.load(f)
    from ..model.spec import Spec

    with open(dataset / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    spec = Spec.from_metadata(metadata)
    schema = metadata["schema"]
    want = sorted(cards)
    tracker = Behaviour(spec.header, metadata["vocabulary"]["names"], spec.hand_slots, spec.tiles,
                        spec.first_ability, int(metadata["config"]["decision_ticks"]), 1)
    games = []
    with open(dataset / "index.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            for seat in r.get("piggies_seats") or []:
                if sorted(r["decks"][seat]) != want:
                    continue
                z = np.load(dataset / "battles" / f"{r['tag']}.npz")
                rows = z["seats"] == seat
                tracker.reset(1)
                tracker.add(np.zeros(int(rows.sum()), np.int64),
                            z["header"][rows][:, :schema["header_ints"]], z["actions"][rows])
                games.append(tracker.game(0))
    out = summarize(games)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    return out
