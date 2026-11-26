"""Opponent deck sets for scorecards.

- ``reference``: the 13 hand-picked archetypes (:data:`clashai.decks.REFERENCE_DECKS`)
- ``exit``: the most played held-out decks of the November replays, as played (cards and slot
  flags). PPO's league never draws a held-out deck (:func:`clashai.decks.is_held_out`), so these
  are decks the learner has never practised against. Built once from the conversion summaries
  and kept in ``<data>/eval/exit-decks.json``, so the set stays fixed as the data grows.

    python -m clashai.eval.deck_sets --build [--count 20]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .. import paths
from ..data.convert import summary_files
from ..decks import PIGGIES_EQ, REFERENCE_DECKS, Deck, is_held_out

CONVERTED = "converted/all-screen-team0-v0-t0"
EXIT_FILE = "eval/exit-decks.json"


def most_played_held_out(summaries: list[Path], count: int) -> list[dict]:
    """The ``count`` most played held-out decks (Piggies EQ aside) in conversion summaries, each
    with its most common slot flags, how often it was played and its share of all decks."""
    decks: Counter = Counter()
    flags: dict[tuple, Counter] = {}
    total = 0
    for path in summaries:
        with open(path, encoding="utf-8") as f:
            for line in f:
                s = json.loads(line)
                for cards, slots in zip(s.get("decks") or [], s.get("slots") or [], strict=False):
                    total += 1
                    order = sorted(range(8), key=lambda i: cards[i])
                    key = tuple(cards[i] for i in order)
                    decks[key] += 1
                    flags.setdefault(key, Counter())[tuple(slots[i] for i in order)] += 1
    piggies = tuple(sorted(PIGGIES_EQ.cards))
    out = []
    for key, n in decks.most_common():
        if key == piggies or not is_held_out(key):
            continue
        out.append({"name": f"held_{len(out) + 1:02d}", "cards": list(key),
                    "flags": list(flags[key].most_common(1)[0][0]), "played": n,
                    "share": n / total})
        if len(out) == count:
            break
    return out


def build(count: int = 20) -> Path:
    path = paths.data_root() / EXIT_FILE
    entries = most_played_held_out(summary_files(paths.data_root() / CONVERTED), count)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"source": CONVERTED, "decks": entries}, f, indent=1)
    return path


def deck_set(name: str) -> dict[str, Deck]:
    """The decks of a set by name, in order."""
    if name == "reference":
        return dict(REFERENCE_DECKS)
    if name == "exit":
        path = paths.data_root() / EXIT_FILE
        if not path.exists():
            raise SystemExit(f"{path} is missing: run python -m clashai.eval.deck_sets --build")
        with open(path, encoding="utf-8") as f:
            entries = json.load(f)["decks"]
        return {e["name"]: Deck(e["name"], tuple(e["cards"]), archetype="meta",
                                flags=tuple(e["flags"])) for e in entries}
    raise SystemExit(f"unknown deck set {name!r}: reference or exit")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--build", action="store_true", help="write the exit set from the summaries")
    p.add_argument("--count", type=int, default=20)
    args = p.parse_args(argv)
    if args.build:
        print(f"written to {build(args.count)}")
    for name, d in deck_set("exit").items():
        print(f"{name}: {', '.join(d.cards)}")


if __name__ == "__main__":
    main()
