"""The per-matchup scorecard: our Piggies EQ pilot against every reference deck.

    python -m clashai.eval.scorecard --policy random --opponent scripted --seeds 50

Prints a Markdown table (one row per opponent deck, held-out decks marked) and writes the games
as JSON under ``<data>/runs/scorecards``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

from .. import paths
from ..decks import PIGGIES_EQ, REFERENCE_DECKS, is_held_out
from ..sim.policies import BASELINES
from .arena import Side, play
from .stats import summarize


def make_policy(name: str, temperature: float = 1.0):
    if name in BASELINES:
        return BASELINES[name]()
    if name.endswith(".pt"):
        from ..model.policy import NetPolicy

        return NetPolicy(name, temperature=temperature)
    raise SystemExit(f"unknown policy {name!r}: one of {sorted(BASELINES)} or a .pt checkpoint")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--policy", default="random",
                        help=f"one of {sorted(BASELINES)} or a checkpoint (.pt) path")
    parser.add_argument("--opponent", default="scripted", help="as --policy")
    parser.add_argument("--temperature", type=float, default=1.0, help="for checkpoint policies")
    parser.add_argument("--seeds", type=int, default=50, help="seeds per deck; 2 games each")
    parser.add_argument("--first-seed", type=int, default=1_000_000)
    parser.add_argument("--decks", nargs="*", default=list(REFERENCE_DECKS))
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    policy = make_policy(args.policy, args.temperature)
    opponent = make_policy(args.opponent, args.temperature)
    rows, record, every = [], {}, []
    t0 = time.time()
    for name in args.decks:
        deck = REFERENCE_DECKS[name]
        games = play(Side(policy, PIGGIES_EQ), Side(opponent, deck), seeds, threads=args.threads)
        summary = summarize(g.score for g in games)
        held = "yes" if is_held_out(deck) else ""
        rows.append(f"| {name} | {deck.archetype} | {held} | {summary.games} | {summary.row()} |")
        record[name] = [asdict(g) for g in games]
        every += games
        print(rows[-1], flush=True)
    elapsed = time.time() - t0
    total = summarize(g.score for g in every)
    faults = sum(1 for g in every if g.fault != "none")
    print()
    print(f"Piggies EQ ({policy.name}) vs reference decks ({opponent.name}), {args.seeds} seeds "
          "x 2 seats per deck")
    print()
    print("| deck | archetype | held out | games | W-D-L | score | 95% interval |")
    print("|---|---|---|---:|---|---:|---|")
    print("\n".join(rows))
    print(f"| **all** | | | {total.games} | {total.row()} |")
    print()
    print(f"{len(every)} games in {elapsed:.0f} s ({len(every) / elapsed * 3600:,.0f} games/hour), "
          f"{faults} faulted")
    out = paths.runs_dir() / "scorecards" if args.out is None else None
    path = (out / f"{time.strftime('%Y%m%d-%H%M%S')}-{policy.name}-vs-{opponent.name}.json"
            if out else args.out)
    if out:
        out.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "seconds": elapsed, "games": record}, f)
    print(f"games written to {path}")


if __name__ == "__main__":
    main()
