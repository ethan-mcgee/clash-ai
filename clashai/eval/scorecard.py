"""The per-matchup scorecard: our Piggies EQ pilot against every deck of a set.

    python -m clashai.eval.scorecard --policy random --opponent scripted --seeds 50

Every deck is played side by side in one vector (:func:`clashai.eval.arena.play_matches`). Prints
a Markdown table (one row per opponent deck, held-out decks marked) and writes the games as JSON
under ``<data>/runs/scorecards``. ``--deck-set exit`` plays the held-out meta decks
(:mod:`clashai.eval.deck_sets`); ``--behaviour`` records how our side plays in each game.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from .. import paths
from ..decks import PIGGIES_EQ, Deck, is_held_out
from ..sim.policies import BASELINES
from .arena import Side, play_matches
from .stats import summarize

SAMPLING_TAGS = {"card_temperature": "ct", "tile_temperature": "tt", "tile_top_p": "tp"}


def make_policy(name: str, temperature: float = 1.0, **sampling):
    """A baseline by name, or a checkpoint; ``sampling`` takes NetPolicy's card and tile
    settings, and any not at 1 are added to the policy's name."""
    if name in BASELINES:
        return BASELINES[name]()
    if name.endswith(".pt"):
        from ..model.policy import NetPolicy

        label = Path(name).parent.name + "".join(
            f"-{SAMPLING_TAGS[k]}{v:g}" for k, v in sampling.items() if v != 1.0)
        return NetPolicy(name, temperature=temperature, name=label, **sampling)
    raise SystemExit(f"unknown policy {name!r}: one of {sorted(BASELINES)} or a .pt checkpoint")


def run_scorecard(policy, opponent, decks: dict[str, Deck], seeds: list[int], *,
                  deck: Deck = PIGGIES_EQ, threads: int | None = None, batch: int = 1024,
                  behaviour: bool = False, tag: str = "", args: dict | None = None,
                  out: Path | None = None, quiet: bool = False) -> dict:
    """Plays ``deck`` piloted by ``policy`` against every deck of ``decks`` piloted by
    ``opponent``, on ``seeds`` from both seats; prints the table, writes the games as JSON and
    returns ``{"path", "games", "score", "by_deck"}``."""
    t0 = time.time()
    matches = [(Side(policy, deck), Side(opponent, d)) for d in decks.values()]
    played = play_matches(matches, seeds, batch=batch, threads=threads, behaviour=behaviour)
    elapsed = time.time() - t0
    rows, record, every = [], {}, []
    for (name, d), games in zip(decks.items(), played, strict=True):
        summary = summarize(g.score for g in games)
        held = "yes" if is_held_out(d) else ""
        rows.append(f"| {name} | {d.archetype} | {held} | {summary.games} | {summary.row()} |")
        record[name] = [asdict(g) for g in games]
        every += games
    total = summarize(g.score for g in every)
    faults = sum(1 for g in every if g.fault != "none")
    if not quiet:
        print(f"{deck.name} ({policy.name}) vs {tag or 'reference'} decks ({opponent.name}), "
              f"{len(seeds)} seeds x 2 seats per deck")
        print()
        print("| deck | archetype | held out | games | W-D-L | score | 95% interval |")
        print("|---|---|---|---:|---|---:|---|")
        print("\n".join(rows))
        print(f"| **all** | | | {total.games} | {total.row()} |")
        print()
        print(f"{len(every)} games in {elapsed:.0f} s ({len(every) / elapsed * 3600:,.0f} "
              f"games/hour), {faults} faulted")
    folder = paths.runs_dir() / "scorecards"
    folder.mkdir(parents=True, exist_ok=True)
    suffix = f"-{tag}" if tag else ""
    path = out or folder / (f"{time.strftime('%Y%m%d-%H%M%S')}-{policy.name}-vs-"
                            f"{opponent.name}{suffix}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"args": args or {}, "deck": deck.name, "decks": tag or "reference",
                   "seconds": elapsed, "games": record}, f, default=str)
    if not quiet:
        print(f"games written to {path}")
    return {"path": path, "games": record, "score": total.score,
            "by_deck": {n: summarize(g["score"] for g in gs).score for n, gs in record.items()}}


def main(argv: list[str] | None = None) -> None:
    from .deck_sets import deck_set

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--policy", default="random",
                        help=f"one of {sorted(BASELINES)} or a checkpoint (.pt) path")
    parser.add_argument("--opponent", default="scripted", help="as --policy")
    parser.add_argument("--temperature", type=float, default=1.0, help="for checkpoint policies")
    for who in ("", "opponent-"):
        side = "--opponent" if who else "--policy"
        parser.add_argument(f"--{who}card-temperature", type=float, default=1.0,
                            help=f"{side} only: which card, given that it plays")
        parser.add_argument(f"--{who}tile-temperature", type=float, default=1.0,
                            help=f"{side} only: where it plays")
        parser.add_argument(f"--{who}tile-top-p", type=float, default=1.0,
                            help=f"{side} only: sample among the likeliest tiles holding this share")
    parser.add_argument("--seeds", type=int, default=50, help="seeds per deck; 2 games each")
    parser.add_argument("--first-seed", type=int, default=1_000_000)
    parser.add_argument("--deck-set", default="reference", help="reference or exit")
    parser.add_argument("--decks", nargs="*", help="names within the set (default: all)")
    parser.add_argument("--behaviour", action="store_true", help="record how our side plays")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--batch", type=int, default=1024, help="battles played at once")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    decks = deck_set(args.deck_set)
    if args.decks:
        missing = set(args.decks) - set(decks)
        if missing:
            raise SystemExit(f"not in the {args.deck_set} set: {sorted(missing)}")
        decks = {n: decks[n] for n in args.decks}
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    policy = make_policy(args.policy, args.temperature, card_temperature=args.card_temperature,
                         tile_temperature=args.tile_temperature, tile_top_p=args.tile_top_p)
    opponent = make_policy(args.opponent, args.temperature,
                           card_temperature=args.opponent_card_temperature,
                           tile_temperature=args.opponent_tile_temperature,
                           tile_top_p=args.opponent_tile_top_p)
    run_scorecard(policy, opponent, decks, seeds, threads=args.threads, batch=args.batch,
                  behaviour=args.behaviour,
                  tag="" if args.deck_set == "reference" else args.deck_set, args=vars(args),
                  out=Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
