"""The data audit: how much Piggies EQ the replays hold, and whom it meets.

    python -m clashai.data.audit

Writes ``<data>/datasets/audit/piggies_eq.json`` (every count) and prints a Markdown summary:
Piggies EQ games by evolution variant with win rates, one-card swaps, the opponent decks it meets
most often (with the held-out split marked) and its win rate against each, and the card and tower
levels in the data.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from .. import paths
from ..decks import PIGGIES_EQ, is_held_out
from ..eval.stats import wilson
from .cards import CardNames, split_key
from .download import dataset_dir
from .replays import read_replays

PIGGIES_KEYS = {"royal-hogs", "archer-queen", "cannon", "skeletons", "ice-spirit", "the-log",
                "earthquake", "royal-delivery"}
PIGGIES_EVOS = ("cannon-ev1", "royal-hogs-ev1")


def _base(deck) -> frozenset[str]:
    return frozenset(split_key(k)[0] for k in deck)


def _variant(deck) -> tuple[str, ...]:
    return tuple(sorted(k for k in deck if split_key(k)[1]))


def audit(root=None) -> dict:
    variants, wins = Counter(), defaultdict(lambda: [0, 0, 0])  # wins, losses, draws
    swaps = Counter()
    opponents, vs = Counter(), defaultdict(lambda: [0, 0, 0])
    levels, towers, leagues = Counter(), Counter(), Counter()
    replays = 0
    for r in read_replays(root or dataset_dir()):
        replays += 1
        leagues[r.league or r.mode] += 1
        for side, other in (("team", "opponent"), ("opponent", "team")):
            p = r.side(side)
            levels.update(p.levels)
            towers[p.tower] += 1
            base = _base(p.deck)
            won = r.won(side)
            outcome = 2 if won is None else (0 if won else 1)
            if base == PIGGIES_KEYS:
                v = _variant(p.deck)
                variants[v] += 1
                wins[v][outcome] += 1
                if v == PIGGIES_EVOS:
                    opp = r.side(other).deck
                    key = ",".join(sorted(_base(opp)))
                    opponents[key] += 1
                    vs[key][outcome] += 1
            elif len(base & PIGGIES_KEYS) == 7 and {"royal-hogs", "earthquake"} <= base:
                swaps[(tuple(sorted(base - PIGGIES_KEYS)), tuple(sorted(PIGGIES_KEYS - base)))] += 1

    def rate(w):
        n = sum(w)
        return {"games": n, "wins": w[0], "losses": w[1], "draws": w[2],
                "score": (w[0] + 0.5 * w[2]) / n if n else None,
                "interval": wilson(w[0] + 0.5 * w[2], n)}

    return {
        "replays": replays,
        "leagues": dict(leagues),
        "piggies_variants": [{"evolutions": list(v), **rate(wins[v])}
                             for v, _ in variants.most_common()],
        "one_card_swaps": [{"added": list(a), "removed": list(b), "games": c}
                           for (a, b), c in swaps.most_common(30)],
        "opponents": [{"deck": k.split(","), "held_out": is_held_out(k.split(",")), **rate(vs[k])}
                      for k, _ in opponents.most_common()],
        "card_levels": dict(sorted(levels.items())),
        "towers": dict(towers.most_common()),
    }


def _held_out_crforge(report: dict, names: CardNames | None) -> None:
    """Re-mark the held-out split on crforge names, the names training will see decks by."""
    if names is None:
        return
    for o in report["opponents"]:
        o["held_out"] = is_held_out([names.name(k) for k in o["deck"]])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit the replays for Piggies EQ")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args(argv)
    from ..sim.env import make_vector

    with make_vector(1, 1) as v:
        names = CardNames(v.vocabulary[i] for i in v.card_types())
    report = audit()
    _held_out_crforge(report, names)
    out = paths.data_root() / "datasets" / "audit"
    out.mkdir(parents=True, exist_ok=True)
    (out / "piggies_eq.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    def pct(x):
        return f"{x:.1%}"

    print(f"{report['replays']:,} replays; leagues {report['leagues']}\n")
    print("| Piggies EQ evolutions | games | score | 95% interval |\n|---|---:|---:|---|")
    for v in report["piggies_variants"]:
        lo, hi = v["interval"]
        print(f"| {' + '.join(v['evolutions']) or 'none'} | {v['games']} | {pct(v['score'])} | "
              f"{pct(lo)}-{pct(hi)} |")
    print("\n| one-card swap: added | removed | games |\n|---|---|---:|")
    for s in report["one_card_swaps"][:10]:
        print(f"| {', '.join(s['added'])} | {', '.join(s['removed'])} | {s['games']} |")
    total = sum(o["games"] for o in report["opponents"])
    held = sum(o["games"] for o in report["opponents"] if o["held_out"])
    print(f"\nOpponents of {' + '.join(PIGGIES_EVOS)} Piggies EQ: {total} games, "
          f"{len(report['opponents'])} distinct decks, {pct(held / total)} of games against "
          f"held-out decks\n")
    print("| opponent deck | held out | games | Piggies score | 95% interval |\n|---|---|---:|---:|---|")
    for o in report["opponents"][:args.top]:
        lo, hi = o["interval"]
        print(f"| {', '.join(o['deck'])} | {'yes' if o['held_out'] else ''} | {o['games']} | "
              f"{pct(o['score'])} | {pct(lo)}-{pct(hi)} |")
    print(f"\nCard levels: {report['card_levels']}\nTowers: {report['towers']}")
    print(f"\nWritten to {out / 'piggies_eq.json'}  (deck under training: {PIGGIES_EQ.name})")


if __name__ == "__main__":
    main()
