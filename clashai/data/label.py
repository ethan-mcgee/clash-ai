"""Converted battles into training rows, through crforge's replay labeller.

    python -m clashai.data.label --run piggies_eq-screen-team0-v0-t0

Reads a conversion run's ``summary.jsonl``, keeps the battles that pass the quality filter, and
labels each: at every decision (5 ticks) of each seat, the observation and mask the environment
shows that seat and the action that gives the player's command, or the wait. ``--check`` also
drives each battle back from its labels alone; at 5-tick decisions it stops at the first command
given between two decisions (driven up to 4 ticks early), so it proves little there.

Writes ``<data>/labelled/<run>/``: one ``.npz`` of rows per battle and crforge's ``metadata.json``
and ``counts.jsonl``, plus ``index.jsonl`` naming, for each battle, which seat played which deck,
which seats played Piggies EQ and each seat's result.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from .. import paths
from ..decks import PIGGIES_EQ

FILTERS = {
    "strict": lambda r: r["towers_exact"] and r["dropped"] == 0,
    "standard": lambda r: r["crowns_match"] and r["result_match"] and r["dropped"] == 0,
    "all": lambda r: True,
}


def piggies_seats(row: dict) -> list[int]:
    out = []
    for seat, (deck, slots) in enumerate(zip(row["decks"], row["slots"], strict=True)):
        evolved = {c for c, f in zip(deck, slots, strict=True) if f & 1}
        if set(deck) == set(PIGGIES_EQ.cards) and evolved == set(PIGGIES_EQ.evolutions):
            out.append(seat)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Label converted battles for imitation")
    parser.add_argument("--run", required=True, help="a folder under <data>/converted")
    parser.add_argument("--filter", choices=FILTERS, default="standard")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    source = paths.data_root() / "converted" / args.run
    rows = [json.loads(line) for line in open(source / "summary.jsonl", encoding="utf-8")]
    keep = [r for r in rows if not r["problem"] and FILTERS[args.filter](r)]
    out = paths.data_root() / "labelled" / f"{args.run}-{args.filter}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(keep)} of {len(rows)} battles pass the '{args.filter}' filter", flush=True)

    paths.ensure_java()
    from crforge_env import label_replays

    run = label_replays([source / "docs" / f"{r['tag']}.json" for r in keep], paths.install_dir(),
                        paths.tables_dir(), decision_ticks=5,
                        threads=args.threads or paths.default_threads(),
                        check=args.check, out_dir=out)
    with open(out / "index.jsonl", "w", encoding="utf-8") as f:
        for r in keep:
            results = [_result(r, seat) for seat in (0, 1)]
            f.write(json.dumps({"tag": r["tag"], "decks": r["decks"], "slots": r["slots"],
                                "piggies_seats": piggies_seats(r), "results": results,
                                "towers_exact": r["towers_exact"]}) + "\n")
    print("unreadable:", run.unreadable or "none")
    print("totals:", json.dumps(run.totals(), indent=1)[:3000])
    counts = Counter(len(piggies_seats(r)) for r in keep)
    print(f"battles by Piggies EQ seats: {dict(counts)}; written to {out}")


def _result(row: dict, seat: int) -> str:
    """A seat's result by the converted battle's crowns, which equal the recording's under the
    standard and strict filters."""
    side, other = row["sides"][seat], row["sides"][1 - seat]
    mine, theirs = row["crowns"].get(side, 0), row["crowns"].get(other, 0)
    return "win" if mine > theirs else "loss" if mine < theirs else "draw"


if __name__ == "__main__":
    main()
