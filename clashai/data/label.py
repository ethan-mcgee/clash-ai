"""Converted battles into compact training rows, through crforge's replay labeller.

    python -m clashai.data.label --run piggies_eq-screen-team0-v0-t0

Reads a conversion run's ``summary.jsonl``, keeps the battles that pass the quality filter, and
labels each: at every decision (5 ticks) of each seat, the observation and mask the environment
shows that seat and the action that gives the player's command, or the wait. ``--check`` also
drives each battle back from its labels alone; at 5-tick decisions it stops at the first command
given between two decisions (driven up to 4 ticks early), so it proves little there.

crforge writes about 54 MB of uncompressed rows a battle, so battles are labelled in chunks, each
compacted (:mod:`.dataset`) and its raw rows deleted before the next. Writes
``<data>/dataset/<run>-<filter>/``: ``battles/<tag>.npz``, crforge's ``metadata.json``,
``counts.jsonl``, and ``index.jsonl`` naming each battle's decks, slot flags, Piggies EQ seats,
each seat's result and whether any deck is in the held-out split.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter

from .. import paths
from ..decks import PIGGIES_EQ, is_held_out
from .dataset import Layout, compact

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


def _result(row: dict, seat: int) -> str:
    """A seat's result by the converted battle's crowns, which equal the recording's under the
    standard and strict filters."""
    side, other = row["sides"][seat], row["sides"][1 - seat]
    mine, theirs = row["crowns"].get(side, 0), row["crowns"].get(other, 0)
    return "win" if mine > theirs else "loss" if mine < theirs else "draw"


def index_row(r: dict) -> dict:
    return {"tag": r["tag"], "decks": r["decks"], "slots": r["slots"],
            "piggies_seats": piggies_seats(r), "results": [_result(r, s) for s in (0, 1)],
            "held_out": [is_held_out(d) for d in r["decks"]], "towers_exact": r["towers_exact"]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Label converted battles for imitation")
    parser.add_argument("--run", required=True, help="a folder under <data>/converted")
    parser.add_argument("--filter", choices=FILTERS, default="standard")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=400, help="battles labelled at a time")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    source = paths.data_root() / "converted" / args.run
    rows = [json.loads(line) for line in open(source / "summary.jsonl", encoding="utf-8")]
    keep = [r for r in rows if not r["problem"] and FILTERS[args.filter](r)]
    out = paths.data_root() / "dataset" / f"{args.run}-{args.filter}"
    raw = paths.data_root() / "labelled" / f"{args.run}-{args.filter}-chunk"
    (out / "battles").mkdir(parents=True, exist_ok=True)
    print(f"{len(keep)} of {len(rows)} battles pass the '{args.filter}' filter", flush=True)

    paths.ensure_java()
    from crforge_env import label_replays

    totals: Counter = Counter()
    layout = None
    with open(out / "counts.jsonl", "w", encoding="utf-8") as counts:
        for start in range(0, len(keep), args.chunk):
            chunk = keep[start:start + args.chunk]
            if raw.exists():
                shutil.rmtree(raw)
            run = label_replays([source / "docs" / f"{r['tag']}.json" for r in chunk],
                                paths.install_dir(), paths.tables_dir(), decision_ticks=5,
                                threads=args.threads or paths.cpu_budget(),
                                check=args.check, out_dir=raw)
            if layout is None:
                metadata = (raw / "metadata.json").read_text()
                (out / "metadata.json").write_text(metadata)
                layout = Layout.from_metadata(json.loads(metadata))
            totals.update({k: v for k, v in run.totals().items() if isinstance(v, int)})
            counts.write((raw / "counts.jsonl").read_text())
            for f in raw.glob("*.npz"):
                tag = f.name.split("_docs_")[-1].removesuffix(".json.npz")
                compact(f, out / "battles" / f"{tag}.npz", layout)
            shutil.rmtree(raw)
            print(f"{min(start + args.chunk, len(keep))}/{len(keep)} battles labelled", flush=True)
    with open(out / "index.jsonl", "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(index_row(r)) + "\n")
    print("totals:", json.dumps(dict(totals)))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
