"""Converted battles into compact training rows, through crforge's replay labeller.

    python -m clashai.data.label --run piggies_eq-screen-team0-v0-t0
    python -m clashai.data.label --run all-screen-team0-v0-t0 --filter prefix \
        --reuse all-screen-team0-v0-t0-standard --resume

``--resume`` skips the battles already in the dataset folder, so a conversion run that grew since it
was last labelled costs only its new battles. ``--reuse`` hard-links the battles another dataset
folder already holds in the same form (no cut, see below) instead of labelling them again.

Filters (:data:`FILTERS`): ``strict`` needs every tower's final hit points exact and no dropped
play; ``standard`` needs the recorded winner and crowns and no dropped play; ``prefix`` needs the
winner and crowns but allows dropped plays, and keeps a battle only up to its first: the decisions
from one decision before the first dropped play on are cut (:func:`stop_tick`), since from there
the player reacts to a battle the simulation no longer shows. Both seats are cut at the same tick.
A cut battle's ``counts.jsonl`` line gives the decisions kept and ``cut_at``.

Reads a conversion run's summaries (:func:`.convert.summary_files`), keeps the battles that pass the quality filter, and
labels each: at every decision (5 ticks) of each seat, the observation and mask the environment
shows that seat and the action that gives the player's command, or the wait. ``--check`` also
drives each battle back from its labels alone; at 5-tick decisions it stops at the first command
given between two decisions (driven up to 4 ticks early), so it proves little there.

crforge writes about 54 MB of uncompressed rows a battle, so battles are labelled in chunks, each
compacted (:mod:`.dataset`, by ``--workers`` processes: one core would take as long as the engine's
labelling) and its raw rows deleted before the next. Writes
``<data>/dataset/<run>-<filter>/``: ``battles/<tag>.npz``, crforge's ``metadata.json``,
``counts.jsonl``, and ``index.jsonl`` naming each battle's decks, slot flags, Piggies EQ seats,
each seat's result and whether any deck is in the held-out split.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from .. import paths
from ..decks import PIGGIES_EQ, is_held_out
from .convert import summary_files
from .dataset import PART, Layout, compact

FILTERS = {
    "strict": lambda r: r["towers_exact"] and r["dropped"] == 0,
    "standard": lambda r: r["crowns_match"] and r["result_match"] and r["dropped"] == 0,
    "prefix": lambda r: r["crowns_match"] and r["result_match"],
    "all": lambda r: True,
}
DECISION_TICKS = 5


def stop_tick(row: dict, filter_name: str) -> int | None:
    """Under ``prefix``, the tick from which a battle's decisions are cut: one decision before its
    first dropped play, whose own decision would otherwise be labelled a wait. None: keep all."""
    if filter_name != "prefix" or not row["drops"]:
        return None
    return min(d[0] for d in row["drops"]) - DECISION_TICKS


def _tag(counts_line: dict) -> str:
    return counts_line["source"].replace("\\", "/").rsplit("/", 1)[-1].removesuffix(".json")


def _cut_counts(line: dict, rows: int, stop: int) -> dict:
    """A counts line after the battle was cut at ``stop``: its decisions and rows as kept."""
    c = dict(line["counts"])
    c.update(decisions=rows // 2, rows=rows, cut_at=stop)
    return {**line, "counts": c}


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
    parser.add_argument("--workers", type=int, default=None,
                        help="processes compacting labelled battles; half the CPU budget by default")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="skip battles already labelled into the dataset folder")
    parser.add_argument("--reuse", help="a dataset folder whose uncut battles are hard-linked")
    parser.add_argument("--limit", type=int, default=0, help="label at most this many (checks)")
    args = parser.parse_args(argv)

    source = paths.data_root() / "converted" / args.run
    rows = []
    for summary in summary_files(source):
        with open(summary, encoding="utf-8") as f:
            rows += [json.loads(line) for line in f]
    keep = [r for r in rows if not r["problem"] and FILTERS[args.filter](r)]
    out = paths.data_root() / "dataset" / f"{args.run}-{args.filter}"
    raw = paths.data_root() / "labelled" / f"{args.run}-{args.filter}-chunk"
    (out / "battles").mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume:
        # Done: written and counted (a run stopped between the two labels the battle again).
        done = {f.name.removesuffix(".npz") for f in (out / "battles").glob("*.npz")
                if not f.name.endswith(PART)}
        counted = set()
        if (out / "counts.jsonl").exists():
            with open(out / "counts.jsonl", encoding="utf-8") as f:
                counted = {_tag(json.loads(line)) for line in f}
        done &= counted
    reused: set = set()
    if args.reuse:
        reused = _reuse(paths.data_root() / "dataset" / args.reuse, out, keep, done, args.filter)
        done |= reused
    todo = [r for r in keep if r["tag"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    cut = sum(stop_tick(r, args.filter) is not None for r in keep)
    print(f"{len(keep)} of {len(rows)} battles pass the '{args.filter}' filter ({cut} cut at their "
          f"first dropped play); {len(done) - len(reused)} labelled before, {len(reused)} "
          f"reused; {len(todo)} to label", flush=True)

    paths.ensure_java()
    from crforge_env import label_replays

    totals: Counter = Counter()
    layout = None
    if (args.resume or args.reuse) and (out / "metadata.json").exists():
        layout = Layout.from_metadata(json.loads((out / "metadata.json").read_text()))
    workers = args.workers or max(1, paths.cpu_budget() // 2)
    with (open(out / "counts.jsonl", "a" if args.resume or args.reuse else "w",
               encoding="utf-8") as counts,
          ProcessPoolExecutor(workers) as pool):
        for start in range(0, len(todo), args.chunk):
            chunk = todo[start:start + args.chunk]
            if raw.exists():
                shutil.rmtree(raw)
            run = label_replays([source / "docs" / f"{r['tag']}.json" for r in chunk],
                                paths.install_dir(), paths.tables_dir(), decision_ticks=5,
                                threads=args.threads or paths.cpu_budget(),
                                check=args.check, out_dir=raw, jvm_args=paths.jvm_args())
            if layout is None:
                metadata = (raw / "metadata.json").read_text()
                (out / "metadata.json").write_text(metadata)
                layout = Layout.from_metadata(json.loads(metadata))
            totals.update({k: v for k, v in run.totals().items() if isinstance(v, int)})
            files = sorted(raw.glob("*.npz"))
            tags = [f.name.split("_docs_")[-1].removesuffix(".json.npz") for f in files]
            stops = {r["tag"]: stop_tick(r, args.filter) for r in chunk}
            targets = [out / "battles" / f"{t}.npz" for t in tags]
            kept = dict(zip(tags, pool.map(compact, files, targets, [layout] * len(files),
                                           [stops[t] for t in tags]), strict=True))
            with open(raw / "counts.jsonl", encoding="utf-8") as f:
                for line in map(json.loads, f):
                    t = _tag(line)
                    if stops.get(t) is not None:
                        line = _cut_counts(line, kept[t], stops[t])
                    counts.write(json.dumps(line) + "\n")
            shutil.rmtree(raw)
            print(f"{min(start + args.chunk, len(todo))}/{len(todo)} battles labelled", flush=True)
    with open(out / "index.jsonl", "w", encoding="utf-8") as f:
        # The battles written so far: all of them, unless --limit stopped short.
        f.writelines(json.dumps(index_row(r)) + "\n" for r in keep
                     if (out / "battles" / f"{r['tag']}.npz").exists())
    print("totals:", json.dumps(dict(totals)))
    print(f"written to {out}")


def _reuse(source, out, keep: list, done: set, filter_name: str) -> set:
    """Hard-links into ``out`` the battles of dataset folder ``source`` that this filter keeps
    whole, with their counts lines; returns their tags."""
    whole = {r["tag"] for r in keep if stop_tick(r, filter_name) is None} - done
    have = {f.name.removesuffix(".npz") for f in (source / "battles").glob("*.npz")
            if not f.name.endswith(PART)}
    tags = whole & have
    if not (out / "metadata.json").exists():
        shutil.copyfile(source / "metadata.json", out / "metadata.json")
    for t in sorted(tags):
        target = out / "battles" / f"{t}.npz"
        if not target.exists():
            os.link(source / "battles" / f"{t}.npz", target)
    with (open(source / "counts.jsonl", encoding="utf-8") as f,
          open(out / "counts.jsonl", "a", encoding="utf-8") as counts):
        for line in f:
            if _tag(json.loads(line)) in tags:
                counts.write(line)
    return tags


if __name__ == "__main__":
    main()
