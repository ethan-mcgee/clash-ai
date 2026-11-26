"""Labelled rows in a compact, lossless form for training.

crforge's labeller writes every row's full v4 observation (7,024 int32, mostly empty entity slots)
and mask (2,307 bytes): about 30 KB a row, 127 GB for the Piggies EQ battles. Here each battle
keeps:

- ``header``: the observation's header, ``(rows, header_ints)`` int32
- ``entities`` and ``offsets``: only the entity rows in use, stacked, and where each row's start
  (``offsets[r]:offsets[r + 1]``)
- ``masks``: the masks packed to bits, ``(rows, ceil(actions / 8))`` uint8
- ``actions``, ``ticks``, ``seats``, ``flags``: as labelled

compressed with zlib. :func:`observations` and :func:`masks` rebuild the originals exactly.

    python -m clashai.data.dataset --run piggies_eq-screen-team0-v0-t0-standard
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import paths

PART = ".part.npz"  # a battle being written


@dataclass(frozen=True)
class Layout:
    header_ints: int
    entity_ints: int
    max_entities: int
    actions: int
    entity_count: int  # the header index of the entity count

    @classmethod
    def from_metadata(cls, metadata: dict) -> Layout:
        s = metadata["schema"]
        return cls(int(s["header_ints"]), int(s["entity_ints"]), int(s["max_entities"]),
                   int(s["actions"]), list(s["header_fields"]).index("entity_count"))


def compact(source: Path, target: Path, layout: Layout, stop_tick: int | None = None) -> int:
    """Writes ``source``'s rows compacted to ``target``; with ``stop_tick``, only the rows of
    decisions before that tick. Returns the rows written."""
    z = np.load(source, allow_pickle=False)
    keep = slice(None) if stop_tick is None else z["ticks"] < stop_tick
    z = {k: z[k][keep] for k in ("observations", "masks", "actions", "ticks", "seats", "flags")}
    obs = z["observations"]
    rows = len(obs)
    header = obs[:, :layout.header_ints]
    counts = header[:, layout.entity_count].astype(np.int64)
    table = obs[:, layout.header_ints:].reshape(rows, layout.max_entities, layout.entity_ints)
    used = np.arange(layout.max_entities)[None, :] < counts[:, None]
    # Written beside the target and renamed into place, so an interrupted run never leaves a
    # partial battle under the target's name.
    part = target.with_name(target.name.removesuffix(".npz") + PART)
    np.savez_compressed(
        part, header=header, entities=table[used],
        offsets=np.concatenate([[0], np.cumsum(counts)]),
        masks=np.packbits(z["masks"], axis=1), actions=z["actions"], ticks=z["ticks"],
        seats=z["seats"].astype(np.int8), flags=z["flags"])
    os.replace(part, target)
    return rows


def observations(battle, layout: Layout) -> np.ndarray:
    """The full v4 observations of a compacted battle."""
    header, entities, offsets = battle["header"], battle["entities"], battle["offsets"]
    rows = len(header)
    out = np.zeros((rows, layout.header_ints + layout.max_entities * layout.entity_ints), np.int32)
    out[:, :layout.header_ints] = header
    table = out[:, layout.header_ints:].reshape(rows, layout.max_entities, layout.entity_ints)
    counts = np.diff(offsets)
    table[np.arange(layout.max_entities)[None, :] < counts[:, None]] = entities
    return out


def masks(battle, layout: Layout) -> np.ndarray:
    return np.unpackbits(battle["masks"], axis=1, count=layout.actions)


def _job(args) -> tuple[str, int, bool]:
    source, target, layout, verify = args
    rows = compact(source, target, layout)
    ok = True
    if verify:
        z, c = np.load(source), np.load(target)
        ok = (np.array_equal(observations(c, layout), z["observations"])
              and np.array_equal(masks(c, layout), z["masks"])
              and np.array_equal(c["actions"], z["actions"]))
    return source.name, rows, ok


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compact labelled rows for training")
    parser.add_argument("--run", required=True, help="a folder under <data>/labelled")
    parser.add_argument("--verify", type=int, default=50,
                        help="battles checked to rebuild exactly (all with -1)")
    parser.add_argument("--delete-source", action="store_true",
                        help="delete each crforge .npz once its compact copy is written")
    args = parser.parse_args(argv)
    source = paths.data_root() / "labelled" / args.run
    target = paths.data_root() / "dataset" / args.run
    (target / "battles").mkdir(parents=True, exist_ok=True)
    layout = Layout.from_metadata(json.loads((source / "metadata.json").read_text()))
    files = sorted(source.glob("*.npz"))
    jobs = [(f, target / "battles" / (f.name.split("_docs_")[-1].removesuffix(".json.npz") + ".npz"),
             layout, args.verify < 0 or i < args.verify) for i, f in enumerate(files)]
    rows = bad = 0
    with ProcessPoolExecutor(max_workers=max(1, paths.cpu_budget() // 2)) as pool:
        for name, n, ok in pool.map(_job, jobs, chunksize=4):
            rows += n
            bad += not ok
    for name in ("metadata.json", "index.jsonl", "counts.jsonl"):
        if (source / name).exists():
            (target / name).write_bytes((source / name).read_bytes())
    size = sum(f.stat().st_size for f in (target / "battles").glob("*.npz"))
    print(f"{len(files)} battles, {rows:,} rows, {size / 1e9:.2f} GB; "
          f"{min(args.verify, len(files)) if args.verify >= 0 else len(files)} checked, {bad} "
          "failed to rebuild")
    if args.delete_source and bad == 0:
        for f in files:
            f.unlink()
        print(f"deleted {len(files)} source files")


if __name__ == "__main__":
    main()
