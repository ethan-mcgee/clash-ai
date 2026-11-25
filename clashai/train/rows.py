"""Training rows for imitation, gathered from compacted battles into flat arrays.

    python -m clashai.train.rows --dataset piggies_eq-screen-team0-v0-t0-standard --wait-keep 0.25

Every row where the player gave a command is kept. Of the rest (waits, about 95%), a random
``wait_keep`` share is kept with weight ``1 / wait_keep``, so the weighted loss is the loss over all
rows at a fraction of the cost.

Per row, beside the observation and mask:

- ``weight``: the wait sampling weight
- ``result``: the seat's result, 1 won, -1 lost, 0 draw
- ``piggies``: the seat played Piggies EQ
- ``opp_hand`` (4) and ``opp_next``: the enemy's hand at this decision and the card of its next
  play (-1 for none): targets for the belief heads, read from the other seat's rows
- ``battle``, ``seat``, ``tick``, ``val``: the battle's index, the side, the time, and whether the
  battle is in the validation split (3%, by a hash of its tag, so both seats stay together)
- ``label_masked``: the labelled action is closed in the row's mask

Arrays are written as raw files with a ``rows.json`` describing them, and loaded whole into RAM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .. import paths

ARRAYS = {  # name: (dtype, columns or None for one value a row)
    "header": ("int32", None), "masks": ("uint8", None), "actions": ("int32", None),
    "weight": ("float32", None), "result": ("int8", None), "piggies": ("bool", None),
    "opp_hand": ("int16", 4), "opp_next": ("int16", None), "battle": ("int32", None),
    "seat": ("int8", None), "tick": ("int32", None), "val": ("bool", None),
    "label_masked": ("bool", None), "entity_offsets": ("int64", None), "entities": ("int32", None),
}
MAX_ENTITIES = 80


def is_val(tag: str, percent: float = 3.0) -> bool:
    return int.from_bytes(hashlib.sha256(tag.encode()).digest()[:4], "big") < percent / 100 * 2**32


def battle_rows(path: Path, info: dict, battle: int, wait_keep: float, seed: int,
                header_fields: list[str]) -> dict:
    z = np.load(path)
    actions, seats, ticks = z["actions"], z["seats"].astype(np.int64), z["ticks"].astype(np.int64)
    header, offsets, entities = z["header"], z["offsets"], z["entities"]
    n = len(actions)
    hand_idx = [i for i, f in enumerate(header_fields) if f == "hand_card"]
    tiles = 18 * 32

    # The other seat's row at the same decision.
    key = ticks * 2 + seats
    order = np.argsort(key)
    want = ticks * 2 + (1 - seats)
    pos = np.clip(np.searchsorted(key[order], want), 0, n - 1)
    other = np.where(key[order][pos] == want, order[pos], -1)

    # Each row's card for its own play, then each seat's next play card at or after each row.
    play = (actions > 0) & (actions < 2305)
    slot = np.where(play, (actions - 1) // tiles, 0)
    played = np.where(play, header[np.arange(n), np.array(hand_idx)[slot]], -1)
    next_card = np.full(n, -1, np.int64)
    for s in (0, 1):
        rows = np.flatnonzero(seats == s)
        rows = rows[np.argsort(ticks[rows], kind="stable")]
        upcoming = -1
        for r in rows[::-1]:
            if played[r] >= 0:
                upcoming = played[r]
            next_card[r] = upcoming
    has_other = other >= 0
    opp_hand = np.where(has_other[:, None], header[np.maximum(other, 0)][:, hand_idx], -1)
    opp_next = np.where(has_other, next_card[np.maximum(other, 0)], -1)

    rng = np.random.default_rng([seed, battle])
    command = actions != 0
    keep = np.flatnonzero(command | (rng.random(n) < wait_keep))
    weight = np.where(command[keep], 1.0, 1.0 / wait_keep).astype(np.float32)

    counts = np.minimum(np.diff(offsets)[keep], MAX_ENTITIES)
    starts = offsets[keep]
    take = np.concatenate([np.arange(s, s + c) for s, c in zip(starts, counts)]) if len(keep) else \
        np.zeros(0, np.int64)
    results = {"win": 1, "loss": -1, "draw": 0}
    seat_k = seats[keep]
    mask = z["masks"][keep]
    label_open = np.unpackbits(mask, axis=1, count=2307)[np.arange(len(keep)), actions[keep]]
    return {
        "header": header[keep], "masks": mask, "actions": actions[keep], "weight": weight,
        "result": np.array([results[info["results"][s]] for s in seat_k], np.int8),
        "piggies": np.isin(seat_k, info["piggies_seats"]),
        "opp_hand": opp_hand[keep].astype(np.int16), "opp_next": opp_next[keep].astype(np.int16),
        "battle": np.full(len(keep), battle, np.int32), "seat": seat_k.astype(np.int8),
        "tick": ticks[keep].astype(np.int32), "val": np.full(len(keep), is_val(info["tag"])),
        "label_masked": label_open == 0, "entity_counts": counts, "entities": entities[take],
    }


def _job(args):
    return battle_rows(*args)


def build(dataset: str, wait_keep: float, seed: int = 0, out_name: str | None = None) -> Path:
    source = paths.data_root() / "dataset" / dataset
    out = paths.data_root() / "rows" / (out_name or f"{dataset}-w{wait_keep:g}")
    out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((source / "metadata.json").read_text())
    header_fields = metadata["schema"]["header_fields"]
    index = [json.loads(line) for line in open(source / "index.jsonl", encoding="utf-8")]
    index = [r for r in index if (source / "battles" / f"{r['tag']}.npz").exists()]
    files = {name: open(out / f"{name}.bin", "wb") for name in ARRAYS if name != "entity_offsets"}
    counts_all = []
    rows = 0
    jobs = [(source / "battles" / f"{r['tag']}.npz", r, i, wait_keep, seed, header_fields)
            for i, r in enumerate(index)]
    with ProcessPoolExecutor(max_workers=max(1, paths.cpu_budget() // 2)) as pool:
        for k, part in enumerate(pool.map(_job, jobs, chunksize=8)):
            for name, f in files.items():
                f.write(np.ascontiguousarray(part[name]).tobytes())
            counts_all.append(part["entity_counts"])
            rows += len(part["actions"])
            if (k + 1) % 2000 == 0:
                print(f"{k + 1}/{len(jobs)} battles, {rows:,} rows", flush=True)
    for f in files.values():
        f.close()
    counts = np.concatenate(counts_all)
    np.concatenate([[0], np.cumsum(counts)]).astype(np.int64).tofile(out / "entity_offsets.bin")
    shapes = {"header": [rows, len(header_fields)], "masks": [rows, (2307 + 7) // 8],
              "opp_hand": [rows, 4], "entity_offsets": [rows + 1],
              "entities": [int(counts.sum()), len(metadata["schema"]["entity_fields"])]}
    (out / "metadata.json").write_text(json.dumps(metadata))
    (out / "battles.json").write_text(json.dumps([r["tag"] for r in index]))
    (out / "rows.json").write_text(json.dumps({
        "dataset": dataset, "wait_keep": wait_keep, "seed": seed, "rows": rows,
        "battles": len(index), "max_entities": MAX_ENTITIES,
        "arrays": {n: {"dtype": d, "shape": shapes.get(n, [rows])} for n, (d, _) in ARRAYS.items()},
    }, indent=1))
    print(f"{len(index)} battles, {rows:,} rows written to {out}")
    return out


class Rows:
    """A rows folder, its arrays mapped from disk (``mmap``, the default) or loaded into RAM.

    Mapped, a folder larger than RAM still trains: the operating system keeps the pages read
    most in its cache and reads the rest from the SSD as batches need them.
    """

    def __init__(self, folder: Path, mmap: bool = True):
        self.folder = Path(folder)
        self.info = json.loads((self.folder / "rows.json").read_text())
        self.metadata = json.loads((self.folder / "metadata.json").read_text())
        for name, a in self.info["arrays"].items():
            path = self.folder / f"{name}.bin"
            shape = tuple(a["shape"])
            if mmap and np.prod(shape) > 0:
                data = np.memmap(path, dtype=a["dtype"], mode="r", shape=shape)
            else:
                data = np.fromfile(path, dtype=a["dtype"]).reshape(shape)
            setattr(self, name, data)
        self.n = self.info["rows"]

    def gather(self, idx: np.ndarray, max_entities: int) -> dict:
        """A batch: arrays for rows ``idx``, entities padded to the batch's longest row."""
        starts = self.entity_offsets[idx]
        counts = np.minimum(self.entity_offsets[idx + 1] - starts, max_entities)
        # Widths rounded up to a multiple of 16, so the GPU sees few distinct shapes and its memory
        # cache stays small.
        width = min(max(16, -(-int(counts.max()) // 16) * 16), max(max_entities, 1))
        valid = np.arange(width)[None, :] < counts[:, None]
        ents = np.zeros((len(idx), width, self.entities.shape[1]), np.int32)
        ents[valid] = self.entities[(starts[:, None] + np.arange(width)[None, :])[valid]]
        return {"header": self.header[idx], "entities": ents, "valid": valid,
                "masks": self.masks[idx], "actions": self.actions[idx], "weight": self.weight[idx],
                "result": self.result[idx], "piggies": self.piggies[idx],
                "opp_hand": self.opp_hand[idx], "opp_next": self.opp_next[idx],
                "label_masked": self.label_masked[idx]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Gather training rows from compacted battles")
    parser.add_argument("--dataset", required=True, help="a folder under <data>/dataset")
    parser.add_argument("--wait-keep", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    build(args.dataset, args.wait_keep, args.seed)


if __name__ == "__main__":
    main()
