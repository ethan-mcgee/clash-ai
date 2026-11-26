"""Training runs as TensorBoard curves, mirrored from the files the trainers already write.

    python -m clashai.train.board --follow
    tensorboard --logdir D:\\ml-data\\clash-ai\\runs\\tb

Reads every run under ``<data>/runs/{bc,seq,ppo}/<name>``: its ``metrics.jsonl`` (one validation
or iteration a line), ``evals.jsonl`` (PPO's in-run scorecards) and ``train.log`` (the whole-battle
trainer's ``step N loss ...`` lines), and writes TensorBoard event files to
``<data>/runs/tb/<kind>/<name>``. The trainers are not touched and ``metrics.jsonl`` stays the
source of truth, so a run that is training now shows up too; ``--follow`` keeps reading new lines.

Curves are grouped: ``loss/`` (the policy loss and its gate, card and tile parts), ``accuracy/``,
``memory/`` (the cost of zeroing a recurrent model's state or hiding its previous action),
``train/`` (training-side means between validations), ``live/`` and ``speed/`` (from the log),
``ppo/``, ``eval/`` and ``run/``. Their step is the training step, or the PPO iteration.

``by_plays/`` repeats the main curves against the play examples trained on so far, in thousands,
so runs with different batches compare at equal data: the whole-battle trainer records
``plays_seen``; for the row trainer it is the step times the batch times the rows' share of plays.

Each run folder keeps ``.board.json``: how far each file was read, so a restart of this mirror
adds only what is new. A run started over (its file shorter than before) is mirrored afresh.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path

import numpy as np

from .. import paths

KINDS = ("bc", "seq", "ppo")
STATE = ".board.json"
BY_PLAYS = ("policy_nll", "gate_nll", "card_nll", "tile_nll", "card_acc", "cell_acc",
            "act_calibration", "zero_state_cost")
RUN_KEYS = {"minutes", "max_memory_gb", "skipped_chunks", "passes", "plays_seen", "decisions",
            "plays", "rollout_s", "update_s", "games", "mirror_games", "game_seconds"}
LIVE = re.compile(r"^\s+step (\d+) loss ([\d.eE+-]+|nan) policy ([\d.eE+-]+|nan) "
                  r"\(([\d,]+) rows/s, waiting for data (\d+)%(?:, lr ([\d.eE+-]+))?\)")


def tag(kind: str, key: str) -> str:
    """The TensorBoard tag of a metrics key: its group, then the key."""
    if key in RUN_KEYS:
        return f"run/{key}"
    if key.startswith(("zero_state", "hide_prev")):
        return f"memory/{key}"
    if kind == "ppo":
        return f"ppo/{key}"
    if key == "loss" or key.endswith(("_nll", "_mse", "_loss")):
        return f"loss/{key}"
    return f"accuracy/{key}"


def scalars(kind: str, row: dict) -> dict[str, float]:
    """Every number in one metrics line, by tag: nested ``train`` means and lists included."""
    out = {}
    for key, value in row.items():
        if key in ("step", "iteration") or isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int | float):
            out[tag(kind, key)] = float(value)
        elif isinstance(value, dict):
            out.update({f"{key}/{k}": float(v) for k, v in value.items()
                        if isinstance(v, int | float) and not isinstance(v, bool)})
        elif isinstance(value, list) and all(isinstance(v, int | float) for v in value):
            out.update({f"{key}/{i + 1}": float(v) for i, v in enumerate(value)})
    return out


def plays_per_step(run: Path, data: Path) -> float | None:
    """For a row-trainer run: play examples a training step, from its batch and its rows' share
    of plays (read once from the rows' actions)."""
    try:
        args = json.loads((run / "args.json").read_text())
        folder = data / "rows" / args["rows"]
        info = json.loads((folder / "rows.json").read_text())
        actions = np.memmap(folder / "actions.bin", dtype=info["arrays"]["actions"]["dtype"],
                            mode="r")
        return float(args["batch"] * np.count_nonzero(actions) / len(actions))
    except (OSError, KeyError, ValueError):
        return None


def new_lines(path: Path, state: dict) -> tuple[list[str], bool]:
    """The complete lines of ``path`` after the offset in ``state``, which is advanced; and
    whether the file was started over (shorter than the offset)."""
    if not path.exists():
        return [], False
    offset = state.get(path.name, 0)
    size = path.stat().st_size
    restarted = size < offset
    if restarted:
        offset = 0
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read()
    end = data.rfind(b"\n") + 1  # a line still being written waits for the next pass
    state[path.name] = offset + end
    return data[:end].decode("utf-8", "replace").splitlines(), restarted


class Run:
    """One run's mirror: reads what is new in its files and writes it as events."""

    def __init__(self, kind: str, folder: Path, out: Path, data: Path):
        self.kind, self.folder, self.out, self.data = kind, folder, out, data
        self.writer = None
        self.state = json.loads((out / STATE).read_text()) if (out / STATE).exists() else {}
        self._per_step: float | None = None

    def _write(self, name: str, value: float, step: int, wall: float) -> None:
        if self.writer is None:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(str(self.out))
        self.writer.add_scalar(name, value, step, walltime=wall)

    def _start(self) -> float:
        first = self.folder / "args.json"
        return (first if first.exists() else self.folder / "metrics.jsonl").stat().st_mtime

    def _plays(self, row: dict) -> float | None:
        if "plays_seen" in row:
            return float(row["plays_seen"])
        if self.kind == "bc" and "step" in row:
            if self._per_step is None:
                self._per_step = plays_per_step(self.folder, self.data) or 0.0
            return row["step"] * self._per_step if self._per_step else None
        return None

    def update(self) -> int:
        """Mirrors what is new; returns the number of points written."""
        state = dict(self.state)
        read = {name: new_lines(self.folder / name, state)
                for name in ("metrics.jsonl", "evals.jsonl", "train.log")}
        if any(restarted for _, restarted in read.values()):
            self.close()
            shutil.rmtree(self.out, ignore_errors=True)
            self.state = {}
            return self.update()
        start, written = self._start(), 0
        for line in read["metrics.jsonl"][0]:
            row = json.loads(line)
            step = int(row.get("step", row.get("iteration", 0)))
            wall = start + 60 * float(row.get("minutes", 0))
            values = scalars(self.kind, row)
            for name, value in values.items():
                self._write(name, value, step, wall)
            plays = self._plays(row)
            if plays is not None:
                for key in BY_PLAYS:
                    if isinstance(row.get(key), int | float):
                        self._write(f"by_plays/{key}", float(row[key]), round(plays / 1000), wall)
            written += len(values)
        for line in read["evals.jsonl"][0]:
            row = json.loads(line)
            step = int(row.get("iteration", 0))
            if isinstance(row.get("score"), int | float):
                self._write("eval/score", float(row["score"]), step, start)
                written += 1
            for deck, score in (row.get("by_deck") or {}).items():
                if isinstance(score, int | float):
                    self._write(f"eval_deck/{deck}", float(score), step, start)
        now = time.time()
        for line in read["train.log"][0]:
            m = LIVE.match(line)
            if not m:
                continue
            step = int(m[1])
            self._write("live/loss", float(m[2]), step, now)
            self._write("live/policy", float(m[3]), step, now)
            self._write("speed/rows_per_s", float(m[4].replace(",", "")), step, now)
            self._write("speed/waiting_for_data_percent", float(m[5]), step, now)
            if m[6]:
                self._write("live/lr", float(m[6]), step, now)
            written += 1
        if state != self.state:
            self.state = state
            if self.writer is not None:
                self.writer.flush()
            self.out.mkdir(parents=True, exist_ok=True)
            (self.out / STATE).write_text(json.dumps(state))
        return written

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def mirror(runs: Path, out: Path, data: Path, mirrors: dict, only: list[str] | None = None) -> int:
    """One pass over every run under ``runs``; ``mirrors`` keeps each run's state between passes."""
    written = 0
    for kind in KINDS:
        for metrics in sorted((runs / kind).glob("*/metrics.jsonl")):
            folder = metrics.parent
            if only and folder.name not in only:
                continue
            key = (kind, folder.name)
            if key not in mirrors:
                mirrors[key] = Run(kind, folder, out / kind / folder.name, data)
            written += mirrors[key].update()
    return written


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Mirror training runs into TensorBoard event files")
    p.add_argument("--follow", action="store_true", help="keep mirroring new lines until stopped")
    p.add_argument("--interval", type=float, default=30, help="seconds between passes")
    p.add_argument("--runs", nargs="*", help="run names (default: all)")
    args = p.parse_args(argv)
    runs = paths.runs_dir()
    out = runs / "tb"
    mirrors: dict = {}
    try:
        n = mirror(runs, out, paths.data_root(), mirrors, args.runs)
        print(f"{len(mirrors)} runs, {n:,} points written to {out}\n"
              f"view with: tensorboard --logdir {out}", flush=True)
        while args.follow:
            time.sleep(args.interval)
            mirror(runs, out, paths.data_root(), mirrors, args.runs)
    except KeyboardInterrupt:
        pass
    finally:
        for run in mirrors.values():
            run.close()


if __name__ == "__main__":
    main()
