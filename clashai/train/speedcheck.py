"""The whole-battle trainer's speed and GPU load in each ``--graph`` mode, as one table.

    python -m clashai.train.speedcheck --modes off cuda              # Windows
    python -m clashai.train.speedcheck --modes off cuda compile      # Linux

Each mode trains a new network for ``--steps`` steps in a run of its own (``zz-speed-<mode>``,
removed afterwards) while the GPU's utilisation, power and memory are sampled once a second. The
first log windows are left out: they hold the starting validation, the warm-up steps and the
recording or compiling. About 8 minutes a mode at the default 1,000 steps.

Run it on an otherwise idle GPU: it measures whatever else is running too.
"""

from __future__ import annotations

import argparse
import re
import shutil
import statistics
import subprocess
import sys
import threading

from .. import paths
from .graphed import MODES

WINDOW = re.compile(r"^\s+step (\d+) loss \S+ policy \S+ \(([\d,]+) rows/s, waiting for data "
                    r"(\d+)%, lr \S+?(?:, recorded steps (\d+)%)?\)")
SKIP = 2  # log windows left out at the start


def windows(lines) -> list[dict]:
    """The trainer's 200-step log lines as rows/s, the data wait and the recorded share."""
    out = []
    for line in lines:
        m = WINDOW.match(line)
        if m:
            out.append({"step": int(m[1]), "rows_per_s": float(m[2].replace(",", "")),
                        "waiting": int(m[3]), "recorded": None if m[4] is None else int(m[4])})
    return out


def summary(mode: str, found: list[dict], gpu: list[tuple[float, float, float]],
            skip: int = SKIP) -> dict:
    """One mode's row of the table from its log windows and GPU samples (utilisation %, watts,
    MiB), the samples of the left-out windows left out in proportion."""
    kept = found[skip:]
    if not kept:
        return {"mode": mode, "error": f"only {len(found)} log windows; need more than {skip}"}
    samples = gpu[len(gpu) * skip // max(1, len(found)):]
    recorded = [w["recorded"] for w in kept if w["recorded"] is not None]
    return {"mode": mode, "rows_per_s": statistics.median(w["rows_per_s"] for w in kept),
            "waiting": max(w["waiting"] for w in kept),
            "recorded": statistics.mean(recorded) if recorded else None,
            "gpu_util": statistics.mean(s[0] for s in samples) if samples else None,
            "gpu_watts": statistics.mean(s[1] for s in samples) if samples else None,
            "gpu_mib": max(s[2] for s in samples) if samples else None}


def _sample_gpu(stop: threading.Event, out: list) -> None:
    query = ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw,memory.used",
             "--format=csv,noheader,nounits", "-l", "1"]
    try:
        with subprocess.Popen(query, stdout=subprocess.PIPE, text=True) as p:
            for line in p.stdout:
                if stop.is_set():
                    p.terminate()
                    break
                parts = line.split(",")
                if len(parts) == 3:
                    try:
                        out.append(tuple(float(x) for x in parts))
                    except ValueError:
                        pass
    except OSError:
        pass  # no nvidia-smi: the table goes without the GPU columns


def measure(mode: str, steps: int, extra: list[str]) -> dict:
    name = f"zz-speed-{mode}"
    folder = paths.runs_dir() / "seq" / name
    shutil.rmtree(folder, ignore_errors=True)
    command = [sys.executable, "-m", "clashai.train.seq", "--name", name, "--graph", mode,
               "--steps", str(steps), "--eval-every", "1000000", "--val-seats", "16", *extra]
    gpu: list = []
    stop = threading.Event()
    sampler = threading.Thread(target=_sample_gpu, args=(stop, gpu), daemon=True)
    sampler.start()
    lines = []
    try:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True) as p:
            for line in p.stdout:
                lines.append(line)
                if WINDOW.match(line):
                    print(f"  [{mode}] {line.strip()}", flush=True)
            code = p.wait()
    finally:
        stop.set()
        shutil.rmtree(folder, ignore_errors=True)
    if code != 0:
        return {"mode": mode, "error": f"exit {code}: " + "".join(lines[-12:]).strip()}
    return summary(mode, windows(lines), gpu)


def table(rows: list[dict]) -> str:
    base = next((r["rows_per_s"] for r in rows if r["mode"] == "off" and "error" not in r), None)
    out = ["mode      rows/s   against off   recorded   GPU busy   power   GPU memory   data wait"]
    for r in rows:
        if "error" in r:
            out.append(f"{r['mode']:<8}  failed: {r['error']}")
            continue

        def cell(value, form, width):
            return ("-" if value is None else form.format(value)).rjust(width)

        out.append(f"{r['mode']:<8}{r['rows_per_s']:>8,.0f}"
                   + cell(r["rows_per_s"] / base if base else None, "{:.2f}x", 14)
                   + cell(r["recorded"], "{:.0f}%", 11) + cell(r["gpu_util"], "{:.0f}%", 11)
                   + cell(r["gpu_watts"], "{:.0f} W", 8)
                   + cell(None if r["gpu_mib"] is None else r["gpu_mib"] / 1024, "{:.1f} GB", 13)
                   + cell(r["waiting"], "{:.0f}%", 12))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Speed and GPU load of each --graph mode")
    p.add_argument("--modes", nargs="+", choices=MODES, default=["off", "cuda"])
    p.add_argument("--steps", type=int, default=1000)
    args, extra = p.parse_known_args(argv)  # anything else goes to clashai.train.seq
    rows = []
    for mode in args.modes:
        print(f"{mode}: {args.steps} steps", flush=True)
        rows.append(measure(mode, args.steps, extra))
    print("\n" + table(rows), flush=True)


if __name__ == "__main__":
    main()
