"""M3's exit test: does the PPO specialist beat the imitation model on decks and a pilot it never
trained against, and does it still play like a person?

    python -m clashai.eval.exit_test [--candidate <ppo best.pt>] [--baseline <imitation best.pt>]

Five scorecards, every model at the same sampling (card 0.7, tile 0.3 by default: the best of the
temperature sweep for every model), on seeds no training or model choice has used:

1. candidate vs the held-out meta decks (``deck_sets`` ``exit``), piloted by the held-out pilot
2. baseline, the same
3. candidate vs the held-out meta decks, piloted by the training pilot (PPO's reference)
4. baseline, the same
5. the mirror: candidate's Piggies EQ against baseline's Piggies EQ

Checks, decided before the results:

- **A** (pass needed): 1 minus 2, paired game by game, has a 95% interval above 0
- **B** (reported): 3 minus 4, the same, against the pilot PPO trained with
- **C** (pass needed): the candidate's mirror score has a 95% interval above 50%
- **D** (pass needed): behaviour near human play. The candidate's median game is inside the human
  10th-90th percentile range on each measure (:mod:`clashai.eval.behaviour`), and each of the
  deck's cards is played at between half and twice its human share

Writes a Markdown report and the numbers as JSON under ``<data>/runs/exit``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .. import paths
from ..decks import PIGGIES_EQ
from .behaviour import MEASURES, human_reference, summarize
from .deck_sets import deck_set
from .scorecard import make_policy, run_scorecard
from .stats import paired_difference, wilson

HUMAN_DATASET = "dataset/all-screen-team0-v0-t0-standard"


def keyed(record: dict) -> dict:
    return {(deck, g["seed"], g["a_seat"]): g["score"] for deck, gs in record.items() for g in gs}


def training_pilot(candidate: Path) -> Path | None:
    """The pilot a PPO run trained against: its ``--reference``, else its ``--init``."""
    args = candidate.parent / "args.json"
    if not args.exists():
        return None
    with open(args, encoding="utf-8") as f:
        a = json.load(f)
    ref = a.get("reference") or a.get("init")
    return Path(ref) if ref else None


def behaviour_check(human: dict, bot: dict, cards) -> tuple[bool, list[str]]:
    lines, ok = [], True
    lines.append("| measure | human 10th | human median | human 90th | candidate median | inside |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for m in MEASURES:
        h, b = human.get(m), bot.get(m)
        if h is None or b is None:
            ok = False
            lines.append(f"| {m} | - | - | - | - | **no data** |")
            continue
        inside = h["p10"] <= b["p50"] <= h["p90"]
        ok &= inside
        lines.append(f"| {m} | {h['p10']:.2f} | {h['p50']:.2f} | {h['p90']:.2f} | "
                     f"{b['p50']:.2f} | {'yes' if inside else '**no**'} |")
    lines += ["", "| card | human share | candidate share | inside 0.5x-2x |", "|---|---:|---:|---|"]
    hs, bs = human["card_share"], bot["card_share"]
    for card in sorted(hs, key=hs.get, reverse=True)[:len(cards)]:
        inside = 0.5 * hs[card] <= bs.get(card, 0.0) <= 2 * hs[card]
        ok &= inside
        lines.append(f"| {card} | {hs[card]:.1%} | {bs.get(card, 0.0):.1%} | "
                     f"{'yes' if inside else '**no**'} |")
    return ok, lines


def main(argv: list[str] | None = None) -> None:
    runs = paths.runs_dir()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--candidate", type=Path, default=runs / "ppo" / "ppo2" / "best.pt")
    p.add_argument("--baseline", type=Path, default=runs / "bc" / "generalist-v2" / "best.pt")
    p.add_argument("--held-out-pilot", type=Path, default=runs / "bc" / "generalist" / "best.pt",
                   help="a pilot the candidate never trained against")
    p.add_argument("--training-pilot", type=Path,
                   help="default: the candidate run's reference (its --init)")
    p.add_argument("--card-temperature", type=float, default=0.7)
    p.add_argument("--tile-temperature", type=float, default=0.3)
    p.add_argument("--seeds", type=int, default=50, help="per held-out deck; 2 games each")
    p.add_argument("--mirror-seeds", type=int, default=200)
    p.add_argument("--first-seed", type=int, default=1_500_000,
                   help="clear of the in-run scorecards' and the sweep's seeds (from 1,000,000)")
    p.add_argument("--threads", type=int, default=None)
    args = p.parse_args(argv)

    trainer = args.training_pilot or training_pilot(args.candidate) or args.baseline
    sampling = {"card_temperature": args.card_temperature,
                "tile_temperature": args.tile_temperature}
    loaded: dict = {}

    def policy(path: Path):
        key = str(Path(path).resolve())
        if key not in loaded:
            loaded[key] = make_policy(str(path), **sampling)
        return loaded[key]

    cand, base = policy(args.candidate), policy(args.baseline)
    held_pilot, train_pilot = policy(args.held_out_pilot), policy(trainer)
    if held_pilot is train_pilot:
        raise SystemExit("the held-out pilot is the training pilot: pass another --held-out-pilot")
    decks = deck_set("exit")
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    mirror_seeds = list(range(args.first_seed, args.first_seed + args.mirror_seeds))
    print(f"exit test: {cand.name} against {base.name}; {len(decks)} held-out decks; held-out "
          f"pilot {held_pilot.name}, training pilot {train_pilot.name}", flush=True)

    t0 = time.time()
    results = {}
    plan = [("heldout_candidate", cand, held_pilot, decks, seeds, True),
            ("heldout_baseline", base, held_pilot, decks, seeds, True),
            ("training_candidate", cand, train_pilot, decks, seeds, False),
            ("training_baseline", base, train_pilot, decks, seeds, False),
            ("mirror", cand, base, {"piggies_eq": PIGGIES_EQ}, mirror_seeds, False)]
    for label, a, b, ds, ss, record_behaviour in plan:
        t = time.time()
        r = run_scorecard(a, b, ds, ss, threads=args.threads, behaviour=record_behaviour,
                          tag="exit" if label != "mirror" else "mirror",
                          args={**vars(args), "part": label}, quiet=True)
        results[label] = r
        print(f"  {label}: {r['score']:.1%} over {sum(len(g) for g in r['games'].values())} games "
              f"[{time.time() - t:.0f} s] -> {Path(r['path']).name}", flush=True)

    def diff(x, y):
        return paired_difference(keyed(results[x]["games"]), keyed(results[y]["games"]))

    a_mean, a_low, a_high, a_n = diff("heldout_candidate", "heldout_baseline")
    b_mean, b_low, b_high, b_n = diff("training_candidate", "training_baseline")
    mirror = [g["score"] for gs in results["mirror"]["games"].values() for g in gs
              if g["score"] is not None]
    c_low, c_high = wilson(sum(mirror), len(mirror))
    c_score = sum(mirror) / len(mirror)

    human = human_reference(paths.data_root() / HUMAN_DATASET, PIGGIES_EQ.cards,
                            cache=paths.data_root() / "eval" / "human-behaviour-piggies_eq.json")
    bot = summarize([g["behaviour"] for gs in results["heldout_candidate"]["games"].values()
                     for g in gs if g.get("behaviour")])
    base_bot = summarize([g["behaviour"] for gs in results["heldout_baseline"]["games"].values()
                          for g in gs if g.get("behaviour")])
    d_ok, d_lines = behaviour_check(human, bot, PIGGIES_EQ.cards)

    a_ok, c_ok = a_low > 0, c_low > 0.5
    passed = a_ok and c_ok and d_ok

    def yes(ok):
        return "**pass**" if ok else "**fail**"

    def pts(m, lo, hi):
        return f"{100 * m:+.1f} points ({100 * lo:+.1f} to {100 * hi:+.1f})"

    lines = [
        f"# M3 exit test: {cand.name} against {base.name}",
        "",
        (f"{time.strftime('%Y-%m-%d %H:%M')}. Every model samples at card {args.card_temperature:g}, "
        f"tile {args.tile_temperature:g}. Held-out decks: {len(decks)} most played held-out decks "
        f"of the replays, {args.seeds} seeds x 2 seats each, from seed {args.first_seed:,}. "
        f"Held-out pilot: {held_pilot.name}. Training pilot: {train_pilot.name}."),
        "",
        f"**Result: {'PASS' if passed else 'FAIL'}**",
        "",
        "| check | result | |",
        "|---|---|---|",
        (f"| A. Held-out decks and pilot: candidate minus baseline | {pts(a_mean, a_low, a_high)}, "
        f"{a_n} pairs | {yes(a_ok)} |"),
        (f"| B. Held-out decks, training pilot: candidate minus baseline | "
        f"{pts(b_mean, b_low, b_high)}, {b_n} pairs | reported |"),
        (f"| C. Mirror: candidate's score against baseline | {c_score:.1%} ({c_low:.1%} to "
        f"{c_high:.1%}), {len(mirror)} games | {yes(c_ok)} |"),
        f"| D. Behaviour near human play | see below | {yes(d_ok)} |",
        "",
        "## Scores",
        "",
        ("| deck | candidate, held-out pilot | baseline, held-out pilot | candidate, training pilot "
        "| baseline, training pilot |"),
        "|---|---:|---:|---:|---:|",
    ]
    for name, d in decks.items():
        row = [results[k]["by_deck"][name] for k in ("heldout_candidate", "heldout_baseline",
                                                      "training_candidate", "training_baseline")]
        lines.append(f"| {name} ({', '.join(d.cards[:3])}...) | "
                     + " | ".join(f"{x:.0%}" for x in row) + " |")
    lines.append("| **all** | " + " | ".join(
        f"**{results[k]['score']:.1%}**" for k in ("heldout_candidate", "heldout_baseline",
                                                    "training_candidate", "training_baseline"))
        + " |")
    lines += ["", (f"## Behaviour (candidate against the held-out pilot; humans: {human['games']} "
              f"Piggies EQ games)"), "", *d_lines, "",
              "Baseline medians for comparison: " + ", ".join(
                  f"{m} {base_bot[m]['p50']:.2f}" for m in MEASURES if base_bot.get(m)), "",
              f"Took {(time.time() - t0) / 60:.0f} min."]

    out = runs / "exit"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{time.strftime('%Y%m%d-%H%M%S')}-{cand.name}-vs-{base.name}"
    (out / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(out / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump({"args": {k: str(v) for k, v in vars(args).items()}, "passed": passed,
                   "A": [a_mean, a_low, a_high, a_n], "B": [b_mean, b_low, b_high, b_n],
                   "C": [c_score, c_low, c_high, len(mirror)], "D": d_ok,
                   "human": human, "candidate_behaviour": bot, "baseline_behaviour": base_bot,
                   "scorecards": {k: str(r["path"]) for k, r in results.items()}}, f, indent=1)
    print("\n".join(lines))
    print(f"\nreport written to {out / (stem + '.md')}")


if __name__ == "__main__":
    main()
