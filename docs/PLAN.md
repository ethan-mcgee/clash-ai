# Piggies EQ specialist on crforge, with our own training stack

## Context

Goal: train one agent to play **one deck, Piggies EQ, extremely well**, on one desktop: an RTX 3090 Ti, 64 GB RAM and a 12-core / 24-thread Ryzen 9 7900X. Offline only; it never plays real people.

The first plan built on FirstLight_CR. That route is dead: Null's server now forces game version 16, and the content FirstLight needs can't be downloaded any more.

Decision: **train on crforge only** (voonhous/crforge, Apache-2.0, Java 21). It is a separate simulator from RoyaleSim, and the more accurate one:
- 20,763 of 20,763 recorded real battles match on every tick, on the current game data (16.635.8).
- All 1,137 targeted test battles match, including Evo Royal Hogs, Evo Cannon and Archer Queen's ability.
- About 127,000 steps a second per CPU thread.

The model, data pipeline, league and evaluation are written in this repo.

Deck: Evo Royal Hogs, Archer Queen, Evo Cannon, Skeletons, Ice Spirit, The Log, Earthquake, Royal Delivery.

## What crforge already gives us

Read from its docs before any of it had run here; M0 verifies each item.

| Need | crforge feature |
|---|---|
| Many battles at once | `VectorBattle(count, threads)`: battles step on JVM threads inside one Python process (via JPype); observations are numpy arrays that share memory with the engine |
| Fair observations | Schema v4 shows a seat only what a player sees; `require_fair` rejects any model input that reads a hidden field |
| Opponent elixir | Included in the observation "as a player counts it" |
| Legal moves | A per-decision action mask; plays run 20 ticks after they're given, as in the game |
| Levels and tower troops | `SideSetup` per side |
| Search | `fork_from` copies a battle and re-deals the opponent's unseen cards from a belief we supply |
| Baselines | `ScriptedBot` and `evaluate()` |
| Demo | Its desktop visualizer can play a human against a model exported to ONNX |

What it does **not** give us: it can't read our replay dataset directly. Its replay labeller needs the game's own replay files (with the random seed). Our data is RoyaleAPI's format: plays with card, coordinates and tick, but no seed and no opening hand. We write that conversion (M1).

## Lessons carried over from the projects we studied

| Lesson | Source | Where it lands |
|---|---|---|
| Imitation first, then PPO anchored to the human policy | FirstLight, Daybreak, RoyaleFish, ClashAI | M2–M3 |
| Specialists beat generalists | FirstLight 88%, RoyaleFish 66–68% | One fixed deck |
| Drop replays the simulator can't reproduce | Daybreak | Consistency filter, M1 |
| Paired seeds and confidence intervals | ClashAI | Evaluation harness, M0 |
| Lookahead search, distilled back | itzik | M4 |
| Train against the matchups you lose | RoyaleFish | League, M3 |

| Failure seen elsewhere | Our guard |
|---|---|
| Scored against opponents from the training pool | Held-out decks and pilots, fixed before training |
| Trained with a 1-tick card delay; 68% fell to 36% with the real one | crforge enforces the 20-tick delay; a test confirms it |
| Dozens of hand-tuned reward bonuses | Win/loss plus potential-based tower shaping only |
| Imitation from won games only | Lost games included, outcome-weighted; lane mirroring |
| Opponent's hand ignored | Hand-belief head, used by the policy and by search |
| Live ladder play | Offline only, stated in the README |

## Approach

### Environment
- crforge's `VectorBattle`, decisions every 5 ticks (0.25 s), levels and tower troops per side.
- Model input is crforge's v4 observation and mask, so the trained model can be exported to ONNX for the visualizer.
- Action: crforge's own action integer (wait, play slot × tile, ability press), masked.

### Model
- Entity transformer over the observation's rows, about 10–13M parameters, bf16 on the GPU. No recurrent state; recent plays are already in the observation.
- Heads: action (card, then cell given the card), value.
- Auxiliary heads: the opponent's unseen cards and next play. Targets come from crforge's hidden fields during training; the policy input never reads them.

### Data: RoyaleAPI replays into crforge
- Dataset: `A1A2CV3/IL_Replay_1115`, 156,160 top-ladder replays from Nov 4–15, 2026. That is the 16.635.8 period crforge supports.
- Our own Parquet parser (schema already mapped: decks, forms, levels, tower troops, plays with coordinates and 20 Hz tick, final tower HP).
- **Opening-hand solver:** find a starting hand and draw order under which every recorded play was possible.
- **Converter:** set up the battle, feed both sides' recorded plays at their ticks through the normal environment, and save (observation, mask, action) at every decision.
- **Consistency filter:** compare final tower HP and crowns with the recorded result; keep samples up to the first refused play. The pass rate is reported.
- Three things to settle by experiment in M1:
  - how to force a chosen opening hand through crforge's Python API;
  - whether RoyaleAPI's tick is when a play was sent or when it landed;
  - how much the unknown random seed makes battles drift.

### Training
1. **Generalist by imitation** on a broad slice of the replays (start near 30k games). It pilots every opponent deck.
2. **Specialist by imitation:** all Piggies EQ games, wins and losses, outcome-weighted, lanes mirrored.
3. **PPO league:**
   - Rewards: win/loss plus potential-based tower-HP shaping. Nothing deck-specific.
   - KL penalty toward the imitation specialist.
   - Opponent decks drawn at the rate Piggies EQ meets them in the replays, weighted toward its worst matchups.
   - Pilots: the generalist, past versions picked by how often we lose to them, and one exploiter trained only to beat the specialist.
   - Promotion only at 55% or better against the current champion on paired seeds.
4. **Search:** fork the battle, deal the opponent's unseen cards from the belief head, roll out the top candidate plays, pick the best; distil into the network.

### Evaluation, built before any training
- Held-out opponent decks and pilots, fixed in M0.
- Paired seeds, alternating sides, Wilson 95% intervals, per-archetype scorecard.
- Behaviour metrics against human Piggies EQ players: time to first play, elixir wasted at 10, ability presses per game, Earthquake targets.

## Setup

**1. Secure the game data first (time-sensitive)**
- Install JDK 21 (`winget install EclipseAdoptium.Temurin.21.JDK`).
- Add crforge as a pinned submodule at `third_party/crforge` (since vendored with `git subtree`; see `docs/CRFORGE.md`).
- Build its engine and the 16.635.8 tables: `gradlew :env:installDist :tables:dataVersionTables -Pcrforge.assetSource=cdn -Pcrforge.cache=D:\ml-data\clash-ai\crforge-cache`.
  - This downloads the game's data files from Supercell's asset server to your machine. They are never committed.
  - Supercell's server reportedly stops serving old data versions after an update, so this comes first, with a backup copy of the cache on D:.
- If the build fails on native Windows, fall back to WSL2.

**2. Python environment**
- New Python 3.12 venv at `D:\ml-data\clash-ai\venv`: PyTorch 2.5.1 CUDA 12.4, `crforge-env` (from `third_party/crforge/python`), `pyarrow`, `pytest`.
- Run crforge's own Python tests and benchmark.

## Milestones

**M0. Foundations**
- Package skeleton, environment wrapper, model-input adapter, evaluation harness, held-out deck split.
- Tests: plays land 20 ticks after they're given; `require_fair` passes; same seeds give identical results; every unmasked action is accepted; mirroring round-trips.
- A Piggies EQ smoke match: every card, both evolutions and the Archer Queen ability play without a "not modelled" refusal.
- Throughput at 1, 6, 12 and 24 threads (24 measured after the BIOS change re-enabled all 12 cores).
- Exit: tests pass; a games-per-hour number; a scripted-bot scorecard with intervals.

**M1. Data**
- Download and audit: Piggies EQ count by variant, win rate, opponent-deck frequencies.
- Parser, opening-hand solver, converter, consistency filter.
- Exit: the audit table, the consistency pass rate, a converted Piggies EQ set. If the exact deck has under about 1,500 games, add one-card-swap variants.

**M2. Imitation**
- Generalist, then specialist.
- Exit: held-out card and cell accuracy reported (RoyaleFish's 63.5% and 31.5% as a rough yardstick); the specialist beats the generalist piloting Piggies EQ.
- Result (`docs/IMITATION.md`): card 70.1% and tile 34.7% on Piggies EQ validation, above the yardstick. The specialist does not beat the generalist in games (41.3% against 42.6%, intervals overlapping), so that criterion is not met; M3 starts from the generalist.

**M3. PPO league**
- Exit: on held-out decks and pilots, the PPO specialist beats the imitation specialist with an interval clear of 50%, and behaviour metrics stay near human ranges.

**M4. Search**
- Exit: measured win-rate gain and decision latency.

**M5. Ablations, demo and write-up**
- Remove one idea at a time: lost-game data, hand belief, matchup weighting, KL anchor, search.
- Export to ONNX and play against the agent in crforge's visualizer.
- Scorecard, curves, a recorded match, and a limits section.

## Layout

```
clash-ai/
  pyproject.toml  NOTICE  README.md
  third_party/crforge/          vendored crforge (git subtree)
  configs/piggies_eq.yaml
  clashai/
    sim/     env.py  adapter.py  mirror.py
    data/    download.py  replay.py  hand_solver.py  convert.py  audit.py  dataset.py
    model/   net.py  heads.py
    train/   bc.py  ppo.py  league.py
    eval/    arena.py  stats.py  scorecard.py  behaviour.py
    search/  lookahead.py
    export/  onnx.py
  tests/
  docs/      PLAN.md  RESEARCH.md  SETUP.md
```

Data, the crforge cache, checkpoints and runs live under `D:\ml-data\clash-ai`, never in the repo.

## Risks

- **Unproven on this PC.** crforge's docs target macOS and Linux. JPype and Gradle both support Windows, but nobody has shown this stack running there. WSL2 is the fallback.
- **Replay conversion is the hard part.** No project has published RoyaleAPI-to-crforge conversion. If the consistency pass rate is low, imitation data shrinks; the fallbacks are looser filtering and more self-play.
- **Game data provenance.** crforge's tables are built from Supercell's files, and the replays come from RoyaleAPI. The write-up says so plainly; the repo ships neither.
- **Game updates.** Supercell will move past 16.635.8. We pin crforge's commit and keep our cached tables, so the project stays reproducible on the November data.
- **Piggies EQ sample size** is unknown until the audit.
- **Throughput** is likely limited by GPU inference, not the engine. The estimate is tens of thousands of games an hour; M0 measures it.
- **Scope.** M0–M3 are the project. M4 and M5's demo can be cut without losing a complete result.

## Verification

- `pytest` green on every commit, including the delay, fairness, determinism and mask tests.
- First steps: crforge's own test suite and benchmark pass; the tables' data version reads 16.635.8.
- M0: the arena gives identical results from the same seeds.
- M1: audit counts spot-checked against the Hugging Face viewer; one replay hand-checked play by play.
- M2: validation split grouped by `replay_tag`, so both sides of a battle stay together.
- M3–M5: every claimed gain is a paired-seed comparison of at least 400 games with a 95% interval, on held-out decks and pilots only.
