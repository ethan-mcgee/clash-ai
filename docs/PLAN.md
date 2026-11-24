# Clash Royale specialist agent: Piggies EQ on FirstLight_CR

## Context

Goal: a project that trains one agent to play **one deck extremely well** on a single RTX 3090 Ti (24 GB), 64 GB RAM, Windows 11.

Decisions so far:
- **Deck:** Piggies EQ (Evo Royal Hogs, Archer Queen, Evo Cannon, Skeletons, Ice Spirit, The Log, Earthquake, Royal Delivery).
- **Base:** build on FirstLight_CR, borrowing ideas from Daybreak. This replaces the earlier open-simulator plan.

## What the two repos actually give us

| | FirstLight_CR | Daybreak CR |
|---|---|---|
| Licence | Apache-2.0, so we can fork it | None, so we can read it but not copy code |
| Runnable from public files | Yes: engine build, replay cache, evaluation, play-against-it console all documented for Windows | No: needs a separate engine repo, a private dataset and a deck pool that aren't published |
| Checkpoints | Ships General, IL, Active IL, and two Hog 2.6 specialists | None published |
| Training scripts | bash/Slurm launchers written for an 8-GPU Linux cluster | Slurm, 3–4 A100s |
| Best use for us | **The foundation** | **A design reference** |

In practice: build on FirstLight, and reimplement the Daybreak ideas worth having.

Facts that shape the plan:
- FirstLight's engine is the real game's battle code, run headless inside a MuMu Android 12 emulator through a patched Null's Royale 15.535.13 (arm64) APK. **You supply that APK yourself**; neither FirstLight nor this repo ships it. Treat it as untrusted: it stays inside the emulator, which the setup script cuts off from the network.
- All eight Piggies EQ cards are in FirstLight's supported list. Evolution and Archer Queen ability bindings live in `form_evidence.json` and are not yet confirmed.
- FirstLight's Hog 2.6 specialist started from the shipped General checkpoint and beat it 88% (760 games). We can start from the same checkpoint and skip the cluster-scale imitation stage.
- The replay set (`VanguardX101/IL_Replay`, 252k replays) has plenty of Royal Hogs: 38,009 of the 91,970 replays indexed so far, with Earthquake in 7,814. Two of four sampled Royal Hogs replays were exactly this deck.
- FirstLight's docs admit two weaknesses: specialists play passively in some openings, and a high win rate once came from exploiting a passive opponent.
- Daybreak's live results came from queueing ranked ladder against real players. That is out of scope here: it breaks Supercell's terms and is unfair to the people on the other side. Everything in this plan is offline.

## Approach

Warm-start from FirstLight's General checkpoint and fine-tune a Piggies EQ specialist with PPO, on one PC.

Our own contributions:

1. **A single-GPU training driver.** FirstLight only documents multi-node Slurm training. A one-box driver (one learner, engines in one MuMu VM) is the piece that makes this reproducible on a desktop, and it's a real engineering result on its own.
2. **Human-anchored specialist training.** Add an imitation term on Piggies EQ human replays during PPO (Daybreak's recipe: coefficient 0.2, one imitation step per update) plus a KL penalty to the starting policy. This targets the passive-opening and opponent-exploitation failures FirstLight documents.
3. **A matchup-weighted opponent field.** Opponent decks are sampled at the rate Piggies EQ actually faces them in the replay data, with extra weight on its worst matchups. Opponent pilots are the shipped IL and General checkpoints plus frozen past selves.
4. **A per-matchup scorecard.** Win rate by opponent archetype, before and after, using FirstLight's own protocol (760 games per opponent, argmax, alternating sides) so the numbers are directly comparable to their Hog 2.6 results.

### Compute

- **GPU (CUDA, via PyTorch):** batched move selection for all concurrent matches, plus imitation and PPO gradient updates, with mixed precision. A 13M-parameter model leaves the 3090 Ti plenty of headroom.
- **CPU:** the game engine runs inside the emulator on CPU cores. This, not the GPU, limits matches per hour.
- **No Slurm.** Slurm is a cluster job scheduler that the upstream launchers assume. On one PC a plain launcher script replaces it.

### Milestones

**M0. Get FirstLight running**
- Add FirstLight_CR as the pinned submodule. Python 3.12 venv, MuMu Player (Android 12, root, ARM64), JDK 17, Android NDK and build-tools.
- Run `check_original_game.ps1`, `build_offline_engine.ps1`, `install_offline_engine.ps1`, then `launch_interface.cmd`.
- Exit: a human plays a match against the shipped Hog 2.6 specialist in the console, and `evaluate` reproduces a slice of the published General vs IL result.

**M1. Measure and audit**
- Throughput: games per hour at 1, 8, 24 engine processes, with CPU and RAM use. This sets the whole RL budget.
- Read `battle_env.py`, the observation and action contracts, `form_evidence.json`; confirm Evo Royal Hogs, Evo Cannon and the Archer Queen ability work in a scripted match.
- Download the replay set and audit it: exact Piggies EQ count by variant, win rate, opponent-deck frequencies.
- Baseline: shipped General and IL checkpoints piloting Piggies EQ against the matchup-weighted field.
- Exit: a throughput number, a variant table, a baseline scorecard. If the exact variant has under about 1,500 replays, include one-card-swap variants.

**M2. Piggies EQ imitation data**
- Build the replay cache for Piggies EQ replays only, with `cache_builder` and `--verify-roundtrip`.
- Short imitation fine-tune of General on that cache as a sanity check.
- Exit: held-out card and tile accuracy on Piggies EQ decisions, and a first scorecard versus the baseline.

**M3. Single-GPU PPO specialist**
- Write the one-box driver around FirstLight's v4 trainer (`train_ppo_self_play_cluster.py`, env keys `CR_PPO_CHECKPOINT`, `CR_PPO_DECK_POOL`). Try native Windows first, since one GPU doesn't need the Linux-only multi-GPU backend; fall back to WSL2.
- Fixed deck, matchup-weighted opponents, imitation term and KL penalty.
- Rewards as in FirstLight's specialist 2: win/loss, tower-health shaping, elixir-overflow penalty. No deck-specific rewards.
- Promote a new champion only at 55% or better against the current one (Daybreak's gate).
- Exit: specialist beats General-on-Piggies-EQ with a confidence interval that clears 50%, and opening passivity is measured (time to first play).

**M4. Ablations and write-up**
- Three runs with one idea removed each: no imitation term, uniform opponent decks, no KL penalty.
- Scorecard, training curves, a recorded match from the console, and an honest limits section.

**Optional experiments, after M4**
- Bigger board encoder. Daybreak notes FirstLight's 12.7M-parameter net has only 0.29M in its board network. Distil the M3 specialist into a wider one, then continue PPO.
- Opponent-deck prediction as an auxiliary head.
- Move to FirstLight's `yes-delay` branch, which models the real game's 1-second deploy delay. Its checkpoints don't load on main, so this is a restart.

### Layout

```
clash-ai/                      our public repo
  third_party/FirstLight_CR/   upstream submodule, pinned; patches go in our fork of it
  specialist/                  ours
    audit/      replay download, deck and matchup tables
    driver/     single-GPU PPO launcher, engine pool manager
    train/      imitation term, KL penalty, opponent sampler, promotion gate
    eval/       scorecard, passivity metric
  docs/  tests/
```

Keep replay data, caches, checkpoints and run outputs **outside OneDrive** (for example `C:\ml-data\clash-ai`) so multi-GB files don't sync.

### Upstream pieces to reuse

Paths are relative to `third_party/FirstLight_CR/`.

- `native_runner/battle_env.py`: reset/step environment.
- `native_runner/training/v4/`: `model.py`, `cache_builder.py`, `train_imitation_cache.py`, `train_ppo_self_play_cluster.py`, `evaluate.py`, `serve_policy.py`.
- `checkpoints/General/checkpoint-step-00000460.pt` as the warm start; `checkpoints/IL/` and `checkpoints/2_6hog_expert/` as opponents and references.
- `native_runner/data/competitive/card_support.json` and `form_evidence.json` for card and form ids.

## Risks

- **Throughput is unknown.** FirstLight's General took roughly 156k games' worth of experience on 8 GPUs (460 updates of 1,528 matches × 40 game-seconds). If one PC manages about 96 concurrent matches at real-time speed, that same volume is about 3.5 days. That is an estimate, not a documented figure; M1 replaces it with a measurement. A specialist fine-tune should need far less.
- **The training code targets a Linux cluster.** Adapting it to one Windows box may take real work. WSL2 is the fallback.
- **Evolutions and the Archer Queen ability** may not be fully bound. If so, start with the non-evo deck.
- **The game client is unofficial.** The APK is a modified copy of Supercell's game. This is offline research with no accounts involved, but the write-up should say plainly what it runs on.
- **Small upstream.** FirstLight has one author and 12 commits. Pin the fork to a known commit.

## Verification

- M0: console launches; `python -m native_runner.training.v4.evaluate --checkpoint-a ... --checkpoint-b ... --games N --output DIR` writes `summary.json` with a win rate near the published one.
- M1: throughput script output; audit table spot-checked against the dataset viewer.
- M2: cache round-trip verification passes; held-out accuracy logged.
- M3–M4: 760-game evaluations per opponent with fixed seeds and alternating sides; every claimed gain is an A/B on the same seeds; unit tests for the opponent sampler, promotion gate and loss terms.
