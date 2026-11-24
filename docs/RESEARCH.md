# Research: open Clash Royale AI projects (November 2026)

What existing projects do, where they fall short, and what this project takes from them. Everything here was read from public repositories, papers and datasets between 2026-11-24 and the start of M0. Numbers are as each project reports them, not independently verified unless stated.

## Engines

| Engine | Licence | What it is | Accuracy | Status |
|---|---|---|---|---|
| [crforge](https://github.com/voonhous/crforge) | Apache-2.0 | Java battle core built from the game's own data tables; Python vector environment via JPype | 20,763 of 20,763 recorded real battles match on every tick (data version 16.635.8); all 1,137 targeted reference battles match | Active, daily commits |
| [RoyaleSim](https://github.com/RoyaleGym/RoyaleSim) | MIT | Rust engine, `pip install royalesim`, Windows wheels | 743 of 744 recorded unit routes; 56.5% of unit positions within tolerance | Active, daily releases |
| [ClashRoyaleAi](https://github.com/itzik123/ClashRoyaleAi) | MIT | C++17 simulator with pybind11, about 10 ms per match | Movement calibrated on 8 recorded matches; known stat errors (spell damage to crown towers, level-9 towers) | Active |
| [FirstLight_CR](https://github.com/Jaasssoooonnnnn/FirstLight_CR) | Apache-2.0 | The game's own engine, hooked inside a Null's Royale 15.535.13 client in an emulator | Exact by construction | **Broken since Null's forced v16**: the 15.535.86 content it needs is gone |

Why crforge: this deck has two evolutions and a champion, which is where the engines differ most. crforge has targeted reference battles for Evo Royal Hogs, Evo Cannon and the Archer Queen ability, all matching. An agent trained on a less exact engine can learn that engine's quirks rather than the game.

Why not FirstLight: the server-side content it pins can no longer be downloaded. Porting to v16 means re-deriving about 150 addresses and struct offsets in a stripped binary (upstream issue #9), and Null's updates every 3–4 months.

## Training projects

| Project | Approach | Reported result | Licence |
|---|---|---|---|
| [KataCR](https://github.com/wty-yy/katacr) ([paper](https://arxiv.org/abs/2504.04783)) | Screen reading with YOLOv8, offline RL (StARformer) on 105 games | 0–10% against the game's built-in bot | MIT |
| [cochon123/clash-royale-ai](https://github.com/cochon123/clash-royale-ai) | Behaviour cloning on scraped replays, no simulator | 56% next-card, 12.8% card+cell accuracy | not stated |
| [ClashRoyaleAi](https://github.com/itzik123/ClashRoyaleAi) | Recurrent PPO, 1-move lookahead search, expert iteration, PFSP league | Search lifts 62.5% to 94.4% against a heuristic bot; "doesn't beat competent human players" | MIT |
| [FirstLight_CR](https://github.com/Jaasssoooonnnnn/FirstLight_CR) | Imitation on 252k replays, PPO league on 8 GPUs, deck specialists | Hog 2.6 specialist beats its generalist 88% | Apache-2.0 |
| [Daybreak CR](https://github.com/Jason-XII/daybreak-cr) | Imitation on 176k replays filtered by exact tower-HP replay, PPO with promotion gates | Small self-reported live samples | **none** |
| [RoyaleFish](https://github.com/fabiospalla9-tech/RoyaleFish) | RoyaleSim plus FirstLight's framework; exact-deck imitation, league with strategy-guide rewards | Specialists 66–68% against its generalist (400 games); 40 h from scratch | Apache-2.0 |
| [vegetableleaf/ClashAI](https://github.com/vegetableleaf/ClashAI) | RoyaleSim, public-only observations, paired evaluation, bridge to crforge | 33–48% on live Ranked | **none** |

## What works

- **Imitation first, then self-play anchored to the human policy.** Every project that got far converged on it. KataCR (offline RL on 105 games) and pure-PPO attempts did not get far.
- **Specialists.** One deck, trained against a field of others, consistently beats a generalist on that deck.
- **Filtering replays the engine can't reproduce** (Daybreak: keep a game only if every tower ends at its recorded HP; 95.6% pass).
- **Fair observations** (ClashAI): the opponent's hand and elixir are never inputs; elixir is estimated from visible plays.
- **Paired evaluation with confidence intervals** (ClashAI), and **promotion gates** (Daybreak: 55% against the champion).
- **Lookahead search and distillation** (itzik).
- **Exact-deck data selection and matchup-weighted opponents** (RoyaleFish).

## Where they fall short

1. **Evaluation.** Most report win rates against opponents in their own training pool, which RoyaleFish's README admits can overstate strength. Live claims are small, self-reported samples.
2. **Training conditions that drift from the game.** RoyaleFish trained with cards landing after 1 tick instead of 20; with the real delay one specialist fell from 68% to 36%. FirstLight documents passive openings; RoyaleFish needed a forced-first-play rule.
3. **Hand-written reward rules.** RoyaleFish pays dozens of capped, deck-specific bonuses that need constant re-tuning, and its known weaknesses are mostly reward-tuning debt.
4. **Narrow imitation data.** Won games only, 3–5k per deck; exact-cell accuracy around 31%; no use of the arena's left-right symmetry.
5. **Little planning at decision time**, despite both engines supporting cheap state copies.
6. **Hidden information mostly ignored.** For a cycle deck, knowing when the opponent's counters are out of cycle is the game.
7. **Reproducibility and ethics.** Private engines, missing licences, a dead dependency, and two projects that played the live ladder against real people.

## What this project does differently

| Shortfall | This project |
|---|---|
| Evaluation on the training pool | Held-out decks and pilots fixed before training; paired seeds; Wilson intervals; per-archetype scorecard |
| Conditions drift | crforge enforces the 20-tick play delay; tests for delay, fairness and determinism |
| Reward rules | Win/loss plus potential-based tower shaping; KL anchor to human play |
| Narrow data | Wins and losses, outcome-weighted; lane mirroring; a generalist pretrained on all replays |
| No hand model | A belief head over the opponent's unseen cards, used by the policy and by search |
| No search | Forked rollouts with opponent hands dealt from the belief; distilled back |
| Live play | Offline only |

## Data

| Dataset | Contents | Use here |
|---|---|---|
| [A1A2CV3/IL_Replay_1115](https://huggingface.co/datasets/A1A2CV3/IL_Replay_1115) | 156,160 top-ladder replays, Nov 4–15 2026, on 16.635; decks, levels, tower troops, every play with tile and 20 Hz tick; Apache-2.0 for the dataset's own contributions | Imitation data (matches crforge's data version) |
| [VanguardX101/IL_Replay](https://huggingface.co/datasets/VanguardX101/IL_Replay) | 252,238 replays, mid-2026, older game version, no licence stated | Not used: wrong data version for crforge |

Both are anonymised scrapes of RoyaleAPI replays. Neither is redistributed by this repository.
