# clash-ai

A Clash Royale agent trained to play **one deck** as well as possible, instead of a generalist that plays any deck passably.

**Deck: Piggies EQ.** Evo Royal Hogs, Archer Queen, Evo Cannon, Skeletons, Ice Spirit, The Log, Earthquake, Royal Delivery.

## Approach

- **Foundation:** [FirstLight_CR](https://github.com/Jaasssoooonnnnn/FirstLight_CR) (Apache-2.0), an offline research environment that runs the game's real battle logic headless in an Android emulator. It ships imitation-learned and PPO checkpoints, including a Hog 2.6 specialist that beats its generalist 88% of the time.
- **Data:** the public [IL_Replay](https://huggingface.co/datasets/VanguardX101/IL_Replay) set: 252k human replays with both decks and every play's card, tile and tick.
- **Hardware:** one RTX 3090 Ti, 64 GB RAM. No cluster.

What this project adds:

1. **Single-GPU training driver.** FirstLight's training launchers target a multi-node GPU cluster. This runs the same pipeline on one desktop.
2. **Human-anchored specialist training.** PPO with an imitation term on Piggies EQ human replays and a KL penalty to the starting policy, aimed at the passive openings and opponent exploitation FirstLight reports.
3. **Matchup-weighted opponents.** Opponent decks sampled at the rate Piggies EQ actually meets them in human play, with extra weight on bad matchups.
4. **Per-matchup scorecard.** Win rate by opponent archetype, using FirstLight's evaluation protocol so results compare directly with theirs.

Full plan and milestones: [docs/PLAN.md](docs/PLAN.md).

## Status

| Milestone | State |
|---|---|
| M0. FirstLight running locally | not started |
| M1. Throughput, card support, replay audit, baseline | not started |
| M2. Piggies EQ imitation data | not started |
| M3. Single-GPU PPO specialist | not started |
| M4. Ablations and write-up | not started |

## Scope and ground rules

- **Offline research only.** The agent never plays ranked ladder or any live match against real people; that breaks Supercell's terms of service and is unfair to the other player.
- **No game files in this repo.** FirstLight needs a game client you provide yourself. APKs, game resources, datasets and checkpoints are never committed.
- Not affiliated with or endorsed by Supercell. Clash Royale is a trademark of Supercell.

## Licence

MIT for the code in this repo. FirstLight_CR is Apache-2.0 and keeps its own licence.
