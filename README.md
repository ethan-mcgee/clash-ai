# clash-ai

A Clash Royale agent trained to play **one deck** as well as possible, instead of a generalist that plays any deck passably.

**Deck: Piggies EQ.** Evo Royal Hogs, Archer Queen, Evo Cannon, Skeletons, Ice Spirit, The Log, Earthquake, Royal Delivery.

## Approach

- **Engine:** [crforge](https://github.com/voonhous/crforge) (Apache-2.0), a battle simulator built from the game's own data tables. It matches 20,763 of 20,763 recorded real battles on every tick, including the evolutions and champion this deck relies on.
- **Data:** [IL_Replay_1115](https://huggingface.co/datasets/A1A2CV3/IL_Replay_1115), 156k top-ladder replays from November 2026 on the same game version, converted into crforge battles by this repo.
- **Training:** imitation of human players first (a generalist, then the deck specialist), then a self-play league anchored to the human policy, then lookahead search distilled back into the network.
- **Hardware:** one RTX 3090 Ti, 64 GB RAM.

What this project adds over existing work ([docs/RESEARCH.md](docs/RESEARCH.md)):

1. **Evaluation first.** Held-out opponent decks and pilots fixed before training, paired seeds and confidence intervals, a per-matchup scorecard.
2. **No hand-written strategy rewards.** Win/loss plus tower-damage shaping, with behaviour anchored to human play.
3. **More from the replays.** Lost games as well as won ones, lane mirroring, and a converter from RoyaleAPI replays to crforge.
4. **Opponent modelling.** A belief over the opponent's unseen cards, used by the policy and by search.

Full plan and milestones: [docs/PLAN.md](docs/PLAN.md). Machine setup: [docs/SETUP.md](docs/SETUP.md).

## Status

| Milestone | State |
|---|---|
| M0. Foundations: environment, tests, evaluation harness, throughput | in progress |
| M1. Data: audit, replay conversion, consistency filter | not started |
| M2. Imitation: generalist, then specialist | not started |
| M3. PPO league | not started |
| M4. Search | not started |
| M5. Ablations, demo and write-up | not started |

## Scope and ground rules

- **Offline research only.** The agent never plays ranked, ladder or any live match against real people. That breaks Supercell's terms of service and is unfair to the other player.
- **No game files in this repo.** crforge builds its data tables on your machine from the game's own files; replays are downloaded separately. Neither is ever committed.
- Not affiliated with or endorsed by Supercell. Clash Royale is a trademark of Supercell.

## Licence

MIT for the code in this repo. crforge is Apache-2.0 and keeps its own licence; see [NOTICE](NOTICE).
