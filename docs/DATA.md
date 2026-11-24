# Data audit

`python -m clashai.data.audit` on [IL_Replay_1115](https://huggingface.co/datasets/A1A2CV3/IL_Replay_1115) (downloaded and checked against its SHA256SUMS by `python -m clashai.data.download`). Run 2026-11-24.

## The dataset

- 156,160 replays from Nov 4–15 2026: 119,539 Ultimate Champion, 35,506 Royal Champion, 365 Grand Champion, 750 Grand Challenge.
- Checked by its authors against game data `16.635.8-062a11e0`: the same content hash as the crforge tables built here.
- Card levels: 99.0% level 16, 0.6% level 15, 0.5% level 11 (Grand Challenge). Training uses level 16.
- Tower troops: 92.1% princess, 4.8% Royal Chef, 2.2% Dagger Duchess, 0.9% Cannoneer.
- Every card and tower troop in the replays maps to a crforge card (`clashai/data/cards.py`, tested).
- Missing, to be reconstructed by the converter (M1): the battle's random seed, the opening hand and cycle, king levels, and card forms at each play.

## Piggies EQ

| Evolutions | Games | Score | 95% interval |
|---|---:|---:|---|
| **Royal Hogs + Cannon** (the deck we train) | **3,395** | 50.9% | 49.2–52.5% |
| Royal Hogs + Skeletons | 47 | 48.9% | 35.3–62.8% |
| Cannon + Skeletons | 10 | 50.0% | 23.7–76.3% |
| others | 15 | | |

3,395 games is above the 1,500-game threshold in the plan, so imitation uses the exact deck. One-card swaps add about 700 more if needed: Electro Spirit for Ice Spirit (349), Bomber for The Log (191), Fire Spirit for Ice Spirit (78).

## Whom it meets

1,368 distinct opponent decks in 3,395 games; the most common appears 108 times. 15.5% of these games are against decks in the held-out split, which training never samples.

Selected human matchups (Piggies EQ's score):

| Opponent | Games | Score |
|---|---:|---:|
| Hog 2.6 | 64 | 82.8% |
| Log Bait (Princess, Dart Goblin) | 28 | 89.3% |
| Mortar Bait | 40 | 82.5% |
| Bandit / Battle Ram / Lumberjack | 108 | 57.4% |
| Sparky / Mini P.E.K.K.A | 62 | 50.0% |
| Lavaloon | 39 | 33.3% |
| Minion Giant, Lightning | 38 | 26.3% |
| Golem, Night Witch | 53 | 20.8% |

The full table is in `D:\ml-data\clash-ai\datasets\audit\piggies_eq.json`.

## Conversion into crforge battles

`python -m clashai.data.convert`, then `python -m clashai.data.label` and `python -m clashai.data.dataset`. Run 2026-11-24 on every replay with Evo Royal Hogs + Evo Cannon Piggies EQ on either side.

### How

The replays lack the battle's random seed and both players' deck orders. crforge deals each side `DeckShuffle.order(deck, draw + accountID.lo)`, where `draw` is the battle random source's third or fourth draw (each side's player data takes one draw first). `clashai/data/shuffle.py` ports that deal to Python; a test checks it against the engine for random seeds and account words.

1. **Deck order.** Each side's sequence of plays rules out most of the 1,680 possible opening hand and queue orders (a played card must be in hand; it goes to the back of the queue). One fitting order is picked; the Mirror and Elixir Collector are never dealt into the opening hand.
2. **Deal.** A shuffle seed giving that order is looked up, and each side's `accountID.lo` is set so crforge deals exactly it. The battle seed is a hash of the replay's tag.
3. **Drive.** The battle starts from a crforge replay document with that setup and is stepped one tick at a time. Each recorded play is given on its recorded tick as the action for the tile holding its point, in the player's frame. A play the engine can't take yet is retried for up to 1 s, then dropped.
4. **Check.** Final crowns, winner and all six towers' hit points are compared with the recording.

Settled by measurement on the first 64 replays:

| Question | Answer | Evidence |
|---|---|---|
| Recorded coordinates | The team player's own screen view: x mirrored for crforge side 0 | Plays on their recorded point 37% → 94%; winners 79% → 89% |
| Which recorded side is crforge side 0 | Always the team (`data_i` is not the side) | Every `data_i` variant reproduced fewer battles |
| Recorded tick | The tick a play was sent | Constant offsets from −10 to +20 ticks are no better than 0 |
| Does the unknown battle seed matter | Barely | 61 of 64 battles end identically under four seeds |
| Do the 1-unit troop offsets matter | No | Rewriting every play to its exact recorded point changes no result |

crforge places a tapped troop 1 unit off the tile centre (the game's symmetrical snap); RoyaleAPI's recorded point is not snapped. Same tile, same placement.

### Results (3,376 replays)

| | |
|---|---:|
| Replays converted | 100% |
| Dealt the planned deck order | 100% |
| Commands sent on their recorded tick / late (≤ 1 s) / dropped | 93.8% / 0.5% / 0.7% |
| Plays landing on their recorded point (within the 1-unit snap) | 93.7% |
| Plays landing on another tile | 0.09% |
| Recorded winner reproduced | 86.8% |
| Recorded crowns reproduced | 83.4% |
| All six towers' final hit points exact | 43.8% |
| Faults, or commands crforge does not model | 0 |

Exactness falls with game length (62% of games under 2 minutes, 41% of overtime games): a small difference early compounds. Its cause is not yet found; the unknown seed, tick alignment, positions, levels and unmodelled mechanics are ruled out above.

### Training set

Kept: crowns and winner reproduced and no command dropped, **2,379 battles** (70%), 16 with Piggies EQ on both sides.

- crforge's labeller turned them into **4,476,350 rows** (2,238,175 decisions per seat): 99.75% of the 206,334 human commands labelled; 275 two-commands-in-one-window collisions and 240 cards that only came into hand within the window are not.
- **5,205 commands (2.5%) are labelled with an action their row's mask closes**: given a tick or two after the decision, when elixir or the hand had changed. Training must allow these labels (or drop those rows) rather than let a masked softmax assign them zero probability.
- Compacted losslessly from 127 GB to 0.77 GB (`clashai/data/dataset.py`; every battle checked to rebuild exactly), in `D:\ml-data\clash-ai\dataset\piggies_eq-screen-team0-v0-t0-standard`.
