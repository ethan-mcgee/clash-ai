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
