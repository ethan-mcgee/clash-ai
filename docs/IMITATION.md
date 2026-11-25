# M2: imitation results

Run 2026-11-24 and 25 on the RTX 3090 Ti. One network for every run: the entity transformer in `clashai/model/net.py`, 14.4M parameters (d 384, 6 layers, 6 heads). Training is `python -m clashai.train.bc`; the arena scorecard is `python -m clashai.eval.scorecard`.

## Data

| Rows | Battles | Rows | Waits kept |
|---|---:|---:|---:|
| Piggies EQ (`piggies_eq-screen-team0-v0-t0-standard-w0.25`) | 2,379 | 1,275,852 | 25% |
| All decks (`all-screen-team0-v0-t0-standard-w0.1`) | 20,550 | 4,783,292 | 10% |

The all-deck set is the first 30,000 replays converted, of which 20,550 pass the standard filter. Validation is 3% of battles, chosen by a hash of the replay tag, so both seats of a battle, and a battle that appears in both sets, are always on the same side. There is no separate test split: validation also chooses the checkpoint and stops training early, so its figures are slightly optimistic.

## Runs

| Run | Rows | Start | Steps | Time | Best step |
|---|---|---|---:|---:|---:|
| `piggies-scratch` | Piggies EQ | random | 20,000 | 58 min | 4,000 |
| `piggies-scratch-v2` | Piggies EQ | random | 12,000 (stopped early) | 38 min | 6,000 |
| `generalist` | all decks | random | 40,000 | 134 min | 36,000 |
| `specialist` | Piggies EQ, Piggies seat only | `generalist` | 5,000 (stopped early) | 17 min | 2,000 |

The first run had no dropout and a value loss that reached the trunk. It overfit after 4,000 steps: validation loss rose from 0.355 to 1.017 while training loss fell to 0.06, and its value error (1.64) was worse than always guessing a draw (1.0). Every later run has dropout 0.1, early stopping after 3 validations without improvement, and a value head trained on a detached trunk at weight 0.1.

## Accuracy on Piggies EQ validation rows

The Piggies EQ seat's decisions in the 59 validation battles.

| | Scratch v2 | Generalist | Specialist |
|---|---:|---:|---:|
| Policy loss | 0.346 | 0.313 | **0.312** |
| Card (when a card is played) | 64.3% | 69.3% | **70.1%** |
| Tile, exact | 29.6% | **34.7%** | **34.7%** |
| Tile, within one | 54.5% | 60.7% | **61.2%** |
| Enemy hand, top 4 | 77.7% | **85.3%** | 85.0% |
| Value error (1.0 = a constant guess) | 1.00 | **0.86** | 0.92 |

RoyaleFish's 63.5% card and 31.5% tile are a rough yardstick (a different simulator and data). Over all decks, the generalist reaches 68.4% card and 31.3% tile.

## In the arena

Piggies EQ against the 13 reference decks, every opponent piloted by the generalist. 50 seeds per deck, each played from both seats: 1,300 games per row. Sampling at temperature 1.

| Piggies EQ piloted by | Score | 95% interval |
|---|---:|---|
| Generalist | **42.6%** | 39.9–45.3% |
| Specialist | 41.3% | 38.7–44.0% |
| Scratch v2 | 12.0% | 10.3–13.9% |

| Opponent | Specialist | Generalist | Scratch v2 |
|---|---:|---:|---:|
| Piggies EQ (mirror) | 55% | 49% | 19% |
| Hog 2.6 | 83% | 76% | 37% |
| Log Bait | 67% | 58% | 16% |
| X-Bow | 30% | 38% | 13% |
| Mortar Bait | 51% | 59% | 11% |
| Golem | 15% | 25% | 6% |
| Lavaloon (held out) | 28% | 33% | 7% |
| Giant Double Prince | 27% | 20% | 3% |
| Royal Giant | 59% | 64.5% | 21% |
| P.E.K.K.A Bridge Spam (held out) | 9% | 11% | 3% |
| Mega Knight | 16% | 19% | 1% |
| Miner Control (held out) | 57% | 55% | 11% |
| Graveyard Freeze | 40% | 46% | 8% |

About 7,500 to 8,200 network-against-network games an hour, 100 battles at a time, 0 faults.

## Conclusions

1. **Breadth of data decides strength.** The generalist, trained on every deck, scores 3.5 times the network trained on Piggies EQ alone, and wins the mirror match 81 to 19 against it.
2. **Small accuracy gaps are large in games.** About five points of card accuracy separate 42.6% from 12.0%: a mistake repeats over hundreds of decisions, and one missed defence costs a tower.
3. **Fine-tuning on Piggies EQ adds nothing.** Under a point of accuracy, and no difference in games. 2,379 battles are too few to add to what 20,550 battles of general play teach. **M2's exit criterion, the specialist beating the generalist piloting Piggies EQ, is not met.**
4. **The matchups agree with human play.** Piggies EQ beats Hog 2.6 and Log Bait and loses to Golem and heavy beatdown, as in the replays (`docs/DATA.md`: 82.8% against Hog 2.6, 20.8% against Golem Night Witch).
5. **The score below 50% is the deck, not a fault.** Both sides play at one network's strength; the mirror match is close to even.

M3 starts from the generalist: it is as strong as the specialist, less overfit, and has the better value head. Specialisation is left to PPO, on Piggies EQ's own games, weighted toward its losing matchups.
