# Model improvements: what to try after generalist-v2

Written 2026-11-25, during the overnight pipeline. Nothing here is built yet. Each idea is a
change to `clashai/model/net.py` or `clashai/train/bc.py`, tested as a probe (below).

## Where the error is today

The generalist's validation loss per decision (120,000 decisions):

| Part | Loss | Share |
|---|---:|---:|
| Timing (play now or wait) | 0.122 | 50% |
| Tile | 0.095 | 38% |
| Card | 0.029 | 12% |

The value head picks the winner 53% of the time in the first minute, 61% in the second, 70% in
the third and 72% in overtime.

## FirstLight_CR compared with ours

Read from FirstLight_CR's source (`native_runner/training/v4/`, main branch, Apache-2.0).
Their results are against their own models, with argmax play.

| | FirstLight_CR | Ours |
|---|---|---|
| Input | Entity rows: cards, towers, unit groups holding child units, 32 recent events, 18 match scalars. Also a 14-channel 32×18 grid of counts and HP per side | Entity tokens (40 fixed plus one per unit); unit outputs scattered onto a 32×18 grid |
| Trunk | 4 pre-norm transformer layers, width 256, 8 heads. Attention bias learned per relation type (14 types: targets, spawned by, and so on) and from pairwise dx, dy, distance and overlap | 6 pre-norm layers, width 384, 6 heads. No attention bias: units see positions only as input numbers |
| Grid CNN | 3 residual blocks, GroupNorm, SiLU, coordinate planes | 4 dilated residual blocks (1, 2, 4, 1), GroupNorm, GELU, FiLM on the chosen card |
| Memory | LSTMCell(512) after the trunk; trained over 32-decision chunks | None: one frame per decision |
| Activation | SiLU | GELU |
| Size | About 12M | 14.4M |
| Decision step | 5 ticks (250 ms) | 5 ticks (250 ms) |
| Actions | Act or wait → card (pointer) → tile (576) → delay 0–200 ms (5 bins) → optional second action | Wait, card slot or ability → tile (576), at 5-tick decisions |
| Value | One scalar; Huber on returns discounted per decision (γ 0.99) | One scalar on the final result (tanh); card-belief heads |
| Auxiliary losses | None | Hand and next-card belief |
| Imitation | 252k replays, both seats; act rows ×8 in the gate loss; delay label smoothed 0.2 to each neighbour; AdamW 3e-5; 8 GPUs | About 107k battles from 156k replays, our seat; waits subsampled to 5%; AdamW 3e-4 cosine, EMA |
| PPO | lr 1e-5, clip 0.1, no entropy bonus, early stop at KL 0.01 to the behaviour policy. **No anchor to the imitation model** | lr 3e-6, KL anchor to the generalist with adaptive weight, separate critic |
| League | Self-play, frozen checkpoints, snapshot history sampled by PFSP | Reference and replay decks, drawn more where it loses |
| Reward | ±1 plus potential shaping on crowns and tower HP | ±1 plus tower-HP potential |
| Specialists | PPO from the general model with the deck fixed. The best one dropped the Hog-specific rewards and demonstrations, and trained longer | One fixed deck from the start of PPO |
| Reported | Hog specialist beats their IL 90% and their general model 88% | – |

Their own caution: one 80% result came almost entirely from an IL opponent that barely played.

## What to try, in order

Each idea is a 20,000-step probe (about 2.7 h) on the full rows, compared at the same step with
`generalist-v2`. Run one change per probe. A winner then gets a 1,300-game scorecard against the
old generalist, game by game. Keep a change only if it is better in the arena, not just on loss.

| # | Change | Targets | Evidence | Effort |
|---|---|---|---|---|
| 1 | **Distance and relation attention bias**: an MLP over pairwise dx, dy, distance and same lane adds a bias per head to each attention score; later, relation types such as targets | Tile and timing, through unit interactions | FirstLight uses both | Medium; `trunk_explicit` (ONNX) changes too |
| 2 | **Cheap history features**: seconds since our last play and since the enemy's, elixir 1 s and 2 s ago, our last 4 plays as tokens | Timing (50% of the loss) | FirstLight's LSTM carries this | Small to medium; new header fields mean relabelling |
| 3 | **Auxiliary targets**: final crown difference, each tower's final HP, tower damage over the next 10–20 s | Value head; a richer trunk for the policy and critic | Not in FirstLight: a chance to get ahead | Small; targets are already in the labels |
| 4 | **Soft timing targets**: spread some weight of a play onto the neighbouring decisions | Calibration of play or wait | FirstLight smooths its delay bins | Small; judge in the arena only, since it changes what the loss measures |
| 5 | **Recurrent core** (LSTM or GRU over decisions) | Timing and memory | FirstLight's main structural difference | Large: rows become sequences, PPO and search carry state. Only if #2 shows memory matters |
| 6 | **Size probe** (512 wide, 8 layers) and SwiGLU | Capacity | – | Small; makes PPO and search slower |

Worth weighing, not copying blindly:

- Act rows weighted in the timing loss (they use ×8).
- A delay head within the 250 ms decision.
- A second action per decision.

Keep: tokens, the tile grid with card conditioning, card-then-tile order, the belief heads, the
KL anchor in PPO (ppo1 drifted, and we have one GPU), and a reward of winning plus tower damage.

## What generalist-v3 takes (decided 2026-11-26)

We looked at seven other public Clash Royale models: FirstLight_CR, RoyaleFish, Daybreak CR, ClashAI, itzik's ClashRoyaleAi, KataCR and cochon123's. Every strong one has memory, so v3 skips the probes above and puts these into one model, trained from scratch on whole battles:

| Part | Why |
|---|---|
| LSTM after the trunk, reading the match token and the previous own action (card, tile, decisions since we last acted) | Timing is half the loss, and one frame can't say what we just did. FirstLight, Daybreak and itzik use an LSTM |
| Distance and relation (targets, targeted by) bias in every attention layer | FirstLight uses both; Daybreak measured a small gain |
| fp32 scores at play time; argmax card and tile as an option | bf16 rounding reorders near-ties; play or wait stays sampled |

Left out, and why:

| Left out | Why |
|---|---|
| A delay head (0–200 ms) | Our replay play times are quantised at the source (tick offsets within a decision are uneven), and it needs an engine change |
| Recent events | crforge doesn't export them, and the observation already shows units appearing, dying and moving |
| Auxiliary heads | No measured gain elsewhere; the LSTM covers their purpose |
| Weighting act rows ×8 | It would change the play rate; every decision has weight 1 instead |

Each part is a setting, off in older checkpoints, so it can be switched off to diagnose a poor result. The plan, budget and go/no-go are in `docs/PLAN.md` (M3b).

## Tooling: faster scorecards

Not a model change, but every idea above is judged by scorecards, so it speeds up all of them.

**Problem.** `clashai/eval/scorecard.py` plays one deck at a time, 100 battles (50 seeds × 2
seats), through `arena.play`. Each decision is a serial round trip: crforge steps the battles on
the CPU, Python and JPype gather the observations, the GPU runs the model, and the moves go back.
With only 100 battles, each side mostly waits for the other. Measured on 2026-11-26 during
`ppo2-tt0.7`:

- The scorecard process used about 1.4 cores of its 16.
- GPU busy 40–42% (`nvidia-smi`), at about 197 W of 450 W.
- One 1,300-game scorecard takes about 9 minutes.

PPO's in-loop `Evaluator` in `clashai/train/ppo.py` does the same, deck by deck. Its 520-game
scorecards took about 8.5 min each, about 15% of ppo2's 3 hours.

**Change (done 2026-11-26).** `arena.play_matches` plays every deck of a scorecard side by side
in one vector, each battle with its own decks (crforge's `reset` already takes a deck per battle,
so the engine needed no change). Each policy acts once per decision on all of its rows. The
scorecard and PPO's in-loop `Evaluator` both use it. A single match still plays exactly as
`arena.play` did.

**Measured.** generalist-v2 at card 0.7, tile 0.3, 1,300 games against the reference decks:

| | Time | Score |
|---|---:|---:|
| Deck by deck (sweep) | 557 s | 74.6% |
| Batched | 240 s | 75.9% |

2.3× faster, less than the 3–4× hoped for. The results agree: +1.3 points (−1.8 to +4.4) game by
game. Games are not identical, since each policy's random draws are shared across all decks.

## Not yet

No model change goes in while a long run is going: each pipeline step starts a fresh Python
process and imports the model code at that moment.
