"""PPO for the Piggies EQ seat, against a league of opponent decks, anchored to imitation.

    python -m clashai.train.ppo --name ppo2 --hours 3
    python -m clashai.train.ppo --name ppo2 --resume --hours 8      # carries on a stopped run

The learner starts from an imitation checkpoint (``--init``, the generalist) and plays Piggies EQ
from a random seat in ``--envs`` battles at once. Every game draws its opponent:

- a deck from :class:`clashai.train.league.League` (the decks Piggies EQ meets in the replays,
  never a held-out one, weighted toward those it loses to), piloted by the reference network
  (``--reference``, the generalist), frozen;
- or, with probability ``--mirror-share``, Piggies EQ piloted by one of the learner's own recent
  snapshots, so it also plays against what it is becoming.

Each decision's reward is the game's result at its end (+1, -1, 0 for a draw) plus potential-based
shaping, ``gamma * phi(next) - phi(now)`` with ``phi = --shaping * (own tower share - enemy tower
share)`` and ``phi = 0`` at the end, which changes no optimal policy.

The critic is a network of its own (``--critic-init``), so its errors never reach the policy's
features; it values every decision of a rollout in one pass after it. The first
``--critic-warmup`` iterations train the critic alone, the policy frozen, so it learns the value of
the learner's own games before it steers anything. The policy's loss is PPO's clipped surrogate, a
small entropy bonus, and a weight times the KL divergence to the reference network on the
learner's own decisions, which keeps its play near human play; the weight is adjusted each
iteration to hold that divergence near ``--kl-target``.

Every ``--eval-every`` iterations after the warm-up, the snapshot plays a short scorecard
(``--eval-seeds`` seeds against each reference deck, piloted by the reference), and the best is
copied to ``best.pt``: the training
score drifts with the league's draws, the scorecard does not.

A run writes ``<runs>/ppo/<name>/``: ``last.pt`` (with the optimiser and the league, to resume),
``pool/iter_NNNNN.pt`` every ``--snapshot-every`` iterations (playable by the scorecard),
``best.pt``, ``metrics.jsonl`` (one line an iteration) and ``evals.jsonl`` (one per scorecard).
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .. import paths
from ..decks import PIGGIES_EQ
from ..model.policy import load_net
from ..model.spec import unpack_masks
from ..sim.env import make_vector
from .league import League, opponent_decks

NEG = -1e9
DEFAULT_INIT = "bc/generalist/best.pt"
EVAL_SEEDS = (1_000_000, 1_999_999)
"""Scorecards play seeds from 1,000,000 up; training battles start above this range."""
TOWERS = ("king", "left", "right")


# -- the network on observations --------------------------------------------------------------


def seat_inputs(obs: np.ndarray, spec, max_entities: int, device):
    """Header, entities and their validity for ``(B, seat_ints)`` observations, on ``device``;
    entities are cut to the fullest row, rounded up to 16."""
    hi, ei = spec.header_ints, spec.entity_ints
    counts = np.minimum(obs[:, spec.header["entity_count"][0]], max_entities)
    width = min(max(16, -(-int(counts.max(initial=1)) // 16) * 16), max_entities)
    ents = obs[:, hi:hi + width * ei].reshape(len(obs), width, ei)
    valid = np.arange(width)[None, :] < counts[:, None]
    return (torch.from_numpy(np.ascontiguousarray(obs[:, :hi])).to(device),
            torch.from_numpy(np.ascontiguousarray(ents)).to(device),
            torch.from_numpy(valid).to(device))


def _type_logits(net, out, mask):
    type_mask = net.spec.type_mask(mask)
    type_mask[:, 0] |= ~type_mask.any(1)  # a closed row (a finished battle) may wait
    return out["type_logits"].float().masked_fill(~type_mask, NEG)


def _tile_logits(net, out, mask, rows, slot):
    logits = net.tile_logits(out, rows, slot).float()
    return logits.masked_fill(~net.spec.tile_mask(mask[rows], slot), NEG)


@torch.no_grad()
def sample(net, out, mask, generator=None):
    """Actions sampled from the network's output under ``mask``, with their log-probabilities."""
    spec = net.spec
    type_logits = _type_logits(net, out, mask)
    kind = torch.multinomial(torch.softmax(type_logits, -1), 1, generator=generator).squeeze(1)
    logp = torch.log_softmax(type_logits, -1).gather(1, kind[:, None]).squeeze(1)
    tile = torch.zeros_like(kind)
    rows = torch.nonzero((kind >= 1) & (kind <= spec.hand_slots)).squeeze(1)
    if rows.numel():
        tl = _tile_logits(net, out, mask, rows, kind[rows] - 1)
        t = torch.multinomial(torch.softmax(tl, -1), 1, generator=generator).squeeze(1)
        tile[rows] = t
        logp[rows] += torch.log_softmax(tl, -1).gather(1, t[:, None]).squeeze(1)
    return spec.encode(kind, tile), logp


def evaluate_actions(net, out, mask, actions):
    """Log-probabilities of ``actions`` and the policy's entropy (the type's, plus the tile's
    where a card is played)."""
    spec = net.spec
    kind, tile = spec.decode(actions)
    type_logp = torch.log_softmax(_type_logits(net, out, mask), -1)
    logp = type_logp.gather(1, kind[:, None]).squeeze(1)
    entropy = -(type_logp.exp() * type_logp).sum(1)
    rows = torch.nonzero((kind >= 1) & (kind <= spec.hand_slots)).squeeze(1)
    if rows.numel():
        tile_logp = torch.log_softmax(_tile_logits(net, out, mask, rows, kind[rows] - 1), -1)
        logp = logp.index_add(0, rows, tile_logp.gather(1, tile[rows, None]).squeeze(1))
        entropy = entropy.index_add(0, rows, -(tile_logp.exp() * tile_logp).sum(1))
    return logp, entropy


def critic(net, out, trunk_grad: float) -> torch.Tensor:
    """The value head's estimate, -1 to 1, the same number as ``out["value"]``. Its gradient
    reaches the shared trunk scaled by ``trunk_grad``, so the critic, wrong at first, cannot pull
    the policy's features far from imitation's."""
    match = out["match"]
    match = match.detach() + trunk_grad * (match - match.detach())
    return torch.tanh(net.value_head(match).squeeze(-1).float())


def _rows(out: dict, rows: slice) -> dict:
    """A network output cut to some of its rows."""
    return {key: val[rows] if torch.is_tensor(val) else val for key, val in out.items()}


# -- returns --------------------------------------------------------------------------------


def tower_potential(header: np.ndarray, spec, scale: float) -> np.ndarray:
    """``scale`` times the seat's share of its towers' hit points left less the enemy's."""
    def share(side):
        hp = sum(header[:, spec.header[f"{side}_{t}_hp"][0]].astype(np.float64) for t in TOWERS)
        top = sum(header[:, spec.header[f"{side}_{t}_max_hp"][0]].astype(np.float64)
                  for t in TOWERS)
        return np.where(top > 0, hp / np.maximum(top, 1), 0.0)

    return (scale * (share("own") - share("enemy"))).astype(np.float32)


def gae(rewards, values, dones, live, last_values, gamma: float, lam: float):
    """Generalised advantage estimates and returns over ``(T, N)`` arrays. A row not ``live``
    (the step that restarts a finished battle) has no advantage and cuts nothing: the row before
    it always ends its episode."""
    steps = len(rewards)
    adv = np.zeros_like(rewards)
    last = np.zeros(rewards.shape[1], np.float32)
    next_value = last_values.astype(np.float32)
    for t in range(steps - 1, -1, -1):
        go_on = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * go_on - values[t]
        last = np.where(live[t], delta + gamma * lam * go_on * last, 0.0)
        adv[t] = last
        next_value = np.where(live[t], values[t], next_value)
    return adv, adv + values


# -- the run ----------------------------------------------------------------------------------


class Pilots:
    """Frozen networks by checkpoint path, the most recent ``keep`` kept on the GPU."""

    def __init__(self, device, keep: int):
        self.device, self.keep, self.nets = device, keep, {}

    def get(self, path: str):
        if path not in self.nets:
            if len(self.nets) >= self.keep:
                self.nets.pop(next(iter(self.nets)))
            net, _ = load_net(path, self.device)
            self.nets[path] = net.eval()
        return self.nets[path]


def adapt_kl(coef: float, kl: float, target: float, low: float = 1e-3, high: float = 10.0) -> float:
    """The KL weight for the next iteration: raised by half when the drift ``kl`` is past 1.5
    times ``target``, lowered by a third when it is under half of it. A target of 0 keeps it."""
    if target <= 0:
        return coef
    if kl > 1.5 * target:
        coef *= 1.5
    elif kl < 0.5 * target:
        coef /= 1.5
    return float(min(max(coef, low), high))


def game_outcomes(dones: np.ndarray, results: np.ndarray) -> np.ndarray:
    """Each row's game result, read back from the step its game ended (``results`` holds the
    result on ``dones`` rows); NaN for a game still going at the end of the rollout."""
    out = np.full(dones.shape, np.nan, np.float32)
    carry = np.full(dones.shape[1], np.nan, np.float32)
    for t in range(len(dones) - 1, -1, -1):
        carry = np.where(dones[t] > 0, results[t], carry)
        out[t] = carry
    return out


class Evaluator:
    """A short scorecard: Piggies EQ, piloted by a checkpoint, against every reference deck
    piloted by the reference network, on fixed seeds, 2 games each (both seats)."""

    def __init__(self, reference_path: str, seeds: int, threads: int | None):
        from ..model.policy import NetPolicy

        self.opponent = NetPolicy(reference_path, name="reference")
        self.seeds = list(range(EVAL_SEEDS[0], EVAL_SEEDS[0] + seeds))
        self.threads = threads

    def __call__(self, checkpoint: str) -> dict:
        from ..decks import REFERENCE_DECKS
        from ..eval.arena import Side, play
        from ..model.policy import NetPolicy

        policy = NetPolicy(checkpoint, name="learner")
        by_deck = {}
        for name, d in REFERENCE_DECKS.items():
            games = play(Side(policy, PIGGIES_EQ), Side(self.opponent, d), self.seeds,
                         threads=self.threads)
            by_deck[name] = float(np.mean([g.score for g in games]))
        del policy
        torch.cuda.empty_cache()
        return {"score": float(np.mean(list(by_deck.values()))), "by_deck": by_deck,
                "games": len(by_deck) * 2 * len(self.seeds)}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="PPO for Piggies EQ against a league")
    p.add_argument("--name", required=True)
    p.add_argument("--init", help=f"checkpoint to start from (default <runs>/{DEFAULT_INIT})")
    p.add_argument("--critic-init",
                   help="checkpoint the critic starts from (default: --init's), such as one whose "
                        "value head was fitted with bc --value-only")
    p.add_argument("--reference", help="the frozen pilot and KL anchor (default: --init's)")
    p.add_argument("--resume", action="store_true", help="carry on from the run's last.pt")
    p.add_argument("--iters", type=int, default=100000)
    p.add_argument("--hours", type=float, default=0, help="stop after this long (0: never)")
    p.add_argument("--envs", type=int, default=1024)
    p.add_argument("--steps", type=int, default=128, help="decisions per battle an iteration")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--minibatch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-6)
    p.add_argument("--critic-lr", type=float, default=3e-5)
    p.add_argument("--gamma", type=float, default=0.998)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.001)
    p.add_argument("--kl-coef", type=float, default=0.1, help="the KL weight's starting value")
    p.add_argument("--kl-target", type=float, default=0.01,
                   help="the drift from the reference the KL weight is adjusted to hold (0: fixed)")
    p.add_argument("--kl-min", type=float, default=0.02,
                   help="the KL weight's floor: the learner starts equal to the reference, so its "
                        "drift reads 0 at first, and an unbounded weight would fall far meanwhile")
    p.add_argument("--shaping", type=float, default=0.5)
    p.add_argument("--critic-warmup", type=int, default=30,
                   help="iterations at the start that train the critic alone, the policy frozen")
    p.add_argument("--mirror-share", type=float, default=0.1)
    p.add_argument("--hard-share", type=float, default=0.5,
                   help="share of games drawn toward lost matchups (the rest by replay frequency)")
    p.add_argument("--snapshot-every", type=int, default=20)
    p.add_argument("--eval-seeds", type=int, default=20,
                   help="seeds per reference deck for the in-run scorecard (0: none)")
    p.add_argument("--eval-every", type=int, default=40,
                   help="iterations between in-run scorecards, a multiple of --snapshot-every "
                        "(each takes about 8-10 minutes at 20 seeds)")
    p.add_argument("--pool", type=int, default=5, help="recent snapshots the mirror games draw on")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.eval_seeds and args.eval_every % args.snapshot_every:
        raise SystemExit("--eval-every must be a multiple of --snapshot-every: it scores snapshots")

    out_dir = paths.runs_dir() / "ppo" / args.name
    (out_dir / "pool").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(0.8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    resume = torch.load(out_dir / "last.pt", map_location=device, weights_only=False) \
        if args.resume else None
    if resume and "critic" not in resume:
        raise SystemExit("this run was made before the separate critic; start a new --name")
    init = Path(args.init) if args.init else paths.runs_dir() / DEFAULT_INIT
    reference_path = str(args.reference or (resume["reference"] if resume else init))
    net, init_ckpt = load_net(init, device)
    metadata = init_ckpt.get("metadata")
    net.set_dropout(0.0)  # rollout and update must see the same policy
    critic_net, _ = load_net(Path(args.critic_init) if args.critic_init else init, device)
    critic_net.set_dropout(0.0)
    if resume:
        net.load_state_dict(resume["model"])
        critic_net.load_state_dict(resume["critic"])
    net.train()
    critic_net.train()
    spec, k = net.spec, net.config.max_entities
    pilots = Pilots(device, args.pool + 1)
    reference = pilots.get(reference_path)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-5,
                            weight_decay=0.0)
    critic_opt = torch.optim.AdamW(critic_net.parameters(), lr=args.critic_lr,
                                   betas=(0.9, 0.999), eps=1e-5, weight_decay=0.0)
    decks, held = opponent_decks()
    league = League(decks, uniform=1 - args.hard_share)
    start_iter = 0
    kl_coef = args.kl_coef
    best_eval = -1.0
    if resume:
        opt.load_state_dict(resume["optimizer"])
        critic_opt.load_state_dict(resume["critic_optimizer"])
        league.load_state_dict(resume["league"])
        start_iter = resume["iteration"] + 1
        kl_coef = resume["kl_coef"]
        best_eval = resume.get("best_eval", -1.0)
    pool = sorted(str(f) for f in (out_dir / "pool").glob("iter_*.pt"))
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=1))
    print(f"{len(decks)} opponent decks ({sum(d.met for d in decks):,} games in the replays; "
          f"{held} against held-out decks left out); reference {reference_path}", flush=True)

    n, steps = args.envs, args.steps
    value_scale = 1.0 + args.shaping  # returns reach past +-1 when the towers and the result differ
    v = make_vector(n, args.threads)
    s = v.schema
    v.set_autoreset_seeds(EVAL_SEEDS[1] + 1, 2**31 - 1)
    slot_evo = s.slot_evolution
    piggies_slots = PIGGIES_EQ.slots(slot_evo)
    evaluator = Evaluator(reference_path, args.eval_seeds, args.threads) if args.eval_seeds else None

    # Each battle's current game: the learner's seat, its opponent and how it was drawn.
    seat = rng.integers(0, 2, n)
    deck = np.full(n, -1)          # league deck index, -1 for a mirror game
    pilot = [reference_path] * n
    weight = np.ones(n)            # replay frequency / draw probability, for the weighted score

    def draw(i):
        """The next game of battle ``i``: its seat, opponent deck and pilot."""
        if rng.random() < args.mirror_share:
            snapshots = pool[-args.pool:] or [str(init)]
            return (int(rng.integers(0, 2)), -1, snapshots[rng.integers(len(snapshots))], 0.0)
        probs = league_probs[0]
        d = int(rng.choice(len(decks), p=probs))
        freq = decks[d].met / league.met.sum()
        return int(rng.integers(0, 2)), d, reference_path, freq / probs[d]

    def decks_for(i, seat_i, d):
        opp = (list(PIGGIES_EQ.cards), piggies_slots) if d < 0 else \
            (list(decks[d].cards), list(decks[d].slots))
        own = (list(PIGGIES_EQ.cards), piggies_slots)
        return (own, opp) if seat_i == 0 else (opp, own)

    league_probs = [league.probabilities()]  # refreshed each iteration
    games = [draw(i) for i in range(n)]
    seat = np.array([g[0] for g in games])
    deck = np.array([g[1] for g in games])
    pilot = [g[2] for g in games]
    weight = np.array([g[3] for g in games])
    sides = [decks_for(i, seat[i], deck[i]) for i in range(n)]
    v.reset(rng.integers(EVAL_SEEDS[1] + 1, 2**31 - 1, n).tolist(),
            [sd[0][0] for sd in sides], [sd[1][0] for sd in sides],
            [sd[0][1] for sd in sides], [sd[1][1] for sd in sides])
    pending: dict[int, tuple] = {}  # battle -> its next game, taking effect after the restart step
    restarting = np.zeros(n, bool)

    hi, ei = spec.header_ints, spec.entity_ints
    width_ints = k * ei
    buf = {
        "header": np.zeros((steps, n, hi), np.int32),
        "entities": np.zeros((steps, n, width_ints), np.int32),
        "masks": np.zeros((steps, n, (spec.actions + 7) // 8), np.uint8),
        "actions": np.zeros((steps, n), np.int64),
        "logp": np.zeros((steps, n), np.float32),
        "ref_logp": np.zeros((steps, n), np.float32),
        "values": np.zeros((steps, n), np.float32),
        "rewards": np.zeros((steps, n), np.float32),
        "results": np.zeros((steps, n), np.float32),
        "phi": np.zeros((steps, n), np.float32),
        "dones": np.zeros((steps, n), np.float32),
        "live": np.zeros((steps, n), bool),
    }
    log = open(out_dir / "metrics.jsonl", "a", encoding="utf-8")
    evals = open(out_dir / "evals.jsonl", "a", encoding="utf-8")
    t_start = time.time()
    arange = np.arange(n)

    def forward(model, obs):
        h, e, valid = seat_inputs(obs, spec, k, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(h, e, valid)
        return out

    @torch.no_grad()
    def values_of(obs: np.ndarray) -> np.ndarray:
        """The critic's values of ``(B, seat_ints)`` observations, in batches."""
        out = np.zeros(len(obs), np.float32)
        for a in range(0, len(obs), 4096):
            o = forward(critic_net, obs[a:a + 4096])
            out[a:a + 4096] = (o["value"].float() * value_scale).cpu().numpy()
        return out

    def run_eval(label: str, checkpoint: str, it: int) -> float:
        t = time.time()
        r = evaluator(checkpoint)
        evals.write(json.dumps({"iteration": it, "checkpoint": label, **r}) + "\n")
        evals.flush()
        worst = sorted(r["by_deck"].items(), key=lambda kv: kv[1])[:3]
        print(f"  scorecard {label}: {r['score']:.3f} over {r['games']} games "
              f"(weakest: {', '.join(f'{d} {x:.0%}' for d, x in worst)}) [{time.time() - t:.0f} s]",
              flush=True)
        return r["score"]

    try:
        if evaluator and start_iter == 0:
            run_eval("start", str(init), -1)
        for it in range(start_iter, args.iters):
            t0 = time.time()
            warm = it < args.critic_warmup
            league_probs[0] = league.probabilities()
            net.eval()
            finished = []  # (score, deck, weight, ticks) of the league games, mirror games apart
            mirror_scores = []
            plays = 0
            for t in range(steps):
                obs_all = v.observations
                masks_all = v.masks
                own = obs_all[arange, seat]
                own_mask = masks_all[arange, seat].astype(bool)
                opp = obs_all[arange, 1 - seat]
                opp_mask = masks_all[arange, 1 - seat].astype(bool)
                with torch.no_grad():
                    out = forward(net, own)
                    m = torch.from_numpy(own_mask).to(device)
                    actions, logp = sample(net, out, m, gen)
                    # The reference pilots its opponents and scores the learner's actions in one
                    # forward pass.
                    ref_rows = np.flatnonzero([pl == reference_path for pl in pilot])
                    both = np.concatenate([opp[ref_rows], own])
                    both_mask = torch.from_numpy(np.concatenate([opp_mask[ref_rows], own_mask])
                                                 ).to(device)
                    rout = forward(reference, both)
                    r = len(ref_rows)
                    opp_actions = np.zeros(n, np.int64)
                    if r:
                        a, _ = sample(reference, _rows(rout, slice(0, r)), both_mask[:r], gen)
                        opp_actions[ref_rows] = a.cpu().numpy()
                    ref_logp, _ = evaluate_actions(reference, _rows(rout, slice(r, None)),
                                                   both_mask[r:], actions)
                    for path in {pl for pl in pilot if pl != reference_path}:
                        rows = np.flatnonzero([pl == path for pl in pilot])
                        snap = pilots.get(path)
                        sout = forward(snap, opp[rows])
                        a, _ = sample(snap, sout, torch.from_numpy(opp_mask[rows]).to(device),
                                      gen)
                        opp_actions[rows] = a.cpu().numpy()
                acts = actions.cpu().numpy()
                buf["header"][t] = own[:, :hi]
                buf["entities"][t] = own[:, hi:hi + width_ints]
                buf["masks"][t] = np.packbits(own_mask, axis=1)
                buf["actions"][t] = acts
                buf["logp"][t] = logp.cpu().numpy()
                buf["ref_logp"][t] = ref_logp.float().cpu().numpy()
                buf["live"][t] = ~restarting
                phi_now = tower_potential(own[:, :hi], spec, args.shaping)
                buf["phi"][t] = phi_now
                both_actions = np.zeros((n, 2), np.int32)
                both_actions[arange, seat] = acts
                both_actions[arange, 1 - seat] = opp_actions
                plays += int(((acts > 0) & ~restarting).sum())
                v.actions[:] = both_actions
                v.step()

                dones = v.dones.astype(bool) & ~restarting
                reward = v.rewards[arange, seat].astype(np.float32)
                phi_next = np.where(dones, 0.0, tower_potential(
                    v.observations[arange, seat][:, :hi], spec, args.shaping))
                faulted = v.infos[:, s.info["fault"]] != 0
                shaped = reward + args.gamma * phi_next - phi_now
                buf["rewards"][t] = np.where(restarting | (dones & faulted), 0.0, shaped)
                buf["results"][t] = np.where(dones & ~faulted, reward, 0.0)
                buf["dones"][t] = dones
                # Battles restarted by this step start the game drawn for them.
                for i in np.flatnonzero(restarting):
                    seat[i], deck[i], pilot[i], weight[i] = pending.pop(i)
                restarting[:] = False
                for i in np.flatnonzero(dones):
                    winner = int(v.infos[i, s.info["winner"]])
                    if not faulted[i]:
                        score = 1.0 if winner == seat[i] else 0.0 if winner == 1 - seat[i] \
                            else 0.5 if winner == s.winner_draw else None
                        if score is not None:
                            if deck[i] >= 0:
                                league.record(int(deck[i]), score)
                                finished.append((score, int(deck[i]), weight[i],
                                                 int(v.infos[i, s.info["tick"]])))
                            else:
                                mirror_scores.append(score)
                    nxt = draw(i)
                    (d0, s0), (d1, s1) = decks_for(i, nxt[0], nxt[1])
                    v.set_decks(i, d0, d1, s0, s1)
                    pending[i] = nxt
                    restarting[i] = True
            rollout_s = time.time() - t0

            # The critic values every row in one pass, and the battles as the rollout leaves them.
            flat = {key: val.reshape(steps * n, *val.shape[2:]) for key, val in buf.items()}
            obs_flat = np.concatenate([flat["header"], flat["entities"]], 1)
            buf["values"][:] = values_of(obs_flat).reshape(steps, n)
            last_values = values_of(v.observations[arange, seat])
            adv, ret = gae(buf["rewards"], buf["values"], buf["dones"], buf["live"],
                           last_values, args.gamma, args.lam)

            # How well the critic foresees the result: its value less the shaping still to come
            # (the potential, which the returns pay back) is its estimate of the result.
            outcome = game_outcomes(buf["dones"], buf["results"])
            known = buf["live"] & ~np.isnan(outcome)
            win_pred = (buf["values"] + buf["phi"])[known]
            result_known = outcome[known]
            decided = result_known != 0
            win_sign = float(((win_pred > 0) == (result_known > 0))[decided].mean()) \
                if decided.any() else None
            win_mse = float(((win_pred - result_known) ** 2).mean()) if known.any() else None

            # -- update --
            t1 = time.time()
            net.train()
            idx_all = np.flatnonzero(buf["live"].ravel())
            adv_f, ret_f = adv.ravel(), ret.ravel()
            stats: dict = {}
            for _ in range(args.epochs):
                order = rng.permutation(idx_all)
                for start in range(0, len(order), args.minibatch):
                    idx = np.sort(order[start:start + args.minibatch])
                    if len(idx) < args.minibatch // 4:
                        continue
                    h, e, valid = seat_inputs(obs_flat[idx], spec, k, device)
                    target = torch.from_numpy(ret_f[idx]).to(device)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        value = critic(critic_net, critic_net(h, e, valid), 1.0) * value_scale
                    v_loss = F.mse_loss(value, target)
                    critic_opt.zero_grad(set_to_none=True)
                    v_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic_net.parameters(), 0.5)
                    critic_opt.step()
                    batch_stats = {"value_loss": v_loss}
                    if not warm:
                        mask = unpack_masks(torch.from_numpy(flat["masks"][idx]).to(device),
                                            spec.actions)
                        acts_t = torch.from_numpy(flat["actions"][idx]).to(device)
                        old = torch.from_numpy(flat["logp"][idx]).to(device)
                        ref = torch.from_numpy(flat["ref_logp"][idx]).to(device)
                        a = torch.from_numpy(adv_f[idx]).to(device)
                        a = (a - a.mean()) / (a.std() + 1e-8)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            logp, entropy = evaluate_actions(net, net(h, e, valid), mask, acts_t)
                        ratio = torch.exp(logp - old)
                        pg = -torch.min(ratio * a,
                                        ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                        d = ref - logp
                        kl_ref = (torch.exp(d) - d - 1).mean()
                        ent = entropy.mean()
                        loss = pg - args.ent_coef * ent + kl_coef * kl_ref
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                        opt.step()
                        batch_stats.update({
                            "policy_loss": pg, "entropy": ent, "kl_ref": kl_ref,
                            "approx_kl": (old - logp).mean(),
                            "clip_frac": ((ratio - 1).abs() > args.clip).float().mean()})
                    with torch.no_grad():
                        for key, val in batch_stats.items():
                            stats[key] = stats.get(key, 0.0) + float(val)
                        stats["_n"] = stats.get("_n", 0) + 1
            update_s = time.time() - t1
            nb = max(1, stats.pop("_n", 1))
            stats = {key: val / nb for key, val in stats.items()}
            used_kl_coef = kl_coef
            if not warm:
                kl_coef = adapt_kl(kl_coef, stats["kl_ref"], args.kl_target, low=args.kl_min)
            live = buf["live"]
            var = float(np.var(ret[live]))
            ev = 1 - float(np.var(ret[live] - buf["values"][live])) / var if var > 0 else 0.0

            scores = np.array([f[0] for f in finished])
            w = np.array([f[2] for f in finished])
            row = {
                "iteration": it, "minutes": round((time.time() - t_start) / 60, 1),
                "decisions": int(live.sum()), "games": len(finished),
                "score": float(scores.mean()) if len(scores) else None,
                "score_by_frequency": float((scores * w).sum() / w.sum()) if len(scores) else None,
                "mirror_games": len(mirror_scores),
                "mirror_score": float(np.mean(mirror_scores)) if mirror_scores else None,
                "game_seconds": float(np.mean([f[3] for f in finished]) / 20) if finished else None,
                "plays_per_minute": plays / max(1, int(live.sum())) * 240,
                "explained_variance": ev, "result_sign_acc": win_sign, "result_mse": win_mse,
                "kl_coef": used_kl_coef, "warmup": warm,
                "rollout_s": round(rollout_s, 1), "update_s": round(update_s, 1), **stats,
            }
            log.write(json.dumps(row) + "\n")
            log.flush()
            fmt = lambda x, f="{:.3f}": "-" if x is None else f.format(x)
            print(f"iter {it}: {len(finished)} games, score {fmt(row['score'])} "
                  f"(by frequency {fmt(row['score_by_frequency'])}), mirror "
                  f"{fmt(row['mirror_score'])} of {len(mirror_scores)}; critic: value loss "
                  f"{stats.get('value_loss', 0):.4f}, ev {ev:.2f}, result sign "
                  f"{fmt(win_sign, '{:.0%}')}, result mse {fmt(win_mse)}"
                  + (" [critic warm-up]" if warm else
                     f"; policy: kl_ref {stats['kl_ref']:.4f} (weight {used_kl_coef:.3g}), "
                     f"approx_kl {stats['approx_kl']:.4f}, clip {stats['clip_frac']:.3f}, "
                     f"entropy {stats['entropy']:.3f}")
                  + f" [{rollout_s:.0f}+{update_s:.0f} s, {row['minutes']} min]", flush=True)

            state = {"model": net.state_dict(), "description": net.describe(), "step": it,
                     "metadata": metadata, "optimizer": opt.state_dict(),
                     "critic": critic_net.state_dict(), "critic_optimizer": critic_opt.state_dict(),
                     "kl_coef": kl_coef, "best_eval": best_eval,
                     "league": league.state_dict(), "iteration": it, "reference": reference_path,
                     "args": vars(args)}
            torch.save(state, out_dir / "last.pt")
            if (it + 1) % args.snapshot_every == 0:
                snap = out_dir / "pool" / f"iter_{it + 1:05d}.pt"
                torch.save({k2: state[k2] for k2 in ("model", "description", "step", "metadata")},
                           snap)
                pool.append(str(snap))
                hard = ", ".join(f"{'/'.join(d.cards[:2])} {rate:.0%}"
                                 for d, rate, _ in league.hardest(5))
                print(f"  snapshot {snap.name}; hardest matchups drawn more: {hard}", flush=True)
                if evaluator and not warm and (it + 1) % args.eval_every == 0:
                    score = run_eval(snap.name, str(snap), it)
                    if score > best_eval:
                        best_eval = score
                        shutil.copyfile(snap, out_dir / "best.pt")
                        print(f"  best so far: {snap.name} -> best.pt", flush=True)
                    torch.save({**state, "best_eval": best_eval}, out_dir / "last.pt")
            if args.hours and time.time() - t_start > args.hours * 3600:
                print(f"stopping: {args.hours} h reached", flush=True)
                break
    finally:
        log.close()
        evals.close()
        v.close()
    print(f"done; checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
