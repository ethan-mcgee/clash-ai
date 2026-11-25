"""PPO for the Piggies EQ seat, against a league of opponent decks, anchored to imitation.

    python -m clashai.train.ppo --name ppo1 --hours 8
    python -m clashai.train.ppo --name ppo1 --resume --hours 8      # carries on a stopped run

The learner starts from an imitation checkpoint (``--init``, the generalist) and plays Piggies EQ
from a random seat in ``--envs`` battles at once. Every game draws its opponent:

- a deck from :class:`clashai.train.league.League` (the decks Piggies EQ meets in the replays,
  never a held-out one, weighted toward those it loses to), piloted by the reference network
  (``--reference``, the generalist), frozen;
- or, with probability ``--mirror-share``, Piggies EQ piloted by one of the learner's own recent
  snapshots, so it also plays against what it is becoming.

Each decision's reward is the game's result at its end (+1, -1, 0 for a draw) plus potential-based
shaping, ``gamma * phi(next) - phi(now)`` with ``phi = --shaping * (own tower share - enemy tower
share)`` and ``phi = 0`` at the end, which changes no optimal policy. The loss is PPO's clipped
surrogate, a value loss, a small entropy bonus, and ``--kl-coef`` times the KL divergence to the
reference network on the learner's own decisions, which keeps its play near human play. The first
``--critic-warmup`` iterations train the value alone, while the policy waits for a usable critic.

A run writes ``<runs>/ppo/<name>/``: ``last.pt`` (with the optimiser and the league, to resume),
``pool/iter_NNNNN.pt`` every ``--snapshot-every`` iterations (playable by the scorecard), and
``metrics.jsonl``, one line an iteration.
"""

from __future__ import annotations

import argparse
import json
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


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="PPO for Piggies EQ against a league")
    p.add_argument("--name", required=True)
    p.add_argument("--init", help=f"checkpoint to start from (default <runs>/{DEFAULT_INIT})")
    p.add_argument("--reference", help="the frozen pilot and KL anchor (default: --init's)")
    p.add_argument("--resume", action="store_true", help="carry on from the run's last.pt")
    p.add_argument("--iters", type=int, default=100000)
    p.add_argument("--hours", type=float, default=0, help="stop after this long (0: never)")
    p.add_argument("--envs", type=int, default=512)
    p.add_argument("--steps", type=int, default=128, help="decisions per battle an iteration")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--minibatch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--gamma", type=float, default=0.998)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--value-trunk-grad", type=float, default=0.1,
                   help="scale of the value loss's gradient into the shared trunk")
    p.add_argument("--ent-coef", type=float, default=0.001)
    p.add_argument("--kl-coef", type=float, default=0.05)
    p.add_argument("--shaping", type=float, default=0.5)
    p.add_argument("--critic-warmup", type=int, default=5)
    p.add_argument("--mirror-share", type=float, default=0.1)
    p.add_argument("--hard-share", type=float, default=0.5,
                   help="share of games drawn toward lost matchups (the rest by replay frequency)")
    p.add_argument("--snapshot-every", type=int, default=20)
    p.add_argument("--pool", type=int, default=5, help="recent snapshots the mirror games draw on")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

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
    init = Path(args.init) if args.init else paths.runs_dir() / DEFAULT_INIT
    reference_path = str(args.reference or (resume["reference"] if resume else init))
    net, init_ckpt = load_net(init, device)
    metadata = init_ckpt.get("metadata")
    net.set_dropout(0.0)  # rollout and update must see the same policy
    if resume:
        net.load_state_dict(resume["model"])
    net.train()
    spec, k = net.spec, net.config.max_entities
    pilots = Pilots(device, args.pool + 1)
    reference = pilots.get(reference_path)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-5,
                            weight_decay=0.0)
    decks, held = opponent_decks()
    league = League(decks, uniform=1 - args.hard_share)
    start_iter = 0
    if resume:
        opt.load_state_dict(resume["optimizer"])
        league.load_state_dict(resume["league"])
        start_iter = resume["iteration"] + 1
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
        "dones": np.zeros((steps, n), np.float32),
        "live": np.zeros((steps, n), bool),
    }
    log = open(out_dir / "metrics.jsonl", "a", encoding="utf-8")
    t_start = time.time()
    arange = np.arange(n)

    def forward(model, obs):
        h, e, valid = seat_inputs(obs, spec, k, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(h, e, valid)
        return out

    try:
        for it in range(start_iter, args.iters):
            t0 = time.time()
            league_probs[0] = league.probabilities()
            net.eval()
            finished = []  # (score, deck, weight) of the league games, mirror games apart
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
                    values = out["value"].float() * value_scale
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
                buf["values"][t] = values.cpu().numpy()
                buf["live"][t] = ~restarting
                phi_now = tower_potential(own[:, :hi], spec, args.shaping)
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

            with torch.no_grad():
                last_values = (forward(net, v.observations[arange, seat])["value"].float()
                               * value_scale).cpu().numpy()
            adv, ret = gae(buf["rewards"], buf["values"], buf["dones"], buf["live"],
                           last_values, args.gamma, args.lam)

            # -- update --
            t1 = time.time()
            net.train()
            idx_all = np.flatnonzero(buf["live"].ravel())
            flat = {key: val.reshape(steps * n, *val.shape[2:]) for key, val in buf.items()}
            adv_f, ret_f = adv.ravel(), ret.ravel()
            warm = it < args.critic_warmup
            stats: dict = {}
            for _ in range(args.epochs):
                order = rng.permutation(idx_all)
                for start in range(0, len(order), args.minibatch):
                    idx = np.sort(order[start:start + args.minibatch])
                    if len(idx) < args.minibatch // 4:
                        continue
                    obs = np.concatenate([flat["header"][idx], flat["entities"][idx]], 1)
                    h, e, valid = seat_inputs(obs, spec, k, device)
                    mask = unpack_masks(torch.from_numpy(flat["masks"][idx]).to(device),
                                        spec.actions)
                    acts_t = torch.from_numpy(flat["actions"][idx]).to(device)
                    old = torch.from_numpy(flat["logp"][idx]).to(device)
                    ref = torch.from_numpy(flat["ref_logp"][idx]).to(device)
                    a = torch.from_numpy(adv_f[idx]).to(device)
                    a = (a - a.mean()) / (a.std() + 1e-8)
                    target = torch.from_numpy(ret_f[idx]).to(device)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        o = net(h, e, valid)
                        logp, entropy = evaluate_actions(net, o, mask, acts_t)
                        value = critic(net, o, args.value_trunk_grad) * value_scale
                    ratio = torch.exp(logp - old)
                    pg = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                    d = ref - logp
                    kl_ref = (torch.exp(d) - d - 1).mean()
                    v_loss = F.mse_loss(value, target)
                    ent = entropy.mean()
                    if warm:
                        loss = args.vf_coef * v_loss
                    else:
                        loss = pg + args.vf_coef * v_loss - args.ent_coef * ent \
                            + args.kl_coef * kl_ref
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                    if warm:  # the policy's own heads stay as they are
                        for name, prm in net.named_parameters():
                            if not name.startswith("value_head.") and prm.grad is not None:
                                prm.grad = None
                    opt.step()
                    with torch.no_grad():
                        for key, val in (("policy_loss", pg), ("value_loss", v_loss),
                                         ("entropy", ent), ("kl_ref", kl_ref),
                                         ("approx_kl", (old - logp).mean()),
                                         ("clip_frac", ((ratio - 1).abs() > args.clip).float().mean())):
                            stats[key] = stats.get(key, 0.0) + float(val)
                        stats["_n"] = stats.get("_n", 0) + 1
            update_s = time.time() - t1
            nb = max(1, stats.pop("_n", 1))
            stats = {key: val / nb for key, val in stats.items()}
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
                "explained_variance": ev, "warmup": warm,
                "rollout_s": round(rollout_s, 1), "update_s": round(update_s, 1), **stats,
            }
            log.write(json.dumps(row) + "\n")
            log.flush()
            fmt = lambda x, f="{:.3f}": "-" if x is None else f.format(x)
            print(f"iter {it}: {len(finished)} games, score {fmt(row['score'])} "
                  f"(by frequency {fmt(row['score_by_frequency'])}), mirror "
                  f"{fmt(row['mirror_score'])} of {len(mirror_scores)}; kl_ref "
                  f"{stats.get('kl_ref', 0):.4f}, approx_kl {stats.get('approx_kl', 0):.4f}, clip "
                  f"{stats.get('clip_frac', 0):.3f}, value loss {stats.get('value_loss', 0):.4f}, "
                  f"ev {ev:.2f}, entropy {stats.get('entropy', 0):.3f}"
                  f"{' [critic warm-up]' if warm else ''} "
                  f"[{rollout_s:.0f}+{update_s:.0f} s, {row['minutes']} min]", flush=True)

            state = {"model": net.state_dict(), "description": net.describe(), "step": it,
                     "metadata": metadata, "optimizer": opt.state_dict(),
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
            if args.hours and time.time() - t_start > args.hours * 3600:
                print(f"stopping: {args.hours} h reached", flush=True)
                break
    finally:
        log.close()
        v.close()
    print(f"done; checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
