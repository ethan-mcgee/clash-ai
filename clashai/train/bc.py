"""Imitation: train the policy on human rows.

    python -m clashai.train.bc --name generalist --rows <rows folder> --steps 40000
    python -m clashai.train.bc --name specialist --rows <piggies rows> --init <generalist ckpt> \
        --policy-rows piggies --steps 10000 --lr 1e-4 --eval-every 1000

Losses, per row weighted by its wait-sampling weight:

- policy: the human action's negative log-likelihood, as type (wait, slot, ability) then tile given
  the slot, under the row's mask. A label the mask closes (given a tick or two after the decision)
  is opened for its row rather than dropped. Rows of a lost game count ``--loss-weight`` as much as
  a won one; with ``--policy-rows piggies`` only the Piggies EQ seat's rows train the policy.
- value: squared error to the seat's result (-1, 0, 1), weighted ``--value-weight``. Its head
  reads the trunk's output detached: a battle has one result and hundreds of nearly identical
  rows, so a value loss that reaches the trunk teaches it to recognise battles, not positions
  (the Piggies EQ baseline's value error rose above a constant guess's while its policy overfit).
- belief: the enemy's hand now (a soft target over its four cards) and its next card

Rows are mirrored left to right with probability ``--mirror``. Validation is on held-out battles.
The trunk trains with ``--dropout``. A run stops early once ``--patience`` validations in a row
fail to improve the policy loss; ``best.pt`` is the checkpoint with the lowest one.

``--ema`` keeps an exponential moving average of the weights, which is what is validated and
saved (``last.pt`` also keeps the raw weights). ``--profile N`` runs N steps under the PyTorch
profiler, writes ``profile.json`` (open it at ui.perfetto.dev) and stops. The progress lines show
the share of time spent waiting for batches: above a few percent, the data loader limits speed.

``--value-only`` fits a fresh value head on the frozen network of ``--init`` and nothing else,
stopping on the validation value error:

    python -m clashai.train.bc --name generalist-value --rows <rows> --init <ckpt> --value-only \\
        --steps 10000 --eval-every 500
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .. import paths
from ..model.net import Config, PolicyNet
from ..model.policy import load_net
from ..model.spec import Spec, unpack_masks
from .rows import Rows

NEG = -1e9


class Batches:
    """Background batch assembly into pinned memory, sampled uniformly from ``indices``."""

    def __init__(self, rows: Rows, indices: np.ndarray, batch: int, max_entities: int, seed: int,
                 workers: int = 3, depth: int = 6):
        self.rows, self.indices, self.batch, self.max_entities = rows, indices, batch, max_entities
        self.queue: queue.Queue = queue.Queue(depth)
        self.stop = False
        self.threads = [threading.Thread(target=self._run, args=(seed + k,), daemon=True)
                        for k in range(workers)]
        for t in self.threads:
            t.start()

    def _run(self, seed: int):
        # Rows drawn 16 batches at a time and cut into batches by entity count, so a batch pads
        # little; every row drawn is used once, so the sample stays uniform.
        rng = np.random.default_rng(seed)
        o = self.rows.entity_offsets
        while not self.stop:
            pool = rng.choice(self.indices, self.batch * 16)
            pool = pool[np.argsort(o[pool + 1] - o[pool], kind="stable")]
            for k in rng.permutation(16):
                idx = np.sort(pool[k * self.batch:(k + 1) * self.batch])
                b = self.rows.gather(idx, self.max_entities)
                self.queue.put({key: torch.from_numpy(np.ascontiguousarray(v)).pin_memory()
                                for key, v in b.items()})
                if self.stop:
                    return

    def next(self, device) -> dict:
        return {k: v.to(device, non_blocking=True) for k, v in self.queue.get().items()}

    def close(self):
        self.stop = True
        while not self.queue.empty():
            self.queue.get_nowait()


def to_device(b: dict, device) -> dict:
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in b.items()}


class EMA:
    """An exponential moving average of a network's weights. The decay ramps up from about 0.1
    over the first steps, so the average is not dominated by the starting weights."""

    def __init__(self, net: PolicyNet, decay: float):
        self.net, self.decay = copy.deepcopy(net).eval(), decay
        for p in self.net.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, net: PolicyNet, step: int) -> None:
        d = min(self.decay, (1 + step) / (10 + step))
        torch._foreach_lerp_(list(self.net.parameters()), list(net.parameters()), 1 - d)


MINUTE = 1200  # ticks
PHASES = 4  # the value's sign accuracy is reported for minutes 1, 2, 3 and overtime


def _mirrored(spec, b: dict, mirror: float, flip: torch.Tensor | None = None) -> tuple:
    """The batch's inputs, mask and actions, each row reflected with probability ``mirror`` or
    where ``flip`` (B,) bool says."""
    header, entities, valid = b["header"], b["entities"], b["valid"]
    mask = unpack_masks(b["masks"], spec.actions)
    actions = b["actions"].long()
    if flip is None and mirror > 0:
        flip = torch.rand(len(header), device=header.device) < mirror
    if flip is not None:
        mh, me, mm, ma = spec.mirror(header, entities, valid, mask, actions)
        header = torch.where(flip[:, None], mh, header)
        entities = torch.where(flip[:, None, None], me, entities)
        mask = torch.where(flip[:, None], mm, mask)
        actions = torch.where(flip, ma, actions)
    return header, entities, valid, mask, actions


def value_loss(net: PolicyNet, b: dict, *, mirror: float, dropout: float) -> tuple[torch.Tensor, dict]:
    """The value head's weighted squared error on a frozen network, its input under dropout."""
    header, entities, valid, _, _ = _mirrored(net.spec, b, mirror)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=header.is_cuda):
        match = net(header, entities, valid)["match"]
    x = F.dropout(match.float(), dropout, training=True)
    pred = torch.tanh(net.value_head(x).squeeze(-1))
    weight = b["weight"].float()
    loss = (weight * (pred - b["result"].float()) ** 2).sum() / weight.sum()
    return loss, {"loss": loss.item(), "value": loss.item()}


def forward_losses(net: PolicyNet, b: dict, *, policy_rows: str, loss_weight: float,
                   mirror: float = 0.0, value_weight: float = 0.1,
                   metrics: bool = False) -> tuple[torch.Tensor, dict]:
    header, entities, valid, mask, actions = _mirrored(net.spec, b, mirror)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=header.is_cuda):
        out = net(header, entities, valid)
    return losses(net, out, b, mask, actions, header[:, net.spec.header["tick"][0]],
                  policy_rows=policy_rows, loss_weight=loss_weight, value_weight=value_weight,
                  metrics=metrics)


def losses(net: PolicyNet, out: dict, b: dict, mask: torch.Tensor, actions: torch.Tensor,
           ticks: torch.Tensor, *, policy_rows: str, loss_weight: float, value_weight: float = 0.1,
           metrics: bool = False) -> tuple[torch.Tensor, dict]:
    """The losses of a batch from the network's output on it (see the module notes). ``b`` holds
    the rows' targets and weights; ``mask`` and ``actions`` are as the network saw them."""
    spec = net.spec
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=mask.is_cuda):
        kind, tile = spec.decode(actions)
        n = len(kind)
        ar = torch.arange(n, device=kind.device)
        type_mask = spec.type_mask(mask)
        type_mask[ar, kind] = True
        type_logits = out["type_logits"].float().masked_fill(~type_mask, NEG)
        nll_type = F.cross_entropy(type_logits, kind, reduction="none")
        play_rows = torch.nonzero((kind >= 1) & (kind <= spec.hand_slots)).squeeze(1)
        slot = kind[play_rows] - 1
        tile_mask = spec.tile_mask(mask[play_rows], slot)
        tile_mask[torch.arange(len(play_rows), device=kind.device), tile[play_rows]] = True
        tile_logits = net.tile_logits(out, play_rows, slot).float().masked_fill(~tile_mask, NEG)
        nll_tile = F.cross_entropy(tile_logits, tile[play_rows], reduction="none")
        # The value head trains on a detached trunk (see the module notes); out["value"] is the
        # same number, with the gradient path into the trunk.
        value_pred = torch.tanh(net.value_head(out["match"].detach()).squeeze(-1))

    weight = b["weight"].float()
    result = b["result"].float()
    outcome = torch.where(result < 0, loss_weight, 1.0)
    on = b["piggies"].float() if policy_rows == "piggies" else torch.ones_like(weight)
    w = weight * outcome * on
    nll = nll_type.clone()
    nll[play_rows] += nll_tile
    policy = (w * nll).sum() / w.sum().clamp(min=1e-6)

    value = (weight * (value_pred.float() - result) ** 2).sum() / weight.sum()

    hand = b["opp_hand"].long()
    known = hand >= 0
    logp = F.log_softmax(out["hand_belief"].float(), -1)
    hand_nll = -(logp.gather(1, hand.clamp(min=0)) * known).sum(1) / known.sum(1).clamp(min=1)
    belief_hand = (weight * hand_nll).sum() / weight.sum()
    nxt = b["opp_next"].long()
    next_nll = F.cross_entropy(out["next_belief"].float(), nxt.clamp(min=0), reduction="none")
    belief_next = (weight * next_nll * (nxt >= 0)).sum() / (weight * (nxt >= 0)).sum().clamp(min=1)

    loss = policy + value_weight * value + 0.1 * belief_hand + 0.1 * belief_next
    stats = {"loss": loss.item(), "policy": policy.item(), "value": value.item(),
             "belief_hand": belief_hand.item(), "belief_next": belief_next.item()}
    if metrics:
        stats["sums"] = (_metrics(spec, out, type_logits, tile_logits, kind, tile, play_rows, w,
                              hand, known, nxt, logp, result, weight, ticks, nll_tile))
    return loss, stats


@torch.no_grad()
def _metrics(spec, out, type_logits, tile_logits, kind, tile, play_rows, w, hand, known, nxt,
             logp, result, weight, ticks, nll_tile) -> dict:
    """Sums and counts, merged over validation batches by :func:`evaluate`.

    The policy loss splits into three parts per decision: the gate (play or ability, against
    wait), which card or ability given that it acts, and the tile given the card. ``act_p`` and
    ``act_w`` compare the expected and the true number of actions (calibration)."""
    phase = torch.div(ticks.long(), MINUTE, rounding_mode="floor").clamp(max=PHASES - 1)
    decided = (result != 0).float() * weight
    right = ((out["value"].float() > 0) == (result > 0)).float() * decided
    by_phase = {}
    for k in range(PHASES):
        at = (phase == k).float()
        by_phase[f"sign_ok_{k}"] = (right * at).sum().item()
        by_phase[f"sign_n_{k}"] = (decided * at).sum().item()
    on = w > 0
    play_on = on[play_rows]
    slot_logits = type_logits[play_rows, 1:1 + spec.hand_slots]
    card_ok = (slot_logits.argmax(1) == kind[play_rows] - 1) & play_on
    pred_tile = tile_logits.argmax(1)
    cell_ok = (pred_tile == tile[play_rows]) & play_on
    dr = (torch.div(pred_tile, spec.cols, rounding_mode="floor")
          - torch.div(tile[play_rows], spec.cols, rounding_mode="floor")).abs()
    dc = (pred_tile % spec.cols - tile[play_rows] % spec.cols).abs()
    near = (torch.maximum(dr, dc) <= 1) & play_on
    type_ok = (type_logits.argmax(1) == kind)
    act = kind != 0
    pred_act = type_logits.argmax(1) != 0
    top4 = logp.topk(4, -1).indices
    hand_hit = ((top4.unsqueeze(1) == hand.unsqueeze(2)).any(2) & known).sum(1).float()
    next_ok = (out["next_belief"].float().argmax(1) == nxt) & (nxt >= 0)
    logp_type = F.log_softmax(type_logits, -1)
    logp_wait = logp_type[:, 0]
    logp_act = torch.logsumexp(logp_type[:, 1:], -1)
    gate = torch.where(act, -logp_act, -logp_wait)
    card = torch.where(act, logp_act - logp_type.gather(1, kind[:, None]).squeeze(1),
                       torch.zeros_like(logp_act))
    tile_nll = torch.zeros_like(gate)
    tile_nll[play_rows] = nll_tile.float()
    heads = {"nll_gate": (w * gate).sum().item(), "nll_card": (w * card).sum().item(),
             "nll_tile": (w * tile_nll).sum().item(), "act_p": (w * logp_act.exp()).sum().item(),
             "act_w": (w * act).sum().item()}
    return {**heads, "n_play": play_on.sum().item(), "card_ok": card_ok.sum().item(),
            "cell_ok": cell_ok.sum().item(), "cell_near": near.sum().item(),
            "w": w.sum().item(), "type_ok_w": (w * type_ok).sum().item(),
            "act_true": (on & act).sum().item(), "act_recall": (on & act & pred_act).sum().item(),
            "act_pred_w": (w * pred_act).sum().item(),
            "act_precision_w": (w * (pred_act & act)).sum().item(),
            "value_se": (weight * (out["value"].float() - result) ** 2).sum().item(),
            "weight": weight.sum().item(),
            "hand_hit": hand_hit.sum().item(), "hand_known": known.sum().item(),
            "next_ok": next_ok.sum().item(), "next_known": (nxt >= 0).sum().item(), **by_phase}


@torch.no_grad()
def evaluate(net, rows: Rows, indices: np.ndarray, *, policy_rows: str, loss_weight: float,
             batch: int, device, value_weight: float = 0.1) -> dict:
    net.eval()
    sums: dict = {}
    for start in range(0, len(indices), batch):
        b = to_device(rows.gather(indices[start:start + batch], net.config.max_entities), device)
        _, s = forward_losses(net, b, policy_rows=policy_rows, loss_weight=loss_weight,
                              value_weight=value_weight, metrics=True)
        for k, v in s.pop("sums").items():
            sums[k] = sums.get(k, 0.0) + v
        for k, v in s.items():  # batch means, weighted by batch size
            sums[k] = sums.get(k, 0.0) + v * len(b["actions"])
    n = len(indices)
    net.train()
    return summary(sums, n)


def summary(sums: dict, n: int) -> dict:
    """Validation metrics from the sums of :func:`_metrics` (and the batch means of the losses,
    each times its batch size) over ``n`` rows."""
    return {
        "loss": sums["loss"] / n, "policy_nll": sums["policy"] / n,
        "card_acc": sums["card_ok"] / max(1, sums["n_play"]),
        "cell_acc": sums["cell_ok"] / max(1, sums["n_play"]),
        "cell_within_1": sums["cell_near"] / max(1, sums["n_play"]),
        "type_acc": sums["type_ok_w"] / max(1e-9, sums["w"]),
        "act_recall": sums["act_recall"] / max(1, sums["act_true"]),
        "act_precision": sums["act_precision_w"] / max(1e-9, sums["act_pred_w"]),
        "value_mse": sums["value_se"] / max(1e-9, sums["weight"]),
        "hand_top4_recall": sums["hand_hit"] / max(1, sums["hand_known"]),
        "next_card_acc": sums["next_ok"] / max(1, sums["next_known"]),
        "value_sign_by_minute": [round(sums[f"sign_ok_{k}"] / max(1e-9, sums[f"sign_n_{k}"]), 4)
                                 for k in range(PHASES)],
        "plays": int(sums["n_play"]),
        **{f"{k}_nll": sums[f"nll_{k}"] / max(1e-9, sums["w"]) for k in ("gate", "card", "tile")},
        "act_calibration": sums["act_p"] / max(1e-9, sums["act_w"]),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Imitation training")
    p.add_argument("--name", required=True)
    p.add_argument("--rows", required=True, help="a folder under <data>/rows")
    p.add_argument("--init", help="a checkpoint to start from")
    p.add_argument("--policy-rows", choices=("all", "piggies"), default="all")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.1,
                   help="trunk dropout (replaces an --init checkpoint's)")
    p.add_argument("--value-weight", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=3,
                   help="stop after this many validations without a better policy loss (0: never)")
    p.add_argument("--loss-weight", type=float, default=0.5, help="policy weight of lost games")
    p.add_argument("--mirror", type=float, default=0.5)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--eval-rows", type=int, default=60000)
    p.add_argument("--d", type=int, default=384)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--attn-bias", action="store_true",
                   help="new models: bias unit-to-unit attention by their relative positions")
    p.add_argument("--relations", action="store_true",
                   help="new models: bias unit-to-unit attention where one targets the other")
    p.add_argument("--ema", type=float, default=0.9995,
                   help="decay of the averaged weights that are validated and saved (0: off)")
    p.add_argument("--value-only", action="store_true",
                   help="fit a fresh value head on the frozen --init network, stopping on its error")
    p.add_argument("--value-dropout", type=float, default=0.1,
                   help="with --value-only: dropout on the value head's input")
    p.add_argument("--profile", type=int, default=0,
                   help="run this many steps under the profiler, write profile.json and stop")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.value_only and not args.init:
        raise SystemExit("--value-only needs --init: the network whose value head to fit")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    # Past this share PyTorch frees its cache instead of growing: on Windows the driver would
    # otherwise spill GPU memory into system RAM and slow training about tenfold.
    torch.cuda.set_per_process_memory_fraction(0.8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    t0 = time.time()
    rows = Rows(paths.data_root() / "rows" / args.rows)
    print(f"opened {rows.n:,} rows in {time.time() - t0:.0f} s", flush=True)
    spec = Spec.from_metadata(rows.metadata)
    if args.init:
        net, _ = load_net(Path(args.init), device)
        net.set_dropout(args.dropout)
        net.train()
        assert net.spec.to_dict() == spec.to_dict(), "the rows' layout differs from the checkpoint's"
    else:
        net = PolicyNet(spec, Config(d=args.d, layers=args.layers, heads=args.heads,
                                    dropout=args.dropout, attn_bias=args.attn_bias,
                                    relations=args.relations)).to(device)
    print(f"{sum(p.numel() for p in net.parameters()) / 1e6:.1f}M parameters", flush=True)

    keep = rows.piggies if args.policy_rows == "piggies" else np.ones(rows.n, bool)
    train_idx = np.flatnonzero(~rows.val)
    val_idx = np.flatnonzero(rows.val & keep)
    rng = np.random.default_rng(args.seed)
    if len(val_idx) > args.eval_rows:
        val_idx = np.sort(rng.choice(val_idx, args.eval_rows, replace=False))
    print(f"train rows {len(train_idx):,}; validation rows {len(val_idx):,}", flush=True)

    if args.value_only:
        # Everything but the value head is frozen; the head starts afresh.
        net.eval()
        for prm in net.parameters():
            prm.requires_grad_(False)
        for m in net.value_head:
            if isinstance(m, torch.nn.Linear):
                m.reset_parameters()
        for prm in net.value_head.parameters():
            prm.requires_grad_(True)
        decay = [prm for prm in net.value_head.parameters() if prm.dim() >= 2]
        other = [prm for prm in net.value_head.parameters() if prm.dim() < 2]
    else:
        embeddings = ("card.", "cast.", "buff.", "token_type.", "position.", "kind.", "side.",
                      "state.")
        decay = [prm for n, prm in net.named_parameters()
                 if prm.dim() >= 2 and not n.startswith(embeddings) and n != "relation_bias"]
        other = [prm for prm in net.parameters() if all(prm is not q for q in decay)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": other, "weight_decay": 0.0}], lr=args.lr,
                            betas=(0.9, 0.98), fused=True)
    ema = EMA(net, args.ema) if args.ema and not args.value_only else None
    # The selection metric: the policy loss, or for --value-only the value error.
    key = "value_mse" if args.value_only else "policy_nll"

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * t)))

    out = paths.runs_dir() / "bc" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    batches = Batches(rows, train_idx, args.batch, net.config.max_entities, args.seed)
    common = {"policy_rows": args.policy_rows, "loss_weight": args.loss_weight,
              "value_weight": args.value_weight}

    def train_step(step: int) -> tuple[dict, float]:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        t = time.perf_counter()
        b = batches.next(device)
        waited = time.perf_counter() - t
        if args.value_only:
            loss, s = value_loss(net, b, mirror=args.mirror, dropout=args.value_dropout)
        else:
            loss, s = forward_losses(net, b, mirror=args.mirror, **common)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([q for g in opt.param_groups for q in g["params"]], 1.0)
        opt.step()
        if ema:
            ema.update(net, step)
        return s, waited

    if args.profile:
        try:
            _profile(train_step, args.profile, out)
        finally:
            batches.close()
        return

    log = open(out / "metrics.jsonl", "a", encoding="utf-8")
    best = math.inf
    stale = 0  # validations since the best
    running: dict = {}
    waited = 0.0
    t0 = tick = time.time()
    try:
        for step in range(args.steps + 1):
            if step % args.eval_every == 0 or step == args.steps:
                judged = ema.net if ema else net
                m = evaluate(judged, rows, val_idx, batch=args.batch, device=device, **common)
                if args.value_only:
                    net.eval()  # evaluate() leaves it training; the frozen trunk stays in eval
                m.update(step=step, minutes=round((time.time() - t0) / 60, 1),
                         train={k: v / max(1, running.get("_n", 1)) for k, v in running.items()
                                if k != "_n"})
                running = {}
                log.write(json.dumps(m) + "\n")
                log.flush()
                print(f"step {step}: val policy nll {m['policy_nll']:.4f}, card {m['card_acc']:.3f}, "
                      f"cell {m['cell_acc']:.3f} (within 1: {m['cell_within_1']:.3f}), act recall "
                      f"{m['act_recall']:.3f} precision {m['act_precision']:.3f}, value mse "
                      f"{m['value_mse']:.3f} (sign right by minute "
                      f"{', '.join(f'{x:.0%}' for x in m['value_sign_by_minute'])}), hand top4 "
                      f"{m['hand_top4_recall']:.3f}, next card {m['next_card_acc']:.3f} "
                      f"[{m['minutes']} min]", flush=True)
                state = {"model": judged.state_dict(), "description": judged.describe(),
                         "step": step, "metadata": rows.metadata, "args": vars(args), "metrics": m}
                if ema:
                    torch.save({**state, "raw_model": net.state_dict()}, out / "last.pt")
                else:
                    torch.save(state, out / "last.pt")
                if m[key] < best - 1e-4:
                    best, stale = m[key], 0
                    torch.save(state, out / "best.pt")
                else:
                    stale += 1
                if step == args.steps:
                    break
                if args.patience and stale >= args.patience:
                    print(f"stopping early: no better validation {key} in {stale} "
                          f"validations ({stale * args.eval_every} steps)", flush=True)
                    break
            s, w = train_step(step)
            waited += w
            for k, v in s.items():
                running[k] = running.get(k, 0.0) + v
            running["_n"] = running.get("_n", 0) + 1
            if step % 200 == 0 and step:
                now = time.time()
                rate = step * args.batch / (now - t0)
                print(f"  step {step} loss {s['loss']:.4f} {key.split('_')[0]} "
                      f"{s['value' if args.value_only else 'policy']:.4f} ({rate:,.0f} rows/s, "
                      f"waiting for data {waited / max(1e-9, now - tick):.0%})", flush=True)
                waited, tick = 0.0, now
    finally:
        batches.close()
        log.close()
    print(f"done; best validation {key} {best:.4f}; checkpoints in {out}")


def _profile(train_step, steps: int, out: Path) -> None:
    """Runs ``steps`` training steps under the profiler, after 5 to warm up, and writes the trace."""
    from torch.profiler import ProfilerActivity, profile, schedule

    trace = out / "profile.json"
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=schedule(wait=2, warmup=3, active=steps),
                 on_trace_ready=lambda prof: prof.export_chrome_trace(str(trace))) as prof:
        waited = 0.0
        t0 = time.perf_counter()
        for step in range(5 + steps):
            if step == 5:
                torch.cuda.synchronize()
                waited, t0 = 0.0, time.perf_counter()
            waited += train_step(step)[1]
            prof.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    print(f"{steps} steps in {elapsed:.1f} s ({elapsed / steps * 1000:.0f} ms a step), "
          f"{waited / elapsed:.0%} of it waiting for data")
    print(f"trace written to {trace}: open it at https://ui.perfetto.dev (it stays on this machine)")


if __name__ == "__main__":
    main()
