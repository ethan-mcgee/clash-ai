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
"""

from __future__ import annotations

import argparse
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


def forward_losses(net: PolicyNet, b: dict, *, policy_rows: str, loss_weight: float,
                   mirror: float = 0.0, value_weight: float = 0.1,
                   metrics: bool = False) -> tuple[torch.Tensor, dict]:
    spec = net.spec
    header, entities, valid = b["header"], b["entities"], b["valid"]
    mask = unpack_masks(b["masks"], spec.actions)
    actions = b["actions"].long()
    if mirror > 0:
        flip = torch.rand(len(header), device=header.device) < mirror
        mh, me, mm, ma = spec.mirror(header, entities, valid, mask, actions)
        header = torch.where(flip[:, None], mh, header)
        entities = torch.where(flip[:, None, None], me, entities)
        mask = torch.where(flip[:, None], mm, mask)
        actions = torch.where(flip, ma, actions)

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=header.is_cuda):
        out = net(header, entities, valid)
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
                              hand, known, nxt, logp, result, weight))
    return loss, stats


@torch.no_grad()
def _metrics(spec, out, type_logits, tile_logits, kind, tile, play_rows, w, hand, known, nxt,
             logp, result, weight) -> dict:
    """Sums and counts, merged over validation batches by :func:`evaluate`."""
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
    return {"n_play": play_on.sum().item(), "card_ok": card_ok.sum().item(),
            "cell_ok": cell_ok.sum().item(), "cell_near": near.sum().item(),
            "w": w.sum().item(), "type_ok_w": (w * type_ok).sum().item(),
            "act_true": (on & act).sum().item(), "act_recall": (on & act & pred_act).sum().item(),
            "act_pred_w": (w * pred_act).sum().item(),
            "act_precision_w": (w * (pred_act & act)).sum().item(),
            "value_se": (weight * (out["value"].float() - result) ** 2).sum().item(),
            "weight": weight.sum().item(),
            "hand_hit": hand_hit.sum().item(), "hand_known": known.sum().item(),
            "next_ok": next_ok.sum().item(), "next_known": (nxt >= 0).sum().item()}


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
        "plays": int(sums["n_play"]),
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
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    # Past this share PyTorch frees its cache instead of growing: on Windows the driver would
    # otherwise spill GPU memory into system RAM and slow training about tenfold.
    torch.cuda.set_per_process_memory_fraction(0.8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    t0 = time.time()
    rows = Rows(paths.data_root() / "rows" / args.rows)
    print(f"loaded {rows.n:,} rows in {time.time() - t0:.0f} s", flush=True)
    spec = Spec.from_metadata(rows.metadata)
    if args.init:
        net, _ = load_net(Path(args.init), device)
        net.set_dropout(args.dropout)
        net.train()
        assert net.spec.to_dict() == spec.to_dict(), "the rows' layout differs from the checkpoint's"
    else:
        net = PolicyNet(spec, Config(d=args.d, layers=args.layers, heads=args.heads,
                                    dropout=args.dropout)).to(device)
    print(f"{sum(p.numel() for p in net.parameters()) / 1e6:.1f}M parameters", flush=True)

    keep = rows.piggies if args.policy_rows == "piggies" else np.ones(rows.n, bool)
    train_idx = np.flatnonzero(~rows.val)
    val_idx = np.flatnonzero(rows.val & keep)
    rng = np.random.default_rng(args.seed)
    if len(val_idx) > args.eval_rows:
        val_idx = np.sort(rng.choice(val_idx, args.eval_rows, replace=False))
    print(f"train rows {len(train_idx):,}; validation rows {len(val_idx):,}", flush=True)

    embeddings = ("card.", "cast.", "buff.", "token_type.", "position.", "kind.", "side.", "state.")
    decay = [p for n, p in net.named_parameters() if p.dim() >= 2 and not n.startswith(embeddings)]
    other = [p for p in net.parameters() if all(p is not q for q in decay)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": other, "weight_decay": 0.0}], lr=args.lr,
                            betas=(0.9, 0.98), fused=True)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * t)))

    out = paths.runs_dir() / "bc" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    log = open(out / "metrics.jsonl", "a", encoding="utf-8")
    batches = Batches(rows, train_idx, args.batch, net.config.max_entities, args.seed)
    common = {"policy_rows": args.policy_rows, "loss_weight": args.loss_weight,
              "value_weight": args.value_weight}
    best = math.inf
    stale = 0  # validations since the best
    running: dict = {}
    t0 = time.time()
    try:
        for step in range(args.steps + 1):
            if step % args.eval_every == 0 or step == args.steps:
                m = evaluate(net, rows, val_idx, batch=args.batch, device=device, **common)
                m.update(step=step, minutes=round((time.time() - t0) / 60, 1),
                         train={k: v / max(1, running.get("_n", 1)) for k, v in running.items()
                                if k != "_n"})
                running = {}
                log.write(json.dumps(m) + "\n")
                log.flush()
                print(f"step {step}: val policy nll {m['policy_nll']:.4f}, card {m['card_acc']:.3f}, "
                      f"cell {m['cell_acc']:.3f} (within 1: {m['cell_within_1']:.3f}), act recall "
                      f"{m['act_recall']:.3f} precision {m['act_precision']:.3f}, value mse "
                      f"{m['value_mse']:.3f}, hand top4 {m['hand_top4_recall']:.3f}, next card "
                      f"{m['next_card_acc']:.3f} [{m['minutes']} min]", flush=True)
                state = {"model": net.state_dict(), "description": net.describe(), "step": step,
                         "metadata": rows.metadata,
                         "args": vars(args), "metrics": m}
                torch.save(state, out / "last.pt")
                if m["policy_nll"] < best - 1e-4:
                    best, stale = m["policy_nll"], 0
                    torch.save(state, out / "best.pt")
                else:
                    stale += 1
                if step == args.steps:
                    break
                if args.patience and stale >= args.patience:
                    print(f"stopping early: no better validation policy loss in {stale} "
                          f"validations ({stale * args.eval_every} steps)", flush=True)
                    break
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            b = batches.next(device)
            loss, s = forward_losses(net, b, mirror=args.mirror, **common)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            for k, v in s.items():
                running[k] = running.get(k, 0.0) + v
            running["_n"] = running.get("_n", 0) + 1
            if step % 200 == 0 and step:
                rate = step * args.batch / (time.time() - t0)
                print(f"  step {step} loss {s['loss']:.4f} policy {s['policy']:.4f} "
                      f"({rate:,.0f} rows/s)", flush=True)
    finally:
        batches.close()
        log.close()
    print(f"done; best validation policy nll {best:.4f}; checkpoints in {out}")


if __name__ == "__main__":
    main()
