"""Imitation on whole battles in order, for recurrent models (and for scoring any model).

    python -m clashai.train.seq --name generalist-v3 --resume
    python -m clashai.train.seq --name v2-whole --evaluate <checkpoint>

Each step trains on ``--lanes`` battle-seats side by side, ``--chunk`` consecutive decisions of
each. A lane's recurrent state carries from one chunk to the next (detached: truncated
backpropagation through time) and restarts at zero where a new battle-seat begins, which can be
inside a chunk. Every decision is a row, waits included, each of weight 1: the same objective in
expectation as the row trainer's waits sampled at 5% with weight 20. Lost games count
``--loss-weight`` as much as won ones. The value head reads the core's output detached.

Per battle-seat, drawn once from the seed: whether it is mirrored left to right, and whether its
previous actions are hidden (``--drop-prev``), so the network also learns to play without them.
Rows are grouped by unit count and the trunk runs per group, then the rows go back in order for
the core.

Validation plays whole validation battle-seats (the row trainer's split, by tag), state carried,
and reports the policy loss split into gate, card and tile, the play-rate calibration, and for a
recurrent model the same loss with the state zeroed at every decision and with the previous
action hidden: how much the model uses its memory. ``--evaluate`` scores a checkpoint the same
way, a non-recurrent one too, so models compare on equal terms.

Built for runs of days: ``last.pt`` is written atomically every ``--checkpoint-minutes`` and at
every validation, with the optimiser, the averaged weights, the lanes' places and states and the
random state; ``--resume`` carries on exactly (and starts afresh when there is nothing to resume).
``--hours`` stops after a while. The log also goes to ``train.log`` in the run's folder.
"""

from __future__ import annotations

import argparse
import itertools
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
from ..model import memory as mem
from ..model.net import Config, PolicyNet
from ..model.policy import load_net
from ..model.spec import Spec, unpack_masks
from .bc import EMA, losses, summary
from .longrun import atomic_save, keep_awake, tee
from .rows import MAX_ENTITIES, battle_rows, is_val

DATASET = "dataset/all-screen-team0-v0-t0-standard"
WIDTHS = (16, 32)  # rows with at most this many units run the trunk together; the rest in one


# -- data -----------------------------------------------------------------------------------------

class Battles:
    """The labelled battles: battle-seat items, loading one seat's rows in time order."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.metadata = json.loads((self.folder / "metadata.json").read_text())
        self.spec = Spec.from_metadata(self.metadata)
        with open(self.folder / "index.jsonl", encoding="utf-8") as f:
            self.index = [json.loads(line) for line in f]
        self.val = np.array([is_val(r["tag"]) for r in self.index])
        self._cache: tuple[int, dict] | None = None
        self._lengths: np.ndarray | None = None

    def items(self, val: bool, limit: int = 0) -> list[tuple[int, int]]:
        battles = np.flatnonzero(self.val == val)
        if limit:
            battles = battles[:limit]
        return [(int(b), s) for b in battles for s in (0, 1)]

    def length(self, battle: int) -> int:
        """Decisions of each seat of ``battle``, from the labelling counts (read once)."""
        if self._lengths is None:
            by_tag = {}
            with open(self.folder / "counts.jsonl", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    by_tag[Path(r["source"]).stem] = r["counts"]["decisions"]
            self._lengths = np.array([by_tag.get(r["tag"], 0) for r in self.index])
        return int(self._lengths[battle])

    def decisions(self, items: list[tuple[int, int]]) -> int:
        return sum(self.length(b) for b, _ in items)

    def load(self, battle: int, seat: int) -> dict:
        if self._cache is None or self._cache[0] != battle:
            info = self.index[battle]
            d = battle_rows(self.folder / "battles" / f"{info['tag']}.npz", info, battle, 1.0, 0,
                            self.metadata["schema"]["header_fields"])
            self._cache = (battle, d)
        d = self._cache[1]
        rows = np.flatnonzero(d["seat"] == seat)
        offsets = np.concatenate([[0], np.cumsum(d["entity_counts"])])
        counts = d["entity_counts"][rows]
        take = (np.concatenate([np.arange(offsets[r], offsets[r] + c) for r, c in zip(rows, counts)])
                if len(rows) else np.zeros(0, np.int64))
        spec = self.spec
        actions = d["actions"][rows].astype(np.int64)
        header = d["header"][rows]
        cards = mem.played_cards(header, actions, spec.header["hand_card"], spec.first_ability,
                                 spec.tiles)
        prev, prev_card, since = mem.history(actions, cards)
        return {"header": header, "masks": d["masks"][rows], "actions": actions,
                "result": d["result"][rows], "opp_hand": d["opp_hand"][rows],
                "opp_next": d["opp_next"][rows], "counts": counts,
                "offsets": np.concatenate([[0], np.cumsum(counts)]), "entities": d["entities"][take],
                "prev_action": prev, "prev_card": prev_card, "since": since}


def _flag(seed: int, epoch: int, item: int, which: int, p: float) -> bool:
    return p > 0 and np.random.default_rng([seed, epoch, item, which]).random() < p


class Stream:
    """Chunks of ``lanes`` x ``chunk`` rows, lane-major, from items in a seeded order per epoch.

    Deterministic: the same seed and snapshot give the same chunks, so a resumed run continues
    exactly. Each chunk carries the snapshot to resume after it."""

    def __init__(self, battles: Battles, items: list, lanes: int, chunk: int, seed: int, *,
                 mirror: float, drop_prev: float, snapshot: dict | None = None):
        self.battles, self.items, self.lanes, self.chunk, self.seed = battles, items, lanes, chunk, seed
        self.mirror, self.drop_prev = mirror, drop_prev
        self._cache: dict[int, dict] = {}
        self._order_cache: tuple[int, np.ndarray | None] = (-1, None)
        if snapshot is None:
            self.epoch, self.cursor = 0, 0
            self.lane = []
            for j in range(lanes):  # staggered starts, so lanes are not all in minute one at once
                k, ep = self._pull()
                n = len(self._data(k)["actions"])
                self.lane.append({"item": k, "epoch": ep, "offset": int(j * n / lanes),
                                  "fresh": True})
        else:
            self.epoch, self.cursor = snapshot["epoch"], snapshot["cursor"]
            self.lane = [dict(x) for x in snapshot["lanes"]]

    def _order(self, epoch: int) -> np.ndarray:
        return np.random.default_rng([self.seed, epoch]).permutation(len(self.items))

    def _pull(self) -> tuple[int, int]:
        if self.cursor >= len(self.items):
            self.epoch, self.cursor = self.epoch + 1, 0
        if self._order_cache[0] != self.epoch:
            self._order_cache = (self.epoch, self._order(self.epoch))
        k = int(self._order_cache[1][self.cursor])
        self.cursor += 1
        return k, self.epoch

    def _data(self, k: int) -> dict:
        if k not in self._cache:  # one entry per lane: a finished seat's is dropped
            self._cache[k] = self.battles.load(*self.items[k])
        return self._cache[k]

    def snapshot(self) -> dict:
        return {"epoch": self.epoch, "cursor": self.cursor, "lanes": [dict(x) for x in self.lane]}

    def next(self) -> dict:
        parts = []  # (data, start, count, first, flip, drop)
        for ln in self.lane:
            need = self.chunk
            while need:
                data = self._data(ln["item"])
                n = len(data["actions"])
                if ln["offset"] >= n:
                    self._cache.pop(ln["item"], None)
                    k, ep = self._pull()
                    ln.update(item=k, epoch=ep, offset=0, fresh=True)
                    continue
                take = min(need, n - ln["offset"])
                flip = _flag(self.seed, ln["epoch"], ln["item"], 0, self.mirror)
                drop = _flag(self.seed, ln["epoch"], ln["item"], 1, self.drop_prev)
                parts.append((data, ln["offset"], take, ln["fresh"], flip, drop))
                ln["offset"] += take
                ln["fresh"] = False
                need -= take
        return {**assemble(parts), "snapshot": self.snapshot()}


def assemble(parts: list) -> dict:
    """Rows from (data, start, count, first, flip, drop) parts, in order, with the units grouped
    for the trunk (``groups``: row indices, padded units, valid)."""
    def cat(key):
        return np.concatenate([d[key][s:s + c] for d, s, c, *_ in parts])

    b = {k: cat(k) for k in ("header", "masks", "actions", "result", "opp_hand", "opp_next",
                              "prev_action", "prev_card", "since")}
    n = len(b["actions"])
    first = np.zeros(n, bool)
    flip = np.zeros(n, bool)
    at = 0
    counts, ents = [], []
    for d, s, c, fresh, f, drop in parts:
        first[at] = fresh
        flip[at:at + c] = f
        if drop:
            b["prev_action"][at:at + c] = mem.UNKNOWN
        cs = np.minimum(d["counts"][s:s + c], MAX_ENTITIES)
        starts = d["offsets"][s:s + c]
        counts.append(cs)
        ents.extend(d["entities"][o:o + k] for o, k in zip(starts, cs))
        at += c
    counts = np.concatenate(counts)
    b.update(first=first, flip=flip, weight=np.ones(n, np.float32), counts=counts)
    b["groups"] = groups(counts, ents)
    return b


def groups(counts: np.ndarray, ents: list) -> list:
    """Rows sorted by unit count and cut at :data:`WIDTHS`: (rows, units (k, w, ints), valid)."""
    order = np.argsort(counts, kind="stable")
    out = []
    edges = [0, *WIDTHS, MAX_ENTITIES]
    ints = ents[0].shape[1] if ents else 34
    for lo, hi in itertools.pairwise(edges):
        rows = order[(counts[order] > lo if lo else counts[order] >= 0) & (counts[order] <= hi)]
        if not len(rows):
            continue
        w = max(1, int(counts[rows].max()))
        w = min(MAX_ENTITIES, -(-w // 16) * 16)
        units = np.zeros((len(rows), w, ints), np.int32)
        valid = np.zeros((len(rows), w), bool)
        for i, r in enumerate(rows):
            k = counts[r]
            units[i, :k] = ents[r][:k]
            valid[i, :k] = True
        out.append((rows, units, valid))
    return out


class Loader:
    """A background thread filling a queue with pinned chunks from a :class:`Stream`."""

    def __init__(self, stream: Stream, depth: int = 4):
        self.stream, self.queue = stream, queue.Queue(depth)
        self.stop = False
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self.stop:
                self.queue.put(pin(self.stream.next()))
        except BaseException as e:  # noqa: BLE001 (surfaced in next())
            self.error = e
            self.queue.put(None)

    def next(self) -> dict:
        b = self.queue.get()
        if b is None:
            raise RuntimeError("the data loader failed") from self.error
        return b

    def close(self):
        self.stop = True
        while not self.queue.empty():
            self.queue.get_nowait()


def pin(b: dict) -> dict:
    out = {}
    for k, v in b.items():
        if k == "groups":
            out[k] = [tuple(torch.from_numpy(np.ascontiguousarray(a)).pin_memory() for a in g)
                      for g in v]
        elif k == "snapshot":
            out[k] = v
        else:
            out[k] = torch.from_numpy(np.ascontiguousarray(v)).pin_memory()
    return out


def to(b: dict, device) -> dict:
    out = {}
    for k, v in b.items():
        if k == "groups":
            out[k] = [tuple(a.to(device, non_blocking=True) for a in g) for g in v]
        elif k == "snapshot":
            out[k] = v
        else:
            out[k] = v.to(device, non_blocking=True)
    return out


# -- the network over a chunk ---------------------------------------------------------------------

def run(net: PolicyNet, b: dict, lanes: int, state=None, *, zero_state: bool = False,
        hide_prev: bool = False) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The network's output on a chunk (mirrored where ``flip``), and the mask, actions and
    header as it saw them. ``zero_state`` restarts the state at every row; ``hide_prev`` hides the previous
    action everywhere."""
    spec = net.spec
    n = len(b["actions"])
    mask_all = unpack_masks(b["masks"], spec.actions)
    actions_all = b["actions"].long()
    flip = b["flip"]
    parts, rows_all = [], []
    header = torch.empty_like(b["header"])
    mask = torch.empty_like(mask_all)
    actions = torch.empty_like(actions_all)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=b["header"].is_cuda):
        for rows, units, valid in b["groups"]:
            rows = rows.long()
            h, m, a = b["header"][rows], mask_all[rows], actions_all[rows]
            mh, mu, mm, ma = spec.mirror(h, units, valid, m, a)
            f = flip[rows]
            h = torch.where(f[:, None], mh, h)
            units = torch.where(f[:, None, None], mu, units)
            header[rows], mask[rows] = h, torch.where(f[:, None], mm, m)
            actions[rows] = torch.where(f, ma, a)
            parts.append((rows, net.encode(h, units, valid)))
            rows_all.append(rows)
        fixed = parts[0][1]["fixed"]
        width = max(p["trunk"].shape[1] for _, p in parts)
        inverse = torch.empty(n, dtype=torch.long, device=actions.device)
        inverse[torch.cat(rows_all)] = torch.arange(n, device=actions.device)

        def gather(key, value=0):
            xs = []
            for _, p in parts:
                x = p[key]
                pad = width - fixed - x.shape[1] if key != "trunk" else width - x.shape[1]
                if pad:
                    x = F.pad(x, (0, 0, 0, pad) if x.dim() == 3 else (0, pad), value=value)
                xs.append(x)
            return torch.cat(xs)[inverse]

        enc = {"trunk": gather("trunk"), "fixed": fixed, "valid": gather("valid", False),
               "entity_tile": gather("entity_tile")}
        prev = b["prev_action"].long()
        prev = torch.where(flip, spec.mirror_actions(prev), prev)
        if hide_prev:
            prev = torch.full_like(prev, mem.UNKNOWN)
        first = torch.ones_like(b["first"]) if zero_state else b["first"]
        memory = {"lanes": lanes, "first": first, "state": state, "prev_action": prev,
                  "prev_card": b["prev_card"], "since": b["since"]}
        out = net.heads(enc, memory)
    return out, mask, actions, header


def chunk_losses(net, b, lanes, state, *, loss_weight, value_weight, metrics=False, **variant):
    out, mask, actions, header = run(net, b, lanes, state, **variant)
    targets = {"weight": b["weight"], "result": b["result"], "opp_hand": b["opp_hand"],
               "opp_next": b["opp_next"], "piggies": torch.zeros_like(b["first"])}
    loss, stats = losses(net, out, targets, mask, actions, header[:, net.spec.header["tick"][0]],
                         policy_rows="all", loss_weight=loss_weight, value_weight=value_weight,
                         metrics=metrics)
    return loss, stats, out["state"]


# -- validation -----------------------------------------------------------------------------------

@torch.no_grad()
def validate(net: PolicyNet, battles: Battles, items: list, device, *, lanes: int = 128,
             chunk: int = 64, loss_weight: float = 0.5, value_weight: float = 0.1,
             variants: bool = True) -> dict:
    """Whole validation battle-seats, state carried; metrics as the row trainer's, the policy
    loss exact over every decision, plus the memory checks for a recurrent model. Seats run in
    groups of ``lanes`` of similar length, loaded a group at a time."""
    was_training = net.training
    net.eval()
    order = sorted(items, key=lambda it: -battles.length(it[0]))  # a battle's seats stay together
    kinds = {"": {}}
    if variants and net.config.lstm:
        kinds["zero_state"] = {"zero_state": True}
        if net.config.prev_action:
            kinds["hide_prev"] = {"hide_prev": True}
    sums: dict = {name: {} for name in kinds}
    rows = 0
    for g in range(0, len(order), lanes):
        group = [battles.load(*it) for it in order[g:g + lanes]]
        longest = max(len(d["actions"]) for d in group)
        rows += sum(len(d["actions"]) for d in group)
        for name, variant in kinds.items():
            state = None
            for start in range(0, longest, chunk):
                parts, real = [], []
                for d in group:
                    take = max(0, min(chunk, len(d["actions"]) - start))
                    if take:
                        parts.append((d, start, take, start == 0, False, False))
                    if take < chunk:  # after the seat's end: padding of weight 0
                        parts.append((_padding(d, chunk - take), 0, chunk - take, False, False,
                                      False))
                    real += [1.0] * take + [0.0] * (chunk - take)
                b = assemble(parts)
                b["weight"] = np.array(real, np.float32)
                _, s, state = chunk_losses(net, to(pin(b), device), len(group), state,
                                           loss_weight=loss_weight, value_weight=value_weight,
                                           metrics=True, **variant)
                k = int(b["weight"].sum())
                into = sums[name]
                for key, v in s.pop("sums").items():
                    into[key] = into.get(key, 0.0) + v
                if k:  # batch means: an all-padding chunk's are 0/0
                    for key, v in s.items():
                        into[key] = into.get(key, 0.0) + v * k
    result = {}
    for name in kinds:
        m = summary(sums[name], max(1, rows))
        # Exact, rather than a mean of chunk means: the weighted sum over every decision.
        m["policy_nll"] = (sums[name]["nll_gate"] + sums[name]["nll_card"]
                           + sums[name]["nll_tile"]) / max(1e-9, sums[name]["w"])
        if name:
            result[f"{name}_policy_nll"] = m["policy_nll"]
            result[f"{name}_cost"] = m["policy_nll"] - result["policy_nll"]
        else:
            result.update(m, decisions=rows)
    if was_training:
        net.train()
    return result


def _padding(d: dict, n: int) -> dict:
    """``n`` rows to pad a finished seat with: its last row repeated, as a wait with no units
    and no targets (the caller gives them weight 0)."""
    out = {k: np.repeat(v[-1:], n, axis=0) for k, v in d.items()
           if k not in ("offsets", "entities")}
    out["actions"][:] = 0
    out["opp_hand"][:] = -1
    out["opp_next"][:] = -1
    out["counts"][:] = 0
    out["offsets"] = np.zeros(n + 1, np.int64)
    out["entities"] = d["entities"][:0]
    return out


def describe(m: dict) -> str:
    s = (f"policy nll {m['policy_nll']:.4f} (gate {m['gate_nll']:.4f}, card {m['card_nll']:.4f}, "
         f"tile {m['tile_nll']:.4f}), calibration {m['act_calibration']:.3f}, card "
         f"{m['card_acc']:.3f}, cell {m['cell_acc']:.3f}, value mse {m['value_mse']:.3f} (sign by "
         f"minute {', '.join(f'{x:.0%}' for x in m['value_sign_by_minute'])}), next card "
         f"{m['next_card_acc']:.3f}")
    if "zero_state_cost" in m:
        s += f"; memory zeroed +{m['zero_state_cost']:.4f}"
    if "hide_prev_cost" in m:
        s += f", previous action hidden +{m['hide_prev_cost']:.4f}"
    return s


# -- training -------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    bool_flag = argparse.BooleanOptionalAction
    p = argparse.ArgumentParser(description="Imitation on whole battles (recurrent models)")
    p.add_argument("--name", required=True)
    p.add_argument("--dataset", default=DATASET, help="a labelled dataset under <data>")
    p.add_argument("--evaluate", help="only score this checkpoint on the validation battles")
    p.add_argument("--resume", action="store_true", help="carry on from last.pt if there is one")
    p.add_argument("--init", help="a checkpoint to start from (default: a new network)")
    p.add_argument("--lanes", type=int, default=64, help="battle-seats per step")
    p.add_argument("--chunk", type=int, default=32, help="consecutive decisions per lane a step")
    p.add_argument("--passes", type=float, default=3.0, help="passes over the training decisions")
    p.add_argument("--steps", type=int, default=0, help="steps (default: from --passes)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=3000)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--value-weight", type=float, default=0.1)
    p.add_argument("--loss-weight", type=float, default=0.5, help="policy weight of lost games")
    p.add_argument("--mirror", type=float, default=0.5)
    p.add_argument("--drop-prev", type=float, default=0.1,
                   help="share of battle-seats whose previous actions are hidden")
    p.add_argument("--ema", type=float, default=0.9995)
    p.add_argument("--d", type=int, default=384)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--lstm", type=int, default=512, help="recurrent core width (0: none)")
    p.add_argument("--attn-bias", action=bool_flag, default=True)
    p.add_argument("--relations", action=bool_flag, default=True)
    p.add_argument("--prev-action", action=bool_flag, default=True)
    p.add_argument("--eval-every", type=int, default=4000)
    p.add_argument("--val-seats", type=int, default=600,
                   help="validation battle-seats for the periodic checks (0: all)")
    p.add_argument("--patience", type=int, default=3,
                   help="after the first pass, stop after this many validations without a "
                        "better policy loss (0: never)")
    p.add_argument("--checkpoint-minutes", type=float, default=15)
    p.add_argument("--hours", type=float, default=0, help="stop (resumably) after this long")
    p.add_argument("--limit-battles", type=int, default=0,
                   help="train on only the first N training battles (checks)")
    p.add_argument("--val-on-train", action="store_true",
                   help="validate on the training battles (with --limit-battles: an overfit check)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    out = paths.runs_dir() / "seq" / args.name
    out.mkdir(parents=True, exist_ok=True)
    tee(out / "train.log")
    keep_awake()
    device = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(0.8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    battles = Battles(paths.data_root() / args.dataset)
    spec = battles.spec
    train_items = battles.items(False, args.limit_battles)
    val_items = train_items if args.val_on_train else battles.items(True)
    if args.val_seats and len(val_items) > args.val_seats:
        val_items = val_items[:args.val_seats]
    common = {"loss_weight": args.loss_weight, "value_weight": args.value_weight}

    if args.evaluate:
        net, _ = load_net(Path(args.evaluate), device)
        t = time.time()
        m = validate(net, battles, val_items, device, **common)
        print(f"{args.evaluate}: {describe(m)} [{m['decisions']:,} decisions, "
              f"{time.time() - t:.0f} s]", flush=True)
        (out / "evaluate.json").write_text(json.dumps({"checkpoint": args.evaluate, **m}, indent=1))
        return

    last = out / "last.pt"
    resumed = torch.load(last, map_location="cpu", weights_only=False) if (
        args.resume and last.exists()) else None
    if resumed:
        net = PolicyNet.from_description(resumed["description"]).to(device)
        net.load_state_dict(resumed["raw_model"])
        print(f"resuming {args.name} at step {resumed['step']}", flush=True)
    elif args.init:
        net, _ = load_net(Path(args.init), device)
        net.set_dropout(args.dropout)
    else:
        net = PolicyNet(spec, Config(d=args.d, layers=args.layers, heads=args.heads,
                                     dropout=args.dropout, attn_bias=args.attn_bias,
                                     relations=args.relations, lstm=args.lstm,
                                     prev_action=args.prev_action and args.lstm > 0)).to(device)
    net.train()
    print(f"{sum(q.numel() for q in net.parameters()) / 1e6:.1f}M parameters; config "
          f"{net.describe()['config']}", flush=True)

    rows_per_step = args.lanes * args.chunk
    decisions = battles.decisions(train_items)
    steps = args.steps or int(args.passes * decisions / rows_per_step)
    per_pass = decisions / rows_per_step
    print(f"{len(train_items):,} training battle-seats ({decisions:,} decisions); "
          f"{len(val_items):,} validation battle-seats; {steps:,} steps of {rows_per_step:,} rows "
          f"({steps / per_pass:.2f} passes)", flush=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))

    embeddings = ("card.", "cast.", "buff.", "token_type.", "position.", "kind.", "side.", "state.",
                  "prev_kind.", "prev_since.")
    decay = [q for n, q in net.named_parameters()
             if q.dim() >= 2 and not n.startswith(embeddings) and n != "relation_bias"]
    other = [q for q in net.parameters() if all(q is not x for x in decay)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": other, "weight_decay": 0.0}], lr=args.lr,
                            betas=(0.9, 0.98), fused=True)
    ema = EMA(net, args.ema) if args.ema else None
    step, best, stale = 0, math.inf, 0
    state = None
    snapshot = None
    if resumed:
        opt.load_state_dict(resumed["optimizer"])
        if ema:
            ema.net.load_state_dict(resumed["model"])
        step, best, stale = resumed["step"], resumed["best"], resumed["stale"]
        snapshot = resumed["stream"]
        if resumed["lstm_state"] is not None:
            state = tuple(s.to(device) for s in resumed["lstm_state"])
        torch.set_rng_state(resumed["rng"]["cpu"])
        torch.cuda.set_rng_state(resumed["rng"]["cuda"])

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        t = (s - args.warmup) / max(1, steps - args.warmup)
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))

    stream = Stream(battles, train_items, args.lanes, args.chunk, args.seed, mirror=args.mirror,
                    drop_prev=args.drop_prev, snapshot=snapshot)
    loader = Loader(stream)
    consumed = snapshot

    def save(m: dict | None):
        judged = ema.net if ema else net
        atomic_save({"model": judged.state_dict(), "raw_model": net.state_dict(),
                     "description": judged.describe(), "step": step,
                     "metadata": battles.metadata, "args": vars(args),
                     "optimizer": opt.state_dict(), "stream": consumed,
                     "lstm_state": None if state is None else tuple(s.cpu() for s in state),
                     "rng": {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()},
                     "best": best, "stale": stale, "metrics": m}, last)

    log = out / "metrics.jsonl"
    t0 = time.time()
    saved = tick = time.time()
    waited, running, seen = 0.0, {}, 0
    try:
        while step <= steps:
            if step % args.eval_every == 0 or step == steps:
                judged = ema.net if ema else net
                t = time.time()
                m = validate(judged, battles, val_items, device, **common)
                m.update(step=step, minutes=round((time.time() - t0) / 60, 1),
                         passes=round(step / per_pass, 3),
                         train={k: v / max(1, running.get("_n", 1)) for k, v in running.items()
                                if k != "_n"},
                         max_memory_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
                running = {}
                with open(log, "a", encoding="utf-8") as f:
                    f.write(json.dumps(m) + "\n")
                print(f"step {step} ({m['passes']:.2f} passes): {describe(m)} "
                      f"[validation {time.time() - t:.0f} s, {m['minutes']} min]", flush=True)
                if m["policy_nll"] < best - 1e-4:
                    best, stale = m["policy_nll"], 0
                    atomic_save({"model": judged.state_dict(), "description": judged.describe(),
                                 "step": step, "metadata": battles.metadata, "args": vars(args),
                                 "metrics": m}, out / "best.pt")
                elif step >= per_pass:
                    stale += 1
                save(m)
                saved = time.time()
                if step == steps:
                    break
                if args.patience and stale >= args.patience:
                    print(f"stopping early: no better validation loss in {stale} validations",
                          flush=True)
                    break
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            t = time.perf_counter()
            b = loader.next()
            waited += time.perf_counter() - t
            consumed = b["snapshot"]
            b = to(b, device)
            loss, s, new_state = chunk_losses(net, b, args.lanes, state, **common)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([q for g in opt.param_groups for q in g["params"]], 1.0)
            opt.step()
            if ema:
                ema.update(net, step)
            state = None if new_state is None else tuple(x.detach() for x in new_state)
            step += 1
            seen += rows_per_step
            for k, v in s.items():
                running[k] = running.get(k, 0.0) + v
            running["_n"] = running.get("_n", 0) + 1
            if step % 200 == 0:
                now = time.time()
                print(f"  step {step} loss {s['loss']:.4f} policy {s['policy']:.4f} "
                      f"({seen / (now - tick):,.0f} rows/s, waiting for data "
                      f"{waited / max(1e-9, now - tick):.0%}, lr {lr_at(step):.2e})", flush=True)
                waited, seen, tick = 0.0, 0, now
            if time.time() - saved > args.checkpoint_minutes * 60:
                save(None)
                saved = time.time()
            if args.hours and time.time() - t0 > args.hours * 3600:
                save(None)
                print(f"stopping after {args.hours} h at step {step}; --resume carries on",
                      flush=True)
                break
    finally:
        loader.close()
    print(f"done at step {step}; best validation policy loss {best:.4f}; checkpoints in {out}",
          flush=True)


if __name__ == "__main__":
    main()
