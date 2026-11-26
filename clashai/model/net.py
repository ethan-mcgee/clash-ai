"""The policy: an entity transformer over one seat's v4 observation.

Tokens: one for the match's scalars, one per hand slot, the next card, the queue, the deck, the
enemy's seen cards, the enemy's last 8 plays, the seat's waiting plays, the two champion slots,
then one per entity on the arena. A pre-norm transformer mixes them.

Heads:

- type: wait (from the match token), a play of each hand slot (from that slot's token), each
  ability (from its slot's token); masked to what is open
- tile, given the slot: entity outputs scattered onto the 32 x 18 tile grid, conditioned on the slot
  and the match token, through a small dilated conv net, one logit per tile
- value: the seat's result, -1 to 1
- belief: the enemy's hand now and the enemy's next card, as distributions over the vocabulary.
  Training targets come from the other seat's rows; the input never holds them.

Optional, off in older checkpoints (each a :class:`Config` field):

- ``attn_bias``: each trunk layer adds to the attention score between two units a per-head bias
  from where they stand relative to each other (offset, distance, same lane, overlap), through a
  small network of its own. Its last layer starts at zero, so a new model starts as without it.
- ``relations``: each layer adds a learned per-head bias where one unit targets the other, and
  another where it is targeted by it (an entity's ``target`` is the other's ``id``).
- ``lstm``: a recurrent core after the trunk. Its input is the match token (with ``prev_action``,
  joined by the seat's own previous action, the card it played and how long ago it last acted;
  :mod:`clashai.model.memory`). Its output is added to the match, hand and ability tokens, so
  every head reads it; the value head reads it too (the trainer detaches it). Such a model is
  called with ``memory`` (see :meth:`PolicyNet.heads`) and runs its heads in fp32, since its
  state is carried over a whole battle.

Every amount is scaled here from crforge's integers, so the model takes the raw observation.
"""

from __future__ import annotations

import itertools
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from . import memory as mem
from .spec import Spec

TOWERS = [f"{who}_{t}_{k}" for who in ("own", "enemy") for t in ("king", "left", "right")
          for k in ("hp", "max_hp")]
SCALARS = [("tick", 6000), ("elixir", 100000), ("elixir_reserved", 100000), ("crowns", 3),
           ("crowns_enemy", 3), ("hand_refill_ms", 1000), ("entity_count", 50),
           ("entity_overflow", 10), ("pending_count", 4), ("time_ms", 300000), ("section", 3),
           ("section_type", 3), ("section_left_ms", 180000), ("full_bar_ms", 28000),
           ("overtime", 1), ("play_open_in_ticks", 90), ("enemy_play_count", 50),
           ("enemy_elixir", 100000)] + [(t, 8000) for t in TOWERS]
REFUSALS = ["last_refusal", "last_run_refusal", "last_ability_run_refusal"]
ENTITY_NUMERIC = 28 + 12


@dataclass
class Config:
    d: int = 384
    layers: int = 6
    heads: int = 6
    grid_channels: int = 64
    max_entities: int = 80
    dropout: float = 0.0  # in the trunk's attention and feed-forward layers, while training
    attn_bias: bool = False  # a bias on unit-to-unit attention from their positions
    relations: bool = False  # a bias on unit-to-unit attention where one targets the other
    lstm: int = 0  # width of a recurrent core after the trunk (0: none)
    prev_action: bool = False  # the core also reads the seat's own previous action


PAIR_FEATURES = 8
PAIR_HIDDEN = 32


def _onehot(x: torch.Tensor, n: int) -> torch.Tensor:
    return F.one_hot(x.long().clamp(0, n - 1), n).float()


def _bits(x: torch.Tensor, n: int) -> torch.Tensor:
    """The low n bits of x, each 0 or 1. Arithmetic rather than shifts and masks, which ONNX does
    not take on signed integers; equal to ``(x >> k) & 1`` for every integer x, negative too."""
    powers = 2 ** torch.arange(n, device=x.device)
    return torch.remainder(torch.div(x.long().unsqueeze(-1), powers, rounding_mode="floor"), 2).float()


class Block(nn.Module):
    def __init__(self, c: int, dilation: int):
        super().__init__()
        self.a = nn.Conv2d(c, c, 3, padding=dilation, dilation=dilation)
        self.b = nn.Conv2d(c, c, 3, padding=1)
        self.norm = nn.GroupNorm(8, c)

    def forward(self, x):
        return x + self.b(F.gelu(self.a(F.gelu(self.norm(x)))))


class PolicyNet(nn.Module):
    def __init__(self, spec: Spec, config: Config | None = None):
        super().__init__()
        self.spec, self.config = spec, config or Config()
        d, c = self.config.d, self.config.grid_channels
        self.card = nn.Embedding(spec.vocab + 1, d)  # vocabulary id + 1; 0 for none or unknown
        self.cast = nn.Embedding(spec.vocab + 1, d)
        self.buff = nn.Embedding(spec.buffs + 1, d)
        self.token_type = nn.Embedding(10, d)
        self.position = nn.Embedding(8, d)  # the place within a group: slot, queue place, recency
        self.kind = nn.Embedding(5, d)
        self.side = nn.Embedding(3, d)
        self.state = nn.Embedding(18, d)
        self.match_in = nn.Linear(len(SCALARS) + 6 + len(REFUSALS), d)
        self.hand_in = nn.Linear(6, d)
        self.deck_in = nn.Linear(4, d)
        self.seen_in = nn.Linear(3, d)
        self.play_in = nn.Linear(20, d)
        self.pending_in = nn.Linear(4, d)
        self.ability_in = nn.Linear(21, d)
        self.entity_in = nn.Linear(ENTITY_NUMERIC, d)
        self.buff_ms_in = nn.Linear(3, d)
        self.norm_in = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(d, self.config.heads, 4 * d, dropout=self.config.dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.trunk = nn.TransformerEncoder(layer, self.config.layers, enable_nested_tensor=False)
        self.norm_out = nn.LayerNorm(d)

        def head(out):
            return nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, out))

        self.wait_head, self.slot_head, self.ability_head = head(1), head(1), head(1)
        self.value_head = head(1)
        self.hand_belief, self.next_belief = head(spec.vocab), head(spec.vocab)
        self.grid_in = nn.Linear(d, c)
        self.grid_position = nn.Parameter(torch.zeros(1, c, spec.rows, spec.cols))
        self.film = nn.Linear(2 * d, 2 * c)
        self.grid = nn.Sequential(*(Block(c, k) for k in (1, 2, 4, 1)))
        self.grid_out = nn.Conv2d(c, 1, 1)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        heads, layers = self.config.heads, self.config.layers
        if self.config.attn_bias:
            self.pair_mlp = nn.ModuleList(
                nn.Sequential(nn.Linear(PAIR_FEATURES, PAIR_HIDDEN), nn.GELU(),
                              nn.Linear(PAIR_HIDDEN, heads)) for _ in range(layers))
            for mlp in self.pair_mlp:
                nn.init.zeros_(mlp[2].weight)
                nn.init.zeros_(mlp[2].bias)
        if self.config.relations:
            self.relation_bias = nn.Parameter(torch.zeros(layers, 2, heads))
        if self.config.prev_action and not self.config.lstm:
            raise ValueError("prev_action feeds the recurrent core: it needs lstm > 0")
        if self.config.lstm:
            h = self.config.lstm
            if self.config.prev_action:
                self.prev_kind = nn.Embedding(5, d)  # first decision, wait, play, ability, unknown
                self.prev_tile = nn.Linear(2, d)
                self.prev_since = nn.Embedding(mem.SINCE_BUCKETS, d)
            self.core_in = nn.Sequential(nn.Linear(2 * d if self.config.prev_action else d, h),
                                         nn.GELU())
            self.core = nn.LSTM(h, h, batch_first=True)
            with torch.no_grad():  # forget gate bias 1: remember by default at the start
                self.core.bias_ih_l0[h:2 * h].fill_(1.0)
                self.core.bias_hh_l0[h:2 * h].zero_()
            self.core_out = nn.Linear(h, 3 * d)
            self.core_norm = nn.ModuleList(nn.LayerNorm(d) for _ in range(3))

    def set_dropout(self, p: float) -> None:
        """Change the trunk's dropout, for a run that starts from another run's checkpoint."""
        self.config.dropout = p
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.p = p
            elif isinstance(m, nn.MultiheadAttention):
                m.dropout = p

    def describe(self) -> dict:
        return {"spec": self.spec.to_dict(), "config": asdict(self.config)}

    @classmethod
    def from_description(cls, description: dict) -> PolicyNet:
        return cls(Spec(**description["spec"]), Config(**description["config"]))

    # -- inputs --------------------------------------------------------------------------------

    def _ids(self, header: torch.Tensor, name: str, table: int | None = None) -> torch.Tensor:
        """Vocabulary ids of a header field, + 1, with -1 (none, unknown) at 0."""
        n = self.spec.vocab if table is None else table
        return header[:, self.spec.header[name]].long().clamp(-1, n - 1) + 1

    def _tokens(self, header: torch.Tensor, entities: torch.Tensor, valid: torch.Tensor):
        hf = header.float()

        def H(name):
            return hf[:, self.spec.header[name]]

        def col(x):
            return x.unsqueeze(-1)

        pos = self.position.weight
        typ = self.token_type.weight

        scalars = [H(n)[:, :1] / s for n, s in SCALARS]
        towers = [H(f"{w}_{t}_hp")[:, :1] / H(f"{w}_{t}_max_hp")[:, :1].clamp(min=1)
                  for w in ("own", "enemy") for t in ("king", "left", "right")]
        refusals = [(H(n)[:, :1] != 0).float() for n in REFUSALS]
        match = self.match_in(torch.cat(scalars + towers + refusals, 1)).unsqueeze(1) + typ[0]

        hand = self.card(self._ids(header, "hand_card")) + self.hand_in(torch.cat([
            col(H("hand_cost") / 10), col(H("hand_pending")), _onehot(H("hand_form"), 3),
            col((H("hand_card") < 0).float())], -1)) + pos[:4] + typ[1]
        nxt = self.card(self._ids(header, "next_card")) + typ[2]  # (B, 1, d)
        queue = self.card(self._ids(header, "queue")) + pos[:4] + typ[3]

        slot_flags = header[:, self.spec.header["deck_slot"]].long()
        deck = self.card(self._ids(header, "deck")) + self.deck_in(torch.stack([
            torch.remainder(slot_flags, 2).float(),
            torch.remainder(torch.div(slot_flags, 2, rounding_mode="floor"), 2).float(),
            (H("deck_evolution") + 1) / 3, H("deck_level") / 16], -1)) + typ[4]

        seen = self.card(self._ids(header, "enemy_seen")) + self.seen_in(torch.stack([
            (H("enemy_seen_evolution") + 1) / 3, H("enemy_seen_evolution_need") / 3,
            (H("enemy_seen") >= 0).float()], -1)) + pos[:8] + typ[5]

        ticks, px, py = H("enemy_play_ticks"), H("enemy_play_x"), H("enemy_play_y")
        plays = (self.card(self._ids(header, "enemy_play_card"))
                 + self.cast(self._ids(header, "enemy_play_cast"))
                 + self.play_in(torch.cat([
                     _onehot(H("enemy_play_form").clamp(min=0), 3), col(ticks / 1000),
                     col(torch.log1p(ticks.clamp(min=0)) / 8), col(px.clamp(min=0) / 18000),
                     col(py.clamp(min=0) / 32000), col((px != -1).float()),
                     _onehot(H("enemy_play_kind"), 6), _bits(H("enemy_play_flags"), 4),
                     col(H("enemy_play_point_ticks").clamp(min=0) / 1000),
                     col((H("enemy_play_card") >= 0).float())], -1))
                 + pos[:8] + typ[6])

        pending = self.card(self._ids(header, "pending_card")) + self.pending_in(torch.stack([
            (H("pending_card") >= 0).float(), H("pending_x") / 18000, H("pending_y") / 32000,
            H("pending_ticks") / 20], -1)) + pos[:4] + typ[7]

        ability = self.card(self._ids(header, "ability_champion")) + self.ability_in(torch.cat([
            col((H("ability_champion") >= 0).float()), col((H("ability_unit") >= 0).float()),
            _onehot(H("ability_state"), 16), col(H("ability_cooldown_ms") / 3000),
            col((H("ability_charges") + 1) / 3), col(H("ability_command_ticks") / 20)], -1)) + pos[:2] + typ[8]

        ents = self._entity_tokens(entities) + typ[9]
        tokens = torch.cat([match, hand, nxt, queue, deck, seen, plays, pending,
                            ability, ents], 1)
        fixed = tokens.shape[1] - entities.shape[1]
        pad = torch.cat([torch.zeros(header.shape[0], fixed, dtype=torch.bool, device=header.device),
                         ~valid], 1)
        return self.norm_in(tokens), pad, fixed

    def _entity_tokens(self, entities: torch.Tensor) -> torch.Tensor:
        e = entities.float()

        def f(name):
            return e[..., self.spec.entity[name]]

        hp, max_hp = f("hp"), f("max_hp")
        known = (hp >= 0).float()
        target = (f("target") >= 0).float()
        x, y = f("x"), f("y")
        numeric = torch.cat([torch.stack([
            x / 18000, y / 32000, known, known * hp.clamp(min=0) / max_hp.clamp(min=1),
            hp.clamp(min=0) / 8000, max_hp.clamp(min=0) / 8000, f("shield").clamp(min=0) / 1000,
            f("level").clamp(min=0) / 16, f("dx") / 1000, f("dy") / 1000, target,
            target * f("target_x") / 18000, target * f("target_y") / 32000,
            target * (f("target_x") - x) / 18000, target * (f("target_y") - y) / 32000,
            f("deploy_ms") / 1000, f("attack_ms").clamp(max=20000) / 1000, f("load_ms") / 1000,
            f("speed_percent") / 100, f("hit_speed_percent") / 100, f("buff_count") / 3,
            f("aim_x") / 18000, f("aim_y") / 32000, f("time_left_ms").clamp(max=20000) / 1000,
            f("radius") / 5000, (f("ability_ticks") >= 0).float(),
            f("ability_ticks").clamp(0, 1000) / 200, (f("row") < 0).float()], -1),
            _bits(f("flags"), 12)], -1)
        ids = entities.long()
        row = ids[..., self.spec.entity["row"]].clamp(-1, self.spec.vocab - 1) + 1
        buffs = sum(self.buff(ids[..., self.spec.entity[f"buff_{k}"]].clamp(-1, self.spec.buffs - 1) + 1)
                    for k in range(3))
        ms = torch.stack([f(f"buff_ms_{k}") for k in range(3)], -1)
        ms = torch.where(ms < 0, torch.full_like(ms, 60000), ms.clamp(0, 60000))
        return (self.card(row) + self.kind(ids[..., self.spec.entity["kind"]].clamp(0, 4))
                + self.side(ids[..., self.spec.entity["side"]].clamp(0, 2))
                + self.state(ids[..., self.spec.entity["state"]].clamp(-1, 16) + 1)
                + buffs + self.buff_ms_in(torch.log1p(ms) / 11) + self.entity_in(numeric))

    # -- forward -------------------------------------------------------------------------------

    fp32_heads: bool = False
    """Run everything after the trunk in fp32 even under bf16 autocast: the type and tile scores
    are then exact enough to rank, which matters when sampling sharpened or taking the most likely
    card and tile. Not part of the configuration: a play-time choice (NetPolicy sets it)."""

    def forward(self, header: torch.Tensor, entities: torch.Tensor, valid: torch.Tensor,
                memory: dict | None = None) -> dict:
        """header (B, header_ints) int, entities (B, N, entity_ints) int, valid (B, N) bool;
        ``memory`` for a recurrent model (:meth:`heads`)."""
        return self.heads(self.encode(header, entities, valid), memory)

    def encode(self, header: torch.Tensor, entities: torch.Tensor, valid: torch.Tensor) -> dict:
        """The tokens through the trunk: the trunk's normalised output with what the heads need."""
        tokens, pad, fixed = self._tokens(header, entities, valid)
        if self.config.attn_bias or self.config.relations:
            trunk = self.trunk_explicit(tokens, pad, self._pairs(entities, valid), fixed)
        elif torch.onnx.is_in_onnx_export():
            trunk = self.trunk_explicit(tokens, pad)
        else:
            trunk = self.trunk(tokens, src_key_padding_mask=pad)
        ex = entities[..., self.spec.entity["x"]].long()
        ey = entities[..., self.spec.entity["y"]].long()
        tile = ((ey // 1000).clamp(0, self.spec.rows - 1) * self.spec.cols
                + (ex // 1000).clamp(0, self.spec.cols - 1))
        return {"trunk": self.norm_out(trunk), "fixed": fixed, "valid": valid,
                "entity_tile": tile}

    def heads(self, enc: dict, memory: dict | None = None) -> dict:
        """Every head's output from :meth:`encode`'s.

        A recurrent model (``lstm`` > 0) needs ``memory``. Its B rows are ``lanes`` sequences of
        T = B / lanes consecutive decisions, lane-major (row = lane * T + t):

        - ``lanes``: the number of sequences
        - ``first`` (B,) bool: the decision starts a battle-seat; the state restarts from zero
        - ``state``: (h, c), each (1, lanes, lstm), carried from before the first row; None for
          zeros
        - ``prev_action``, ``prev_card``, ``since`` (B,): with ``prev_action``, the seat's own
          past (:mod:`clashai.model.memory`)

        The output's ``state`` is the state after each lane's last row. Other models ignore
        ``memory``.
        """
        out, fixed = enc["trunk"], enc["fixed"]
        with self._head_precision(out) as out:
            match, hand, ability = out[:, 0], out[:, 1:5], out[:, fixed - 2:fixed]
            state = None
            if self.config.lstm:
                if memory is None:
                    raise ValueError("a recurrent model needs memory (see PolicyNet.heads)")
                match, hand, ability, state = self._core(match, hand, ability, memory)
            type_logits = torch.cat([self.wait_head(match), self.slot_head(hand).squeeze(-1),
                                     self.ability_head(ability).squeeze(-1)], 1)
            return {"type_logits": type_logits,
                    "value": torch.tanh(self.value_head(match).squeeze(-1)),
                    "hand_belief": self.hand_belief(match), "next_belief": self.next_belief(match),
                    "match": match, "hand": hand, "entities": out[:, fixed:],
                    "valid": enc["valid"], "entity_tile": enc["entity_tile"], "state": state}

    def _prev_embedding(self, memory: dict) -> torch.Tensor:
        """(B, d): the seat's previous action, the card it played and how long since it acted."""
        a = memory["prev_action"].long()
        first_ability = self.spec.first_ability
        kind = torch.full_like(a, 4)  # unknown
        kind = torch.where(a == mem.NO_ACTION, torch.zeros_like(a), kind)
        kind = torch.where(a == 0, torch.ones_like(a), kind)
        play = (a > 0) & (a < first_ability)
        kind = torch.where(play, torch.full_like(a, 2), kind)
        kind = torch.where(a >= first_ability, torch.full_like(a, 3), kind)
        tile = torch.where(play, (a - 1) % self.spec.tiles, torch.zeros_like(a))
        xy = torch.stack([(tile % self.spec.cols + 0.5) / self.spec.cols,
                          (torch.div(tile, self.spec.cols, rounding_mode="floor") + 0.5)
                          / self.spec.rows], -1) * play[:, None]
        card = torch.where(play, memory["prev_card"].long().clamp(-1, self.spec.vocab - 1) + 1,
                           torch.zeros_like(a))
        return (self.prev_kind(kind) + self.card(card) + self.prev_tile(xy)
                + self.prev_since(mem.since_bucket(memory["since"].long(), a)))

    def _core(self, match, hand, ability, memory: dict):
        """The recurrent core over lane-major rows, restarting at each ``first``."""
        x = match
        if self.config.prev_action:
            x = torch.cat([match, self._prev_embedding(memory).to(match.dtype)], -1)
        x = self.core_in(x)
        lanes = int(memory["lanes"])
        n, width = x.shape
        steps = n // lanes
        x = x.view(lanes, steps, width)
        first = memory["first"].view(lanes, steps)
        if memory.get("state") is None:
            h = x.new_zeros(1, lanes, width)
            c = x.new_zeros(1, lanes, width)
        else:
            h, c = (s.to(x.dtype) for s in memory["state"])
        # Run in segments, cut wherever some lane restarts, so a restart is always a segment's
        # first step and nn.LSTM runs each segment whole.
        cuts = [0, *(torch.nonzero(first[:, 1:].any(0)).squeeze(1) + 1).tolist(), steps]
        ys = []
        for a, b in itertools.pairwise(cuts):
            keep = (~first[:, a]).to(x.dtype)[None, :, None]
            y, (h, c) = self.core(x[:, a:b], (h * keep, c * keep))
            ys.append(y)
        y = torch.cat(ys, 1).reshape(n, width)
        add = self.core_out(y).view(n, 3, -1)
        match = self.core_norm[0](match + add[:, 0])
        hand = self.core_norm[1](hand + add[:, 1, None])
        ability = self.core_norm[2](ability + add[:, 2, None])
        return match, hand, ability, (h, c)

    def _pairs(self, entities: torch.Tensor, valid: torch.Tensor) -> dict:
        """What every layer's attention bias reads, computed once in fp32: for each pair of units
        (query i, key j), their relative position (``features``, (B, N, N, 8)) and whether i
        targets j (``targets``, (B, N, N) float). A pair with a padded unit gets zeros."""
        e, ix = entities.float(), self.spec.entity
        pair = (valid[:, :, None] & valid[:, None, :]).float()
        out = {}
        if self.config.attn_bias:
            x, y, r = e[..., ix["x"]], e[..., ix["y"]], e[..., ix["radius"]].clamp(min=0)
            dx = x[:, None, :] - x[:, :, None]
            dy = y[:, None, :] - y[:, :, None]
            dist = torch.sqrt(dx * dx + dy * dy)
            lane = x < self.spec.width / 2
            same_lane = (lane[:, :, None] == lane[:, None, :]).float()
            overlap = (dist < r[:, :, None] + r[:, None, :]).float()
            s = 8000.0
            features = torch.stack([dx / s, dy / s, dx.abs() / s, dy.abs() / s, dist / s,
                                    torch.exp(-dist / 3000), same_lane, overlap], -1)
            out["features"] = features * pair[..., None]
        if self.config.relations:
            target, ids = e[..., ix["target"]], e[..., ix["id"]]
            hits = (target[:, :, None] == ids[:, None, :]) & (target >= 0)[:, :, None]
            out["targets"] = hits.float() * pair
        return out

    def _layer_bias(self, layer: int, pairs: dict, fixed: int, pad: torch.Tensor) -> torch.Tensor:
        """Layer ``layer``'s additive attention mask (B, heads, L, L): the units' pair bias, zero
        for the first ``fixed`` tokens, and -inf at padded keys."""
        bias = None
        if "features" in pairs:
            bias = self.pair_mlp[layer](pairs["features"])  # (B, N, N, heads)
        if "targets" in pairs:
            t = pairs["targets"]
            w = self.relation_bias[layer].to(t.dtype)
            rel = t[..., None] * w[0] + t.transpose(1, 2)[..., None] * w[1]
            bias = rel if bias is None else bias + rel
        bias = F.pad(bias.permute(0, 3, 1, 2).float(), (fixed, 0, fixed, 0))
        return bias.masked_fill(pad[:, None, None, :], float("-inf"))

    @contextmanager
    def _head_precision(self, x: torch.Tensor):
        """Yields ``x`` as is, or in fp32 with autocast off (:attr:`fp32_heads`, and always for a
        recurrent model)."""
        if not (self.fp32_heads or self.config.lstm):
            yield x
            return
        with torch.autocast(x.device.type, enabled=False):
            yield x.float()

    def trunk_explicit(self, x: torch.Tensor, pad: torch.Tensor, pairs: dict | None = None,
                       fixed: int = 0) -> torch.Tensor:
        """The trunk's pre-norm layers written out with the same weights: for ONNX export, and
        for the attention biases (``pairs`` from :meth:`_pairs`; the first ``fixed`` tokens are
        not units and get none).

        PyTorch's attention layer records the sequence length as a constant when traced; this
        reshapes with -1 so the exported graph takes any number of units. Equal to ``self.trunk``
        (tests/test_model.py), in training too: each layer's dropouts apply while training.
        """
        b, heads, d = x.shape[0], self.config.heads, self.config.d
        keep = (~pad)[:, None, None, :]  # True where a key may be attended to
        for i, layer in enumerate(self.trunk.layers):
            attn = layer.self_attn
            q, k, v = F.linear(layer.norm1(x), attn.in_proj_weight, attn.in_proj_bias).chunk(3, -1)
            q, k, v = (t.reshape(b, -1, heads, d // heads).transpose(1, 2) for t in (q, k, v))
            p = attn.dropout if self.training else 0.0
            mask = keep if pairs is None else self._layer_bias(i, pairs, fixed, pad).to(q.dtype)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p)
            x = x + layer.dropout1(attn.out_proj(a.transpose(1, 2).reshape(b, -1, d)))
            x = x + layer.dropout2(layer.linear2(layer.dropout(F.gelu(
                layer.linear1(layer.norm2(x))))))
        return x

    def _grid(self, out: dict, rows: torch.Tensor) -> torch.Tensor:
        """The entity outputs of the given rows scattered onto the tile grid, (rows, C, 32, 18)."""
        ents, valid = out["entities"][rows], out["valid"][rows]
        c = self.config.grid_channels
        v = self.grid_in(ents) * valid.unsqueeze(-1).to(ents.dtype)
        grid = torch.zeros(rows.shape[0], self.spec.tiles, c, device=v.device, dtype=v.dtype)
        grid = grid.scatter_add(1, out["entity_tile"][rows].unsqueeze(-1).expand(-1, -1, c), v)
        return grid.view(rows.shape[0], self.spec.rows, self.spec.cols, c).permute(0, 3, 1, 2)

    def _tile_head(self, grid: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(cond).chunk(2, -1)
        grid = (grid + self.grid_position) * (1 + gamma[..., None, None]) + beta[..., None, None]
        return self.grid_out(self.grid(grid)).flatten(1)

    def tile_logits(self, out: dict, rows: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        """(len(rows), tiles) logits for a play of hand slot `slot` (0-based) in each given row."""
        with self._head_precision(out["match"]):
            cond = torch.cat([out["hand"][rows, slot], out["match"][rows]], -1)
            return self._tile_head(self._grid(out, rows), cond)

    def tile_logits_all(self, out: dict) -> torch.Tensor:
        """(B, hand slots, tiles) logits for a play of each hand slot in every row."""
        b, slots = out["hand"].shape[0], self.spec.hand_slots
        with self._head_precision(out["match"]):
            return self._tile_logits_all(out, b, slots)

    def _tile_logits_all(self, out: dict, b: int, slots: int) -> torch.Tensor:
        grid = self._grid(out, torch.arange(b, device=out["hand"].device))
        grid = grid.unsqueeze(1).expand(-1, slots, -1, -1, -1).flatten(0, 1)
        cond = torch.cat([out["hand"], out["match"].unsqueeze(1).expand(-1, slots, -1)], -1)
        return self._tile_head(grid, cond.flatten(0, 1)).view(b, slots, -1)
