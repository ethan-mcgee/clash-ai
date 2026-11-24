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

Every amount is scaled here from crforge's integers, so the model takes the raw observation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

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


def _onehot(x: torch.Tensor, n: int) -> torch.Tensor:
    return F.one_hot(x.long().clamp(0, n - 1), n).float()


def _bits(x: torch.Tensor, n: int) -> torch.Tensor:
    return ((x.long().unsqueeze(-1) >> torch.arange(n, device=x.device)) & 1).float()


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
        layer = nn.TransformerEncoderLayer(d, self.config.heads, 4 * d, dropout=0.0,
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
            (slot_flags & 1).float(), ((slot_flags >> 1) & 1).float(),
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
        pad = torch.cat([torch.zeros(len(header), fixed, dtype=torch.bool, device=header.device),
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

    def forward(self, header: torch.Tensor, entities: torch.Tensor, valid: torch.Tensor) -> dict:
        """header (B, header_ints) int, entities (B, N, entity_ints) int, valid (B, N) bool."""
        tokens, pad, fixed = self._tokens(header, entities, valid)
        out = self.norm_out(self.trunk(tokens, src_key_padding_mask=pad))
        match, hand, ability = out[:, 0], out[:, 1:5], out[:, fixed - 2:fixed]
        type_logits = torch.cat([self.wait_head(match), self.slot_head(hand).squeeze(-1),
                                 self.ability_head(ability).squeeze(-1)], 1)
        ex = entities[..., self.spec.entity["x"]].long()
        ey = entities[..., self.spec.entity["y"]].long()
        tile = ((ey // 1000).clamp(0, self.spec.rows - 1) * self.spec.cols
                + (ex // 1000).clamp(0, self.spec.cols - 1))
        return {"type_logits": type_logits, "value": torch.tanh(self.value_head(match).squeeze(-1)),
                "hand_belief": self.hand_belief(match), "next_belief": self.next_belief(match),
                "match": match, "hand": hand, "entities": out[:, fixed:], "valid": valid,
                "entity_tile": tile}

    def tile_logits(self, out: dict, rows: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        """(len(rows), tiles) logits for a play of hand slot `slot` (0-based) in each given row."""
        ents, valid = out["entities"][rows], out["valid"][rows]
        c = self.config.grid_channels
        v = self.grid_in(ents) * valid.unsqueeze(-1).to(ents.dtype)
        grid = torch.zeros(len(rows), self.spec.tiles, c, device=v.device, dtype=v.dtype)
        grid.scatter_add_(1, out["entity_tile"][rows].unsqueeze(-1).expand(-1, -1, c), v)
        grid = grid.view(len(rows), self.spec.rows, self.spec.cols, c).permute(0, 3, 1, 2)
        cond = torch.cat([out["hand"][rows, slot], out["match"][rows]], -1)
        gamma, beta = self.film(cond).chunk(2, -1)
        grid = (grid + self.grid_position) * (1 + gamma[..., None, None]) + beta[..., None, None]
        return self.grid_out(self.grid(grid)).flatten(1)
