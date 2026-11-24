"""Where each field sits in crforge's v4 observation and action, read from labelled metadata.

Kept as a plain dict inside every checkpoint, so a model always knows the layout it was trained on.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import torch

TOWER_SWAP = (("own_left_hp", "own_right_hp"), ("own_left_max_hp", "own_right_max_hp"),
              ("enemy_left_hp", "enemy_right_hp"), ("enemy_left_max_hp", "enemy_right_max_hp"))


@dataclass
class Spec:
    header_fields: list[str]
    entity_fields: list[str]
    vocab: int
    buffs: int
    cols: int
    rows: int
    hand_slots: int
    first_ability: int
    actions: int
    width: int
    vocab_hash: str = ""
    header: dict[str, list[int]] = field(init=False, repr=False)
    entity: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self):
        header = defaultdict(list)
        for i, name in enumerate(self.header_fields):
            header[name].append(i)
        self.header = dict(header)
        self.entity = {n: i for i, n in enumerate(self.entity_fields)}

    @classmethod
    def from_metadata(cls, metadata: dict) -> Spec:
        s, v = metadata["schema"], metadata["vocabulary"]
        return cls(list(s["header_fields"]), list(s["entity_fields"]), len(v["names"]),
                   len(v["buff_names"]), int(s["cols"]), int(s["rows"]), int(s["hand_slots"]),
                   int(s["first_ability"]), int(s["actions"]), int(s["scales"]["width_units"]),
                   str(v.get("hash", "")))

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("header_fields", "entity_fields", "vocab", "buffs",
                                              "cols", "rows", "hand_slots", "first_ability",
                                              "actions", "width", "vocab_hash")}

    @property
    def tiles(self) -> int:
        return self.cols * self.rows

    @property
    def header_ints(self) -> int:
        return len(self.header_fields)

    @property
    def entity_ints(self) -> int:
        return len(self.entity_fields)

    # Actions: 0 wait, 1 + slot * tiles + row * cols + col a play, first_ability + k an ability.
    # A "type" is 0 wait, 1 + slot a play of that slot, 1 + hand_slots + k an ability.

    @property
    def types(self) -> int:
        return 1 + self.hand_slots + (self.actions - self.first_ability)

    def decode(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(type, tile) of each action; tile is 0 for a wait or an ability."""
        play = (actions > 0) & (actions < self.first_ability)
        slot = torch.div(actions - 1, self.tiles, rounding_mode="floor")
        kind = torch.where(play, 1 + slot, torch.zeros_like(actions))
        kind = torch.where(actions >= self.first_ability,
                           1 + self.hand_slots + actions - self.first_ability, kind)
        tile = torch.where(play, (actions - 1) % self.tiles, torch.zeros_like(actions))
        return kind, tile

    def encode(self, kind: torch.Tensor, tile: torch.Tensor) -> torch.Tensor:
        play = (kind >= 1) & (kind <= self.hand_slots)
        action = torch.where(play, 1 + (kind - 1) * self.tiles + tile, torch.zeros_like(kind))
        return torch.where(kind > self.hand_slots, self.first_ability + kind - 1 - self.hand_slots,
                           action)

    def type_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """(B, types) bool: wait, each slot with any open tile, each ability."""
        plays = mask[:, 1:self.first_ability].view(-1, self.hand_slots, self.tiles).any(-1)
        return torch.cat([mask[:, :1], plays, mask[:, self.first_ability:]], 1).bool()

    def tile_mask(self, mask: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        """(B, tiles) bool: the open tiles of each row's slot (0-based)."""
        plays = mask[:, 1:self.first_ability].view(-1, self.hand_slots, self.tiles)
        return plays[torch.arange(len(slot), device=slot.device), slot].bool()

    # Mirroring, batched on the device.

    def mirror(self, header, entities, valid, mask, actions):
        """Each row reflected left to right (see clashai.sim.mirror for the rule); new tensors."""
        header, entities = header.clone(), entities.clone()
        w = self.width
        h, e = self.header, self.entity
        pend_x, pend_card = h["pending_x"], h["pending_card"]
        header[:, pend_x] = torch.where(header[:, pend_card] >= 0, w - header[:, pend_x],
                                        header[:, pend_x])
        play_x = h["enemy_play_x"]
        header[:, play_x] = torch.where(header[:, play_x] != -1, w - header[:, play_x],
                                        header[:, play_x])
        for left, right in TOWER_SWAP:
            a, b = h[left][0], h[right][0]
            header[:, [a, b]] = header[:, [b, a]]
        x, tx, ax, ay = e["x"], e["target_x"], e["aim_x"], e["aim_y"]
        entities[..., x] = torch.where(valid, w - entities[..., x], entities[..., x])
        has_target = valid & (entities[..., e["target"]] >= 0)
        entities[..., tx] = torch.where(has_target, w - entities[..., tx], entities[..., tx])
        aimed = valid & ((entities[..., ax] != 0) | (entities[..., ay] != 0)) & (entities[..., ax] != -1)
        entities[..., ax] = torch.where(aimed, w - entities[..., ax], entities[..., ax])
        entities[..., e["dx"]] = torch.where(valid, -entities[..., e["dx"]], entities[..., e["dx"]])
        plays = mask[:, 1:self.first_ability].view(-1, self.hand_slots, self.rows, self.cols)
        mask = torch.cat([mask[:, :1], plays.flip(-1).reshape(len(mask), -1),
                          mask[:, self.first_ability:]], 1)
        kind, tile = self.decode(actions)
        row, col = torch.div(tile, self.cols, rounding_mode="floor"), tile % self.cols
        actions = self.encode(kind, row * self.cols + (self.cols - 1 - col))
        return header, entities, mask, actions


def unpack_masks(packed: torch.Tensor, actions: int) -> torch.Tensor:
    """np.packbits(mask, axis=1) rows back to (B, actions) bool, on the device."""
    shifts = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    bits = (packed.unsqueeze(-1) >> shifts) & 1
    return bits.flatten(1)[:, :actions].bool()

