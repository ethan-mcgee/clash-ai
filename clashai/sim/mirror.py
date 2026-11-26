"""Left-right mirroring of actions, masks and observations, for doubling imitation data.

The arena is symmetric across its long axis: a play on tile column ``c`` mirrors to column
``cols - 1 - c``, and an x coordinate in game units to ``width - x``. Rows, ids, cards and hit
points are unchanged, except that the header's left and right princess towers (ordered as the seat's
player sees them) swap. Values the seat is not shown (``schema.unknown``, -1) stay unknown.
:meth:`clashai.model.spec.Spec.mirror` is the same rule on batches of training rows.
"""

from __future__ import annotations

import numpy as np

# Header fields ordered left then right as the seat's player sees them.
_HEADER_SWAP = (("own_left_hp", "own_right_hp"), ("own_left_max_hp", "own_right_max_hp"),
                ("enemy_left_hp", "enemy_right_hp"), ("enemy_left_max_hp", "enemy_right_max_hp"))


def mirror_actions(schema, actions) -> np.ndarray:
    """Each play moved to the mirror tile of the same row; wait and abilities unchanged."""
    a = np.asarray(actions)
    out = a.copy()
    plays = (a > schema.wait) & (a < schema.first_ability)
    tiles = schema.cols * schema.rows
    slot, tile = np.divmod(a[plays] - 1, tiles)
    row, col = np.divmod(tile, schema.cols)
    out[plays] = 1 + slot * tiles + row * schema.cols + (schema.cols - 1 - col)
    return out


def mirror_mask(schema, mask: np.ndarray) -> np.ndarray:
    """A mask (or a batch of them) with every play tile moved to its mirror tile."""
    out = np.array(mask, copy=True)
    lead = out.shape[:-1]
    plays = out[..., 1:schema.first_ability].reshape(*lead, schema.hand_slots, schema.rows,
                                                     schema.cols)
    out[..., 1:schema.first_ability] = plays[..., ::-1].reshape(*lead, -1)
    return out


def mirror_observation(schema, observation: np.ndarray) -> np.ndarray:
    """One v4 seat observation reflected left to right.

    Only real points move: an unused pending row's 0, an enemy play's or a spell's unknown -1, and
    the 0 of no target or no aim stay as they are.
    """
    obs = np.array(observation, copy=True)
    width = schema.width_units
    h, e = schema.header, schema.entity
    px, pcard = list(h["pending_x"]), list(h["pending_card"])
    obs[px] = np.where(obs[pcard] >= 0, width - obs[px], obs[px])
    ex = list(h["enemy_play_x"])
    obs[ex] = np.where(obs[ex] == schema.unknown, obs[ex], width - obs[ex])
    for left, right in _HEADER_SWAP:
        a, b = h[left][0], h[right][0]
        obs[a], obs[b] = obs[b], obs[a]
    count = int(obs[h["entity_count"][0]])
    rows = obs[schema.header_ints:].reshape(-1, schema.entity_ints)[:count]
    rows[:, e["x"]] = width - rows[:, e["x"]]
    target = rows[:, e["target"]] >= 0
    rows[:, e["target_x"]] = np.where(target, width - rows[:, e["target_x"]], rows[:, e["target_x"]])
    aim_x, aim_y = rows[:, e["aim_x"]], rows[:, e["aim_y"]]
    aimed = ((aim_x != 0) | (aim_y != 0)) & (aim_x != schema.unknown)
    rows[:, e["aim_x"]] = np.where(aimed, width - aim_x, aim_x)
    rows[:, e["dx"]] = -rows[:, e["dx"]]
    return obs
