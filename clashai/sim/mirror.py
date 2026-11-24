"""Left-right mirroring of actions, masks and observations, for doubling imitation data.

The arena is symmetric across its long axis: a play on tile column ``c`` mirrors to column
``cols - 1 - c``, and an x coordinate in game units to ``width - x``. Rows, ids, cards and hit
points are unchanged. Values the seat is not shown (``schema.unknown``, -1) stay unknown.
"""

from __future__ import annotations

import numpy as np

# Header fields holding x coordinates in the seat's frame, in game units.
_HEADER_X = ("pending_x", "enemy_play_x")
# Entity columns holding x coordinates; the x velocity is negated instead.
_ENTITY_X = ("x", "target_x", "aim_x")
_ENTITY_DX = ("dx",)


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
    """One v4 seat observation reflected left to right."""
    obs = np.array(observation, copy=True)
    width = schema.width_units
    for name in _HEADER_X:
        idx = list(schema.header[name])
        values = obs[idx]
        obs[idx] = np.where(values == schema.unknown, values, width - values)
    count = int(obs[schema.header["entity_count"][0]])
    rows = obs[schema.header_ints:].reshape(schema.max_entities, schema.entity_ints)[:count]
    for name in _ENTITY_X:
        col = rows[:, schema.entity[name]]
        rows[:, schema.entity[name]] = np.where(col == schema.unknown, col, width - col)
    for name in _ENTITY_DX:
        rows[:, schema.entity[name]] = -rows[:, schema.entity[name]]
    return obs
