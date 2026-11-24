"""Baseline policies, batched: given many seats' observations and masks, one action each.

A policy has ``name``, ``bind(vector)`` (called once with the vector it plays in) and
``act(observations, masks, rng)``, which takes ``(B, seat_ints)`` and ``(B, actions)`` arrays and
returns ``(B,)`` actions. Every action returned must be open in its mask.
"""

from __future__ import annotations

import numpy as np


class WaitPolicy:
    """Never plays: the weakest possible opponent, for sanity checks."""

    name = "wait"

    def bind(self, vector) -> None:
        self.wait = vector.schema.wait

    def act(self, observations, masks, rng) -> np.ndarray:
        return np.full(len(masks), self.wait, np.int32)


class RandomLegal:
    """Plays a random legal move at random moments.

    At each decision it acts with probability ``act_prob``. It then picks uniformly among the open
    hand slots and ability buttons, and for a card a uniform open tile. Picking the slot first
    gives every card an even chance: picking uniformly over all open actions would favour spells,
    which have far more legal tiles than troops.
    """

    name = "random"

    def __init__(self, act_prob: float = 0.1):
        self.act_prob = act_prob

    def bind(self, vector) -> None:
        s = vector.schema
        self.wait, self.first_ability = s.wait, s.first_ability
        self.slots, self.tiles = s.hand_slots, s.cols * s.rows

    def act(self, observations, masks, rng) -> np.ndarray:
        b = len(masks)
        out = np.full(b, self.wait, np.int32)
        plays = masks[:, 1:self.first_ability].reshape(b, self.slots, self.tiles)
        slot_open = plays.any(axis=2)
        ability_open = masks[:, self.first_ability:] != 0
        for i in np.flatnonzero(rng.random(b) < self.act_prob):
            choices = [("slot", int(k)) for k in np.flatnonzero(slot_open[i])]
            choices += [("ability", int(j)) for j in np.flatnonzero(ability_open[i])]
            if not choices:
                continue
            kind, k = choices[rng.integers(len(choices))]
            if kind == "ability":
                out[i] = self.first_ability + k
            else:
                tiles = np.flatnonzero(plays[i, k])
                out[i] = 1 + k * self.tiles + tiles[rng.integers(len(tiles))]
        return out


class ScriptedPolicy:
    """crforge's ``ScriptedBot``: a few readable rules on what its player sees."""

    name = "scripted"

    def __init__(self, **bot_options):
        self.bot_options = bot_options

    def bind(self, vector) -> None:
        from crforge_env import ScriptedBot

        self.bot = ScriptedBot(**self.bot_options).bind(vector)

    def act(self, observations, masks, rng) -> np.ndarray:
        return np.fromiter(
            (self.bot(o, m, rng) for o, m in zip(observations, masks, strict=True)),
            np.int32, len(masks))


BASELINES = {"wait": WaitPolicy, "random": RandomLegal, "scripted": ScriptedPolicy}
