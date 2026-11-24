from types import SimpleNamespace

import numpy as np
import pytest

from clashai.decks import PIGGIES_EQ
from clashai.sim.env import make_vector
from clashai.sim.mirror import mirror_actions, mirror_mask, mirror_observation
from clashai.sim.policies import RandomLegal

# The v4 layout's action numbering, enough for the pure tests.
SCHEMA = SimpleNamespace(wait=0, cols=18, rows=32, hand_slots=4, first_ability=2305, actions=2307)


def _play(slot, col, row):
    return 1 + slot * 18 * 32 + row * 18 + col


def test_a_play_mirrors_to_the_same_row_on_the_other_side():
    assert mirror_actions(SCHEMA, [_play(2, 0, 5)])[0] == _play(2, 17, 5)
    assert mirror_actions(SCHEMA, [_play(0, 8, 9)])[0] == _play(0, 9, 9)


def test_wait_and_abilities_are_unchanged():
    np.testing.assert_array_equal(mirror_actions(SCHEMA, [0, 2305, 2306]), [0, 2305, 2306])


def test_mirroring_actions_and_masks_twice_changes_nothing():
    every = np.arange(SCHEMA.actions)
    np.testing.assert_array_equal(mirror_actions(SCHEMA, mirror_actions(SCHEMA, every)), every)
    mask = np.random.default_rng(0).integers(0, 2, (3, SCHEMA.actions), dtype=np.uint8)
    np.testing.assert_array_equal(mirror_mask(SCHEMA, mirror_mask(SCHEMA, mask)), mask)


def test_a_mirrored_action_is_open_in_the_mirrored_mask():
    mask = np.zeros(SCHEMA.actions, np.uint8)
    mask[_play(1, 3, 12)] = 1
    assert mirror_mask(SCHEMA, mask)[mirror_actions(SCHEMA, [_play(1, 3, 12)])[0]] == 1


@pytest.mark.engine
def test_mirroring_a_real_observation_twice_changes_nothing_and_moves_units():
    with make_vector(4) as v:
        s = v.schema
        slots = PIGGIES_EQ.slots(s.slot_evolution)
        v.reset([1, 2, 3, 4], list(PIGGIES_EQ.cards), list(PIGGIES_EQ.cards), slots, slots)
        policy = RandomLegal(0.3)
        policy.bind(v)
        rng = np.random.default_rng(0)
        for _ in range(400):
            v.actions[:] = policy.act(v.observations.reshape(8, -1), v.masks.reshape(8, -1),
                                      rng).reshape(4, 2)
            v.step()
        for i in range(4):
            for seat in (0, 1):
                o = v.observations[i, seat]
                m = mirror_observation(s, o)
                np.testing.assert_array_equal(mirror_observation(s, m), o)
                rows = s.entities(o)
                assert len(rows) > 0
                x = s.entity["x"]
                np.testing.assert_array_equal(s.entities(m)[:, x], s.width_units - rows[:, x])


@pytest.mark.engine
def test_the_mask_of_an_empty_arena_is_symmetric():
    """Mirroring is only sound if the game's legal tiles are left-right symmetric."""
    with make_vector(1, 1) as v:
        s = v.schema
        slots = PIGGIES_EQ.slots(s.slot_evolution)
        v.reset([3], list(PIGGIES_EQ.cards), list(PIGGIES_EQ.cards), slots, slots)
        while not v.masks[0, 0, 1:s.first_ability].any():  # the start cooldown
            v.step()
        for seat in (0, 1):
            mask = v.masks[0, seat]
            np.testing.assert_array_equal(mirror_mask(s, mask), mask)
