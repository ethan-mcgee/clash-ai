"""The engine behaves as the game does where training depends on it."""

import numpy as np
import pytest

from clashai.decks import PIGGIES_EQ, REFERENCE_DECKS
from clashai.sim.env import make_vector
from clashai.sim.inputs import critic_inputs, policy_inputs
from clashai.sim.policies import RandomLegal

pytestmark = pytest.mark.engine

DECK = list(PIGGIES_EQ.cards)
HOG = REFERENCE_DECKS["hog_26"]


def _reset(v, seeds, deck1=HOG):
    s = v.schema
    v.reset(list(seeds), DECK, list(deck1.cards), PIGGIES_EQ.slots(s.slot_evolution),
            deck1.slots(s.slot_evolution))


def test_a_play_lands_20_ticks_after_it_is_sent_and_the_opponent_sees_it_then():
    """RoyaleFish trained with plays landing after 1 tick and lost half its win rate when the
    real 20-tick delay was restored. This pins the delay."""
    with make_vector(1, 1, decision_ticks=1) as v:
        s, h, e = v.schema, v.schema.header, v.schema.entity
        assert s.play_delay_ticks == 20
        v.reset([7], DECK, DECK, PIGGIES_EQ.slots(s.slot_evolution),
                PIGGIES_EQ.slots(s.slot_evolution))
        troops = {v.card_id(c) for c in ("Skeletons", "IceSpirits", "RoyalHogs", "ArcherQueen")}
        tiles = s.cols * s.rows

        def own_units():
            rows = s.entities(v.observations[0, 0])
            return int(((rows[:, e["side"]] == 0) & (rows[:, e["kind"]] == 1)).sum())

        def seen_by_opponent():
            return int(v.observations[0, 1][h["enemy_play_count"][0]])

        sent = None
        for _ in range(3000):
            o = v.observations[0, 0]
            action = s.wait
            for k, card in enumerate(int(o[i]) for i in h["hand_card"]):
                open_tiles = np.flatnonzero(v.masks[0, 0, 1 + k * tiles:1 + (k + 1) * tiles])
                if card in troops and len(open_tiles):
                    action = 1 + k * tiles + int(open_tiles[len(open_tiles) // 2])
                    break
            if action != s.wait:
                sent, units, seen = int(o[h["tick"][0]]), own_units(), seen_by_opponent()
                v.actions[0] = [action, s.wait]
                v.step()
                break
            v.actions[0] = [s.wait, s.wait]
            v.step()
        assert sent is not None, "no troop was ever playable"

        for waited in range(1, 22):
            o = v.observations[0, 0]
            assert int(o[h["tick"][0]]) == sent + waited
            if waited <= 20:
                assert int(o[h["pending_count"][0]]) == 1
                assert int(o[h["pending_ticks"][0]]) == 20 - waited
                assert own_units() == units and seen_by_opponent() == seen
            else:  # ran on tick sent + 20, first shown in the observation after it
                assert int(o[h["pending_count"][0]]) == 0
                assert own_units() > units and seen_by_opponent() == seen + 1
            v.actions[0] = [s.wait, s.wait]
            v.step()


def test_the_same_seeds_and_actions_give_the_same_battles_on_any_thread_count():
    with make_vector(4, 1) as one, make_vector(4, 4) as four:
        for v in (one, four):
            _reset(v, [11, 12, 13, 14])
        policies = [RandomLegal(0.3), RandomLegal(0.3)]
        for p, v in zip(policies, (one, four), strict=True):
            p.bind(v)
        rngs = [np.random.default_rng(5), np.random.default_rng(5)]
        for _ in range(800):
            np.testing.assert_array_equal(one.observations, four.observations)
            np.testing.assert_array_equal(one.masks, four.masks)
            for p, v, rng in zip(policies, (one, four), rngs, strict=True):
                v.actions[:] = p.act(v.observations.reshape(8, -1), v.masks.reshape(8, -1),
                                     rng).reshape(4, 2)
                v.step()


def test_actions_the_mask_allows_are_never_refused():
    with make_vector(16) as v:
        s = v.schema
        _reset(v, range(16))
        policy = RandomLegal(0.5)
        policy.bind(v)
        rng = np.random.default_rng(0)
        illegal = 0
        for _ in range(2000):
            v.actions[:] = policy.act(v.observations.reshape(32, -1), v.masks.reshape(32, -1),
                                      rng).reshape(16, 2)
            v.step()
            illegal += int(v.infos[:, [s.info["illegal_0"], s.info["illegal_1"]]].sum())
            assert not v.infos[:, s.info["not_modelled"]].any()
        assert illegal == 0
        assert v.fault_kind_counts() == {"refusal": 0, "bug": 0, "env_bug": 0}


def test_the_policy_reads_nothing_a_player_cannot_see():
    from crforge_env.adapters import AdapterSet, HiddenFieldError, OtherSeat, V4Adapter
    from crforge_env.fields import Registry

    with make_vector(4, 2, fields=True) as v:
        registry = Registry.from_metadata(v.metadata["fields"])
        inputs = policy_inputs()
        inputs.bind(registry)  # refuses any adapter reading a hidden field
        assert not inputs.hidden
        critic = critic_inputs()
        critic.bind(registry)
        assert critic.hidden  # the critic's extra view is hidden, and the guard knows it
        with pytest.raises(HiddenFieldError):
            AdapterSet([OtherSeat(V4Adapter())], forbid_hidden=True).bind(registry)

        # The policy's v4 input is the observation the engine shows the seat, bit for bit.
        _reset(v, [1, 2, 3, 4])
        policy = RandomLegal(0.3)
        policy.bind(v)
        rng = np.random.default_rng(1)
        view = inputs.view(v.fields)
        for _ in range(300):
            for seat in (0, 1):
                out = inputs.observe(view, seat)
                np.testing.assert_array_equal(out["v4.observation"], v.observations[:, seat])
            done = v.dones.astype(bool)
            v.actions[:] = policy.act(v.observations.reshape(8, -1), v.masks.reshape(8, -1),
                                      rng).reshape(4, 2)
            v.step()
            for seat in (0, 1):
                inputs.reset(done, seat)
