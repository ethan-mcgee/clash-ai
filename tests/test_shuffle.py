"""The opening deal: the Python port against the engine, and the cycle solver."""

import json

import numpy as np
import pytest

from clashai import paths
from clashai.data.cycle import all_keys, feasible_keys, fits, key_of
from clashai.data.shuffle import deal_order, match_orders

PIGGIES = ["RoyalHogs", "ArcherQueen", "Cannon", "Skeletons", "IceSpirits", "Log", "Earthquake",
           "RoyalDelivery"]
HELD = ["Mirror", "Elixir Collector", "Golem", "DarkWitch", "BabyDragon", "Lightning", "Zap",
        "Tornado"]


def test_a_deal_is_a_permutation_and_depends_on_the_seed():
    orders = {tuple(deal_order(PIGGIES, seed)) for seed in range(50)}
    assert all(sorted(o) == list(range(8)) for o in orders)
    assert len(orders) > 40


def test_mirror_and_elixir_collector_are_never_in_the_opening_hand():
    for seed in range(500):
        hand = {HELD[i] for i in deal_order(HELD, seed)[:4]}
        assert not hand & {"Mirror", "Elixir Collector"}


def test_there_are_1680_orders_up_to_the_hands_slot_order():
    assert len(set(all_keys())) == 1680


def test_a_play_must_be_in_hand_and_cycles_to_the_back():
    key = key_of([0, 1, 2, 3, 4, 5, 6, 7])  # hand 0-3, queue 4,5,6,7
    assert fits(key, [0, 1, 4])  # 4 came into the hand after the first play
    assert not fits(key, [4])
    assert fits(key, [0, 1, 2, 3, 4, 5, 6, 7, 0])  # 0 is back after four more plays


def test_every_fitting_order_explains_the_plays_and_most_plays_narrow_it_down():
    plays = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1]
    keys = feasible_keys(plays)
    assert key_of([0, 1, 2, 3, 4, 5, 6, 7]) in keys
    assert all(fits(k, plays) for k in keys)
    assert len(keys) < len(feasible_keys(plays[:3]))


@pytest.mark.engine
def test_the_ported_deal_is_the_engines_for_any_seed_and_account_words():
    from crforge_env import LadderConfig, VectorBattle

    paths.ensure_java()
    with VectorBattle(1, 1, install=paths.install_dir(), tables=paths.tables_dir(),
                      config=LadderConfig(decision_ticks=1, arena="Arena_L")) as v:
        h = v.schema.header
        v.reset([1], PIGGIES, HELD, [1, 0, 1, 0, 0, 0, 0, 0], [0] * 8)
        v.step()
        template = json.loads(v.replay_json(0))
        rng = np.random.default_rng(0)
        for _ in range(40):
            seed, w0, w1 = (int(x) for x in rng.integers(-2**31, 2**31, 3))
            doc = json.loads(json.dumps(template))
            doc["rndSeed"] = seed
            doc["battle"]["avatar0"]["accountID.lo"] = w0
            doc["battle"]["avatar1"]["accountID.lo"] = w1
            v.start_from_replay(0, doc, 0)
            for seat, (deck, order) in enumerate(zip((PIGGIES, HELD),
                                                     match_orders(seed, [PIGGIES, HELD], [w0, w1]),
                                                     strict=True)):
                o = v.observations[0, seat]
                dealt = [v.vocabulary[int(o[i])] for i in (*h["hand_card"], *h["queue"])]
                assert dealt == [deck[k] for k in order]
