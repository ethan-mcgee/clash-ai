"""PPO's pieces: returns, shaping, action log-probabilities, the league, and crforge's set_decks."""

import numpy as np
import pytest
import torch

from clashai import paths
from clashai.decks import PIGGIES_EQ, REFERENCE_DECKS, is_held_out
from clashai.train.league import League, PoolDeck
from clashai.train.ppo import critic, evaluate_actions, gae, sample, tower_potential

ROWS = paths.data_root() / "rows" / "piggies_eq-screen-team0-v0-t0-standard-w0.25"
needs_rows = pytest.mark.skipif(not (ROWS / "rows.json").exists(), reason="training rows not built")


def test_gae_matches_a_hand_computation_and_cuts_at_episode_ends():
    gamma, lam = 0.9, 0.5
    # One battle: two steps of an episode that ends, the restart step (not live), then one step.
    rewards = np.array([[0.0], [1.0], [0.0], [0.5]], np.float32)
    values = np.array([[0.2], [0.4], [9.0], [0.1]], np.float32)
    dones = np.array([[0], [1], [0], [0]], np.float32)
    live = np.array([[True], [True], [False], [True]])
    adv, ret = gae(rewards, values, dones, live, np.array([0.3]), gamma, lam)
    d3 = 0.5 + gamma * 0.3 - 0.1
    d1 = 1.0 - 0.4  # ends: no bootstrap
    d0 = 0.0 + gamma * 0.4 - 0.2
    np.testing.assert_allclose(adv[:, 0], [d0 + gamma * lam * d1, d1, 0.0, d3], rtol=1e-6)
    np.testing.assert_allclose(ret, adv + values)


def test_the_kl_weight_rises_above_the_target_falls_below_it_and_holds_between():
    from clashai.train.ppo import adapt_kl

    assert adapt_kl(0.1, 0.03, 0.01) == pytest.approx(0.15)
    assert adapt_kl(0.1, 0.004, 0.01) == pytest.approx(0.1 / 1.5)
    assert adapt_kl(0.1, 0.012, 0.01) == 0.1
    assert adapt_kl(0.1, 0.5, 0.0) == 0.1  # no target: fixed
    assert adapt_kl(9.0, 1.0, 0.01) == 10.0 and adapt_kl(0.001, 0.0, 0.01) == 0.001


def test_game_outcomes_reach_back_to_each_games_rows_and_leave_unfinished_games_unknown():
    from clashai.train.ppo import game_outcomes

    # Battle 0: a game lost at step 1, the restart at 2, then a game still going.
    # Battle 1: a game won at step 3.
    dones = np.array([[0, 0], [1, 0], [0, 0], [0, 1]], np.float32)
    results = np.array([[0, 0], [-1, 0], [0, 0], [0, 1]], np.float32)
    out = game_outcomes(dones, results)
    np.testing.assert_array_equal(out[:2, 0], [-1, -1])
    assert np.isnan(out[2:, 0]).all()
    np.testing.assert_array_equal(out[:, 1], [1, 1, 1, 1])


@pytest.fixture(scope="module")
def spec_and_batch():
    from clashai.model.spec import Spec
    from clashai.train.rows import Rows

    rows = Rows(ROWS)
    spec = Spec.from_metadata(rows.metadata)
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(rows.n, 64, replace=False))
    return spec, rows.gather(idx, 80)


@needs_rows
def test_the_tower_potential_is_zero_when_even_and_signed_by_who_is_ahead(spec_and_batch):
    spec, _ = spec_and_batch
    h = np.zeros((3, spec.header_ints), np.int32)
    for side in ("own", "enemy"):
        for t in ("king", "left", "right"):
            h[:, spec.header[f"{side}_{t}_max_hp"][0]] = 1000
            h[:, spec.header[f"{side}_{t}_hp"][0]] = 1000
    h[1, spec.header["enemy_left_hp"][0]] = 0      # we took a princess tower
    h[2, spec.header["own_right_hp"][0]] = 400     # we are behind
    phi = tower_potential(h, spec, 0.5)
    np.testing.assert_allclose(phi, [0.0, 0.5 / 3, -0.5 * 600 / 3000], rtol=1e-6)


@needs_rows
def test_sampled_log_probabilities_equal_the_evaluated_ones_and_actions_are_legal(spec_and_batch):
    from clashai.model.net import Config, PolicyNet
    from clashai.model.spec import unpack_masks

    spec, b = spec_and_batch
    torch.manual_seed(0)
    net = PolicyNet(spec, Config(d=64, layers=2, heads=4, grid_channels=16)).eval()
    t = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in b.items()}
    mask = unpack_masks(t["masks"], spec.actions)
    with torch.no_grad():
        out = net(t["header"], t["entities"], t["valid"])
        actions, logp = sample(net, out, mask, torch.Generator().manual_seed(1))
        again, _ = evaluate_actions(net, out, mask, actions)
        value = critic(net, out, 0.1)
    assert mask[torch.arange(len(actions)), actions].all()
    torch.testing.assert_close(again, logp, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(value, out["value"].float(), rtol=1e-5, atol=1e-5)


def _deck(name, met=1):
    d = REFERENCE_DECKS[name]
    return PoolDeck(d.cards, tuple(d.slots(1)), met)


def test_the_league_draws_more_of_what_it_loses_to_and_of_its_neighbours():
    decks = [_deck("hog_26", 10), _deck("xbow", 10), _deck("golem", 10)]
    league = League(decks, uniform=0.5, share=6)
    np.testing.assert_allclose(league.probabilities(), [1 / 3] * 3)
    for _ in range(20):
        league.record(2, 0.0)  # loses to Golem
        league.record(0, 1.0)  # beats Hog 2.6
    p = league.probabilities()
    assert p[2] > p[1] > p[0] and abs(p.sum() - 1) < 1e-9
    again = League(decks, uniform=0.5)
    again.load_state_dict(league.state_dict())
    np.testing.assert_allclose(again.probabilities(), p)
    # Hog 2.6 and X-Bow share 6 of 8 cards in name only if they do: neighbours pool results.
    shared = len(set(decks[0].cards) & set(decks[1].cards))
    assert league.near[0, 1] == (shared >= 6)


@needs_rows
def test_the_opponent_pool_holds_no_held_out_deck():
    from clashai.train.league import opponent_decks

    decks, held = opponent_decks()
    assert decks and held > 0
    assert not any(is_held_out(d.cards) for d in decks)


@pytest.mark.engine
def test_set_decks_changes_a_battles_next_game_only():
    from clashai.sim.env import make_vector

    hog, golem = REFERENCE_DECKS["hog_26"], REFERENCE_DECKS["golem"]
    with make_vector(2, 2) as v:
        s = v.schema
        names = list(v.vocabulary)
        deck = s.header["deck"]

        def seat1_deck(i):
            return sorted(names[int(x)] for x in v.observations[i, 1, deck])

        slots = PIGGIES_EQ.slots(s.slot_evolution)
        v.reset([1, 2], list(PIGGIES_EQ.cards), list(hog.cards), slots, hog.slots(s.slot_evolution))
        v.set_decks(0, list(PIGGIES_EQ.cards), list(golem.cards), slots,
                    golem.slots(s.slot_evolution))
        assert seat1_deck(0) == sorted(hog.cards)  # the game in play keeps its decks
        while not v.dones[0]:
            v.actions[:] = 0
            v.step()
        v.actions[:] = 0
        v.step()  # the restart
        assert v.infos[0, s.info["reset"]] == 1
        assert seat1_deck(0) == sorted(golem.cards) and seat1_deck(1) == sorted(hog.cards)
        with pytest.raises(ValueError):
            v.set_decks(0, ["Knight"] * 8, list(golem.cards))
