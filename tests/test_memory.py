"""The recurrent core and the seat's own history: chunked and stepwise runs agree, restarts cut
the state, and training and play compute the same history."""

import numpy as np
import pytest
import torch

from clashai import paths
from clashai.model import memory as mem
from clashai.model.net import Config, PolicyNet
from clashai.model.spec import Spec

ROWS = paths.data_root() / "rows" / "piggies_eq-screen-team0-v0-t0-standard-w0.25"
needs_rows = pytest.mark.skipif(not (ROWS / "rows.json").exists(),
                                reason="training rows not built (python -m clashai.train.rows)")


@pytest.fixture(scope="module")
def data():
    from clashai.train.bc import to_device
    from clashai.train.rows import Rows

    rows = Rows(ROWS)
    spec = Spec.from_metadata(rows.metadata)
    idx = np.sort(np.random.default_rng(0).choice(rows.n, 64, replace=False))
    return spec, to_device(rows.gather(idx, 80), "cpu")


def _net(spec, **flags):
    torch.manual_seed(0)
    flags = {"lstm": 32, "prev_action": True, **flags}
    return PolicyNet(spec, Config(d=64, layers=2, heads=4, grid_channels=16, **flags)).eval()


def _memory(n, lanes, first=None, state=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    prev = torch.randint(0, 2307, (n,), generator=g)
    prev[torch.rand(n, generator=g) < 0.6] = 0  # mostly waits, as in play
    prev[::7] = mem.NO_ACTION
    prev[3::11] = mem.UNKNOWN
    return {"lanes": lanes, "first": torch.zeros(n, dtype=torch.bool) if first is None else first,
            "state": state, "prev_action": prev,
            "prev_card": torch.randint(-1, 50, (n,), generator=g),
            "since": torch.randint(1, 300, (n,), generator=g)}


def _rows(b, idx):
    return b["header"][idx], b["entities"][idx], b["valid"][idx]


def _sub(m, idx, lanes, state):
    return {**{k: m[k][idx] for k in ("first", "prev_action", "prev_card", "since")},
            "lanes": lanes, "state": state}


@needs_rows
def test_one_pass_equals_two_chunks_equals_single_steps(data):
    spec, b = data
    net = _net(spec)
    n = 64
    m = _memory(n, 1)
    idx = torch.arange(n)
    with torch.no_grad():
        whole = net(*_rows(b, idx), m)
        a = net(*_rows(b, idx[:32]), _sub(m, idx[:32], 1, None))
        c = net(*_rows(b, idx[32:]), _sub(m, idx[32:], 1, a["state"]))
        state, steps = None, []
        for t in range(n):
            one = net(*_rows(b, idx[t:t + 1]), _sub(m, idx[t:t + 1], 1, state))
            state = one["state"]
            steps.append(one["type_logits"])
    torch.testing.assert_close(torch.cat([a["type_logits"], c["type_logits"]]),
                               whole["type_logits"], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(torch.cat(steps), whole["type_logits"], rtol=1e-5, atol=1e-5)
    for x, y in zip(state, whole["state"], strict=True):
        torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-5)


@needs_rows
def test_a_restart_inside_a_chunk_equals_starting_that_lane_afresh(data):
    spec, b = data
    net = _net(spec)
    lanes, steps = 2, 16
    first = torch.zeros(lanes * steps, dtype=torch.bool)
    first[7] = True  # lane 0 restarts at its step 7
    m = _memory(lanes * steps, lanes, first)
    idx = torch.arange(lanes * steps)
    with torch.no_grad():
        both = net(*_rows(b, idx), m)["type_logits"]
        head = net(*_rows(b, idx[:7]), _sub(m, idx[:7], 1, None))["type_logits"]
        tail = net(*_rows(b, idx[7:16]), _sub(m, idx[7:16], 1, None))["type_logits"]
        other = net(*_rows(b, idx[16:]), _sub(m, idx[16:], 1, None))["type_logits"]
    torch.testing.assert_close(both, torch.cat([head, tail, other]), rtol=1e-5, atol=1e-5)


@needs_rows
def test_the_core_and_the_previous_action_learn_through_the_policy_loss(data):
    from clashai.model.spec import unpack_masks
    from clashai.train.bc import losses

    spec, b = data
    net = _net(spec).train()
    out = net(b["header"], b["entities"], b["valid"], _memory(64, 4))
    mask = unpack_masks(b["masks"], spec.actions)
    loss, _ = losses(net, out, b, mask, b["actions"].long(), b["header"][:, 0],
                     policy_rows="all", loss_weight=0.5)
    loss.backward()
    assert out["type_logits"].dtype == torch.float32
    for name in ("core.weight_ih_l0", "core_out.weight", "prev_kind.weight", "prev_since.weight",
                 "prev_tile.weight"):
        assert dict(net.named_parameters())[name].grad.abs().sum() > 0, name


@needs_rows
def test_a_recurrent_model_needs_memory_and_others_ignore_it(data):
    spec, b = data
    with pytest.raises(ValueError):
        _net(spec)(b["header"], b["entities"], b["valid"])
    with pytest.raises(ValueError):
        PolicyNet(spec, Config(d=64, layers=2, heads=4, grid_channels=16, prev_action=True))
    plain = _net(spec, lstm=0, prev_action=False)
    with torch.no_grad():
        a = plain(b["header"], b["entities"], b["valid"])["type_logits"]
        c = plain(b["header"], b["entities"], b["valid"], _memory(64, 4))["type_logits"]
    torch.testing.assert_close(a, c)
    again = PolicyNet.from_description(_net(spec).describe())
    assert again.config.lstm == 32 and again.config.prev_action
    h = again.config.lstm
    assert (again.core.bias_ih_l0[h:2 * h] == 1).all()


def test_the_whole_history_equals_advancing_one_decision_at_a_time():
    rng = np.random.default_rng(0)
    actions = rng.integers(0, 2307, 200)
    actions[rng.random(200) < 0.8] = 0
    cards = np.where((actions > 0) & (actions < 2305), rng.integers(0, 100, 200), -1)
    prev, prev_card, since = mem.history(actions, cards)
    h = mem.start(1)
    for t in range(200):
        assert h["prev_action"].item() == prev[t] and h["prev_card"].item() == prev_card[t]
        assert h["since"].item() == since[t]
        h = mem.advance(h, torch.tensor([actions[t]]), torch.tensor([cards[t]]))
    assert since[0] == mem.NEVER and prev[0] == mem.NO_ACTION


def test_played_cards_read_the_hand_slot_of_each_play_in_numpy_and_torch():
    tiles, first_ability, hand_card = 576, 2305, [7, 8, 9, 10]
    header = np.zeros((4, 12), np.int32)
    header[:, 7:11] = [[11, 12, 13, 14]] * 4
    actions = np.array([0, 1 + 2 * tiles + 5, first_ability, 1 + 0 * tiles])
    want = [-1, 13, -1, 11]
    assert played_list(header, actions, hand_card, first_ability, tiles) == want
    got = mem.played_cards(torch.from_numpy(header), torch.from_numpy(actions), hand_card,
                           first_ability, tiles)
    assert got.tolist() == want


def played_list(header, actions, hand_card, first_ability, tiles):
    return mem.played_cards(header, actions, hand_card, first_ability, tiles).tolist()


def test_since_buckets_are_log_spaced_and_unknown_has_its_own():
    since = torch.tensor([1, 2, 4, 5, 6, 7, 128, 129, mem.NEVER, 3])
    prev = torch.tensor([0] * 9 + [mem.UNKNOWN])
    assert mem.since_bucket(since, prev).tolist() == [0, 1, 3, 4, 4, 5, 13, 14, 14,
                                                      mem.SINCE_BUCKETS - 1]
