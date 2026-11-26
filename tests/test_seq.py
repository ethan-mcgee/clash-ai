"""The whole-battle trainer: its data stream resumes exactly, grouping rows by unit count changes
nothing, and validation runs over whole seats."""

import numpy as np
import pytest
import torch

from clashai import paths
from clashai.model import memory as mem
from clashai.model.net import Config, PolicyNet

DATASET = paths.data_root() / "dataset" / "all-screen-team0-v0-t0-standard"
pytestmark = pytest.mark.skipif(not (DATASET / "index.jsonl").exists(),
                                reason="labelled dataset not built")


@pytest.fixture(scope="module")
def battles():
    from clashai.train.seq import Battles

    return Battles(DATASET)


def _stream(battles, snapshot=None):
    from clashai.train.seq import Stream

    return Stream(battles, battles.items(False, 12), 4, 48, 7, mirror=0.5, drop_prev=0.3,
                  snapshot=snapshot)


def _same(a, b):
    for k in a:
        if k == "groups":
            for ga, gb in zip(a[k], b[k], strict=True):
                for x, y in zip(ga, gb, strict=True):
                    np.testing.assert_array_equal(x, y)
        elif k != "snapshot":
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)


def test_a_seat_loads_in_order_with_its_counted_length_and_history(battles):
    b, _ = battles.items(False, 1)[0]
    d = battles.load(b, 1)
    assert len(d["actions"]) == battles.length(b)
    assert (np.diff(d["header"][:, battles.spec.header["tick"][0]]) > 0).all()
    assert d["prev_action"][0] == mem.NO_ACTION
    np.testing.assert_array_equal(d["prev_action"][1:], d["actions"][:-1])
    assert d["offsets"][-1] == len(d["entities"])


def test_the_stream_resumes_from_a_snapshot_exactly(battles):
    s = _stream(battles)
    for _ in range(5):
        b = s.next()
    snap = b["snapshot"]
    want = [s.next() for _ in range(30)]  # past several seats' ends and restarts
    again = _stream(battles, snap)
    for w in want:
        _same(again.next(), w)
    assert sum(w["first"].sum() for w in want) > 0  # some seats began inside these chunks


def test_lanes_are_lane_major_and_continue_their_seat(battles):
    s = _stream(battles)
    a, b = s.next(), s.next()
    tick = battles.spec.header["tick"][0]
    for lane in range(4):
        ta = a["header"][lane * 48:(lane + 1) * 48, tick]
        tb = b["header"][lane * 48:(lane + 1) * 48, tick]
        if not b["first"][lane * 48:(lane + 1) * 48].any():
            assert (np.diff(np.concatenate([ta, tb])) > 0).all()


def test_the_grouped_trunk_gives_what_one_padded_batch_gives(battles):
    from clashai.train.seq import pin, run

    torch.manual_seed(0)
    net = PolicyNet(battles.spec, Config(d=64, layers=2, heads=4, grid_channels=16,
                                         attn_bias=True, relations=True, lstm=32,
                                         prev_action=True)).eval()
    b = pin(_stream(battles).next())
    assert len(b["groups"]) >= 2  # the test needs rows of different widths
    n, w = len(b["actions"]), 80
    units = torch.zeros(n, w, battles.spec.entity_ints, dtype=torch.int32)
    valid = torch.zeros(n, w, dtype=torch.bool)
    for rows, u, v in b["groups"]:
        units[rows.long(), :u.shape[1]] = u
        valid[rows.long(), :v.shape[1]] = v
    with torch.no_grad():
        got, _, actions, header = run(net, {**b, "flip": torch.zeros_like(b["flip"])}, 4)
        memory = {"lanes": 4, "first": b["first"], "state": None,
                  "prev_action": b["prev_action"], "prev_card": b["prev_card"],
                  "since": b["since"]}
        want = net(b["header"], units, valid, memory)
    torch.testing.assert_close(got["type_logits"], want["type_logits"], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(header, b["header"])
    assert torch.equal(actions, b["actions"].long())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_validation_scores_whole_seats_for_recurrent_and_plain_models(battles):
    from clashai.train.seq import validate

    items = battles.items(True)[:6]
    for flags in ({"lstm": 32, "prev_action": True}, {}):
        torch.manual_seed(0)
        net = PolicyNet(battles.spec, Config(d=64, layers=2, heads=4, grid_channels=16,
                                             **flags)).cuda()
        m = validate(net, battles, items, "cuda", lanes=4, chunk=64)
        assert m["decisions"] == battles.decisions(items)
        assert np.isfinite(m["policy_nll"]) and m["policy_nll"] > 0
        assert m["policy_nll"] == pytest.approx(m["gate_nll"] + m["card_nll"] + m["tile_nll"])
        assert ("zero_state_cost" in m) == bool(flags)


def _pair(battles, keep):
    """The same chunk from two streams: every row's trunk gradient, and a ``keep`` share of the
    waits'."""
    from clashai.train.seq import Stream

    items = battles.items(False, 12)
    full = Stream(battles, items, 4, 48, 7, mirror=0.5, drop_prev=0.3)
    part = Stream(battles, items, 4, 48, 7, mirror=0.5, drop_prev=0.3, wait_grad=keep)
    return full.next(), part.next()


def test_every_play_and_a_seeded_share_of_waits_train_the_trunk_and_the_groups_cover_every_row(
        battles):
    from clashai.train.seq import Stream

    full, part = _pair(battles, 0.1)
    _same({k: v for k, v in part.items() if k not in ("grad", "group_grad", "groups",
                                                     "wait_grad")},
          {k: v for k, v in full.items() if k != "groups"})
    plays = part["actions"] != 0
    assert part["grad"][plays].all()
    share = part["grad"][~plays].mean()
    assert 0.03 < share < 0.2, share
    rows = np.concatenate([g[0] for g in part["groups"]])
    assert sorted(rows.tolist()) == list(range(len(part["actions"])))
    for (r, _, _), flag in zip(part["groups"], part["group_grad"], strict=True):
        assert (part["grad"][r] == flag).all()
    # Resumes exactly, the draw included.
    s = Stream(battles, battles.items(False, 12), 4, 48, 7, mirror=0.5, drop_prev=0.3,
               wait_grad=0.1)
    snap = s.next()["snapshot"]
    want = s.next()
    again = Stream(battles, battles.items(False, 12), 4, 48, 7, mirror=0.5, drop_prev=0.3,
                   wait_grad=0.1, snapshot=snap)
    _same(again.next(), want)


def test_sampling_the_trunks_waits_changes_no_output_and_no_gradient_after_the_trunk(battles):
    from clashai.train.seq import chunk_losses, pin

    full, part = (pin(b) for b in _pair(battles, 0.1))
    grads, outs = [], []
    for b in (full, part):
        torch.manual_seed(0)
        net = PolicyNet(battles.spec, Config(d=64, layers=2, heads=4, grid_channels=16,
                                             attn_bias=True, relations=True, lstm=32,
                                             prev_action=True)).train()  # dropout 0
        loss, s, _ = chunk_losses(net, b, 4, None, loss_weight=0.5, value_weight=0.1)
        loss.backward()
        outs.append(loss.detach())
        grads.append({n: q.grad.clone() for n, q in net.named_parameters() if q.grad is not None})
    torch.testing.assert_close(outs[0], outs[1])
    after = ("core.", "core_in.", "core_out.", "wait_head.", "slot_head.", "prev_kind.")
    for n in grads[0]:
        if n.startswith(after):
            torch.testing.assert_close(grads[1][n], grads[0][n], rtol=1e-5, atol=1e-6, msg=n)
    trunk = "trunk.layers.0.linear1.weight"
    assert not torch.allclose(grads[1][trunk], grads[0][trunk])  # fewer waits, scaled up


def _packed_pair(battles, profiles):
    """The same chunk from two streams: grouped by its rows, and packed into ``profiles``."""
    from clashai.train.seq import Stream

    items = battles.items(False, 12)
    common = {"mirror": 0.5, "drop_prev": 0.3, "wait_grad": 0.1}
    return (Stream(battles, items, 4, 48, 7, **common).next(),
            Stream(battles, items, 4, 48, 7, **common, profiles=profiles).next())


def test_a_packed_chunk_has_its_profiles_shapes_and_places_every_row_once(battles):
    from clashai.train.seq import Profile

    small = Profile((8, 8, 8), (16, 16, 16), 4)  # 72 places for 192 rows: never fits
    roomy = Profile((64, 48, 32), (192, 96, 64), 40)
    free, packed = _packed_pair(battles, (small, roomy))
    assert packed["packed"] == 1
    n = len(packed["actions"])
    sizes = [len(g[0]) for g in packed["groups"]]
    assert sizes == [*roomy.grad, *roomy.plain]
    assert [g[1].shape[1] for g in packed["groups"]] == [16, 32, 80, 16, 32, 80]
    assert packed["group_grad"] == [True] * 3 + [False] * 3
    # Each row sits at its place, in a group wide enough for its units and of its gradient flag.
    rows = np.concatenate([g[0] for g in packed["groups"]])
    np.testing.assert_array_equal(rows[packed["inverse"]], np.arange(n))
    assert len(set(packed["inverse"].tolist())) == n
    start = 0
    for (r, units, valid), flag in zip(packed["groups"], packed["group_grad"], strict=True):
        mine = packed["inverse"][(packed["inverse"] >= start) & (packed["inverse"] < start + len(r))]
        real = r[mine - start]
        assert (packed["grad"][real] == flag).all()
        assert (packed["counts"][real] <= units.shape[1]).all()
        np.testing.assert_array_equal(valid[mine - start].sum(1), packed["counts"][real])
        spare = np.setdiff1d(np.arange(len(r)), mine - start)
        assert not valid[spare].any() and not r[spare].any()
        start += len(r)
    plays = np.flatnonzero((packed["actions"] > 0)
                           & (packed["actions"] < battles.spec.first_ability))
    assert len(packed["play_rows"]) == 40 and packed["play_on"].sum() == len(plays)
    np.testing.assert_array_equal(packed["play_rows"][packed["play_on"]], plays)
    # The rows themselves are untouched, and a chunk that fits no profile is grouped as usual.
    same = ("header", "masks", "actions", "first", "flip", "prev_action", "counts", "grad")
    _same({k: packed[k] for k in same}, {k: free[k] for k in same})
    _, unfit = _packed_pair(battles, (small,))
    assert unfit["packed"] == -1 and "inverse" not in unfit
    _same({k: v for k, v in unfit.items() if k != "packed"}, free)


def test_a_packed_chunk_with_the_core_run_by_step_trains_as_the_chunk_grouped_by_its_rows(battles):
    """The step of fixed shapes computes the same loss, state and gradients as the usual one."""
    from clashai.train.seq import Profile, chunk_losses, pin

    free, packed = (pin(b) for b in _packed_pair(battles, (Profile((64, 48, 32), (192, 96, 64), 40),)))
    assert packed["packed"] == 0
    torch.manual_seed(0)
    net = PolicyNet(battles.spec, Config(d=64, layers=2, heads=4, grid_channels=16,
                                         attn_bias=True, relations=True, lstm=32,
                                         prev_action=True)).train()  # dropout 0
    g = torch.Generator().manual_seed(3)
    state = tuple(torch.randn(1, 4, 32, generator=g) for _ in range(2))
    got = []
    for b, by_step, sync in ((free, False, True), (packed, True, False)):
        net.core_steps = by_step
        net.zero_grad()
        loss, stats, new_state = chunk_losses(net, b, 4, state, loss_weight=0.5,
                                              value_weight=0.1, sync=sync)
        loss.backward()
        got.append((loss.detach(), stats, new_state,
                    {n: q.grad.clone() for n, q in net.named_parameters() if q.grad is not None}))
    (loss_a, stats_a, state_a, grads_a), (loss_b, stats_b, state_b, grads_b) = got
    torch.testing.assert_close(loss_b, loss_a, rtol=1e-5, atol=1e-6)
    assert all(isinstance(v, float) for v in stats_a.values())
    for k, v in stats_b.items():  # left as tensors, read later
        assert isinstance(v, torch.Tensor) and not v.requires_grad
        assert abs(v.item() - stats_a[k]) < 1e-5, k
    for x, y in zip(state_a, state_b, strict=True):
        torch.testing.assert_close(y, x, rtol=1e-4, atol=1e-5)
    assert grads_a.keys() == grads_b.keys()
    for name in grads_a:
        torch.testing.assert_close(grads_b[name], grads_a[name], rtol=2e-3, atol=2e-6, msg=name)
