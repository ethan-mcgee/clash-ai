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
