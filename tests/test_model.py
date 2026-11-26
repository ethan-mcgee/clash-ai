"""The policy network, its action layout and its batched mirror, on real labelled rows."""

import numpy as np
import pytest
import torch

from clashai import paths
from clashai.model.net import Config, PolicyNet
from clashai.model.spec import Spec, unpack_masks
from clashai.sim.mirror import mirror_actions, mirror_mask, mirror_observation

ROWS = paths.data_root() / "rows" / "piggies_eq-screen-team0-v0-t0-standard-w0.25"
pytestmark = pytest.mark.skipif(not (ROWS / "rows.json").exists(),
                                reason="training rows not built (python -m clashai.train.rows)")


@pytest.fixture(scope="module")
def data():
    from clashai.train.rows import Rows

    rows = Rows(ROWS)
    spec = Spec.from_metadata(rows.metadata)
    rng = np.random.default_rng(0)
    commands = np.flatnonzero(rows.actions != 0)
    idx = np.sort(np.concatenate([rng.choice(commands, 48, replace=False),
                                  rng.choice(rows.n, 16, replace=False)]))
    return rows, spec, rows.gather(idx, 80)


def test_mapped_rows_gather_the_same_batches_as_loaded_rows(data):
    from clashai.train.rows import Rows

    rows, _, _ = data
    loaded = Rows(ROWS, mmap=False)
    assert isinstance(rows.header, np.memmap) and not isinstance(loaded.header, np.memmap)
    idx = np.sort(np.random.default_rng(1).choice(rows.n, 300, replace=False))
    a, b = rows.gather(idx, 80), loaded.gather(idx, 80)
    assert a.keys() == b.keys()
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_actions_decode_and_encode_back(data):
    _, spec, _ = data
    a = torch.arange(spec.actions)
    kind, tile = spec.decode(a)
    assert torch.equal(spec.encode(kind, tile), a)
    assert kind.max() == spec.types - 1 and kind[0] == 0 and kind[-1] == spec.types - 1


def test_masks_unpack_exactly_and_every_label_type_is_open_or_flagged(data):
    _, spec, b = data
    packed = torch.from_numpy(b["masks"])
    mask = unpack_masks(packed, spec.actions)
    np.testing.assert_array_equal(mask.numpy(), np.unpackbits(b["masks"], axis=1,
                                                              count=spec.actions).astype(bool))
    kind, _ = spec.decode(torch.from_numpy(b["actions"]).long())
    open_type = spec.type_mask(mask)[torch.arange(len(kind)), kind]
    assert torch.all(open_type | torch.from_numpy(b["label_masked"]))


def test_the_batched_mirror_matches_the_engine_side_mirror(data):
    rows, spec, b = data
    from types import SimpleNamespace

    s = SimpleNamespace(header=spec.header, entity=spec.entity, header_ints=spec.header_ints,
                        entity_ints=spec.entity_ints, max_entities=rows.info["max_entities"],
                        width_units=spec.width, unknown=-1, wait=0, cols=spec.cols,
                        rows=spec.rows, hand_slots=spec.hand_slots,
                        first_ability=spec.first_ability)
    header = torch.from_numpy(b["header"])
    ents = torch.from_numpy(b["entities"])
    valid = torch.from_numpy(b["valid"])
    mask = unpack_masks(torch.from_numpy(b["masks"]), spec.actions)
    actions = torch.from_numpy(b["actions"]).long()
    mh, me, mm, ma = spec.mirror(header, ents, valid, mask, actions)
    np.testing.assert_array_equal(ma.numpy(), mirror_actions(s, b["actions"]))
    np.testing.assert_array_equal(mm.numpy(), mirror_mask(s, mask.numpy()))
    for i in range(len(header)):
        n = int(valid[i].sum())
        obs = np.zeros(s.header_ints + s.max_entities * s.entity_ints, np.int32)
        obs[:s.header_ints] = b["header"][i]
        obs[s.header_ints:s.header_ints + n * s.entity_ints] = b["entities"][i, :n].ravel()
        want = mirror_observation(s, obs)
        np.testing.assert_array_equal(mh[i].numpy(), want[:s.header_ints])
        np.testing.assert_array_equal(
            me[i, :n].numpy(), want[s.header_ints:s.header_ints + n * s.entity_ints].reshape(n, -1))
    assert torch.equal(spec.mirror(mh, me, valid, mm, ma)[0], header)


def test_the_network_runs_and_learns_a_batch(data):
    from clashai.train.bc import forward_losses, to_device

    _, spec, b = data
    torch.manual_seed(0)
    net = PolicyNet(spec, Config(d=64, layers=2, heads=4, grid_channels=16))
    batch = to_device(b, "cpu")
    out = net(batch["header"], batch["entities"], batch["valid"])
    assert out["type_logits"].shape == (len(b["actions"]), spec.types)
    rows = torch.arange(3)
    assert net.tile_logits(out, rows, torch.tensor([0, 1, 3])).shape == (3, spec.tiles)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    first = None
    for _ in range(30):
        loss, s = forward_losses(net, batch, policy_rows="all", loss_weight=0.5, mirror=0.5)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first or s["policy"]
    assert s["policy"] < 0.7 * first


@pytest.mark.engine
def test_a_network_policy_plays_only_legal_actions_in_real_battles(data, tmp_path):
    from clashai.decks import PIGGIES_EQ
    from clashai.model.policy import NetPolicy
    from clashai.sim.env import make_vector

    _, spec, _ = data
    torch.manual_seed(0)
    net = PolicyNet(spec, Config(d=64, layers=2, heads=4, grid_channels=16))
    path = tmp_path / "tiny.pt"
    torch.save({"model": net.state_dict(), "description": net.describe(), "step": 0}, path)
    policy = NetPolicy(path, device="cuda" if torch.cuda.is_available() else "cpu")
    with make_vector(4, 2) as v:
        s = v.schema
        policy.bind(v)
        slots = PIGGIES_EQ.slots(s.slot_evolution)
        v.reset([1, 2, 3, 4], list(PIGGIES_EQ.cards), list(PIGGIES_EQ.cards), slots, slots)
        rng = np.random.default_rng(0)
        plays = 0
        for _ in range(400):
            obs, masks = v.observations.reshape(8, -1), v.masks.reshape(8, -1)
            a = policy.act(obs, masks, rng)
            assert masks[np.arange(8), a].all()
            plays += int((a != 0).sum())
            v.actions[:] = a.reshape(4, 2)
            v.step()
        assert plays > 0
        assert not v.infos[:, s.info["illegal_0"]].any() and not v.infos[:, s.info["illegal_1"]].any()


def test_arithmetic_bits_equal_shifts_and_masks_for_every_integer():
    from clashai.model.net import _bits

    x = torch.arange(-5000, 5000)
    want = torch.stack([((x >> k) & 1) for k in range(12)], -1).float()
    assert torch.equal(_bits(x, 12), want)


def test_the_all_slots_tile_head_equals_the_per_slot_one(data):
    from clashai.train.bc import to_device

    _, spec, b = data
    torch.manual_seed(0)
    net = PolicyNet(spec, Config(d=64, layers=2, heads=4, grid_channels=16)).eval()
    batch = to_device(b, "cpu")
    with torch.no_grad():
        out = net(batch["header"], batch["entities"], batch["valid"])
        every = net.tile_logits_all(out)
        rows = torch.arange(len(b["actions"]))
        for slot in range(spec.hand_slots):
            one = net.tile_logits(out, rows, torch.full_like(rows, slot))
            torch.testing.assert_close(every[:, slot], one, rtol=1e-5, atol=1e-5)


def test_the_explicit_trunk_for_export_equals_the_trunk(data):
    from clashai.train.bc import to_device

    _, spec, b = data
    torch.manual_seed(0)
    net = PolicyNet(spec, Config(d=64, layers=2, heads=4, grid_channels=16)).eval()
    batch = to_device(b, "cpu")
    with torch.no_grad():
        tokens, pad, _ = net._tokens(batch["header"], batch["entities"], batch["valid"])
        want = net.trunk(tokens, src_key_padding_mask=pad)
        got = net.trunk_explicit(tokens, pad)
    live = ~pad  # padded tokens' outputs are never read
    torch.testing.assert_close(got[live], want[live], rtol=1e-4, atol=1e-4)


def test_dropout_is_saved_with_the_model_and_can_be_changed(data):
    _, spec, _ = data
    net = PolicyNet(spec, Config(d=64, layers=2, heads=4, grid_channels=16, dropout=0.1))
    again = PolicyNet.from_description(net.describe())
    assert again.config.dropout == 0.1
    assert {m.p for m in again.modules() if isinstance(m, torch.nn.Dropout)} == {0.1}
    again.set_dropout(0.0)
    assert {m.p for m in again.modules() if isinstance(m, torch.nn.Dropout)} == {0.0}
    assert all(m.dropout == 0.0 for m in again.modules()
               if isinstance(m, torch.nn.MultiheadAttention))
    old = net.describe()
    del old["config"]["dropout"]  # checkpoints from before dropout existed
    assert PolicyNet.from_description(old).config.dropout == 0.0


def test_the_value_loss_trains_the_value_head_but_not_the_trunk(data):
    from clashai.train.bc import forward_losses, to_device

    _, spec, b = data
    torch.manual_seed(0)
    net = PolicyNet(spec, Config(d=64, layers=2, heads=4, grid_channels=16))
    batch = to_device(b, "cpu")
    grads = []
    for value_weight in (0.0, 1.0):
        net.zero_grad()
        loss, _ = forward_losses(net, batch, policy_rows="all", loss_weight=0.5,
                                 value_weight=value_weight)
        loss.backward()
        grads.append({n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None})
    without, with_value = grads
    for name, g in with_value.items():
        if name.startswith("value_head."):
            assert name not in without or not torch.equal(g, without[name])
        else:
            torch.testing.assert_close(g, without[name])
