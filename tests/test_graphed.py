"""The step runner: every mode trains as the plain loop does."""

import copy

import pytest
import torch

from clashai import paths
from clashai.model.net import Config, PolicyNet

DATASET = paths.data_root() / "dataset" / "all-screen-team0-v0-t0-standard"
pytestmark = pytest.mark.skipif(not (DATASET / "index.jsonl").exists(),
                                reason="labelled dataset not built")
LANES, CHUNK = 4, 48
LOSS = {"loss_weight": 0.5, "value_weight": 0.1}


@pytest.fixture(scope="module")
def battles():
    from clashai.train.seq import Battles

    return Battles(DATASET)


def _chunks(battles, n, profiles=(), device="cpu"):
    from clashai.train.seq import Stream, pin, to

    stream = Stream(battles, battles.items(False, 12), LANES, CHUNK, 7, mirror=0.5,
                    drop_prev=0.3, wait_grad=0.1, profiles=profiles)
    return [to(pin(stream.next()), device) for _ in range(n)]


def _setup(battles, device="cpu"):
    from clashai.train.bc import EMA

    torch.manual_seed(0)
    net = PolicyNet(battles.spec, Config(d=64, layers=2, heads=4, grid_channels=16,
                                         attn_bias=True, relations=True, lstm=32,
                                         prev_action=True)).to(device).train()  # dropout 0
    # A large eps: Adam otherwise turns rounding noise in a gradient that is exactly zero (a bias
    # before a normalisation) into full-sized steps, different in every way of running.
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=0.05,
                            eps=1e-4, fused=device != "cpu")
    return net, opt, EMA(net, 0.9)


def _plain_loop(battles, chunks, lrs, device="cpu"):
    """The trainer's loop before the step runner, kept here as the reference."""
    from clashai.train.seq import chunk_losses

    net, opt, ema = _setup(battles, device)
    state, stats = None, []
    for at, (b, lr) in enumerate(zip(chunks, lrs, strict=True)):
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss, s, new_state = chunk_losses(net, b, LANES, state, **LOSS)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(net.parameters()), 1.0)
        opt.step()
        ema.update(net, at)
        state = tuple(x.detach() for x in new_state)
        stats.append(s)
    return net, ema, state, stats


def _run(battles, chunks, lrs, mode, device="cpu", profiles=0):
    from clashai.train.graphed import Stepper

    net, opt, ema = _setup(battles, device)
    stepper = Stepper(net, opt, ema, LANES, mode=mode, profiles=profiles, **LOSS)
    state, stats = None, []
    for at, (b, lr) in enumerate(zip(chunks, lrs, strict=True)):
        s, state = stepper.step(b, state, lr=lr, at=at)
        stats.append({k: float(v) for k, v in s.items()})
    return net, ema, state, stats, stepper


def _close(a, b, rtol, atol):
    for (name, x), y in zip(a.named_parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(y, x, rtol=rtol, atol=atol, msg=name)


LRS = [1e-3, 5e-4, 2e-3, 1e-3, 1e-3, 7e-4]


def test_the_step_runner_off_is_the_plain_loop(battles):
    chunks = _chunks(battles, 3)
    net, ema, state, stats = _plain_loop(battles, chunks, LRS[:3])
    net2, ema2, state2, stats2, stepper = _run(battles, chunks, LRS[:3], "off")
    _close(net, net2, 0, 0)
    _close(ema.net, ema2.net, 0, 0)
    for x, y in zip(state, state2, strict=True):
        assert torch.equal(x, y)
    for s, s2 in zip(stats, stats2, strict=True):
        assert s == pytest.approx(s2, abs=1e-7)
    assert stepper.counts == {"recorded": 0, "ordinary": 3}


def test_packed_chunks_the_core_by_step_and_rates_held_in_tensors_train_as_the_plain_loop(
        battles, monkeypatch):
    """Everything the recorded mode changes around the recording itself, checked on the CPU: the
    warm-up steps here stand for all of them."""
    from clashai.train import graphed
    from clashai.train.seq import Profile

    profiles = (Profile((64, 48, 32), (192, 96, 64), 40),)
    plain = _chunks(battles, 4)
    packed = _chunks(battles, 4, profiles)
    assert all(b["packed"] == 0 for b in packed)
    monkeypatch.setattr(graphed, "WARMUP", 10**9)
    monkeypatch.setattr(graphed.Stepper, "_warm_up", graphed.Stepper._ordinary)
    net, ema, state, stats = _plain_loop(battles, plain, LRS[:4])
    net2, ema2, state2, stats2, _ = _run(battles, packed, LRS[:4], "cuda")
    assert net2.core_steps
    _close(net, net2, 2e-3, 2e-5)
    _close(ema.net, ema2.net, 2e-3, 2e-5)
    for x, y in zip(state, state2, strict=True):
        torch.testing.assert_close(y, x, rtol=1e-3, atol=1e-4)
    for s, s2 in zip(stats, stats2, strict=True):
        assert s == pytest.approx(s2, rel=1e-4, abs=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_recorded_steps_train_as_ordinary_steps(battles):
    """On the GPU: a run whose steps are replayed CUDA graphs ends with the weights of the plain
    loop, through warm-up, the recording, replays and a chunk that fits no profile."""
    from clashai.train.seq import Profile

    profiles = (Profile((64, 48, 32), (192, 96, 64), 40),)
    packed = _chunks(battles, 6, profiles, "cuda")
    plain = _chunks(battles, 6, (), "cuda")
    unfit = copy.copy(plain[4])  # the fifth chunk fits no profile
    unfit["packed"] = -1
    packed[4] = unfit
    net, ema, state, stats = _plain_loop(battles, plain, LRS, "cuda")
    net2, ema2, state2, stats2, stepper = _run(battles, packed, LRS, "cuda", "cuda")
    assert stepper.counts == {"recorded": 2, "ordinary": 4}
    _close(net, net2, 5e-3, 5e-5)
    _close(ema.net, ema2.net, 5e-3, 5e-5)
    for x, y in zip(state, state2, strict=True):
        torch.testing.assert_close(y, x, rtol=5e-3, atol=5e-4)
    for s, s2 in zip(stats, stats2, strict=True):
        assert s == pytest.approx(s2, rel=1e-3, abs=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_recordings_sharing_memory_replay_in_turn_as_ordinary_steps(battles):
    """On the GPU: two recordings in one memory pool, replayed in turn, train as the plain loop
    does; neither overwrites what the other needs. Neither is recorded before both are warmed
    up. (Both profiles have the same shapes; what matters is that they are two recordings.)"""
    from clashai.train.seq import Profile

    profiles = (Profile((64, 48, 32), (192, 96, 64), 40),)
    n = 10
    packed = _chunks(battles, n, profiles, "cuda")
    labels = [0, 0, 0, 0, 1, 1, 1, 0, 0, 1]
    for b, label in zip(packed, labels, strict=True):
        assert b["packed"] == 0
        b["packed"] = label
    plain = _chunks(battles, n, (), "cuda")
    lrs = [2e-4 * (1 + i % 3) for i in range(n)]
    net, ema, state, stats = _plain_loop(battles, plain, lrs, "cuda")
    net2, ema2, state2, stats2, stepper = _run(battles, packed, lrs, "cuda", "cuda", profiles=2)
    # Profile 0 is warm after step 2 but waits for profile 1 (steps 4-6): 7 records it, 8
    # replays it, 9 records profile 1.
    assert stepper.counts == {"recorded": 3, "ordinary": 7}
    assert len(stepper.recorded) == 2
    # Over 10 steps the GPU's arithmetic order alone moves weights about 1e-4 from the plain
    # loop (8e-5 with no recording at all, 1.7e-4 with one recording or two); a replay that
    # overwrote another's results would be far off.
    _close(net, net2, 5e-3, 5e-4)
    _close(ema.net, ema2.net, 5e-3, 5e-4)
    for x, y in zip(state, state2, strict=True):
        torch.testing.assert_close(y, x, rtol=5e-3, atol=5e-4)
    for s, s2 in zip(stats, stats2, strict=True):
        assert s == pytest.approx(s2, rel=1e-3, abs=1e-4)
