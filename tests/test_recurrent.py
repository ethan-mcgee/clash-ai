"""Memory in PPO: what the rollout carries per battle is what the update rebuilds per sequence."""

import numpy as np
import pytest
import torch

from clashai import paths
from clashai.model import memory as mem
from clashai.train.recurrent import (
    Seats,
    batch_memory,
    join,
    sequence_rows,
    split_state,
)

ROWS = paths.data_root() / "rows" / "piggies_eq-screen-team0-v0-t0-standard-w0.25"
needs_rows = pytest.mark.skipif(not (ROWS / "rows.json").exists(),
                                reason="training rows not built (python -m clashai.train.rows)")
N, STEPS, SEQ = 4, 16, 8


def test_sequence_rows_are_lane_major_rows_of_a_rollout_laid_out_step_by_step():
    # 3 battles, sequences of 4: sequence 5 is battle 2's steps 4 to 7; sequence 0 battle 0's 0-3.
    rows = sequence_rows(np.array([5, 0]), 4, 3)
    assert rows.tolist() == [4 * 3 + 2, 5 * 3 + 2, 6 * 3 + 2, 7 * 3 + 2, 0, 3, 6, 9]


@pytest.fixture(scope="module")
def data():
    from clashai.model.spec import Spec, unpack_masks
    from clashai.train.rows import Rows

    rows = Rows(ROWS)
    spec = Spec.from_metadata(rows.metadata)
    idx = np.sort(np.random.default_rng(0).choice(rows.n, N * STEPS, replace=False))
    b = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in rows.gather(idx, 80).items()}
    b["mask"] = unpack_masks(b["masks"], spec.actions)
    return spec, b


def _net(spec, lstm=32):
    from clashai.model.net import Config, PolicyNet

    torch.manual_seed(0)
    return PolicyNet(spec, Config(d=64, layers=2, heads=4, grid_channels=16, lstm=lstm,
                                  prev_action=bool(lstm))).eval()


@needs_rows
def test_the_update_recomputes_the_rollouts_log_probabilities_from_the_stored_states(data):
    """A rollout in miniature: 4 battles, 16 decisions, battle 2 restarting after its 7th and
    battle 0 after its 12th. Each sequence, rebuilt from the state stored at its start and the
    stored history, gives the log-probabilities the rollout sampled with."""
    from clashai.train.ppo import evaluate_actions, sample

    spec, b = data
    net = _net(spec)
    seats = Seats(N, 32, spec, "cpu")
    gen = torch.Generator().manual_seed(1)
    buf = {"first": np.zeros((STEPS, N), bool), "logp": np.zeros((STEPS, N), np.float32),
           "actions": np.zeros((STEPS, N), np.int64),
           **{k: np.zeros((STEPS, N), np.int64) for k in ("prev_action", "prev_card", "since")}}
    states = [np.zeros((STEPS // SEQ, N, 32), np.float32) for _ in range(2)]
    at = lambda t: torch.arange(N) * STEPS + t  # the data rows battle i sees at step t
    with torch.no_grad():
        for t in range(STEPS):
            buf["first"][t] = seats.fresh.numpy()
            for name, val in seats.hist.items():
                buf[name][t] = val.numpy()
            if t % SEQ == 0:
                states[0][t // SEQ], states[1][t // SEQ] = seats.h.numpy(), seats.c.numpy()
            r = at(t)
            out = net(b["header"][r], b["entities"][r], b["valid"][r], seats.memory(net))
            actions, logp = sample(net, out, b["mask"][r], gen)
            seats.update(out["state"], b["header"][r], actions)
            buf["actions"][t], buf["logp"][t] = actions.numpy(), logp.numpy()
            if t == 6:
                seats.restart([2])
            if t == 11:
                seats.restart(np.array([0]))
    assert buf["first"][0].all() and buf["first"][7, 2] and buf["first"][12, 0]
    assert buf["first"].sum() == N + 2
    assert buf["prev_action"][7, 2] == mem.NO_ACTION and buf["since"][12, 0] == mem.NEVER
    np.testing.assert_array_equal(buf["prev_action"][1:6], buf["actions"][:5])

    flat = {k: v.reshape(STEPS * N, *v.shape[2:]) for k, v in buf.items()}
    pick = np.array([1, 2, 4, 7])  # sequences of battles 1 and 2 (first half), 0 and 3 (second)
    idx = sequence_rows(pick, SEQ, N)
    j, i = np.divmod(pick, N)
    memory = batch_memory(flat, idx, len(pick), (states[0][j, i], states[1][j, i]), "cpu")
    steps, battles = np.divmod(idx, N)
    r = torch.from_numpy(battles * STEPS + steps)
    net.train()  # as in the update; dropout is 0
    out = net(b["header"][r], b["entities"][r], b["valid"][r], memory)
    logp, _ = evaluate_actions(net, out, b["mask"][r], torch.from_numpy(flat["actions"][idx]))
    torch.testing.assert_close(logp, torch.from_numpy(flat["logp"][idx]), rtol=1e-4, atol=1e-4)
    # And it matters: from a zero state the recurrent network scores them differently.
    blank = {**memory, "state": tuple(torch.zeros_like(s) for s in memory["state"])}
    other, _ = evaluate_actions(net, net(b["header"][r], b["entities"][r], b["valid"][r], blank),
                                b["mask"][r], torch.from_numpy(flat["actions"][idx]))
    assert not torch.allclose(other, logp, atol=1e-4)


@needs_rows
def test_one_joined_pass_equals_two_and_some_seats_update_alone(data):
    spec, b = data
    net = _net(spec)
    opp, shadow = Seats(N, 32, spec, "cpu"), Seats(N, 32, spec, "cpu")
    g = torch.Generator().manual_seed(2)
    for seats in (opp, shadow):
        seats.h, seats.c = torch.randn(N, 32, generator=g), torch.randn(N, 32, generator=g)
        seats.fresh[:] = False
    rows = np.array([1, 3])
    ra, rb = torch.tensor([5, 9]), torch.arange(N) + 20
    both = torch.cat([ra, rb])
    with torch.no_grad():
        joined = net(b["header"][both], b["entities"][both], b["valid"][both],
                     join(opp.memory(net, rows), shadow.memory(net)))
        first = net(b["header"][ra], b["entities"][ra], b["valid"][ra], opp.memory(net, rows))
        second = net(b["header"][rb], b["entities"][rb], b["valid"][rb], shadow.memory(net))
    torch.testing.assert_close(joined["type_logits"],
                               torch.cat([first["type_logits"], second["type_logits"]]),
                               rtol=1e-5, atol=1e-5)
    state_a, state_b = split_state(joined["state"], 2)
    torch.testing.assert_close(state_a[0], first["state"][0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(state_b[1], second["state"][1], rtol=1e-5, atol=1e-5)
    before = opp.h.clone()
    opp.update(state_a, b["header"][ra], torch.tensor([0, 7]), rows)
    assert torch.equal(opp.h[[0, 2]], before[[0, 2]])  # the other seats are untouched
    torch.testing.assert_close(opp.h[[1, 3]], first["state"][0][0], rtol=1e-5, atol=1e-5)
    assert opp.hist["prev_action"][[1, 3]].tolist() == [0, 7]
    assert opp.hist["prev_action"][0] == mem.NO_ACTION


@needs_rows
def test_a_network_without_a_core_has_no_memory_and_seats_still_keep_the_history(data):
    spec, b = data
    plain = _net(spec, lstm=0)
    seats = Seats(N, 0, spec, "cpu")
    assert seats.memory(plain) is None and join(None, None) is None
    assert split_state(None, 2) == (None, None)
    seats.update(None, b["header"][:N], torch.tensor([0, 5, 0, 0]))
    assert seats.hist["prev_action"].tolist() == [0, 5, 0, 0] and not seats.fresh.any()
    assert seats.hist["since"].tolist() == [mem.NEVER, 1, mem.NEVER, mem.NEVER]
    with pytest.raises(ValueError, match="core"):
        Seats(N, 16, spec, "cpu").memory(_net(spec))
    with pytest.raises(ValueError, match="both"):
        join(None, {"lanes": 1})


@needs_rows
@pytest.mark.engine
def test_ppo_with_a_recurrent_learner_reference_and_critic_plays_and_trains(tmp_path, monkeypatch):
    """The whole loop on the CPU with a small recurrent network, long enough for games to end
    and restart: the first policy update recomputes the rollout's log-probabilities (so its KL
    from the rollout's policy is about zero), every statistic is a number, and the run resumes."""
    import json

    from clashai.model.spec import Spec
    from clashai.train import ppo

    metadata = json.loads((ROWS / "metadata.json").read_text())
    net = _net(Spec.from_metadata(metadata))
    init = tmp_path / "init.pt"
    torch.save({"model": net.state_dict(), "description": net.describe(), "step": 0,
                "metadata": metadata}, init)
    monkeypatch.setattr(paths, "runs_dir", lambda: tmp_path)
    common = ["--name", "trial", "--init", str(init), "--device", "cpu", "--envs", "8",
              "--steps", "64", "--minibatch", "128", "--critic-warmup", "1", "--eval-seeds", "0",
              "--snapshot-every", "2", "--mirror-share", "0.5", "--threads", "2", "--lr", "1e-9"]
    ppo.main([*common, "--iters", "24"])
    rows = [json.loads(line) for line in open(tmp_path / "ppo" / "trial" / "metrics.jsonl")]
    assert len(rows) == 24
    assert sum(r["games"] + r["mirror_games"] for r in rows) >= 4  # games ended and restarted
    assert rows[0]["warmup"] and "approx_kl" not in rows[0]
    for r in rows[1:]:
        assert abs(r["approx_kl"]) < 2e-3, r["approx_kl"]
        assert r["clip_frac"] == 0
        assert all(np.isfinite(r[k]) for k in ("value_loss", "policy_loss", "entropy", "kl_ref"))
        assert r["kl_ref"] < 2e-3  # the reference is the learner's start, read with its own memory
    assert sorted(f.name for f in (tmp_path / "ppo" / "trial" / "pool").iterdir())[0] == "iter_00002.pt"
    ppo.main([*common, "--iters", "25", "--resume"])
    assert len(open(tmp_path / "ppo" / "trial" / "metrics.jsonl").readlines()) == 25
