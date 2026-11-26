"""The TensorBoard mirror writes what the runs' files say, once, and follows new lines."""

import json

import pytest

pytest.importorskip("tensorboard")


def _events(folder):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(folder), size_guidance={"scalars": 0})
    acc.Reload()
    return {t: [(e.step, e.value) for e in acc.Scalars(t)] for t in acc.Tags()["scalars"]}


def test_a_runs_files_become_curves_once_and_new_lines_are_followed(tmp_path):
    from clashai.train.board import mirror

    run = tmp_path / "runs" / "seq" / "v3"
    run.mkdir(parents=True)
    rows = [{"step": 0, "minutes": 1.5, "policy_nll": 1.25, "gate_nll": 1.0, "card_acc": 0.25,
             "zero_state_cost": 0.5, "plays_seen": 0, "train": {"policy": 0.75},
             "value_sign_by_minute": [0.5, 0.75], "warmup": True, "score": None},
            {"step": 4000, "minutes": 30.0, "policy_nll": 0.25, "gate_nll": 0.125,
             "card_acc": 0.5, "zero_state_cost": 0.25, "plays_seen": 632000,
             "train": {"policy": 0.5}, "value_sign_by_minute": [0.5, 0.625]}]
    (run / "metrics.jsonl").write_text(json.dumps(rows[0]) + "\n")
    (run / "train.log").write_text(
        "17.5M parameters\n"
        "  step 200 loss 1.5000 policy 0.3750 (8,608 rows/s, waiting for data 2%, lr 4.00e-05)\n"
        "  step 400 loss 1.2500 policy 0.31")  # the last line is still being written
    mirrors: dict = {}
    out = tmp_path / "runs" / "tb"
    mirror(tmp_path / "runs", out, tmp_path, mirrors)
    with open(run / "metrics.jsonl", "a") as f:
        f.write(json.dumps(rows[1]) + "\n")
    with open(run / "train.log", "a") as f:
        f.write("25 (9,000 rows/s, waiting for data 0%, lr 8.00e-05)\n")
    mirror(tmp_path / "runs", out, tmp_path, mirrors)
    assert mirror(tmp_path / "runs", out, tmp_path, mirrors) == 0  # nothing new, nothing written
    for m in mirrors.values():
        m.close()
    e = _events(out / "seq" / "v3")
    assert e["loss/policy_nll"] == [(0, 1.25), (4000, 0.25)]
    assert e["loss/gate_nll"] == [(0, 1.0), (4000, 0.125)]
    assert e["accuracy/card_acc"] == [(0, 0.25), (4000, 0.5)]
    assert e["memory/zero_state_cost"] == [(0, 0.5), (4000, 0.25)]
    assert e["train/policy"] == [(0, 0.75), (4000, 0.5)]
    assert e["value_sign_by_minute/2"] == [(0, 0.75), (4000, 0.625)]
    assert e["by_plays/policy_nll"] == [(0, 1.25), (632, 0.25)]
    assert e["speed/rows_per_s"] == [(200, 8608.0), (400, 9000.0)]
    assert e["live/loss"] == [(200, 1.5), (400, 1.25)]
    assert "accuracy/warmup" not in e and "accuracy/score" not in e


def test_a_run_started_over_is_mirrored_afresh_and_ppo_scorecards_become_curves(tmp_path):
    from clashai.train.board import mirror

    run = tmp_path / "runs" / "ppo" / "p"
    run.mkdir(parents=True)
    (run / "metrics.jsonl").write_text(
        "".join(json.dumps({"iteration": i, "minutes": i, "score": 0.5, "kl_ref": 0.25}) + "\n"
                for i in range(3)))
    (run / "evals.jsonl").write_text(json.dumps(
        {"iteration": 2, "score": 0.75, "by_deck": {"golem": 0.25}}) + "\n")
    mirrors: dict = {}
    out = tmp_path / "runs" / "tb"
    mirror(tmp_path / "runs", out, tmp_path, mirrors)
    (run / "metrics.jsonl").write_text(json.dumps({"iteration": 0, "minutes": 0, "score": 0.25})
                                       + "\n")
    mirror(tmp_path / "runs", out, tmp_path, mirrors)
    for m in mirrors.values():
        m.close()
    e = _events(out / "ppo" / "p")
    assert e["ppo/score"] == [(0, 0.25)]
    assert e["eval/score"] == [(2, 0.75)] and e["eval_deck/golem"] == [(2, 0.25)]
