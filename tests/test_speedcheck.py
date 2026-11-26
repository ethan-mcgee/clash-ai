"""The speed check reads the trainer's log lines and lays the modes side by side."""

from clashai.train.speedcheck import summary, table, windows

LOG = """17.5M parameters; config {...}
step 0 (0.00 passes): policy nll 1.2360 (gate 0.9858, card 0.0496, tile 0.2006)
  step 200 loss 1.4945 policy 0.3647 (4,370 rows/s, waiting for data 0%, lr 4.02e-05)
  step 400 loss 1.2694 policy 0.3269 (9,109 rows/s, waiting for data 0%, lr 8.02e-05, recorded steps 40%)
  step 600 loss 1.0760 policy 0.3227 (11,000 rows/s, waiting for data 2%, lr 1.20e-04, recorded steps 96%)
  step 800 loss 1.0100 policy 0.3100 (12,000 rows/s, waiting for data 1%, lr 1.60e-04, recorded steps 98%)
done at step 800; best validation policy loss 1.2360; checkpoints in x
""".splitlines()


def test_log_windows_are_read_with_and_without_the_recorded_share():
    found = windows(LOG)
    assert [w["step"] for w in found] == [200, 400, 600, 800]
    assert found[0] == {"step": 200, "rows_per_s": 4370.0, "waiting": 0, "recorded": None}
    assert found[2] == {"step": 600, "rows_per_s": 11000.0, "waiting": 2, "recorded": 96}


def test_a_modes_row_leaves_out_the_first_windows_and_their_gpu_samples():
    gpu = [(10.0, 100.0, 1000.0)] * 4 + [(90.0, 400.0, 9000.0)] * 4
    row = summary("cuda", windows(LOG), gpu)
    assert row["rows_per_s"] == 11500.0 and row["recorded"] == 97 and row["waiting"] == 2
    assert row["gpu_util"] == 90.0 and row["gpu_watts"] == 400.0 and row["gpu_mib"] == 9000.0
    assert "error" in summary("off", windows(LOG[:4]), [])


def test_the_table_compares_each_mode_with_off_and_shows_a_failure():
    rows = [{"mode": "off", "rows_per_s": 9000.0, "waiting": 0, "recorded": None,
             "gpu_util": 76.0, "gpu_watts": 314.0, "gpu_mib": 6700.0},
            {"mode": "cuda", "rows_per_s": 11250.0, "waiting": 1, "recorded": 97.0,
             "gpu_util": 95.0, "gpu_watts": 380.0, "gpu_mib": 10240.0},
            {"mode": "compile", "error": "exit 1: no compiler"}]
    lines = table(rows).splitlines()
    assert "1.00x" in lines[1] and "76%" in lines[1] and "314 W" in lines[1]
    assert "1.25x" in lines[2] and "97%" in lines[2] and "10.0 GB" in lines[2]
    assert lines[3].startswith("compile") and "no compiler" in lines[3]
