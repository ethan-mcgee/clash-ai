"""Labelling filters: battles with a dropped play are kept up to it under the prefix filter."""

import numpy as np

from clashai.data.dataset import Layout, compact
from clashai.data.label import FILTERS, _cut_counts, _tag, stop_tick


def _row(**kw):
    return {"crowns_match": True, "result_match": True, "dropped": 0, "drops": [],
            "towers_exact": False, **kw}


def test_prefix_keeps_dropped_play_battles_and_cuts_them_one_decision_before_the_first():
    clean = _row()
    dropped = _row(dropped=2, drops=[[3371, 0, "play", "Hut"], [2200, 1, "ability", None]])
    wrong = _row(dropped=1, drops=[[100, 0, "play", "X"]], result_match=False)
    assert FILTERS["prefix"](clean) and FILTERS["prefix"](dropped) and not FILTERS["prefix"](wrong)
    assert FILTERS["standard"](clean) and not FILTERS["standard"](dropped)
    assert stop_tick(clean, "prefix") is None
    assert stop_tick(dropped, "prefix") == 2195  # the earliest drop, less one decision
    assert stop_tick(dropped, "standard") is None


def test_compact_with_a_stop_tick_keeps_only_the_earlier_decisions_of_both_seats(tmp_path):
    layout = Layout(header_ints=3, entity_ints=2, max_entities=4, actions=10, entity_count=0)
    ticks = np.repeat(np.arange(0, 50, 5), 2)
    n = len(ticks)
    obs = np.zeros((n, 3 + 4 * 2), np.int32)
    obs[:, 0] = 1  # one unit a row
    obs[:, 3] = np.arange(n)
    np.savez(tmp_path / "raw.npz", observations=obs, masks=np.ones((n, 10), bool),
             actions=np.arange(n, dtype=np.int32), ticks=ticks,
             seats=np.tile([0, 1], n // 2), flags=np.zeros(n, np.int32))
    rows = compact(tmp_path / "raw.npz", tmp_path / "cut.npz", layout, stop_tick=22)
    c = np.load(tmp_path / "cut.npz")
    assert rows == 10 and (c["ticks"] < 22).all() and c["ticks"].max() == 20
    assert np.bincount(c["seats"]).tolist() == [5, 5]
    assert c["entities"][:, 0].tolist() == list(range(10)) and c["offsets"][-1] == 10
    assert compact(tmp_path / "raw.npz", tmp_path / "all.npz", layout) == n


def test_a_cut_battles_counts_line_gives_what_was_kept():
    line = {"source": r"D:\data\docs\abc123.json", "counts": {"decisions": 700, "rows": 1400}}
    assert _tag(line) == "abc123"
    cut = _cut_counts(line, 1200, 3000)
    assert cut["counts"]["decisions"] == 600 and cut["counts"]["cut_at"] == 3000
    assert line["counts"]["decisions"] == 700
