"""The replay converter, end to end on a few real replays."""

import numpy as np
import pytest

from clashai import paths
from clashai.data import convert as C
from clashai.data.cards import CardNames

pytestmark = pytest.mark.engine


def test_a_point_maps_to_its_tile_in_each_seats_frame():
    # Measured in the engine: side 0's frame mirrors x, side 1's mirrors y.
    assert C._tile(0, 15500, 5500) == (2, 5)
    assert C._tile(1, 2500, 26500) == (2, 5)
    assert C._tile(0, 18000, 32000) == (0, 31)  # the far edges clamp onto the grid


def test_real_replays_convert_with_the_planned_deal_and_land_where_recorded(tmp_path):
    if not (C.dataset_dir() / "replays").is_dir():
        pytest.skip("replay dataset not downloaded (python -m clashai.data.download)")
    from crforge_env import LadderConfig, VectorBattle

    replays = []
    for r in C.read_replays(C.dataset_dir()):
        if any(C.is_deck(r, side, *C.PIGGIES) for side in ("team", "opponent")):
            replays.append(r)
        if len(replays) == 8:
            break
    events = C.load_events(C.dataset_dir(), {r.tag for r in replays})
    paths.ensure_java()
    with VectorBattle(8, 4, install=paths.install_dir(), tables=paths.tables_dir(),
                      config=LadderConfig(decision_ticks=1, level=16, tower_level=16,
                                          arena=C.ARENA)) as v:
        names = CardNames(v.vocabulary[i] for i in v.card_types())
        rng = np.random.default_rng(0)
        plans = [C.plan_replay(r, events[r.tag], names, C.SeedTable(), rng) for r in replays]
        assert not [p.problem for p in plans if p.problem]
        outcomes = C.drive(plans, v, tmp_path)
    assert all(o.deal_ok for o in outcomes)
    assert all(o.fault == "none" and o.not_modelled == 0 for o in outcomes)
    plays = sum(o.plays for o in outcomes)
    assert sum(o.exact_points for o in outcomes) / plays > 0.85
    assert sum(o.dropped for o in outcomes) / plays < 0.03
    assert sum(o.result_match for o in outcomes) >= 6
    assert len(list((tmp_path / "docs").glob("*.json"))) == 8
