import numpy as np

from clashai.data.dataset import Layout, compact, masks, observations

LAYOUT = Layout(header_ints=6, entity_ints=3, max_entities=5, actions=11, entity_count=2)


def test_compaction_rebuilds_observations_and_masks_exactly(tmp_path):
    rng = np.random.default_rng(0)
    rows = 7
    counts = rng.integers(0, LAYOUT.max_entities + 1, rows)
    obs = np.zeros((rows, LAYOUT.header_ints + LAYOUT.max_entities * LAYOUT.entity_ints), np.int32)
    obs[:, :LAYOUT.header_ints] = rng.integers(-5, 50, (rows, LAYOUT.header_ints))
    obs[:, LAYOUT.entity_count] = counts
    for r, c in enumerate(counts):
        start = LAYOUT.header_ints
        obs[r, start:start + c * LAYOUT.entity_ints] = rng.integers(-1, 30000, c * LAYOUT.entity_ints)
    mask = rng.integers(0, 2, (rows, LAYOUT.actions), dtype=np.uint8)
    source = tmp_path / "battle.npz"
    np.savez(source, observations=obs, masks=mask, actions=np.arange(rows, dtype=np.int32),
             ticks=np.arange(rows, dtype=np.int32) * 5, seats=np.zeros(rows, np.int32) % 2,
             flags=np.zeros(rows, np.int32))
    target = tmp_path / "compact.npz"
    assert compact(source, target, LAYOUT) == rows
    c = np.load(target)
    np.testing.assert_array_equal(observations(c, LAYOUT), obs)
    np.testing.assert_array_equal(masks(c, LAYOUT), mask)
    np.testing.assert_array_equal(c["actions"], np.arange(rows))
