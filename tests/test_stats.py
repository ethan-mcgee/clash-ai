import math

import pytest

from clashai.eval.stats import summarize, wilson


def test_wilson_matches_known_values():
    assert wilson(50, 100) == pytest.approx((0.4038, 0.5962), abs=1e-4)
    assert wilson(0, 10) == pytest.approx((0.0, 0.2775), abs=1e-4)
    assert wilson(10, 10) == pytest.approx((0.7225, 1.0), abs=1e-4)


def test_wilson_with_no_games_is_uninformative():
    assert wilson(0, 0) == (0.0, 1.0)


def test_summary_counts_draws_as_half_and_drops_unfinished_games():
    s = summarize([1, 1, 0.5, 0, None])
    assert (s.games, s.wins, s.draws, s.losses, s.unfinished) == (5, 2, 1, 1, 1)
    assert s.score == pytest.approx(2.5 / 4)
    assert s.low < s.score < s.high


def test_summary_of_nothing_has_no_score():
    assert math.isnan(summarize([None]).score)
