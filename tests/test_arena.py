import pytest

from clashai.decks import PIGGIES_EQ, REFERENCE_DECKS
from clashai.eval.arena import Side, play
from clashai.eval.stats import summarize
from clashai.sim.policies import RandomLegal, WaitPolicy

pytestmark = pytest.mark.engine


def test_every_seed_is_played_from_both_seats_and_replays_exactly():
    a = Side(RandomLegal(0.2), PIGGIES_EQ)
    b = Side(WaitPolicy(), REFERENCE_DECKS["hog_26"])
    # A batch of 4 for 6 games: the second chunk is padded, and the padding is not counted.
    first = play(a, b, [10, 11, 12], batch=4)
    again = play(a, b, [10, 11, 12], batch=4)
    assert first == again
    assert [(g.seed, g.a_seat) for g in first] == [(10, 0), (10, 1), (11, 0), (11, 1),
                                                    (12, 0), (12, 1)]
    assert all(g.fault == "none" for g in first)


def test_a_player_who_plays_beats_one_who_never_does():
    games = play(Side(RandomLegal(0.2), PIGGIES_EQ), Side(WaitPolicy(), PIGGIES_EQ), range(4))
    assert summarize(g.score for g in games).score == 1.0
