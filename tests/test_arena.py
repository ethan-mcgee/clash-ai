import pytest

from clashai.decks import PIGGIES_EQ, REFERENCE_DECKS
from clashai.eval.arena import Side, play, play_matches
from clashai.eval.stats import summarize
from clashai.sim.policies import RandomLegal, WaitPolicy

pytestmark = pytest.mark.engine


def test_every_seed_is_played_from_both_seats_and_replays_exactly():
    a = Side(RandomLegal(0.2), PIGGIES_EQ)
    b = Side(WaitPolicy(), REFERENCE_DECKS["hog_26"])
    # A batch of 4 for 10 games: 3 chunks of 4, the last padded, and the padding not counted.
    first = play(a, b, [10, 11, 12, 13, 14], batch=4)
    again = play(a, b, [10, 11, 12, 13, 14], batch=4)
    assert first == again
    assert [(g.seed, g.a_seat) for g in first] == [(s, seat) for s in range(10, 15)
                                                    for seat in (0, 1)]
    assert all(g.fault == "none" for g in first)


def test_a_player_who_plays_beats_one_who_never_does():
    games = play(Side(RandomLegal(0.2), PIGGIES_EQ), Side(WaitPolicy(), PIGGIES_EQ), range(4))
    assert summarize(g.score for g in games).score == 1.0


def test_matches_played_side_by_side_keep_their_own_decks_and_one_match_plays_as_before():
    from clashai.decks import PIGGIES_EQ, REFERENCE_DECKS

    hog, golem = REFERENCE_DECKS["hog_26"], REFERENCE_DECKS["golem"]
    a = RandomLegal(0.3)
    b = WaitPolicy()
    alone = play(Side(a, PIGGIES_EQ), Side(b, hog), [5, 6])
    together = play_matches([(Side(a, PIGGIES_EQ), Side(b, hog))], [5, 6])[0]
    assert alone == together
    # Two matches in one vector: each side A plays only its own deck's cards.
    games = play_matches([(Side(a, PIGGIES_EQ), Side(b, hog)), (Side(a, golem), Side(b, hog))],
                         [5, 6, 7], behaviour=True)
    assert [len(g) for g in games] == [6, 6]
    for deck, played in zip((PIGGIES_EQ, golem), games, strict=True):
        assert [(g.seed, g.a_seat) for g in played] == [(s, x) for s in (5, 6, 7) for x in (0, 1)]
        cards = {c for g in played for c in g.behaviour["cards"]}
        assert cards and cards <= set(deck.cards), (deck.name, cards)
        assert all(g.behaviour["plays_per_min"] > 0 for g in played)
