"""Behaviour measures, held-out meta decks, recorded slot flags and paired differences."""

import json

import numpy as np
import pytest

from clashai.decks import PIGGIES_EQ, Deck, is_held_out
from clashai.eval.behaviour import FULL_ELIXIR, Behaviour, summarize
from clashai.eval.stats import paired_difference

HEADER = {"tick": [0], "elixir": [1], "hand_card": [2, 3, 4, 5]}
VOCAB = ["Knight", "Log", "Zap", "Golem"]
TILES, FIRST_ABILITY = 10, 41  # 4 slots x 10 tiles: plays are 1..40, abilities 41 and 42


def _header(tick, elixir, hand=(0, 1, 2, 3)):
    return [tick, elixir, *hand]


def test_behaviour_counts_plays_abilities_full_elixir_and_the_first_play_per_game():
    b = Behaviour(HEADER, VOCAB, 4, TILES, FIRST_ABILITY, 5, 2)
    # Game 0: a wait at full elixir, a play of slot 1 (Log) at tick 20, an ability at tick 1195.
    # Game 1: two plays of slot 3 (Golem) at ticks 40 and 45, fed one decision at a time.
    b.add(np.array([0, 0, 0]),
          np.array([_header(0, FULL_ELIXIR), _header(20, 5000), _header(1195, 0)]),
          np.array([0, 1 + 1 * TILES + 3, FIRST_ABILITY]))
    b.add(np.array([1]), np.array([_header(40, 0)]), np.array([1 + 3 * TILES]))
    b.add(np.array([1]), np.array([_header(45, 0)]), np.array([1 + 3 * TILES + 9]))
    g0, g1 = b.game(0), b.game(1)
    assert g0["minutes"] == 1.0  # (1195 + 5) / 1200
    assert g0["plays_per_min"] == 1 and g0["abilities_per_min"] == 1
    assert g0["full_elixir_share"] == pytest.approx(1 / 3)
    assert g0["first_play_s"] == 1.0 and g0["cards"] == {"Log": 1}
    assert g1["first_play_s"] == 2.0 and g1["cards"] == {"Golem": 2}
    b.reset(1)
    assert b.game(0)["first_play_s"] is None and b.game(0)["cards"] == {}


def test_summarize_gives_percentiles_and_card_shares():
    games = [{"plays_per_min": x, "abilities_per_min": 0.0, "full_elixir_share": 0.1,
              "first_play_s": None if x == 0 else x, "cards": {"Log": 1, "Zap": int(x)}}
             for x in range(11)]
    s = summarize(games)
    assert s["games"] == 11
    assert s["plays_per_min"]["p50"] == 5 and s["plays_per_min"]["p10"] == 1
    assert s["first_play_s"]["mean"] == pytest.approx(5.5)  # the game with no play left out
    assert s["card_share"]["Zap"] == pytest.approx(55 / 66)


def test_recorded_slot_flags_replace_the_evolution_flags():
    cards = PIGGIES_EQ.cards
    assert Deck("d", cards, flags=(1, 2, 0, 0, 0, 0, 0, 1)).slots(1) == [1, 2, 0, 0, 0, 0, 0, 1]
    assert Deck("d", cards, ("Log",)).slots(1) == [1 if c == "Log" else 0 for c in cards]
    with pytest.raises(ValueError):
        Deck("d", cards, flags=(1, 2))


def test_the_exit_set_takes_the_most_played_held_out_decks_with_their_usual_flags(tmp_path):
    from clashai.eval.deck_sets import most_played_held_out

    pool = [tuple(f"C{i}{j}" for j in range(8)) for i in range(40)]
    held = [d for d in pool if is_held_out(d)]
    kept = [d for d in pool if not is_held_out(d)]
    assert len(held) >= 2 and kept
    a, b = held[0], held[1]
    lines = []
    for _ in range(3):  # a: 3 times, twice with its first card evolved
        lines.append({"decks": [list(a), list(kept[0])], "slots": [[1] + [0] * 7, [0] * 8]})
    lines[-1]["slots"][0] = [0] * 8
    lines.append({"decks": [list(reversed(b)), list(PIGGIES_EQ.cards)],
                  "slots": [[0] * 7 + [2], [0] * 8]})
    path = tmp_path / "summary.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
    out = most_played_held_out([path], 5)
    assert [e["cards"] for e in out] == [sorted(a), sorted(b)]
    assert out[0]["played"] == 3 and out[0]["flags"][sorted(a).index(a[0])] == 1
    assert out[1]["flags"][sorted(b).index(b[0])] == 2  # flags follow their cards when sorted
    assert out[0]["share"] == pytest.approx(3 / 8)


def test_paired_difference_pairs_games_by_key_and_skips_games_without_a_result():
    a = {("d", 1, 0): 1.0, ("d", 1, 1): 1.0, ("d", 2, 0): 0.5, ("d", 2, 1): None, ("x", 0, 0): 1}
    b = {("d", 1, 0): 0.0, ("d", 1, 1): 1.0, ("d", 2, 0): 0.0, ("d", 2, 1): 0.0}
    mean, low, high, n = paired_difference(a, b)
    assert n == 3 and mean == pytest.approx(0.5)
    sd = np.std([1.0, 0.0, 0.5], ddof=1)
    assert high - mean == pytest.approx(1.959964 * sd / np.sqrt(3))
    assert low < mean < high
