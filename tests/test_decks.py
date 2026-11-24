import numpy as np
import pytest

from clashai.decks import PIGGIES_EQ, REFERENCE_DECKS, Deck, is_held_out
from clashai.sim.env import make_vector


def test_piggies_eq_is_the_deck_we_train():
    assert set(PIGGIES_EQ.cards) == {"RoyalHogs", "ArcherQueen", "Cannon", "Skeletons",
                                     "IceSpirits", "Log", "Earthquake", "RoyalDelivery"}
    assert set(PIGGIES_EQ.evolutions) == {"RoyalHogs", "Cannon"}


def test_a_deck_is_eight_different_cards_with_at_most_two_evolutions():
    with pytest.raises(ValueError):
        Deck("short", ("Knight",) * 8)
    with pytest.raises(ValueError):
        Deck("evo", tuple(PIGGIES_EQ.cards), evolutions=("Zap",))


def test_the_held_out_split_is_fixed_and_ignores_card_order():
    deck = REFERENCE_DECKS["hog_26"]
    assert is_held_out(deck) == is_held_out(list(reversed(deck.cards)))
    held = [d.name for d in REFERENCE_DECKS.values() if is_held_out(d)]
    assert 0 < len(held) < len(REFERENCE_DECKS), held


def test_the_held_out_split_holds_out_about_a_fifth_of_decks():
    rng = np.random.default_rng(0)
    pool = [f"card{i}" for i in range(120)]
    decks = [list(rng.choice(pool, 8, replace=False)) for _ in range(5000)]
    assert 0.18 < np.mean([is_held_out(d) for d in decks]) < 0.22


@pytest.mark.engine
def test_every_reference_deck_starts_a_battle_with_its_evolutions():
    names = list(REFERENCE_DECKS)
    decks = [REFERENCE_DECKS[n] for n in names]
    with make_vector(len(decks), 2) as v:
        s = v.schema
        piggies = [list(PIGGIES_EQ.cards)] * len(decks)
        v.reset(list(range(len(decks))), piggies, [list(d.cards) for d in decks],
                [PIGGIES_EQ.slots(s.slot_evolution)] * len(decks),
                [d.slots(s.slot_evolution) for d in decks])
        for _ in range(100):
            v.step()
        assert v.fault_kind_counts() == {"refusal": 0, "bug": 0, "env_bug": 0}
        # Each deck's evolution slots are the ones the deck names.
        for i, d in enumerate(decks):
            o = v.observations[i, 1]
            deck = [v.vocabulary[int(o[j])] for j in s.header["deck"]]
            flags = [int(o[j]) for j in s.header["deck_slot"]]
            evolved = {c for c, f in zip(deck, flags, strict=True) if f & s.slot_evolution}
            assert evolved == set(d.evolutions), names[i]
