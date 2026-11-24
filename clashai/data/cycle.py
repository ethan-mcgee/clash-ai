"""Which opening hands and queue orders explain a player's recorded plays.

A deck's order is its four hand cards and its queue of four. A played card leaves the hand for the
back of the queue, and the queue's front refills the hand. So a play sequence is possible from an
order exactly when every played card is in the hand at its play, simulated card by card. The refill
is treated as instant: the real one-second cooldown only forbids sequences no recorded game holds.

There are C(8,4) * 4! = 1680 distinct orders (the hand's slot order changes nothing here). Early in
a game several fit; once a player has played most of the deck usually one does.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from itertools import combinations, permutations

Key = tuple[frozenset, tuple[int, ...]]
"""An order up to the hand's slot order: (hand cards, queue front first), as deck indices."""


@lru_cache(maxsize=1)
def all_keys() -> tuple[Key, ...]:
    keys = []
    for hand in combinations(range(8), 4):
        rest = [i for i in range(8) if i not in hand]
        for queue in permutations(rest):
            keys.append((frozenset(hand), queue))
    return tuple(keys)


def fits(key: Key, plays: Sequence[int]) -> bool:
    hand, queue = set(key[0]), list(key[1])
    for card in plays:
        if card not in hand:
            return False
        hand.remove(card)
        queue.append(card)
        hand.add(queue.pop(0))
    return True


def feasible_keys(plays: Sequence[int]) -> list[Key]:
    """Every order under which the plays (deck indices, in order) were each in hand."""
    return [k for k in all_keys() if fits(k, plays)]


def key_of(order: Sequence[int]) -> Key:
    return frozenset(order[:4]), tuple(order[4:])
