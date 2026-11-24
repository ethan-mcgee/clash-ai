"""crforge's opening deal, ported to Python so the converter can choose it.

A Ladder match deals each side's deck once, at set-up, side 0 first:
``order = DeckShuffle.order(deck, draw + word)``, where ``draw`` is a draw below ``0x0fffffff`` of
the battle's random source (seeded with the replay's ``rndSeed``) and ``word`` is the side's
``accountID.lo``. Before the shuffles, each side's player data (``hbd``) takes one draw of one
choice, which answers 0 but still advances the source: the shuffle draws are the third and fourth. The first four cards of the order are the hand's slots, the rest the
queue, its front first. A card the tables mark ``OmitFromStartingHand`` (the Mirror, the Elixir
Collector) is held back until four others are drawn.

Ported from crforge's ``BattleRandom``, ``DeckShuffle`` and ``LadderMatch`` (Apache-2.0,
third_party/crforge at the pinned commit), with Java's 32-bit integer arithmetic.
``tests/test_shuffle.py`` checks the port against the engine's own deals.
"""

from __future__ import annotations

from collections.abc import Sequence

SHUFFLE_BOUND = 0x0FFFFFFF
OMIT_FROM_STARTING_HAND = frozenset({"Mirror", "Elixir Collector"})
_WORDS = 624
_FIRST_DRAWS = 4


def _i32(x: int) -> int:
    x &= 0xFFFFFFFF
    return x - (1 << 32) if x & 0x80000000 else x


def _shl(x: int, n: int) -> int:
    return _i32(x << n)


class BattleRandom:
    """The battle's random source: an xorshift on a 32-bit state."""

    def __init__(self, state: int):
        self.state = _i32(state)

    def next(self, bound: int) -> int:
        if bound < 1:
            return 0
        s = -1 if self.state == 0 else self.state
        s = _i32(s ^ _shl(s, 13))
        s = _i32(s ^ (s >> 17))
        s = _i32(s ^ _shl(s, 5))
        self.state = s
        magnitude = _i32(-s) if s < 0 else s  # Java's -MIN_VALUE stays MIN_VALUE
        return (magnitude & 0xFFFFFFFF) % bound


class _Twister:
    """DeckShuffle's Mersenne-twister variant."""

    def __init__(self, seed: int):
        w = [0] * _WORDS
        w[0] = _i32(seed)
        for i in range(1, _WORDS):
            prev = w[i - 1]
            x = _i32(prev ^ (prev >> 30))
            w[i] = _i32(x * 0x6C078965 + 0x6C078965)
        self.words, self.index = w, 0

    def next(self, bound: int) -> int:
        if bound < 1:
            return 0
        w = self.words
        if self.index == 0:
            for i in range(_WORDS):
                nxt = w[0 if i == _WORDS - 1 else i + 1]
                far = w[i + 397 if i < 227 else i - 227]
                y = _i32((w[i] & 0x80000000) | (nxt & 0x7FFFFFFE))
                y = _i32(far ^ (y >> 1))
                w[i] = _i32(y ^ 0x9908B0DF) if nxt & 1 else y
        y = w[self.index]
        y = _i32(y ^ (y >> 11))
        y = _i32(y ^ (_shl(y, 7) & 0x9D2C5680))
        y = _i32(y ^ (_shl(y, 15) & 0xEFC60000))
        y = _i32(y ^ (y >> 18))
        y &= 0x7FFFFFFF
        self.index = (self.index + 1) % _WORDS
        return y % bound


def deal_order(deck: Sequence[str], seed: int) -> list[int]:
    """``DeckShuffle.order``: deck indices in the order dealt, hand first."""
    rest = list(range(len(deck)))
    order: list[int] = []
    held: list[int] = []
    for k in range(len(rest) - 1, -1, -1):
        if deck[rest[k]] in OMIT_FROM_STARTING_HAND:
            held.append(rest.pop(k))
    gen = _Twister(seed)
    while rest and len(order) < _FIRST_DRAWS:
        order.append(rest.pop(gen.next(len(rest))))
    for k in range(len(held) - 1, -1, -1):
        rest.append(held.pop(k))
    while rest:
        order.append(rest.pop(gen.next(len(rest))))
    return order


PLAYER_DATA_DRAWS = 2
"""Draws taken before the shuffles: one per side's player data, each of one choice."""


def shuffle_draws(rnd_seed: int) -> tuple[int, int]:
    """The two sides' shuffle draws of a battle seeded ``rnd_seed``, side 0 first."""
    r = BattleRandom(rnd_seed)
    for _ in range(PLAYER_DATA_DRAWS):
        r.next(1)
    return r.next(SHUFFLE_BOUND), r.next(SHUFFLE_BOUND)


def match_orders(rnd_seed: int, decks: Sequence[Sequence[str]], words: Sequence[int]
                 ) -> list[list[int]]:
    """Each side's dealt order in a battle seeded ``rnd_seed`` with these account words."""
    draws = shuffle_draws(rnd_seed)
    return [deal_order(decks[s], _i32(draws[s] + words[s])) for s in (0, 1)]
