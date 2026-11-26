"""Win rates with honest uncertainty.

A score counts a win as 1, a draw as 1/2 and a loss as 0. Its 95% interval is Wilson's score
interval, which stays inside [0, 1] and behaves at small samples and extreme rates, where the
normal approximation does not. Games cut short with no result (truncated or faulted) are counted
apart and left out of the score.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


def wilson(successes: float, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for ``successes`` out of ``n`` (95% by default)."""
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass(frozen=True)
class Summary:
    games: int
    wins: int
    draws: int
    losses: int
    unfinished: int
    score: float
    low: float
    high: float

    def row(self) -> str:
        return (f"{self.wins}-{self.draws}-{self.losses} | {self.score:.1%} | "
                f"{self.low:.1%}-{self.high:.1%}")


def paired_difference(a: dict, b: dict, z: float = 1.959964) -> tuple[float, float, float, int]:
    """``a`` minus ``b`` over the games both played with a result: mean and 95% interval of the
    per-game differences, and the number of pairs. ``a`` and ``b`` map a game's key (such as deck,
    seed and seat) to its score."""
    d = [a[k] - b[k] for k in a.keys() & b.keys() if a[k] is not None and b[k] is not None]
    n = len(d)
    if n < 2:
        return float("nan"), float("nan"), float("nan"), n
    mean = sum(d) / n
    half = z * math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1) / n)
    return mean, mean - half, mean + half, n


def summarize(scores: Iterable[float | None]) -> Summary:
    """Summary of per-game scores (1, 0.5, 0, or None for a game with no result)."""
    scores = list(scores)
    done = [x for x in scores if x is not None]
    wins = sum(1 for x in done if x == 1)
    draws = sum(1 for x in done if x == 0.5)
    losses = sum(1 for x in done if x == 0)
    n = len(done)
    total = wins + 0.5 * draws
    low, high = wilson(total, n)
    return Summary(len(scores), wins, draws, losses, len(scores) - n,
                   total / n if n else float("nan"), low, high)
