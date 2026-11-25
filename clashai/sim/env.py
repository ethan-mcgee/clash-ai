"""Building crforge vectors with this project's defaults."""

from __future__ import annotations

from collections.abc import Sequence

from .. import paths

DEFAULT_LEVEL = 16
"""Card and tower level for both sides: 99% of the cards in the November replays are level 16, the
maximum (crforge counts levels from 1 across all rarities, as the game shows them)."""

DECISION_TICKS = 5
"""A decision every 5 ticks, 0.25 s of game time."""


def make_vector(count: int, threads: int | None = None, *, decision_ticks: int = DECISION_TICKS,
                level: int = DEFAULT_LEVEL, tower_level: int | None = None, fields: bool = False,
                max_ticks: int | None = None, jvm_args: Sequence[str] = ()):
    """A crforge ``VectorBattle`` of ``count`` Ladder battles on this project's tables.

    Sets ``JAVA_HOME`` to JDK 21 first (:func:`paths.ensure_java`); ``threads`` defaults to one
    JVM thread per physical core.
    """
    paths.ensure_java()
    from crforge_env import LadderConfig, VectorBattle  # no JVM starts until a vector is made

    config = LadderConfig(decision_ticks=decision_ticks, level=level,
                          tower_level=level if tower_level is None else tower_level,
                          max_ticks=max_ticks)
    return VectorBattle(count, threads or paths.default_threads(), install=paths.install_dir(),
                        tables=paths.tables_dir(), config=config, fields=fields,
                        jvm_args=[*paths.jvm_args(), *jvm_args])
