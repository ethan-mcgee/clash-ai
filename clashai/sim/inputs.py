"""What the model reads: crforge adapters over the field export, behind the fairness guard.

The policy's inputs are only what a player could see on screen:

- ``v4``: the seat's v4 observation (board units, towers, own hand and elixir, both sides' plays)
- ``enemy_cycle``: the opponent's card queue and the cards known to be in their hand, tracked from
  the plays shown, as a player tracks them
- ``elixir_lead``: own elixir less the opponent's as a player counts it
- ``unit_age``, ``attack_cooldown``, ``enemy_hit_points``: per-unit facts a player can see or
  infer from what was shown
- ``own_plays``: the seat's own last plays

:func:`policy_inputs` is built with ``forbid_hidden=True``: binding it fails if any adapter reads
a field the seat is not shown. :func:`critic_inputs` adds the other seat's view, which only the
value network may read, during training (an asymmetric actor-critic); it is never given to the
policy.
"""

from __future__ import annotations


def policy_inputs():
    from crforge_env.adapters import (AdapterSet, AttackCooldown, ElixirLead, EnemyCycle,
                                      EnemyHitPoints, OwnPlays, UnitAge, V4Adapter)

    return AdapterSet([V4Adapter(), EnemyCycle(), ElixirLead(), UnitAge(), AttackCooldown(),
                       EnemyHitPoints(), OwnPlays()], forbid_hidden=True)


def critic_inputs():
    from crforge_env.adapters import AdapterSet, OtherSeat, V4Adapter

    return AdapterSet([*policy_inputs().adapters, OtherSeat(V4Adapter())])
