"""A recurrent policy's inputs from its own past: the action it took at the previous decision,
the card that action played, and how many decisions ago it last acted.

One definition, used three ways, so training and play see exactly the same numbers:

- :func:`history` for a whole battle-seat at once (training on recorded battles)
- :func:`advance` one decision at a time (play)
- :func:`played_cards` names the card each action played, from that decision's hand

The previous action is an encoded action (:mod:`clashai.model.spec`), :data:`NO_ACTION` at a
battle's first decision, or :data:`UNKNOWN` where it is hidden (training drops it for some
battle-seats, so the network also learns to play without it). ``since`` counts decisions since
the last card play or ability: 1 right after one, :data:`NEVER` before the first.
"""

from __future__ import annotations

import numpy as np
import torch

NO_ACTION = -1
UNKNOWN = -2
NEVER = 1_000_000
SINCE_EDGES = (1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129)
"""``since`` is bucketed by these lower edges: 1, 2, 3, 4, 5-6, ..., 129 or more and never."""
SINCE_BUCKETS = len(SINCE_EDGES) + 1  # the last for unknown


def played_cards(header, actions, hand_card: list[int], first_ability: int, tiles: int):
    """The vocabulary id of the card each action played, -1 for a wait or an ability.

    ``header`` (T, H) and ``actions`` (T,) as numpy arrays or tensors; ``hand_card`` the header
    columns of the hand slots."""
    is_torch = isinstance(actions, torch.Tensor)
    xp = torch if is_torch else np
    a = actions.long() if is_torch else actions.astype(np.int64)
    play = (a > 0) & (a < first_ability)
    slot = xp.where(play, (a - 1) // tiles, xp.zeros_like(a))
    cols = (torch.as_tensor(hand_card, device=a.device) if is_torch
            else np.asarray(hand_card))[slot]
    if is_torch:
        card = header.long().gather(1, cols[:, None]).squeeze(1)
    else:
        card = np.take_along_axis(header.astype(np.int64), cols[:, None], 1)[:, 0]
    return xp.where(play, card, xp.full_like(card, -1))


def history(actions: np.ndarray, cards: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each decision of one battle-seat, in order from its first: the previous action, the
    card it played and the decisions since the last action. ``cards`` from :func:`played_cards`."""
    t = len(actions)
    prev_action = np.concatenate([[NO_ACTION], actions[:-1]]).astype(np.int64)
    prev_card = np.concatenate([[-1], cards[:-1]]).astype(np.int64)
    idx = np.arange(t)
    last = np.maximum.accumulate(np.where(actions != 0, idx, -1))
    before = np.concatenate([[-1], last[:-1]])
    since = np.where(before >= 0, idx - before, NEVER).astype(np.int64)
    return prev_action, prev_card, since


def start(n: int, device=None) -> dict:
    """The history of ``n`` battle-seats before their first decision."""
    return {"prev_action": torch.full((n,), NO_ACTION, dtype=torch.long, device=device),
            "prev_card": torch.full((n,), -1, dtype=torch.long, device=device),
            "since": torch.full((n,), NEVER, dtype=torch.long, device=device)}


def advance(hist: dict, actions: torch.Tensor, cards: torch.Tensor) -> dict:
    """The history after one more decision, at which each battle-seat took ``actions``."""
    acted = actions != 0
    since = torch.where(acted, torch.ones_like(hist["since"]),
                        (hist["since"] + 1).clamp(max=NEVER))
    return {"prev_action": actions.long(), "prev_card": cards.long(), "since": since}


_EDGES: dict = {}


def since_bucket(since: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
    """``since`` as a bucket index; the last bucket where the previous action is unknown."""
    edges = _EDGES.get(since.device)
    if edges is None:  # made once per device, not copied anew each call (see Spec.columns)
        edges = _EDGES[since.device] = torch.tensor(SINCE_EDGES, device=since.device)
    bucket = torch.bucketize(since, edges, right=True) - 1
    bucket = bucket.clamp(0, len(SINCE_EDGES) - 1)
    return torch.where(prev_action == UNKNOWN, torch.full_like(bucket, SINCE_BUCKETS - 1),
                       bucket)
