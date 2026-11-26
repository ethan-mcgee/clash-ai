"""Memory for recurrent networks in PPO: carried game by game in the rollout, and rebuilt for
sequences in the update.

A recurrent network (``Config.lstm``) reads each decision with its core's state and the seat's
own history (:mod:`clashai.model.memory`). In the rollout every battle-seat a network plays has a
place in a :class:`Seats`, read before the decision and written after it. The update trains on
sequences of consecutive decisions of one battle, from the state the rollout had at each
sequence's start (:func:`sequence_rows`, :func:`batch_memory`).

A network with no core gets ``None`` for its memory everywhere, and nothing here changes what it
computes.
"""

from __future__ import annotations

import numpy as np
import torch

from ..model import memory as mem


class Seats:
    """The core state and own history of ``n`` battle-seats, for networks whose core is ``width``
    wide (0: networks without one; the history is still kept, for a critic that reads it)."""

    def __init__(self, n: int, width: int, spec, device):
        self.n, self.width, self.spec, self.device = n, width, spec, device
        self.h = torch.zeros(n, max(1, width), device=device)
        self.c = torch.zeros(n, max(1, width), device=device)
        self.hist = mem.start(n, device)
        self.fresh = torch.ones(n, dtype=torch.bool, device=device)

    def _rows(self, rows):
        if rows is None:
            return slice(None), self.n
        r = torch.as_tensor(np.asarray(rows), dtype=torch.long, device=self.device)
        return r, len(r)

    def memory(self, net, rows=None) -> dict | None:
        """What ``net`` needs for one decision of each of ``rows`` (default: all), each a lane of
        its own; None for a network without a core."""
        if not net.config.lstm:
            return None
        if net.config.lstm != self.width:
            raise ValueError(f"these seats hold a core of {self.width}, the network's is "
                             f"{net.config.lstm}")
        r, k = self._rows(rows)
        return {"lanes": k, "first": self.fresh[r].clone(),
                "state": (self.h[r][None], self.c[r][None]),
                **{name: v[r] for name, v in self.hist.items()}}

    def update(self, state, header: torch.Tensor, actions: torch.Tensor, rows=None) -> None:
        """After a decision: ``state`` is the network's (None without a core), ``header`` what it
        saw and ``actions`` what the seats did."""
        r, _ = self._rows(rows)
        if state is not None:
            self.h[r], self.c[r] = state[0][0].float(), state[1][0].float()
        spec = self.spec
        a = actions.to(self.device).long()
        cards = mem.played_cards(header, a, spec.header["hand_card"], spec.first_ability,
                                 spec.tiles)
        new = mem.advance({name: v[r] for name, v in self.hist.items()}, a, cards)
        for name, v in new.items():
            self.hist[name][r] = v
        self.fresh[r] = False

    def restart(self, rows) -> None:
        """The seats of ``rows`` start a new game at their next decision."""
        r, k = self._rows(rows)
        if not k:
            return
        self.h[r] = 0
        self.c[r] = 0
        for name, v in mem.start(k, self.device).items():
            self.hist[name][r] = v
        self.fresh[r] = True


def join(a: dict | None, b: dict | None) -> dict | None:
    """Two memories as one, ``a``'s lanes first, for one forward pass over both."""
    if a is None or b is None:
        if a is not b:
            raise ValueError("both memories or neither: one network reads them together")
        return None
    out = {"lanes": a["lanes"] + b["lanes"],
           "state": tuple(torch.cat([x, y], 1) for x, y in zip(a["state"], b["state"], strict=True))}
    for name in ("first", "prev_action", "prev_card", "since"):
        out[name] = torch.cat([a[name], b[name]])
    return out


def split_state(state, k: int):
    """A joined pass's state as its first ``k`` lanes' and the rest's."""
    if state is None:
        return None, None
    return tuple(s[:, :k] for s in state), tuple(s[:, k:] for s in state)


def sequence_rows(sequences: np.ndarray, length: int, n: int) -> np.ndarray:
    """The rows of whole sequences, lane-major, as indices into a rollout laid out step by step
    (row = step * n + battle). Sequence ``j * n + i`` is battle ``i``'s steps ``j * length`` to
    ``(j + 1) * length - 1``."""
    j, i = np.divmod(np.asarray(sequences), n)
    steps = j[:, None] * length + np.arange(length)[None, :]
    return (steps * n + i[:, None]).ravel()


def batch_memory(flat: dict, rows: np.ndarray, lanes: int, state, device) -> dict:
    """The memory of ``lanes`` sequences whose rows are ``rows`` (lane-major) of a rollout's
    flat ``first``, ``prev_action``, ``prev_card`` and ``since``; ``state`` (h, c), each
    (lanes, width), is the core's before each sequence's first row."""
    def t(name):
        return torch.from_numpy(flat[name][rows]).to(device)

    return {"lanes": lanes, "first": t("first"),
            "state": tuple(torch.as_tensor(s, device=device)[None] for s in state),
            "prev_action": t("prev_action"), "prev_card": t("prev_card"), "since": t("since")}
