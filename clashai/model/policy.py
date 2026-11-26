"""A trained network as an arena policy (see clashai.sim.policies for the interface)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from . import memory as mem
from .net import PolicyNet


def load_net(path: str | Path, device="cpu") -> tuple[PolicyNet, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = PolicyNet.from_description(ckpt["description"]).to(device)
    net.load_state_dict(ckpt["model"])
    return net.eval(), ckpt


class NetPolicy:
    """Samples the network's action at each decision: the type, then the tile for a play.

    Sampling, not the most likely action: an imitation policy waits at almost every single
    decision, so its most likely action is nearly always a wait. Settings that shape the
    distributions (see :func:`type_probs` and :func:`tile_probs`):

    - ``temperature``: the whole type distribution, so also how often it plays at all
    - ``card_temperature``: which card or ability, given that it plays; the chance of playing at
      all is kept
    - ``tile_temperature`` and ``tile_top_p``: where the card goes. Top-p keeps the most likely
      tiles holding that share of the probability and samples among them

    A card or tile temperature of 0 takes the most likely card or tile; whether to play at all is
    still drawn. All at 1 is the network's own distribution. The scores are computed in fp32
    (``PolicyNet.fp32_heads``), so near-ties are ranked exactly. Random numbers come from the
    arena's generator, so games are reproducible.

    A recurrent network (``lstm`` > 0) makes the policy ``stateful``: it keeps the core's state and
    its own history (:mod:`clashai.model.memory`) per seat, keyed by the arena (``keys``: battle
    row * 2 + seat), from :meth:`reset` until the next. One object can play both seats.
    """

    def __init__(self, checkpoint: str | Path, device: str = "cuda", temperature: float = 1.0,
                 name: str | None = None, *, card_temperature: float = 1.0,
                 tile_temperature: float = 1.0, tile_top_p: float = 1.0):
        self.net, ckpt = load_net(checkpoint, device)
        self.net.fp32_heads = True
        self.device, self.temperature = torch.device(device), temperature
        self.card_temperature, self.tile_temperature = card_temperature, tile_temperature
        self.tile_top_p = tile_top_p
        self.name = name or Path(checkpoint).parent.name
        self.step = ckpt.get("step")
        self.stateful = bool(self.net.config.lstm)
        self._state: tuple[torch.Tensor, torch.Tensor] | None = None
        self._hist: dict | None = None

    def reset(self, n: int) -> None:
        """Fresh state for ``n`` battles (keys 0 to 2n - 1), before their first decision."""
        if self.stateful:
            h = self.net.config.lstm
            self._state = (torch.zeros(2 * n, h, device=self.device),
                           torch.zeros(2 * n, h, device=self.device))
            self._hist = mem.start(2 * n, self.device)

    def inputs(self, observations) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The network's header, units and valid from flat observations, on the device."""
        spec = self.net.spec
        obs = np.asarray(observations)
        hi, ei = spec.header_ints, spec.entity_ints
        counts = np.minimum(obs[:, self.count], self.net.config.max_entities)
        width = max(1, int(counts.max()))
        ents = obs[:, hi:hi + width * ei].reshape(len(obs), width, ei)
        valid = np.arange(width)[None, :] < counts[:, None]
        dev = self.device
        return (torch.from_numpy(np.ascontiguousarray(obs[:, :hi])).to(dev),
                torch.from_numpy(np.ascontiguousarray(ents)).to(dev),
                torch.from_numpy(valid).to(dev))

    def bind(self, vector) -> None:
        s, spec = vector.schema, self.net.spec
        assert s.header_ints == spec.header_ints and s.entity_ints == spec.entity_ints
        assert s.actions == spec.actions and len(vector.vocabulary) == spec.vocab, \
            "the engine's layout or vocabulary differs from the model's"
        assert all(list(s.header[k]) == v for k, v in spec.header.items() if k != "reserved"), \
            "header fields moved"
        self.count = s.header["entity_count"][0]

    @torch.no_grad()
    def act(self, observations, masks, rng, keys=None) -> np.ndarray:
        spec, net, dev = self.net.spec, self.net, self.device
        header, ents, valid = self.inputs(observations)
        b = len(header)
        mask = torch.from_numpy(np.asarray(masks, dtype=bool)).to(dev)
        memory = k = None
        if self.stateful:
            if keys is None or self._state is None:
                raise ValueError("a recurrent policy needs reset(n) before play and keys per row")
            k = torch.as_tensor(np.asarray(keys), dtype=torch.long, device=dev)
            memory = {"lanes": b, "first": torch.zeros(b, dtype=torch.bool, device=dev),
                      "state": (self._state[0][k][None], self._state[1][k][None]),
                      **{name: v[k] for name, v in self._hist.items()}}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out = net(header, ents, valid, memory)
            probs = type_probs(out["type_logits"].float(), spec.type_mask(mask), self.temperature,
                               self.card_temperature)
            kind = _sample(probs.cpu().numpy(), rng)
            tile = np.zeros(b, np.int64)
            play = np.flatnonzero((kind >= 1) & (kind <= spec.hand_slots))
            if len(play):
                rows = torch.from_numpy(play).to(dev)
                slot = torch.from_numpy(kind[play] - 1).to(dev)
                probs = tile_probs(net.tile_logits(out, rows, slot).float(),
                                   spec.tile_mask(mask[rows], slot),
                                   self.temperature * self.tile_temperature, self.tile_top_p)
                tile[play] = _sample(probs.cpu().numpy(), rng)
        actions = spec.encode(torch.from_numpy(kind), torch.from_numpy(tile))
        if self.stateful:
            h, c = out["state"]
            self._state[0][k], self._state[1][k] = h[0].float(), c[0].float()
            a = actions.to(dev)
            cards = mem.played_cards(header, a, spec.header["hand_card"], spec.first_ability,
                                     spec.tiles)
            new = mem.advance({name: v[k] for name, v in self._hist.items()}, a, cards)
            for name, v in new.items():
                self._hist[name][k] = v
        return actions.numpy().astype(np.int32)


def type_probs(logits: torch.Tensor, mask: torch.Tensor, temperature: float = 1.0,
               card_temperature: float = 1.0) -> torch.Tensor:
    """The type distribution (wait first, then each slot and ability) under ``mask``.

    ``card_temperature`` reshapes only the choice among the non-wait types: the wait keeps its
    probability, and the rest shares what is left; at 0 it all goes to the most likely open
    card or ability. At 1 this is the plain softmax.
    """
    probs = _softmax(logits, mask, temperature)
    if card_temperature == 1.0:
        return probs
    act = _softmax(logits[:, 1:], mask[:, 1:], temperature * card_temperature)
    # A row whose every play is closed keeps its wait at 1.
    act = act * mask[:, 1:].any(1, keepdim=True)
    return torch.cat([probs[:, :1], (1 - probs[:, :1]) * act], 1)


def tile_probs(logits: torch.Tensor, mask: torch.Tensor, temperature: float = 1.0,
               top_p: float = 1.0) -> torch.Tensor:
    """The tile distribution under ``mask``, sharpened by ``temperature`` and cut to the most
    likely tiles holding ``top_p`` of the probability (always at least the most likely one). At
    temperature 0, the most likely open tile."""
    probs = _softmax(logits, mask, temperature)
    if top_p >= 1.0:
        return probs
    sorted_p, order = probs.sort(-1, descending=True)
    before = sorted_p.cumsum(-1) - sorted_p  # the probability of the tiles ahead of each
    keep = torch.zeros_like(before, dtype=torch.bool).scatter(-1, order, before < top_p)
    probs = probs * keep
    return probs / probs.sum(-1, keepdim=True)


def _softmax(logits: torch.Tensor, mask: torch.Tensor, temperature: float) -> torch.Tensor:
    """Softmax of ``logits / temperature`` under ``mask``; at temperature 0, all on the most
    likely open entry (a row with none open gets all zeros)."""
    if temperature == 0:
        best = logits.masked_fill(~mask, float("-inf")).argmax(-1, keepdim=True)
        return torch.zeros_like(logits).scatter(-1, best, 1.0) * mask.any(-1, keepdim=True)
    return torch.softmax((logits / temperature).masked_fill(~mask, -1e9), -1)


def _sample(probs: np.ndarray, rng) -> np.ndarray:
    cdf = np.cumsum(probs, axis=1)
    u = np.maximum(rng.random(len(probs))[:, None], 1e-9) * cdf[:, -1:]  # never 0: index 0 may be closed
    return np.minimum((cdf < u).sum(1), probs.shape[1] - 1).astype(np.int64)
