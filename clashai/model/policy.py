"""A trained network as an arena policy (see clashai.sim.policies for the interface)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .net import PolicyNet


def load_net(path: str | Path, device="cpu") -> tuple[PolicyNet, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = PolicyNet.from_description(ckpt["description"]).to(device)
    net.load_state_dict(ckpt["model"])
    return net.eval(), ckpt


class NetPolicy:
    """Samples the network's action at each decision: the type, then the tile for a play.

    Sampling, not the most likely action: an imitation policy waits at almost every single
    decision, so its most likely action is nearly always a wait. ``temperature`` scales both
    distributions. Random numbers come from the arena's generator, so games are reproducible.
    """

    def __init__(self, checkpoint: str | Path, device: str = "cuda", temperature: float = 1.0,
                 name: str | None = None):
        self.net, ckpt = load_net(checkpoint, device)
        self.device, self.temperature = torch.device(device), temperature
        self.name = name or Path(checkpoint).parent.name
        self.step = ckpt.get("step")

    def bind(self, vector) -> None:
        s, spec = vector.schema, self.net.spec
        assert s.header_ints == spec.header_ints and s.entity_ints == spec.entity_ints
        assert s.actions == spec.actions and len(vector.vocabulary) == spec.vocab, \
            "the engine's layout or vocabulary differs from the model's"
        assert all(list(s.header[k]) == v for k, v in spec.header.items() if k != "reserved"), \
            "header fields moved"
        self.count = s.header["entity_count"][0]

    @torch.no_grad()
    def act(self, observations, masks, rng) -> np.ndarray:
        spec, net = self.net.spec, self.net
        obs = np.asarray(observations)
        b = len(obs)
        hi, ei = spec.header_ints, spec.entity_ints
        counts = np.minimum(obs[:, self.count], net.config.max_entities)
        width = max(1, int(counts.max()))
        ents = obs[:, hi:hi + width * ei].reshape(b, width, ei)
        valid = np.arange(width)[None, :] < counts[:, None]
        dev = self.device
        header = torch.from_numpy(np.ascontiguousarray(obs[:, :hi])).to(dev)
        mask = torch.from_numpy(np.asarray(masks, dtype=bool)).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out = net(header, torch.from_numpy(np.ascontiguousarray(ents)).to(dev),
                      torch.from_numpy(valid).to(dev))
            type_mask = spec.type_mask(mask)
            logits = (out["type_logits"].float() / self.temperature).masked_fill(~type_mask, -1e9)
            kind = _sample(torch.softmax(logits, -1).cpu().numpy(), rng)
            tile = np.zeros(b, np.int64)
            play = np.flatnonzero((kind >= 1) & (kind <= spec.hand_slots))
            if len(play):
                rows = torch.from_numpy(play).to(dev)
                slot = torch.from_numpy(kind[play] - 1).to(dev)
                tile_logits = net.tile_logits(out, rows, slot).float() / self.temperature
                tile_logits = tile_logits.masked_fill(~spec.tile_mask(mask[rows], slot), -1e9)
                tile[play] = _sample(torch.softmax(tile_logits, -1).cpu().numpy(), rng)
        actions = spec.encode(torch.from_numpy(kind), torch.from_numpy(tile)).numpy()
        return actions.astype(np.int32)


def _sample(probs: np.ndarray, rng) -> np.ndarray:
    cdf = np.cumsum(probs, axis=1)
    u = np.maximum(rng.random(len(probs))[:, None], 1e-9) * cdf[:, -1:]  # never 0: index 0 may be closed
    return np.minimum((cdf < u).sum(1), probs.shape[1] - 1).astype(np.int64)
