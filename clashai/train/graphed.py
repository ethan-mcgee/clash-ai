"""One training step of the whole-battle trainer, run three ways.

- ``off``: operation by operation from Python, as PyTorch normally runs.
- ``cuda``: the whole step (forward, losses, backward, gradient clipping, AdamW and the weight
  average) recorded once as a CUDA graph and replayed. Python then launches one graph a step
  instead of some thousands of small operations, which is where about a quarter of a step went.
- ``compile``: the forward and backward passes compiled with ``torch.compile`` (Linux only; it
  generates fused kernels and replays them as CUDA graphs itself).

A recorded step replays the same operations on the same memory, so it needs a chunk of fixed
shapes: one packed into a :class:`clashai.train.seq.Profile`. Each profile gets its own recording,
made after :data:`WARMUP` ordinary steps on it. A chunk that fits no profile runs the ordinary
way, on the same weights and optimiser, so nothing about what is learned depends on the mode
beyond floating-point rounding.

What changes with the mode, and why:

- The recurrent core runs a decision at a time (``PolicyNet.core_steps``), not in segments cut
  at restarts, whose number depends on the data.
- The learning rate and the weight average's rate are tensors filled before each step: a number
  from Python would be frozen into the recording.
- Statistics stay on the GPU and are read when they are printed.
"""

from __future__ import annotations

import torch

from .bc import EMA

MODES = ("off", "cuda", "compile")
WARMUP = 3
"""Ordinary steps on a profile before it is recorded: PyTorch asks for a few, so that everything
set up on first use (the optimiser's state, library handles) exists before the recording."""


def ema_rate(ema: EMA, step: int) -> float:
    """The share of the new weights the average takes at ``step`` (see :class:`EMA`)."""
    return 1 - min(ema.decay, (1 + step) / (10 + step))


def _copy(dst, src) -> None:
    """Copies a chunk's tensors into buffers of the same structure, in place."""
    if isinstance(dst, torch.Tensor):
        dst.copy_(src, non_blocking=True)
    elif isinstance(dst, dict):
        for k, v in dst.items():
            _copy(v, src[k])
    elif isinstance(dst, list | tuple):
        for a, b in zip(dst, src, strict=True):
            _copy(a, b)


def _buffers(b, device):
    """Device tensors with a chunk's structure and values; other values are kept as they are."""
    if isinstance(b, torch.Tensor):
        return b.to(device, copy=True)
    if isinstance(b, dict):
        return {k: _buffers(v, device) for k, v in b.items() if k != "snapshot"}
    if isinstance(b, list | tuple) and b and isinstance(b[0], torch.Tensor | list | tuple):
        return type(b)(_buffers(v, device) for v in b)
    return b


class Stepper:
    """Runs training steps in one of :data:`MODES`.

    ``step(b, state, lr=, at=)`` trains on chunk ``b`` (on the device, from ``seq.to``) from the
    lanes' ``state`` and returns ``(stats, state)``: the step's statistics as tensors, and the
    lanes' state after it, detached.
    """

    def __init__(self, net, opt, ema: EMA | None, lanes: int, *, mode: str = "off",
                 clip: float = 1.0, loss_weight: float, value_weight: float, profiles: int = 0):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.net, self.opt, self.ema, self.lanes, self.mode, self.clip = net, opt, ema, lanes, mode, clip
        self.loss_args = {"loss_weight": loss_weight, "value_weight": value_weight}
        self.params = [q for g in opt.param_groups for q in g["params"]]
        self.device = self.params[0].device
        self.recorded: dict[int, _Recording] = {}
        self.warm: dict[int, int] = {}
        self.pool = None  # the recordings' shared memory (see _Recording)
        # With the number of profiles known, none is recorded until every one is warmed up: an
        # ordinary step beside a recording needs as much memory again, which a 24 GB card does
        # not have. Until then every step is ordinary; after, only chunks that fit no profile.
        self.profiles = profiles
        self.counts = {"recorded": 0, "ordinary": 0}
        self.lr = self.rate = None
        if mode != "off":
            net.core_steps = bool(net.config.lstm)
        if mode == "cuda":
            # Read by the recording at each replay; see the module notes.
            self.lr = torch.zeros((), device=self.device)
            self.rate = torch.zeros((), device=self.device)
            for g in opt.param_groups:
                g["lr"] = self.lr
                g["capturable"] = self.device.type == "cuda"  # AdamW's own flag for recording
        self._compiled = None
        if mode == "compile":
            self._compiled = torch.compile(self._forward, mode="reduce-overhead")

    # -- the step's parts ---------------------------------------------------------------------

    def _forward(self, b: dict, h: torch.Tensor, c: torch.Tensor):
        from .seq import chunk_losses

        return chunk_losses(self.net, b, self.lanes, (h, c), sync=False, **self.loss_args)

    def _state(self, state):
        if state is not None:
            return state
        width = self.net.config.lstm or 1
        return tuple(torch.zeros(1, self.lanes, width, device=self.device) for _ in range(2))

    def _update(self, loss: torch.Tensor, rate) -> None:
        """Backward, clip, AdamW and the weight average; ``rate`` a number or a tensor."""
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, self.clip)
        self.opt.step()
        if self.ema is not None:
            with torch.no_grad():
                if isinstance(rate, torch.Tensor):  # a number here would be frozen when recorded
                    for mean, new in zip(self.ema.net.parameters(), self.net.parameters(),
                                         strict=True):
                        mean.lerp_(new, rate)
                else:
                    torch._foreach_lerp_(list(self.ema.net.parameters()),
                                         list(self.net.parameters()), rate)

    def _set_rates(self, lr: float, at: int):
        rate = ema_rate(self.ema, at) if self.ema is not None else 0.0
        if self.lr is None:
            for g in self.opt.param_groups:
                g["lr"] = lr
            return rate
        self.lr.fill_(lr)
        self.rate.fill_(rate)
        return self.rate

    def _ordinary(self, b: dict, state, rate):
        self.opt.zero_grad(set_to_none=True)
        forward = self._compiled if self._compiled is not None and b.get("packed", -1) >= 0 \
            else self._forward
        tensors = {k: v for k, v in b.items() if k not in ("snapshot", "packed")}
        loss, stats, new_state = forward(tensors, *self._state(state))
        self._update(loss, rate)
        return stats, None if new_state is None else tuple(x.detach() for x in new_state)

    # -- a step -------------------------------------------------------------------------------

    def step(self, b: dict, state, *, lr: float, at: int):
        rate = self._set_rates(lr, at)
        profile = b.get("packed", -1)
        if self.mode != "cuda" or profile < 0:
            self.counts["ordinary"] += 1
            return self._ordinary(b, state, rate)
        if profile not in self.recorded:
            if self.warm.get(profile, 0) < WARMUP or not self._all_warm():
                self.warm[profile] = self.warm.get(profile, 0) + 1
                self.counts["ordinary"] += 1
                return self._warm_up(b, state, rate)
            self.recorded[profile] = _Recording(self, b)
        self.counts["recorded"] += 1
        return self.recorded[profile].replay(b, self._state(state))

    def _all_warm(self) -> bool:
        return sum(n >= WARMUP for n in self.warm.values()) >= self.profiles

    def _warm_up(self, b: dict, state, rate):
        """An ordinary step on a side stream, where PyTorch asks for warm-up steps to run."""
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            out = self._ordinary(b, state, rate)
        torch.cuda.current_stream().wait_stream(side)
        return out


class _Recording:
    """One profile's step as a CUDA graph, with the buffers it reads and writes."""

    def __init__(self, owner: Stepper, b: dict):
        self.owner = owner
        self.inputs = _buffers({k: v for k, v in b.items() if k != "packed"}, owner.device)
        self.h, self.c = owner._state(None)
        self.graph = torch.cuda.CUDAGraph()
        # Gradients are made anew inside the recording, in its own memory, and refilled in place
        # by each replay.
        owner.opt.zero_grad(set_to_none=True)
        # All recordings share one memory pool: each holds about a whole step's memory, and one
        # per profile does not fit on a 24 GB card. Sharing is safe because the recordings never
        # run at once and replay() copies out every result before another replay can overwrite
        # it. The memory ordinary steps keep cached is freed first, for the pool to use: running
        # out of memory while recording cannot be recovered from.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if owner.pool is None:
            owner.pool = torch.cuda.graph_pool_handle()
        with torch.cuda.graph(self.graph, pool=owner.pool):
            loss, self.stats, self.state = owner._forward(self.inputs, self.h, self.c)
            owner._update(loss, owner.rate)

    def replay(self, b: dict, state):
        _copy(self.inputs, b)
        self.h.copy_(state[0])
        self.c.copy_(state[1])
        self.graph.replay()
        # Copies: the recording's own outputs are overwritten by its next replay.
        return ({k: v.clone() for k, v in self.stats.items()},
                None if self.state is None else tuple(x.detach().clone() for x in self.state))
