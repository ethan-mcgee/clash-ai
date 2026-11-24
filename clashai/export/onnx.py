"""A trained policy as an ONNX file, in the format crforge's visualizer plays (export format 1).

    python -m clashai.export.onnx --checkpoint <run>/best.pt [--metadata <rows>/metadata.json]
        [--deck piggies_eq] [--verify <rows folder>] [--bench]

The graph takes what crforge's environment writes, nothing prepared in Python:

- ``observations`` ``(batch, 7024)`` int32, one seat's v4 observation
- ``masks`` ``(batch, 2307)`` bool, its legal actions

and gives:

- ``first_logp`` ``(batch, 7)``: log-probabilities of the wait, each hand slot, each ability
- ``tile_logp`` ``(batch, 4, 576)``: each hand slot's tile log-probabilities
- ``value`` ``(batch,)``: the seat's expected result, -1 to 1

A closed choice's log-probability is -1e9 (crforge reads anything below -1e8 as closed). The
entity rows in use are counted inside the graph, so any batch and any number of units run.

The file's metadata carries ``crforge.export_format`` and ``crforge.metadata`` (the observation
schema and vocabulary the model was trained on, which the visualizer checks before it plays), and
the deck and levels it plays at. Model files stay out of the repository (``.gitignore``).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .. import paths
from ..decks import REFERENCE_DECKS
from ..model.policy import load_net

EXPORT_FORMAT = 1
CLOSED = -1e9
OUTPUTS = ("first_logp", "tile_logp", "value")


class OnnxPolicy(nn.Module):
    """The network between crforge's raw observation and mask and the exported outputs."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, observations: torch.Tensor, masks: torch.Tensor):
        net, spec = self.net, self.net.spec
        hi, ei, k = spec.header_ints, spec.entity_ints, net.config.max_entities
        header = observations[:, :hi]
        table = observations[:, hi:hi + k * ei].reshape(observations.shape[0], k, ei)
        count = header[:, spec.header["entity_count"][0]].clamp(0, k)
        place = torch.arange(k, device=observations.device)
        # Only as many entity rows as the fullest observation uses: a data-dependent width the
        # exported graph keeps (NonZero), so a quiet arena costs little.
        keep = torch.nonzero(place < count.max().clamp(min=1)).squeeze(1)
        entities = table.index_select(1, keep)
        valid = keep.unsqueeze(0) < count.unsqueeze(1)
        out = net(header, entities, valid)
        masks = masks.bool()
        closed = torch.tensor(CLOSED, device=observations.device)
        type_mask = spec.type_mask(masks)
        first = torch.log_softmax(out["type_logits"].float().masked_fill(~type_mask, CLOSED), -1)
        first = torch.where(type_mask, first, closed)
        tile_mask = masks[:, 1:spec.first_ability].reshape(-1, spec.hand_slots, spec.tiles)
        tiles = net.tile_logits_all(out).float().masked_fill(~tile_mask, CLOSED)
        tiles = torch.where(tile_mask, torch.log_softmax(tiles, -1), closed)
        return first, tiles, out["value"].float()


def full_observations(batch: dict, spec) -> np.ndarray:
    """v4 observations rebuilt from a :class:`clashai.train.rows.Rows` batch (padding is zero)."""
    hi, ei = spec.header_ints, spec.entity_ints
    b, width, _ = batch["entities"].shape
    obs = np.zeros((b, 7024), np.int32)
    obs[:, :hi] = batch["header"]
    ents = np.where(batch["valid"][..., None], batch["entities"], 0)
    obs[:, hi:hi + width * ei] = ents.reshape(b, -1)
    return obs


def export(checkpoint: Path, out: Path, metadata: dict | None = None, deck: str = "piggies_eq",
           level: int = 16, example: tuple[np.ndarray, np.ndarray] | None = None) -> Path:
    import onnx

    net, ckpt = load_net(checkpoint, "cpu")
    metadata = metadata or ckpt.get("metadata")
    if metadata is None:
        raise SystemExit(f"{checkpoint} holds no schema metadata: pass --metadata <rows>/metadata.json")
    if metadata["vocabulary"].get("hash") != net.spec.vocab_hash:
        raise SystemExit("the metadata's vocabulary is not the one the model was trained on")
    model = OnnxPolicy(net.float().eval())
    if example is None:
        obs = np.zeros((2, 7024), np.int32)
        obs[:, net.spec.header["entity_count"][0]] = [3, 1]
        masks = np.ones((2, net.spec.actions), bool)
    else:
        obs, masks = example
    out.parent.mkdir(parents=True, exist_ok=True)
    batch_axis = {0: "batch"}
    torch.backends.mha.set_fastpath_enabled(False)  # the fused kernel has no ONNX form
    try:
        with torch.no_grad():
            torch.onnx.export(
                model, (torch.from_numpy(obs), torch.from_numpy(masks)), str(out),
                input_names=["observations", "masks"], output_names=list(OUTPUTS),
                dynamic_axes={n: batch_axis for n in ("observations", "masks", *OUTPUTS)},
                opset_version=17, dynamo=False)
    finally:
        torch.backends.mha.set_fastpath_enabled(True)
    proto = onnx.load(str(out))
    d = REFERENCE_DECKS[deck]
    props = {
        "crforge.export_format": str(EXPORT_FORMAT),
        "crforge.metadata": json.dumps(metadata),
        "crforge.name": out.stem,
        "crforge.deck": ",".join(d.cards),
        "crforge.deck_slots": ",".join(str(f) for f in d.slots(1)),
        "crforge.level": str(level),
        "crforge.tower_level": str(level),
        "clashai.checkpoint": str(checkpoint),
        "clashai.step": str(ckpt.get("step")),
    }
    del proto.metadata_props[:]
    for k, v in props.items():
        proto.metadata_props.add(key=k, value=v)
    onnx.checker.check_model(proto)
    onnx.save(proto, str(out))
    return out


def session(path: Path, device: str = "cuda"):
    import onnxruntime as ort  # after torch, whose CUDA libraries ONNX Runtime then finds

    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda"
                 else ["CPUExecutionProvider"])
    options = ort.SessionOptions()
    options.log_severity_level = 3
    s = ort.InferenceSession(str(path), options, providers=providers)
    if device == "cuda" and "CUDAExecutionProvider" not in s.get_providers():
        # ONNX Runtime falls back to the CPU silently when its CUDA libraries do not load: 1.24.4
        # is built for the CUDA 12 libraries PyTorch 2.5.1 ships; newer builds need CUDA 13.
        raise RuntimeError("ONNX Runtime could not start CUDA: install onnxruntime-gpu==1.24.4")
    return s


def compare(path: Path, checkpoint: Path, obs: np.ndarray, masks: np.ndarray,
            device: str = "cuda") -> dict:
    """Largest differences between ONNX Runtime and PyTorch (fp32) on the same rows, over open
    choices (closed ones are -1e9 in both)."""
    net, _ = load_net(checkpoint, device)
    model = OnnxPolicy(net.float().eval())
    with torch.no_grad():
        want = [t.cpu().numpy() for t in model(torch.from_numpy(obs).to(device),
                                               torch.from_numpy(masks).to(device))]
    got = session(path, device).run(None, {"observations": obs, "masks": masks})
    diff = {}
    for name, w, g in zip(OUTPUTS, want, got, strict=True):
        open_ = w > CLOSED / 10
        assert np.array_equal(open_, g > CLOSED / 10), f"{name}: open choices differ"
        diff[name] = float(np.abs(w[open_] - g[open_]).max())
    first_w, first_g = want[0], got[0]
    diff["same_first_choice"] = float(np.mean(first_w.argmax(1) == first_g.argmax(1)))
    return diff


def bench(path: Path, checkpoint: Path, obs: np.ndarray, masks: np.ndarray,
          batches=(64, 256, 1024), repeats: int = 20) -> list[tuple]:
    """Observations a second from numpy in to numpy out: ONNX Runtime (CUDA, fp32), PyTorch fp32,
    and PyTorch bf16 autocast (as the arena's NetPolicy runs)."""
    net, _ = load_net(checkpoint, "cuda")
    model = OnnxPolicy(net.float().eval())
    sess = session(path, "cuda")
    rows = []
    for b in batches:
        o, m = obs[:b], masks[:b]
        if len(o) < b:
            reps = -(-b // len(o))
            o, m = np.tile(o, (reps, 1))[:b], np.tile(m, (reps, 1))[:b]

        def ort_run():
            sess.run(None, {"observations": o, "masks": m})

        def torch_run(bf16):
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=bf16):
                [t.cpu().numpy() for t in model(torch.from_numpy(o).cuda(),
                                                torch.from_numpy(m).cuda())]

        result = [b]
        for fn in (ort_run, lambda: torch_run(False), lambda: torch_run(True)):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(repeats):
                fn()
            torch.cuda.synchronize()
            result.append(b * repeats / (time.perf_counter() - t))
        rows.append(tuple(result))
    return rows


def sample_rows(rows_folder: Path, n: int, seed: int = 0):
    from ..model.spec import Spec
    from ..train.rows import Rows

    rows = Rows(rows_folder)
    spec = Spec.from_metadata(rows.metadata)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(rows.n, n, replace=False))
    b = rows.gather(idx, 80)
    masks = np.unpackbits(b["masks"], axis=1, count=spec.actions).astype(bool)
    return full_observations(b, spec), masks


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Export a checkpoint to ONNX for crforge")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--metadata", type=Path, help="metadata.json, for checkpoints without it")
    p.add_argument("--out", type=Path)
    p.add_argument("--deck", default="piggies_eq", choices=sorted(REFERENCE_DECKS))
    p.add_argument("--verify", help="a rows folder under <data>/rows to compare outputs on")
    p.add_argument("--bench", action="store_true")
    args = p.parse_args(argv)
    out = args.out or paths.runs_dir() / "onnx" / f"{args.checkpoint.parent.name}.onnx"
    metadata = json.loads(args.metadata.read_text()) if args.metadata else None
    example = sample_rows(paths.data_root() / "rows" / args.verify, 2048) if args.verify else None
    export(args.checkpoint, out, metadata, args.deck,
           example=(example[0][:8], example[1][:8]) if example else None)
    print(f"exported {out} ({out.stat().st_size / 1e6:.1f} MB)")
    if example:
        obs, masks = example
        for device in ("cpu", "cuda"):
            d = compare(out, args.checkpoint, obs[:512], masks[:512], device)
            print(f"ONNX Runtime vs PyTorch on {device}, 512 real rows: largest difference "
                  + ", ".join(f"{k} {v:.2e}" for k, v in d.items() if k != "same_first_choice")
                  + f"; same most likely first choice {d['same_first_choice']:.1%}")
        if args.bench:
            print("observations a second, numpy in to numpy out:")
            print("| batch | ONNX Runtime CUDA fp32 | PyTorch fp32 | PyTorch bf16 |")
            print("|---:|---:|---:|---:|")
            for b, ort_rate, fp32, bf16 in bench(out, args.checkpoint, obs, masks):
                print(f"| {b} | {ort_rate:,.0f} | {fp32:,.0f} | {bf16:,.0f} |")


if __name__ == "__main__":
    main()
