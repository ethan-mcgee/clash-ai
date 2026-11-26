"""ONNX export: the exported graph gives what PyTorch gives, on real rows, in crforge's format."""

import json

import numpy as np
import pytest
import torch

from clashai import paths
from clashai.model.net import Config, PolicyNet
from clashai.model.spec import Spec

ROWS = paths.data_root() / "rows" / "piggies_eq-screen-team0-v0-t0-standard-w0.25"
pytestmark = pytest.mark.skipif(not (ROWS / "rows.json").exists(),
                                reason="training rows not built (python -m clashai.train.rows)")
onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")


def test_an_exported_policy_matches_pytorch_and_carries_crforges_metadata(tmp_path):
    from clashai.export.onnx import CLOSED, EXPORT_FORMAT, compare, export, sample_rows

    metadata = json.loads((ROWS / "metadata.json").read_text())
    spec = Spec.from_metadata(metadata)
    torch.manual_seed(0)
    net = PolicyNet(spec, Config(d=64, layers=2, heads=4, grid_channels=16))
    ckpt = tmp_path / "tiny.pt"
    torch.save({"model": net.state_dict(), "description": net.describe(), "step": 0,
                "metadata": metadata}, ckpt)
    obs, masks = sample_rows(ROWS, 96)
    path = export(ckpt, tmp_path / "tiny.onnx", example=(obs[:4], masks[:4]))

    d = compare(path, ckpt, obs, masks, device="cpu")  # other batch sizes and unit counts too
    assert d["first_logp"] < 1e-4 and d["tile_logp"] < 1e-4 and d["value"] < 1e-4
    assert d["same_first_choice"] == 1.0

    props = {p.key: p.value for p in onnx.load(str(path)).metadata_props}
    assert props["crforge.export_format"] == str(EXPORT_FORMAT)
    assert json.loads(props["crforge.metadata"])["schema"] == metadata["schema"]
    assert props["crforge.deck"].split(",")[0] == "RoyalHogs"
    assert props["crforge.deck_slots"] == "1,0,1,0,0,0,0,0"

    from clashai.export.onnx import session

    first, tiles, value = session(path, "cpu").run(None, {"observations": obs, "masks": masks})
    assert first.shape == (96, spec.types) and tiles.shape == (96, spec.hand_slots, spec.tiles)
    open_first = spec.type_mask(torch.from_numpy(masks)).numpy()
    assert np.array_equal(first > CLOSED / 10, open_first)
    np.testing.assert_allclose(np.exp(first).sum(1), 1, rtol=1e-4)


def test_a_model_with_attention_biases_exports_and_matches_pytorch(tmp_path):
    from clashai.export.onnx import compare, export, sample_rows

    metadata = json.loads((ROWS / "metadata.json").read_text())
    spec = Spec.from_metadata(metadata)
    torch.manual_seed(0)
    net = PolicyNet(spec, Config(d=64, layers=2, heads=4, grid_channels=16, attn_bias=True,
                                 relations=True))
    with torch.no_grad():  # weights away from their zero start, so the biases matter
        for mlp in net.pair_mlp:
            mlp[2].weight.normal_(0, 0.5)
        net.relation_bias.normal_(0, 0.5)
    ckpt = tmp_path / "biased.pt"
    torch.save({"model": net.state_dict(), "description": net.describe(), "step": 0,
                "metadata": metadata}, ckpt)
    obs, masks = sample_rows(ROWS, 64)
    path = export(ckpt, tmp_path / "biased.onnx", example=(obs[:4], masks[:4]))
    d = compare(path, ckpt, obs, masks, device="cpu")
    assert d["first_logp"] < 1e-4 and d["tile_logp"] < 1e-4 and d["value"] < 1e-4
