"""Synthetic technical checks for reset E4; no pretrained accuracy claims."""

import ast
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "notebooks/multitrait_reset"
E2 = FOLDER / "AES_Reset_E2_S3_MultiTrait_TraitHeads.ipynb"
E4 = FOLDER / "AES_Reset_E4_S3_MultiTrait_InterTraitAttention.ipynb"
TRAITS = [(name,) for name in ("isi", "organisasi", "kebahasaan", "mekanik")]


def cells(path):
    return {
        c["metadata"].get("aes_multitrait_role"): "".join(c["source"])
        for c in json.loads(path.read_text())["cells"]
        if c["cell_type"] == "code"
    }


def namespace(path):
    source = cells(path)
    ns = dict(torch=torch, nn=nn, math=math, TRAITS=TRAITS)
    for role in (
        "definition:mean_by_essay",
        "definition:SingleScoreModel",
        "definition:optimizer_groups",
    ):
        exec(source[role], ns)
    return ns


def make_pair():
    current, base = namespace(E4), namespace(E2)
    torch.manual_seed(42)
    model = current["MultiTraitModel"]("S3", nn.Linear(4, 4)).eval()
    rng = torch.get_rng_state()
    torch.manual_seed(42)
    control = base["MultiTraitModel"]("S3", nn.Linear(4, 4)).eval()
    assert torch.equal(rng, torch.get_rng_state())
    return current, model, control


def test_e2_initialization_predictions_rng_and_parameter_budget():
    _, model, control = make_pair()
    for name, original in control.state_dict().items():
        actual = model.state_dict()[name]
        if name.startswith("head.") and name.endswith(".3.weight"):
            torch.testing.assert_close(actual[:, :64], original, rtol=0, atol=0)
            assert actual[:, 64:].abs().sum() == 0
        else:
            assert torch.equal(actual, original), name
    fusion = torch.randn(5, 1024)
    torch.testing.assert_close(model.head(fusion), control.head(fusion))
    assert sum(p.numel() for p in model.head.parameters()) == 262916
    assert (
        sum(p.numel() for p in model.parameters())
        - sum(p.numel() for p in control.parameters())
        == 256
    )


def test_attention_reads_other_traits_within_each_answer_and_is_normalized():
    _, model, _ = make_pair()
    fusion = torch.randn(5, 1024)
    prediction, features = model.head(fusion, return_features=True)
    weights = features["inter_trait_attention"]
    hidden = features["trait_hidden"]
    assert weights.shape == (5, 4, 4)
    assert weights.diagonal(dim1=1, dim2=2).abs().sum() == 0
    torch.testing.assert_close(weights.sum(-1), torch.ones(5, 4))
    assert features["trait_context"].shape == (5, 4, 64)
    assert features["trait_concat"].shape == (5, 4, 128)
    expected = torch.matmul(hidden, hidden.transpose(1, 2)) / math.sqrt(64)
    expected.masked_fill_(torch.eye(4, dtype=torch.bool)[None], float("-inf"))
    torch.testing.assert_close(weights, expected.softmax(-1))
    torch.testing.assert_close(features["trait_context"], weights @ hidden)
    # Other answers cannot affect this answer's features or attention.
    single_prediction, single = model.head(fusion[:1], return_features=True)
    torch.testing.assert_close(prediction[:1], single_prediction)
    torch.testing.assert_close(weights[:1], single["inter_trait_attention"])


def test_context_weights_learn_and_then_connect_gradients_between_traits():
    ns, model, _ = make_pair()
    optimizer = torch.optim.SGD(model.head.parameters(), lr=0.01)
    fusion = torch.randn(3, 1024)
    model.head(fusion)[:, 1].sum().backward()
    organization = model.head.heads["organisasi"]
    assert organization[3].weight.grad[:, 64:].abs().sum() > 0
    optimizer.step()
    assert organization[3].weight[:, 64:].abs().sum() > 0
    optimizer.zero_grad(set_to_none=True)
    model.head(fusion)[:, 1].sum().backward()
    for trait, head in model.head.heads.items():
        assert head[0].weight.grad.abs().sum() > 0, trait
    cfg = dict(
        s3_encoder_lr=5e-6, projection_lr=1e-4, s3_mlp_lr=1e-4, s3_layers_lr=1e-4
    )
    groups = ns["optimizer_groups"](model, cfg)
    head_group = next(g for g in groups if g["name"] == "head")
    assert head_group["lr"] == 1e-4
    assert {id(p) for p in head_group["params"]} == {
        id(p) for p in model.head.parameters()
    }


def test_mixed_precision_attention_stays_finite_and_backpropagates():
    _, model, _ = make_pair()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        prediction, features = model.head(torch.randn(2, 1024), return_features=True)
    assert features["inter_trait_attention"].dtype == torch.float32
    assert torch.isfinite(prediction).all()
    prediction.float().square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.head.parameters()
    )


def test_notebook_identity_copies_and_unchanged_e2_training_recipe():
    notebook = json.loads(E4.read_text())
    code_hash = notebook["metadata"]["aes_standalone"]["code_sha256"]
    code = "\n".join(
        "".join(c["source"]).replace(code_hash, "__RESET_E4_CODE_SHA256__")
        for c in notebook["cells"]
        if c["cell_type"] == "code"
    )
    assert hashlib.sha256(code.encode()).hexdigest() == code_hash
    copy = ROOT / "eksperimen/03_tiga_input_multi_trait/reset/notebooks" / E4.name
    assert copy.read_bytes() == E4.read_bytes()
    base, current = cells(E2), cells(E4)
    for role in (
        "hyperparameters",
        "definition:fit",
        "definition:optimizer_groups",
        "definition:Engine",
        "definition:make_split",
        "metrics",
    ):
        assert ast.dump(ast.parse(base[role])) == ast.dump(
            ast.parse(current[role])
        ), role
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and not cell["outputs"]
