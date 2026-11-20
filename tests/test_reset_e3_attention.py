"""Technical E3 checks using synthetic tensors, not student scores or pretrained results."""

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "notebooks/multitrait_reset"
E2 = FOLDER / "AES_Reset_E2_S3_MultiTrait_TraitHeads.ipynb"
E3 = FOLDER / "AES_Reset_E3_S3_MultiTrait_TraitAttention.ipynb"
TRAITS = [(name,) for name in ("isi", "organisasi", "kebahasaan", "mekanik")]


def cells(path):
    return {
        c["metadata"].get("aes_multitrait_role"): "".join(c["source"])
        for c in json.loads(path.read_text())["cells"]
        if c["cell_type"] == "code"
    }


def namespace(path=E3):
    source = cells(path)
    ns = dict(torch=torch, nn=nn, math=math, TRAITS=TRAITS)
    for role in (
        "definition:mean_by_essay",
        "definition:SingleScoreModel",
        "definition:optimizer_groups",
    ):
        exec(source[role], ns)
    return ns


class SyntheticEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 768)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_mask_excludes_context_and_padding_on_either_side():
    ns = namespace()
    mapping = torch.tensor([0, 0, 1])
    details = [
        {"answer_positions": [[2, 3], [2]]},
        {"answer_positions": [[1, 2]]},
    ]
    right = {"attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 0]])}
    mask = ns["answer_token_mask"](right, mapping, details)
    assert mask.tolist() == [
        [False, False, True, True],
        [False, False, True, False],
        [False, True, True, False],
    ]
    left = {"attention_mask": torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1]])}
    mask = ns["answer_token_mask"](left, mapping, details, "left")
    assert mask.tolist() == [
        [False, False, True, True],
        [False, False, False, True],
        [False, False, True, True],
    ]
    with pytest.raises(ValueError, match="jawaban"):
        ns["answer_token_mask"](right, torch.tensor([1, 0, 1]), details)


def test_attention_is_normalized_masked_and_differentiable():
    ns = namespace()
    pool = ns["TraitAttentionPool"]()
    hidden = torch.randn(2, 6, 768, requires_grad=True)
    mask = torch.tensor([[0, 0, 1, 1, 1, 0], [0, 1, 1, 0, 0, 0]], dtype=torch.bool)
    pooled, weights = pool(hidden, mask)
    assert pooled.shape == (2, 4, 768)
    assert torch.equal(weights.masked_select(~mask[:, None, :]), torch.zeros(28))
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4))
    changed = hidden.detach().clone()
    changed[~mask] = 10000
    torch.testing.assert_close(pool(changed, mask)[0], pooled)
    pooled[:, 0].square().mean().backward()
    assert pool.queries.grad[0].abs().sum() > 0
    assert pool.queries.grad[1:].abs().sum() == 0
    assert hidden.grad[mask].abs().sum() > 0
    assert hidden.grad[~mask].abs().sum() == 0
    with pytest.raises(ValueError, match="token jawaban"):
        pool(hidden, torch.zeros_like(mask))


def test_common_initialization_rng_and_optimizer_match_e2():
    e3, e2 = namespace(E3), namespace(E2)
    torch.manual_seed(42)
    model = e3["MultiTraitModel"]("S3", SyntheticEncoder())
    rng = torch.get_rng_state()
    torch.manual_seed(42)
    control = e2["MultiTraitModel"]("S3", SyntheticEncoder())
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in control.state_dict().items():
        assert torch.equal(value, model.state_dict()[name]), name
    assert (
        sum(p.numel() for p in model.parameters())
        - sum(p.numel() for p in control.parameters())
        == 3072
    )
    assert sum(p.numel() for p in model.head.parameters()) == 262660
    cfg = dict(
        s3_encoder_lr=5e-6, projection_lr=1e-4, s3_mlp_lr=1e-4, s3_layers_lr=1e-4
    )
    groups = e3["optimizer_groups"](model, cfg)
    query_groups = [
        g for g in groups if any(p is model.trait_pool.queries for p in g["params"])
    ]
    assert len(query_groups) == 1
    assert query_groups[0]["name"] == "pooling_norm" and query_groups[0]["lr"] == 1e-4


def test_multiwindow_forward_and_gradient_to_the_matching_head_and_query():
    ns = namespace()
    model = ns["MultiTraitModel"]("S3", SyntheticEncoder()).eval()
    windows = dict(
        input_ids=torch.randint(0, 32, (3, 6)),
        attention_mask=torch.ones(3, 6, dtype=torch.long),
    )
    mask = torch.tensor([[0, 0, 1, 1, 1, 0]] * 3, dtype=torch.bool)
    prediction, features = model(
        windows, torch.tensor([0, 0, 1]), 2, torch.randn(2, 1024), mask, True
    )
    assert prediction.shape == (2, 4)
    assert features["roberta"].shape == (2, 4, 768)
    assert features["concat"].shape == (2, 4, 1024)
    prediction[:, 1].sum().backward()
    assert model.trait_pool.queries.grad[1].abs().sum() > 0
    assert model.trait_pool.queries.grad[[0, 2, 3]].abs().sum() == 0
    assert model.roberta.embedding.weight.grad.abs().sum() > 0
    for i, (trait, *_) in enumerate(TRAITS):
        norm = model.head.heads[trait][0].weight.grad.abs().sum()
        assert (norm > 0) if i == 1 else (norm == 0)


def test_notebook_identity_copies_and_unchanged_training_recipe():
    notebook = json.loads(E3.read_text())
    code_hash = notebook["metadata"]["aes_standalone"]["code_sha256"]
    code = "\n".join(
        "".join(c["source"]).replace(code_hash, "__RESET_E3_CODE_SHA256__")
        for c in notebook["cells"]
        if c["cell_type"] == "code"
    )
    assert hashlib.sha256(code.encode()).hexdigest() == code_hash
    copy = ROOT / "eksperimen/03_tiga_input_multi_trait/notebooks" / E3.name
    assert copy.read_bytes() == E3.read_bytes()
    base, current = cells(E2), cells(E3)
    for role in (
        "hyperparameters",
        "definition:fit",
        "definition:optimizer_groups",
        "definition:make_split",
        "metrics",
    ):
        assert base[role] == current[role], role
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and not cell["outputs"]
