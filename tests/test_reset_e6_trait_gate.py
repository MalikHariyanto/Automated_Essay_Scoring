import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / 'notebooks/multitrait_reset'
E2 = NOTEBOOKS / 'AES_Reset_E2_S3_MultiTrait_TraitHeads.ipynb'
E6 = NOTEBOOKS / 'AES_Reset_E6_S3_MultiTrait_TraitGatedConcat.ipynb'
TRAITS = [(name,) for name in ('isi', 'organisasi', 'kebahasaan', 'mekanik')]


def cells(notebook):
    return {
        cell['metadata']['aes_multitrait_role']: ''.join(cell['source'])
        for cell in json.loads(notebook.read_text())['cells']
        if cell['cell_type'] == 'code'
    }


def model_for(notebook):
    namespace = dict(nn=nn, torch=torch, TRAITS=TRAITS)
    source = cells(notebook)
    for role in ('definition:mean_by_essay', 'definition:SingleScoreModel', 'definition:optimizer_groups'):
        exec(source[role], namespace)

    class Encoder(nn.Embedding):
        def forward(self, input_ids):
            return SimpleNamespace(last_hidden_state=super().forward(input_ids))

    torch.manual_seed(42)
    return namespace['MultiTraitModel']('S3', Encoder(8, 768)), namespace['optimizer_groups']


def test_gate_starts_at_e2_predictions_and_can_learn_different_mix_per_trait():
    e2, _ = model_for(E2)
    e6, optimizer_groups = model_for(E6)
    for name, value in e2.state_dict().items():
        assert torch.equal(value, e6.state_dict()[name]), name
    e2.eval()
    e6.eval()
    args = ({'input_ids': torch.randint(8, (3, 5))}, torch.tensor([0, 0, 1]), 2, torch.randn(2, 1024))
    expected = e2(*args)
    prediction, features = e6(*args, return_features=True)
    assert torch.equal(expected, prediction)
    assert torch.equal(features['gate'], torch.full((2, 4, 2), 0.5))

    prediction.square().mean().backward()
    groups = optimizer_groups(e6, dict(s3_encoder_lr=5e-6, projection_lr=1e-4,
                                      s3_mlp_lr=1e-4, s3_layers_lr=1e-4, gate_lr=1e-4))
    gate_group = next(group for group in groups if group['name'] == 'gate')
    assert gate_group['lr'] == 1e-4
    assert sum(p.numel() for p in gate_group['params']) == 8200
    assert all(p.grad.abs().sum() > 0 for p in gate_group['params'])
    torch.optim.AdamW(groups).step()
    updated = e6(*args, return_features=True)[1]['gate']
    torch.testing.assert_close(updated.sum(dim=-1), torch.ones(2, 4))
    assert not torch.equal(updated[:, 0], updated[:, 1])


def test_e6_keeps_training_recipe_and_runs_quick_export_before_inspection():
    control, current = cells(E2), cells(E6)
    for role in ('hyperparameters', 'definition:fit', 'definition:make_split', 'metrics'):
        assert control[role] == current[role], role
    order = list(current)
    assert order.index('test_execute') < order.index('inspection') < order.index('final_features')
    mirror = ROOT / 'eksperimen/03_tiga_input_multi_trait/notebooks' / E6.name
    assert mirror.read_bytes() == E6.read_bytes()
