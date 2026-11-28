"""E33 technical fixtures only; no claim about pretrained AES performance."""
import ast
import hashlib
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'notebooks/multitrait_reset'
E31 = FOLDER / 'AES_Reset_E31_S3_PerTrait_EMA.ipynb'
E33 = FOLDER / 'AES_Reset_E33_S3_PerTrait_EMA_TraitGate.ipynb'


def cells(path):
    return {c['metadata']['aes_multitrait_role']: c['source']
            for c in json.loads(path.read_text())['cells'] if c['cell_type'] == 'code'}


def namespace(path):
    ns = {}
    source = cells(path)
    for role in ('imports', 'columns', 'definition:mean_by_essay',
                 'definition:SingleScoreModel', 'definition:optimizer_groups'):
        exec(source[role], ns)
    return ns


def test_control_equivalence_gate_gradients_capture_and_roundtrip():
    torch.set_num_threads(1)
    control, current = namespace(E31), namespace(E33)
    torch.manual_seed(44)
    before = control['PromptAwareHeads'](0.)
    rng_before = torch.random.get_rng_state()
    torch.manual_seed(44)
    after = current['PromptAwareHeads'](0.)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    for key, value in before.state_dict().items():
        assert torch.equal(value, after.state_dict()[key])
    assert sum(p.numel() for p in after.parameters()) - sum(p.numel() for p in before.parameters()) == 260
    states = [torch.randn(7, 64, requires_grad=True) for _ in range(4)]
    # Equivalence also holds once the shared residual carries nonzero information.
    with torch.no_grad():
        before.trait_residual.weight.normal_(0, .1)
        after.trait_residual.weight.copy_(before.trait_residual.weight)
    after.begin_gate_capture()
    prediction = after.finish(states)
    torch.testing.assert_close(prediction, before.finish(states), rtol=0, atol=0)
    report = after.end_gate_capture()
    assert all(r['n'] == 7 and r['gate mean'] == 1 and r['gate std'] == 0 for r in report)
    prediction.square().mean().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in after.trait_gates.parameters())
    optimizer = torch.optim.AdamW(after.parameters(), lr=1e-4)
    optimizer.step()
    after.begin_gate_capture()
    after.finish([x[:3] for x in states]); after.finish([x[3:] for x in states])
    report = after.end_gate_capture()
    assert all(r['n'] == 7 and 0 < r['gate min'] <= r['gate max'] < 2 for r in report)
    assert any(r['gate std'] > 0 for r in report)
    restored = current['PromptAwareHeads'](0.)
    restored.load_state_dict(after.state_dict())
    torch.testing.assert_close(restored.finish(states), after.finish(states))


def test_notebook_identity_and_controlled_recipe():
    source, before = cells(E33), cells(E31)
    for role in ('hyperparameters', 'configuration_defaults', 'definition:fit',
                 'definition:optimizer_groups', 'definition:Engine', 'definition:make_split', 'metrics'):
        assert source[role] == before[role]
    n = json.loads(E33.read_text())
    fp = n['metadata']['aes_standalone']['code_sha256']
    codes = []
    for c in n['cells']:
        if c['cell_type'] == 'code':
            ast.parse(c['source'])
            assert c['outputs'] == [] and c['execution_count'] is None
            codes.append(c['source'].replace(fp, '__RESET_E33_CODE_SHA256__'))
    assert hashlib.sha256('\n'.join(codes).encode()).hexdigest() == fp
    assert E33.read_bytes() == (ROOT / 'eksperimen/03_tiga_input_multi_trait/notebooks' / E33.name).read_bytes()
    assert list(source).index('fold_1') < list(source).index('fold_5') < list(source).index('final_report')
    assert 'ordinal_threshold_loss' not in '\n'.join(codes)


def test_five_fold_fit_ema_resume_and_excel_export(tmp_path):
    torch.set_num_threads(1)
    ns, source = namespace(E33), cells(E33)
    for role in ('dataset_definitions', 'definition:device_name', 'definition:seed_everything',
                 'definition:trait_metrics', 'metrics', 'definition:fit',
                 'definition:excel_value', 'definition:write_excel', 'report',
                 'validation_report', 'fold_helpers'):
        exec(source[role], ns)

    class Encoder(nn.Embedding):
        config = SimpleNamespace(num_hidden_layers=4)

        def forward(self, input_ids, **kwargs):
            hidden = super().forward(input_ids)
            return SimpleNamespace(hidden_states=tuple(hidden * (1 + i * .01) for i in range(5)))

    class FixtureEngine:
        def __init__(self, model, texts, bge, cfg, device, **kwargs):
            self.model, self.bge = model, bge
            self.tokenizer, self.contexts = None, None

        def forward(self, ids):
            tokens = torch.tensor([[i % 24, (i + 1) % 24, (i + 2) % 24] for i in ids])
            question = torch.tensor([[True, False, False]] * len(ids))
            return self.model({'input_ids': tokens, 'attention_mask': torch.ones_like(tokens)},
                              torch.arange(len(ids)), len(ids), self.bge[ids], question, ~question)

        @torch.no_grad()
        def predict(self, ids):
            self.model.eval()
            return self.forward(ids).numpy() * ns['SCORE_SPAN'] + ns['SCORE_MIN']

    cfg = dict(experiment='TECHNICAL_FIXTURE_E33', seed=44, device='cpu', epochs=2,
               early_stopping_patience=5, effective_batch=4, micro_batch=2,
               s3_encoder_lr=5e-6, projection_lr=1e-4, s3_mlp_lr=1e-4, s3_layers_lr=1e-4,
               ema_maximum_decay=.99, ema_offset=10, trait_corr_threshold=.7,
               trait_relation_weight=0., isi_loss_ratio=1., organization_loss_ratio=1.2,
               checkpoint_rmse_margin=.01, weight_decay=.01, warmup_ratio=.1, gradient_clip=1.)
    labels = np.array([[1 + i % 4, 1 + i % 2, 1 + (i // 2) % 2, 1 + (i // 3) % 2]
                       for i in range(20)], dtype=np.float32)
    bundle = SimpleNamespace(y=labels, texts=['fixture'] * 20, summary={'rows': 20},
                             groups=np.arange(20), rows=[dict(row_id=i, source_row=i+2, response_number=1)
                                                        for i in range(20)])
    folds = [dict(fold=i+1, validation=list(range(i*4, i*4+4)),
                  train=[j for j in range(20) if not i*4 <= j < i*4+4]) for i in range(5)]
    ns.update(cfg=cfg, bundle=bundle, bge=torch.randn(20, 1024), output=tmp_path,
              manifest=dict(folds=folds, development=list(range(20)), split_sha256='fixture'),
              Engine=FixtureEngine, contexts_from_bundle=lambda b: None, release=lambda: None,
              make_model=lambda *a: ns['MultiTraitModel']('S3', Encoder(24, 768), s3_mlp_dropout=0.))
    # Extract only embedded aggregate references, without reading private data/Drive.
    for line in source['prepare'].splitlines():
        if line.startswith(('E31_REFERENCE =', 'E31_FOLD_REFERENCE =', 'E28_REFERENCE =', 'E25_REFERENCE =')):
            exec(line, ns)
    for fold in range(1, 6):
        result = ns['run_validation_fold'](fold)
        ns['fold_results'][fold] = result
        assert ns['run_validation_fold'](fold) == result
        assert all(row['n'] == 4 for row in result['training']['gate_B'])
        assert result['training']['ema_updates_final'] > 0
    destination = tmp_path / 'Hasil_AES_Reset_E33_Validasi_04_05.zip'
    tables = ns['export_validation'](destination, final=True)
    assert len(tables['Prediksi OOF A B']) == 40
    assert len(tables['E33 vs E31 per Trait']) == 16
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read(next(n for n in archive.namelist() if n.endswith('manifest.json'))))
        books = [name for name in archive.namelist() if name.endswith('.xlsx')]
        assert len(books) == 2
        assert len(manifest['workbooks']) == 2
        training = next(name for name in books if '05_Riwayat' in name)
        wb = ns['load_workbook'](io.BytesIO(archive.read(training)), read_only=True)
        metadata = dict(list(wb['Info'].values)[1:])
        assert metadata['test_dievaluasi'] is False
        rows = list(wb['Gate Validasi B'].values)
        assert len(rows) == 21
        assert 'Diagnostik EMA' in wb.sheetnames
