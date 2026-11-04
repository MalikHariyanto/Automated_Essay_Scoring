"""E43 masked mean, preservation of shared E42 initialization, and full native run."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import nbformat
import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests'))
from test_e41_and_e30_native import runtime, setup, definitions
from test_e36_notebook import execute_full_native_fixture

NOTEBOOK = ROOT/'notebooks/experiments/AES_300_E43_E42_S3_MaskedMeanPooling_Native_Learnable_GPU.ipynb'
BASE = ROOT/'notebooks/experiments/AES_300_E42_E41_S3_ScalarMix4_Gamma_Native_Learnable_GPU.ipynb'


def test_structure_protocol_and_deterministic_builder():
    raw, old = NOTEBOOK.read_bytes(), BASE.read_bytes()
    n = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(n)
    m = n.metadata.aes_experiment
    assert m.scenarios == ['S3'] and m.cv_fits == 5 and m.refits == m.test_evaluations == 1
    assert not m.multiple_seed_runs and not m.control_retrained and not m.ensemble
    assert m.scalar_mix_layers == 4 and m.scalar_mix_gamma_init == 1.
    assert m.s3_encoder_lr == 5e-6 and m.scalar_mix_lr == 1e-4 and m.s3_sequence_hidden == 64
    assert m.s3_pooling == 'bigru_coattention_mean_pooling' and not m.attention_pooling
    assert m.removed_parameters_vs_e42 == 8320 and m.net_parameters_vs_e30 == -8315
    code = [c for c in n.cells if c.cell_type == 'code']
    fp = n.metadata.aes_standalone.code_sha256
    assert hashlib.sha256('\n'.join(c.source.replace(fp, '__NOTEBOOK_CODE_SHA256__') for c in code).encode()).hexdigest() == fp
    assert hashlib.sha256(old).hexdigest() == m.format_sha256
    for c in code:
        assert c.metadata.aes43_role and not c.outputs and c.execution_count is None
        assert not any(line.lstrip().startswith(('%', '!')) for line in c.source.splitlines())
        assert not any(isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id in ('exec', 'eval')
                       for v in ast.walk(ast.parse(c.source)))
    text = '\n'.join(c.source for c in code)
    assert all(x not in text for x in ['cosine', 'WORKER_SOURCE', 'TRAINING_SEEDS', 'encoder_layer_decay'])
    a, b = definitions(BASE), definitions(NOTEBOOK)
    for name in ['optimizer_groups', 'fit', 'train_all', 'make_split', 'metrics', 'gwet_ac2', 'Engine',
                 'encoder', 'bge_features', 'ScalarMix', 'AnswerBiGRUAttention',
                 'pair_content_windows', 'pair_batch_windows']:
        actual = ast.dump(b[name]).replace('Engine E43 hanya menjalankan S3.', 'Engine E42 hanya menjalankan S3.')
        assert ast.dump(a[name]) == actual, name
    subprocess.run([sys.executable, 'scripts/build_e43_notebook.py'], cwd=ROOT, check=True, capture_output=True)
    assert raw == NOTEBOOK.read_bytes() == (ROOT/'aes-learning/notebooks/experiments'/NOTEBOOK.name).read_bytes()
    assert old == BASE.read_bytes()


def test_masked_mean_matches_uniform_control_padding_and_gradients():
    ns, old = runtime(NOTEBOOK, 'aes43_role'), runtime(BASE, 'aes42_role')
    torch.manual_seed(71)
    pool = ns['AnswerBiGRUCoAttentionMeanPool']()
    rng = torch.get_rng_state()
    torch.manual_seed(71)
    control = old['AnswerBiGRUCoAttentionPool']()
    assert torch.equal(rng, torch.get_rng_state())
    assert sum(p.numel() for p in control.parameters()) - sum(p.numel() for p in pool.parameters()) == 8320
    assert set(control.state_dict()) - set(pool.state_dict()) == {
        'attention_hidden.weight', 'attention_hidden.bias', 'attention_score.weight'}
    for key, value in pool.state_dict().items():
        torch.testing.assert_close(value, control.state_dict()[key], rtol=0, atol=0)
    control.attention_score.weight.data.zero_()  # Softmax of zeros gives the same mean.
    hidden = torch.randn(2, 9, 768, requires_grad=True)
    mask = torch.tensor([[0,0,0,1,1,1,0,0,0], [0,0,0,1,1,0,0,0,0]], dtype=torch.bool)
    ref = torch.tensor([[0,1,1,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0]], dtype=torch.bool)
    pool.capture_alignment = True
    result = pool(hidden, mask, ref)
    torch.testing.assert_close(result, control(hidden, mask, ref), rtol=1e-6, atol=1e-7)
    expected = mask.float() / mask.sum(1, keepdim=True)
    torch.testing.assert_close(pool.last_weights, expected)
    torch.testing.assert_close(pool.last_weights.sum(1), torch.ones(2))
    assert pool.last_summary.shape == (2,128)
    assert torch.isfinite(pool.last_cross_attention).all()
    for weights in [pool.last_response_coattention_weights, pool.last_reference_weights]:
        torch.testing.assert_close(weights.sum(1), torch.ones(2))
    # Context/padding outside the two explicitly selected sequences cannot change this module.
    changed = hidden.detach().clone()
    changed[~(mask | ref)] = 10000.
    torch.testing.assert_close(result, pool(changed, mask, ref), rtol=0, atol=0)
    result.square().mean().backward()
    for name, p in pool.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
    assert hidden.grad[~(mask | ref)].abs().sum() == 0
    assert hidden.grad[ref].abs().sum() > 0 and hidden.grad[mask].abs().sum() > 0


def test_shared_model_initialization_learns_and_checkpoint_recipe_guard(tmp_path):
    ns, old = runtime(NOTEBOOK, 'aes43_role'), runtime(BASE, 'aes42_role')
    cfg, bundle = setup(ns, tmp_path)
    oldcfg, _ = setup(old, tmp_path)
    assert cfg['epochs'] == 25 and cfg['folds'] == 5 and cfg['early_stopping_patience'] == 5
    for key in old['DEFAULTS']:
        if key != 's3_pooling':
            assert cfg[key] == oldcfg[key], key
    assert cfg['s3_pooling'] != oldcfg['s3_pooling']
    ns['seed_everything'](71)
    model = ns['make_model']('S3', cfg).eval()
    rng = torch.get_rng_state()
    old['seed_everything'](71)
    control = old['make_model']('S3', oldcfg).eval()
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, control.state_dict()[key], rtol=0, atol=0)
    assert sum(p.numel() for p in control.parameters()) - sum(p.numel() for p in model.parameters()) == 8320
    bge = torch.randn(24, 1024)
    engine = ns['Engine'](model, bundle.texts, bge, cfg, 'cpu', contexts=ns['contexts_from_bundle'](bundle))
    pred, features = engine.forward([0,1], return_features=True)
    assert features['concat'].shape == (2,1024) and features['bigru_summary'].shape == (2,128)
    assert features['answer_pool768'].shape == (2,768)
    (pred-.8).square().mean().backward()
    assert model.scalar_mix.gamma.grad.abs().sum() > 0
    assert model.scalar_mix.logits.grad.abs().sum() > 0
    assert next(model.roberta.parameters()).grad.abs().sum() > 0
    assert model.head[0].weight.grad.abs().sum() > 0
    groups = ns['optimizer_groups'](model, cfg)
    pooled = next(g for g in groups if g['name'] == 'pooling_norm')
    assert pooled['lr'] == 1e-4
    assert sum(p.numel() for g in groups for p in g['params']) == sum(p.numel() for p in model.parameters())
    assert not any('attention_hidden' in name or 'attention_score' in name for name, _ in model.named_parameters())
    file = tmp_path/'TECHNICAL_mean.pt'
    torch.save({'scope':ns['SCOPE'], 'trained':True, 'scenario':'S3', 'config':cfg, 'state':model.state_dict()}, file)
    loaded, _ = ns['load_model'](file, cfg)
    torch.testing.assert_close(loaded.scalar_mix.gamma, model.scalar_mix.gamma)
    with pytest.raises(ValueError, match='s3_pooling'):
        ns['load_model'](file, {**cfg, 's3_pooling':'bigru_coattention_pooling'})


def test_full_native_notebook_mean_exports_cv_errors_and_resume(tmp_path):
    execute_full_native_fixture(tmp_path, NOTEBOOK, 'aes43_role', 5e-6, encoder_layers=4)
    info = json.loads((tmp_path/'standalone/workspace/outputs/dataset300_download.json').read_text())
    run = Path(info['run_folder'])
    checkpoint = torch.load(run/'S3/final.pt', map_location='cpu', weights_only=True)
    assert checkpoint['config']['s3_pooling'] == 'bigru_coattention_mean_pooling'
    assert not any('attention_hidden' in key or 'attention_score' in key for key in checkpoint['state'])
    vectors = run/'hasil_ringkas/02_Vektor_Jawaban.xlsx'
    book = pd.ExcelFile(vectors)
    assert 'Bobot Mean Pooling S3' in book.sheet_names and 'Bobot Attention S3' not in book.sheet_names
    assert {'Bobot CoAttention Respons S3','Bobot Referensi S3','Pasangan Token Top1 S3'} <= set(book.sheet_names)
    weights = pd.read_excel(vectors, sheet_name='Bobot Mean Pooling S3')
    assert 'Bobot Pooling' in weights and 'Bobot Attention' not in weights
    for _, group in weights.groupby(['ID Jawaban', 'Bagian Teks']):
        np.testing.assert_allclose(group['Bobot Pooling'], np.full(len(group), 1/len(group)), rtol=1e-6)
    history = run/'hasil_ringkas/05_Riwayat_Pelatihan.xlsx'
    mix = pd.read_excel(history, sheet_name='Bobot Scalar Mix')
    assert len(mix) == 24 and np.isfinite(mix.Gamma).all()
    summary = pd.read_excel(history, sheet_name='MAE dan Toleransi CV')
    assert summary.shape == (6,6)
    all_errors, ids = [], []
    for fold in range(1,6):
        raw = pd.read_excel(run/f'excel/09_S3_fold_{fold}.xlsx', sheet_name='Prediksi_validation')
        errors = (raw.prediksi_mentah - raw.skor_guru).abs().to_numpy()
        all_errors.extend(errors.tolist())
        ids.extend(raw.row_id.tolist())
        row = summary[summary.Fold == fold].iloc[0]
        expected = [len(errors), errors.mean(), (errors<=1).sum(), (errors<=5).sum(), (errors<=10).sum()]
        np.testing.assert_allclose(row.iloc[1:].astype(float), expected)
    assert len(ids) == len(set(ids))
    errors = np.asarray(all_errors)
    oof = summary.iloc[-1]
    assert oof.Fold == 'OOF development'
    np.testing.assert_allclose(oof.iloc[1:].astype(float),
        [len(errors), errors.mean(), (errors<=1).sum(), (errors<=5).sum(), (errors<=10).sum()])
    test = pd.read_excel(run/'hasil_ringkas/04_Prediksi_dan_Evaluasi.xlsx', sheet_name='Nilai Guru dan Prediksi')
    assert set(test['ID Jawaban']).isdisjoint(ids) and len(test) + len(ids) == 24
    gamma = float(checkpoint['state']['scalar_mix.gamma'].item())
    before = pd.read_excel(vectors, sheet_name='Norma Campuran Sebelum')
    after = pd.read_excel(vectors, sheet_name='Norma Campuran Sesudah')
    np.testing.assert_allclose(after.iloc[:,1], abs(gamma)*before.iloc[:,1], rtol=1e-5)
    assert Path(info['archive']).name == 'Hasil_' + NOTEBOOK.stem + '.zip'
