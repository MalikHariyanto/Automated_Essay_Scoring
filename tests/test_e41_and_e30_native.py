"""Native E30 equivalence, Scalar Mix learning and two complete notebook runs."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import nbformat
import numpy as np
import pandas as pd
import pytest
import torch
from transformers import RobertaConfig, RobertaModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests'))
import test_context_pair_notebook as context
from test_e36_notebook import execute_full_native_fixture
from test_e6_three_input_notebook import ThreeTechnicalTokenizer
from test_teacher_score import make_book

BASE = ROOT/'notebooks/experiments/AES_300_E30_E26_BiGRU_CoAttention_Pooling_GPU.ipynb'
NATIVE = ROOT/'notebooks/experiments/AES_300_E30_S3_MLP_Native_Learnable_GPU.ipynb'
MIX = ROOT/'notebooks/experiments/AES_300_E41_E30_S3_ScalarMix4_Native_Learnable_GPU.ipynb'
FORMAT = ROOT/'notebooks/experiments/AES_300_E39_E30_S3_BiGRU128_Native_Learnable_GPU.ipynb'


def runtime(path, key):
    notebook = nbformat.read(path, as_version=4)
    module = types.ModuleType(key+'_fixture')
    sys.modules[module.__name__] = module
    ns = module.__dict__
    safe = {'imports', 'model_names', 'input_columns', 'hyperparameters', 'runtime_flags',
            'configuration_defaults', 'presentation_helpers', 'word_helpers'}
    for cell in notebook.cells:
        if cell.cell_type == 'code' and cell.metadata[key] in safe:
            exec(compile(cell.source, str(path), 'exec'), ns)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            definitions = [node for node in ast.parse(cell.source).body
                           if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
            exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), ns)
    ns['show'] = lambda *args, **kwargs: None
    return ns


def definitions(path):
    return {node.name: node for cell in nbformat.read(path, as_version=4).cells if cell.cell_type == 'code'
            for node in ast.parse(cell.source.removeprefix('%%aes\n')).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def tiny_encoder(cfg, role):
    assert role == 'roberta'
    with torch.random.fork_rng():
        torch.manual_seed(170)
        return RobertaModel(RobertaConfig(vocab_size=64, hidden_size=768, num_hidden_layers=4,
            num_attention_heads=12, intermediate_size=16, max_position_embeddings=64,
            pad_token_id=1, bos_token_id=0, eos_token_id=2, hidden_dropout_prob=0.,
            attention_probs_dropout_prob=0.), add_pooling_layer=False)


def setup(env, tmp_path):
    env['encoder'] = tiny_encoder
    env['tokenizer'] = lambda *args, **kwargs: ThreeTechnicalTokenizer()
    file = tmp_path/'TECHNICAL_FIXTURE_ONLY.xlsx'
    if not file.exists():
        make_book(file)
    cfg = env['config'](dict(dataset=str(file), output=str(tmp_path/'run'), expected_rows=24,
        device='cpu', roberta_max_length=32, roberta_overlap=4, bge_max_length=32, bge_overlap=4,
        context_max_tokens=12, question_max_tokens=4))
    torch.set_num_threads(2)
    return cfg, env['read_dataset'](cfg)


def test_both_notebooks_native_hash_protocol_and_deterministic_builder():
    original = {path: path.read_bytes() for path in (BASE, FORMAT, NATIVE, MIX)}
    historical = definitions(BASE)
    for path, key, mixed in [(NATIVE, 'aes30native_role', False), (MIX, 'aes41_role', True)]:
        n = nbformat.read(path, as_version=4)
        nbformat.validate(n)
        meta = n.metadata.aes_experiment
        assert meta.scenarios == ['S3'] and meta.cv_fits == 5 and meta.refits == meta.test_evaluations == 1
        assert not meta.multiple_seed_runs and not meta.control_retrained and not meta.ensemble
        assert meta.s3_sequence_hidden == 64 and meta.sequence_output_dimensions == 128
        assert meta.s3_encoder_lr == 5e-6 and meta.s3_mlp_dropout == .1 and meta.scalar_mix == mixed
        assert meta.format_sha256 == hashlib.sha256(FORMAT.read_bytes()).hexdigest()
        code = [cell for cell in n.cells if cell.cell_type == 'code']
        digest = n.metadata.aes_standalone.code_sha256
        assert hashlib.sha256('\n'.join(c.source.replace(digest, '__NOTEBOOK_CODE_SHA256__') for c in code).encode()).hexdigest() == digest
        text = '\n'.join(c.source for c in code)
        for token in ('%%aes', 'WORKER_SOURCE', 'cosine', 'TRAINING_SEEDS', 'encoder_layer_decay'):
            assert token not in text
        if not mixed:
            assert 'scalar_mix' not in text
        for cell in code:
            assert cell.metadata[key] and not cell.outputs and cell.execution_count is None
            assert not any(line.lstrip().startswith(('%', '!')) for line in cell.source.splitlines())
            assert not any(isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id in ('exec', 'eval')
                           for v in ast.walk(ast.parse(cell.source)))
        current = definitions(path)
        for name in ('train_all', 'make_split', 'metrics', 'gwet_ac2', 'bge_features', 'encoder',
                     'pair_content_windows', 'pair_batch_windows', 'answer_token_mask', 'reference_token_mask', 'save_model'):
            assert ast.dump(current[name]) == ast.dump(historical[name]), name
        if not mixed:
            assert ast.dump(current['fit']) == ast.dump(historical['fit'])
        else:
            epoch = lambda fn: next(v for v in fn.body if isinstance(v, ast.For)
                                    and isinstance(v.target, ast.Name) and v.target.id == 'epoch')
            assert ast.dump(epoch(current['fit'])) == ast.dump(epoch(historical['fit']))
    subprocess.run([sys.executable, 'scripts/build_e41_and_e30_native.py'], cwd=ROOT, check=True, capture_output=True)
    assert all(path.read_bytes() == value for path, value in original.items())
    for path in (NATIVE, MIX):
        assert path.read_bytes() == (ROOT/'aes-learning/notebooks/experiments'/path.name).read_bytes()


def test_scalar_mix_selects_only_last_four_layers_and_learns_weights():
    ns = runtime(MIX, 'aes41_role')
    scalar = ns['ScalarMix'](4)
    assert sum(p.numel() for p in scalar.parameters()) == 4
    states = tuple(torch.full((2, 3, 5), float(i), requires_grad=True) for i in range(7))
    # Embeddings and first two Transformer layers are excluded; average of3,4,5,6 is4.5.
    mixed = scalar(states)
    torch.testing.assert_close(mixed, torch.full_like(mixed, 4.5))
    torch.testing.assert_close(scalar.weights(), torch.full((4,), .25))
    mixed.sum().backward()
    assert all(value.grad is None for value in states[:3])
    for value in states[3:]:
        torch.testing.assert_close(value.grad, torch.full_like(value, .25))
    assert scalar.logits.grad is not None and torch.isfinite(scalar.logits.grad).all()
    assert (scalar.logits.grad != 0).all()
    torch.optim.AdamW(scalar.parameters(), lr=1e-4).step()
    assert not torch.equal(scalar.weights(), torch.full((4,), .25))
    assert scalar.weights().sum().item() == pytest.approx(1.)
    assert (scalar.weights() > 0).all()
    half = scalar(tuple(value.detach().half() for value in states))
    assert half.dtype == torch.float16 and torch.isfinite(half).all()
    with pytest.raises(ValueError, match='cukup layer'):
        scalar(states[:4])
    with pytest.raises(ValueError, match='shape'):
        scalar((*states[:-1], torch.zeros(1, 2, 3)))
    for invalid in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            ns['config']({'scalar_mix_layers': invalid})
    for invalid in (0, -1, float('nan')):
        with pytest.raises(ValueError):
            ns['config']({'scalar_mix_lr': invalid})


def test_native_e30_initialization_predictions_and_adamw_update_match_original(tmp_path, monkeypatch):
    ns = runtime(NATIVE, 'aes30native_role')
    monkeypatch.setattr(context, 'NOTEBOOK', BASE)
    old = context.runtime()
    cfg, bundle = setup(ns, tmp_path)
    control_cfg, control_bundle = setup(old, tmp_path)
    for key in old['DEFAULTS']:
        if key != 's2_mlp_lr':
            assert cfg[key] == control_cfg[key], key
    assert ns['make_split'](bundle, cfg) == old['make_split'](control_bundle, control_cfg)
    ns['seed_everything'](71)
    model = ns['make_model']('S3', cfg).eval()
    old['seed_everything'](71)
    control = old['make_model']('S3', control_cfg).eval()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, control.state_dict()[key], rtol=0, atol=0)
    bge = torch.randn(24, 1024)
    predictions = []
    for env, current, settings in [(ns, model, cfg), (old, control, control_cfg)]:
        engine = env['Engine'](current, bundle.texts, bge, settings, 'cpu', contexts=env['contexts_from_bundle'](bundle))
        pred = engine.forward([0, 1])
        predictions.append(pred.detach())
        (pred-.8).square().mean().backward()
        torch.optim.AdamW(env['optimizer_groups'](current, settings), weight_decay=.01).step()
    torch.testing.assert_close(*predictions, rtol=0, atol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, control.state_dict()[key], rtol=0, atol=0)


def test_e41_gradients_optimizer_checkpoint_and_selected_weights_export(tmp_path):
    ns = runtime(MIX, 'aes41_role')
    base = runtime(NATIVE, 'aes30native_role')
    cfg, bundle = setup(ns, tmp_path)
    base_cfg, _ = setup(base, tmp_path)
    ns['seed_everything'](71)
    model = ns['make_model']('S3', cfg).eval()
    base['seed_everything'](71)
    control = base['make_model']('S3', base_cfg).eval()
    assert sum(p.numel() for p in model.parameters())-sum(p.numel() for p in control.parameters()) == 4
    for key, value in control.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
    groups = ns['optimizer_groups'](model, cfg)
    assert {g['name']: g['lr'] for g in groups} == {'roberta': 5e-6, 'head': 1e-4,
        'projection': 1e-4, 'pooling_norm': 1e-4, 'scalar_mix': 1e-4}
    assert sum(p.numel() for g in groups if g['name'] == 'scalar_mix' for p in g['params']) == 4
    ids = [id(p) for g in groups for p in g['params']]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    bge = torch.randn(24, 1024)
    engine = ns['Engine'](model, bundle.texts, bge, cfg, 'cpu', contexts=ns['contexts_from_bundle'](bundle))
    pred, features = engine.forward([0, 1], return_features=True)
    assert pred.shape == (2, 1) and features['bigru_summary'].shape == (2, 128)
    assert features['concat'].shape == (2, 1024)
    (pred-.8).square().mean().backward()
    for p in [model.scalar_mix.logits, model.roberta.embeddings.word_embeddings.weight,
              model.roberta.encoder.layer[0].attention.self.query.weight,
              model.roberta.encoder.layer[3].attention.self.query.weight,
              model.sequence_pool.gru.weight_ih_l0, model.sequence_pool.reference_projection.weight]:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    weights_before = model.scalar_mix.weights().detach().clone()
    torch.optim.AdamW(groups, weight_decay=.01).step()
    assert not torch.equal(weights_before, model.scalar_mix.weights())
    checkpoint = tmp_path/'TECHNICAL_checkpoint.pt'
    torch.save({'scope': ns['SCOPE'], 'trained': True, 'scenario': 'S3', 'config': cfg,
                'state': model.state_dict()}, checkpoint)
    loaded, _ = ns['load_model'](checkpoint, cfg)
    torch.testing.assert_close(loaded.scalar_mix.weights(), model.scalar_mix.weights())
    with pytest.raises(ValueError, match='scalar_mix_layers'):
        ns['load_model'](checkpoint, {**cfg, 'scalar_mix_layers': 3})
    with pytest.raises(ValueError, match='scalar_mix_lr'):
        ns['load_model'](checkpoint, {**cfg, 'scalar_mix_lr': 5e-5})
    for scenario in ('S1', 'S2'):
        with pytest.raises(ValueError):
            ns['make_model'](scenario, cfg)
    with pytest.raises(ValueError, match='lebih sedikit layer'):
        ns['make_model']('S3', {**cfg, 'scalar_mix_layers': 5})
    # Force epoch1 to be the validation choice, then verify exports use its restored weights.
    recorded = []
    ns['save_training_progress'] = lambda path, value: recorded.append(value['model']['scalar_mix.logits'].clone())
    ns['load_training_progress'] = lambda *args: None
    actual_metrics = ns['metrics']
    counter = []
    def controlled_validation(*args):
        counter.append(1)
        return {**actual_metrics(*args), 'RMSE': float(len(counter))}
    ns['metrics'] = controlled_validation
    fitted, _, info = ns['fit']('S3', list(range(8)), [8, 9], bundle, bge,
        {**cfg, 'early_stopping_patience': 1}, 3, 71, checkpoint=tmp_path/'unused_progress.json')
    assert len(recorded) == 2 and not torch.equal(recorded[0], recorded[1])
    assert info['best_epoch'] == 1 and info['epochs_completed'] == 2
    torch.testing.assert_close(fitted.scalar_mix.logits, recorded[0], rtol=0, atol=0)
    np.testing.assert_allclose([r['Bobot Softmax'] for r in info['scalar_mix_weights']],
                               torch.softmax(recorded[0], dim=0).numpy(), rtol=0, atol=0)


@pytest.mark.parametrize('path,key,layers', [(NATIVE, 'aes30native_role', 1), (MIX, 'aes41_role', 4)])
def test_full_native_notebook_exports_and_fresh_kernel_resume(tmp_path, path, key, layers):
    execute_full_native_fixture(tmp_path, path, key, 5e-6, encoder_layers=layers)
    info = json.loads((tmp_path/'standalone/workspace/outputs/dataset300_download.json').read_text())
    run = Path(info['run_folder'])
    cfg = json.loads((run/'effective_config.json').read_text())
    assert cfg['s3_sequence_hidden'] == cfg['pooling_attention_hidden'] == 64 and cfg['s3_mlp_dropout'] == .1
    book = run/'hasil_ringkas/05_Riwayat_Pelatihan.xlsx'
    checkpoint = torch.load(run/'S3/final.pt', map_location='cpu', weights_only=True)
    if path == MIX:
        assert cfg['scalar_mix_layers'] == 4 and cfg['scalar_mix_lr'] == 1e-4
        weights = pd.read_excel(book, sheet_name='Bobot Scalar Mix')
        assert len(weights) == 24
        for stage, fold in [('Validasi silang', i) for i in range(1, 6)]+[('Latihan final', None)]:
            rows = weights[(weights.Tahap == stage) & (weights.Fold.isna() if fold is None else weights.Fold == fold)]
            assert rows['Indeks Layer RoBERTa'].tolist() == [0, 1, 2, 3]
            assert (rows['Bobot Awal'] == .25).all() and (rows['Bobot Softmax'] > 0).all()
            assert rows['Bobot Softmax'].sum() == pytest.approx(1., abs=1e-6)
        final_rows = weights[weights.Tahap == 'Latihan final']
        expected = torch.softmax(checkpoint['state']['scalar_mix.logits'], dim=0).numpy()
        np.testing.assert_allclose(final_rows['Bobot Softmax'].to_numpy(), expected, rtol=1e-7)
        assert checkpoint['training']['optimizer_learning_rates']['scalar_mix'] == 1e-4
        assert not np.allclose(expected, [.25]*4, rtol=0, atol=1e-7)
    else:
        assert 'scalar_mix_layers' not in cfg
        assert not any(k.startswith('scalar_mix') for k in checkpoint['state'])
        assert 'Bobot Scalar Mix' not in pd.ExcelFile(book).sheet_names
    vectors = pd.read_excel(run/'hasil_ringkas/02_Vektor_Jawaban.xlsx', sheet_name='Ringkasan CoAttn Pooling S3')
    assert vectors.shape == (24, 129)
    assert Path(info['archive']).name == 'Hasil_'+path.stem+'.zip'
