"""E40: actual encoder group mapping, E30 math equivalence, native execution."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import nbformat
import pandas as pd
import pytest
import torch
from torch import nn
from transformers import RobertaConfig, RobertaModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests'))
import test_context_pair_notebook as context
from test_e36_notebook import execute_full_native_fixture
from test_e6_three_input_notebook import ThreeTechnicalTokenizer
from test_teacher_score import make_book

NAME = 'AES_300_E40_E30_S3_LLRD095_Native_Learnable_GPU.ipynb'
NOTEBOOK = ROOT/'notebooks/experiments'/NAME
BASE = ROOT/'notebooks/experiments/AES_300_E30_E26_BiGRU_CoAttention_Pooling_GPU.ipynb'
FORMAT = ROOT/'notebooks/experiments/AES_300_E39_E30_S3_BiGRU128_Native_Learnable_GPU.ipynb'


def native_runtime():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    module = types.ModuleType('e40_native_fixture')
    sys.modules[module.__name__] = module
    namespace = module.__dict__
    safe = {'imports', 'model_names', 'input_columns', 'hyperparameters', 'runtime_flags',
            'configuration_defaults', 'presentation_helpers', 'word_helpers'}
    for cell in notebook.cells:
        if cell.cell_type == 'code' and cell.metadata.aes40_role in safe:
            exec(compile(cell.source, str(NOTEBOOK), 'exec'), namespace)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            definitions = [node for node in ast.parse(cell.source).body
                           if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
            exec(compile(ast.Module(body=definitions, type_ignores=[]), str(NOTEBOOK), 'exec'), namespace)
    namespace['show'] = lambda *args, **kwargs: None
    return namespace


def test_native_structure_protocol_and_deterministic_builder():
    raw = NOTEBOOK.read_bytes()
    notebook = nbformat.reads(raw.decode(), as_version=4)
    nbformat.validate(notebook)
    meta = notebook.metadata.aes_experiment
    assert meta.scenarios == ['S3'] and meta.cv_fits == 5 and meta.refits == meta.test_evaluations == 1
    assert not meta.multiple_seed_runs and not meta.control_retrained and not meta.ensemble
    assert meta.s3_sequence_hidden == 64 and meta.sequence_output_dimensions == 128
    assert meta.encoder_layer_decay == .95 and meta.s3_encoder_lr == 5e-6 and meta.s3_mlp_dropout == .1
    assert meta.format_sha256 == hashlib.sha256(FORMAT.read_bytes()).hexdigest()
    code = [cell for cell in notebook.cells if cell.cell_type == 'code']
    fingerprint = notebook.metadata.aes_standalone.code_sha256
    assert hashlib.sha256('\n'.join(c.source.replace(fingerprint, '__NOTEBOOK_CODE_SHA256__') for c in code).encode()).hexdigest() == fingerprint
    text = '\n'.join(cell.source for cell in code)
    assert all(token not in text for token in ('%%aes', 'WORKER_SOURCE', 'cosine', 'TRAINING_SEEDS', 'run_paired_cv', 's2_mlp_lr'))
    for cell in code:
        assert cell.execution_count is None and not cell.outputs
        tree = ast.parse(cell.source)
        assert not any(line.lstrip().startswith(('%', '!')) for line in cell.source.splitlines())
        assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                       and node.func.id in ('exec', 'eval') for node in ast.walk(tree))
    def definitions(path):
        return {node.name: ast.dump(node, include_attributes=False)
                for cell in nbformat.read(path, as_version=4).cells if cell.cell_type == 'code'
                for node in ast.parse(cell.source.removeprefix('%%aes\n')).body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    old, new = definitions(BASE), definitions(NOTEBOOK)
    for name in ('train_all', 'make_split', 'metrics', 'gwet_ac2', 'bge_features', 'encoder',
                 'pair_content_windows', 'pair_batch_windows', 'answer_token_mask', 'reference_token_mask', 'save_model'):
        assert old[name] == new[name], name
    # Fit may add tables/metadata; numerical statements remain exactly the E30 recipe.
    def fit_math(path):
        n = nbformat.read(path, as_version=4)
        function = next(node for cell in n.cells if cell.cell_type == 'code'
                        for node in ast.parse(cell.source.removeprefix('%%aes\n')).body
                        if isinstance(node, ast.FunctionDef) and node.name == 'fit')
        # Compare complete epoch loop including loss, gradients, scheduler, selection and early stopping.
        return ast.dump(next(node for node in function.body if isinstance(node, ast.For)
                             and isinstance(node.target, ast.Name) and node.target.id == 'epoch'),
                        include_attributes=False)
    assert fit_math(BASE) == fit_math(NOTEBOOK)
    historical = {path: path.read_bytes() for path in (BASE, FORMAT)}
    subprocess.run([sys.executable, 'scripts/build_e40_notebook.py'], cwd=ROOT, check=True, capture_output=True)
    assert raw == NOTEBOOK.read_bytes() == (ROOT/'aes-learning/notebooks/experiments'/NAME).read_bytes()
    assert all(path.read_bytes() == contents for path, contents in historical.items())


def test_llrd_maps_all_twelve_layers_and_scheduler_preserves_ratios():
    ns = native_runtime()
    cfg = ns['config']({})
    # Compact technical parameter container: twelve separate blocks, no research data/model.
    model = nn.Module()
    model.scenario = 'S3'
    model.roberta = nn.Module()
    model.roberta.config = types.SimpleNamespace(num_hidden_layers=12)
    model.roberta.embeddings = nn.Linear(2, 2)
    model.roberta.encoder = nn.Module()
    model.roberta.encoder.layer = nn.ModuleList([nn.Linear(2, 2) for _ in range(12)])
    model.projection = nn.Linear(2, 2)
    model.head = nn.Linear(2, 1)
    model.sequence_pool = nn.Linear(2, 2)
    model.roberta_norm = nn.LayerNorm(2)
    model.double()
    groups = ns['optimizer_groups'](model, cfg)
    assert len(groups) == 16  # embeddings + twelve blocks + three non-encoder groups
    expected = {'roberta.embeddings': 5e-6*.95**12,
                **{f'roberta.encoder.layer.{i}': 5e-6*.95**(11-i) for i in range(12)},
                'projection': 1e-4, 'head': 1e-4, 'pooling_norm': 1e-4}
    assert {group['name']: group['lr'] for group in groups} == expected
    for group in groups:
        for parameter in group['params']:
            name = next(name for name, value in model.named_parameters() if value is parameter)
            actual_group = '.'.join(name.split('.')[:4]) if name.startswith('roberta.encoder.layer.') else (
                'roberta.embeddings' if name.startswith('roberta.embeddings.') else name.split('.')[0])
            if actual_group in ('roberta_norm', 'sequence_pool'):
                actual_group = 'pooling_norm'
            assert group['name'] == actual_group
    ids = [id(p) for group in groups for p in group['params']]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg['weight_decay'])
    schedule = ns['get_linear_schedule_with_warmup'](optimizer, 2, 6)
    initial = {name: value.clone() for name, value in model.named_parameters()}
    for step in range(6):
        if step in (1, 3):
            rates = {g['name']: g['lr'] for g in optimizer.param_groups}
            assert rates['roberta.embeddings']/rates['roberta.encoder.layer.11'] == pytest.approx(.95**12)
            assert rates['head']/rates['roberta.encoder.layer.11'] == pytest.approx(20.)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        schedule.step()
    assert all(not torch.equal(value, initial[name]) for name, value in model.named_parameters())
    for invalid in (0, -1, 1.01, float('nan'), float('inf'), True):
        with pytest.raises(ValueError, match='encoder_layer_decay'):
            ns['config']({'encoder_layer_decay': invalid})
    model.extra = nn.Linear(2, 2)
    with pytest.raises(ValueError, match='belum mempunyai kelompok'):
        ns['optimizer_groups'](model, cfg)


def test_e30_predictions_gradients_and_uniform_decay_one_update_identical(tmp_path, monkeypatch):
    ns = native_runtime()
    monkeypatch.setattr(context, 'NOTEBOOK', BASE)
    old = context.runtime()
    def tiny_encoder(cfg, role):
        assert role == 'roberta'
        with torch.random.fork_rng():
            torch.manual_seed(170)
            return RobertaModel(RobertaConfig(vocab_size=64, hidden_size=768, num_hidden_layers=2,
                num_attention_heads=12, intermediate_size=16, max_position_embeddings=64,
                pad_token_id=1, bos_token_id=0, eos_token_id=2, hidden_dropout_prob=0.,
                attention_probs_dropout_prob=0.), add_pooling_layer=False)
    for env in (ns, old):
        env['encoder'] = tiny_encoder
        env['tokenizer'] = lambda *args, **kwargs: ThreeTechnicalTokenizer()
    fixture = tmp_path/'TECHNICAL_FIXTURE_ONLY.xlsx'
    make_book(fixture)
    values = dict(dataset=str(fixture), output=str(tmp_path/'run'), expected_rows=24, device='cpu',
                  roberta_max_length=32, roberta_overlap=4, bge_max_length=32, bge_overlap=4,
                  context_max_tokens=12, question_max_tokens=4)
    cfg, control_cfg = ns['config'](values), old['config'](values)
    for key in old['DEFAULTS']:
        if key != 's2_mlp_lr':
            assert cfg[key] == control_cfg[key], key
    bundle = ns['read_dataset'](cfg)
    assert ns['make_split'](bundle, cfg) == old['make_split'](bundle, control_cfg)
    torch.set_num_threads(2)
    ns['seed_everything'](71)
    model = ns['make_model']('S3', cfg).eval()
    old['seed_everything'](71)
    control = old['make_model']('S3', control_cfg).eval()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, control.state_dict()[key], rtol=0, atol=0)
    assert model.sequence_pool.gru.hidden_size == 64 and model.head[2].p == .1
    bge = torch.randn(24, 1024)
    predictions = []
    cfg_uniform = {**cfg, 'encoder_layer_decay': 1.}
    for env, current, settings in ((ns, model, cfg_uniform), (old, control, control_cfg)):
        engine = env['Engine'](current, bundle.texts, bge, settings, 'cpu', contexts=env['contexts_from_bundle'](bundle))
        pred, features = engine.forward([0, 1], return_features=True)
        predictions.append(pred.detach())
        assert features['concat'].shape == (2, 1024)
        (pred-.8).square().mean().backward()
        torch.optim.AdamW(env['optimizer_groups'](current, settings), weight_decay=.01).step()
    torch.testing.assert_close(*predictions, rtol=0, atol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, control.state_dict()[key], rtol=0, atol=0)
    # Reuse identical gradients/state: only lower encoder updates may change at decay0.95.
    model.load_state_dict(control.state_dict())
    before = {key: value.clone() for key, value in model.state_dict().items()}
    for env, current, settings in ((ns, model, cfg), (old, control, control_cfg)):
        torch.optim.AdamW(env['optimizer_groups'](current, settings), weight_decay=.01).step()
    for key, value in model.state_dict().items():
        if not key.startswith('roberta.') or key.startswith('roberta.encoder.layer.1.'):
            torch.testing.assert_close(value, control.state_dict()[key], rtol=0, atol=0)
    key = 'roberta.embeddings.word_embeddings.weight'
    ratio = (model.state_dict()[key]-before[key]).norm()/(control.state_dict()[key]-before[key]).norm()
    assert float(ratio) == pytest.approx(.95**2, rel=.01)
    assert model.sequence_pool.gru.weight_ih_l0.grad.abs().sum() > 0
    assert model.roberta.encoder.layer[0].attention.self.query.weight.grad.abs().sum() > 0
    checkpoint = tmp_path/'TECHNICAL_checkpoint.pt'
    torch.save({'scope': ns['SCOPE'], 'trained': True, 'scenario': 'S3', 'config': cfg,
                'state': model.state_dict()}, checkpoint)
    loaded, _ = ns['load_model'](checkpoint, cfg)
    assert not loaded.training and loaded.sequence_pool.gru.hidden_size == 64
    with pytest.raises(ValueError, match='encoder_layer_decay'):
        ns['load_model'](checkpoint, cfg_uniform)
    for scenario in ('S1', 'S2'):
        with pytest.raises(ValueError):
            ns['make_model'](scenario, cfg)


def test_full_native_notebook_exports_actual_layer_lrs_and_fresh_kernel_resume(tmp_path):
    expected_rates = {'roberta.embeddings': 5e-6*.95**2,
                      'roberta.encoder.layer.0': 5e-6*.95,
                      'roberta.encoder.layer.1': 5e-6}
    execute_full_native_fixture(tmp_path, NOTEBOOK, 'aes40_role', 5e-6,
                                encoder_rate_groups=expected_rates, encoder_layers=2)
    info = json.loads((tmp_path/'standalone/workspace/outputs/dataset300_download.json').read_text())
    run = Path(info['run_folder'])
    cfg = json.loads((run/'effective_config.json').read_text())
    assert cfg['encoder_layer_decay'] == .95 and cfg['s3_sequence_hidden'] == 64
    book = run/'hasil_ringkas/05_Riwayat_Pelatihan.xlsx'
    settings = pd.read_excel(book, sheet_name='Pengaturan Eksperimen')
    settings = dict(zip(settings.Pengaturan, settings.Nilai))
    assert settings['Faktor LR per layer RoBERTa'] == .95 and settings['Hidden pooling S3'] == 64
    lr = pd.read_excel(book, sheet_name='Learning Rate per Layer')
    assert len(lr) == 36  # six groups, five folds and one refit
    for stage, fold in [('Validasi silang', i) for i in range(1, 6)]+[('Latihan final', None)]:
        rows = lr[(lr.Tahap == stage) & (lr.Fold.isna() if fold is None else lr.Fold == fold)]
        actual = dict(zip(rows.Kelompok, rows['LR Puncak']))
        assert actual == pytest.approx({**expected_rates, 'projection': 1e-4, 'head': 1e-4, 'pooling_norm': 1e-4})
        assert (rows['Jumlah Parameter'] > 0).all()
    vectors = pd.read_excel(run/'hasil_ringkas/02_Vektor_Jawaban.xlsx', sheet_name='Ringkasan CoAttn Pooling S3')
    assert vectors.shape == (24, 129) and vectors.iloc[:, 1:].notna().all().all()
    checkpoint = torch.load(run/'S3/final.pt', map_location='cpu', weights_only=True)
    assert checkpoint['training']['encoder_layer_decay'] == .95
    assert checkpoint['training']['encoder_num_layers'] == 2
    assert Path(info['archive']).name == 'Hasil_AES_300_E40_E30_S3_LLRD095_Native_Learnable_GPU.zip'
