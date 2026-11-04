"""Complete E30 reference: all original scenarios and actual native execution."""
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests'))
import test_context_pair_notebook as context
from test_e41_and_e30_native import runtime
from test_e6_three_input_notebook import ThreeTechnicalTokenizer
from test_teacher_score import TechnicalTokenizer, make_book, encoders

NOTEBOOK = ROOT/'notebooks/experiments/AES_300_E30_Lengkap_S1_S2_S3_Native_Learnable_GPU.ipynb'
BASE = ROOT/'notebooks/experiments/AES_300_E30_E26_BiGRU_CoAttention_Pooling_GPU.ipynb'
FORMAT = ROOT/'notebooks/experiments/AES_300_E35_E30_LayersLR5e5_Native_Learnable_GPU.ipynb'
ROLE = 'aes30full_role'


def test_complete_native_structure_original_e30_core_and_deterministic_builder():
    raw = NOTEBOOK.read_bytes()
    n = nbformat.reads(raw.decode(), as_version=4)
    nbformat.validate(n)
    meta = n.metadata.aes_experiment
    assert meta.scenarios == ['S1', 'S2', 'S3'] and meta.cv_fits == 15 and meta.refits == 3
    assert meta.s3_layers_lr == meta.s2_mlp_lr == meta.s3_mlp_lr == meta.head_lr == meta.projection_lr == 1e-4
    assert meta.s1_encoder_lr == meta.s3_encoder_lr == 5e-6 and not meta.multiple_seed_runs
    assert meta.format_sha256 == hashlib.sha256(FORMAT.read_bytes()).hexdigest()
    code = [c for c in n.cells if c.cell_type == 'code']
    fingerprint = n.metadata.aes_standalone.code_sha256
    assert hashlib.sha256('\n'.join(c.source.replace(fingerprint, '__NOTEBOOK_CODE_SHA256__') for c in code).encode()).hexdigest() == fingerprint
    text = '\n'.join(c.source for c in code)
    for token in ('%%aes', 'WORKER_SOURCE', 'cosine', 'TRAINING_SEEDS', 'scalar_mix', 'encoder_layer_decay'):
        assert token not in text
    for c in code:
        assert not c.outputs and c.execution_count is None and c.metadata[ROLE]
        assert not any(line.lstrip().startswith(('%', '!')) for line in c.source.splitlines())
        assert not any(isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id in ('exec', 'eval')
                       for v in ast.walk(ast.parse(c.source)))
    def definitions(path):
        return {node.name: ast.dump(node) for cell in nbformat.read(path, as_version=4).cells if cell.cell_type == 'code'
                for node in ast.parse(cell.source.removeprefix('%%aes\n')).body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    old, new = definitions(BASE), definitions(NOTEBOOK)
    for name in ['AnswerBiGRUAttention', 'AnswerBiGRUCoAttentionPool', 'SingleScoreModel', 'Engine',
                 'make_model', 'optimizer_groups', 'fit', 'train_all', 'make_split', 'bge_features',
                 'metrics', 'gwet_ac2', 'pair_content_windows', 'pair_batch_windows', 'answer_token_mask',
                 'reference_token_mask', 'save_model', 'load_model', 'export_final_features']:
        assert old[name] == new[name], name
    historical = {p: p.read_bytes() for p in [BASE, FORMAT,
        ROOT/'notebooks/experiments/AES_300_E30_S3_MLP_Native_Learnable_GPU.ipynb',
        ROOT/'notebooks/experiments/AES_300_E41_E30_S3_ScalarMix4_Native_Learnable_GPU.ipynb']}
    subprocess.run([sys.executable, 'scripts/build_e30_full_native.py'], cwd=ROOT, check=True, capture_output=True)
    assert raw == NOTEBOOK.read_bytes() == (ROOT/'aes-learning/notebooks/experiments'/NOTEBOOK.name).read_bytes()
    assert all(path.read_bytes() == content for path, content in historical.items())


@pytest.mark.parametrize('scenario', ['S1', 'S2', 'S3'])
def test_all_scenarios_match_original_initial_weights_predictions_and_update(scenario, tmp_path, monkeypatch, encoders):
    ns = runtime(NOTEBOOK, ROLE)
    monkeypatch.setattr(context, 'NOTEBOOK', BASE)
    old = context.runtime()
    for env in (ns, old):
        env['encoder'] = encoders
        env['tokenizer'] = lambda *args, **kwargs: ThreeTechnicalTokenizer()
    file = tmp_path/'TECHNICAL_FIXTURE_ONLY.xlsx'
    make_book(file)
    values = dict(dataset=str(file), output=str(tmp_path/'run'), expected_rows=24, device='cpu',
        roberta_max_length=32, roberta_overlap=4, bge_max_length=32, bge_overlap=4,
        context_max_tokens=12, question_max_tokens=4)
    cfg, control_cfg = ns['config'](values), old['config'](values)
    assert ns['SCENARIOS'] == ('S1', 'S2', 'S3')
    for key in old['DEFAULTS']:
        assert cfg[key] == control_cfg[key], key
    bundle = ns['read_dataset'](cfg)
    assert ns['make_split'](bundle, cfg) == old['make_split'](bundle, control_cfg)
    torch.set_num_threads(2)
    ns['seed_everything'](71)
    model = ns['make_model'](scenario, cfg).eval()
    old['seed_everything'](71)
    control = old['make_model'](scenario, control_cfg).eval()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, control.state_dict()[key], rtol=0, atol=0)
    bge = torch.randn(24, 1024)
    predictions = []
    for env, current, settings in [(ns, model, cfg), (old, control, control_cfg)]:
        engine = env['Engine'](current, bundle.texts, bge, settings, 'cpu', contexts=env['contexts_from_bundle'](bundle))
        pred = engine.forward([0, 1])
        predictions.append(pred.detach())
        groups = env['optimizer_groups'](current, settings)
        ids = [id(p) for g in groups for p in g['params']]
        assert len(ids) == len(set(ids)) == sum(p.requires_grad for p in current.parameters())
        rates = {g['name']: g['lr'] for g in groups}
        assert rates['head'] == 1e-4
        if scenario in ('S1', 'S3'):
            assert rates['roberta'] == 5e-6
        if scenario == 'S3':
            assert rates['pooling_norm'] == rates['projection'] == 1e-4
        (pred-.8).square().mean().backward()
        torch.optim.AdamW(groups, weight_decay=.01).step()
    torch.testing.assert_close(*predictions, rtol=0, atol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, control.state_dict()[key], rtol=0, atol=0)


def test_full_native_notebook_all_three_scenarios_excel_and_fresh_kernel_resume(tmp_path):
    n = nbformat.read(NOTEBOOK, as_version=4)
    isolated = tmp_path/'standalone'
    isolated.mkdir()
    fixture = isolated/'TECHNICAL_FIXTURE_ONLY.xlsx'
    make_book(fixture)
    workspace = isolated/'workspace'
    guard = """import builtins
_original_import=builtins.__import__
def _no_project_import(name,*args,**kwargs):
    if name=='aes' or name.startswith('aes.'):raise ImportError('Native fixture forbids project imports')
    return _original_import(name,*args,**kwargs)
builtins.__import__=_no_project_import
"""
    tiny = """
from transformers import RobertaConfig,RobertaModel,XLMRobertaConfig,XLMRobertaModel
def encoder(cfg,role):
    config,model,dim,heads=(RobertaConfig,RobertaModel,768,12) if role=='roberta' else (XLMRobertaConfig,XLMRobertaModel,1024,16)
    with torch.random.fork_rng():
        torch.manual_seed(170)
        result=model(config(vocab_size=64,hidden_size=dim,num_hidden_layers=1,num_attention_heads=heads,
            intermediate_size=16,max_position_embeddings=64,pad_token_id=1,bos_token_id=0,eos_token_id=2,
            hidden_dropout_prob=0.,attention_probs_dropout_prob=0.),add_pooling_layer=False)
    result.requires_grad_(role=='roberta');return result
def tokenizer(*a,**k):return ThreeTechnicalTokenizer()
"""
    for cell in n.cells:
        if cell.cell_type != 'code':
            continue
        role = cell.metadata[ROLE]
        if role == 'hyperparameters':
            replacements = {'EPOCHS = 25': 'EPOCHS = 1', "DEVICE = 'cuda'": "DEVICE = 'cpu'",
                'REQUIRE_GPU = True': 'REQUIRE_GPU = False', 'ROBERTA_MAX_TOKENS = 512': 'ROBERTA_MAX_TOKENS = 32',
                'BGE_MAX_TOKENS = 1024': 'BGE_MAX_TOKENS = 32', 'WINDOW_OVERLAP = 128': 'WINDOW_OVERLAP = 4',
                'CONTEXT_MAX_TOKENS = 192': 'CONTEXT_MAX_TOKENS = 12', 'QUESTION_MAX_TOKENS = 64': 'QUESTION_MAX_TOKENS = 4'}
            for before, after in replacements.items():
                assert before in cell.source
                cell.source = cell.source.replace(before, after)
        if role == 'input_columns':
            cell.source = cell.source.replace('EXPECTED_ROWS = 300', 'EXPECTED_ROWS = 24')
        if role == 'definition:tokenizer':
            cell.source += '\n'+inspect.getsource(TechnicalTokenizer)+'\n'+inspect.getsource(context.PairTechnicalTokenizer)+'\n'+inspect.getsource(ThreeTechnicalTokenizer)+tiny
            cell.source = cell.source.replace('context.PairTechnicalTokenizer', 'PairTechnicalTokenizer')
    n.cells.insert(0, nbformat.v4.new_code_cell(guard))
    n.cells.append(nbformat.v4.new_code_cell("""
assert SCENARIOS==('S1','S2','S3')
assert list(X.columns)==INPUT_COLUMNS and X.shape==(24,3) and y.shape==(24,)
assert TARGET_COLUMN not in X.columns and X[ANSWER_COLUMN].tolist()==bundle.texts
assert list(X[[QUESTION_COLUMN,REFERENCE_COLUMN]].itertuples(index=False,name=None))==contexts_from_bundle(bundle)
assert np.array_equal(y.to_numpy(),bundle.y)
assert fit.__globals__ is globals() and make_model.__globals__ is globals()
"""))
    path = isolated/'TECHNICAL_input.ipynb'
    nbformat.write(n, path)
    kernels = tmp_path/'kernels'
    spec = kernels/'e30full_fixture'
    spec.mkdir(parents=True)
    (spec/'kernel.json').write_text(json.dumps({'argv': [sys.executable, '-I', '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
                                              'display_name': 'E30 full fixture', 'language': 'python'}))
    env = {**os.environ, 'AES300_WORKSPACE': str(workspace), 'AES300_DATASET': str(fixture),
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'OMP_NUM_THREADS': '2',
           'MKL_NUM_THREADS': '2', 'IPYTHONDIR': str(tmp_path/'ipython')}
    env.pop('PYTHONPATH', None)
    for attempt in (1, 2):
        current = nbformat.read(path, as_version=4)
        manager = KernelManager(kernel_name='e30full_fixture', kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
        try:
            NotebookClient(current, timeout=360, km=manager, resources={'metadata': {'path': str(isolated)}}).execute(env=env)
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        nbformat.write(current, isolated/f'TECHNICAL_attempt_{attempt}.ipynb')
        info = json.loads((workspace/'outputs/dataset300_download.json').read_text())
        run = Path(info['run_folder'])
        report = json.loads((run/'final_report.json').read_text())
        assert info['excel_count'] == 5 and len(report['cv']) == 15 and report['test_evaluations_per_scenario'] == 1
        assert {r['Skenario'] for r in report['rows']} == set(report['checkpoints']) == set(report['selected_epochs']) == {'S1', 'S2', 'S3'}
        hashes = {p.parent.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.glob('S*/final.pt')}
        assert len(hashes) == 3
        if attempt == 1:
            first_hashes = hashes
        else:
            assert first_hashes == hashes
    config = json.loads((run/'effective_config.json').read_text())
    assert config['s3_layers_lr'] == config['s3_mlp_lr'] == config['s2_mlp_lr'] == config['projection_lr'] == 1e-4
    history = run/'hasil_ringkas/05_Riwayat_Pelatihan.xlsx'
    for name in ['RoBERTa (S1)', 'BGE-M3 (S2)', 'Fusion (S3)']:
        assert len(pd.read_excel(history, sheet_name=name)) == 6
    summary = pd.read_excel(history, sheet_name='Ringkasan Lima Fold')
    assert len(summary) == 15 and set(summary.Skenario) == {'S1', 'S2', 'S3'}
    settings = pd.read_excel(history, sheet_name='Pengaturan Eksperimen')
    settings = dict(zip(settings.Pengaturan, settings.Nilai))
    assert float(settings['LR pooling dan norm S3']) == float(settings['LR MLP S3']) == 1e-4
    predictions = pd.read_excel(run/'hasil_ringkas/04_Prediksi_dan_Evaluasi.xlsx', sheet_name='Nilai Guru dan Prediksi')
    for scenario in ['S1', 'S2', 'S3']:
        assert predictions['Prediksi '+scenario].notna().all()
        checkpoint = torch.load(run/scenario/'final.pt', map_location='cpu', weights_only=True)
        assert checkpoint['training']['optimizer_learning_rates']['head'] == 1e-4
        if scenario == 'S3':
            assert checkpoint['training']['optimizer_learning_rates']['pooling_norm'] == 1e-4
        assert len(list((run/scenario).glob('cv_fold_*.json'))) == 5
    with ZipFile(info['archive']) as z:
        assert len(z.namelist()) == 5
    assert Path(info['archive']).name == 'Hasil_'+NOTEBOOK.stem+'.zip'
    assert not list(workspace.rglob('*.py'))
    proof = {'scope': 'TECHNICAL RANDOM TRANSFORMER FIXTURE ONLY', 'cv_fits': 15, 'refits': 3,
             'test_evaluations_per_scenario': 1, 'reports': 5, 'execution_device': 'cpu',
             'fresh_kernel_resume_unchanged_checkpoints': True, 'complete_scenarios': ['S1', 'S2', 'S3']}
    (isolated/'proof.json').write_text(json.dumps(proof, indent=2))
    print(json.dumps(proof))
