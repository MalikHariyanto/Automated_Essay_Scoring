"""E29 Ridge: standalone architecture and full technical notebook execution."""
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import nbformat
import test_s3_ridge_notebook as base

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT/'notebooks/experiments/AES_300_E29_BiGRU_Attention_Ridge_Colab.ipynb'


def test_e29_source_optimizer_and_reproducible_build():
    raw = NOTEBOOK.read_bytes()
    n = nbformat.reads(raw.decode(), as_version=4)
    nbformat.validate(n)
    identity = n.metadata.aes_standalone.code_sha256
    codes = [c.source.removeprefix('%%aes\n') for c in n.cells if c.cell_type == 'code']
    for s in codes:
        ast.parse(s)
    assert hashlib.sha256('\n'.join(codes).replace(identity, '__S3_RIDGE_CODE_HASH__').encode()).hexdigest() == identity
    original = json.loads(subprocess.check_output(['git', 'show',
        '47ddb8280592231fbb682f2e6a8b3c5735d185fb:notebooks/experiments/AES_300_E29_E26_BiGRU_Attention_MLPLR5e6_GPU.ipynb'], cwd=ROOT))
    from build_bge_readout_notebook import definitions
    architecture = definitions(''.join(original['cells'][30]['source']).removeprefix('%%aes\n'))
    assert architecture in '\n'.join(codes)
    optimizer = definitions(''.join(original['cells'][36]['source']).removeprefix('%%aes\n'), {'optimizer_groups'})
    assert optimizer in '\n'.join(codes)
    assert 's3_mlp_lr=5e-6' in '\n'.join(codes)
    assert 's3_layers_lr=1e-4' in '\n'.join(codes)
    assert all(not c.outputs and c.execution_count is None for c in n.cells if c.cell_type == 'code')
    subprocess.run([sys.executable, 'scripts/build_e29_ridge_notebook.py'], cwd=ROOT, check=True)
    assert raw == NOTEBOOK.read_bytes()
    assert raw == (ROOT/'aes-learning/notebooks/experiments'/NOTEBOOK.name).read_bytes()


def test_full_e29_notebook_technical_training_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(base, 'NOTEBOOK', NOTEBOOK)
    base.test_all_cells_with_random_transformers_and_training_resume(tmp_path)
    pointer = json.loads((tmp_path/'standalone/workspace/outputs/s3_ridge_download.json').read_text())
    result = json.loads((Path(pointer['run_folder'])/'s3_ridge_result.json').read_text())
    assert result['protocol']['experiment'] == 'E29_BIGRU_ATTENTION_RIDGE'
    for stage in result['representation_stages']:
        info = stage['proof']['info']
        assert info['optimizer_learning_rates']['head'] == 5e-6
        assert info['optimizer_learning_rates']['pooling_norm'] == 1e-4
        assert info['parameter_samples_changed']['sequence_pool']


def test_e29_ridge_label_isolation_and_cache_integrity(tmp_path, monkeypatch):
    monkeypatch.setattr(base, 'NOTEBOOK', NOTEBOOK)
    original = base.fixture
    def fixture(folder):
        values = original(folder)
        values[1]['representation_policy'] = 'E29_fusion_fixed_epochs_train_only'
        return values
    monkeypatch.setattr(base, 'fixture', fixture)
    first = tmp_path/'isolation'
    first.mkdir()
    base.test_supervised_inner_folds_never_see_outer_labels_and_pair_cache_is_shared(first)
    second = tmp_path/'cache'
    second.mkdir()
    base.test_completed_cache_resume_export_and_raw_feature_model_reload(second)
