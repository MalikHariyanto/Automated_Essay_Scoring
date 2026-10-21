"""Putus/lanjut pada fixture teknis; tidak mengakses akun Drive atau data siswa."""
import ast
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from aes.teacher_score import data, experiment, models
from test_teacher_score import dataset, encoders


@pytest.mark.parametrize('scenario', ['S1', 'S2', 'S3'])
def test_interrupted_epoch_resume_matches_uninterrupted_training(dataset, encoders, monkeypatch, tmp_path, scenario):
    cfg, bundle = dataset
    cfg = {**cfg, 'dropout': .2, 'early_stopping_patience': 5}
    bge = torch.nn.functional.normalize(torch.randn(24, 1024), dim=1)
    train, valid = list(range(8)), list(range(8, 12))
    baseline, baseline_engine, expected = experiment.fit(scenario, train, valid, bundle, bge, cfg, 4, 42)
    expected_state = {k: v.clone() for k, v in baseline.state_dict().items()}
    expected_predictions = baseline_engine.predict([12, 13])
    del baseline_engine, baseline
    manifest = tmp_path/'progress.json'
    original_save = experiment.save_training_progress
    def disconnect_after_saved_epoch(path, payload):
        original_save(path, payload)
        if payload['epoch'] == 2:
            raise RuntimeError('SIMULATED RUNTIME DISCONNECT')
    monkeypatch.setattr(experiment, 'save_training_progress', disconnect_after_saved_epoch)
    with pytest.raises(RuntimeError, match='SIMULATED'):
        experiment.fit(scenario, train, valid, bundle, bge, cfg, 4, 42, checkpoint=manifest)
    monkeypatch.setattr(experiment, 'save_training_progress', original_save)
    actual, engine, info = experiment.fit(scenario, train, valid, bundle, bge, cfg, 4, 42, checkpoint=manifest)
    assert info == expected
    for name, value in actual.state_dict().items():
        torch.testing.assert_close(value, expected_state[name], rtol=0, atol=0)
    np.testing.assert_array_equal(engine.predict([12, 13]), expected_predictions)
    assert len(json.loads(manifest.read_text())['snapshots']) == 2
    experiment.clear_training_progress(manifest)
    assert not manifest.exists() and not list(tmp_path.glob('progress_epoch_*'))


def test_checkpoint_rejects_different_training_and_uses_intact_backup(tmp_path):
    path = tmp_path/'progress.json'
    contract = {'scenario': 'TECHNICAL', 'split': [0, 1]}
    for epoch in (1, 2):
        experiment.save_training_progress(path, {'contract': contract, 'epoch': epoch, 'model': {'x': torch.tensor([epoch])}})
    latest = tmp_path/json.loads(path.read_text())['snapshots'][0]['file']
    latest.write_bytes(b'SIMULATED INCOMPLETE TRANSFER')
    assert experiment.load_training_progress(path, contract)['epoch'] == 1
    with pytest.raises(ValueError, match='berbeda'):
        experiment.load_training_progress(path, {**contract, 'split': [2, 3]})
    for record in json.loads(path.read_text())['snapshots']:
        (tmp_path/record['file']).write_bytes(b'CORRUPT')
    with pytest.raises(RuntimeError, match='utuh'):
        experiment.load_training_progress(path, contract)


def test_failed_checkpoint_write_preserves_previous_snapshot(tmp_path, monkeypatch):
    path = tmp_path/'progress.json'; contract = {'version': 'TECHNICAL'}
    experiment.save_training_progress(path, {'contract': contract, 'epoch': 1})
    def failed_save(payload, destination):
        Path(destination).write_bytes(b'PARTIAL'); raise OSError('SIMULATED DRIVE FULL')
    monkeypatch.setattr(torch, 'save', failed_save)
    with pytest.raises(OSError, match='FULL'):
        experiment.save_training_progress(path, {'contract': contract, 'epoch': 2})
    assert experiment.load_training_progress(path, contract)['epoch'] == 1


def test_interrupted_final_training_resumes_and_completed_refit_is_reused(dataset, encoders, monkeypatch, tmp_path):
    cfg, bundle = dataset
    split = data.make_split(bundle, cfg)
    bge = torch.nn.functional.normalize(torch.randn(24, 1024), dim=1)
    original_save = experiment.save_training_progress
    def disconnect_final(path, payload):
        original_save(path, payload)
        if path.name == 'progress_refit.json' and payload['contract']['scenario'] == 'S2':
            raise RuntimeError('SIMULATED RESTART DURING S2 REFIT')
    monkeypatch.setattr(experiment, 'save_training_progress', disconnect_final)
    with pytest.raises(RuntimeError, match='S2 REFIT'):
        experiment.train_all(bundle, split, bge, cfg, tmp_path)
    assert (tmp_path/'S1/refit_complete.json').exists()
    assert not (tmp_path/'refit_locked.json').exists()
    checksum = data.digest(tmp_path/'S1/final.pt')
    monkeypatch.setattr(experiment, 'save_training_progress', original_save)
    original_fit = experiment.fit
    calls = []
    def tracked_fit(scenario, *args, **kwargs):
        calls.append(scenario)
        return original_fit(scenario, *args, **kwargs)
    monkeypatch.setattr(experiment, 'fit', tracked_fit)
    report = experiment.train_all(bundle, split, bge, cfg, tmp_path)
    assert calls == ['S2', 'S3'] and data.digest(tmp_path/'S1/final.pt') == checksum
    assert report['test_evaluations_per_scenario'] == 1
    assert not list(tmp_path.glob('S*/progress_*.pt'))


def storage_function():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root/'notebooks/03_AES_dataset_guru_300.ipynb').read_text())
    source = next(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code' and 'def prepare_persistent_storage' in ''.join(c['source']))
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_persistent_storage')
    import os
    namespace = {'Path': Path, 'os': os}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<storage>', 'exec'), namespace)
    return namespace['prepare_persistent_storage']


def test_persistent_storage_survives_local_workspace_replacement(tmp_path, monkeypatch):
    persistent = tmp_path/'SIMULATED_PRIVATE_DRIVE'
    monkeypatch.setenv('AES300_PERSISTENT_DIR', str(persistent))
    setup = storage_function()
    first = setup(tmp_path/'runtime_a', False)
    (first/'saved_epoch.txt').write_text('TECHNICAL SAVED EPOCH')
    second = setup(tmp_path/'runtime_b', False)
    assert first == second and (second/'saved_epoch.txt').exists()


def test_drive_authorization_failure_stops_without_local_fallback(tmp_path, monkeypatch):
    module = ModuleType('google.colab')
    def denied(*args):
        raise RuntimeError('SIMULATED DRIVE AUTHORIZATION FAILURE')
    module.drive = SimpleNamespace(mount=denied)
    monkeypatch.setitem(sys.modules, 'google.colab', module)
    with pytest.raises(RuntimeError, match='AUTHORIZATION'):
        storage_function()(tmp_path/'runtime', True)


@pytest.mark.parametrize('name,is_t4', [('Tesla T4', True), ('NVIDIA L4', False)])
def test_gpu_check_reports_actual_gpu_type_and_memory(monkeypatch, name, is_t4):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda index: name)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda index: SimpleNamespace(total_memory=15*1024**3))
    report = models.check_gpu(require_gpu=True)
    assert report == {'name': name, 'cuda_available': True, 'is_t4': is_t4, 'memory_gib': 15.}


def test_gpu_check_stops_colab_training_on_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    assert models.check_gpu()['cuda_available'] is False
    with pytest.raises(RuntimeError, match='T4 GPU'):
        models.check_gpu(require_gpu=True)


def test_early_stopping_counter_and_best_weights_survive_restart(dataset, encoders, monkeypatch, tmp_path):
    cfg, bundle = dataset
    cfg = {**cfg, 'early_stopping_patience': 3}
    bge = torch.nn.functional.normalize(torch.randn(24, 1024), dim=1)
    errors = iter([10., 9., 10., 11., 12.])
    monkeypatch.setattr(experiment, 'metrics', lambda *args: {'RMSE': next(errors)})
    original_save = experiment.save_training_progress
    def disconnected(path, payload):
        original_save(path, payload)
        if payload['epoch'] == 3:
            raise RuntimeError('SIMULATED DISCONNECT')
    monkeypatch.setattr(experiment, 'save_training_progress', disconnected)
    path = tmp_path/'progress.json'
    with pytest.raises(RuntimeError, match='DISCONNECT'):
        experiment.fit('S2', list(range(8)), list(range(8,12)), bundle, bge, cfg, 10, 42, checkpoint=path)
    metadata = json.loads(path.read_text())
    saved = torch.load(tmp_path/metadata['snapshots'][0]['file'], weights_only=True)
    best = saved['best_state']
    monkeypatch.setattr(experiment, 'save_training_progress', original_save)
    model, _, info = experiment.fit('S2', list(range(8)), list(range(8,12)), bundle, bge, cfg, 10, 42, checkpoint=path)
    assert info['early_stopped'] and info['epochs_completed'] == 5 and info['best_epoch'] == 2
    assert [r['epochs_without_improvement'] for r in info['history']] == [0,0,1,2,3]
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, best[key], rtol=0, atol=0)
