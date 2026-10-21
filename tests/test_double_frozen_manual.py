"""Manual checkpoint retention and full notebook execution on technical fixtures."""
import ast
import hashlib
import json
from pathlib import Path

import nbformat
import pytest
import torch

from aes.teacher_score import manual_checkpoints as checkpoints
from aes.teacher_score.experiment import load_training_progress
from test_teacher_score import dataset, encoders
import test_double_frozen_notebook as frozen


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT/'notebooks/experiments/AES_300_S3_DoubleFrozen_MLP_Manual.ipynb'


def test_manual_notebook_is_separate_clean_and_has_no_cleanup_protocol():
    n = nbformat.read(NOTEBOOK, as_version=4)
    assert len(n.cells) == 47 and sum(c.cell_type == 'code' for c in n.cells) == 30
    assert n.metadata.aes_experiment.name == 'S3_DOUBLE_FROZEN_MLP_MANUAL'
    assert n.metadata.aes_experiment.automatic_checkpoint_deletion is False
    assert n.metadata.aes_experiment.drive_api_cleanup is False
    for name in ('notebooks/03_AES_dataset_guru_300.ipynb',
                 'aes-learning/notebooks/03_AES_dataset_guru_300.ipynb'):
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == frozen.SEMIFINAL_SHA256
    for cell in n.cells:
        assert not cell.get('attachments')
        if cell.cell_type != 'code':continue
        assert cell.execution_count is None and cell.outputs == []
        for forbidden in ('DriveCheckpointCleaner', 'googleapiclient', 'auth.authenticate_user',
                          '_drive_cleanup', 'retry_pending_checkpoint_cleanup',
                          'sweep_completed_checkpoint_cleanup', 'clear_training_progress'):
            assert forbidden not in cell.source
    joined = '\n'.join(c.source for c in n.cells)
    assert 'AES_300_DoubleFrozen_MLP_Manual_Privat' in joined
    assert "lesson_version='s3_double_frozen_mlp_manual_v1'" in joined


def test_manual_model_methods_equal_double_frozen_model_methods():
    def model_source(path):
        n = nbformat.read(path, as_version=4)
        cell = next(c for c in n.cells if c.cell_type == 'code' and 'class SingleScoreModel' in c.source)
        tree = ast.parse(cell.source.removeprefix('%%aes\n'))
        return ast.dump(next(node for node in tree.body
                             if isinstance(node, ast.ClassDef) and node.name == 'SingleScoreModel'),
                        include_attributes=False)
    assert model_source(NOTEBOOK) == model_source(frozen.NOTEBOOK)


def test_all_epochs_survive_save_and_recovery_uses_latest_or_backup(tmp_path):
    path = tmp_path/'S3/progress_fold_1.json'
    for epoch in range(1, 6):
        checkpoints.save_training_progress(path, {'epoch': epoch, 'contract': {},
                                                  'model': torch.tensor([float(epoch)])})
    snapshots = json.loads(path.read_text())['snapshots']
    assert [s['file'] for s in snapshots] == ['progress_fold_1_epoch_005.pt', 'progress_fold_1_epoch_004.pt']
    assert len(list(path.parent.glob('*.pt'))) == 5
    assert load_training_progress(path, {})['epoch'] == 5
    # Recovery can use the backup without deleting the failed latest file.
    (path.parent/snapshots[0]['file']).write_bytes(b'CORRUPT TECHNICAL CHECKPOINT')
    assert load_training_progress(path, {})['epoch'] == 4
    assert len(list(path.parent.glob('*.pt'))) == 5


def test_failed_save_never_deletes_previous_progress(tmp_path, monkeypatch):
    path = tmp_path/'S3/progress_refit.json'
    checkpoints.save_training_progress(path, {'epoch': 1, 'contract': {}})
    before = {p.name: p.read_bytes() for p in path.parent.iterdir()}
    def fail(payload, target):
        Path(target).write_bytes(b'PARTIAL TECHNICAL CHECKPOINT')
        raise OSError('TECHNICAL WRITE FAILURE')
    monkeypatch.setattr(checkpoints.torch, 'save', fail)
    with pytest.raises(OSError, match='WRITE FAILURE'):
        checkpoints.save_training_progress(path, {'epoch': 2, 'contract': {}})
    assert all((path.parent/name).read_bytes() == data for name, data in before.items())
    assert load_training_progress(path, {})['epoch'] == 1


def test_manual_s3_encoder_frozen_and_scoring_layers_trainable(monkeypatch):
    monkeypatch.setattr(frozen, 'NOTEBOOK', NOTEBOOK)
    frozen.test_s3_freezes_encoder_without_freezing_projection_layernorm_or_mlp()


@pytest.mark.parametrize('cuda_available', [False, True])
def test_manual_cpu_and_gpu_configurations(monkeypatch, tmp_path, capsys, cuda_available):
    monkeypatch.setattr(frozen, 'NOTEBOOK', NOTEBOOK)
    frozen.test_configuration_accepts_cpu_and_still_detects_gpu(monkeypatch, tmp_path, capsys, cuda_available)


def test_manual_training_resumes_identically_after_interruption(dataset, encoders, tmp_path, monkeypatch):
    monkeypatch.setattr(frozen, 'NOTEBOOK', NOTEBOOK)
    frozen.test_interrupted_s3_fit_resumes_identical_frozen_state_history_and_predictions(
        dataset, encoders, tmp_path, monkeypatch)


def test_manual_complete_pipeline_five_excel_and_reuse_keeps_all_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(frozen, 'NOTEBOOK', NOTEBOOK)
    frozen.test_experiment_runs_to_five_excel_and_reuses_persistent_results(tmp_path)
    pointer = json.loads((tmp_path/'only_notebook_and_data/private_results/outputs/dataset300_download.json').read_text())
    run = Path(pointer['run_folder'])
    for scenario in ('S1', 'S2', 'S3'):
        folder = run/scenario
        assert len(list(folder.glob('progress_fold_*_epoch_001.pt'))) == 5
        assert len(list(folder.glob('progress_fold_*.json'))) == 5
        assert list(folder.glob('progress_refit_epoch_*.pt'))
        assert (folder/'progress_refit.json').is_file()
        assert (folder/'final.pt').is_file()
    assert not (run/'_checkpoint_cleanup_pending.json').exists()
