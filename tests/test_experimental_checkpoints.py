"""Checkpoint and actual worker/kernel cleanup protocol; no real Drive account."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest
import torch

from aes.teacher_score import experimental_checkpoints as checkpoints
from aes.teacher_score.experiment import load_training_progress
from aes.teacher_score.drive_checkpoint_cleanup import CleanupDeferred


def test_local_saver_keeps_two_intact_generations_and_checksums(tmp_path, monkeypatch):
    monkeypatch.delenv('AES300_DRIVE_CLEANUP', raising=False)
    path = tmp_path/'S3/progress_refit.json'
    contract = {'fixture': 'TECHNICAL ONLY'}
    for epoch in range(1, 5):
        checkpoints.save_training_progress(path, {'epoch': epoch, 'contract': contract, 'value': torch.tensor([epoch])})
    manifest = json.loads(path.read_text())
    assert [r['file'] for r in manifest['snapshots']] == ['progress_refit_epoch_004.pt', 'progress_refit_epoch_003.pt']
    assert len(list(path.parent.glob('*.pt'))) == 2
    for row in manifest['snapshots']:
        value = (path.parent/row['file']).read_bytes()
        assert row['sha256'] == hashlib.sha256(value).hexdigest()
        assert row['md5'] == hashlib.md5(value, usedforsecurity=False).hexdigest()
        assert row['size'] == len(value)
    assert load_training_progress(path, contract)['epoch'] == 4


def test_failed_save_preserves_previous_checkpoint_and_does_not_request_delete(tmp_path, monkeypatch):
    path = tmp_path/'S3/progress_refit.json'
    monkeypatch.delenv('AES300_DRIVE_CLEANUP', raising=False)
    checkpoints.save_training_progress(path, {'epoch': 1, 'contract': {}})
    before = path.read_bytes()
    calls = []
    monkeypatch.setenv('AES300_DRIVE_CLEANUP', '1')
    monkeypatch.setattr(checkpoints, '_drive_cleanup', lambda r: calls.append(r), raising=False)
    def fail(payload, target):
        Path(target).write_bytes(b'PARTIAL FIXTURE')
        raise OSError('SIMULATED DRIVE FULL')
    monkeypatch.setattr(checkpoints.torch, 'save', fail)
    with pytest.raises(OSError, match='FULL'):
        checkpoints.save_training_progress(path, {'epoch': 2, 'contract': {}})
    assert path.read_bytes() == before and calls == []
    assert load_training_progress(path, {})['epoch'] == 1


def test_drive_failure_preserves_progress_and_retries_persistent_queue(tmp_path, monkeypatch):
    path = tmp_path/'abcdef0123456789/S3/progress_refit.json'
    monkeypatch.setenv('AES300_DRIVE_CLEANUP', '1')
    delays, calls = [], []
    monkeypatch.setattr(checkpoints.time, 'sleep', delays.append)
    def unavailable(request):
        calls.append(request.copy())
        assert path.exists()
        return {'ok': False, 'retryable': True, 'error': 'SIMULATED INDEX LAG'}
    monkeypatch.setattr(checkpoints, '_drive_cleanup', unavailable, raising=False)
    with pytest.raises(RuntimeError, match='Training dihentikan'):
        checkpoints.save_training_progress(path, {'epoch': 1, 'contract': {}})
    assert len(calls) == 5 and delays == [2, 4, 8, 16]
    assert load_training_progress(path, {})['epoch'] == 1
    queue = path.parent.parent/'_checkpoint_cleanup_pending.json'
    assert len(json.loads(queue.read_text())['requests']) == 1
    monkeypatch.setattr(checkpoints, '_drive_cleanup', lambda r: {'ok': True, 'result': {'deleted': 0}}, raising=False)
    checkpoints.retry_pending_checkpoint_cleanup(path.parent.parent)
    assert json.loads(queue.read_text())['requests'] == {}
    assert (path.parent/'progress_refit_epoch_001.pt').exists()


def test_permission_failure_is_not_retried_and_clear_keeps_manifest(tmp_path, monkeypatch):
    path = tmp_path/'abcdef0123456789/S3/progress_refit.json'
    monkeypatch.delenv('AES300_DRIVE_CLEANUP', raising=False)
    checkpoints.save_training_progress(path, {'epoch': 1, 'contract': {}})
    calls = []
    monkeypatch.setenv('AES300_DRIVE_CLEANUP', '1')
    def forbidden(request):
        calls.append(request)
        return {'ok': False, 'retryable': False, 'error': 'SIMULATED PERMISSION FAILURE'}
    monkeypatch.setattr(checkpoints, '_drive_cleanup', forbidden, raising=False)
    with pytest.raises(RuntimeError, match='PERMISSION'):
        checkpoints.clear_training_progress(path)
    assert len(calls) == 1 and calls[0]['mode'] == 'complete'
    assert path.exists() and list(path.parent.glob('*.pt'))


def test_drive_save_uses_broker_without_filesystem_unlink_for_checkpoint(tmp_path, monkeypatch):
    path = tmp_path/'abcdef0123456789/S3/progress_refit.json'
    monkeypatch.setenv('AES300_DRIVE_CLEANUP', '1')
    calls = []
    def broker(request):
        calls.append(request)
        # API transport is simulated; saver must not substitute unlink().
        return {'ok': True, 'result': {'deleted': 0}}
    monkeypatch.setattr(checkpoints, '_drive_cleanup', broker, raising=False)
    for epoch in range(1, 4):
        checkpoints.save_training_progress(path, {'epoch': epoch, 'contract': {}})
    assert len(list(path.parent.glob('*.pt'))) == 3
    assert len(json.loads(path.read_text())['snapshots']) == 2
    assert all(r['mode'] == 'active' for r in calls)


@pytest.mark.parametrize('failure', [None, 'deferred', 'permission'])
def test_actual_kernel_worker_cleanup_round_trip_always_replies(tmp_path, failure):
    notebook = json.loads((Path(__file__).resolve().parents[1]/'notebooks/experiments/AES_300_S3_DoubleFrozen_MLP.ipynb').read_text())
    worker_cell = next(c for c in notebook['cells'] if 'WORKER_SOURCE=' in c['source'])
    worker_source = ast.literal_eval(next(n for n in ast.parse(worker_cell['source']).body if isinstance(n, ast.Assign)).value)
    setup = next(c['source'] for c in notebook['cells'] if 'def aes_cell(' in c['source'])
    aes_node = next(n for n in ast.parse(setup).body if isinstance(n, ast.FunctionDef) and n.name == 'aes_cell')
    prefix = '__TECHNICAL_BROKER__'
    process = subprocess.Popen([sys.executable, '-u', '-c', worker_source, prefix, str(tmp_path)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    calls = []
    def cleaner(request):
        calls.append(request)
        if failure == 'deferred':
            raise CleanupDeferred('SIMULATED INDEX LAG')
        if failure == 'permission':
            raise PermissionError('SIMULATED PERMISSION FAILURE')
        return {'deleted': 1, 'bytes_deleted': 123}
    namespace = {'_aes_process': process, '_aes_prefix': prefix, '_aes_cleaner': cleaner,
                 'CleanupDeferred': CleanupDeferred, 'json': json, 'display_aes_table': lambda e: None}
    exec(compile(ast.Module(body=[aes_node], type_ignores=[]), '<actual-cleanup-kernel>', 'exec'), namespace)
    errors = []
    expected_ok = failure is None
    expected_retry = failure == 'deferred'
    code = f"reply=_drive_cleanup({{'run_id':'abcdef0123456789','scenario':'S3','stage':'progress_refit','mode':'active'}})\nassert reply['ok'] is {expected_ok}\n"
    if failure:
        code += f"assert reply['retryable'] is {expected_retry}\n"
    def invoke():
        try:
            namespace['aes_cell']('', code)
        except BaseException as error:
            errors.append(error)
    try:
        ready = json.loads(process.stdout.readline().removeprefix(prefix))
        assert ready['kind'] == 'ready'
        thread = threading.Thread(target=invoke, daemon=True)
        thread.start()
        thread.join(8)
        assert not thread.is_alive(), 'Kernel failed to reply; worker deadlocked'
        assert not errors, errors
        assert len(calls) == 1
    finally:
        process.terminate()
        process.communicate(timeout=8)
