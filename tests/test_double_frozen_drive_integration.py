"""Composition: real Torch saver/queue plus real cleaner on simulated mounted Drive.

Only transport and filesystem synchronization are simulated. No real account,
GPU, research essay, or student score is used.
"""
import hashlib
import json
from pathlib import Path

import pytest
import torch

from aes.teacher_score import experimental_checkpoints as checkpoints
from aes.teacher_score.data import dump
from aes.teacher_score.experiment import load_training_progress
from aes.teacher_score.drive_checkpoint_cleanup import CleanupDeferred, DRIVE_CLEANUP_SCOPE
from test_drive_checkpoint_cleanup import drive_state


class MountedDriveBridge:
    """Simulate Drive mount upload and API deletions becoming visible locally."""

    def __init__(self, state):
        self.state = state
        self.bindings = {}
        self.requests = []
        self.deleted_metadata = []
        self.server_checksum_lag = False
        state.api.on_after_delete = self.reflect_api_delete

    def sync(self, force=False):
        for path in sorted(self.state.run.rglob('*')):
            if not path.is_file():
                continue
            if path.parent == self.state.run:
                parent = self.state.remote_run
            elif path.parent == self.state.folder:
                parent = self.state.remote_folder
            else:
                raise AssertionError('Unexpected fixture folder: '+str(path.parent))
            raw = path.read_bytes()
            checksum = hashlib.sha256(raw).hexdigest()
            previous = self.bindings.get(path)
            if previous and previous['sha256'] == checksum and not force:
                continue
            if previous and previous['id'] in self.state.api.items:
                identifier = previous['id']
                self.state.api.items[identifier].update(
                    _content=raw, size=str(len(raw)), sha256Checksum=checksum,
                    md5Checksum=hashlib.md5(raw, usedforsecurity=False).hexdigest())
            else:
                identifier = self.state.api.add(path.name, parent, raw)
            self.bindings[path] = {'id': identifier, 'sha256': checksum}

    def reflect_api_delete(self, identifier):
        # This callback is the simulated mount, not the saver substituting unlink.
        for path, binding in list(self.bindings.items()):
            if binding['id'] == identifier:
                self.deleted_metadata.append({'id': identifier, 'name': path.name})
                if path.exists():
                    path.unlink()
                self.bindings.pop(path)
                return

    def __call__(self, request):
        self.requests.append(dict(request))
        self.sync()
        if self.server_checksum_lag and request['mode'] == 'active':
            marker = self.state.folder/(request['stage']+'.json')
            latest = json.loads(marker.read_text())['snapshots'][0]['file']
            identifier = self.bindings[self.state.folder/latest]['id']
            self.state.api.items[identifier]['sha256Checksum'] = '0'*64
        try:
            return {'ok': True, 'result': self.state.cleaner(request)}
        except CleanupDeferred as error:
            return {'ok': False, 'retryable': True, 'error': str(error)}
        except PermissionError as error:
            return {'ok': False, 'retryable': False, 'error': str(error)}


def install_bridge(state, monkeypatch):
    bridge = MountedDriveBridge(state)
    monkeypatch.setenv('AES300_DRIVE_CLEANUP', '1')
    monkeypatch.setattr(checkpoints, '_drive_cleanup', bridge, raising=False)
    return bridge


def payload(epoch):
    return {'epoch': epoch, 'contract': {'scope': 'TECHNICAL COMPOSITION FIXTURE'},
            'model': {'head': torch.tensor([[epoch*.1, epoch*.2]])},
            'history': [{'epoch': epoch, 'loss': .1/epoch}]}


def assert_snapshot_proofs(path):
    manifest = json.loads(path.read_text())
    assert manifest['scope'] == DRIVE_CLEANUP_SCOPE
    for row in manifest['snapshots']:
        file = path.parent/row['file']
        raw = file.read_bytes()
        assert row['sha256'] == hashlib.sha256(raw).hexdigest()
        assert row['md5'] == hashlib.md5(raw, usedforsecurity=False).hexdigest()
        assert row['size'] == len(raw)
        saved = torch.load(file, map_location='cpu', weights_only=True)
        assert saved['epoch'] == int(file.stem.rsplit('_', 1)[1])
    return manifest


def test_real_saver_and_drive_cleaner_keep_latest_two_and_delete_old_trash(drive_state, monkeypatch):
    s = drive_state
    bridge = install_bridge(s, monkeypatch)
    path = s.folder/'progress_fold_1.json'
    checkpoints.save_training_progress(path, payload(1))
    first_id = bridge.bindings[s.folder/'progress_fold_1_epoch_001.pt']['id']
    marker_id = bridge.bindings[path]['id']
    checkpoints.save_training_progress(path, payload(2))
    assert bridge.bindings[path]['id'] == marker_id
    # Simulate an old entry still present in mounted cache after moving to Trash.
    # The new generation makes epoch 1 obsolete before cleaner releases it.
    s.api.items[first_id]['trashed'] = True
    checkpoints.save_training_progress(path, payload(3))
    manifest = assert_snapshot_proofs(path)
    assert [x['file'] for x in manifest['snapshots']] == [
        'progress_fold_1_epoch_003.pt', 'progress_fold_1_epoch_002.pt']
    remote_names = {v['name'] for v in s.api.items.values()
                    if s.remote_folder in v.get('parents', []) and v['name'].endswith('.pt')}
    assert remote_names == {x['file'] for x in manifest['snapshots']}
    assert first_id in s.api.deleted and first_id not in s.api.items
    assert len(list(s.folder.glob('*.pt'))) == 2
    assert load_training_progress(path, payload(1)['contract'])['epoch'] == 3
    assert json.loads((s.run/'_checkpoint_cleanup_pending.json').read_text())['requests'] == {}
    assert len(bridge.requests) == 3 and all(r['mode'] == 'active' for r in bridge.requests)


def test_real_saver_continues_with_stale_server_pointer_and_cleans_after_sync(drive_state, monkeypatch):
    import copy
    s = drive_state
    class LaggedPointerBridge(MountedDriveBridge):
        def __init__(self, state):
            super().__init__(state)
            self.frozen_pointer = None
        def sync(self, force=False):
            super().sync(force=force)
            path = s.folder/'progress_fold_1.json'
            binding = self.bindings.get(path)
            if self.frozen_pointer is not None and binding:
                s.api.items[binding['id']] = copy.deepcopy(self.frozen_pointer)
    bridge = LaggedPointerBridge(s)
    monkeypatch.setenv('AES300_DRIVE_CLEANUP', '1')
    monkeypatch.setattr(checkpoints, '_drive_cleanup', bridge, raising=False)
    path = s.folder/'progress_fold_1.json'
    checkpoints.save_training_progress(path, payload(1))
    bridge.frozen_pointer = copy.deepcopy(s.api.items[bridge.bindings[path]['id']])
    for epoch in (2, 3, 4):
        checkpoints.save_training_progress(path, payload(epoch))
    # Latest two plus the older server-referenced epoch stay available.
    assert {p.name for p in s.folder.glob('*.pt')} == {
        'progress_fold_1_epoch_001.pt', 'progress_fold_1_epoch_003.pt', 'progress_fold_1_epoch_004.pt'}
    assert load_training_progress(path, payload(1)['contract'])['epoch'] == 4
    marker = json.loads(bridge.frozen_pointer['_content'])
    server_snapshot = marker['snapshots'][0]
    server_file = s.folder/server_snapshot['file']
    assert hashlib.sha256(server_file.read_bytes()).hexdigest() == server_snapshot['sha256']
    assert torch.load(server_file, weights_only=True)['epoch'] == 1
    bridge.frozen_pointer = None
    bridge.sync(force=True)
    checkpoints.request_checkpoint_cleanup(path, 'active')
    assert {p.name for p in s.folder.glob('*.pt')} == {
        'progress_fold_1_epoch_003.pt', 'progress_fold_1_epoch_004.pt'}
    assert load_training_progress(path, payload(1)['contract'])['epoch'] == 4
    assert json.loads((s.run/'_checkpoint_cleanup_pending.json').read_text())['requests'] == {}
    assert len(bridge.requests) == 5 and all(r['mode'] == 'active' for r in bridge.requests)


def test_real_completed_fold_clear_checks_result_then_keeps_final_and_other_stage(drive_state, monkeypatch):
    s = drive_state
    bridge = install_bridge(s, monkeypatch)
    path = s.folder/'progress_fold_1.json'
    for epoch in range(1, 4):
        checkpoints.save_training_progress(path, payload(epoch))
    split = {
        'scope': DRIVE_CLEANUP_SCOPE,
        'source_sha256': hashlib.sha256(b'TECHNICAL WORKBOOK ONLY').hexdigest(),
        'development': [0, 1, 2, 3], 'test': [4, 5],
        'folds': [{'fold': 1, 'train': [0, 1], 'validation': [2, 3]}],
    }
    split['split_sha256'] = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
    dump(s.run/'split_manifest.json', split)
    result = {'fold': 1, 'split_sha256': split['split_sha256'],
              'training': {'scenario': s.scenario, 'best_epoch': 2,
                           'history': [payload(1)['history'][0], payload(2)['history'][0]]},
              'metrics': {'RMSE': 2., 'Pearson': .8, 'QWK': .7, 'AC2': .9},
              'validation_indices': [2, 3], 'prediction': [80., 90.]}
    dump(s.folder/'cv_fold_1.json', result)
    cv_before = (s.folder/'cv_fold_1.json').read_bytes()
    torch.save({'scope': 'TECHNICAL FINAL SENTINEL'}, s.folder/'final.pt')
    torch.save(payload(1), s.folder/'progress_fold_2_epoch_001.pt')
    (s.folder/'progress_fold_1_epoch_004.pt.pending').write_bytes(b'PARTIAL TECHNICAL CHECKPOINT')
    bridge.sync()
    final_id = bridge.bindings[s.folder/'final.pt']['id']
    other_id = bridge.bindings[s.folder/'progress_fold_2_epoch_001.pt']['id']
    kept = {name: (s.folder/name).read_bytes()
            for name in ['final.pt', 'progress_fold_2_epoch_001.pt']}
    checkpoints.clear_training_progress(path)
    assert bridge.requests[-1]['mode'] == 'complete'
    assert not path.exists()
    assert not list(s.folder.glob('progress_fold_1_epoch_*'))
    assert final_id in s.api.items and other_id in s.api.items
    assert all((s.folder/name).read_bytes() == content for name, content in kept.items())
    assert (s.folder/'cv_fold_1.json').read_bytes() == cv_before
    assert not any(item['name'] == 'final.pt' for item in bridge.deleted_metadata)
    assert json.loads((s.run/'_checkpoint_cleanup_pending.json').read_text())['requests'] == {}


def test_real_saver_pauses_on_bad_server_proof_then_pending_retry_recovers(drive_state, monkeypatch):
    s = drive_state
    bridge = install_bridge(s, monkeypatch)
    delays = []
    monkeypatch.setattr(checkpoints.time, 'sleep', delays.append)
    path = s.folder/'progress_fold_1.json'
    for epoch in [1, 2]:
        checkpoints.save_training_progress(path, payload(epoch))
    first_id = bridge.bindings[s.folder/'progress_fold_1_epoch_001.pt']['id']
    bridge.server_checksum_lag = True
    with pytest.raises(RuntimeError, match='Training dihentikan'):
        checkpoints.save_training_progress(path, payload(3))
    assert delays == [2, 4, 8, 16]
    assert first_id in s.api.items and s.api.deleted == []
    assert load_training_progress(path, payload(1)['contract'])['epoch'] == 3
    assert_snapshot_proofs(path)
    queue = s.run/'_checkpoint_cleanup_pending.json'
    requests = json.loads(queue.read_text())['requests']
    assert list(requests) == ['S3/progress_fold_1']
    bridge.server_checksum_lag = False
    bridge.sync(force=True)
    checkpoints.retry_pending_checkpoint_cleanup(s.run)
    assert first_id not in s.api.items and first_id in s.api.deleted
    assert len(list(s.folder.glob('*.pt'))) == 2
    assert json.loads(queue.read_text())['requests'] == {}
    assert load_training_progress(path, payload(1)['contract'])['epoch'] == 3
