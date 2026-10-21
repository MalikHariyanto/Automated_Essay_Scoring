"""Fixture Drive palsu: bukan akses akun pengguna atau file penelitian."""
import copy
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from aes.teacher_score.drive_checkpoint_cleanup import (
    CleanupDeferred, DriveCheckpointCleaner, DRIVE_EXPERIMENT_ROOT,
    DRIVE_FOLDER_MIME, DRIVE_CLEANUP_SCOPE,
)


class FakeHttpError(RuntimeError):
    def __init__(self, status):
        super().__init__('SIMULATED HTTP '+str(status))
        self.resp = SimpleNamespace(status=status)


class FakeRequest:
    def __init__(self, api, operation, arguments, action):
        self.api, self.operation, self.arguments, self.action = api, operation, arguments, action

    def execute(self, num_retries=0):
        self.api.requests.append((self.operation, copy.deepcopy(self.arguments), num_retries))
        for attempt in range(num_retries+1):
            try:
                return self.action()
            except FakeHttpError as error:
                if error.resp.status not in (429, 500, 502, 503, 504) or attempt == num_retries:
                    raise


class FakeDrive:
    def __init__(self):
        self.items = {}
        self.requests = []
        self.deleted = []
        self.delete_attempts = []
        self.counter = 0
        self.page_size = 1000
        self.fail_list = None
        self.delete_failures = {}
        self.delete_then_500 = set()
        self.on_after_delete = None
        self.on_get = None

    def files(self):
        return self

    def add(self, name, parent, content=b'', trashed=False, folder=False):
        self.counter += 1
        identifier = f'file-{self.counter:04d}'
        self.items[identifier] = {
            'id': identifier, 'name': name, 'parents': [parent], 'trashed': trashed,
            'mimeType': DRIVE_FOLDER_MIME if folder else 'application/octet-stream',
            'size': str(len(content)), 'sha256Checksum': hashlib.sha256(content).hexdigest(),
            'md5Checksum': hashlib.md5(content).hexdigest(), '_content': content,
            'capabilities': {'canDelete': True, 'canDeleteChildren': True, 'canAddChildren': True},
        }
        return identifier

    def metadata(self, identifier):
        if identifier not in self.items:
            raise FakeHttpError(404)
        return copy.deepcopy({k: v for k, v in self.items[identifier].items() if k != '_content'})

    def list(self, **arguments):
        def action():
            if self.fail_list:
                raise FakeHttpError(self.fail_list)
            query = arguments['q']
            parent = re.search(r"'([^']+)' in parents", query)[1]
            name = re.search(r"name='([^']+)'", query)
            mime = re.search(r"mimeType='([^']+)'", query)
            result = []
            for identifier, item in sorted(self.items.items()):
                if parent not in item['parents']:
                    continue
                if name and item['name'] != name[1]:
                    continue
                if mime and item['mimeType'] != mime[1]:
                    continue
                if 'trashed=false' in query and item['trashed']:
                    continue
                if 'trashed=true' in query and not item['trashed']:
                    continue
                result.append(self.metadata(identifier))
            start = int(arguments.get('pageToken') or 0)
            size = min(arguments.get('pageSize', 1000), self.page_size)
            response = {'files': result[start:start+size]}
            if start+size < len(result):
                response['nextPageToken'] = str(start+size)
            return response
        return FakeRequest(self, 'list', arguments, action)

    def get(self, **arguments):
        def action():
            if self.on_get:
                self.on_get(arguments['fileId'])
            return self.metadata(arguments['fileId'])
        return FakeRequest(self, 'get', arguments, action)

    def get_media(self, **arguments):
        def action():
            identifier = arguments['fileId']
            if identifier not in self.items:
                raise FakeHttpError(404)
            return self.items[identifier]['_content']
        return FakeRequest(self, 'get_media', arguments, action)

    def delete(self, **arguments):
        def action():
            identifier = arguments['fileId']
            self.delete_attempts.append(identifier)
            failures = self.delete_failures.get(identifier, [])
            if failures:
                raise FakeHttpError(failures.pop(0))
            if identifier not in self.items:
                raise FakeHttpError(404)
            self.items.pop(identifier)
            self.deleted.append(identifier)
            if self.on_after_delete:
                self.on_after_delete(identifier)
            if identifier in self.delete_then_500:
                self.delete_then_500.remove(identifier)
                raise FakeHttpError(500)
            return {}
        return FakeRequest(self, 'delete', arguments, action)


@pytest.fixture
def drive_state(tmp_path):
    api = FakeDrive()
    root = tmp_path/DRIVE_EXPERIMENT_ROOT
    run_id, scenario = 'abc123def4567890', 'S3'
    run = root/'outputs/dataset300_inline'/run_id
    folder = run/scenario
    folder.mkdir(parents=True)
    remote, remote_run = 'root', None
    folders = {}
    for name in [DRIVE_EXPERIMENT_ROOT, 'outputs', 'dataset300_inline', run_id, scenario]:
        remote = api.add(name, remote, folder=True)
        folders[name] = remote
        if name == run_id:
            remote_run = remote
    state = SimpleNamespace(api=api, root=root, run=run, folder=folder, run_id=run_id,
                            scenario=scenario, remote_run=remote_run, remote_folder=remote,
                            folders=folders, cleaner=DriveCheckpointCleaner(api, root))
    return state


def local_json(state, path, remote_folder, value):
    raw = (json.dumps(value, sort_keys=True)+'\n').encode()
    path.write_bytes(raw)
    return state.api.add(path.name, remote_folder, raw)


def active(state, stage='progress_fold_1'):
    protected = []
    identities = {}
    for epoch in [4, 3]:
        name = f'{stage}_epoch_{epoch:03d}.pt'
        raw = f'TECHNICAL CHECKPOINT {epoch}'.encode()
        (state.folder/name).write_bytes(raw)
        identities[epoch] = state.api.add(name, state.remote_folder, raw)
        protected.append({'file': name, 'size': len(raw),
                          'sha256': hashlib.sha256(raw).hexdigest(), 'md5': hashlib.md5(raw).hexdigest()})
    marker = state.folder/(stage+'.json')
    marker_id = local_json(state, marker, state.remote_folder,
                           {'scope': DRIVE_CLEANUP_SCOPE, 'snapshots': protected})
    old = []
    for epoch in [1, 2]:
        old.append(state.api.add(f'{stage}_epoch_{epoch:03d}.pt', state.remote_folder,
                                 b'OLD TECHNICAL CHECKPOINT', trashed=epoch == 1))
    return SimpleNamespace(stage=stage, marker=marker, marker_id=marker_id,
                           protected=protected, identities=identities, old=old)


def request(state, stage='progress_fold_1', mode='active'):
    return {'run_id': state.run_id, 'scenario': state.scenario, 'stage': stage, 'mode': mode}


def split(state):
    value = {'scope': DRIVE_CLEANUP_SCOPE, 'source_sha256': 'a'*64, 'split_sha256': 'b'*64,
             'development': [0, 1, 2, 3], 'test': [4, 5],
             'folds': [{'fold': 1, 'train': [0, 1], 'validation': [2, 3]}]}
    local_json(state, state.run/'split_manifest.json', state.remote_run, value)
    return value


def completed_cv(state):
    division = split(state)
    result = {'fold': 1, 'split_sha256': division['split_sha256'],
              'training': {'best_epoch': 2, 'scenario': state.scenario},
              'metrics': {'RMSE': 3.5}, 'validation_indices': [2, 3], 'prediction': [80., 90.]}
    marker = state.folder/'cv_fold_1.json'
    identifier = local_json(state, marker, state.remote_folder, result)
    return marker, identifier


def completed_refit(state):
    division = split(state)
    selection = {'scope': DRIVE_CLEANUP_SCOPE, 'split_sha256': division['split_sha256'],
                 'selected_epochs': {'S1': 2, 'S2': 3, 'S3': 4}}
    local_json(state, state.run/'selection_locked.json', state.remote_run, selection)
    raw = b'TECHNICAL FINAL MODEL ONLY'
    (state.folder/'final.pt').write_bytes(raw)
    final_id = state.api.add('final.pt', state.remote_folder, raw)
    value = {'selection': selection, 'source': division['source_sha256'],
             'sha256': hashlib.sha256(raw).hexdigest()}
    marker = state.folder/'refit_complete.json'
    identifier = local_json(state, marker, state.remote_folder, value)
    return marker, identifier, final_id


def test_only_older_stage_files_are_permanently_deleted(drive_state):
    s = drive_state
    a = active(s)
    extras = [s.api.add(name, s.remote_folder, b'MUST STAY', trashed=True) for name in [
        'progress_fold_1_epoch_005.pt', 'progress_fold_1_epoch_001.pt.pending',
        'progress_fold_2_epoch_001.pt', 'progress_refit_epoch_001.pt',
        'final.pt', 'cv_fold_1.json', '04_Prediksi_dan_Evaluasi.xlsx', 'dataset.xlsx']]
    # The same checkpoint filename in another run/folder cannot be selected.
    sibling = s.api.add('S1', s.remote_run, folder=True)
    other = s.api.add('progress_fold_1_epoch_001.pt', sibling, b'OTHER', trashed=True)
    result = s.cleaner(request(s))
    assert result['deleted'] == 2 and result['retained'] == 2
    assert set(s.api.deleted) == set(a.old)
    assert all(identifier in s.api.items for identifier in [*extras, other, *a.identities.values(), a.marker_id])
    assert all(retries == 5 for _, _, retries in s.api.requests)
    assert not any('emptyTrash' in operation for operation, _, _ in s.api.requests)


@pytest.mark.parametrize('fault', ['missing_local', 'size_local', 'missing_remote', 'empty_remote',
                                   'checksum_remote', 'trashed_remote', 'duplicate_remote', 'server_marker'])
def test_incomplete_protected_generation_prevents_all_deletion(drive_state, fault):
    s = drive_state
    a = active(s)
    latest = a.identities[4]
    path = s.folder/a.protected[0]['file']
    if fault == 'missing_local':path.unlink()
    elif fault == 'size_local':path.write_bytes(b'PARTIAL')
    elif fault == 'missing_remote':s.api.items.pop(latest)
    elif fault == 'empty_remote':s.api.items[latest]['size'] = '0'
    elif fault == 'checksum_remote':s.api.items[latest]['sha256Checksum'] = '0'*64
    elif fault == 'trashed_remote':s.api.items[latest]['trashed'] = True
    elif fault == 'duplicate_remote':s.api.add(path.name, s.remote_folder, path.read_bytes())
    elif fault == 'server_marker':s.api.items[a.marker_id]['_content'] = b'UNSYNCHRONIZED'
    with pytest.raises(CleanupDeferred):
        s.cleaner(request(s))
    assert s.api.deleted == []
    assert all(x in s.api.items for x in a.old)


def test_md5_fallback_when_drive_does_not_supply_sha256(drive_state):
    s = drive_state
    a = active(s)
    for identifier in a.identities.values():
        s.api.items[identifier].pop('sha256Checksum')
    assert s.cleaner(request(s))['deleted'] == 2


def test_changed_manifest_before_next_delete_stops_cleanup(drive_state):
    s = drive_state
    a = active(s)
    def change_marker(identifier):
        a.marker.write_text('{"scope":"CHANGED"}')
    s.api.on_after_delete = change_marker
    with pytest.raises(CleanupDeferred, match='Metadata berubah'):
        s.cleaner(request(s))
    assert len(s.api.deleted) == 1
    assert all(identifier in s.api.items for identifier in a.identities.values())


def test_completed_cv_releases_only_that_fold_including_pending(drive_state):
    s = drive_state
    a = active(s)
    completed_cv(s)
    pending = s.api.add('progress_fold_1_epoch_005.pt.pending', s.remote_folder, b'PARTIAL', trashed=True)
    sibling = s.api.add('progress_fold_2_epoch_001.pt', s.remote_folder, b'OTHER FOLD', trashed=True)
    final = s.api.add('final.pt', s.remote_folder, b'FINAL MUST STAY')
    result = s.cleaner(request(s, mode='complete'))
    assert result['deleted'] == 5
    assert set(s.api.deleted) == {*a.old, *a.identities.values(), pending}
    assert sibling in s.api.items and final in s.api.items
    assert s.api.items[a.marker_id]['name'] == 'progress_fold_1.json'


@pytest.mark.parametrize('fault', ['fold', 'split', 'validation', 'prediction', 'training', 'remote_marker'])
def test_completed_cv_requires_matching_complete_result(drive_state, fault):
    s = drive_state
    active(s)
    marker, marker_id = completed_cv(s)
    result = json.loads(marker.read_text())
    if fault == 'fold':result['fold'] = 2
    elif fault == 'split':result['split_sha256'] = 'c'*64
    elif fault == 'validation':result['validation_indices'] = [1, 2]
    elif fault == 'prediction':result['prediction'] = []
    elif fault == 'training':result['training'] = {}
    elif fault == 'remote_marker':s.api.items[marker_id]['_content'] = b'OLD RESULT'
    if fault != 'remote_marker':marker.write_text(json.dumps(result))
    with pytest.raises(CleanupDeferred):
        s.cleaner(request(s, mode='complete'))
    assert s.api.deleted == []


def test_completed_refit_requires_final_checksum_and_keeps_final(drive_state):
    s = drive_state
    a = active(s, 'progress_refit')
    marker, marker_id, final_id = completed_refit(s)
    result = s.cleaner(request(s, 'progress_refit', 'complete'))
    assert result['deleted'] == 4 and result['retained'] == 1
    assert set(s.api.deleted) == {*a.old, *a.identities.values()}
    assert final_id in s.api.items and marker_id in s.api.items


@pytest.mark.parametrize('fault', ['local_final', 'remote_final', 'source', 'selection', 'remote_marker'])
def test_refit_cleanup_preserves_progress_when_final_proof_is_bad(drive_state, fault):
    s = drive_state
    active(s, 'progress_refit')
    marker, marker_id, final_id = completed_refit(s)
    if fault == 'local_final':(s.folder/'final.pt').write_bytes(b'CORRUPT')
    elif fault == 'remote_final':s.api.items[final_id]['md5Checksum'] = '0'*32;s.api.items[final_id].pop('sha256Checksum')
    elif fault == 'remote_marker':s.api.items[marker_id]['_content'] = b'NOT FINAL'
    else:
        result = json.loads(marker.read_text())
        if fault == 'source':result['source'] = 'c'*64
        else:result['selection']['selected_epochs']['S3'] = 1
        marker.write_text(json.dumps(result))
    with pytest.raises(CleanupDeferred):
        s.cleaner(request(s, 'progress_refit', 'complete'))
    assert s.api.deleted == []


def test_delete_retry_handles_server_commit_then_500_then_404(drive_state):
    s = drive_state
    a = active(s)
    s.api.delete_then_500.add(a.old[0])
    s.api.delete_failures[a.old[1]] = [500, 503, 429]
    result = s.cleaner(request(s))
    assert result['deleted'] == 1 and result['already_absent'] == 1
    assert set(s.api.deleted) == set(a.old)
    assert s.api.delete_attempts.count(a.old[0]) == 2
    assert s.api.delete_attempts.count(a.old[1]) == 4


def test_candidate_already_deleted_between_listing_and_get_is_idempotent(drive_state):
    s = drive_state
    a = active(s)
    removed = set()
    def disappear(identifier):
        if identifier == a.old[0] and identifier not in removed:
            s.api.items.pop(identifier)
            removed.add(identifier)
    s.api.on_get = disappear
    result = s.cleaner(request(s))
    assert result['already_absent'] == 1 and result['deleted'] == 1


def test_preflight_resolves_only_experimental_root_and_checks_permission(drive_state):
    s = drive_state
    assert s.cleaner.preflight() == s.folders[DRIVE_EXPERIMENT_ROOT]
    s.api.items[s.folders[DRIVE_EXPERIMENT_ROOT]]['capabilities']['canDeleteChildren'] = False
    with pytest.raises(PermissionError):s.cleaner.preflight()
    assert s.api.deleted == []


def test_permission_error_is_not_disguised_as_success_or_retryable_indexlag(drive_state):
    s = drive_state
    s.api.fail_list = 403
    with pytest.raises(FakeHttpError) as error:s.cleaner.preflight()
    assert error.value.resp.status == 403


def test_duplicate_experimental_root_is_not_guessed(drive_state):
    s = drive_state
    s.api.add(DRIVE_EXPERIMENT_ROOT, 'root', folder=True)
    with pytest.raises(CleanupDeferred):s.cleaner.preflight()
    assert s.api.deleted == []


def test_semifinal_root_and_path_traversal_are_rejected(drive_state, tmp_path):
    s = drive_state
    with pytest.raises(ValueError):DriveCheckpointCleaner(s.api, tmp_path/'AES_300_MLP_Privat')
    for change in [{'run_id': '../other'}, {'scenario': '../S1'}, {'stage': 'progress_fold_6'},
                   {'stage': 'final'}, {'mode': 'emptyTrash'}]:
        with pytest.raises(ValueError):s.cleaner({**request(s), **change})
    assert s.api.requests == [] and s.api.deleted == []


def test_candidate_pagination_does_not_leave_older_stage_files(drive_state):
    s = drive_state
    a = active(s)
    s.api.page_size = 2
    assert s.cleaner(request(s))['deleted'] == 2
    assert set(s.api.deleted) == set(a.old)
    assert any(args.get('pageToken') for op, args, _ in s.api.requests if op == 'list')


def test_completed_stage_without_visible_checkpoint_skips_model_hashing(drive_state, monkeypatch):
    s = drive_state
    def must_not_read(*args):raise AssertionError('No model or metadata should be read for zero deletions')
    monkeypatch.setattr(s.cleaner, '_prepare', must_not_read)
    assert s.cleaner(request(s, 'progress_refit', 'complete'))['deleted'] == 0


def test_stale_candidate_permission_failure_does_not_delete_file(drive_state):
    s = drive_state
    a = active(s)
    s.api.items[a.old[0]]['capabilities']['canDelete'] = False
    with pytest.raises(PermissionError):s.cleaner(request(s))
    assert s.api.deleted == []


def test_only_two_protected_generations_and_unrelated_trash_means_zero_deletions(drive_state):
    s = drive_state
    a = active(s)
    for identifier in a.old:s.api.items.pop(identifier)
    unrelated = s.api.add('progress_fold_2_epoch_001.pt', s.remote_folder, b'OTHER', trashed=True)
    result = s.cleaner(request(s))
    assert result['deleted'] == 0 and result['retained'] == 2
    assert unrelated in s.api.items
    assert all(identifier in s.api.items for identifier in a.identities.values())


def test_server_manifest_changes_between_deletions_preserves_remaining_files(drive_state):
    s = drive_state
    a = active(s)
    s.api.on_after_delete = lambda _:s.api.items[a.marker_id].update(_content=b'NEW SERVER MANIFEST')
    with pytest.raises(CleanupDeferred, match='Metadata Drive berubah'):
        s.cleaner(request(s))
    assert len(s.api.deleted) == 1
    assert all(identifier in s.api.items for identifier in a.identities.values())


@pytest.mark.parametrize('fault', ['scope', 'path', 'epoch_zero', 'missing_hash'])
def test_invalid_snapshot_proof_stops_before_any_deletion(drive_state, fault):
    s = drive_state
    a = active(s)
    value = json.loads(a.marker.read_text())
    if fault == 'scope':value['scope'] = 'OTHER WORKFLOW'
    elif fault == 'path':value['snapshots'][0]['file'] = '../progress_fold_1_epoch_004.pt'
    elif fault == 'epoch_zero':value['snapshots'][0]['file'] = 'progress_fold_1_epoch_000.pt'
    else:value['snapshots'][0].pop('sha256')
    a.marker.write_text(json.dumps(value))
    with pytest.raises(CleanupDeferred):s.cleaner(request(s))
    assert s.api.deleted == []


def test_scenario_symlink_cannot_escape_experimental_storage(drive_state, tmp_path):
    s = drive_state
    s.folder.rmdir()
    target = tmp_path/'SEMIFINAL_KEEP';target.mkdir()
    s.folder.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match='keluar'):
        s.cleaner(request(s))
    assert s.api.requests == []


def test_zero_epoch_name_is_not_treated_as_known_checkpoint(drive_state):
    s = drive_state
    active(s)
    identifier = s.api.add('progress_fold_1_epoch_000.pt', s.remote_folder, b'UNKNOWN', trashed=True)
    assert s.cleaner(request(s))['deleted'] == 2
    assert identifier in s.api.items


def lagged_server_manifest(state):
    current = active(state)
    snapshots = []
    for identifier in reversed(current.old):
        item = state.api.items[identifier]
        item['trashed'] = False
        snapshots.append({'file': item['name'], 'size': int(item['size']),
                          'sha256': item['sha256Checksum'], 'md5': item['md5Checksum']})
    server = {'scope': DRIVE_CLEANUP_SCOPE, 'snapshots': snapshots}
    state.api.items[current.marker_id]['_content'] = json.dumps(server).encode()
    return current, server


def test_valid_old_server_manifest_keeps_both_resume_paths_until_sync(drive_state):
    s = drive_state
    a, server = lagged_server_manifest(s)
    result = s.cleaner(request(s))
    assert result['retained'] == 4 and result['deleted'] == 0
    assert all(identifier in s.api.items for identifier in [*a.old, *a.identities.values()])
    # Once the pointer catches up, release only obsolete generations.
    s.api.items[a.marker_id]['_content'] = a.marker.read_bytes()
    result = s.cleaner(request(s))
    assert result['retained'] == 2 and result['deleted'] == 2
    assert all(identifier in s.api.items for identifier in a.identities.values())


def test_folder_search_requests_only_folder_metadata(drive_state):
    s = drive_state
    active(s)
    s.cleaner(request(s))
    searches = [args for op, args, _ in s.api.requests
                if op == 'list' and 'mimeType=' in args['q']]
    assert searches
    assert all('Checksum' not in args['fields'] and 'capabilities' not in args['fields']
               for args in searches)


@pytest.mark.parametrize('fault', ['scope', 'stage', 'future', 'checksum', 'missing_file',
                                   'trashed_file', 'duplicate', 'same_name_other_content'])
def test_unverifiable_old_server_pointer_still_prevents_deletion(drive_state, fault):
    s = drive_state
    a, server = lagged_server_manifest(s)
    row = server['snapshots'][0]
    if fault == 'scope':server['scope'] = 'UNRELATED'
    elif fault == 'stage':row['file'] = 'progress_fold_2_epoch_002.pt'
    elif fault == 'future':row['file'] = 'progress_fold_1_epoch_005.pt'
    elif fault == 'checksum':row['sha256'] = '0'*64
    elif fault == 'missing_file':s.api.items.pop(a.old[1])
    elif fault == 'trashed_file':s.api.items[a.old[1]]['trashed'] = True
    elif fault == 'duplicate':server['snapshots'].append(dict(row))
    else:
        row['file'] = a.protected[0]['file']
    s.api.items[a.marker_id]['_content'] = json.dumps(server).encode()
    with pytest.raises(CleanupDeferred):s.cleaner(request(s))
    assert s.api.deleted == []


def test_server_pointer_advancing_during_cleanup_stops_before_deletion(drive_state):
    s = drive_state
    a, server = lagged_server_manifest(s)
    server['snapshots'] = server['snapshots'][:1]
    s.api.items[a.marker_id]['_content'] = json.dumps(server).encode()
    original = s.cleaner._remote_marker
    calls = []
    def advance(folder, name):
        calls.append(name)
        if len(calls) == 2:
            s.api.items[a.marker_id]['_content'] = a.marker.read_bytes()
        return original(folder, name)
    s.cleaner._remote_marker = advance
    with pytest.raises(CleanupDeferred, match='Metadata Drive berubah'):
        s.cleaner(request(s))
    assert s.api.deleted == [] and a.old[0] in s.api.items


def test_documented_kernel_repair_replaces_strict_cleaner_without_touching_worker(drive_state, monkeypatch):
    import io
    import urllib.request
    s = drive_state
    a, server = lagged_server_manifest(s)
    class LegacyStrictCleaner(DriveCheckpointCleaner):
        def _protect_server_snapshots(self, state, folder):
            self._marker(folder, *state['primary'])
    legacy = LegacyStrictCleaner(s.api, s.root)
    with pytest.raises(CleanupDeferred, match='Metadata Drive belum cocok'):
        legacy(request(s))
    root = Path(__file__).resolve().parents[1]
    module = (root/'src/aes/teacher_score/drive_checkpoint_cleanup.py').read_bytes()
    doc = (root/'docs/perbaikan_sinkronisasi_checkpoint_drive.md').read_text()
    code = doc.split('```python\n', 1)[1].split('```', 1)[0]
    requests = []
    def public_code_only(url, timeout):
        requests.append((url, timeout))
        return io.BytesIO(module)
    monkeypatch.setattr(urllib.request, 'urlopen', public_code_only)
    worker, configuration, fingerprint = object(), {'epochs': 25}, 'ORIGINAL_RUNTIME_HASH'
    namespace = {'_aes_cleaner': legacy, '_aes_process': worker, 'cfg': configuration,
                 'WORKER_SOURCE': 'UNCHANGED NUMERICAL WORKER', 'fingerprint': fingerprint}
    exec(compile(code, '<documented-repair-cell>', 'exec'), namespace)
    assert namespace['_aes_process'] is worker and namespace['cfg'] is configuration
    assert namespace['fingerprint'] == fingerprint
    assert namespace['WORKER_SOURCE'] == 'UNCHANGED NUMERICAL WORKER'
    cleaner = namespace['_aes_cleaner']
    assert cleaner.api is s.api and cleaner.root == s.root
    assert type(cleaner) is not type(legacy)
    result = cleaner(request(s))
    assert result['retained'] == 4 and result['deleted'] == 0
    assert all(identifier in s.api.items for identifier in [*a.old, *a.identities.values()])
    assert len(requests) == 1 and requests[0][0].endswith('/drive_checkpoint_cleanup.py')
