"""Hapus checkpoint usang milik eksperimen double frozen melalui Drive API.

Tidak membaca kredensial, melatih model, atau mengosongkan Sampah global.
Client API dibuat oleh kernel Colab. Pemanggil baru boleh meminta cleanup
sesudah saver menghitung checksum file lengkap dan menerbitkan manifest.
"""
import hashlib
import json
import math
from pathlib import Path
import re


DRIVE_CLEANUP_SCOPE = 'teacher_single_score_dataset300'
DRIVE_EXPERIMENT_ROOT = 'AES_300_DoubleFrozen_MLP_Privat'
DRIVE_FOLDER_MIME = 'application/vnd.google-apps.folder'
DRIVE_FILE_FIELDS = (
    'id,name,size,trashed,parents,mimeType,sha256Checksum,md5Checksum,'
    'capabilities(canDelete)'
)


class CleanupDeferred(RuntimeError):
    """Bukti belum lengkap; tidak boleh melemahkan pemeriksaan untuk menghapus."""


class DriveCheckpointCleaner:
    """Pembersih dengan scope tetap; API menerima request, bukan path arbitrer.

    Request: run_id (16 hex), scenario, stage, mode active|complete.
    Hasil: deleted, bytes_deleted, already_absent, retained, scenario, stage, mode.
    Semua pembacaan API memakai retry bawaan client; 404 delete idempoten.
    """

    def __init__(self, api, storage_root):
        root = Path(storage_root).expanduser().absolute()
        if root.name != DRIVE_EXPERIMENT_ROOT:
            raise ValueError('Pembersihan hanya diizinkan pada folder eksperimen double frozen.')
        self.root = root.resolve()
        if self.root.name != DRIVE_EXPERIMENT_ROOT:
            raise ValueError('Folder eksperimen tidak boleh mengarah ke folder lain.')
        self.api = api
        parts = self.root.parts
        self.storage_parts = parts[parts.index('MyDrive')+1:] if 'MyDrive' in parts else (DRIVE_EXPERIMENT_ROOT,)

    def _inside(self, path):
        path = Path(path)
        try:
            path.resolve().relative_to(self.root)
        except ValueError as error:
            raise ValueError('Path pembersihan keluar dari folder eksperimen.') from error
        return path

    def _read(self, path):
        path = self._inside(path)
        try:
            if not path.is_file():
                raise FileNotFoundError(path.name)
            raw = path.read_bytes()
            data = json.loads(raw)
        except (OSError, ValueError, UnicodeDecodeError) as error:
            raise CleanupDeferred('Bukti penyimpanan belum dapat dibaca: '+path.name) from error
        if not isinstance(data, dict):
            raise CleanupDeferred('Bukti penyimpanan bukan objek yang sah: '+path.name)
        return raw, data

    def _stat(self, path):
        path = self._inside(path)
        try:
            if not path.is_file():
                raise FileNotFoundError(path.name)
            result = path.stat()
        except OSError as error:
            raise CleanupDeferred('File yang dilindungi belum tersedia: '+path.name) from error
        if result.st_size <= 0:
            raise CleanupDeferred('File yang dilindungi masih kosong: '+path.name)
        return result.st_size, result.st_mtime_ns, result.st_ino

    @staticmethod
    def _checksum(value, length):
        return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{'+str(length)+'}', value) is not None

    @staticmethod
    def _status(error):
        response = getattr(error, 'resp', None)
        return getattr(response, 'status', None)

    def _execute(self, request):
        try:
            return request.execute(num_retries=5)
        except Exception as error:
            if self._status(error) == 404:
                raise CleanupDeferred('File Drive belum terlihat atau sudah berubah; pembersihan ditunda.') from error
            raise

    def _list(self, query):
        files, page = [], None
        while True:
            response = self._execute(self.api.files().list(
                q=query, spaces='drive', pageSize=1000, pageToken=page,
                fields='nextPageToken,files('+DRIVE_FILE_FIELDS+')',
                supportsAllDrives=True, includeItemsFromAllDrives=True))
            files.extend(response.get('files', []))
            page = response.get('nextPageToken')
            if not page:
                return files

    @staticmethod
    def _parent(file, parent):
        if parent != 'root' and parent not in file.get('parents', []):
            raise CleanupDeferred('Lokasi file Drive tidak sesuai folder eksperimen.')

    def _folder(self, parent, name):
        files = self._list(
            f"'{parent}' in parents and name='{name}' and trashed=false "
            f"and mimeType='{DRIVE_FOLDER_MIME}'")
        if len(files) != 1:
            raise CleanupDeferred('Folder Drive belum ditemukan secara unik: '+name)
        file = files[0]
        self._parent(file, parent)
        if file.get('name') != name or file.get('trashed') or file.get('mimeType') != DRIVE_FOLDER_MIME:
            raise CleanupDeferred('Identitas folder Drive tidak cocok: '+name)
        return file['id']

    def _folders(self, run_id, scenario):
        parent = 'root'
        for name in (*self.storage_parts, 'outputs', 'dataset300_inline', run_id):
            parent = self._folder(parent, name)
        return parent, self._folder(parent, scenario)

    def preflight(self):
        """Pastikan folder eksperimen dan izin hapus sebelum pelatihan panjang."""
        folder = 'root'
        for name in self.storage_parts:
            folder = self._folder(folder, name)
        info = self._execute(self.api.files().get(
            fileId=folder, supportsAllDrives=True,
            fields='id,name,mimeType,trashed,capabilities(canDeleteChildren,canAddChildren)'))
        if (info.get('id') != folder or info.get('name') != DRIVE_EXPERIMENT_ROOT
                or info.get('mimeType') != DRIVE_FOLDER_MIME or info.get('trashed')):
            raise CleanupDeferred('Folder eksperimen Drive berubah saat pemeriksaan awal.')
        if info.get('capabilities', {}).get('canDeleteChildren') is False:
            raise PermissionError('Akun Drive tidak mempunyai izin membersihkan isi folder eksperimen.')
        return folder

    def _single(self, folder, name):
        files = self._list(f"'{folder}' in parents and name='{name}' and trashed=false")
        if len(files) != 1:
            raise CleanupDeferred('File Drive yang dilindungi belum ditemukan secara unik: '+name)
        file = files[0]
        self._parent(file, folder)
        if file.get('name') != name or file.get('trashed'):
            raise CleanupDeferred('File Drive yang dilindungi berubah: '+name)
        try:
            size = int(file.get('size', 0))
        except (TypeError, ValueError) as error:
            raise CleanupDeferred('Ukuran file Drive belum sah: '+name) from error
        if size <= 0:
            raise CleanupDeferred('File Drive yang dilindungi masih kosong: '+name)
        return file

    def _marker(self, folder, name, expected):
        file = self._single(folder, name)
        raw = self._execute(self.api.files().get_media(
            fileId=file['id'], supportsAllDrives=True))
        if isinstance(raw, str):
            raw = raw.encode()
        if not isinstance(raw, bytes) or hashlib.sha256(raw).digest() != hashlib.sha256(expected).digest():
            raise CleanupDeferred('Metadata Drive belum cocok dengan hasil lokal: '+name)

    def _binary(self, folder, proof):
        file = self._single(folder, proof['file'])
        if int(file['size']) != proof['size']:
            raise CleanupDeferred('Ukuran checkpoint Drive belum lengkap: '+proof['file'])
        sha = file.get('sha256Checksum')
        if sha:
            valid = sha == proof['sha256']
        else:
            valid = bool(file.get('md5Checksum')) and file['md5Checksum'] == proof['md5']
        if not valid:
            raise CleanupDeferred('Checksum checkpoint Drive belum cocok: '+proof['file'])

    def _split(self, run):
        path = run/'split_manifest.json'
        raw, split = self._read(path)
        if (split.get('scope') != DRIVE_CLEANUP_SCOPE
                or not self._checksum(split.get('split_sha256'), 64)
                or not self._checksum(split.get('source_sha256'), 64)
                or not isinstance(split.get('folds'), list)):
            raise CleanupDeferred('Manifest pembagian data belum sah.')
        return path, raw, split

    def _prepare(self, run, folder, scenario, stage, mode):
        state = {'markers': {}, 'stats': {}, 'protected': [], 'stage': stage, 'mode': mode}
        if mode == 'active':
            marker = folder/(stage+'.json')
            raw, data = self._read(marker)
            snapshots = data.get('snapshots')
            if (data.get('scope') != DRIVE_CLEANUP_SCOPE or not isinstance(snapshots, list)
                    or not 1 <= len(snapshots) <= 2):
                raise CleanupDeferred('Manifest checkpoint aktif belum sah.')
            names, epochs = set(), []
            for item in snapshots:
                if not isinstance(item, dict):
                    raise CleanupDeferred('Bukti snapshot belum sah.')
                name = item.get('file', '')
                match = re.fullmatch(re.escape(stage)+r'_epoch_(\d+)\.pt', name) if isinstance(name, str) else None
                size = item.get('size')
                if (not match or int(match[1]) < 1 or name in names
                        or not self._checksum(item.get('sha256'), 64)
                        or not self._checksum(item.get('md5'), 32)
                        or type(size) is not int or size <= 0):
                    raise CleanupDeferred('Nama, ukuran, atau checksum snapshot belum sah.')
                path = self._inside(folder/name)
                stats = self._stat(path)
                if stats[0] != size:
                    raise CleanupDeferred('Ukuran checkpoint lokal belum cocok: '+name)
                state['stats'][path] = stats
                state['protected'].append(dict(item))
                names.add(name)
                epochs.append(int(match[1]))
            state['oldest'] = min(epochs)
            state['markers'][marker] = raw
            state['primary'] = (marker.name, raw)
            return state

        split_path, split_raw, split = self._split(run)
        state['markers'][split_path] = split_raw
        state['split'] = (split_path.name, split_raw)
        if stage.startswith('progress_fold_'):
            number = int(stage.rsplit('_', 1)[1])
            marker = folder/f'cv_fold_{number}.json'
            raw, result = self._read(marker)
            matches = [f for f in split['folds'] if isinstance(f, dict) and f.get('fold') == number]
            if len(matches) != 1:
                raise CleanupDeferred('Fold pada manifest pembagian data tidak ditemukan secara unik.')
            valid = matches[0].get('validation')
            predictions = result.get('prediction')
            training = result.get('training')
            rmse = result.get('metrics', {}).get('RMSE') if isinstance(result.get('metrics'), dict) else None
            if (result.get('fold') != number or result.get('split_sha256') != split['split_sha256']
                    or not isinstance(valid, list) or not valid
                    or result.get('validation_indices') != valid
                    or not isinstance(predictions, list) or len(predictions) != len(valid)
                    or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in predictions)
                    or not isinstance(training, dict) or type(training.get('best_epoch')) is not int
                    or training['best_epoch'] < 1
                    or training.get('scenario', scenario) != scenario
                    or not isinstance(rmse, (int, float)) or not math.isfinite(rmse) or rmse < 0):
                raise CleanupDeferred('Hasil fold belum lengkap atau tidak cocok dengan split.')
        else:
            selection_path = run/'selection_locked.json'
            selection_raw, selection = self._read(selection_path)
            marker = folder/'refit_complete.json'
            raw, result = self._read(marker)
            if (selection.get('scope') != DRIVE_CLEANUP_SCOPE
                    or selection.get('split_sha256') != split['split_sha256']
                    or not isinstance(selection.get('selected_epochs'), dict)
                    or type(selection['selected_epochs'].get(scenario)) is not int
                    or selection['selected_epochs'][scenario] < 1
                    or result.get('selection') != selection
                    or result.get('source') != split['source_sha256']
                    or not self._checksum(result.get('sha256'), 64)):
                raise CleanupDeferred('Penanda model final belum cocok dengan sumber dan pilihan CV.')
            state['markers'][selection_path] = selection_raw
            final = folder/'final.pt'
            before = self._stat(final)
            sha, md5 = hashlib.sha256(), hashlib.md5()
            with final.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                    sha.update(chunk)
                    md5.update(chunk)
            if self._stat(final) != before or sha.hexdigest() != result['sha256']:
                raise CleanupDeferred('Model final belum lolos pemeriksaan checksum.')
            state['stats'][final] = before
            state['protected'].append({'file': 'final.pt', 'size': before[0],
                                       'sha256': sha.hexdigest(), 'md5': md5.hexdigest()})
        state['markers'][marker] = raw
        state['primary'] = (marker.name, raw)
        return state

    def _guard(self, state, run_id, folder_id):
        for path, expected in state['markers'].items():
            try:
                actual = self._inside(path).read_bytes()
            except OSError as error:
                raise CleanupDeferred('Bukti penyimpanan berubah; pembersihan dihentikan.') from error
            if actual != expected:
                raise CleanupDeferred('Metadata berubah saat pembersihan; file lain dipertahankan.')
        for path, expected in state['stats'].items():
            if self._stat(path) != expected:
                raise CleanupDeferred('File yang dilindungi berubah saat pembersihan.')
        name, raw = state['primary']
        self._marker(folder_id, name, raw)
        if 'split' in state:
            name, raw = state['split']
            self._marker(run_id, name, raw)
        for proof in state['protected']:
            self._binary(folder_id, proof)

    def __call__(self, request):
        if not isinstance(request, dict):
            raise ValueError('Permintaan cleanup harus berupa objek.')
        run_id, scenario, stage, mode = (request.get(k) for k in ('run_id', 'scenario', 'stage', 'mode'))
        if (not isinstance(run_id, str) or not re.fullmatch(r'[0-9a-f]{16}', run_id)
                or scenario not in ('S1', 'S2', 'S3')
                or not isinstance(stage, str)
                or not re.fullmatch(r'progress_fold_[1-5]|progress_refit', stage)
                or mode not in ('active', 'complete')):
            raise ValueError('Run, skenario, tahap, atau mode cleanup tidak diizinkan.')
        run = self._inside(self.root/'outputs/dataset300_inline'/run_id)
        folder = self._inside(run/scenario)
        remote_run, remote_folder = self._folders(run_id, scenario)
        pattern = re.compile(re.escape(stage)+r'_epoch_(\d+)\.pt'+(r'(?:\.pending)?' if mode == 'complete' else ''))
        files = self._list(f"'{remote_folder}' in parents")
        # Hasil lengkap sering dibaca ulang. Jika tidak ada checkpoint tahap ini
        # di Drive, jangan membaca ulang model final besar hanya untuk menghapus nol file.
        if mode == 'complete' and not any(pattern.fullmatch(f.get('name', '')) for f in files):
            return {'deleted': 0, 'bytes_deleted': 0, 'already_absent': 0,
                    'retained': 0, 'scenario': scenario, 'stage': stage, 'mode': mode}
        state = self._prepare(run, folder, scenario, stage, mode)
        self._guard(state, remote_run, remote_folder)
        protected = {x['file'] for x in state['protected']}
        candidates = []
        for file in files:
            self._parent(file, remote_folder)
            name = file.get('name', '')
            match = pattern.fullmatch(name)
            if not match or int(match[1]) < 1 or name in protected:
                continue
            if mode == 'active' and int(match[1]) >= state['oldest']:
                continue
            candidates.append(file)
        deleted, absent, size = 0, 0, 0
        for file in candidates:
            self._guard(state, remote_run, remote_folder)
            try:
                fresh = self._execute(self.api.files().get(
                    fileId=file['id'], fields=DRIVE_FILE_FIELDS, supportsAllDrives=True))
            except CleanupDeferred as error:
                if self._status(error.__cause__) == 404:
                    absent += 1
                    continue
                raise
            self._parent(fresh, remote_folder)
            if fresh.get('name') != file.get('name') or fresh.get('id') != file.get('id'):
                raise CleanupDeferred('Identitas kandidat berubah; pembersihan dihentikan.')
            if fresh.get('mimeType') == DRIVE_FOLDER_MIME:
                raise CleanupDeferred('Kandidat bukan file checkpoint biasa.')
            if fresh.get('capabilities', {}).get('canDelete') is False:
                raise PermissionError('Akun Drive tidak mempunyai izin menghapus checkpoint usang.')
            try:
                self.api.files().delete(fileId=file['id'], supportsAllDrives=True).execute(num_retries=5)
            except Exception as error:
                if self._status(error) == 404:
                    absent += 1
                    continue
                raise
            deleted += 1
            try:
                size += max(0, int(fresh.get('size', 0)))
            except (TypeError, ValueError):
                pass
        return {'deleted': deleted, 'bytes_deleted': size, 'already_absent': absent,
                'retained': len(state['protected']), 'scenario': scenario, 'stage': stage, 'mode': mode}
