"""Checkpoint eksperimen terpisah; Drive dibersihkan melalui broker kernel Colab."""
import hashlib
import json
import os
from pathlib import Path
import re
import time

import torch

from . import SCOPE
from .data import dump


def checkpoint_file_proof(path):
    """Baca sekali untuk SHA-256, MD5 transport Drive, dan ukuran file."""
    path = Path(path)
    before = path.stat()
    sha = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    size = 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(chunk)
            md5.update(chunk)
            size += len(chunk)
    after = path.stat()
    if before.st_size != size or after.st_size != size or before.st_mtime_ns != after.st_mtime_ns:
        raise RuntimeError('Checkpoint berubah ketika diperiksa; file lama dipertahankan.')
    return {'sha256': sha.hexdigest(), 'md5': md5.hexdigest(), 'size': size}


def drive_cleanup_enabled():
    return os.environ.get('AES300_DRIVE_CLEANUP') == '1'


def request_checkpoint_cleanup(path, mode):
    """Hanya tahap/run saat ini; gagal bersih menghentikan epoch berikutnya dengan progres aman."""
    path = Path(path)
    if not re.fullmatch(r'progress_(?:fold_[1-5]|refit)', path.stem):
        raise ValueError('Nama tahap checkpoint tidak sah.')
    run = path.parent.parent
    request = {'run_id': run.name, 'scenario': path.parent.name,
               'stage': path.stem, 'mode': mode}
    queue_path = run / '_checkpoint_cleanup_pending.json'
    queue = json.loads(queue_path.read_text()) if queue_path.exists() else {'scope': SCOPE, 'requests': {}}
    if queue.get('scope') != SCOPE:
        raise RuntimeError('Antrean pembersihan bukan milik eksperimen AES ini.')
    key = request['scenario'] + '/' + request['stage']
    queue['requests'][key] = request
    dump(queue_path, queue)
    broker = globals().get('_drive_cleanup')
    if broker is None:
        raise RuntimeError('Broker pembersihan belum siap. Jalankan setup notebook dari atas; checkpoint dipertahankan.')
    last_error = ''
    for attempt in range(5):
        reply = broker(request)
        if reply.get('ok'):
            queue['requests'].pop(key, None)
            dump(queue_path, queue)
            result = reply.get('result', {})
            count = result.get('deleted', 0)
            if count:
                print(f"Penyimpanan {request['scenario']}: {count} checkpoint usang dihapus permanen "
                      f"({result.get('bytes_deleted', 0) / 1024**3:.2f} GiB).", flush=True)
            return result
        last_error = reply.get('error', 'Pembersihan Drive belum terverifikasi.')
        if not reply.get('retryable', False):
            break
        if attempt < 4:
            wait = 2 ** (attempt + 1)
            print(f'Drive belum siap diverifikasi; mencoba pembersihan lagi dalam {wait} detik.', flush=True)
            time.sleep(wait)
    raise RuntimeError(
        'Checkpoint/hasil tahap tetap disimpan. Training dihentikan sebelum tahap berikutnya '
        'agar checkpoint tidak menumpuk. Jalankan notebook yang sama dari atas untuk '
        'mencoba pembersihan dan melanjutkan progres. Rincian: ' + last_error)


def retry_pending_checkpoint_cleanup(output):
    queue_path = Path(output) / '_checkpoint_cleanup_pending.json'
    if not drive_cleanup_enabled() or not queue_path.exists():
        return
    queue = json.loads(queue_path.read_text())
    if queue.get('scope') != SCOPE:
        raise RuntimeError('Antrean pembersihan tidak cocok eksperimen ini.')
    for request in list(queue.get('requests', {}).values()):
        path = Path(output) / request['scenario'] / (request['stage'] + '.json')
        request_checkpoint_cleanup(path, request['mode'])


def sweep_completed_checkpoint_cleanup(output):
    """Retry pembersihan sesudah hasil lengkap dipakai ulang, tanpa training lagi."""
    if not drive_cleanup_enabled():
        return
    for scenario in ('S1', 'S2', 'S3'):
        folder = Path(output) / scenario
        for fold in range(1, 6):
            if (folder / f'cv_fold_{fold}.json').is_file():
                request_checkpoint_cleanup(folder / f'progress_fold_{fold}.json', 'complete')
        if (folder / 'refit_complete.json').is_file():
            request_checkpoint_cleanup(folder / 'progress_refit.json', 'complete')


def save_training_progress(path, payload):
    """Simpan dua generasi; bersihkan versi lama hanya sesudah bukti pengganti sah."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(path.read_text())['snapshots'] if path.exists() else []
    target = path.with_name(f'{path.stem}_epoch_{payload["epoch"]:03d}.pt')
    temporary = target.with_name(target.name + '.pending')
    torch.save(payload, temporary)
    proof = checkpoint_file_proof(temporary)
    temporary.replace(target)
    snapshots = [{'file': target.name, **proof}] + [r for r in previous if r['file'] != target.name][:1]
    dump(path, {'scope': SCOPE, 'snapshots': snapshots})
    if drive_cleanup_enabled():
        # Jangan unlink .pt di Drive: itu dapat memenuhi Sampah dengan file besar.
        request_checkpoint_cleanup(path, 'active')
    else:
        keep = {r['file'] for r in snapshots}
        for old in path.parent.glob(path.stem + '_epoch_*.pt'):
            if old.name not in keep:
                old.unlink()


def clear_training_progress(path):
    path = Path(path)
    if drive_cleanup_enabled():
        # Broker memeriksa hasil CV/model final sebelum menghapus file tahap selesai.
        request_checkpoint_cleanup(path, 'complete')
    else:
        for file in path.parent.glob(path.stem + '_epoch_*.pt*'):
            file.unlink()
    if path.exists():
        path.unlink()
