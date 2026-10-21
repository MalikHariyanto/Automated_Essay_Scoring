"""Simpan progres eksperimen tanpa penghapusan checkpoint otomatis."""
import json
from pathlib import Path

import torch

from . import SCOPE
from .data import digest, dump


def save_training_progress(path, payload):
    """Simpan tiap epoch; penunjuk memuat dua terbaru, file lama tetap ada."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(path.read_text())['snapshots'] if path.exists() else []
    target = path.with_name(f'{path.stem}_epoch_{payload["epoch"]:03d}.pt')
    temporary = target.with_name(target.name + '.pending')
    torch.save(payload, temporary)
    checksum = digest(temporary)
    temporary.replace(target)
    snapshots = [{'file': target.name, 'sha256': checksum}]
    snapshots += [row for row in previous if row['file'] != target.name][:1]
    # Semua file epoch tetap disimpan; tidak ada unlink atau permintaan Drive API.
    dump(path, {'scope': SCOPE, 'snapshots': snapshots})
