"""Split an uploaded Excel file into verified pieces suitable for attachment upload."""

import hashlib
import json
from pathlib import Path


def split_excel_file(source, destination, part_size=30 * 1024 * 1024):
    """Return a manifest after checking that concatenated pieces match the source."""
    source = Path(source)
    destination = Path(destination)
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError("Pilih file Excel yang ada dan berisi data.")
    if not 1 <= part_size <= 30 * 1024 * 1024:
        raise ValueError("Ukuran bagian harus antara 1 byte dan 30 MiB.")
    destination.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256()
    parts = []

    with source.open("rb") as stream:
        number = 1
        while chunk := stream.read(part_size):
            name = f"{source.name}.part{number:03d}"
            path = destination / name
            with path.open("xb") as target:
                target.write(chunk)
            source_hash.update(chunk)
            parts.append({
                "name": name,
                "size_bytes": len(chunk),
                "sha256": hashlib.sha256(chunk).hexdigest(),
            })
            number += 1

    reconstructed_hash = hashlib.sha256()
    total_bytes = 0
    for part in parts:
        part_hash = hashlib.sha256()
        size = 0
        with (destination / part["name"]).open("rb") as stream:
            while block := stream.read(1024 * 1024):
                reconstructed_hash.update(block)
                part_hash.update(block)
                size += len(block)
        if size != part["size_bytes"] or part_hash.hexdigest() != part["sha256"]:
            raise IOError("Verifikasi bagian gagal: " + part["name"])
        total_bytes += size

    if total_bytes != source.stat().st_size or reconstructed_hash.hexdigest() != source_hash.hexdigest():
        raise IOError("Gabungan pecahan berbeda dari file asli.")

    manifest = {
        "original_name": source.name,
        "size_bytes": total_bytes,
        "sha256": source_hash.hexdigest(),
        "part_size_bytes": part_size,
        "parts": parts,
    }
    manifest_path = destination / f"{source.name}.manifest.json"
    with manifest_path.open("x", encoding="utf-8") as target:
        json.dump(manifest, target, ensure_ascii=False, indent=2)
    return manifest, manifest_path
