"""Build a stage launcher that reuses the existing E17E training contract."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'eksperimen/03_tiga_input_multi_trait/notebooks'
BASE = FOLDER / 'AES_Reset_E17E_S3_Optuna_TPE_Native_Learnable_GPU.ipynb'
NAME = 'AES_Reset_E17E_Lanjut_Konfirmasi_dan_Test.ipynb'
INTRO = """# Lanjut E17E · Konfirmasi lima fold sampai evaluasi test

Buka runtime T4, hubungkan Drive yang menyimpan study pencarian E17E, lalu
Run all. Notebook memulihkan database, menguji lima kandidat terbaik dan
baseline pada lima fold, memeriksa stabilitas sesuai protokol E17E, mengunci
pilihan dari validasi, lalu refit dan menghitung test.

Study harus sudah mencapai 60 trial COMPLETE atau PRUNED. Ranking Excel
dipakai sebagai ringkasan; pemulihan memerlukan folder study asli di Drive.
Log menampilkan study yang dipilih dan jumlah trial sebelum mulai konfirmasi.
Jika snapshot study belum ditemukan, tahap ini berhenti sebelum training.
Versi runtime, dataset, split dan konfigurasi perlu cocok dengan study asal.

Hasil final berupa ZIP 04–05, berisi empat metrik per trait untuk OOF/test,
prediksi, konfigurasi terpilih, riwayat training, dan tabel pencarian.
Tombol unduhan muncul setelah laporan diperiksa. Hasil tersimpan di Drive.
Jika runtime putus, Run all memakai ulang trial, fold dan hasil test yang selesai.
Fit yang terputus dimulai ulang dari awal.

Notebook ini merupakan peluncur tahap dari E17E. Identitas kode training dan
kontrak study berasal dari notebook pencarian asli. Metadata peluncur mencatat
checksum sumber dan seluruh cell kode peluncur.
"""


def build():
    raw = BASE.read_bytes()
    notebook = json.loads(raw)
    mode_changes = dispatch_changes = 0
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        if "STAGE = 'search'" in source:
            source = source.replace("STAGE = 'search'", "STAGE = 'confirm_final'")
            source = source.replace("('search', 'confirm', 'final')",
                                    "('search', 'confirm', 'final', 'confirm_final')")
            source = source.replace("# 'search', 'confirm', atau 'final'",
                                    "# 'confirm_final' menjalankan konfirmasi lalu final.")
            mode_changes += 1
        if source.startswith('# Satu mode per Run all.'):
            source = """# Peluncur konfirmasi dan final memakai kontrak training E17E asli.
if STAGE == 'search':
    run_search()
elif STAGE == 'confirm':
    run_confirmation()
elif STAGE == 'final':
    run_final()
elif STAGE == 'confirm_final':
    if not SNAPSHOT_POINTER.exists() or len(finished_trials()) < HPO_TRIALS:
        raise RuntimeError(
            f'Study belum siap: {STUDY_DIR}. '
            f'Ditemukan {len(finished_trials())}/{HPO_TRIALS} trial selesai. '
            'Gunakan Drive dan versi runtime study pencarian yang sama.'
        )
    print('Study konfirmasi:', STUDY_DIR, flush=True)
    print('Trial selesai:', len(finished_trials()), flush=True)
    run_confirmation()
    # run_final verifies selection lock and fold checksums before refit/test.
    run_final()
print('Mode selesai:', STAGE)
print('Folder study privat:', STUDY_DIR)
"""
            dispatch_changes += 1
        cell['source'] = source
        if cell['cell_type'] == 'code':
            compile(source, cell.get('id', 'e17e'), 'exec')
            cell['outputs'] = []
            cell['execution_count'] = None
    if (mode_changes, dispatch_changes) != (1, 1):
        raise ValueError('Struktur mode notebook E17E berubah.')
    notebook['cells'][0]['source'] = INTRO
    notebook['metadata']['colab']['name'] = NAME
    notebook['metadata']['aes_standalone']['experiment'] = 'Reset_E17E'
    notebook['metadata']['aes_standalone']['code_id'] = 'E17E_TPE_GPU_CONCAT_V1'
    notebook['metadata']['aes_stage_launcher'] = {
        'source': BASE.relative_to(ROOT).as_posix(),
        'source_sha256': hashlib.sha256(raw).hexdigest(),
        'mode': 'confirm_final',
        'launcher_code_sha256': hashlib.sha256('\n'.join(
            c['source'] for c in notebook['cells'] if c['cell_type'] == 'code'
        ).encode()).hexdigest(),
    }
    (FOLDER / NAME).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')


if __name__ == '__main__':
    build()
