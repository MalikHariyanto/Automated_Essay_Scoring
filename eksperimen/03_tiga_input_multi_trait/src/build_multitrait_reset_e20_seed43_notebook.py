"""Build the E20 confirmation notebook with a second training seed."""

import hashlib
import json

from build_multitrait_reset_e6_notebook import BASE, GROUP, ROOT, replace
from build_multitrait_reset_e20_notebook import NAME as E20_NAME

NAME = 'AES_Reset_E20_Konfirmasi_Seed43.ipynb'
MARKER = '__RESET_E20_SEED43_CODE_SHA256__'

INTRO = """# E20 · Konfirmasi aturan checkpoint A/B dengan seed 43

Tujuannya melihat apakah pola E20 pertama muncul lagi ketika seed training
berubah. Seed dasar sekarang **43**, sehingga seed kelima fit adalah 44–48.
Pembagian kelompok dan kelima fold tetap sama dengan E20 pertama.

- **A:** pilih epoch dengan RMSE gabungan paling rendah.
- **B:** pilih epoch dengan rata-rata QWK empat aspek tertinggi dari epoch
  dengan RMSE maksimal **0,01** di atas A. Jika seri, pilih RMSE lebih rendah,
  lalu epoch lebih awal.

Arsitektur, loss E19 (rasio bobot organisasi 1,2×), LR, ambang 1,50,
maksimal 25 epoch, dan patience 5 mengikuti E20. Satu training per fold
menghasilkan prediksi A dan B. Lihat hasil tiap fold langsung di bawah cell.

Pilih **T4 GPU**, hubungkan Drive, lalu jalankan dari atas. Fold yang selesai
tersimpan di folder run khusus seed 43. Run ini berakhir di validasi lima
fold. Kirim ZIP 04–05 hasilnya untuk dibandingkan dengan E20 pertama.
"""


def build():
    base = BASE.parent / E20_NAME
    raw = base.read_bytes()
    notebook = json.loads(raw)
    old_hash = notebook['metadata']['aes_standalone']['code_sha256']

    for index, cell in enumerate(notebook['cells']):
        source = ''.join(cell['source']).replace(old_hash, MARKER)
        role = cell['metadata'].get('aes_multitrait_role')
        if index == 0:
            source = INTRO
        elif role == 'hyperparameters':
            source = replace(source, {
                'TRAIN_SEED = 42': 'TRAIN_SEED = 43',
            })
        elif role == 'storage':
            source = replace(source, {
                "OUTPUT_NAME = 'aes_reset_e20_e19_checkpoint_ab'":
                    "OUTPUT_NAME = 'aes_reset_e20_checkpoint_ab_seed43'",
            })
        elif role == 'prepare':
            source = replace(source, {
                'Reset_E20_E19_Pilih_Checkpoint_AB': 'Reset_E20_Konfirmasi_Seed43',
            })
        elif role == 'fold_helpers':
            source = replace(source, {
                'Hasil_AES_Reset_E20_Progres_04_05.zip':
                    'Hasil_AES_Reset_E20_Seed43_Progres_04_05.zip',
            })
        elif role == 'validation_report':
            source = replace(source, {
                "'organization_loss_ratio': cfg['organization_loss_ratio'],":
                    "'organization_loss_ratio': cfg['organization_loss_ratio'], 'training_seed': cfg['seed'],",
            })
        elif role == 'final_report':
            source = replace(source, {
                'Hasil_AES_Reset_E20_Validasi_Checkpoint_AB_04_05.zip':
                    'Hasil_AES_Reset_E20_Seed43_Validasi_Checkpoint_AB_04_05.zip',
            })
        cell['source'] = source
        cell['id'] = f'reset-e20-seed43-{index}'

    code = '\n'.join(cell['source'] for cell in notebook['cells'] if cell['cell_type'] == 'code')
    fingerprint = hashlib.sha256(code.encode()).hexdigest()
    for cell in notebook['cells']:
        cell['source'] = cell['source'].replace(MARKER, fingerprint)
        if cell['cell_type'] == 'code':
            compile(cell['source'], cell['id'], 'exec')
    notebook['metadata']['colab']['name'] = NAME
    notebook['metadata']['aes_standalone'] = {
        'code_sha256': fingerprint, 'base': base.relative_to(ROOT).as_posix(),
        'base_sha256': hashlib.sha256(raw).hexdigest(), 'experiment': 'Reset_E20_Konfirmasi_Seed43',
    }
    content = json.dumps(notebook, ensure_ascii=False, indent=1) + '\n'
    for folder in (BASE.parent, GROUP / 'notebooks'):
        (folder / NAME).write_text(content, encoding='utf-8')


if __name__ == '__main__':
    build()
