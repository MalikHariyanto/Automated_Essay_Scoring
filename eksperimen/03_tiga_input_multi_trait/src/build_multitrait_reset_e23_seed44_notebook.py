"""Derive the E23 confirmation notebook with training seed 44."""

import hashlib
import json

from build_multitrait_reset_e23_notebook import BASE, EXPERIMENT, GROUP, NAME as SOURCE_NAME, ROOT, replace

NAME = 'AES_Reset_E23_Konfirmasi_Seed44.ipynb'
MARKER = '__RESET_E23_SEED44_CODE_SHA256__'
CONFIRMATION = 'Reset_E23_Konfirmasi_Seed44'


def build():
    base = BASE.parent / SOURCE_NAME
    original = base.read_bytes()
    notebook = json.loads(original)
    old_hash = notebook['metadata']['aes_standalone']['code_sha256']
    for index, cell in enumerate(notebook['cells']):
        source = ''.join(cell['source']).replace(old_hash, MARKER)
        role = cell['metadata'].get('aes_multitrait_role')
        if index == 0:
            source = replace(source, {
                '# E23 · Bobot loss isi skor 1 sebesar 2×': '# E23 · Konfirmasi bobot isi 2× dengan seed 44',
                'seed dasar 43,': 'seed dasar 44 (seed fit 45–49),',
                'Tahap ini berhenti di development.': 'Tahap ini berhenti di development. Bandingkan dengan E23 seed 43.',
            })
        elif role == 'hyperparameters':
            source = replace(source, {'TRAIN_SEED = 43': 'TRAIN_SEED = 44'})
        elif role == 'prepare':
            source = replace(source, {EXPERIMENT: CONFIRMATION})
        elif role == 'storage':
            source = replace(source, {'aes_reset_e23_s3_bobot_isi_2x': 'aes_reset_e23_seed44'})
        elif role == 'fold_helpers':
            source = replace(source, {'Hasil_AES_Reset_E23_Progres': 'Hasil_AES_Reset_E23_Seed44_Progres'})
        elif role == 'final_report':
            source = replace(source, {'Hasil_AES_Reset_E23_Validasi': 'Hasil_AES_Reset_E23_Seed44_Validasi'})
        cell['source'] = source
        cell['id'] = f'reset-e23-seed44-{index}'

    code = '\n'.join(c['source'] for c in notebook['cells'] if c['cell_type'] == 'code')
    fingerprint = hashlib.sha256(code.encode()).hexdigest()
    for cell in notebook['cells']:
        cell['source'] = cell['source'].replace(MARKER, fingerprint)
        if cell['cell_type'] == 'code':
            compile(cell['source'], cell['id'], 'exec')
    notebook['metadata']['colab']['name'] = NAME
    notebook['metadata']['aes_standalone'] = {
        'code_sha256': fingerprint, 'base': base.relative_to(ROOT).as_posix(),
        'base_sha256': hashlib.sha256(original).hexdigest(), 'experiment': CONFIRMATION,
    }
    content = json.dumps(notebook, ensure_ascii=False, indent=1) + '\n'
    for folder in (BASE.parent, GROUP / 'notebooks'):
        (folder / NAME).write_text(content, encoding='utf-8')


if __name__ == '__main__':
    build()
