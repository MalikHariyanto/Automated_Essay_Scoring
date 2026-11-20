"""Generate E10, the zero-weight relation-loss ablation of E9."""

import hashlib
import json

from build_multitrait_reset_e6_notebook import BASE, GROUP, ROOT, replace
from build_multitrait_reset_e9_notebook import NAME as E9_NAME

NAME = 'AES_Reset_E10_S3_MultiTrait_NoRelationLoss.ipynb'
MARKER = '__RESET_E10_CODE_SHA256__'
INTRO = """# E10 reset · S3 multi-trait · Ablasi loss relasi E9

E10 menguji bobot loss relasi 0 dibandingkan bobot 0,1 di E9.
Attention soal–jawaban, attention antartrait, encoder, split, seed, optimizer,
dan jadwal training mengikuti E9. MSE menentukan gradien. Riwayat masih
mencatat kandidat pasangan dan nilai loss relasi mentah untuk diagnosis.

Bandingkan RMSE validasi pada lima fold yang sama, khususnya kebahasaan,
mekanik, dan organisasi. Pilih T4 GPU dan jalankan dari atas dengan workbook
semi-final. Ambil ZIP ringkas 04–05 setelah evaluasi test.
"""


def build():
    base = BASE.parent / E9_NAME
    raw = base.read_bytes()
    notebook = json.loads(raw)
    previous_hash = notebook['metadata']['aes_standalone']['code_sha256']
    for index, cell in enumerate(notebook['cells']):
        source = ''.join(cell['source']).replace(previous_hash, MARKER)
        source = (source.replace('Reset_E9', 'Reset_E10')
                  .replace('E9 reset', 'E10 reset')
                  .replace('reset_e9_s3_multitrait', 'reset_e10_s3_multitrait')
                  .replace('aes_reset_e9_protact', 'aes_reset_e10_no_relation_loss'))
        if cell['metadata'].get('aes_multitrait_role') == 'prepare':
            source = replace(source, {"'trait_relation_weight': 0.1": "'trait_relation_weight': 0.0"})
        if cell['cell_type'] == 'markdown':
            source = source.replace('dan loss relasi antartrait', 'serta pencatatan loss relasi antartrait')
        cell['source'] = source
        cell['id'] = f'reset-e10-{index}'
    notebook['cells'][0]['source'] = INTRO
    code = '\n'.join(cell['source'] for cell in notebook['cells'] if cell['cell_type'] == 'code')
    fingerprint = hashlib.sha256(code.encode()).hexdigest()
    for cell in notebook['cells']:
        cell['source'] = cell['source'].replace(MARKER, fingerprint)
        if cell['cell_type'] == 'code':
            compile(cell['source'], cell['id'], 'exec')
    notebook['metadata']['colab']['name'] = NAME
    notebook['metadata']['aes_standalone'] = {
        'code_sha256': fingerprint, 'base': base.relative_to(ROOT).as_posix(),
        'base_sha256': hashlib.sha256(raw).hexdigest(), 'experiment': 'Reset_E10',
    }
    content = json.dumps(notebook, ensure_ascii=False, indent=1) + '\n'
    for folder in (BASE.parent, GROUP / 'notebooks'):
        (folder / NAME).write_text(content)


if __name__ == '__main__':
    build()
