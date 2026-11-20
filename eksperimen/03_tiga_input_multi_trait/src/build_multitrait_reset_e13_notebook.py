"""Generate E13, an E10 ablation of prompt attention for organization."""

import hashlib
import json

from build_multitrait_reset_e6_notebook import BASE, GROUP, ROOT, replace
from build_multitrait_reset_e10_notebook import NAME as E10_NAME

NAME = 'AES_Reset_E13_S3_MultiTrait_OrganisasiPromptAblation.ipynb'
MARKER = '__RESET_E13_CODE_SHA256__'
INTRO = """# E13 reset · S3 multi-trait · Ablasi attention soal–jawaban untuk organisasi

Head organisasi melewati penambahan langsung sinyal attention soal–jawaban.
Attention antartrait tetap aktif, sehingga organisasi masih bisa menerima
informasi dari tiga trait lain. Soal, kunci, dan jawaban tetap masuk ke dua
encoder. Inisialisasi, seed, lima fold, dan resep MSE mengikuti E10.

[ProTACT](https://aclanthology.org/2023.findings-acl.98/) menjadi rujukan
arsitektur dasar. Ablasi khusus organisasi berasal dari hasil E10–E12.
Bandingkan RMSE organisasi, terutama fold 4, serta RMSE keseluruhan,
kebahasaan, dan mekanik di validasi.

Pilih T4 GPU, jalankan dari atas dengan workbook semi-final, lalu ambil
ZIP ringkas 04–05 setelah evaluasi test.
"""


def build():
    base = BASE.parent / E10_NAME
    raw = base.read_bytes()
    notebook = json.loads(raw)
    previous_hash = notebook['metadata']['aes_standalone']['code_sha256']
    for index, cell in enumerate(notebook['cells']):
        source = ''.join(cell['source']).replace(previous_hash, MARKER)
        source = (source.replace('Reset_E10', 'Reset_E13')
                  .replace('E10 reset', 'E13 reset')
                  .replace('reset_e10_s3_multitrait', 'reset_e13_s3_multitrait')
                  .replace('aes_reset_e10_no_relation_loss', 'aes_reset_e13_organisasi_prompt_ablation'))
        role = cell['metadata'].get('aes_multitrait_role')
        if role == 'definition:SingleScoreModel':
            source = replace(source, {
                """        hidden = torch.stack([
            self.heads[name][2](self.heads[name][1](self.heads[name][0](fusion) + signal[:, i]))
            for i, (name, *_) in enumerate(TRAITS)
        ], dim=1)""": """        states = []
        for i, (name, *_) in enumerate(TRAITS):
            linear, activation, dropout, _ = self.heads[name]
            state = linear(fusion)
            if name != 'organisasi':
                state = state + signal[:, i]
            states.append(dropout(activation(state)))
        hidden = torch.stack(states, dim=1)""",
            })
        elif role == 'prepare':
            source = replace(source, {"'fusion_variant': 'protact_adaptation'":
                                      "'fusion_variant': 'organisasi_prompt_ablation'"})
        elif role == 'architecture':
            source = replace(source, {"    ['Attention soal–jawaban', '4 × 64'],":
                                      "    ['Sinyal attention soal–jawaban', 'Isi, kebahasaan, mekanik'],"})
        cell['source'] = source
        cell['id'] = f'reset-e13-{index}'

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
        'base_sha256': hashlib.sha256(raw).hexdigest(), 'experiment': 'Reset_E13',
    }
    content = json.dumps(notebook, ensure_ascii=False, indent=1) + '\n'
    for folder in (BASE.parent, GROUP / 'notebooks'):
        (folder / NAME).write_text(content)


if __name__ == '__main__':
    build()
