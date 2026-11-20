"""Generate the E11 prompt-attention ablation of E10."""

import hashlib
import json

from build_multitrait_reset_e6_notebook import BASE, GROUP, ROOT, replace
from build_multitrait_reset_e10_notebook import NAME as E10_NAME

NAME = 'AES_Reset_E11_S3_MultiTrait_PromptAttentionOnly.ipynb'
MARKER = '__RESET_E11_CODE_SHA256__'
INTRO = """# E11 reset · S3 multi-trait · Attention soal–jawaban

E11 menguji manfaat tambahan attention antartrait dengan menghapus jalur itu
dari E10. Setiap trait tetap mencari token jawaban yang relevan dengan soal.
Encoder, inisialisasi parameter bersama, split, seed, dan jadwal mengikuti
E10. Training memakai MSE dengan bobot loss relasi nol.

Bandingkan RMSE validasi pada lima fold yang sama dan cek hasil per trait.
Pilih T4 GPU, jalankan dari atas, lalu ambil ZIP ringkas 04–05 setelah test.
"""


def build():
    base = BASE.parent / E10_NAME
    raw = base.read_bytes()
    notebook = json.loads(raw)
    previous_hash = notebook['metadata']['aes_standalone']['code_sha256']
    for index, cell in enumerate(notebook['cells']):
        source = ''.join(cell['source']).replace(previous_hash, MARKER)
        source = (source.replace('Reset_E10', 'Reset_E11')
                  .replace('E10 reset', 'E11 reset')
                  .replace('reset_e10_s3_multitrait', 'reset_e11_s3_multitrait')
                  .replace('aes_reset_e10_no_relation_loss', 'aes_reset_e11_prompt_attention_only'))
        role = cell['metadata'].get('aes_multitrait_role')
        if role == 'definition:SingleScoreModel':
            source = replace(source, {
                '        self.trait_residual = nn.Linear(64, 64, bias=False)\n': '',
                '        nn.init.zeros_(self.trait_residual.weight)\n': '',
                """        other_logits = hidden.float() @ hidden.float().transpose(1, 2) / 8
        diagonal = torch.eye(4, dtype=torch.bool, device=hidden.device)
        other_weights = other_logits.masked_fill(diagonal, float('-inf')).softmax(dim=-1)
        updated = hidden + self.trait_residual(other_weights.to(hidden.dtype) @ hidden)
        return torch.cat([self.heads[name][3](updated[:, i]) for i, (name, *_) in enumerate(TRAITS)], dim=1)""":
                    '        return torch.cat([self.heads[name][3](hidden[:, i]) for i, (name, *_) in enumerate(TRAITS)], dim=1)',
            })
        elif role == 'prepare':
            source = replace(source, {"'fusion_variant': 'protact_adaptation'":
                                      "'fusion_variant': 'prompt_attention_only'"})
        elif role == 'architecture':
            source = replace(source, {"    ['Attention antartrait', '4 × 4'],\n": ''})
        cell['source'] = source
        cell['id'] = f'reset-e11-{index}'

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
        'base_sha256': hashlib.sha256(raw).hexdigest(), 'experiment': 'Reset_E11',
    }
    content = json.dumps(notebook, ensure_ascii=False, indent=1) + '\n'
    for folder in (BASE.parent, GROUP / 'notebooks'):
        (folder / NAME).write_text(content)


if __name__ == '__main__':
    build()
