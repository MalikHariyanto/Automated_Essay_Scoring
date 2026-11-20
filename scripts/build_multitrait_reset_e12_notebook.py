"""Generate E12, the inter-trait-only attention ablation of E10."""

import hashlib
import json

from build_multitrait_reset_e6_notebook import BASE, GROUP, ROOT, replace
from build_multitrait_reset_e10_notebook import NAME as E10_NAME

NAME = 'AES_Reset_E12_S3_MultiTrait_InterTraitAttentionOnly.ipynb'
MARKER = '__RESET_E12_CODE_SHA256__'
INTRO = """# E12 reset · S3 multi-trait · Attention antartrait

E12 menguji attention antartrait E10 tanpa jalur attention soal–jawaban.
Soal, kunci, dan jawaban tetap dibaca oleh kedua encoder melalui input E2.
Empat head menerima concat E2, lalu masing-masing membaca tiga trait lain.
Encoder, parameter bersama, split, seed, dan jadwal mengikuti E10.
Training memakai MSE dengan bobot loss relasi nol.

Pilih T4 GPU, jalankan notebook dari atas dengan workbook semi-final,
lalu ambil ZIP ringkas 04–05 setelah evaluasi test.
"""


def intertrait_model(source):
    start = source.index('    def forward(self, fusion, tokens, question_mask')
    end = source.index('        hidden = torch.stack([', start)
    source = source[:start] + '    def forward(self, fusion):\n' + source[end:]
    start = source.index('\n\ndef prompt_answer_masks(')
    end = source.index('\n\ndef correlated_trait_pairs(', start)
    source = source[:start] + source[end:]
    return replace(source, {
        '        self.question_query = nn.Linear(768, 4 * 64, bias=False)\n': '',
        '        self.answer_key = nn.Linear(768, 64, bias=False)\n': '',
        '        self.prompt_residual = nn.Linear(768, 64, bias=False)\n': '',
        '        nn.init.zeros_(self.prompt_residual.weight)\n': '',
        'self.heads[name][0](fusion) + signal[:, i]': 'self.heads[name][0](fusion)',
        '            for i, (name, *_) in enumerate(TRAITS)\n': '            for name, *_ in TRAITS\n',
        'bge, question_mask, answer_mask, return_features=False):': 'bge, return_features=False):',
        'self.head(fusion, hidden, question_mask, answer_mask, mapping)': 'self.head(fusion)',
    }).replace('PromptAwareHeads', 'InterTraitHeads')


def build():
    base = BASE.parent / E10_NAME
    raw = base.read_bytes()
    notebook = json.loads(raw)
    previous_hash = notebook['metadata']['aes_standalone']['code_sha256']
    for index, cell in enumerate(notebook['cells']):
        source = ''.join(cell['source']).replace(previous_hash, MARKER)
        source = (source.replace('Reset_E10', 'Reset_E12')
                  .replace('E10 reset', 'E12 reset')
                  .replace('reset_e10_s3_multitrait', 'reset_e12_s3_multitrait')
                  .replace('aes_reset_e10_no_relation_loss', 'aes_reset_e12_intertrait_only'))
        role = cell['metadata'].get('aes_multitrait_role')
        if role == 'definition:SingleScoreModel':
            source = intertrait_model(source)
        elif role == 'definition:Engine':
            source = replace(source, {
                """        question_mask, answer_mask = prompt_answer_masks(
            details, windows['input_ids'].shape[1], self.tokenizer.padding_side, self.device)
        return self.model(question_mask=question_mask, answer_mask=answer_mask, windows=windows""":
                    '        return self.model(windows=windows',
                'windows, mapping, details = pair_batch_windows(': 'windows, mapping, _ = pair_batch_windows(',
            })
        elif role == 'prepare':
            source = replace(source, {"'fusion_variant': 'protact_adaptation'":
                                      "'fusion_variant': 'intertrait_attention_only'"})
        elif role == 'architecture':
            source = replace(source, {"    ['Attention soal–jawaban', '4 × 64'],\n": ''})
        cell['source'] = source
        cell['id'] = f'reset-e12-{index}'

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
        'base_sha256': hashlib.sha256(raw).hexdigest(), 'experiment': 'Reset_E12',
    }
    content = json.dumps(notebook, ensure_ascii=False, indent=1) + '\n'
    for folder in (BASE.parent, GROUP / 'notebooks'):
        (folder / NAME).write_text(content)


if __name__ == '__main__':
    build()
