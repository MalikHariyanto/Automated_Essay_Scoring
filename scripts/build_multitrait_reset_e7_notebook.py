"""Build E7 from E6, replacing input-dependent gates with learned trait weights."""

import ast
import hashlib
import json

from build_multitrait_reset_e6_notebook import BASE, GROUP, NAME as E6_NAME, replace

NAME = 'AES_Reset_E7_S3_MultiTrait_StaticTraitMix.ipynb'
PLACEHOLDER = '__RESET_E7_CODE_SHA256__'
INTRO = """# E7 reset · S3 multi-trait · Campuran tetap per trait

Setiap trait belajar satu pasangan bobot RoBERTa/BGE dari data training.
Setelah checkpoint dipilih, pasangan itu dipakai untuk semua jawaban.
Ini hipotesis dari hasil E6 tentang apakah campuran per trait sudah cukup.

Dua logit per trait dimulai dari nol. Softmax menghasilkan bobot 50:50 dan
pengali 1 di setiap cabang, sehingga prediksi awal sama dengan E2 untuk seed
yang sama. Delapan parameter baru memakai learning rate 1e-4. Resep E2
memakai MSE biasa, dropout 0,2, split siswa, dan lima fold yang sama.

Pilih T4 GPU, hubungkan Drive ke dataset semi-final, lalu jalankan dari atas.
ZIP ringkas 04–05 tersedia sesudah test sebelum inspeksi vektor. Ringkasan
Gate OOF menggabungkan beberapa checkpoint fold, jadi bobot bisa berbeda
antar-fold. Di dalam satu fold atau pada model refit, bobot tetap per trait.
Output berisi data privat.
"""


def build():
    base = BASE.parent / E6_NAME
    raw = base.read_bytes()
    notebook = json.loads(raw)
    old_hash = notebook['metadata']['aes_standalone']['code_sha256']
    cells = notebook['cells']
    for cell in cells:
        source = ''.join(cell['source']).replace(old_hash, PLACEHOLDER)
        source = (source.replace('Reset_E6', 'Reset_E7')
                  .replace('reset_e6_s3_multitrait', 'reset_e7_s3_multitrait')
                  .replace('aes_reset_e6_trait_gated_concat', 'aes_reset_e7_static_trait_mix')
                  .replace('E6 reset', 'E7 reset')
                  .replace('Validasi silang E6', 'Validasi silang E7')
                  .replace('trait_softmax_2x_identity', 'static_trait_softmax_2x_identity'))
        role = cell['metadata'].get('aes_multitrait_role')
        if role == 'definition:SingleScoreModel':
            source = replace(source, {
                'from the original concat': 'with weights shared across answers',
                """self.gates = nn.ModuleDict({name: nn.Linear(1024, 2) for name, *_ in TRAITS})
        for gate in self.gates.values():
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)""": """self.gates = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(2)) for name, *_ in TRAITS
        })""",
                'self.gates[name](fusion).float().softmax(dim=1)':
                    'self.gates[name].float().softmax(dim=0).expand(fusion.shape[0], -1)',
            })
        if role in ('architecture', 'final_features'):
            source = source.replace("    ['Concat pembaca gate', '1024'],\n", '')
            source = source.replace('1024 → 2', '2 logit tetap')
        cell['source'] = source

    cells[0]['source'] = INTRO
    fingerprint = hashlib.sha256(
        '\n'.join(c['source'] for c in cells if c['cell_type'] == 'code').encode()
    ).hexdigest()
    for i, cell in enumerate(cells):
        cell['id'] = f'reset-e7-{i}'
        cell['source'] = cell['source'].replace(PLACEHOLDER, fingerprint)
        if cell['cell_type'] == 'code':
            ast.parse(cell['source'])
    notebook['metadata']['colab']['name'] = NAME
    notebook['metadata']['aes_standalone'] = {
        'code_sha256': fingerprint, 'base': base.relative_to(BASE.parents[2]).as_posix(),
        'base_sha256': hashlib.sha256(raw).hexdigest(), 'experiment': 'Reset_E7',
    }
    serialized = json.dumps(notebook, ensure_ascii=False, indent=1) + '\n'
    for folder in (BASE.parent, GROUP / 'notebooks'):
        (folder / NAME).write_text(serialized)
    print(f'{NAME} · {fingerprint}')


if __name__ == '__main__':
    build()
