"""Build E8 with explicit BGE distances in the E2 content head."""

import ast
import hashlib
import json

from build_multitrait_reset_e6_notebook import BASE, GROUP, NAME as E6_NAME, ROOT, replace

NAME = 'AES_Reset_E8_S3_MultiTrait_BGERelevance.ipynb'
MARKER = '__RESET_E8_CODE_SHA256__'
INTRO = """# E8 reset · S3 multi-trait · Kecocokan BGE untuk isi

E8 menambahkan dua jarak ke head isi E2: jawaban–soal dan jawaban–kunci.
BGE-M3 frozen membaca setiap teks secara terpisah. Representasi teks adalah
rerata CLS window. Jarak berupa RMS selisih antardimensi, lalu log1p.
Jarak kecil menunjukkan representasi yang lebih dekat dalam ruang BGE.
Ide menilai kesesuaian dengan soal terinspirasi dari [MOOSE](https://aclanthology.org/2025.acl-long.884/).

Concat E2 masuk ke keempat head. Head isi juga menerima dua jarak melalui
proyeksi 2 → 64 yang mulai dari bobot nol dan dilatih dengan LR 1e-4.
E8 mengikuti split, MSE, optimizer, dan jadwal E2. BGE tetap frozen.

Pilih GPU T4 dan jalankan dari atas. ZIP ringkas 04–05 keluar sebelum
inspeksi vektor. Excel 04 mencatat dua jarak untuk OOF dan test.
"""

DISTANCES = """

def bge_relevance(bundle, cfg):
    from functools import lru_cache

    device = device_name(cfg)
    model = encoder(cfg, 'bge').to(device).eval()
    tok = tokenizer(cfg, 'bge')
    maximum, overlap = cfg['bge_max_length'], cfg['bge_overlap']
    capacity = maximum - tok.num_special_tokens_to_add(pair=False)
    stride = capacity - overlap
    if stride <= 0:
        raise ValueError('Window BGE harus lebih panjang dari overlap.')

    @lru_cache(maxsize=None)
    def pooled(text):
        ids = tok.encode(text, add_special_tokens=False)
        starts = [0]
        while starts[-1] + capacity < len(ids):
            starts.append(starts[-1] + stride)
        vectors = []
        for start in starts:
            window = tok.build_inputs_with_special_tokens(ids[start:start + capacity])
            tokens = torch.tensor([window], device=device)
            vectors.append(model(input_ids=tokens, attention_mask=torch.ones_like(tokens))
                           .last_hidden_state[0, 0].float().cpu())
        return torch.stack(vectors).mean(dim=0)

    rows = []
    with torch.inference_mode():
        for answer, (question, reference) in zip(bundle.texts, contexts_from_bundle(bundle)):
            response = pooled(answer)
            rows.append(torch.stack([
                (response - pooled(other)).square().mean().sqrt().log1p()
                for other in (question, reference)
            ]))
    result = torch.stack(rows)
    if result.shape != (len(bundle.rows), 2) or not torch.isfinite(result).all():
        raise ValueError('Dimensi atau nilai jarak BGE tidak sah.')
    return result
"""


def build():
    template = BASE.parent / E6_NAME
    notebook = json.loads(template.read_bytes())
    original = json.loads(BASE.read_bytes())
    original_roles = {c['metadata'].get('aes_multitrait_role'): ''.join(c['source']) for c in original['cells']}
    restore = {'definition:SingleScoreModel', 'definition:Engine', 'definition:optimizer_groups',
               'cv_definition', 'test_definition', 'test_execute', 'architecture', 'final_features'}
    for cell in notebook['cells']:
        role = cell['metadata'].get('aes_multitrait_role')
        source = original_roles[role] if role in restore else ''.join(cell['source'])
        source = (source.replace('Reset_E6', 'Reset_E8').replace('Reset_E2', 'Reset_E8')
                  .replace('reset_e6_s3_multitrait', 'reset_e8_s3_multitrait')
                  .replace('aes_reset_e6_trait_gated_concat', 'aes_reset_e8_bge_relevance')
                  .replace('E2 reset', 'E8 reset').replace('E6 reset', 'E8 reset'))
        if role == 'prepare':
            source = replace(source, {"'fusion_gate': 'trait_softmax_2x_identity', 'gate_lr': 1e-4,":
                                      "'content_relevance': 'bge_separate_text_log1p_rms_distance',"})
        elif role == 'definition:bge_features':
            source += DISTANCES
        elif role == 'bge_execute':
            source += "\n    bge = torch.cat([bge, bge_relevance(bundle, cfg)], dim=1)\n"
        elif role == 'definition:SingleScoreModel':
            source = replace(source, {
                '    def forward(self, fusion):': """        self.relevance = nn.Linear(2, 64, bias=False)
        nn.init.zeros_(self.relevance.weight)

    def forward(self, fusion, relevance):""",
                '        return torch.cat(\n            [self.heads[name](fusion) for name, *_ in TRAITS], dim=1\n        )': """        content = self.heads['isi']
        hidden = content[0](fusion) + self.relevance(relevance)
        scores = [content[3](content[2](content[1](hidden)))]
        scores.extend(self.heads[name](fusion) for name, *_ in TRAITS if name != 'isi')
        return torch.cat(scores, dim=1)""",
                '        projected = self.projection(bge)': '        bge, relevance = bge.split((1024, 2), dim=1)\n        bge = bge.contiguous()\n        projected = self.projection(bge)',
                'prediction = self.head(fusion)': 'prediction = self.head(fusion, relevance)',
                'shared four-output MLP': 'two distances for the content head',
            })
        elif role == 'architecture':
            source = replace(source, {"    ['Jumlah head', '4'],": "    ['Jarak BGE untuk isi', '2 → 64'],\n    ['Jumlah head', '4'],"})
        elif role == 'report':
            source = source.replace('test_gate, inspection=None', 'distance, inspection=None')
            source = replace(source, {
                '    oof_gate_rows = []\n': '',
                "        oof_gate_rows.extend(gate_rows(bundle, valid, result['gate'], fold=fold))\n": '',
                '    oof_gate = pd.DataFrame(oof_gate_rows)\n': '',
                '    test_gate = pd.DataFrame(gate_rows(bundle, test, test_gate))\n': '',
                """        'Gate OOF': oof_gate,
        'Gate Test': test_gate,
        'Ringkasan Gate': pd.DataFrame(
            gate_summary(oof_gate, 'OOF') + gate_summary(test_gate, 'test')
        ),""": """        'Jarak BGE OOF': distance_rows(bundle, indices, distance),
        'Jarak BGE Test': distance_rows(bundle, test, distance),""",
            })
            source += """

def distance_rows(bundle, indices, distance):
    return pd.DataFrame({
        'id_jawaban': [bundle.rows[i]['row_id'] for i in indices],
        'jarak_jawaban_soal': distance[indices, 0],
        'jarak_jawaban_kunci': distance[indices, 1],
    })
"""
        elif role == 'test_execute':
            source += """

    quick_path = output / 'Hasil_AES_Reset_E8_Ringkas_04_05.zip'
    export_results(quick_path, bundle, manifest, cfg, cv_summary, folds, final_info,
                   test_prediction, bge[:, -2:].numpy())
    print(f'ZIP ringkas siap diunduh: {quick_path}')
    if 'google.colab' in sys.modules:
        from google.colab import files
        files.download(str(quick_path))
"""
        elif role == 'export_execute':
            source = source.replace("'test_gate'", "'test_prediction'").replace(
                'test_prediction, test_gate,', 'test_prediction, bge[:, -2:].numpy(),')
        elif role == 'final_features':
            source = source.replace('"MLP hidden", "dimensi": 256', '"Hidden per trait", "dimensi": 64')
        elif role == 'inspection':
            source = replace(source, {'bge.numpy()': 'bge[:, :1024].numpy()'})
        cell['source'] = source.replace(notebook['metadata']['aes_standalone']['code_sha256'], MARKER)

    notebook['cells'][0]['source'] = INTRO
    code = '\n'.join(c['source'] for c in notebook['cells'] if c['cell_type'] == 'code')
    fingerprint = hashlib.sha256(code.encode()).hexdigest()
    for index, cell in enumerate(notebook['cells']):
        cell['id'] = f'reset-e8-{index}'
        cell['source'] = cell['source'].replace(MARKER, fingerprint)
        if cell['cell_type'] == 'code':
            ast.parse(cell['source'])
    notebook['metadata']['colab']['name'] = NAME
    notebook['metadata']['aes_standalone'] = {
        'code_sha256': fingerprint, 'base': BASE.relative_to(ROOT).as_posix(),
        'base_sha256': hashlib.sha256(BASE.read_bytes()).hexdigest(), 'experiment': 'Reset_E8',
    }
    content = json.dumps(notebook, ensure_ascii=False, indent=1) + '\n'
    for folder in (BASE.parent, GROUP / 'notebooks'):
        (folder / NAME).write_text(content)


if __name__ == '__main__':
    build()
