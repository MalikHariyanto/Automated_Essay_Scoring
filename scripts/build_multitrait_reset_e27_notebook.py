"""Build E27: E43's representation path with E25's multi-trait recipe."""

import hashlib
import json

from build_multitrait_reset_e25_notebook import BASE, GROUP, ROOT, NAME as E25_NAME, replace

NAME = 'AES_Reset_E27_S3_E43_CoAttention_MultiTrait.ipynb'
EXPERIMENT = 'Reset_E27_S3_E43_CoAttention_MultiTrait'
MARKER = '__RESET_E27_CODE_SHA256__'
E43 = ROOT / ('eksperimen/02_tiga_input_single_trait/notebooks/'
              'AES_300_E43_E42_S3_MaskedMeanPooling_Native_Learnable_GPU.ipynb')

INTRO = """# E27 · Latih ulang jalur representasi E43 untuk empat trait

RoBERTa → ScalarMix empat layer + gamma → BiGRU 64 per arah pada token jawaban
→ co-attention dua arah dengan token referensi → masked mean → proyeksi RoBERTa 768.
Gabungkan dengan BGE-M3 frozen yang diproyeksikan ke 256, lalu prediksi empat
trait dengan empat head E25. Skor guru menjadi target loss.

Pilih T4, hubungkan Drive ke workbook semi-final, dan jalankan dari atas.
Lima fit development memakai seed dasar 44, bobot isi 1×, bobot organisasi
1,2×, dropout 0,2, dan LR RoBERTa 5e-6 serta modul lain 1e-4. Aturan A/B,
split, dan ambang kategori mengikuti E25. Fold selesai dapat dipakai ulang.
Setelah validasi, bandingkan dengan E25-B dan unduh ZIP 04–05.

Intervensi ini menguji satu paket arsitektur. Mulai dari pretrained encoder
yang sama dan latih ulang modul representasi serta empat head. Bobot single-trait,
paket Trial 60, dan label test tidak digunakan.
"""

MODEL = '''
class MultiTraitModel(nn.Module):
    """Jalur representasi E43 dengan empat head skor dari E25."""

    def __init__(self, scenario, roberta, s3_mlp_dropout, scalar_mix_layers, gamma_initial, bigru_hidden):
        super().__init__()
        self.scenario, self.roberta = scenario, roberta
        self.roberta.requires_grad_(True)
        self.roberta_norm = nn.LayerNorm(768)
        self.projection = nn.Sequential(nn.Linear(1024, 256), nn.LayerNorm(256))
        # Pertahankan urutan inisialisasi head E25, termasuk RNG dari head awal.
        self.head = nn.Sequential(nn.Linear(1024, 256), nn.GELU(),
                                  nn.Dropout(s3_mlp_dropout), nn.Linear(256, 4))
        with torch.random.fork_rng(devices=[]):
            self.head = IndependentTraitHeads(s3_mlp_dropout)
        self.sequence_pool = AnswerBiGRUCoAttentionMeanPool(bigru_hidden)
        self.scalar_mix = ScalarMix(scalar_mix_layers, gamma_initial)

    def forward(self, windows, mapping, n_essays, bge, answer_mask, reference_mask, return_features=False):
        encoded = self.roberta(**windows, output_hidden_states=True, return_dict=True)
        mixed = self.scalar_mix(encoded.hidden_states, windows['attention_mask'])
        answer_windows = self.sequence_pool(mixed, answer_mask, reference_mask)
        roberta = mean_by_essay(answer_windows, mapping, n_essays)
        projection = self.projection(bge)
        fusion = torch.cat([self.roberta_norm(roberta), projection], dim=1)
        prediction = self.head(fusion)
        if return_features:
            return prediction, {'roberta': roberta, 'bge': bge, 'projection': projection, 'concat': fusion}
        return prediction
'''

ARCHITECTURE = """display(pd.DataFrame([
    ['ScalarMix RoBERTa', '4 layer · softmax awal 0,25; gamma awal 1'],
    ['BiGRU jawaban', '64 per arah → 128'],
    ['Co-attention jawaban–referensi', 'Dua arah · matching 640 → 128'],
    ['Masked mean → proyeksi RoBERTa', '128 → 768'],
    ['BGE-M3 frozen → proyeksi', '1024 → 256'],
    ['Concat → empat head', '1024 → 64 → 1 per trait'],
], columns=['Bagian', 'Dimensi / setelan']))
"""

def reference_snapshot():
    run = json.loads((GROUP / 'output/checkpoints/E25.json').read_text())
    return {
        'split_sha256': run['metadata']['split_sha256'],
        'seed': run['config']['seed'],
        'zip_sha256': run['source']['sha256'],
        'metrics': [{key: row[key] for key in ('trait', 'RMSE', 'Pearson', 'QWK', 'Gwet AC2')}
                    for row in run['tables']['OOF A B per Trait'] if row['pilihan'] == 'B'],
        'overview': run['tables']['E25 vs E23 Ringkasan'][0],
    }


def source_map(notebook, key):
    return {c['metadata'].get(key): ''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code'}


def build():
    base = BASE.parent / E25_NAME
    raw, e43_raw = base.read_bytes(), E43.read_bytes()
    notebook = json.loads(raw)
    e43 = source_map(json.loads(e43_raw), 'aes43_role')
    e25 = source_map(notebook, 'aes_multitrait_role')
    old_hash = notebook['metadata']['aes_standalone']['code_sha256']
    independent = e25['definition:SingleScoreModel'].split('class PromptAwareHeads', 1)[0]
    trait_loss = e25['definition:SingleScoreModel'].split('def correlated_trait_pairs', 1)[1].split('class MultiTraitModel', 1)[0]
    scalar_mix = e43['definition:SingleScoreModel'].split('class SingleScoreModel', 1)[0]
    reused = ['answer_token_mask', 'reference_token_mask', 'check_answer_pool_inputs',
              'AnswerBiGRUAttention', 'AnswerBiGRUCoAttentionMeanPool']
    architecture = '\n\n'.join([independent, *(e43['definition:'+key] for key in reused), scalar_mix, 'def correlated_trait_pairs' + trait_loss, MODEL])
    engine = replace(e25['definition:Engine'], {
        'Engine E43 hanya menjalankan S3.': 'Engine E27 hanya menjalankan S3.',
        '        question_mask, answer_mask = prompt_answer_masks(\n'
        "            details, windows['input_ids'].shape[1], self.tokenizer.padding_side, self.device)\n"
        '        return self.model(question_mask=question_mask, answer_mask=answer_mask, windows=windows, mapping=mapping, n_essays=len(ids),':
            '        answer_mask = answer_token_mask(windows, mapping, details, self.tokenizer.padding_side)\n'
            '        reference_mask = reference_token_mask(windows, mapping, details, self.tokenizer.padding_side)\n'
            '        return self.model(answer_mask=answer_mask, reference_mask=reference_mask, windows=windows,\n'
            '                          mapping=mapping, n_essays=len(ids),',
    })
    reference = reference_snapshot()
    for index, cell in enumerate(notebook['cells']):
        source = ''.join(cell['source']).replace(old_hash, MARKER)
        role = cell['metadata'].get('aes_multitrait_role')
        if index == 0:
            source = INTRO
        elif role == 'prepare':
            reference_line = next(line for line in source.splitlines() if line.startswith('E23_SEED44_REFERENCE ='))
            source = replace(source, {
                'Reset_E25_S3_Kontrol_Isi_1x_Seed44': EXPERIMENT,
                "'fusion_variant': 'organisasi_prompt_ablation',":
                    "'fusion_variant': 'e43_bigru_coattention_masked_mean',\n"
                    "    'scalar_mix_layers': 4, 'scalar_mix_gamma_initial': 1.0, 'bigru_hidden': 64,",
                reference_line: f'E25_SEED44_REFERENCE = {reference!r}',
                'Split E25 berbeda dari E23 seed 44.': 'Split E27 berbeda dari E25 seed 44.',
                'Seed E25 harus sama dengan E23 seed 44.': 'Seed E27 harus sama dengan E25 seed 44.',
            })
            source = source.replace('E23_SEED44_REFERENCE[', 'E25_SEED44_REFERENCE[')
        elif role == 'definition:SingleScoreModel':
            source = architecture
        elif role == 'definition:Engine':
            source = engine
        elif role == 'definition:make_model':
            source = """def make_model(scenario, cfg):
    return MultiTraitModel(
        scenario, encoder(cfg, 'roberta'), s3_mlp_dropout=cfg['s3_mlp_dropout'],
        scalar_mix_layers=cfg['scalar_mix_layers'], gamma_initial=cfg['scalar_mix_gamma_initial'],
        bigru_hidden=cfg['bigru_hidden'],
    )
"""
        elif role == 'architecture':
            source = ARCHITECTURE
        elif role in ('storage', 'fold_helpers', 'validation_report', 'final_report'):
            for old, new in {
                'aes_reset_e25_kontrol_1x_seed44': 'aes_reset_e27_s3_e43_seed44',
                'Hasil_AES_Reset_E25_Progres': 'Hasil_AES_Reset_E27_Progres',
                'Hasil_AES_Reset_E25_Kontrol_Seed44_Validasi': 'Hasil_AES_Reset_E27_Validasi',
                'E23_SEED44_REFERENCE': 'E25_SEED44_REFERENCE',
                'e23_seed44_source_sha256': 'e25_seed44_source_sha256',
                'compare_seed44_control': 'compare_architecture',
                'E25 1x (B)': 'E27 E43 (B)', 'E23 2x (B)': 'E25 kontrol (B)',
                'delta E23 minus E25': 'delta E27 minus E25',
                'weighted - control': 'control - weighted',
                'E25 kontrol 1x': 'E27 jalur E43', 'E25 vs E23': 'E27 vs E25',
                'Perbandingan bobot isi dengan seed 44 yang sama': 'Perbandingan arsitektur pada seed 44',
            }.items():
                source = source.replace(old, new)
        elif cell['cell_type'] == 'markdown':
            source = source.replace('Arsitektur S3 dari E13', 'Arsitektur E43 untuk S3 multi-trait')
        cell['source'] = source
        cell['id'] = f'reset-e27-{index}'

    fingerprint = hashlib.sha256('\n'.join(c['source'] for c in notebook['cells']
                                            if c['cell_type'] == 'code').encode()).hexdigest()
    for cell in notebook['cells']:
        cell['source'] = cell['source'].replace(MARKER, fingerprint)
        if cell['cell_type'] == 'code':
            compile(cell['source'], cell['id'], 'exec')
    notebook['metadata']['colab']['name'] = NAME
    notebook['metadata']['aes_standalone'] = {
        'code_sha256': fingerprint, 'base': base.relative_to(ROOT).as_posix(),
        'base_sha256': hashlib.sha256(raw).hexdigest(), 'experiment': EXPERIMENT,
        'architecture_source': E43.relative_to(ROOT).as_posix(),
        'architecture_source_sha256': hashlib.sha256(e43_raw).hexdigest(),
    }
    content = json.dumps(notebook, ensure_ascii=False, indent=1) + '\n'
    for folder in (BASE.parent, GROUP / 'notebooks'):
        (folder / NAME).write_text(content, encoding='utf-8')


if __name__ == '__main__':
    build()
