"""Build E28: ScalarMix-4 on the E25 multi-trait architecture."""

import hashlib
import json

from build_multitrait_reset_e25_notebook import BASE, GROUP, ROOT, NAME as E25_NAME, replace

NAME = 'AES_Reset_E28_S3_ScalarMix4_MultiTrait.ipynb'
EXPERIMENT = 'Reset_E28_S3_ScalarMix4_MultiTrait'
MARKER = '__RESET_E28_CODE_SHA256__'
E43 = ROOT / ('eksperimen/02_tiga_input_single_trait/notebooks/'
              'AES_300_E43_E42_S3_MaskedMeanPooling_Native_Learnable_GPU.ipynb')

INTRO = """# E28 · ScalarMix-4 untuk S3 multi-trait

Campurkan empat layer terakhir RoBERTa dengan empat bobot softmax terlatih
dan gamma terlatih. Awalnya bobot 0,25 per layer dan gamma 1. Campuran ini
menggantikan fitur layer terakhir yang masuk ke CLS dan attention E25.
BGE frozen, attention soal–jawaban/antartrait, dan empat head mengikuti E25.

Pilih T4 dan Run all. Lima fit development memakai seed dasar 44, LR encoder
5e-6, LR modul lain 1e-4, dropout 0,2, bobot isi 1× dan organisasi 1,2×.
Hasil tampil per fold, fold selesai tersimpan, lalu tabel final membandingkan
empat metrik per trait dengan E25-B. Unduh ZIP 04–05.

ScalarMix menambah lima parameter, serta memerlukan keluaran empat layer
encoder saat training. Uji ini dimulai dari pretrained encoder dan latihan baru
pada empat label. Hasil evaluasi berhenti di development.
"""

ARCHITECTURE = """display(pd.DataFrame([
    ['ScalarMix RoBERTa', '4 layer · softmax awal 0,25; gamma awal 1'],
    ['Attention soal–jawaban', 'Isi, kebahasaan, mekanik seperti E25'],
    ['Attention antartrait', 'E25'],
    ['BGE frozen → proyeksi', '1024 → 256'],
    ['Concat → empat head', '1024 → 64 → 1 per trait'],
    ['Tambahan parameter', '5 skalar terlatih'],
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
    scalar_mix = e43['definition:SingleScoreModel'].split('class SingleScoreModel', 1)[0]
    architecture = scalar_mix + '\n\n' + replace(e25['definition:SingleScoreModel'], {
        'def __init__(self, scenario, roberta, s3_mlp_dropout=0.2):':
            'def __init__(self, scenario, roberta, s3_mlp_dropout=0.2, scalar_mix_layers=4, gamma_initial=1.0):',
        '        hidden = self.roberta(**windows).last_hidden_state':
            '        encoded = self.roberta(**windows, output_hidden_states=True, return_dict=True)\n'
            "        hidden = self.scalar_mix(encoded.hidden_states, windows['attention_mask'])",
        '            self.head = PromptAwareHeads(s3_mlp_dropout)':
            '            self.head = PromptAwareHeads(s3_mlp_dropout)\n'
            '        self.scalar_mix = ScalarMix(scalar_mix_layers, gamma_initial)',
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
                    "'fusion_variant': 'e25_scalar_mix4',\n"
                    "    'scalar_mix_layers': 4, 'scalar_mix_gamma_initial': 1.0,",
                reference_line: f'E25_SEED44_REFERENCE = {reference!r}',
                'Split E25 berbeda dari E23 seed 44.': 'Split E28 berbeda dari E25 seed 44.',
                'Seed E25 harus sama dengan E23 seed 44.': 'Seed E28 harus sama dengan E25 seed 44.',
            })
            source = source.replace('E23_SEED44_REFERENCE[', 'E25_SEED44_REFERENCE[')
        elif role == 'definition:SingleScoreModel':
            source = architecture
        elif role == 'definition:make_model':
            source = replace(source, {
                "s3_mlp_dropout=cfg['s3_mlp_dropout'])":
                    "s3_mlp_dropout=cfg['s3_mlp_dropout'],\n"
                    "                           scalar_mix_layers=cfg['scalar_mix_layers'],\n"
                    "                           gamma_initial=cfg['scalar_mix_gamma_initial'])",
            })
        elif role == 'architecture':
            source = ARCHITECTURE
        elif role in ('storage', 'fold_helpers', 'validation_report', 'final_report'):
            for old, new in {
                'aes_reset_e25_kontrol_1x_seed44': 'aes_reset_e28_s3_scalarmix4_seed44',
                'Hasil_AES_Reset_E25_Progres': 'Hasil_AES_Reset_E28_Progres',
                'Hasil_AES_Reset_E25_Kontrol_Seed44_Validasi': 'Hasil_AES_Reset_E28_Validasi',
                'E23_SEED44_REFERENCE': 'E25_SEED44_REFERENCE',
                'e23_seed44_source_sha256': 'e25_seed44_source_sha256',
                'compare_seed44_control': 'compare_architecture',
                'E25 1x (B)': 'E28 ScalarMix (B)', 'E23 2x (B)': 'E25 kontrol (B)',
                'delta E23 minus E25': 'delta E28 minus E25',
                'weighted - control': 'control - weighted',
                'E25 kontrol 1x': 'E28 ScalarMix4', 'E25 vs E23': 'E28 vs E25',
                'Perbandingan bobot isi dengan seed 44 yang sama': 'Perbandingan arsitektur pada seed 44',
            }.items():
                source = source.replace(old, new)
        elif cell['cell_type'] == 'markdown':
            source = source.replace('Arsitektur S3 dari E13', 'Arsitektur E25 dengan ScalarMix-4')
        if role == 'fold_helpers':
            source = replace(source, {
                "        del engine, model":
                    "        info['scalar_mix_B'] = model.scalar_mix.report(model.roberta.config.num_hidden_layers)\n"
                    "        del engine, model",
            })
        if role == 'validation_report':
            source = replace(source, {
                "            'Konfigurasi': pd.DataFrame":
                    "            'ScalarMix Checkpoint B': pd.DataFrame([{'fold': number, **row}\n"
                    "                for number, result in sorted(fold_results.items())\n"
                    "                for row in result['training']['scalar_mix_B']]),\n"
                    "            'Konfigurasi': pd.DataFrame",
            })
        cell['source'] = source
        cell['id'] = f'reset-e28-{index}'

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
