"""Build the matched seed-44 control for the E23 content-weight experiment."""

import hashlib
import json

from build_multitrait_reset_e23_seed44_notebook import BASE, GROUP, NAME as SOURCE_NAME, ROOT, replace

NAME = 'AES_Reset_E25_S3_Kontrol_Isi_1x_Seed44.ipynb'
MARKER = '__RESET_E25_CODE_SHA256__'
EXPERIMENT = 'Reset_E25_S3_Kontrol_Isi_1x_Seed44'
REFERENCE = GROUP / 'output/checkpoints/E23-Seed44.json'

INTRO = """# E25 · Kontrol bobot isi 1× pada seed 44

Pilih T4, hubungkan Drive dengan workbook semi-final yang sama, lalu klik Run all.
Lima fit development mengikuti E23 seed 44. Bobot loss isi menjadi 1× untuk
skor 1–4. Bobot organisasi tetap 1,2×; arsitektur, optimizer, seed fit 45–49,
split, ambang kategori, early stopping dan aturan A/B mengikuti E23.

Setelah tiap fold, lihat deteksi isi 1 di latih dan validasi. Setelah lima fold,
tabel final membandingkan E25 1× dan E23 2× pada checkpoint B untuk empat metrik
per trait, rata-rata fold, dan deteksi skor rendah. Angka E23 berasal dari arsip
run seed 44 yang sudah selesai. Tahap ini berakhir di development.

Unduh ZIP 04–05. Workbook privat berisi prediksi validasi per jawaban.
"""

COMPARE = '''
def compare_seed44_control(worksheets):
    reference = E23_SEED44_REFERENCE
    current = worksheets['OOF A B per Trait'].query("pilihan == 'B'").set_index('trait')
    results = []
    for row in reference['metrics']:
        trait = row['trait']
        for key in ('RMSE', 'Pearson', 'QWK', 'Gwet AC2'):
            control = current.loc[trait, key]
            weighted = row[key]
            delta = None if pd.isna(control) or pd.isna(weighted) else weighted - control
            results.append({'trait': trait, 'metrik': key, 'E25 1x (B)': control,
                            'E23 2x (B)': weighted, 'delta E23 minus E25': delta})
    comparisons = pd.DataFrame(results)
    checkpoint_b = worksheets['Pilihan Checkpoint'].query("pilihan == 'B'")
    validation = worksheets['Isi 1 Latih dan Validasi B'].query("bagian == 'validasi'")
    overview = pd.DataFrame([
        {'konfigurasi': 'E25 kontrol 1x', 'RMSE rata-rata fold': checkpoint_b['RMSE'].mean(),
         'macro QWK rata-rata fold': checkpoint_b['macro_QWK'].mean(),
         'label isi 1': validation['label 1'].sum(),
         'isi 1 terdeteksi': validation['label 1 terdeteksi'].sum(),
         'label 2–4 ke 1': validation['label 2–4 salah jadi 1'].sum()},
        reference['overview'],
    ])
    return comparisons, overview
'''


def reference_snapshot():
    run = json.loads(REFERENCE.read_text())
    choices = [r for r in run['tables']['Pilihan Checkpoint'] if r['pilihan'] == 'B']
    low = [r for r in run['tables']['Isi 1 Latih dan Validasi B'] if r['bagian'] == 'validasi']
    return {
        'split_sha256': run['metadata']['split_sha256'], 'seed': run['config']['seed'],
        'zip_sha256': run['source']['sha256'],
        'metrics': [{key: row[key] for key in ('trait', 'RMSE', 'Pearson', 'QWK', 'Gwet AC2')}
                    for row in run['tables']['OOF A B per Trait'] if row['pilihan'] == 'B'],
        'overview': {
            'konfigurasi': 'E23 bobot 2x',
            'RMSE rata-rata fold': sum(r['RMSE'] for r in choices) / len(choices),
            'macro QWK rata-rata fold': sum(r['macro_QWK'] for r in choices) / len(choices),
            'label isi 1': sum(r['label 1'] for r in low),
            'isi 1 terdeteksi': sum(r['label 1 terdeteksi'] for r in low),
            'label 2–4 ke 1': sum(r['label 2–4 salah jadi 1'] for r in low),
        },
    }


def build():
    base = BASE.parent / SOURCE_NAME
    raw = base.read_bytes()
    notebook = json.loads(raw)
    old_hash = notebook['metadata']['aes_standalone']['code_sha256']
    reference = reference_snapshot()

    for index, cell in enumerate(notebook['cells']):
        source = ''.join(cell['source']).replace(old_hash, MARKER)
        role = cell['metadata'].get('aes_multitrait_role')
        if index == 0:
            source = INTRO
        elif role == 'prepare':
            source = replace(source, {
                'Reset_E23_Konfirmasi_Seed44': EXPERIMENT,
                "'isi_loss_ratio': 2.0": "'isi_loss_ratio': 1.0",
                "output = run_folder(cfg, bundle)": (
                    f"E23_SEED44_REFERENCE = {reference!r}\n"
                    "if manifest['split_sha256'] != E23_SEED44_REFERENCE['split_sha256']:\n"
                    "    raise ValueError('Split E25 berbeda dari E23 seed 44. Periksa dataset dan versi scikit-learn.')\n"
                    "if cfg['seed'] != E23_SEED44_REFERENCE['seed']:\n"
                    "    raise ValueError('Seed E25 harus sama dengan E23 seed 44.')\n"
                    "output = run_folder(cfg, bundle)"
                ),
            })
        elif role == 'storage':
            source = replace(source, {'aes_reset_e23_seed44': 'aes_reset_e25_kontrol_1x_seed44'})
        elif role == 'fold_helpers':
            source = replace(source, {'Hasil_AES_Reset_E23_Seed44_Progres': 'Hasil_AES_Reset_E25_Progres'})
        elif role == 'validation_report':
            source = replace(source, {
                'def export_validation(path, final=False):': (
                    COMPARE + '\n\ndef export_validation(path, final=False):'
                ),
                '    export_excel_zip(path, {': (
                    "    if final:\n"
                    "        comparison, overview = compare_seed44_control(worksheets)\n"
                    "        worksheets['E25 vs E23 per Trait'] = comparison\n"
                    "        worksheets['E25 vs E23 Ringkasan'] = overview\n"
                    "    export_excel_zip(path, {"
                ),
                "'test_dievaluasi': False,": (
                    "'test_dievaluasi': False, 'e23_seed44_source_sha256': E23_SEED44_REFERENCE['zip_sha256'],"
                ),
            })
        elif role == 'final_report':
            source = replace(source, {
                'Hasil_AES_Reset_E23_Seed44_Validasi': 'Hasil_AES_Reset_E25_Kontrol_Seed44_Validasi',
                "        print(f'ZIP validasi A/B · {final_path}')": (
                    "        print('Perbandingan bobot isi dengan seed 44 yang sama')\n"
                    "        display(sheets['E25 vs E23 Ringkasan'].round(4))\n"
                    "        display(sheets['E25 vs E23 per Trait'].round(4))\n"
                    "        print(f'ZIP validasi A/B · {final_path}')"
                ),
            })
        cell['source'] = source
        cell['id'] = f'reset-e25-{index}'

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
    }
    content = json.dumps(notebook, ensure_ascii=False, indent=1) + '\n'
    for folder in (BASE.parent, GROUP / 'notebooks'):
        (folder / NAME).write_text(content, encoding='utf-8')


if __name__ == '__main__':
    build()
