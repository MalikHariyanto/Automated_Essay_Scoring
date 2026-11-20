"""Build the standalone reset E5 notebook from the reset E2 control."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'notebooks/multitrait_reset/AES_Reset_E2_S3_MultiTrait_TraitHeads.ipynb'
NAME = 'AES_Reset_E5_S3_MultiTrait_WeightedLoss.ipynb'
PLACEHOLDER = '__RESET_E5_CODE_SHA256__'

INTRO = """# E5 reset · S3 multi-trait · Weighted MSE

E5 menguji pembobotan ringan untuk skor yang jarang muncul. Pembandingnya
**E2 reset**, dengan model, split siswa, seed, dropout, learning rate, dan
jadwal training yang sama. Pilih runtime **T4 GPU**, hubungkan Drive ke
dataset semi-final, lalu jalankan notebook dari atas.

Untuk setiap trait, bobot awal kategori adalah akar dari rasio jumlah kategori
terbanyak terhadap jumlah kategori itu, dengan rasio bobot maksimum 2 sebelum
normalisasi. Rerata bobot pada baris training dibuat 1 untuk setiap trait.
Bobot memakai label training di fold tersebut, lalu dihitung ulang dari seluruh
development saat refit. Kategori yang absen dicatat dengan bobot kosong.
Objektif adalah rerata bobot × kuadrat error pada target 0–1. Bobot tidak
dinormalisasi ulang per microbatch. RMSE validasi pada skala asli tetap memilih
epoch, dan evaluasi OOF serta test memakai metrik biasa.

Unduh satu ZIP hasil. Setelah diekstrak, setiap bagian Excel .xlsx bisa
langsung dibuka. Kelompok 01–05 mengikuti laporan single-trait. Jika ada
workbook berukuran 30 MB atau lebih, notebook membaginya lagi dan mengecek
ukuran file aktual sebelum membuat ZIP. Mulai dari 04 untuk prediksi, evaluasi,
dan hitungan skor terendah yang dikenali, lalu 05 untuk audit bobot dan riwayat.
Output berisi data siswa dan disimpan secara privat.
"""

FIT_EDITS = {
    '    seed_everything(seed)': """    fit_weights, weight_audit = training_loss_weights(
        bundle.y[np.asarray(training, dtype=np.int64)], cfg['loss_weight_max_ratio']
    )
    seed_everything(seed)""",
    '    # Buat model baru': """    weights_by_row = dict(zip(map(int, training), fit_weights))
    # Buat model baru""",
    '        loss_total = 0.': '        loss_total = mse_total = 0.',
    '                # Normalisasi per trait memberi bobot sama pada empat dimensi loss.':
        '                # Target tetap pada skala 0–1; bobot tiap label berasal dari fold ini.',
    '                with torch.autocast': """                weights = torch.as_tensor(np.stack([weights_by_row[i] for i in ids]),
                                          dtype=torch.float32, device=device)
                with torch.autocast""",
    '                    # MSE mengukur selisih kuadrat antara prediksi dan target; optimizer berusaha memperkecilnya.\n                    loss = torch.mean((prediction-targets)**2)':
        """                    squared_error = (prediction-targets).square()
                    loss = (weights * squared_error).mean()""",
    '                loss_total += loss.item()*len(ids)': """                loss_total += loss.item()*len(ids)
                mse_total += squared_error.detach().mean().item()*len(ids)""",
    "'train_mse_scaled': loss_total/len(training),":
        "'train_mse_scaled': mse_total/len(training),\n               'train_weighted_mse_scaled': loss_total/len(training),",
    'loss latih (MSE, target 0–1): {row["train_mse_scaled"]:.6f}':
        'weighted MSE latih (target 0–1): {row["train_weighted_mse_scaled"]:.6f}',
    "'history': history, 'optimizer_steps'": "'history': history, 'audit_bobot_loss': weight_audit, 'optimizer_steps'",
}

REPORT_HELPERS = """def lowest_score_rows(bundle, indices, prediction, stage):
    scores = predictions_array(prediction, indices)
    rows = []
    for column, trait in enumerate(RUBRIC['traits']):
        predicted = ordinal_prediction(scores[:, column], trait)
        lowest = bundle.y[indices, column] == trait['min']
        total = int(lowest.sum())
        correct = int(np.sum(lowest & (predicted == trait['min'])))
        rows.append({
            'bagian': stage, 'trait': trait['name'], 'skor_terendah': trait['min'],
            'n_label_terendah': total, 'n_terdeteksi': correct,
            'proporsi_terdeteksi': correct / total if total else None,
        })
    return rows


"""

REPORT_EDITS = {
    "    cv_fold, oof_rows, history, indices, predictions = [], [], [], [], []":
        "    cv_fold, oof_rows, history, indices, predictions, audit = [], [], [], [], [], []",
    "        history.extend(history_rows(info, fold=fold))":
        "        history.extend(history_rows(info, fold=fold))\n        audit.extend({'fit': f'fold {fold}', **row} for row in info['audit_bobot_loss'])",
    "    return write_excel(path, {": """    audit.extend({'fit': 'refit', **row} for row in final_info['audit_bobot_loss'])
    evaluation = {""",
    """        'Riwayat Training': pd.DataFrame(history),
        **inspection_sheets,
    }, {""": """        'Skor Terendah': pd.DataFrame(
            lowest_score_rows(bundle, indices, predictions, 'OOF')
            + lowest_score_rows(bundle, test, prediction, 'test')
        ),
    }
    groups = {
        '01_Data_dan_Tokenisasi': {},
        '02_Vektor_Jawaban': {},
        '03_Vektor_Kata_dan_Kalimat': {},
        '04_Prediksi_dan_Evaluasi': evaluation,
        '05_Riwayat_Pelatihan': {
            'Riwayat Training': pd.DataFrame(history),
            'Bobot Loss per Fit': pd.DataFrame(audit),
        },
    }
    for name, frame in inspection_sheets.items():
        if name.startswith('Kata'):
            workbook = '03_Vektor_Kata_dan_Kalimat'
        elif name.startswith('Final_') or name in ('RoBERTa pretrained', 'BGE frozen'):
            workbook = '02_Vektor_Jawaban'
        elif name in ('Konfigurasi', 'Arsitektur'):
            workbook = '05_Riwayat_Pelatihan'
        else:
            workbook = '01_Data_dan_Tokenisasi'
        groups[workbook][name] = frame

    return export_excel_zip(path, groups, {""",
}

EXPORT = """if model_stage_ready('Ekspor ZIP', ('cv_summary', 'folds', 'final_info', 'test_prediction', 'inspection_sheets')):
    report_path = output / 'Hasil_AES_Reset_E5_S3_MultiTrait.zip'
    workbooks = export_results(report_path, bundle, manifest, cfg, cv_summary, folds, final_info, test_prediction)
    display(pd.DataFrame(workbooks)[['file', 'bytes']].assign(MB=lambda files: files['bytes'] / 1_000_000))
    print(f'ZIP hasil: {report_path}. Ekstrak untuk membuka setiap Excel secara terpisah.')
"""


def replace(source, edits):
    for original, updated in edits.items():
        if source.count(original) != 1:
            raise ValueError(f'Sumber E2 berubah pada: {original}')
        source = source.replace(original, updated)
    return source


def build():
    raw = BASE.read_bytes()
    notebook = json.loads(raw)
    base_hash = notebook['metadata']['aes_standalone']['code_sha256']
    cells = notebook['cells']
    for index, cell in enumerate(cells):
        source = ''.join(cell['source']).replace(base_hash, PLACEHOLDER)
        source = (source.replace('Reset_E2', 'Reset_E5')
                  .replace('reset_e2_s3_multitrait', 'reset_e5_s3_multitrait')
                  .replace('E2 reset', 'E5 reset'))
        role = cell['metadata'].get('aes_multitrait_role')
        if role == 'storage':
            source = replace(source, {"'outputs/aes_e43_multitrait'": "'outputs/aes_reset_e5_weighted_loss'"})
        elif role == 'prepare':
            source = replace(source, {"'runtime': runtime_information(),": """'runtime': runtime_information(),
    'training_loss': 'trait_inverse_sqrt_frequency_weighted_mse',
    'loss_weight_max_ratio': LOSS_WEIGHT_MAX_RATIO,"""})
        elif role == 'hyperparameters':
            assignments = [line.split('#', 1)[0].rstrip() for line in source.splitlines()]
            source = '# Resep training E2 dipertahankan.\n' + '\n'.join(assignments).strip()
            source += '\nLOSS_WEIGHT_MAX_RATIO = 2.0  # Rasio maksimum antarkategori dalam satu trait.\n'
        elif role == 'definition:fit':
            helper = (ROOT / 'scripts/reset_e5_weighted_loss.py').read_text()
            source = helper + '\n\n' + replace(source, FIT_EDITS)
        elif role == 'cv_definition':
            source = source.replace('Validasi silang satu resep E3', 'Validasi silang E5')
        elif role == 'final_features':
            source = replace(source, {'"MLP hidden", "dimensi": 256': '"Hidden per trait", "dimensi": 64'})
        elif role == 'definition:write_excel':
            source += '\n\n' + (ROOT / 'scripts/reset_e5_excel_zip.py').read_text()
        elif role == 'report':
            source = REPORT_HELPERS + replace(source, REPORT_EDITS)
        elif role == 'export_execute':
            source = EXPORT
        elif role == 'download':
            source = source.replace('Unduh workbook', 'Unduh ZIP')
        elif cell['cell_type'] == 'markdown':
            source = (source.replace('Susun Excel hasil', 'Susun ZIP berisi Excel hasil')
                      .replace('Unduh Excel', 'Unduh ZIP')
                      .replace('dengan AdamW dan MSE dari empat target skala 0–1',
                               'dengan AdamW dan MSE berbobot dari empat target skala 0–1')
                      .replace('### export execute', '### Buat ZIP dan periksa ukuran Excel'))
        cell['source'] = source
        cell['id'] = f'reset-e5-{index}'

    cells[0]['source'] = INTRO
    fingerprint = hashlib.sha256(
        '\n'.join(c['source'] for c in cells if c['cell_type'] == 'code').encode()
    ).hexdigest()
    for cell in cells:
        cell['source'] = cell['source'].replace(PLACEHOLDER, fingerprint)
        if cell['cell_type'] == 'code':
            ast.parse(cell['source'])
            cell['outputs'] = []
            cell['execution_count'] = None

    notebook['metadata']['colab']['name'] = NAME
    notebook['metadata']['aes_standalone'] = {
        'code_sha256': fingerprint, 'base': BASE.relative_to(ROOT).as_posix(),
        'base_sha256': hashlib.sha256(raw).hexdigest(), 'experiment': 'Reset_E5',
    }
    serialized = json.dumps(notebook, ensure_ascii=False, indent=1) + '\n'
    for folder in ('notebooks/multitrait_reset', 'eksperimen/03_tiga_input_multi_trait/notebooks'):
        (ROOT / folder / NAME).write_text(serialized)
    print(f'{NAME} · {fingerprint}')


if __name__ == '__main__':
    build()
