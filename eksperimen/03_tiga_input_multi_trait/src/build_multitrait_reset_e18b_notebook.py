"""Build the development-only E13 + Trial 60 Colab from the E13 template."""

import hashlib
import json

from build_multitrait_reset_e6_notebook import BASE, GROUP, ROOT, replace
from build_multitrait_reset_e13_notebook import NAME as E13_NAME


NAME = 'AES_Reset_E18B_E13_Trial60_Validasi.ipynb'
MARKER = '__RESET_E18B_CODE_SHA256__'
INTRO = """# E18B · E13 dengan setelan Trial 60

Tujuan: lihat apakah paket hyperparameter yang menang di E17E membantu arsitektur
E13. S3 tetap memakai RoBERTa + BGE, empat head, dan attention E13. Sinyal
attention soal–jawaban langsung tetap dilewati untuk organisasi.

Pilih **T4 GPU**, lalu jalankan section dari atas. Data, split, loss, seed, batch,
maksimal 25 epoch dan patience 5 mengikuti E13. Bagian 5 memperlihatkan lima
perubahan hyperparameter. Hanya ada **lima fit validasi** pada satu seed dasar.

Setiap fold punya cell sendiri. Begitu selesai, lihat tabel metrik dan skor
rendahnya tepat di bawah cell. Hasil fold dan ZIP progres disimpan ke Drive, jadi
fold yang sudah lengkap bisa diteruskan setelah runtime putus. Bagian terakhir
merangkum prediksi validasi dan membuat ZIP final 04–05. Run ini berakhir di
development. Holdout test tetap disisihkan.

Untuk membandingkan dengan hasil E13 lama, opsional isi path ZIP ringkas E13 di
bagian 5. File hasil siswa tetap privat di Drive.
"""


FOLD_HELPERS = """from time import perf_counter


def run_validation_fold(number):
    fold = manifest['folds'][number - 1]
    if fold['fold'] != number:
        raise ValueError(f'Nomor fold tidak sesuai: {number}.')
    training, validation = fold['train'], fold['validation']
    seed = cfg['seed'] + number
    path = output / f'fold_{number}.json'
    identity = {
        'split': manifest['split_sha256'],
        'fold': number,
        'train': training,
        'validation': validation,
        'seed': seed,
    }
    if path.exists():
        result = json.loads(path.read_text())
        if result['identity'] != identity:
            raise ValueError(f'Cache fold {number} memakai pembagian atau seed berbeda.')
        print(f'Fold {number}/{len(manifest["folds"])} · hasil tersimpan dipakai ulang.')
    else:
        print(f'Fold {number}/{len(manifest["folds"])} · seed {seed} · '
              f'{len(training)} jawaban latih, {len(validation)} validasi', flush=True)
        started = perf_counter()
        model, engine, info = fit('S3', training, validation, bundle, bge,
                                  cfg, cfg['epochs'], seed)
        prediction = predictions_array(engine.predict(validation), validation)
        result = {'identity': identity, 'prediction': prediction.tolist(),
                  'training': info, 'seconds': perf_counter() - started}
        assert_best_epoch(result)
        dump(path, result)
        del model, engine
        release()
    assert_best_epoch(result)
    return result


def assert_best_epoch(result):
    valid = result['identity']['validation']
    prediction = predictions_array(result['prediction'], valid)
    score = metrics(bundle.y[valid], prediction)['RMSE']
    if not math.isclose(score, result['training']['best_validation_rmse'], abs_tol=1e-6):
        raise ValueError('Prediksi tersimpan tidak cocok dengan RMSE epoch terpilih.')


def show_fold(number):
    if not model_stage_ready(f'Fold {number}', ('bge', 'fold_results')):
        return
    result = run_validation_fold(number)
    fold_results[number] = result
    valid, prediction = result['identity']['validation'], result['prediction']
    info = result['training']
    result_metrics = metrics(bundle.y[valid], prediction)
    print(f'Fold {number} tersimpan · terbaik epoch {info["best_epoch"]} '
          f'dari {info["epochs_completed"]} · RMSE {result_metrics["RMSE"]:.4f} · '
          f'{result["seconds"] / 60:.1f} menit saat training pertama.')
    display(pd.DataFrame(trait_rows(result_metrics, n=len(valid))).round(4))
    display(pd.DataFrame(lowest_score_rows(bundle, valid, prediction, 'validasi')))
    progress_path = output / 'Hasil_AES_Reset_E18B_Progres_04_05.zip'
    export_validation(progress_path)
    print(f'Progres {len(fold_results)}/{len(manifest["folds"])} fold · {progress_path}')


fold_results = {}
"""


VALIDATION_REPORT = """def export_validation(path, comparison=None, final=False):
    summaries, per_trait, predictions, history, low_scores = [], [], [], [], []
    indices, scores = [], []
    for number, result in sorted(fold_results.items()):
        valid, predicted = result['identity']['validation'], result['prediction']
        info = result['training']
        evaluation = metrics(bundle.y[valid], predicted)
        summaries.append({
            'fold': number, 'RMSE': evaluation['RMSE'],
            'epoch_terpilih': info['best_epoch'], 'epoch_dijalankan': info['epochs_completed'],
            'menit_training': result['seconds'] / 60,
        })
        per_trait.extend(trait_rows(evaluation, fold=number, epoch_terpilih=info['best_epoch'], n=len(valid)))
        predictions.extend(prediction_rows(bundle, valid, predicted, fold=number))
        history.extend(history_rows(info, fold=number))
        low_scores.extend({'fold': number, **row} for row in
                          lowest_score_rows(bundle, valid, predicted, 'validasi'))
        indices.extend(valid)
        scores.extend(predicted)

    completed = len(fold_results)
    if final and sorted(indices) != sorted(manifest['development']):
        raise ValueError('ZIP final membutuhkan tepat satu prediksi untuk setiap jawaban development.')
    summary = {
        'eksperimen': cfg['experiment'], 'fold_selesai': completed,
        'jumlah_fold': len(manifest['folds']), 'jawaban_OOF': len(indices),
        'mean_cv_RMSE': float(np.mean([row['RMSE'] for row in summaries])),
        'status': 'lengkap' if final else 'progres',
    }
    worksheets = {
        'Ringkasan CV': pd.DataFrame([summary]),
        'Progres Fold': pd.DataFrame(summaries),
        'CV per Fold dan Trait': pd.DataFrame(per_trait),
        'Prediksi OOF': pd.DataFrame(predictions),
        'Skor Terendah': pd.DataFrame(low_scores),
    }
    if final:
        evaluation = metrics(bundle.y[indices], scores)
        worksheets['CV OOF per Trait'] = pd.DataFrame(trait_rows(evaluation, n=len(indices)))
        worksheets['Skor Terendah'] = pd.concat([
            worksheets['Skor Terendah'],
            pd.DataFrame(lowest_score_rows(bundle, indices, scores, 'OOF')),
        ], ignore_index=True)
    if comparison is not None:
        worksheets.update(comparison)

    export_excel_zip(path, {
        '04_Prediksi_dan_Evaluasi': worksheets,
        '05_Riwayat_Pelatihan': {
            'Riwayat Training': pd.DataFrame(history),
            'Konfigurasi': pd.DataFrame(cfg.items(), columns=['parameter', 'nilai']),
        },
    }, {
        **bundle.summary, 'experiment': cfg['experiment'], 'trial_sumber': 60,
        'development_rows': len(manifest['development']),
        'split_sha256': manifest['split_sha256'], 'notebook_code_sha256': NOTEBOOK_CODE_SHA256,
        'final_config': cfg, 'test_dipakai_untuk_seleksi_epoch': False,
        'test_dievaluasi': False, 'fold_selesai': completed,
    })
    return worksheets
"""


COMPARE_CONTROL = """def compare_e13(path):
    from io import BytesIO

    with zipfile.ZipFile(path) as archive:
        matches = [name for name in archive.namelist()
                   if Path(name).name == '04_Prediksi_dan_Evaluasi.xlsx']
        if len(matches) != 1:
            raise ValueError('ZIP kontrol harus berisi satu workbook 04 E13.')
        content = BytesIO(archive.read(matches[0]))
    with pd.ExcelFile(content, engine='openpyxl') as workbook:
        info_sheet = workbook.parse('Info')
        info = dict(zip(info_sheet['field'], info_sheet['value']))
        control = workbook.parse('Prediksi OOF')

    if info.get('eksperimen') != 'Reset_E13' or info.get('source_sha256') != bundle.source_sha256:
        raise ValueError('Gunakan hasil S3 E13 yang memakai dataset yang sama.')
    expected_scope = {**manifest, 'scope': 'reset_e13_s3_multitrait'}
    expected_scope.pop('split_sha256')
    expected_sha = hashlib.sha256(json.dumps(expected_scope, sort_keys=True).encode()).hexdigest()
    if info.get('split_sha256') != expected_sha:
        raise ValueError('Hash pembagian data E13 berbeda dari pembagian notebook ini.')

    candidate_rows = [
        row for number, result in sorted(fold_results.items())
        for row in prediction_rows(bundle, result['identity']['validation'],
                                   result['prediction'], fold=number)
    ]
    candidate = pd.DataFrame(candidate_rows)
    paired_keys = ['id_jawaban', 'fold', 'kelompok',
                   *(f'{name}_label' for name, *_ in TRAITS)]
    old = control.sort_values('id_jawaban').reset_index(drop=True)
    new = candidate.sort_values('id_jawaban').reset_index(drop=True)
    if not old[paired_keys].equals(new[paired_keys]):
        raise ValueError('ID, label, atau fold berbeda antara kandidat dan kontrol.')

    trait_names = [name for name, *_ in TRAITS]
    y = new[[f'{name}_label' for name in trait_names]].to_numpy()
    old_scores = old[[f'{name}_prediksi_mentah' for name in trait_names]].to_numpy()
    new_scores = new[[f'{name}_prediksi_mentah' for name in trait_names]].to_numpy()
    old_error, new_error = (y - old_scores)**2, (y - new_scores)**2
    fold_rows = []
    for number in sorted(fold_results):
        mask = new['fold'].eq(number).to_numpy()
        control_rmse = float(np.sqrt(old_error[mask].mean()))
        candidate_rmse = float(np.sqrt(new_error[mask].mean()))
        fold_rows.append({'fold': number, 'RMSE E13': control_rmse, 'RMSE kandidat': candidate_rmse,
                          'selisih kandidat - E13': candidate_rmse - control_rmse})

    old_metrics, new_metrics = metrics(y, old_scores)['traits'], metrics(y, new_scores)['traits']
    trait_rows_comparison, low_rows = [], []
    for name in trait_names:
        for metric in ('RMSE', 'Pearson', 'QWK', 'AC2'):
            trait_rows_comparison.append({
                'trait': name, 'metrik': metric, 'E13': old_metrics[name][metric],
                'kandidat': new_metrics[name][metric],
            })
        for label, frame in [('E13', old), ('kandidat', new)]:
            true = frame[f'{name}_label']
            pred = frame[f'{name}_prediksi_kategori']
            low_rows.append({
                'trait': name, 'model': label, 'label 1': int(true.eq(1).sum()),
                'skor 1 terdeteksi': int((true.eq(1) & pred.eq(1)).sum()),
                'skor 2 salah jadi 1': int((true.eq(2) & pred.eq(1)).sum()),
            })
    return {
        'Banding E13 per Fold': pd.DataFrame(fold_rows),
        'Banding E13 per Trait': pd.DataFrame(trait_rows_comparison),
        'Banding E13 Skor Rendah': pd.DataFrame(low_rows),
    }
"""


FINAL_REPORT = """if model_stage_ready('Rekap validasi', ('fold_results',)):
    expected_folds = {fold['fold'] for fold in manifest['folds']}
    if set(fold_results) != expected_folds:
        print('Rekap final menunggu fold:', ', '.join(map(str, sorted(expected_folds - fold_results.keys()))))
    else:
        comparison = None
        if E13_CONTROL_ZIP:
            control_path = Path(E13_CONTROL_ZIP)
            if not control_path.is_file():
                raise FileNotFoundError(f'ZIP E13 belum ditemukan: {control_path}. '
                                        'Perbaiki path di bagian 5 atau kosongkan, lalu ulangi rekap.')
            comparison = compare_e13(control_path)

        final_path = output / 'Hasil_AES_Reset_E18B_Validasi_04_05.zip'
        sheets = export_validation(final_path, comparison=comparison, final=True)
        display(sheets['Ringkasan CV'].round(4))
        display(sheets['Progres Fold'].round(4))
        display(sheets['CV OOF per Trait'].round(4))
        display(sheets['Skor Terendah'].query("bagian == 'OOF'").round(4))
        if comparison is not None:
            display(comparison['Banding E13 per Fold'].round(4))
            display(comparison['Banding E13 per Trait'].round(4))
            display(comparison['Banding E13 Skor Rendah'])
        else:
            print('ZIP E13 belum disertakan. Rekap ini berisi hasil kandidat.')
        print(f'ZIP validasi final · {final_path}')
        if 'google.colab' in sys.modules:
            from google.colab import files
            files.download(str(final_path))
"""


def update_settings(source):
    source = replace(source, {
        '# HIPERPARAMETER BASELINE E43; Scalar Mix empat layer terakhir.':
            '# Paket Trial 60 dari E17E. Parameter lain mengikuti E13.',
        'TRAIN_SEED = 42  # Satu konstanta pengacakan, tanpa loop multi-seed.':
            'TRAIN_SEED = 42  # Seed fit = seed dasar + nomor fold, seperti E13.',
        'ENCODER_LR = 5e-6             # Tetap E30; bukan LR E37 7e-6.':
            'ENCODER_LR = 1.640743321580258e-05',
        'S3_MLP_LR = 1e-4              # MLP S3 tetap nilai asli E30.':
            'AUX_LR = 3.345252006826264e-05  # Untuk head, proyeksi, dan layer di luar RoBERTa.',
        'S3_LAYERS_LR = 1e-4           # Kembali ke E30: BiGRU/co-attention/pooling/norm = 1e-4.':
            'S3_MLP_LR = S3_LAYERS_LR = AUX_LR',
        'PROJECTION_LR = 1e-4          # Proyeksi BGE projection.* tidak ikut diturunkan.':
            'PROJECTION_LR = AUX_LR',
        'S3_MLP_DROPOUT = 0.2             # E3: dropout MLP E1 dinaikkan dari 0,1 ke 0,2.':
            'S3_MLP_DROPOUT = 0.30',
        'WEIGHT_DECAY = 0.01': 'WEIGHT_DECAY = 0.03',
    })
    return source + """
# Opsional: isi dengan path lengkap ZIP ringkas kontrol di Drive.
E13_CONTROL_ZIP = ''
"""


def build():
    base = BASE.parent / E13_NAME
    raw = base.read_bytes()
    notebook = json.loads(raw)
    previous_hash = notebook['metadata']['aes_standalone']['code_sha256']
    cells = notebook['cells']
    notebook['cells'] = []
    for index, cell in enumerate(cells[:81]):
        source = ''.join(cell['source']).replace(previous_hash, MARKER)
        source = (source.replace('Reset_E13', 'Reset_E18B_E13_Trial60')
                  .replace('E13 reset', 'E18B')
                  .replace('reset_e13_s3_multitrait', 'reset_e18b_e13_trial60')
                  .replace('aes_reset_e13_organisasi_prompt_ablation', 'aes_reset_e18b_e13_trial60'))
        role = cell['metadata'].get('aes_multitrait_role')
        if role == 'hyperparameters':
            source = update_settings(source)
        if role == 'configuration_defaults':
            source = replace(source, {"'head_lr': 1e-4": "'head_lr': AUX_LR"})
        if role == 'storage':
            source = replace(source, {
                "OUTPUT_ROOT = Path('outputs/aes_reset_e18b_e13_trial60')":
                    """OUTPUT_NAME = 'aes_reset_e18b_e13_trial60'
IN_COLAB = 'google.colab' in sys.modules or Path('/content').is_dir()
OUTPUT_ROOT = (DRIVE_MOUNT / 'MyDrive' / 'AES' / 'outputs' / OUTPUT_NAME
               if IN_COLAB else Path('outputs') / OUTPUT_NAME)""",
            })
        if role == 'prepare':
            source += '\nprint(f\'Folder penyimpanan run: {output}\')\n'
        if role == 'bge_execute':
            source = replace(source, {
                '    bge = bge_features(bundle, cfg, output)':
                    """    development = manifest['development']
    dev_bundle = Bundle([bundle.rows[i] for i in development],
                        bundle.y[development], bundle.groups[development],
                        bundle.source_sha256, bundle.summary)
    dev_features = bge_features(dev_bundle, cfg, output)
    # Pertahankan indeks workbook asli yang dipakai fold.
    bge = dev_features.new_full((len(bundle.rows), dev_features.shape[1]), float('nan'))
    bge[development] = dev_features""",
                "'jumlah_vektor': bge.shape[0]": "'vektor_development': len(dev_features)",
            })
        if index == 0:
            source = INTRO
        if index == 60:
            source = '## 12 · Arsitektur S3 dari E13'
        if index == 61:
            continue
        cell['source'] = source
        notebook['cells'].append(cell)
        if role == 'hyperparameters':
            append_cell(notebook, 'code', """display(pd.DataFrame([
    ('LR RoBERTa', 5e-6, ENCODER_LR),
    ('LR head / proyeksi / layer lain', 1e-4, AUX_LR),
    ('Dropout head', 0.2, S3_MLP_DROPOUT),
    ('Weight decay', 0.01, WEIGHT_DECAY),
    ('Warmup ratio', 0.10, WARMUP_RATIO),
], columns=['Setelan', 'E13', 'Trial 60']))
print(f'Maksimum {EPOCHS} epoch per fold · patience {EARLY_STOPPING_PATIENCE} · '
      f'{FOLDS} fold · seed dasar {TRAIN_SEED}')""", 'recipe_preview')

    append_cell(notebook, 'markdown', """## 16 · Mulai validasi dan siapkan penyimpanan

Definisi di bawah menyiapkan tabel dan ZIP. Lima cell sesudahnya menjalankan
training satu fold per cell, menampilkan metrik, lalu menyimpan hasilnya.
Jalankan kembali dari atas untuk memulihkan progres dari Drive. Fold yang sempat
putus di tengah dilatih lagi dari awal.""")
    for index in (120, 122, 124):
        source = ''.join(cells[index]['source'])
        if index == 124:
            source = source[:source.index('def export_results(')].rstrip()
        append_cell(notebook, 'code', source, cells[index]['metadata']['aes_multitrait_role'])
    append_cell(notebook, 'code', VALIDATION_REPORT, 'validation_report')
    append_cell(notebook, 'code', FOLD_HELPERS, 'fold_helpers')
    for number in range(1, 6):
        append_cell(notebook, 'markdown', f'### Fold {number} dari 5')
        append_cell(notebook, 'code', f'show_fold({number})', f'fold_{number}')
    append_cell(notebook, 'markdown', """## 17 · Rekap lima fold dan unduh ZIP 04–05

Di sini semua prediksi validasi disatukan. Lihat RMSE tiap fold, empat metrik
per trait, dan jumlah skor 1 yang berhasil dikenali. Jika path ZIP E13 diisi,
bandingkan hasilnya pada baris validasi yang sama.""")
    append_cell(notebook, 'code', COMPARE_CONTROL, 'compare_control')
    append_cell(notebook, 'code', FINAL_REPORT, 'final_report')

    for index, cell in enumerate(notebook['cells']):
        cell['id'] = f'reset-e18b-{index}'
    code = '\n'.join(cell['source'] for cell in notebook['cells'] if cell['cell_type'] == 'code')
    fingerprint = hashlib.sha256(code.encode()).hexdigest()
    for cell in notebook['cells']:
        cell['source'] = cell['source'].replace(MARKER, fingerprint)
        if cell['cell_type'] == 'code':
            compile(cell['source'], cell['id'], 'exec')
    notebook['metadata']['colab']['name'] = NAME
    notebook['metadata']['aes_standalone'] = {
        'code_sha256': fingerprint, 'base': base.relative_to(ROOT).as_posix(),
        'base_sha256': hashlib.sha256(raw).hexdigest(), 'experiment': 'Reset_E18B_E13_Trial60',
    }
    content = json.dumps(notebook, ensure_ascii=False, indent=1) + '\n'
    for folder in (BASE.parent, GROUP / 'notebooks'):
        (folder / NAME).write_text(content, encoding='utf-8')


def append_cell(notebook, kind, source, role=None):
    metadata = {'aes_multitrait_role': role} if role else {}
    cell = {'cell_type': kind, 'metadata': metadata, 'source': source}
    if kind == 'code':
        cell.update(execution_count=None, outputs=[])
    notebook['cells'].append(cell)


if __name__ == '__main__':
    build()
