"""Build E20, a paired comparison of two checkpoint rules on the same fits."""

import hashlib
import json

from build_multitrait_reset_e6_notebook import BASE, GROUP, ROOT, replace
from build_multitrait_reset_e19_notebook import NAME as E19_NAME

NAME = 'AES_Reset_E20_E19_Pilih_Checkpoint_AB.ipynb'
MARKER = '__RESET_E20_CODE_SHA256__'
INTRO = """# E20 · Satu training, dua aturan pilih checkpoint

Resep E19 dipakai untuk lima fold S3. Bobot loss organisasi tetap 1,2× untuk
skor 1, ambang tetap 1,50, dan early stopping tetap menurut RMSE gabungan
dengan patience 5. Semua epoch yang selesai menyimpan prediksi validasi
di memori untuk memilih dua keluaran dari training yang sama.

- **A** memakai epoch dengan RMSE gabungan terendah.
- **B** memakai epoch dengan rata-rata QWK empat trait tertinggi di antara epoch
  yang RMSE-nya paling banyak 0,01 lebih besar daripada A.
  Jika seri, pilih RMSE lebih rendah, lalu epoch lebih awal.

Pilih T4 GPU dan jalankan dari atas. Setiap fold menampilkan perbandingan dan
menyimpan progres ke Drive. Setelah lima fold, ZIP 04–05 memuat prediksi
validasi A/B, metrik per trait, salah kategori, dan riwayat per epoch.
Run ini berhenti pada validasi. Hasil siswa tetap privat.
"""

SELECTION = """def choose_checkpoints(history, predictions_by_epoch, margin):
    # Early stopping dan checkpoint A memakai RMSE, seperti pada E19.
    evaluated = [row for row in history if 'validation' in row]
    best_rmse = min(evaluated, key=lambda row: (row['validation']['RMSE'], row['epoch']))
    rmse_limit = best_rmse['validation']['RMSE'] + margin
    candidates = []
    for row in evaluated:
        validation = row['validation']
        qwk = [validation['traits'][name]['QWK'] for name, *_ in TRAITS]
        if validation['RMSE'] <= rmse_limit + 1e-12 and all(value is not None for value in qwk):
            candidates.append((row, float(np.mean(qwk))))
    # Jika QWK suatu trait tidak terdefinisi untuk semua kandidat, pakai A.
    best_qwk = best_rmse
    if candidates:
        best_qwk, _ = min(
            candidates, key=lambda item: (-item[1], item[0]['validation']['RMSE'], item[0]['epoch']),
        )

    return {
        choice: {'epoch': row['epoch'], 'prediction': predictions_by_epoch[row['epoch']].tolist()}
        for choice, row in [('A', best_rmse), ('B', best_qwk)]
    }, len(candidates)


"""


def update_fit(source):
    return SELECTION + replace(source, {
        '    history, active_steps, optimizer_steps, skipped_steps = [], 0, 0, 0':
            '    predictions_by_epoch = {}\n    history, active_steps, optimizer_steps, skipped_steps = [], 0, 0, 0',
        '            row[\'validation\'] = metrics(bundle.y[validation], predicted)':
            """            predictions_by_epoch[epoch] = predicted.copy()
            row['validation'] = metrics(bundle.y[validation], predicted)""",
        '    return model, engine, info':
            """    choices, eligible = choose_checkpoints(history, predictions_by_epoch, cfg['checkpoint_rmse_margin'])
    if choices['A']['epoch'] != best_epoch:
        raise ValueError('Epoch A berbeda dari checkpoint RMSE yang dipulihkan.')
    info['eligible_epochs_for_B'] = eligible
    return model, engine, info, choices""",
    })


REPORT_HELPER = """def checkpoint_tables():
    per_fold, per_trait, predictions, low_scores, epoch_metrics = [], [], [], [], []
    for number, result in sorted(fold_results.items()):
        valid = result['identity']['validation']
        for choice, checkpoint in result['checkpoints'].items():
            prediction = checkpoint['prediction']
            evaluation = metrics(bundle.y[valid], prediction)
            per_fold.append({
                'fold': number, 'pilihan': choice, 'epoch': checkpoint['epoch'],
                'RMSE': evaluation['RMSE'],
                'macro_QWK': mean_defined_qwk(evaluation),
            })
            tags = {'pilihan': choice, 'fold': number}
            per_trait.extend(trait_rows(evaluation, **tags, n=len(valid)))
            predictions.extend(prediction_rows(bundle, valid, prediction, **tags))
            low_scores.extend(checkpoint_score_errors(valid, prediction, **tags))
        for row in result['training']['history']:
            epoch_metrics.extend(trait_rows(row['validation'], fold=number, epoch=row['epoch'], n=len(valid)))

    return {
        'Pilihan Checkpoint': pd.DataFrame(per_fold),
        'A B per Fold dan Trait': pd.DataFrame(per_trait),
        'Prediksi OOF A B': pd.DataFrame(predictions),
        'Kesalahan Skor Rendah': pd.DataFrame(low_scores),
    }, pd.DataFrame(epoch_metrics)


def mean_defined_qwk(evaluation):
    values = [stats['QWK'] for stats in evaluation['traits'].values()]
    return float(np.mean(values)) if all(x is not None for x in values) else None


def checkpoint_score_errors(indices, predicted, **keys):
    errors = []
    for column, trait in enumerate(RUBRIC['traits']):
        labels = bundle.y[indices, column]
        category = ordinal_prediction(np.asarray(predicted)[:, column], trait)
        errors.append({
            **keys, 'trait': trait['name'], 'jumlah label 1': int(np.sum(labels == 1)),
            '1 benar': int(np.sum((labels == 1) & (category == 1))),
            '1 salah jadi 2': int(np.sum((labels == 1) & (category == 2))),
            '2 salah jadi 1': int(np.sum((labels == 2) & (category == 1))),
        })
    return errors


"""


def update_fold_helpers(source):
    source = replace(source, {
        "        model, engine, info = fit('S3'": "        model, engine, info, checkpoints = fit('S3'",
        "                  'training': info, 'seconds': perf_counter() - started}":
            "                  'training': info, 'checkpoints': checkpoints, 'seconds': perf_counter() - started}",
        "        raise ValueError('Prediksi tersimpan tidak cocok dengan RMSE epoch terpilih.')":
            """        raise ValueError('Prediksi tersimpan tidak cocok dengan RMSE epoch terpilih.')
    history = {row['epoch']: row['validation'] for row in result['training']['history']}
    for choice, checkpoint in result['checkpoints'].items():
        selected = predictions_array(checkpoint['prediction'], valid)
        expected = history[checkpoint['epoch']]
        actual = metrics(bundle.y[valid], selected)
        if not math.isclose(actual['RMSE'], expected['RMSE'], abs_tol=1e-6):
            raise ValueError(f'Prediksi checkpoint {choice} berbeda dari riwayat epoch.')
        if mean_defined_qwk(actual) != mean_defined_qwk(expected):
            raise ValueError(f'QWK checkpoint {choice} berbeda dari riwayat epoch.')""",
        """    display(pd.DataFrame(lowest_score_rows(bundle, valid, prediction, 'validasi')))
    display(pd.DataFrame(info['organization_loss_weights']).round(4))""":
            """    options = []
    for choice, checkpoint in result['checkpoints'].items():
        evaluation = metrics(bundle.y[valid], checkpoint['prediction'])
        org = next(row for row in checkpoint_score_errors(valid, checkpoint['prediction'])
                   if row['trait'] == 'organisasi')
        options.append({'pilihan': choice, 'epoch': checkpoint['epoch'],
                        'RMSE': evaluation['RMSE'], 'macro_QWK': mean_defined_qwk(evaluation),
                        'QWK organisasi': evaluation['traits']['organisasi']['QWK'],
                        'skor organisasi 1 benar': org['1 benar'],
                        'skor organisasi 2 salah jadi 1': org['2 salah jadi 1']})
    display(pd.DataFrame(options).round(4))
    if not info['eligible_epochs_for_B']:
        print('QWK empat trait tidak lengkap pada kandidat yang memenuhi batas; B memakai A.')""",
    })
    return source


def update_report(source):
    return REPORT_HELPER + replace(source, {
        '    if comparison is not None:\n        worksheets.update(comparison)':
            """    paired, epoch_metrics = checkpoint_tables()
    worksheets.update(paired)
    if final:
        pooled_rows, pooled_errors = [], []
        ab_predictions = paired['Prediksi OOF A B']
        for choice, frame in ab_predictions.groupby('pilihan', sort=True):
            label_columns = [f'{name}_label' for name, *_ in TRAITS]
            pred_columns = [f'{name}_prediksi_mentah' for name, *_ in TRAITS]
            if frame['id_jawaban'].duplicated().any() or len(frame) != len(indices):
                raise ValueError(f'Prediksi OOF {choice} harus lengkap dan unik.')
            evaluation = metrics(frame[label_columns].to_numpy(), frame[pred_columns].to_numpy())
            pooled_rows.extend(trait_rows(evaluation, pilihan=choice, n=len(indices)))
            pooled_errors.extend(checkpoint_score_errors(
                indices, frame[pred_columns].to_numpy(), pilihan=choice,
            ))
        worksheets['OOF A B per Trait'] = pd.DataFrame(pooled_rows)
        worksheets['Kesalahan Skor Rendah OOF'] = pd.DataFrame(pooled_errors)""",
        "'Riwayat Training': pd.DataFrame(history),":
            "'Riwayat Training': pd.DataFrame(history),\n            'Metrik Validasi per Epoch': epoch_metrics,",
    })


FINAL_REPORT = """if model_stage_ready('Rekap validasi A dan B', ('fold_results',)):
    expected_folds = {fold['fold'] for fold in manifest['folds']}
    if set(fold_results) != expected_folds:
        print('Rekap final menunggu fold:', ', '.join(map(str, sorted(expected_folds - fold_results.keys()))))
    else:
        final_path = output / 'Hasil_AES_Reset_E20_Validasi_Checkpoint_AB_04_05.zip'
        sheets = export_validation(final_path, final=True)
        print('RMSE dan macro QWK · rata-rata lima fold')
        display(sheets['Pilihan Checkpoint'].groupby('pilihan')[['RMSE', 'macro_QWK']].mean().round(4))
        print('Metrik gabungan dari prediksi OOF masing-masing pilihan')
        display(sheets['OOF A B per Trait'].round(4))
        display(sheets['Kesalahan Skor Rendah OOF'])
        print(f'ZIP validasi A/B · {final_path}')
        if 'google.colab' in sys.modules:
            from google.colab import files
            files.download(str(final_path))
"""


def build():
    base = BASE.parent / E19_NAME
    raw = base.read_bytes()
    notebook = json.loads(raw)
    old_hash = notebook['metadata']['aes_standalone']['code_sha256']
    cells = []
    for index, cell in enumerate(notebook['cells']):
        role = cell['metadata'].get('aes_multitrait_role')
        source = (''.join(cell['source']).replace(old_hash, MARKER)
                  .replace('Reset_E19_E13_Bobot_Organisasi', 'Reset_E20_E19_Pilih_Checkpoint_AB')
                  .replace('reset_e19_e13_bobot_organisasi', 'reset_e20_e19_checkpoint_ab')
                  .replace('Hasil_AES_Reset_E19_', 'Hasil_AES_Reset_E20_'))
        if index == 0:
            source = INTRO
        elif role == 'hyperparameters':
            source = replace(source, {
                "# Opsional: isi path lengkap ZIP ringkas kontrol E13 di Drive.\nE13_CONTROL_ZIP = ''":
                    "# Margin RMSE untuk kandidat checkpoint B, ditetapkan sebelum run.\nCHECKPOINT_RMSE_MARGIN = 0.01",
            })
        elif role == 'recipe_preview':
            source += "\nprint(f'Checkpoint B · margin RMSE {CHECKPOINT_RMSE_MARGIN:g} di atas A.')"
        elif role == 'prepare':
            source = replace(source, {
                "'organization_loss_ratio': ORGANIZATION_LOSS_RATIO,":
                    "'organization_loss_ratio': ORGANIZATION_LOSS_RATIO,\n    'checkpoint_rmse_margin': CHECKPOINT_RMSE_MARGIN,",
            })
        elif role == 'definition:fit':
            source = update_fit(source)
        elif role == 'fold_helpers':
            source = update_fold_helpers(source)
        elif role == 'validation_report':
            source = update_report(source).replace(
                'def export_validation(path, comparison=None, final=False):',
                'def export_validation(path, final=False):',
            )
        elif role == 'compare_control':
            continue
        elif role == 'final_report':
            source = FINAL_REPORT
        elif cell['cell_type'] == 'markdown' and source.startswith('## 17'):
            source = '## 17 · Bandingkan checkpoint A dan B\n\nRekap prediksi validasi dan unduh ZIP 04–05.'
        cell['source'] = source
        cell['id'] = f'reset-e20-{index}'
        cells.append(cell)

    notebook['cells'] = cells
    code = '\n'.join(cell['source'] for cell in cells if cell['cell_type'] == 'code')
    fingerprint = hashlib.sha256(code.encode()).hexdigest()
    for cell in cells:
        cell['source'] = cell['source'].replace(MARKER, fingerprint)
        if cell['cell_type'] == 'code':
            compile(cell['source'], cell['id'], 'exec')
    notebook['metadata']['colab']['name'] = NAME
    notebook['metadata']['aes_standalone'] = {
        'code_sha256': fingerprint, 'base': base.relative_to(ROOT).as_posix(),
        'base_sha256': hashlib.sha256(raw).hexdigest(), 'experiment': 'Reset_E20_E19_Pilih_Checkpoint_AB',
    }
    content = json.dumps(notebook, ensure_ascii=False, indent=1) + '\n'
    for folder in (BASE.parent, GROUP / 'notebooks'):
        (folder / NAME).write_text(content, encoding='utf-8')


if __name__ == '__main__':
    build()
