"""Notebook cells for the S3 four-trait experiment, assembled by build_s3_multitrait_notebook.py."""

# %% columns
SCOPE = 'e43_s3_multitrait_three_schools'
SHEET_NAME = 'Sheet1'
QUESTION_COLUMN = 'Pertanyaan'
REFERENCE_COLUMN = 'Kunci_Jawaban'
ANSWER_COLUMN = 'Jawaban'
GROUP_COLUMN = 'Email Address'
RESPONSE_NUMBER_COLUMN = 'No_Soal'
INPUT_COLUMNS = (QUESTION_COLUMN, REFERENCE_COLUMN, ANSWER_COLUMN)

# Setel sesuai pedoman penilaian sebelum training. Rentang awal sesuai kategori di file contoh.
TRAITS = (
    ('isi', 'Skor_Isi', 1, 4),
    ('organisasi', 'Skor_Organisasi', 1, 2),
    ('kebahasaan', 'Skor_Tata_Bahasa', 1, 2),
    ('mekanik', 'Skor_Mekanik', 1, 2),
)
TARGET_COLUMNS = tuple(column for _, column, _, _ in TRAITS)
REQUIRED = (*INPUT_COLUMNS, GROUP_COLUMN, RESPONSE_NUMBER_COLUMN, *TARGET_COLUMNS)
RUBRIC = {'traits': [
    {'name': name, 'min': low, 'max': high, 'categories': list(range(low, high + 1))}
    for name, _, low, high in TRAITS
]}
SCORE_MIN = np.asarray([low for _, _, low, _ in TRAITS], dtype=np.float32)
SCORE_SPAN = np.asarray([high - low for _, _, low, high in TRAITS], dtype=np.float32)

# %% storage
# Notebook eksperimen berikutnya bisa memakai path dan SHA-256 yang sama.
DATASET_VERSION = 'semi-final'
DATASET_NAME = 'data_cleaned_3sekolah.xlsx'
DATASET_SHA256 = 'dfbd97128b09a4eb488dd890ccc3efd9b9fe232e74fcc00ee5504a03acc58226'
DRIVE_MOUNT = Path('/content/drive')
DRIVE_DATASET = DRIVE_MOUNT / 'MyDrive' / 'AES' / 'datasets' / 'semi_final' / DATASET_NAME
LOCAL_DATASET = Path(DATASET_NAME)  # Untuk eksekusi di luar Colab.
OUTPUT_ROOT = Path('outputs/aes_e43_multitrait')


def resolve_colab_dataset(mount_drive, upload, drive_mount, drive_dataset):
    """Gunakan salinan privat Drive atau simpan workbook yang cocok pada upload pertama."""
    from tempfile import NamedTemporaryFile

    mount_drive(str(drive_mount))
    if not (drive_mount / 'MyDrive').is_dir():
        raise RuntimeError('MyDrive belum tersedia. Hubungkan Google Drive di Colab.')
    if not drive_dataset.exists():
        print(f'Unggah {DATASET_NAME} untuk penyimpanan pertama di Drive.')
        uploads = upload()
        if len(uploads) != 1:
            raise ValueError(f'Unggah satu workbook asli bernama {DATASET_NAME}.')
        workbook = next(iter(uploads.values()))
        if hashlib.sha256(workbook).hexdigest() != DATASET_SHA256:
            raise ValueError('File unggahan berbeda dari workbook semi-final yang dipakai eksperimen ini.')
        drive_dataset.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=drive_dataset.parent, suffix='.pending', delete=False) as stream:
            staging = Path(stream.name)
            try:
                stream.write(workbook)
                stream.close()
                if digest(staging) != DATASET_SHA256:
                    raise IOError('Salinan di Drive gagal diverifikasi.')
                if not drive_dataset.exists():
                    staging.replace(drive_dataset)
            finally:
                staging.unlink(missing_ok=True)
        print(f'Workbook semi-final tersedia di Drive: {drive_dataset}')
    return drive_dataset


if 'google.colab' in sys.modules or Path('/content').is_dir():
    from google.colab import drive, files
    DATASET_PATH = resolve_colab_dataset(drive.mount, files.upload, DRIVE_MOUNT, DRIVE_DATASET)
else:
    DATASET_PATH = LOCAL_DATASET

if not DATASET_PATH.is_file():
    raise FileNotFoundError(f'Workbook semi-final belum ditemukan: {DATASET_PATH}')
if digest(DATASET_PATH) != DATASET_SHA256:
    raise ValueError(f'Workbook di {DATASET_PATH} berbeda dari versi semi-final terpilih. '
                     'Gunakan file asli atau tetapkan path dan SHA-256 versi baru secara eksplisit.')
print(f'Dataset {DATASET_VERSION} terverifikasi: {DATASET_PATH}')

# %% prepare
cfg = config({
    'dataset': str(DATASET_PATH), 'output': str(OUTPUT_ROOT),
    'experiment': 'E43', 'dataset_version': DATASET_VERSION, 'traits': TRAITS,
    'runtime': {'torch': str(torch.__version__), 'transformers': version('transformers'),
                'scikit-learn': version('scikit-learn'), 'device': device_name(DEFAULTS),
                'accelerator': torch.cuda.get_device_name() if DEVICE == 'cuda' else None},
})
bundle = read_dataset(cfg)
print(f"{len(bundle.rows)} jawaban, {bundle.summary['email_groups']} akun, "
      f"{bundle.summary['connected_groups']} kelompok setelah penggabungan duplikat.")
print(f"X berisi {len(INPUT_COLUMNS)} kolom teks: {', '.join(INPUT_COLUMNS)}")
print(f"y berukuran {bundle.y.shape}")
display(pd.DataFrame([{
    'trait': name, 'kolom_target': column, 'minimum': low, 'maksimum': high,
} for name, column, low, high in TRAITS]))

# %% split_execute
manifest = make_split(bundle, cfg)
output = run_folder(cfg, bundle)
dump(output / 'split.json', manifest)

print(f"Development: {len(manifest['development'])} | Test: {len(manifest['test'])}")
display(pd.DataFrame([{
    'fold': f['fold'], 'training': len(f['train']), 'validation': len(f['validation']),
} for f in manifest['folds']]))

# %% token_preview
# Ambil satu jawaban development dan perlihatkan ukuran window serta posisi yang dipakai E43.
example_index = manifest['development'][0]
example_row = bundle.rows[example_index]
example_context = (example_row['question'], example_row['reference'])
example_tokenizer = tokenizer(cfg, 'roberta')
example_windows, example_detail = pair_content_windows(
    example_row['text'], example_context, example_tokenizer,
    cfg['roberta_max_length'], cfg['roberta_overlap'], cfg,
)
display(pd.DataFrame([{
    'id_jawaban': example_row['row_id'],
    'window': n,
    'panjang_input': len(window['input_ids']),
    'token_referensi': len(reference),
    'token_jawaban': len(answer),
} for n, (window, reference, answer) in enumerate(zip(
    example_windows, example_detail['reference_positions'], example_detail['answer_positions'],
), 1)]))

# %% bge_execute
bge = bge_features(bundle, cfg, output)
coverage = json.loads((output / 'bge_encoding.json').read_text())
display(pd.DataFrame([{
    'encoder_frozen': cfg['bge']['checkpoint'],
    'jumlah_vektor': bge.shape[0], 'dimensi': bge.shape[1],
    'cache_dipakai_ulang': coverage['reused'],
    'token_jawaban_tidak_tercakup': coverage['answer_tokens_uncovered'],
}]))

# %% architecture
display(pd.DataFrame([
    ('RoBERTa', '768 per token', cfg['s3_encoder_lr']),
    ('Scalar Mix 4 + gamma', '768 per token', cfg['scalar_mix_lr']),
    ('BiGRU + co-attention', 2 * cfg['s3_sequence_hidden'], cfg['s3_layers_lr']),
    ('Masked mean + proyeksi RoBERTa', 768, cfg['s3_layers_lr']),
    ('BGE-M3 frozen', 1024, None),
    ('Proyeksi BGE', 256, cfg['projection_lr']),
    ('Concat → MLP', f'1024 → 256 → {len(TRAITS)}', cfg['s3_mlp_lr']),
], columns=['bagian', 'dimensi', 'learning_rate']))
print('Urutan empat keluaran MLP:', ', '.join(name for name, *_ in TRAITS))

# %% metrics
def metrics(teacher, prediction):
    """RMSE dalam poin asli per trait, ditambah makro RMSE ternormalisasi untuk seleksi."""
    per_trait = evaluate(teacher, prediction, RUBRIC)
    macro = float(np.mean([
        per_trait[name]['RMSE'] / span
        for (name, _, _, _), span in zip(TRAITS, SCORE_SPAN)
    ]))
    return {'macro_nRMSE': macro, 'traits': per_trait}


def predictions_array(values, indices):
    array = np.asarray(values, dtype=float)
    if array.shape != (len(indices), len(TRAITS)) or not np.isfinite(array).all():
        raise ValueError('Prediksi harus finite dan berukuran [jumlah jawaban, 4].')
    return array

# %% cv_definition
def run_cv(bundle, manifest, bge, cfg, output):
    """Validasi silang satu resep E43 pada development."""
    results = []
    for fold in manifest['folds']:
        number, validation = fold['fold'], fold['validation']
        path = output / f'fold_{number}.json'
        identity = {'split': manifest['split_sha256'], 'fold': number, 'validation': validation}
        if path.exists():
            result = json.loads(path.read_text())
            if result['identity'] != identity:
                raise ValueError(f'Cache fold memakai split berbeda: {path}')
            print(f'E43 · fold {number}: hasil selesai dipakai ulang.', flush=True)
        else:
            print(f'E43 · fold {number}/{cfg["folds"]}', flush=True)
            model, engine, info = fit('S3', fold['train'], validation, bundle, bge,
                                      cfg, cfg['epochs'], cfg['seed'] + number)
            prediction = predictions_array(engine.predict(validation), validation)
            result = {'identity': identity, 'prediction': prediction.tolist(), 'training': info}
            dump(path, result)
            del model, engine
            release()
        prediction = predictions_array(result['prediction'], validation)
        evaluation = metrics(bundle.y[validation], prediction)
        if not math.isclose(evaluation['macro_nRMSE'],
                            result['training']['best_validation_macro_nrmse'], abs_tol=1e-6):
            raise ValueError(f'Prediksi dan epoch terpilih tidak cocok: {path}')
        results.append({**result, 'metrics': evaluation})
    indices = [i for result in results for i in result['identity']['validation']]
    if sorted(indices) != sorted(manifest['development']):
        raise ValueError('Setiap jawaban development harus memiliki tepat satu prediksi OOF.')
    return {
        'eksperimen': 'E43', 'mean_cv_macro_nRMSE': float(np.mean([
            r['metrics']['macro_nRMSE'] for r in results
        ])),
        'epoch_refit': math.ceil(np.median([r['training']['best_epoch'] for r in results])),
    }, results


# %% final_model
def refit_checkpoint(epochs, bundle, manifest, bge, cfg, output):
    """Latih pada development dan simpan checkpoint sebelum cell test dijalankan."""
    identity = {'refit_epochs': epochs, 'split': manifest['split_sha256']}
    model_path = output / 'final.pt'
    refit_path = output / 'refit_result.json'
    test_path = output / 'test_result.json'
    # Format lama menyimpan informasi refit bersama hasil test.
    for saved in (refit_path, test_path):
        if saved.is_file():
            result = json.loads(saved.read_text())
            if (result['identity'] != identity or not model_path.is_file()
                or digest(model_path) != result['model_sha256']):
                raise ValueError('Metadata refit dan checkpoint final berbeda.')
            print('Checkpoint refit yang sudah selesai dipakai ulang.', flush=True)
            return result

    print(f'Refit E43 · {epochs} epoch pada seluruh development.', flush=True)
    model, engine, info = fit('S3', manifest['development'], None, bundle, bge, cfg,
                              epochs, cfg['seed'])
    pending = model_path.with_suffix('.pending.pt')
    save_model(pending, model, cfg, bundle, manifest, info)
    pending.replace(model_path)
    result = {'identity': identity, 'training': info, 'model_sha256': digest(model_path)}
    dump(refit_path, result)
    del model, engine
    release()
    return result


# %% test_definition
def evaluate_test(refit_result, bundle, manifest, bge, cfg, output):
    """Baca checkpoint refit dan hitung satu prediksi untuk setiap jawaban test."""
    complete = output / 'test_result.json'
    model_path = output / 'final.pt'
    if digest(model_path) != refit_result['model_sha256']:
        raise ValueError('Checkpoint yang akan dievaluasi berbeda dari checkpoint refit.')
    test = manifest['test']
    if complete.exists():
        result = json.loads(complete.read_text())
        if (result['identity'] != refit_result['identity']
            or result['model_sha256'] != refit_result['model_sha256']):
            raise ValueError('Hasil test dan checkpoint final berbeda.')
        print('Hasil test final yang sudah selesai dipakai ulang.', flush=True)
        return predictions_array(result['prediction'], test)

    payload = torch.load(model_path, map_location='cpu', weights_only=True)
    device = device_name(cfg)
    model = make_model('S3', cfg)
    model.load_state_dict(payload['state'], strict=True)
    del payload
    model.to(device)
    engine = Engine(model, bundle.texts, bge, cfg, device, contexts=contexts_from_bundle(bundle))
    prediction = predictions_array(engine.predict(test), test)
    dump(complete, {**refit_result, 'prediction': prediction.tolist()})
    del model, engine
    release()
    return prediction

# %% report
def trait_rows(values, **keys):
    return [{**keys, 'trait': name, 'macro_nRMSE': values['macro_nRMSE'],
             'RMSE': stats['RMSE'], 'Pearson': stats['Pearson'],
             'QWK': stats['QWK'], 'Gwet AC2': stats['AC2'],
             'proporsi_di_luar_skala': stats['out_of_range_fraction'],
             'alasan_metrik_kosong': stats['undefined_reasons']}
            for name, stats in values['traits'].items()]


def prediction_rows(bundle, indices, prediction, **keys):
    scores = predictions_array(prediction, indices)
    categories = np.column_stack([
        ordinal_prediction(scores[:, i], trait) for i, trait in enumerate(RUBRIC['traits'])
    ])
    rows = []
    for j, i in enumerate(indices):
        source = bundle.rows[i]
        row = {**keys, 'id_jawaban': source['row_id'], 'baris_excel': source['source_row'],
               'no_soal': source['response_number'], 'kelompok': int(bundle.groups[i])}
        for col, (name, _, _, _) in enumerate(TRAITS):
            actual, raw = bundle.y[i, col], scores[j, col]
            row.update({f'{name}_label': int(actual), f'{name}_prediksi_mentah': float(raw),
                        f'{name}_prediksi_kategori': int(categories[j, col]),
                        f'{name}_error_abs': float(abs(raw - actual))})
        rows.append(row)
    return rows


def history_rows(info, **keys):
    return [{**keys, **{k: v for k, v in row.items() if k != 'validation'},
             'validation_macro_nRMSE': row.get('validation', {}).get('macro_nRMSE')}
            for row in info['history']]


def export_results(path, bundle, manifest, cfg, cv_summary, folds, final_info, prediction):
    cv_fold, oof_rows, history, mixture, indices, predictions = [], [], [], [], [], []
    for result in folds:
        fold = result['identity']['fold']
        valid = result['identity']['validation']
        info = result['training']
        cv_fold.extend(trait_rows(result['metrics'], fold=fold,
                                  epoch_terpilih=info['best_epoch'], n=len(valid)))
        oof_rows.extend(prediction_rows(bundle, valid, result['prediction'], fold=fold))
        history.extend(history_rows(info, fold=fold))
        mixture.extend({'fold': fold, **r} for r in info['scalar_mix_weights'])
        indices.extend(valid)
        predictions.extend(result['prediction'])
    cv_oof = trait_rows(metrics(bundle.y[indices], predictions), n=len(indices))

    test = manifest['test']
    test_metrics = trait_rows(metrics(bundle.y[test], prediction), n=len(test))
    history.extend(history_rows(final_info, fold='refit'))
    mixture.extend({'fold': 'refit', **r} for r in final_info['scalar_mix_weights'])
    return write_excel(path, {
        'Ringkasan CV': pd.DataFrame([cv_summary]),
        'CV per Fold dan Trait': pd.DataFrame(cv_fold),
        'CV OOF per Trait': pd.DataFrame(cv_oof),
        'Prediksi OOF': pd.DataFrame(oof_rows),
        'Test per Trait': pd.DataFrame(test_metrics),
        'Prediksi Test': pd.DataFrame(prediction_rows(bundle, test, prediction)),
        'Riwayat Training': pd.DataFrame(history),
        'Scalar Mix': pd.DataFrame(mixture),
    }, {
        **bundle.summary, 'eksperimen': 'E43', 'dataset_version': cfg['dataset_version'],
        'development_rows': len(manifest['development']), 'test_rows': len(test),
        'split_sha256': manifest['split_sha256'], 'notebook_code_sha256': NOTEBOOK_CODE_SHA256,
        'training_code_sha256': TRAINING_CODE_SHA256,
        'test_dipakai_untuk_seleksi_epoch': False, 'final_config': cfg,
        'kriteria_early_stopping': 'Rata-rata empat RMSE per trait setelah dibagi lebar skala trait.',
    })

# %% cv_execute
cv_summary, folds = run_cv(bundle, manifest, bge, cfg, output)
display(pd.DataFrame([{
    'fold': result['identity']['fold'],
    'epoch_terpilih': result['training']['best_epoch'],
    'macro_nRMSE_validasi': result['metrics']['macro_nRMSE'],
} for result in folds]))
display(pd.DataFrame([cv_summary]))

# %% refit_execute
refit_result = refit_checkpoint(cv_summary['epoch_refit'], bundle, manifest, bge, cfg, output)
final_info = refit_result['training']
display(pd.DataFrame(final_info['scalar_mix_weights']))
print(f'Checkpoint final: {output / "final.pt"}')

# %% test_execute
test_prediction = evaluate_test(refit_result, bundle, manifest, bge, cfg, output)
test_metrics = metrics(bundle.y[manifest['test']], test_prediction)
display(pd.DataFrame(trait_rows(test_metrics)))
print(f"Jawaban test dievaluasi: {len(test_prediction)}")

# %% export_execute
report_path = output / 'Hasil_AES_E43_S3_MultiTrait.xlsx'
sheets = export_results(report_path, bundle, manifest, cfg, cv_summary, folds, final_info, test_prediction)
display(pd.DataFrame.from_dict(sheets, orient='index').rename_axis('sheet'))
print(f'Excel hasil: {report_path}')

# %% download
from google.colab import files
files.download(str(report_path))
