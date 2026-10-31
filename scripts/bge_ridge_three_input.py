"""Embedded runtime for the standalone three-input frozen-BGE Ridge experiment.

Imports and the unchanged dataset/encoder/metric helpers are embedded by the builder.
This module contains no dataset, scores, trained weights, or research results.
"""
from sklearn.linear_model import Ridge

RIDGE_BOOKS = ('01_Data_dan_Split.xlsx', '02_Vektor_BGE_dan_Model.xlsx',
               '03_Prediksi_OOF_dan_Test.xlsx', '04_Evaluasi_dan_Seleksi.xlsx',
               '05_Pengaturan_dan_Referensi.xlsx')
RIDGE_PAPERS = [
    dict(method='BGE-M3', title='BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation',
         authors='Chen et al. (2024)', url='https://arxiv.org/abs/2402.03216',
         use='Dasar encoder; bukan bukti alpha terbaik atau jaminan performa AES.'),
    dict(method='Ridge', title='Ridge Regression: Biased Estimation for Nonorthogonal Problems',
         authors='Hoerl & Kennard (1970)', url='https://doi.org/10.1080/00401706.1970.10488634',
         use='Regresi linear dengan penalti L2; alpha tetap harus dipilih pada data development.'),
    dict(method='Seleksi dan evaluasi terpisah', title='On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation',
         authors='Cawley & Talbot (2010)', url='https://www.jmlr.org/papers/v11/cawley10a.html',
         use='Dasar pemisahan seleksi parameter dari evaluasi; nested CV untuk OOF prosedur seleksi.'),
]


def validate_ridge_inputs(bundle, X, manifest, cfg):
    X = np.asarray(X, dtype=np.float64)
    if X.shape != (len(bundle.rows), 1024) or not np.isfinite(X).all():
        raise ValueError('Ridge perlu fitur BGE finite [jumlah jawaban, 1024].')
    if not np.allclose(np.linalg.norm(X, axis=1), 1., atol=1e-4):
        raise ValueError('Fitur harus menggunakan L2 per jawaban yang sama dengan acuan.')
    alphas = cfg['ridge_alphas']
    if not alphas or len(set(alphas)) != len(alphas) or any(
            isinstance(a, bool) or not math.isfinite(a) or a <= 0 for a in alphas):
        raise ValueError('Alpha harus unik, positif, finite dan ditentukan sebelum evaluasi.')
    dev, test = set(manifest['development']), set(manifest['test'])
    if not dev or not test or dev & test or dev | test != set(range(len(X))):
        raise ValueError('Development dan test harus membagi seluruh baris tanpa overlap.')
    if len(dev) != len(manifest['development']) or len(test) != len(manifest['test']):
        raise ValueError('Indeks partisi duplikat.')
    if set(bundle.groups[list(dev)]) & set(bundle.groups[list(test)]):
        raise ValueError('Kelompok menyeberang development dan test.')
    covered = []
    for fold in manifest['folds']:
        tr, va = set(fold['train']), set(fold['validation'])
        if not tr or not va or tr & va or tr | va != dev:
            raise ValueError('Fold harus membagi development; test tidak boleh masuk fold.')
        if len(tr) != len(fold['train']) or len(va) != len(fold['validation']):
            raise ValueError('Indeks fold duplikat.')
        if set(bundle.groups[list(tr)]) & set(bundle.groups[list(va)]):
            raise ValueError('Kelompok menyeberang train dan validation.')
        covered.extend(fold['validation'])
    if len(manifest['folds']) < 3 or len(covered) != len(dev) or set(covered) != dev:
        raise ValueError('Setiap development wajib menjadi validation tepat sekali; minimal3fold.')
    return X


def select_ridge_alpha(X, y, allowed, blocks, alphas, stage):
    """Only labels in allowed participate; blocks are the existing grouped fold blocks."""
    allowed = set(allowed)
    pieces = [sorted(allowed & set(block)) for block in blocks]
    pieces = [piece for piece in pieces if piece]
    flat = [i for piece in pieces for i in piece]
    if len(pieces) < 2 or len(flat) != len(allowed) or set(flat) != allowed:
        raise ValueError('Blok inner/final CV tidak membagi train yang diizinkan tepat sekali.')
    search, details = [], []
    for alpha in alphas:
        squared, count = 0., 0
        for number, va in enumerate(pieces, 1):
            tr = sorted(allowed - set(va))
            model = Ridge(alpha=float(alpha), fit_intercept=True, solver='cholesky')
            model.fit(X[tr], y[tr])
            p = model.predict(X[va])
            sse = float(np.sum((p - y[va]) ** 2))
            squared += sse; count += len(va)
            details.append(dict(stage=stage, alpha=float(alpha), inner_fold=number,
                                train_rows=len(tr), validation_rows=len(va), RMSE=math.sqrt(sse / len(va))))
        search.append(dict(stage=stage, alpha=float(alpha), selection_RMSE=math.sqrt(squared / count),
                           selection_rows=count, selection_folds=len(pieces)))
    # Fixed tie rule: smaller alpha. No test metric is available to this function.
    selected = min(search, key=lambda r: (r['selection_RMSE'], r['alpha']))['alpha']
    for row in search:
        row['selected'] = row['alpha'] == selected
    return selected, search, details


def ridge_metric_row(partition, teacher, prediction, cfg, **extra):
    m = metrics(teacher, prediction, cfg)
    errors = np.asarray(prediction) - np.asarray(teacher)
    absolute = np.abs(errors)
    m.update(n=len(errors), MAE=float(absolute.mean()), bias=float(errors.mean()),
             median_absolute_error=float(np.median(absolute)), p90_absolute_error=float(np.quantile(absolute, .90)),
             p95_absolute_error=float(np.quantile(absolute, .95)), max_absolute_error=float(absolute.max()))
    for threshold in (1, 2, 5, 10):
        m['within' + str(threshold)] = int(np.sum(absolute <= threshold))
    reasons = json.dumps(m.pop('undefined_reasons'), ensure_ascii=False)
    return dict(partition=partition, **extra, **m, undefined_reasons=reasons)


def ridge_prediction_rows(bundle, indices, prediction, partition, fold, alpha):
    rows = []
    for i, p in zip(indices, prediction):
        y = float(bundle.y[i]); error = float(p - y)
        row = dict(row_id=bundle.rows[i]['row_id'], partition=partition, fold=fold,
                   model='BGE_M3_RIDGE_3INPUT', alpha=alpha, teacher=y, prediction=float(p),
                   error=error, absolute_error=abs(error))
        row.update({f'within{t}': abs(error) <= t for t in (1, 2, 5, 10)})
        rows.append(row)
    return rows


def run_ridge_three_input(bundle, features, manifest, cfg, output):
    X = validate_ridge_inputs(bundle, features, manifest, cfg)
    y = np.asarray(bundle.y, dtype=np.float64)
    blocks = [f['validation'] for f in manifest['folds']]
    dev = np.asarray(manifest['development'], dtype=int)
    test = np.asarray(manifest['test'], dtype=int)
    run = Path(output); run.mkdir(parents=True, exist_ok=True)
    evaluation, predictions, searches, details, outer = [], [], [], [], []
    oof = np.full(len(y), np.nan)
    for fold in manifest['folds']:
        tr, va, number = fold['train'], fold['validation'], fold['fold']
        alpha, grid, inner = select_ridge_alpha(X, y, tr, blocks, cfg['ridge_alphas'], f'OUTER_{number}')
        model = Ridge(alpha=alpha, fit_intercept=True, solver='cholesky').fit(X[tr], y[tr])
        p = model.predict(X[va]); oof[va] = p
        predictions.extend(ridge_prediction_rows(bundle, va, p, 'DEVELOPMENT_OOF', number, alpha))
        evaluation.append(ridge_metric_row('DEVELOPMENT_OUTER_FOLD', y[va], p, cfg, fold=number, alpha=alpha))
        evaluation.append(ridge_metric_row('TRAIN_OUTER_OPTIMISTIC', y[tr], model.predict(X[tr]), cfg, fold=number, alpha=alpha))
        outer.append(dict(fold=number, alpha=alpha, train_rows=len(tr), validation_rows=len(va)))
        searches.extend(grid); details.extend(inner)
        print(f'Ridge outer fold{number}: alpha={alpha}; validation={len(va)}', flush=True)
    if not np.isfinite(oof[dev]).all() or np.isfinite(oof[test]).any():
        raise ValueError('OOF development tidak lengkap atau berisi test.')
    evaluation.append(ridge_metric_row('DEVELOPMENT_POOLED_NESTED_OOF', y[dev], oof[dev], cfg, fold='ALL', alpha=None))
    # Final alpha chosen on development only; this score is for selection, not independent OOF performance.
    alpha, grid, inner = select_ridge_alpha(X, y, dev.tolist(), blocks, cfg['ridge_alphas'], 'FINAL_DEVELOPMENT_SELECTION')
    searches.extend(grid); details.extend(inner)
    feature_hash = hashlib.sha256(np.asarray(features, dtype=np.float32).tobytes()).hexdigest()
    membership = dict(development=manifest['development'], test=manifest['test'], folds=manifest['folds'])
    contract = dict(experiment='BGE_M3_RIDGE_3INPUT', dataset_sha256=bundle.source_sha256,
                    code_sha256=NOTEBOOK_CODE_SHA256, feature_sha256=feature_hash,
                    split_sha256=manifest['split_sha256'],
                    membership_sha256=hashlib.sha256(json.dumps(membership, sort_keys=True).encode()).hexdigest(),
                    selected_alpha=alpha, alpha_grid=cfg['ridge_alphas'],
                    selection_partition='development_only', selection_metric='pooled_RMSE',
                    selection_tie_rule='smaller_alpha', fit_intercept=True, solver='cholesky',
                    features='L2 per answer; no StandardScaler', head_dtype='float64',
                    target='rater 0–100, unscaled', test_used_for_selection=False,
                    evaluation_context='Retrospective shared historical holdout; not a new unseen test set.')
    dump(run / 'alpha_locked_before_test.json', contract)
    model = Ridge(alpha=alpha, fit_intercept=True, solver='cholesky').fit(X[dev], y[dev])
    model_path = run / 'ridge_final_model.npz'
    np.savez_compressed(model_path, coef=np.asarray(model.coef_, dtype=np.float64),
                        intercept=np.asarray(model.intercept_, dtype=np.float64), alpha=np.asarray(alpha))
    # Test is evaluated only after the selection contract and final coefficients have been saved.
    p = model.predict(X[test])
    predictions.extend(ridge_prediction_rows(bundle, test, p, 'TEST_HISTORICAL_HOLDOUT', 'FINAL_REFIT', alpha))
    evaluation.append(ridge_metric_row('TEST_HISTORICAL_HOLDOUT', y[test], p, cfg, fold='FINAL_REFIT', alpha=alpha))
    evaluation.append(ridge_metric_row('TRAIN_FINAL_OPTIMISTIC', y[dev], model.predict(X[dev]), cfg, fold='FINAL_REFIT', alpha=alpha))
    evaluation.append(ridge_metric_row('TEST_TRAIN_MEAN_BASELINE', y[test], np.full(len(test), y[dev].mean()), cfg, fold='FINAL_REFIT', alpha=None))
    result = dict(run_folder=str(run), contract=contract, evaluation=evaluation, predictions=predictions,
                  alpha_search=searches, inner_folds=details, outer_folds=outer,
                  test_evaluated=True, BGE_encoder_trained=False,
                  model_sha256=digest(model_path), model_file=model_path.name)
    dump(run / 'ridge_result.json', result)
    return result


def export_ridge_three_input(bundle, X, manifest, cfg, result):
    run = Path(result['run_folder']); destination = run / 'hasil_excel'; destination.mkdir(exist_ok=True)
    roles = {i: 'development' for i in manifest['development']}
    roles.update({i: 'test' for i in manifest['test']})
    records = [dict(row_id=r['row_id'], source_row=r['source_row'], soal=r['question'],
                    jawaban_referensi=r['reference'], response_text=r['text'], rater=float(bundle.y[i]),
                    response_number=r['response_number'], connected_group=int(bundle.groups[i]), role=roles[i])
               for i, r in enumerate(bundle.rows)]
    split = [dict(fold=f['fold'], role=role, row_id=bundle.rows[i]['row_id'])
             for f in manifest['folds'] for role in ('train', 'validation') for i in f[role]]
    guide = [dict(Keterangan=k, Nilai=v) for k, v in dict(
        model='BGE-M3 frozen → mean window CLS → L2 → Ridge; tiga input; satu rater',
        selection='Nested5fold OOF development; final alpha dari5fold development; final fit seluruh development',
        comparison='Bandingkan TEST dengan TEST S2; OOF dengan OOF yang protokolnya sepadan. Jangan campur OOF dan test.',
        historical='Holdout lama sudah dilihat pada eksperimen lain: perbandingan retrospektif, bukan test baru.',
        metrics='RMSE/MAE dan toleransi memakai prediksi kontinu tanpa clipping; skor agreement memakai salinan kategori.',
        training='Ridge bukan MLP; tanpa epoch/LR/dropout/early stopping; encoder tidak dilatih.',
        controls='E22 resep token context_pair berbeda; E29/E31 LR MLP5e-6. Catat kontrol S2 yang prediksinya identik.',
        join='Cocokkan ketiga teks, label, keanggotaan test dan fold; row_id saja tidak cukup.',
        privacy='Seluruh Excel dan model privat; jangan masukkan data siswa ke repository publik.').items()]
    frame = pd.DataFrame(X, columns=[f'Komponen {i+1}' for i in range(X.shape[1])])
    frame.insert(0, 'row_id', [r['row_id'] for r in bundle.rows])
    with np.load(run / result['model_file'], allow_pickle=False) as model:
        weights = pd.DataFrame(dict(component=np.arange(1, X.shape[1]+1), coefficient=model['coef']))
        intercept = float(model['intercept'])
    enc_path = run / 'bge_encoding.json'
    encoding = json.loads(enc_path.read_text()) if enc_path.is_file() else {'proof': 'not provided; technical fixture only'}
    write_readout_workbook(destination / RIDGE_BOOKS[0], dict(Panduan=guide, Data=records, Lima_Fold=split))
    write_readout_workbook(destination / RIDGE_BOOKS[1], dict(Panduan=guide, BGE_Frozen=frame, Ridge_Coef=weights,
        Model=[dict(intercept=intercept, alpha=result['contract']['selected_alpha'])],
        Bukti_Encoding=[dict(parameter=k, value=v) for k, v in encoding.items()]))
    write_readout_workbook(destination / RIDGE_BOOKS[2], dict(Panduan=guide,
        Prediksi_Development_OOF=[r for r in result['predictions'] if r['partition']=='DEVELOPMENT_OOF'],
        Prediksi_Test=[r for r in result['predictions'] if r['partition']=='TEST_HISTORICAL_HOLDOUT']))
    write_readout_workbook(destination / RIDGE_BOOKS[3], dict(Panduan=guide, Evaluasi=result['evaluation'],
        Alpha_Seleksi=result['alpha_search'], Inner_Fold=result['inner_folds'], Outer_Fold=result['outer_folds']))
    settings = dict(cfg, **result['contract'])
    write_readout_workbook(destination / RIDGE_BOOKS[4], dict(Panduan=guide,
        Pengaturan=[dict(parameter=k, value=v) for k, v in settings.items()], Referensi=RIDGE_PAPERS,
        Versi=[dict(library=k, version=version(k)) for k in ('torch','transformers','numpy','pandas','scikit-learn','openpyxl')]))
    dump(run / 'split_manifest.json', manifest)
    members = {name: destination / name for name in RIDGE_BOOKS}
    for name in ('ridge_final_model.npz', 'split_manifest.json', 'alpha_locked_before_test.json'):
        members[name] = run / name
    archive = run / 'Hasil_AES_300_BGE_M3_Ridge_Tiga_Input.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, path in members.items():
            z.write(path, name)
        z.writestr('manifest.json', json.dumps(dict(**result['contract'], test_evaluated=True,
            BGE_encoder_trained=False, files={name: digest(path) for name, path in members.items()}),
            ensure_ascii=False, indent=2))
    return dict(archive=str(archive), books=list(RIDGE_BOOKS), run_folder=str(run),
                selected_alpha=result['contract']['selected_alpha'], test_evaluated=True)
