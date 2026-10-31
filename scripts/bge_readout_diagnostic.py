"""Runtime embedded by build_bge_readout_notebook.py; no encoder training."""
import copy
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

CONDITIONS = ('MLP_ASLI', 'MLP_TARGET_CENTER', 'MLP_STANDARD_SCALER', 'RIDGE')
DESCRIPTIONS = {
    'MLP_ASLI': 'BGE L2 → Linear1024/256 → GELU → dropout → Linear256/1',
    'MLP_TARGET_CENTER': 'Head sama; target dikurangi rerata train fold, ditambah kembali saat prediksi',
    'MLP_STANDARD_SCALER': 'Head sama; StandardScaler per komponen dipasang hanya pada train fold',
    'RIDGE': 'BGE L2 yang sama; Ridge alpha dipilih melalui inner CV berkelompok',
}
REPORT_BOOKS = ('01_Data_dan_Split.xlsx', '02_Vektor_BGE_dan_Transformasi.xlsx',
                '03_Prediksi_OOF.xlsx', '04_Evaluasi_dan_Duel.xlsx', '05_Riwayat_Pelatihan.xlsx')


def readout_config(values):
    result = dict(values)
    if not result['seeds'] or len(set(result['seeds'])) != len(result['seeds']):
        raise ValueError('SEEDS harus unik dan tidak kosong.')
    if any(type(s) is not int or s < 0 for s in result['seeds']):
        raise ValueError('Seed harus integer nonnegatif.')
    if not result['alphas'] or any(not math.isfinite(a) or a <= 0 for a in result['alphas']):
        raise ValueError('Alpha Ridge harus positif dan finite.')
    if result['primary_tolerance'] not in (1, 2, 5):
        raise ValueError('Toleransi utama harus 1, 2, atau 5 poin.')
    if not math.isfinite(result['lr']) or result['lr'] <= 0:
        raise ValueError('LR head tidak sah.')
    return result


def prepare_transform(X, y, training, condition):
    """No validation/test statistics enter either transformation."""
    ids = np.asarray(training, dtype=int)
    scaler = StandardScaler().fit(X[ids]) if condition == 'MLP_STANDARD_SCALER' else None
    offset = float(np.mean(y[ids] / 100.)) if condition == 'MLP_TARGET_CENTER' else 0.
    return scaler, offset


def transform_features(X, scaler):
    values = np.asarray(X, dtype=np.float32)
    return values if scaler is None else scaler.transform(values).astype(np.float32)


def readout_metrics(y, prediction, cfg):
    result = metrics(np.asarray(y), np.asarray(prediction), cfg)
    errors = np.asarray(prediction) - np.asarray(y)
    result.update(MAE=float(np.abs(errors).mean()), bias=float(errors.mean()),
                  max_absolute_error=float(np.abs(errors).max()), n=len(errors),
                  rounded_exact=int(np.sum(np.ceil(np.clip(prediction, 0, 100) - .5) == y)))
    for threshold in (1, 2, 5, 10):
        result['within' + str(threshold)] = int(np.sum(np.abs(errors) <= threshold))
    return result


def metric_row(condition, seed, fold, y, p, cfg, **extra):
    m = readout_metrics(y, p, cfg)
    reasons = m.pop('undefined_reasons')
    return dict(condition=condition, seed=seed, fold=fold, **m,
                undefined_reasons=json.dumps(reasons, ensure_ascii=False), **extra)


def make_readout_head(seed, dropout):
    # Same initialization across conditions for the same seed/fold.
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(1024, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 1))


def read_progress(path, contract):
    if not path.exists():
        return None
    index = json.loads(path.read_text())
    if index['contract'] != contract:
        raise ValueError('Checkpoint berasal dari data/fitur/konfigurasi/kode yang berbeda.')
    for item in index['snapshots']:
        if Path(item['file']).name != item['file']:
            raise ValueError('Nama checkpoint tidak sah.')
        file = path.parent / item['file']
        if file.is_file() and digest(file) == item['sha256']:
            return torch.load(file, map_location='cpu', weights_only=True)
    raise RuntimeError('Kedua checkpoint lokal tidak utuh. Jangan menimpa hasil lama.')


def write_progress(path, payload, contract):
    path.parent.mkdir(parents=True, exist_ok=True)
    index = json.loads(path.read_text()) if path.exists() else {'snapshots': []}
    slot = 1 - index.get('slot', 1)
    file = path.with_name(path.stem + '_slot' + str(slot) + '.pt')
    temp = file.with_suffix('.tmp'); torch.save(payload, temp); temp.replace(file)
    snapshots = [dict(file=file.name, sha256=digest(file))]
    snapshots += [v for v in index['snapshots'] if v['file'] != file.name][:1]
    dump(path, dict(contract=contract, slot=slot, snapshots=snapshots))


def fit_readout(condition, X, y, training, validation, cfg, diagnostic, seed, path, contract):
    """CPU float32 head, identical optimization budget and stopping rule."""
    if condition not in CONDITIONS[:3]:
        raise ValueError('fit_readout khusus kondisi MLP.')
    tr = np.asarray(training, dtype=int); va = np.asarray(validation, dtype=int)
    if not len(tr) or not len(va) or set(tr) & set(va):
        raise ValueError('Train/validation harus terpisah dan tidak kosong.')
    scaler, offset = prepare_transform(X, y, tr, condition)
    tx = torch.from_numpy(transform_features(X, scaler))
    targets = torch.as_tensor(y / 100. - offset, dtype=torch.float32).reshape(-1, 1)
    head = make_readout_head(seed, cfg['dropout'])
    initial_hash = hashlib.sha256(b''.join(t.detach().numpy().tobytes() for t in head.parameters())).hexdigest()
    optimizer = torch.optim.AdamW(head.parameters(), lr=diagnostic['lr'], weight_decay=cfg['weight_decay'])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(tr.tolist(), batch_size=cfg['effective_batch'], shuffle=True, generator=generator)
    total = len(loader) * cfg['epochs']
    warmup = min(int(cfg['warmup_ratio'] * total), max(0, total - 1))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total)
    start, best, best_epoch, stale, history, best_state = 1, float('inf'), 0, 0, [], None
    saved = read_progress(path, contract)
    if saved:
        if saved['complete']:
            return saved
        head.load_state_dict(saved['state']); optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']); generator.set_state(saved['generator'])
        torch.set_rng_state(saved['torch_rng'])
        start, best, best_epoch, stale = saved['epoch'] + 1, saved['best'], saved['best_epoch'], saved['stale']
        history, best_state = saved['history'], saved['best_state']
    for epoch in range(start, cfg['epochs'] + 1):
        head.train(); loss_sum = 0.; steps = 0
        for effective in loader:
            ids = effective.tolist(); optimizer.zero_grad(set_to_none=True)
            for begin in range(0, len(ids), cfg['micro_batch']):
                batch = ids[begin:begin + cfg['micro_batch']]
                loss = (head(tx[batch]) - targets[batch]).square().mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('Loss nonfinite; tidak dianggap hasil berhasil.')
                (loss * (len(batch) / len(ids))).backward()
                loss_sum += loss.item() * len(batch)
            torch.nn.utils.clip_grad_norm_(head.parameters(), cfg['gradient_clip'])
            optimizer.step(); scheduler.step(); steps += 1
        head.eval()
        with torch.no_grad():
            prediction = (head(tx[va]).reshape(-1).numpy() + offset) * 100.
        score = readout_metrics(y[va], prediction, cfg)['RMSE']
        if score < best:
            best, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
        history.append(dict(epoch=epoch, train_mse_scaled=loss_sum / len(tr), validation_RMSE=score,
                            best_epoch=best_epoch, lr_end=optimizer.param_groups[0]['lr'],
                            optimizer_steps=steps, target_offset=offset))
        complete = stale >= cfg['early_stopping_patience'] or epoch == cfg['epochs']
        saved = dict(complete=complete, epoch=epoch, state=head.state_dict(), optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), generator=generator.get_state(), torch_rng=torch.get_rng_state(),
                     best=best, best_epoch=best_epoch, stale=stale, history=history, best_state=best_state,
                     initial_parameter_sha256=initial_hash, target_offset=offset,
                     scaler_mean=None if scaler is None else torch.from_numpy(scaler.mean_),
                     scaler_scale=None if scaler is None else torch.from_numpy(scaler.scale_))
        if complete:
            head.load_state_dict(best_state)
            with torch.no_grad():
                saved['prediction'] = torch.from_numpy(((head(tx[va]).reshape(-1).numpy() + offset) * 100.).copy())
            saved['state'] = best_state
        write_progress(path, saved, contract)
        print(f'{condition} | seed {seed} | epoch {epoch}/{cfg["epochs"]} | RMSE validasi {score:.4f} | terbaik {best:.4f}', flush=True)
        if complete:
            break
    return saved


def ridge_fold(X, y, training, validation, folds, alphas):
    """Alpha never selected on the outer validation or historical test."""
    tr = np.asarray(training, dtype=int); va = np.asarray(validation, dtype=int)
    if set(tr) & set(va):
        raise ValueError('Outer train/validation bertumpang tindih.')
    grid = []
    for alpha in alphas:
        truth, predictions = [], []
        for fold in folds:
            iv = np.intersect1d(tr, fold['validation'])
            it = np.setdiff1d(tr, iv)
            if not len(iv):
                continue
            if not len(it):
                raise ValueError('Inner CV tidak cukup blok.')
            model = Ridge(alpha=alpha, fit_intercept=True, solver='cholesky').fit(X[it], y[it])
            truth.extend(y[iv]); predictions.extend(model.predict(X[iv]))
        if len(truth) != len(tr):
            raise ValueError('Inner OOF tidak meliputi seluruh outer training tepat sekali.')
        score = float(np.sqrt(np.mean((np.asarray(truth) - np.asarray(predictions)) ** 2)))
        grid.append(dict(alpha=alpha, inner_RMSE=score, inner_rows=len(truth)))
    chosen = min(grid, key=lambda row: row['inner_RMSE'])['alpha']
    model = Ridge(alpha=chosen, fit_intercept=True, solver='cholesky').fit(X[tr], y[tr])
    return model.predict(X[va]), chosen, grid


def run_readout_cv(bundle, features, manifest, cfg, diagnostic, folder):
    X = np.asarray(features, dtype=np.float32); y = np.asarray(bundle.y, dtype=np.float64)
    if X.shape != (len(y), 1024) or not np.isfinite(X).all():
        raise ValueError('Vektor BGE harus [jumlah_jawaban,1024] dan finite.')
    if not np.allclose(np.linalg.norm(X, axis=1), 1., atol=1e-4):
        raise ValueError('Kontrol memerlukan BGE L2 asli.')
    dev, test = set(manifest['development']), set(manifest['test'])
    feature_hash = hashlib.sha256(X.tobytes()).hexdigest()
    semantic = dict(source=bundle.source_sha256, split=manifest['split_sha256'], features=feature_hash,
                    config=cfg, diagnostic=diagnostic, code=NOTEBOOK_CODE_SHA256,
                    torch=str(torch.__version__), precision='cpu_float32', sklearn=version('scikit-learn'))
    key = hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()[:20]
    run = Path(folder) / key; run.mkdir(parents=True, exist_ok=True); dump(run / 'contract.json', semantic)
    evaluation, histories, predictions, transforms, grids = [], [], [], [], []
    for condition in CONDITIONS:
        seeds = [None] if condition == 'RIDGE' else diagnostic['seeds']
        for seed in seeds:
            oof = np.full(len(y), np.nan)
            for fold in manifest['folds']:
                tr, va = fold['train'], fold['validation']; number = fold['fold']
                if set(tr) & test or set(va) & test or set(tr) | set(va) != dev:
                    raise ValueError('Test memasuki proses CV atau partisi tidak lengkap.')
                if set(bundle.groups[tr]) & set(bundle.groups[va]):
                    raise ValueError('Kelompok peserta menyeberang fold.')
                if condition == 'RIDGE':
                    p, alpha, search = ridge_fold(X, y, tr, va, manifest['folds'], diagnostic['alphas'])
                    grids.extend(dict(fold=number, **row) for row in search)
                    transforms.append(dict(condition=condition, seed=seed, fold=number, target_offset=0., chosen_alpha=alpha))
                    best_epoch = None
                else:
                    contract = dict(key=key, condition=condition, seed=seed, fold=number)
                    state = fit_readout(condition, X, y, tr, va, cfg, diagnostic, seed + number,
                                        run / condition / str(seed) / f'fold_{number}.json', contract)
                    p = state['prediction'].numpy(); best_epoch = state['best_epoch']
                    histories.extend(dict(condition=condition, seed=seed, fold=number, **row) for row in state['history'])
                    transforms.append(dict(condition=condition, seed=seed, fold=number, target_offset=state['target_offset'],
                                           initial_parameter_sha256=state['initial_parameter_sha256'], best_epoch=best_epoch))
                    if state['scaler_mean'] is not None:
                        transforms.extend(dict(condition=condition, seed=seed, fold=number, component=i + 1,
                                               mean=float(m), scale=float(s)) for i, (m, s) in enumerate(zip(state['scaler_mean'], state['scaler_scale'])))
                if np.isfinite(oof[va]).any():
                    raise ValueError('Jawaban validasi diprediksi lebih dari sekali.')
                oof[va] = p
                evaluation.append(metric_row(condition, seed, number, y[va], p, cfg, best_epoch=best_epoch))
                predictions.extend(dict(condition=condition, seed=seed, fold=number, row_id=bundle.rows[i]['row_id'],
                                        teacher=float(y[i]), prediction=float(v), error=float(v - y[i]),
                                        absolute_error=float(abs(v - y[i]))) for i, v in zip(va, p))
            dv = np.asarray(sorted(dev))
            if not np.isfinite(oof[dv]).all() or np.isfinite(oof[list(test)]).any():
                raise ValueError('OOF tidak lengkap atau mengandung prediksi test.')
            evaluation.append(metric_row(condition, seed, 'POOLED_OOF', y[dv], oof[dv], cfg))
    result = dict(run_folder=str(run), feature_sha256=feature_hash, evaluation=evaluation, history=histories,
                  predictions=predictions, transforms=transforms, ridge_grid=grids,
                  test_evaluated=False, BGE_encoder_trained=False, head_device='cpu', head_precision='float32')
    dump(run / 'readout_result.json', result)
    return result


def summarize_readouts(result, tolerance):
    scores = pd.DataFrame([r for r in result['evaluation'] if r['fold'] == 'POOLED_OOF'])
    rows = []
    for condition in CONDITIONS:
        group = scores[scores.condition == condition]; n = len(group)
        rows.append(dict(condition=condition, repetitions=n, n_per_repetition=int(group.n.iloc[0]),
                         RMSE_mean=float(group.RMSE.mean()), RMSE_seed_sd=float(group.RMSE.std(ddof=1)) if n > 1 else None,
                         MAE_mean=float(group.MAE.mean()), within1_mean=float(group.within1.mean()),
                         within2_mean=float(group.within2.mean()), within5_mean=float(group.within5.mean()),
                         primary_within_mean=float(group['within' + str(tolerance)].mean())))
    for row in rows:
        row['pareto_RMSE_and_tolerance'] = not any(
            other['RMSE_mean'] <= row['RMSE_mean'] and other['primary_within_mean'] >= row['primary_within_mean']
            and (other['RMSE_mean'] < row['RMSE_mean'] or other['primary_within_mean'] > row['primary_within_mean'])
            for other in rows)
    return pd.DataFrame(rows)


def readout_duels(result, diagnostic):
    rows = []; data = pd.DataFrame(result['predictions'])
    for seed in diagnostic['seeds']:
        selected = data[(data.seed == seed) | (data.condition == 'RIDGE')]
        for left, right in itertools.combinations(CONDITIONS, 2):
            a = selected[selected.condition == left].set_index('row_id')
            b = selected[selected.condition == right].set_index('row_id')
            if set(a.index) != set(b.index):
                raise ValueError('Duel tidak memakai jawaban yang sama.')
            diff = a.absolute_error - b.loc[a.index].absolute_error
            rows.append(dict(seed=seed, left=left, right=right, n=len(diff),
                             left_closer=int((diff < -1e-5).sum()), right_closer=int((diff > 1e-5).sum()),
                             ties=int((diff.abs() <= 1e-5).sum())))
    return pd.DataFrame(rows)


def write_readout_workbook(path, sheets):
    # Strings are literal text, never spreadsheet formulas from uploaded essays.
    book = Workbook(write_only=True)
    for title, data in sheets.items():
        frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        sheet = book.create_sheet(title[:31])
        sheet.append([str(v) for v in frame.columns])
        for record in frame.itertuples(index=False, name=None):
            cells = []
            for value in record:
                if isinstance(value, (list, dict, tuple)):
                    value = json.dumps(value, ensure_ascii=False)
                if isinstance(value, np.generic):
                    value = value.item()
                if isinstance(value, float) and not math.isfinite(value):
                    value = None
                cell = WriteOnlyCell(sheet, value=value)
                if isinstance(value, str):
                    cell.data_type = 's'
                cells.append(cell)
            sheet.append(cells)
    book.save(path)


def export_readout(bundle, features, manifest, cfg, diagnostic, result):
    run = Path(result['run_folder']); destination = run / 'hasil_ringkas'; destination.mkdir(exist_ok=True)
    roles = {i: 'development' for i in manifest['development']}; roles.update({i: 'test_locked' for i in manifest['test']})
    records = pd.DataFrame([dict(row_id=r['row_id'], soal=r['question'], jawaban_referensi=r['reference'],
                                 response_text=r['text'], rater=r['score'], connected_group=int(bundle.groups[i]),
                                 role=roles[i]) for i, r in enumerate(bundle.rows)])
    split = pd.DataFrame([dict(fold=f['fold'], role=role, row_id=bundle.rows[i]['row_id'])
                          for f in manifest['folds'] for role in ('train', 'validation') for i in f[role]])
    guide = pd.DataFrame([dict(Keterangan=k, Nilai=v) for k, v in dict(
        scope='DIAGNOSTIK BGE FROZEN; empat kondisi readout; bukan S1/S2/S3 lengkap',
        target='rater 0–100; tiga input; data guru asli',
        test='Tidak dievaluasi; seleksi hanya development/OOF',
        seed='Ridge deterministik dihitung sekali; mean seed bukan ensemble prediksi',
        comparison='Target-centering dan StandardScaler terpisah; statistik train fold saja',
        precision='Hitungan toleransi dan RMSE dilaporkan terpisah; Pareto bukan klaim satu pemenang',
        training='MSE target0–1; CPU float32; BGE tidak mendapat optimizer',
        privacy='Seluruh Excel/cache/checkpoint privat; jangan unggah ke repo publik').items()])
    feature_frame = pd.DataFrame(np.asarray(features), columns=[f'Komponen {i+1}' for i in range(1024)])
    feature_frame.insert(0, 'row_id', records.row_id)
    summary = summarize_readouts(result, diagnostic['primary_tolerance'])
    write_readout_workbook(destination / REPORT_BOOKS[0], dict(Panduan=guide, Data=records, Lima_Fold=split))
    write_readout_workbook(destination / REPORT_BOOKS[1], dict(Panduan=guide, BGE_Frozen=feature_frame,
                                                            Transformasi_Train_Fold=pd.DataFrame(result['transforms'])))
    write_readout_workbook(destination / REPORT_BOOKS[2], dict(Panduan=guide, Prediksi_Development_OOF=pd.DataFrame(result['predictions'])))
    write_readout_workbook(destination / REPORT_BOOKS[3], dict(Panduan=guide, Ringkasan=summary,
        Fold_dan_Pooled=pd.DataFrame(result['evaluation']), Duel=readout_duels(result, diagnostic), Ridge_Inner_CV=pd.DataFrame(result['ridge_grid'])))
    parameters = dict(**cfg, **diagnostic, code_sha256=NOTEBOOK_CODE_SHA256, feature_sha256=result['feature_sha256'])
    write_readout_workbook(destination / REPORT_BOOKS[4], dict(Panduan=guide,
        Pengaturan=pd.DataFrame([dict(parameter=k, value=v) for k,v in parameters.items()]),
        Riwayat_MLP=pd.DataFrame(result['history']),
        Versi=pd.DataFrame([dict(library=k, version=version(k)) for k in ('torch','transformers','numpy','pandas','scikit-learn','openpyxl')])))
    archive = run / 'Hasil_AES_300_BGE_Frozen_Readout.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in REPORT_BOOKS:
            z.write(destination / name, name)
        z.writestr('manifest.json', json.dumps(dict(scope=guide.iloc[0].Nilai, dataset_sha256=bundle.source_sha256,
            code_sha256=NOTEBOOK_CODE_SHA256, feature_sha256=result['feature_sha256'], test_evaluated=False,
            books={n:digest(destination / n) for n in REPORT_BOOKS}), ensure_ascii=False, indent=2))
    return dict(archive=str(archive), books=list(REPORT_BOOKS), run_folder=str(run), test_evaluated=False)
