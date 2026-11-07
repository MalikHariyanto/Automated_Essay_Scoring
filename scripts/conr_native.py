"""E54 ConR adaptation; embedded visibly in the standalone notebook.

ConR algorithm/source: Keramati, Meng & Evans, ICLR 2024, BorealisAI/ConR.
Adapted loss code: CC BY-NC-SA 4.0 (original: Royal Bank of Canada).
https://github.com/BorealisAI/ConR/blob/main/LICENSE
Changes: same-question groups, point units, stable log-space, empty-anchor masks,
uniform weights, explicit configuration, and batch/resume/export diagnostics.
The surrounding historical AES implementation retains its existing rights.
"""


def validate_conr_config(cfg):
    if cfg['loss_recipe'] != 'mse_same_question_conr_v1':
        raise ValueError('loss_recipe harus mse_same_question_conr_v1 untuk E54.')
    for key in ('conr_weight', 'conr_window_points', 'conr_temperature', 'conr_eta_per_point'):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(key+' harus finite dan tidak negatif.')
    if cfg['conr_window_points'] <= 0 or cfg['conr_temperature'] <= 0:
        raise ValueError('Window dan temperature ConR harus positif.')
    if cfg['conr_window_points'] >= cfg['score_max']-cfg['score_min']:
        raise ValueError('Window ConR harus lebih kecil dari rentang nilai.')
    if type(cfg['conr_recompute']) is not bool:
        raise ValueError('conr_recompute harus boolean.')


def same_question_conr(features, targets, predictions, questions, cfg):
    """Uniform-weight ConR in point units; average over the actual full batch.

    Positives: same question, different answer, |teacher_i-teacher_j| <= w.
    Hard negatives: same question, teacher distance > w, prediction distance <= w.
    Original denominator sum(exp(masked-positive logits)) is retained per group:
    masked entries contribute exp(0)=1. Hard negatives add exp(sim/t+eta*distance).
    Anchors without positives or hard negatives contribute an exact graph-linked 0.
    Cosine similarity is only a TRAINING regularizer, never the predicted score.
    """
    n = len(questions)
    if features.ndim != 2 or features.shape[0] != n or n == 0 or features.shape[1] == 0:
        raise ValueError('Fitur ConR harus [B,D] dan batch tidak kosong.')
    if targets.shape != (n, 1) or predictions.shape != (n, 1):
        raise ValueError('Target/prediksi ConR harus [B,1].')
    if features.device != targets.device or features.device != predictions.device:
        raise ValueError('Fitur/target/prediksi ConR harus pada perangkat yang sama.')
    if any(not torch.isfinite(x).all() for x in (features, targets, predictions)):
        raise ValueError('Fitur, target dan prediksi ConR harus finite.')
    if not all(isinstance(q, str) and q.strip() for q in questions):
        raise ValueError('Soal ConR harus teks yang tidak kosong.')
    counts = dict(conr_answers=n, conr_positive_anchors=0, conr_eligible_anchors=0,
                  conr_active_anchors=0, conr_positive_pairs=0, conr_hard_negative_pairs=0)
    # Labels and discrete hard-negative mining are not differentiable inputs.
    span = cfg['score_max']-cfg['score_min']
    # Normalized FP32 teacher scores can reconstruct 5 points as 5.000004.
    # A machine-precision tolerance preserves the declared inclusive boundary.
    boundary = cfg['conr_window_points']+8*torch.finfo(torch.float32).eps*span
    teacher = targets.detach().float().reshape(-1)*span
    predicted = predictions.detach().float().reshape(-1)*span
    groups = {}
    for i, q in enumerate(questions):
        groups.setdefault(q, []).append(i)
    zero = features.float().sum()*0.
    terms = []
    # Disable outer autocast: logsumexp/normalization and all pair logits use FP32.
    with torch.autocast(device_type=features.device.type, enabled=False):
        for positions in groups.values():
            if len(positions) < 2:
                continue
            ix = torch.tensor(positions, device=features.device)
            z = torch.nn.functional.normalize(features.index_select(0, ix).float(), dim=1)
            label = teacher[ix]; prediction = predicted[ix]
            distance = (label[:, None]-label[None, :]).abs()
            prediction_distance = (prediction[:, None]-prediction[None, :]).abs()
            off_diagonal = ~torch.eye(len(ix), dtype=torch.bool, device=ix.device)
            positive = (distance <= boundary) & off_diagonal
            far = (distance > boundary) & off_diagonal
            negative = far & (prediction_distance <= boundary)
            has_positive = positive.any(1)
            eligible = has_positive & far.any(1)
            active = has_positive & negative.any(1)
            counts['conr_positive_anchors'] += int(has_positive.sum().item())
            counts['conr_eligible_anchors'] += int(eligible.sum().item())
            counts['conr_active_anchors'] += int(active.sum().item())
            counts['conr_positive_pairs'] += int(positive.sum().item())
            counts['conr_hard_negative_pairs'] += int(negative.sum().item())
            logits = (z @ z.T)/cfg['conr_temperature']
            for i in torch.nonzero(active, as_tuple=False).flatten().tolist():
                # Same denominator as source on well-defined single-question cases,
                # but without exp overflow or division by zero for invalid anchors.
                positive_logits = torch.where(positive[i], logits[i], torch.zeros_like(logits[i]))
                negative_logits = logits[i][negative[i]]+cfg['conr_eta_per_point']*distance[i][negative[i]]
                log_denominator = torch.logsumexp(torch.cat((positive_logits, negative_logits)), dim=0)
                terms.append(log_denominator-logits[i][positive[i]].mean())
    loss = torch.stack(terms).sum()/n if terms else zero
    if not torch.isfinite(loss):
        raise RuntimeError('Loss ConR nonfinite; run dihentikan.')
    return loss, counts


def conr_micro_forward(device_marker, indices, engine):
    """Empty device marker lets checkpoint preserve CPU/CUDA dropout RNG."""
    return engine.forward(indices, return_conr_features=True)


def training_batch_loss(engine, batch, bundle, cfg, device, dtype, use_cuda):
    predictions, features = [], []
    marker = torch.empty(0, device=device)
    for start in range(0, len(batch), cfg['micro_batch']):
        ids = batch[start:start+cfg['micro_batch']]
        with torch.autocast(device_type=device, dtype=dtype, enabled=use_cuda):
            if cfg['conr_recompute']:
                p, z = activation_checkpoint(conr_micro_forward, marker, ids, engine,
                    use_reentrant=False, preserve_rng_state=True)
            else:
                p, z = engine.forward(ids, return_conr_features=True)
        predictions.append(p.float()); features.append(z.float())
    prediction = torch.cat(predictions); representation = torch.cat(features)
    targets = torch.as_tensor((bundle.y[batch]-cfg['score_min'])/(cfg['score_max']-cfg['score_min']),
                              dtype=torch.float32, device=device).reshape(-1, 1)
    if prediction.shape != targets.shape:
        raise ValueError('Prediksi/target harus [B,1], tanpa broadcasting.')
    mse = (prediction-targets).square().mean()
    conr, counts = same_question_conr(representation, targets, prediction,
        [bundle.rows[i]['question'] for i in batch], cfg)
    return mse+cfg['conr_weight']*conr, mse, conr, counts


def conr_preflight(bundle, manifest, cfg):
    """Check TRAINING labels only before model downloads; preserve original sampler."""
    result = []
    for stage, ids in [(f"fold {f['fold']}", f['train']) for f in manifest['folds']]+[('refit', manifest['development'])]:
        eligible = 0
        for i in ids:
            distances = [abs(float(bundle.y[i])-float(bundle.y[j])) for j in ids
                         if i != j and bundle.rows[i]['question'] == bundle.rows[j]['question']]
            eligible += bool(distances) and any(d <= cfg['conr_window_points'] for d in distances) and any(d > cfg['conr_window_points'] for d in distances)
        result.append({'Tahap':stage, 'Jawaban Training':len(ids), 'Anchor Potensial':eligible})
        if cfg['conr_weight'] > 0 and eligible == 0:
            raise ValueError('ConR tidak memiliki anchor dengan pasangan dekat dan jauh pada training '+stage+'. Sampler tidak diubah otomatis.')
    return result


def conr_export_tables(source):
    rows = []
    for file in sorted(source.files):
        match = re.fullmatch(r'09_S3_fold_(\d+)\.xlsx', file)
        if not match and file != '10_S3_refit.xlsx':
            continue
        for r in source.records(file, 'Riwayat_epoch'):
            rows.append(['Validasi silang' if match else 'Latihan final', int(match[1]) if match else None,
                r['epoch'], r['train_mse_scaled'], r['train_conr'], r['train_combined'],
                r['conr_weight'], r['conr_answers'], r['conr_positive_anchors'], r['conr_eligible_anchors'],
                r['conr_active_anchors'], r['conr_positive_pairs'], r['conr_hard_negative_pairs'],
                r['conr_active_batches'], r['conr_successful_batches'], 100*r['conr_active_anchors']/r['conr_answers']])
    return {'Diagnostik ConR':Table(['Tahap','Fold','Epoch','MSE','ConR','Loss Gabungan','Bobot ConR',
        'Jumlah Jawaban','Anchor Positif','Anchor Potensial','Anchor Aktif','Pasangan Positif Terarah',
        'Hard Negative Terarah','Batch Aktif','Langkah ConR Berhasil','Cakupan Aktif (%)'],rows)}


def verify_conr_exports(rows, history, summaries, cfg):
    """Check six fits, nonempty coverage and loss identity in serialized Excel."""
    validate_conr_config(cfg)
    if not rows or len(rows) != len(history) or len(summaries) != cfg['folds']:
        raise ValueError('Riwayat/diagnostik ConR atau ringkasan fold tidak lengkap.')
    for fold in [*range(1, cfg['folds']+1), None]:
        stage = 'Latihan final' if fold is None else 'Validasi silang'
        part = [r for r in rows if r['Tahap'] == stage and r['Fold'] == fold]
        h = [r for r in history if r['Tahap'] == stage and r['Fold'] == fold]
        if not part or len(part) != len(h) or [r['Epoch'] for r in part] != list(range(1,len(part)+1)):
            raise ValueError('Urutan epoch ConR tidak lengkap.')
        if fold is not None and next(r for r in summaries if r['Fold']==fold)['Epoch Dijalankan'] != len(part):
            raise ValueError('Jumlah epoch ConR berbeda dari ringkasan fold.')
        if cfg['conr_weight'] > 0 and sum(r['Langkah ConR Berhasil'] for r in part) == 0:
            raise ValueError('ConR tidak pernah aktif pada satu fit.')
        for r, v in zip(part, h):
            for key in ['MSE','ConR','Loss Gabungan','Bobot ConR','Cakupan Aktif (%)']:
                if not isinstance(r[key], (int,float)) or not math.isfinite(r[key]) or r[key] < 0:
                    raise ValueError('Angka diagnostik ConR tidak finite/sah.')
            n = r['Jumlah Jawaban']
            counts = ['Jumlah Jawaban','Anchor Positif','Anchor Potensial','Anchor Aktif',
                      'Pasangan Positif Terarah','Hard Negative Terarah','Batch Aktif','Langkah ConR Berhasil']
            if any(type(r[k]) is not int or r[k]<0 for k in counts) or n<1:
                raise ValueError('Counter ConR tidak sah.')
            if not 0 <= r['Anchor Aktif'] <= r['Anchor Potensial'] <= r['Anchor Positif'] <= n:
                raise ValueError('Cakupan anchor ConR tidak sah.')
            if not 0 <= r['Langkah ConR Berhasil'] <= r['Batch Aktif'] <= r['Anchor Aktif']:
                raise ValueError('Langkah optimizer ConR tidak sah.')
            if not math.isclose(r['Bobot ConR'],cfg['conr_weight']) or not math.isclose(r['Loss Gabungan'],r['MSE']+r['Bobot ConR']*r['ConR'],rel_tol=1e-6,abs_tol=1e-9):
                raise ValueError('Loss gabungan ConR tidak cocok.')
            if not math.isclose(r['Cakupan Aktif (%)'],100*r['Anchor Aktif']/n,abs_tol=1e-7) or not math.isclose(v['Loss Training (MSE)'],r['MSE'],rel_tol=1e-6,abs_tol=1e-9):
                raise ValueError('Riwayat/cakupan ConR tidak cocok.')
    if len(rows) != sum(len([r for r in rows if r['Tahap']==('Latihan final' if f is None else 'Validasi silang') and r['Fold']==f]) for f in [*range(1,cfg['folds']+1),None]):
        raise ValueError('Identitas fit ConR asing.')
    return {'conr_fits':cfg['folds']+1,'conr_epoch_rows':len(rows),'conr_loss_and_coverage_verified':True}
