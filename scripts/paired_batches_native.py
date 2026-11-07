"""Shared E55/E56 deterministic training-only batch plans, each answer once."""


def paired_batch_plan(training, bundle, cfg, seed, epoch):
    """Prefer same-question near/far triples; no replacement, dropping, or held-out data.

    Greedily choose a random eligible anchor, a near peer and a far peer from
    remaining TRAINING answers, then optionally a fourth same-question answer.
    Two groups of up to four form a batch of eight. Shuffle leftovers and every
    batch. Local RNG depends only on seed/epoch, not model, loss or early stopping.
    """
    ids = [int(i) for i in training]
    if not ids or len(set(ids)) != len(ids) or any(i < 0 or i >= len(bundle.rows) for i in ids):
        raise ValueError('Training IDs harus unik, sah dan tidak kosong.')
    size = cfg['effective_batch']
    if type(size) is not int or size < 3 or type(epoch) is not int or epoch < 1:
        raise ValueError('Batch pasangan minimal3; epoch harus positif.')
    rng = random.Random(int(seed)+1000003*epoch)
    groups = {}
    for i in ids:
        q = bundle.rows[i]['question']
        if not isinstance(q,str) or not q.strip() or not math.isfinite(float(bundle.y[i])):
            raise ValueError('Soal/label training untuk sampler tidak sah.')
        groups.setdefault(q, []).append(i)
    plans = []
    window = cfg['conr_window_points']
    while any(groups.values()):
        batch = []
        while len(batch) < size and any(groups.values()):
            capacity = size-len(batch)
            candidates = []
            if capacity >= 3:
                for q, members in groups.items():
                    for anchor in members:
                        near = [j for j in members if j != anchor and abs(float(bundle.y[j])-float(bundle.y[anchor])) <= window]
                        far = [j for j in members if abs(float(bundle.y[j])-float(bundle.y[anchor])) > window]
                        if near and far:
                            candidates.append((q,anchor,near,far))
            if candidates:
                q,anchor,near,far = rng.choice(candidates)
                chosen = [anchor,rng.choice(near),rng.choice(far)]
                for i in chosen:groups[q].remove(i)
                if capacity >= 4 and groups[q]:
                    peer = rng.choice(groups[q]);chosen.append(peer);groups[q].remove(peer)
                batch.extend(chosen)
            else:
                remaining = [i for members in groups.values() for i in members]
                rng.shuffle(remaining)
                chosen = remaining[:capacity]
                for i in chosen:groups[bundle.rows[i]['question']].remove(i)
                batch.extend(chosen)
        rng.shuffle(batch);plans.append(batch)
    flat = [i for batch in plans for i in batch]
    if sorted(flat) != sorted(ids) or len(flat) != len(set(flat)) or len(plans) != math.ceil(len(ids)/size):
        raise RuntimeError('Sampler mengulang/menghilangkan jawaban atau mengubah jumlah batch.')
    positives = eligible = 0
    for batch in plans:
        for i in batch:
            distance = [abs(float(bundle.y[i])-float(bundle.y[j])) for j in batch
                        if i != j and bundle.rows[i]['question'] == bundle.rows[j]['question']]
            has_near = any(d <= window for d in distance)
            positives += has_near
            eligible += has_near and any(d > window for d in distance)
    # A hash permits comparison without placing private question text in reports.
    identity = hashlib.sha256(json.dumps(plans,separators=(',',':')).encode()).hexdigest()
    return plans, {'batch_plan_sha256':identity,'batch_count':len(plans),'batch_answers':len(flat),
                   'batch_unique_answers':len(set(flat)),'batch_positive_anchors':positives,
                   'batch_eligible_anchors':eligible,'batch_seed':int(seed),'batch_epoch':epoch}


def paired_batch_report(source):
    rows = []
    for file in sorted(source.files):
        match = re.fullmatch(r'09_S3_fold_(\d+)\.xlsx',file)
        if not match and file != '10_S3_refit.xlsx':continue
        for r in source.records(file,'Riwayat_epoch'):
            rows.append(['Validasi silang' if match else 'Latihan final',int(match[1]) if match else None,
                r['epoch'],r['batch_seed'],r['batch_count'],r['batch_answers'],r['batch_unique_answers'],
                r['batch_positive_anchors'],r['batch_eligible_anchors'],r['batch_plan_sha256'],
                r['epoch_checkpoints'],r['conr_weight']])
    return {'Rencana Batch':Table(['Tahap','Fold','Epoch','Seed Fit','Jumlah Batch','Jumlah Jawaban',
        'Jawaban Unik','Anchor Positif Rencana','Anchor Potensial Rencana','SHA256 Rencana',
        'Checkpoint Tiap Epoch','Bobot ConR'],rows)}


def verify_paired_batch_exports(rows, diagnostics, cfg, bundle, manifest):
    if cfg['batch_recipe'] != 'same_question_near_far_once_v1' or cfg['epoch_checkpoints'] is not False:
        raise ValueError('Resep batch/checkpoint bukan pasangan E55/E56.')
    if not rows or len(rows) != len(diagnostics):
        raise ValueError('Rencana batch tidak lengkap.')
    seen = set()
    for r,d in zip(rows,diagnostics):
        identity = (r['Tahap'],r['Fold'],r['Epoch'])
        if identity in seen or identity != (d['Tahap'],d['Fold'],d['Epoch']):
            raise ValueError('Identitas rencana batch duplikat/tidak cocok.')
        seen.add(identity)
        n = r['Jumlah Jawaban']
        if type(n) is not int or n<1 or r['Jawaban Unik'] != n or d['Jumlah Jawaban'] != n:
            raise ValueError('Jawaban batch diulang atau hilang.')
        if r['Jumlah Batch'] != math.ceil(n/cfg['effective_batch']) or r['Checkpoint Tiap Epoch'] is not False:
            raise ValueError('Jumlah batch/checkpoint epoch tidak cocok.')
        if (r['Anchor Positif Rencana'] != d['Anchor Positif'] or r['Anchor Potensial Rencana'] != d['Anchor Potensial']
                or r['Bobot ConR'] != cfg['conr_weight'] or not re.fullmatch('[0-9a-f]{64}',str(r['SHA256 Rencana']))):
            raise ValueError('Cakupan/hash batch tidak cocok dengan diagnostik.')
        if r['Tahap']=='Latihan final' and r['Fold'] is None:
            training,seed=manifest['development'],cfg['seed']
        elif r['Tahap']=='Validasi silang' and r['Fold'] in range(1,cfg['folds']+1):
            training=next(f['train'] for f in manifest['folds'] if f['fold']==r['Fold'])
            seed=cfg['seed']+r['Fold']
        else:
            raise ValueError('Tahap/fold batch tidak dikenal.')
        _,expected=paired_batch_plan(training,bundle,cfg,seed,r['Epoch'])
        if r['Seed Fit']!=seed or r['SHA256 Rencana']!=expected['batch_plan_sha256']:
            raise ValueError('Hash/seed rencana berbeda dari training fold yang seharusnya.')
    return {'paired_batch_epoch_rows':len(rows),'each_training_answer_once_verified':True,
            'epoch_checkpoint_disabled_verified':True}
