"""Visible E48 helpers: fold-local LDS and report validation. Embedded in Colab."""


def lds_training_weights(bundle, training, cfg):
    """Only index training labels. Return global-mean-one weights, never batch normalized."""
    indices = [int(i) for i in training]
    if not indices or len(set(indices)) != len(indices) or min(indices)<0 or max(indices)>=len(bundle.y):
        raise ValueError('LDS memerlukan indeks training unik dan tidak kosong.')
    lower, upper = cfg['score_min'], cfg['score_max']
    size, sigma = cfg['lds_kernel_size'], cfg['lds_sigma']
    if cfg['lds_bin_width']!=1 or cfg['lds_reweight']!='inverse':
        raise ValueError('E48 mengunci bin 1 poin dan inverse frekuensi LDS.')
    if type(size) is not int or size<1 or size%2!=1 or size>upper-lower+1 or not math.isfinite(sigma) or sigma<=0:
        raise ValueError('Kernel LDS harus ganjil, muat rentang skor, dan sigma positif finite.')
    labels = np.asarray(bundle.y[indices], dtype=np.float64)
    if not np.isfinite(labels).all() or not np.equal(labels,np.floor(labels)).all() or np.any(labels<lower) or np.any(labels>upper):
        raise ValueError('Label training LDS harus integer finite dalam rentang skor.')
    bins = (labels-lower).astype(np.int64)
    counts = np.bincount(bins,minlength=int(upper-lower+1)).astype(np.float64)
    positions = np.arange(size,dtype=np.float64)-size//2
    kernel = np.exp(-.5*(positions/sigma)**2)
    kernel /= kernel.sum()
    # Zero padding, sama dengan convolve1d(mode=constant) pada kernel simetris.
    effective = np.convolve(counts,kernel,mode='same')
    density = effective[bins]
    if not np.isfinite(density).all() or np.any(density<=0):
        raise ValueError('Frekuensi efektif LDS pada training harus positif finite.')
    inverse = 1./density
    normalizer = float(inverse.mean())
    weights = inverse/normalizer
    by_index = {i:float(w) for i,w in zip(indices,weights)}
    sample_rows = [{'ID Jawaban':friendly_value('row_id',bundle.rows[i]['row_id']),
                    'Skor Guru':float(label),'Bobot Loss LDS':float(w)} for i,label,w in zip(indices,labels,weights)]
    distribution = [{'Skor Guru':int(lower+j),'Jumlah Training':int(n),
                     'Frekuensi Dihaluskan':float(effective[j]),
                     'Bobot Normalisasi':float(1./effective[j]/normalizer) if n>0 else None}
                    for j,n in enumerate(counts)]
    fingerprint = hashlib.sha256(json.dumps({'indices':indices,'labels':labels.tolist(),
        'weights':weights.tolist(),'kernel':kernel.tolist()},sort_keys=True).encode()).hexdigest()
    audit = {'method':'LDS inverse Gaussian; mean-one on training only',
             'training_count':len(indices),'kernel_size':size,'sigma':float(sigma),'bin_width_points':1,
             'normalization':'mean(weight over training answers)=1; no minibatch renormalization',
             'weight_mean':float(weights.mean()),'weight_min':float(weights.min()),'weight_max':float(weights.max()),
             'weights_sha256':fingerprint,'samples':sample_rows,'distribution':distribution}
    return by_index,audit


def lds_weighted_mse(prediction, targets, weights):
    """Batch mean(w_i * error_i²); weights already normalized on the full training split."""
    if prediction.ndim!=2 or prediction.shape[1]!=1 or targets.shape!=prediction.shape or weights.shape!=prediction.shape:
        raise ValueError('Prediksi, target dan bobot harus [B,1], tanpa broadcasting.')
    if not len(prediction) or not torch.isfinite(weights).all() or bool((weights<=0).any()):
        raise ValueError('Bobot loss harus positif finite; batch tidak kosong.')
    squared = (prediction.float()-targets.float()).square()
    return (weights.detach().float()*squared).mean(),squared.mean()


def lds_report_tables(source):
    """Materialize all report rows; avoid generators capturing a changing source file."""
    samples,distributions,info = [],[],[]
    for file in sorted(source.files):
        match = re.fullmatch(r'09_S3_fold_(\d+)\.xlsx',file)
        if not match and file!='10_S3_refit.xlsx':continue
        stage = 'Validasi silang' if match else 'Latihan final'
        fold = int(match[1]) if match else None
        for raw,target in [('LDS_bobot_jawaban',samples),('LDS_distribusi',distributions),('LDS_info',info)]:
            for row in source.records(file,raw):target.append({'Tahap':stage,'Fold':fold,**row})
    result={}
    for title,rows in [('Bobot LDS Training',samples),('Distribusi LDS Training',distributions),('Audit LDS',info)]:
        if rows:
            keys=list(rows[0]);result[title]=Table(keys,[[r[k] for k in keys] for r in rows])
    return result


def lds_journal_tables(info):
    audit=info['lds']
    return {'LDS_bobot_jawaban':pd.DataFrame(audit['samples']),
            'LDS_distribusi':pd.DataFrame(audit['distribution']),
            'LDS_info':pd.DataFrame([{k:v for k,v in audit.items() if k not in ('samples','distribution')}])}


def validate_e48_download(folder, bundle, manifest, cfg):
    """Read saved Excel values, validate predictions/OOF/LDS; fail before ZIP creation."""
    folder=Path(folder)
    names=['01_Data_dan_Tokenisasi.xlsx','02_Vektor_Jawaban.xlsx','03_Vektor_Kata_dan_Kalimat.xlsx',
           '04_Prediksi_dan_Evaluasi.xlsx','05_Riwayat_Pelatihan.xlsx']
    if any(not (folder/name).is_file() for name in names):
        raise RuntimeError('Unduhan dihentikan: lima Excel belum lengkap.')
    def read(file,sheet):
        book=load_workbook(folder/file,read_only=True,data_only=True)
        try:
            if sheet not in book:raise RuntimeError('Unduhan dihentikan: sheet hilang: '+sheet)
            iterator=iter(book[sheet].values);header=next(iterator,())
            return [dict(zip(header,r)) for r in iterator]
        finally:book.close()
    def same(a,b):return math.isclose(float(a),float(b),rel_tol=1e-6,abs_tol=1e-5)
    file=names[3]
    detail=read(file,'Nilai Guru dan Prediksi')
    try:
        verify_prediction_report_tables({'Nilai Guru dan Prediksi':detail,'Selisih Tepat':read(file,'Selisih Tepat'),
            'Toleransi Selisih':read(file,'Toleransi Selisih')},cfg['score_min'],cfg['score_max'])
    except ValueError as error:raise RuntimeError('Unduhan dihentikan: rincian selisih tidak cocok: '+str(error)) from error
    test_ids={friendly_value('row_id',bundle.rows[i]['row_id']) for i in manifest['test']}
    teachers={friendly_value('row_id',r['row_id']):r['score'] for r in bundle.rows}
    if len(detail)!=len(test_ids) or {r['ID Jawaban'] for r in detail}!=test_ids or any(r['Nilai Guru']!=teachers[r['ID Jawaban']] for r in detail):
        raise RuntimeError('Unduhan dihentikan: ID/jumlah/label test tidak cocok.')
    models=read(file,'Perbandingan Model');summary=read(file,'Ringkasan Validasi');folds=read(file,'Hasil Tiap Fold')
    if len(models)!=1 or len(summary)!=1 or len(folds)!=cfg['folds']:
        raise RuntimeError('Unduhan dihentikan: ringkasan test/CV kosong atau tidak lengkap.')
    score=metrics([r['Nilai Guru'] for r in detail],[r['Prediksi S3'] for r in detail],cfg)
    columns={'RMSE':'RMSE (poin)','Pearson':'Pearson','QWK':'QWK','AC2':'Gwet AC2'}
    for key,column in columns.items():
        if not same(score[key],models[0][column]):raise RuntimeError('Unduhan dihentikan: metrik test tidak cocok: '+key)
        values=[r[column] for r in folds]
        display_value=summary[0][{'AC2':'Gwet AC2'}.get(key,key)+' (rata-rata ± SD)']
        expected=f'{np.mean(values):.4f} ± {np.std(values,ddof=1):.4f} (n={cfg["folds"]})'
        if display_value!=expected:raise RuntimeError('Unduhan dihentikan: mean/SD CV tidak cocok.')
    oof=read(file,'Prediksi Validasi OOF');seen=set()
    for fold in manifest['folds']:
        part=[r for r in oof if r['Fold']==fold['fold']]
        ids={friendly_value('row_id',bundle.rows[i]['row_id']) for i in fold['validation']}
        if len(part)!=len(ids) or {r['ID Jawaban'] for r in part}!=ids or ids&seen or ids&test_ids:
            raise RuntimeError('Unduhan dihentikan: OOF tidak cocok dengan fold validasi.')
        if any(r['Nilai Guru']!=teachers[r['ID Jawaban']] for r in part):
            raise RuntimeError('Unduhan dihentikan: label OOF berbeda dari dataset.')
        actual=metrics([r['Nilai Guru'] for r in part],[r['Prediksi S3'] for r in part],cfg)
        original=next(r for r in folds if r['Fold']==fold['fold'])
        if any(not same(actual[k],original[c]) for k,c in columns.items()):
            raise RuntimeError('Unduhan dihentikan: metrik OOF/fold tidak cocok.')
        seen|=ids
    if len(oof)!=len(manifest['development']) or len(seen)!=len(oof):
        raise RuntimeError('Unduhan dihentikan: jumlah OOF development tidak lengkap.')
    samples=read(names[4],'Bobot LDS Training')
    expected_total=0
    for fold in [*manifest['folds'],{'fold':None,'train':manifest['development']}]:
        expected,audit=lds_training_weights(bundle,fold['train'],cfg)
        wanted={r['ID Jawaban']:r for r in audit['samples']}
        rows=[r for r in samples if r['Fold']==fold['fold'] and r['Tahap']==('Latihan final' if fold['fold'] is None else 'Validasi silang')]
        expected_total+=len(expected)
        if len(rows)!=len(expected) or {r['ID Jawaban'] for r in rows}!=set(wanted) or any(
            r['Skor Guru']!=wanted[r['ID Jawaban']]['Skor Guru'] or not same(r['Bobot Loss LDS'],wanted[r['ID Jawaban']]['Bobot Loss LDS']) for r in rows):
            raise RuntimeError('Unduhan dihentikan: bobot LDS bukan berasal dari training fold yang sesuai.')
    if len(samples)!=expected_total:raise RuntimeError('Unduhan dihentikan: bobot LDS tercampur atau tidak lengkap.')
    history=read(names[4],'Fusion (S3)')
    if not history or any(not math.isfinite(r['Loss Training (MSE)']) or not math.isfinite(r['Loss Training (LDS-MSE)']) for r in history):
        raise RuntimeError('Unduhan dihentikan: history loss kosong/tidak finite.')
    proof={'status':'PASSED','excel_count':5,'test_rows':len(detail),'oof_rows':len(oof),'new_cv_fits':cfg['folds'],
           'metrics_unweighted_verified':True,'prediction_counts_verified':True,'training_only_lds_verified':True,
           'lds_training_rows':expected_total,'summaries_nonempty':True}
    dump(folder/'verifikasi_unduhan.json',proof)
    return proof
