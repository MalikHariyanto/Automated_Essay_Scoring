"""Visible notebook helpers for verified CV reuse and refit diagnostics.

The builder embeds these functions as ordinary Python cells, without project imports.
"""


def resolve_cv_archive(workspace, configured, expected_name, browser_upload=False):
    import shutil
    destination = Path(workspace)/'data/cv_source'/expected_name
    if configured:
        source = Path(configured).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError('CV_SOURCE_FILE/AES300_CV_ZIP tidak ditemukan: '+str(source))
    else:
        candidates = [p/expected_name for p in [Path('/content'),Path.cwd(),Path(workspace)] if (p/expected_name).is_file()]
        if destination.is_file():
            candidates.append(destination)
        if not candidates and browser_upload:
            from google.colab import files
            destination.parent.mkdir(parents=True,exist_ok=True)
            previous = Path.cwd()
            try:
                os.chdir(destination.parent)
                print('Pilih ZIP hasil CV acuan:',expected_name,flush=True)
                uploaded = files.upload()
            finally:
                os.chdir(previous)
            if expected_name not in uploaded:
                raise ValueError('Pilih ZIP acuan yang namanya tercantum di atas, lalu jalankan ulang cell ini.')
            candidates = [destination]
            del uploaded
        if not candidates:
            raise FileNotFoundError('ZIP CV acuan belum ada. Upload '+expected_name+' atau isi CV_SOURCE_FILE.')
        hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in candidates}
        if len(hashes)>1:
            raise ValueError('Ada beberapa ZIP acuan berbeda; isi CV_SOURCE_FILE secara eksplisit.')
        source = candidates[0]
    destination.parent.mkdir(parents=True,exist_ok=True)
    if source.resolve()!=destination.resolve():
        shutil.copy2(source,destination)
    print('ZIP CV acuan:',destination,flush=True)
    return destination


def cv_archive_records(path, file, sheet):
    with zipfile.ZipFile(path) as archive:
        with archive.open(file) as stream:
            book = load_workbook(BytesIO(stream.read()),read_only=True,data_only=True)
        try:
            if sheet not in book.sheetnames:
                return None
            rows = iter(book[sheet].iter_rows(values_only=True))
            header = next(rows,())
            return [dict(zip(header,row)) for row in rows]
        finally:
            book.close()


def verify_reused_cv(path, bundle, manifest, cfg):
    """Baca CV lama saja: tabel hasil test lama tidak dibaca untuk memilih resep."""
    actual_hash = digest(path)
    if actual_hash != cfg['cv_archive_sha256']:
        raise ValueError('SHA256 ZIP acuan berbeda. Gunakan ZIP hasil asli '+cfg['cv_source_experiment']+' yang ditentukan notebook.')
    with zipfile.ZipFile(path) as archive:
        expected = {f'{i:02d}_'+name+'.xlsx' for i,name in enumerate(
            ['Data_dan_Tokenisasi','Vektor_Jawaban','Vektor_Kata_dan_Kalimat','Prediksi_dan_Evaluasi','Riwayat_Pelatihan'],1)}
        if len(archive.namelist())!=5 or set(archive.namelist())!=expected:
            raise ValueError('ZIP acuan harus berisi tepat lima Excel standar.')
    read = lambda file,sheet: cv_archive_records(path,file,sheet)
    old_data = read('01_Data_dan_Tokenisasi.xlsx','Data dan Nilai Guru')
    expected_data = {friendly_value('row_id',r['row_id']):
        (r['question'],r['text'],r['reference'],r['score']) for r in bundle.rows}
    if old_data is None or len(old_data)!=len(bundle.rows):
        raise ValueError('Jumlah data CV acuan berbeda dari dataset sekarang.')
    seen = set()
    for r in old_data:
        identifier = r['ID Jawaban']
        values = (r['Soal'],r['Jawaban Siswa'],r['jawaban referensi'],r['Nilai Guru'])
        if identifier in seen or identifier not in expected_data or values!=expected_data[identifier]:
            raise ValueError('Teks, nilai guru, urutan atau ID dataset berbeda dari CV acuan.')
        seen.add(identifier)
    old_split = read('01_Data_dan_Tokenisasi.xlsx','Pembagian Data')
    expected_split = {(friendly_value('row_id',bundle.rows[i]['row_id']),friendly_value('role',role))
                      for role,key in [('development','development'),('test','test')] for i in manifest[key]}
    if old_split is None or len(old_split)!=len(expected_split) or {
        (r['ID Jawaban'],r['Penggunaan Data']) for r in old_split}!=expected_split:
        raise ValueError('Pembagian development/test berbeda dari CV acuan.')
    old_folds = read('01_Data_dan_Tokenisasi.xlsx','Pembagian Lima Fold')
    expected_folds = {(f['fold'],friendly_value('row_id',bundle.rows[i]['row_id']),friendly_value('role',role))
                      for f in manifest['folds'] for role,key in [('train','train'),('validation','validation')] for i in f[key]}
    if old_folds is None or len(old_folds)!=len(expected_folds) or {
        (r['Fold'],r['ID Jawaban'],r['Penggunaan Data']) for r in old_folds}!=expected_folds:
        raise ValueError('Anggota lima fold berbeda dari CV acuan.')
    settings_rows = read('05_Riwayat_Pelatihan.xlsx','Pengaturan Eksperimen')
    settings = {r['Pengaturan']:r['Nilai'] for r in settings_rows or []}
    mapping = {'Skor minimum':'score_min','Skor maksimum':'score_max','Jumlah fold':'folds',
        'Seed training tetap':'seed','Maksimum epoch per fold':'epochs','Batas epoch tanpa perbaikan':'early_stopping_patience',
        'Batch efektif':'effective_batch','Micro batch':'micro_batch','Learning rate encoder RoBERTa S3':'s3_encoder_lr',
        'Learning rate proyeksi BGE':'projection_lr','Dropout parameter lama':'dropout','Dropout MLP S3':'s3_mlp_dropout',
        'Batas token RoBERTa':'roberta_max_length','Batas token BGE-M3':'bge_max_length',
        'Batas token konteks semua encoder':'context_max_tokens','Batas token soal semua encoder':'question_max_tokens',
        'Format input tiga teks':'input_format','Pooling jawaban S3':'s3_pooling','Hidden pooling S3':'s3_sequence_hidden',
        'Ukuran init attention acuan (dihapus)':'pooling_attention_hidden','Gamma awal Scalar Mix':'scalar_mix_gamma_init',
        'Jumlah layer Scalar Mix':'scalar_mix_layers','LR Scalar Mix':'scalar_mix_lr','LR MLP S3':'s3_mlp_lr',
        'LR pooling dan norm S3':'s3_layers_lr','Fusion S3':'s3_fusion'}
    for title,key in mapping.items():
        a,b = settings.get(title),cfg[key]
        same = math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-12) if isinstance(a,(int,float)) and isinstance(b,(int,float)) else a==b
        if not same:
            raise ValueError('Pengaturan berbeda dari CV acuan: '+title)
    metric_rows = read('04_Prediksi_dan_Evaluasi.xlsx','Hasil Tiap Fold')
    summaries = read('05_Riwayat_Pelatihan.xlsx','Ringkasan Lima Fold')
    all_history = read('05_Riwayat_Pelatihan.xlsx','Fusion (S3)')
    if not metric_rows or not summaries or not all_history:
        raise ValueError('Metrik, ringkasan atau history lima fold tidak lengkap.')
    metric_rows = [r for r in metric_rows if r['Skenario']=='S3']
    summaries = [r for r in summaries if r['Skenario']=='S3']
    history = [r for r in all_history if r['Tahap']=='Validasi silang']
    if len(metric_rows)!=5 or len(summaries)!=5 or {r['Fold'] for r in metric_rows}!=set(range(1,6)) or {r['Fold'] for r in summaries}!=set(range(1,6)):
        raise ValueError('CV acuan harus berisi tepat lima fold S3.')
    metrics_by_fold = {r['Fold']:r for r in metric_rows}
    for row in summaries:
        fold = row['Fold'];entries = [r for r in history if r['Fold']==fold]
        if len(entries)!=row['Epoch Dijalankan'] or row['Epoch Maksimal']!=cfg['epochs'] or row['Batas Tanpa Perbaikan']!=cfg['early_stopping_patience']:
            raise ValueError('History/anggaran epoch CV berbeda dari konfigurasi acuan.')
        if [r['Epoch'] for r in entries]!=list(range(1,len(entries)+1)):
            raise ValueError('Urutan epoch CV tidak lengkap.')
        best = min(entries,key=lambda r:r['RMSE Validasi (poin)'])
        if best['Epoch']!=row['Epoch Terpilih'] or not math.isclose(best['RMSE Validasi (poin)'],metrics_by_fold[fold]['RMSE (poin)'],abs_tol=1e-8):
            raise ValueError('Epoch terbaik/metric CV tidak konsisten dengan history.')
    selected = int(math.ceil(np.median([r['Epoch Terpilih'] for r in summaries])))
    if selected!=cfg['expected_refit_epochs'] or not 1<=selected<=cfg['refit_scheduler_epochs']:
        raise ValueError('Median epoch CV berbeda dari refit yang sudah dikunci untuk eksperimen ini.')
    cv_rows = [{'Skenario':'S3','Fold':r['Fold'],'RMSE':r['RMSE (poin)'],'Pearson':r['Pearson'],
                'QWK':r['QWK'],'AC2':r['Gwet AC2']} for r in sorted(metric_rows,key=lambda r:r['Fold'])]
    if any(not math.isfinite(float(r[k])) for r in cv_rows for k in ['RMSE','Pearson','QWK','AC2']):
        raise ValueError('Metrik CV acuan tidak finite.')
    weights = [r for r in read('05_Riwayat_Pelatihan.xlsx','Bobot Scalar Mix') or [] if r['Tahap']=='Validasi silang']
    if len(weights)!=5*cfg['scalar_mix_layers']:
        raise ValueError('Bobot Scalar Mix lima checkpoint CV tidak lengkap.')
    for fold in range(1,6):
        group = [r for r in weights if r['Fold']==fold]
        values = np.array([r['Bobot Softmax'] for r in group],float)
        if len(group)!=cfg['scalar_mix_layers'] or not np.isfinite(values).all() or (values<0).any() or abs(values.sum()-1)>1e-6:
            raise ValueError('Bobot campuran CV acuan tidak sah.')
    error_summary = read('05_Riwayat_Pelatihan.xlsx','MAE dan Toleransi CV')
    if error_summary is None or len(error_summary)!=6 or error_summary[-1]['Jumlah Jawaban']!=len(manifest['development']):
        raise ValueError('Ringkasan MAE/toleransi CV acuan tidak lengkap.')
    for r in error_summary[:5]:
        if r['Jumlah Jawaban']!=len(manifest['folds'][r['Fold']-1]['validation']) or not math.isfinite(r['MAE Validasi (poin)']) or r['MAE Validasi (poin)']<0:
            raise ValueError('Jumlah/MAE validasi acuan tidak sah.')
        if not 0<=r['Dalam ±1 poin']<=r['Dalam ±5 poin']<=r['Dalam ±10 poin']<=r['Jumlah Jawaban']:
            raise ValueError('Jumlah toleransi CV acuan tidak sah.')
    aggregate = error_summary[-1]
    if abs(sum(r['Jumlah Jawaban']*r['MAE Validasi (poin)'] for r in error_summary[:5])/len(manifest['development'])-aggregate['MAE Validasi (poin)'])>1e-8 or any(
        sum(r[k] for r in error_summary[:5])!=aggregate[k] for k in ['Dalam ±1 poin','Dalam ±5 poin','Dalam ±10 poin']):
        raise ValueError('Agregat MAE/toleransi CV acuan tidak cocok.')
    oof = read('04_Prediksi_dan_Evaluasi.xlsx','Prediksi Validasi OOF')
    if oof is not None:
        expected_oof = {(f['fold'],friendly_value('row_id',bundle.rows[i]['row_id'])) for f in manifest['folds'] for i in f['validation']}
        if len(oof)!=len(expected_oof) or {(r['Fold'],r['ID Jawaban']) for r in oof}!=expected_oof:
            raise ValueError('ID/fold OOF acuan berbeda dari development.')
        for f in range(1,6):
            group = [r for r in oof if r['Fold']==f]
            for r in group:
                if r['Nilai Guru']!=expected_data[r['ID Jawaban']][3]:
                    raise ValueError('Nilai guru OOF berbeda.')
            actual = metrics([r['Nilai Guru'] for r in group],[r['Prediksi S3'] for r in group],cfg)
            for key in ['RMSE','Pearson','QWK','AC2']:
                if not math.isclose(actual[key],cv_rows[f-1][key],rel_tol=1e-6,abs_tol=1e-5):
                    raise ValueError('Metrik OOF acuan tidak cocok dengan tabel fold.')
    return {'base_experiment':cfg['cv_source_experiment'],'archive_sha256':actual_hash,
        'dataset_sha256':bundle.source_sha256,'split_sha256':manifest['split_sha256'],
        'selected_epoch':selected,'cv_rows':cv_rows,'history':history,'fold_summaries':summaries,
        'mix_weights':weights,'error_summary':error_summary,'oof':oof,
        'verification':{'rows':len(bundle.rows),'folds':5,'settings_checked':len(mapping),
          'test_results_used_for_selection':False,'new_cv_fits':0,
          'limitations':'Verifikasi field yang tersimpan di lima Excel. ZIP tidak mencatat semua versi runtime/revisi/parameter; bukan bukti seluruh runtime lama identik.'}}


def export_reused_cv(reused, journal):
    provenance = {'Eksperimen acuan':reused['base_experiment'],'SHA256 ZIP acuan':reused['archive_sha256'],
        'CV baru dijalankan':0,'Fold CV lama dipakai ulang':5,'Epoch refit dari median CV':reused['selected_epoch'],
        'OOF acuan tersedia':reused['oof'] is not None,'Hasil test lama untuk memilih epoch':False,
        'Batas verifikasi':reused['verification']['limitations']}
    sheets = {'Provenansi':pd.DataFrame([{'Keterangan':k,'Nilai':v} for k,v in provenance.items()]),
        'Riwayat_CV_acuan':pd.DataFrame(reused['history']),
        'Ringkasan_fold_acuan':pd.DataFrame(reused['fold_summaries']),
        'Bobot_mix_CV_acuan':pd.DataFrame(reused['mix_weights']),
        'MAE_toleransi_CV_acuan':pd.DataFrame(reused['error_summary'])}
    if reused['oof'] is not None:
        sheets['OOF_acuan'] = pd.DataFrame(reused['oof'])
    journal.save('09_CV_acuan_dipakai_ulang',sheets,'CV LAMA DIPAKAI ULANG; bukan training/evaluasi CV baru',
                 status='REUSED VERIFIED CV',base_experiment=reused['base_experiment'],archive_sha256=reused['archive_sha256'])


def load_reused_bge(bundle, cfg, output):
    if digest(cfg['cv_source_path'])!=cfg['cv_archive_sha256']:
        raise ValueError('ZIP acuan berubah setelah verifikasi.')
    rows = cv_archive_records(cfg['cv_source_path'],'02_Vektor_Jawaban.xlsx','BGE-M3')
    if rows is None or len(rows)!=len(bundle.rows):
        raise ValueError('Vektor BGE acuan tidak lengkap.')
    lookup = {r['ID Jawaban']:r for r in rows}
    if len(lookup)!=len(rows):
        raise ValueError('ID vektor BGE acuan duplikat.')
    values = []
    for row in bundle.rows:
        identifier = friendly_value('row_id',row['row_id'])
        if identifier not in lookup:
            raise ValueError('ID BGE acuan tidak cocok dengan dataset.')
        values.append([lookup[identifier][f'Komponen {i}'] for i in range(1,1025)])
    array = np.asarray(values,dtype=np.float32)
    if array.shape!=(len(bundle.rows),1024) or not np.isfinite(array).all():
        raise ValueError('Vektor BGE harus finite dengan shape [jumlah jawaban,1024].')
    dump(Path(output)/'bge_encoding.json',{'source':'frozen vectors reused from verified CV archive',
         'archive_sha256':cfg['cv_archive_sha256'],'base_experiment':cfg['cv_source_experiment'],
         'BGE_encoder_trained':False,'rows':len(array),'dimensions':1024})
    return torch.from_numpy(array)


def validate_refit_download(folder, bundle, manifest, cfg):
    """Wajib lolos sebelum ZIP ditawarkan untuk download; tidak menjalankan model."""
    folder = Path(folder)
    names = ['01_Data_dan_Tokenisasi.xlsx','02_Vektor_Jawaban.xlsx','03_Vektor_Kata_dan_Kalimat.xlsx',
             '04_Prediksi_dan_Evaluasi.xlsx','05_Riwayat_Pelatihan.xlsx']
    if any(not (folder/name).is_file() for name in names):
        raise RuntimeError('Unduhan dihentikan: lima Excel standar belum lengkap.')
    def rows(file,sheet):
        book = load_workbook(folder/file,read_only=True,data_only=True)
        try:
            if sheet not in book.sheetnames:
                raise RuntimeError('Unduhan dihentikan: sheet '+sheet+' tidak ada.')
            iterator = iter(book[sheet].values);header=next(iterator,())
            return [dict(zip(header,r)) for r in iterator]
        finally:book.close()
    data = rows(names[0],'Data dan Nilai Guru')
    expected_data_ids = {friendly_value('row_id',r['row_id']) for r in bundle.rows}
    if len(data)!=len(bundle.rows) or {r['ID Jawaban'] for r in data}!=expected_data_ids:
        raise RuntimeError('Unduhan dihentikan: tabel dataset kosong atau ID tidak lengkap.')
    for sheet,dimensions in [('BGE-M3',1024),('Fusion S3',1024),('Ringkasan CoAttn Pooling S3',128)]:
        vector_rows = rows(names[1],sheet)
        if len(vector_rows)!=len(bundle.rows) or {r['ID Jawaban'] for r in vector_rows}!=expected_data_ids or any(
            len(r)!=dimensions+1 or not all(isinstance(r.get(f'Komponen {i}'),(int,float)) and math.isfinite(r[f'Komponen {i}']) for i in range(1,dimensions+1)) for r in vector_rows):
            raise RuntimeError('Unduhan dihentikan: vektor '+sheet+' tidak lengkap/tidak finite.')
    file = names[3]
    models = rows(file,'Perbandingan Model');cv = rows(file,'Hasil Tiap Fold');summary=rows(file,'Ringkasan Validasi')
    prediction = rows(file,'Nilai Guru dan Prediksi')
    if len(models)!=1 or len(cv)!=5 or len(summary)!=1:
        raise RuntimeError('Unduhan dihentikan: tabel metrik test atau ringkasan lima fold kosong/tidak lengkap.')
    expected_ids = {friendly_value('row_id',bundle.rows[i]['row_id']) for i in manifest['test']}
    if len(prediction)!=len(expected_ids) or {r['ID Jawaban'] for r in prediction}!=expected_ids:
        raise RuntimeError('Unduhan dihentikan: jumlah/ID prediksi test tidak cocok dengan split.')
    lookup = {friendly_value('row_id',r['row_id']):r['score'] for r in bundle.rows}
    if any(r['Nilai Guru']!=lookup[r['ID Jawaban']] for r in prediction):
        raise RuntimeError('Unduhan dihentikan: nilai guru pada prediksi berbeda dari sumber.')
    actual = metrics([r['Nilai Guru'] for r in prediction],[r['Prediksi S3'] for r in prediction],cfg)
    for key,column in [('RMSE','RMSE (poin)'),('Pearson','Pearson'),('QWK','QWK'),('AC2','Gwet AC2')]:
        value = models[0].get(column)
        if value is None or actual[key] is None or not math.isclose(float(value),actual[key],rel_tol=1e-6,abs_tol=1e-5):
            raise RuntimeError('Unduhan dihentikan: metrik test tidak cocok dengan prediksi: '+key)
    for key in ['RMSE','Pearson','QWK','AC2']:
        values = [r[{'RMSE':'RMSE (poin)','Pearson':'Pearson','QWK':'QWK','AC2':'Gwet AC2'}[key]] for r in cv]
        title = {'RMSE':'RMSE','Pearson':'Pearson','QWK':'QWK','AC2':'Gwet AC2'}[key]+' (rata-rata ± SD)'
        display_value = summary[0].get(title)
        if not isinstance(display_value,str) or '(n=5)' not in display_value or abs(float(display_value.split(' ± ')[0])-np.mean(values))>5e-4 or abs(float(display_value.split(' ± ')[1].split(' (n=')[0])-np.std(values,ddof=1))>5e-4:
            raise RuntimeError('Unduhan dihentikan: ringkasan CV tidak cocok dengan lima fold: '+key)
    try:
        verify_prediction_report_tables({'Nilai Guru dan Prediksi':prediction,
            'Selisih Tepat':rows(file,'Selisih Tepat'),'Toleransi Selisih':rows(file,'Toleransi Selisih')},
            cfg['score_min'],cfg['score_max'])
    except ValueError as error:
        raise RuntimeError('Unduhan dihentikan: rincian/rekap selisih prediksi tidak cocok: '+str(error)) from error
    history = rows(names[4],'Fusion (S3)')
    if len(history)!=cfg['expected_refit_epochs'] or any(r['Tahap']!='Latihan final' for r in history):
        raise RuntimeError('Unduhan dihentikan: history refit baru tidak lengkap atau tercampur CV lama.')
    old_folds = rows(names[4],'Ringkasan Fold CV Acuan')
    if len(old_folds)!=5:
        raise RuntimeError('Unduhan dihentikan: provenance lima fold CV lama tidak lengkap.')
    weights = rows(names[4],'Bobot Scalar Mix')
    if len(weights)!=cfg['scalar_mix_layers'] or any(not math.isfinite(r['Gamma']) for r in weights) or abs(sum(r['Bobot Softmax'] for r in weights)-1)>1e-6:
        raise RuntimeError('Unduhan dihentikan: bobot campuran checkpoint refit tidak lengkap.')
    trace = rows(names[4],'Jadwal LR Refit')
    if len(trace)!=cfg['expected_refit_epochs']*5:
        raise RuntimeError('Unduhan dihentikan: riwayat LR lima kelompok per epoch tidak lengkap.')
    for row in trace:
        step,total,warmup = row['Optimizer Steps'],row['Total Steps Jadwal'],row['Warmup Steps']
        ratio = step/max(1,warmup) if step<warmup else max(0.,(total-step)/max(1,total-warmup))
        if row['Horizon Jadwal (epoch)']!=cfg['refit_scheduler_epochs'] or not math.isclose(row['LR Akhir Epoch'],row['LR Puncak']*ratio,rel_tol=1e-6,abs_tol=1e-12):
            raise RuntimeError('Unduhan dihentikan: jejak LR tidak cocok dengan scheduler.')
    proof = {'status':'PASSED','excel_count':5,'test_rows':len(prediction),'cv_rows_reused':5,
             'new_cv_fits':0,'refit_epochs':len(history),'test_metrics_recomputed_match':True,
             'summaries_nonempty':True,'lr_trace_matches':True,'prediction_breakdown_verified':True}
    dump(folder/'verifikasi_unduhan.json',proof)
    return proof
