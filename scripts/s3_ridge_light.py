"""Budgeted, fixed-alpha S3 with ordinary grouped CV and portable completed-stage backups."""


def light_recipe(cfg):
    # Completed numerical features are portable. In-progress optimizer snapshots remain hardware-specific.
    return {k:v for k,v in cfg.items() if k not in ('dataset','output','device')}


def representation_contract(bundle,bge,cfg,training,inference):
    training,inference=sorted(map(int,training)),sorted(map(int,inference))
    if not training or len(training)!=len(set(training)) or not set(training)<=set(inference):
        raise ValueError('Indeks training representasi tidak sah.')
    return dict(protocol='S3_LIGHT_FIXED_ALPHA_portable_complete_v1',
        source_sha256=bundle.source_sha256,code_sha256=NOTEBOOK_CODE_SHA256,
        training=training,inference=inference,
        train_target_sha256=hashlib.sha256(np.asarray(bundle.y[training],dtype=np.float64).tobytes()).hexdigest(),
        bge_sha256=hashlib.sha256(np.asarray(bge,dtype=np.float32).tobytes()).hexdigest(),
        config=light_recipe(cfg),seed=cfg['seed'],epochs=cfg['epochs'],validation_used_for_training=False)


def light_membership(manifest):
    return hashlib.sha256(json.dumps({k:manifest[k] for k in ('development','test','folds')},sort_keys=True).encode()).hexdigest()


def initialize_s3_light(bundle,manifest,cfg,output):
    if cfg['folds']!=5 or len(manifest['folds'])!=5:
        raise ValueError('Varian hemat ini mempertahankan5fold yang sama dengan eksperimen sebelumnya.')
    if cfg['s3_ridge_alphas'] != [cfg['s3_fixed_alpha']]:
        raise ValueError('Alpha S3 tetap: tidak ada pencarian parameter pada CV.')
    if not math.isfinite(cfg['s3_fixed_alpha']) or cfg['s3_fixed_alpha']<=0:
        raise ValueError('Alpha tetap harus positif dan finite.')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    protocol=dict(experiment='S3_FUSION_RIDGE_LIGHT_FIXED_ALPHA',code_sha256=NOTEBOOK_CODE_SHA256,
        dataset_sha256=bundle.source_sha256,split_sha256=manifest['split_sha256'],config=cfg,
        representation='E21: RoBERTa CLS768 LayerNorm + BGE1024 projected256 LayerNorm → concat1024',
        stage_training='MLP auxiliary representation; fixed5epoch default; train labels only',
        ridge='Post-hoc Ridge on the same raw fusion features as MLP control',
        evaluation_protocol='Ordinary grouped5fold CV with predeclared fixed alpha; NOT nested CV',
        alpha_selection='None: fixed before execution; no grid search or CV-based alpha selection',
        fixed_alpha=cfg['s3_fixed_alpha'],alpha_rationale='1.0 is sklearn Ridge default starting value, not a paper optimum or an S2-transferred optimum',
        unique_representation_fits=6,epoch_budget=6*cfg['epochs'],test_used_for_selection=False,BGE_encoder_trained=False,
        historical_caveat='Resource-budget ablation; epochs and fixed-alpha protocol differ from nested25epoch notebook. Historical test reused.',
        backup='Each completed outer fold is archived for laptop download; active epoch snapshots and final encoder are not included.')
    dump(output/'protocol_before_test.json',protocol);dump(output/'split_manifest.json',manifest)
    print('Output aktual:',output,flush=True)
    print(f'Budget:6pelatihan × {cfg["epochs"]}epoch = {6*cfg["epochs"]}epoch total; alpha tetap={cfg["s3_fixed_alpha"]}.',flush=True)
    return protocol


def light_backup(bundle,manifest,cfg,output,fold):
    run=Path(output);members={}
    for name in ('bge_frozen.npy','bge_encoding.json'):
        if (run/name).is_file():members[name]=run/name
    completed=0
    for proof_path in sorted((run/'representations').glob('*/complete.json')):
        proof=json.loads(proof_path.read_text())
        if proof['final_model_saved']:continue
        f=proof_path.parent/'features.npz'
        if digest(f)!=proof['features_sha256']:raise ValueError('Cache tidak utuh saat membuat backup.')
        members[str(proof_path.relative_to(run))]=proof_path
        members[str(f.relative_to(run))]=f;completed+=1
    if 'bge_frozen.npy' not in members:raise ValueError('Fitur BGE belum tersimpan untuk backup.')
    meta=dict(format='S3_LIGHT_COMPLETED_BACKUP_V1',code_sha256=NOTEBOOK_CODE_SHA256,
        dataset_sha256=bundle.source_sha256,membership_sha256=light_membership(manifest),recipe=light_recipe(cfg),
        completed_outer_stages=completed,in_progress_training_included=False,final_encoder_included=False,
        files={n:digest(p) for n,p in members.items()})
    directory=run/'backups';directory.mkdir(exist_ok=True)
    archive=directory/f'Backup_S3_Ridge_Hemat_Fold_{fold}.zip';temporary=archive.with_suffix('.pending')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
        for name,path in members.items():z.write(path,name)
        z.writestr('backup_manifest.json',json.dumps(meta,ensure_ascii=False,indent=2))
    temporary.replace(archive)
    info=dict(archive=str(archive),completed_outer_stages=completed,bytes=archive.stat().st_size,
              note='Simpan berkas yang diunduh di laptop. Berkas pada disk Colab saja belum merupakan backup eksternal.')
    dump(ROOT/'outputs/s3_light_backup_download.json',info)
    print(f'Backup fold{fold}: {archive} ({info["bytes"]/1024**2:.1f}MiB), {completed}tahap selesai.',flush=True)
    return info


def restore_light_backup(path,bundle,manifest,cfg,output):
    """Validate every member before writing; never extract arbitrary uploaded paths or pickle."""
    import io
    import stat
    run=Path(output)
    with zipfile.ZipFile(path) as z:
        entries=z.infolist();names=[v.filename for v in entries]
        if len(names)!=len(set(names)) or len(names)>64 or sum(v.file_size for v in entries)>128*1024**2:
            raise ValueError('Ukuran/jumlah/duplikasi berkas backup tidak sah.')
        for v in entries:
            if stat.S_ISLNK(v.external_attr>>16):raise ValueError('Symlink tidak diizinkan dalam backup.')
        meta=json.loads(z.read('backup_manifest.json'))
        if meta.get('format')!='S3_LIGHT_COMPLETED_BACKUP_V1':raise ValueError('Ini bukan backup varian hemat.')
        if meta['code_sha256']!=NOTEBOOK_CODE_SHA256 or meta['dataset_sha256']!=bundle.source_sha256:
            raise ValueError('Backup berasal dari kode atau dataset berbeda.')
        if meta['membership_sha256']!=light_membership(manifest) or meta['recipe']!=light_recipe(cfg):
            raise ValueError('Split atau resep training backup berbeda.')
        if set(names)!=set(meta['files'])|{'backup_manifest.json'}:raise ValueError('Daftar berkas backup tidak cocok.')
        payload={}
        for name,h in meta['files'].items():
            p=Path(name)
            allowed=name in ('bge_frozen.npy','bge_encoding.json') or (
                len(p.parts)==3 and p.parts[0]=='representations' and re.fullmatch('[0-9a-f]{20}',p.parts[1])
                and p.parts[2] in ('complete.json','features.npz'))
            if not allowed or p.is_absolute() or '..' in p.parts:raise ValueError('Path backup tidak diizinkan.')
            raw=z.read(name)
            if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('Checksum backup tidak cocok.')
            payload[name]=raw
        with io.BytesIO(payload['bge_frozen.npy']) as stream:X=np.load(stream,allow_pickle=False)
        validate_s3_protocol(bundle,X,manifest,cfg)
        count=0
        for name,raw in payload.items():
            if not name.endswith('/complete.json'):continue
            proof=json.loads(raw);contract=proof['contract'];tr=contract['training'];infer=contract['inference']
            if contract!=representation_contract(bundle,X,cfg,tr,infer) or proof['final_model_saved']:
                raise ValueError('Kontrak cache fold berbeda.')
            if set(infer)!=set(manifest['development']) or not any(set(tr)==set(f['train']) for f in manifest['folds']):
                raise ValueError('Cache tidak cocok dengan train fold development.')
            feature_name=str(Path(name).with_name('features.npz'))
            if feature_name not in payload or hashlib.sha256(payload[feature_name]).hexdigest()!=proof['features_sha256']:
                raise ValueError('Fitur dalam backup tidak cocok dengan bukti cache.')
            expected_key=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()[:20]
            if Path(name).parts[1]!=expected_key:raise ValueError('Identitas folder cache tidak cocok.')
            with np.load(io.BytesIO(payload[feature_name]),allow_pickle=False) as arrays:
                if arrays['concat'].shape!=(len(infer),1024) or not np.isfinite(arrays['concat']).all():
                    raise ValueError('Shape fitur backup tidak sah.')
                if arrays['indices'].tolist()!=infer:raise ValueError('Urutan fitur backup berubah.')
            count+=1
        if count!=meta['completed_outer_stages']:raise ValueError('Jumlah tahap backup berubah.')
        for name,raw in payload.items():
            target=run/name
            if target.exists() and target.read_bytes()!=raw:raise ValueError('Restore akan menimpa berkas berbeda; gunakan workspace baru.')
        for name,raw in payload.items():
            target=run/name;target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_name(target.name+'.restore_pending');temporary.write_bytes(raw);temporary.replace(target)
    print(f'Backup dipulihkan: {count}fold selesai; tahap yang sedang berjalan harus diulang jika runtime hilang.',flush=True)
    return dict(completed_outer_stages=count,source=str(path))


def run_s3_light_fold(number,bundle,bge,manifest,cfg,output):
    X=np.asarray(bge,dtype=np.float32);validate_s3_protocol(bundle,X,manifest,cfg)
    f=manifest['folds'][number-1]
    if f['fold']!=number:raise ValueError('Nomor fold tidak sesuai manifest.')
    tr,va=f['train'],f['validation'];run=Path(output)
    stage=fit_s3_representation(bundle,torch.from_numpy(X),cfg,tr,manifest['development'],run/'representations',
                                f'Representasi {number}/6: outer {number}')
    alpha=cfg['s3_fixed_alpha'];y=np.asarray(bundle.y,dtype=float)
    ridge=Ridge(alpha=alpha,fit_intercept=True,solver='cholesky').fit(stage_values(stage,tr).astype(float),y[tr])
    s2=Ridge(alpha=cfg['s2_ridge_alpha'],fit_intercept=True,solver='cholesky').fit(X[tr].astype(float),y[tr])
    values={'S3_RIDGE':ridge.predict(stage_values(stage,va).astype(float)),
            'S3_MLP_CONTROL':stage_values(stage,va,'mlp_prediction'),'S2_RIDGE_FIXED':s2.predict(X[va].astype(float))}
    evaluation=[];predictions=[]
    for model,p in values.items():
        a=alpha if model=='S3_RIDGE' else cfg['s2_ridge_alpha'] if model=='S2_RIDGE_FIXED' else None
        evaluation.append(s3_evaluation(y[va],p,cfg,model,'DEVELOPMENT_OUTER_FOLD',number,a))
        predictions.extend(s3_predictions(bundle,va,p,model,'DEVELOPMENT_OOF',number,a))
    result=dict(fold=number,stage=dict(kind='OUTER',fold=number,folder=stage['folder'],proof=stage['proof']),
                evaluation=evaluation,predictions=predictions)
    dump(run/f'fold_{number}_result.json',result)
    light_backup(bundle,manifest,cfg,run,number)
    show(pd.DataFrame(evaluation),f'Fold{number}/5 selesai; parameter tetap, bukan pemilihan alpha')
    return result


def finalize_s3_light(bundle,bge,manifest,cfg,output,fold_results):
    run=Path(output);X=np.asarray(bge,dtype=np.float32);y=np.asarray(bundle.y,dtype=float)
    if sorted(v['fold'] for v in fold_results)!=[1,2,3,4,5]:raise ValueError('Selesaikan kelima fold terlebih dahulu.')
    dev,test=manifest['development'],manifest['test'];evaluations=[];predictions=[];stages=[]
    for result in fold_results:
        evaluations.extend(result['evaluation']);predictions.extend(result['predictions']);stages.append(result['stage'])
    for model in S3_MODELS:
        rows={r['row_id']:r for r in predictions if r['model']==model}
        if len(rows)!=len(dev) or set(rows)!={bundle.rows[i]['row_id'] for i in dev}:raise ValueError('OOF tidak lengkap.')
        p=np.array([rows[bundle.rows[i]['row_id']]['prediction'] for i in dev])
        evaluations.append(s3_evaluation(y[dev],p,cfg,model,'DEVELOPMENT_POOLED_OOF','ALL',
            cfg['s3_fixed_alpha'] if model=='S3_RIDGE' else cfg['s2_ridge_alpha'] if model=='S2_RIDGE_FIXED' else None))
    alpha=cfg['s3_fixed_alpha']
    lock=dict(fixed_alpha=alpha,selection_partition='none_fixed_before_run',test_used_for_selection=False,
              selection_metric=None,code_sha256=NOTEBOOK_CODE_SHA256,split_sha256=manifest['split_sha256'],dataset_sha256=bundle.source_sha256)
    dump(run/'alpha_locked_before_test.json',lock)
    stage=fit_s3_representation(bundle,torch.from_numpy(X),cfg,dev,list(range(len(y))),run/'representations',
                                'Representasi 6/6: final development',save_final=True)
    stages.append(dict(kind='FINAL',fold='FINAL',folder=stage['folder'],proof=stage['proof']))
    ridge=Ridge(alpha=alpha,fit_intercept=True,solver='cholesky').fit(stage_values(stage,dev).astype(float),y[dev])
    s2=Ridge(alpha=cfg['s2_ridge_alpha'],fit_intercept=True,solver='cholesky').fit(X[dev].astype(float),y[dev])
    for name,model in [('s3_ridge_final.npz',ridge),('s2_ridge_control.npz',s2)]:
        np.savez_compressed(run/name,coef=np.asarray(model.coef_,dtype=float),intercept=np.asarray(model.intercept_,dtype=float),alpha=np.asarray(model.alpha))
    protocol=json.loads((run/'protocol_before_test.json').read_text())
    final_contract=dict(**lock,representation_checkpoint=str(Path(stage['folder'])/'final_representation.pt'),
        representation_checkpoint_sha256=stage['proof']['model_sha256'],fusion_features_sha256=stage['proof']['features_sha256'],
        BGE=cfg['bge'],RoBERTa=cfg['roberta'],feature_recipe=protocol['representation'],
        s3_ridge_sha256=digest(run/'s3_ridge_final.npz'),s2_ridge_sha256=digest(run/'s2_ridge_control.npz'))
    dump(run/'final_model_contract_before_test.json',final_contract)
    for part,indices in [('TRAIN_FINAL_OPTIMISTIC',dev),('TEST_HISTORICAL_HOLDOUT',test)]:
        values={'S3_RIDGE':ridge.predict(stage_values(stage,indices).astype(float)),
                'S3_MLP_CONTROL':stage_values(stage,indices,'mlp_prediction'),'S2_RIDGE_FIXED':s2.predict(X[indices].astype(float))}
        for model,p in values.items():
            a=alpha if model=='S3_RIDGE' else cfg['s2_ridge_alpha'] if model=='S2_RIDGE_FIXED' else None
            evaluations.append(s3_evaluation(y[indices],p,cfg,model,part,'FINAL',a))
            if part=='TEST_HISTORICAL_HOLDOUT':predictions.extend(s3_predictions(bundle,indices,p,model,part,'FINAL',a))
    result=dict(protocol=protocol,final_contract=final_contract,run_folder=str(run),evaluation=evaluations,
        predictions=predictions,representation_stages=stages,final_features=str(Path(stage['folder'])/'features.npz'),
        final_representation_checkpoint=final_contract['representation_checkpoint'],test_evaluated=True,
        alpha_contract=[dict(model='S3_RIDGE',fixed_alpha=alpha,tuned=False,rationale=protocol['alpha_rationale']),
                        dict(model='S2_RIDGE_FIXED',fixed_alpha=cfg['s2_ridge_alpha'],tuned=False,rationale='Existing S2 baseline')],
        outer_summary=[dict(fold=v['fold'],alpha=alpha,epochs=cfg['epochs'],tuned=False) for v in fold_results])
    dump(run/'s3_ridge_result.json',result)
    return result
