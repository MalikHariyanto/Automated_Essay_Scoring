"""E30 co-attention fusion with five-fold CV, early stopping and fixed-alpha Ridge.

The builder embeds this runtime together with pinned E30 helpers. No research data.
"""
from sklearn.linear_model import Ridge

S3_MODELS = ('S3_RIDGE', 'S3_MLP_CONTROL', 'S2_RIDGE_FIXED')
S3_BOOKS = ('01_Data_dan_Split.xlsx', '02_Fitur_Fusion_dan_Model.xlsx',
            '03_Prediksi_OOF_dan_Test.xlsx', '04_Evaluasi_dan_Alpha.xlsx',
            '05_Riwayat_dan_Referensi.xlsx')
S3_REFERENCES = [
    dict(component='Ridge', authors='Hoerl & Kennard (1970)',
         title='Ridge Regression: Biased Estimation for Nonorthogonal Problems',
         url='https://doi.org/10.1080/00401706.1970.10488634',
         limit='Dasar penalti L2, bukan bukti S3 Ridge pasti mengungguli MLP atau alpha tertentu optimal.'),
    dict(component='Pemisahan seleksi dan evaluasi', authors='Cawley & Talbot (2010)',
         title='On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation',
         url='https://www.jmlr.org/papers/v11/cawley10a.html',
         limit='Seleksi epoch memakai validation; skor CV bukan evaluasi independen dari seleksi checkpoint. Test historis tetap eksploratif.'),
    dict(component='Pembaca representasi tetap', authors='Conneau & Kiela (2018)',
         title='SentEval: An Evaluation Toolkit for Universal Sentence Representations',
         url='https://aclanthology.org/L18-1269/',
         limit='Prinsip evaluasi embedding downstream; bukan resep khusus BGE-M3 Ridge AES.'),
    dict(component='BGE-M3', authors='Chen et al. (2024)', title='BGE M3-Embedding',
         url='https://arxiv.org/abs/2402.03216', limit='Dasar encoder frozen, bukan head atau parameter optimum AES.'),
    dict(component='RoBERTa', authors='Liu et al. (2019)', title='RoBERTa: A Robustly Optimized BERT Pretraining Approach',
         url='https://arxiv.org/abs/1907.11692', limit='Arsitektur/pretraining; checkpoint Indonesia dan LR diwarisi source E30.'),
    dict(component='AdamW', authors='Loshchilov & Hutter (2019)', title='Decoupled Weight Decay Regularization',
         url='https://arxiv.org/abs/1711.05101', limit='Dasar optimizer representasi, bukan penetapan LR/epoch terbaik untuk data ini.'),
]


def validate_s3_protocol(bundle, bge, manifest, cfg):
    validate_ridge_inputs(bundle, np.asarray(bge), manifest,
                          dict(cfg, ridge_alphas=cfg['s3_ridge_alphas']))
    if not cfg['s3_ridge_alphas'] or cfg['s2_ridge_alpha'] <= 0:
        raise ValueError('Grid S3 dan alpha kontrol S2 harus positif.')
    if type(cfg['epochs']) is not int or cfg['epochs'] < 1:
        raise ValueError('Maksimum epoch wajib positif.')
    if cfg['representation_policy'] != 'E30_five_fold_early_stopping':
        raise ValueError('Notebook ini memakai CV E30 dengan early stopping.')
    if cfg['folds'] != 5 or len(manifest['folds']) != 5:
        raise ValueError('Protokol E30 memakai tepat 5 fold.')
    if cfg['early_stopping_patience'] != 5:
        raise ValueError('Early stopping E30 memakai patience 5.')
    if len(cfg['s3_ridge_alphas']) != 1:
        raise ValueError('Tetapkan satu alpha Ridge sebelum run; tidak ada nested CV.')
    y = np.asarray(bundle.y)
    if not np.isfinite(y).all() or not np.isin(y, np.arange(cfg['score_min'], cfg['score_max']+1)).all():
        raise ValueError('Seluruh skor guru harus kategori sah.')


def representation_contract(bundle, bge, cfg, training, inference, validation=None):
    training, inference = sorted(map(int, training)), sorted(map(int, inference))
    if not training or not inference or len(training) != len(set(training)) or len(inference) != len(set(inference)):
        raise ValueError('Indeks representasi harus unik dan tidak kosong.')
    if not set(training) <= set(inference):
        raise ValueError('Cache fitur harus mencakup train.')
    validation = [] if validation is None else sorted(map(int, validation))
    if set(training) & set(validation) or not set(validation) <= set(inference):
        raise ValueError('Indeks train dan validation harus terpisah.')
    semantic = {k:v for k,v in cfg.items() if k not in ('dataset','output')}
    return dict(protocol='S3_supervised_representation_train_only_v1',
        source_sha256=bundle.source_sha256, code_sha256=NOTEBOOK_CODE_SHA256,
        training=training, inference=inference,
        train_target_sha256=hashlib.sha256(np.asarray(bundle.y[training],dtype=np.float64).tobytes()).hexdigest(),
        bge_sha256=hashlib.sha256(np.asarray(bge,dtype=np.float32).tobytes()).hexdigest(),
        config=semantic, seed=cfg['seed'], epochs=cfg['epochs'], validation_used_for_training=False, validation=validation,
        validation_used_for_epoch_selection=bool(validation),
        validation_target_sha256=hashlib.sha256(np.asarray(bundle.y[validation],dtype=np.float64).tobytes()).hexdigest(),
        torch=str(torch.__version__), device=device_name(cfg),
        gpu=torch.cuda.get_device_name() if device_name(cfg)=='cuda' else None)


def collect_s3_features(model, engine, indices, cfg):
    model.eval()
    arrays = {key:[] for key in ('roberta','roberta_layernorm','projection','concat')}
    predictions = []
    with torch.no_grad():
        for start in range(0,len(indices),cfg['micro_batch']):
            ids=indices[start:start+cfg['micro_batch']]
            p,features=engine.forward(ids,return_features=True)
            for key in arrays:
                arrays[key].append(features[key].detach().float().cpu().numpy())
            predictions.append(p.detach().reshape(-1).float().cpu().numpy() *
                               (cfg['score_max']-cfg['score_min'])+cfg['score_min'])
    arrays={k:np.concatenate(v).astype(np.float32) for k,v in arrays.items()}
    arrays['mlp_prediction']=np.concatenate(predictions).astype(np.float32)
    arrays['indices']=np.asarray(indices,dtype=np.int64)
    if arrays['concat'].shape!=(len(indices),1024) or any(not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('Fitur/prediksi representasi tidak sah.')
    np.testing.assert_array_equal(arrays['concat'], np.concatenate([arrays['roberta_layernorm'],arrays['projection']],axis=1))
    return arrays


def fit_s3_representation(bundle, bge, cfg, training, inference, cache_root, label, save_final=False, validation=None):
    """Train gradients use train only; fold validation selects epoch, never test."""
    contract=representation_contract(bundle,bge,cfg,training,inference,validation)
    key=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()[:20]
    folder=Path(cache_root)/key;folder.mkdir(parents=True,exist_ok=True)
    complete=folder/'complete.json'; features_path=folder/'features.npz'
    if complete.exists():
        proof=json.loads(complete.read_text())
        if proof['contract']!=contract or digest(features_path)!=proof['features_sha256']:
            raise ValueError('Cache representasi tidak utuh atau kontraknya berbeda; tidak ditimpa.')
        if proof['final_model_saved'] != save_final:
            raise ValueError('Cache final dan cache fold tidak boleh ditukar.')
        if save_final and digest(folder/'final_representation.pt')!=proof['model_sha256']:
            raise ValueError('Checkpoint representasi final berubah.')
        with np.load(features_path,allow_pickle=False) as f: arrays={k:f[k].copy() for k in f.files}
        print(label+': fitur utuh dipakai ulang; training tidak diulang.',flush=True)
        return dict(folder=str(folder),arrays=arrays,proof=proof)
    # Train labels drive gradients; fold validation selects epoch; test labels stay masked.
    masked=np.full(len(bundle.y),np.nan,dtype=float);masked[training]=bundle.y[training]
    if validation is not None:
        masked[validation]=bundle.y[validation]
    training_bundle=Bundle(rows=bundle.rows,y=masked,groups=bundle.groups,
                           source_sha256=bundle.source_sha256,summary=bundle.summary)
    checkpoint=folder/'epoch_progress.json'
    rule='patience 5 pada RMSE validasi' if validation is not None else 'epoch dari median CV; tanpa early stopping test'
    print(label+f': train={len(training)}, maksimal {cfg["epochs"]} epoch; '+rule,flush=True)
    model,engine,info=fit('S3',training,validation,training_bundle,bge,cfg,cfg['epochs'],cfg['seed'],checkpoint=checkpoint)
    arrays=collect_s3_features(model,engine,contract['inference'],cfg)
    model_hash=hashlib.sha256()
    for name,t in model.state_dict().items():
        model_hash.update(name.encode());model_hash.update(t.detach().cpu().contiguous().numpy().tobytes())
    model_sha=model_hash.hexdigest()
    if save_final:
        path=folder/'final_representation.pt';temp=path.with_suffix('.pending')
        torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                        contract=contract,info=info),temp);temp.replace(path);model_sha=digest(path)
    temporary=folder/'features.pending.npz';np.savez_compressed(temporary,**arrays);temporary.replace(features_path)
    proof=dict(contract=contract,features_sha256=digest(features_path),model_sha256=model_sha,
               final_model_saved=save_final,info=info,label=label)
    dump(complete,proof)
    # Only this stage's two temporary optimizer snapshots are reclaimed after an intact cache is committed.
    # Interrupted stages retain both snapshots. No older experiment directories are touched.
    if checkpoint.exists():
        progress=json.loads(checkpoint.read_text())
        for snap in progress['snapshots']:
            if Path(snap['file']).name!=snap['file']:raise ValueError('Nama snapshot tidak sah.')
            own_file=folder/snap['file']
            if own_file.is_file():own_file.unlink()
        checkpoint.unlink()
    del engine,model;release()
    return dict(folder=str(folder),arrays=arrays,proof=proof)


def stage_values(stage, indices, key='concat'):
    a=stage['arrays'];lookup={int(v):i for i,v in enumerate(a['indices'])}
    return a[key][[lookup[int(i)] for i in indices]]


def s3_predictions(bundle, indices, p, model, partition, fold, alpha):
    return [dict(model=model,partition=partition,fold=fold,alpha=alpha,row_id=bundle.rows[i]['row_id'],
                 teacher=float(bundle.y[i]),prediction=float(v),error=float(v-bundle.y[i]),
                 absolute_error=float(abs(v-bundle.y[i])),
                 **{f'within{t}':bool(abs(v-bundle.y[i])<=t) for t in (1,2,5,10)}) for i,v in zip(indices,p)]


def s3_evaluation(y,p,cfg,model,partition,fold,alpha):
    return ridge_metric_row(partition,y,p,cfg,model=model,fold=fold,alpha=alpha)


def run_s3_ridge(bundle,bge,manifest,cfg,output):
    X=np.asarray(bge,dtype=np.float32);validate_s3_protocol(bundle,X,manifest,cfg)
    bge_tensor=torch.from_numpy(X);y=np.asarray(bundle.y,dtype=float)
    dev=manifest['development'];test=manifest['test'];folds=manifest['folds']
    run=Path(output);run.mkdir(parents=True,exist_ok=True);cache=run/'representations'
    protocol=dict(experiment='E30_BIGRU_ATTENTION_RIDGE',code_sha256=NOTEBOOK_CODE_SHA256,
        dataset_sha256=bundle.source_sha256,split_sha256=manifest['split_sha256'],config=cfg,
        representation='E30: RoBERTa token respons768 → BiGRU64/arah → co-attention respons-referensi → attention128 → Linear768 → LayerNorm; BGE1024 → Linear256 → LayerNorm; concat1024',
        stage_training='S3 MLP auxiliary supervised representation; 5-fold CV; max25 epochs, patience5; validation selects checkpoint',
        ridge='Post-hoc readout on frozen final fusion features; no gradient through Ridge',
        alpha_grid=cfg['s3_ridge_alphas'],alpha_selection='One alpha declared before CV; no alpha search',
        comparison='MLP control uses the exact same fitted fusion features; control head learned jointly during representation training',
        historical_caveat='E30 epoch selection via fold validation, final refit ceil median best epochs. Historical test is retrospective; CV validation selects checkpoints.',
        unique_representation_fits=len(folds)+1,
        test_used_for_selection=False,BGE_encoder_trained=False)
    dump(run/'protocol_before_test.json',protocol);dump(run/'split_manifest.json',manifest)
    stage_records=[];outer_stages={}
    for f in folds:
        label=f'S3 · Fold {f["fold"]}/{len(folds)}'
        s=fit_s3_representation(bundle,bge_tensor,cfg,f['train'],dev,cache,label,validation=f['validation'])
        outer_stages[f['fold']]=s
        stage_records.append(dict(kind='CV_FOLD',fold=f['fold'],folder=s['folder'],proof=s['proof']))
    evaluations=[];predictions=[];searches=[];inner=[];selected=[]
    oof={model:np.full(len(y),np.nan) for model in S3_MODELS}
    for f in folds:
        tr,va=f['train'],f['validation'];stage=outer_stages[f['fold']]
        alpha=float(cfg['s3_ridge_alphas'][0])
        grid=[dict(stage=f'FOLD_{f["fold"]}',alpha=alpha,selected=True,selection='declared_before_CV')]
        detail=[]
        searches.extend(grid);inner.extend(detail);selected.append(dict(fold=f['fold'],alpha=alpha))
        ridge=Ridge(alpha=alpha,fit_intercept=True,solver='cholesky').fit(stage_values(stage,tr).astype(float),y[tr])
        control=Ridge(alpha=cfg['s2_ridge_alpha'],fit_intercept=True,solver='cholesky').fit(X[tr].astype(float),y[tr])
        values={'S3_RIDGE':ridge.predict(stage_values(stage,va).astype(float)),
                'S3_MLP_CONTROL':stage_values(stage,va,'mlp_prediction'),
                'S2_RIDGE_FIXED':control.predict(X[va].astype(float))}
        for model,p in values.items():
            a=alpha if model=='S3_RIDGE' else cfg['s2_ridge_alpha'] if model=='S2_RIDGE_FIXED' else None
            oof[model][va]=p
            predictions.extend(s3_predictions(bundle,va,p,model,'DEVELOPMENT_OOF',f['fold'],a))
            evaluations.append(s3_evaluation(y[va],p,cfg,model,'DEVELOPMENT_CV_FOLD',f['fold'],a))
    for model,p in oof.items():
        if not np.isfinite(p[dev]).all() or np.isfinite(p[test]).any():raise ValueError('OOF tidak lengkap atau bercampur test.')
        evaluations.append(s3_evaluation(y[dev],p[dev],cfg,model,'DEVELOPMENT_POOLED_OOF','ALL',None))
    final_alpha=float(cfg['s3_ridge_alphas'][0])
    final_epochs=int(math.ceil(np.median([s['proof']['info']['best_epoch'] for s in outer_stages.values()])))
    lock=dict(selected_alpha=final_alpha,selection_partition='declared_before_CV', final_epochs=final_epochs,test_used_for_selection=False,
              selection_metric='none_fixed_alpha',tie_rule='not_applicable',code_sha256=NOTEBOOK_CODE_SHA256,
              split_sha256=manifest['split_sha256'],dataset_sha256=bundle.source_sha256)
    dump(run/'alpha_locked_before_test.json',lock)
    stage=fit_s3_representation(bundle,bge_tensor,dict(cfg,epochs=final_epochs),dev,list(range(len(y))),cache,
                                f'S3 · Refit final development · {final_epochs} epoch dari median CV',save_final=True)
    stage_records.append(dict(kind='FINAL',fold='FINAL',folder=stage['folder'],proof=stage['proof']))
    ridge=Ridge(alpha=final_alpha,fit_intercept=True,solver='cholesky').fit(stage_values(stage,dev).astype(float),y[dev])
    s2=Ridge(alpha=cfg['s2_ridge_alpha'],fit_intercept=True,solver='cholesky').fit(X[dev].astype(float),y[dev])
    for name,model in [('s3_ridge_final.npz',ridge),('s2_ridge_control.npz',s2)]:
        np.savez_compressed(run/name,coef=np.asarray(model.coef_,dtype=float),intercept=np.asarray(model.intercept_,dtype=float),alpha=np.asarray(model.alpha))
    final_features=Path(stage['folder'])/'features.npz'
    final_contract=dict(**lock,representation_checkpoint=str(Path(stage['folder'])/'final_representation.pt'),
        representation_checkpoint_sha256=stage['proof']['model_sha256'],fusion_features_sha256=stage['proof']['features_sha256'],
        BGE=cfg['bge'],RoBERTa=cfg['roberta'],feature_recipe=protocol['representation'],
        s3_ridge_sha256=digest(run/'s3_ridge_final.npz'),s2_ridge_sha256=digest(run/'s2_ridge_control.npz'))
    dump(run/'final_model_contract_before_test.json',final_contract)
    # No test metric or test label was needed to produce either final model.
    for partition,indices in [('TRAIN_FINAL_OPTIMISTIC',dev),('TEST_HISTORICAL_HOLDOUT',test)]:
        values={'S3_RIDGE':ridge.predict(stage_values(stage,indices).astype(float)),
                'S3_MLP_CONTROL':stage_values(stage,indices,'mlp_prediction'),
                'S2_RIDGE_FIXED':s2.predict(X[indices].astype(float))}
        for model,p in values.items():
            a=final_alpha if model=='S3_RIDGE' else cfg['s2_ridge_alpha'] if model=='S2_RIDGE_FIXED' else None
            evaluations.append(s3_evaluation(y[indices],p,cfg,model,partition,'FINAL',a))
            if partition=='TEST_HISTORICAL_HOLDOUT':
                predictions.extend(s3_predictions(bundle,indices,p,model,partition,'FINAL',a))
    result=dict(protocol=protocol,final_contract=final_contract,run_folder=str(run),evaluation=evaluations,
        predictions=predictions,alpha_search=searches,inner_validation=inner,outer_alpha=selected,
        representation_stages=stage_records,final_features=str(final_features),
        final_representation_checkpoint=final_contract['representation_checkpoint'],test_evaluated=True)
    dump(run/'s3_ridge_result.json',result)
    return result


def s3_duels(result):
    rows=[]
    for partition in ('DEVELOPMENT_OOF','TEST_HISTORICAL_HOLDOUT'):
        records=pd.DataFrame([r for r in result['predictions'] if r['partition']==partition])
        for a,b in itertools.combinations(S3_MODELS,2):
            left=records[records.model==a].set_index('row_id');right=records[records.model==b].set_index('row_id').loc[left.index]
            d=left.absolute_error.to_numpy()-right.absolute_error.to_numpy()
            row=dict(partition=partition,left=a,right=b,n=len(d),left_closer=int((d< -1e-8).sum()),
                     right_closer=int((d>1e-8).sum()),ties=int((abs(d)<=1e-8).sum()))
            for t in (1,2,5,10):
                la=left.absolute_error.to_numpy()<=t;ra=right.absolute_error.to_numpy()<=t
                row[f'within{t}_only_left']=int((la&~ra).sum());row[f'within{t}_only_right']=int((ra&~la).sum())
            rows.append(row)
    return rows


def export_s3_ridge(bundle,bge,manifest,cfg,result):
    run=Path(result['run_folder']);out=run/'hasil_excel';out.mkdir(exist_ok=True)
    roles={i:'development' for i in manifest['development']};roles.update({i:'test' for i in manifest['test']})
    data=[dict(row_id=r['row_id'],soal=r['question'],jawaban_referensi=r['reference'],response_text=r['text'],
               rater=float(bundle.y[i]),connected_group=int(bundle.groups[i]),role=roles[i]) for i,r in enumerate(bundle.rows)]
    folds=[dict(fold=f['fold'],row_id=bundle.rows[i]['row_id'],role=role)
           for f in manifest['folds'] for role in ('train','validation') for i in f[role]]
    guide=[dict(Keterangan=k,Nilai=v) for k,v in dict(
        method='S3 feature fusion E30 → Ridge; bukan late fusion dua skor',
        supervision='RoBERTa/norm/proyeksi/MLP belajar train saja; BGE encoder frozen',
        control='MLP kontrol dari checkpoint yang menghasilkan fitur sama; bukan angka lama yang disalin',
        evaluation='Lima fold CV, validation memilih epoch; alpha tetap sebelum run; test setelah model dikunci',
        epochs='Maksimal25, patience5; refit ceil median best epoch lima fold tanpa membuka test',
        alpha='Grid pada fitur concat1024 mentah; alpha tidak disamakan skalanya dengan BGE L2',
        limits='Test historis retrospektif; satu seed bukan pengukuhan multi-seed',
        weights='Checkpoint encoder final tetap lokal di path kontrak; NPZ Ridge saja tidak cukup untuk teks baru',
        privacy='Seluruh hasil berisi data privat; jangan commit ke GitHub').items()]
    with np.load(result['final_features'],allow_pickle=False) as f:features={k:f[k].copy() for k in f.files}
    def frame(values,ids):
        df=pd.DataFrame(values,columns=[f'Komponen {i+1}' for i in range(values.shape[1])]);df.insert(0,'row_id',[bundle.rows[int(i)]['row_id'] for i in ids]);return df
    vectors=dict(Panduan=guide,BGE_Frozen=frame(np.asarray(bge),range(len(bundle.rows))))
    for key,title in [('roberta','RoBERTa_Final'),('roberta_layernorm','RoBERTa_LayerNorm'),('projection','BGE_Proyeksi256'),('concat','Fusion1024_Final')]:
        vectors[title]=frame(features[key],features['indices'])
    with np.load(run/'s3_ridge_final.npz',allow_pickle=False) as f:
        vectors['Ridge_Coef']=pd.DataFrame(dict(component=np.arange(1,1025),coefficient=f['coef']))
        vectors['Ridge_Intercept']=[dict(intercept=float(f['intercept']),alpha=float(f['alpha']))]
    write_readout_workbook(out/S3_BOOKS[0],dict(Panduan=guide,Data=data,Lima_Fold=folds))
    write_readout_workbook(out/S3_BOOKS[1],vectors)
    write_readout_workbook(out/S3_BOOKS[2],dict(Panduan=guide,
        Prediksi_Development_OOF=[v for v in result['predictions'] if v['partition']=='DEVELOPMENT_OOF'],
        Prediksi_Test=[v for v in result['predictions'] if v['partition']=='TEST_HISTORICAL_HOLDOUT']))
    write_readout_workbook(out/S3_BOOKS[3],dict(Panduan=guide,Evaluasi=result['evaluation'],Duel=s3_duels(result),
        Alpha_Tetap=result['alpha_search'],Fold_Alpha=result['outer_alpha']))
    history=[];stages=[]
    for s in result['representation_stages']:
        p=s['proof'];c=p['contract'];info=p['info']
        stages.append(dict(kind=s['kind'],fold=s['fold'],train_rows=len(c['training']),inference_rows=len(c['inference']),
            training_indices=c['training'],model_sha256=p['model_sha256'],features_sha256=p['features_sha256'],
            validation_used_for_training=False,validation_used_for_epoch_selection=c['validation_used_for_epoch_selection'],parameter_samples_changed=info.get('parameter_samples_changed'),
            optimizer_learning_rates=info.get('optimizer_learning_rates')))
        history.extend(dict(stage_kind=s['kind'],fold=s['fold'],**h) for h in info['history'])
    write_readout_workbook(out/S3_BOOKS[4],dict(Panduan=guide,
        Pengaturan=[dict(parameter=k,value=v) for k,v in dict(cfg,**result['final_contract']).items()],
        Representasi=stages,Riwayat=history,Referensi=S3_REFERENCES,
        Versi=[dict(library=k,version=version(k)) for k in ('torch','transformers','numpy','pandas','scikit-learn','openpyxl')]))
    names={name:out/name for name in S3_BOOKS}
    for name in ('s3_ridge_final.npz','s2_ridge_control.npz','split_manifest.json',
                 'alpha_locked_before_test.json','final_model_contract_before_test.json','protocol_before_test.json'):
        names[name]=run/name
    archive=run/'Hasil_AES_300_S3_Fusion_Ridge_Tiga_Input.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,path in names.items():z.write(path,name)
        z.writestr('manifest.json',json.dumps(dict(experiment='E30_BIGRU_ATTENTION_RIDGE',
            dataset_sha256=bundle.source_sha256,code_sha256=NOTEBOOK_CODE_SHA256,
            split_sha256=manifest['split_sha256'],selected_alpha=result['final_contract']['selected_alpha'],
            test_evaluated=True,test_used_for_selection=False,BGE_encoder_trained=False,
            representation_fits=result['protocol']['unique_representation_fits'],
            full_representation_checkpoint_included=False,
            files={name:digest(path) for name,path in names.items()}),ensure_ascii=False,indent=2))
    return dict(archive=str(archive),books=list(S3_BOOKS),run_folder=str(run),
                final_representation_checkpoint=result['final_representation_checkpoint'],test_evaluated=True)
