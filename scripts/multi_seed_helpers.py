"""Template fungsi notebook: dependency berasal dari cell AES sebelumnya, bukan import proyek."""

def runtime_code_sha256():
    import types
    # Bandingkan bytecode/konstanta, bukan lokasi file atau identitas objek Python.
    def signature(value):
        if isinstance(value,types.CodeType):
            return {'code':value.co_code.hex(),'args':[value.co_argcount,value.co_posonlyargcount,value.co_kwonlyargcount],
                    'flags':value.co_flags,'names':value.co_names,'variables':value.co_varnames,
                    'free':value.co_freevars,'cells':value.co_cellvars,'constants':[signature(x) for x in value.co_consts]}
        if value is None or isinstance(value,(str,int,bool)):return value
        if isinstance(value,float):return {'float':repr(value)}
        if isinstance(value,complex):return {'complex':repr(value)}
        if isinstance(value,bytes):return {'bytes':value.hex()}
        if isinstance(value,tuple):return {'tuple':[signature(x) for x in value]}
        if isinstance(value,frozenset):return {'frozenset':sorted([signature(x) for x in value],key=lambda x:json.dumps(x,sort_keys=True))}
        if value is Ellipsis:return {'ellipsis':True}
        raise TypeError('Konstanta kode belum didukung: '+type(value).__name__)
    functions={name:globals()[name].__code__ for name in ('config','clean','read_dataset','make_split','encoder','tokenizer','bge_features','batch_windows','mean_by_essay','seed_everything','make_model','optimizer_groups','fit','train_all','metrics','gwet_ac2','ordinal_copy','test_score_comparison','prepare_seed_run','run_three_seeds','initialization_audit','runtime_code_sha256')}
    functions.update({'model_init':SingleScoreModel.__init__.__code__,'model_forward':SingleScoreModel.forward.__code__,
                      'engine_forward':Engine.forward.__code__,'engine_predict':Engine.predict.__code__})
    return hashlib.sha256(json.dumps({name:signature(code) for name,code in functions.items()},sort_keys=True).encode()).hexdigest()


def initialization_audit(model, engine, training, cfg, seed):
    import platform
    modules=[]
    for name,module in model.named_modules():
        row={'name':name,'class':type(module).__name__}
        for key in ('p','eps','normalized_shape','in_features','out_features'):
            if hasattr(module,key):row[key]=getattr(module,key)
        modules.append(row)
    structure={'modules':modules,'parameters':[{'name':n,'shape':list(p.shape),'trainable':p.requires_grad,'dtype':str(p.dtype)} for n,p in model.named_parameters()]}
    architecture=hashlib.sha256(json.dumps(structure,sort_keys=True).encode()).hexdigest()
    head=hashlib.sha256();encoder_sample=hashlib.sha256()
    for name,parameter in model.named_parameters():
        values=parameter.detach().float().cpu().contiguous()
        if name.startswith('roberta.'):
            encoder_sample.update(name.encode());encoder_sample.update(values.reshape(-1)[:32].numpy().tobytes())
        else:
            head.update(name.encode());head.update(values.numpy().tobytes())
    # Dua forward tanpa dropout: memeriksa pembacaan bobot yang sama, tanpa memakai label atau test.
    first=engine.predict([int(training[0])]);second=engine.predict([int(training[0])])
    return {'runtime_code_sha256':runtime_code_sha256(),'scenario':model.scenario,'run_seed':cfg['seed'],'fit_seed':seed,
            'architecture_sha256':architecture,'initial_head_projection_sha256':head.hexdigest(),
            'initial_encoder_sample_sha256':encoder_sample.hexdigest() if model.scenario!='S2' else None,
            'encoder_hash_scope':'32 angka pertama per parameter; bukan hash semua bobot encoder',
            'repeat_eval_max_abs_delta':float(np.max(np.abs(first-second))),
            'torch':str(torch.__version__),'transformers':version('transformers'),'numpy':version('numpy'),
            'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            'device':device_name(cfg),'python':platform.python_version(),
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'cublas_workspace':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'precision':'bfloat16' if device_name(cfg)=='cuda' and torch.cuda.is_bf16_supported() else 'float16' if device_name(cfg)=='cuda' else 'float32',
            'encoder_lr':cfg['s3_encoder_lr'] if model.scenario=='S3' else cfg['encoder_lr'] if model.scenario=='S1' else None,
            'head_lr':cfg['head_lr'],'projection_lr':cfg['projection_lr'] if model.scenario=='S3' else None}


def prepare_seed_run(bundle,manifest,cfg,root,shared_journal,seed):
    import copy
    import shutil
    seed_cfg=config({**copy.deepcopy(cfg),'seed':int(seed)})
    if make_split(bundle,seed_cfg)!=manifest:raise ValueError('Anggota split/fold berubah ketika seed training diganti.')
    output=Path(root)/f'seed_{seed}';output.mkdir(parents=True,exist_ok=True)
    lock={'source_sha256':bundle.source_sha256,'split_sha256':manifest['split_sha256'],
          'code_sha256':NOTEBOOK_CODE_SHA256,'config':seed_cfg}
    path=output/'seed_plan.json'
    if path.exists() and json.loads(path.read_text())!=lock:raise ValueError('Pengaturan seed ini berubah. Gunakan RUN_TAG baru; hasil lama tidak ditimpa.')
    dump(path,lock);dump(output/'effective_config.json',seed_cfg);dump(output/'split_manifest.json',manifest)
    folder=output/'excel';folder.mkdir(exist_ok=True)
    # XLSX yang dibuat ulang bisa berbeda byte karena timestamp, walau isinya sama.
    # Cocokkan salinan lama dengan checksum yang dicatat saat copy, bukan XLSX baru.
    copy_path=output/'shared_excel_files.json'
    copied=json.loads(copy_path.read_text()) if copy_path.exists() else {}
    for source in shared_journal.folder.glob('*.xlsx'):
        if source.name in ('00_indeks_output.xlsx','01_konfigurasi.xlsx'):continue
        target=folder/source.name
        if source.name in copied:
            if not target.exists() or digest(target)!=copied[source.name]:raise ValueError('Salinan Excel bersama rusak: '+str(target))
            continue
        shutil.copyfile(source,target)
        if digest(source)!=digest(target):raise RuntimeError('Penyalinan Excel belum utuh: '+source.name)
        copied[source.name]=digest(target);dump(copy_path,copied)
    journal=ExcelJournal(folder,{**shared_journal.metadata,'run_folder':str(output),'run_seed':seed})
    for key,record in shared_journal.records.items():
        if key!='01_konfigurasi':journal.records.setdefault(key,record)
    journal.save('01_konfigurasi',{'Parameter':pd.DataFrame([{'name':k,'value':v} for k,v in seed_cfg.items()]),
                 'Versi':pd.DataFrame([{'library':p,'version':version(p)} for p in ('torch','transformers','numpy','pandas','scikit-learn','openpyxl')])},
                 'Satu seed; semua model baru dari pretrained dan head baru')
    return {'seed':seed,'cfg':seed_cfg,'output':output,'journal':journal}


def run_three_seeds(bundle,manifest,bge,cfg,root,shared_journal):
    seeds=cfg['training_seeds']
    if len(seeds)!=3 or len(set(seeds))!=3 or any(type(x)is not int or x<0 for x in seeds):raise ValueError('Isi tepat tiga seed integer berbeda.')
    feature_hash=hashlib.sha256(bge.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()
    plan={'scope':SCOPE,'run_tag':cfg['run_tag'],'training_seeds':seeds,'source_sha256':bundle.source_sha256,
          'split_sha256':manifest['split_sha256'],'code_sha256':NOTEBOOK_CODE_SHA256,'bge_features_sha256':feature_hash,
          'runtime_code_sha256':runtime_code_sha256(),'config':cfg,'scenario_policy':'S1 S2 S3 dilatih baru pada setiap seed; tidak ada reuse kontrol run lama',
          'selection':'Tidak memilih seed dari test; semua tiga seed dilaporkan'}
    path=Path(root)/'three_seed_plan.json'
    if path.exists():
        previous=json.loads(path.read_text())
        changed=[key for key in plan if previous.get(key)!=plan[key]]
        if changed:raise ValueError('Rencana tiga seed berubah: '+', '.join(changed)+'. Pilih RUN_TAG baru.')
    dump(path,plan)
    runs=[]
    for position,seed in enumerate(seeds,1):
        print(f'\nSEED {position}/3: {seed}. S1, S2, S3 dimulai dari model baru; split/fold tetap.',flush=True)
        run=prepare_seed_run(bundle,manifest,cfg,root,shared_journal,seed)
        def observer(kind,scenario,result):export_training_event(kind,scenario,result,bundle,run['cfg'],run['journal'])
        report=train_all(bundle,manifest,bge,run['cfg'],run['output'],observer=observer)
        export_completed_training(report,bundle,run['cfg'],run['output'],run['journal'])
        run['report']=report;runs.append(run)
        print(f'Seed {seed} selesai. Hasil disimpan di {run["output"]}',flush=True)
    return runs


def summarize_seed_runs(runs,bundle):
    cv=[];tests=[];folds=[];audit=[];predictions=None;rounded=None
    for run in runs:
        seed=run['seed'];report=run['report'];cfg=run['cfg'];output=run['output']
        for scenario in SCENARIOS:
            selected=[r for r in report['cv'] if r['Skenario']==scenario]
            if len(selected)!=cfg['folds']:raise ValueError('Hasil fold tidak lengkap.')
            row={'Seed':seed,'Skenario':scenario}
            for key,label in [('RMSE','RMSE (poin)'),('Pearson','Pearson'),('QWK','QWK'),('AC2','Gwet AC2')]:
                values=[r[key] for r in selected if r[key] is not None and np.isfinite(r[key])]
                row[label]=float(np.mean(values)) if len(values)==cfg['folds'] else None
            cv.append(row)
            result=next(r for r in report['rows'] if r['Skenario']==scenario)
            tests.append({'Seed':seed,'Skenario':scenario,**{label:result[k] for k,label in [('RMSE','RMSE (poin)'),('Pearson','Pearson'),('QWK','QWK'),('AC2','Gwet AC2')]}})
            for path in sorted((output/scenario).glob('progress_*_audit.json')):
                record=json.loads(path.read_text());audit.append({'Seed':seed,'Skenario':scenario,'Tahap':path.stem.removesuffix('_audit'),
                    'Seed latihan':record['fit_seed'],'Hash arsitektur':record['architecture_sha256'],'Hash kode runtime':record['runtime_code_sha256'],
                    'Hash awal head dan proyeksi':record['initial_head_projection_sha256'],
                    'Hash sampel encoder awal':record['initial_encoder_sample_sha256'],
                    'Selisih dua prediksi eval':record['repeat_eval_max_abs_delta'],
                    'GPU':record['gpu'],'Presisi training':record['precision'],'PyTorch':record['torch']})
        for row in report['cv']:
            folds.append({'Seed':seed,'Skenario':row['Skenario'],'Fold':row['Fold'],**{label:row[k] for k,label in [('RMSE','RMSE (poin)'),('Pearson','Pearson'),('QWK','QWK'),('AC2','Gwet AC2')]}})
        frame=test_score_comparison(report,bundle,cfg,output).set_index('row_id')
        if predictions is None:
            predictions=frame[['skor_guru']].rename(columns={'skor_guru':'Nilai Guru'});rounded=predictions.copy()
        if set(frame.index)!=set(predictions.index) or not np.array_equal(frame.loc[predictions.index,'skor_guru'],predictions['Nilai Guru']):raise ValueError('ID/nilai guru test berbeda antar-seed.')
        for scenario in SCENARIOS:
            predictions[f'{scenario} · seed {seed}']=frame.loc[predictions.index,'Prediksi '+scenario]
            rounded[f'{scenario} · seed {seed}']=frame.loc[rounded.index,'Pembulatan '+scenario]
    def aggregate(rows):
        result=[]
        for scenario in SCENARIOS:
            selected=[r for r in rows if r['Skenario']==scenario];row={'Skenario':scenario,'Jumlah seed':len(selected)}
            for key in ('RMSE (poin)','Pearson','QWK','Gwet AC2'):
                values=[r[key] for r in selected if r[key] is not None and np.isfinite(r[key])]
                row[key+' rata-rata']=float(np.mean(values)) if len(values)==len(selected) else None
                row['SD '+key]=float(np.std(values,ddof=1)) if len(values)==len(selected) and len(values)>1 else None
            result.append(row)
        return pd.DataFrame(result)
    frames={'Rerata CV 3 Seed':aggregate(cv),'Rerata Uji 3 Seed':aggregate(tests),
            'CV per Seed':pd.DataFrame(cv),'Uji per Seed':pd.DataFrame(tests),'Seluruh Fold':pd.DataFrame(folds),
            'Guru dan Prediksi':predictions.reset_index().rename(columns={'row_id':'ID Jawaban'}),
            'Skor Dibulatkan':rounded.reset_index().rename(columns={'row_id':'ID Jawaban'}),'Audit Model dan Seed':pd.DataFrame(audit)}
    if not frames['Audit Model dan Seed'].empty:
        for scenario,group in frames['Audit Model dan Seed'].groupby('Skenario'):
            if group['Hash arsitektur'].nunique()!=1:raise ValueError('Arsitektur berbeda antar-seed pada '+scenario)
    return frames
