import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from .io import load_data,load_rubric,targets,sha256,dump
from .audit import validate_inputs_for_training
from .cleaning import clean
from .split import make_manifest
from .encoders import resolve_models,BGEEncoder,cached_bge
from .trainer import train_with_oom_retry,device_name,BatchEngine
from .checkpoint import load_checkpoint
from .metrics import evaluate


def summarize(results,rubric):
    rows=[]
    for scenario,runs in results.items():
        for trait in rubric['traits']:
            for metric in ['RMSE','Pearson','QWK','AC2']:
                values=[r['metrics'][trait['name']][metric] for r in runs if r['metrics'] is not None and r['metrics'][trait['name']][metric] is not None]
                rows.append({'scenario':scenario,'trait':trait['name'],'metric':metric,'completed_runs':len(runs),
                   'n_valid':len(values),'mean':float(np.mean(values)) if values else None,
                   'sample_sd':float(np.std(values,ddof=1)) if len(values)>1 else None})
    return rows

def prepare(cfg):
    df=load_data(cfg['csv']);r=load_rubric(cfg['rubric']);review=json.loads(Path(cfg['split']['review_file']).read_text())
    y=validate_inputs_for_training(df,r,review,sha256(cfg['csv']))
    out=Path(cfg['output']);manifest_path=out/'manifest.json'
    manifest=make_manifest(df,r,review,cfg)
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:
        raise ValueError('Manifest lama berbeda; pilih output baru agar eksperimen tidak tercampur')
    dump(manifest_path,manifest)
    resolved_path=out/'model_revisions.json'
    resolved=json.loads(resolved_path.read_text()) if resolved_path.exists() else resolve_models(cfg)
    for role,spec in cfg['models'].items():
        if resolved[role]['checkpoint']!=spec['checkpoint'] or (spec['revision'] and spec['revision']!=resolved[role]['revision']):
            raise ValueError('Model revision tidak cocok manifest')
    dump(resolved_path,resolved)
    texts=[clean(t,**cfg['preprocessing']) for t in df.text_raw]
    encoder=BGEEncoder(resolved['bge'],device_name());bge,_=cached_bge(df,texts,encoder,cfg);del encoder
    index={i:j for j,i in enumerate(df.essay_id)}
    return df,r,y,texts,bge,resolved,manifest,index

def run(cfg,mode='learning',scenarios=('S1','S2','S3')):
    df,r,y,texts,bge,resolved,manifest,index=prepare(cfg);out=Path(cfg['output']);all_results={};final_runs={}
    def positions(ids):return [index[i] for i in ids]
    f0=manifest['folds'][0]
    for s in scenarios:
        base=out/'runs'/mode/s
        if mode=='learning':
            result=train_with_oom_retry(scenario=s,candidate='A',train_ids=positions(f0['train']),valid_ids=positions(f0['validation']),
               texts=texts,y=y,bge=bge,resolved=resolved,cfg=cfg,rubric=r,seed=f0['seed'],out=base,epochs=1)
            all_results[s]=[result];continue
        candidates=[]
        for c in ['A','B']:
            candidates.append(train_with_oom_retry(scenario=s,candidate=c,train_ids=positions(f0['train']),valid_ids=positions(f0['validation']),
                texts=texts,y=y,bge=bge,resolved=resolved,cfg=cfg,rubric=r,seed=f0['seed'],out=base/'tuning'/c))
        # Jika satu kandidat menggunakan fallback, bandingkan keduanya pada MSE, bukan unit campuran.
        fallback=any(v['monitor_kind']=='validation_mse' for v in candidates)
        selected=min(candidates,key=lambda v:v['best_validation_mse']) if fallback else max(candidates,key=lambda v:v['best_monitor'])
        choice=selected['candidate'];dump(base/'locked_config.json',{'candidate':choice,'tuning_fold':'repeat1-fold1',
                 'comparison':'validation_mse' if fallback else 'mean_qwk_all_four','test_used':False})
        results=[]
        for f in manifest['folds']:
            result=train_with_oom_retry(scenario=s,candidate=choice,train_ids=positions(f['train']),valid_ids=positions(f['validation']),
                texts=texts,y=y,bge=bge,resolved=resolved,cfg=cfg,rubric=r,seed=f['seed'],out=base/f"repeat{f['repeat']}-fold{f['fold']}")
            results.append({**result,'repeat':f['repeat'],'fold':f['fold']});dump(base/'cv_results.json',results)
        all_results[s]=results
        final_epochs=max(1,min(5,math.ceil(np.median([v['best_epoch'] for v in results]))))
        final=train_with_oom_retry(scenario=s,candidate=choice,train_ids=positions(manifest['development']),valid_ids=None,
            texts=texts,y=y,bge=bge,resolved=resolved,cfg=cfg,rubric=r,seed=42,out=base/'final',epochs=final_epochs,early_stop=False)
        final_runs[s]={'result':final,'epochs':final_epochs}
        lock_path=base/'locked_config.json';lock=json.loads(lock_path.read_text())
        lock.update({'final_epochs':final_epochs,'final_seed':42,'final_refit_completed':True})
        dump(lock_path,lock)
    # Semua skenario selesai memilih konfigurasi dan refit sebelum skor test dilaporkan.
    if mode=='proposal':
        for s in scenarios:
            base=out/'runs'/mode/s;final=final_runs[s]['result'];final_epochs=final_runs[s]['epochs']
            model,payload=load_checkpoint(final['checkpoint'],device_name(),cfg['rubric'],cfg['preprocessing'])
            engine=BatchEngine(model,texts,bge,resolved,cfg,device_name());test_idx=positions(manifest['test'])
            pred=engine.predict(test_idx,cfg['training']['micro_batch']);test_metrics=evaluate(y[test_idx],pred,r)
            dump(base/'final'/'test_metrics.json',{'configuration_locked':True,'all_requested_scenarios_locked':list(scenarios),
                 'metrics':test_metrics,'refit_epochs':final_epochs,
                 'label_distribution':{t['name']:{str(c):int((y[test_idx,j]==c).sum()) for c in t['categories']} for j,t in enumerate(r['traits'])}})
            table=pd.DataFrame(pred,columns=['pred_'+t['name'] for t in r['traits']]);table.insert(0,'essay_id',df.iloc[test_idx].essay_id.to_numpy());table.to_csv(base/'final'/'test_predictions.csv',index=False)
            del model,engine
    dump(out/f'{mode}_summary.json',summarize(all_results,r))
    pd.DataFrame(summarize(all_results,r)).to_csv(out/f'{mode}_summary.csv',index=False)
    return all_results
