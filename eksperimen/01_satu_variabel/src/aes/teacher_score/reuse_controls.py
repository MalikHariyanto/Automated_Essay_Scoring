"""Reuse S1/S2 selesai dari run tepercaya; tidak menghapus atau menulis sumber."""
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from . import SCOPE
from .data import digest, dump
from .experiment import load_model
from .models import release

REUSE_SOURCE_PROFILES=[
    {'name':'MLP acuan','lesson_version':'s3_mlp_account2_v1',
     'code_sha256':'49f80d493aa531197b6a01e57fcb80e51ff18fb4f2ecda09abfc1f37814f1da7'},
    {'name':'LoRA, hanya kontrol S1/S2','lesson_version':'s3_lora_mlp_gpu_v1',
     'code_sha256':'86bcfd7dff74cf2a47517244cadbc12b2c8e6b61e2345da27781f9bd2368b9f5'}]


def inspect_reuse_runtime(old, cfg):
    """Checkpoint final berisi state_dict; bukan resume optimizer lintas versi."""
    from packaging.version import InvalidVersion, Version
    source=old.get('runtime_versions',{})
    current=cfg.get('runtime_versions',{})
    for key in ('transformers','numpy'):
        if not source.get(key) or source[key]!=current.get(key):
            raise ValueError('Versi numerik sumber berbeda: '+key)
    # Build CUDA/CPU, patch dan minor Torch boleh berbeda untuk final state_dict.
    # Keutuhan dan kecocokan layer tetap wajib melalui weights_only + strict=True.
    for label,versions in (('sumber',source),('aktif',current)):
        try:
            value=Version(versions.get('torch',''))
        except (InvalidVersion,TypeError) as error:
            raise ValueError('Versi PyTorch '+label+' tidak valid.') from error
        if value.major!=2 or value.minor<6 or value.is_prerelease or value.is_devrelease:
            raise ValueError('Reuse final memerlukan PyTorch stabil 2.x mulai 2.6: '+label)
    return {'source':dict(source),'current':dict(current),
            'torch_changed':source['torch']!=current['torch'],
            'policy':'final_state_dict_cpu_strict_load; no_optimizer_resume'}


def inspect_control_source(source, bundle, manifest, cfg):
    """Validasi metode/config, split, lima fold, final dan hasil uji S1/S2."""
    source=Path(source)
    old=json.loads((source/'effective_config.json').read_text())
    report=json.loads((source/'final_report.json').read_text())
    if report['scope']!=SCOPE or report['source_sha256']!=bundle.source_sha256 or report['split_sha256']!=manifest['split_sha256']:
        raise ValueError('Dataset atau split sumber S1/S2 berbeda.')
    semantic={k:v for k,v in old.items() if k not in ('dataset','output','trace_position','vector_source','new_answer','focus_word')}
    profile=None
    for item in REUSE_SOURCE_PROFILES:
        key=hashlib.sha256(json.dumps({'source':bundle.source_sha256,'config':semantic,'code':item['code_sha256']},sort_keys=True).encode()).hexdigest()[:16]
        if source.name==key and old.get('lesson_version')==item['lesson_version']:
            profile=item['name'];break
    if profile is None:
        raise ValueError('Identitas source bukan MLP acuan atau LoRA yang didukung untuk reuse S1/S2.')
    keys=('sheet','expected_rows','score_min','score_max','test_fraction','folds','seed','split_seed','split_candidates',
          'epochs','early_stopping_patience','effective_batch','micro_batch','encoder_lr','projection_lr','head_lr',
          'weight_decay','warmup_ratio','gradient_clip','dropout','roberta_max_length','roberta_overlap',
          'bge_max_length','bge_batch','roberta','bge')
    for key in keys:
        if old.get(key)!=cfg.get(key):raise ValueError('Pengaturan sumber S1/S2 berbeda: '+key)
    # S3/LoRA/source lesson berbeda memang disengaja; metode S1/S2 kedua profile sama.
    runtime=inspect_reuse_runtime(old,cfg)
    files={};epochs={};training={}
    for scenario in ('S1','S2'):
        best=[]
        for fold in manifest['folds']:
            relative=f'{scenario}/cv_fold_{fold["fold"]}.json'
            result=json.loads((source/relative).read_text())
            if result['fold']!=fold['fold'] or result['split_sha256']!=manifest['split_sha256'] or result['validation_indices']!=fold['validation']:
                raise ValueError('Fold sumber berbeda: '+relative)
            if len(result['prediction'])!=len(fold['validation']):raise ValueError('Prediksi CV sumber tidak lengkap.')
            original=next(r for r in report['cv'] if r['Skenario']==scenario and r['Fold']==fold['fold'])
            if any(original[k]!=result['metrics'][k] for k in ('RMSE','Pearson','QWK','AC2')):
                raise ValueError('Metrik CV berbeda dari laporan sumber.')
            best.append(result['training']['best_epoch']);files[relative]=digest(source/relative)
        epochs[scenario]=int(math.ceil(np.median(best)))
        if report['selected_epochs'][scenario]!=epochs[scenario]:raise ValueError('Epoch final sumber tidak cocok dengan CV.')
        checkpoint=source/scenario/'final.pt'
        if digest(checkpoint)!=report['checkpoints'][scenario]:raise ValueError('Checksum model final sumber berubah.')
        model,payload=load_model(checkpoint,cfg)
        if payload['scenario']!=scenario or payload['source_sha256']!=bundle.source_sha256 or payload['split_sha256']!=manifest['split_sha256']:
            raise ValueError('Metadata model final sumber berbeda.')
        training[scenario]=payload['training']
        if training[scenario]['epochs_completed']!=epochs[scenario]:raise ValueError('Refit sumber belum lengkap.')
        rates=training[scenario]['optimizer_learning_rates']
        if rates.get('head')!=cfg['head_lr'] or (scenario=='S1' and rates.get('roberta')!=cfg['encoder_lr']):
            raise ValueError('Learning rate aktual kontrol sumber berbeda.')
        del model,payload
        release()
        result=json.loads((source/scenario/'test_result.json').read_text())
        predictions=source/scenario/'test_predictions.csv'
        if (result['checkpoint_sha256']!=report['checkpoints'][scenario]
                or result['split_sha256']!=manifest['split_sha256']
                or digest(predictions)!=result['prediction_sha256']
                or digest(predictions)!=report['prediction_files'][scenario]
                or result['row']!=next(r for r in report['rows'] if r['Skenario']==scenario)):
            raise ValueError('Hasil test kontrol sumber tidak cocok.')
        frame=pd.read_csv(predictions)
        ids=[bundle.rows[i]['row_id'] for i in manifest['test']]
        if frame['row_id'].tolist()!=ids or not np.array_equal(frame['skor_guru'].to_numpy(),bundle.y[manifest['test']]):
            raise ValueError('ID atau nilai guru pada prediksi test sumber berbeda.')
        for relative in (f'{scenario}/final.pt',f'{scenario}/test_result.json',f'{scenario}/test_predictions.csv'):
            files[relative]=digest(source/relative)
    return {'scope':SCOPE,'source_run':str(source),'profile':profile,'source_sha256':bundle.source_sha256,
            'split_sha256':manifest['split_sha256'],'files':files,'selected_epochs':epochs,
            'training':training,'source_device':old['device'],'runtime_compatibility':runtime,
            'source_report_sha256':digest(source/'final_report.json')}


def prepare_reused_controls(bundle, manifest, cfg, output):
    """Kunci satu sumber berdasarkan urutan folder/tanggal, tanpa memilih dari skor."""
    import shutil
    output=Path(output);plan_path=output/'s1_s2_reuse.json'
    if plan_path.exists():
        plan=json.loads(plan_path.read_text())
        if plan.get('scope')!=SCOPE or plan['source_sha256']!=bundle.source_sha256 or plan['split_sha256']!=manifest['split_sha256']:
            raise ValueError('Rencana reuse tersimpan berbeda data/split.')
    else:
        if str(cfg.get('reuse_run','')).strip():
            candidates=[Path(cfg['reuse_run']).expanduser()]
        else:
            # output = MyDrive/<folder eksperimen>/outputs/dataset300_inline/<hash>
            drive_root=output.parents[3]
            candidates=[]
            for folder in ('AES_300_MLP_Privat','AES_300_S3_LoRA_MLP_Privat'):
                parent=drive_root/folder/'outputs/dataset300_inline'
                if parent.is_dir():
                    complete=[p for p in parent.iterdir() if p.is_dir() and (p/'final_report.json').is_file()]
                    candidates.extend(sorted(complete,key=lambda p:(-(p/'final_report.json').stat().st_mtime,str(p))))
        failures=[];plan=None
        for candidate in candidates:
            if candidate.resolve()==output.resolve():continue
            try:
                plan=inspect_control_source(candidate,bundle,manifest,cfg);break
            except (OSError,ValueError,KeyError,StopIteration,RuntimeError) as error:
                failures.append(str(candidate)+': '+str(error))
        if plan is None:
            detail='\n'.join(failures[-4:]) or 'Tidak ada folder sumber lengkap pada lokasi default.'
            raise RuntimeError('S1/S2 yang cocok belum ditemukan. Isi REUSE_RUN dengan folder run lama berisi effective_config.json, final_report.json, S1 dan S2. Training tidak dimulai.\n'+detail)
        plan['state']='copy_pending'
        # Rencana dikunci sebelum copy agar interupsi tidak memilih sumber lain.
        dump(plan_path,plan)
    allowed={f'{scenario}/cv_fold_{fold["fold"]}.json' for scenario in ('S1','S2') for fold in manifest['folds']}
    allowed.update(f'{scenario}/{name}' for scenario in ('S1','S2') for name in ('final.pt','test_result.json','test_predictions.csv'))
    if set(plan['files'])!=allowed or Path(plan['source_run']).resolve()==output.resolve():
        raise ValueError('Daftar file atau folder rencana reuse tidak sah.')
    for relative,checksum in plan['files'].items():
        target=output/relative
        if target.is_file():
            if digest(target)!=checksum:raise ValueError('Hasil kontrol tujuan berbeda dari sumber terkunci: '+relative)
            continue
        origin=Path(plan['source_run'])/relative
        if digest(origin)!=checksum:raise ValueError('Sumber berubah sebelum penyalinan: '+relative)
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_name(target.name+'.copy_pending')
        shutil.copyfile(origin,temporary)
        if digest(temporary)!=checksum:raise ValueError('Penyalinan tidak utuh: '+relative)
        temporary.replace(target)
    plan['state']='complete';dump(plan_path,plan)
    print('S1/S2 memakai hasil selesai dari:',plan['source_run'],flush=True)
    print('S1/S2 tidak dilatih atau diuji ulang. Hanya S3 kandidat LR baru yang dilatih.',flush=True)
    runtime=plan.get('runtime_compatibility',{})
    if runtime.get('torch_changed'):
        print('Model final S1/S2 berhasil dimuat di CPU dengan layer yang cocok. PyTorch sumber: '
              +runtime['source']['torch']+'; aktif: '+runtime['current']['torch']+'. '
              'Metrik S1/S2 tetap hasil run sumber; optimizer lama tidak dilanjutkan.',flush=True)
    return plan
