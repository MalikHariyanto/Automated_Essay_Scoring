"""Matched training-only plans, no epoch file saves, completed models and exports."""
import ast
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import nbformat
import numpy as np
from openpyxl import load_workbook
import pandas as pd
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from test_e41_and_e30_native import runtime,setup,definitions
from test_e36_notebook import execute_full_native_fixture


def path(number):
    return ROOT/'notebooks/experiments'/f'AES_300_E{number}_E43_S3_PairedBatch_{"MSE" if number==55 else "ConR"}_Native_Learnable_GPU.ipynb'


def env(number):
    return runtime(path(number),f'aes{number}_role')


def test_native_protocol_hash_rebuild_and_unchanged_parent_models():
    parents=[ROOT/'notebooks/experiments/AES_300_E54_E43_S3_ConR_Native_Learnable_GPU.ipynb',
             ROOT/'notebooks/experiments/AES_300_E43_E42_S3_MaskedMeanPooling_Native_Learnable_GPU.ipynb']
    original={p:p.read_bytes() for p in [*parents,path(55),path(56)]}
    for num in [55,56]:
        p=path(num);n=nbformat.read(p,as_version=4);nbformat.validate(n);m=n.metadata.aes_experiment
        assert m.conr_weight==(0 if num==55 else .001) and m.epoch_checkpoints is False
        assert m.scenarios==['S3'] and m.cv_fits==5 and m.refits==m.test_evaluations==1
        assert m.batch_recipe=='same_question_near_far_once_v1' and m.added_parameters_vs_e43==0
        fp=n.metadata.aes_standalone.code_sha256;code=[c for c in n.cells if c.cell_type=='code']
        assert hashlib.sha256('\n'.join(c.source.replace(fp,'__NOTEBOOK_CODE_SHA256__') for c in code).encode()).hexdigest()==fp
        for c in code:
            assert c.metadata[f'aes{num}_role'] and not c.outputs and c.execution_count is None
            assert not any(line.lstrip().startswith(('%','!')) for line in c.source.splitlines())
            assert not any(isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id in ('exec','eval') for x in ast.walk(ast.parse(c.source)))
        text='\n'.join(c.source for c in code)
        assert 'save_training_progress' not in text and 'load_training_progress' not in text and 'progress_refit.json' not in text
        a,b=definitions(parents[0]),definitions(p)
        for name in ['SingleScoreModel','Engine','optimizer_groups','make_split','make_model','same_question_conr','save_model','metrics']:
            assert ast.dump(a[name])==ast.dump(b[name]).replace(f'E{num}','E54'),name
    subprocess.run([sys.executable,'scripts/build_e55_e56_notebooks.py'],cwd=ROOT,check=True,capture_output=True)
    for p,raw in original.items():assert p.read_bytes()==raw
    for num in [55,56]:assert path(num).read_bytes()==(ROOT/'aes-learning/notebooks/experiments'/path(num).name).read_bytes()


def test_plans_identical_every_row_once_no_heldout_or_rng_leakage(tmp_path):
    a,b=env(55),env(56);cfg,bundle=setup(a,tmp_path);other,_=setup(b,tmp_path)
    manifest=a['make_split'](bundle,cfg);ids=manifest['folds'][0]['train']
    for i in ids:bundle.rows[i]['question']='TECHNICAL group '+str(i%2)
    state=random.getstate();torch_state=torch.get_rng_state();np_state=np.random.get_state()
    hashes=[]
    for epoch in [1,2,3]:
        plan,info=a['paired_batch_plan'](ids,bundle,cfg,43,epoch)
        assert b['paired_batch_plan'](ids,bundle,other,43,epoch)==(plan,info)
        flat=[i for batch in plan for i in batch]
        assert sorted(flat)==sorted(ids) and len(flat)==len(set(flat)) and all(0<len(batch)<=8 for batch in plan)
        assert not set(flat)&set(manifest['test']) and not set(flat)&set(manifest['folds'][0]['validation'])
        hashes.append(info['batch_plan_sha256'])
        assert info['batch_eligible_anchors']>0
    assert len(set(hashes))==3 and random.getstate()==state and torch.equal(torch_state,torch.get_rng_state())
    assert np.array_equal(np.random.get_state()[1],np_state[1])
    before=a['paired_batch_plan'](ids,bundle,cfg,43,1)
    for i in set(range(24))-set(ids):
        bundle.y[i]=float('nan');bundle.rows[i]['question']='CHANGED HELDOUT'
    assert a['paired_batch_plan'](ids,bundle,cfg,43,1)==before


def test_sampler_edge_cases_and_coverage_vs_random():
    ns=env(55);cfg=ns['config']({})
    # Twelve prompts, sixteen answers each, with nearby and distant labels.
    bundle=SimpleNamespace(rows=[{'question':f'TECHNICAL q{i//16}'} for i in range(192)],
                           y=np.asarray([65+(i%16)*2 for i in range(192)]))
    ids=list(range(192));plan,info=ns['paired_batch_plan'](ids,bundle,cfg,43,1)
    rng=random.Random(43);control=ids.copy();rng.shuffle(control)
    def eligible(batches):
        count=0
        for batch in batches:
            for i in batch:
                ds=[abs(bundle.y[i]-bundle.y[j]) for j in batch if i!=j and bundle.rows[i]['question']==bundle.rows[j]['question']]
                count+=any(d<=5 for d in ds) and any(d>5 for d in ds)
        return count
    assert info['batch_eligible_anchors']==eligible(plan)>eligible([control[i:i+8] for i in range(0,192,8)])
    for count in [1,2,3,7,8,9,17]:
        tiny=SimpleNamespace(rows=[{'question':f'unique{i}'} for i in range(count)],y=np.arange(count))
        p,d=ns['paired_batch_plan'](list(range(count)),tiny,cfg,1,1)
        assert sorted(i for batch in p for i in batch)==list(range(count)) and d['batch_eligible_anchors']==0
    for bad in [[],[0,0],[-1],[999]]:
        with pytest.raises(ValueError):ns['paired_batch_plan'](bad,bundle,cfg,1,1)


@pytest.mark.parametrize('number',[55,56])
def test_fit_never_saves_epoch_files_best_state_preserved_and_recipe_guards(tmp_path,monkeypatch,number):
    ns=env(number);cfg,bundle=setup(ns,tmp_path);bge=torch.randn(24,1024)
    def forbidden_save(*args,**kwargs):raise AssertionError('fit wrote a checkpoint')
    with monkeypatch.context() as m:
        m.setattr(torch,'save',forbidden_save)
        model,engine,info=ns['fit']('S3',list(range(16)),[16,17],bundle,bge,cfg,2,43)
    assert info['epoch_checkpoints'] is False and info['epochs_completed']==2
    assert not list(tmp_path.rglob('progress*'))
    for epoch,row in enumerate(info['history'],1):
        plan,proof=ns['paired_batch_plan'](list(range(16)),bundle,cfg,43,epoch)
        assert row['batch_plan_sha256']==proof['batch_plan_sha256'] and row['batch_unique_answers']==16
        assert row['batch_eligible_anchors']==row['conr_eligible_anchors']
        assert row['train_combined']==pytest.approx(row['train_mse_scaled']+cfg['conr_weight']*row['train_conr'])
        if number==55:assert row['conr_successful_batches']==0
    assert info['best_epoch']==min(info['history'],key=lambda r:r['validation']['RMSE'])['epoch']
    assert ns['metrics'](bundle.y[[16,17]],engine.predict([16,17]),cfg)['RMSE']==pytest.approx(info['best_validation_rmse'])
    file=tmp_path/'TECHNICAL_final.pt';manifest=ns['make_split'](bundle,cfg)
    ns['save_model'](file,model,cfg,bundle,manifest,info);ns['load_model'](file,cfg)
    for key,value in [('batch_recipe','random'),('epoch_checkpoints',True),('conr_weight',.001 if number==55 else 0.)]:
        with pytest.raises(ValueError):ns['load_model'](file,{**cfg,key:value})


def test_control_only_difference_objective_and_shared_initialization(tmp_path):
    a,b=env(55),env(56);ca,bundle=setup(a,tmp_path);cb,_=setup(b,tmp_path)
    a['seed_everything'](42);ma=a['make_model']('S3',ca).train()
    b['seed_everything'](42);mb=b['make_model']('S3',cb).train()
    for k,v in ma.state_dict().items():torch.testing.assert_close(v,mb.state_dict()[k],rtol=0,atol=0)
    features=torch.randn(24,1024);batch=a['paired_batch_plan'](list(range(24)),bundle,ca,43,1)[0][0]
    values=[]
    for ns,cfg,model in [(a,ca,ma),(b,cb,mb)]:
        engine=ns['Engine'](model,bundle.texts,features,cfg,'cpu',contexts=ns['contexts_from_bundle'](bundle));ns['seed_everything'](123)
        total,mse,conr,counts=ns['training_batch_loss'](engine,batch,bundle,cfg,'cpu',torch.float32,False)
        values.append((total,mse,conr,counts));total.backward()
    torch.testing.assert_close(values[0][1],values[1][1],rtol=0,atol=0)
    torch.testing.assert_close(values[0][2],values[1][2],rtol=0,atol=0)
    assert values[0][3]==values[1][3] and values[0][3]['conr_active_anchors']>0
    torch.testing.assert_close(values[0][0],values[0][1])
    assert not torch.equal(ma.roberta.embeddings.word_embeddings.weight.grad,mb.roberta.embeddings.word_embeddings.weight.grad)


def test_interrupted_between_folds_reuses_completed_fold_and_rejects_changed_model(tmp_path):
    ns=env(55);cfg,bundle=setup(ns,tmp_path);cfg={**cfg,'epochs':1}
    manifest=ns['make_split'](bundle,cfg);bge=torch.randn(24,1024);output=tmp_path/'completed';output.mkdir()
    original=ns['fit'];calls=[]
    def record(*args,**kwargs):
        calls.append(tuple(args[1]));return original(*args,**kwargs)
    ns['fit']=record
    def stop_at(fold):
        def observer(kind,scenario,result):
            if kind=='cv_fold' and result['fold']==fold:raise RuntimeError('TECHNICAL between-fold interruption')
        return observer
    for fold in [1,2]:
        with pytest.raises(RuntimeError,match='between-fold'):
            ns['train_all'](bundle,manifest,bge,cfg,output,observer=stop_at(fold))
    assert calls==[tuple(manifest['folds'][0]['train']),tuple(manifest['folds'][1]['train'])]
    assert not (output/'S3/test_predictions.csv').exists() and not list(output.rglob('progress*'))
    (output/'S3/cv_fold_1.pt').write_bytes(b'TECHNICAL corrupted model')
    with pytest.raises(ValueError,match='Model fold'):
        ns['train_all'](bundle,manifest,bge,cfg,output)
    assert len(calls)==2


@pytest.mark.parametrize('number',[55,56])
def test_full_notebook_twice_completed_models_reports_and_tamper_gate(tmp_path,number):
    ns=env(number);cols=list(ns['prediction_report_tables']([{'ID Jawaban':'TECHNICAL','Nilai Guru':80,'Prediksi S3':79.5}])['Nilai Guru dan Prediksi'][0])
    execute_full_native_fixture(tmp_path,path(number),f'aes{number}_role',5e-6,encoder_layers=4,prediction_columns=cols)
    info=json.loads((tmp_path/'standalone/workspace/outputs/dataset300_download.json').read_text());run=Path(info['run_folder'])
    assert not list(run.rglob('progress*'))
    for fold in range(1,6):
        f=run/f'S3/cv_fold_{fold}.pt';p=torch.load(f,map_location='cpu',weights_only=True)
        assert p['training']['epoch_checkpoints'] is False and p['training']['best_epoch']==1
        r=json.loads((run/f'S3/cv_fold_{fold}.json').read_text())
        assert hashlib.sha256(f.read_bytes()).hexdigest()==r['model_sha256']
    cfg=json.loads((run/'effective_config.json').read_text());bundle=ns['read_dataset'](cfg);manifest=ns['make_split'](bundle,cfg);folder=run/'hasil_ringkas'
    verify=ns[f'validate_e{number}_download'];proof=verify(folder,bundle,manifest,cfg)
    assert proof['test_rows']==6 and proof['oof_rows']==18 and proof['epoch_checkpoint_disabled_verified']
    report=folder/'05_Riwayat_Pelatihan.xlsx';batch=pd.read_excel(report,sheet_name='Rencana Batch');conr=pd.read_excel(report,sheet_name='Diagnostik ConR')
    assert len(batch)==6 and (batch['Jumlah Jawaban']==batch['Jawaban Unik']).all() and not batch['Checkpoint Tiap Epoch'].any()
    np.testing.assert_allclose(conr['Loss Gabungan'],conr['MSE']+cfg['conr_weight']*conr['ConR'])
    if number==55:assert (conr['Langkah ConR Berhasil']==0).all()
    raw=report.read_bytes()
    for name,value in [('Jawaban Unik',0),('SHA256 Rencana','0'*64),('Seed Fit',0),('Checkpoint Tiap Epoch',True)]:
        w=load_workbook(report);s=w['Rencana Batch'];column=next(c.column for c in s[1] if c.value==name);s.cell(2,column,value);w.save(report);w.close()
        with pytest.raises(RuntimeError,match='batch'):verify(folder,bundle,manifest,cfg)
        report.write_bytes(raw)
