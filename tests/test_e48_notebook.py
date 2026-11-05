"""Fold-local LDS objective, preserved E43 model, full native export/resume."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import nbformat
import numpy as np
import pandas as pd
from openpyxl import load_workbook
import pytest
import torch
from scipy.ndimage import convolve1d

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from test_e41_and_e30_native import runtime,setup,definitions
from test_e36_notebook import execute_full_native_fixture

NOTEBOOK=ROOT/'notebooks/experiments/AES_300_E48_E43_S3_LDS_WeightedMSE_Native_Learnable_GPU.ipynb'
BASE=ROOT/'notebooks/experiments/AES_300_E43_E42_S3_MaskedMeanPooling_Native_Learnable_GPU.ipynb'


def test_structure_protocol_and_deterministic_builder():
    old,source=BASE.read_bytes(),NOTEBOOK.read_bytes()
    n=nbformat.read(NOTEBOOK,as_version=4);nbformat.validate(n);m=n.metadata.aes_experiment
    assert m.cv_fits==5 and m.refits==m.test_evaluations==1 and m.scenarios==['S3']
    assert m.lds_training_only and not m.requires_previous_zip and m.loss=='LDS-weighted MSE'
    code=[c for c in n.cells if c.cell_type=='code'];fp=n.metadata.aes_standalone.code_sha256
    assert hashlib.sha256('\n'.join(c.source.replace(fp,'__NOTEBOOK_CODE_SHA256__') for c in code).encode()).hexdigest()==fp
    text='\n'.join(c.source for c in code)
    assert all(token not in text for token in ['CV_SOURCE','reused_cv','WORKER_SOURCE','cosine','ListNet','TRAINING_SEEDS','REFIT_SCHEDULER_EPOCHS'])
    for c in code:
        assert c.metadata.aes48_role and not c.outputs and c.execution_count is None
        assert not any(line.lstrip().startswith(('%','!')) for line in c.source.splitlines())
        assert not any(isinstance(v,ast.Call) and isinstance(v.func,ast.Name) and v.func.id in ['exec','eval'] for v in ast.walk(ast.parse(c.source)))
    a,b=definitions(BASE),definitions(NOTEBOOK)
    for name in ['ScalarMix','SingleScoreModel','AnswerBiGRUCoAttentionMeanPool','AnswerBiGRUAttention','Engine','optimizer_groups',
                 'train_all','metrics','gwet_ac2','make_split','pair_content_windows','pair_batch_windows','encoder','bge_features']:
        assert ast.dump(a[name])==ast.dump(b[name]).replace('E48','E43'),name
    subprocess.run([sys.executable,'scripts/build_e48_notebook.py'],cwd=ROOT,check=True,capture_output=True)
    assert BASE.read_bytes()==old and NOTEBOOK.read_bytes()==source==(ROOT/'aes-learning/notebooks/experiments'/NOTEBOOK.name).read_bytes()


def test_lds_matches_gaussian_inverse_and_never_reads_other_labels(tmp_path):
    ns=runtime(NOTEBOOK,'aes48_role');cfg,bundle=setup(ns,tmp_path)
    bundle.y[:8]=[70,70,70,70,70,70,95,100];training=list(range(8))
    weights,audit=ns['lds_training_weights'](bundle,training,cfg)
    counts=np.bincount(bundle.y[training].astype(int),minlength=101)
    positions=np.arange(-2,3);kernel=np.exp(-.5*(positions/2)**2);kernel/=kernel.sum()
    density=convolve1d(counts.astype(float),kernel,mode='constant')
    expected=1/density[bundle.y[training].astype(int)];expected/=expected.mean()
    np.testing.assert_allclose(list(weights.values()),expected,rtol=1e-12)
    assert np.mean(list(weights.values()))==pytest.approx(1) and weights[6]>weights[0]
    changed=copy.deepcopy(bundle);changed.y[8:]=-999
    changed.rows[8]['text']='HELD OUT MUST NEVER BE READ'
    assert ns['lds_training_weights'](changed,training,cfg)==(weights,audit)
    assert audit['training_count']==8 and len(audit['distribution'])==101
    for indices in [[],[0,0],[-1],[len(bundle.y)]]:
        with pytest.raises(ValueError):ns['lds_training_weights'](bundle,indices,cfg)
    with pytest.raises(ValueError):ns['config']({**cfg,'lds_kernel_size':4})


def test_weighted_objective_preserves_microbatch_weights_and_gradients():
    ns=runtime(NOTEBOOK,'aes48_role')
    p=torch.tensor([[.2],[.3],[.6],[.8]],requires_grad=True);y=torch.tensor([[.7],[.8],[.9],[.95]])
    weights=torch.tensor([[.4],[.4],[1.2],[2.]])
    loss,raw=ns['lds_weighted_mse'](p,y,weights)
    assert raw.item()==pytest.approx(((p-y)**2).mean().item())
    expected=((p-y)**2*weights).mean()
    torch.testing.assert_close(loss,expected)
    gradient=torch.autograd.grad(loss,p,retain_graph=True)[0]
    torch.testing.assert_close(gradient,2*weights*(p-y)/4)
    for micro in [1,2,3,4]:
        accumulated=sum(ns['lds_weighted_mse'](p[s:s+micro],y[s:s+micro],weights[s:s+micro])[0]*len(p[s:s+micro])/4 for s in range(0,4,micro))
        torch.testing.assert_close(accumulated,loss)
        torch.testing.assert_close(torch.autograd.grad(accumulated,p,retain_graph=True)[0],gradient)
    unit,_=ns['lds_weighted_mse'](p,y,torch.ones_like(p));torch.testing.assert_close(unit,raw)
    with pytest.raises(ValueError):ns['lds_weighted_mse'](p,y,weights[:,0])


def test_model_initialization_recipe_and_interrupted_resume(tmp_path):
    ns=runtime(NOTEBOOK,'aes48_role');old=runtime(BASE,'aes43_role');cfg,bundle=setup(ns,tmp_path)
    old['encoder']=ns['encoder'];old['tokenizer']=ns['tokenizer']
    baseline=old['config']({k:v for k,v in cfg.items() if not k.startswith('lds_')})
    ns['seed_everything'](71);model=ns['make_model']('S3',cfg)
    old['seed_everything'](71);control=old['make_model']('S3',baseline)
    for name,p in model.state_dict().items():torch.testing.assert_close(p,control.state_dict()[name],rtol=0,atol=0)
    assert ns['make_split'](bundle,cfg)==old['make_split'](bundle,baseline)
    del model,control
    bge=torch.randn(24,1024);training=list(range(8));path=tmp_path/'progress.json'
    save=ns['save_training_progress']
    def stop(file,state):
        save(file,state)
        if state['epoch']==1:raise RuntimeError('TECHNICAL INTERRUPT')
    ns['save_training_progress']=stop
    with pytest.raises(RuntimeError,match='TECHNICAL INTERRUPT'):ns['fit']('S3',training,None,bundle,bge,cfg,2,42,checkpoint=path)
    ns['save_training_progress']=save
    resumed,_,info=ns['fit']('S3',training,None,bundle,bge,cfg,2,42,checkpoint=path)
    plain,_,other=ns['fit']('S3',training,None,bundle,bge,cfg,2,42)
    assert info['history']==other['history'] and info['lds']==other['lds']
    assert all('train_mse_scaled' in r and 'train_lds_mse_scaled' in r for r in info['history'])
    for name,p in plain.state_dict().items():torch.testing.assert_close(p,resumed.state_dict()[name],rtol=0,atol=0)
    with pytest.raises(ValueError,match='tumpang tindih'):ns['fit']('S3',training,[0],bundle,bge,cfg,2,42)


def test_full_notebook_cv_refit_exports_gate_and_fresh_kernel_resume(tmp_path):
    ns=runtime(NOTEBOOK,'aes48_role')
    columns=list(ns['prediction_report_tables']([{'ID Jawaban':'TECHNICAL','Nilai Guru':80,'Prediksi S3':79.5}])['Nilai Guru dan Prediksi'][0])
    execute_full_native_fixture(tmp_path,NOTEBOOK,'aes48_role',5e-6,encoder_layers=4,prediction_columns=columns)
    info=json.loads((tmp_path/'standalone/workspace/outputs/dataset300_download.json').read_text());run=Path(info['run_folder'])
    cfg=json.loads((run/'effective_config.json').read_text());bundle=ns['read_dataset'](cfg);manifest=ns['make_split'](bundle,cfg)
    folder=run/'hasil_ringkas';proof=ns['validate_e48_download'](folder,bundle,manifest,cfg)
    assert proof['status']=='PASSED' and proof['new_cv_fits']==5 and proof['test_rows']==6 and proof['oof_rows']==18
    lds=pd.read_excel(folder/'05_Riwayat_Pelatihan.xlsx',sheet_name='Bobot LDS Training')
    assert len(lds)==sum(len(f['train']) for f in manifest['folds'])+len(manifest['development'])
    history=pd.read_excel(folder/'05_Riwayat_Pelatihan.xlsx',sheet_name='Fusion (S3)')
    assert history['Loss Training (MSE)'].notna().all() and history['Loss Training (LDS-MSE)'].notna().all()
    cv=pd.read_excel(folder/'05_Riwayat_Pelatihan.xlsx',sheet_name='MAE dan Toleransi CV')
    assert 'Dalam ±3 poin' in cv and 'Selisih Maksimal (poin)' in cv
    f=folder/'04_Prediksi_dan_Evaluasi.xlsx';original=f.read_bytes()
    for title,action in [('Perbandingan Model','empty'),('Ringkasan Validasi','empty'),('Selisih Tepat','count'),('Prediksi Validasi OOF','oof')]:
        w=load_workbook(f);s=w[title]
        if action=='empty':s.delete_rows(2,s.max_row-1)
        elif action=='count':s.cell(2,4,s.cell(2,4).value+1)
        else:s.cell(2,2,'WRONG_OOF_ID')
        w.save(f);w.close()
        with pytest.raises(RuntimeError,match='Unduhan dihentikan'):ns['validate_e48_download'](folder,bundle,manifest,cfg)
        f.write_bytes(original)
    f=folder/'05_Riwayat_Pelatihan.xlsx';original=f.read_bytes();w=load_workbook(f);s=w['Bobot LDS Training']
    s.cell(2,5,s.cell(2,5).value+1);w.save(f);w.close()
    with pytest.raises(RuntimeError,match='bobot LDS'):ns['validate_e48_download'](folder,bundle,manifest,cfg)
    f.write_bytes(original)
    print(json.dumps({'scope':'SYNTHETIC CPU ONLY','full_notebook_twice':True,'fresh_kernel_resume':True,'gate':proof,'corrupted_summary_counts_oof_weights_rejected':True}))
