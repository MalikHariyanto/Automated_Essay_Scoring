"""Refit-only notebooks: verified CV reuse, exact schedule, and download gate."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
import numpy as np
from openpyxl import load_workbook
import pandas as pd
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from test_e41_and_e30_native import runtime,setup,definitions
from test_e36_notebook import execute_full_native_fixture
from test_e45_notebook import twelve_layer_encoder

SPECS={
 46:dict(parent=ROOT/'notebooks/experiments/AES_300_E43_E42_S3_MaskedMeanPooling_Native_Learnable_GPU.ipynb',
         path=ROOT/'notebooks/experiments/AES_300_E46_E43_S3_Refit18_Schedule25_Native_Learnable_GPU.ipynb',
         base='E43',role='aes46_role',old_role='aes43_role',layers=4,epochs=18),
 47:dict(parent=ROOT/'notebooks/experiments/AES_300_E45_E43_S3_ScalarMix12_Native_Learnable_GPU.ipynb',
         path=ROOT/'notebooks/experiments/AES_300_E47_E45_S3_Refit7_Schedule25_Native_Learnable_GPU.ipynb',
         base='E45',role='aes47_role',old_role='aes45_role',layers=12,epochs=7)}


def test_structure_protocol_and_deterministic_builder():
    old={s['parent']:s['parent'].read_bytes() for s in SPECS.values()}
    original={s['path']:s['path'].read_bytes() for s in SPECS.values()}
    for number,s in SPECS.items():
        n=nbformat.read(s['path'],as_version=4);nbformat.validate(n);m=n.metadata.aes_experiment
        assert m.scenarios==['S3'] and m.cv_fits==0 and m.cv_fits_reused==5 and m.refit_only
        assert m.refits==m.test_evaluations==1 and not m.new_cv_results
        assert m.refit_epochs==s['epochs'] and m.scheduler_horizon_epochs==25
        assert m.download_validation_required and m.export_summary_closure_fixed
        assert not m.multiple_seed_runs and not m.control_retrained and not m.ranking
        fp=n.metadata.aes_standalone.code_sha256;code=[c for c in n.cells if c.cell_type=='code']
        assert hashlib.sha256('\n'.join(c.source.replace(fp,'__NOTEBOOK_CODE_SHA256__') for c in code).encode()).hexdigest()==fp
        assert hashlib.sha256(s['parent'].read_bytes()).hexdigest()==m.format_sha256
        for c in code:
            assert c.metadata[s['role']] and not c.outputs and c.execution_count is None
            assert not any(l.lstrip().startswith(('%','!')) for l in c.source.splitlines())
            assert not any(isinstance(v,ast.Call) and isinstance(v.func,ast.Name) and v.func.id in ['exec','eval']
                           for v in ast.walk(ast.parse(c.source)))
        text='\n'.join(c.source for c in code)
        assert all(x not in text for x in ['cosine','WORKER_SOURCE','TRAINING_SEEDS','ListNet'])
        a,b=definitions(s['parent']),definitions(s['path'])
        for name in ['ScalarMix','SingleScoreModel','AnswerBiGRUCoAttentionMeanPool','optimizer_groups',
                     'Engine','metrics','gwet_ac2','make_split','pair_batch_windows','encoder']:
            assert ast.dump(a[name])==ast.dump(b[name]).replace('E'+str(number),s['base']),name
    subprocess.run([sys.executable,'scripts/build_e46_e47_notebooks.py'],cwd=ROOT,check=True,capture_output=True)
    assert all(p.read_bytes()==raw for p,raw in {**old,**original}.items())
    for s in SPECS.values():assert s['path'].read_bytes()==(ROOT/'aes-learning/notebooks/experiments'/s['path'].name).read_bytes()


def test_scheduler_matches_cv_at_seven_and_eighteen_epochs():
    ns=runtime(SPECS[46]['path'],'aes46_role')
    for steps,epochs,expected in [(24,7,.8),(30,7,.8),(30,18,14/45)]:
        p=torch.nn.Parameter(torch.tensor(1.));optimizer=torch.optim.AdamW([p],lr=1e-4)
        total=steps*25;warmup=int(.1*total)
        scheduler=ns['get_linear_schedule_with_warmup'](optimizer,warmup,total)
        for _ in range(steps*epochs):optimizer.step();scheduler.step()
        assert optimizer.param_groups[0]['lr']==pytest.approx(1e-4*expected)


@pytest.mark.parametrize('number',[46,47])
def test_fit_refit_only_horizon_and_interrupted_resume(tmp_path,number):
    s=SPECS[number];ns=runtime(s['path'],s['role']);cfg,bundle=setup(ns,tmp_path)
    if number==47:ns['encoder']=twelve_layer_encoder
    cfg={**cfg,'expected_refit_epochs':2};bge=torch.randn(24,1024);training=list(range(8))
    with pytest.raises(ValueError,match='hanya refit'):
        ns['fit']('S3',training,[8,9],bundle,bge,cfg,2,42)
    original=ns['save_training_progress'];path=tmp_path/'progress.json'
    def stop(file,state):
        original(file,state)
        if state['epoch']==1:raise RuntimeError('TECHNICAL INTERRUPT')
    ns['save_training_progress']=stop
    with pytest.raises(RuntimeError,match='TECHNICAL INTERRUPT'):
        ns['fit']('S3',training,None,bundle,bge,cfg,2,42,checkpoint=path)
    ns['save_training_progress']=original
    resumed,_,info=ns['fit']('S3',training,None,bundle,bge,cfg,2,42,checkpoint=path)
    control,_,control_info=ns['fit']('S3',training,None,bundle,bge,cfg,2,42)
    assert info['history']==control_info['history']
    assert info['scheduler_horizon_epochs']==25 and info['scheduler_total_steps']==25 and info['warmup_steps']==2
    assert len(info['history'])==2 and all(v>0 for v in info['history'][-1]['scheduled_learning_rates_end'].values())
    for key,value in control.state_dict().items():torch.testing.assert_close(value,resumed.state_dict()[key],rtol=0,atol=0)


@pytest.fixture(scope='module',params=[46,47])
def old_cv_fixture(request,tmp_path_factory):
    number=request.param;s=SPECS[number];folder=tmp_path_factory.mktemp('old_cv_'+s['base'])
    execute_full_native_fixture(folder,s['parent'],s['old_role'],5e-6,encoder_layers=s['layers'])
    info=json.loads((folder/'standalone/workspace/outputs/dataset300_download.json').read_text())
    run=Path(info['run_folder']);ns=runtime(s['path'],s['role'])
    old=json.loads((run/'effective_config.json').read_text())
    cfg=ns['config']({**old,'cv_source_path':info['archive'],'cv_archive_sha256':ns['digest'](info['archive']),
        'cv_source_experiment':s['base'],'expected_refit_epochs':1,'refit_scheduler_epochs':1})
    bundle=ns['read_dataset'](cfg);manifest=ns['make_split'](bundle,cfg)
    return dict(number=number,spec=s,folder=folder,info=info,ns=ns,cfg=cfg,bundle=bundle,manifest=manifest)


def test_verification_rejects_mismatch_and_never_reads_old_test_results(old_cv_fixture,monkeypatch):
    x=old_cv_fixture;ns=x['ns'];cfg=x['cfg'];bundle=x['bundle'];manifest=x['manifest'];path=cfg['cv_source_path']
    original=ns['cv_archive_records'];read=[]
    def observe(*args):read.append(args[1:]);return original(*args)
    monkeypatch.setitem(ns,'cv_archive_records',observe)
    reused=ns['verify_reused_cv'](path,bundle,manifest,cfg)
    assert reused['selected_epoch']==1 and len(reused['cv_rows'])==5
    assert not any(sheet in ['Perbandingan Model','Nilai Guru dan Prediksi','Prediksi Satu Jawaban'] for _,sheet in read)
    with pytest.raises(ValueError,match='SHA256'):
        ns['verify_reused_cv'](path,bundle,manifest,{**cfg,'cv_archive_sha256':'0'*64})
    changed=copy.deepcopy(bundle);changed.rows[0]['text']+=' altered'
    with pytest.raises(ValueError,match='Teks'):
        ns['verify_reused_cv'](path,changed,manifest,cfg)
    changed=copy.deepcopy(manifest);changed['folds'][0]['validation'][0]=changed['test'][0]
    with pytest.raises(ValueError,match='Anggota lima fold'):
        ns['verify_reused_cv'](path,bundle,changed,cfg)
    for extra,match in [({'s3_mlp_lr':.001},'Pengaturan'),({'expected_refit_epochs':2},'Median epoch')]:
        with pytest.raises(ValueError,match=match):ns['verify_reused_cv'](path,bundle,manifest,{**cfg,**extra})
    def missing_history(file,workbook,sheet):
        r=original(file,workbook,sheet)
        return r[1:] if sheet=='Fusion (S3)' else r
    monkeypatch.setitem(ns,'cv_archive_records',missing_history)
    with pytest.raises(ValueError,match='History'):
        ns['verify_reused_cv'](path,bundle,manifest,cfg)


def execute_refit_fixture(tmp_path,x):
    s=x['spec'];n=nbformat.read(s['path'],as_version=4)
    isolated=tmp_path/'standalone';isolated.mkdir();workspace=isolated/'workspace'
    source=nbformat.read(x['folder']/'standalone/TECHNICAL_executed.ipynb',as_version=4)
    technical_tokenizer=next(c.source for c in source.cells if c.metadata.get(s['old_role'])=='definition:tokenizer')
    for c in n.cells:
        if c.cell_type!='code':continue
        role=c.metadata[s['role']]
        if role=='hyperparameters':
            replacements={'EPOCHS = 25':'EPOCHS = 1',"DEVICE = 'cuda'":"DEVICE = 'cpu'",'REQUIRE_GPU = True':'REQUIRE_GPU = False',
                'ROBERTA_MAX_TOKENS = 512':'ROBERTA_MAX_TOKENS = 32','BGE_MAX_TOKENS = 1024':'BGE_MAX_TOKENS = 32',
                'WINDOW_OVERLAP = 128':'WINDOW_OVERLAP = 4','CONTEXT_MAX_TOKENS = 192':'CONTEXT_MAX_TOKENS = 12',
                'QUESTION_MAX_TOKENS = 64':'QUESTION_MAX_TOKENS = 4',
                f"EXPECTED_REFIT_EPOCHS = {s['epochs']}":'EXPECTED_REFIT_EPOCHS = 1'}
            for a,b in replacements.items():assert a in c.source;c.source=c.source.replace(a,b)
            c.source=__import__('re').sub(r"CV_SOURCE_SHA256 = '[0-9a-f]{64}'",f"CV_SOURCE_SHA256 = '{x['cfg']['cv_archive_sha256']}'",c.source)
        if role=='input_columns':c.source=c.source.replace('EXPECTED_ROWS = 300','EXPECTED_ROWS = 24')
        if role=='definition:tokenizer':c.source=technical_tokenizer
    fit_index=next(i for i,c in enumerate(n.cells) if c.metadata.get(s['role'])=='definition:fit')
    n.cells.insert(fit_index+1,nbformat.v4.new_code_cell('''_fit_calls=[]
_actual_fit=fit
def fit(*args,**kwargs):
    assert args[2] is None, 'Diagnostic must never run a new CV fold'
    _fit_calls.append(args[6])
    return _actual_fit(*args,**kwargs)
'''))
    n.cells.insert(0,nbformat.v4.new_code_cell(source.cells[0].source))
    n.cells.append(nbformat.v4.new_code_cell('''assert len(_fit_calls)==(0 if report['reused_completed_run'] else 1)
assert report['new_cv_fits']==0 and report['cv_folds_reused']==5
assert X.shape==(24,3) and y.shape==(24,)
assert summary['download_verification']['status']=='PASSED'
'''))
    notebook=isolated/'TECHNICAL_refit.ipynb';nbformat.write(n,notebook)
    kernels=tmp_path/'kernels';spec=kernels/'refit_fixture';spec.mkdir(parents=True)
    (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-I','-m','ipykernel_launcher','-f','{connection_file}'],
        'display_name':'Refit technical fixture','language':'python'}))
    env={**os.environ,'AES300_WORKSPACE':str(workspace),'AES300_DATASET':x['cfg']['dataset'],
         'AES300_CV_ZIP':x['cfg']['cv_source_path'],'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
         'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','IPYTHONDIR':str(tmp_path/'ipython')}
    env.pop('PYTHONPATH',None)
    for attempt in [1,2]:
        current=nbformat.read(notebook,as_version=4)
        manager=KernelManager(kernel_name='refit_fixture',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
        try:NotebookClient(current,timeout=360,km=manager,resources={'metadata':{'path':str(isolated)}}).execute(env=env)
        finally:
            if manager.has_kernel:manager.shutdown_kernel(now=True)
        nbformat.write(current,isolated/f'TECHNICAL_attempt{attempt}.ipynb')
        info=json.loads((workspace/'outputs/dataset300_download.json').read_text());run=Path(info['run_folder'])
        sha=hashlib.sha256((run/'S3/final.pt').read_bytes()).hexdigest()
        if attempt==1:first_sha=sha
        else:assert sha==first_sha
    return info,run


def test_full_refit_no_cv_and_download_gate(tmp_path,old_cv_fixture):
    x=old_cv_fixture;info,run=execute_refit_fixture(tmp_path,x)
    cfg=json.loads((run/'effective_config.json').read_text());ns=x['ns'];bundle=ns['read_dataset'](cfg);manifest=ns['make_split'](bundle,cfg)
    assert not list((run/'S3').glob('cv_fold_*.json'))
    report=json.loads((run/'final_report.json').read_text());assert len(report['cv'])==5 and report['new_cv_fits']==0
    assert report['selected_epochs']=={'S3':1} and report['test_evaluations_per_scenario']==1
    folder=run/'hasil_ringkas'
    assert ns['validate_refit_download'](folder,bundle,manifest,cfg)['status']=='PASSED'
    test=pd.read_excel(folder/'04_Prediksi_dan_Evaluasi.xlsx',sheet_name='Perbandingan Model')
    cv=pd.read_excel(folder/'04_Prediksi_dan_Evaluasi.xlsx',sheet_name='Ringkasan Validasi')
    assert len(test)==len(cv)==1 and test['RMSE (poin)'].notna().all()
    weights=pd.read_excel(folder/'05_Riwayat_Pelatihan.xlsx',sheet_name='Bobot Scalar Mix')
    assert len(weights)==x['spec']['layers'] and set(weights.Tahap)=={'Latihan final'}
    history=pd.read_excel(folder/'05_Riwayat_Pelatihan.xlsx',sheet_name='Fusion (S3)');assert len(history)==1
    prior=pd.read_excel(folder/'05_Riwayat_Pelatihan.xlsx',sheet_name='CV Acuan (dipakai ulang)')
    assert len(prior)==5 and set(prior.Tahap)=={'Validasi silang'}
    bge_old=pd.read_excel(Path(x['info']['run_folder'])/'hasil_ringkas/02_Vektor_Jawaban.xlsx',sheet_name='BGE-M3')
    bge_new=pd.read_excel(folder/'02_Vektor_Jawaban.xlsx',sheet_name='BGE-M3')
    np.testing.assert_array_equal(bge_old.iloc[:,1:],bge_new.iloc[:,1:])
    book=pd.ExcelFile(folder/'04_Prediksi_dan_Evaluasi.xlsx')
    assert ('OOF Acuan (dipakai ulang)' in book.sheet_names)==(x['number']==47)
    with ZipFile(info['archive']) as z:assert len(z.namelist())==5
    # Corrupt the same tables that were empty in E44/E45, then numeric data and LR.
    damaged=tmp_path/'damaged';shutil.copytree(folder,damaged)
    file=damaged/'04_Prediksi_dan_Evaluasi.xlsx';original=file.read_bytes()
    for sheet,action,match in [
        ('Perbandingan Model','empty','tabel metrik'),('Ringkasan Validasi','empty','tabel metrik'),
        ('Nilai Guru dan Prediksi','prediction','metrik test'),('Nilai Guru dan Prediksi','missing','jumlah/ID')]:
        file.write_bytes(original);w=load_workbook(file);ws=w[sheet]
        if action=='empty':ws.delete_rows(2,ws.max_row-1)
        elif action=='missing':ws.delete_rows(2)
        else:ws.cell(2,3,ws.cell(2,3).value+3)
        w.save(file);w.close()
        with pytest.raises(RuntimeError,match=match):ns['validate_refit_download'](damaged,bundle,manifest,cfg)
    file.write_bytes(original)
    hfile=damaged/'05_Riwayat_Pelatihan.xlsx';w=load_workbook(hfile);ws=w['Jadwal LR Refit']
    headers={c.value:c.column for c in ws[1]};ws.cell(2,headers['LR Akhir Epoch'],1.)
    w.save(hfile);w.close()
    with pytest.raises(RuntimeError,match='jejak LR'):ns['validate_refit_download'](damaged,bundle,manifest,cfg)
    proof={'scope':'SYNTHETIC CPU ONLY','new_cv_fits':0,'prior_folds_reused':5,'new_refits':1,
           'new_test_evaluations':1,'download_gate_passed':True,'blank_summary_and_bad_numbers_rejected':True,
           'fresh_kernel_resume_unchanged_checkpoint':True,'model_layers':x['spec']['layers']}
    (tmp_path/'refit_proof.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
