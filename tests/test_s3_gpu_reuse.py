"""Reuse completed controls, never old S3; all results are synthetic fixtures."""
import ast
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
import pandas as pd
import pytest
import torch

import test_lora_notebook as technical
from test_teacher_score import TechnicalTokenizer,make_book

ROOT=Path(__file__).resolve().parents[1]
CPU=ROOT/'notebooks/experiments/AES_300_S3_MLP_LR5e6_CPU.ipynb'
NOTEBOOK=ROOT/'notebooks/experiments/AES_300_E6_S3_LR5e6_GPU_Reuse.ipynb'


def runtime(path):
    original=technical.NOTEBOOK
    try:
        technical.NOTEBOOK=path
        ns=technical.runtime()
        for cell in nbformat.read(path,as_version=4).cells:
            if cell.cell_type!='code':continue
            for node in ast.parse(cell.source.removeprefix('%%aes\n')).body:
                if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REUSE_SOURCE_PROFILES' for t in node.targets):
                    exec(compile(ast.Module(body=[node],type_ignores=[]),'<reuse-profiles>','exec'),ns)
        return ns
    finally:technical.NOTEBOOK=original


@pytest.fixture(scope='module')
def source_run(tmp_path_factory):
    root=tmp_path_factory.mktemp('TECHNICAL_REUSE_SOURCE')
    path=root/'TECHNICAL_FIXTURE_ONLY.xlsx';make_book(path)
    ns=runtime(CPU)
    cfg=ns['config']({'dataset':str(path),'output':str(root/'MyDrive/AES_300_MLP_Privat/outputs/dataset300_inline'),
        'expected_rows':24,'device':'cpu','epochs':1,'roberta_max_length':32,'roberta_overlap':4,
        'bge_max_length':32,'effective_batch':4,'micro_batch':1,'dropout':0.,
        's3_encoder_lr':1e-5,'lesson_version':'s3_mlp_account2_v1',
        'runtime_versions':{p:__import__('importlib.metadata',fromlist=['version']).version(p) for p in ('torch','transformers','numpy')}})
    # Technical fixture uses the trusted source profile identity with equivalent
    # S1/S2 runtime methods; tiny random encoders are explicitly substituted.
    ns['NOTEBOOK_CODE_SHA256']='49f80d493aa531197b6a01e57fcb80e51ff18fb4f2ecda09abfc1f37814f1da7'
    bundle=ns['read_dataset'](cfg);manifest=ns['make_split'](bundle,cfg)
    output=ns['run_folder'](cfg,bundle)
    bge=torch.nn.functional.normalize(torch.arange(24*1024,dtype=torch.float32).reshape(24,1024)+1,dim=1)
    report=ns['train_all'](bundle,manifest,bge,cfg,output)
    return root,path,cfg,bundle,manifest,bge,output,report


def target(source_run,tmp_path,explicit=True):
    root,path,old,bundle,manifest,bge,source,report=source_run
    ns=runtime(NOTEBOOK)
    cfg=ns['config']({**old,'s3_encoder_lr':5e-6,'lesson_version':'e6_s3_lr5e6_gpu_reuse_v1',
        'output':str(tmp_path/'MyDrive/AES_300_E6_S3_LR5e6_GPU_Reuse_Privat/outputs/dataset300_inline'),
        'reuse_run':str(source) if explicit else ''})
    destination=ns['run_folder'](cfg,bundle)
    return ns,cfg,bundle,manifest,bge,source,destination


def test_notebook_clean_gpu_and_cpu_baseline_unchanged():
    n=nbformat.read(NOTEBOOK,as_version=4);nbformat.validate(n)
    assert n.metadata.aes_experiment.base_sha256==hashlib.sha256(CPU.read_bytes()).hexdigest()
    assert n.metadata.accelerator=='GPU'
    assert (ROOT/'aes-learning/notebooks/experiments'/NOTEBOOK.name).read_bytes()==NOTEBOOK.read_bytes()
    for c in n.cells:
        if c.cell_type=='code':
            assert c.outputs==[] and c.execution_count is None
            assert not any(x in c.source for x in ('import peft','get_peft_model','clear_training_progress','googleapiclient'))
            ast.parse(c.source.removeprefix('%%aes\n'))
    config=next(c.source for c in n.cells if c.cell_type=='code' and 'ROBERTA_CHECKPOINT=' in c.source)
    assert "device='cuda'" in config and 'REQUIRE_GPU=True' in config
    before=NOTEBOOK.read_bytes()
    subprocess.run([sys.executable,str(ROOT/'scripts/build_s3_lr_gpu_reuse_notebook.py')],check=True,capture_output=True)
    assert NOTEBOOK.read_bytes()==before


def test_only_s3_fit_runs_source_and_source_s3_remain_unchanged(source_run,tmp_path):
    ns,cfg,bundle,manifest,bge,source,dest=target(source_run,tmp_path)
    before={str(p.relative_to(source)):ns['digest'](p) for p in source.rglob('*') if p.is_file()}
    calls=[];fit=ns['fit']
    def only_s3(scenario,*args,**kwargs):
        calls.append(scenario);assert scenario=='S3','S1/S2 must never train again'
        return fit(scenario,*args,**kwargs)
    ns['fit']=only_s3
    report=ns['train_all'](bundle,manifest,bge,cfg,dest)
    assert calls==['S3']*6 and report['test_evaluations_this_run']=={'S1':0,'S2':0,'S3':1}
    assert report['reused_control_scenarios']==['S1','S2'] and len(report['cv'])==15
    for scenario in ('S1','S2'):
        assert ns['digest'](source/scenario/'final.pt')==ns['digest'](dest/scenario/'final.pt')
        assert ns['digest'](source/scenario/'test_predictions.csv')==ns['digest'](dest/scenario/'test_predictions.csv')
        assert not list((dest/scenario).glob('progress_*.pt'))
        assert json.loads((dest/scenario/'refit_complete.json').read_text())['trained_in_this_run'] is False
    payload=torch.load(dest/'S3/final.pt',weights_only=True,map_location='cpu')
    assert payload['training']['optimizer_learning_rates']['roberta']==5e-6
    assert not torch.equal(payload['state']['head.0.weight'],torch.load(source/'S3/final.pt',weights_only=True,map_location='cpu')['state']['head.0.weight'])
    assert before=={str(p.relative_to(source)):ns['digest'](p) for p in source.rglob('*') if p.is_file()}
    again=ns['train_all'](bundle,manifest,bge,cfg,dest)
    assert again['reused_completed_run'] and calls==['S3']*6


@pytest.mark.parametrize('change',['data','split','lr','runtime','checksum','profile'])
def test_incompatible_sources_rejected_before_training(source_run,tmp_path,change):
    ns,cfg,bundle,manifest,bge,source,dest=target(source_run,tmp_path)
    local=tmp_path/'isolated_source'/source.name;shutil.copytree(source,local)
    other_bundle=copy.copy(bundle);other_manifest=copy.deepcopy(manifest)
    if change=='data':other_bundle.source_sha256='different'
    elif change=='split':other_manifest['split_sha256']='different'
    elif change=='lr':cfg={**cfg,'head_lr':5e-5}
    elif change=='runtime':cfg={**cfg,'runtime_versions':{**cfg['runtime_versions'],'torch':'different'}}
    elif change=='checksum':(local/'S1/final.pt').write_bytes(b'TECHNICAL_CORRUPT_CHECKPOINT')
    elif change=='profile':local=local.rename(local.with_name('unknown_source_identity'))
    with pytest.raises(ValueError):ns['inspect_control_source'](local,other_bundle,other_manifest,cfg)


def test_missing_source_fails_without_training_or_deleting(source_run,tmp_path):
    ns,cfg,bundle,manifest,bge,source,dest=target(source_run,tmp_path)
    cfg={**cfg,'reuse_run':str(tmp_path/'no_source')}
    ns['fit']=lambda *a,**k:pytest.fail('No source must stop before training')
    with pytest.raises(RuntimeError,match='Training tidak dimulai'):
        ns['train_all'](bundle,manifest,bge,cfg,dest)
    assert not (dest/'S3').exists()


def test_partial_copy_recovers_using_locked_source_and_checks_existing_files(source_run,tmp_path,monkeypatch):
    ns,cfg,bundle,manifest,bge,source,dest=target(source_run,tmp_path)
    import shutil as copy_module
    actual=copy_module.copyfile;count=0
    def interrupted(*args,**kwargs):
        nonlocal count
        count+=1
        if count==3:raise OSError('TECHNICAL COPY INTERRUPT')
        return actual(*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(copy_module,'copyfile',interrupted)
        with pytest.raises(OSError,match='COPY INTERRUPT'):ns['prepare_reused_controls'](bundle,manifest,cfg,dest)
    plan=json.loads((dest/'s1_s2_reuse.json').read_text());assert plan['state']=='copy_pending'
    result=ns['prepare_reused_controls'](bundle,manifest,cfg,dest)
    assert result['state']=='complete'
    (dest/'S1/final.pt').write_bytes(b'TECHNICAL_CHANGED_DESTINATION')
    with pytest.raises(ValueError,match='tujuan berbeda'):ns['prepare_reused_controls'](bundle,manifest,cfg,dest)


def test_auto_source_selection_by_folder_priority_not_metrics(source_run,tmp_path):
    ns,cfg,bundle,manifest,bge,source,dest=target(source_run,tmp_path,explicit=False)
    root=dest.parents[3]
    local=root/'AES_300_MLP_Privat/outputs/dataset300_inline'/source.name
    shutil.copytree(source,local)
    result=ns['prepare_reused_controls'](bundle,manifest,cfg,dest)
    assert result['source_run']==str(local)


def test_notebook_executes_all_reports_without_control_training(source_run,tmp_path):
    root,dataset,old,bundle,manifest,bge,source,report=source_run
    n=nbformat.read(NOTEBOOK,as_version=4);isolated=tmp_path/'isolated';isolated.mkdir()
    guard="""
import builtins
_original_import=builtins.__import__
def _no_project_import(name,*args,**kwargs):
    if name=='aes' or name.startswith('aes.') or name=='peft' or name.startswith('peft.'):
        raise ImportError('Project/PEFT imports forbidden in reuse fixture')
    return _original_import(name,*args,**kwargs)
builtins.__import__=_no_project_import
"""
    tiny="""
from transformers import RobertaConfig,RobertaModel,XLMRobertaConfig,XLMRobertaModel
def encoder(cfg,role):
    cls,model,dim,heads=(RobertaConfig,RobertaModel,768,12) if role=='roberta' else (XLMRobertaConfig,XLMRobertaModel,1024,16)
    with torch.random.fork_rng():
        torch.manual_seed(170)
        result=model(cls(vocab_size=64,hidden_size=dim,num_hidden_layers=1,num_attention_heads=heads,
            intermediate_size=16,max_position_embeddings=64,pad_token_id=1,bos_token_id=0,eos_token_id=2,
            hidden_dropout_prob=0.,attention_probs_dropout_prob=0.),add_pooling_layer=False)
    result.requires_grad_(role=='roberta')
    return result
def tokenizer(*args,**kwargs):return TechnicalTokenizer()
"""
    for c in n.cells:
        if c.cell_type!='code':continue
        if 'ROBERTA_CHECKPOINT=' in c.source:
            c.source=c.source.replace('REQUIRE_GPU=True','REQUIRE_GPU=False').replace("device='cuda'","device='cpu'")
            c.source=c.source.replace("REUSE_RUN=''",'REUSE_RUN='+repr(str(source)))
            c.source=c.source.replace('EPOCHS,FOLDS=25,5','EPOCHS,FOLDS=1,5').replace('expected_rows=300','expected_rows=24')
            c.source=c.source.replace('epochs=EPOCHS,folds=FOLDS,','epochs=EPOCHS,folds=FOLDS,roberta_max_length=32,roberta_overlap=4,bge_max_length=32,dropout=0.,effective_batch=4,')
            # effective_batch already passed later: replace its value instead.
            c.source=c.source.replace('dropout=0.,effective_batch=4,','dropout=0.,').replace('MICRO_BATCH,EFFECTIVE_BATCH=1,8','MICRO_BATCH,EFFECTIVE_BATCH=1,4')
        if 'def encoder(cfg, role):' in c.source:c.source+='\n'+inspect.getsource(TechnicalTokenizer)+'\n'+tiny
        if 'def fit(' in c.source:
            c.source+='''\n_original_fit=fit
def fit(scenario,*args,**kwargs):
    if scenario!='S3':raise RuntimeError('CONTROL_TRAINING_FORBIDDEN')
    return _original_fit(scenario,*args,**kwargs)
'''
        if c.source.startswith('# Penghubung proses Python:'):
            node=next(x for x in ast.parse(c.source).body if isinstance(x,ast.Assign))
            c.source='WORKER_SOURCE='+repr(guard+ast.literal_eval(node.value))+'\n'
    first=nbformat.v4.new_code_cell(guard);first.pop('id',None);n.cells.insert(0,first)
    kernels=tmp_path/'kernels';spec=kernels/'aes_fixture';spec.mkdir(parents=True)
    (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-I','-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'AES fixture','language':'python'}))
    env={**os.environ,'AES300_DATASET':str(dataset),'AES300_PERSISTENT_DIR':str(tmp_path/'TECHNICAL_PERSISTENT'),
        'AES300_PYTHON':sys.executable,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','IPYTHONDIR':str(tmp_path/'ipython')}
    env.pop('PYTHONPATH',None);env.pop('AES300_AUDIT_ONLY',None)
    first_hashes=None
    for attempt in (1,2):
        current=nbformat.reads(nbformat.writes(n),as_version=4);workspace=isolated/f'runtime_{attempt}'
        manager=KernelManager(kernel_name='aes_fixture',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
        try:
            NotebookClient(current,timeout=360,km=manager,resources={'metadata':{'path':str(isolated)}}).execute(env={**env,'AES300_WORKSPACE':str(workspace)})
        finally:
            if manager.has_kernel:manager.shutdown_kernel(now=True)
        nbformat.write(current,isolated/f'TECHNICAL_executed_{attempt}.ipynb')
        pointer=json.loads((workspace/'outputs/dataset300_download.json').read_text());assert pointer['excel_count']==5
        run=Path(pointer['run_folder']);report=json.loads((run/'final_report.json').read_text())
        assert len(report['cv'])==15 and report['test_evaluations_this_run']=={'S1':0,'S2':0,'S3':1}
        hashes={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in run.glob('S*/final.pt')}
        if attempt==1:first_hashes=hashes
        else:assert first_hashes==hashes
    book=pd.ExcelFile(run/'hasil_ringkas/05_Riwayat_Pelatihan.xlsx')
    assert 'Asal Model' in book.sheet_names
    comparison=pd.read_excel(run/'hasil_ringkas/04_Prediksi_dan_Evaluasi.xlsx',sheet_name='Nilai Guru dan Prediksi')
    assert {'Nilai Guru','Prediksi S1','Prediksi S2','Prediksi S3'}<=set(comparison.columns)
