"""CPU-only MLP candidate B; synthetic fixtures, not research scores."""
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
import numpy as np
import pandas as pd
import pytest
import torch

import test_lora_notebook as technical
import test_lora_setup as setup_fixture
from test_teacher_score import TechnicalTokenizer,make_book

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/'notebooks/experiments/AES_300_S3_MLP_LR5e6_CPU.ipynb'
BASE=ROOT/'notebooks/experiments/AES_300_S3_MLP.ipynb'


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(technical,'NOTEBOOK',NOTEBOOK)
    ns=technical.runtime()
    path=tmp_path/'TECHNICAL_FIXTURE_ONLY.xlsx';make_book(path)
    cfg=ns['config']({'dataset':str(path),'output':str(tmp_path/'results'),'expected_rows':24,
        'device':'cpu','epochs':3,'roberta_max_length':32,'roberta_overlap':4,
        'bge_max_length':32,'effective_batch':4,'micro_batch':1,'dropout':0.})
    return ns,cfg,ns['read_dataset'](cfg)


def test_clean_standalone_and_acuan_model_split_metrics_unchanged():
    n=nbformat.read(NOTEBOOK,as_version=4);nbformat.validate(n)
    assert n.metadata.aes_experiment.base_sha256==hashlib.sha256(BASE.read_bytes()).hexdigest()
    assert 'accelerator' not in n.metadata
    assert n.metadata.aes_experiment.device=='cpu'
    assert (ROOT/'aes-learning/notebooks/experiments'/NOTEBOOK.name).read_bytes()==NOTEBOOK.read_bytes()
    def definitions(notebook):
        result={}
        for c in notebook.cells:
            if c.cell_type!='code':continue
            assert not c.outputs and c.execution_count is None
            code=c.source.removeprefix('%%aes\n')
            assert all(x not in code for x in ('import peft','get_peft_model','pair_batch_windows','clear_training_progress','googleapiclient'))
            for node in ast.parse(code).body:
                if isinstance(node,(ast.FunctionDef,ast.ClassDef)):result[node.name]=ast.dump(node,include_attributes=False)
        return result
    new=definitions(n)
    # The original has cleanup functions; compare only the selected definitions.
    original={}
    for c in nbformat.read(BASE,as_version=4).cells:
        if c.cell_type=='code':
            for node in ast.parse(c.source.removeprefix('%%aes\n')).body:
                if isinstance(node,(ast.FunctionDef,ast.ClassDef)):original[node.name]=ast.dump(node,include_attributes=False)
    for name in ('SingleScoreModel','Engine','make_model','make_split','metrics','batch_windows','fit','score_answer','export_final_features'):
        assert new[name]==original[name],name
    before=NOTEBOOK.read_bytes()
    subprocess.run([sys.executable,str(ROOT/'scripts/build_s3_lr_cpu_notebook.py')],check=True,capture_output=True)
    assert NOTEBOOK.read_bytes()==before


@pytest.mark.parametrize('scenario',['S1','S2','S3'])
def test_s3_lr_only_and_encoder_is_fine_tuned(fixture,scenario):
    ns,cfg,bundle=fixture
    model=ns['make_model'](scenario,cfg)
    groups=ns['optimizer_groups'](model,cfg)
    rates={g['name']:g['lr'] for g in groups}
    assert rates['head']==1e-4
    if scenario!='S2':
        assert rates['roberta']==(5e-6 if scenario=='S3' else 1e-5)
        assert all(p.requires_grad for p in model.roberta.parameters())
    else:assert 'roberta' not in rates and model.roberta is None
    if scenario=='S3':assert rates['projection']==1e-4
    params=[id(p) for g in groups for p in g['params']]
    assert len(params)==len(set(params)) and set(params)=={id(p) for p in model.parameters() if p.requires_grad}


@pytest.mark.parametrize('available',[False,True])
def test_visible_config_stays_cpu_even_with_gpu_flag(fixture,monkeypatch,available):
    ns,cfg,bundle=fixture
    monkeypatch.setattr(torch.cuda,'is_available',lambda:available)
    monkeypatch.setenv('AES300_REQUIRE_GPU','1')
    monkeypatch.setenv('AES300_DATASET',cfg['dataset']);monkeypatch.setenv('AES300_RESULTS',cfg['output'])
    monkeypatch.delenv('AES300_AUDIT_ONLY',raising=False)
    n=nbformat.read(NOTEBOOK,as_version=4)
    cell=next(c.source for c in n.cells if c.cell_type=='code' and 'ROBERTA_CHECKPOINT=' in c.source)
    exec(cell.removeprefix('%%aes\n'),ns)
    assert ns['cfg']['device']=='cpu' and ns['device_name'](ns['cfg'])=='cpu'
    assert ns['REQUIRE_GPU'] is False and ns['cfg']['s3_encoder_lr']==5e-6
    assert ns['cfg']['encoder_lr']==1e-5


@pytest.mark.parametrize('value',[0,-1,float('nan')])
def test_invalid_s3_rate_rejected(fixture,value):
    ns,cfg,_=fixture
    with pytest.raises(ValueError):ns['config']({**cfg,'s3_encoder_lr':value})


def test_interrupted_cpu_s3_training_resumes_identical(fixture,tmp_path):
    ns,cfg,bundle=fixture
    bge=torch.nn.functional.normalize(torch.arange(24*1024,dtype=torch.float32).reshape(24,1024)+1,dim=1)
    model,engine,expected=ns['fit']('S3',list(range(8)),list(range(8,12)),bundle,bge,cfg,3,42)
    before={k:v.detach().clone() for k,v in model.state_dict().items()};prediction=engine.predict([12,13])
    path=tmp_path/'progress_fold_1.json';save=ns['save_training_progress']
    def interrupt(path,payload):
        save(path,payload)
        if payload['epoch']==1:raise RuntimeError('TECHNICAL INTERRUPT')
    ns['save_training_progress']=interrupt
    with pytest.raises(RuntimeError,match='TECHNICAL INTERRUPT'):
        ns['fit']('S3',list(range(8)),list(range(8,12)),bundle,bge,cfg,3,42,checkpoint=path)
    ns['save_training_progress']=save
    actual,engine,info=ns['fit']('S3',list(range(8)),list(range(8,12)),bundle,bge,cfg,3,42,checkpoint=path)
    assert expected==info and info['optimizer_learning_rates']['roberta']==5e-6
    for key,value in before.items():torch.testing.assert_close(value,actual.state_dict()[key],rtol=0,atol=0)
    np.testing.assert_array_equal(prediction,engine.predict([12,13]))
    assert len(list(path.parent.glob('progress_fold_1_epoch_*.pt')))==3


def test_cpu_setup_probe_handles_broken_vision_without_peft(tmp_path,monkeypatch):
    monkeypatch.setattr(setup_fixture,'NOTEBOOK',NOTEBOOK)
    ns=setup_fixture.setup_namespace(tmp_path);fake=setup_fixture.broken_vision(tmp_path)
    requirements=ns['active_requirements'](tmp_path/'requirements.dataset300.txt',sys.version_info)
    assert not any(x.startswith(('peft','accelerate')) for x in requirements)
    assert ns['environment_folder'](tmp_path,sys.version_info).name.endswith('_text_s3_lr_cpu')
    process=subprocess.run([sys.executable,'-c',ns['IMPORTS']],capture_output=True,text=True,
        env={**os.environ,'PYTHONPATH':str(fake),'HF_HOME':str(tmp_path/'hf')})
    assert process.returncode==0,process.stderr
    assert json.loads(process.stdout)['imports_verified']


def test_full_cpu_notebook_to_five_reports_then_fresh_runtime_resume(tmp_path):
    notebook=nbformat.read(NOTEBOOK,as_version=4)
    isolated=tmp_path/'isolated';isolated.mkdir();dataset=isolated/'TECHNICAL_FIXTURE_ONLY.xlsx';make_book(dataset)
    guard="""
import builtins
_original_import=builtins.__import__
def _no_project_import(name,*args,**kwargs):
    if name=='aes' or name.startswith('aes.') or name=='peft' or name.startswith('peft.'):
        raise ImportError('Project/PEFT imports forbidden in CPU baseline fixture')
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
    for c in notebook.cells:
        if c.cell_type!='code':continue
        if 'EPOCHS,FOLDS=25,5' in c.source:
            c.source=c.source.replace('EPOCHS,FOLDS=25,5','EPOCHS,FOLDS=1,5').replace('expected_rows=300','expected_rows=24')
            c.source=c.source.replace('epochs=EPOCHS,folds=FOLDS,','epochs=EPOCHS,folds=FOLDS,roberta_max_length=32,roberta_overlap=4,bge_max_length=32,')
            c.source=c.source.replace("lesson_version='s3_mlp_lr5e6_cpu_v1'","lesson_version='TECHNICAL_FIXTURE_ONLY'")
        if 'def encoder(cfg, role):' in c.source:c.source+='\n'+inspect.getsource(TechnicalTokenizer)+'\n'+tiny
        if c.source.startswith('# Penghubung proses Python:'):
            node=next(n for n in ast.parse(c.source).body if isinstance(n,ast.Assign))
            c.source='WORKER_SOURCE='+repr(guard+ast.literal_eval(node.value))+'\n'
    first=nbformat.v4.new_code_cell(guard);first.pop('id',None);notebook.cells.insert(0,first)
    kernels=tmp_path/'kernels';spec=kernels/'aes_fixture';spec.mkdir(parents=True)
    (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-I','-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'AES fixture','language':'python'}))
    env={**os.environ,'AES300_DATASET':str(dataset),'AES300_PERSISTENT_DIR':str(tmp_path/'TECHNICAL_PERSISTENT'),
        'AES300_PYTHON':sys.executable,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','IPYTHONDIR':str(tmp_path/'ipython')}
    env.pop('PYTHONPATH',None);env.pop('AES300_AUDIT_ONLY',None)
    first_hashes=None
    for attempt in (1,2):
        current=nbformat.reads(nbformat.writes(notebook),as_version=4);workspace=isolated/f'runtime_{attempt}'
        manager=KernelManager(kernel_name='aes_fixture',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
        try:
            NotebookClient(current,timeout=360,km=manager,resources={'metadata':{'path':str(isolated)}}).execute(env={**env,'AES300_WORKSPACE':str(workspace)})
        finally:
            if manager.has_kernel:manager.shutdown_kernel(now=True)
        nbformat.write(current,isolated/f'TECHNICAL_executed_{attempt}.ipynb')
        pointer=json.loads((workspace/'outputs/dataset300_download.json').read_text());assert pointer['excel_count']==5
        run=Path(pointer['run_folder']);report=json.loads((run/'final_report.json').read_text())
        assert len(report['cv'])==15 and report['test_evaluations_per_scenario']==1
        hashes={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in run.glob('S*/final.pt')}
        if attempt==1:first_hashes=hashes
        else:
            assert first_hashes==hashes
            stream='\n'.join(str(o.get('text','')) for c in current.cells for o in c.get('outputs',[]) if o.output_type=='stream')
            assert 'Memakai hasil tersimpan yang cocok' in stream and ' · epoch ' not in stream
    for scenario in ('S1','S2','S3'):
        payload=torch.load(run/scenario/'final.pt',weights_only=True,map_location='cpu')
        assert payload['config']['device']=='cpu'
        rates=payload['training']['optimizer_learning_rates']
        assert rates['head']==1e-4
        if scenario!='S2':assert rates['roberta']==(5e-6 if scenario=='S3' else 1e-5)
        if scenario=='S3':assert rates['projection']==1e-4
        assert len(list((run/scenario).glob('progress_fold_*_epoch_001.pt')))==5
    reports=run/'hasil_ringkas'
    comparison=pd.read_excel(reports/'04_Prediksi_dan_Evaluasi.xlsx',sheet_name='Nilai Guru dan Prediksi')
    assert {'Nilai Guru','Prediksi S1','Prediksi S2','Prediksi S3'}<=set(comparison.columns)
    settings=pd.read_excel(reports/'05_Riwayat_Pelatihan.xlsx',sheet_name='Pengaturan Eksperimen')
    assert 'Learning rate encoder RoBERTa S3' in settings['Pengaturan'].tolist()
    vectors=pd.ExcelFile(reports/'02_Vektor_Jawaban.xlsx')
    assert 'RoBERTa S3 Setelah Latihan' in vectors.sheet_names
    assert not any('Pair' in x for x in vectors.sheet_names)
