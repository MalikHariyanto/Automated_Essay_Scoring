"""Separate double-frozen notebook; technical fixtures, never student scores."""
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
import pytest
import torch

from aes.windows import mean_by_essay
from test_teacher_score import TechnicalTokenizer,make_book,dataset,encoders

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/'notebooks/experiments/AES_300_S3_DoubleFrozen_MLP.ipynb'
SEMIFINAL=ROOT/'notebooks/03_AES_dataset_guru_300.ipynb'
SEMIFINAL_SHA256='b06735cb45d632b92ad8ed30990e2e34d0fb66c46c612c528b812dd0d090440b'


def notebook_definition(path,name):
    notebook=nbformat.read(path,as_version=4)
    for cell in notebook.cells:
        if cell.cell_type!='code':continue
        for node in ast.parse(cell.source.removeprefix('%%aes\n')).body:
            if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name==name:
                namespace={'torch':torch,'nn':torch.nn,'mean_by_essay':mean_by_essay}
                exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
                return namespace[name]
    raise AssertionError('Definition missing: '+name)


class TechnicalEncoder(torch.nn.Module):
    """A dropout-bearing fixture; not a pretrained model or research result."""
    def __init__(self,dropout=.7):
        super().__init__()
        self.embedding=torch.nn.Embedding(16,768)
        self.dropout=torch.nn.Dropout(dropout)
        self.gradient_modes=[]

    def forward(self,input_ids):
        self.gradient_modes.append(torch.is_grad_enabled())
        return SimpleNamespace(last_hidden_state=self.dropout(self.embedding(input_ids)))


def example_inputs():
    return dict(windows={'input_ids':torch.tensor([[1,2],[3,4],[5,6]])},
                mapping=torch.tensor([0,0,1]),n_essays=2,
                bge=torch.nn.functional.normalize(torch.arange(2048,dtype=torch.float32).reshape(2,1024)+1,dim=1))


def test_experiment_is_separate_clean_and_has_new_resume_identity():
    assert hashlib.sha256(SEMIFINAL.read_bytes()).hexdigest()==SEMIFINAL_SHA256
    mirror=ROOT/'aes-learning/notebooks/03_AES_dataset_guru_300.ipynb'
    assert mirror.read_bytes()==SEMIFINAL.read_bytes()
    notebook=nbformat.read(NOTEBOOK,as_version=4)
    original=nbformat.read(SEMIFINAL,as_version=4)
    assert len(notebook.cells)==len(original.cells)==47
    assert notebook.metadata.aes_standalone.repository_required is False
    fingerprint=notebook.metadata.aes_standalone.code_sha256
    assert fingerprint!=original.metadata.aes_standalone.code_sha256
    assert notebook.metadata.aes_experiment.name=='S3_DOUBLE_FROZEN_MLP'
    sources=[]
    for cell in notebook.cells:
        assert not cell.get('attachments')
        if cell.cell_type!='code':continue
        assert cell.execution_count is None and cell.outputs==[]
        source=cell.source.removeprefix('%%aes\n')
        sources.append(source)
        for node in ast.walk(ast.parse(source)):
            if isinstance(node,ast.ImportFrom):
                assert not node.module or not (node.module=='aes' or node.module.startswith('aes.'))
            if isinstance(node,ast.Import):
                assert not any(a.name=='aes' or a.name.startswith('aes.') for a in node.names)
            if isinstance(node,ast.Constant) and isinstance(node.value,str):
                assert node.value!='git'
    joined='\n'.join(sources)
    assert 'AES_300_DoubleFrozen_MLP_Privat' in joined
    assert 'AES_300_DoubleFrozen_MLP' in joined
    assert "lesson_version='s3_double_frozen_mlp_autocleanup_v1'" in joined
    assert fingerprint in joined


def test_s3_freezes_encoder_without_freezing_projection_layernorm_or_mlp():
    cls=notebook_definition(NOTEBOOK,'SingleScoreModel')
    torch.manual_seed(38)
    encoder=TechnicalEncoder()
    initial={k:v.detach().clone() for k,v in encoder.state_dict().items()}
    model=cls('S3',encoder,dropout=.5)
    model.train()
    assert model.training and not model.roberta.training
    assert model.head.training and model.projection.training and model.roberta_norm.training
    assert not any(p.requires_grad for p in model.roberta.parameters())
    assert all(p.requires_grad for p in model.head.parameters())
    kwargs=example_inputs()
    prediction,features=model(**kwargs,return_features=True)
    _,repeated_features=model(**kwargs,return_features=True)
    torch.testing.assert_close(features['roberta'],repeated_features['roberta'],atol=0,rtol=0)
    assert not features['roberta'].requires_grad
    assert encoder.gradient_modes==[False,False]
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
    (prediction-torch.tensor([[.81],[.74]])).square().mean().backward()
    assert all(p.grad is None for p in encoder.parameters())
    for parameter in [model.projection[0].weight,model.projection[1].weight,
                      model.roberta_norm.weight,model.head[0].weight,model.head[3].weight]:
        assert parameter.grad is not None and parameter.grad.abs().sum()>0
    optimizer.step()
    for key,value in initial.items():torch.testing.assert_close(encoder.state_dict()[key],value,atol=0,rtol=0)
    assert not torch.equal(before['projection.0.weight'],model.projection[0].weight)
    assert not torch.equal(before['head.0.weight'],model.head[0].weight)
    model.eval();model.train()
    assert not encoder.training and model.head.training


@pytest.mark.parametrize('cuda_available', [False, True])
def test_configuration_accepts_cpu_and_still_detects_gpu(monkeypatch, tmp_path, capsys, cuda_available):
    import math
    from importlib.metadata import version
    import pandas as pd
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: cuda_available)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda index=0: 'Tesla T4')
    monkeypatch.setattr(torch.cuda, 'get_device_properties',
                        lambda index=0: SimpleNamespace(total_memory=16*1024**3))
    monkeypatch.setenv('AES300_DATASET', str(tmp_path/'TECHNICAL_INPUT_ONLY.xlsx'))
    monkeypatch.setenv('AES300_RESULTS', str(tmp_path/'results'))
    # A previous setup cell may have left this flag set. The visible notebook
    # switch must allow CPU without asking the user to change hidden state.
    monkeypatch.setenv('AES300_REQUIRE_GPU', '1')
    monkeypatch.delenv('AES300_AUDIT_ONLY', raising=False)
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    defaults = next(c.source for c in notebook.cells
                    if c.cell_type == 'code' and 'def config(' in c.source)
    configuration = next(c.source for c in notebook.cells
                         if c.cell_type == 'code' and 'ROBERTA_CHECKPOINT=' in c.source)
    tables = []
    namespace = {'math': math, 'os': os, 'pd': pd, 'version': version,
                 'check_gpu': notebook_definition(NOTEBOOK, 'check_gpu'),
                 'show': lambda frame, title, **kwargs: tables.append((frame, title))}
    exec(compile(defaults.removeprefix('%%aes\n'), '<actual-defaults>', 'exec'), namespace)
    exec(compile(configuration.removeprefix('%%aes\n'), '<actual-config>', 'exec'), namespace)
    assert namespace['REQUIRE_GPU'] is False
    assert namespace['RUN_TRAINING'] is True
    assert namespace['cfg']['device'] == 'auto'
    assert notebook_definition(NOTEBOOK, 'device_name')(namespace['cfg']) == ('cuda' if cuda_available else 'cpu')
    assert namespace['gpu']['name'] == ('Tesla T4' if cuda_available else 'CPU')
    assert namespace['gpu']['is_t4'] is cuda_available
    assert len(tables) == 1
    assert ('Mode CPU aktif' in capsys.readouterr().out) is (not cuda_available)


def test_s3_optimizer_has_only_learned_projection_and_head_parameters():
    cls=notebook_definition(NOTEBOOK,'SingleScoreModel')
    groups_fn=notebook_definition(NOTEBOOK,'optimizer_groups')
    model=cls('S3',TechnicalEncoder())
    groups=groups_fn(model,{'encoder_lr':1e-5,'projection_lr':1e-4,'head_lr':1e-4})
    assert {g['name'] for g in groups}=={'projection','head'}
    actual=[id(p) for g in groups for p in g['params']]
    expected=[id(p) for p in model.parameters() if p.requires_grad]
    assert len(actual)==len(set(actual)) and set(actual)==set(expected)
    assert all(g['lr']==1e-4 for g in groups)


@pytest.mark.parametrize('scenario',['S1','S2'])
def test_control_models_retain_semifinal_parameters_predictions_and_training_mode(scenario):
    models=[]
    for path in [SEMIFINAL,NOTEBOOK]:
        cls=notebook_definition(path,'SingleScoreModel')
        torch.manual_seed(89)
        model=cls(scenario,TechnicalEncoder() if scenario=='S1' else None,dropout=.2)
        models.append(model)
    assert models[0].state_dict().keys()==models[1].state_dict().keys()
    for key,value in models[0].state_dict().items():
        torch.testing.assert_close(value,models[1].state_dict()[key],atol=0,rtol=0)
    kwargs=example_inputs()
    if scenario=='S2':kwargs={'bge':kwargs['bge']}
    for model in models:model.eval()
    torch.testing.assert_close(models[0](**kwargs),models[1](**kwargs),atol=0,rtol=0)
    if scenario=='S1':
        models[1].train()
        assert models[1].roberta.training
        assert all(p.requires_grad for p in models[1].roberta.parameters())
        models[1](**kwargs).square().mean().backward()
        assert models[1].roberta.embedding.weight.grad.abs().sum()>0
        assert models[1].roberta.gradient_modes[-1] is True


def test_interrupted_s3_fit_resumes_identical_frozen_state_history_and_predictions(dataset,encoders,tmp_path,monkeypatch):
    """Run actual notebook fit with random tiny Transformer fixture encoders."""
    import numpy as np
    from aes.teacher_score import experiment,experimental_checkpoints,models
    cfg,bundle=dataset
    cfg={**cfg,'dropout':.2,'early_stopping_patience':5}
    monkeypatch.setenv('AES300_DRIVE_CLEANUP','0')
    namespace={**vars(experiment),**vars(experimental_checkpoints),
               'nn':torch.nn,'SingleScoreModel':notebook_definition(NOTEBOOK,'SingleScoreModel'),
               'encoder':encoders,'Engine':models.Engine}
    notebook=nbformat.read(NOTEBOOK,as_version=4)
    cell=next(c for c in notebook.cells if c.cell_type=='code' and 'def fit(' in c.source)
    definitions=[n for n in ast.parse(cell.source.removeprefix('%%aes\n')).body
                 if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(NOTEBOOK),'exec'),namespace)
    # make_model lives in a different notebook cell; avoid the baseline module's
    # function, whose globals still refer to the original linear S3 class.
    namespace['make_model']=lambda scenario,settings: namespace['SingleScoreModel'](
        scenario,encoders(settings,'roberta') if scenario!='S2' else None,
        dropout=settings['dropout'])
    bge=torch.nn.functional.normalize(torch.randn(24,1024),dim=1)
    fit=namespace['fit']
    baseline,baseline_engine,expected=fit('S3',list(range(8)),list(range(8,12)),bundle,bge,cfg,4,42)
    state={k:v.detach().clone() for k,v in baseline.state_dict().items()}
    predictions=baseline_engine.predict([12,13])
    del baseline_engine,baseline
    checkpoint=tmp_path/'S3/progress_fold_1.json'
    save=namespace['save_training_progress']
    def disconnect(path,payload):
        save(path,payload)
        if payload['epoch']==2:
            raise RuntimeError('TECHNICAL SIMULATED RUNTIME DISCONNECT')
    namespace['save_training_progress']=disconnect
    with pytest.raises(RuntimeError,match='SIMULATED'):
        fit('S3',list(range(8)),list(range(8,12)),bundle,bge,cfg,4,42,checkpoint=checkpoint)
    namespace['save_training_progress']=save
    resumed,engine,actual=fit('S3',list(range(8)),list(range(8,12)),bundle,bge,cfg,4,42,checkpoint=checkpoint)
    assert actual==expected
    assert not any(p.requires_grad for p in resumed.roberta.parameters())
    assert not resumed.roberta.training
    for key,value in state.items():
        torch.testing.assert_close(resumed.state_dict()[key],value,atol=0,rtol=0)
    np.testing.assert_array_equal(engine.predict([12,13]),predictions)
    assert len(json.loads(checkpoint.read_text())['snapshots'])==2


def test_experiment_runs_to_five_excel_and_reuses_persistent_results(tmp_path):
    pytest.importorskip('ipykernel',reason='Eksekusi notebook memerlukan ipykernel pada lingkungan pengujian.')
    notebook=nbformat.read(NOTEBOOK,as_version=4)
    isolated=tmp_path/'only_notebook_and_data';isolated.mkdir()
    workspace=isolated/'private_results'
    persistent=tmp_path/'SIMULATED_PRIVATE_DRIVE'
    fixture=isolated/'TECHNICAL_FIXTURE_ONLY.xlsx';make_book(fixture)
    guard='''
import builtins
_original_import=builtins.__import__
def _no_project_import(name,*args,**kwargs):
    if name=='aes' or name.startswith('aes.'):
        raise ImportError('Repository imports are blocked in this standalone test')
    return _original_import(name,*args,**kwargs)
builtins.__import__=_no_project_import
'''
    tiny_encoder='''
# PRIVATE TECHNICAL FIXTURE: random tiny transformers, not pretrained models.
from transformers import RobertaConfig,RobertaModel,XLMRobertaConfig,XLMRobertaModel
def encoder(cfg,role):
    cls,model,dim,heads=(RobertaConfig,RobertaModel,768,12) if role=='roberta' else (XLMRobertaConfig,XLMRobertaModel,1024,16)
    with torch.random.fork_rng():
        torch.manual_seed(170)
        result=model(cls(vocab_size=64,hidden_size=dim,num_hidden_layers=1,num_attention_heads=heads,
                         intermediate_size=16,max_position_embeddings=64,pad_token_id=1,bos_token_id=0,
                         eos_token_id=2,hidden_dropout_prob=0.,attention_probs_dropout_prob=0.),add_pooling_layer=False)
    result.requires_grad_(role=='roberta')
    return result
def tokenizer(*args,**kwargs):return TechnicalTokenizer()
'''
    for cell in notebook.cells:
        if cell.cell_type!='code':continue
        if 'EPOCHS,FOLDS=25,5' in cell.source:
            cell.source=cell.source.replace('EPOCHS,FOLDS=25,5','EPOCHS,FOLDS=1,5').replace('expected_rows=300','expected_rows=24')
            cell.source=cell.source.replace("lesson_version='s3_double_frozen_mlp_autocleanup_v1'","lesson_version='TECHNICAL_FIXTURE_ONLY'")
        if "journal.save('01_konfigurasi'" in cell.source:
            cell.source=cell.source.replace("journal.save('01_konfigurasi'","journal.metadata['evidence_scope']='TECHNICAL RANDOM TRANSFORMER FIXTURE ONLY; NOT STUDENT RESULTS'\njournal.save('01_konfigurasi'")
        if 'def encoder(cfg, role):' in cell.source:
            cell.source+='\n'+inspect.getsource(TechnicalTokenizer)+tiny_encoder
        if cell.source.startswith('# Penghubung proses Python:'):
            node=next(n for n in ast.parse(cell.source).body if isinstance(n,ast.Assign))
            cell.source='WORKER_SOURCE='+repr(guard+ast.literal_eval(node.value))+'\n'
    guard_cell=nbformat.v4.new_code_cell(guard);guard_cell.pop('id',None)
    notebook.cells.insert(0,guard_cell)
    notebook_path=isolated/'03_AES_dataset_guru_300.ipynb'
    nbformat.write(notebook,notebook_path)
    kernels=tmp_path/'kernels';spec=kernels/'aes_isolated';spec.mkdir(parents=True)
    (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-I','-m','ipykernel_launcher','-f','{connection_file}'],
                                              'display_name':'AES technical isolated','language':'python'}))
    # Git tidak tersedia. Kalau notebook mencoba clone/pull, proses langsung gagal.
    bin_folder=tmp_path/'bin';bin_folder.mkdir()
    (bin_folder/'git').write_text('#!/bin/sh\nexit 97\n');(bin_folder/'git').chmod(0o755)
    environment={**os.environ,'AES300_WORKSPACE':str(workspace),'AES300_DATASET':str(fixture),
                 'AES300_PERSISTENT_DIR':str(persistent),'AES300_REQUIRE_GPU':'0',
                 'AES300_PYTHON':sys.executable,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
                 'PATH':str(bin_folder)+os.pathsep+os.environ['PATH'],'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
                 'IPYTHONDIR':str(tmp_path/'ipython')}
    environment.pop('PYTHONPATH',None);environment.pop('AES300_AUDIT_ONLY',None)
    manager=KernelManager(kernel_name='aes_isolated',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
    start=time.time()
    try:
        NotebookClient(notebook,timeout=360,km=manager,resources={'metadata':{'path':str(isolated)}}).execute(env=environment)
    finally:
        if manager.has_kernel:manager.shutdown_kernel(now=True)
    nbformat.write(notebook,isolated/'TECHNICAL_executed.ipynb')
    pointer=json.loads((workspace/'outputs/dataset300_download.json').read_text())
    assert pointer['excel_count']==5
    report=json.loads((Path(pointer['run_folder'])/'final_report.json').read_text())
    settings=json.loads((Path(pointer['run_folder'])/'effective_config.json').read_text())
    assert settings['epochs']==1 and settings['lesson_version']=='TECHNICAL_FIXTURE_ONLY'
    assert settings['encoder_lr']==1e-5 and settings['projection_lr']==settings['head_lr']==1e-4
    assert [r['Skenario'] for r in report['rows']]==['S1','S2','S3']
    assert len(report['cv'])==15
    assert report['test_evaluations_per_scenario']==1
    # Notebook utama memakai S3 MLP; angka di sini hanya fixture teknis.
    import torch
    import pandas as pd
    import numpy as np
    run=Path(pointer['run_folder'])
    payload=torch.load(run/'S3/final.pt',map_location='cpu',weights_only=True)
    assert payload['state']['head.0.weight'].shape==(256,1024)
    assert payload['state']['head.3.weight'].shape==(1,256)
    assert payload['training']['BGE_encoder_trained'] is False
    assert not payload['training']['parameter_samples_changed'].get('roberta', False)
    assert payload['training']['parameter_samples_changed']['projection'] is True
    assert payload['training']['parameter_samples_changed']['head'] is True
    s1=torch.load(run/'S1/final.pt',map_location='cpu',weights_only=True)
    assert s1['state']['head.weight'].shape==(1,768)
    book=run/'excel/14_S3_fitur_setelah_training.xlsx'
    hidden=pd.read_excel(book,sheet_name='MLP_dropout_eval').iloc[:,1:].to_numpy()
    gelu=pd.read_excel(book,sheet_name='MLP_GELU256').iloc[:,1:].to_numpy()
    assert hidden.shape==(24,256)
    np.testing.assert_array_equal(hidden,gelu)
    # Frozen encoder retains pretrained features while the scoring layers learn.
    initial_roberta=pd.read_excel(run/'excel/05_roberta_pretrained.xlsx',sheet_name='Vektor_jawaban_768').iloc[:,1:].to_numpy()
    final_roberta=pd.read_excel(book,sheet_name='roberta').iloc[:,1:].to_numpy()
    np.testing.assert_allclose(final_roberta,initial_roberta,atol=1e-6,rtol=0)
    word_report=json.loads((run/'vectors/S3/report.json').read_text())
    source_label=word_report['roberta']['source'].casefold()
    assert 'frozen' in source_label or 'dibekukan' in source_label
    raw=pd.read_excel(book,sheet_name='Prediksi_semua_jawaban')['prediksi_mentah'].to_numpy()
    reconstructed=torch.nn.functional.linear(torch.as_tensor(hidden,dtype=torch.float32),
                    payload['state']['head.3.weight'],payload['state']['head.3.bias']).numpy().reshape(-1)*100
    np.testing.assert_allclose(reconstructed,raw,atol=5e-5,rtol=1e-6)
    comparison_views=[out.data.get('text/html','') for cell in notebook.cells for out in cell.get('outputs',[])
                      if out.output_type=='display_data' and 'Nilai guru, prediksi mentah, dan pembulatan pada data uji' in out.data.get('text/html','')]
    assert len(comparison_views)==1
    assert 'Nilai Guru' in comparison_views[0] and 'Pembulatan S3' in comparison_views[0]
    assert not list(workspace.rglob('*.py'))
    assert not (workspace/'src').exists() and not (workspace/'docs').exists() and not (workspace/'.git').exists()
    # Mesin lokal diganti; hasil tetap ada di penyimpanan permanen yang sama.
    run_folder=Path(pointer['run_folder'])
    assert persistent in run_folder.parents
    hashes={str(p.relative_to(run_folder)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in run_folder.glob('S*/*') if p.name in ('final.pt','test_predictions.csv')}
    second=nbformat.read(notebook_path,as_version=4)
    second_workspace=isolated/'new_runtime_after_disconnect'
    second_manager=KernelManager(kernel_name='aes_isolated',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
    try:
        NotebookClient(second,timeout=360,km=second_manager,resources={'metadata':{'path':str(isolated)}}).execute(
            env={**environment,'AES300_WORKSPACE':str(second_workspace)})
    finally:
        if second_manager.has_kernel:second_manager.shutdown_kernel(now=True)
    nbformat.write(second,isolated/'TECHNICAL_resumed.ipynb')
    second_pointer=json.loads((second_workspace/'outputs/dataset300_download.json').read_text())
    assert second_pointer['run_folder']==pointer['run_folder'] and second_pointer['excel_count']==5
    streams='\n'.join(out.get('text','') for cell in second.cells for out in cell.get('outputs',[]) if out.output_type=='stream')
    assert 'Memakai hasil tersimpan yang cocok' in streams and ' · epoch ' not in streams
    assert hashes=={name:hashlib.sha256((run_folder/name).read_bytes()).hexdigest() for name in hashes}
    proof={'scope':'TECHNICAL FIXTURE ONLY; NOT STUDENT SCORES','code_cells_executed':sum(c.cell_type=='code' and c.execution_count is not None for c in notebook.cells),
           'production_code_cells':sum(c.cell_type=='code' for c in notebook.cells)-1,'fixture_guard_cells':1,
           'elapsed_seconds':round(time.time()-start,2),'notebook_and_dataset_only':True,'project_imports_blocked':True,
           'git_unavailable':True,'private_python_files_created':False,'output':pointer}
    proof['fresh_runtime_reuses_persistent_results_without_training']=True
    (isolated/'standalone_proof.json').write_text(json.dumps(proof,indent=2))
    print(json.dumps(proof))
