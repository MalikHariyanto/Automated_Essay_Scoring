"""E37 paired-CV protocol and standalone execution; technical fixtures only."""
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import types
from zipfile import ZipFile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests'))
import test_context_pair_notebook as context
from test_e6_three_input_notebook import ThreeTechnicalTokenizer
from test_teacher_score import TechnicalTokenizer, make_book, encoders

NAME = 'AES_300_E37_E30_S3_EncoderLR5e6_vs7e6_3Seed_Native_Learnable_GPU.ipynb'
NOTEBOOK = ROOT/'notebooks/experiments'/NAME
BASE = ROOT/'notebooks/experiments/AES_300_E30_E26_BiGRU_CoAttention_Pooling_GPU.ipynb'
FORMAT = ROOT/'notebooks/experiments/AES_300_E36_E30_S3_EncoderLR3e6_Native_Learnable_GPU.ipynb'


def native_runtime():
    n = nbformat.read(NOTEBOOK, as_version=4)
    module = types.ModuleType('e37_native_fixture_runtime')
    sys.modules[module.__name__] = module
    ns = module.__dict__
    safe = {'imports','model_names','input_columns','hyperparameters','runtime_flags',
            'configuration_defaults','presentation_helpers','word_helpers'}
    for c in n.cells:
        if c.cell_type == 'code' and c.metadata.aes37_role in safe:
            exec(compile(c.source, str(NOTEBOOK), 'exec'), ns)
    for c in n.cells:
        if c.cell_type == 'code':
            definitions = [v for v in ast.parse(c.source).body if isinstance(v,(ast.FunctionDef,ast.ClassDef))]
            exec(compile(ast.Module(body=definitions,type_ignores=[]),str(NOTEBOOK),'exec'),ns)
    ns['show'] = lambda *a,**k: None
    return ns


def test_native_structure_deterministic_source_and_e30_math_unchanged():
    raw=NOTEBOOK.read_bytes();n=nbformat.reads(raw.decode(),as_version=4);nbformat.validate(n)
    code=[c for c in n.cells if c.cell_type=='code'];fingerprint=n.metadata.aes_standalone.code_sha256
    assert len(code)==94 and n.metadata.aes_experiment.cv_fits==30
    assert n.metadata.aes_experiment.scenarios==['S3'] and n.metadata.aes_experiment.training_seeds==[42,142,242]
    assert n.metadata.aes_experiment.format_sha256==hashlib.sha256(FORMAT.read_bytes()).hexdigest()
    for c in code:
        assert c.execution_count is None and not c.outputs
        tree=ast.parse(c.source)
        assert not any(isinstance(v,ast.Call) and isinstance(v.func,ast.Name) and v.func.id in ('exec','eval') for v in ast.walk(tree))
        assert not any(line.lstrip().startswith(('%','!')) for line in c.source.splitlines())
    text='\n'.join(c.source for c in code)
    assert all(s not in text for s in ('WORKER_SOURCE','%%aes','cosine','register_magic'))
    assert hashlib.sha256('\n'.join(c.source.replace(fingerprint,'__NOTEBOOK_CODE_SHA256__') for c in code).encode()).hexdigest()==fingerprint
    def definitions(path):
        return {v.name:ast.dump(v,include_attributes=False)
                for c in nbformat.read(path,as_version=4).cells if c.cell_type=='code'
                for v in ast.parse(c.source.removeprefix('%%aes\n')).body
                if isinstance(v,(ast.FunctionDef,ast.ClassDef))}
    a,b=definitions(BASE),definitions(NOTEBOOK)
    for name in ('AnswerBiGRUAttention','AnswerBiGRUCoAttentionPool','Engine','optimizer_groups','fit',
                 'metrics','gwet_ac2','make_split','bge_features','save_model','load_model'):
        assert a[name]==b[name],name
    originals={p:p.read_bytes() for p in (BASE,FORMAT)}
    subprocess.run([sys.executable,'scripts/build_e37_notebook.py'],cwd=ROOT,check=True,capture_output=True)
    assert NOTEBOOK.read_bytes()==raw==(ROOT/'aes-learning/notebooks/experiments'/NAME).read_bytes()
    assert all(p.read_bytes()==v for p,v in originals.items())


def test_selection_uses_all_cv_seeds_ties_keep_control_and_partial_results_rejected(tmp_path):
    ns=native_runtime();seeds=[42,142,242]
    rows=[{'encoder_lr':lr,'seed':seed,'fold':fold,'RMSE':6.+(lr==7e-6)*.2}
          for lr in (5e-6,7e-6) for seed in seeds for fold in range(1,6)]
    choice=ns['select_lr_from_cv'](rows,seeds,5e-6,7e-6)
    assert choice['selected_encoder_lr']==5e-6 and choice['primary_seed']==42
    assert not choice['test_used_for_selection']
    for r in rows:r['RMSE']=6.
    assert ns['select_lr_from_cv'](rows,seeds,5e-6,7e-6)['selected_encoder_lr']==5e-6
    # A good candidate on a single seed must not outweigh worse means on the other two.
    for r in rows:
        if r['encoder_lr']==7e-6:r['RMSE']=4. if r['seed']==42 else 8.
    assert ns['select_lr_from_cv'](rows,seeds,5e-6,7e-6)['selected_encoder_lr']==5e-6
    with pytest.raises(ValueError,match='lengkap'):ns['select_lr_from_cv'](rows[:-1],seeds,5e-6,7e-6)
    with pytest.raises(ValueError,match='lengkap'):ns['select_lr_from_cv'](rows+[rows[0]],seeds,5e-6,7e-6)
    rows[0]['RMSE']=float('nan')
    with pytest.raises(ValueError,match='finite'):ns['select_lr_from_cv'](rows,seeds,5e-6,7e-6)
    progress=tmp_path/'progress_fold_1_slot_A.pt';progress.write_text('fixture')
    with pytest.raises(ValueError,match='sebelum'):ns['discard_completed_cv_progress'](tmp_path,1)
    assert progress.exists()
    (tmp_path/'cv_fold_1.json').write_text('{}');ns['discard_completed_cv_progress'](tmp_path,1)
    assert not progress.exists() and (tmp_path/'cv_fold_1.json').exists()


def test_paired_lr_only_changes_roberta_update_and_not_split_or_other_parameters(tmp_path,encoders):
    ns=native_runtime();ns['encoder']=encoders;ns['tokenizer']=lambda *a,**k:ThreeTechnicalTokenizer()
    path=tmp_path/'TECHNICAL_FIXTURE_ONLY.xlsx';make_book(path)
    cfg=ns['config'](dict(dataset=str(path),output=str(tmp_path/'run'),expected_rows=24,device='cpu',dropout=0.,
        roberta_max_length=32,roberta_overlap=4,bge_max_length=32,bge_overlap=4,context_max_tokens=12,question_max_tokens=4))
    bundle=ns['read_dataset'](cfg);manifest=ns['make_split'](bundle,cfg)
    for seed in (42,142,242):assert ns['make_split'](bundle,{**cfg,'seed':seed})==manifest
    torch.set_num_threads(2);bge=torch.randn(24,1024);states=[];outputs=[];initial=None
    for lr in (5e-6,7e-6):
        current={**cfg,'encoder_lr':lr,'s3_encoder_lr':lr};ns['seed_everything'](71)
        model=ns['make_model']('S3',current).eval()
        if initial is None:initial={k:v.clone() for k,v in model.state_dict().items()}
        else:
            for k,v in model.state_dict().items():torch.testing.assert_close(v,initial[k],rtol=0,atol=0)
        engine=ns['Engine'](model,bundle.texts,bge,current,'cpu',contexts=ns['contexts_from_bundle'](bundle))
        groups=ns['optimizer_groups'](model,current)
        assert {g['name']:g['lr'] for g in groups}=={'roberta':lr,'projection':1e-4,'head':1e-4,'pooling_norm':1e-4}
        p=engine.forward([0,1]);outputs.append(p.detach());(p-.8).square().mean().backward()
        torch.optim.AdamW(groups,weight_decay=.01).step();states.append({k:v.clone() for k,v in model.state_dict().items()})
    torch.testing.assert_close(*outputs,rtol=0,atol=0)
    for k,v in states[0].items():
        if not k.startswith('roberta.'):torch.testing.assert_close(v,states[1][k],rtol=0,atol=0)
    key='roberta.embeddings.word_embeddings.weight'
    assert float((states[1][key]-initial[key]).norm()/(states[0][key]-initial[key]).norm())==pytest.approx(1.4,rel=.01)
    for scenario in ('S1','S2'):
        with pytest.raises(ValueError):ns['make_model'](scenario,cfg)


def test_full_notebook_30_cv_fits_selection_then_3_refits_and_fresh_kernel_resume(tmp_path):
    """Execute every native section with random tiny Transformers, never research data."""
    n=nbformat.read(NOTEBOOK,as_version=4)
    isolated=tmp_path/'standalone';isolated.mkdir();fixture=isolated/'TECHNICAL_FIXTURE_ONLY.xlsx';make_book(fixture)
    workspace=isolated/'workspace'
    guard="""import builtins
_original_import=builtins.__import__
def _no_project_import(name,*args,**kwargs):
    if name=='aes' or name.startswith('aes.'):raise ImportError('Standalone fixture forbids project imports')
    return _original_import(name,*args,**kwargs)
builtins.__import__=_no_project_import
"""
    tiny="""
from transformers import RobertaConfig,RobertaModel,XLMRobertaConfig,XLMRobertaModel
def encoder(cfg,role):
    config,model,dim,heads=(RobertaConfig,RobertaModel,768,12) if role=='roberta' else (XLMRobertaConfig,XLMRobertaModel,1024,16)
    with torch.random.fork_rng():
        torch.manual_seed(170)
        result=model(config(vocab_size=64,hidden_size=dim,num_hidden_layers=1,num_attention_heads=heads,
            intermediate_size=16,max_position_embeddings=64,pad_token_id=1,bos_token_id=0,eos_token_id=2,
            hidden_dropout_prob=0.,attention_probs_dropout_prob=0.),add_pooling_layer=False)
    result.requires_grad_(role=='roberta');return result
def tokenizer(*a,**k):return ThreeTechnicalTokenizer()
"""
    for c in n.cells:
        if c.cell_type!='code':continue
        role=c.metadata.aes37_role
        if role=='hyperparameters':
            for a,b in {'EPOCHS = 25':'EPOCHS = 1',"DEVICE = 'cuda'":"DEVICE = 'cpu'",'REQUIRE_GPU = True':'REQUIRE_GPU = False',
                'ROBERTA_MAX_TOKENS = 512':'ROBERTA_MAX_TOKENS = 32','BGE_MAX_TOKENS = 1024':'BGE_MAX_TOKENS = 32',
                'WINDOW_OVERLAP = 128':'WINDOW_OVERLAP = 4','CONTEXT_MAX_TOKENS = 192':'CONTEXT_MAX_TOKENS = 12',
                'QUESTION_MAX_TOKENS = 64':'QUESTION_MAX_TOKENS = 4'}.items():
                assert a in c.source;c.source=c.source.replace(a,b)
        if role=='input_columns':c.source=c.source.replace('EXPECTED_ROWS = 300','EXPECTED_ROWS = 24')
        if role=='definition:tokenizer':
            c.source+='\n'+inspect.getsource(TechnicalTokenizer)+'\n'+inspect.getsource(context.PairTechnicalTokenizer)+'\n'+inspect.getsource(ThreeTechnicalTokenizer)+tiny
            c.source=c.source.replace('context.PairTechnicalTokenizer','PairTechnicalTokenizer')
        if role=='training_execute':
            c.source="""FIT_CALLS=[]
_original_fit=fit
def fit(scenario,training,validation,bundle,bge,cfg,epochs,seed,checkpoint=None):
    # Validate actual train/evaluation boundaries, including when the winner is control.
    assert scenario=='S3'
    assert not set(training)&set(manifest['test'])
    if validation is not None:
        assert not set(validation)&set(manifest['test'])
    else:
        assert (study_output/'paired_selection_locked.json').is_file()
        assert cfg['s3_encoder_lr']==study['selection']['selected_encoder_lr']
    FIT_CALLS.append((cfg['s3_encoder_lr'],cfg['seed'],validation is None))
    return _original_fit(scenario,training,validation,bundle,bge,cfg,epochs,seed,checkpoint=checkpoint)
"""+c.source
    n.cells.insert(0,nbformat.v4.new_code_cell(guard))
    n.cells.append(nbformat.v4.new_code_cell("""
assert list(X.columns)==INPUT_COLUMNS and TARGET_COLUMN not in X.columns
assert X.shape==(24,3) and y.shape==(24,)
assert len(FIT_CALLS) in (0,33), FIT_CALLS
if FIT_CALLS:
    assert sum(not final for lr,seed,final in FIT_CALLS)==30
    assert sum(final for lr,seed,final in FIT_CALLS)==3
    assert all(not final for lr,seed,final in FIT_CALLS[:30])
assert study['selection']['primary_seed']==42
assert cfg['seed']==42 and len(final['test_rows'])==3
assert output==Path(final['primary']['folder'])
"""))
    path=isolated/'TECHNICAL_source.ipynb';nbformat.write(n,path)
    kernels=tmp_path/'kernels';spec=kernels/'aes37_fixture';spec.mkdir(parents=True)
    (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-I','-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'AES37 fixture','language':'python'}))
    env={**os.environ,'AES300_WORKSPACE':str(workspace),'AES300_DATASET':str(fixture),
         'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','IPYTHONDIR':str(tmp_path/'ipython')}
    env.pop('PYTHONPATH',None)
    for attempt in (1,2):
        current=nbformat.read(path,as_version=4)
        manager=KernelManager(kernel_name='aes37_fixture',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
        try:NotebookClient(current,timeout=600,km=manager,resources={'metadata':{'path':str(isolated)}}).execute(env=env)
        finally:
            if manager.has_kernel:manager.shutdown_kernel(now=True)
        nbformat.write(current,isolated/f'TECHNICAL_attempt_{attempt}.ipynb')
        info=json.loads((workspace/'outputs/dataset300_download.json').read_text());run=Path(info['run_folder'])
        reports=list(workspace.rglob('final_report.json'));assert len(reports)==3
        cv=list(workspace.rglob('cv_report.json'));assert len(cv)==6
        assert sum(len(json.loads(p.read_text())['cv']) for p in cv)==30
        assert all(json.loads(p.read_text())['test_evaluations_per_scenario']==0 for p in cv)
        assert not list(workspace.rglob('progress_fold_*_slot_*.pt'))
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in workspace.rglob('final.pt')};assert len(hashes)==3
        if attempt==1:first_hashes=hashes
        else:assert hashes==first_hashes
    assert not list(workspace.rglob('S1')) and not list(workspace.rglob('S2'))
    evaluation=pd.ExcelFile(run/'hasil_ringkas/04_Prediksi_dan_Evaluasi.xlsx')
    training=pd.ExcelFile(run/'hasil_ringkas/05_Riwayat_Pelatihan.xlsx')
    assert {'Test per Seed','Test Mean SD Seed','Prediksi Semua Seed'}<=set(evaluation.sheet_names)
    assert {'CV Semua LR Seed','CV per Seed','CV Ringkasan LR','Selisih CV Berpasangan','Seleksi LR Terkunci','Riwayat Semua LR Seed'}<=set(training.sheet_names)
    assert len(pd.read_excel(training,sheet_name='CV Semua LR Seed'))==30
    assert len(pd.read_excel(evaluation,sheet_name='Test per Seed'))==3
    assert len(pd.read_excel(evaluation,sheet_name='Prediksi Semua Seed'))==3*json.loads((run/'final_report.json').read_text())['test_rows']
    with ZipFile(info['archive']) as z:assert len(z.namelist())==5
    assert not list(workspace.rglob('*.py'))
    proof={'scope':'TECHNICAL RANDOM TRANSFORMER FIXTURE ONLY','native_kernel':True,
           'cv_fits':30,'refits':3,'reports':5,'fresh_kernel_resume_unchanged_checkpoints':True,
           'fixed_holdout_and_folds':True,'primary_seed':42,'test_not_used_for_selection':True,
           'execution_device':'cpu','published_runtime':'cuda','research_training_executed':False}
    (isolated/'proof.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
