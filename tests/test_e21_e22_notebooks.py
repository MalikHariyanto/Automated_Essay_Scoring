"""Ablation format input: fixture teknis, bukan hasil penelitian pretrained."""
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
import test_context_pair_notebook as context
from test_e6_three_input_notebook import ThreeTechnicalTokenizer
from test_teacher_score import TechnicalTokenizer,make_book,encoders

NAMES={'single_sequence':'AES_300_E21_Tiga_Input_CPU.ipynb',
       'context_pair':'AES_300_E22_Tiga_Input_ContextPair_CPU.ipynb'}


@pytest.fixture(params=NAMES)
def runtime(request,monkeypatch,tmp_path):
    mode=request.param
    monkeypatch.setattr(context,'NOTEBOOK',ROOT/'notebooks/experiments'/NAMES[mode])
    ns=context.runtime();path=tmp_path/'TECHNICAL_FIXTURE_ONLY.xlsx';make_book(path)
    cfg=ns['config']({'dataset':str(path),'output':str(tmp_path/'results'),'expected_rows':24,
        'device':'cpu','epochs':2,'roberta_max_length':32,'roberta_overlap':4,
        'bge_max_length':32,'bge_overlap':4,'context_max_tokens':12,'question_max_tokens':4,
        'effective_batch':4,'micro_batch':1,'dropout':0.})
    assert cfg['input_format']==mode
    ns['tokenizer']=lambda *a,**k:ThreeTechnicalTokenizer()
    return ns,cfg,ns['read_dataset'](cfg)


def test_both_sources_are_standalone_deterministic_and_differ_only_by_format_and_identity():
    original={name:(ROOT/'notebooks/experiments'/name).read_bytes() for name in NAMES.values()}
    normalized=[]
    for mode,name in NAMES.items():
        raw=original[name];n=json.loads(raw);meta=n['metadata']['aes_experiment']
        assert meta['drive_mount'] is False and meta['drive_autosave'] is False
        assert meta['input_columns']==['soal','response_text','jawaban_referensi'] and meta['target']=='rater'
        assert (ROOT/'aes-learning/notebooks/experiments'/name).read_bytes()==raw
        fingerprint=n['metadata']['aes_standalone']['code_sha256'];codes=[]
        for c in n['cells']:
            if c['cell_type']!='code':continue
            assert c['execution_count'] is None and c['outputs']==[]
            s=''.join(c['source']).removeprefix('%%aes\n');ast.parse(s)
            assert 'drive.mount' not in s and 'googleapiclient' not in s and 'AES300_PERSISTENT_DIR' not in s
            assert 'files.upload(' not in s
            codes.append(s.replace(fingerprint,'__NOTEBOOK_CODE_SHA256__'))
        assert len(codes)==30
        assert hashlib.sha256('\n'.join(codes).encode()).hexdigest()==fingerprint
        normalized.append('\n'.join(codes).replace(name.removesuffix('.ipynb'),'EXPERIMENT').replace("'input_format':'"+mode+"'","'input_format':'FORMAT'").replace("INPUT_FORMAT='"+mode+"'","INPUT_FORMAT='FORMAT'"))
    assert normalized[0]==normalized[1]
    subprocess.run([sys.executable,'scripts/build_e21_e22_notebooks.py'],cwd=ROOT,check=True,capture_output=True)
    assert all((ROOT/'notebooks/experiments'/name).read_bytes()==raw for name,raw in original.items())


def test_local_storage_ignores_persistent_drive_override_and_blocks_drive_path(runtime,monkeypatch,tmp_path):
    ns,_,_=runtime
    monkeypatch.setenv('AES300_PERSISTENT_DIR','/content/drive/MyDrive/DO_NOT_WRITE')
    local=tmp_path/'only_local'
    assert ns['prepare_local_storage'](local,True)==local.resolve()
    with pytest.raises(ValueError,match='lokal'):
        ns['prepare_local_storage'](Path('/content/drive/MyDrive/DO_NOT_WRITE'),True)


def test_native_editor_upload_is_found_copied_and_explicit_path_resolves_conflicts(runtime,tmp_path):
    ns,_,_=runtime;folder=tmp_path/'native_content';folder.mkdir()
    source=folder/'fix_dataset_300.xlsx';make_book(source)
    workspace=tmp_path/'local_workspace'
    dest=ns['resolve_dataset_upload'](workspace,search_paths=[folder])
    assert dest==workspace/'data/dataset300/fix_dataset_300.xlsx'
    assert dest.read_bytes()==source.read_bytes()
    # Dua upload berbeda tidak boleh dipilih diam-diam.
    other=tmp_path/'other_upload';other.mkdir();make_book(other/source.name,rows=12)
    with pytest.raises(ValueError,match='DATASET_FILE'):
        ns['resolve_dataset_upload'](workspace,search_paths=[folder,other])
    assert ns['resolve_dataset_upload'](workspace,configured=str(source)).read_bytes()==source.read_bytes()
    source.unlink()
    assert ns['resolve_dataset_upload'](workspace,search_paths=[]).is_file()


def test_missing_native_upload_explains_retry_without_widget_and_rejects_empty_file(runtime,tmp_path):
    ns,_,_=runtime;workspace=tmp_path/'empty_workspace';native=tmp_path/'incoming';native.mkdir()
    with pytest.raises(FileNotFoundError,match='Upload to Colab'):
        ns['resolve_dataset_upload'](workspace,search_paths=[native])
    with pytest.raises(FileNotFoundError,match='DATASET_FILE'):
        ns['resolve_dataset_upload'](workspace,configured=str(native/'missing.xlsx'))
    source=native/'fix_dataset_300.xlsx';source.touch()
    with pytest.raises(ValueError,match='kosong'):
        ns['resolve_dataset_upload'](workspace,search_paths=[native])
    make_book(source)
    assert ns['resolve_dataset_upload'](workspace,search_paths=[native]).read_bytes()==source.read_bytes()


def test_single_and_pair_preserve_identical_information_answer_windows_and_coverage(runtime):
    ns,cfg,_=runtime;tok=ThreeTechnicalTokenizer()
    text=' '.join('answer'+str(i) for i in range(73));ctx=('question '*12,'reference '*14)
    layouts={mode:ns['pair_content_windows'](text,ctx,tok,32,4,{**cfg,'input_format':mode}) for mode in NAMES}
    a,ai=layouts['single_sequence'];b,bi=layouts['context_pair']
    assert ai['starts']==bi['starts'] and ai['capacity']==bi['capacity']
    assert len(a)==len(b)>1 and ai['pair_special_tokens']==2 and bi['pair_special_tokens']==4
    ids=tok.encode(text,add_special_tokens=False);covered={}
    for i,(single,pair) in enumerate(zip(a,b)):
        assert len(pair['input_ids'])==len(single['input_ids'])+2<=32
        strip=lambda r:[v for v in r['input_ids'] if v not in (0,2)]
        assert strip(single)==strip(pair)
        for j,p in enumerate(ai['answer_positions'][i]):covered[ai['starts'][i]+j]=single['input_ids'][p]
        assert min(pair['input_ids'])>=0 and 'Penanda jawaban' in ai['segments'][i]
    assert [covered[i] for i in range(len(ids))]==ids
    assert ai['answer_tokens_uncovered']==bi['answer_tokens_uncovered']==0


def test_three_input_features_change_with_context_not_teacher_and_cache_isolated(runtime,encoders):
    ns,cfg,bundle=runtime
    def stable(cfg,role):
        with torch.random.fork_rng():
            torch.manual_seed(170);return encoders(cfg,role)
    ns['encoder']=stable
    old=ns['bge_features'](bundle,cfg,Path(cfg['output'])/'run').numpy()
    bundle.y[:]=-999
    for row in bundle.rows:row['score']=999
    np.testing.assert_array_equal(old,ns['bge_features'](bundle,cfg,Path(cfg['output'])/'run').numpy())
    assert np.allclose(np.linalg.norm(old,axis=1),1)
    for key,value in [('question','unfamiliar question context'),('reference','changed reference words')]:
        original=bundle.rows[0][key];bundle.rows[0][key]=value
        changed=ns['bge_features'](bundle,cfg,Path(cfg['output'])/'run').numpy()
        assert not np.array_equal(old[0],changed[0]);np.testing.assert_array_equal(old[1:],changed[1:])
        bundle.rows[0][key]=original
    ns['bge_features'](bundle,{**cfg,'input_format':next(m for m in NAMES if m!=cfg['input_format'])},Path(cfg['output'])/'run')
    assert len(list((Path(cfg['output'])/'cache').glob('bge_three_inputs_*.npz')))==4


@pytest.mark.parametrize('scenario',['S1','S2','S3'])
def test_three_scenarios_forward_backward_heads_lr_and_context(runtime,encoders,scenario):
    ns,cfg,bundle=runtime;ns['encoder']=encoders
    bge=ns['bge_features'](bundle,cfg,Path(cfg['output'])/'run')
    model=ns['make_model'](scenario,cfg).eval()
    assert isinstance(model.head,torch.nn.Linear if scenario=='S1' else torch.nn.Sequential)
    engine=ns['Engine'](model,bundle.texts,bge,cfg,'cpu',roberta_tokenizer=ThreeTechnicalTokenizer(),contexts=ns['contexts_from_bundle'](bundle))
    prediction,features=engine.forward([0],True);assert prediction.shape==(1,1)
    if scenario!='S2':
        for key in (0,1):
            original=engine.contexts[0];changed=list(original);changed[key]='unfamiliar changed context words'
            engine.contexts[0]=tuple(changed);_,after=engine.forward([0],True)
            assert not torch.equal(features['roberta'],after['roberta']);engine.contexts[0]=original
        assert all(p.requires_grad for p in model.roberta.parameters())
    prediction.square().mean().backward()
    assert sum(p.grad.abs().sum().item() for p in model.head.parameters() if p.grad is not None)>0
    rates={g['name']:g['lr'] for g in ns['optimizer_groups'](model,cfg)}
    assert rates.get('roberta')==(None if scenario=='S2' else 5e-6) and rates['head']==1e-4
    if scenario=='S3':
        assert rates['projection']==1e-4 and features['concat'].shape==(1,1024)
        assert any(p.grad is not None for p in model.projection.parameters())


def test_both_encoder_token_exports_and_word_vectors_align_with_actual_format(runtime,encoders):
    ns,cfg,bundle=runtime
    for role in ('roberta','bge'):
        tok=ThreeTechnicalTokenizer();frame,coverage=ns['pair_token_frames'](bundle,cfg,tok,role)
        assert {'Soal/referensi','Jawaban siswa','Token khusus','Penanda jawaban'}<=set(frame['input_segment'])
        assert all(c['input_format']==cfg['input_format'] and c['answer_tokens_uncovered']==0 for c in coverage)
        model=encoders(cfg,role).eval()
        trace=ns['token_vectors']('answer '*55,model,tok,tok,role,{**cfg,'pair_context':ns['contexts_from_bundle'](bundle)[0]},'cpu')
        assert np.all(trace['counts']>0) and trace['contextual'].shape[0]==55
        if role=='bge':assert np.isclose(np.linalg.norm(trace['cls']),1)


@pytest.mark.parametrize('mode',NAMES)
def test_full_notebook_thirty_cells_and_resume_same_local_folder(mode,tmp_path):
    """15 CV+3 refit par notebook; bobot Transformer acak, bukan penelitian."""
    notebook=nbformat.read(ROOT/'notebooks/experiments'/NAMES[mode],as_version=4)
    isolated=tmp_path/'standalone';isolated.mkdir();fixture=isolated/'TECHNICAL_FIXTURE_ONLY.xlsx';make_book(fixture)
    workspace=isolated/'workspace'
    guard="""
import builtins
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
    for cell in notebook.cells:
        if cell.cell_type!='code':continue
        if 'EPOCHS,FOLDS=25,5' in cell.source:
            cell.source=cell.source.replace('EPOCHS,FOLDS=25,5','EPOCHS,FOLDS=1,5').replace('expected_rows=300','expected_rows=24')
            cell.source=cell.source.replace('CONTEXT_MAX_TOKENS,QUESTION_MAX_TOKENS=192,64','CONTEXT_MAX_TOKENS,QUESTION_MAX_TOKENS=12,4')
            cell.source=cell.source.replace('context_max_tokens=CONTEXT_MAX_TOKENS,question_max_tokens=QUESTION_MAX_TOKENS,',
                'context_max_tokens=CONTEXT_MAX_TOKENS,question_max_tokens=QUESTION_MAX_TOKENS,roberta_max_length=32,roberta_overlap=4,bge_max_length=32,bge_overlap=4,')
        if 'def encoder(cfg, role):' in cell.source:
            cell.source+='\n'+inspect.getsource(TechnicalTokenizer)+'\n'+inspect.getsource(context.PairTechnicalTokenizer)+'\n'+inspect.getsource(ThreeTechnicalTokenizer)+tiny
            cell.source=cell.source.replace('context.PairTechnicalTokenizer','PairTechnicalTokenizer')
        if cell.source.startswith('# Penghubung proses Python:'):
            node=next(n for n in ast.parse(cell.source).body if isinstance(n,ast.Assign))
            cell.source='WORKER_SOURCE='+repr(guard+ast.literal_eval(node.value))+'\n'
    notebook.cells.insert(0,nbformat.v4.new_code_cell(guard));path=isolated/'TECHNICAL_notebook.ipynb';nbformat.write(notebook,path)
    kernels=tmp_path/'kernels';spec=kernels/'aes_fixture';spec.mkdir(parents=True)
    (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-I','-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'AES fixture','language':'python'}))
    env={**os.environ,'AES300_WORKSPACE':str(workspace),'AES300_DATASET':str(fixture),
         'AES300_PERSISTENT_DIR':'/content/drive/MyDrive/SHOULD_NEVER_BE_WRITTEN',
         'AES300_PYTHON':sys.executable,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
         'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','IPYTHONDIR':str(tmp_path/'ipython')}
    env.pop('PYTHONPATH',None);env.pop('AES300_AUDIT_ONLY',None);started=time.time()
    for attempt in (1,2):
        current=nbformat.read(path,as_version=4)
        manager=KernelManager(kernel_name='aes_fixture',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
        try:NotebookClient(current,timeout=360,km=manager,resources={'metadata':{'path':str(isolated)}}).execute(env=env)
        finally:
            if manager.has_kernel:manager.shutdown_kernel(now=True)
        pointer=json.loads((workspace/'outputs/dataset300_download.json').read_text())
        run=Path(pointer['run_folder']);assert workspace in run.parents
        report=json.loads((run/'final_report.json').read_text())
        assert pointer['excel_count']==5 and len(report['cv'])==15 and report['test_evaluations_per_scenario']==1
        assert json.loads((run/'effective_config.json').read_text())['input_format']==mode
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in run.glob('S*/final.pt')}
        if attempt==1:first_hashes=hashes
        else:assert hashes==first_hashes
    import pandas as pd
    data=pd.ExcelFile(run/'hasil_ringkas/01_Data_dan_Tokenisasi.xlsx')
    coverage=pd.read_excel(data,sheet_name='Cakupan Token dan Konteks')
    assert set(coverage['Format Input'])=={mode}
    evaluation=pd.read_excel(run/'hasil_ringkas/05_Riwayat_Pelatihan.xlsx',sheet_name='Pengaturan Eksperimen')
    settings=dict(zip(evaluation['Pengaturan'],evaluation['Nilai']))
    assert settings['Format input tiga teks']==mode and float(settings['Learning rate encoder RoBERTa S1'])==5e-6
    assert not list(workspace.rglob('*.py'))
    proof={'scope':'TECHNICAL RANDOM TRANSFORMER FIXTURE ONLY','format':mode,'code_cells':30,
        'cv_fits':15,'refits':3,'reports':5,'resume_fresh_kernel_same_local_folder':True,
        'drive_not_used':True,'elapsed_seconds':round(time.time()-started,2)}
    (isolated/'execution_proof.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
