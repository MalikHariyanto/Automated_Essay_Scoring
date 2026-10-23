"""LoRA notebook: technical fixtures only, never research performance evidence."""
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
import numpy as np
import pandas as pd
import pytest
import torch
from transformers import RobertaConfig, RobertaModel

from test_teacher_score import TechnicalTokenizer, make_book

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'notebooks/experiments/AES_300_S3_MLP.ipynb'
NOTEBOOK=ROOT/'notebooks/experiments/AES_300_S3_LoRA_MLP_GPU.ipynb'


def runtime():
    notebook=nbformat.read(NOTEBOOK,as_version=4)
    module=types.ModuleType('lora_technical_runtime')
    sys.modules[module.__name__]=module
    ns=module.__dict__
    for index in (9,10,11):
        exec(compile(notebook.cells[index].source.removeprefix('%%aes\n'),str(NOTEBOOK),'exec'),ns)
    for cell in notebook.cells:
        if cell.cell_type!='code':continue
        nodes=[n for n in ast.parse(cell.source.removeprefix('%%aes\n')).body
               if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(NOTEBOOK),'exec'),ns)
    ns['WORD']=__import__('re').compile(r"\b\w+(?:[-’']\w+)*\b",__import__('re').UNICODE)
    ns['show']=lambda *a,**kw:None
    def encoder(cfg,role):
        assert role=='roberta'
        with torch.random.fork_rng():
            torch.manual_seed(170)
            return RobertaModel(RobertaConfig(vocab_size=64,hidden_size=768,
                num_hidden_layers=1,num_attention_heads=12,intermediate_size=16,
                max_position_embeddings=64,pad_token_id=1,bos_token_id=0,eos_token_id=2,
                hidden_dropout_prob=0.,attention_probs_dropout_prob=0.),add_pooling_layer=False)
    ns['encoder']=encoder
    ns['tokenizer']=lambda *a,**kw:TechnicalTokenizer()
    torch.set_num_threads(2)
    return ns


@pytest.fixture
def fixture(tmp_path):
    ns=runtime();path=tmp_path/'TECHNICAL_FIXTURE_ONLY.xlsx';make_book(path)
    cfg=ns['config']({'dataset':str(path),'output':str(tmp_path/'results'),'expected_rows':24,
        'device':'cpu','epochs':3,'roberta_max_length':32,'roberta_overlap':4,
        'bge_max_length':32,'effective_batch':4,'micro_batch':1,'dropout':0.})
    return ns,cfg,ns['read_dataset'](cfg)


def test_standalone_gpu_notebook_and_unchanged_acuan():
    n=nbformat.read(NOTEBOOK,as_version=4);nbformat.validate(n)
    assert n.metadata.aes_experiment.base_sha256==hashlib.sha256(BASE.read_bytes()).hexdigest()
    assert n.metadata.accelerator=='GPU' and not n.metadata.aes_experiment.context_pair
    assert (ROOT/'aes-learning/notebooks/experiments'/NOTEBOOK.name).read_bytes()==NOTEBOOK.read_bytes()
    sources=[]
    for c in n.cells:
        if c.cell_type!='code':continue
        assert c.outputs==[] and c.execution_count is None
        text=c.source.removeprefix('%%aes\n');sources.append(text)
        assert all(x not in text for x in ('clear_training_progress','googleapiclient','pair_batch_windows'))
        for node in ast.walk(ast.parse(text)):
            if isinstance(node,ast.ImportFrom):assert not node.module or not node.module.startswith('aes')
    assert "device='cuda'" in '\n'.join(sources) and 'REQUIRE_GPU=True' in '\n'.join(sources)
    before=NOTEBOOK.read_bytes()
    subprocess.run([sys.executable,str(ROOT/'scripts/build_lora_notebook.py')],check=True,capture_output=True)
    assert NOTEBOOK.read_bytes()==before


def test_gpu_guard_stops_configuration_before_models(fixture,monkeypatch):
    ns,cfg,bundle=fixture
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='GPU belum aktif'):
        ns['check_gpu'](require_gpu=True)
    with pytest.raises(RuntimeError,match='CUDA diminta'):
        ns['device_name']({**cfg,'device':'cuda'})


@pytest.mark.parametrize('key,value',[('lora_rank',0),('lora_lr',0),('lora_alpha',float('nan')),
    ('lora_dropout',1),('lora_targets',['dense'])])
def test_invalid_lora_parameters_fail_before_training(fixture,key,value):
    ns,cfg,_=fixture
    with pytest.raises(ValueError):ns['config']({**cfg,key:value})


def test_only_s3_adapters_train_and_base_remains_identical(fixture):
    ns,cfg,bundle=fixture
    torch.manual_seed(9);model=ns['make_model']('S3',cfg)
    bge=torch.nn.functional.normalize(torch.randn(24,1024),dim=1)
    engine=ns['Engine'](model,bundle.texts,bge,cfg,'cpu')
    before={k:v.detach().clone() for k,v in model.roberta.named_parameters() if not v.requires_grad}
    params=[n for n,p in model.roberta.named_parameters() if p.requires_grad]
    assert params and all('.lora_A.' in n or '.lora_B.' in n for n in params)
    groups=ns['optimizer_groups'](model,cfg)
    assert {g['name'] for g in groups}=={'lora','projection','head'}
    assert next(g for g in groups if g['name']=='lora')['lr']==cfg['lora_lr']
    actual=[id(p) for g in groups for p in g['params']]
    assert len(actual)==len(set(actual)) and set(actual)=={id(p) for p in model.parameters() if p.requires_grad}
    optim=torch.optim.AdamW(groups)
    for _ in range(2):
        optim.zero_grad();p,features=engine.forward([0,1],True)
        assert p.shape==(2,1) and features['concat'].shape==(2,1024)
        (p-.8).square().mean().backward();optim.step()
    for name,value in before.items():torch.testing.assert_close(dict(model.roberta.named_parameters())[name],value,rtol=0,atol=0)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.roberta.named_parameters() if '.lora_B.' in n)
    assert model.head[0].weight.grad.abs().sum()>0 and model.projection[0].weight.grad.abs().sum()>0
    assert len(ns['training_state'](model))<len(model.state_dict())
    assert all('base_layer.weight' not in key for key in ns['training_state'](model))
    assert any(r['Bagian']=='Bobot asli RoBERTa' and r['Parameter Dilatih']==0 for r in ns['parameter_report'](model))


@pytest.mark.parametrize('scenario',['S1','S2'])
def test_control_architectures_and_predictions_match_acuan(fixture,scenario):
    ns,cfg,bundle=fixture
    base=nbformat.read(BASE,as_version=4)
    text=next(c.source for c in base.cells if c.cell_type=='code' and 'class SingleScoreModel' in c.source)
    other=dict(ns);exec(text.removeprefix('%%aes\n'),other)
    torch.manual_seed(22);control=other['SingleScoreModel'](scenario,ns['encoder'](cfg,'roberta') if scenario=='S1' else None,dropout=0.)
    torch.manual_seed(22);actual=ns['make_model'](scenario,cfg)
    assert control.state_dict().keys()==actual.state_dict().keys()
    for key,value in control.state_dict().items():torch.testing.assert_close(value,actual.state_dict()[key],rtol=0,atol=0)
    if scenario=='S1':assert all(p.requires_grad for p in actual.roberta.parameters())
    else:assert actual.roberta is None


def test_lora_starts_as_same_function_and_keeps_original_split_metrics(fixture):
    ns,cfg,bundle=fixture
    base=nbformat.read(BASE,as_version=4)
    for name in ('make_split','metrics','batch_windows'):
        def definition(n):
            for c in n.cells:
                if c.cell_type!='code':continue
                for node in ast.parse(c.source.removeprefix('%%aes\n')).body:
                    if isinstance(node,ast.FunctionDef) and node.name==name:
                        return ast.dump(node,include_attributes=False)
            raise AssertionError(name)
        assert definition(base)==definition(nbformat.read(NOTEBOOK,as_version=4))
    torch.manual_seed(22)
    plain=ns['SingleScoreModel']('S3',ns['encoder'](cfg,'roberta'),dropout=0.).eval()
    torch.manual_seed(22);adapted=ns['make_model']('S3',cfg).eval()
    bge=torch.nn.functional.normalize(torch.randn(24,1024),dim=1)
    expected=ns['Engine'](plain,bundle.texts,bge,cfg,'cpu').predict([0,1])
    actual=ns['Engine'](adapted,bundle.texts,bge,cfg,'cpu').predict([0,1])
    np.testing.assert_array_equal(expected,actual)


def test_compact_checkpoint_roundtrip_and_vector_adapter_active(fixture,tmp_path):
    ns,cfg,bundle=fixture
    model=ns['make_model']('S3',cfg).eval()
    with torch.no_grad():
        for name,p in model.roberta.named_parameters():
            if '.lora_B.' in name:p.normal_(0,.02)
    bge=torch.nn.functional.normalize(torch.randn(24,1024),dim=1)
    expected=ns['Engine'](model,bundle.texts,bge,cfg,'cpu').predict([0,1])
    state=ns['training_state'](model);assert set(state)=={n for n,p in model.named_parameters() if p.requires_grad}
    manifest=ns['make_split'](bundle,cfg);file=tmp_path/'final.pt'
    ns['save_model'](file,model,cfg,bundle,manifest,{'technical_fixture':True})
    restored,payload=ns['load_model'](file,cfg)
    assert payload['checkpoint_format']=='lora_trainable_only_v1'
    np.testing.assert_array_equal(expected,ns['Engine'](restored,bundle.texts,bge,cfg,'cpu').predict([0,1]))
    damaged=dict(state);damaged.pop(next(iter(damaged)))
    with pytest.raises(ValueError,match='seluruh LoRA'):ns['restore_training_state'](restored,damaged)
    with pytest.raises(ValueError,match='Metadata checkpoint'):ns['load_model'](file,{**cfg,'lora_alpha':32})
    tok=TechnicalTokenizer();text=bundle.texts[0]
    actual=ns['token_vectors'](text,restored.roberta,tok,tok,'roberta',cfg,'cpu')
    base=ns['token_vectors'](text,ns['encoder'](cfg,'roberta'),tok,tok,'roberta',cfg,'cpu')
    np.testing.assert_array_equal(actual['initial'],base['initial'])
    assert not np.array_equal(actual['contextual'],base['contextual'])


def test_resume_compact_progress_matches_uninterrupted_training(fixture,tmp_path):
    ns,cfg,bundle=fixture
    bge=torch.nn.functional.normalize(torch.arange(24*1024,dtype=torch.float32).reshape(24,1024)+1,dim=1)
    model,engine,expected=ns['fit']('S3',list(range(8)),list(range(8,12)),bundle,bge,cfg,3,42)
    expected_state={k:v.detach().clone() for k,v in model.state_dict().items()};predicted=engine.predict([12,13])
    path=tmp_path/'progress_fold_1.json';save=ns['save_training_progress']
    def interrupt(path,payload):
        save(path,payload)
        if payload['epoch']==1:raise RuntimeError('TECHNICAL INTERRUPT')
    ns['save_training_progress']=interrupt
    with pytest.raises(RuntimeError,match='TECHNICAL INTERRUPT'):
        ns['fit']('S3',list(range(8)),list(range(8,12)),bundle,bge,cfg,3,42,checkpoint=path)
    ns['save_training_progress']=save
    actual,engine,info=ns['fit']('S3',list(range(8)),list(range(8,12)),bundle,bge,cfg,3,42,checkpoint=path)
    assert expected==info
    for name,value in expected_state.items():torch.testing.assert_close(value,actual.state_dict()[name],rtol=0,atol=0)
    np.testing.assert_array_equal(predicted,engine.predict([12,13]))
    assert len(list(tmp_path.glob('progress_fold_1_epoch_*.pt')))==3
    assert len(json.loads(path.read_text())['snapshots'])==2


def test_execute_notebook_all_stages_and_resume_fresh_runtime(tmp_path):
    """Actual cell execution with local random Transformers; not a GPU/pretrained trial."""
    notebook=nbformat.read(NOTEBOOK,as_version=4)
    isolated=tmp_path/'isolated';isolated.mkdir();path=isolated/'TECHNICAL_FIXTURE_ONLY.xlsx';make_book(path)
    guard="""
import builtins
_original_import=builtins.__import__
def _no_project_import(name,*args,**kwargs):
    if name=='aes' or name.startswith('aes.'):
        raise ImportError('Project imports forbidden in standalone fixture')
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
    for cell in notebook.cells:
        if cell.cell_type!='code':continue
        if 'EPOCHS,FOLDS=25,5' in cell.source:
            # Test override only; published notebook remains CUDA-required.
            cell.source=cell.source.replace('REQUIRE_GPU=True','REQUIRE_GPU=False').replace("device='cuda'","device='cpu'")
            cell.source=cell.source.replace('EPOCHS,FOLDS=25,5','EPOCHS,FOLDS=1,5').replace('expected_rows=300','expected_rows=24')
            cell.source=cell.source.replace('epochs=EPOCHS,folds=FOLDS,','epochs=EPOCHS,folds=FOLDS,roberta_max_length=32,roberta_overlap=4,bge_max_length=32,')
            cell.source=cell.source.replace("lesson_version='s3_lora_mlp_gpu_v1'","lesson_version='TECHNICAL_FIXTURE_ONLY'")
        if 'def encoder(cfg, role):' in cell.source:
            cell.source+='\n'+inspect.getsource(TechnicalTokenizer)+'\n'+tiny
        if cell.source.startswith('# Penghubung proses Python:'):
            node=next(n for n in ast.parse(cell.source).body if isinstance(n,ast.Assign))
            cell.source='WORKER_SOURCE='+repr(guard+ast.literal_eval(node.value))+'\n'
    first=nbformat.v4.new_code_cell(guard);first.pop('id',None)
    notebook.cells.insert(0,first)
    kernels=tmp_path/'kernels';spec=kernels/'aes_fixture';spec.mkdir(parents=True)
    (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-I','-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'AES fixture','language':'python'}))
    env={**os.environ,'AES300_DATASET':str(path),'AES300_PERSISTENT_DIR':str(tmp_path/'TECHNICAL_PERSISTENT'),
         'AES300_PYTHON':sys.executable,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
         'IPYTHONDIR':str(tmp_path/'ipython')}
    env.pop('PYTHONPATH',None);env.pop('AES300_AUDIT_ONLY',None)
    hashes=None
    for attempt in (1,2):
        current=nbformat.reads(nbformat.writes(notebook),as_version=4)
        workspace=isolated/f'runtime_{attempt}'
        manager=KernelManager(kernel_name='aes_fixture',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
        try:
            NotebookClient(current,timeout=360,km=manager,resources={'metadata':{'path':str(isolated)}}).execute(env={**env,'AES300_WORKSPACE':str(workspace)})
        finally:
            if manager.has_kernel:manager.shutdown_kernel(now=True)
        nbformat.write(current,isolated/f'TECHNICAL_executed_{attempt}.ipynb')
        pointer=json.loads((workspace/'outputs/dataset300_download.json').read_text())
        assert pointer['excel_count']==5
        run=Path(pointer['run_folder']);report=json.loads((run/'final_report.json').read_text())
        assert len(report['cv'])==15 and report['test_evaluations_per_scenario']==1
        current_hashes={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in run.glob('S*/final.pt')}
        if attempt==1:hashes=current_hashes
        else:
            assert hashes==current_hashes
            streams='\n'.join(str(o.get('text','')) for c in current.cells for o in c.get('outputs',[]) if o.output_type=='stream')
            assert 'Memakai hasil tersimpan yang cocok' in streams and ' · epoch ' not in streams
    reports=run/'hasil_ringkas'
    train=pd.ExcelFile(reports/'05_Riwayat_Pelatihan.xlsx')
    assert 'Parameter LoRA dan Fusion' in train.sheet_names
    settings=pd.read_excel(train,sheet_name='Pengaturan Eksperimen')
    assert 'Rank LoRA S3' in settings['Pengaturan'].tolist()
    score=pd.ExcelFile(reports/'04_Prediksi_dan_Evaluasi.xlsx')
    assert any('Guru' in sheet for sheet in score.sheet_names)
    comparison=pd.read_excel(score,sheet_name='Nilai Guru dan Prediksi')
    assert {'Nilai Guru','Prediksi S1','Prediksi S2','Prediksi S3'}<=set(comparison.columns)
    vectors=pd.ExcelFile(reports/'02_Vektor_Jawaban.xlsx')
    assert 'RoBERTa S3 Setelah Latihan' in vectors.sheet_names
    assert not any('Pair' in s for s in vectors.sheet_names)
    before=pd.read_excel(vectors,sheet_name='RoBERTa Sebelum Latihan').iloc[:,1:].to_numpy()
    after=pd.read_excel(vectors,sheet_name='RoBERTa S3 Setelah Latihan').iloc[:,1:].to_numpy()
    assert before.shape==after.shape==(24,768) and not np.array_equal(before,after)
    payload=torch.load(run/'S3/final.pt',weights_only=True,map_location='cpu')
    assert payload['checkpoint_format']=='lora_trainable_only_v1'
    assert any('.lora_B.' in key for key in payload['state'])
    assert not any('base_layer.weight' in key for key in payload['state'])
    assert list((run/'S3').glob('progress_fold_1_epoch_*.pt'))
