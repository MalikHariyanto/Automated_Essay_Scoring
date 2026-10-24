"""Pasangan input dan notebook mandiri; seluruh data uji merupakan fixture teknis."""
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import subprocess
import time
import types

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
import numpy as np
import pytest
import torch

from aes.teacher_score.context_pair import (context_token_ids, contexts_from_bundle,
    pair_content_windows, pair_batch_windows, paired_token_vectors)
from test_teacher_score import TechnicalTokenizer, make_book, encoders

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT/'notebooks/experiments'/os.environ.get('AES_CONTEXT_NOTEBOOK','AES_300_S3_ContextPair_MLP.ipynb')
IS_E8 = NOTEBOOK.name.startswith('AES_300_E8_ContextPair_LR5e6_')
IS_CPU = IS_E8 and NOTEBOOK.name.endswith('_CPU.ipynb')
BASE = ROOT/'notebooks/experiments'/('AES_300_S3_ContextPair_MLP.ipynb' if IS_E8 else 'AES_300_S3_MLP.ipynb')


class PairTechnicalTokenizer(TechnicalTokenizer):
    """Template RoBERTa fixture; bukan kosakata pretrained atau teks siswa."""
    def num_special_tokens_to_add(self, pair=False):
        return 4 if pair else 2

    def build_inputs_with_special_tokens(self, ids, pair_ids=None):
        return [0]+list(ids)+[2] if pair_ids is None else [0]+list(ids)+[2,2]+list(pair_ids)+[2]


def runtime():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    module = types.ModuleType('context_pair_fixture_runtime')
    sys.modules[module.__name__] = module
    ns = module.__dict__
    for index in (9,10,11):
        exec(compile(notebook.cells[index].source.removeprefix('%%aes\n'), str(NOTEBOOK), 'exec'), ns)
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        nodes = [n for n in ast.parse(cell.source.removeprefix('%%aes\n')).body
                 if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), 'exec'), ns)
    ns['WORD'] = __import__('re').compile(r"\b\w+(?:[-’']\w+)*\b", __import__('re').UNICODE)
    ns['show'] = lambda *args, **kwargs: None
    return ns


@pytest.fixture
def pair_dataset(tmp_path):
    ns = runtime()
    path = tmp_path/'TECHNICAL_FIXTURE_ONLY.xlsx'
    make_book(path)
    cfg = ns['config']({'dataset': str(path), 'output': str(tmp_path/'results'), 'expected_rows':24,
        'device':'cpu', 'epochs':3, 'roberta_max_length':32, 'roberta_overlap':4,
        'bge_max_length':32, 'context_max_tokens':12, 'question_max_tokens':4,
        'effective_batch':4, 'micro_batch':1, 'dropout':0.})
    return ns, cfg, ns['read_dataset'](cfg)


def test_clean_isolated_standalone_notebook_and_originals_unchanged():
    n = nbformat.read(NOTEBOOK, as_version=4)
    assert len(n.cells)==47 and sum(c.cell_type=='code' for c in n.cells)==30
    assert n.metadata.aes_experiment.base_sha256==hashlib.sha256(BASE.read_bytes()).hexdigest()
    assert n.metadata.aes_experiment.name==('E8_CONTEXT_PAIR_LR5E6_CPU' if IS_CPU else 'E8_CONTEXT_PAIR_LR5E6_GPU' if IS_E8 else 'S3_CONTEXT_PAIR_MLP')
    if IS_CPU:assert 'accelerator' not in n.metadata
    else:assert n.metadata.accelerator=='GPU'
    assert not n.metadata.aes_experiment.automatic_checkpoint_deletion
    assert (ROOT/'aes-learning/notebooks/experiments'/NOTEBOOK.name).read_bytes()==NOTEBOOK.read_bytes()
    assert n.metadata.aes_standalone.code_sha256!=nbformat.read(BASE,as_version=4).metadata.aes_standalone.code_sha256
    for cell in n.cells:
        if cell.cell_type!='code':continue
        assert cell.outputs==[] and cell.execution_count is None
        code=cell.source.removeprefix('%%aes\n')
        for forbidden in ('DriveCheckpointCleaner','googleapiclient','_drive_cleanup','clear_training_progress'):
            assert forbidden not in code
        for node in ast.walk(ast.parse(code)):
            if isinstance(node,ast.ImportFrom):assert not node.module or not node.module.startswith('aes')
            if isinstance(node,ast.Import):assert not any(v.name=='aes' or v.name.startswith('aes.') for v in node.names)


def test_builder_is_deterministic_across_python_versions():
    before=NOTEBOOK.read_bytes()
    for executable in ('python',sys.executable):
        script='build_e8_context_lr_cpu_notebook.py' if IS_CPU else 'build_e8_context_lr_notebook.py' if IS_E8 else 'build_context_pair_notebook.py'
        subprocess.run([executable,str(ROOT/'scripts'/script)],cwd=ROOT,check=True,capture_output=True)
        assert NOTEBOOK.read_bytes()==before


def test_gpu_guard_blocks_cpu_and_accepts_reported_cuda_device():
    """Pemeriksaan perangkat dengan CUDA tiruan; bukan eksekusi GPU asli."""
    notebook=nbformat.read(NOTEBOOK,as_version=4)
    code=next(c.source.removeprefix('%%aes\n') for c in notebook.cells
              if c.cell_type=='code' and 'def check_gpu(' in c.source)
    node=next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name=='check_gpu')
    cuda=types.SimpleNamespace(is_available=lambda:False)
    ns={'torch':types.SimpleNamespace(cuda=cuda)}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(NOTEBOOK),'exec'),ns)
    with pytest.raises(RuntimeError,match='GPU belum aktif'):
        ns['check_gpu'](require_gpu=True)
    assert ns['check_gpu'](require_gpu=False)['name']=='CPU'
    cuda.is_available=lambda:True
    cuda.get_device_name=lambda index:'NVIDIA T4'
    cuda.get_device_properties=lambda index:types.SimpleNamespace(total_memory=16*1024**3)
    assert ns['check_gpu'](require_gpu=True)=={'name':'NVIDIA T4','cuda_available':True,'is_t4':True,'memory_gib':16.0}
    cuda.get_device_name=lambda index:'NVIDIA L4'
    assert not ns['check_gpu'](require_gpu=True)['is_t4']


def test_pair_layout_repeats_context_and_covers_entire_answer():
    tok=PairTechnicalTokenizer();cfg={'context_max_tokens':12,'question_max_tokens':4}
    context=('technical question '*30,'technical reference '*30)
    text=' '.join('answer'+str(i) for i in range(73))
    chunks,info=pair_content_windows(text,context,tok,32,4,cfg)
    cids,detail=context_token_ids(context,tok,cfg)
    assert detail['question_truncated'] and detail['reference_truncated']
    assert detail['question_kept']==4 and detail['reference_kept']==8
    assert info['capacity']==16 and info['pair_special_tokens']==4
    ids=tok.encode(text,add_special_tokens=False);observed={}
    for chunk,start,positions in zip(chunks,info['starts'],info['answer_positions']):
        assert chunk['input_ids'][1:13]==cids
        assert chunk['input_ids'][13:15]==[2,2]
        assert max(chunk['input_ids'])<64 and min(chunk['input_ids'])>=0
        assert len(chunk['input_ids'])<=32
        assert [chunk['input_ids'][i] for i in positions]==ids[start:start+16]
        observed.update({start+j:chunk['input_ids'][p] for j,p in enumerate(positions)})
    assert [observed[i] for i in range(len(ids))]==ids
    assert info['answer_tokens_uncovered']==0


def test_batch_alignment_padding_and_invalid_contexts():
    tok=PairTechnicalTokenizer();cfg={'context_max_tokens':12,'question_max_tokens':4}
    texts=['long technical answer '*30,'short answer']
    contexts=[('question one','reference one'),('question two','reference two')]
    batch,mapping,info=pair_batch_windows(texts,contexts,tok,'cpu',32,4,cfg)
    assert mapping.tolist()==[0]*info[0]['windows']+[1]
    assert batch['input_ids'].shape[1]<=32
    assert batch['attention_mask'][-1].sum()<batch['input_ids'].shape[1]
    assert (batch['input_ids'][batch['attention_mask']==0]==1).all()
    with pytest.raises(ValueError,match='jumlah jawaban'):
        pair_batch_windows(texts,contexts[:1],tok,'cpu',32,4,cfg)
    with pytest.raises(ValueError,match='soal'):
        context_token_ids(('', 'reference'),tok,cfg)
    with pytest.raises(ValueError,match='overlap'):
        pair_content_windows('answer',contexts[0],tok,16,4,cfg)


def test_real_roberta_tokenizer_template_with_local_technical_vocabulary(tmp_path):
    from tokenizers import ByteLevelBPETokenizer
    from transformers import RobertaTokenizer, RobertaTokenizerFast
    trainer=ByteLevelBPETokenizer()
    corpus=tmp_path/'TECHNICAL_CORPUS.txt'
    corpus.write_text('Soal: technical question\nJawaban referensi: technical reference\nalpha response alpha final tail token\n')
    trainer.train(files=[str(corpus)],vocab_size=300,special_tokens=['<s>','<pad>','</s>','<unk>','<mask>'])
    trainer.save_model(str(tmp_path))
    slow=RobertaTokenizer(vocab_file=str(tmp_path/'vocab.json'),merges_file=str(tmp_path/'merges.txt'))
    fast=RobertaTokenizerFast(vocab_file=str(tmp_path/'vocab.json'),merges_file=str(tmp_path/'merges.txt'))
    cfg={'context_max_tokens':40,'question_max_tokens':12}
    text='alpha response '*70+'final tail token'
    chunks,stats=pair_content_windows(text,('technical question','technical reference'),slow,96,16,cfg)
    assert slow.num_special_tokens_to_add(True)==4
    ids=slow.encode(text,add_special_tokens=False)
    assert ids==fast(text,add_special_tokens=False)['input_ids']
    context,_=context_token_ids(('technical question','technical reference'),slow,cfg)
    for chunk,start,positions in zip(chunks,stats['starts'],stats['answer_positions']):
        part=ids[start:start+stats['capacity']]
        expected=slow.prepare_for_model(context,part,add_special_tokens=True,return_token_type_ids=False)
        assert chunk['input_ids']==expected['input_ids']
        assert [chunk['input_ids'][p] for p in positions]==part
    assert stats['starts'][-1]+len(stats['answer_positions'][-1])==len(ids)


def test_context_never_reads_teacher_score_and_split_matches_acuan(pair_dataset):
    ns,cfg,bundle=pair_dataset
    context=contexts_from_bundle(bundle)
    bundle.y[:]=999
    for row in bundle.rows:row['score']=-999
    assert contexts_from_bundle(bundle)==context
    old=nbformat.read(BASE,as_version=4)
    cell=next(c for c in old.cells if c.cell_type=='code' and 'def make_split(' in c.source)
    function=next(n for n in ast.parse(cell.source.removeprefix('%%aes\n')).body if isinstance(n,ast.FunctionDef))
    other=dict(ns);exec(compile(ast.Module(body=[function],type_ignores=[]),'baseline-split','exec'),other)
    assert ns['make_split'](bundle,cfg)==other['make_split'](bundle,cfg)


def test_new_prediction_requires_its_own_question_and_reference(pair_dataset):
    ns,cfg,bundle=pair_dataset
    with pytest.raises(ValueError,match='NEW_QUESTION'):
        ns['config']({**cfg,'new_answer':'technical new answer'})
    valid=ns['config']({**cfg,'new_answer':'technical new answer','new_question':'technical question','new_reference':'technical reference'})
    assert valid['new_reference']=='technical reference'
    row=dict(bundle.rows[0]);row['question']='technical\r\nquestion';row['reference']='technical\x00reference'
    assert contexts_from_bundle(types.SimpleNamespace(rows=[row]))==[('technical\nquestion','technicalreference')]


@pytest.mark.parametrize('scenario',['S1','S2','S3'])
def test_forward_features_context_effect_and_trainable_layers(pair_dataset,encoders,scenario):
    ns,cfg,bundle=pair_dataset
    torch.manual_seed(99)
    model=ns['SingleScoreModel'](scenario,encoders(cfg,'roberta') if scenario!='S2' else None)
    model.eval();bge=torch.nn.functional.normalize(torch.randn(24,1024),dim=1)
    contexts=contexts_from_bundle(bundle)
    engine=ns['Engine'](model,bundle.texts,bge,cfg,'cpu',roberta_tokenizer=PairTechnicalTokenizer(),contexts=contexts)
    prediction,features=engine.forward([0],return_features=True)
    assert prediction.shape==(1,1)
    if scenario=='S3':
        assert features['roberta'].shape==(1,768) and features['concat'].shape==(1,1024)
        assert all(p.requires_grad for p in model.roberta.parameters())
        engine.contexts=[('changed question','changed reference')]*24
        _,changed=engine.forward([0],return_features=True)
        assert not torch.equal(features['roberta'],changed['roberta'])
        assert torch.equal(features['bge'],changed['bge'])
        prediction.square().mean().backward()
        assert model.roberta.embeddings.word_embeddings.weight.grad.abs().sum()>0
        assert model.projection[0].weight.grad.abs().sum()>0
        assert model.head[0].weight.grad.abs().sum()>0
    elif scenario=='S1':
        engine.contexts=[('changed','changed')]*24
        torch.testing.assert_close(prediction,engine.forward([0]),atol=0,rtol=0)
    else:
        torch.testing.assert_close(prediction,model.head(bge[:1]),atol=0,rtol=0)


def test_pair_word_vectors_use_actual_answer_offsets(pair_dataset,encoders):
    ns,cfg,bundle=pair_dataset;tok=PairTechnicalTokenizer();model=encoders(cfg,'roberta').eval()
    context=('technical question','technical reference')
    text='alpha technical response. alpha '*8
    trace=paired_token_vectors(text,model,tok,tok,cfg,'cpu',context)
    assert (trace['counts']>0).all()
    assert trace['contextual'].shape==(len(trace['ids']),768)
    _,changed=pair_content_windows(text,context,tok,32,4,cfg)
    assert trace['starts']==changed['starts']
    other=paired_token_vectors(text,model,tok,tok,cfg,'cpu',('different question','different reference'))
    np.testing.assert_array_equal(trace['initial'],other['initial'])
    assert not np.array_equal(trace['contextual'],other['contextual'])


def test_interrupted_fine_tuning_resumes_exactly_and_keeps_old_checkpoints(pair_dataset,encoders,tmp_path):
    ns,cfg,bundle=pair_dataset
    ns['encoder']=encoders;ns['tokenizer']=lambda *args,**kwargs:PairTechnicalTokenizer()
    torch.set_num_threads(2)
    bge=torch.nn.functional.normalize(torch.arange(24*1024,dtype=torch.float32).reshape(24,1024)+1,dim=1)
    model,engine,expected=ns['fit']('S3',list(range(8)),list(range(8,12)),bundle,bge,cfg,3,42)
    state={k:v.detach().clone() for k,v in model.state_dict().items()};prediction=engine.predict([12,13])
    checkpoint=tmp_path/'progress_fold_1.json';save=ns['save_training_progress']
    def interrupted(path,payload):
        save(path,payload)
        if payload['epoch']==1:raise RuntimeError('TECHNICAL INTERRUPT')
    ns['save_training_progress']=interrupted
    with pytest.raises(RuntimeError,match='TECHNICAL INTERRUPT'):
        ns['fit']('S3',list(range(8)),list(range(8,12)),bundle,bge,cfg,3,42,checkpoint=checkpoint)
    ns['save_training_progress']=save
    resumed,engine,actual=ns['fit']('S3',list(range(8)),list(range(8,12)),bundle,bge,cfg,3,42,checkpoint=checkpoint)
    assert expected==actual
    for key,value in state.items():torch.testing.assert_close(value,resumed.state_dict()[key],atol=0,rtol=0)
    np.testing.assert_array_equal(prediction,engine.predict([12,13]))
    assert len(list(tmp_path.glob('progress_fold_1_slot_*.pt' if IS_E8 else 'progress_fold_1_epoch_*.pt')))==(2 if IS_E8 else 3)
    assert len(json.loads(checkpoint.read_text())['snapshots'])==2


def test_notebook_executes_all_stages_and_resumes_from_persistent_results(tmp_path):
    """Actual standalone notebook; random tiny Transformers, never research metrics."""
    notebook=nbformat.read(NOTEBOOK,as_version=4)
    isolated=tmp_path/'standalone';isolated.mkdir()
    fixture=isolated/'TECHNICAL_FIXTURE_ONLY.xlsx';make_book(fixture)
    persistent=tmp_path/'TECHNICAL_PERSISTENT_DRIVE';workspace=isolated/'workspace'
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
def tokenizer(*args,**kwargs):return PairTechnicalTokenizer()
"""
    for cell in notebook.cells:
        if cell.cell_type!='code':continue
        if 'EPOCHS,FOLDS=25,5' in cell.source:
            # Fixture teknis berjalan di CPU; notebook pengguna tetap mewajibkan GPU.
            assert ('REQUIRE_GPU=False' if IS_CPU else 'REQUIRE_GPU=True') in cell.source
            cell.source=cell.source.replace('REQUIRE_GPU=True','REQUIRE_GPU=False')
            cell.source=cell.source.replace('EPOCHS,FOLDS=25,5','EPOCHS,FOLDS=1,5').replace('expected_rows=300','expected_rows=24')
            cell.source=cell.source.replace('CONTEXT_MAX_TOKENS,QUESTION_MAX_TOKENS=192,64','CONTEXT_MAX_TOKENS,QUESTION_MAX_TOKENS=12,4')
            cell.source=cell.source.replace("context_max_tokens=CONTEXT_MAX_TOKENS,question_max_tokens=QUESTION_MAX_TOKENS,", "context_max_tokens=CONTEXT_MAX_TOKENS,question_max_tokens=QUESTION_MAX_TOKENS,roberta_max_length=32,roberta_overlap=4,bge_max_length=32,")
            cell.source=cell.source.replace("lesson_version='s3_context_pair_mlp_v1'","lesson_version='TECHNICAL_FIXTURE_ONLY'")
        if "journal.save('01_konfigurasi'" in cell.source:
            cell.source=cell.source.replace("journal.save('01_konfigurasi'","journal.metadata['evidence_scope']='TECHNICAL RANDOM TRANSFORMER FIXTURE; NOT STUDENT RESULTS'\njournal.save('01_konfigurasi'")
        if 'def encoder(cfg, role):' in cell.source:
            cell.source+='\n'+inspect.getsource(TechnicalTokenizer)+'\n'+inspect.getsource(PairTechnicalTokenizer)+tiny
        if cell.source.startswith('# Penghubung proses Python:'):
            node=next(n for n in ast.parse(cell.source).body if isinstance(n,ast.Assign))
            cell.source='WORKER_SOURCE='+repr(guard+ast.literal_eval(node.value))+'\n'
    notebook.cells.insert(0,nbformat.v4.new_code_cell(guard))
    path=isolated/'TECHNICAL_notebook.ipynb';nbformat.write(notebook,path)
    kernels=tmp_path/'kernels';spec=kernels/'aes_fixture';spec.mkdir(parents=True)
    (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-I','-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'AES fixture','language':'python'}))
    environment={**os.environ,'AES300_WORKSPACE':str(workspace),'AES300_DATASET':str(fixture),
        'AES300_PERSISTENT_DIR':str(persistent),'AES300_PYTHON':sys.executable,
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
        'IPYTHONDIR':str(tmp_path/'ipython')}
    environment.pop('PYTHONPATH',None);environment.pop('AES300_AUDIT_ONLY',None)
    start=time.time()
    for attempt in (1,2):
        current=nbformat.read(path,as_version=4)
        manager=KernelManager(kernel_name='aes_fixture',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
        try:
            NotebookClient(current,timeout=360,km=manager,resources={'metadata':{'path':str(isolated)}}).execute(
                env={**environment,'AES300_WORKSPACE':str(workspace if attempt==1 else isolated/'fresh_runtime')})
        finally:
            if manager.has_kernel:manager.shutdown_kernel(now=True)
        nbformat.write(current,isolated/f'TECHNICAL_executed_{attempt}.ipynb')
        pointer=json.loads(((workspace if attempt==1 else isolated/'fresh_runtime')/'outputs/dataset300_download.json').read_text())
        assert pointer['excel_count']==5
        run=Path(pointer['run_folder']);report=json.loads((run/'final_report.json').read_text())
        assert len(report['cv'])==15 and report['test_evaluations_per_scenario']==1
        if attempt==1:
            initial_hashes={p.name+str(p.parent):hashlib.sha256(p.read_bytes()).hexdigest() for p in run.glob('S*/final.pt')}
        else:
            assert initial_hashes=={p.name+str(p.parent):hashlib.sha256(p.read_bytes()).hexdigest() for p in run.glob('S*/final.pt')}
            streams='\n'.join(out.get('text','') for c in current.cells for out in c.get('outputs',[]) if out.output_type=='stream')
            assert 'Memakai hasil tersimpan yang cocok' in streams and ' · epoch ' not in streams
    import pandas as pd
    root=run/'hasil_ringkas'
    data=pd.ExcelFile(root/'01_Data_dan_Tokenisasi.xlsx')
    assert 'Token RoBERTa S3 Pair' in data.sheet_names and 'Cakupan Token dan Konteks' in data.sheet_names
    tokens=pd.read_excel(data,sheet_name='Token RoBERTa S3 Pair')
    assert set(tokens['Bagian Input'])=={'Soal/referensi','Jawaban siswa','Token khusus'}
    features=pd.ExcelFile(root/'02_Vektor_Jawaban.xlsx')
    assert 'RoBERTa S3 Pair Awal' in features.sheet_names
    before=pd.read_excel(features,sheet_name='RoBERTa S3 Pair Awal').iloc[:,1:].to_numpy()
    after=pd.read_excel(features,sheet_name='RoBERTa S3 Setelah Latihan').iloc[:,1:].to_numpy()
    assert before.shape==after.shape==(24,768) and not np.array_equal(before,after)
    words=pd.ExcelFile(root/'03_Vektor_Kata_dan_Kalimat.xlsx')
    assert 'Kata S3 Pair Sebelum' in words.sheet_names
    pair_report=json.loads((run/'vectors/S3/report.json').read_text())
    assert pair_report['roberta']['input_policy']=='soal/referensi berpasangan dengan jawaban'
    assert pair_report['roberta']['uncovered_tokens']==0
    assert pair_report['bge']['input_policy']=='jawaban siswa saja'
    evaluation=pd.read_excel(root/'04_Prediksi_dan_Evaluasi.xlsx',sheet_name='Nilai Guru dan Prediksi')
    assert all(k in evaluation for k in ('Nilai Guru','Prediksi S3','Pembulatan S3'))
    for scenario in ('S1','S2','S3'):
        payload=torch.load(run/scenario/'final.pt',map_location='cpu',weights_only=True)
        assert payload['training']['BGE_encoder_trained'] is False
        assert (run/scenario/'progress_refit.json').is_file()
        assert len(list((run/scenario).glob('progress_fold_*_slot_A.pt' if IS_E8 else 'progress_fold_*_epoch_001.pt')))==5
    assert not list(workspace.rglob('*.py')) and not (workspace/'src').exists()
    proof={'scope':'TECHNICAL RANDOM TRANSFORMER FIXTURE ONLY; NO STUDENT RESULTS',
        'production_code_cells_executed':30,'cv_folds':15,'reports':5,'resume_fresh_runtime':True,
        'project_imports_blocked':True,'elapsed_seconds':round(time.time()-start,2)}
    (isolated/'context_pair_execution_proof.json').write_text(json.dumps(proof,indent=2))
    print(json.dumps(proof))
