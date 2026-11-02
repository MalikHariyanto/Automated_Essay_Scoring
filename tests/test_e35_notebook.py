"""Native E35: inspectable namespace, unchanged model/training math, actual notebook execution."""
import ast
import copy
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
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests'))
import test_context_pair_notebook as context
from test_e6_three_input_notebook import ThreeTechnicalTokenizer
from test_teacher_score import TechnicalTokenizer, make_book, encoders

NAME = 'AES_300_E35_E30_LayersLR5e5_Native_Learnable_GPU.ipynb'
NOTEBOOK = ROOT/'notebooks/experiments'/NAME
BASE = ROOT/'notebooks/experiments/AES_300_E30_E26_BiGRU_CoAttention_Pooling_GPU.ipynb'


def native_runtime():
    n = nbformat.read(NOTEBOOK, as_version=4)
    module = types.ModuleType('e35_native_fixture_runtime')
    sys.modules[module.__name__] = module
    ns = module.__dict__
    safe = {'imports','model_names','input_columns','hyperparameters','runtime_flags',
            'configuration_defaults','presentation_helpers','word_helpers'}
    for c in n.cells:
        if c.cell_type == 'code' and c.metadata.aes35_role in safe:
            exec(compile(c.source, str(NOTEBOOK), 'exec'), ns)
    for c in n.cells:
        if c.cell_type != 'code':
            continue
        nodes = [v for v in ast.parse(c.source).body if isinstance(v,(ast.FunctionDef,ast.ClassDef))]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(NOTEBOOK),'exec'),ns)
    ns['show'] = lambda *a, **k: None
    return ns


def test_native_structure_parameters_code_hash_and_deterministic_builder():
    raw = NOTEBOOK.read_bytes()
    n = nbformat.reads(raw.decode(), as_version=4)
    nbformat.validate(n)
    fp = n.metadata.aes_standalone.code_sha256
    cells = [c for c in n.cells if c.cell_type == 'code']
    assert n.metadata.aes_standalone.native_kernel
    assert n.metadata.aes_experiment.s3_mlp_lr == 1e-4
    assert n.metadata.aes_experiment.s3_layers_lr == 5e-5
    assert n.metadata.aes_experiment.base_sha256 == hashlib.sha256(BASE.read_bytes()).hexdigest()
    assert not n.metadata.aes_experiment.inspection_cosine
    all_code = '\n'.join(c.source for c in cells)
    for banned in ['%%aes','aes_cell(', 'WORKER_SOURCE','register_magic','cosine']:
        assert banned not in all_code
    for c in cells:
        assert not c.outputs and c.execution_count is None
        tree = ast.parse(c.source)
        assert not any(isinstance(v,ast.Call) and isinstance(v.func,ast.Name) and v.func.id in ['exec','eval'] for v in ast.walk(tree))
        assert not any(line.lstrip().startswith(('%','!')) for line in c.source.splitlines())
    assert hashlib.sha256('\n'.join(c.source.replace(fp,'__NOTEBOOK_CODE_SHA256__') for c in cells).encode()).hexdigest() == fp
    roles = {c.metadata.aes35_role for c in cells}
    assert {'model_names','input_columns','hyperparameters','dataset_execute_xy',
            'definition:SingleScoreModel','definition:AnswerBiGRUCoAttentionPool',
            'definition:fit','training_execute','download_execute'} <= roles
    before = BASE.read_bytes()
    subprocess.run([sys.executable,'scripts/build_e35_notebook.py'],cwd=ROOT,check=True,capture_output=True)
    assert NOTEBOOK.read_bytes() == raw == (ROOT/'aes-learning/notebooks/experiments'/NAME).read_bytes()
    assert BASE.read_bytes() == before


def test_core_model_tokenization_training_and_exports_match_e30_ast():
    old = nbformat.read(BASE,as_version=4); new = nbformat.read(NOTEBOOK,as_version=4)
    def definitions(n):
        return {node.name:ast.dump(node,include_attributes=False)
                for c in n.cells if c.cell_type == 'code'
                for node in ast.parse(c.source.removeprefix('%%aes\n')).body
                if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
    a,b = definitions(old),definitions(new)
    core = ['AnswerBiGRUAttention','AnswerBiGRUCoAttentionPool','SingleScoreModel','Engine',
            'seed_everything','make_model','optimizer_groups','fit','train_all','metrics',
            'gwet_ac2','make_split','contexts_from_bundle','pair_content_windows','pair_batch_windows',
            'paired_pretrained_values','paired_token_vectors','bge_features','score_answer',
            'export_final_features','save_model','load_model','save_training_progress','load_training_progress']
    for name in core:
        assert a[name] == b[name], name
    assert 'cosine' in a and 'cosine' not in b


@pytest.mark.parametrize('scenario',['S1','S2','S3'])
def test_native_forward_first_update_and_only_s3_layer_lr_changes(scenario,tmp_path,monkeypatch,encoders):
    ns = native_runtime()
    monkeypatch.setattr(context,'NOTEBOOK',BASE); old = context.runtime()
    for env in [ns,old]:
        env['encoder'] = encoders
        env['tokenizer'] = lambda *a,**k:ThreeTechnicalTokenizer()
    fixture = tmp_path/'TECHNICAL_FIXTURE_ONLY.xlsx'; make_book(fixture)
    values = dict(dataset=str(fixture),output=str(tmp_path/'run'),expected_rows=24,device='cpu',
        dropout=0.,roberta_max_length=32,roberta_overlap=4,bge_max_length=32,bge_overlap=4,
        context_max_tokens=12,question_max_tokens=4)
    cfg,control_cfg = ns['config'](values),old['config'](values)
    assert cfg['s3_layers_lr']==5e-5 and cfg['s3_mlp_lr']==cfg['projection_lr']==1e-4
    bundle = ns['read_dataset'](cfg); other = old['read_dataset'](control_cfg)
    assert bundle.rows == other.rows
    np.testing.assert_array_equal(bundle.y,other.y)
    assert ns['make_split'](bundle,cfg) == old['make_split'](other,control_cfg)
    torch.set_num_threads(2);torch.manual_seed(71);bge = torch.randn(24,1024)
    model,control = ns['make_model'](scenario,cfg).eval(),old['make_model'](scenario,control_cfg).eval()
    control.load_state_dict(model.state_dict());before = {k:v.clone() for k,v in model.state_dict().items()}
    predictions = []
    for env,m,c in [(ns,model,cfg),(old,control,control_cfg)]:
        groups = env['optimizer_groups'](m,c)
        ids = [id(p) for g in groups for p in g['params']]
        assert len(ids)==len(set(ids))==sum(p.requires_grad for p in m.parameters())
        if m is model:
            rates = {g['name']:g['lr'] for g in groups}
            assert rates['head']==1e-4
            if scenario=='S3':
                assert rates == {'roberta':5e-6,'projection':1e-4,'head':1e-4,'pooling_norm':5e-5}
        engine = env['Engine'](m,bundle.texts,bge,c,'cpu',contexts=env['contexts_from_bundle'](bundle))
        p = engine.forward([0,1]);predictions.append(p.detach())
        (p-.8).square().mean().backward();torch.optim.AdamW(groups,weight_decay=.01).step()
    torch.testing.assert_close(*predictions,rtol=0,atol=0)
    for key,value in model.state_dict().items():
        if scenario!='S3' or not key.startswith(('sequence_pool.','roberta_norm.')):
            torch.testing.assert_close(value,control.state_dict()[key],rtol=0,atol=0)
    if scenario=='S3':
        key = 'sequence_pool.reference_projection.weight'
        small = (model.state_dict()[key]-before[key]).norm()
        large = (control.state_dict()[key]-before[key]).norm()
        assert float(large/small)==pytest.approx(2.,rel=.002)
        assert model.sequence_pool.gru.weight_ih_l0.grad.abs().sum()>0
        assert model.sequence_pool.reference_projection.weight.grad.abs().sum()>0


def test_input_column_mapping_and_repeated_word_inspection_without_similarity(tmp_path):
    ns = native_runtime(); path=tmp_path/'TECHNICAL_FIXTURE_ONLY.xlsx';make_book(path)
    cfg=ns['config']({'dataset':str(path),'expected_rows':24})
    original=ns['read_dataset'](cfg)
    import openpyxl
    book=openpyxl.load_workbook(path);names={'soal':'q','jawaban_referensi':'ref','response_text':'answer','rater':'teacher'}
    for c in book.active[1]:
        if c.value in names:c.value=names[c.value]
    book.save(path);book.close()
    for key,value in [('QUESTION_COLUMN','q'),('REFERENCE_COLUMN','ref'),('ANSWER_COLUMN','answer'),('TARGET_COLUMN','teacher')]:
        ns[key]=value
    ns['INPUT_COLUMNS']=['q','ref','answer'];ns['REQUIRED']=['Email address','response_number','q','ref','answer','teacher']
    renamed=ns['read_dataset'](cfg)
    assert original.rows == renamed.rows
    np.testing.assert_array_equal(original.y,renamed.y)
    assert renamed.summary['target_columns']==['teacher']
    rows=[{'word_index':i,'word':'alpha','covered':True,'token_ids':[3]} for i in [0,1]]
    vectors=np.array([[1.,2.],[2.,3.]],dtype=np.float32)
    result=ns['compare_repetitions'](rows,vectors,vectors)
    assert len(result)==1 and result[0]['contextual_max_abs_difference']==1
    assert not any('cosine' in k for k in result[0])
    assert len(list(ns['repeated_word_rows'](result,'roberta'))[0])==len(ns['REPEATED_COLUMNS'])==8


def test_native_gpu_guard_and_setup_rejects_incompatible_loaded_core(monkeypatch):
    ns = native_runtime()
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='GPU belum aktif'):
        ns['check_gpu'](require_gpu=True)
    n=nbformat.read(NOTEBOOK,as_version=4)
    setup=next(c.source for c in n.cells if c.cell_type=='code' and c.metadata.aes35_role=='dependency_setup')
    from importlib import metadata
    original=metadata.version
    monkeypatch.setattr(metadata,'version',lambda p:'0.1' if p=='numpy' else original(p))
    with pytest.raises(RuntimeError,match='runtime ini memiliki'):
        exec(setup,{})


def test_full_native_notebook_and_fresh_kernel_resume(tmp_path):
    """Actual native notebook execution; random Transformer fixture, not GPU research."""
    n=nbformat.read(NOTEBOOK,as_version=4)
    isolated=tmp_path/'standalone';isolated.mkdir();fixture=isolated/'TECHNICAL_FIXTURE_ONLY.xlsx';make_book(fixture)
    workspace=isolated/'workspace'
    guard="""import builtins
_original_import=builtins.__import__
def _no_project_import(name,*args,**kwargs):
    if name=='aes' or name.startswith('aes.'):raise ImportError('Standalone native fixture forbids project imports')
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
        role=c.metadata.aes35_role
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
    n.cells.insert(0,nbformat.v4.new_code_cell(guard))
    n.cells.append(nbformat.v4.new_code_cell("""
assert list(X.columns)==INPUT_COLUMNS and X.shape==(24,3) and y.shape==(24,)
assert TARGET_COLUMN not in X.columns
assert X[ANSWER_COLUMN].tolist()==bundle.texts
assert list(X[[QUESTION_COLUMN,REFERENCE_COLUMN]].itertuples(index=False,name=None))==contexts_from_bundle(bundle)
assert np.array_equal(y.to_numpy(),bundle.y)
assert fit.__globals__ is globals() and make_model.__globals__ is globals()
"""))
    path=isolated/'TECHNICAL_executed.ipynb';nbformat.write(n,path)
    kernels=tmp_path/'kernels';spec=kernels/'aes35_fixture';spec.mkdir(parents=True)
    (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-I','-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'AES35 fixture','language':'python'}))
    env={**os.environ,'AES300_WORKSPACE':str(workspace),'AES300_DATASET':str(fixture),
         'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
         'IPYTHONDIR':str(tmp_path/'ipython')}
    env.pop('PYTHONPATH',None)
    for attempt in [1,2]:
        current=nbformat.read(path,as_version=4)
        manager=KernelManager(kernel_name='aes35_fixture',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
        try:
            NotebookClient(current,timeout=360,km=manager,resources={'metadata':{'path':str(isolated)}}).execute(env=env)
        finally:
            if manager.has_kernel:manager.shutdown_kernel(now=True)
        nbformat.write(current,isolated/f'TECHNICAL_attempt_{attempt}.ipynb')
        info=json.loads((workspace/'outputs/dataset300_download.json').read_text());run=Path(info['run_folder'])
        report=json.loads((run/'final_report.json').read_text())
        assert info['excel_count']==5 and len(report['cv'])==15 and report['test_evaluations_per_scenario']==1
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in run.glob('S*/final.pt')}
        if attempt==1:first_hashes=hashes
        else:assert hashes==first_hashes
    import pandas as pd
    config=json.loads((run/'effective_config.json').read_text())
    assert config['s3_mlp_lr']==config['projection_lr']==1e-4 and config['s3_layers_lr']==5e-5
    settings=pd.read_excel(run/'hasil_ringkas/05_Riwayat_Pelatihan.xlsx',sheet_name='Pengaturan Eksperimen')
    settings=dict(zip(settings.Pengaturan,settings.Nilai))
    assert float(settings['LR MLP S3'])==1e-4 and float(settings['LR pooling dan norm S3'])==5e-5
    headers=pd.read_excel(run/'hasil_ringkas/03_Vektor_Kata_dan_Kalimat.xlsx',sheet_name='Kata Berulang',nrows=0).columns
    assert len(headers)==8 and not any('cosine' in h.lower() for h in headers)
    for s in ['S1','S2','S3']:
        checkpoint=torch.load(run/s/'final.pt',map_location='cpu',weights_only=True)
        rates=checkpoint['training']['optimizer_learning_rates'];assert rates['head']==1e-4
        if s=='S3':assert rates['pooling_norm']==5e-5 and rates['projection']==1e-4
    with ZipFile(info['archive']) as z:assert len(z.namelist())==5
    assert not list(workspace.rglob('*.py'))
    proof={'scope':'TECHNICAL RANDOM TRANSFORMER FIXTURE ONLY','native_kernel':True,
           'cv_fits':15,'refits':3,'reports':5,'fresh_kernel_resume_unchanged_checkpoints':True,
           'visible_X_y_same_as_training_inputs':True,'code_cells':sum(c.cell_type=='code' for c in n.cells)-2,
           'execution_device':'cpu','published_runtime':'cuda','cosine_removed':True}
    (isolated/'proof.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
