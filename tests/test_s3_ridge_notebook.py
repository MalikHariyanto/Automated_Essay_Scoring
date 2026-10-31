"""S3 Ridge technical fixtures: no real student data or pretrained accuracy claims."""
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import zipfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/'notebooks/experiments/AES_300_S3_Fusion_Ridge_Tiga_Input_Colab.ipynb'
sys.path.insert(0,str(ROOT/'tests'))
from test_teacher_score import TechnicalTokenizer,make_book,encoders
from test_context_pair_notebook import PairTechnicalTokenizer


def runtime(tmp_path):
    n=nbformat.read(NOTEBOOK,as_version=4);m=types.ModuleType('s3_ridge_fixture');sys.modules[m.__name__]=m
    ns=m.__dict__;ns['ROOT']=tmp_path
    for i in (7,10,11,14,17,18,21,22,23,25,26,27,28,29,30):
        exec(compile(n.cells[i].source.removeprefix('%%aes\n'),'<s3-standalone>','exec'),ns)
    ns['show']=lambda *a,**k:None
    return ns


def fixture(tmp_path):
    ns=runtime(tmp_path);path=tmp_path/'TECHNICAL_FIXTURE_ONLY.xlsx';make_book(path)
    cfg=dict(dataset=str(path),output=str(tmp_path/'results'),sheet='Sheet1',expected_rows=24,score_min=0,score_max=100,
        folds=5,test_fraction=.2,split_seed=2026,split_candidates=128,seed=42,epochs=1,micro_batch=1,
        effective_batch=8,dropout=.1,early_stopping_patience=5,representation_policy='E21_fusion_fixed_epochs_train_only',
        s3_ridge_alphas=[.01,1.],s2_ridge_alpha=.01,device='cpu',code_sha256=ns['NOTEBOOK_CODE_SHA256'],
        roberta={'checkpoint':'TECHNICAL_RANDOM_ONLY','revision':'fixture'},bge={'checkpoint':'TECHNICAL_RANDOM_ONLY','revision':'fixture'})
    bundle=ns['read_dataset'](cfg);manifest=ns['make_split'](bundle,cfg)
    rng=np.random.default_rng(71);X=rng.normal(size=(24,1024)).astype(np.float32);X/=np.linalg.norm(X,axis=1,keepdims=True)
    return ns,cfg,bundle,manifest,X


def fake_training(ns,calls):
    """Label-dependent mock representation exposes any forbidden target reaching training."""
    class Engine:
        def __init__(self,X,mean):self.X=X;self.mean=mean
        def forward(self,ids,return_features=False):
            z=self.X[ids]*(1+self.mean/100.)
            p=torch.full((len(ids),1),self.mean/100.,dtype=torch.float32)
            return p,dict(roberta=z[:,:768],roberta_layernorm=z[:,:768],projection=z[:,768:],concat=z)
    def fit(scenario,training,validation,bundle,bge,cfg,epochs,seed,checkpoint=None):
        assert scenario=='S3' and validation is None
        unused=sorted(set(range(len(bundle.y)))-set(training))
        assert np.isnan(bundle.y[unused]).all()
        assert np.isfinite(bundle.y[training]).all()
        calls.append(tuple(training));torch.manual_seed(seed)
        model=torch.nn.Linear(1,1)
        info=dict(history=[dict(epoch=1,train_mse_scaled=.01)],parameter_samples_changed={'technical_fixture':True},
                  optimizer_learning_rates={'technical_fixture':1e-4})
        return model,Engine(bge,float(bundle.y[training].mean())),info
    ns['fit']=fit


def test_structure_identity_builder_and_existing_notebooks_unchanged():
    raw=NOTEBOOK.read_bytes();n=nbformat.reads(raw.decode(),as_version=4);nbformat.validate(n)
    assert raw==(ROOT/'aes-learning/notebooks/experiments'/NOTEBOOK.name).read_bytes()
    identity=n.metadata.aes_standalone.code_sha256
    s='\n'.join(c.source.removeprefix('%%aes\n').replace(identity,'__S3_RIDGE_CODE_HASH__') for c in n.cells if c.cell_type=='code')
    assert hashlib.sha256(s.encode()).hexdigest()==identity
    assert all(not c.outputs and c.execution_count is None for c in n.cells if c.cell_type=='code')
    assert n.metadata.aes_experiment.supervised_representation_refit_inside_inner_cv
    assert not n.metadata.aes_experiment.research_results_included
    assert 'from aes.' not in s and 'import s3_ridge_experiment' not in s
    old={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'notebooks/experiments').glob('*.ipynb') if p!=NOTEBOOK}
    subprocess.run([sys.executable,'scripts/build_s3_ridge_notebook.py'],cwd=ROOT,check=True,capture_output=True)
    assert raw==NOTEBOOK.read_bytes()
    assert old=={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in old}


def test_supervised_inner_folds_never_see_outer_labels_and_pair_cache_is_shared(tmp_path):
    ns,cfg,bundle,m,X=fixture(tmp_path);calls=[];fake_training(ns,calls)
    result=ns['run_s3_ridge'](bundle,X,m,cfg,tmp_path/'base')
    assert len(calls)==16 and result['protocol']['unique_representation_fits']==16
    assert len({tuple(sorted(v)) for v in calls})==16
    for stage in result['representation_stages']:
        tr=set(stage['proof']['contract']['training'])
        assert tr.isdisjoint(m['test'])
        if stage['kind']=='INNER_PAIR':
            a,b=map(int,stage['fold'].split('-'))
            excluded=set(m['folds'][a-1]['validation'])|set(m['folds'][b-1]['validation'])
            assert tr==set(m['development'])-excluded
    outer=m['folds'][0];changed=copy.deepcopy(bundle);changed.y[outer['validation']]=100-changed.y[outer['validation']]
    other=ns['run_s3_ridge'](changed,X,m,cfg,tmp_path/'outer_changed')
    assert result['outer_alpha'][0]==other['outer_alpha'][0]
    selection=lambda r:[(v['row_id'],v['prediction']) for v in r['predictions'] if v['model']=='S3_RIDGE' and v['partition']=='DEVELOPMENT_OOF' and v['fold']==1]
    assert selection(result)==selection(other)
    # Test labels cannot affect any alpha, representation or final coefficients.
    changed=copy.deepcopy(bundle);changed.y[m['test']]=100-changed.y[m['test']]
    third=ns['run_s3_ridge'](changed,X,m,cfg,tmp_path/'test_changed')
    assert result['alpha_search']==third['alpha_search']
    assert result['final_contract']['s3_ridge_sha256']==third['final_contract']['s3_ridge_sha256']
    assert result['final_contract']['representation_checkpoint_sha256']==third['final_contract']['representation_checkpoint_sha256']


def test_completed_cache_resume_export_and_raw_feature_model_reload(tmp_path):
    ns,cfg,bundle,m,X=fixture(tmp_path);calls=[];fake_training(ns,calls)
    result=ns['run_s3_ridge'](bundle,X,m,cfg,tmp_path/'run')
    def forbidden(*a,**k):raise AssertionError('Complete representation cache must not train again')
    ns['fit']=forbidden
    again=ns['run_s3_ridge'](bundle,X,m,cfg,tmp_path/'run')
    assert result['predictions']==again['predictions']
    for model in ns['S3_MODELS']:
        for part,indices in [('DEVELOPMENT_OOF',m['development']),('TEST_HISTORICAL_HOLDOUT',m['test'])]:
            rows=[r for r in result['predictions'] if r['model']==model and r['partition']==part]
            assert len(rows)==len(indices) and {r['row_id'] for r in rows}=={bundle.rows[i]['row_id'] for i in indices}
    with np.load(result['final_features'],allow_pickle=False) as f:Z=f['concat'];ids=f['indices']
    assert np.linalg.norm(Z,axis=1).mean()!=pytest.approx(1.) # No hidden L2 on fusion features.
    with np.load(tmp_path/'run/s3_ridge_final.npz',allow_pickle=False) as f:p=Z.astype(float)@f['coef']+f['intercept']
    lookup={int(i):float(v) for i,v in zip(ids,p)}
    rows=[r for r in result['predictions'] if r['model']=='S3_RIDGE' and r['partition']=='TEST_HISTORICAL_HOLDOUT']
    np.testing.assert_allclose([r['prediction'] for r in rows],[lookup[i] for i in m['test']],atol=1e-10)
    info=ns['export_s3_ridge'](bundle,X,m,cfg,result)
    with zipfile.ZipFile(info['archive']) as z:
        meta=json.loads(z.read('manifest.json'));assert len([v for v in z.namelist() if v.endswith('.xlsx')])==5
        assert not meta['full_representation_checkpoint_included']
        assert meta['test_evaluated'] and not meta['test_used_for_selection']
        assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in meta['files'].items())
    duels=ns['s3_duels'](result);assert len(duels)==6
    assert all(v['left_closer']+v['right_closer']+v['ties']==v['n'] for v in duels)
    cache=Path(result['representation_stages'][0]['folder'])/'features.npz';cache.write_bytes(b'corrupt technical cache')
    with pytest.raises(ValueError,match='Cache representasi'):
        ns['run_s3_ridge'](bundle,X,m,cfg,tmp_path/'run')


def test_all_cells_with_random_transformers_and_training_resume(tmp_path):
    n=nbformat.read(NOTEBOOK,as_version=4);original_cells=sum(c.cell_type=='code' for c in n.cells)
    standalone=tmp_path/'standalone';standalone.mkdir();dataset=standalone/'TECHNICAL_FIXTURE_ONLY.xlsx';make_book(dataset)
    tiny='''
from transformers import XLMRobertaConfig, XLMRobertaModel
class ThreeTechnicalTokenizer(PairTechnicalTokenizer):
    padding_side='right'
def tokenizer(*args,**kwargs):return ThreeTechnicalTokenizer()
def encoder(cfg,role):
    dim,heads=(768,12) if role=='roberta' else (1024,16)
    with torch.random.fork_rng():
        torch.manual_seed(170)
        result=XLMRobertaModel(XLMRobertaConfig(vocab_size=64,hidden_size=dim,num_hidden_layers=1,
            num_attention_heads=heads,intermediate_size=16,max_position_embeddings=64,pad_token_id=1,
            bos_token_id=0,eos_token_id=2,hidden_dropout_prob=0.,attention_probs_dropout_prob=0.),add_pooling_layer=False)
    result.requires_grad_(role=='roberta')
    return result
'''
    for c in n.cells:
        if c.cell_type!='code':continue
        if 'EPOCHS, FOLDS = 25, 5' in c.source:
            c.source=c.source.replace('EPOCHS, FOLDS = 25, 5','EPOCHS, FOLDS = 1, 5').replace('EXPECTED_ROWS = 300','EXPECTED_ROWS = 24').replace("DEVICE = 'cuda'","DEVICE = 'cpu'")
            c.source=c.source.replace('roberta_max_length=512, roberta_overlap=128, bge_max_length=1024, bge_overlap=128','roberta_max_length=32, roberta_overlap=4, bge_max_length=32, bge_overlap=4').replace('context_max_tokens=192, question_max_tokens=64','context_max_tokens=12, question_max_tokens=4')
            c.source=c.source.replace('S3_RIDGE_ALPHAS = [.01, .1, 1., 10., 100.]','S3_RIDGE_ALPHAS = [.01, 1.]')
        if 'def encoder(cfg, role):' in c.source:
            c.source+='\n'+inspect.getsource(TechnicalTokenizer)+'\n'+inspect.getsource(PairTechnicalTokenizer)+tiny
    n.cells.append(nbformat.v4.new_code_cell('''%%aes
original_fit=fit
def reject_training(*args,**kwargs):raise AssertionError('Completed stages must resume without training')
fit=reject_training
repeated=run_s3_ridge(bundle,X,manifest,cfg,output)
assert repeated['predictions']==result['predictions']
fit=original_fit
'''))
    kernels=tmp_path/'kernels';spec=kernels/'s3_fixture';spec.mkdir(parents=True)
    (spec/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable,'-I','-m','ipykernel_launcher','-f','{connection_file}'],display_name='S3 fixture',language='python')))
    workspace=standalone/'workspace'
    env=dict(os.environ,AES300_WORKSPACE=str(workspace),AES300_DATASET=str(dataset),AES300_PYTHON=sys.executable,
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',IPYTHONDIR=str(tmp_path/'ipython'))
    env.pop('PYTHONPATH',None);env.pop('AES300_AUDIT_ONLY',None)
    manager=KernelManager(kernel_name='s3_fixture',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
    try:
        NotebookClient(n,timeout=420,km=manager,resources={'metadata':{'path':str(standalone)}}).execute(env=env)
    finally:
        if manager.has_kernel:manager.shutdown_kernel(now=True)
    pointer=json.loads((workspace/'outputs/s3_ridge_download.json').read_text());run=Path(pointer['run_folder'])
    r=json.loads((run/'s3_ridge_result.json').read_text());assert len(r['representation_stages'])==16
    assert all(v['proof']['info']['parameter_samples_changed']['roberta'] for v in r['representation_stages'])
    assert all(not v['proof']['contract']['validation_used_for_training'] for v in r['representation_stages'])
    assert not list(run.glob('representations/*/epoch_progress*'))
    assert Path(r['final_representation_checkpoint']).is_file()
    saved=torch.load(r['final_representation_checkpoint'],map_location='cpu',weights_only=True)
    assert saved['contract']['training']==sorted(json.loads((run/'split_manifest.json').read_text())['development'])
    head=torch.nn.Sequential(torch.nn.Linear(1024,256),torch.nn.GELU(),torch.nn.Dropout(.1),torch.nn.Linear(256,1)).eval()
    head.load_state_dict({k.removeprefix('head.'):v for k,v in saved['state_dict'].items() if k.startswith('head.')})
    with np.load(r['final_features'],allow_pickle=False) as f:
        with torch.no_grad():control=head(torch.from_numpy(f['concat'])).reshape(-1).numpy()*100.
        np.testing.assert_allclose(control,f['mlp_prediction'],atol=1e-4,rtol=1e-5)
    assert len(r['predictions'])==24*3
    assert all(c.execution_count is not None for c in n.cells if c.cell_type=='code')
    evidence=ROOT/'outputs/s3_ridge_verification';evidence.mkdir(exist_ok=True,parents=True)
    nbformat.write(n,evidence/'TECHNICAL_executed.ipynb')
    (evidence/'proof.json').write_text(json.dumps(dict(scope='SYNTHETIC24 + RANDOM TRANSFORMERS; NO PRETRAINED AES ACCURACY',
        original_code_cells=original_cells,representation_fits=16,resume_completed_stages=True,
        encoder_gradients_observed=True,research_training=False,
        code_sha256=nbformat.read(NOTEBOOK,as_version=4).metadata.aes_standalone.code_sha256),indent=2))


def test_interrupted_s3_epoch_resumes_exactly_with_random_transformer(tmp_path,encoders):
    ns,cfg,bundle,m,X=fixture(tmp_path)
    cfg.update(epochs=2,s3_encoder_lr=5e-6,encoder_lr=5e-6,projection_lr=1e-4,head_lr=1e-4,
               weight_decay=.01,warmup_ratio=.1,gradient_clip=1.,roberta_max_length=32,roberta_overlap=4,
               context_max_tokens=12,question_max_tokens=4,input_format='single_sequence')
    class Three(PairTechnicalTokenizer):padding_side='right'
    ns['encoder']=encoders;ns['tokenizer']=lambda *a,**k:Three()
    torch.set_num_threads(2);tr=m['folds'][0]['train'];va=m['folds'][0]['validation'];bge=torch.from_numpy(X)
    expected,engine,info=ns['fit']('S3',tr,None,bundle,bge,cfg,2,42,checkpoint=tmp_path/'whole.json')
    p=engine.predict(va)
    original=ns['save_training_progress'];calls=[]
    def interrupted(path,payload):
        original(path,payload);calls.append(payload['epoch'])
        if len(calls)==1:raise InterruptedError('Technical interruption after an intact epoch checkpoint')
    ns['save_training_progress']=interrupted
    with pytest.raises(InterruptedError):
        ns['fit']('S3',tr,None,bundle,bge,cfg,2,42,checkpoint=tmp_path/'resume.json')
    ns['save_training_progress']=original
    actual,engine2,info2=ns['fit']('S3',tr,None,bundle,bge,cfg,2,42,checkpoint=tmp_path/'resume.json')
    assert info['history']==info2['history']
    np.testing.assert_array_equal(p,engine2.predict(va))
    for k,v in expected.state_dict().items():torch.testing.assert_close(v,actual.state_dict()[k],atol=0,rtol=0)
