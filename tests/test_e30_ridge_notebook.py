"""E30 Ridge: standalone architecture and full technical notebook execution."""
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import nbformat
import inspect, os, types, zipfile
import numpy as np
import pytest
import torch
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from test_teacher_score import TechnicalTokenizer, make_book
from test_context_pair_notebook import PairTechnicalTokenizer
from test_teacher_score import encoders
from test_e6_three_input_notebook import ThreeTechnicalTokenizer

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT/'notebooks/experiments/AES_300_E30_BiGRU_CoAttention_Ridge_Colab.ipynb'


def test_e30_source_optimizer_and_reproducible_build():
    raw = NOTEBOOK.read_bytes()
    n = nbformat.reads(raw.decode(), as_version=4)
    nbformat.validate(n)
    identity = n.metadata.aes_standalone.code_sha256
    codes = [c.source.removeprefix('%%aes\n') for c in n.cells if c.cell_type == 'code']
    for s in codes:
        ast.parse(s)
    assert hashlib.sha256('\n'.join(codes).replace(identity, '__S3_RIDGE_CODE_HASH__').encode()).hexdigest() == identity
    original = json.loads(subprocess.check_output(['git', 'show',
        '47ddb8280592231fbb682f2e6a8b3c5735d185fb:notebooks/experiments/AES_300_E30_E26_BiGRU_CoAttention_Pooling_GPU.ipynb'], cwd=ROOT))
    from build_bge_readout_notebook import definitions
    architecture = definitions(''.join(original['cells'][30]['source']).removeprefix('%%aes\n'))
    assert architecture in '\n'.join(codes)
    optimizer = definitions(''.join(original['cells'][36]['source']).removeprefix('%%aes\n'), {'optimizer_groups'})
    assert optimizer in '\n'.join(codes)
    assert 's3_mlp_lr=1e-4' in '\n'.join(codes)
    assert 's3_layers_lr=1e-4' in '\n'.join(codes)
    assert all(not c.outputs and c.execution_count is None for c in n.cells if c.cell_type == 'code')
    subprocess.run([sys.executable, 'scripts/build_e30_ridge_notebook.py'], cwd=ROOT, check=True)
    assert raw == NOTEBOOK.read_bytes()
    assert raw == (ROOT/'aes-learning/notebooks/experiments'/NOTEBOOK.name).read_bytes()


def test_full_e30_notebook_training_exports_and_resume(tmp_path):
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
            c.source=c.source
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
    r=json.loads((run/'s3_ridge_result.json').read_text());assert len(r['representation_stages'])==6
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
    assert r['final_contract']['final_epochs']==1
    assert all(s['proof']['contract']['validation_used_for_epoch_selection'] for s in r['representation_stages'][:5])
    assert not r['representation_stages'][-1]['proof']['contract']['validation_used_for_epoch_selection']
    assert all(s['proof']['info']['optimizer_learning_rates']['head']==1e-4 for s in r['representation_stages'])
    assert all(s['proof']['info']['parameter_samples_changed']['sequence_pool'] for s in r['representation_stages'])
    assert all(c.execution_count is not None for c in n.cells if c.cell_type=='code')
    evidence=ROOT/'outputs/e30_ridge_verification';evidence.mkdir(exist_ok=True,parents=True)
    nbformat.write(n,evidence/'TECHNICAL_executed.ipynb')
    (evidence/'proof.json').write_text(json.dumps(dict(scope='SYNTHETIC24 + RANDOM TRANSFORMERS; NO PRETRAINED AES ACCURACY',
        original_code_cells=original_cells,representation_fits=6,resume_completed_stages=True,
        encoder_gradients_observed=True,research_training=False,
        code_sha256=nbformat.read(NOTEBOOK,as_version=4).metadata.aes_standalone.code_sha256),indent=2))


def fixture_runtime(tmp_path):
    module=types.ModuleType('e30_ridge_fixture');sys.modules[module.__name__]=module
    ns=module.__dict__;ns['ROOT']=tmp_path
    n=nbformat.read(NOTEBOOK,as_version=4)
    for i in (7,10,11,14,17,18,21,22,23,25,26,27,28,29,30):
        exec(n.cells[i].source.removeprefix('%%aes\n'),ns)
    path=tmp_path/'TECHNICAL_FIXTURE_ONLY.xlsx';make_book(path)
    cfg=dict(dataset=str(path),output=str(tmp_path/'run'),sheet='Sheet1',expected_rows=24,
        score_min=0,score_max=100,folds=5,test_fraction=.2,split_seed=2026,split_candidates=128,
        seed=42,epochs=8,early_stopping_patience=5,effective_batch=8,micro_batch=1,dropout=0.,
        s3_encoder_lr=5e-6,encoder_lr=5e-6,projection_lr=1e-4,head_lr=1e-4,
        s2_mlp_lr=1e-4,s3_mlp_lr=1e-4,s3_layers_lr=1e-4,s3_sequence_hidden=64,
        s3_pooling='bigru_coattention_pooling',s3_fusion='concat',weight_decay=.01,warmup_ratio=.1,
        gradient_clip=1.,roberta_max_length=32,roberta_overlap=4,bge_max_length=32,bge_overlap=4,
        bge_batch=4,device='cpu',context_max_tokens=12,question_max_tokens=4,input_format='single_sequence',
        s3_ridge_alphas=[1.],s2_ridge_alpha=.01,representation_policy='E30_five_fold_early_stopping',
        roberta={'checkpoint':'TECHNICAL_RANDOM_ONLY','revision':'fixture'},bge={'checkpoint':'TECHNICAL_RANDOM_ONLY','revision':'fixture'})
    bundle=ns['read_dataset'](cfg);manifest=ns['make_split'](bundle,cfg)
    return ns,cfg,bundle,manifest


def test_patience_five_restores_best_epoch_and_resumes(tmp_path,encoders):
    ns,cfg,bundle,manifest=fixture_runtime(tmp_path)
    ns['encoder']=encoders;ns['tokenizer']=lambda *a,**k:ThreeTechnicalTokenizer()
    ns['metrics']=lambda *a,**k:dict(RMSE=10.)  # Deliberate plateau, technical fixture only.
    torch.set_num_threads(2);torch.manual_seed(19)
    X=torch.randn(24,1024);f=manifest['folds'][0]
    path=tmp_path/'progress.json'
    model,engine,info=ns['fit']('S3',f['train'],f['validation'],bundle,X,cfg,8,42,checkpoint=path)
    assert info['early_stopped'] and info['epochs_completed']==6 and info['best_epoch']==1
    assert info['history'][-1]['epochs_without_improvement']==5
    snap=json.loads(path.read_text())['snapshots'][0]['file']
    payload=torch.load(path.parent/snap,map_location='cpu',weights_only=True)
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,payload['best_state'][key],rtol=0,atol=0)
    selected=engine.predict(f['validation'])
    resumed,engine2,again=ns['fit']('S3',f['train'],f['validation'],bundle,X,cfg,8,42,checkpoint=path)
    np.testing.assert_array_equal(selected,engine2.predict(f['validation']))
    assert info['history']==again['history']
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,resumed.state_dict()[key],rtol=0,atol=0)


def test_five_cv_fits_median_refit_and_test_label_isolation(tmp_path):
    ns,cfg,bundle,manifest=fixture_runtime(tmp_path)
    ns['show']=lambda *a,**k:None
    calls=[];best=[1,4,5,2,7]
    class Engine:
        def __init__(self,X,mean):self.X=X;self.mean=mean
        def forward(self,ids,return_features=False):
            z=self.X[ids]*(1+self.mean/100.)
            p=torch.full((len(ids),1),self.mean/100.)
            return p,dict(roberta=z[:,:768],roberta_layernorm=z[:,:768],projection=z[:,768:],concat=z)
    def fit(scenario,tr,va,data,X,config,epochs,seed,checkpoint=None):
        assert np.isnan(data.y[manifest['test']]).all()
        assert set(tr).isdisjoint(va or [])
        calls.append((list(tr),va,epochs))
        if va is None:
            assert sorted(tr)==sorted(manifest['development']) and epochs==4
            epoch=epochs
        else:
            f=next(f for f in manifest['folds'] if f['validation']==va)
            assert tr==f['train']
            epoch=best[f['fold']-1]
        torch.manual_seed(seed)
        info=dict(best_epoch=epoch,history=[dict(epoch=epoch)],parameter_samples_changed={})
        return torch.nn.Linear(1,1),Engine(X,float(data.y[tr].mean())),info
    ns['fit']=fit
    rng=np.random.default_rng(33);X=rng.normal(size=(24,1024)).astype(np.float32);X/=np.linalg.norm(X,axis=1,keepdims=True)
    r=ns['run_s3_ridge'](bundle,X,manifest,cfg,tmp_path/'first')
    assert len(calls)==6 and r['final_contract']['final_epochs']==4
    assert not r['inner_validation'] and r['protocol']['unique_representation_fits']==6
    calls.clear()
    ns['run_s3_ridge'](bundle,X,manifest,cfg,tmp_path/'first')
    assert not calls
    import copy
    changed=copy.deepcopy(bundle);changed.y[manifest['test']]=100-changed.y[manifest['test']]
    other=ns['run_s3_ridge'](changed,X,manifest,cfg,tmp_path/'changed_test')
    assert other['final_contract']['final_epochs']==4
    assert r['final_contract']['s3_ridge_sha256']==other['final_contract']['s3_ridge_sha256']
    assert r['final_contract']['representation_checkpoint_sha256']==other['final_contract']['representation_checkpoint_sha256']
    first=r['representation_stages'][0]
    stage=Path(first['folder'])/'features.npz';stage.write_bytes(b'TECHNICAL_CORRUPT_CACHE')
    with pytest.raises(ValueError,match='Cache representasi'):
        ns['run_s3_ridge'](bundle,X,manifest,cfg,tmp_path/'first')
