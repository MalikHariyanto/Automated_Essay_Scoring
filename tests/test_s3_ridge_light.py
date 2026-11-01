"""Budget and backup tests use synthetic fixtures, never research accuracy."""
import hashlib,inspect,json,os,subprocess,sys,zipfile
from pathlib import Path
import nbformat,numpy as np,pytest
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
import test_s3_ridge_notebook as old
ROOT=old.ROOT
NOTEBOOK=ROOT/'notebooks/experiments/AES_300_S3_Fusion_Ridge_Hemat_Colab.ipynb'

def setup(tmp_path,monkeypatch):
 monkeypatch.setattr(old,'NOTEBOOK',NOTEBOOK)
 ns,cfg,b,m,X=old.fixture(tmp_path)
 cfg.update(s3_fixed_alpha=1.,s3_ridge_alphas=[1.])
 calls=[];old.fake_training(ns,calls)
 out=tmp_path/'run';ns['initialize_s3_light'](b,m,cfg,out);np.save(out/'bge_frozen.npy',X)
 return ns,cfg,b,m,X,out,calls

def test_budget_and_portable_restore(tmp_path,monkeypatch):
 ns,cfg,b,m,X,out,calls=setup(tmp_path,monkeypatch)
 folds=[ns['run_s3_light_fold'](i,b,X,m,cfg,out) for i in range(1,6)]
 backup=out/'backups/Backup_S3_Ridge_Hemat_Fold_5.zip'
 target=tmp_path/'new_runtime';target.mkdir()
 # Completed features may move to another hardware setting; no model training in this check.
 changed=dict(cfg,device='auto',output=str(target))
 ns['initialize_s3_light'](b,m,changed,target)
 ns['restore_light_backup'](backup,b,m,changed,target)
 ns['device_name']=lambda cfg:'cpu'
 before=len(calls)
 restored=[ns['run_s3_light_fold'](i,b,X,m,changed,target) for i in range(1,6)]
 assert len(calls)==before==5
 assert [f['predictions'] for f in folds]==[f['predictions'] for f in restored]
 r=ns['finalize_s3_light'](b,X,m,changed,target,restored)
 assert len(calls)==6 and r['protocol']['unique_representation_fits']==6
 assert not r.get('alpha_search') and r['final_contract']['selection_partition']=='none_fixed_before_run'
 info=ns['export_s3_ridge'](b,X,m,changed,r)
 with zipfile.ZipFile(info['archive']) as z:
  meta=json.loads(z.read('manifest.json'));assert meta['fixed_alpha']==1.
  assert len([n for n in z.namelist() if n.endswith('.xlsx')])==5
  assert all(hashlib.sha256(z.read(n)).hexdigest()==v for n,v in meta['files'].items())
 with pytest.raises(ValueError,match='resep'):
  ns['restore_light_backup'](backup,b,m,dict(cfg,epochs=9),tmp_path/'bad')

def test_identity_and_old_source_unchanged():
 n=nbformat.read(NOTEBOOK,as_version=4);nbformat.validate(n);identity=n.metadata.aes_standalone.code_sha256
 s='\n'.join(c.source.removeprefix('%%aes\n').replace(identity,'__LIGHT_HASH__') for c in n.cells if c.cell_type=='code')
 assert hashlib.sha256(s.encode()).hexdigest()==identity
 assert all(not c.outputs and c.execution_count is None for c in n.cells if c.cell_type=='code')
 assert 'def choose_s3_alpha' not in s and 'def run_s3_ridge(' not in s
 assert NOTEBOOK.read_bytes()==(ROOT/'aes-learning/notebooks/experiments'/NOTEBOOK.name).read_bytes()
 original=ROOT/'notebooks/experiments/AES_300_S3_Fusion_Ridge_Tiga_Input_Colab.ipynb';before=original.read_bytes()
 subprocess.run([sys.executable,'scripts/build_s3_ridge_light_notebook.py'],cwd=ROOT,check=True,capture_output=True)
 assert original.read_bytes()==before

def test_all_cells(tmp_path):
 n=nbformat.read(NOTEBOOK,as_version=4);folder=tmp_path/'alone';folder.mkdir();dataset=folder/'TECHNICAL.xlsx';old.make_book(dataset)
 tiny='''
from transformers import XLMRobertaConfig,XLMRobertaModel
class ThreeTechnicalTokenizer(PairTechnicalTokenizer):padding_side='right'
def tokenizer(*a,**k):return ThreeTechnicalTokenizer()
def encoder(cfg,role):
    dim,heads=(768,12) if role=='roberta' else (1024,16)
    with torch.random.fork_rng():
        torch.manual_seed(170)
        model=XLMRobertaModel(XLMRobertaConfig(vocab_size=64,hidden_size=dim,num_hidden_layers=1,num_attention_heads=heads,intermediate_size=16,max_position_embeddings=64,pad_token_id=1,bos_token_id=0,eos_token_id=2,hidden_dropout_prob=0.,attention_probs_dropout_prob=0.),add_pooling_layer=False)
    model.requires_grad_(role=='roberta');return model
'''
 for c in n.cells:
  if c.cell_type!='code':continue
  if 'EPOCHS, FOLDS = 5, 5' in c.source:
   c.source=c.source.replace('EPOCHS, FOLDS = 5, 5','EPOCHS, FOLDS = 1, 5').replace('EXPECTED_ROWS = 300','EXPECTED_ROWS = 24').replace("DEVICE = 'cuda'","DEVICE = 'cpu'").replace('roberta_max_length=512, roberta_overlap=128, bge_max_length=1024, bge_overlap=128','roberta_max_length=32, roberta_overlap=4, bge_max_length=32, bge_overlap=4').replace('context_max_tokens=192, question_max_tokens=64','context_max_tokens=12, question_max_tokens=4')
  if 'def encoder(cfg, role):' in c.source:c.source+='\n'+inspect.getsource(old.TechnicalTokenizer)+'\n'+inspect.getsource(old.PairTechnicalTokenizer)+tiny
 kernels=tmp_path/'kernels';spec=kernels/'light';spec.mkdir(parents=True)
 (spec/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable,'-I','-m','ipykernel_launcher','-f','{connection_file}'],display_name='Light fixture',language='python')))
 env=dict(os.environ,AES300_WORKSPACE=str(folder/'workspace'),AES300_DATASET=str(dataset),AES300_PYTHON=sys.executable,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',IPYTHONDIR=str(tmp_path/'ipy'))
 env.pop('PYTHONPATH',None)
 km=KernelManager(kernel_name='light',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
 try:NotebookClient(n,timeout=240,km=km,resources={'metadata':{'path':str(folder)}}).execute(env=env)
 finally:
  if km.has_kernel:km.shutdown_kernel(now=True)
 pointer=json.loads((folder/'workspace/outputs/s3_light_download.json').read_text());run=Path(pointer['run_folder'])
 r=json.loads((run/'s3_ridge_result.json').read_text());assert len(r['representation_stages'])==6
 assert len(list((run/'backups').glob('*.zip')))==5
 assert all(c.execution_count is not None for c in n.cells if c.cell_type=='code')
 evidence=ROOT/'outputs/s3_light_verification';evidence.mkdir(parents=True,exist_ok=True)
 nbformat.write(n,evidence/'TECHNICAL_executed.ipynb')
 (evidence/'proof.json').write_text(json.dumps(dict(scope='Synthetic24 + random transformer; no pretrained research results',code_sha256=nbformat.read(NOTEBOOK,as_version=4).metadata.aes_standalone.code_sha256,representation_fits=6,backup_archives=5,code_cells=sum(c.cell_type=='code' for c in n.cells)),indent=2))

def test_held_labels_and_tampered_backup(tmp_path,monkeypatch):
 import copy
 ns,cfg,b,m,X,out,calls=setup(tmp_path,monkeypatch)
 first=ns['run_s3_light_fold'](1,b,X,m,cfg,out)
 changed=copy.deepcopy(b)
 changed.y[m['folds'][0]['validation']]=100-changed.y[m['folds'][0]['validation']]
 changed.y[m['test']]=100-changed.y[m['test']]
 second=ns['run_s3_light_fold'](1,changed,X,m,cfg,out)
 assert len(calls)==1
 assert [r['prediction'] for r in first['predictions']]==[r['prediction'] for r in second['predictions']]
 original=out/'backups/Backup_S3_Ridge_Hemat_Fold_1.zip';bad=tmp_path/'bad.zip'
 with zipfile.ZipFile(original) as z,zipfile.ZipFile(bad,'w') as w:
  for name in z.namelist():w.writestr(name,b'changed' if name=='bge_frozen.npy' else z.read(name))
 target=tmp_path/'unwritten'
 with pytest.raises(ValueError,match='Checksum'):ns['restore_light_backup'](bad,b,m,cfg,target)
 assert not target.exists()
