"""Technical fixture: real Optuna/SQLite, simulated model. Not AES performance."""
import json,math,hashlib,random,os,sys,sqlite3,shutil,tempfile,time,traceback,warnings,gc,re,zipfile,io
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import version
import numpy as np
import pandas as pd
import optuna
from optuna.trial import TrialState
from sklearn.model_selection import GroupKFold,GroupShuffleSplit
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score
from openpyxl import Workbook,load_workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Font,PatternFill
optuna.logging.set_verbosity(optuna.logging.WARNING)
import ast
ROOT=Path(__file__).parent
NOTEBOOK=ROOT/'e17e/AES_Reset_E17E_S3_Optuna_TPE_Native_Learnable_GPU.ipynb'
if not NOTEBOOK.exists():
 NOTEBOOK=ROOT.parent/'eksperimen/03_tiga_input_multi_trait/notebooks/AES_Reset_E17E_S3_Optuna_TPE_Native_Learnable_GPU.ipynb'
nb=json.loads(NOTEBOOK.read_text())
def function_cell(name):
 for cell in nb['cells']:
  if cell['cell_type']!='code':continue
  s=''.join(cell['source']) if isinstance(cell['source'],list) else cell['source']
  if any(isinstance(n,ast.FunctionDef) and n.name==name for n in ast.parse(s).body):return s
 raise AssertionError('Missing function: '+name)
def module_cell(name):
 return function_cell({'hpo_contract':'stable_hash','hpo_storage':'restore_database',
                       'hpo_search':'suggest_parameters','hpo_confirm':'search_fingerprint',
                       'hpo_final':'run_final'}[name])
ns=globals()
display=lambda x:None
NOTEBOOK_CODE_ID='fixture';HPO_CODE_SHA256='fixture-code'
EPOCHS=25;TRAIN_SEED=42;HPO_TRIALS=18;TOP_K=2;HPO_SEED=2026;HPO_SPLIT_SEED=2029;STABILITY_SEEDS=(42,43,44)
SCOPE='fixture';DATASET_VERSION='fixture'
TRAITS=[('isi','isi',1,4),('organisasi','org',1,2),('kebahasaan','keb',1,2),('mekanik','mek',1,2)]
RUBRIC={'traits':[{'name':a,'min':lo,'max':hi,'categories':list(range(lo,hi+1))} for a,_,lo,hi in TRAITS]}
for name in ['clean','make_split','ordinal_prediction','metrics','excel_value','write_excel','lowest_score_rows']:
 exec(function_cell(name),ns)
tmp=tempfile.TemporaryDirectory();OUTPUT_ROOT=Path(tmp.name)
runtime_information=lambda:{'torch':'fixture','transformers':'fixture','device':'cpu','accelerator':None}
version_real=version
version=lambda name:'4.6.0' if name=='optuna' else version_real(name)
config=lambda x:x
cfg={'folds':5,'split_seed':2026,'split_candidates':128,'test_fraction':.2,
     'seed':42,'epochs':25,'dataset_version':'fixture','experiment':'E17E_fixture',
     's3_fusion':'concat','weight_decay':.01,'trait_corr_threshold':.7,'trait_relation_weight':0.}
groups=np.repeat(np.arange(30),2)
y=np.column_stack([1+np.arange(60)%4,1+np.arange(60)%2,1+(np.arange(60)//2)%2,1+(np.arange(60)//3)%2]).astype(float)
bundle=Bundle([{'row_id':f'row_{i:03d}','source_row':i+2,'response_number':str(i%2),'text':'fixture'} for i in range(60)],y,groups,'fixture-source',{})
manifest=make_split(bundle,cfg)
for module in ['hpo_contract','hpo_storage']:exec(module_cell(module),ns)
fit_calls=[];predict_calls=[]
class FakeModel:
 pass
class FakeEngine:
 def __init__(self,c):self.cfg=c
 def predict(self,ids):
  predict_calls.append(list(ids))
  # Same deterministic predictor at each epoch: isolates orchestration behavior.
  err=.04+abs(math.log(self.cfg['s3_encoder_lr']/3e-6))*.03+abs(self.cfg['s3_mlp_dropout']-.2)*.2
  return bundle.y[ids]+err
def fake_fit(scenario,training,validation,bundle,bge,cfg,epochs,seed,epoch_callback=None):
 assert not set(training)&set(manifest['test'])
 if validation is not None:assert not set(validation)&set(manifest['test'])
 fit_calls.append((tuple(training),tuple(validation or []),seed))
 engine=FakeEngine(cfg);history=[];pruned=False
 value=metrics(bundle.y[validation],engine.predict(validation))['RMSE'] if validation else None
 for epoch in range(1,7):
  row={'epoch':epoch,'validation':{'RMSE':value},'train_mse_scaled':.1/epoch}
  history.append(row)
  if epoch_callback is not None and epoch_callback(epoch,row,history):pruned=True;break
 return FakeModel(),engine,{'history':history,'best_epoch':1,'best_validation_rmse':value,'pruned':pruned}
fit=fake_fit;bge=None;release=lambda:gc.collect()
exec(function_cell('run_cv'),ns)
for module in ['hpo_search','hpo_confirm','hpo_final']:exec(module_cell(module),ns)
run_search()
assert len(finished_trials())==18
assert all(not(set(ids)&set(manifest['test'])) for ids in predict_calls)
assert len([t for t in study.trials if t.user_attrs.get('anchor') and t.state==TrialState.COMPLETE])==16
saved_trial_states=[(t.number,t.state.name,t.params,t.value) for t in study.trials]
oldpointer=json.loads(SNAPSHOT_POINTER.read_text());assert digest(STUDY_DIR/oldpointer['file'])==oldpointer['sha256']
# Simulate a dropped runtime with one RUNNING fit. Restore and ensure exactly one retry.
orphan=study.ask();suggest_parameters(orphan);backup_database(); orphan_number=orphan.number
exec(module_cell('hpo_storage'),ns)
assert next(t for t in study.trials if t.number==orphan_number).state==TrialState.FAIL
assert len([t for t in study.trials if t.user_attrs.get('retry_of')==orphan_number])==1
assert [(t.number,t.state.name,t.params,t.value) for t in study.trials[:18]]==saved_trial_states
exec(module_cell('hpo_storage'),ns)
assert len([t for t in study.trials if t.user_attrs.get('retry_of')==orphan_number])==1
before=len(fit_calls);run_search();assert len(fit_calls)==before # budget already complete
try:load_locked_selection()
except RuntimeError:pass
else:raise AssertionError('Final gate allowed missing confirmation')
run_confirmation();root,seal=load_locked_selection()
assert seal['test_used_for_selection'] is False
assert all(not(set(ids)&set(manifest['test'])) for ids in predict_calls)
before=len(fit_calls);run_confirmation();assert len(fit_calls)==before
# Corrupt a fold: final must reject it rather than silently evaluate test.
name=next(iter(seal['files_sha256']));p=root/name;original=p.read_bytes();p.write_text('{}')
try:load_locked_selection()
except ValueError:pass
else:raise AssertionError('Corrupted CV was not rejected')
p.write_bytes(original)
# Exercise the actual baseline-derived checkpoint/refit/test cache functions.
import pickle
class FakeTorch:
 @staticmethod
 def save(payload,path):Path(path).write_bytes(pickle.dumps(payload))
 @staticmethod
 def load(path,**kwargs):return pickle.loads(Path(path).read_bytes())
torch=FakeTorch()
FakeModel.scenario='S3'
FakeModel.state_dict=lambda self:{}
FakeModel.load_state_dict=lambda self,state,strict:None
FakeModel.to=lambda self,device:self
make_model=lambda scenario,cfg:FakeModel()
device_name=lambda cfg:'cpu'
contexts_from_bundle=lambda bundle:[('q','r')]*len(bundle.rows)
Engine=lambda model,texts,bge,cfg,device,**kwargs:FakeEngine(cfg)
for name in ['save_model','refit_checkpoint','evaluate_test']:exec(function_cell(name),ns)
winner_cfg=candidate_config(cfg,seal['winner']['params']); cache_folder=root/'cache_fixture';cache_folder.mkdir()
before=len(fit_calls)
r1=refit_checkpoint(4,bundle,manifest,bge,winner_cfg,cache_folder,'S3')
r2=refit_checkpoint(4,bundle,manifest,bge,winner_cfg,cache_folder,'S3')
assert r1==r2 and len(fit_calls)==before+1
before=len(predict_calls)
p1=evaluate_test(r1,bundle,manifest,bge,winner_cfg,cache_folder)
p2=evaluate_test(r1,bundle,manifest,bge,winner_cfg,cache_folder)
assert np.array_equal(p1,p2) and len(predict_calls)==before+1
try:refit_checkpoint(5,bundle,manifest,bge,winner_cfg,cache_folder,'S3')
except ValueError:pass
else:raise AssertionError('Changed refit duration accepted stale checkpoint')
# Exercise final reporting with simulated trained checkpoint/test, using real ZIP/XLSX.
refit_calls=[];test_calls=[]
def fake_refit(epochs,bundle,manifest,bge,cfg,output,scenario):
 refit_calls.append(epochs)
 return {'training':{'history':[]},'model_sha256':'fixture-checkpoint'}
def fake_test(refit,bundle,manifest,bge,cfg,output):
 test_calls.append(1);return FakeEngine(cfg).predict(manifest['test'])
refit_checkpoint=fake_refit;evaluate_test=fake_test
run_final();assert len(test_calls)==1 and len(refit_calls)==1
reports=list(root.glob('final/*/*.zip'));assert len(reports)==1
with zipfile.ZipFile(reports[0]) as z:
 assert z.testzip() is None
HPO_TRIALS=19
try:load_locked_selection()
except RuntimeError:pass
else:raise AssertionError('Extended study budget did not block final')
print('PASS: real Optuna search, SQLite recovery/retry, no duplicate completed trials, group isolation, CV reuse, selection seal, corruption guard, final ZIP/XLSX verification.')
prune_study=optuna.create_study(direction='minimize',pruner=optuna.pruners.MedianPruner(
 n_startup_trials=16,n_warmup_steps=5,interval_steps=1,n_min_trials=5))
for i in range(16):
 t=prune_study.ask()
 for epoch in range(1,7):t.report(.1,epoch)
 prune_study.tell(t,.1)
bad=prune_study.ask()
for epoch in range(1,5):
 bad.report(.5,epoch);assert not bad.should_prune()
bad.report(.5,5);assert bad.should_prune()
prune_study.tell(bad,state=TrialState.PRUNED)
print('PASS: pruning waits through epoch4 and stops a worse trial at epoch5 after startup.')
