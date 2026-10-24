"""Tiga seed/fold tetap/resume: fixture teknis, bukan hasil penelitian atau benchmark GPU."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import nbformat
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
import test_lora_notebook as technical
from test_teacher_score import TechnicalTokenizer,make_book

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/'notebooks/experiments'/os.environ.get('AES_THREE_SEED_NOTEBOOK','AES_300_E6_3Seed_GPU.ipynb')
BASE=ROOT/'notebooks/experiments/AES_300_S3_MLP_LR5e6_CPU.ipynb'


class TechnicalEncoder(nn.Module):
    """Encoder kecil untuk memeriksa protokol; bukan RoBERTa pretrained."""
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(64,768);self.scale=nn.Parameter(torch.ones(768))
    def forward(self,input_ids,attention_mask=None):
        value=self.embedding(input_ids)
        mask=attention_mask.unsqueeze(-1) if attention_mask is not None else torch.ones_like(value[:,:,:1])
        context=(value*mask).sum(1,keepdim=True)/mask.sum(1,keepdim=True).clamp_min(1)
        return types.SimpleNamespace(last_hidden_state=(value+context)*self.scale)


def runtime():
    old=technical.NOTEBOOK
    try:technical.NOTEBOOK=NOTEBOOK;ns=technical.runtime()
    finally:technical.NOTEBOOK=old
    def encoder(cfg,role):
        with torch.random.fork_rng():
            torch.manual_seed(170);return TechnicalEncoder()
    ns['encoder']=encoder;ns['tokenizer']=lambda *a,**k:TechnicalTokenizer()
    torch.set_num_threads(2)
    return ns


@pytest.fixture(scope='module')
def full_run(tmp_path_factory):
    root=tmp_path_factory.mktemp('THREE_SEED_TECHNICAL_ONLY');dataset=root/'TECHNICAL_FIXTURE_ONLY.xlsx';make_book(dataset)
    ns=runtime();cfg=ns['config']({'dataset':str(dataset),'output':str(root/'results'),'expected_rows':24,'device':'cpu',
        'epochs':1,'roberta_max_length':32,'roberta_overlap':4,'bge_max_length':32,'effective_batch':4,'micro_batch':1,
        'training_seeds':[42,123,2026],'run_tag':'technical_01','checkpoint_strategy':'two_slots','dropout':.1})
    bundle=ns['read_dataset'](cfg);manifest=ns['make_split'](bundle,cfg);output=ns['run_folder'](cfg,bundle)
    journal=ns['ExcelJournal'](output/'excel',{'scope':ns['SCOPE']});journal.save('02_dataset_dan_cleaning',{'Sebelum_sesudah':pd.DataFrame([{'row_id':r['row_id'],'teks_model':r['text'],'skor_guru':r['score']} for r in bundle.rows])},'TECHNICAL FIXTURE ONLY')
    bge=torch.nn.functional.normalize(torch.arange(24*1024,dtype=torch.float32).reshape(24,1024)+1,dim=1)
    calls=[];original_fit=ns['fit']
    def fit(scenario,training,validation,bundle,bge,cfg,epochs,seed,checkpoint=None):
        calls.append((cfg['seed'],scenario,validation is not None,seed))
        return original_fit(scenario,training,validation,bundle,bge,cfg,epochs,seed,checkpoint=checkpoint)
    ns['fit']=fit
    runs=ns['run_three_seeds'](bundle,manifest,bge,cfg,output,journal)
    return ns,cfg,bundle,manifest,bge,output,journal,runs,calls,dataset


def test_clean_gpu_notebook_mirror_fingerprint_and_reproducible_builder():
    n=nbformat.read(NOTEBOOK,as_version=4);nbformat.validate(n)
    cpu=n.metadata.aes_experiment.device=='cpu'
    assert n.metadata.aes_experiment.reused_controls==[]
    if cpu:assert 'accelerator' not in n.metadata
    else:assert n.metadata.accelerator=='GPU'
    assert n.metadata.aes_experiment.base_sha256==hashlib.sha256(BASE.read_bytes()).hexdigest()
    assert (ROOT/'aes-learning/notebooks/experiments'/NOTEBOOK.name).read_bytes()==NOTEBOOK.read_bytes()
    codes=[]
    for cell in n.cells:
        if cell.cell_type!='code':continue
        assert not cell.outputs and cell.execution_count is None
        code=cell.source.removeprefix('%%aes\n');ast.parse(code);codes.append(code.replace(n.metadata.aes_standalone.code_sha256,'__NOTEBOOK_CODE_SHA256__'))
        assert not any(word in code for word in ('googleapiclient','clear_training_progress','prepare_reused_controls','shutil.rmtree'))
    assert hashlib.sha256('\n'.join(codes).encode()).hexdigest()==n.metadata.aes_standalone.code_sha256
    if cpu:
        assert "device='cpu'" in n.cells[12].source and 'REQUIRE_GPU=False' in n.cells[12].source
        assert "'CUDA_VISIBLE_DEVICES':''" in n.cells[7].source
        assert 'torch.set_num_threads(2)' in n.cells[12].source
    else:
        assert "device='cuda'" in n.cells[12].source and 'REQUIRE_GPU=True' in n.cells[12].source
        assert 'CUBLAS_WORKSPACE_CONFIG' in n.cells[7].source
    builder='build_e6_three_seed_cpu_notebook.py' if cpu else 'build_e6_three_seed_notebook.py'
    before=NOTEBOOK.read_bytes();subprocess.run([sys.executable,str(ROOT/'scripts'/builder)],check=True,capture_output=True)
    assert NOTEBOOK.read_bytes()==before


def test_all_45_fits_plus_9_refits_keep_partitions_architecture_and_e6_rates(full_run):
    ns,cfg,bundle,manifest,bge,output,journal,runs,calls,dataset=full_run
    assert len(calls)==54 and sum(row[2] for row in calls)==45
    architectures={s:set() for s in ns['SCENARIOS']};heads={s:set() for s in ns['SCENARIOS']}
    for run in runs:
        assert ns['make_split'](bundle,run['cfg'])==manifest
        for scenario in ns['SCENARIOS']:
            selected=[r for r in calls if r[:2]==(run['seed'],scenario)]
            assert [r[3] for r in selected if r[2]]==[run['seed']+i for i in range(1,6)]
            assert [r[3] for r in selected if not r[2]]==[run['seed']]
            audits=list((run['output']/scenario).glob('progress_*_audit.json'));assert len(audits)==6
            for path in audits:
                audit=json.loads(path.read_text());architectures[scenario].add(audit['architecture_sha256']);heads[scenario].add(audit['initial_head_projection_sha256'])
                assert audit['repeat_eval_max_abs_delta']==0
                assert audit['encoder_lr']==(5e-6 if scenario=='S3' else 1e-5 if scenario=='S1' else None)
                assert audit['head_lr']==1e-4
            report=run['report'];assert report['test_evaluations_per_scenario']==1 and report['test_rows']==manifest['test_rows']
    assert all(len(v)==1 for v in architectures.values())
    assert all(len(v)>3 for v in heads.values())
    assert len({json.loads(p.read_text())['split_sha256'] for p in output.glob('seed_*/split_manifest.json')})==1


def test_same_plan_resumes_without_any_new_training(full_run):
    ns,cfg,bundle,manifest,bge,output,journal,runs,calls,dataset=full_run
    previous=len(calls)
    old=ns['digest'](runs[0]['output']/'excel/02_dataset_dan_cleaning.xlsx')
    journal.save('02_dataset_dan_cleaning',{'Sebelum_sesudah':pd.DataFrame([{'row_id':r['row_id'],'teks_model':r['text'],'skor_guru':r['score']} for r in bundle.rows])},'TECHNICAL SAME DATA REGENERATED ON RESUME')
    assert ns['digest'](journal.folder/'02_dataset_dan_cleaning.xlsx')!=old
    again=ns['run_three_seeds'](bundle,manifest,bge,cfg,output,journal)
    assert len(calls)==previous and all(run['report']['reused_completed_run'] for run in again)
    assert ns['digest'](runs[0]['output']/'excel/02_dataset_dan_cleaning.xlsx')==old


def test_two_slot_checkpoint_recovers_previous_epoch_after_corruption(full_run,tmp_path):
    ns=full_run[0];path=tmp_path/'progress_fold_1.json';contract={'technical_fixture':True}
    for epoch in range(1,6):ns['save_training_progress'](path,{'epoch':epoch,'contract':contract})
    assert len(list(tmp_path.glob('*.pt')))==2
    assert ns['load_training_progress'](path,contract)['epoch']==5
    manifest=json.loads(path.read_text());(tmp_path/manifest['snapshots'][0]['file']).write_bytes(b'TECHNICAL INTERRUPTED WRITE')
    assert ns['load_training_progress'](path,contract)['epoch']==4


def test_summary_has_45_folds_and_true_between_seed_sd_not_15_independent_runs(full_run):
    ns,cfg,bundle,manifest,bge,output,journal,runs,calls,dataset=full_run
    frames=ns['summarize_seed_runs'](runs,bundle)
    assert len(frames['Seluruh Fold'])==45 and len(frames['Audit Model dan Seed'])==54
    assert len(frames['Guru dan Prediksi'])==manifest['test_rows'] and frames['Guru dan Prediksi'].shape[1]==11
    for scenario in ns['SCENARIOS']:
        values=frames['CV per Seed'].query('Skenario == @scenario')['RMSE (poin)']
        row=frames['Rerata CV 3 Seed'].set_index('Skenario').loc[scenario]
        assert row['Jumlah seed']==3 and row['SD RMSE (poin)']==pytest.approx(np.std(values,ddof=1))
    destination=output/'TECHNICAL_SUMMARY.xlsx'
    ns['write_excel'](destination,frames,{'scope':'TECHNICAL_FIXTURE_ONLY'})
    assert destination.is_file()


def test_changed_plan_or_shared_features_is_rejected_before_training(full_run):
    ns,cfg,bundle,manifest,bge,output,journal,runs,calls,dataset=full_run
    previous=len(calls)
    with pytest.raises(ValueError,match='Rencana'):
        ns['run_three_seeds'](bundle,manifest,bge,{**cfg,'head_lr':2e-4},output,journal)
    with pytest.raises(ValueError,match='Rencana'):
        ns['run_three_seeds'](bundle,manifest,bge+1,cfg,output,journal)
    assert len(calls)==previous


def test_same_seed_repeats_initialization_and_training_on_cpu_fixture(full_run,tmp_path):
    ns,cfg,bundle,manifest,bge,output,journal,runs,calls,dataset=full_run
    fold=manifest['folds'][0];predictions=[];audits=[]
    for repetition in range(2):
        model,engine,info=ns['fit']('S3',fold['train'],fold['validation'],bundle,bge,cfg,1,43)
        predictions.append(engine.predict(fold['validation']));audits.append(info['reproducibility']);del engine,model
    np.testing.assert_array_equal(predictions[0],predictions[1])
    assert audits[0]['initial_head_projection_sha256']==audits[1]['initial_head_projection_sha256']


def test_complete_download_cell_executes_and_preserves_seed_paths(full_run,monkeypatch):
    import zipfile
    ns,cfg,bundle,manifest,bge,output,journal,runs,calls,dataset=full_run
    workspace=output/'technical_workspace';(workspace/'outputs').mkdir(parents=True,exist_ok=True)
    ns.update(seed_runs=runs,bundle=bundle,manifest=manifest,output=output,journal=journal,ROOT=workspace,IN_COLAB=False)
    ns['aes_cell']=lambda line,code:exec(code,ns)
    notebook=nbformat.read(NOTEBOOK,as_version=4)
    previous=len(calls)
    ns.update(RUN_TRAINING=True,RUN_FINAL_FEATURES=True,RUN_WORD_VECTORS=False,bge=bge,trace_index=manifest['test'][0],trace_text=bundle.texts[manifest['test'][0]])
    monkeypatch.setitem(ns,'bge_features',lambda sample,cfg,output:bge[:len(sample.rows)])
    exec(compile(notebook.cells[41].source.removeprefix('%%aes\n'),'<technical-scoring-cell>','exec'),ns)
    exec(compile(notebook.cells[43].source.removeprefix('%%aes\n'),'<technical-final-vector-cell>','exec'),ns)
    # Fixture kata dihitung dari encoder kecil; bukan vektor RoBERTa/BGE penelitian.
    toy=ns['encoder'](cfg,'roberta')
    vector=toy(input_ids=torch.tensor([[0,2]]),attention_mask=torch.ones(1,2)).last_hidden_state[:,0].detach().numpy()
    words=pd.DataFrame({'word_index':[0],'word':['TECHNICAL_FIXTURE_ONLY'],**{f'v_{i:04d}':vector[:,i] for i in range(768)}})
    for run in runs:run['journal'].save('07_vektor_kata_kalimat_pretrained',{'roberta_word_contextual':words},'TECHNICAL ENCODER FIXTURE ONLY')
    exec(compile(notebook.cells[46].source,'<technical-final-download-cell>','exec'),ns)
    package=json.loads((workspace/'outputs/dataset300_download.json').read_text())
    with zipfile.ZipFile(package['archive']) as z:
        assert len(z.namelist())==16
        assert 'Ringkasan_3_Seed.xlsx' in z.namelist()
        assert any(name.startswith('seed_42/') for name in z.namelist())
        assert any(name.startswith('seed_123/') for name in z.namelist())
        assert any(name.startswith('seed_2026/') for name in z.namelist())
    assert len(calls)==previous


def test_edited_runtime_function_cannot_silently_reuse_old_results(full_run,monkeypatch):
    ns,cfg,bundle,manifest,bge,output,journal,runs,calls,dataset=full_run
    previous=len(calls)
    with monkeypatch.context() as patch:
        patch.setitem(ns,'optimizer_groups',lambda model,cfg:[])
        with pytest.raises(ValueError,match='Rencana'):
            ns['run_three_seeds'](bundle,manifest,bge,cfg,output,journal)
    assert len(calls)==previous


def test_actual_code_fingerprint_survives_two_fresh_python_workers():
    import os
    command=[sys.executable,'-c',"from test_three_seed_notebook import runtime; ns=runtime(); print(ns['runtime_code_sha256']())"]
    env={**os.environ,'PYTHONPATH':str(ROOT/'tests')+os.pathsep+str(ROOT/'src'),'PYTHONHASHSEED':'0'}
    hashes=[subprocess.check_output(command,cwd=ROOT,env=env,text=True).splitlines()[-1] for _ in range(2)]
    assert len(hashes[0])==64 and hashes[0]==hashes[1]


def test_published_configuration_selects_device_and_keeps_all_e6_rates(tmp_path,monkeypatch):
    ns=runtime();checks=[]
    monkeypatch.setenv('AES300_DATASET',str(tmp_path/'TECHNICAL_ONLY.xlsx'))
    monkeypatch.setenv('AES300_RESULTS',str(tmp_path/'results'))
    monkeypatch.delenv('AES300_AUDIT_ONLY',raising=False)
    ns['check_gpu']=lambda require_gpu:checks.append(require_gpu) or {'name':'TECHNICAL DEVICE CHECK'}
    ns['show']=lambda *args,**kwargs:None
    notebook=nbformat.read(NOTEBOOK,as_version=4)
    exec(compile(notebook.cells[12].source.removeprefix('%%aes\n'),'<published-configuration>','exec'),ns)
    cfg=ns['cfg'];cpu=notebook.metadata.aes_experiment.device=='cpu'
    assert cfg['device']==('cpu' if cpu else 'cuda')
    assert checks==[not cpu]
    assert cfg['encoder_lr']==1e-5 and cfg['s3_encoder_lr']==5e-6
    assert cfg['projection_lr']==cfg['head_lr']==1e-4
    assert cfg['training_seeds']==[42,123,2026] and cfg['split_seed']==2026
    assert cfg['epochs']==25 and cfg['early_stopping_patience']==5
