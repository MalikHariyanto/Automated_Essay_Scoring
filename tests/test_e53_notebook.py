"""E53 EMA recurrence, online trajectory, selection, resume, and complete native export."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import nbformat
import numpy as np
import pandas as pd
from openpyxl import load_workbook
import pytest
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from test_e41_and_e30_native import runtime,setup,definitions
from test_e36_notebook import execute_full_native_fixture
NAME = 'AES_300_E53_E43_S3_EMA099_Native_Learnable_GPU'
NOTEBOOK = ROOT/'notebooks/experiments'/(NAME+'.ipynb')
BASE = ROOT/'notebooks/experiments/AES_300_E43_E42_S3_MaskedMeanPooling_Native_Learnable_GPU.ipynb'


def make_env(tmp_path):
    ns = runtime(NOTEBOOK,'aes53_role'); cfg,bundle = setup(ns,tmp_path)
    cfg['lesson_version'] = 'e53_native_ema099_v1'
    return ns,cfg,bundle


def latest_progress(path):
    manifest = json.loads(path.read_text())
    return torch.load(path.parent/manifest['snapshots'][0]['file'],map_location='cpu',weights_only=True)


def test_structure_protocol_and_rebuild_preserve_e43_architecture():
    parent,raw = BASE.read_bytes(),NOTEBOOK.read_bytes()
    n = nbformat.read(NOTEBOOK,as_version=4);nbformat.validate(n);m = n.metadata.aes_experiment
    assert m.scenarios == ['S3'] and m.cv_fits == 5 and m.refits == m.test_evaluations == 1
    assert m.inference_weights == 'ema' and m.ema_decay_max == .99 and m.ema_warmup_offset == 10
    assert m.full_encoder_fine_tuning and m.s3_encoder_lr == 5e-6 and m.loss == 'MSE'
    code = [c for c in n.cells if c.cell_type == 'code'];fp = n.metadata.aes_standalone.code_sha256
    assert hashlib.sha256('\n'.join(c.source.replace(fp,'__NOTEBOOK_CODE_SHA256__') for c in code).encode()).hexdigest() == fp
    text = '\n'.join(c.source for c in code)
    for bad in ['use_dora','from peft','multihead','fusion_gate','cosine','ListNet','LDS_POWER','WORKER_SOURCE','CV_SOURCE','TRAINING_SEEDS','REFIT_SCHEDULER_EPOCHS']:
        assert bad not in text,bad
    assert 'Hasil_'+NAME+'.zip' in text
    for c in code:
        assert c.metadata.aes53_role and not c.outputs and c.execution_count is None
        assert not any(line.lstrip().startswith(('%','!')) for line in c.source.splitlines())
        assert not any(isinstance(v,ast.Call) and isinstance(v.func,ast.Name) and v.func.id in ['exec','eval'] for v in ast.walk(ast.parse(c.source)))
    a,b = definitions(BASE),definitions(NOTEBOOK)
    for name in ['SingleScoreModel','ScalarMix','AnswerBiGRUAttention','AnswerBiGRUCoAttentionMeanPool','Engine','make_model',
                 'optimizer_groups','train_all','make_split','metrics','gwet_ac2','encoder','bge_features','pair_content_windows','pair_batch_windows','prediction_frame']:
        assert ast.dump(a[name]) == ast.dump(b[name]).replace('E53','E43'),name
    subprocess.run([sys.executable,'scripts/build_e53_notebook.py'],cwd=ROOT,check=True,capture_output=True)
    assert BASE.read_bytes() == parent and NOTEBOOK.read_bytes() == raw == (ROOT/'aes-learning/notebooks/experiments'/NOTEBOOK.name).read_bytes()


def test_ema_recurrence_pytorch_reference_buffers_skipped_steps_and_state(tmp_path):
    ns,cfg,_ = make_env(tmp_path)
    online = nn.Linear(3,2);online.register_buffer('counter',torch.tensor([0],dtype=torch.long))
    rng = torch.get_rng_state();ema = ns['ExponentialWeightAverage'](online,cfg)
    assert torch.equal(rng,torch.get_rng_state()) and not any(p.requires_grad for p in ema.model.parameters())
    ref = AveragedModel(online,avg_fn=lambda average,current,count:
                        average*ns['ema_decay_at'](int(count)+1)+current*(1-ns['ema_decay_at'](int(count)+1)))
    manual = {n:p.detach().clone() for n,p in online.named_parameters()}
    for step in range(1,6):
        with torch.no_grad():
            for p in online.parameters():p.add_(step*.07)
            online.counter.add_(1)
        before = copy.deepcopy(ema.state_dict())
        assert not ema.update_after_optimizer(online,lr_active=False,skipped=False)
        assert not ema.update_after_optimizer(online,lr_active=True,skipped=True)
        assert ema.updates == before['updates']
        for key,value in before['model'].items():torch.testing.assert_close(ema.model.state_dict()[key],value,rtol=0,atol=0)
        assert ema.update_after_optimizer(online,lr_active=True,skipped=False)
        decay = ns['ema_decay_at'](step)
        for name,p in online.named_parameters():
            manual[name] = p.detach().clone() if step == 1 else decay*manual[name]+(1-decay)*p.detach()
            torch.testing.assert_close(dict(ema.model.named_parameters())[name],manual[name],rtol=1e-6,atol=1e-7)
        ref.update_parameters(online)
        for name,p in ema.model.named_parameters():torch.testing.assert_close(p,dict(ref.module.named_parameters())[name],rtol=1e-6,atol=1e-7)
        assert torch.equal(ema.model.counter,online.counter)
    restored = ns['ExponentialWeightAverage'](online,cfg);restored.load_state_dict(ema.state_dict())
    assert restored.updates == 5
    for name,value in ema.model.state_dict().items():torch.testing.assert_close(restored.model.state_dict()[name],value,rtol=0,atol=0)
    for key,value in [('ema_decay_max',.9),('ema_decay_max',True),('ema_warmup_offset',False),('inference_weights','online')]:
        with pytest.raises(ValueError):ns['config']({**cfg,key:value})
    for patch in [{'updates':-1},{'maximum':.9},{'offset':3}]:
        with pytest.raises(ValueError):restored.load_state_dict({**ema.state_dict(),**patch})
    assert ns['ema_decay_at'](100000) == .99 and ns['ema_decay_at'](2) == .25


def test_initial_predictions_and_ema_copy_do_not_change_model_or_rng(tmp_path):
    ns,cfg,bundle = make_env(tmp_path); ns['seed_everything'](71)
    model = ns['make_model']('S3',cfg).eval();before = {n:v.detach().clone() for n,v in model.state_dict().items()}
    rng = torch.get_rng_state();ema = ns['ExponentialWeightAverage'](model,cfg)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(p.requires_grad for p in model.parameters()) and not any(p.requires_grad for p in ema.model.parameters())
    bge = torch.randn(24,1024)
    def engine(m):return ns['Engine'](m,bundle.texts,bge,cfg,'cpu',contexts=ns['contexts_from_bundle'](bundle))
    torch.testing.assert_close(engine(model).forward([0,1]),engine(ema.model).forward([0,1]),rtol=0,atol=0)
    model.train();out = engine(model).forward([0,1]);out.square().mean().backward()
    assert model.roberta.encoder.layer[0].attention.self.query.weight.grad.abs().sum() > 0
    assert model.projection[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in ema.model.parameters())
    for name,value in ema.model.state_dict().items():torch.testing.assert_close(value,before[name],rtol=0,atol=0)


def test_online_training_trajectory_matches_e43_and_refit_returns_ema(tmp_path):
    ns,cfg,bundle = make_env(tmp_path);old = runtime(BASE,'aes43_role')
    old['encoder'],old['tokenizer'] = ns['encoder'],ns['tokenizer']
    bge = torch.randn(24,1024);path = tmp_path/'online_and_ema.json'
    selected,_,info = ns['fit']('S3',list(range(8)),None,bundle,bge,cfg,2,42,checkpoint=path)
    payload = latest_progress(path)
    original,_,control = old['fit']('S3',list(range(8)),None,bundle,bge,cfg,2,42)
    assert [r['train_mse_scaled'] for r in info['history']] == [r['train_mse_scaled'] for r in control['history']]
    assert info['optimizer_learning_rates'] == control['optimizer_learning_rates']
    for name,value in original.state_dict().items():
        torch.testing.assert_close(payload['model'][name],value,rtol=0,atol=0)
        torch.testing.assert_close(selected.state_dict()[name],payload['ema_state']['model'][name],rtol=0,atol=0)
    assert any(not torch.equal(v,payload['model'][n]) for n,v in selected.state_dict().items())
    assert payload['ema_state']['updates'] == info['active_lr_steps'] == 2
    assert selected.inference_weights_kind == info['inference_weights'] == 'ema'
    final = tmp_path/'final.pt';manifest = ns['make_split'](bundle,cfg)
    ns['save_model'](final,selected,cfg,bundle,manifest,info)
    restored,saved = ns['load_model'](final,cfg)
    for name,value in selected.state_dict().items():torch.testing.assert_close(restored.state_dict()[name],value,rtol=0,atol=0)
    assert saved['inference_weights'] == restored.inference_weights_kind == 'ema'
    with pytest.raises(ValueError):ns['save_model'](tmp_path/'wrong.pt',original,cfg,bundle,manifest,info)
    with pytest.raises(ValueError,match='Metadata checkpoint'):ns['load_model'](final,{**cfg,'ema_decay_max':.9})
    saved['inference_weights'] = 'online';torch.save(saved,final)
    with pytest.raises(ValueError,match='bukan bobot EMA'):ns['load_model'](final,cfg)


def test_checkpoint_and_early_stop_follow_ema_not_online_minimum(tmp_path):
    ns,cfg,bundle = make_env(tmp_path);cfg['early_stopping_patience'] = 2
    online_errors,ema_errors = [6.,5.,4.,3.,2.,1.],[5.,2.,4.,6.,7.,8.]
    calls = {'online':0,'ema':0};original_predict = ns['Engine'].predict
    def controlled(self,indices):
        key = 'online' if next(self.model.parameters()).requires_grad else 'ema'
        values = online_errors if key == 'online' else ema_errors
        error = values[calls[key]];calls[key] += 1
        return bundle.y[indices]+error
    ns['Engine'].predict = controlled
    path = tmp_path/'selection.json'
    selected,_,info = ns['fit']('S3',list(range(8)),[8,9],bundle,torch.randn(24,1024),cfg,6,42,checkpoint=path)
    ns['Engine'].predict = original_predict
    assert info['best_epoch'] == 2 and info['epochs_completed'] == 4 and info['early_stopped']
    assert info['best_validation_rmse'] == 2.0 and min(info['history'],key=lambda r:r['validation_online']['RMSE'])['epoch'] == 4
    saved = latest_progress(path)
    for name,value in selected.state_dict().items():torch.testing.assert_close(value,saved['best_state'][name],rtol=0,atol=0)
    assert info['ema']['Update EMA Checkpoint Terpilih'] == 2 and info['ema']['Update EMA Total Latihan'] == 4


def test_interrupted_fit_restores_online_ema_optimizer_and_rng(tmp_path):
    ns,cfg,bundle = make_env(tmp_path);bge = torch.randn(24,1024);path = tmp_path/'progress.json'
    save = ns['save_training_progress']
    def stop(file,payload):
        save(file,payload)
        if payload['epoch'] == 1:raise RuntimeError('TECHNICAL INTERRUPT')
    ns['save_training_progress'] = stop
    with pytest.raises(RuntimeError,match='TECHNICAL INTERRUPT'):
        ns['fit']('S3',list(range(8)),None,bundle,bge,cfg,2,42,checkpoint=path)
    ns['save_training_progress'] = save
    resumed,_,a = ns['fit']('S3',list(range(8)),None,bundle,bge,cfg,2,42,checkpoint=path)
    plain,_,b = ns['fit']('S3',list(range(8)),None,bundle,bge,cfg,2,42)
    assert a['history'] == b['history'] and a['ema'] == b['ema'] and a['scalar_mix_weights'] == b['scalar_mix_weights']
    for name,value in plain.state_dict().items():torch.testing.assert_close(resumed.state_dict()[name],value,rtol=0,atol=0)


def test_complete_notebook_twice_and_serialized_ema_reports(tmp_path):
    ns = runtime(NOTEBOOK,'aes53_role')
    cols = list(ns['prediction_report_tables']([{'ID Jawaban':'TECHNICAL','Nilai Guru':80,'Prediksi S3':79.5}])['Nilai Guru dan Prediksi'][0])
    execute_full_native_fixture(tmp_path,NOTEBOOK,'aes53_role',5e-6,encoder_layers=4,prediction_columns=cols)
    info = json.loads((tmp_path/'standalone/workspace/outputs/dataset300_download.json').read_text());run = Path(info['run_folder'])
    cfg = json.loads((run/'effective_config.json').read_text());bundle = ns['read_dataset'](cfg);manifest = ns['make_split'](bundle,cfg);folder = run/'hasil_ringkas'
    proof = ns['validate_e53_download'](folder,bundle,manifest,cfg)
    assert proof['test_rows'] == 6 and proof['oof_rows'] == 18 and proof['ema_checkpoints'] == 6
    assert proof['ema_comparison_rows'] == 5 and proof['ema_epoch_update_rows'] == 6 and proof['ema_selection_counts_and_decay_verified']
    checkpoint = torch.load(run/'S3/final.pt',map_location='cpu',weights_only=True)
    assert checkpoint['inference_weights'] == checkpoint['training']['inference_weights'] == 'ema'
    f = folder/'05_Riwayat_Pelatihan.xlsx';original = f.read_bytes()
    comparisons = pd.read_excel(f,sheet_name='Perbandingan EMA');config = pd.read_excel(f,sheet_name='Konfigurasi EMA')
    assert len(comparisons) == 5 and len(config) == 6
    for title,column,value in [('Konfigurasi EMA','Jenis Bobot Prediksi','online'),
                               ('Update EMA per Epoch','Update EMA',999),
                               ('Perbandingan EMA','Checkpoint Terpilih',False),
                               ('Perbandingan EMA','Selisih EMA - Online (poin)',999.)]:
        w = load_workbook(f);s = w[title];col = next(c.column for c in s[1] if c.value == column)
        s.cell(2,col,value);w.save(f);w.close()
        with pytest.raises(RuntimeError,match='EMA'):ns['validate_e53_download'](folder,bundle,manifest,cfg)
        f.write_bytes(original)
    evaluation = folder/'04_Prediksi_dan_Evaluasi.xlsx';original = evaluation.read_bytes()
    for title in ['Perbandingan Model','Ringkasan Validasi']:
        w = load_workbook(evaluation);s = w[title];s.delete_rows(2,s.max_row-1);w.save(evaluation);w.close()
        with pytest.raises(RuntimeError,match='Unduhan dihentikan'):ns['validate_e53_download'](folder,bundle,manifest,cfg)
        evaluation.write_bytes(original)
    print(json.dumps({'scope':'SYNTHETIC CPU ONLY','notebook_twice':True,'proof':proof,'bad_ema_metadata_and_empty_summaries_rejected':True}))
