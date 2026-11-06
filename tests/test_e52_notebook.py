"""E52 official PEFT DoRA tests; synthetic CPU evidence, not AES research scores."""
import ast
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests'))
from test_e41_and_e30_native import runtime, setup, definitions
from test_e36_notebook import execute_full_native_fixture

NAME = 'AES_300_E52_E43_S3_DoRA8_Native_Learnable_GPU'
NOTEBOOK = ROOT/'notebooks/experiments'/(NAME+'.ipynb')
BASE = ROOT/'notebooks/experiments/AES_300_E43_E42_S3_MaskedMeanPooling_Native_Learnable_GPU.ipynb'


def test_source_protocol_and_deterministic_builder():
    parent, raw = BASE.read_bytes(), NOTEBOOK.read_bytes()
    n = nbformat.read(NOTEBOOK, as_version=4); nbformat.validate(n)
    m = n.metadata.aes_experiment
    assert m.scenarios == ['S3'] and m.cv_fits == 5 and m.refits == m.test_evaluations == 1
    assert m.dora_use_dora and m.original_roberta_frozen and m.dora_rank == 8 and m.dora_alpha == 16
    assert m.s3_encoder_lr == m.dora_lr == 1e-4 and m.loss == 'MSE'
    code = [c for c in n.cells if c.cell_type == 'code']; fp = n.metadata.aes_standalone.code_sha256
    assert hashlib.sha256('\n'.join(c.source.replace(fp, '__NOTEBOOK_CODE_SHA256__') for c in code).encode()).hexdigest() == fp
    text = '\n'.join(c.source for c in code)
    for bad in ['cosine','multihead','fusion_gate','ListNet','LDS_POWER','WORKER_SOURCE','CV_SOURCE','TRAINING_SEEDS','REFIT_SCHEDULER_EPOCHS']:
        assert bad not in text, bad
    assert 'use_dora=True' in text and 'peft==0.14.0' in text and 'accelerate==0.34.2' in text
    assert 'Hasil_'+NAME+'.zip' in text
    for c in code:
        assert c.metadata.aes52_role and not c.outputs and c.execution_count is None
        assert not any(line.lstrip().startswith(('%','!')) for line in c.source.splitlines())
        assert not any(isinstance(v,ast.Call) and isinstance(v.func,ast.Name) and v.func.id in ['exec','eval'] for v in ast.walk(ast.parse(c.source)))
    a,b = definitions(BASE),definitions(NOTEBOOK)
    for name in ['SingleScoreModel','ScalarMix','AnswerBiGRUAttention','AnswerBiGRUCoAttentionMeanPool','Engine','train_all',
                 'make_split','metrics','gwet_ac2','encoder','bge_features','pair_content_windows','pair_batch_windows','prediction_frame']:
        assert ast.dump(a[name]) == ast.dump(b[name]).replace('E52','E43'),name
    loop = lambda fn: next(v for v in fn.body if isinstance(v,ast.For) and isinstance(v.target,ast.Name) and v.target.id == 'epoch')
    assert ast.dump(loop(a['fit'])) == ast.dump(loop(b['fit']))
    subprocess.run([sys.executable,'scripts/build_e52_notebook.py'],cwd=ROOT,check=True,capture_output=True)
    assert BASE.read_bytes() == parent and NOTEBOOK.read_bytes() == raw == (ROOT/'aes-learning/notebooks/experiments'/NOTEBOOK.name).read_bytes()


def make_env(tmp_path):
    ns = runtime(NOTEBOOK,'aes52_role'); cfg,bundle = setup(ns,tmp_path)
    cfg['lesson_version'] = 'e52_native_dora8_v1'
    return ns,cfg,bundle


def test_official_dora_initial_function_rng_and_common_weights_match_e43(tmp_path):
    ns,cfg,bundle = make_env(tmp_path); old = runtime(BASE,'aes43_role')
    old['encoder'],old['tokenizer'] = ns['encoder'],ns['tokenizer']
    oldcfg = old['config']({k:v for k,v in cfg.items() if not k.startswith('dora_') and k != 'encoder_adaptation'})
    ns['seed_everything'](71); model = ns['make_model']('S3',cfg).eval(); rng = torch.get_rng_state()
    old['seed_everything'](71); control = old['make_model']('S3',oldcfg).eval()
    assert torch.equal(rng,torch.get_rng_state())
    assert model.roberta.peft_config['default'].use_dora is True
    weights = model.roberta.get_base_model().state_dict()
    original = {k.replace('.base_layer.','.'):v for k,v in weights.items() if '.lora_' not in k}
    assert set(original) == set(control.roberta.state_dict())
    for name,value in control.roberta.state_dict().items():
        torch.testing.assert_close(original[name],value,rtol=0,atol=0)
    for name,value in control.state_dict().items():
        if not name.startswith('roberta.'):
            torch.testing.assert_close(model.state_dict()[name],value,rtol=0,atol=0)
    bge = torch.randn(24,1024)
    eng = ns['Engine'](model,bundle.texts,bge,cfg,'cpu',contexts=ns['contexts_from_bundle'](bundle))
    ref = old['Engine'](control,bundle.texts,bge,oldcfg,'cpu',contexts=old['contexts_from_bundle'](bundle))
    torch.testing.assert_close(eng.forward([0,1]),ref.forward([0,1]),rtol=1e-6,atol=1e-6)
    assert ns['make_split'](bundle,cfg) == old['make_split'](bundle,oldcfg)
    report = ns['dora_adapter_report'](model)
    assert len(report) == 8 and sum(r['Parameter Adapter'] for r in report) == 104448
    assert all(r['DoRA Aktif'] and not r['Bobot Dasar Dilatih'] and r['Norma B'] == 0 for r in report)


def test_learning_changes_direction_magnitude_and_fusion_but_not_frozen_base(tmp_path):
    ns,cfg,bundle = make_env(tmp_path); ns['seed_everything'](42)
    model = ns['make_model']('S3',cfg); bge = torch.randn(24,1024)
    engine = ns['Engine'](model,bundle.texts,bge,cfg,'cpu',contexts=ns['contexts_from_bundle'](bundle))
    before = {n:p.detach().clone() for n,p in model.named_parameters()}
    groups = ns['optimizer_groups'](model,cfg)
    encoder = next(g for g in groups if g['name'] == 'roberta')
    assert encoder['lr'] == 1e-4 and all(p.requires_grad for p in encoder['params'])
    optimizer = torch.optim.AdamW(groups,weight_decay=cfg['weight_decay'])
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        prediction = engine.forward([0,1]); loss = (prediction-.83).square().mean(); loss.backward()
        assert model.projection[0].weight.grad.abs().sum() > 0
        assert model.sequence_pool.matching_projection.weight.grad.abs().sum() > 0
        optimizer.step()
    for name,p in model.named_parameters():
        if not p.requires_grad:
            assert name.startswith('roberta.') and p.grad is None
            torch.testing.assert_close(p,before[name],rtol=0,atol=0)
    for tag in ['.lora_A.','.lora_B.','.lora_magnitude_vector.']:
        params = [(n,p) for n,p in model.named_parameters() if tag in n]
        assert len(params) == 8
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for _,p in params)
        assert any(not torch.equal(p,before[n]) for n,p in params),tag
    assert all(not any(p.requires_grad for p in module.get_base_layer().parameters())
               for module in model.roberta.modules() if hasattr(module,'lora_magnitude_vector'))
    with torch.autocast('cpu',dtype=torch.bfloat16):
        out = engine.forward([0]); amp_loss = out.float().square().mean()
    assert torch.isfinite(out).all(); amp_loss.backward()
    assert torch.isfinite(model.head[0].weight.grad).all()


def test_config_checkpoint_roundtrip_and_wrong_or_missing_adapter_rejected(tmp_path):
    ns,cfg,bundle = make_env(tmp_path)
    for key,value in [('dora_use_dora',False),('encoder_adaptation','lora'),('dora_rank',0),('dora_rank',True),
                      ('dora_targets',['dense']),('dora_alpha',32),('dora_dropout',.1),('dora_lr',float('nan')),
                      ('s3_encoder_lr',5e-6)]:
        with pytest.raises(ValueError): ns['config']({**cfg,key:value})
    model = ns['make_model']('S3',cfg).eval(); manifest = ns['make_split'](bundle,cfg)
    path = tmp_path/'final.pt'; ns['save_model'](path,model,cfg,bundle,manifest,{})
    restored,payload = ns['load_model'](path,cfg)
    for name,value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name],value,rtol=0,atol=0)
    bge = torch.randn(24,1024)
    def predict(m):return ns['Engine'](m,bundle.texts,bge,cfg,'cpu',contexts=ns['contexts_from_bundle'](bundle)).predict([0,1])
    np.testing.assert_array_equal(predict(model),predict(restored))
    with pytest.raises(ValueError,match='Metadata checkpoint'): ns['load_model'](path,{**cfg,'dora_use_dora':False})
    magnitude = next(k for k in payload['state'] if '.lora_magnitude_vector.' in k)
    del payload['state'][magnitude]; torch.save(payload,path)
    with pytest.raises(RuntimeError,match='Missing key'):ns['load_model'](path,cfg)


def test_interrupted_training_resume_restores_dora_and_optimizer_exactly(tmp_path):
    ns,cfg,bundle = make_env(tmp_path); bge = torch.randn(24,1024); path = tmp_path/'progress.json'
    save = ns['save_training_progress']
    def interrupt(file,payload):
        save(file,payload)
        if payload['epoch'] == 1:raise RuntimeError('TECHNICAL INTERRUPT')
    ns['save_training_progress'] = interrupt
    with pytest.raises(RuntimeError,match='TECHNICAL INTERRUPT'):
        ns['fit']('S3',list(range(8)),None,bundle,bge,cfg,2,42,checkpoint=path)
    ns['save_training_progress'] = save
    resumed,_,a = ns['fit']('S3',list(range(8)),None,bundle,bge,cfg,2,42,checkpoint=path)
    plain,_,b = ns['fit']('S3',list(range(8)),None,bundle,bge,cfg,2,42)
    assert a['history'] == b['history'] and a['dora_adapters'] == b['dora_adapters'] and a['parameter_counts'] == b['parameter_counts']
    for name,value in plain.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name],value,rtol=0,atol=0)


def test_complete_native_notebook_twice_and_serialized_dora_report_gate(tmp_path):
    ns = runtime(NOTEBOOK,'aes52_role')
    cols = list(ns['prediction_report_tables']([{'ID Jawaban':'TECHNICAL','Nilai Guru':80,'Prediksi S3':79.5}])['Nilai Guru dan Prediksi'][0])
    execute_full_native_fixture(tmp_path,NOTEBOOK,'aes52_role',1e-4,encoder_layers=4,prediction_columns=cols)
    info = json.loads((tmp_path/'standalone/workspace/outputs/dataset300_download.json').read_text())
    run = Path(info['run_folder']); cfg = json.loads((run/'effective_config.json').read_text())
    bundle = ns['read_dataset'](cfg); manifest = ns['make_split'](bundle,cfg); folder = run/'hasil_ringkas'
    proof = ns['validate_e52_download'](folder,bundle,manifest,cfg)
    assert proof['test_rows'] == 6 and proof['oof_rows'] == 18 and proof['dora_module_rows'] == 48
    assert proof['dora_checkpoints'] == 6 and proof['dora_enabled_and_original_roberta_frozen_verified']
    f = folder/'05_Riwayat_Pelatihan.xlsx'; original = f.read_bytes()
    table = pd.read_excel(f,sheet_name='Adapter DoRA')
    assert len(table) == 48 and table['DoRA Aktif'].all() and not table['Bobot Dasar Dilatih'].any()
    for column,value in [('DoRA Aktif',False),('Bobot Dasar Dilatih',True),('Rank',4),('Parameter Adapter',1)]:
        w = load_workbook(f); sheet = w['Adapter DoRA']
        col = next(c.column for c in sheet[1] if c.value == column)
        sheet.cell(2,col,value); w.save(f); w.close()
        with pytest.raises(RuntimeError,match='DoRA'):ns['validate_e52_download'](folder,bundle,manifest,cfg)
        f.write_bytes(original)
    w = load_workbook(f); sheet = w['Adapter DoRA']; sheet.delete_rows(2); w.save(f); w.close()
    with pytest.raises(RuntimeError,match='DoRA'):ns['validate_e52_download'](folder,bundle,manifest,cfg)
    f.write_bytes(original)
    evaluation = folder/'04_Prediksi_dan_Evaluasi.xlsx'; original = evaluation.read_bytes()
    for title in ['Perbandingan Model','Ringkasan Validasi']:
        w = load_workbook(evaluation); sheet = w[title]; sheet.delete_rows(2,sheet.max_row-1); w.save(evaluation);w.close()
        with pytest.raises(RuntimeError,match='Unduhan dihentikan'):ns['validate_e52_download'](folder,bundle,manifest,cfg)
        evaluation.write_bytes(original)
    print(json.dumps({'scope':'SYNTHETIC CPU ONLY','full_notebook_twice':True,'gate':proof,'bad_adapter_and_empty_summaries_rejected':True}))
