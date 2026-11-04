"""E45 all-layer selection/gradients, E43 equivalence, and twelve-layer native run."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import nbformat
import numpy as np
import pandas as pd
import pytest
import torch
from transformers import RobertaConfig, RobertaModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from test_e41_and_e30_native import runtime, setup, definitions
from test_e36_notebook import execute_full_native_fixture

NOTEBOOK = ROOT/'notebooks/experiments/AES_300_E45_E43_S3_ScalarMix12_Native_Learnable_GPU.ipynb'
BASE = ROOT/'notebooks/experiments/AES_300_E43_E42_S3_MaskedMeanPooling_Native_Learnable_GPU.ipynb'
PARALLEL = ROOT/'notebooks/experiments/AES_300_E44_E43_S3_MSE_ListNet_Native_Learnable_GPU.ipynb'


def twelve_layer_encoder(cfg,role):
    assert role == 'roberta'
    with torch.random.fork_rng():
        torch.manual_seed(170)
        return RobertaModel(RobertaConfig(vocab_size=64,hidden_size=768,num_hidden_layers=12,
            num_attention_heads=12,intermediate_size=16,max_position_embeddings=64,
            pad_token_id=1,bos_token_id=0,eos_token_id=2,hidden_dropout_prob=0.,attention_probs_dropout_prob=0.),
            add_pooling_layer=False)


def test_structure_mse_protocol_and_deterministic_builder():
    raw,old,parallel = NOTEBOOK.read_bytes(),BASE.read_bytes(),PARALLEL.read_bytes()
    n = nbformat.read(NOTEBOOK,as_version=4);nbformat.validate(n)
    m = n.metadata.aes_experiment
    assert m.scenarios == ['S3'] and m.cv_fits == 5 and m.refits == m.test_evaluations == 1
    assert not m.multiple_seed_runs and not m.control_retrained and not m.ensemble
    assert m.scalar_mix_layers == 12 and m.scalar_mix_all_layers and not m.embedding_layer_included
    assert m.scalar_mix_trainable_parameters == 13 and m.added_parameters_vs_e43 == 8
    assert m.s3_encoder_lr == 5e-6 and m.scalar_mix_lr == 1e-4 and m.loss == 'MSE' and not m.ranking
    code = [c for c in n.cells if c.cell_type == 'code'];assert len(code) == 89
    fp = n.metadata.aes_standalone.code_sha256
    assert hashlib.sha256('\n'.join(c.source.replace(fp,'__NOTEBOOK_CODE_SHA256__') for c in code).encode()).hexdigest() == fp
    assert hashlib.sha256(old).hexdigest() == m.format_sha256
    for c in code:
        assert c.metadata.aes45_role and not c.outputs and c.execution_count is None
        assert not any(line.lstrip().startswith(('%','!')) for line in c.source.splitlines())
        assert not any(isinstance(v,ast.Call) and isinstance(v.func,ast.Name) and v.func.id in ('exec','eval')
                       for v in ast.walk(ast.parse(c.source)))
    text = '\n'.join(c.source for c in code)
    assert all(x not in text for x in ['cosine','WORKER_SOURCE','TRAINING_SEEDS','encoder_layer_decay',
                                     'ListNet','ranking_weight','activation_checkpoint'])
    a,b = definitions(BASE),definitions(NOTEBOOK)
    for name in ['optimizer_groups','fit','train_all','make_split','metrics','gwet_ac2','Engine',
                 'encoder','bge_features','AnswerBiGRUCoAttentionMeanPool','pair_content_windows',
                 'pair_batch_windows','export_training_event','load_model']:
        assert ast.dump(a[name]) == ast.dump(b[name]).replace('E45','E43'),name
    subprocess.run([sys.executable,'scripts/build_e45_notebook.py'],cwd=ROOT,check=True,capture_output=True)
    assert raw == NOTEBOOK.read_bytes() == (ROOT/'aes-learning/notebooks/experiments'/NOTEBOOK.name).read_bytes()
    assert old == BASE.read_bytes() and parallel == PARALLEL.read_bytes()


def test_mix_all_twelve_excludes_embedding_and_backpropagates_each_layer():
    ns = runtime(NOTEBOOK,'aes45_role');mix = ns['ScalarMix']()
    states = (torch.full((2,3,5),10000.,requires_grad=True),)+tuple(
        torch.full((2,3,5),float(i),requires_grad=True) for i in range(1,13))
    result = mix(states)
    torch.testing.assert_close(result,torch.full_like(result,6.5))
    torch.testing.assert_close(mix.weights(),torch.full((12,),1/12))
    assert sum(p.numel() for p in mix.parameters()) == 13
    result.sum().backward()
    assert states[0].grad is None
    for state in states[1:]:torch.testing.assert_close(state.grad,torch.full_like(state,1/12))
    assert torch.isfinite(mix.logits.grad).all() and (mix.logits.grad!=0).all()
    assert mix.gamma.grad.abs().sum() > 0
    with pytest.raises(ValueError,match='cukup layer'):mix(states[:12])


def test_shared_initialization_last_four_equivalence_and_all_layer_training(tmp_path):
    ns,old = runtime(NOTEBOOK,'aes45_role'),runtime(BASE,'aes43_role')
    cfg,bundle = setup(ns,tmp_path);oldcfg,_ = setup(old,tmp_path)
    ns['encoder'] = old['encoder'] = twelve_layer_encoder
    assert cfg['epochs'] == 25 and cfg['folds'] == 5 and cfg['early_stopping_patience'] == 5
    for key in old['DEFAULTS']:
        if key != 'scalar_mix_layers':assert cfg[key] == oldcfg[key],key
    ns['seed_everything'](75);model = ns['make_model']('S3',cfg).eval();rng = torch.get_rng_state()
    old['seed_everything'](75);control = old['make_model']('S3',oldcfg).eval()
    assert torch.equal(rng,torch.get_rng_state())
    assert sum(p.numel() for p in model.parameters())-sum(p.numel() for p in control.parameters()) == 8
    for key,value in control.state_dict().items():
        if key != 'scalar_mix.logits':torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
    bge = torch.randn(24,1024)
    engine = ns['Engine'](model,bundle.texts,bge,cfg,'cpu',contexts=ns['contexts_from_bundle'](bundle))
    oldengine = old['Engine'](control,bundle.texts,bge,oldcfg,'cpu',contexts=old['contexts_from_bundle'](bundle))
    # With only the last four weights nonzero, the wider mechanism recovers E43.
    model.scalar_mix.logits.data[:8] = -1000.
    p,features = engine.forward([0,1],return_features=True)
    previous,oldfeatures = oldengine.forward([0,1],return_features=True)
    torch.testing.assert_close(p,previous,rtol=0,atol=0)
    for key in ['concat','bigru_summary','answer_pool768','mix_norm_before']:
        torch.testing.assert_close(features[key],oldfeatures[key],rtol=0,atol=0)
    model.scalar_mix.logits.data.zero_()
    model.zero_grad(set_to_none=True)
    current,features = engine.forward([0,1],return_features=True)
    assert not torch.equal(current,previous)
    (current-.8).square().mean().backward()
    assert model.scalar_mix.logits.grad.shape == (12,) and torch.isfinite(model.scalar_mix.logits.grad).all()
    assert model.scalar_mix.logits.grad.abs().sum() > 0 and model.scalar_mix.gamma.grad.abs().sum() > 0
    for i,layer in enumerate(model.roberta.encoder.layer):
        assert sum(p.grad.abs().sum().item() for p in layer.parameters() if p.grad is not None)>0,i
    assert features['concat'].shape == (2,1024) and features['bigru_summary'].shape == (2,128)
    group = next(g for g in ns['optimizer_groups'](model,cfg) if g['name']=='scalar_mix')
    assert group['lr'] == 1e-4 and sum(p.numel() for p in group['params']) == 13


def test_all_layer_guard_and_checkpoint_metadata(tmp_path):
    ns = runtime(NOTEBOOK,'aes45_role');cfg,_ = setup(ns,tmp_path);ns['encoder'] = twelve_layer_encoder
    with pytest.raises(ValueError,match='seluruh layer Transformer'):
        ns['make_model']('S3',{**cfg,'scalar_mix_layers':4})
    model = ns['make_model']('S3',cfg)
    file = tmp_path/'TECHNICAL_all12.pt'
    torch.save({'scope':ns['SCOPE'],'trained':True,'scenario':'S3','config':cfg,'state':model.state_dict()},file)
    loaded,_ = ns['load_model'](file,cfg)
    assert loaded.scalar_mix.number_of_layers == loaded.roberta.config.num_hidden_layers == 12
    with pytest.raises(ValueError,match='scalar_mix_layers'):
        ns['load_model'](file,{**cfg,'scalar_mix_layers':4})


def test_full_native_twelve_layer_notebook_exports_oof_and_resume(tmp_path):
    execute_full_native_fixture(tmp_path,NOTEBOOK,'aes45_role',5e-6,encoder_layers=12)
    info = json.loads((tmp_path/'standalone/workspace/outputs/dataset300_download.json').read_text())
    run = Path(info['run_folder'])
    checkpoint = torch.load(run/'S3/final.pt',map_location='cpu',weights_only=True)
    assert checkpoint['config']['scalar_mix_layers'] == 12
    assert not any('ranking' in k for k in checkpoint['config'])
    history = run/'hasil_ringkas/05_Riwayat_Pelatihan.xlsx'
    weights = pd.read_excel(history,sheet_name='Bobot Scalar Mix')
    assert len(weights)==72 and set(weights['Layer ke-'])==set(range(1,13))
    assert set(weights['Indeks Layer RoBERTa'])==set(range(12)) and np.isfinite(weights.Gamma).all()
    np.testing.assert_allclose(weights['Bobot Awal'],1/12)
    for _,rows in weights.groupby(['Tahap','Fold'],dropna=False):
        assert len(rows)==12 and rows.Gamma.nunique()==1
        assert rows['Bobot Softmax'].sum()==pytest.approx(1.,abs=1e-6)
    final = weights[weights.Tahap=='Latihan final']
    np.testing.assert_allclose(final['Bobot Softmax'],torch.softmax(checkpoint['state']['scalar_mix.logits'],0),rtol=1e-6)
    train = pd.read_excel(history,sheet_name='Fusion (S3)')
    assert 'Loss Training (MSE)' in train and not any('ListNet' in c for c in train.columns)
    vectors = run/'hasil_ringkas/02_Vektor_Jawaban.xlsx'
    gamma = float(checkpoint['state']['scalar_mix.gamma'].item())
    before = pd.read_excel(vectors,sheet_name='Norma Campuran Sebelum')
    after = pd.read_excel(vectors,sheet_name='Norma Campuran Sesudah')
    np.testing.assert_allclose(after.iloc[:,1],abs(gamma)*before.iloc[:,1],rtol=1e-5)
    mean = pd.read_excel(vectors,sheet_name='Bobot Mean Pooling S3')
    for _,g in mean.groupby(['ID Jawaban','Bagian Teks']):np.testing.assert_allclose(g['Bobot Pooling'],1/len(g),rtol=1e-6)
    evaluation = run/'hasil_ringkas/04_Prediksi_dan_Evaluasi.xlsx'
    oof = pd.read_excel(evaluation,sheet_name='Prediksi Validasi OOF')
    test = pd.read_excel(evaluation,sheet_name='Nilai Guru dan Prediksi')
    assert oof['ID Jawaban'].is_unique and len(oof)+len(test)==24 and oof.Soal.notna().all()
    assert set(oof['ID Jawaban']).isdisjoint(test['ID Jawaban']) and set(oof.Fold)==set(range(1,6))
    np.testing.assert_allclose(oof['Selisih S3'],oof['Prediksi S3']-oof['Nilai Guru'])
    summary = pd.read_excel(history,sheet_name='MAE dan Toleransi CV')
    for f in range(1,6):
        values = oof[oof.Fold==f];errors=values['Selisih Absolut'].to_numpy()
        row = summary[summary.Fold==f].iloc[0]
        np.testing.assert_allclose(row.iloc[1:].astype(float),
            [len(errors),errors.mean(),(errors<=1).sum(),(errors<=5).sum(),(errors<=10).sum()])
        raw = pd.read_excel(run/f'excel/09_S3_fold_{f}.xlsx',sheet_name='Prediksi_validation')
        np.testing.assert_allclose(values['Prediksi S3'],raw.prediksi_mentah)
    assert Path(info['archive']).name=='Hasil_'+NOTEBOOK.stem+'.zip'
