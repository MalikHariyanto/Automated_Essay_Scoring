"""E44 ListNet math, end-to-end/recompute gradients, interrupted resume and exports."""
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
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests'))
from test_e41_and_e30_native import runtime, setup, definitions
from test_e36_notebook import execute_full_native_fixture

NOTEBOOK = ROOT/'notebooks/experiments/AES_300_E44_E43_S3_MSE_ListNet_Native_Learnable_GPU.ipynb'
BASE = ROOT/'notebooks/experiments/AES_300_E43_E42_S3_MaskedMeanPooling_Native_Learnable_GPU.ipynb'


def test_native_structure_protocol_model_and_deterministic_builder():
    raw, old = NOTEBOOK.read_bytes(), BASE.read_bytes()
    n = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(n)
    m = n.metadata.aes_experiment
    assert m.scenarios == ['S3'] and m.cv_fits == 5 and m.refits == m.test_evaluations == 1
    assert not m.multiple_seed_runs and not m.control_retrained and not m.ensemble
    assert m.ranking_weight == .1 and m.ranking_recompute and m.model_unchanged_vs_e43
    assert m.added_parameters_vs_e43 == 0 and m.s3_encoder_lr == 5e-6 and m.scalar_mix_lr == 1e-4
    assert m.s3_pooling == 'bigru_coattention_mean_pooling'
    fp = n.metadata.aes_standalone.code_sha256
    code = [c for c in n.cells if c.cell_type == 'code']
    assert len(code) == 90
    assert hashlib.sha256('\n'.join(c.source.replace(fp,'__NOTEBOOK_CODE_SHA256__') for c in code).encode()).hexdigest() == fp
    assert hashlib.sha256(old).hexdigest() == m.format_sha256
    for c in code:
        assert c.metadata.aes44_role and not c.outputs and c.execution_count is None
        assert not any(line.lstrip().startswith(('%','!')) for line in c.source.splitlines())
        assert not any(isinstance(v,ast.Call) and isinstance(v.func,ast.Name) and v.func.id in ('exec','eval')
                       for v in ast.walk(ast.parse(c.source)))
    text = '\n'.join(c.source for c in code)
    assert all(x not in text for x in ['cosine','WORKER_SOURCE','TRAINING_SEEDS','encoder_layer_decay'])
    a, b = definitions(BASE), definitions(NOTEBOOK)
    for name in ['optimizer_groups','train_all','make_split','metrics','gwet_ac2','Engine','SingleScoreModel',
                 'encoder','bge_features','ScalarMix','AnswerBiGRUCoAttentionMeanPool',
                 'pair_content_windows','pair_batch_windows','export_training_event']:
        assert ast.dump(a[name]) == ast.dump(b[name]).replace('E44','E43'), name
    subprocess.run([sys.executable,'scripts/build_e44_notebook.py'],cwd=ROOT,check=True,capture_output=True)
    assert raw == NOTEBOOK.read_bytes() == (ROOT/'aes-learning/notebooks/experiments'/NOTEBOOK.name).read_bytes()
    assert old == BASE.read_bytes()


def test_listnet_analytic_gradients_grouping_ties_and_singletons():
    ns = runtime(NOTEBOOK,'aes44_role')
    p = torch.tensor([[.8],[.85],[.2],[.5],[.7]], requires_grad=True)
    y = torch.tensor([[.9],[.75],[.3],[.6],[.99]], requires_grad=True)
    q = ['A','A','B','B','C']
    loss, counts = ns['same_question_listnet'](p,y,q)
    expected = sum(-(torch.softmax(y.detach()[ids,0],0)*torch.log_softmax(p[ids,0],0)).sum()
                   for ids in [[0,1],[2,3]])/2
    torch.testing.assert_close(loss, expected)
    assert counts == {'ranking_groups':2,'ranking_answers':4}
    loss.backward()
    expected_gradient = torch.zeros_like(p)
    for ids in [[0,1],[2,3]]:
        expected_gradient[ids,0] = (torch.softmax(p.detach()[ids,0],0)-torch.softmax(y.detach()[ids,0],0))/2
    torch.testing.assert_close(p.grad,expected_gradient)
    assert p.grad[0] < 0 and p.grad[1] > 0  # Gradient descent corrects reversed ordering.
    assert p.grad[4] == 0 and y.grad is None
    shifted = p.detach().clone();shifted[:2] += 100;shifted[2:4] -= 50
    torch.testing.assert_close(ns['same_question_listnet'](shifted,y,q)[0],loss.detach(),atol=1e-6,rtol=1e-5)
    permutation = [4,2,0,3,1]
    torch.testing.assert_close(ns['same_question_listnet'](p[permutation],y[permutation],[q[i] for i in permutation])[0],loss)
    singleton = p.detach().clone().requires_grad_()
    zero, count = ns['same_question_listnet'](singleton,y,['a','b','c','d','e'])
    assert zero.item() == 0 and count['ranking_answers'] == 0
    zero.backward();assert torch.equal(singleton.grad,torch.zeros_like(singleton))
    tied = torch.ones(2,1,requires_grad=True)
    entropy,_ = ns['same_question_listnet'](tied,torch.full_like(tied,.8),['a','a'])
    assert entropy.item() == pytest.approx(np.log(2))
    entropy.backward();torch.testing.assert_close(tied.grad,torch.zeros_like(tied))
    with pytest.raises(ValueError,match='finite'):
        ns['same_question_listnet'](torch.full((2,1),float('nan')),torch.ones(2,1),['a','a'])


def test_recompute_preserves_dropout_predictions_gradients_and_rng(tmp_path,monkeypatch):
    ns = runtime(NOTEBOOK,'aes44_role')
    cfg,bundle = setup(ns,tmp_path)
    assert cfg['epochs'] == 25 and cfg['folds'] == 5 and cfg['early_stopping_patience'] == 5
    for i,q in enumerate(['a','b','a','c','b','d','e','f']):
        bundle.rows[i]['question'] = 'TECHNICAL question '+q
    ns['seed_everything'](77)
    model = ns['make_model']('S3',cfg).train()
    bge = torch.randn(24,1024)
    engine = ns['Engine'](model,bundle.texts,bge,cfg,'cpu',contexts=ns['contexts_from_bundle'](bundle))
    original_checkpoint = ns['activation_checkpoint']
    markers = []
    def inspect_marker(function,marker,*args,**kwargs):
        assert isinstance(marker,torch.Tensor) and marker.numel() == 0
        assert marker.device == next(model.parameters()).device
        assert kwargs['preserve_rng_state'] and not kwargs['use_reentrant']
        markers.append(marker.device)
        return original_checkpoint(function,marker,*args,**kwargs)
    monkeypatch.setitem(ns,'activation_checkpoint',inspect_marker)
    results=[]
    for recompute in [False,True]:
        ns['seed_everything'](176)
        model.zero_grad(set_to_none=True)
        total,mse,rank,counts = ns['training_batch_loss'](engine,list(range(8)),bundle,
            {**cfg,'ranking_recompute':recompute},'cpu',torch.float32,False)
        assert counts == {'ranking_groups':2,'ranking_answers':4}
        torch.testing.assert_close(total,mse+.1*rank)
        total.backward()
        results.append(([x.detach().clone() for x in [total,mse,rank]],
                        {n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None},
                        torch.get_rng_state().clone()))
    for a,b in zip(results[0][0],results[1][0]):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert results[0][1].keys() == results[1][1].keys()
    for name,gradient in results[0][1].items():
        torch.testing.assert_close(gradient,results[1][1][name],rtol=1e-5,atol=1e-7,msg=name)
    assert torch.equal(results[0][2],results[1][2])
    assert len(markers) == 8
    for prefix in ['roberta.','sequence_pool.gru.','sequence_pool.reference_projection.',
                   'sequence_pool.matching_projection.','scalar_mix.','projection.','head.']:
        assert sum(g.abs().sum().item() for n,g in results[1][1].items() if n.startswith(prefix)) > 0, prefix


def test_zero_ranking_matches_e43_and_recipe_guards(tmp_path):
    ns,old = runtime(NOTEBOOK,'aes44_role'),runtime(BASE,'aes43_role')
    cfg,bundle = setup(ns,tmp_path);oldcfg,_ = setup(old,tmp_path)
    for k in old['DEFAULTS']:assert cfg[k] == oldcfg[k],k
    ns['seed_everything'](72);model = ns['make_model']('S3',cfg).train()
    old['seed_everything'](72);control = old['make_model']('S3',oldcfg).train()
    for k,v in control.state_dict().items():torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
    bge = torch.randn(24,1024);batch = list(range(4))
    engine = ns['Engine'](model,bundle.texts,bge,cfg,'cpu',contexts=ns['contexts_from_bundle'](bundle))
    oldengine = old['Engine'](control,bundle.texts,bge,oldcfg,'cpu',contexts=old['contexts_from_bundle'](bundle))
    ns['seed_everything'](83)
    total,mse,_,_ = ns['training_batch_loss'](engine,batch,bundle,{**cfg,'ranking_weight':0.},'cpu',torch.float32,False)
    total.backward()
    old['seed_everything'](83);losses=[]
    for i in batch:
        p = oldengine.forward([i]);target = torch.tensor([[bundle.y[i]/100]],dtype=torch.float32)
        loss = (p-target).square().mean();losses.append(loss.detach());(loss/len(batch)).backward()
    torch.testing.assert_close(mse,torch.stack(losses).mean())
    for name,p in control.named_parameters():
        if p.grad is not None:torch.testing.assert_close(p.grad,dict(model.named_parameters())[name].grad,rtol=1e-4,atol=1e-6,msg=name)
    file=tmp_path/'TECHNICAL_listnet.pt'
    torch.save({'scope':ns['SCOPE'],'trained':True,'scenario':'S3','config':cfg,'state':model.state_dict()},file)
    ns['load_model'](file,cfg)
    for change,key in [({'ranking_weight':.2},'ranking_weight'),({'loss_recipe':'mse'},'loss_recipe')]:
        with pytest.raises(ValueError,match=key):ns['load_model'](file,{**cfg,**change})
    for bad in [-1,float('nan'),float('inf'),True]:
        with pytest.raises(ValueError,match='ranking_weight'):ns['config']({'ranking_weight':bad})


def test_interrupted_epoch_resume_matches_uninterrupted(tmp_path):
    ns = runtime(NOTEBOOK,'aes44_role');cfg,bundle = setup(ns,tmp_path)
    cfg = {**cfg,'epochs':2,'effective_batch':4}
    bge = torch.randn(24,1024);training=list(range(8));validation=[8,9]
    original = ns['save_training_progress'];path = tmp_path/'progress.json'
    def stop_after_save(file,state):
        original(file,state)
        if state['epoch'] == 1:raise RuntimeError('TECHNICAL simulated interruption')
    ns['save_training_progress'] = stop_after_save
    with pytest.raises(RuntimeError,match='simulated interruption'):
        ns['fit']('S3',training,validation,bundle,bge,cfg,2,84,checkpoint=path)
    ns['save_training_progress'] = original
    resumed,_,resumed_info = ns['fit']('S3',training,validation,bundle,bge,cfg,2,84,checkpoint=path)
    full,_,full_info = ns['fit']('S3',training,validation,bundle,bge,cfg,2,84)
    assert resumed_info['history'] == full_info['history']
    assert resumed_info['best_epoch'] == full_info['best_epoch']
    for k,v in full.state_dict().items():torch.testing.assert_close(v,resumed.state_dict()[k],rtol=0,atol=0)
    assert all(r['ranking_answers'] == len(training) for r in resumed_info['history'])


def test_full_native_notebook_losses_oof_mean_exports_and_resume(tmp_path,monkeypatch):
    import test_e36_notebook as fixture_module
    original_book = fixture_module.make_book
    def three_question_book(path):
        original_book(path)
        book = load_workbook(path)
        columns = {c.value:c.column for c in book.active[1] if c.value is not None}
        for i in range(24):
            book.active.cell(i+2,columns['soal'],f'fixture question group{i%3}')
            book.active.cell(i+2,columns['jawaban_referensi'],f'fixture reference group{i%3}')
        book.save(path);book.close()
    monkeypatch.setattr(fixture_module,'make_book',three_question_book)
    execute_full_native_fixture(tmp_path,NOTEBOOK,'aes44_role',5e-6,encoder_layers=4)
    info = json.loads((tmp_path/'standalone/workspace/outputs/dataset300_download.json').read_text())
    run = Path(info['run_folder'])
    checkpoint = torch.load(run/'S3/final.pt',map_location='cpu',weights_only=True)
    assert checkpoint['config']['ranking_weight'] == .1 and checkpoint['config']['ranking_recompute']
    assert checkpoint['training']['loss_recipe'] == 'mse_same_question_listnet_v1'
    h = pd.read_excel(run/'hasil_ringkas/05_Riwayat_Pelatihan.xlsx',sheet_name='Fusion (S3)')
    assert len(h) == 6 and np.isfinite(h['Loss Ranking (ListNet)']).all()
    np.testing.assert_allclose(h['Loss Gabungan'],h['Loss Training (MSE)']+.1*h['Loss Ranking (ListNet)'],rtol=1e-6)
    assert h['Cakupan Ranking (%)'].between(0,100,inclusive='right').all()
    assert (h['Kelompok Soal untuk Ranking']>0).all()
    evaluation = run/'hasil_ringkas/04_Prediksi_dan_Evaluasi.xlsx'
    oof = pd.read_excel(evaluation,sheet_name='Prediksi Validasi OOF')
    test = pd.read_excel(evaluation,sheet_name='Nilai Guru dan Prediksi')
    assert len(oof) + len(test) == 24 and oof['ID Jawaban'].is_unique
    assert set(oof['ID Jawaban']).isdisjoint(test['ID Jawaban'])
    assert oof.Soal.notna().all() and set(oof.Fold) == set(range(1,6))
    np.testing.assert_allclose(oof['Selisih S3'],oof['Prediksi S3']-oof['Nilai Guru'])
    np.testing.assert_allclose(oof['Selisih Kuadrat'],oof['Selisih S3']**2)
    summary = pd.read_excel(run/'hasil_ringkas/05_Riwayat_Pelatihan.xlsx',sheet_name='MAE dan Toleransi CV')
    for f in range(1,6):
        values = oof[oof.Fold==f]
        error = values['Selisih Absolut'].to_numpy()
        expected = summary[summary.Fold==f].iloc[0]
        np.testing.assert_allclose(expected.iloc[1:].astype(float),
            [len(error),error.mean(),(error<=1).sum(),(error<=5).sum(),(error<=10).sum()])
        raw = pd.read_excel(run/f'excel/09_S3_fold_{f}.xlsx',sheet_name='Prediksi_validation')
        np.testing.assert_allclose(values['Prediksi S3'],raw.prediksi_mentah)
    vectors = run/'hasil_ringkas/02_Vektor_Jawaban.xlsx'
    weights = pd.read_excel(vectors,sheet_name='Bobot Mean Pooling S3')
    for _,group in weights.groupby(['ID Jawaban','Bagian Teks']):
        np.testing.assert_allclose(group['Bobot Pooling'],1/len(group),rtol=1e-6)
    assert Path(info['archive']).name == 'Hasil_'+NOTEBOOK.stem+'.zip'
