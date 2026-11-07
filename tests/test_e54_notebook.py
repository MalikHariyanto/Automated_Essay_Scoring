"""E54 numerical ConR, real graph/recompute/resume and standalone notebook exports."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import nbformat
import numpy as np
from openpyxl import load_workbook
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests'))
from test_e41_and_e30_native import runtime, setup, definitions
from test_e36_notebook import execute_full_native_fixture

NAME = 'AES_300_E54_E43_S3_ConR_Native_Learnable_GPU'
NOTEBOOK = ROOT/'notebooks/experiments'/(NAME+'.ipynb')
BASE = ROOT/'notebooks/experiments/AES_300_E43_E42_S3_MaskedMeanPooling_Native_Learnable_GPU.ipynb'


def env():
    return runtime(NOTEBOOK, 'aes54_role')


def test_structure_rebuild_original_architecture_and_native_protocol():
    raw, parent = NOTEBOOK.read_bytes(), BASE.read_bytes()
    n = nbformat.read(NOTEBOOK, as_version=4); nbformat.validate(n)
    m = n.metadata.aes_experiment
    assert m.scenarios == ['S3'] and m.cv_fits == 5 and m.refits == m.test_evaluations == 1
    assert m.added_parameters_vs_e43 == 0 and m.conr_weight == .001 and m.conr_window_points == 5
    assert m.conr_temperature == .07 and m.conr_eta_per_point == .01 and m.conr_recompute
    code = [c for c in n.cells if c.cell_type == 'code']; fp = n.metadata.aes_standalone.code_sha256
    assert hashlib.sha256('\n'.join(c.source.replace(fp,'__NOTEBOOK_CODE_SHA256__') for c in code).encode()).hexdigest() == fp
    for c in code:
        assert c.metadata.aes54_role and not c.outputs and c.execution_count is None
        assert not any(line.lstrip().startswith(('%','!')) for line in c.source.splitlines())
        assert not any(isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id in ['exec','eval'] for v in ast.walk(ast.parse(c.source)))
    text = '\n'.join(c.source for c in code)
    for bad in ['ExponentialWeightAverage','from peft','multihead_report','ListNet','WORKER_SOURCE','TRAINING_SEEDS']:
        assert bad not in text
    a,b = definitions(BASE),definitions(NOTEBOOK)
    for name in ['optimizer_groups','train_all','make_split','metrics','gwet_ac2','encoder','bge_features',
                 'ScalarMix','AnswerBiGRUCoAttentionMeanPool','make_model','save_model']:
        assert ast.dump(a[name]) == ast.dump(b[name]).replace('E54','E43'), name
    subprocess.run([sys.executable,'scripts/build_e54_notebook.py'],cwd=ROOT,check=True,capture_output=True)
    assert NOTEBOOK.read_bytes() == raw == (ROOT/'aes-learning/notebooks/experiments'/NOTEBOOK.name).read_bytes()
    assert BASE.read_bytes() == parent


def test_conr_matches_source_equation_with_well_defined_anchors_and_gradients():
    ns=env(); cfg=ns['config']({})
    # Every anchor has a near neighbour and a far label, but predictions collapse.
    z=torch.tensor([[1.,.1],[.8,.6],[-.6,.8],[-1.,.1]],requires_grad=True)
    y=torch.tensor([[.65],[.67],[.85],[.87]],requires_grad=True)
    p=torch.full((4,1),.75,requires_grad=True)
    loss,c=ns['same_question_conr'](z,y,p,['A']*4,cfg)
    q=torch.nn.functional.normalize(z,dim=1);delta=(y.detach()*100-(y.detach()*100).T).abs()
    pos=(delta<=5)&~torch.eye(4,dtype=torch.bool);neg=delta>5
    logits=q@q.T/.07
    # Independent direct exponential expression from official ConR source.
    den=(torch.exp(logits*pos).sum(1)+(torch.exp(logits*neg)*neg*torch.exp(delta*.01)).sum(1))
    reference=((-torch.log(torch.exp(logits*pos)/den[:,None])*pos).sum(1)/pos.sum(1)).mean()
    torch.testing.assert_close(loss,reference,rtol=1e-6,atol=1e-6)
    expected_grad=torch.autograd.grad(reference,z,retain_graph=True)[0]
    loss.backward();torch.testing.assert_close(z.grad,expected_grad,rtol=1e-5,atol=2e-6)
    assert z.grad.abs().sum()>0 and y.grad is None and p.grad is None
    assert c['conr_active_anchors']==4 and c['conr_positive_pairs']==4 and c['conr_hard_negative_pairs']==8
    order=[2,0,3,1]
    torch.testing.assert_close(ns['same_question_conr'](z[order],y[order],p[order],['A']*4,cfg)[0],loss)


def test_empty_anchors_singleton_cross_question_ties_and_numerical_stability():
    ns=env();cfg=ns['config']({})
    for labels,pred,q in [([.1,.3,.5,.7],[.5]*4,['A']*4), # no positives (original NaN case)
                          ([.7]*4,[.7]*4,['A']*4), # no negatives
                          ([.65,.67,.85,.87],[.65,.67,.85,.87],['A']*4), # no hard negatives
                          ([.65,.67,.85,.87],[.75]*4,['A','B','C','D']),
                          ([.7],[.7],['A'])]:
        z=torch.randn(len(labels),8,requires_grad=True)
        loss,c=ns['same_question_conr'](z,torch.tensor(labels)[:,None],torch.tensor(pred)[:,None],q,cfg)
        assert torch.isfinite(loss) and loss.item()==0 and c['conr_active_anchors']==0
        loss.backward();assert torch.equal(z.grad,torch.zeros_like(z))
    z=torch.randn(4,8,requires_grad=True);y=torch.tensor([[.65],[.67],[.85],[.87]]);p=torch.full_like(y,.75)
    loss,_=ns['same_question_conr'](z,y,p,['A']*4,{**cfg,'conr_temperature':1e-6})
    loss.backward();assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
    # Cross-question similarities cannot create false positives/negatives.
    loss,c=ns['same_question_conr'](z,y,p,['A','A','B','B'],cfg)
    assert loss.item()==0 and c['conr_hard_negative_pairs']==0
    # Inclusive 5-point boundary survives normalization/reconstruction in FP32.
    scores=torch.arange(101,dtype=torch.float32)[:,None]/100
    _,count=ns['same_question_conr'](torch.randn(101,8),scores,scores,['A']*101,cfg)
    assert count['conr_positive_pairs']==980 and count['conr_hard_negative_pairs']==0
    for changes in [{'conr_temperature':0},{'conr_weight':True},{'conr_weight':float('nan')},
                    {'conr_window_points':100},{'conr_eta_per_point':-1},{'conr_recompute':1}]:
        with pytest.raises(ValueError):ns['config'](changes)


def test_recompute_matches_predictions_gradients_rng_and_keeps_model_parameters(tmp_path):
    ns=env();cfg,bundle=setup(ns,tmp_path);ns['seed_everything'](71)
    model=ns['make_model']('S3',cfg).train();bge=torch.randn(24,1024)
    old=runtime(BASE,'aes43_role');oldcfg,_=setup(old,tmp_path);old['seed_everything'](71)
    control=old['make_model']('S3',oldcfg)
    for k,v in model.state_dict().items():torch.testing.assert_close(v,control.state_dict()[k],rtol=0,atol=0)
    engine=ns['Engine'](model,bundle.texts,bge,cfg,'cpu',contexts=ns['contexts_from_bundle'](bundle))
    # Mixed label distances in one original batch; no teacher enters forward.
    batch=[0,1,2,3,12,13,14,15];results=[]
    for recompute in [False,True]:
        ns['seed_everything'](77);model.zero_grad(set_to_none=True)
        total,mse,conr,counts=ns['training_batch_loss'](engine,batch,bundle,{**cfg,'conr_recompute':recompute},'cpu',torch.float32,False)
        torch.testing.assert_close(total,mse+.001*conr)
        assert counts['conr_active_anchors']>0
        total.backward()
        results.append(([v.detach().clone() for v in [total,mse,conr]],{n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None},torch.get_rng_state()))
    for a,b in zip(results[0][0],results[1][0]):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert torch.equal(results[0][2],results[1][2])
    for k,g in results[0][1].items():torch.testing.assert_close(g,results[1][1][k],rtol=1e-5,atol=1e-7,msg=k)
    for prefix in ['roberta.','sequence_pool.gru.','sequence_pool.reference_projection.',
                   'sequence_pool.matching_projection.','scalar_mix.','projection.','head.']:
        assert sum(g.abs().sum().item() for k,g in results[1][1].items() if k.startswith(prefix))>0,prefix
    # ConR alone reaches representations/encoder but not the discrete prediction gate/head.
    model.zero_grad(set_to_none=True);ns['seed_everything'](77)
    _,_,conr,_=ns['training_batch_loss'](engine,batch,bundle,cfg,'cpu',torch.float32,False);conr.backward()
    assert model.roberta.embeddings.word_embeddings.weight.grad.abs().sum()>0
    assert all(p.grad is None or p.grad.count_nonzero()==0 for p in model.head.parameters())


def test_zero_weight_matches_e43_and_checkpoint_recipe_guards(tmp_path):
    ns=env();old=runtime(BASE,'aes43_role');cfg,bundle=setup(ns,tmp_path);oldcfg,_=setup(old,tmp_path)
    ns['seed_everything'](19);model=ns['make_model']('S3',cfg).train()
    old['seed_everything'](19);control=old['make_model']('S3',oldcfg).train()
    bge=torch.randn(24,1024);ids=[0,1,12,13]
    engine=ns['Engine'](model,bundle.texts,bge,cfg,'cpu',contexts=ns['contexts_from_bundle'](bundle))
    other=old['Engine'](control,bundle.texts,bge,oldcfg,'cpu',contexts=old['contexts_from_bundle'](bundle))
    ns['seed_everything'](101)
    total,*_=ns['training_batch_loss'](engine,ids,bundle,{**cfg,'conr_weight':0.},'cpu',torch.float32,False);total.backward()
    old['seed_everything'](101)
    for i in ids:
        pred=other.forward([i]);((pred-torch.tensor([[bundle.y[i]/100]],dtype=torch.float32)).square().mean()/len(ids)).backward()
    for k,p in control.named_parameters():
        if p.grad is not None:torch.testing.assert_close(p.grad,dict(model.named_parameters())[k].grad,rtol=1e-4,atol=1e-6,msg=k)
    file=tmp_path/'TECHNICAL_conr.pt'
    torch.save({'scope':ns['SCOPE'],'trained':True,'scenario':'S3','config':cfg,'state':model.state_dict()},file)
    ns['load_model'](file,cfg)
    for key,value in [('loss_recipe','mse'),('conr_weight',.1),('conr_window_points',3),('conr_temperature',.2),('conr_eta_per_point',0),('conr_recompute',False)]:
        with pytest.raises(ValueError,match=key):ns['load_model'](file,{**cfg,key:value})


def test_preflight_training_only_and_inactive_fit_is_rejected(tmp_path):
    ns=env();cfg,bundle=setup(ns,tmp_path);manifest=ns['make_split'](bundle,cfg)
    result=ns['conr_preflight'](bundle,manifest,cfg)
    assert len(result)==6 and all(r['Anchor Potensial']>0 for r in result)
    before=[dict(r) for r in result]
    # Deliberately mutate test labels: preflight must never consult them.
    bundle.y[manifest['test']]=-10000
    assert ns['conr_preflight'](bundle,manifest,cfg)==before
    for i in manifest['development']:bundle.rows[i]['question']='TECHNICAL unique '+str(i)
    with pytest.raises(ValueError,match='anchor'):ns['conr_preflight'](bundle,manifest,cfg)
    bge=torch.randn(24,1024)
    with pytest.raises(RuntimeError,match='tidak pernah aktif'):
        ns['fit']('S3',manifest['development'][:8],None,bundle,bge,cfg,1,42)


def test_resume_restores_joint_loss_model_and_rng(tmp_path):
    ns=env();cfg,bundle=setup(ns,tmp_path);bge=torch.randn(24,1024)
    save=ns['save_training_progress'];file=tmp_path/'progress.json'
    def interrupt(path,state):
        save(path,state)
        if state['epoch']==1:raise RuntimeError('TECHNICAL interrupt')
    ns['save_training_progress']=interrupt
    with pytest.raises(RuntimeError,match='TECHNICAL interrupt'):
        ns['fit']('S3',list(range(16)),[16,17],bundle,bge,cfg,2,42,checkpoint=file)
    ns['save_training_progress']=save
    resumed,_,a=ns['fit']('S3',list(range(16)),[16,17],bundle,bge,cfg,2,42,checkpoint=file)
    full,_,b=ns['fit']('S3',list(range(16)),[16,17],bundle,bge,cfg,2,42)
    assert a['history']==b['history'] and a['conr_config']==b['conr_config']
    for k,v in full.state_dict().items():torch.testing.assert_close(v,resumed.state_dict()[k],rtol=0,atol=0)
    with pytest.raises(ValueError):ns['fit']('S3',list(range(16)),[16,17],bundle,bge,{**cfg,'conr_weight':.02},2,42,checkpoint=file)


def test_standalone_notebook_twice_oof_loss_coverage_and_tampered_export_gate(tmp_path):
    ns=env()
    cols=list(ns['prediction_report_tables']([{'ID Jawaban':'TECHNICAL','Nilai Guru':80,'Prediksi S3':79.5}])['Nilai Guru dan Prediksi'][0])
    execute_full_native_fixture(tmp_path,NOTEBOOK,'aes54_role',5e-6,encoder_layers=4,prediction_columns=cols)
    info=json.loads((tmp_path/'standalone/workspace/outputs/dataset300_download.json').read_text());run=Path(info['run_folder'])
    cfg=json.loads((run/'effective_config.json').read_text());bundle=ns['read_dataset'](cfg);manifest=ns['make_split'](bundle,cfg);folder=run/'hasil_ringkas'
    proof=ns['validate_e54_download'](folder,bundle,manifest,cfg)
    assert proof['test_rows']==6 and proof['oof_rows']==18 and proof['conr_fits']==6 and proof['conr_epoch_rows']==6
    diagnostics=pd.read_excel(folder/'05_Riwayat_Pelatihan.xlsx',sheet_name='Diagnostik ConR')
    assert (diagnostics['Anchor Aktif']>0).all() and diagnostics['Cakupan Aktif (%)'].between(0,100).all()
    np.testing.assert_allclose(diagnostics['Loss Gabungan'],diagnostics['MSE']+.001*diagnostics['ConR'])
    f=folder/'05_Riwayat_Pelatihan.xlsx';original=f.read_bytes()
    for column,value in [('Loss Gabungan',999.),('Anchor Aktif',0),('Bobot ConR',.5),('Cakupan Aktif (%)',999.)]:
        w=load_workbook(f);s=w['Diagnostik ConR'];col=next(c.column for c in s[1] if c.value==column);s.cell(2,col,value);w.save(f);w.close()
        with pytest.raises(RuntimeError,match='ConR'):ns['validate_e54_download'](folder,bundle,manifest,cfg)
        f.write_bytes(original)
    f=folder/'04_Prediksi_dan_Evaluasi.xlsx';w=load_workbook(f);s=w['Ringkasan Validasi'];s.delete_rows(2,s.max_row-1);w.save(f);w.close()
    with pytest.raises(RuntimeError,match='Unduhan dihentikan'):ns['validate_e54_download'](folder,bundle,manifest,cfg)
