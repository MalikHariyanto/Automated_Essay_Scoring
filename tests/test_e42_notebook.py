"""E42 gamma math, E41 equivalence at gamma1, exports and native execution."""
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from test_e41_and_e30_native import runtime,setup,definitions
from test_e36_notebook import execute_full_native_fixture

NOTEBOOK=ROOT/'notebooks/experiments/AES_300_E42_E41_S3_ScalarMix4_Gamma_Native_Learnable_GPU.ipynb'
BASE=ROOT/'notebooks/experiments/AES_300_E41_E30_S3_ScalarMix4_Native_Learnable_GPU.ipynb'


def test_structure_protocol_and_deterministic_builder():
    raw=NOTEBOOK.read_bytes();old=BASE.read_bytes();n=nbformat.read(NOTEBOOK,as_version=4);nbformat.validate(n)
    m=n.metadata.aes_experiment
    assert m.scenarios==['S3'] and m.cv_fits==5 and m.refits==m.test_evaluations==1
    assert not m.multiple_seed_runs and not m.control_retrained
    assert m.scalar_mix_layers==4 and m.scalar_mix_gamma_init==1. and m.added_parameters_vs_e41==1
    assert m.s3_encoder_lr==5e-6 and m.scalar_mix_lr==1e-4 and m.s3_sequence_hidden==64
    fp=n.metadata.aes_standalone.code_sha256;code=[c for c in n.cells if c.cell_type=='code']
    assert hashlib.sha256('\n'.join(c.source.replace(fp,'__NOTEBOOK_CODE_SHA256__') for c in code).encode()).hexdigest()==fp
    assert hashlib.sha256(old).hexdigest()==m.format_sha256
    for c in code:
        assert not c.outputs and c.execution_count is None
        assert not any(line.lstrip().startswith(('%','!')) for line in c.source.splitlines())
        assert not any(isinstance(v,ast.Call) and isinstance(v.func,ast.Name) and v.func.id in ('exec','eval') for v in ast.walk(ast.parse(c.source)))
    text='\n'.join(c.source for c in code)
    assert all(x not in text for x in ['cosine','WORKER_SOURCE','TRAINING_SEEDS','encoder_layer_decay'])
    a,b=definitions(BASE),definitions(NOTEBOOK)
    for name in ['optimizer_groups','fit','train_all','make_split','metrics','gwet_ac2','Engine','encoder','bge_features',
                 'AnswerBiGRUAttention','AnswerBiGRUCoAttentionPool','pair_content_windows','pair_batch_windows']:
        # Only the version name in Engine's scenario error message changes.
        actual=ast.dump(b[name]).replace('Engine E42 hanya menjalankan S3.', 'Engine E41 hanya menjalankan S3.')
        assert ast.dump(a[name])==actual,name
    subprocess.run([sys.executable,'scripts/build_e42_notebook.py'],cwd=ROOT,check=True,capture_output=True)
    assert raw==NOTEBOOK.read_bytes()==(ROOT/'aes-learning/notebooks/experiments'/NOTEBOOK.name).read_bytes()
    assert old==BASE.read_bytes()


def test_gamma_scaling_gradients_and_masked_norms():
    ns=runtime(NOTEBOOK,'aes42_role');mix=ns['ScalarMix'](4)
    states=tuple(torch.full((1,3,2),float(i),requires_grad=True) for i in range(5))
    mix.capture_stats=True;mask=torch.tensor([[1,1,0]])
    y=mix(states,mask);torch.testing.assert_close(y,torch.full_like(y,2.5))
    assert sum(p.numel() for p in mix.parameters())==5
    y.sum().backward();assert mix.gamma.grad is not None and mix.gamma.grad.abs().sum()>0
    assert mix.logits.grad is not None and mix.logits.grad.abs().sum()>0
    assert mix.norm_before.shape==(1,1)
    # Padding is excluded even if its magnitude differs greatly.
    large=tuple(torch.cat([s.detach()[:,:2],torch.full((1,1,2),10000.)],dim=1) for s in states)
    original_norm=mix.norm_before.clone();mix.gamma.data.fill_(2.)
    result=mix(large,mask)
    torch.testing.assert_close(mix.norm_before,original_norm)
    torch.testing.assert_close(mix.norm_after,2*original_norm)
    torch.testing.assert_close(result[:,:2],2*y.detach()[:,:2])
    with pytest.raises(ValueError,match='gamma_init'):ns['config']({'scalar_mix_gamma_init':float('nan')})


def test_initial_predictions_and_frozen_gamma_update_identical_to_e41(tmp_path):
    ns=runtime(NOTEBOOK,'aes42_role');old=runtime(BASE,'aes41_role')
    cfg,bundle=setup(ns,tmp_path);oldcfg,_=setup(old,tmp_path)
    for key in old['DEFAULTS']:assert cfg[key]==oldcfg[key],key
    ns['seed_everything'](71);model=ns['make_model']('S3',cfg).eval()
    old['seed_everything'](71);control=old['make_model']('S3',oldcfg).eval()
    assert sum(p.numel() for p in model.parameters())-sum(p.numel() for p in control.parameters())==1
    for key,value in control.state_dict().items():torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
    model.scalar_mix.gamma.requires_grad_(False);bge=torch.randn(24,1024);preds=[]
    for env,current,settings in [(ns,model,cfg),(old,control,oldcfg)]:
        engine=env['Engine'](current,bundle.texts,bge,settings,'cpu',contexts=env['contexts_from_bundle'](bundle))
        pred=engine.forward([0,1]);preds.append(pred.detach());(pred-.8).square().mean().backward()
        torch.optim.AdamW(env['optimizer_groups'](current,settings),weight_decay=.01).step()
    torch.testing.assert_close(*preds,rtol=0,atol=0)
    for key,value in control.state_dict().items():torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
    model.scalar_mix.gamma.requires_grad_(True)
    groups=ns['optimizer_groups'](model,cfg)
    assert sum(p.numel() for g in groups if g['name']=='scalar_mix' for p in g['params'])==5
    assert next(g['lr'] for g in groups if g['name']=='scalar_mix')==1e-4
    engine=ns['Engine'](model,bundle.texts,bge,cfg,'cpu',contexts=ns['contexts_from_bundle'](bundle))
    model.zero_grad(set_to_none=True);pred,features=engine.forward([0,1],return_features=True);(pred-.8).square().mean().backward()
    assert model.scalar_mix.gamma.grad.abs().sum()>0
    assert all(features[k].shape==(2,1) for k in ['mix_norm_before','mix_norm_after','mix_norm_last'])
    path=tmp_path/'TECHNICAL_gamma.pt'
    torch.save({'scope':ns['SCOPE'],'trained':True,'scenario':'S3','config':cfg,'state':model.state_dict()},path)
    loaded,_=ns['load_model'](path,cfg);torch.testing.assert_close(loaded.scalar_mix.gamma,model.scalar_mix.gamma)
    with pytest.raises(ValueError,match='gamma_init'):ns['load_model'](path,{**cfg,'scalar_mix_gamma_init':2.})


def test_full_native_notebook_gamma_norm_exports_and_resume(tmp_path):
    execute_full_native_fixture(tmp_path,NOTEBOOK,'aes42_role',5e-6,encoder_layers=4)
    info=json.loads((tmp_path/'standalone/workspace/outputs/dataset300_download.json').read_text());run=Path(info['run_folder'])
    checkpoint=torch.load(run/'S3/final.pt',map_location='cpu',weights_only=True)
    gamma=float(checkpoint['state']['scalar_mix.gamma'].item());assert np.isfinite(gamma) and gamma!=1.
    assert checkpoint['config']['scalar_mix_gamma_init']==1.
    assert checkpoint['training']['optimizer_learning_rates']['scalar_mix']==1e-4
    weights=pd.read_excel(run/'hasil_ringkas/05_Riwayat_Pelatihan.xlsx',sheet_name='Bobot Scalar Mix')
    assert len(weights)==24 and np.isfinite(weights.Gamma).all()
    final=weights[weights.Tahap=='Latihan final'];assert np.allclose(final.Gamma,gamma)
    expected=torch.softmax(checkpoint['state']['scalar_mix.logits'],dim=0).numpy()
    np.testing.assert_allclose(final['Bobot Softmax'],expected,rtol=1e-6)
    before=pd.read_excel(run/'hasil_ringkas/02_Vektor_Jawaban.xlsx',sheet_name='Norma Campuran Sebelum')
    after=pd.read_excel(run/'hasil_ringkas/02_Vektor_Jawaban.xlsx',sheet_name='Norma Campuran Sesudah')
    last=pd.read_excel(run/'hasil_ringkas/02_Vektor_Jawaban.xlsx',sheet_name='Norma Layer Terakhir')
    assert before.shape==after.shape==last.shape==(24,2)
    np.testing.assert_allclose(after.iloc[:,1],abs(gamma)*before.iloc[:,1],rtol=1e-5)
    assert (last.iloc[:,1]>0).all()
    assert Path(info['archive']).name=='Hasil_'+NOTEBOOK.stem+'.zip'
