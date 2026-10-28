"""Uji tambahan S3 serta eksekusi notebook mandiri; fixture teknis, bukan penelitian."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
import test_context_pair_notebook as context
import test_e21_e22_notebooks as base
from test_e6_three_input_notebook import ThreeTechnicalTokenizer
from test_teacher_score import encoders,make_book
NAMES={'E25':'AES_300_E25_Tiga_Input_Attention_GatedFusion_GPU.ipynb','E26':'AES_300_E26_Tiga_Input_BiGRU_Attention_GPU.ipynb'}

@pytest.fixture(params=NAMES)
def runtime(request,monkeypatch,tmp_path):
    e=request.param;monkeypatch.setattr(context,'NOTEBOOK',ROOT/'notebooks/experiments'/NAMES[e]);ns=context.runtime()
    path=tmp_path/'TECHNICAL_FIXTURE_ONLY.xlsx';make_book(path)
    cfg=ns['config']({'dataset':str(path),'output':str(tmp_path/'results'),'expected_rows':24,'device':'cpu','epochs':1,'micro_batch':1,'effective_batch':4,'dropout':0.,'roberta_max_length':32,'roberta_overlap':4,'bge_max_length':32,'bge_overlap':4,'context_max_tokens':12,'question_max_tokens':4})
    ns['tokenizer']=lambda *a,**k:ThreeTechnicalTokenizer()
    return e,ns,cfg,ns['read_dataset'](cfg)


def test_standalone_sources_deterministic_gpu_and_no_base_change():
    baseline=(ROOT/'notebooks/experiments/AES_300_E24_Tiga_Input_AttentionPooling_GPU.ipynb').read_bytes()
    paths=[ROOT/'notebooks/experiments'/n for n in NAMES.values()];before=[p.read_bytes() for p in paths]
    subprocess.run([sys.executable,'scripts/build_e25_e26_notebooks.py'],cwd=ROOT,check=True,capture_output=True)
    assert baseline==(ROOT/'notebooks/experiments/AES_300_E24_Tiga_Input_AttentionPooling_GPU.ipynb').read_bytes()
    for path,raw in zip(paths,before):
        assert raw==path.read_bytes()==(ROOT/'aes-learning/notebooks/experiments'/path.name).read_bytes()
        n=json.loads(raw);meta=n['metadata']['aes_experiment'];assert meta['drive_mount'] is meta['drive_autosave'] is False
        assert meta['target']=='rater' and len(meta['input_columns'])==3 and meta['input_format']=='single_sequence'
        assert "device='cuda'" in n['cells'][12]['source'] and 'require_gpu=True' in n['cells'][12]['source']
        fingerprint=n['metadata']['aes_standalone']['code_sha256'];codes=[]
        for c in n['cells']:
            if c['cell_type']!='code':continue
            assert c['outputs']==[] and c['execution_count'] is None
            code=c['source'].removeprefix('%%aes\n');ast.parse(code);assert not any(x in code for x in ['drive.mount','googleapiclient','from aes.'])
            codes.append(code.replace(fingerprint,'__NOTEBOOK_CODE_SHA256__'))
        assert len(codes)==30 and hashlib.sha256('\n'.join(codes).encode()).hexdigest()==fingerprint

@pytest.mark.parametrize('scenario',['S1','S2','S3'])
def test_forward_backward_lr_freeze_controls_and_state_reload(runtime,encoders,scenario,tmp_path):
    e,ns,cfg,bundle=runtime;ns['encoder']=encoders;bge=ns['bge_features'](bundle,cfg,Path(cfg['output'])/'run');assert not bge.requires_grad
    torch.manual_seed(4);model=ns['make_model'](scenario,cfg).eval();engine=ns['Engine'](model,bundle.texts,bge,cfg,'cpu',contexts=ns['contexts_from_bundle'](bundle))
    pred,feat=engine.forward([0,1],True);assert pred.shape==(2,1) and torch.isfinite(pred).all();pred.square().mean().backward()
    groups=ns['optimizer_groups'](model,cfg);rates={g['name']:g['lr'] for g in groups};assert rates['head']==1e-4
    if scenario=='S3':
        assert rates['roberta']==5e-6 and rates['projection']==1e-4
        assert feat['concat'].shape==(2,1024) and feat['answer_pool768'].shape==(2,768)
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.roberta.parameters())
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.sequence_pool.parameters())
        new=model.fusion_gate if e=='E25' else model.sequence_pool
        head_ids={id(p) for g in groups if g['name']=='head' for p in g['params']};assert all(id(p) in head_ids for p in new.parameters())
        if e=='E25':
            torch.testing.assert_close(feat['fusion_gate'],torch.full((2,1),.5),rtol=0,atol=0)
            torch.testing.assert_close(feat['concat'],torch.cat([feat['roberta_layernorm'],feat['projection']],1),rtol=0,atol=0)
            assert model.fusion_gate.score.weight.grad.abs().sum()>0
        else:
            assert model.sequence_pool.gru.bidirectional and model.sequence_pool.attention_hidden.in_features==128
            assert feat['bigru_hidden128'].shape==(2,128)
            assert model.sequence_pool.attention_score.weight.grad.abs().sum()>0
    else:
        assert not hasattr(model,'fusion_gate') and not hasattr(model,'sequence_pool')
        old=json.loads(subprocess.check_output(['git','show','f3d25c4:notebooks/experiments/AES_300_E24_Tiga_Input_AttentionPooling_GPU.ipynb'],cwd=ROOT));tree=ast.parse(old['cells'][30]['source'].removeprefix('%%aes\n'));oldns=dict(ns);cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='SingleScoreModel');exec(compile(ast.Module(body=[cls],type_ignores=[]),'E24_control','exec'),oldns)
        torch.manual_seed(4);control=oldns['SingleScoreModel'](scenario,encoders(cfg,'roberta') if scenario=='S1' else None,dropout=0.).eval()
        assert all(torch.equal(v,control.state_dict()[k]) for k,v in model.state_dict().items())
        prev=ns['Engine'](control,bundle.texts,bge,cfg,'cpu',contexts=ns['contexts_from_bundle'](bundle));torch.testing.assert_close(pred,prev.forward([0,1]),rtol=0,atol=0)
    copy=ns['make_model'](scenario,cfg).eval();p=tmp_path/'technical_state.pt';torch.save(model.state_dict(),p);copy.load_state_dict(torch.load(p,weights_only=True));restored=ns['Engine'](copy,bundle.texts,bge,cfg,'cpu',contexts=ns['contexts_from_bundle'](bundle));torch.testing.assert_close(pred,restored.forward([0,1]),rtol=0,atol=0)


def test_pool_weights_mask_padding_gradients_and_token_sequence(runtime):
    e,ns,cfg,_=runtime;pool=(ns['AnswerAttentionPool'] if e=='E25' else ns['AnswerBiGRUAttention'])(64).eval();torch.manual_seed(21)
    h=torch.randn(2,9,768,requires_grad=True);mask=torch.tensor([[0,0,1,1,1,0,0,0,0],[0,1,1,1,1,1,1,0,0]],dtype=torch.bool)
    out=pool(h,mask);w=pool.last_weights.clone();assert (w[~mask]==0).all();torch.testing.assert_close(w.sum(1),torch.ones(2));alt=h.detach().clone();alt[~mask]=10000;torch.testing.assert_close(out,pool(alt,mask),rtol=0,atol=0)
    individual=torch.cat([pool(row[keep][None],torch.ones(1,int(keep.sum()),dtype=torch.bool)) for row,keep in zip(h,mask)]);torch.testing.assert_close(out,individual,rtol=1e-5,atol=1e-6)
    out.square().mean().backward();assert h.grad[~mask].abs().sum()==0 and h.grad[mask].abs().sum()>0
    if e=='E26':
        reverse=h.detach().clone();reverse[0,mask[0]]=reverse[0,mask[0]].flip(0);assert not torch.allclose(pool(h.detach(),mask)[0],pool(reverse,mask)[0])
    with pytest.raises(ValueError,match='minimal satu'):pool(h,torch.zeros_like(mask))


def test_gate_learns_sample_dependent_weights_without_changing_concat_dimension():
    sys.path.insert(0,str(ROOT/'scripts'));from s3_gated_bigru_attention_helpers import AnswerFusionGate
    torch.manual_seed(44);gate=AnswerFusionGate();r=torch.randn(4,768);b=torch.randn(4,256);opt=torch.optim.AdamW(gate.parameters(),lr=.01)
    for _ in range(3):
        opt.zero_grad();fused,g=gate(r,b);loss=(fused[:,:768]**2).mean();loss.backward();opt.step()
    _,g=gate(r,b);assert ((g>0)&(g<1)).all() and g.std()>0
    assert gate.hidden.weight.grad.abs().sum()>0

@pytest.mark.parametrize('experiment',NAMES)
def test_full_notebook_thirty_cells_fifteen_cv_three_refit_excel_and_resume(experiment,tmp_path,monkeypatch):
    monkeypatch.setattr(base,'NAMES',{'single_sequence':NAMES[experiment]});monkeypatch.setattr(base,'GPU',True)
    base.test_full_notebook_thirty_cells_and_resume_same_local_folder('single_sequence',tmp_path)
    import pandas as pd
    pointer=json.loads((tmp_path/'standalone/workspace/outputs/dataset300_download.json').read_text());folder=Path(pointer['run_folder']);book=folder/'hasil_ringkas/02_Vektor_Jawaban.xlsx'
    weights=pd.read_excel(book,sheet_name='Bobot Attention S3');assert weights['Bobot Attention'].ge(0).all()
    import numpy as np
    np.testing.assert_allclose(weights.groupby(['ID Jawaban','Bagian Teks'])['Bobot Attention'].sum(),1,atol=1e-6)
    if experiment=='E25':
        assert len(pd.read_excel(book,sheet_name='Gate Fusion S3').columns)==2
        assert len(pd.read_excel(book,sheet_name='Pengali BGE S3'))==24
    else:assert len(pd.read_excel(book,sheet_name='Ringkasan BiGRU Attention S3').columns)==129
