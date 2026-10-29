"""Ablasi E25 dengan fixture teknis, bukan training penelitian."""
import ast,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tests'))
import test_context_pair_notebook as context
import test_e21_e22_notebooks as base
from test_e6_three_input_notebook import ThreeTechnicalTokenizer
from test_teacher_score import encoders,make_book
NAMES={'E31':'AES_300_E31_E25_GatedFusion_MLPLR5e6_GPU.ipynb','E32':'AES_300_E32_E25_CoAttention_GatedFusion_GPU.ipynb'}
BASE=ROOT/'notebooks/experiments/AES_300_E25_Tiga_Input_Attention_GatedFusion_GPU.ipynb'
def runtime(monkeypatch,e):
 monkeypatch.setattr(context,'NOTEBOOK',ROOT/'notebooks/experiments'/NAMES[e]);return context.runtime()

def test_pins_source_hashes_mirrors_and_unchanged_base():
 before=BASE.read_bytes();paths=[ROOT/'notebooks/experiments'/v for v in NAMES.values()];raw=[p.read_bytes() for p in paths]
 subprocess.run([sys.executable,'scripts/build_e31_e32_notebooks.py'],cwd=ROOT,check=True,capture_output=True)
 assert BASE.read_bytes()==before
 for p,b in zip(paths,raw):
  assert p.read_bytes()==b==(ROOT/'aes-learning/notebooks/experiments'/p.name).read_bytes();n=json.loads(b);fp=n['metadata']['aes_standalone']['code_sha256'];codes=[]
  assert n['metadata']['aes_experiment']['base_sha256']==hashlib.sha256(before).hexdigest()
  assert "device='cuda'" in n['cells'][12]['source'] and 'check_gpu(require_gpu=True)' in n['cells'][12]['source']
  for c in n['cells']:
   if c['cell_type']=='code':
    assert c['outputs']==[] and c['execution_count'] is None;s=c['source'].removeprefix('%%aes\n');ast.parse(s);assert not any(x in s for x in ['drive.mount','googleapiclient','from aes.']);codes.append(s.replace(fp,'__NOTEBOOK_CODE_SHA256__'))
  assert len(codes)==30 and hashlib.sha256('\n'.join(codes).encode()).hexdigest()==fp
 a=json.loads(raw[0]);b=json.loads(before)
 for i in [14,18,20,21,26,28,30,31,34,35,38,39,41,43]:assert a['cells'][i]['source']==b['cells'][i]['source']

@pytest.mark.parametrize('e',NAMES)
@pytest.mark.parametrize('scenario',['S1','S2','S3'])
def test_lr_gate_controls_gradients_reload(e,scenario,monkeypatch,tmp_path,encoders):
 ns=runtime(monkeypatch,e);ns['encoder']=encoders;ns['tokenizer']=lambda *a,**k:ThreeTechnicalTokenizer();path=tmp_path/'TECHNICAL_FIXTURE_ONLY.xlsx';make_book(path)
 cfg=ns['config']({'dataset':str(path),'output':str(tmp_path/'results'),'expected_rows':24,'device':'cpu','dropout':0.,'roberta_max_length':32,'roberta_overlap':4,'bge_max_length':32,'bge_overlap':4,'context_max_tokens':12,'question_max_tokens':4})
 bundle=ns['read_dataset'](cfg);bge=torch.randn(24,1024);torch.manual_seed(44);m=ns['make_model'](scenario,cfg).eval();engine=ns['Engine'](m,bundle.texts,bge,cfg,'cpu',contexts=ns['contexts_from_bundle'](bundle));pred,f=engine.forward([0,1],True)
 assert pred.shape==(2,1) and torch.isfinite(pred).all();(pred-.8).square().mean().backward();groups=ns['optimizer_groups'](m,cfg);byid={id(p):g['lr'] for g in groups for p in g['params']}
 assert len(byid)==sum(len(g['params']) for g in groups)==sum(p.requires_grad for p in m.parameters())
 for name,p in m.named_parameters():
  if p.requires_grad:assert byid[id(p)]==(5e-6 if name.startswith('roberta.') or (e=='E31' and scenario!='S1' and name.startswith('head.')) else 1e-4),name
 assert not any(isinstance(layer,torch.nn.GRU) for layer in m.modules())
 if scenario=='S3':
  assert f['concat'].shape==(2,1024);torch.testing.assert_close(f['fusion_gate'],torch.full((2,1),.5));assert m.fusion_gate.score.weight.grad.abs().sum()>0 and m.sequence_pool.score.weight.grad.abs().sum()>0
  if e=='E32':assert m.sequence_pool.reference_projection.weight.grad.abs().sum()>0 and m.sequence_pool.matching_projection.weight.grad.abs().sum()>0
 monkeypatch.setattr(context,'NOTEBOOK',BASE);old=context.runtime();torch.manual_seed(44);control=old['SingleScoreModel'](scenario,encoders(cfg,'roberta') if scenario!='S2' else None,dropout=0.).eval()
 assert all(torch.equal(m.state_dict()[k],v) for k,v in control.state_dict().items())
 if scenario!='S3' or e=='E31':
  oldengine=old['Engine'](control,bundle.texts,bge,cfg,'cpu',roberta_tokenizer=ThreeTechnicalTokenizer(),contexts=ns['contexts_from_bundle'](bundle));torch.testing.assert_close(pred,oldengine.forward([0,1]),rtol=0,atol=0)
 restored=ns['make_model'](scenario,cfg).eval();restored.load_state_dict(m.state_dict());second=ns['Engine'](restored,bundle.texts,bge,cfg,'cpu',contexts=ns['contexts_from_bundle'](bundle));torch.testing.assert_close(pred,second.forward([0,1]),rtol=0,atol=0)

def test_two_attention_stages_masks_context_and_amp(monkeypatch):
 ns=runtime(monkeypatch,'E32');pool=ns['AnswerCoAttentionPool']().eval();pool.capture_alignment=True;h=torch.randn(2,10,768,requires_grad=True)
 a=torch.tensor([[0,0,0,1,1,1,1,0,0,0],[0,0,0,0,1,1,1,1,1,0]],dtype=torch.bool);r=torch.tensor([[0,1,1,0,0,0,0,0,0,0],[0,1,1,1,0,0,0,0,0,0]],dtype=torch.bool)
 out=pool(h,a,r);alignment=pool.last_cross_attention.clone()
 for weights,mask in [(pool.last_weights,a),(pool.last_reference_weights,r),(pool.last_response_coattention_weights,a)]:
  torch.testing.assert_close(weights.sum(1),torch.ones(2));assert not weights[~mask].any()
 changed=h.detach().clone();changed[~(a|r)]=1e6;torch.testing.assert_close(out,pool(changed,a,r),rtol=0,atol=0)
 changed=h.detach().clone();changed[r]+=torch.randn_like(changed[r]);assert not torch.allclose(out,pool(changed,a,r))
 single=torch.cat([pool(row[None],am[None],rm[None]) for row,am,rm in zip(h,a,r)]);torch.testing.assert_close(out,single,atol=1e-6,rtol=1e-5)
 out.square().mean().backward();assert h.grad[r].abs().sum()>0 and h.grad[a].abs().sum()>0 and h.grad[~(a|r)].abs().sum()==0
 with torch.no_grad():pool.score.weight.add_(torch.randn_like(pool.score.weight)*5)
 after=pool(h,a,r);torch.testing.assert_close(alignment,pool.last_cross_attention,rtol=0,atol=0);assert not torch.allclose(out,after)
 with torch.autocast('cpu',dtype=torch.bfloat16):assert torch.isfinite(pool(h.detach(),a,r)).all()
 with pytest.raises(ValueError,match='terpisah'):pool(h,a,a)

@pytest.mark.parametrize('e',NAMES)
def test_full_notebook_excel_resume(e,tmp_path,monkeypatch):
 monkeypatch.setattr(base,'NAMES',{'single_sequence':NAMES[e]});monkeypatch.setattr(base,'GPU',True);base.test_full_notebook_thirty_cells_and_resume_same_local_folder('single_sequence',tmp_path)
 import pandas as pd
 pointer=json.loads((tmp_path/'standalone/workspace/outputs/dataset300_download.json').read_text());folder=Path(pointer['run_folder']);cfg=json.loads((folder/'effective_config.json').read_text());assert cfg['s3_fusion']=='gated_concat'
 settings=pd.read_excel(folder/'hasil_ringkas/05_Riwayat_Pelatihan.xlsx',sheet_name='Pengaturan Eksperimen');p=dict(zip(settings.Pengaturan,settings.Nilai));assert float(p['LR MLP S2'])==float(p['LR MLP S3'])==(5e-6 if e=='E31' else 1e-4)
 book=folder/'hasil_ringkas/02_Vektor_Jawaban.xlsx'
 for sheet in ['Gate Fusion S3','Pengali RoBERTa S3','Pengali BGE S3']:assert len(pd.read_excel(book,sheet_name=sheet))==24
 if e=='E32':
  for sheet in ['Bobot Attention S3','Bobot CoAttention Respons S3','Bobot Referensi S3']:
   w=pd.read_excel(book,sheet_name=sheet);np.testing.assert_allclose(w.groupby(['ID Jawaban','Bagian Teks'])['Bobot Attention'].sum(),1.,atol=1e-6)
  assert len(pd.read_excel(book,sheet_name='Pasangan Token Top1 S3'))>24
 # Test-only synthetic checkpoints, created by this test, are temporary artifacts.
 assert cfg['expected_rows']==24 and (tmp_path/'standalone/TECHNICAL_FIXTURE_ONLY.xlsx').exists()
 for checkpoint in folder.rglob('*.pt'):checkpoint.unlink()
