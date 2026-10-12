"""Latihan fixture teknis, tidak pernah memakai target CSV draf."""
import copy,json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from aes.io import load_config,load_rubric
from aes.trainer import train_run
from pathlib import Path

class TinyEncoder(nn.Module):
    def __init__(self):super().__init__();self.embedding=nn.Embedding(20,768)
    def forward(self,input_ids,**kw):
        x=self.embedding(input_ids);return SimpleNamespace(last_hidden_state=x+x.mean(1,keepdim=True))
class TinyTokenizer:
    def num_special_tokens_to_add(self,pair=False):return 2
    def encode(self,text,**kw):return [3,4,5]
    def prepare_for_model(self,ids,**kw):return {'input_ids':[0]+ids+[2],'attention_mask':[1]*(len(ids)+2)}
    def pad(self,chunks,**kw):return {k:torch.tensor([c[k] for c in chunks]) for k in chunks[0]}

@pytest.mark.parametrize('scenario',['S1','S2','S3'])
def test_trainer_effective_batch_and_fallback(scenario,tmp_path,monkeypatch,technical_source):
    import aes.trainer as trainer
    monkeypatch.setattr(trainer,'load_roberta',lambda _:TinyEncoder())
    monkeypatch.setattr(trainer.AutoTokenizer,'from_pretrained',lambda *a,**k:TinyTokenizer())
    _,r,cfg=technical_source
    cfg['training'].update(max_epochs=2,micro_batch=2,patience=2)
    texts=['fixture']*12
    y=np.full((12,4),3.) # Constant teacher fixture triggers undefined Pearson, sometimes QWK.
    # Force undefined QWK fixture to verify deterministic fallback, no imputation.
    original=trainer.evaluate
    def undefined_qwk(*a,**kw):
        result=original(*a,**kw)
        for trait in result.values():trait['QWK']=None;trait['undefined_reasons']['QWK']='technical_fixture'
        return result
    monkeypatch.setattr(trainer,'evaluate',undefined_qwk)
    bge=torch.nn.functional.normalize(torch.randn(12,1024),dim=1)
    resolved={'bge':{'checkpoint':'BAAI/bge-m3','revision':'fixture'},
              'roberta':{'checkpoint':'flax-community/indonesian-roberta-base','revision':'fixture'}}
    result=train_run(scenario,'A',list(range(10)),[10,11],texts,y,bge,resolved,cfg,r,42,tmp_path)
    history=json.loads((tmp_path/'history.json').read_text())
    assert history[0]['optimizer_steps']==2 and history[1]['optimizer_steps']==4
    assert all(v['monitor']=='validation_mse' for v in history)
    assert 'fallback_event' in history[0] and result['best_epoch'] in [1,2]
    # Checkpoint lives only in pytest tmp dir; training metadata from fixture is not research.
    assert (tmp_path/'best.pt').exists()
