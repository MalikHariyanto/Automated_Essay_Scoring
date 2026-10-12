"""Checkpoint asli saja; resolusi SHA disimpan sebelum encode/train."""
from pathlib import Path
import json
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel
from huggingface_hub import model_info, snapshot_download
from .io import dump, text_hash
from .cleaning import POLICY


def resolve_models(cfg):
    resolved={}
    for role,spec in cfg['models'].items():
        revision=spec['revision'] or model_info(spec['checkpoint']).sha
        config=AutoConfig.from_pretrained(spec['checkpoint'], revision=revision)
        if config.hidden_size != spec['dimension']:
            raise ValueError(f'{role} hidden_size aktual {config.hidden_size} berbeda kontrak')
        tok=AutoTokenizer.from_pretrained(spec['checkpoint'],revision=revision,use_fast=(role!='roberta'))
        if tok.num_special_tokens_to_add(False)!=len(tok.build_inputs_with_special_tokens([])):
            raise ValueError('Tokenizer special-token accounting tidak konsisten')
        resolved[role]={'checkpoint':spec['checkpoint'],'revision':revision,
            'dimension':config.hidden_size,'model_type':config.model_type,
            'max_position_embeddings':getattr(config,'max_position_embeddings',None),
            'tokenizer_class':type(tok).__name__, 'special_tokens':tok.num_special_tokens_to_add(False)}
    return resolved

class BGEEncoder:
    def __init__(self, spec, device='cpu'):
        from FlagEmbedding import BGEM3FlagModel
        self.spec=spec
        # FlagEmbedding versi ini menerima path lokal; snapshot mengunci revision.
        path=snapshot_download(spec['checkpoint'],revision=spec['revision'],
           allow_patterns=['config.json','*.safetensors','pytorch_model.bin','tokenizer*','special_tokens_map.json',
                           'sentencepiece*','*.model','*.pt','modules.json','1_Pooling/*'])
        self.tokenizer=AutoTokenizer.from_pretrained(path)
        self.encoder=BGEM3FlagModel(path,use_fp16=False,devices=[device],normalize_embeddings=True)
        self.encoder.model.eval()
        for p in self.encoder.model.parameters(): p.requires_grad_(False)

    def encode(self, texts, batch_size=1):
        self.encoder.model.eval()
        with torch.no_grad():
            result=self.encoder.encode(texts,batch_size=batch_size,max_length=1024,
                      return_dense=True,return_sparse=False,return_colbert_vecs=False)
        dense=np.asarray(result['dense_vecs'],dtype=np.float32)
        if dense.shape != (len(texts),1024) or not np.isfinite(dense).all(): raise ValueError('BGE dense shape/finite salah')
        norm=np.linalg.norm(dense,axis=1,keepdims=True)
        if (norm==0).any(): raise ValueError('BGE zero vector')
        return dense/norm

    def stats(self, text):
        full=len(self.tokenizer.encode(text,add_special_tokens=True))
        return {'tokens_with_special':full,'truncated':full>1024,'tokens_removed':max(0,full-1024)}

def cached_bge(df,texts,encoder,cfg):
    folder=Path(cfg['output'])/'cache'; folder.mkdir(parents=True,exist_ok=True)
    entries=[{'essay_id':str(i),'text_sha256':text_hash(t)} for i,t in zip(df.essay_id,texts)]
    metadata={'entries':entries,'model':encoder.spec,'max_length':1024,'truncation':'right-no-window',
              'pooling':'FlagEmbedding dense_vecs CLS','normalization':'L2',
              'preprocessing':{'policy':POLICY,**cfg['preprocessing']},
              'tokenizer_class':type(encoder.tokenizer).__name__,
              'tokenization':[encoder.stats(t) for t in texts]}
    import hashlib
    key=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()
    npz=folder/(key+'.npy'); meta=folder/(key+'.json')
    if npz.exists() and meta.exists() and json.loads(meta.read_text())==metadata:
        values=np.load(npz,allow_pickle=False)
    else:
        values=encoder.encode(texts); np.save(npz,values);dump(meta,metadata)
    if values.shape!=(len(df),1024) or not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values,axis=1),1,atol=1e-5):
        raise ValueError('Cache BGE rusak')
    return torch.from_numpy(values),metadata

def load_roberta(spec):
    model=AutoModel.from_pretrained(spec['checkpoint'],revision=spec['revision'],add_pooling_layer=False)
    if model.config.hidden_size != 768: raise ValueError('RoBERTa hidden size salah')
    model.requires_grad_(True)
    return model
