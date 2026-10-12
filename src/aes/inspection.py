from pathlib import Path
import json
import torch
from transformers import AutoTokenizer
from .io import load_config,load_data,load_rubric,dump,sha256
from .audit import validate_inputs_for_inspection,training_blockers
from .cleaning import clean
from .encoders import resolve_models,BGEEncoder,cached_bge,load_roberta
from .model import AESModel
from .windows import batch_windows
from .trainer import device_name


def inspect_data(cfg):
    df=load_data(cfg['csv']);r=load_rubric(cfg['rubric'])
    report=validate_inputs_for_inspection(df,r)
    review=json.loads(Path(cfg['split']['review_file']).read_text()) if Path(cfg['split']['review_file']).exists() else None
    report['training_blockers']=training_blockers(df,r,review,sha256(cfg['csv']))
    report['supervised_eligible']=not report['training_blockers']
    report['source_hashes']={'csv':sha256(cfg['csv']),'rubric':sha256(cfg['rubric'])}
    cleaned=[clean(t,**cfg['preprocessing']) for t in df.text_raw]
    report['cleaning']=[{'essay_id':i,'changed':a!=b,'raw_hash':__import__('hashlib').sha256(a.encode()).hexdigest(),
                        'clean_hash':__import__('hashlib').sha256(b.encode()).hexdigest()}
                       for i,a,b in zip(df.essay_id,df.text_raw,cleaned)]
    out=Path(cfg['output']);dump(out/'audit.json',report)
    export=df.copy();export['text_model']=cleaned;export.to_csv(out/'cleaning_review.csv',index=False)
    return df,r,report

def inspect_models(cfg,df=None):
    if df is None:df=load_data(cfg['csv'])
    resolved=resolve_models(cfg);dump(Path(cfg['output'])/'model_revisions.json',resolved)
    device=device_name();texts=[clean(t,**cfg['preprocessing']) for t in df.text_raw]
    bge_encoder=BGEEncoder(resolved['bge'],device)
    bge,metadata=cached_bge(df,texts,bge_encoder,cfg)
    stats=metadata['tokenization']
    bge_ids=bge_encoder.tokenizer.encode(texts[0],add_special_tokens=True,truncation=True,max_length=1024)[:8]
    bge_frozen=not any(p.requires_grad for p in bge_encoder.encoder.model.parameters())
    assert bge_frozen and not bge_encoder.encoder.model.training
    del bge_encoder
    tok=AutoTokenizer.from_pretrained(resolved['roberta']['checkpoint'],revision=resolved['roberta']['revision'], use_fast=False)
    _,_,all_stats=batch_windows(texts,tok,maximum=512,overlap=128)
    win,mapping,_=batch_windows(texts[:1],tok,device,512,128)
    roberta=load_roberta(resolved['roberta'])
    model=AESModel('S3',roberta,**cfg['scenarios']['S3']).to(device).eval()
    with torch.no_grad():p,features=model(windows=win,chunk_to_essay=mapping,bge=bge[:1].to(device),n_essays=1,return_features=True)
    result={'status':'UNTRAINED_DIAGNOSTIC_NOT_ESSAY_ASSESSMENT','device':device,'resolved_models':resolved,
       'bge_tokenization':dict(zip(df.essay_id,stats)),'roberta_windows':dict(zip(df.essay_id,all_stats)),
       'truncated_essays':sum(s['truncated'] for s in stats),'removed_tokens':sum(s['tokens_removed'] for s in stats),
       'first_token_ids':win['input_ids'][0,:8].tolist(),
       'token_ids_by_encoder':{'bge':bge_ids,'roberta':win['input_ids'][0,:8].tolist()},
       'gradient_policy':{'bge_frozen':bge_frozen,'roberta_trainable':all(p.requires_grad for p in model.roberta.parameters()),'projection_trainable':all(p.requires_grad for p in model.projection.parameters())},
       'shapes':{**{k:list(v.shape) for k,v in features.items()},'output':list(p.shape)},
       'vector_first_five':{k:v[0,:5].cpu().tolist() for k,v in features.items()},
       'untrained_output':p[0].cpu().tolist()}
    dump(Path(cfg['output'])/'forward_inspection.json',result)
    return result
