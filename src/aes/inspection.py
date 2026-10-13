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
from .metrics import ordinal_prediction


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
    bge_sample_ids=bge_encoder.tokenizer.encode(texts[0],add_special_tokens=True,truncation=True,max_length=1024)[:16]
    bge_tokens=[{'position':i,'token':token,'token_id':int(token_id),
                 'special':token_id in bge_encoder.tokenizer.all_special_ids}
                for i,(token_id,token) in enumerate(zip(bge_sample_ids,bge_encoder.tokenizer.convert_ids_to_tokens(bge_sample_ids)))]
    bge_frozen=not any(p.requires_grad for p in bge_encoder.encoder.model.parameters())
    assert bge_frozen and not bge_encoder.encoder.model.training
    del bge_encoder
    tok=AutoTokenizer.from_pretrained(resolved['roberta']['checkpoint'],revision=resolved['roberta']['revision'], use_fast=False)
    _,_,all_stats=batch_windows(texts,tok,maximum=512,overlap=128)
    win,mapping,_=batch_windows(texts[:1],tok,device,512,128)
    ro_sample_ids=win['input_ids'][0,:16].tolist()
    ro_tokens=[{'position':i,'token':token,'token_id':int(token_id),'special':token_id in tok.all_special_ids}
               for i,(token_id,token) in enumerate(zip(ro_sample_ids,tok.convert_ids_to_tokens(ro_sample_ids)))]
    roberta=load_roberta(resolved['roberta'])
    diagnostics={}
    rubric=load_rubric(cfg['rubric'])
    for scenario in ['S1','S2','S3']:
        torch.manual_seed(42)  # Reproduksibilitas head diagnostik, bukan training.
        model=AESModel(scenario,roberta if scenario!='S2' else None,
                       **cfg['scenarios'][scenario]).to(device).eval()
        with torch.no_grad():
            p,features=model(windows=win,chunk_to_essay=mapping,bge=bge[:1].to(device),n_essays=1,return_features=True)
            if scenario=='S2':features['mlp_hidden']=model.head[:3](features['bge'])
        diagnostics[scenario]={
            'status':'UNTRAINED_DIAGNOSTIC_NOT_ESSAY_ASSESSMENT',
            'essay_id':df.iloc[0].essay_id,
            'trait_order':[t['name'] for t in rubric['traits']],
            'shapes':{**{k:list(v.shape) for k,v in features.items()},'output':list(p.shape)},
            'vector_first_five':{k:v[0,:5].cpu().tolist() for k,v in features.items()},
            'continuous':dict(zip([t['name'] for t in rubric['traits']],p[0].cpu().tolist())),
            'categories':{t['name']:int(ordinal_prediction(p[0,j].cpu().item(),t))
                          for j,t in enumerate(rubric['traits'])}}
        if scenario!='S2':
            diagnostics[scenario]['cls_per_window_first_five']=features['roberta_window_cls'][:,:5].cpu().tolist()
        if scenario=='S3':
            assert torch.equal(features['concat'][:,:768],features['roberta'])
            assert torch.equal(features['concat'][:,768:],features['projection'])
            diagnostics[scenario]['concat_segments']={'positions_0_to_767':'RoBERTa',
                'positions_768_to_1023':'BGE projected + LayerNorm',
                'roberta_first_five':features['concat'][0,:5].cpu().tolist(),
                'bge_projection_first_five':features['concat'][0,768:773].cpu().tolist()}
    result={'status':'UNTRAINED_DIAGNOSTIC_NOT_ESSAY_ASSESSMENT','device':device,'resolved_models':resolved,
       'bge_tokenization':dict(zip(df.essay_id,stats)),'roberta_windows':dict(zip(df.essay_id,all_stats)),
       'truncated_essays':sum(s['truncated'] for s in stats),'removed_tokens':sum(s['tokens_removed'] for s in stats),
       'first_token_ids':win['input_ids'][0,:8].tolist(),
       'first_essay_flow':{'essay_id':str(df.iloc[0].essay_id),'text_model_excerpt':texts[0][:300],
           'token_samples':{'bge':bge_tokens,'roberta':ro_tokens},
           'roberta_window_inputs_shape':list(win['input_ids'].shape),
           'chunk_to_essay':mapping.cpu().tolist(),
           'window_token_counts_including_special':win['attention_mask'].sum(1).cpu().tolist(),
           'bge_l2_norm':float(torch.linalg.vector_norm(bge[0])),
           'teacher_scores':{t['name']:str(df.iloc[0]['score_'+t['name']]) or None for t in rubric['traits']}},
       'token_ids_by_encoder':{'bge':bge_ids,'roberta':win['input_ids'][0,:8].tolist()},
       'gradient_policy':{'bge_frozen':bge_frozen,'roberta_trainable':all(p.requires_grad for p in model.roberta.parameters()),'projection_trainable':all(p.requires_grad for p in model.projection.parameters())},
       'shapes':{**{k:list(v.shape) for k,v in features.items()},'output':list(p.shape)},
       'vector_first_five':{k:v[0,:5].cpu().tolist() for k,v in features.items()},
       'untrained_output':p[0].cpu().tolist(), 'scenarios':diagnostics}
    dump(Path(cfg['output'])/'forward_inspection.json',result)
    return result
