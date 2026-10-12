"""Checkpoint asli + teks CSV; target fixture teknis bukan skor siswa."""
import gc,json,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from transformers import AutoTokenizer
from aes.io import load_config,load_data,load_rubric,dump
from aes.cleaning import clean
from aes.encoders import BGEEncoder,cached_bge,load_roberta
from aes.model import AESModel,loss_mse
from aes.windows import batch_windows
from aes.checkpoint import save_checkpoint,load_checkpoint

if __name__=='__main__':
    torch.set_num_threads(2)
    root=Path(__file__).resolve().parents[1];cfg=load_config(root/'configs/default.json')
    df=load_data(cfg['csv']);rubric=load_rubric(cfg['rubric'])
    resolved=json.loads((Path(cfg['output'])/'model_revisions.json').read_text())
    texts=[clean(t,**cfg['preprocessing']) for t in df.text_raw]
    encoder=BGEEncoder(resolved['bge'],'cpu');assert not any(p.requires_grad for p in encoder.encoder.model.parameters())
    assert not encoder.encoder.model.training
    bge,metadata=cached_bge(df,texts,encoder,cfg);assert not bge.requires_grad
    frozen_parameter=next(encoder.encoder.model.parameters());frozen_before=frozen_parameter.detach().clone()
    tokenizer=AutoTokenizer.from_pretrained(resolved['roberta']['checkpoint'],revision=resolved['roberta']['revision'],use_fast=False)
    assert tokenizer.num_special_tokens_to_add(False)==2
    results=[]
    for scenario in ['S1','S2','S3']:
        torch.manual_seed(42)
        model=AESModel(scenario,load_roberta(resolved['roberta']) if scenario!='S2' else None,**cfg['scenarios'][scenario])
        cases=[]
        for b in [1,2]:
            win,mapping,stats=batch_windows(texts[:b],tokenizer)
            # Cakupan diuji dengan token IDs asli, bukan fixture tokenizer.
            from aes.windows import content_windows
            for text in texts[:b]:
                chunks,s=content_windows(text,tokenizer);ids=tokenizer.encode(text,add_special_tokens=False);covered={}
                for chunk,start in zip(chunks,s['starts']):
                    for j,v in enumerate(chunk['input_ids'][1:-1]):covered[start+j]=v
                assert [covered[j] for j in range(len(ids))]==ids
            kw={'windows':win,'chunk_to_essay':mapping,'n_essays':b,'bge':bge[:b]}
            model.eval()
            with torch.no_grad():pred=model(**kw)
            assert pred.shape==(b,4) and torch.isfinite(pred).all()
            cases.append({'essays':b,'windows':len(mapping),'shape':list(pred.shape)})
        model.train();optimizer=torch.optim.SGD(model.parameters(),lr=.001)
        # Label 3 hanya fixture backward teknis, tidak diambil/ditulis sebagai skor CSV.
        labels=torch.full((2,4),3.)
        optimizer.zero_grad();pred=model(**kw);loss_mse(pred,labels).backward()
        representatives={}
        for name,p in model.named_parameters():
            group=name.split('.')[0]
            if group not in representatives and p.grad is not None and p.grad.abs().sum()>0:
                representatives[group]=(name,p,p.detach().clone())
        optimizer.step()
        changed={group:bool(not torch.equal(p,old)) for group,(_,p,old) in representatives.items()}
        assert changed['head']
        if scenario!='S2':assert changed['roberta']
        if scenario=='S3':assert changed['projection']
        assert frozen_parameter.grad is None and torch.equal(frozen_parameter,frozen_before)
        model.eval()
        with torch.no_grad():before=model(**kw)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint=Path(folder)/'technical.pt'
            save_checkpoint(checkpoint,model,cfg,resolved,rubric,{'trained':False,'purpose':'technical_backward_fixture_not_research'})
            other,payload=load_checkpoint(checkpoint,'cpu',cfg['rubric'],cfg['preprocessing'])
            with torch.no_grad():after=other(**kw)
            assert torch.allclose(before,after,atol=1e-6)
            del other,payload
        results.append({'scenario':scenario,'forward_cases':cases,'changed_trainable_groups':changed,
                        'bge_frozen_unchanged':True,'save_reload_eval_match':True,'target_source':'technical_fixture_only'})
        del model,optimizer,representatives,pred,before,after,kw;gc.collect()
    dump(Path(cfg['output'])/'real_checkpoint_smoke.json',results)
    print(json.dumps(results,indent=2))
