import contextlib
import math
import random
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from .model import AESModel, loss_mse
from .encoders import load_roberta
from .windows import batch_windows
from .metrics import evaluate, selection_monitor
from .checkpoint import save_checkpoint
from .io import dump


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def device_name():return 'cuda' if torch.cuda.is_available() else 'cpu'

class BatchEngine:
    def __init__(self,model,texts,bge,resolved,cfg,device):
        self.model=model;self.texts=texts;self.bge=bge;self.cfg=cfg;self.device=device
        self.tokenizer=AutoTokenizer.from_pretrained(resolved['roberta']['checkpoint'], revision=resolved['roberta']['revision'], use_fast=False) if model.scenario!='S2' else None
    def forward(self,indices):
        bge=self.bge[indices].to(self.device) if self.model.scenario!='S1' else None
        if self.model.scenario=='S2':return self.model(bge=bge)
        s=self.cfg['models']['roberta']
        win,mapping,_=batch_windows([self.texts[i] for i in indices],self.tokenizer,self.device,s['max_length'],s['overlap'])
        return self.model(windows=win,chunk_to_essay=mapping,bge=bge,n_essays=len(indices))
    def predict(self,indices,micro=1):
        self.model.eval();out=[]
        with torch.no_grad():
            for start in range(0,len(indices),micro):out.append(self.forward(indices[start:start+micro]).cpu().float().numpy())
        return np.concatenate(out)


def train_run(scenario,candidate,train_ids,valid_ids,texts,y,bge,resolved,cfg,rubric,seed,out,epochs=None,early_stop=True):
    seed_everything(seed);device=device_name();out=Path(out)
    roberta=load_roberta(resolved['roberta']) if scenario!='S2' else None
    model=AESModel(scenario,roberta,**cfg['scenarios'][scenario]).to(device)
    engine=BatchEngine(model,texts,bge,resolved,cfg,device)
    c=cfg['training'];rates=c['candidates'][candidate];micro=c['micro_batch'];effective=c['effective_batch']
    if effective!=8 or not 1<=micro<=effective:raise ValueError('Effective batch harus 8, micro-batch 1..8')
    groups=[]
    encoder=[p for name,p in model.named_parameters() if name.startswith('roberta.') and p.requires_grad]
    head=[p for name,p in model.named_parameters() if not name.startswith('roberta.') and p.requires_grad]
    if encoder:groups.append({'params':encoder,'lr':rates['encoder_lr']})
    groups.append({'params':head,'lr':rates['head_lr']})
    optimizer=torch.optim.AdamW(groups,weight_decay=c['weight_decay'])
    n_epochs=epochs or c['max_epochs']
    if not 1<=n_epochs<=5:raise ValueError('Jumlah epoch harus 1–5')
    steps=math.ceil(len(train_ids)/effective)*n_epochs
    scheduler=get_linear_schedule_with_warmup(optimizer,math.ceil(c['warmup_ratio']*steps),steps)
    scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda')
    amp_dtype=torch.bfloat16 if device=='cuda' and torch.cuda.is_bf16_supported() else torch.float16
    best=None;best_epoch=0;stale=0;history=[];fallback=False;best_metrics=None;optimizer_steps=0
    for epoch in range(1,n_epochs+1):
        model.train();order=np.random.permutation(train_ids).tolist();loss_sum=0
        # Satu effective batch berisi <=8 esai, bukan delapan chunks.
        for start in range(0,len(order),effective):
            batch=order[start:start+effective];optimizer.zero_grad(set_to_none=True)
            for offset in range(0,len(batch),micro):
                ids=batch[offset:offset+micro];labels=torch.tensor(y[ids],dtype=torch.float32,device=device)
                with torch.autocast(device_type=device,dtype=amp_dtype,enabled=device=='cuda'):
                    p=engine.forward(ids);loss=loss_mse(p.float(),labels)
                    scaled_loss=loss*(len(ids)/len(batch))
                scaler.scale(scaled_loss).backward();loss_sum+=loss.item()*len(ids)
            scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip'])
            scaler.step(optimizer);scaler.update();scheduler.step();optimizer_steps+=1
        log={'epoch':epoch,'train_mse':loss_sum/len(train_ids),'optimizer_steps':optimizer_steps,
             'effective_batch_essays':8,'micro_batch_essays':micro,'last_batch_can_be_smaller':True}
        if valid_ids is not None:
            pred=engine.predict(valid_ids,micro);metrics=evaluate(y[valid_ids],pred,rubric)
            mse=float(np.mean((y[valid_ids]-pred)**2));kind,value=selection_monitor(metrics,mse,fallback)
            if kind=='validation_mse' and not fallback:
                # Kebijakan sebelum run: fallback sticky; reset pembanding agar unit tidak tercampur.
                fallback=True;best=None;stale=0;log['fallback_event']='QWK tidak terdefinisi pada semua empat trait; MSE sticky dan pembanding direset.'
            log.update({'metrics':metrics,'validation_mse':mse,'monitor':kind,'monitor_value':value})
            improved=best is None or (value<best if fallback else value>best)
        else:improved=True;metrics=None;value=log['train_mse'];kind='fixed_epochs_no_validation'
        if improved:
            best=value;best_epoch=epoch;stale=0;best_metrics=metrics
            save_checkpoint(out/'best.pt',model,cfg,resolved,rubric,{'trained':True,'seed':seed,'candidate':candidate,
               'epoch':epoch,'monitor':kind,'train_essay_count':len(train_ids),'validation_essay_count':len(valid_ids) if valid_ids is not None else 0})
        else:stale+=1
        history.append(log);dump(out/'history.json',history)
        if early_stop and stale>=c['patience']:break
    result={'candidate':candidate,'best_epoch':best_epoch,'best_monitor':best,'monitor_kind':kind,
         'metrics':best_metrics,'best_validation_mse':next((h['validation_mse'] for h in history if h['epoch']==best_epoch),None),
         'checkpoint':str(out/'best.pt'),'completed_epochs':len(history)}
    dump(out/'result.json',result)
    del engine,model,optimizer
    if device=='cuda':torch.cuda.empty_cache()
    return result


def train_with_oom_retry(*args,**kwargs):
    import copy
    cfg=copy.deepcopy(kwargs['cfg']);out=Path(kwargs['out']);events=[]
    while True:
        try:
            kwargs['cfg']=cfg;return train_run(*args,**kwargs)
        except torch.cuda.OutOfMemoryError:
            old=cfg['training']['micro_batch']
            events.append({'event':'OOM','micro_batch':old,'action':'restart_same_seed_from_initial_checkpoint'})
            out.mkdir(parents=True,exist_ok=True);dump(out/'oom.json',events)
            torch.cuda.empty_cache()
            if old<=1:raise RuntimeError('OOM pada satu esai: seluruh windows tetap wajib; gunakan GPU lebih besar/CPU. RoBERTa tidak dibekukan.')
            cfg['training']['micro_batch']=max(1,old//2)
