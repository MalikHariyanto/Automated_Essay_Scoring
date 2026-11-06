"""E50 bidirectional multi-head math, unchanged E43 training, full reports/resume."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import nbformat
import numpy as np
import pandas as pd
from openpyxl import load_workbook
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from test_e41_and_e30_native import runtime,setup,definitions
from test_e36_notebook import execute_full_native_fixture

NOTEBOOK=ROOT/'notebooks/experiments/AES_300_E50_E43_S3_MultiHeadCoAttention4_Native_Learnable_GPU.ipynb'
BASE=ROOT/'notebooks/experiments/AES_300_E43_E42_S3_MaskedMeanPooling_Native_Learnable_GPU.ipynb'


def test_structure_protocol_unchanged_training_and_deterministic_builder():
    old,raw=BASE.read_bytes(),NOTEBOOK.read_bytes()
    n=nbformat.read(NOTEBOOK,as_version=4);nbformat.validate(n);m=n.metadata.aes_experiment
    assert m.scenarios==['S3'] and m.cv_fits==5 and m.refits==m.test_evaluations==1
    assert m.loss=='MSE' and m.coattention_heads==4 and m.coattention_head_dim==32 and m.added_parameters_vs_e43==65536
    code=[c for c in n.cells if c.cell_type=='code'];fp=n.metadata.aes_standalone.code_sha256
    assert hashlib.sha256('\n'.join(c.source.replace(fp,'__NOTEBOOK_CODE_SHA256__') for c in code).encode()).hexdigest()==fp
    text='\n'.join(c.source for c in code)
    assert all(t not in text for t in ['lds_training','lds_reweight','LDS_POWER','LDS-MSE','ListNet','cosine','WORKER_SOURCE','CV_SOURCE','TRAINING_SEEDS','REFIT_SCHEDULER_EPOCHS'])
    for c in code:
        assert c.metadata.aes50_role and not c.outputs and c.execution_count is None
        assert not any(l.lstrip().startswith(('%','!')) for l in c.source.splitlines())
        assert not any(isinstance(v,ast.Call) and isinstance(v.func,ast.Name) and v.func.id in ['exec','eval'] for v in ast.walk(ast.parse(c.source)))
    a,b=definitions(BASE),definitions(NOTEBOOK)
    for name in ['ScalarMix','AnswerBiGRUAttention','optimizer_groups','train_all','metrics','gwet_ac2','make_split','Engine',
                 'pair_content_windows','pair_batch_windows','encoder','bge_features']:
        assert ast.dump(a[name])==ast.dump(b[name]).replace('E50','E43'),name
    loop=lambda fn:next(v for v in fn.body if isinstance(v,ast.For) and isinstance(v.target,ast.Name) and v.target.id=='epoch')
    assert ast.dump(loop(a['fit']))==ast.dump(loop(b['fit']))
    old_forward=next(v for v in a['SingleScoreModel'].body if isinstance(v,ast.FunctionDef) and v.name=='forward')
    new_forward=next(v for v in b['SingleScoreModel'].body if isinstance(v,ast.FunctionDef) and v.name=='forward')
    assert ast.dump(old_forward)==ast.dump(new_forward)
    subprocess.run([sys.executable,'scripts/build_e50_notebook.py'],cwd=ROOT,check=True,capture_output=True)
    assert BASE.read_bytes()==old and NOTEBOOK.read_bytes()==raw==(ROOT/'aes-learning/notebooks/experiments'/NOTEBOOK.name).read_bytes()


def test_bidirectional_heads_manual_equations_padding_and_gradients():
    ns=runtime(NOTEBOOK,'aes50_role');torch.manual_seed(11);module=ns['MultiHeadCoAttention']()
    a=torch.randn(2,7,128,requires_grad=True);r=torch.randn(2,5,128,requires_grad=True)
    valid=torch.tensor([[1]*7,[1]*4+[0]*3],dtype=torch.bool);ref=torch.tensor([[1]*5,[1]*3+[0]*2],dtype=torch.bool)
    aligned,roundtrip,ar,ra=module(a,r,valid,ref)
    assert aligned.shape==roundtrip.shape==(2,7,128) and ar.shape==(2,4,7,5) and ra.shape==(2,4,5,7)
    torch.testing.assert_close(ar.sum(-1),valid[:,None,:].expand(-1,4,-1).float())
    torch.testing.assert_close(ra.sum(-1),ref[:,None,:].expand(-1,4,-1).float())
    expected_aligned=[];expected_round=[]
    for h in range(4):
        lo,hi=h*32,(h+1)*32
        aq,rq=module.q(a)[...,lo:hi],module.q(r)[...,lo:hi]
        ak,rk=module.k(a)[...,lo:hi],module.k(r)[...,lo:hi]
        av,rv=module.v(a)[...,lo:hi],module.v(r)[...,lo:hi]
        forward=torch.softmax((aq@rk.transpose(1,2)/np.sqrt(32)).masked_fill(~ref[:,None,:],-torch.inf),dim=-1).masked_fill(~valid[:,:,None],0.)
        reverse=torch.softmax((rq@ak.transpose(1,2)/np.sqrt(32)).masked_fill(~valid[:,None,:],-torch.inf),dim=-1).masked_fill(~ref[:,:,None],0.)
        torch.testing.assert_close(ar[:,h],forward);torch.testing.assert_close(ra[:,h],reverse)
        expected_aligned.append(forward@rv);expected_round.append(forward@(reverse@av))
    torch.testing.assert_close(aligned,module.out(torch.cat(expected_aligned,-1)))
    torch.testing.assert_close(roundtrip,module.out(torch.cat(expected_round,-1)))
    changed_a=a.detach().clone();changed_r=r.detach().clone();changed_a[~valid]=1e4;changed_r[~ref]=-1e4
    altered=module(changed_a,changed_r,valid,ref)
    torch.testing.assert_close(altered[0][valid],aligned.detach()[valid]);torch.testing.assert_close(altered[1][valid],roundtrip.detach()[valid])
    ((aligned[valid]**2).mean()+(roundtrip[valid]**2).mean()).backward()
    assert a.grad[valid].abs().sum()>0 and r.grad[ref].abs().sum()>0
    for name,p in module.named_parameters():assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0,name
    before=ar.detach().clone()
    with torch.no_grad():module.q.weight[:32].add_(.1)
    updated=module(a.detach(),r.detach(),valid,ref)[2]
    assert not torch.allclose(updated[:,0],before[:,0])
    torch.testing.assert_close(updated[:,1:],before[:,1:],rtol=0,atol=0)
    with pytest.raises(ValueError):module(a,r,torch.zeros_like(valid),ref)
    with pytest.raises(ValueError):ns['MultiHeadCoAttention'](128,3)


def test_shared_initialization_rng_pool_masks_optimizer_and_learning(tmp_path):
    ns=runtime(NOTEBOOK,'aes50_role');old=runtime(BASE,'aes43_role');cfg,bundle=setup(ns,tmp_path)
    old['encoder']=ns['encoder'];old['tokenizer']=ns['tokenizer']
    basecfg={k:v for k,v in cfg.items() if k!='coattention_heads'};basecfg['s3_pooling']='bigru_coattention_mean_pooling';basecfg=old['config'](basecfg)
    ns['seed_everything'](71);model=ns['make_model']('S3',cfg);state=torch.get_rng_state()
    old['seed_everything'](71);control=old['make_model']('S3',basecfg)
    assert torch.equal(state,torch.get_rng_state())
    for name,value in control.state_dict().items():torch.testing.assert_close(value,model.state_dict()[name],rtol=0,atol=0)
    assert sum(p.numel() for p in model.parameters())-sum(p.numel() for p in control.parameters())==65536
    assert ns['make_split'](bundle,cfg)==old['make_split'](bundle,basecfg)
    pool=model.sequence_pool;pool.capture_alignment=True
    hidden=torch.randn(2,10,768,requires_grad=True)
    mask=torch.tensor([[0,0,0,1,1,1,0,0,0,0],[0,0,0,1,1,0,0,0,0,0]],dtype=torch.bool)
    ref=torch.tensor([[0,1,1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0]],dtype=torch.bool)
    result=pool(hidden,mask,ref)
    torch.testing.assert_close(pool.last_cross_attention,pool.last_cross_attention_heads.mean(1))
    torch.testing.assert_close(pool.last_weights,mask.float()/mask.sum(1,keepdim=True))
    changed=hidden.detach().clone();changed[~(mask|ref)]=1e4
    torch.testing.assert_close(result,pool(changed,mask,ref),rtol=0,atol=0)
    before={n:p.detach().clone() for n,p in pool.multihead.named_parameters()}
    result.square().mean().backward()
    assert hidden.grad[mask].abs().sum()>0 and hidden.grad[ref].abs().sum()>0 and hidden.grad[~(mask|ref)].abs().sum()==0
    groups=ns['optimizer_groups'](model,cfg);poolgroup=next(g for g in groups if g['name']=='pooling_norm')
    assert poolgroup['lr']==1e-4 and {id(p) for p in pool.multihead.parameters()}<={id(p) for p in poolgroup['params']}
    torch.optim.AdamW(groups,weight_decay=cfg['weight_decay']).step()
    for name,p in pool.multihead.named_parameters():assert not torch.equal(p,before[name]),name
    with pytest.raises(ValueError):ns['config']({**cfg,'coattention_heads':2})


def test_interrupted_fit_resume(tmp_path):
    ns=runtime(NOTEBOOK,'aes50_role');cfg,bundle=setup(ns,tmp_path);bge=torch.randn(24,1024);training=list(range(8));path=tmp_path/'progress.json'
    save=ns['save_training_progress']
    def stop(file,state):
        save(file,state)
        if state['epoch']==1:raise RuntimeError('TECHNICAL INTERRUPT')
    ns['save_training_progress']=stop
    with pytest.raises(RuntimeError,match='TECHNICAL INTERRUPT'):ns['fit']('S3',training,None,bundle,bge,cfg,2,42,checkpoint=path)
    ns['save_training_progress']=save
    resumed,_,info=ns['fit']('S3',training,None,bundle,bge,cfg,2,42,checkpoint=path)
    plain,_,other=ns['fit']('S3',training,None,bundle,bge,cfg,2,42)
    assert info['history']==other['history'] and info['multihead_projections']==other['multihead_projections']
    for name,p in plain.state_dict().items():torch.testing.assert_close(p,resumed.state_dict()[name],rtol=0,atol=0)
    assert len(info['multihead_projections'])==4


def test_full_native_notebook_reports_and_fresh_kernel_resume(tmp_path):
    ns=runtime(NOTEBOOK,'aes50_role')
    columns=list(ns['prediction_report_tables']([{'ID Jawaban':'TECHNICAL','Nilai Guru':80,'Prediksi S3':79.5}])['Nilai Guru dan Prediksi'][0])
    execute_full_native_fixture(tmp_path,NOTEBOOK,'aes50_role',5e-6,encoder_layers=4,prediction_columns=columns)
    info=json.loads((tmp_path/'standalone/workspace/outputs/dataset300_download.json').read_text());run=Path(info['run_folder'])
    cfg=json.loads((run/'effective_config.json').read_text());bundle=ns['read_dataset'](cfg);manifest=ns['make_split'](bundle,cfg);folder=run/'hasil_ringkas'
    proof=ns['validate_e50_download'](folder,bundle,manifest,cfg)
    assert proof['test_rows']==6 and proof['oof_rows']==18 and proof['coattention_heads']==4 and proof['multihead_projection_rows']==24
    projection=pd.read_excel(folder/'05_Riwayat_Pelatihan.xlsx',sheet_name='Proyeksi Multihead')
    assert len(projection)==24 and set(projection.Head)=={1,2,3,4}
    assert not any('LDS' in x for x in pd.ExcelFile(folder/'05_Riwayat_Pelatihan.xlsx').sheet_names)
    f=folder/'04_Prediksi_dan_Evaluasi.xlsx';original=f.read_bytes()
    for title,action in [('Perbandingan Model','empty'),('Ringkasan Validasi','empty'),('Selisih Tepat','count'),('Prediksi Validasi OOF','oof')]:
        w=load_workbook(f);s=w[title]
        if action=='empty':s.delete_rows(2,s.max_row-1)
        elif action=='count':s.cell(2,4,s.cell(2,4).value+1)
        else:s.cell(2,2,'WRONG_OOF_ID')
        w.save(f);w.close()
        with pytest.raises(RuntimeError,match='Unduhan dihentikan'):ns['validate_e50_download'](folder,bundle,manifest,cfg)
        f.write_bytes(original)
    f=folder/'05_Riwayat_Pelatihan.xlsx';original=f.read_bytes();w=load_workbook(f);s=w['Proyeksi Multihead']
    col=next(c.column for c in s[1] if c.value=='Jumlah Head');s.cell(2,col,1);w.save(f);w.close()
    with pytest.raises(RuntimeError,match='multihead'):ns['validate_e50_download'](folder,bundle,manifest,cfg)
    f.write_bytes(original)
    print(json.dumps({'scope':'SYNTHETIC CPU ONLY','full_notebook_twice':True,'gate':proof,'bad_summary_counts_oof_heads_rejected':True}))
