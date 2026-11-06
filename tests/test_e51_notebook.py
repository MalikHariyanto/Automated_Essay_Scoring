"""E51 identity gate, unchanged E50 training, same-forward held-out gate exports."""
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

NOTEBOOK=ROOT/'notebooks/experiments/AES_300_E51_E50_S3_GatedConcat_Native_Learnable_GPU.ipynb'
BASE=ROOT/'notebooks/experiments/AES_300_E50_E43_S3_MultiHeadCoAttention4_Native_Learnable_GPU.ipynb'


def test_source_protocol_preserves_e50_and_build_is_deterministic():
    parent,raw=BASE.read_bytes(),NOTEBOOK.read_bytes()
    n=nbformat.read(NOTEBOOK,as_version=4);nbformat.validate(n);m=n.metadata.aes_experiment
    assert m.scenarios==['S3'] and m.cv_fits==5 and m.refits==m.test_evaluations==1
    assert m.loss=='MSE' and m.coattention_heads==4 and m.added_parameters_vs_e50==2050 and m.gate_identity_initialization
    code=[c for c in n.cells if c.cell_type=='code'];fp=n.metadata.aes_standalone.code_sha256
    assert hashlib.sha256('\n'.join(c.source.replace(fp,'__NOTEBOOK_CODE_SHA256__') for c in code).encode()).hexdigest()==fp
    text='\n'.join(c.source for c in code)
    assert all(t not in text for t in ['lds_training','lds_reweight','LDS_POWER','LDS-MSE','ListNet','cosine','WORKER_SOURCE','CV_SOURCE','TRAINING_SEEDS','REFIT_SCHEDULER_EPOCHS'])
    assert 'Hasil_'+NOTEBOOK.stem+'.zip' in text and 'AES_300_E51_E51' not in text
    for c in code:
        assert c.metadata.aes51_role and not c.outputs and c.execution_count is None
        assert not any(l.lstrip().startswith(('%','!')) for l in c.source.splitlines())
        assert not any(isinstance(v,ast.Call) and isinstance(v.func,ast.Name) and v.func.id in ['exec','eval'] for v in ast.walk(ast.parse(c.source)))
    a,b=definitions(BASE),definitions(NOTEBOOK)
    for name in ['ScalarMix','AnswerBiGRUAttention','AnswerBiGRUCoAttentionMeanPool','MultiHeadCoAttention','metrics','gwet_ac2','make_split','encoder',
                 'pair_content_windows','pair_batch_windows','bge_features','prediction_frame']:
        assert ast.dump(a[name])==ast.dump(b[name]).replace('E51','E50'),name
    loop=lambda fn:next(v for v in fn.body if isinstance(v,ast.For) and isinstance(v.target,ast.Name) and v.target.id=='epoch')
    assert ast.dump(loop(a['fit']))==ast.dump(loop(b['fit']))
    subprocess.run([sys.executable,'scripts/build_e51_notebook.py'],cwd=ROOT,check=True,capture_output=True)
    assert BASE.read_bytes()==parent and NOTEBOOK.read_bytes()==raw==(ROOT/'aes-learning/notebooks/experiments'/NOTEBOOK.name).read_bytes()


def test_gate_identity_precision_gradients_and_input_dependent_weights():
    ns=runtime(NOTEBOOK,'aes51_role');torch.manual_seed(11);gate=ns['GatedConcatenation']()
    assert sum(p.numel() for p in gate.parameters())==2050
    for dtype in [torch.float32,torch.float16,torch.bfloat16]:
        r=torch.randn(3,768).to(dtype);b=torch.randn(3,256).to(dtype)
        result,alpha=gate(r,b)
        torch.testing.assert_close(result,torch.cat([r,b],-1),rtol=0,atol=0)
        torch.testing.assert_close(alpha,torch.full((3,2),.5),rtol=0,atol=0)
    r=torch.randn(3,768,requires_grad=True);b=torch.randn(3,256,requires_grad=True)
    before=gate.logits.weight.detach().clone();result,alpha=gate(r,b)
    head=torch.nn.Linear(1024,1);(head(result).flatten()-torch.tensor([.7,.9,.8])).square().mean().backward()
    assert gate.logits.weight.grad.abs().sum()>0 and gate.logits.bias.grad.abs().sum()>0
    assert r.grad.abs().sum()>0 and b.grad.abs().sum()>0
    torch.optim.AdamW(gate.parameters(),lr=1e-4).step()
    assert not torch.equal(before,gate.logits.weight)
    _,learned=gate(r.detach(),b.detach())
    assert torch.max(learned[:,0])-torch.min(learned[:,0])>0
    assert torch.isfinite(learned).all();torch.testing.assert_close(learned.sum(1),torch.ones(3))
    assert not gate.last_alpha.requires_grad
    with pytest.raises(ValueError):gate(torch.randn(3,767),torch.randn(3,256))


def test_e50_identity_model_rng_inference_capture_and_optimizer(tmp_path):
    ns=runtime(NOTEBOOK,'aes51_role');old=runtime(BASE,'aes50_role');cfg,bundle=setup(ns,tmp_path)
    old['encoder']=ns['encoder'];old['tokenizer']=ns['tokenizer']
    oldcfg={k:v for k,v in cfg.items() if k!='fusion_gate_mode'};oldcfg['s3_fusion']='concat';oldcfg=old['config'](oldcfg)
    for k in oldcfg:
        if k not in ['s3_fusion']:assert cfg[k]==oldcfg[k],k
    ns['seed_everything'](71);model=ns['make_model']('S3',cfg).eval();state=torch.get_rng_state()
    old['seed_everything'](71);control=old['make_model']('S3',oldcfg).eval()
    assert torch.equal(state,torch.get_rng_state())
    for name,p in control.state_dict().items():torch.testing.assert_close(p,model.state_dict()[name],rtol=0,atol=0)
    assert sum(p.numel() for p in model.parameters())-sum(p.numel() for p in control.parameters())==2050
    assert ns['make_split'](bundle,cfg)==old['make_split'](bundle,oldcfg)
    bge=torch.randn(24,1024)
    current=ns['Engine'](model,bundle.texts,bge,cfg,'cpu',contexts=ns['contexts_from_bundle'](bundle))
    baseline=old['Engine'](control,bundle.texts,bge,oldcfg,'cpu',contexts=old['contexts_from_bundle'](bundle))
    p= current.forward([0,3,5]);q=baseline.forward([0,3,5]);torch.testing.assert_close(p,q,rtol=0,atol=0)
    p.square().mean().backward()
    assert model.fusion_gate.logits.weight.grad.abs().sum()>0
    assert model.projection[0].weight.grad.abs().sum()>0 and model.sequence_pool.multihead.q.weight.grad.abs().sum()>0
    groups=ns['optimizer_groups'](model,cfg);head=next(g for g in groups if g['name']=='head')
    assert head['lr']==1e-4 and {id(p) for p in model.fusion_gate.parameters()}<={id(p) for p in head['params']}
    for micro in [1,2]:
        cfg['micro_batch']=micro
        result=current.predict([0,3,5])
        assert [r['dataset_index'] for r in current.last_gate_rows]==[0,3,5]
        assert all(r['gate_roberta']==r['gate_bge']==.5 for r in current.last_gate_rows)
        np.testing.assert_allclose(result,baseline.predict([0,3,5]),rtol=1e-6,atol=1e-5)
    with pytest.raises(ValueError):ns['config']({**cfg,'fusion_gate_mode':'wrong'})


def test_gate_report_frame_uses_actual_ids_and_rejects_mismatch(tmp_path):
    ns=runtime(NOTEBOOK,'aes51_role');cfg,bundle=setup(ns,tmp_path)
    bundle.rows[1]['row_id']='row_0400'
    frame=ns['gate_prediction_frame'](bundle,[1],[80.], [{'dataset_index':1,'gate_roberta':.7,'gate_bge':.3}])
    assert frame.iloc[0]['ID Jawaban']==ns['friendly_value']('row_id','row_0400')
    assert frame.iloc[0]['Pengali RoBERTa']==1.4
    for rows in [[{'dataset_index':2,'gate_roberta':.7,'gate_bge':.3}],
                 [{'dataset_index':1,'gate_roberta':.7,'gate_bge':.7}],
                 [{'dataset_index':-1,'gate_roberta':.7,'gate_bge':.3}]]:
        with pytest.raises(ValueError):ns['gate_prediction_frame'](bundle,[1],[80.],rows)


def test_interrupted_training_resume_preserves_gate_and_history(tmp_path):
    ns=runtime(NOTEBOOK,'aes51_role');cfg,bundle=setup(ns,tmp_path);bge=torch.randn(24,1024);ids=list(range(8));path=tmp_path/'progress.json'
    save=ns['save_training_progress']
    def stop(file,state):
        save(file,state)
        if state['epoch']==1:raise RuntimeError('TECHNICAL INTERRUPT')
    ns['save_training_progress']=stop
    with pytest.raises(RuntimeError,match='TECHNICAL INTERRUPT'):ns['fit']('S3',ids,None,bundle,bge,cfg,2,42,checkpoint=path)
    ns['save_training_progress']=save
    resumed,engine,info=ns['fit']('S3',ids,None,bundle,bge,cfg,2,42,checkpoint=path)
    model,control,other=ns['fit']('S3',ids,None,bundle,bge,cfg,2,42)
    assert info['history']==other['history'] and info['fusion_gate']==other['fusion_gate']
    assert info['fusion_gate']['Jumlah Parameter']==2050
    for name,p in model.state_dict().items():torch.testing.assert_close(p,resumed.state_dict()[name],rtol=0,atol=0)
    np.testing.assert_array_equal(engine.predict([8,9]),control.predict([8,9]));assert engine.last_gate_rows==control.last_gate_rows


def test_complete_notebook_gate_oof_test_caches_exports_and_resume(tmp_path):
    ns=runtime(NOTEBOOK,'aes51_role')
    columns=list(ns['prediction_report_tables']([{'ID Jawaban':'TECHNICAL','Nilai Guru':80,'Prediksi S3':79.5}])['Nilai Guru dan Prediksi'][0])
    execute_full_native_fixture(tmp_path,NOTEBOOK,'aes51_role',5e-6,encoder_layers=4,prediction_columns=columns)
    info=json.loads((tmp_path/'standalone/workspace/outputs/dataset300_download.json').read_text());run=Path(info['run_folder'])
    assert Path(info['archive']).name=='Hasil_'+NOTEBOOK.stem+'.zip'
    cfg=json.loads((run/'effective_config.json').read_text());bundle=ns['read_dataset'](cfg);manifest=ns['make_split'](bundle,cfg);folder=run/'hasil_ringkas'
    proof=ns['validate_e51_download'](folder,bundle,manifest,cfg)
    assert proof['gate_oof_rows']==18 and proof['gate_test_rows']==6 and proof['gate_config_rows']==6
    evalfile=folder/'04_Prediksi_dan_Evaluasi.xlsx';w=pd.ExcelFile(evalfile)
    oof=pd.read_excel(w,sheet_name='Gate Validasi OOF');test=pd.read_excel(w,sheet_name='Gate Test')
    assert len(oof)==18 and len(test)==6 and set(oof['ID Jawaban']).isdisjoint(test['ID Jawaban'])
    assert len(pd.read_excel(w,sheet_name='Ringkasan Gate'))==7
    csv=pd.read_csv(run/'S3/test_predictions.csv');np.testing.assert_allclose(csv.gate_roberta,test['Alpha RoBERTa'],rtol=1e-7)
    # Fold gate values persisted with the selected-model predictions, not training-epoch gates.
    for path in (run/'S3').glob('cv_fold_*.json'):
        item=json.loads(path.read_text());assert len(item['gate_predictions'])==len(item['prediction'])
        assert [r['dataset_index'] for r in item['gate_predictions']]==item['validation_indices']
    vectors=pd.ExcelFile(folder/'02_Vektor_Jawaban.xlsx')
    assert {'Fusion S3 Sebelum Gate','Fusion S3','Gate Final Semua Jawaban'}<=set(vectors.sheet_names)
    before=pd.read_excel(vectors,sheet_name='Fusion S3 Sebelum Gate').iloc[:,1:].to_numpy()
    after=pd.read_excel(vectors,sheet_name='Fusion S3').iloc[:,1:].to_numpy()
    alpha=pd.read_excel(vectors,sheet_name='Gate Final Semua Jawaban').iloc[:,1:].to_numpy()
    np.testing.assert_allclose(after[:,:768],before[:,:768]*(2*alpha[:,0:1]),rtol=1e-5,atol=1e-6)
    np.testing.assert_allclose(after[:,768:],before[:,768:]*(2*alpha[:,1:2]),rtol=1e-5,atol=1e-6)
    original=evalfile.read_bytes()
    for title,key in [('Gate Test','Alpha RoBERTa'),('Gate Test','Pengali BGE'),('Gate Validasi OOF','ID Jawaban'),
                      ('Ringkasan Gate','Alpha RoBERTa Mean'),('Perbandingan Model',None),('Ringkasan Validasi',None)]:
        book=load_workbook(evalfile);sheet=book[title]
        if key is None:sheet.delete_rows(2,sheet.max_row-1)
        else:
            col=next(c.column for c in sheet[1] if c.value==key)
            value='WRONG_ID' if key=='ID Jawaban' else sheet.cell(2,col).value+.2
            sheet.cell(2,col,value)
        book.save(evalfile);book.close()
        with pytest.raises(RuntimeError,match='Unduhan dihentikan'):ns['validate_e51_download'](folder,bundle,manifest,cfg)
        evalfile.write_bytes(original)
    print(json.dumps({'scope':'SYNTHETIC CPU ONLY','full_notebook_twice':True,'same_forward_gates_cached':True,'gate':proof,'bad_gate_values_scales_ids_summary_rejected':True}))
