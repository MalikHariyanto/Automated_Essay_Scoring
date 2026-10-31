"""Frozen feature readout diagnostics; synthetic fixtures, not research accuracy."""
import hashlib
import json
from pathlib import Path
import sys
import types
import zipfile

import nbformat
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT/'notebooks/experiments/AES_300_BGE_Frozen_Readout_Colab.ipynb'
sys.path.insert(0, str(ROOT/'tests'))
from test_teacher_score import make_book


def runtime(tmp_path):
    n = json.loads(NOTEBOOK.read_text())
    mod = types.ModuleType('readout_technical_fixture'); sys.modules[mod.__name__] = mod
    ns = mod.__dict__;ns['ROOT'] = tmp_path
    ns['show'] = lambda *a, **k: None
    for cell in n['cells']:
        s = ''.join(cell['source'])
        if not s.startswith('%%aes\n'):
            continue
        s = s.removeprefix('%%aes\n')
        # Load definitions/imports, then execute run cells explicitly below.
        if 'def bge_features' in s:
            s = s[:s.index('\nbge = bge_features')]
        if not s.lstrip().startswith('def ') and any(marker in s for marker in ["EPOCHS, FOLDS =", 'bundle = read_dataset', 'manifest = make_split',
                                          'bge = bge_features', 'diagnostic = readout_config',
                                          'result = run_readout_cv', 'summary = summarize_readouts',
                                          'download_info = export_readout']):
            if 'def bge_features' in s:
                s = s[:s.index('\nbge = bge_features')]
            else:
                continue
        exec(compile(s, '<standalone-diagnostic>', 'exec'), ns)
    return ns


def fixture_run(tmp_path):
    ns = runtime(tmp_path);path=tmp_path/'TECHNICAL_FIXTURE_ONLY.xlsx';make_book(path)
    cfg=dict(dataset=str(path), output=str(tmp_path/'results'),sheet='Sheet1',expected_rows=24,
             score_min=0,score_max=100,folds=5,test_fraction=.2,split_seed=2026,split_candidates=128,
             epochs=2,early_stopping_patience=5,effective_batch=8,micro_batch=1,
             dropout=.1,weight_decay=.01,warmup_ratio=.1,gradient_clip=1.)
    bundle=ns['read_dataset'](cfg);manifest=ns['make_split'](bundle,cfg)
    rng=np.random.default_rng(91);X=rng.normal(size=(24,1024)).astype(np.float32)
    X/=np.linalg.norm(X,axis=1,keepdims=True)
    diagnostic=ns['readout_config'](dict(seeds=[42],alphas=[.01,1.],lr=1e-4,primary_tolerance=2))
    torch.set_num_threads(2)
    return ns,cfg,bundle,manifest,X,diagnostic


def test_standalone_structure_hash_and_mirror():
    n=nbformat.read(NOTEBOOK,as_version=4);nbformat.validate(n)
    assert NOTEBOOK.read_bytes()==(ROOT/'aes-learning/notebooks/experiments'/NOTEBOOK.name).read_bytes()
    identity=n.metadata.aes_standalone.code_sha256
    code='\n'.join(c.source.removeprefix('%%aes\n').replace(identity,'__READOUT_CODE_HASH__') for c in n.cells if c.cell_type=='code')
    assert hashlib.sha256(code.encode()).hexdigest()==identity
    assert all(not c.outputs and c.execution_count is None for c in n.cells if c.cell_type=='code')
    assert n.metadata.aes_experiment.BGE_frozen and not n.metadata.aes_experiment.test_evaluated
    assert 'from aes.' not in code and 'import bge_readout_diagnostic' not in code


def test_train_only_transforms_identical_init_and_nested_ridge(tmp_path):
    ns,cfg,bundle,m,X,d=fixture_run(tmp_path);tr=m['folds'][0]['train'];va=m['folds'][0]['validation']
    scaler,off=ns['prepare_transform'](X,bundle.y,tr,'MLP_STANDARD_SCALER')
    np.testing.assert_allclose(scaler.mean_,X[tr].mean(0),atol=1e-7)
    changed=X.copy();changed[va]=10000
    scaler2,_=ns['prepare_transform'](changed,bundle.y,tr,'MLP_STANDARD_SCALER')
    np.testing.assert_array_equal(scaler.mean_,scaler2.mean_)
    _,offset=ns['prepare_transform'](X,bundle.y,tr,'MLP_TARGET_CENTER')
    y=bundle.y.copy();y[va]=-10000;y[m['test']]=10000
    assert offset==ns['prepare_transform'](X,y,tr,'MLP_TARGET_CENTER')[1]
    p,alpha,grid=ns['ridge_fold'](X,bundle.y,tr,va,m['folds'],d['alphas'])
    p2,alpha2,_=ns['ridge_fold'](X,y,tr,va,m['folds'],d['alphas'])
    np.testing.assert_array_equal(p,p2);assert alpha==alpha2
    assert all(r['inner_rows']==len(tr) for r in grid)
    a=ns['make_readout_head'](43,.1);b=ns['make_readout_head'](43,.1)
    assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))


def test_full_cv_resume_oof_report_and_formula_privacy(tmp_path):
    ns,cfg,bundle,m,X,d=fixture_run(tmp_path)
    bundle.rows[0]['text']='=TECHNICAL_FORMULA_TEXT_NOT_TO_EXECUTE'
    result=ns['run_readout_cv'](bundle,X,m,cfg,d,tmp_path/'runs')
    assert not result['test_evaluated'] and not result['BGE_encoder_trained']
    dev_ids={bundle.rows[i]['row_id'] for i in m['development']}
    for condition in ns['CONDITIONS']:
        rows=[r for r in result['predictions'] if r['condition']==condition]
        assert len(rows)==len(dev_ids) and {r['row_id'] for r in rows}==dev_ids
    transforms=[r for r in result['transforms'] if 'initial_parameter_sha256' in r]
    for fold in range(1,6):
        assert len({r['initial_parameter_sha256'] for r in transforms if r['fold']==fold})==1
    original=ns['make_readout_head']
    def fail_optimizer(*a,**k):raise AssertionError('Completed folds must resume without training')
    # Models are initialized before checking progress but optimizer.step must never execute on resume.
    old_step=torch.optim.AdamW.step;torch.optim.AdamW.step=fail_optimizer
    try:again=ns['run_readout_cv'](bundle,X,m,cfg,d,tmp_path/'runs')
    finally:torch.optim.AdamW.step=old_step
    assert result['predictions']==again['predictions'] and result['history']==again['history']
    summary=ns['summarize_readouts'](result,2);assert len(summary)==4 and summary.repetitions.tolist()==[1,1,1,1]
    duels=ns['readout_duels'](result,d);assert len(duels)==6
    assert (duels.left_closer+duels.right_closer+duels.ties==len(dev_ids)).all()
    info=ns['export_readout'](bundle,X,m,cfg,d,result)
    with zipfile.ZipFile(info['archive']) as archive:
        assert set(archive.namelist())==set(ns['REPORT_BOOKS'])|{'manifest.json'}
        metadata=json.loads(archive.read('manifest.json'));assert not metadata['test_evaluated']
        for name in ns['REPORT_BOOKS']:
            assert hashlib.sha256(archive.read(name)).hexdigest()==metadata['books'][name]
    from openpyxl import load_workbook
    b=load_workbook(Path(result['run_folder'])/'hasil_ringkas'/ns['REPORT_BOOKS'][0],data_only=False)
    c=b['Data']['D2'];assert c.value=='=TECHNICAL_FORMULA_TEXT_NOT_TO_EXECUTE' and c.data_type=='s';b.close()
    changed=d|{'lr':2e-4};next_result=ns['run_readout_cv'](bundle,X,m,cfg,changed,tmp_path/'runs')
    assert next_result['run_folder']!=result['run_folder']


def test_interrupted_epoch_resume_reproduces_uninterrupted_training(tmp_path):
    ns,cfg,bundle,m,X,d=fixture_run(tmp_path)
    f=m['folds'][0];contract={'technical_fixture':'same_data_same_code'}
    args=('MLP_TARGET_CENTER',X,bundle.y,f['train'],f['validation'],cfg,d,43)
    expected=ns['fit_readout'](*args,tmp_path/'whole.json',contract)
    original=ns['write_progress'];calls=[]
    def interrupted(path,payload,identity):
        original(path,payload,identity);calls.append(payload['epoch'])
        if len(calls)==1:raise InterruptedError('technical fixture interrupts after an intact epoch')
    ns['write_progress']=interrupted
    with pytest.raises(InterruptedError):ns['fit_readout'](*args,tmp_path/'resume.json',contract)
    ns['write_progress']=original
    actual=ns['fit_readout'](*args,tmp_path/'resume.json',contract)
    assert actual['history']==expected['history'] and actual['best_epoch']==expected['best_epoch']
    torch.testing.assert_close(actual['prediction'],expected['prediction'],atol=0,rtol=0)
    for k in actual['state']:torch.testing.assert_close(actual['state'][k],expected['state'][k],atol=0,rtol=0)
