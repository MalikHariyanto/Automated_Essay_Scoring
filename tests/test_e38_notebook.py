"""E38 MLP-only dropout wiring and native execution; technical fixtures only."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import nbformat
import numpy as np
import pandas as pd
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
import test_context_pair_notebook as context
from test_e36_notebook import execute_full_native_fixture
from test_e6_three_input_notebook import ThreeTechnicalTokenizer
from test_teacher_score import make_book,encoders

NAME='AES_300_E38_E30_S3_MLPDropout02_Native_Learnable_GPU.ipynb'
NOTEBOOK=ROOT/'notebooks/experiments'/NAME
BASE=ROOT/'notebooks/experiments/AES_300_E30_E26_BiGRU_CoAttention_Pooling_GPU.ipynb'
FORMAT=ROOT/'notebooks/experiments/AES_300_E36_E30_S3_EncoderLR3e6_Native_Learnable_GPU.ipynb'


def native_runtime():
    n=nbformat.read(NOTEBOOK,as_version=4)
    module=types.ModuleType('e38_native_fixture');sys.modules[module.__name__]=module;ns=module.__dict__
    safe={'imports','model_names','input_columns','hyperparameters','runtime_flags',
          'configuration_defaults','presentation_helpers','word_helpers'}
    for c in n.cells:
        if c.cell_type=='code' and c.metadata.aes38_role in safe:exec(compile(c.source,str(NOTEBOOK),'exec'),ns)
    for c in n.cells:
        if c.cell_type=='code':
            definitions=[v for v in ast.parse(c.source).body if isinstance(v,(ast.FunctionDef,ast.ClassDef))]
            exec(compile(ast.Module(body=definitions,type_ignores=[]),str(NOTEBOOK),'exec'),ns)
    ns['show']=lambda *a,**k:None
    return ns


def test_native_structure_five_folds_only_e30_core_and_deterministic_builder():
    raw=NOTEBOOK.read_bytes();n=nbformat.reads(raw.decode(),as_version=4);nbformat.validate(n)
    code=[c for c in n.cells if c.cell_type=='code'];assert len(code)==89
    meta=n.metadata.aes_experiment
    assert meta.cv_fits==5 and meta.refits==1 and meta.scenarios==['S3']
    assert meta.s3_encoder_lr==5e-6 and meta.s3_mlp_dropout==.2 and not meta.multiple_seed_runs
    fingerprint=n.metadata.aes_standalone.code_sha256
    assert hashlib.sha256('\n'.join(c.source.replace(fingerprint,'__NOTEBOOK_CODE_SHA256__') for c in code).encode()).hexdigest()==fingerprint
    text='\n'.join(c.source for c in code)
    assert all(s not in text for s in ('cosine','%%aes','WORKER_SOURCE','run_paired_cv','TRAINING_SEEDS'))
    for c in code:
        assert not c.outputs and c.execution_count is None
        tree=ast.parse(c.source)
        assert not any(line.lstrip().startswith(('%','!')) for line in c.source.splitlines())
        assert not any(isinstance(v,ast.Call) and isinstance(v.func,ast.Name) and v.func.id in ('exec','eval') for v in ast.walk(tree))
    def definitions(path):
        return {v.name:ast.dump(v,include_attributes=False)
                for c in nbformat.read(path,as_version=4).cells if c.cell_type=='code'
                for v in ast.parse(c.source.removeprefix('%%aes\n')).body
                if isinstance(v,(ast.FunctionDef,ast.ClassDef))}
    a,b=definitions(BASE),definitions(NOTEBOOK)
    for key in ('AnswerBiGRUAttention','AnswerBiGRUCoAttentionPool','Engine','fit','train_all',
                'optimizer_groups','make_split','bge_features','metrics','gwet_ac2','save_model'):
        assert a[key]==b[key],key
    historical={p:p.read_bytes() for p in (BASE,FORMAT)}
    subprocess.run([sys.executable,'scripts/build_e38_notebook.py'],cwd=ROOT,check=True,capture_output=True)
    assert raw==NOTEBOOK.read_bytes()==(ROOT/'aes-learning/notebooks/experiments'/NAME).read_bytes()
    assert all(p.read_bytes()==value for p,value in historical.items())


def test_only_mlp_dropout_changes_initial_weights_eval_predictions_and_lr_stay_e30(tmp_path,monkeypatch,encoders):
    ns=native_runtime();monkeypatch.setattr(context,'NOTEBOOK',BASE);old=context.runtime()
    for env in (ns,old):
        env['encoder']=encoders;env['tokenizer']=lambda *a,**k:ThreeTechnicalTokenizer()
    file=tmp_path/'TECHNICAL_FIXTURE_ONLY.xlsx';make_book(file)
    values=dict(dataset=str(file),output=str(tmp_path/'run'),expected_rows=24,device='cpu',
                roberta_max_length=32,roberta_overlap=4,bge_max_length=32,bge_overlap=4,
                context_max_tokens=12,question_max_tokens=4)
    cfg=ns['config'](values);control_cfg=old['config'](values)
    for key in old['DEFAULTS']:assert cfg[key]==control_cfg[key],key
    assert cfg['s3_mlp_dropout']==.2 and cfg['dropout']==.1
    for invalid in [-.1,1.,float('nan')]:
        with pytest.raises(ValueError,match='s3_mlp_dropout'):ns['config']({**values,'s3_mlp_dropout':invalid})
    bundle=ns['read_dataset'](cfg)
    assert ns['make_split'](bundle,cfg)==old['make_split'](bundle,control_cfg)
    ns['seed_everything'](71);model=ns['make_model']('S3',cfg)
    old['seed_everything'](71);control=old['make_model']('S3',control_cfg)
    for key,value in model.state_dict().items():torch.testing.assert_close(value,control.state_dict()[key],rtol=0,atol=0)
    dropout=lambda m:{name:layer.p for name,layer in m.named_modules() if isinstance(layer,torch.nn.Dropout)}
    new_ps,old_ps=dropout(model),dropout(control)
    assert set(new_ps)==set(old_ps)
    assert [key for key in new_ps if new_ps[key]!=old_ps[key]]==['head.2']
    assert new_ps['head.2']==.2 and old_ps['head.2']==.1
    assert {g['name']:g['lr'] for g in ns['optimizer_groups'](model,cfg)}=={'roberta':5e-6,'projection':1e-4,'head':1e-4,'pooling_norm':1e-4}
    torch.set_num_threads(2);bge=torch.randn(24,1024);predictions=[]
    for env,current,settings in ((ns,model,cfg),(old,control,control_cfg)):
        current.eval()
        engine=env['Engine'](current,bundle.texts,bge,settings,'cpu',contexts=env['contexts_from_bundle'](bundle))
        with torch.no_grad():predictions.append(engine.forward([0,1]))
    torch.testing.assert_close(*predictions,rtol=0,atol=0)
    # A probability is not a state_dict tensor: reload must validate its config too.
    path=tmp_path/'TECHNICAL_checkpoint.pt'
    torch.save({'scope':ns['SCOPE'],'trained':True,'scenario':'S3','config':cfg,'state':model.state_dict()},path)
    loaded,_=ns['load_model'](path,cfg)
    assert loaded.head[2].p==.2 and not loaded.training
    with pytest.raises(ValueError,match='s3_mlp_dropout'):ns['load_model'](path,{**cfg,'s3_mlp_dropout':.1})
    for scenario in ('S1','S2'):
        with pytest.raises(ValueError):ns['make_model'](scenario,cfg)


def test_full_e38_native_notebook_exported_dropout_and_fresh_kernel_resume(tmp_path):
    execute_full_native_fixture(tmp_path,NOTEBOOK,'aes38_role',5e-6)
    info=json.loads((tmp_path/'standalone/workspace/outputs/dataset300_download.json').read_text())
    run=Path(info['run_folder']);cfg=json.loads((run/'effective_config.json').read_text())
    assert cfg['s3_mlp_dropout']==.2 and cfg['dropout']==.1
    settings=pd.read_excel(run/'hasil_ringkas/05_Riwayat_Pelatihan.xlsx',sheet_name='Pengaturan Eksperimen')
    settings=dict(zip(settings.Pengaturan,settings.Nilai))
    assert settings['Dropout MLP S3']==.2 and settings['Dropout parameter lama']==.1
    assert settings['Seed training tetap']==42
    checkpoint=torch.load(run/'S3/final.pt',map_location='cpu',weights_only=True)
    assert checkpoint['config']['s3_mlp_dropout']==.2
    assert Path(info['archive']).name=='Hasil_AES_300_E38_E30_S3_MLPDropout02_Native_Learnable_GPU.zip'
