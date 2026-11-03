"""E39 capacity ablation, E30 equivalence at hidden64, and standalone execution."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import nbformat
import pandas as pd
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
import test_context_pair_notebook as context
from test_e36_notebook import execute_full_native_fixture
from test_e6_three_input_notebook import ThreeTechnicalTokenizer
from test_teacher_score import make_book,encoders

NAME='AES_300_E39_E30_S3_BiGRU128_Native_Learnable_GPU.ipynb'
NOTEBOOK=ROOT/'notebooks/experiments'/NAME
BASE=ROOT/'notebooks/experiments/AES_300_E30_E26_BiGRU_CoAttention_Pooling_GPU.ipynb'
FORMAT=ROOT/'notebooks/experiments/AES_300_E38_E30_S3_MLPDropout02_Native_Learnable_GPU.ipynb'


def native_runtime():
    n=nbformat.read(NOTEBOOK,as_version=4)
    module=types.ModuleType('e39_native_fixture');sys.modules[module.__name__]=module;ns=module.__dict__
    safe={'imports','model_names','input_columns','hyperparameters','runtime_flags',
          'configuration_defaults','presentation_helpers','word_helpers'}
    for c in n.cells:
        if c.cell_type=='code' and c.metadata.aes39_role in safe:exec(compile(c.source,str(NOTEBOOK),'exec'),ns)
    for c in n.cells:
        if c.cell_type=='code':
            definitions=[v for v in ast.parse(c.source).body if isinstance(v,(ast.FunctionDef,ast.ClassDef))]
            exec(compile(ast.Module(body=definitions,type_ignores=[]),str(NOTEBOOK),'exec'),ns)
    ns['show']=lambda *a,**k:None
    return ns


def test_native_s3_only_hash_and_e30_training_protocol_preserved():
    raw=NOTEBOOK.read_bytes();n=nbformat.reads(raw.decode(),as_version=4);nbformat.validate(n)
    code=[c for c in n.cells if c.cell_type=='code'];assert len(code)==89
    meta=n.metadata.aes_experiment
    assert meta.scenarios==['S3'] and meta.cv_fits==5 and meta.refits==1 and not meta.multiple_seed_runs
    assert meta.s3_encoder_lr==5e-6 and meta.s3_mlp_dropout==.1 and meta.s3_sequence_hidden==128
    assert meta.pooling_attention_hidden==64 and meta.sequence_output_dimensions==256
    assert meta.format_sha256==hashlib.sha256(FORMAT.read_bytes()).hexdigest()
    fingerprint=n.metadata.aes_standalone.code_sha256
    assert hashlib.sha256('\n'.join(c.source.replace(fingerprint,'__NOTEBOOK_CODE_SHA256__') for c in code).encode()).hexdigest()==fingerprint
    text='\n'.join(c.source for c in code)
    assert all(s not in text for s in ('%%aes','WORKER_SOURCE','cosine','run_paired_cv','TRAINING_SEEDS','s2_mlp_lr'))
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
    for name in ['fit','train_all','make_split','metrics','gwet_ac2','bge_features',
                 'pair_content_windows','pair_batch_windows','save_model','answer_token_mask','reference_token_mask']:
        assert a[name]==b[name],name
    historical={p:p.read_bytes() for p in [BASE,FORMAT]}
    subprocess.run([sys.executable,'scripts/build_e39_notebook.py'],cwd=ROOT,check=True,capture_output=True)
    assert NOTEBOOK.read_bytes()==raw==(ROOT/'aes-learning/notebooks/experiments'/NAME).read_bytes()
    assert all(p.read_bytes()==content for p,content in historical.items())


def test_s3_only_cleanup_is_identical_to_e30_at_hidden64_and_wide_model_has_valid_gradients(tmp_path,monkeypatch,encoders):
    ns=native_runtime();monkeypatch.setattr(context,'NOTEBOOK',BASE);old=context.runtime()
    for env in (ns,old):
        env['encoder']=encoders;env['tokenizer']=lambda *a,**k:ThreeTechnicalTokenizer()
    path=tmp_path/'TECHNICAL_FIXTURE_ONLY.xlsx';make_book(path)
    values=dict(dataset=str(path),output=str(tmp_path/'run'),expected_rows=24,device='cpu',
                roberta_max_length=32,roberta_overlap=4,bge_max_length=32,bge_overlap=4,
                context_max_tokens=12,question_max_tokens=4)
    control_cfg=old['config'](values)
    cfg64=ns['config']({**values,'s3_sequence_hidden':64});cfg128=ns['config'](values)
    for key in old['DEFAULTS']:
        if key!='s2_mlp_lr':assert cfg64[key]==control_cfg[key],key
    assert cfg128['s3_sequence_hidden']==128 and cfg128['pooling_attention_hidden']==64
    for invalid in [0,-1,1.5]:
        with pytest.raises(ValueError,match='pooling_attention_hidden'):ns['config']({**values,'pooling_attention_hidden':invalid})
    bundle=ns['read_dataset'](cfg64)
    assert ns['make_split'](bundle,cfg64)==ns['make_split'](bundle,cfg128)==old['make_split'](bundle,control_cfg)
    ns['seed_everything'](71);model64=ns['make_model']('S3',cfg64).eval()
    old['seed_everything'](71);control=old['make_model']('S3',control_cfg).eval()
    for k,v in model64.state_dict().items():torch.testing.assert_close(v,control.state_dict()[k],rtol=0,atol=0)
    torch.set_num_threads(2);bge=torch.randn(24,1024);predictions=[];initial={k:v.clone() for k,v in model64.state_dict().items()}
    for env,model,cfg in [(ns,model64,cfg64),(old,control,control_cfg)]:
        engine=env['Engine'](model,bundle.texts,bge,cfg,'cpu',contexts=env['contexts_from_bundle'](bundle))
        pred=engine.forward([0,1]);predictions.append(pred.detach())
        (pred-.8).square().mean().backward()
        torch.optim.AdamW(env['optimizer_groups'](model,cfg),weight_decay=.01).step()
    torch.testing.assert_close(*predictions,rtol=0,atol=0)
    for k,v in model64.state_dict().items():torch.testing.assert_close(v,control.state_dict()[k],rtol=0,atol=0)
    ns['seed_everything'](71);wide=ns['make_model']('S3',cfg128).eval()
    for k,v in wide.state_dict().items():
        if not k.startswith('sequence_pool.'):torch.testing.assert_close(v,initial[k],rtol=0,atol=0)
    pool=wide.sequence_pool
    assert pool.gru.hidden_size==128 and pool.gru.bidirectional and pool.gru.num_layers==1
    assert pool.reference_projection.out_features==256
    assert (pool.matching_projection.in_features,pool.matching_projection.out_features)==(1280,256)
    assert (pool.attention_hidden.in_features,pool.attention_hidden.out_features)==(256,64)
    assert (pool.output.in_features,pool.output.out_features)==(256,768)
    assert wide.head[0].in_features==1024 and wide.head[0].out_features==256 and wide.head[2].p==.1
    assert sum(p.numel() for p in wide.parameters())-sum(p.numel() for p in control.parameters())==820224
    engine=ns['Engine'](wide,bundle.texts,bge,cfg128,'cpu',contexts=ns['contexts_from_bundle'](bundle))
    pred,features=engine.forward([0,1],return_features=True)
    assert pred.shape==(2,1) and features['bigru_summary'].shape==(2,256)
    assert features['answer_pool768'].shape==(2,768) and features['concat'].shape==(2,1024)
    weights=pool.last_weights;mask=engine.pooling_batch[1]
    torch.testing.assert_close(weights.sum(1),torch.ones(len(weights)),rtol=1e-6,atol=1e-6)
    assert not weights[~mask].any()
    (pred-.8).square().mean().backward()
    for layer in [pool.gru.weight_ih_l0,pool.gru.weight_ih_l0_reverse,pool.reference_projection.weight,
                  pool.matching_projection.weight,pool.attention_hidden.weight,wide.roberta.embeddings.word_embeddings.weight]:
        assert layer.grad is not None and torch.isfinite(layer.grad).all() and layer.grad.abs().sum()>0
    assert {g['name']:g['lr'] for g in ns['optimizer_groups'](wide,cfg128)}=={'roberta':5e-6,'projection':1e-4,'head':1e-4,'pooling_norm':1e-4}
    checkpoint=tmp_path/'TECHNICAL_wide.pt'
    torch.save({'scope':ns['SCOPE'],'trained':True,'scenario':'S3','config':cfg128,'state':wide.state_dict()},checkpoint)
    loaded,_=ns['load_model'](checkpoint,cfg128);assert loaded.sequence_pool.gru.hidden_size==128
    with pytest.raises(ValueError,match='s3_sequence_hidden'):ns['load_model'](checkpoint,cfg64)
    for scenario in ['S1','S2']:
        with pytest.raises(ValueError):ns['make_model'](scenario,cfg128)


def test_full_e39_native_notebook_256_component_export_and_fresh_kernel_resume(tmp_path):
    execute_full_native_fixture(tmp_path,NOTEBOOK,'aes39_role',5e-6)
    info=json.loads((tmp_path/'standalone/workspace/outputs/dataset300_download.json').read_text())
    run=Path(info['run_folder']);cfg=json.loads((run/'effective_config.json').read_text())
    assert cfg['s3_sequence_hidden']==128 and cfg['pooling_attention_hidden']==64 and cfg['s3_mlp_dropout']==.1
    assert 's2_mlp_lr' not in cfg
    settings=pd.read_excel(run/'hasil_ringkas/05_Riwayat_Pelatihan.xlsx',sheet_name='Pengaturan Eksperimen')
    settings=dict(zip(settings.Pengaturan,settings.Nilai))
    assert settings['Hidden pooling S3']==128 and settings['Hidden attention pooling']==64 and settings['Dropout MLP S3']==.1
    vector_file=run/'hasil_ringkas/02_Vektor_Jawaban.xlsx'
    for sheet,dim in [('Ringkasan CoAttn Pooling S3',256),('Pooling Jawaban S3',768),('Fusion S3',1024)]:
        frame=pd.read_excel(vector_file,sheet_name=sheet)
        assert frame.shape==(24,dim+1)
        assert frame.iloc[:,1:].notna().all().all()
    checkpoint=torch.load(run/'S3/final.pt',map_location='cpu',weights_only=True)
    assert checkpoint['config']['s3_sequence_hidden']==128
    assert Path(info['archive']).name=='Hasil_AES_300_E39_E30_S3_BiGRU128_Native_Learnable_GPU.zip'
