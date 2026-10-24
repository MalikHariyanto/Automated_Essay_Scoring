"""E8: pemilihan LR, pasangan input dan retensi checkpoint; bukan skor AES penelitian."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import torch
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
import test_context_pair_notebook as context

NOTEBOOK=ROOT/'notebooks/experiments'/os.environ.get('AES_CONTEXT_NOTEBOOK','AES_300_E8_ContextPair_LR5e6_GPU.ipynb')


def runtime(monkeypatch):
    monkeypatch.setattr(context,'NOTEBOOK',NOTEBOOK)
    return context.runtime()


def test_exact_fingerprint_and_unchanged_pair_pipeline():
    n=json.loads(NOTEBOOK.read_text());base=json.loads((NOTEBOOK.parent/'AES_300_S3_ContextPair_MLP.ipynb').read_text())
    fingerprint=n['metadata']['aes_standalone']['code_sha256'];codes=[]
    for cell in n['cells']:
        if cell['cell_type']=='code':
            code=''.join(cell['source']).removeprefix('%%aes\n').replace(fingerprint,'__NOTEBOOK_CODE_SHA256__');ast.parse(code);codes.append(code)
    assert hashlib.sha256('\n'.join(codes).encode()).hexdigest()==fingerprint
    identities=[]
    for cell in n['cells']:
        if cell['cell_type']!='code':continue
        for node in ast.parse(''.join(cell['source']).removeprefix('%%aes\n')).body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='NOTEBOOK_CODE_SHA256' for t in node.targets):
                identities.append(ast.literal_eval(node.value))
    assert len(identities)==2 and all(value==fingerprint for value in identities)
    # Tokennya, pasangan input, vektor, model, split dan scorer tidak diubah oleh kandidat LR.
    for index in (14,18,20,21,22,24,26,28,30,31,35,38,39,41,43):
        assert ''.join(n['cells'][index]['source'])==''.join(base['cells'][index]['source']),index


@pytest.mark.parametrize('scenario,expected',[('S1',1e-5),('S2',None),('S3',5e-6)])
def test_optimizer_rates_with_actual_heads(monkeypatch,scenario,expected):
    ns=runtime(monkeypatch);cfg=ns['config']({});encoder=torch.nn.Embedding(64,768) if scenario!='S2' else None
    model=ns['SingleScoreModel'](scenario,encoder)
    groups=ns['optimizer_groups'](model,cfg);rates={g['name']:g['lr'] for g in groups}
    assert rates.get('roberta')==expected and rates['head']==1e-4
    if scenario=='S3':assert rates['projection']==1e-4 and all(p.requires_grad for p in model.roberta.parameters())
    parameters=[p for g in groups for p in g['params']];assert len({id(p) for p in parameters})==len(parameters)


def test_slot_corruption_falls_back_without_deleting_latest_good_file(monkeypatch,tmp_path):
    ns=runtime(monkeypatch);path=tmp_path/'progress_fold_1.json';contract={'scope':'TECHNICAL_FIXTURE_ONLY'}
    for epoch in range(1,7):ns['save_training_progress'](path,{'epoch':epoch,'contract':contract})
    assert len(list(tmp_path.glob('*.pt')))==2
    assert ns['load_training_progress'](path,contract)['epoch']==6
    metadata=json.loads(path.read_text());latest=tmp_path/metadata['snapshots'][0]['file'];latest.write_bytes(b'TECHNICAL INTERRUPTED WRITE')
    assert ns['load_training_progress'](path,contract)['epoch']==5
    assert len(list(tmp_path.glob('*.pt')))==2


def test_published_device_configuration_and_gpu_source_preserved(monkeypatch,tmp_path):
    cpu=NOTEBOOK.name.endswith('_CPU.ipynb')
    ns=runtime(monkeypatch);checks=[]
    monkeypatch.setenv('AES300_DATASET',str(tmp_path/'TECHNICAL_ONLY.xlsx'))
    monkeypatch.setenv('AES300_RESULTS',str(tmp_path/'results'))
    ns['check_gpu']=lambda require_gpu:checks.append(require_gpu) or {'name':'TECHNICAL DEVICE CHECK','cuda_available':not cpu,'is_t4':not cpu,'memory_gib':16}
    n=json.loads(NOTEBOOK.read_text());exec(''.join(n['cells'][12]['source']).removeprefix('%%aes\n'),ns)
    assert checks==[not cpu] and ns['cfg']['device']==('cpu' if cpu else 'auto')
    assert ns['cfg']['s3_encoder_lr']==5e-6 and ns['cfg']['head_lr']==1e-4
    if cpu:
        assert "'CUDA_VISIBLE_DEVICES':''" in ''.join(n['cells'][7]['source'])
        gpu=NOTEBOOK.with_name('AES_300_E8_ContextPair_LR5e6_GPU.ipynb')
        assert hashlib.sha256(gpu.read_bytes()).hexdigest()==n['metadata']['aes_experiment']['gpu_notebook_sha256']
