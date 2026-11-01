"""E34: only pooling_norm S3 LR changes; all other updates match E33."""
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
import test_context_pair_notebook as context
import test_e21_e22_notebooks as full
from test_e6_three_input_notebook import ThreeTechnicalTokenizer
from test_teacher_score import encoders,make_book
NAME='AES_300_E34_E33_BiGRU_CoAttention_LayersLR5e5_GPU.ipynb'
NOTEBOOK=ROOT/'notebooks/experiments'/NAME
BASE=ROOT/'notebooks/experiments/AES_300_E33_E30_BiGRU_CoAttention_MLPLR5e5_GPU.ipynb'


def test_source_parity_metadata_identity_and_builder():
    raw=NOTEBOOK.read_bytes();n=nbformat.reads(raw.decode(),as_version=4);nbformat.validate(n)
    old=nbformat.read(BASE,as_version=4);before=BASE.read_bytes()
    fp=n.metadata.aes_standalone.code_sha256;oldfp=old.metadata.aes_standalone.code_sha256
    assert len(n.cells)==47 and sum(c.cell_type=='code' for c in n.cells)==30
    assert n.metadata.aes_experiment.s3_mlp_lr==5e-5 and n.metadata.aes_experiment.s2_mlp_lr==1e-4
    assert n.metadata.aes_experiment.s3_layers_lr==5e-5
    assert n.metadata.aes_experiment.projection_lr==1e-4
    assert n.metadata.aes_experiment.base_sha256==hashlib.sha256(before).hexdigest()
    codes=[];changes=[]
    for i,(c,b) in enumerate(zip(n.cells,old.cells)):
        if c.cell_type!='code':continue
        assert not c.outputs and c.execution_count is None
        s=c.source.removeprefix('%%aes\n');ast.parse(s);codes.append(s.replace(fp,'__NOTEBOOK_CODE_SHA256__'))
        normalized=c.source.replace(fp,'HASH').replace(Path(NAME).stem,'NOTEBOOK')
        original=b.source.replace(oldfp,'HASH').replace(BASE.stem,'NOTEBOOK')
        if normalized!=original:changes.append(i)
    assert changes==[10,12]  # default LR and editable configuration/logging only.
    assert hashlib.sha256('\n'.join(codes).encode()).hexdigest()==fp
    assert 'EPOCHS,FOLDS=25,5' in n.cells[12].source
    assert 'EARLY_STOPPING_PATIENCE=5' in n.cells[12].source
    assert "device='cuda'" in n.cells[12].source and 'gpu=check_gpu(require_gpu=True)' in n.cells[12].source
    subprocess.run([sys.executable,'scripts/build_e34_notebook.py'],cwd=ROOT,check=True,capture_output=True)
    assert NOTEBOOK.read_bytes()==raw==(ROOT/'aes-learning/notebooks/experiments'/NAME).read_bytes()
    assert BASE.read_bytes()==before


@pytest.mark.parametrize('scenario',['S1','S2','S3'])
def test_optimizer_first_step_only_changes_s3_pooling_norm(scenario,tmp_path,monkeypatch,encoders):
    monkeypatch.setattr(context,'NOTEBOOK',NOTEBOOK);ns=context.runtime()
    ns['encoder']=encoders;ns['tokenizer']=lambda *a,**k:ThreeTechnicalTokenizer()
    path=tmp_path/'TECHNICAL_FIXTURE_ONLY.xlsx';make_book(path)
    cfg=ns['config']({'dataset':str(path),'output':str(tmp_path/'out'),'expected_rows':24,'device':'cpu',
        'dropout':0.,'roberta_max_length':32,'roberta_overlap':4,'bge_max_length':32,'bge_overlap':4,
        'context_max_tokens':12,'question_max_tokens':4})
    bundle=ns['read_dataset'](cfg);torch.set_num_threads(2);torch.manual_seed(71);bge=torch.randn(24,1024)
    model=ns['make_model'](scenario,cfg).eval();control=ns['make_model'](scenario,cfg).eval()
    control.load_state_dict(model.state_dict());before={k:v.clone() for k,v in model.state_dict().items()}
    candidates=[]
    for m,config in [(model,cfg),(control,{**cfg,'s3_layers_lr':1e-4})]:
        groups=ns['optimizer_groups'](m,config);ids={id(p):g['lr'] for g in groups for p in g['params']}
        assert len(ids)==sum(len(g['params']) for g in groups)==sum(p.requires_grad for p in m.parameters())
        if m is model:
            for name,p in m.named_parameters():
                if p.requires_grad:
                    rate=5e-6 if name.startswith('roberta.') else 5e-5 if scenario=='S3' and name.startswith(('head.','sequence_pool.','roberta_norm.')) else 1e-4
                    assert ids[id(p)]==rate,name
        engine=ns['Engine'](m,bundle.texts,bge,config,'cpu',contexts=ns['contexts_from_bundle'](bundle))
        p=engine.forward([0,1]);candidates.append(p.detach())
        (p-.8).square().mean().backward();torch.optim.AdamW(groups,weight_decay=.01).step()
    torch.testing.assert_close(candidates[0],candidates[1],rtol=0,atol=0)
    for key,value in model.state_dict().items():
        if scenario!='S3' or not key.startswith(('sequence_pool.','roberta_norm.')):
            torch.testing.assert_close(value,control.state_dict()[key],rtol=0,atol=0)
    if scenario=='S3':
        key='sequence_pool.reference_projection.weight'
        small=(model.state_dict()[key]-before[key]).norm();large=(control.state_dict()[key]-before[key]).norm()
        assert float(large/small)==pytest.approx(2.,rel=.002)
        assert model.sequence_pool.reference_projection.weight.grad.abs().sum()>0
        assert model.sequence_pool.gru.weight_ih_l0.grad.abs().sum()>0


def test_full_notebook_excel_actual_lr_and_resume(tmp_path,monkeypatch):
    monkeypatch.setattr(full,'NAMES',{'single_sequence':NAME});monkeypatch.setattr(full,'GPU',True)
    full.test_full_notebook_thirty_cells_and_resume_same_local_folder('single_sequence',tmp_path)
    import pandas as pd
    info=json.loads((tmp_path/'standalone/workspace/outputs/dataset300_download.json').read_text())
    folder=Path(info['run_folder']);cfg=json.loads((folder/'effective_config.json').read_text())
    assert cfg['s3_mlp_lr']==5e-5 and cfg['s2_mlp_lr']==1e-4
    assert cfg['s3_layers_lr']==5e-5
    book=pd.read_excel(folder/'hasil_ringkas/05_Riwayat_Pelatihan.xlsx',sheet_name='Pengaturan Eksperimen')
    settings=dict(zip(book.Pengaturan,book.Nilai))
    assert float(settings['LR MLP S3'])==5e-5 and float(settings['LR MLP S2'])==1e-4
    assert float(settings['LR pooling dan norm S3'])==5e-5
    for scenario in ('S1','S2','S3'):
        ck=torch.load(folder/scenario/'final.pt',map_location='cpu',weights_only=True)
        rates=ck['training']['optimizer_learning_rates']
        assert rates['head']==(5e-5 if scenario=='S3' else 1e-4)
        if scenario=='S3':
            assert rates['pooling_norm']==5e-5 and rates['projection']==1e-4
