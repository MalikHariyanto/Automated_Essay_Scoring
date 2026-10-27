"""Uji layer dan notebook lengkap dengan fixture teknis, bukan hasil penelitian."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
import test_context_pair_notebook as context
import test_e21_e22_notebooks as base
from test_e6_three_input_notebook import ThreeTechnicalTokenizer
from test_teacher_score import make_book, encoders

NAMES = {'E23':'AES_300_E23_Tiga_Input_BiGRU_GPU.ipynb',
         'E24':'AES_300_E24_Tiga_Input_AttentionPooling_GPU.ipynb'}


@pytest.fixture(params=NAMES)
def runtime(request, monkeypatch, tmp_path):
    experiment = request.param
    monkeypatch.setattr(context, 'NOTEBOOK', ROOT / 'notebooks/experiments' / NAMES[experiment])
    ns = context.runtime()
    path = tmp_path / 'TECHNICAL_FIXTURE_ONLY.xlsx'
    make_book(path)
    cfg = ns['config']({'dataset':str(path),'output':str(tmp_path/'results'), 'expected_rows':24,
        'device':'cpu','epochs':1,'effective_batch':4,'micro_batch':1,'dropout':0.,
        'roberta_max_length':32,'roberta_overlap':4,'bge_max_length':32,'bge_overlap':4,
        'context_max_tokens':12,'question_max_tokens':4})
    ns['tokenizer'] = lambda *a, **k: ThreeTechnicalTokenizer()
    return experiment, ns, cfg, ns['read_dataset'](cfg)


def test_clean_sources_deterministic_and_e21_unchanged():
    files = {e:ROOT/'notebooks/experiments'/n for e,n in NAMES.items()}
    originals = {e:p.read_bytes() for e,p in files.items()}
    e21 = ROOT/'notebooks/experiments/AES_300_E21_Tiga_Input_GPU.ipynb'
    e21bytes = e21.read_bytes()
    subprocess.run([sys.executable,'scripts/build_e23_e24_notebooks.py'],cwd=ROOT,check=True,capture_output=True)
    assert e21.read_bytes() == e21bytes
    for e,path in files.items():
        assert path.read_bytes() == originals[e]
        assert (ROOT/'aes-learning/notebooks/experiments'/path.name).read_bytes() == originals[e]
        n = json.loads(originals[e]);meta = n['metadata']['aes_experiment']
        assert meta['name'] == e and meta['input_format'] == 'single_sequence'
        assert meta['target'] == 'rater' and len(meta['input_columns']) == 3
        assert meta['drive_mount'] is False and meta['drive_autosave'] is False
        assert meta['new_layer_lr'] == meta['head_lr'] == 1e-4
        fingerprint = n['metadata']['aes_standalone']['code_sha256'];codes=[]
        for c in n['cells']:
            if c['cell_type'] != 'code':continue
            assert c['outputs'] == [] and c['execution_count'] is None
            code = ''.join(c['source']).removeprefix('%%aes\n');ast.parse(code)
            assert not any(k in code for k in ['drive.mount','googleapiclient','from aes.'])
            codes.append(code.replace(fingerprint,'__NOTEBOOK_CODE_SHA256__'))
        assert len(codes) == 30 and hashlib.sha256('\n'.join(codes).encode()).hexdigest() == fingerprint
        assert n['metadata']['accelerator'] == 'GPU'
        assert "device='cuda'" in ''.join(n['cells'][12]['source'])
        assert 'require_gpu=True' in ''.join(n['cells'][12]['source'])


@pytest.mark.parametrize('scenario',['S1','S2','S3'])
def test_shapes_gradients_lr_controls_and_reload(runtime,encoders,scenario,tmp_path):
    e,ns,cfg,bundle = runtime
    ns['encoder'] = encoders
    bge = ns['bge_features'](bundle,cfg,Path(cfg['output'])/'run')
    assert not bge.requires_grad
    torch.manual_seed(4)
    model = ns['make_model'](scenario,cfg).eval()
    engine = ns['Engine'](model,bundle.texts,bge,cfg,'cpu',
        roberta_tokenizer=ThreeTechnicalTokenizer(),contexts=ns['contexts_from_bundle'](bundle))
    prediction,features = engine.forward([0,1],True)
    assert prediction.shape == (2,1) and torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    rates = {g['name']:g['lr'] for g in ns['optimizer_groups'](model,cfg)}
    assert rates['head'] == 1e-4
    if scenario == 'S3':
        assert rates['roberta'] == 5e-6 and rates['projection'] == 1e-4
        assert features['concat'].shape == (2,1024) and features['answer_pool768'].shape == (2,768)
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.sequence_pool.parameters())
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.roberta.parameters())
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.projection.parameters())
    else:
        assert not hasattr(model,'sequence_pool')
        if scenario == 'S1':assert isinstance(model.head,torch.nn.Linear)
        else:assert isinstance(model.head,torch.nn.Sequential)
        # Bandingkan kontrol dengan kelas E21 asli, termasuk inisialisasi dan forward.
        original=json.loads(subprocess.check_output(['git','show',
            'ef712b4c035d96178d9baa67caa7453ac9fd8738:notebooks/experiments/AES_300_E21_Tiga_Input_GPU.ipynb'],cwd=ROOT))
        tree=ast.parse(''.join(original['cells'][30]['source']).removeprefix('%%aes\n'))
        oldns=dict(ns)
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='SingleScoreModel')
        exec(compile(ast.Module(body=[cls],type_ignores=[]),'E21_original','exec'),oldns)
        torch.manual_seed(4)
        old=oldns['SingleScoreModel'](scenario,encoders(cfg,'roberta') if scenario=='S1' else None,dropout=0.).eval()
        assert all(torch.equal(value,old.state_dict()[name]) for name,value in model.state_dict().items())
        control=ns['Engine'](old,bundle.texts,bge,cfg,'cpu',roberta_tokenizer=ThreeTechnicalTokenizer(),contexts=ns['contexts_from_bundle'](bundle))
        # Engine baru membawa keyword mask yang tidak diperlukan oleh kelas E21.
        if scenario=='S1':
            windows,mapping,_=ns['pair_batch_windows'](bundle.texts[:2],ns['contexts_from_bundle'](bundle)[:2],ThreeTechnicalTokenizer(),'cpu',32,4,cfg)
            previous=old(windows=windows,mapping=mapping,n_essays=2)
        else:previous=old(bge=bge[:2])
        torch.testing.assert_close(prediction,previous,rtol=0,atol=0)
    copy = ns['make_model'](scenario,cfg).eval()
    path=tmp_path/'technical_state.pt';torch.save(model.state_dict(),path)
    copy.load_state_dict(torch.load(path,weights_only=True))
    restored = ns['Engine'](copy,bundle.texts,bge,cfg,'cpu',
        roberta_tokenizer=ThreeTechnicalTokenizer(),contexts=ns['contexts_from_bundle'](bundle))
    torch.testing.assert_close(prediction,restored.forward([0,1]),rtol=0,atol=0)


def test_pool_mask_padding_context_invariance_and_sequence_axis(runtime):
    e,ns,cfg,_=runtime
    cls=ns['AnswerBiGRU'] if e=='E23' else ns['AnswerAttentionPool']
    torch.manual_seed(21);pool=cls(64).eval()
    hidden=torch.randn(2,9,768,requires_grad=True)
    mask=torch.tensor([[0,0,1,1,1,0,0,0,0],[0,1,1,1,1,1,1,0,0]],dtype=torch.bool)
    out=pool(hidden,mask)
    altered=hidden.detach().clone();altered[~mask]=10000
    torch.testing.assert_close(out,pool(altered,mask),rtol=0,atol=0)
    individual=torch.cat([pool(row[keep][None],torch.ones(1,int(keep.sum()),dtype=torch.bool))
                          for row,keep in zip(hidden,mask)])
    torch.testing.assert_close(out,individual,rtol=1e-5,atol=1e-6)
    out.square().mean().backward()
    assert hidden.grad[~mask].abs().sum()==0 and hidden.grad[mask].abs().sum()>0
    if e=='E24':
        pool(hidden.detach(),mask);weights=pool.last_weights
        assert (weights[~mask]==0).all()
        torch.testing.assert_close(weights.sum(1),torch.ones(2))
        torch.testing.assert_close(out.detach(),(weights[:,:,None]*hidden.detach()).sum(1))
    else:
        assert pool.gru.input_size==768 and pool.gru.bidirectional and pool.output.in_features==128
        reversed_hidden=hidden.detach().clone();reversed_hidden[0,mask[0]]=hidden.detach()[0,mask[0]].flip(0)
        assert not torch.allclose(pool(hidden.detach(),mask)[0],pool(reversed_hidden,mask)[0])
    with pytest.raises(ValueError,match='minimal satu'):pool(hidden,torch.zeros_like(mask))


@pytest.mark.parametrize('side',['left','right'])
def test_mask_uses_positions_not_token_ids_and_preserves_window_mapping(runtime,side):
    _,ns,cfg,_=runtime;tok=ThreeTechnicalTokenizer();tok.padding_side=side
    texts=['question reference answer',' '.join('answer' for _ in range(43))]
    contexts=[('question','reference')]*2
    windows,mapping,stats=ns['pair_batch_windows'](texts,contexts,tok,'cpu',32,4,cfg)
    if side=='left':
        # Tokenizer fixture lama selalu pad kanan. Bentuk batch pad kiri yang benar.
        for i,valid in enumerate(windows['attention_mask'].bool()):
            values=windows['input_ids'][i,valid].clone();length=len(values)
            windows['input_ids'][i].fill_(1);windows['input_ids'][i,-length:]=values
            windows['attention_mask'][i].zero_();windows['attention_mask'][i,-length:]=1
    mask=ns['answer_token_mask'](windows,mapping,stats,side)
    assert len(mask)>2 and (mask & ~windows['attention_mask'].bool()).sum()==0
    for w,(essay,pos) in enumerate((i,p) for i,d in enumerate(stats) for p in d['answer_positions']):
        offset=windows['attention_mask'].shape[1]-int(windows['attention_mask'][w].sum()) if side=='left' else 0
        assert mask[w].nonzero().flatten().tolist()==[p+offset for p in pos]


@pytest.mark.parametrize('experiment',NAMES)
def test_full_thirty_cells_training_reports_and_resume(experiment,tmp_path,monkeypatch):
    # Reuse the existing complete standalone harness: random Transformer fixture on CPU.
    monkeypatch.setattr(base,'NAMES',{'single_sequence':NAMES[experiment]})
    monkeypatch.setattr(base,'GPU',True)
    base.test_full_notebook_thirty_cells_and_resume_same_local_folder('single_sequence',tmp_path)
    import pandas as pd
    workspace=tmp_path/'standalone/workspace'
    pointer=json.loads((workspace/'outputs/dataset300_download.json').read_text())
    folder=Path(pointer['run_folder'])
    x=folder/'hasil_ringkas/02_Vektor_Jawaban.xlsx'
    book=pd.ExcelFile(x)
    assert 'Pooling Jawaban S3' in book.sheet_names
    vectors=pd.read_excel(book,sheet_name='Pooling Jawaban S3')
    assert len(vectors)==24 and len(vectors.columns)==769
    if experiment=='E24':
        weights=pd.read_excel(book,sheet_name='Bobot Attention S3')
        assert len(weights)>24 and np.isfinite(weights['Bobot Attention']).all()
        np.testing.assert_allclose(weights.groupby(['ID Jawaban','Bagian Teks'])['Bobot Attention'].sum(),1.,atol=1e-6)
    else:
        assert 'Bobot Attention S3' not in book.sheet_names
        assert len(pd.read_excel(book,sheet_name='BiGRU S3 State Akhir').columns)==129
