"""Fixture buatan hanya untuk pengujian teknis; bukan data penelitian."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from sklearn.metrics import cohen_kappa_score
from aes.io import load_data,load_rubric,load_config,targets,sha256
from aes.audit import audit,validate_inputs_for_training,EligibilityError
from aes.cleaning import clean
from aes.windows import content_windows,mean_by_essay
from aes.model import AESModel,loss_mse
from aes.metrics import evaluate,ordinal_prediction,gwet_ac2,selection_monitor

class TokenizerFixture:
    def num_special_tokens_to_add(self,pair=False):return 2
    def encode(self,text,add_special_tokens=False):return list(range(int(text)))
    def prepare_for_model(self,ids,**kw):return {'input_ids':[-1]+ids+[-2],'attention_mask':[1]*(len(ids)+2)}

class EncoderFixture(nn.Module):
    def __init__(self):super().__init__();self.embedding=nn.Embedding(20,768)
    def forward(self,input_ids,**kwargs):
        # Context memengaruhi CLS; fixture teknis, bukan checkpoint pengganti.
        x=self.embedding(input_ids);return SimpleNamespace(last_hidden_state=x+x.mean(1,keepdim=True))


def test_draft_loader_and_target_audit(technical_source):
    df,r,c=technical_source
    assert len(df)==10 and df.essay_id.nunique()==10
    assert [t['name'] for t in r['traits']]==['isi','organisasi','kebahasaan','mekanik']
    assert targets(df,r).shape==(10,4) and np.isnan(targets(df,r)).all()
    a=audit(df,r);assert a['complete_label_rows']==0 and a['proposed_group_count']==4
    assert not a['five_group_folds_possible_by_count']
    with pytest.raises(EligibilityError,match='skor guru'):validate_inputs_for_training(df,r)
    altered=df.copy();altered.loc[0,'rubric_version']='different'
    assert audit(altered,r)['version_mismatch_ids']==['FIXTURE_01']
    altered.loc[0,'score_isi']='inf';altered.loc[1,'score_isi']='2.5';altered.loc[2,'score_isi']='6'
    a=audit(altered,r)['traits']['isi'];assert a['invalid_or_nonfinite']==1 and a['noninteger']==1 and a['out_of_range']==1


def test_cleaning():
    s='Pangan LOKAL, salahh!\r\n\r\n“Argumen”—…\x00 e\u0301'
    c=clean(s);assert c=='Pangan LOKAL, salahh!\n\n“Argumen”—… é'
    assert clean(c)==c and clean(c,True)!=c

@pytest.mark.parametrize('n',[0,1,510,511,892,1020,1500])
def test_window_coverage(n):
    chunks,s=content_windows(str(n),TokenizerFixture())
    assert s['capacity']==510 and s['stride']==382
    rebuilt={}
    for chunk,start in zip(chunks,s['starts']):
        assert len(chunk['input_ids'])<=512
        assert chunk['input_ids'][0]==-1 and chunk['input_ids'][-1]==-2
        for i,v in enumerate(chunk['input_ids'][1:-1]):rebuilt[start+i]=v
    assert list(rebuilt.values())==list(range(n))
    if len(s['starts'])>1:assert len(chunks[0]['input_ids'][1:-1][-128:])==128


def test_aggregation_gradients():
    x=torch.randn(3,768,requires_grad=True);mapping=torch.tensor([0,0,1])
    y=mean_by_essay(x,mapping,2);assert torch.allclose(y[0],x[:2].mean(0))
    y.sum().backward();assert torch.allclose(x.grad[0],torch.full((768,),.5))

@pytest.mark.parametrize('scenario',['S1','S2','S3'])
@pytest.mark.parametrize('b',[1,2])
def test_forward_backward_reload(scenario,b,tmp_path):
    torch.manual_seed(42);encoder=EncoderFixture() if scenario!='S2' else None
    model=AESModel(scenario,encoder);bge=torch.randn(b,1024);bge=torch.nn.functional.normalize(bge,dim=1)
    bge_before=bge.clone()
    kw={'windows':{'input_ids':torch.randint(0,20,(b*2,4))},'chunk_to_essay':torch.arange(b).repeat_interleave(2),
        'n_essays':b,'bge':bge}
    old={k:v.clone() for k,v in model.named_parameters()};opt=torch.optim.AdamW(model.parameters(),lr=.01)
    prediction=model(**kw);assert prediction.shape==(b,4)
    labels=torch.full((b,4),3.) # Fixture teknis saja.
    loss_mse(prediction,labels).backward();opt.step()
    assert torch.equal(bge,bge_before) and bge.grad is None
    changed=[k for k,v in model.named_parameters() if not torch.equal(v,old[k])]
    assert any(k.startswith('head.') for k in changed)
    if scenario!='S2':assert any(k.startswith('roberta.') for k in changed)
    if scenario=='S3':assert any(k.startswith('projection.') for k in changed)
    model.eval();before,features=model(**kw,return_features=True)
    if scenario!='S2':
        assert features['roberta_window_cls'].shape==(b*2,768)
        assert torch.allclose(features['roberta'],mean_by_essay(features['roberta_window_cls'],kw['chunk_to_essay'],b))
    if scenario=='S3':
        assert torch.equal(features['concat'][:,:768],features['roberta'])
        assert torch.equal(features['concat'][:,768:],features['projection'])
    torch.save(model.state_dict(),tmp_path/'weights.pt')
    other=AESModel(scenario,EncoderFixture() if scenario!='S2' else None);other.load_state_dict(torch.load(tmp_path/'weights.pt',weights_only=True));other.eval()
    assert torch.allclose(before,other(**kw))


def test_loss_rejects_broadcast():
    with pytest.raises(ValueError,match='broadcasting'):loss_mse(torch.ones(2,4),torch.ones(2,1))
    with pytest.raises(ValueError,match='nonfinite'):loss_mse(torch.full((1,4),float('nan')),torch.ones(1,4))
    assert loss_mse(torch.ones(2,4),torch.full((2,4),3.)).item()==4


def test_metrics(technical_source):
    _,r,_=technical_source;t=r['traits'][0]
    assert ordinal_prediction([-3,1.5,2.5,3.5,4.5,9],t).tolist()==[1,1,2,3,4,5]
    y=np.tile(np.array([1,2,3,4,5])[:,None],(1,4));p=y.astype(float);p[0]=0;p[-1]=6
    m=evaluate(y,p,r)
    for v in m.values():
        assert v['RMSE']==pytest.approx(np.sqrt(.4)) and v['QWK']==1 and v['AC2']==1
        assert v['out_of_range_fraction']==.4
    with pytest.raises(ValueError,match='nonfinite'):ordinal_prediction([np.inf],t)
    p=y+.51;m=evaluate(y,p,r)
    assert m['isi']['QWK']==cohen_kappa_score(y[:,0],ordinal_prediction(p[:,0],t),labels=[1,2,3,4,5],weights='quadratic')
    const=evaluate(np.ones((3,4)),np.ones((3,4)),r)
    assert const['isi']['Pearson'] is None and const['isi']['QWK'] is None
    assert selection_monitor(const,0)==('validation_mse',0.)
    with pytest.raises(ValueError):evaluate(y,np.full((5,4),np.nan),r)
    with pytest.raises(ValueError):evaluate(y,np.ones((5,1)),r)


def test_categories_unobserved():
    assert gwet_ac2([1,1,5,5],[1,2,4,5],[1,2,3,4,5])==pytest.approx(.9120879120879121)


def test_group_split(technical_source,tmp_path):
    from aes.split import make_manifest
    df,r,c=technical_source;df=df.copy();r=copy.deepcopy(r)
    # Hanya fixture teknis untuk membuktikan tidak ada kebocoran group.
    for key,value in {'transcription_status':'verified','text_completeness':'complete','identity_status':'verified',
        'scoring_status':'teacher_scored','ready_for_training':'true','rater_id':'fixture','scoring_date':'2026-01-01'}.items():df[key]=value
    for name in ['isi','organisasi','kebahasaan','mekanik']:df['score_'+name]='3'
    r.update(confirmed=True,teacher_approved_by='fixture',teacher_approved_on='2026-01-01')
    csv=tmp_path/'fixture.csv';rub=tmp_path/'fixture.json';review_path=tmp_path/'review.json'
    df.to_csv(csv,index=False);rub.write_text(json.dumps(r));c['csv']=str(csv);c['rubric']=str(rub);c['split']['review_file']=str(review_path)
    review={'dataset_sha256':sha256(csv),'reviewer':'fixture','reviewed_on':'2026-01-01',
      'independence_verified':True,'near_duplicates_reviewed':True,'mode':'grouped','groups':{i:i for i in df.essay_id}}
    review['pair_decisions']={'|'.join(sorted(v['ids'])):{'relation':'independent','evidence':'technical_fixture_only'} for v in audit(df,r)['near_duplicates']}
    review_path.write_text(json.dumps(review));m=make_manifest(df,r,review,c)
    assert len(m['folds'])==25
    assert 'test' not in m['label_distribution']
    assert m['test_labels_sealed_until_final_evaluation'] is True
    for f in m['folds']:assert not set(f['train'])&set(f['validation']) and not set(f['train'])&set(m['test'])
    review['groups']={i:str(j%4) for j,i in enumerate(df.essay_id)};review_path.write_text(json.dumps(review))
    with pytest.raises(EligibilityError,match='kurang dari lima'):make_manifest(df,r,review,c)
