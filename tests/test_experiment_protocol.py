"""Kontrak jadwal eksperimen pada fixture teknis, tanpa melatih CSV draf."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from aes.io import load_config,load_rubric


def test_all_models_locked_before_any_test_evaluation(tmp_path,monkeypatch,technical_source):
    import aes.experiment as exp
    _,r,cfg=technical_source;cfg['output']=str(tmp_path)
    ids=['D0','D1','V0','T0','T1']
    df=pd.DataFrame({'essay_id':ids});y=np.repeat(np.array([1,2,3,4,5])[:,None],4,axis=1)
    manifest={'development':ids[:3],'test':ids[3:],
              'folds':[{'train':ids[:2],'validation':['V0'],'seed':42,'repeat':i//5+1,'fold':i%5+1} for i in range(25)]}
    index={i:j for j,i in enumerate(ids)};events=[];finalized=set()
    monkeypatch.setattr(exp,'prepare',lambda _: (df,r,y,['technical_fixture']*5,torch.zeros(5,1024),{},manifest,index))
    metric={t['name']:{'QWK':.8,'RMSE':.5,'Pearson':.9,'AC2':.8} for t in r['traits']}
    def train(**kw):
        scenario=kw['scenario'];is_final=kw['valid_ids'] is None
        events.append(('final_refit' if is_final else 'train',scenario))
        if is_final:finalized.add(scenario)
        return {'candidate':kw['candidate'],'monitor_kind':'mean_qwk_all_four','best_monitor':.8,
                'best_validation_mse':.25,'best_epoch':2,'metrics':None if is_final else metric,
                'checkpoint':str(Path(kw['out'])/'best.pt')}
    monkeypatch.setattr(exp,'train_with_oom_retry',train)
    monkeypatch.setattr(exp,'load_checkpoint',lambda *a,**k:(object(),{}))
    class EngineFixture:
        def __init__(self,*a,**kw):pass
        def predict(self,indices,*a):
            assert finalized=={'S1','S2','S3'}
            for scenario in finalized:
                lock=json.loads((tmp_path/'runs/proposal'/scenario/'locked_config.json').read_text())
                assert lock['final_refit_completed'] is True and lock['final_epochs']==2
            events.append(('test_predict',None));return y[indices].astype(float)
    monkeypatch.setattr(exp,'BatchEngine',EngineFixture)
    result=exp.run(cfg,'proposal')
    assert all(len(result[s])==25 for s in ['S1','S2','S3'])
    first_test=next(i for i,event in enumerate(events) if event[0]=='test_predict')
    assert all(event[0]=='test_predict' for event in events[first_test:])
    assert len([e for e in events if e[0]=='train'])==3*27
    for scenario in ['S1','S2','S3']:
        test=json.loads((tmp_path/'runs/proposal'/scenario/'final/test_metrics.json').read_text())
        assert test['label_distribution']['isi']=={'1':0,'2':0,'3':0,'4':1,'5':1}
        assert test['all_requested_scenarios_locked']==['S1','S2','S3']


def test_optional_document_paths_are_configurable(tmp_path):
    root=Path(__file__).resolve().parents[1];cfg=load_config(root/'configs/default.json')
    assert Path(cfg['sources']['proposal']).is_absolute()
    # Path configurable tidak mewajibkan dokumen privat tersedia pada clone publik.
    assert Path(cfg['sources']['proposal']).name.endswith('.docx')
    raw=json.loads((root/'configs/default.json').read_text())
    raw['sources']={'proposal':'project_sources/custom.docx','paper':str(tmp_path/'custom.pdf')}
    config=tmp_path/'configs'/'source_paths.json';config.parent.mkdir();config.write_text(json.dumps(raw))
    adjusted=load_config(config)
    assert adjusted['sources']['proposal']==str(tmp_path/'project_sources/custom.docx')
    assert adjusted['sources']['paper']==str(tmp_path/'custom.pdf')
