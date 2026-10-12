"""Fixture teknis publik; tidak memuat rubrik asli atau teks siswa."""
import json
from pathlib import Path
import pandas as pd
import pytest
from aes.io import TRAITS,load_config,load_data,load_rubric

@pytest.fixture
def technical_source(tmp_path):
    root=Path(__file__).resolve().parents[1]
    cfg=load_config(root/'configs/default.json')
    rubric={'version':'TECHNICAL_FIXTURE_ONLY','confirmed':False,
            'scoring_rule':'Aturan fixture teknis, bukan penilaian guru.',
            'missing_rule':'Missing tetap kosong pada fixture.'}
    rubric['traits']=[{'name':name,'min':1,'max':5,'scale_type':'ordinal','categories':[1,2,3,4,5],
        'definition':'Fixture teknis; bukan definisi rubrik siswa.',
        'indicators':['Indikator fixture teknis saja.'],'exclude':'Tidak digunakan untuk penelitian.',
        'levels':{str(k):f'Deskriptor fixture teknis {k}; bukan rubrik penelitian.' for k in range(1,6)}} for name in TRAITS]
    rows=[]
    for i in range(10):
        text=f'fixture teknis record {i}; tidak berasal dari esai siswa'
        rows.append({'essay_id':f'FIXTURE_{i+1:02d}','text_raw':text,'text_clean':text,
            **{'score_'+t:'' for t in TRAITS},'group_id':f'PROPOSED_FIXTURE_{i%4}',
            'rubric_version':rubric['version'],'scoring_status':'not_scored',
            'transcription_status':'draft_requires_review','ready_for_training':'false',
            'text_completeness':'continuation_unassigned' if i>=8 else 'body_observed_no_known_gap',
            'identity_status':'unverified','group_status':'proposed_similarity_group',
            'rater_id':'','scoring_date':''})
    csv=tmp_path/'technical.csv';rub=tmp_path/'technical.json'
    pd.DataFrame(rows).to_csv(csv,index=False,encoding='utf-8-sig')
    rub.write_text(json.dumps(rubric,ensure_ascii=False),encoding='utf-8')
    cfg['csv']=str(csv);cfg['rubric']=str(rub);cfg['output']=str(tmp_path/'outputs')
    return load_data(csv),load_rubric(rub),cfg
