"""Fixture teknis untuk tabel final; tidak menilai teks siswa."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aes.audit import audit, validate_inputs_for_training
from aes.io import dump, load_data, sha256, targets
from aes.metrics import evaluate
from aes.reporting import final_report


def labelled_artifacts(technical_source):
    df, rubric, cfg = technical_source
    # Label buatan hanya pada fixture unit test di tmp_path, bukan CSV penelitian.
    df = df.copy()
    for j,t in enumerate(rubric['traits']):
        df['score_'+t['name']] = [str((i+j)%5+1) for i in range(len(df))]
    for key,value in {'transcription_status':'verified','text_completeness':'complete',
                      'identity_status':'verified','scoring_status':'teacher_scored',
                      'ready_for_training':'true','rater_id':'TECHNICAL_RATER',
                      'scoring_date':'TECHNICAL_DATE'}.items():
        df[key] = value
    rubric = {**rubric, 'confirmed':True,
              'teacher_approved_by':'TECHNICAL_RATER', 'teacher_approved_on':'TECHNICAL_DATE'}
    df.to_csv(cfg['csv'],index=False)
    dump(cfg['rubric'],rubric)
    pairs=audit(df,rubric)['near_duplicates']
    review={'independence_verified':True,'near_duplicates_reviewed':True,
            'reviewer':'TECHNICAL_REVIEWER','reviewed_on':'TECHNICAL_DATE',
            'dataset_sha256':sha256(cfg['csv']),'mode':'independent',
            'pair_decisions':{'|'.join(sorted(p['ids'])):{'relation':'independent','evidence':'Technical fixture only.'} for p in pairs}}
    cfg['split']['review_file'] = str(Path(cfg['output'])/'technical_review.json')
    dump(cfg['split']['review_file'],review)
    df=load_data(cfg['csv'])
    validate_inputs_for_training(df,rubric,review,sha256(cfg['csv']))
    ids=df.essay_id.tolist()[-2:]
    dump(Path(cfg['output'])/'manifest.json',{'dataset_sha256':sha256(cfg['csv']),
         'rubric_sha256':sha256(cfg['rubric']),'review_sha256':sha256(cfg['split']['review_file']),'test':ids})
    dump(Path(cfg['output'])/'model_revisions.json',cfg['models'])
    y=targets(df.set_index('essay_id').loc[ids].reset_index(),rubric)
    p=y+np.array([[.4,.2,1.1,0.],[.5,-.8,.3,-2.9]])
    for scenario in ['S1','S2','S3']:
        base=Path(cfg['output'])/'runs/proposal'/scenario
        dump(base/'locked_config.json',{'final_refit_completed':True})
        dump(base/'final/test_metrics.json',{'configuration_locked':True,
             'all_requested_scenarios_locked':['S1','S2','S3'],'metrics':evaluate(y,p,rubric)})
        predictions=pd.DataFrame(p,columns=['pred_'+t['name'] for t in rubric['traits']])
        predictions.insert(0,'essay_id',ids)
        predictions.to_csv(base/'final/test_predictions.csv',index=False)
    return cfg,df,rubric


def test_draft_table_keeps_all_twelve_rows_unavailable_even_with_stale_results(technical_source):
    df,rubric,cfg=technical_source
    out=Path(cfg['output'])
    dump(out/'runs/proposal/S3/final/test_metrics.json',{'metrics':'STALE_FIXTURE'})
    report=final_report(cfg,df,rubric)
    assert len(report['rows'])==12
    assert all(row[m] is None for row in report['rows'] for m in ['RMSE','Pearson','QWK','AC2'])
    assert all(row['status']=='not_available' for row in report['rows'])
    assert report['data_blockers']
    assert report['metric_traces']=={}
    assert [row['trait'] for row in report['rows'][:4]]==['isi','organisasi','kebahasaan','mekanik']


def test_final_table_reads_matching_locked_test_artifacts_per_trait(technical_source):
    cfg,df,rubric=labelled_artifacts(technical_source)
    report=final_report(cfg,df,rubric)
    assert len(report['rows'])==12
    assert all(row['status']=='available_final_test' for row in report['rows'])
    assert report['rows'][0]['RMSE']==pytest.approx(np.sqrt((.4**2+.5**2)/2))
    assert report['rows'][0]['out_of_range_fraction']==.5
    assert set(report['metric_traces'])=={'S1','S2','S3'}
    trace=report['metric_traces']['S1']['isi']
    assert [p['essay_id'] for p in trace['pairs']]==df.essay_id.tolist()[-2:]
    assert trace['RMSE_steps']['RMSE']==report['rows'][0]['RMSE']
    # Label/hasil fixture teknis bukan bukti training model asli.


def test_final_table_refuses_test_ids_outside_manifest(technical_source):
    cfg,df,rubric=labelled_artifacts(technical_source)
    path=Path(cfg['output'])/'runs/proposal/S1/final/test_predictions.csv'
    predictions=pd.read_csv(path);predictions.loc[0,'essay_id']=df.essay_id.iloc[0]
    predictions.to_csv(path,index=False)
    with pytest.raises(ValueError,match='ID prediksi'):
        final_report(cfg,df,rubric)


def test_final_table_refuses_results_before_every_scenario_refit(technical_source):
    cfg,df,rubric=labelled_artifacts(technical_source)
    dump(Path(cfg['output'])/'runs/proposal/S3/locked_config.json',{'final_refit_completed':False})
    with pytest.raises(ValueError,match='ketiga konfigurasi'):
        final_report(cfg,df,rubric)


def test_final_table_refuses_changed_independence_review(technical_source):
    cfg,df,rubric=labelled_artifacts(technical_source)
    path=Path(cfg['split']['review_file'])
    review=json.loads(path.read_text());review['reviewer']='ANOTHER_TECHNICAL_REVIEWER'
    dump(path,review)
    with pytest.raises(ValueError,match='review independensi'):
        final_report(cfg,df,rubric)


def test_notebook_renders_metric_steps_only_from_matching_test_pairs(technical_source,capsys):
    # Hanya fixture teknis untuk menjangkau cabang UI berlabel; bukan hasil penelitian.
    cfg,df,rubric=labelled_artifacts(technical_source)
    root=Path(__file__).resolve().parents[1]
    nb=json.loads((root/'notebooks/01_belajar_alur_AES.ipynb').read_text())
    cell=next(''.join(c['source']) for c in nb['cells']
              if c['cell_type']=='code' and ''.join(c['source']).startswith("run_aes([ROOT/'scripts/notebook_step.py', 'report'"))
    tables=[]
    namespace=dict(ROOT=root,OUTPUT=Path(cfg['output']),CONFIG_PATH='TECHNICAL_FIXTURE',
        TRACE_SCENARIO='S3',TRACE_TRAIT='isi',json=json,
        run_aes=lambda _:final_report(cfg,df,rubric),show_table=lambda columns,rows:tables.append((columns,rows)))
    exec(compile(cell,'technical_metric_steps_cell','exec'),namespace)
    pairs=next(rows for columns,rows in tables if 'squared_error' in columns)
    assert [p['essay_id'] for p in pairs]==df.essay_id.tolist()[-2:]
    matrices=[rows for columns,rows in tables if columns[0]=='Kategori guru']
    assert len(matrices)==4 and all(len(rows)==5 for rows in matrices)
    output=capsys.readouterr().out
    assert 'Pa:' in output and 'Expected disagreement:' in output
