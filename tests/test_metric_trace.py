"""Pasangan numerik hanya fixture teknis; tidak menjadi label penelitian."""
import numpy as np
import pytest

from aes.metric_trace import metric_trace
from aes.metrics import evaluate


def test_trace_matches_hand_calculation_and_keeps_trait_order(technical_source):
    _, rubric, _ = technical_source
    y=np.column_stack([[1,1,5,5],[1,2,3,4],[5,4,3,2],[2,3,4,5]])
    p=y.astype(float);p[:,0]=[1,2,4,5]
    trace=metric_trace(y,p,rubric,['PAIR_A','PAIR_B','PAIR_C','PAIR_D'])
    assert list(trace)==['isi','organisasi','kebahasaan','mekanik']
    t=trace['isi']
    assert t['RMSE_steps']['sum_squared_errors']==2
    assert t['RMSE_steps']['RMSE']==pytest.approx(np.sqrt(.5))
    assert t['Pearson_steps']['sum_centered_products']==12
    assert t['Pearson_steps']['Pearson']==pytest.approx(12/np.sqrt(160))
    assert t['QWK_steps']['observed_weighted_disagreement']==.03125
    assert t['QWK_steps']['expected_weighted_disagreement']==.40625
    assert t['QWK_steps']['QWK']==pytest.approx(1-.03125/.40625)
    assert t['AC2_steps']['AC2']==pytest.approx(.9120879120879121)
    assert t['categories']==[1,2,3,4,5]
    matrix=np.asarray(t['QWK_steps']['observed_counts_teacher_rows_prediction_columns'])
    assert matrix.shape==(5,5) and matrix[2].sum()==0
    assert trace['organisasi']['metrics']['RMSE']==0
    assert all(trace[name]['metrics']==values for name,values in evaluate(y,p,rubric).items())


def test_trace_keeps_raw_errors_and_midpoint_policy(technical_source):
    _, rubric, _=technical_source
    y=np.tile([1,2,3,4,5],(4,1)).T
    p=np.tile([0,1.5,2.5,4.5,6],(4,1)).T
    trace=metric_trace(y,p,rubric,list('ABCDE'))['isi']
    assert [pair['ordinal_prediction'] for pair in trace['pairs']]==[1,1,2,4,5]
    assert trace['pairs'][0]['raw_prediction']==0
    assert trace['pairs'][0]['clipped_prediction']==1
    assert trace['pairs'][0]['squared_error']==1
    assert trace['metrics']['out_of_range_fraction']==.4
    assert trace['RMSE_steps']['mean_squared_error']==.55


def test_trace_reports_undefined_and_rejects_missing_or_bad_id_mapping(technical_source):
    _, rubric, _=technical_source
    y=np.ones((3,4))
    trace=metric_trace(y,y,rubric,list('ABC'))['isi']
    assert trace['Pearson_steps']['Pearson'] is None
    assert trace['QWK_steps']['QWK'] is None
    assert trace['AC2_steps']['AC2']==1
    with pytest.raises(ValueError,match='nonfinite'):
        metric_trace(np.full((3,4),np.nan),y,rubric,list('ABC'))
    with pytest.raises(ValueError,match='ID pasangan'):
        metric_trace(y,y,rubric,['A','A','B'])
