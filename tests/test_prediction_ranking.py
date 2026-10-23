"""Uji peringkat yang adil saat kontrol identik, ID berubah, atau prediksi dibulatkan."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_prediction_ranking import compare_predictions


def rows(values):
    return [{'ID Jawaban':str(i),'Nilai Guru':80,**{'Prediksi '+s:value for s in ('S1','S2','S3')}} for i,value in enumerate(values)]


def test_unique_and_joint_winners_conserve_one_point_per_answer():
    data=compare_predictions({'A':rows([80,81,84]),'B':rows([82,79,82]),'C':rows([85,83,82])},expected_rows=3)
    s=data['scenarios']['S3'];entries={r['id']:r for r in s['entries']}
    assert entries['A']=={'id':'A','unique':1,'tied':1,'share':1.5,'closest':2}
    assert entries['B']['share']==1 and entries['C']['share']==.5
    assert s['tie_answers']==2 and sum(r['share'] for r in s['entries'])==3


def test_raw_predictions_do_not_turn_rounded_scores_into_ties():
    data=compare_predictions({'A':rows([80.1]),'B':rows([80.4])},expected_rows=1)
    assert data['scenarios']['S1']['entries'][0]['unique']==1


def test_near_identical_controls_are_tied_and_row_order_does_not_matter():
    a=rows([81,82]);b=list(reversed(rows([81.000005,82.000005])))
    s=compare_predictions({'A':a,'B':b},expected_rows=2)['scenarios']['S2']
    assert s['tie_answers']==2 and [r['share'] for r in s['entries']]==[1,1]


@pytest.mark.parametrize('kind',['duplicate','missing','teacher','nan'])
def test_incomparable_or_invalid_inputs_are_rejected(kind):
    a=rows([80,81]);b=rows([81,82])
    if kind=='duplicate':b[1]['ID Jawaban']='0'
    if kind=='missing':b[1]['ID Jawaban']='other'
    if kind=='teacher':b[0]['Nilai Guru']=81
    if kind=='nan':b[0]['Prediksi S3']=float('nan')
    with pytest.raises(ValueError):compare_predictions({'A':a,'B':b},expected_rows=2)
