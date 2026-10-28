"""Peta rentang skor tidak mengubah evaluator atau menghasilkan skor baru."""
import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_mlp_score_maps import matrix_for


def test_rounding_clipping_and_ties_use_raw_predictions():
    rows = [{'Nilai Guru': t, 'Prediksi S3': p} for t,p in
            [(0,-2),(100,101),(90,89.5),(90,89.5001),(79,79.5)]]
    matrix = matrix_for(rows,'S3')
    assert sum(map(sum,matrix)) == 5
    assert matrix[0][0] == matrix[10][10] == matrix[9][8] == matrix[9][9] == matrix[7][7] == 1


@pytest.mark.parametrize('row',[
    {'Nilai Guru':90,'Prediksi S3':89.5,'Pembulatan S3':90},
    {'Nilai Guru':101,'Prediksi S3':90},
    {'Nilai Guru':90,'Prediksi S3':float('nan')},
])
def test_invalid_teacher_prediction_or_saved_category_is_rejected(row):
    with pytest.raises(ValueError):
        matrix_for([row],'S3')


def test_public_matrices_preserve_old_counts_and_contain_all_twenty_one_maps():
    data = json.loads((ROOT/'docs/snapshots/mlp_score_maps_snapshot.json').read_text())
    assert set(data['matrices']) == {'E0','E2','E3','E4','E5','E6','E8'}
    for run,maps in data['matrices'].items():
        assert set(maps) == {'S1','S2','S3'}
        for matrix in maps.values():
            assert len(matrix) == 11 and all(len(row) == 11 for row in matrix)
            assert sum(map(sum,matrix)) == 60
            assert all(isinstance(n,int) and n >= 0 for row in matrix for n in row)
    for run,path in [('E2','mlp_study_snapshot.json'),('E6','mlp_e6_study_snapshot.json')]:
        assert data['matrices'][run] == json.loads((ROOT/'docs/snapshots'/path).read_text())['matrices']
