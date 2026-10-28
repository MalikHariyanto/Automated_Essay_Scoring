"""Bukti tersimpan E8: source, seleksi epoch, dan agregat tujuh eksperimen."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def test_e8_history_selection_and_cv_are_complete_and_distinct_from_e6():
    data = json.loads((ROOT/'docs/snapshots/mlp_e8_snapshot.json').read_text())
    assert data['id'] == 'E8' and data['teacher_test_pairs_identical']
    assert len(data['books']) == 5 and len(data['folds']) == 15
    for scenario in ('S1', 'S2', 'S3'):
        folds = [f for f in data['folds'] if f['Skenario'] == scenario]
        assert sorted(f['Fold'] for f in folds) == [1, 2, 3, 4, 5]
        for selection in (f for f in data['foldSummary'] if f['Skenario'] == scenario):
            trace = [h for h in data['history'][scenario] if h['Fold'] == selection['Fold']]
            selected = next(h for h in trace if h['Epoch'] == selection['Epoch Terpilih'])
            assert selected['RMSE Validasi (poin)'] == min(h['RMSE Validasi (poin)'] for h in trace)
        assert next(c['mean'] for c in data['cvSummary'] if c['Skenario'] == scenario) == statistics.mean(f['RMSE (poin)'] for f in folds)
    assert data['control_prediction_delta']['S1'] > 1
    assert 'reused_controls' not in data
    parameters = {r['Pengaturan']: r['Nilai'] for r in data['parameters']}
    assert parameters['Learning rate encoder RoBERTa S3'] == 5e-6
    assert parameters['Batas token konteks S3'] == 192
    assert data['metrics'][2]['RMSE (poin)'] == 6.381775529425771


def test_e8_ranking_recomputed_over_all_seven_runs_and_same_sixty_answers():
    data = json.loads((ROOT/'docs/snapshots/mlp_all_results_snapshot.json').read_text())
    for scenario in ('S1', 'S2', 'S3'):
        summary = data['across_experiments']['scenarios'][scenario]
        assert summary['answers'] == 60
        assert {r['id'] for r in summary['entries']} == {'E0', 'E2', 'E3', 'E4', 'E5', 'E6', 'E8'}
        assert abs(sum(r['share'] for r in summary['entries']) - 60) < 1e-8
        assert all(r['closest'] == r['unique'] + r['tied'] for r in summary['entries'])
    assert sum(data['prediction_winners']['E8'].values()) == 60


def test_e8_snapshot_contains_aggregates_only():
    data = json.loads((ROOT/'docs/snapshots/mlp_e8_snapshot.json').read_text())
    forbidden = {'ID Jawaban', 'Jawaban Siswa', 'Nilai Guru', 'pairs', 'responses'}
    def check(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
    check(data)
