"""Audit buku E6: integritas sumber/hasil, bukan pengujian kualitas model baru."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import build_mlp_study as builder


def data():
    return builder.content(json.loads(builder.SNAPSHOT.read_text()))


def test_e6_study_source_and_changed_parameters_are_exact_and_old_snapshot_preserved():
    d = data()
    notebook = json.loads(subprocess.check_output(['git','show',builder.STUDY_COMMIT+':'+builder.STUDY_PATH],cwd=ROOT))
    for cell in d['cells']:
        assert cell['source'] == builder.source(notebook['cells'][cell['index']])
        assert cell['sha256'] == hashlib.sha256(cell['source'].encode()).hexdigest()
    config = next(c['source'] for c in d['cells'] if c['index'] == 12)
    assert 'S3_ENCODER_LR=5e-6' in config and "device='cpu'" in config and "REUSE_RUN=''" in config
    original = json.loads(subprocess.check_output(['git','show','d8c21ac:docs/mlp_study_snapshot.json'],cwd=ROOT))
    assert d['snapshot'] == original
    assert d['studySource']['runtime_match'] is False


def test_e6_metrics_history_and_public_preview_are_from_e6_not_initial_mlp():
    d = data()
    for field in ('metrics','folds','history','foldSummary','books','zip_sha256'):
        assert d['studySnapshot'][field] == d['lrExperiment'][field]
    assert d['studySnapshot']['metrics'][2]['RMSE (poin)'] != d['snapshot']['metrics'][2]['RMSE (poin)']
    assert d['studySnapshot']['native'] == {} and d['studySnapshot']['saved_outputs'] == 0
    assert [r['Epoch Terpilih'] for r in d['studySnapshot']['foldSummary'] if r['Skenario']=='S3'] == [14,10,8,16,10]
    for matrix in d['studySnapshot']['matrices'].values():
        assert len(matrix) == 11 and sum(map(sum,matrix)) == 60
    for records in d['studySnapshot']['publicSheets'].values():
        for row in records:
            assert not {'ID Jawaban','Jawaban Siswa','Nilai Guru'} & row.keys()


def test_each_experiment_source_evidence_matches_pinned_notebook_and_has_scoped_paper_links():
    d = data()
    catalog = d['methodSources']
    assert {e['id'] for e in catalog['experiments']} == {'E0','E2','E3','E4','E5','E6'}
    for exp in catalog['experiments']:
        notebook = json.loads(subprocess.check_output(['git','show',exp['commit']+':'+exp['path']],cwd=ROOT))
        assert '/'+exp['commit']+'/' in exp['notebook']
        assert exp['choice'] and exp['source_status']
        for record in exp['evidence'].values():
            assert record['source'] == builder.source(notebook['cells'][record['cell']])
        for key in exp['papers']:
            paper = catalog['papers'][key]
            assert paper['paper'].startswith('https://') and paper['limit'] and paper['supports'] and paper['evidence']
    assert 'Tidak menetapkan 5e-6' in catalog['papers']['stability']['limit']
    assert 'LwXGBoost' in catalog['papers']['fusion']['limit']


def test_e6_learning_notes_explain_control_reuse_and_per_scenario_rates():
    d = data()
    notes = [note for cell in d['cells'] for note in cell['notes']]
    titles = {note['title'] for note in notes}
    assert {'Perubahan utama E6','Kontrol S1/S2 dipakai ulang','Optimizer membedakan skenario'} <= titles
    reuse = next(n for n in notes if n['title']=='prepare_reused_controls()')
    assert 'Tidak melatih kedua kontrol lagi' in reuse['body']
    checkpoint = next(n for n in notes if n['title']=='save_training_progress()')
    assert 'bukan mekanisme dua slot' in checkpoint['body']
