"""Audit hasil unggahan; tidak menjalankan training penelitian."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def family(path):
    return json.loads(re.search(r'<script id="familyData" type="application/json">(.*?)</script>', path.read_text(), re.S)[1])


def test_new_results_gate_pooling_and_source_provenance():
    for e in ('E31', 'E32'):
        s = json.loads((ROOT / f'docs/snapshots/{e.lower()}_study_snapshot.json').read_text())
        p = {r['Pengaturan']: r['Nilai'] for r in s['parameters']}
        assert s['rows'] == 300 and s['test'] == 60
        assert s['metric_recompute_max_delta'] < 1e-6
        assert s['s3_fusion'] == 'gated_concat'
        assert p['LR MLP S2'] == p['LR MLP S3'] == (5e-6 if e == 'E31' else 1e-4)
        assert p['LR pooling gate norm S3'] == 1e-4
        assert s['gate_audit']['multipliers_verified']
        assert s['attention_audit']['max_sum_error'] < 1e-5
        assert not s['runtime_source_matches_commit']
        source = json.loads(subprocess.check_output(['git', 'show', s['source_commit'] + ':' + s['source_path']], cwd=ROOT))
        assert source['metadata']['aes_experiment']['target'] == 'rater'
        if e == 'E32':
            assert s['coattention_audit']['summary_dimensions'] == 768
            assert s['coattention_audit']['token_positions_verified']
            assert 'bigru_attention_audit' not in s


def test_family_retains_all_results_and_independent_excel_access():
    d = family(ROOT / 'docs/belajar_Tiga_Input_Spider.html')
    c = d['comparison']
    assert list(c['experiments']) == [f'E{n}' for n in range(21,33)]
    assert list(d['documents']) == ['E29', 'E30', 'E26']
    assert len(d['identities']) == 12 and all(x['result_available'] for x in d['identities'].values())
    assert len(c['pairwise']) == 66
    assert sum(len(x['books']) for x in c['experiments'].values()) == 60
    for scenario in ('S1','S2','S3','all'):
        assert abs(sum(x['share'] for x in c['nearest'][scenario].values())-60) < 1e-8
    for e in ('E31','E32'):
        assert not d['identities'][e]['book_available']
        assert d['identities'][e]['base'] == 'E25'
        for scenario in ('S1','S2','S3'):
            a = c['predictionAudit'][e][scenario]
            report = next(r for r in c['experiments'][e]['metrics'] if r['Skenario']==scenario)
            assert abs(a['rmse']-report['RMSE (poin)']) < 1e-6
