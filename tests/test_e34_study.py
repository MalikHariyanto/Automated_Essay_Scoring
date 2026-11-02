"""Audit uploaded E34 evidence and preserve the existing thirteen results."""
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def family(ref=None):
    raw = subprocess.check_output(['git', 'show', ref + ':docs/belajar_Tiga_Input_Spider.html'], cwd=ROOT, text=True) if ref else (ROOT / 'docs/belajar_Tiga_Input_Spider.html').read_text()
    return json.loads(re.search(r'<script id="familyData" type="application/json">(.*?)</script>', raw, re.S)[1])


def test_thirteen_prior_results_and_code_books_are_preserved():
    old, new = family('work'), family()
    assert old['documents'] == new['documents']
    for e, s in old['comparison']['experiments'].items():
        assert new['comparison']['experiments'][e] == s
        assert new['comparison']['predictionAudit'][e] == old['comparison']['predictionAudit'][e]
        assert new['identities'][e] == old['identities'][e]
    pairs = {(p['left'], p['right']): p for p in new['comparison']['pairwise']}
    for p in old['comparison']['pairwise']:
        assert pairs[p['left'], p['right']] == p
    assert len(pairs) == 105


def test_e34_matches_its_source_and_only_one_recorded_parameter_changes():
    c = family()['comparison']
    s, old = c['experiments']['E34'], c['experiments']['E33']
    params = lambda x: {r['Pengaturan']: r['Nilai'] for r in x['parameters']}
    a, b = params(s), params(old)
    assert {k: (b.get(k), v) for k, v in a.items() if b.get(k) != v} == {'LR pooling dan norm S3': (1e-4, 5e-5)}
    assert a['LR MLP S3'] == 5e-5 and a['Learning rate proyeksi BGE'] == 1e-4
    assert a['Jumlah fold'] == a['Batas epoch tanpa perbaikan'] == 5
    assert a['Maksimum epoch per fold'] == 25
    assert s['s3_pooling'] == 'bigru_coattention_pooling' and s['s3_fusion'] == 'concat'
    assert s['rows'] == 300 and s['development'] == 240 and s['test'] == 60
    assert s['metric_recompute_max_delta'] < 1e-6 and not s['runtime_source_matches_commit']
    assert s['selected_epochs'] == {'S1': 9, 'S2': 19, 'S3': 11}
    assert s['attention_audit']['token_positions_verified']
    assert s['coattention_audit']['summary_dimensions'] == 128
    assert s['coattention_audit']['max_sum_error'] < 1e-5
    source = json.loads(subprocess.check_output(['git', 'show', s['source_commit'] + ':' + s['source_path']], cwd=ROOT))
    assert source['metadata']['aes_experiment']['name'] == 'E34'
    for scenario in ['S1', 'S2', 'S3']:
        for r in (r for r in s['foldSummary'] if r['Skenario'] == scenario):
            trace = [h for h in s['history'][scenario] if h['Tahap'] == 'Validasi silang' and h['Fold'] == r['Fold']]
            chosen = next(h for h in trace if h['Epoch'] == r['Epoch Terpilih'])
            assert chosen['RMSE Validasi (poin)'] == min(h['RMSE Validasi (poin)'] for h in trace)
            assert len(trace) == r['Epoch Dijalankan'] <= 25


def test_e34_improvement_is_small_and_does_not_win_all_criteria():
    c = family()['comparison']
    a, b, baseline = [c['predictionAudit'][e]['S3'] for e in ['E34', 'E33', 'E30']]
    assert baseline['rmse'] < a['rmse'] < b['rmse']
    assert baseline['mae'] < a['mae'] < b['mae']
    assert a['within'] == {'1': 13, '2': 19, '5': 39, '10': 54}
    assert a['rounded_exact'] == 6 and a['raw_exact'] == 0
    cv = lambda e: next(r['mean'] for r in c['experiments'][e]['cvSummary'] if r['Skenario'] == 'S3')
    assert cv('E29') < cv('E34') < cv('E33') < cv('E30')
    for e, expected in [('E33', (28, 32)), ('E30', (33, 27))]:
        duel = next(p for p in c['pairwise'] if (p['left'], p['right']) == (e, 'E34'))
        assert duel['impact'][:2] == [{'scenario': s, 'left': 0, 'right': 0, 'ties': 60} for s in ['S1', 'S2']]
        assert duel['impact'][2] == {'scenario': 'S3', 'left': expected[0], 'right': expected[1], 'ties': 0}


def test_all_results_excel_and_precise_e34_identity_are_available():
    f = family(); c = f['comparison']
    assert list(c['experiments']) == [f'E{n}' for n in range(21, 36)]
    assert list(f['documents']) == ['E29', 'E30', 'E26']
    assert sum(len(s['books']) for s in c['experiments'].values()) == 75
    assert len(f['identities']) == 15
    identity = f['identities']['E34']
    assert identity['base'] == 'E33' and identity['result_available'] and not identity['book_available']
    assert 'sequence_pool/roberta_norm S3 5e-5' in identity['other_lr']
    assert 'proyeksi BGE 1e-4' in identity['other_lr']
    for rows in c['nearest'].values():
        assert abs(sum(v['share'] for v in rows.values()) - 60) < 1e-8
