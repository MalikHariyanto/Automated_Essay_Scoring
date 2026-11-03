"""Uploaded E36 S3-only evidence, actual scenario counts and historical integrity."""
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = '349f2e82e876333bf2270d47e7eb86b59e939a55'


def family(ref=None):
    path = 'docs/belajar_Tiga_Input_Spider.html'
    raw = subprocess.check_output(['git','show',ref+':'+path],cwd=ROOT,text=True) if ref else (ROOT/path).read_text()
    return json.loads(re.search(r'<script id="familyData" type="application/json">(.*?)</script>',raw,re.S)[1])


def test_prior_results_predictions_duels_and_retained_code_are_exactly_preserved():
    old,new = family(BASE),family()
    for e,snapshot in old['comparison']['experiments'].items():
        assert new['comparison']['experiments'][e] == snapshot
        assert new['comparison']['predictionAudit'][e] == old['comparison']['predictionAudit'][e]
        identity = {**old['identities'][e], 'book_available': e in new['documents']}
        assert new['identities'][e] == identity
    for e in ('E29','E30'):
        assert new['documents'][e] == old['documents'][e]
    pairs = {(p['left'],p['right']):p for p in new['comparison']['pairwise']}
    for pair in old['comparison']['pairwise']:
        assert pairs[pair['left'],pair['right']] == pair
    assert len(pairs) == 120
    assert subprocess.check_output(['git','show',BASE+':docs/belajar_MLP_Spider.html'],cwd=ROOT) == (ROOT/'docs/belajar_MLP_Spider.html').read_bytes()


def test_e36_recipe_only_s3_and_verified_reported_metrics():
    f=family();c=f['comparison'];s=c['experiments']['E36']
    p={r['Pengaturan']:r['Nilai'] for r in s['parameters']}
    assert s['scenarios']==['S3']
    assert len(s['metrics'])==1 and len(s['folds'])==5
    assert set(s['history'])==set(s['selected_epochs'])=={'S3'}
    assert s['selected_epochs']['S3']==13
    assert p['Learning rate encoder RoBERTa S3']==3e-6
    assert p['LR MLP S3']==p['LR pooling dan norm S3']==p['Learning rate proyeksi BGE']==1e-4
    assert 'LR MLP S2' not in p and 'Learning rate encoder RoBERTa S1' not in p
    assert p['Jumlah fold']==p['Batas epoch tanpa perbaikan']==5 and p['Maksimum epoch per fold']==25
    assert s['metric_recompute_max_delta'] < 1e-6
    assert not s['runtime_source_matches_commit']
    assert s['zip_sha256']=='6eca1930f290ea2cfcde87bcb0cf1acc6f55740ea070f78e013ce816dd7303e4'
    assert s['attention_audit']['token_positions_verified']
    assert s['coattention_audit']['token_positions_verified'] and s['coattention_audit']['summary_dimensions']==128
    assert s['rows']==300 and s['development']==240 and s['test']==60
    for row in s['foldSummary']:
        history=[h for h in s['history']['S3'] if h['Tahap']=='Validasi silang' and h['Fold']==row['Fold']]
        chosen=next(h for h in history if h['Epoch']==row['Epoch Terpilih'])
        assert chosen['RMSE Validasi (poin)']==min(h['RMSE Validasi (poin)'] for h in history)
        assert len(history)==row['Epoch Dijalankan']<=25
        assert history[-1]['Loss Training (MSE)']<history[0]['Loss Training (MSE)']


def test_model_counts_rankings_and_shared_scenario_duels_are_not_filled_with_zeros():
    f=family();c=f['comparison']
    assert list(c['experiments'])==[f'E{n}' for n in range(21,37)]
    assert list(f['documents'])==['E29','E30','E36']
    assert sum(len(v) for v in c['predictionAudit'].values())==46
    assert len(c['nearest']['all'])==46 and len(c['nearest']['S1'])==len(c['nearest']['S2'])==15 and len(c['nearest']['S3'])==16
    assert set(c['predictionAudit']['E36'])=={'S3'}
    a,b,d=[c['predictionAudit'][e]['S3'] for e in ['E36','E30','E26']]
    assert b['rmse']<a['rmse']<d['rmse']
    assert a['within']=={'1':12,'2':18,'5':40,'10':57} and a['rounded_exact']==3
    assert abs(a['mae']-4.2307723)<1e-6
    assert c['experiments']['E36']['cvSummary'][0]['mean']>next(r['mean'] for r in c['experiments']['E30']['cvSummary'] if r['Skenario']=='S3')
    duel=next(p for p in c['pairwise'] if (p['left'],p['right'])==('E30','E36'))
    assert duel['impact']==[{'scenario':'S3','left':33,'right':27,'ties':0}]
    assert sum(duel['overall'].values())==60
    for entries in c['nearest'].values():assert abs(sum(v['share'] for v in entries.values())-60)<1e-8
    assert sum(len(s['books']) for s in c['experiments'].values())==80
    assert sum(len(b['sheets']) for s in c['experiments'].values() for b in s['books'])==748


def test_native_e36_code_book_is_complete_and_annotated_without_legacy_magic():
    f=family();html=f['documents']['E36']
    d=json.loads(re.search(r'<script id="studyData" type="application/json">(.*?)</script>',html,re.S)[1])
    assert d['studySource']['commit']==BASE and d['studySource']['experiment']=='E36'
    assert d['coverage']['code_cells']==89
    assert all(cell['notes'] for cell in d['cells'])
    code='\n'.join(c['source'] for c in d['cells'])
    assert "SCENARIOS = ('S3',)" in code and 'ENCODER_LR = 3e-6' in code
    assert 'S3_LAYERS_LR = 1e-4' in code
    assert '%%aes' not in code and 'WORKER_SOURCE' not in code and 'cosine' not in code
    assert f['identities']['E36']['book_available'] and not f['identities']['E26']['book_available']
    assert '3e-6' in f['identities']['E36']['encoder_lr']
