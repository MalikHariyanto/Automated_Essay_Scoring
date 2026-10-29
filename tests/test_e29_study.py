"""E29 uploaded evidence, reduced code payload and unchanged old predictions."""
import hashlib,json,re,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE='b783fe05896cd4a0efcc4a81508d70b0eedbd717'
PIN='7a76191be20604482fa53f770699563b380120ba'

def family(raw=None):
    raw=raw or (ROOT/'docs/belajar_Tiga_Input_Spider.html').read_text()
    return json.loads(re.search(r'<script id="familyData" type="application/json">(.*?)</script>',raw,re.S)[1])


def test_all_eight_old_results_and_pairwise_outcomes_are_preserved():
    a=family(subprocess.check_output(['git','show',BASE+':docs/belajar_Tiga_Input_Spider.html'],cwd=ROOT,text=True));b=family()
    for e,snapshot in a['comparison']['experiments'].items():
        assert b['comparison']['experiments'][e]==snapshot
        assert b['comparison']['predictionAudit'][e]==a['comparison']['predictionAudit'][e]
    old_pairs={(p['left'],p['right']):p for p in a['comparison']['pairwise']}
    assert all(old_pairs[(p['left'],p['right'])]==p for p in b['comparison']['pairwise'] if (p['left'],p['right']) in old_pairs)
    for e in ('E26','E24'):assert a['documents'][e]==b['documents'][e]
    assert len(b['comparison']['pairwise'])==36


def test_e29_verifies_the_actual_lr_intervention_and_all_three_scenarios():
    c=family()['comparison'];x=c['experiments']['E29'];p={r['Pengaturan']:r['Nilai'] for r in x['parameters']}
    assert p['LR MLP S2']==p['LR MLP S3']==5e-6
    assert p['LR head linear S1']==p['Learning rate proyeksi BGE']==p['LR pooling dan norm S3']==1e-4
    assert x['source_commit']==PIN and x['runtime_source_matches_commit'] is False
    assert x['metric_recompute_max_delta']<1e-6 and x['s3_pooling']=='bigru_attention' and x['s3_fusion']=='concat'
    assert x['attention_audit']['rows']==16946 and x['attention_audit']['max_sum_error']<1e-5
    assert x['bigru_attention_audit']['summary_dimensions']==128
    metrics={r['Skenario']:r for r in x['metrics']}
    assert abs(metrics['S3']['RMSE (poin)']-5.4293216451133)<1e-12
    assert abs(metrics['S2']['RMSE (poin)']-75.9701835491043)<1e-12
    assert metrics['S2']['Gwet AC2'] < -1  # Keep actual recomputed value, never clip agreement.
    assert all(r['Epoch Terpilih']==25 and r['Epoch Dijalankan']==25 for r in x['foldSummary'] if r['Skenario']=='S2')
    assert x['selected_epochs']=={'S1':9,'S2':25,'S3':12}
    assert c['predictionAudit']['E29']['S3']['within']['5']==40
    assert c['predictionAudit']['E29']['S3']['raw_exact']==0
    assert c['predictionAudit']['E29']['S3']['rounded_exact']==4
    assert c['predictionAudit']['E29']['S2']['within']['10']==0


def test_exactly_three_complete_code_books_but_all_nine_result_sets_remain():
    data=family();c=data['comparison'];chosen=sorted(c['predictionAudit'],key=lambda e:c['predictionAudit'][e]['S3']['rmse'])[:3]
    assert chosen==list(data['documents'])==['E29','E26','E24']
    assert len(c['experiments'])==9 and c['studyRecommendation']['learning_default']=='E29'
    x=c['experiments']['E29'];child=json.loads(re.search(r'<script id="studyData" type="application/json">(.*?)</script>',data['documents']['E29'],re.S)[1])
    notebook=json.loads(subprocess.check_output(['git','show',PIN+':'+x['source_path']],cwd=ROOT))
    assert len(child['cells'])==30 and child['coverage']['annotations']==143 and len(child['screenshotMappings'])==10
    for cell in child['cells']:
        source=notebook['cells'][cell['index']]['source'];source=source if isinstance(source,str) else ''.join(source)
        assert cell['source']==source and cell['sha256']==hashlib.sha256(source.encode()).hexdigest()
        assert all(n['anchor'] in source.splitlines()[n['line']-1] for n in cell['notes'])
    assert 'MLP S2 dan S3 juga5e-6' in child['glossary']['learning rate'][1]
    assert 'E30' not in c['experiments'] and 'E30' not in data['documents']
    assert not data['identities']['E30']['result_available']
    assert sum(x['book_available'] for x in data['identities'].values())==3
