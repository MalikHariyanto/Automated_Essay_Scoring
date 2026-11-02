"""Uploaded E35 results: provenance, mixed performance and historical preservation."""
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = '699aaa3be612400ccf264bd8a8b96984263c8c8c'


def family(ref=None):
    raw = subprocess.check_output(['git','show',ref+':docs/belajar_Tiga_Input_Spider.html'],cwd=ROOT,text=True) if ref else (ROOT/'docs/belajar_Tiga_Input_Spider.html').read_text()
    return json.loads(re.search(r'<script id="familyData" type="application/json">(.*?)</script>',raw,re.S)[1])


def test_all_fourteen_prior_results_duels_and_code_books_are_unchanged():
    old,new=family(BASE),family()
    assert old['documents']==new['documents']
    for e,s in old['comparison']['experiments'].items():
        assert new['comparison']['experiments'][e]==s
        assert new['comparison']['predictionAudit'][e]==old['comparison']['predictionAudit'][e]
        assert new['identities'][e]==old['identities'][e]
    pairs={(r['left'],r['right']):r for r in new['comparison']['pairwise']}
    for r in old['comparison']['pairwise']:
        assert pairs[r['left'],r['right']]==r
    assert len(pairs)==105
    for p in ['docs/belajar_MLP_Spider.html',*map(str,(ROOT/'downloads').glob('Belajar_AES_MLP_Lengkap*'))]:
        p=str(Path(p).relative_to(ROOT)) if Path(p).is_absolute() else p
        assert subprocess.check_output(['git','show',BASE+':'+p],cwd=ROOT)==(ROOT/p).read_bytes()


def test_e35_source_recipe_attention_and_selected_epochs():
    f=family();c=f['comparison'];s=c['experiments']['E35'];old=c['experiments']['E30']
    params=lambda x:{r['Pengaturan']:r['Nilai'] for r in x['parameters']}
    a,b=params(s),params(old)
    assert {k:(b.get(k),v) for k,v in a.items() if b.get(k)!=v}=={'LR pooling dan norm S3':(1e-4,5e-5)}
    assert a['LR MLP S3']==a['LR MLP S2']==a['Learning rate proyeksi BGE']==1e-4
    assert a['Learning rate encoder RoBERTa S3']==5e-6
    assert a['Jumlah fold']==a['Batas epoch tanpa perbaikan']==5 and a['Maksimum epoch per fold']==25
    assert s['rows']==300 and s['development']==240 and s['test']==60
    assert s['s3_pooling']=='bigru_coattention_pooling' and s['s3_fusion']=='concat'
    assert s['metric_recompute_max_delta']<1e-6 and not s['runtime_source_matches_commit']
    assert s['selected_epochs']=={'S1':9,'S2':19,'S3':11}
    assert s['attention_audit']['token_positions_verified'] and s['attention_audit']['max_sum_error']<1e-5
    assert s['coattention_audit']['summary_dimensions']==128 and s['coattention_audit']['token_positions_verified']
    source=json.loads(subprocess.check_output(['git','show',s['source_commit']+':'+s['source_path']],cwd=ROOT))
    assert source['metadata']['aes_experiment']['name']=='E35' and source['metadata']['aes_standalone']['native_kernel']
    assert not source['metadata']['aes_experiment']['inspection_cosine']
    assert next(x for x in s['books'][2]['sheets'] if x['name']=='Kata Berulang')['columns']==8
    for scenario in ['S1','S2','S3']:
        for r in (r for r in s['foldSummary'] if r['Skenario']==scenario):
            trace=[h for h in s['history'][scenario] if h['Tahap']=='Validasi silang' and h['Fold']==r['Fold']]
            chosen=next(h for h in trace if h['Epoch']==r['Epoch Terpilih'])
            assert chosen['RMSE Validasi (poin)']==min(h['RMSE Validasi (poin)'] for h in trace)
            assert len(trace)==r['Epoch Dijalankan']<=25


def test_e35_cv_improves_while_test_does_not_replace_e30():
    c=family()['comparison'];a,b,d=[c['predictionAudit'][e]['S3'] for e in ['E35','E30','E34']]
    assert b['rmse']<a['rmse']<d['rmse']
    assert b['mae']<d['mae']<a['mae']
    assert a['within']=={'1':12,'2':18,'5':38,'10':56}
    assert a['rounded_exact']==4 and a['raw_exact']==0
    cv=lambda e:next(r for r in c['experiments'][e]['cvSummary'] if r['Skenario']=='S3')
    assert cv('E29')['mean']<cv('E35')['mean']<cv('E32')['mean']<cv('E34')['mean']<cv('E33')['mean']<cv('E30')['mean']
    assert cv('E35')['sd']<cv('E34')['sd']
    for e,expected in [('E30',(31,29)),('E33',(28,32)),('E34',(28,32))]:
        duel=next(r for r in c['pairwise'] if (r['left'],r['right'])==(e,'E35'))
        assert duel['impact'][:2]==[{'scenario':s,'left':0,'right':0,'ties':60} for s in ['S1','S2']]
        assert duel['impact'][2]=={'scenario':'S3','left':expected[0],'right':expected[1],'ties':0}
    assert c['studyRecommendation']['rmse_best']=='E29'
    assert c['studyRecommendation']['within5_best']=='E30'


def test_fifteen_results_and_all_original_excel_remain_accessible():
    f=family();c=f['comparison']
    assert list(c['experiments'])==[f'E{n}' for n in range(21,36)]
    assert list(f['documents'])==['E29','E30','E26']
    assert len(f['identities'])==15 and sum(len(x['books']) for x in c['experiments'].values())==75
    assert sum(len(b['sheets']) for s in c['experiments'].values() for b in s['books'])==704
    identity=f['identities']['E35']
    assert identity['base']=='E30' and identity['result_available'] and not identity['book_available']
    assert 'MLP S2/S3 1e-4' in identity['head_lr']
    assert 'sequence_pool/roberta_norm S3 5e-5' in identity['other_lr']
    for counts in c['nearest'].values():assert abs(sum(x['share'] for x in counts.values())-60)<1e-8
