"""Uploaded E33 evidence, preserved historical results, and correct public methods."""
import hashlib,json,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE='fa45c47a4d972a20a36445ea58d1fca8e8adcded'

def family(raw=None):
    raw=raw or (ROOT/'docs/belajar_Tiga_Input_Spider.html').read_text()
    return json.loads(re.search(r'<script id="familyData" type="application/json">(.*?)</script>',raw,re.S)[1])


def test_preserve_twelve_experiments_code_books_and_all_old_duels():
    old=family(subprocess.check_output(['git','show',BASE+':docs/belajar_Tiga_Input_Spider.html'],cwd=ROOT,text=True));new=family()
    for e in old['documents'].keys() & new['documents'].keys():assert old['documents'][e]==new['documents'][e]
    for e,s in old['comparison']['experiments'].items():
        assert new['comparison']['experiments'][e]==s
        assert new['comparison']['predictionAudit'][e]==old['comparison']['predictionAudit'][e]
        assert new['identities'][e]=={**old['identities'][e],'book_available':e in new['documents']}
    pairs={(p['left'],p['right']):p for p in new['comparison']['pairwise']}
    assert all(pairs[p['left'],p['right']]==p for p in old['comparison']['pairwise'])
    for f in ['docs/belajar_MLP_Spider.html',*map(str,(ROOT/'downloads').glob('Belajar_AES_MLP_Lengkap*'))]:
        p=Path(f);rel=str(p.relative_to(ROOT)) if p.is_absolute() else str(p)
        assert subprocess.check_output(['git','show',BASE+':'+rel],cwd=ROOT)==(ROOT/rel).read_bytes()


def test_e33_source_hyperparameters_attention_and_epoch_selection():
    s=family()['comparison']['experiments']['E33'];p={r['Pengaturan']:r['Nilai'] for r in s['parameters']}
    assert p['LR MLP S3']==5e-5 and p['LR MLP S2']==p['LR pooling dan norm S3']==p['LR head linear S1']==1e-4
    assert p['Learning rate encoder RoBERTa S3']==5e-6
    assert s['s3_pooling']=='bigru_coattention_pooling' and s['s3_fusion']=='concat'
    assert s['rows']==300 and s['development']==240 and s['test']==60
    assert s['metric_recompute_max_delta']<1e-6 and not s['runtime_source_matches_commit']
    assert s['selected_epochs']=={'S1':9,'S2':19,'S3':11}
    assert s['attention_audit']['token_positions_verified'] and s['coattention_audit']['token_positions_verified']
    assert s['coattention_audit']['summary_dimensions']==128 and s['coattention_audit']['max_sum_error']<1e-5
    source=json.loads(subprocess.check_output(['git','show',s['source_commit']+':'+s['source_path']],cwd=ROOT))
    assert source['metadata']['aes_experiment']['name']=='E33'
    for scenario in ['S1','S2','S3']:
        summary=[r for r in s['foldSummary'] if r['Skenario']==scenario]
        for r in summary:
            trace=[h for h in s['history'][scenario] if h['Tahap']=='Validasi silang' and h['Fold']==r['Fold']]
            chosen=next(h for h in trace if h['Epoch']==r['Epoch Terpilih'])
            assert chosen['RMSE Validasi (poin)']==min(h['RMSE Validasi (poin)'] for h in trace)


def test_e33_comparison_does_not_claim_universal_improvement():
    f=family();c=f['comparison'];a=c['predictionAudit']['E33']['S3'];b=c['predictionAudit']['E30']['S3']
    assert a['rmse']>b['rmse'] and a['mae']>b['mae']
    assert a['within']['5']==37 and b['within']['5']==44
    assert a['within']['1']==14 and a['within']['10']==55
    s=c['experiments']['E33'];old=c['experiments']['E30']
    assert next(r['mean'] for r in s['cvSummary'] if r['Skenario']=='S3')<next(r['mean'] for r in old['cvSummary'] if r['Skenario']=='S3')
    duel=next(p for p in c['pairwise'] if (p['left'],p['right'])==('E30','E33'))
    assert duel['impact']==[{'scenario':'S1','left':0,'right':0,'ties':60},{'scenario':'S2','left':0,'right':0,'ties':60},{'scenario':'S3','left':33,'right':27,'ties':0}]
    assert c['studyRecommendation']['rmse_best']=='E29' and c['studyRecommendation']['within5_best']=='E30'
    assert c['studyRecommendation']['experiment']=='E30'


def test_thirteen_results_independent_workbooks_and_specific_e33_identity():
    f=family();c=f['comparison']
    assert list(c['experiments'])==[f'E{n}' for n in range(21,37)]
    assert len(c['pairwise'])==120 and list(f['documents'])==['E29','E30','E36']
    assert len(f['identities'])==16 and f['identities']['E33']['base']=='E30'
    assert f['identities']['E33']['result_available'] and not f['identities']['E33']['book_available']
    assert 'MLP S2 1e-4 · MLP S3 5e-5' in f['identities']['E33']['head_lr']
    assert sum(len(s['books']) for s in c['experiments'].values())==80
    assert sum(len(b['sheets']) for s in c['experiments'].values() for b in s['books'])==748
    for rows in c['nearest'].values():assert abs(sum(v['share'] for v in rows.values())-60)<1e-8
