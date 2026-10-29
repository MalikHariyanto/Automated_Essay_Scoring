"""Actual E30 evidence, two attention stages, three code books and nine old results."""
import hashlib,json,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE='f07d407288e99e913b01427a9170f7e9d800d92d'
PIN='7a76191be20604482fa53f770699563b380120ba'

def family(raw=None):
 raw=raw or (ROOT/'docs/belajar_Tiga_Input_Spider.html').read_text()
 return json.loads(re.search(r'<script id="familyData" type="application/json">(.*?)</script>',raw,re.S)[1])


def test_nine_old_snapshots_predictions_and_all_36_duels_remain_identical():
 old=family(subprocess.check_output(['git','show',BASE+':docs/belajar_Tiga_Input_Spider.html'],cwd=ROOT,text=True));new=family()
 for e,snapshot in old['comparison']['experiments'].items():
  assert new['comparison']['experiments'][e]==snapshot
  assert new['comparison']['predictionAudit'][e]==old['comparison']['predictionAudit'][e]
 for e in ('E29','E26'):assert new['documents'][e]==old['documents'][e]
 pairs={(p['left'],p['right']):p for p in new['comparison']['pairwise']}
 assert all(pairs[p['left'],p['right']]==p for p in old['comparison']['pairwise'])
 assert len(pairs)==45


def test_e30_metric_fold_mask_and_three_weight_groups_are_verified():
 d=family();c=d['comparison'];s=c['experiments']['E30'];p={r['Pengaturan']:r['Nilai'] for r in s['parameters']}
 assert s['source_commit']==PIN and not s['runtime_source_matches_commit']
 assert s['metric_recompute_max_delta']<1e-6 and s['s3_pooling']=='bigru_coattention_pooling' and s['s3_fusion']=='concat'
 assert p['LR MLP S2']==p['LR MLP S3']==p['LR head linear S1']==p['LR pooling dan norm S3']==p['Learning rate proyeksi BGE']==1e-4
 assert p['Learning rate encoder RoBERTa S1']==p['Learning rate encoder RoBERTa S3']==5e-6
 a=s['coattention_audit'];assert a['answers']==300 and a['summary_dimensions']==128 and a['pooling_dimensions']==768
 assert a['response_weight_rows']==a['top1_rows']==16946 and a['reference_weight_rows']==17600
 assert a['max_sum_error']<1e-5 and a['token_positions_verified']
 assert s['attention_audit']['token_positions_verified'] and s['attention_audit']['max_sum_error']<1e-5
 m=next(r for r in s['metrics'] if r['Skenario']=='S3');assert abs(m['RMSE (poin)']-5.524711834785229)<1e-12
 assert s['selected_epochs']=={'S1':9,'S2':19,'S3':18}
 assert c['predictionAudit']['E30']['S3']['within']['5']==44 and c['predictionAudit']['E30']['S3']['within']['10']==56
 assert c['predictionAudit']['E30']['S3']['raw_exact']==0
 assert c['studyRecommendation']['rmse_best']=='E29' and c['studyRecommendation']['within5_best']=='E30'
 assert c['studyRecommendation']['experiment']=='E30' and c['studyRecommendation']['learning_default']=='E29'
 assert c['nearest']['S3']['E30_S3']['share']==13
 assert c['predictionAudit']['E30']['S3']['mae'] < c['predictionAudit']['E29']['S3']['mae']


def test_three_selected_code_books_and_all_e30_notes_match_the_actual_pinned_source():
 d=family();assert list(d['documents'])==['E29','E30','E26']
 assert len(d['comparison']['experiments'])==10 and all(x['result_available'] for x in d['identities'].values())
 assert sum(x['book_available'] for x in d['identities'].values())==3
 s=d['comparison']['experiments']['E30'];n=json.loads(subprocess.check_output(['git','show',PIN+':'+s['source_path']],cwd=ROOT))
 child=json.loads(re.search(r'<script id="studyData" type="application/json">(.*?)</script>',d['documents']['E30'],re.S)[1])
 assert len(child['cells'])==30 and child['coverage']['annotations']==146 and len(child['screenshotMappings'])==10
 for cell in child['cells']:
  text=n['cells'][cell['index']]['source'];text=text if isinstance(text,str) else ''.join(text)
  assert cell['source']==text and cell['sha256']==hashlib.sha256(text.encode()).hexdigest()
  assert all(note['anchor'] in text.splitlines()[note['line']-1] for note in cell['notes'])
 titles={note['title'] for cell in child['cells'] for note in cell['notes']}
 assert {'Dua tahap attention S3','Attention setelah co-attention','Ringkasan co-attention pooling128'}<=titles
 assert 'MLP S2/S3' in child['glossary']['learning rate'][1] and '1e-4' in child['glossary']['learning rate'][1]
