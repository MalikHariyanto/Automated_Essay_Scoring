"""Integritas delapan hasil; tidak menjalankan training atau memakai baris siswa publik."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE='daa745de0aee5c7bd26090af8fbf7ad99b0cb49c'


def family(raw=None):
    raw=raw or (ROOT/'docs/belajar_Tiga_Input_Spider.html').read_text()
    return json.loads(re.search(r'<script id="familyData" type="application/json">(.*?)</script>',raw,re.S)[1])


def test_six_original_books_all_content_numbers_and_duels_are_unchanged():
    old=family(subprocess.check_output(['git','show',BASE+':docs/belajar_Tiga_Input_Spider.html'],cwd=ROOT,text=True))
    new=family();a=old['comparison'];b=new['comparison']
    for e in a['experiments']:
        assert a['experiments'][e]==b['experiments'][e]
        assert a['predictionAudit'][e]==b['predictionAudit'][e]
        # Seluruh HTML buku lama: kode, sticky note, penjelasan dan sumber utuh.
        if e in new['documents']:assert old['documents'][e]==new['documents'][e]
    old_pairs={(p['left'],p['right']):p for p in a['pairwise']}
    assert all(old_pairs[(p['left'],p['right'])]==p for p in b['pairwise'] if (p['left'],p['right']) in old_pairs)
    assert b['impact']==a['impact'] and b['pairedOverall']==a['pairedOverall']
    assert a['nearest']['S3']['E23_S3']['share']==13 and b['nearest']['S3']['E23_S3']['share']==8
    assert b['studyRecommendation']['learning_default']=='E29'


def test_uploaded_results_lr_and_coattention_are_distinct_and_verified():
    c=family()['comparison'];a=c['experiments']['E27'];b=c['experiments']['E28']
    for e,x,expected in [('E27',a,6.32765108624205),('E28',b,7.044717900803215)]:
        assert x['source_commit']==BASE and x['runtime_source_matches_commit'] is False
        assert x['metric_recompute_max_delta']<1e-6 and x['test']==60 and x['rows']==300
        assert abs(next(r for r in x['metrics'] if r['Skenario']=='S3')['RMSE (poin)']-expected)<1e-12
        assert len(x['books'])==5
        p={r['Pengaturan']:r['Nilai'] for r in x['parameters']}
        assert p['LR MLP S3 saja']==(5e-6 if e=='E28' else 1e-4)
        assert p['LR pooling dan norm S3']==p['Learning rate head kontrol S1 S2']==1e-4
    audit=a['coattention_audit']
    assert audit['response_weight_rows']==audit['top1_rows']==16946 and audit['reference_weight_rows']==17600
    assert audit['max_sum_error']<1e-5 and audit['token_positions_verified'] and audit['finite']
    assert a['s3_pooling']=='bigru_coattention' and b['s3_pooling']=='bigru'
    assert b['bigru_audit']['final_state_dimensions']==128
    assert c['recorded_parameters_equal_except_format'] is False
    assert c['predictionAudit']['E27']['S3']['within']['5']==34
    assert c['predictionAudit']['E28']['S3']['within']['5']==27
    for e in ('E27','E28'):
        pair=next(p for p in c['pairwise'] if (p['left'],p['right'])==('E23',e))
        assert pair['impact'][:2]==[{'scenario':'S1','left':0,'right':0,'ties':60},{'scenario':'S2','left':0,'right':0,'ties':60}]


def test_all_new_source_cells_and_margin_notes_are_traceable_to_published_notebooks():
    data=family()
    for e in ('E27','E28'):
        old=family(subprocess.check_output(['git','show','b783fe05896cd4a0efcc4a81508d70b0eedbd717:docs/belajar_Tiga_Input_Spider.html'],cwd=ROOT,text=True))
        assert e not in data['documents']  # Latest user instruction: only three selected code books.
        child=json.loads(re.search(r'<script id="studyData" type="application/json">(.*?)</script>',old['documents'][e],re.S)[1])
        snap=data['comparison']['experiments'][e]
        notebook=json.loads(subprocess.check_output(['git','show',BASE+':'+snap['source_path']],cwd=ROOT))
        for cell in child['cells']:
            assert cell['source']==notebook['cells'][cell['index']]['source']
            assert cell['sha256']==hashlib.sha256(cell['source'].encode()).hexdigest()
            assert all(n['anchor'] in cell['source'].splitlines()[n['line']-1] for n in cell['notes'])
        assert len(child['screenshotMappings'])==10 and child['coverage']['code_cells']==30
        lr=child['glossary']['learning rate'][1]
        assert 'MLP S3 '+('5e-6' if e=='E28' else '1e-4') in lr
        if e=='E27':
            assert 'Co-attention' in str([n['title'] for c in child['cells'] for n in c['notes']])
            assert child['studySnapshot']['coattention_audit']['top1_rows']==16946
