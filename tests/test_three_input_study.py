"""Provenance dua eksperimen dan kode yang dibaca pengguna, bukan training baru."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def family():
    raw = (ROOT / 'docs/belajar_Tiga_Input_Spider.html').read_text()
    data = json.loads(re.search(r'<script id="familyData" type="application/json">(.*?)</script>', raw, re.S)[1])
    return raw, data


def test_same_data_labels_folds_and_only_recorded_format_changes():
    _, data = family()
    c = data['comparison']
    for key in ('same_300_texts_labels', 'same_60_test_ids_labels', 'same_folds', 'recorded_parameters_equal_except_format'):
        assert c[key] is True
    for experiment, format_ in [('E21', 'single_sequence'), ('E22', 'context_pair')]:
        x = c['experiments'][experiment]
        assert x['input_format'] == format_
        p = {r['Pengaturan']: r['Nilai'] for r in x['parameters']}
        assert p['Learning rate encoder RoBERTa S1'] == p['Learning rate encoder RoBERTa S3'] == 5e-6
        assert x['rows'] == 300 and x['test'] == 60
        assert x['runtime_source_matches_commit'] is False
        assert x['metric_recompute_max_delta'] < 1e-6


def test_nearest_counts_do_not_replace_rmse_comparison():
    _, d = family()
    c = d['comparison']
    for counts in c['nearest'].values():
        assert abs(sum(r['share'] for r in counts.values()) - 60) < 1e-8
    assert c['impact'][2] == {'Skenario': 'S3', 'E21 lebih dekat': 30, 'E22 lebih dekat': 30, 'Seri': 0}
    m = {e: next(r for r in x['metrics'] if r['Skenario'] == 'S3') for e, x in c['experiments'].items()}
    assert m['E21']['RMSE (poin)'] < m['E22']['RMSE (poin)']
    assert m['E21']['Pearson'] < m['E22']['Pearson']


def test_both_embedded_code_books_are_exact_pinned_notebooks():
    _, family_data = family()
    for e, html in family_data['documents'].items():
        # Child code JSON is escaped, parsed as the browser reads it.
        m = re.search(r'<script id="studyData" type="application/json">(.*?)</script>', html, re.S)
        assert m, e
        data = json.loads(m[1])
        snap = family_data['comparison']['experiments'][e]
        source = json.loads(subprocess.check_output(['git', 'show', snap['source_commit'] + ':' + snap['source_path']], cwd=ROOT))
        assert len(data['cells']) == 30
        for cell in data['cells']:
            raw = source['cells'][cell['index']]['source']
            raw = raw if isinstance(raw, str) else ''.join(raw)
            assert cell['source'] == raw, (e, cell['index'])
            assert cell['sha256'] == hashlib.sha256(raw.encode()).hexdigest()
            assert all(1 <= n['line'] <= len(cell['htmlLines']) for n in cell['notes'])
        if e == 'E22':
            token = next(c for c in data['cells'] if c['index'] == 20)
            note = next(n for n in token['notes'] if n['title'] == 'Pasangan konteks dan jawaban')
            assert 'build_inputs_with_special_tokens(context_ids, second)' in token['source'].splitlines()[note['line'] - 1]
        assert '<script id="privatePayload" type="application/octet-stream"></script>' in html


def test_prediction_accuracy_keeps_scopes_and_raw_vs_rounded_distinct():
    _, d = family()
    c = d['comparison']
    assert c['pairedOverall'] == {'answers': 60, 'comparisons': 180, 'E21': 88, 'E22': 92, 'ties': 0}
    for e in ['E21', 'E22']:
        a = c['predictionAudit'][e]['S3']
        assert a['n'] == 60 and a['raw_exact'] == 0 and a['rounded_exact'] == 5
        assert a['within']['1'] == 10
        assert all(a['within'][str(x)] <= a['within'][str(y)] for x, y in [(1,2),(2,5),(5,10)])
        original = next(r for r in c['experiments'][e]['metrics'] if r['Skenario'] == 'S3')
        assert abs(a['rmse'] - original['RMSE (poin)']) < 1e-6
    assert c['predictionAudit']['E21']['S3']['within']['5'] == 32
    assert c['predictionAudit']['E22']['S3']['within']['5'] == 27


def test_public_books_do_not_embed_individual_payloads():
    raw, d = family()
    assert '<script id="familyPrivate" type="application/octet-stream"></script>' in raw
    forbidden = {'ID Jawaban', 'Jawaban Siswa', 'Nilai Guru', 'Prediksi S1', 'Prediksi S2', 'Prediksi S3', 'pairs'}
    def scan(v):
        if isinstance(v, dict):
            assert not forbidden.intersection(v)
            for x in v.values(): scan(x)
        elif isinstance(v, list):
            for x in v: scan(x)
    scan(d['comparison'])


def test_actual_inline_scripts_parse_without_closing_tag_injection(tmp_path):
    raw, _ = family()
    scripts = re.findall(r'<script>(.*?)</script>', raw, re.S)
    assert len(scripts) == 2
    for i, text in enumerate(scripts):
        script = tmp_path / f'inline_{i}.js'
        script.write_text(text)
        r = subprocess.run(['node', '--check', str(script)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-400:]


def test_e24_attention_results_and_comparisons_are_real_and_consistent():
    _, data = family()
    c = data['comparison']
    assert set(c['experiments']) == set(data['documents']) == {'E21','E22','E23','E24','E25','E26'}
    e = c['experiments']['E24']
    assert e['input_format'] == 'single_sequence' and e['s3_pooling'] == 'attention'
    assert e['attention_audit']['rows'] == 16946
    assert e['attention_audit']['answer_windows'] == 300
    assert e['attention_audit']['max_sum_error'] < 1e-5
    assert 'AES_300_E24_Tiga_Input_AttentionPooling_GPU.ipynb' in e['source_path']
    e21=c['experiments']['E21']
    for scenario in ['S1','S2']:
        assert next(r for r in e['metrics'] if r['Skenario']==scenario) == next(r for r in e21['metrics'] if r['Skenario']==scenario)
    assert abs(next(r for r in e['metrics'] if r['Skenario']=='S3')['RMSE (poin)'] - 5.981825270792521)<1e-10
    assert len(c['pairwise']) == 15
    for p in c['pairwise']:
        assert all(r['left']+r['right']+r['ties']==60 for r in p['impact'])
        assert sum(p['overall'].values())==180
    assert abs(sum(v['share'] for v in c['nearest']['all'].values())-60)<1e-8
    assert len(c['nearest']['all']) == 18


def test_e23_biGRU_and_study_default_follow_actual_prediction_criteria():
    _, data=family();c=data['comparison'];e=c['experiments']['E23']
    assert e['s3_pooling']=='bigru' and e['input_format']=='single_sequence'
    assert e['bigru_audit']=={'answers':300,'final_state_dimensions':128,'pooling_dimensions':768,'finite':True}
    assert 'AES_300_E23_Tiga_Input_BiGRU_GPU.ipynb' in e['source_path']
    assert abs(c['predictionAudit']['E23']['S3']['rmse']-6.043921327113497)<1e-6
    assert c['predictionAudit']['E23']['S3']['within']['5']==39
    assert c['predictionAudit']['E24']['S3']['within']['5']==36
    assert c['nearest']['S3']['E23_S3']['share']==13
    assert c['nearest']['S3']['E24_S3']['share']==9
    assert c['studyRecommendation']['experiment']=='E23'
    assert c['studyRecommendation']['nearest_co_leaders']==['E23']
    assert c['studyRecommendation']['within5_best']=='E23'
    assert c['studyRecommendation']['rmse_best']=='E26'
    pair=next(p for p in c['pairwise'] if (p['left'],p['right'])==('E23','E24'))
    assert pair['impact'][2]=={'scenario':'S3','left':28,'right':32,'ties':0}


def test_screenshot_mappings_and_learning_notes_point_to_actual_source_lines():
    _, data=family()
    for experiment,html in data['documents'].items():
        d=json.loads(re.search(r'<script id="studyData" type="application/json">(.*?)</script>',html,re.S)[1])
        cells={cell['index']:cell for cell in d['cells']}
        assert len(d['screenshotMappings'])==10 and d['learningLayout']=='margin'
        for m in d['screenshotMappings']:
            assert m['anchor'] in cells[m['cell']]['source'].splitlines()[m['line']-1]
            assert any(n['line']==m['line'] and n.get('screenshot')==m['title'] for n in cells[m['cell']]['notes'])
        for cell in cells.values():
            for n in cell['notes']:
                assert n['anchor'] in cell['source'].splitlines()[n['line']-1]
        model=cells[30]
        assert any(n['title']=='Head regresi linear S1' for n in model['notes'])
        assert any(n['title']=='Head regresi MLP S3' for n in model['notes'])
        if experiment=='E23':
            assert any(n['title']=='Proyeksi ringkasan BiGRU' and '128' in n['body'] for n in model['notes'])
        if experiment=='E24':
            assert any(n['title']=='Bobot attention bertopeng' for n in model['notes'])


def test_e25_gate_audit_and_e26_attention_use_actual_different_architectures():
    _,data=family();c=data['comparison'];e25=c['experiments']['E25'];e26=c['experiments']['E26']
    assert e25['s3_pooling']=='attention' and e25['s3_fusion']=='gated_concat'
    gate=e25['gate_audit']
    assert gate['answers']==300 and gate['scalar_per_answer'] and gate['multipliers_verified']
    assert 0.495<gate['range'][0]<gate['mean']<gate['range'][1]<0.498
    assert e26['s3_pooling']=='bigru_attention' and e26['s3_fusion']=='concat'
    assert e26['bigru_attention_audit']=={'answers':300,'summary_dimensions':128,'pooling_dimensions':768,'finite':True}
    for s in [e25,e26]:
        assert s['source_commit']=='bc4713872c7a7ce23f7dad8ae3ea73a2b75d9b86'
        assert s['attention_audit']['rows']==16946 and s['attention_audit']['answer_windows']==300
        assert s['attention_audit']['max_sum_error']<1e-5
        assert s['metric_recompute_max_delta']<1e-6
    for experiment,required in [('E25',['Awal sama dengan concat E24','Concat aktual berbobot','Ekspor gate final']),('E26',['Semua output BiGRU','Ringkasan BiGRU attention128','Bobot kembali ke posisi asli'])]:
        child=json.loads(re.search(r'<script id="studyData" type="application/json">(.*?)</script>',data['documents'][experiment],re.S)[1])
        titles={n['title'] for cell in child['cells'] for n in cell['notes']}
        assert set(required)<=titles
        assert child['coverage']['annotations']>=140


def test_e26_rmse_lead_is_not_confused_with_nearest_or_duel_wins():
    _,data=family();c=data['comparison']
    assert abs(c['predictionAudit']['E25']['S3']['rmse']-6.105320784328023)<1e-6
    assert abs(c['predictionAudit']['E26']['S3']['rmse']-5.743464722338123)<1e-6
    assert c['predictionAudit']['E26']['S3']['mae']<c['predictionAudit']['E24']['S3']['mae']
    assert c['predictionAudit']['E26']['S3']['within']['5']==38
    assert c['predictionAudit']['E25']['S3']['within']['5']==35
    assert {k:v['share'] for k,v in c['nearest']['S3'].items()}=={'E21_S3':11,'E22_S3':9,'E23_S3':13,'E24_S3':9,'E25_S3':8,'E26_S3':10}
    pair=next(p for p in c['pairwise'] if (p['left'],p['right'])==('E24','E26'))
    assert pair['impact'][2]=={'scenario':'S3','left':30,'right':30,'ties':0}
    pair=next(p for p in c['pairwise'] if (p['left'],p['right'])==('E25','E26'))
    assert pair['impact'][2]=={'scenario':'S3','left':31,'right':29,'ties':0}
