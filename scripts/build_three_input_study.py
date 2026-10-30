"""Buku keluarga E21/E22 dengan eksperimen, data dan sumber kode terpisah."""
import argparse
import base64
import hashlib
import io
import json
import math
from pathlib import Path
import re
from zipfile import ZipFile,ZIP_STORED
from build_e21_study import ROOT,build,audit
from build_mlp_study import BOOKS,table
from three_input_identities import build_identities


def build_family(e21,e22,e24=None,e23=None,reuse_audit=False,e25=None,e26=None,e27=None,e28=None,e29=None,e30=None,e31=None,e32=None):
    documents={};snapshots={};pairs={};data={};zips={}
    inputs=[('E21',e21),('E22',e22)]+[(e,p) for e,p in [('E23',e23),('E24',e24),('E25',e25),('E26',e26),('E27',e27),('E28',e28),('E29',e29),('E30',e30),('E31',e31),('E32',e32)] if p]
    for experiment,path in inputs:
        zips[experiment]=Path(path).read_bytes()
        if reuse_audit:
            cached=json.loads((ROOT/('docs/snapshots/'+experiment.lower()+'_study_snapshot.json')).read_text())
            assert cached['zip_sha256']==hashlib.sha256(zips[experiment]).hexdigest(),'ZIP berbeda dari audit tersimpan.'
        snapshots[experiment]=cached if reuse_audit else audit(path,experiment)
        if not reuse_audit:(ROOT/('docs/snapshots/'+experiment.lower()+'_study_snapshot.json')).write_text(json.dumps(snapshots[experiment],ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        with ZipFile(io.BytesIO(zips[experiment])) as z:
            pairs[experiment]=table(z.read(BOOKS[3]),'Nilai Guru dan Prediksi')
            data[experiment]=table(z.read(BOOKS[0]),'Data dan Nilai Guru')
    assert all(data['E21']==v for v in data.values()),'Data/teks/label eksperimen berbeda.'
    with ZipFile(io.BytesIO(zips['E21'])) as a:
        for e in zips:
            with ZipFile(io.BytesIO(zips[e])) as b:
                for name in ['Pembagian Data','Pembagian Lima Fold']:
                    assert table(a.read(BOOKS[0]),name)==table(b.read(BOOKS[0]),name),'Split/fold berbeda.'
    lookup={e:{r['ID Jawaban']:r for r in rows} for e,rows in pairs.items()}
    assert len(lookup['E21'])==60
    assert all(set(v)==set(lookup['E21']) for v in lookup.values())
    assert all(r['Nilai Guru']==v[i]['Nilai Guru'] for i,r in lookup['E21'].items() for v in lookup.values())
    keys=[e+'_'+s for e in snapshots for s in ['S1','S2','S3']]
    nearest={};impact=[]
    for scenario in ['S1','S2','S3','all']:
        entries={k:{'unique':0,'tied':0,'share':0.} for k in keys if scenario=='all' or k.endswith(scenario)}
        for i,r in lookup['E21'].items():
            error={k:abs(lookup[k.split('_')[0]][i]['Prediksi '+k.split('_')[1]]-r['Nilai Guru']) for k in entries}
            minimum=min(error.values());best=[k for k,v in error.items() if abs(v-minimum)<=1e-5]
            for k in best:entries[k]['unique' if len(best)==1 else 'tied']+=1;entries[k]['share']+=1/len(best)
        assert abs(sum(x['share'] for x in entries.values())-60)<1e-8
        nearest[scenario]=entries
    for s in ['S1','S2','S3']:
        lower=[abs(lookup['E22'][i]['Prediksi '+s]-r['Nilai Guru'])-abs(r['Prediksi '+s]-r['Nilai Guru']) for i,r in lookup['E21'].items()]
        impact.append({'Skenario':s,'E21 lebih dekat':sum(x>1e-5 for x in lower),'E22 lebih dekat':sum(x< -1e-5 for x in lower),'Seri':sum(abs(x)<=1e-5 for x in lower)})
    parameters={e:{r['Pengaturan']:r['Nilai'] for r in s['parameters']} for e,s in snapshots.items()}
    assert all(parameters['E21'][k]==parameters['E22'][k] for k in parameters['E21'] if k!='Format input tiga teks')
    prediction_audit={}
    for experiment,rows in pairs.items():
        prediction_audit[experiment]={}
        for scenario in ['S1','S2','S3']:
            errors=[r['Prediksi '+scenario]-r['Nilai Guru'] for r in rows]
            # Same nearest-category rule as the evaluator: ties choose the lower integer.
            rounded=[math.ceil(min(100,max(0,r['Prediksi '+scenario]))-.5) for r in rows]
            prediction_audit[experiment][scenario]={
                'n':len(rows),'raw_exact':sum(e==0 for e in errors),
                'rounded_exact':sum(v==r['Nilai Guru'] for v,r in zip(rounded,rows)),
                'within':{str(t):sum(abs(e)<=t for e in errors) for t in [1,2,5,10]},
                'mae':sum(abs(e) for e in errors)/len(rows),
                'rmse':math.sqrt(sum(e*e for e in errors)/len(rows))}
    paired_overall={'answers':60,'comparisons':180,
        'E21':sum(r['E21 lebih dekat'] for r in impact),
        'E22':sum(r['E22 lebih dekat'] for r in impact),
        'ties':sum(r['Seri'] for r in impact)}
    pairwise=[]
    from itertools import combinations
    for left,right in combinations(snapshots,2):
        impacts=[]
        for scenario in ['S1','S2','S3']:
            diff=[abs(lookup[right][i]['Prediksi '+scenario]-r['Nilai Guru'])-abs(lookup[left][i]['Prediksi '+scenario]-r['Nilai Guru']) for i,r in lookup[left].items()]
            impacts.append({'scenario':scenario,'left':sum(v>1e-5 for v in diff),'right':sum(v< -1e-5 for v in diff),'ties':sum(abs(v)<=1e-5 for v in diff)})
        pairwise.append({'left':left,'right':right,'impact':impacts,'overall':{k:sum(r[k] for r in impacts) for k in ['left','right','ties']}})
    for e in ('E23','E24','E25','E26'):
        if e in snapshots:assert all(parameters['E21'][k]==parameters[e][k] for k in parameters['E21'])
    for e in ('E27','E28'):
        if e not in snapshots:continue
        canonical={**parameters[e],'Learning rate head':parameters[e]['Learning rate head kontrol S1 S2']}
        assert all(parameters['E21'][k]==canonical[k] for k in parameters['E21'])
    for e in ('E29','E30','E31','E32'):
        if e in parameters:
            canonical={**parameters[e],'Learning rate head':parameters[e]['LR head linear S1']}
            assert all(parameters['E21'][k]==canonical[k] for k in parameters['E21'])
    most=max(v['share'] for v in nearest['S3'].values())
    candidates=[e for e in snapshots if abs(nearest['S3'][e+'_S3']['share']-most)<1e-8]
    primary=min(candidates,key=lambda e:prediction_audit[e]['S3']['rmse'])
    recommendation={'experiment':primary,'nearest_co_leaders':candidates,'nearest_share':most,
        'rule':'Most nearest S3 predictions among uploaded experiments; break tie using lower test RMSE. Descriptive learning default, not a new validated model selection.',
        'rmse_best':min(snapshots,key=lambda e:prediction_audit[e]['S3']['rmse']),
        'within5_best':max(snapshots,key=lambda e:prediction_audit[e]['S3']['within']['5'])}
    if e27 or e28 or e29 or e30 or e31 or e32:
        recommendation['learning_default']='E23'
        recommendation['learning_default_reason']='Preserve existing E23 study code/content and explicit user choice; rankings are recomputed independently.'
    study_ids=sorted(snapshots,key=lambda e:prediction_audit[e]['S3']['rmse'])[:3] if e29 or e30 or e31 or e32 else list(snapshots)
    if e29 or e30 or e31 or e32:
        recommendation['learning_default']=study_ids[0]
        recommendation['learning_default_reason']='Latest user instruction: embed only three code books with lowest S3 test RMSE. Descriptive shortlist on repeatedly observed test, not fresh validation.'
        recommendation['learning_experiments']=study_ids
        recommendation['learning_rule']='Three lowest S3 test RMSE; other results and all original Excel remain available.'
    previous_path=ROOT/'docs/belajar_Tiga_Input_Spider.html'
    previous=json.loads(re.search(r'<script id="familyData" type="application/json">(.*?)</script>',previous_path.read_text(),re.S)[1]) if previous_path.exists() else {'documents':{},'comparison':{'experiments':{}}}
    documents={e:previous['documents'][e] if e in previous['documents'] and previous['comparison']['experiments'][e]==snapshots[e] else build(None,e,family=True)[0] for e in study_ids}
    comparison={'studyRecommendation':recommendation,'experiments':snapshots,'nearest':nearest,'impact':impact,
        'predictionAudit':prediction_audit,'pairedOverall':paired_overall,'pairwise':pairwise,
        'same_300_texts_labels':True,'same_60_test_ids_labels':True,'same_folds':True,'recorded_parameters_equal_except_format':True}
    if e27 or e28 or e29 or e30 or e31 or e32:
        comparison['recorded_parameters_equal_except_format']=False
        comparison['common_baseline_parameters_equal_except_format']=True
        comparison['parameter_differences']={e:{k:v for k,v in parameters[e].items() if parameters['E21'].get(k)!=v} for e in ('E27','E28','E29','E30','E31','E32') if e in snapshots}
    (ROOT/'docs/snapshots/three_input_comparison_snapshot.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    # Child books keep independent source/curriculum/state. No original data in public srcdoc.
    child_style='<style>.topbar,.hero,.stats-strip,.site-footer,.closing-note,.workspace-switch,#status{display:none!important}main{padding:0!important}.study-layout{margin:0!important;max-width:none!important;width:100%!important}.sticky-note h3{overflow-wrap:anywhere;word-break:break-word}html{scroll-padding-top:0}body{background:#14264d!important}</style>'
    documents={e:d if child_style in d else d.replace('</head>',child_style+'</head>') for e,d in documents.items()}
    stylesheet=re.search(r'<style>(.*?)</style>',next(iter(documents.values())),re.S)[1]
    stylesheet+='\n'+(ROOT/'scripts/templates/three_input_study.css').read_text()+'\n'+(ROOT/'scripts/templates/three_input_identities.css').read_text()
    symbols=(ROOT/'scripts/assets/paper_spider_symbols.svg').read_text()
    family_data={'comparison':comparison,'documents':documents,'identities':build_identities(snapshots,set(study_ids))}
    template=(ROOT/'scripts/templates/three_input_study.html').read_text()
    names={'E21':'single sequence','E22':'context pair','E23':'BiGRU S3','E24':'attention pooling S3',
           'E25':'attention + gated fusion S3','E26':'BiGRU + attention S3','E27':'E23 + co-attention S3','E28':'E23 · LR MLP S3 5e-6','E29':'E26 · LR MLP S2/S3 5e-6','E30':'E26 · co-attention + pooling','E31':'E25 · LR MLP S2/S3 5e-6','E32':'E25 · co-attention + gated fusion'}
    run_ids=list(snapshots);joined=' × '.join(run_ids);dash='–'.join(run_ids)
    word={2:'DUA',3:'TIGA',4:'EMPAT',5:'LIMA',6:'ENAM',7:'TUJUH',8:'DELAPAN',9:'SEMBILAN',10:'SEPULUH',11:'SEBELAS',12:'DUA BELAS'}[len(run_ids)]
    template=template.replace('E21 × E22',joined).replace('E21–E22',dash)
    template=template.replace('DUA EKSPERIMEN',word+' EKSPERIMEN').replace('dua eksperimen',str(len(run_ids))+' eksperimen')
    template=template.replace('SATU BUKU, DUA FORMAT','TIGA INPUT, '+word+' EKSPERIMEN')
    template=template.replace('Hasil dua run','Hasil '+str(len(run_ids))+' run').replace('keenam model',str(len(run_ids)*3)+' model')
    template=template.replace('Format <em>berbeda.</em>','Ringkasan <em>berbeda.</em>')
    desc='E21 acuan; E22 konteks pair.'+(' E23 BiGRU;' if e23 else '')+(' E24 attention;' if e24 else '')+(' E25 attention + gate;' if e25 else '')+(' E26 BiGRU + attention.' if e26 else '')
    desc+=(' E27 co-attention dari E23;' if e27 else '')+(' E28 LR MLP S3 5e-6.' if e28 else '')
    desc+=(' E29 LR MLP S2/S3 5e-6 dari E26.' if e29 else '')
    desc+=(' E30 BiGRU + co-attention + attention pooling dari E26.' if e30 else '')
    desc+=(' E31 LR MLP S2/S3 5e-6 dari E25.' if e31 else '')+(' E32 co-attention + attention pooling + gate dari E25, tanpa BiGRU.' if e32 else '')
    template=template.replace('E21 membaca satu urutan teks. E22 membaca pasangan konteks dan jawaban.',desc)
    template=template.replace('E21 single sequence · E22 context pair.',desc)
    study_default=recommendation.get('learning_default',primary)
    options=''.join('<option value="'+e+'"'+(' selected' if e==study_default else '')+'>'+e+' · '+names[e]+'</option>' for e in study_ids)
    template=re.sub(r'(<select id="familyExperiment"[^>]*>).*?(</select>)',lambda m:m[1]+options+m[2],template,flags=re.S)
    template=template.replace('Meja belajar →','Pelajari kode '+study_default+' →').replace('>Meja belajar</button>','>Meja belajar '+study_default+'</button>')
    if e29 or e30 or e31 or e32:template=template.replace('Pilih kode & hasil','Kode tiga RMSE S3 terendah').replace('kode utama, catatan sejajar dan lima Excel.','kode utama; hanya tiga buku kode dimuat. Semua hasil/Excel tetap tersedia.')
    template=template.replace('Kode lengkap, sticky note, output tahap dan semua sheet.',study_default+' · '+names[study_default]+' · kode utama, catatan sejajar dan lima Excel.')
    document=(template.replace('__STYLE__',stylesheet).replace('__SYMBOLS__',symbols)
        .replace('__DATA__',json.dumps(family_data,ensure_ascii=False,allow_nan=False).replace('<','\\u003c'))
        .replace('__VENDOR__',(ROOT/'scripts/vendor/mlp_jszip.min.js').read_text())
        .replace('__SCRIPT__',(ROOT/'scripts/templates/three_input_study.js').read_text().replace('__IDENTITY_SCRIPT__',(ROOT/'scripts/templates/three_input_identities.js').read_text()).replace('__EXCEL_SCRIPT__',(ROOT/'scripts/templates/three_input_excel.js').read_text()).replace('</script','<\\/script')))
    public=ROOT/'docs/belajar_Tiga_Input_Spider.html';public.write_text(document.replace('__PAYLOAD__',''))
    buf=io.BytesIO()
    with ZipFile(buf,'w',ZIP_STORED) as z:
        for e,raw in zips.items():z.writestr(e+'.zip',raw)
    target=ROOT/'outputs/e21_study/belajar_Tiga_Input_Privat.html'
    encoded=base64.b64encode(buf.getvalue()).decode()
    # Node teks tunggal >300MB mahal diparse browser. Potong transport saja;
    # arsip/Excel di dalamnya tetap byte-identik dan diperiksa dengan SHA256.
    chunks=''.join('<script type="application/octet-stream" data-family-chunk>'+encoded[i:i+1048576]+'</script>' for i in range(0,len(encoded),1048576))
    placeholder='<script id="familyPrivate" type="application/octet-stream"></script>'
    target.write_text(document.replace('__PAYLOAD__','').replace(placeholder,placeholder+chunks))
    print(json.dumps({'public':str(public),'private':str(target),'metrics':{e:s['metrics'] for e,s in snapshots.items()},'impact':impact,'same_data_labels_folds':True}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--e21',type=Path,required=True);p.add_argument('--e22',type=Path,required=True);p.add_argument('--e24',type=Path);p.add_argument('--e23',type=Path);p.add_argument('--e25',type=Path);p.add_argument('--e26',type=Path);p.add_argument('--e27',type=Path);p.add_argument('--e28',type=Path);p.add_argument('--e29',type=Path);p.add_argument('--e30',type=Path);p.add_argument('--e31',type=Path);p.add_argument('--e32',type=Path);p.add_argument('--reuse-audit',action='store_true');a=p.parse_args();build_family(a.e21,a.e22,a.e24,a.e23,a.reuse_audit,a.e25,a.e26,a.e27,a.e28,a.e29,a.e30,a.e31,a.e32)
