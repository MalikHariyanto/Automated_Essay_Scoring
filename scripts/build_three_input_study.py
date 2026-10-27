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
from build_e21_study import ROOT,build
from build_mlp_study import BOOKS,table


def build_family(e21,e22,e24=None):
    documents={};snapshots={};pairs={};data={};zips={}
    inputs=[('E21',e21),('E22',e22)]+([('E24',e24)] if e24 else [])
    for experiment,path in inputs:
        documents[experiment],snapshots[experiment]=build(path,experiment,family=True)
        zips[experiment]=Path(path).read_bytes()
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
    if e24:
        assert all(parameters['E21'][k]==parameters['E24'][k] for k in parameters['E21'])
    comparison={'experiments':snapshots,'nearest':nearest,'impact':impact,
        'predictionAudit':prediction_audit,'pairedOverall':paired_overall,'pairwise':pairwise,
        'same_300_texts_labels':True,'same_60_test_ids_labels':True,'same_folds':True,'recorded_parameters_equal_except_format':True}
    (ROOT/'docs/three_input_comparison_snapshot.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    # Child books keep independent source/curriculum/state. No original data in public srcdoc.
    child_style='<style>.topbar,.hero,.stats-strip,.site-footer,.closing-note,.workspace-switch,#status{display:none!important}main{padding:0!important}.study-layout{margin:0!important;max-width:none!important;width:100%!important}.sticky-note h3{overflow-wrap:anywhere;word-break:break-word}html{scroll-padding-top:0}body{background:#14264d!important}</style>'
    documents={e:d.replace('</head>',child_style+'</head>') for e,d in documents.items()}
    stylesheet=re.search(r'<style>(.*?)</style>',documents['E21'],re.S)[1]
    stylesheet+='\n'+(ROOT/'scripts/templates/three_input_study.css').read_text()
    symbols=(ROOT/'scripts/assets/paper_spider_symbols.svg').read_text()
    family_data={'comparison':comparison,'documents':documents}
    template=(ROOT/'scripts/templates/three_input_study.html').read_text()
    if e24:
        template=template.replace('E21 × E22','E21 × E22 × E24').replace('E21–E22','E21–E22–E24').replace('DUA EKSPERIMEN','TIGA EKSPERIMEN').replace('dua eksperimen','tiga eksperimen').replace('Hasil dua run','Hasil tiga run').replace('keenam model','sembilan model').replace('SATU BUKU, DUA FORMAT','TIGA INPUT, TIGA EKSPERIMEN').replace('Format <em>berbeda.</em>','Ringkasan <em>berbeda.</em>').replace('E22 membaca pasangan konteks dan jawaban.', 'E22 membaca pasangan konteks dan jawaban. E24 menambahkan attention pooling pada S3 dari E21.').replace('E22 context pair.', 'E22 context pair · E24 attention pooling.').replace('</select></label><span id="familyStudyStatus"','<option value="E24">E24 · attention pooling S3</option></select></label><span id="familyStudyStatus"')
    document=(template.replace('__STYLE__',stylesheet).replace('__SYMBOLS__',symbols)
        .replace('__DATA__',json.dumps(family_data,ensure_ascii=False,allow_nan=False).replace('<','\\u003c'))
        .replace('__VENDOR__',(ROOT/'scripts/vendor/mlp_jszip.min.js').read_text())
        .replace('__SCRIPT__',(ROOT/'scripts/templates/three_input_study.js').read_text().replace('</script','<\\/script')))
    public=ROOT/'docs/belajar_Tiga_Input_Spider.html';public.write_text(document.replace('__PAYLOAD__',''))
    buf=io.BytesIO()
    with ZipFile(buf,'w',ZIP_STORED) as z:
        for e,raw in zips.items():z.writestr(e+'.zip',raw)
    target=ROOT/'outputs/e21_study/belajar_Tiga_Input_Privat.html'
    target.write_text(document.replace('__PAYLOAD__',base64.b64encode(buf.getvalue()).decode()))
    print(json.dumps({'public':str(public),'private':str(target),'metrics':{e:s['metrics'] for e,s in snapshots.items()},'impact':impact,'same_data_labels_folds':True}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--e21',type=Path,required=True);p.add_argument('--e22',type=Path,required=True);p.add_argument('--e24',type=Path);a=p.parse_args();build_family(a.e21,a.e22,a.e24)
