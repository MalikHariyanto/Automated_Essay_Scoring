"""Buku E21 baru; audit Excel, agregat publik dan isi lengkap privat terpisah."""
import argparse
import ast
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
from zipfile import ZipFile
import numpy as np
from build_mlp_study import ROOT,BOOKS,table,workbook_inventory,highlighted_lines,source
from e21_study_content import curriculum,PINS,HINTS
from three_input_learning_notes import enrich_learning_notes

PIN='ef712b4c035d96178d9baa67caa7453ac9fd8738'
NOTEBOOK='notebooks/experiments/AES_300_E21_Tiga_Input_GPU.ipynb'
E24_PIN='f3d25c4'  # Source notebook yang dikirim sebelum ZIP hasil E24.
E24_NOTEBOOK='notebooks/experiments/AES_300_E24_Tiga_Input_AttentionPooling_GPU.ipynb'
E23_NOTEBOOK='notebooks/experiments/AES_300_E23_Tiga_Input_BiGRU_GPU.ipynb'
E25_PIN='bc4713872c7a7ce23f7dad8ae3ea73a2b75d9b86'
E25_NOTEBOOK='notebooks/experiments/AES_300_E25_Tiga_Input_Attention_GatedFusion_GPU.ipynb'
E26_NOTEBOOK='notebooks/experiments/AES_300_E26_Tiga_Input_BiGRU_Attention_GPU.ipynb'
E27_PIN='daa745de0aee5c7bd26090af8fbf7ad99b0cb49c'
E27_NOTEBOOK='notebooks/experiments/AES_300_E27_E23_BiGRU_CoAttention_GPU.ipynb'
E28_NOTEBOOK='notebooks/experiments/AES_300_E28_E23_BiGRU_MLPLR5e6_GPU.ipynb'


def audit_coattention(blobs, data):
    """Audit ekspor asli tanpa mempublikasikan token/ID individual."""
    from collections import defaultdict
    weights={name:table(blobs[BOOKS[1]],name) for name in ('Bobot CoAttention Respons S3','Bobot Referensi S3')}
    tokens=table(blobs[BOOKS[0]],'Token RoBERTa')
    lookup={(r['ID Jawaban'],r['Bagian Teks'],r['Posisi Token']):r for r in tokens}
    sums={};ids={r['ID Jawaban'] for r in data}
    for name,rows in weights.items():
        sums[name]=defaultdict(float)
        for r in rows:
            w=r['Bobot Attention'];assert np.isfinite(w) and 0<=w<=1
            key=(r['ID Jawaban'],r['Bagian Teks'],r['Posisi Token']);t=lookup[key]
            assert t['Token']==r['Token'] and t['ID Token']==r['ID Token']
            assert t['Bagian Input']==('Jawaban siswa' if 'Respons' in name else 'Isi referensi')
            sums[name][key[:2]]+=w
        assert {k[0] for k in sums[name]}==ids
        assert all(abs(v-1)<1e-5 for v in sums[name].values())
    top=table(blobs[BOOKS[1]],'Pasangan Token Top1 S3')
    response_keys={(r['ID Jawaban'],r['Bagian Teks'],r['Posisi Token']) for r in weights['Bobot CoAttention Respons S3']}
    assert len(top)==len(response_keys) and len({(r['ID Jawaban'],r['Bagian Teks'],r['Posisi Token Jawaban']) for r in top})==len(top)
    assert {(r['ID Jawaban'],r['Bagian Teks'],r['Posisi Token Jawaban']) for r in top}==response_keys
    for r in top:
        a=lookup[(r['ID Jawaban'],r['Bagian Teks'],r['Posisi Token Jawaban'])]
        b=lookup[(r['ID Jawaban'],r['Bagian Teks'],r['Posisi Token Referensi'])]
        assert a['Token']==r['Token Jawaban'] and a['Bagian Input']=='Jawaban siswa'
        assert b['Token']==r['Token Referensi'] and b['Bagian Input']=='Isi referensi'
        assert np.isfinite(r['Bobot Kecocokan']) and 0<=r['Bobot Kecocokan']<=1
    vectors=table(blobs[BOOKS[1]],'Ringkasan CoAttention S3');pooled=table(blobs[BOOKS[1]],'Pooling Jawaban S3')
    for rows,dimensions in [(vectors,128),(pooled,768)]:
        assert len(rows)==300 and all(len(r)==dimensions+1 for r in rows)
        assert [r['ID Jawaban'] for r in rows]==[r['ID Jawaban'] for r in data]
        assert all(np.isfinite(v) for r in rows for k,v in r.items() if k!='ID Jawaban')
    return {'answers':300,'summary_dimensions':128,'pooling_dimensions':768,'response_weight_rows':len(weights['Bobot CoAttention Respons S3']),
            'reference_weight_rows':len(weights['Bobot Referensi S3']),'top1_rows':len(top),'answer_windows':len(sums['Bobot Referensi S3']),
            'max_sum_error':max(abs(v-1) for group in sums.values() for v in group.values()),'finite':True,'token_positions_verified':True,
            'top1_scope':'Uploaded top-1 summary; full alignment matrix unavailable, maximum not independently recomputed.'}


def audit(path, experiment='E21'):
    sys.path.insert(0,str(ROOT/'src'))
    from aes.teacher_score.models import metrics
    raw=Path(path).read_bytes()
    with ZipFile(io.BytesIO(raw)) as z:
        if set(z.namelist())!=set(BOOKS):raise ValueError('E21 memerlukan lima workbook asli.')
        blobs={n:z.read(n) for n in BOOKS}
    data=table(blobs[BOOKS[0]],'Data dan Nilai Guru')
    pairs=table(blobs[BOOKS[3]],'Nilai Guru dan Prediksi')
    reported=table(blobs[BOOKS[3]],'Perbandingan Model')
    folds=table(blobs[BOOKS[3]],'Hasil Tiap Fold')
    history={s:table(blobs[BOOKS[4]],name) for s,name in [('S1','RoBERTa (S1)'),('S2','BGE-M3 (S2)'),('S3','Fusion (S3)')]}
    summary=table(blobs[BOOKS[4]],'Ringkasan Lima Fold')
    parameters=table(blobs[BOOKS[4]],'Pengaturan Eksperimen');p={r['Pengaturan']:r['Nilai'] for r in parameters}
    assert len(data)==300 and len(pairs)==60 and len(folds)==15
    assert p['Format input tiga teks']==('context_pair' if experiment=='E22' else 'single_sequence')
    if experiment in ('E24','E25','E26'):
        assert p['Pooling jawaban S3']==('bigru_attention' if experiment=='E26' else 'attention') and p['Hidden pooling S3']==64
        weights=table(blobs[BOOKS[1]],'Bobot Attention S3')
        from collections import defaultdict
        sums=defaultdict(float)
        assert all(np.isfinite(r['Bobot Attention']) and 0<=r['Bobot Attention']<=1 for r in weights)
        for r in weights:sums[(r['ID Jawaban'],r['Bagian Teks'])]+=r['Bobot Attention']
        assert {k[0] for k in sums}=={r['ID Jawaban'] for r in data}
        assert all(abs(v-1)<1e-5 for v in sums.values())
        attention_audit={'rows':len(weights),'answer_windows':len(sums),'max_sum_error':max(abs(v-1) for v in sums.values()),'weights_finite_nonnegative':True}
    if experiment in ('E25','E26'):
        pooled=table(blobs[BOOKS[1]],'Pooling Jawaban S3')
        assert len(pooled)==300 and all(len(r)==769 for r in pooled)
        assert [r['ID Jawaban'] for r in pooled]==[r['ID Jawaban'] for r in data]
        assert all(np.isfinite(v) for r in pooled for k,v in r.items() if k!='ID Jawaban')
        if experiment=='E25':
            assert p['Fusion S3']=='gated_concat' and p['Hidden gate S3']==32
            vectors=[table(blobs[BOOKS[1]],name) for name in ('Gate Fusion S3','Pengali RoBERTa S3','Pengali BGE S3')]
            assert all(len(v)==300 and all(len(r)==2 for r in v) for v in vectors)
            assert all([r['ID Jawaban'] for r in v]==[r['ID Jawaban'] for r in data] for v in vectors)
            arrays=[np.array([[v for k,v in r.items() if k!='ID Jawaban'][0] for r in rows]) for rows in vectors]
            gate,roberta,bge=arrays
            assert all(np.isfinite(a).all() for a in arrays) and np.all((gate>=0)&(gate<=1))
            assert np.allclose(roberta,2*gate,atol=1e-6) and np.allclose(bge,2*(1-gate),atol=1e-6)
            gate_audit={'answers':300,'scalar_per_answer':True,'finite':True,'range':[float(gate.min()),float(gate.max())],
                        'mean':float(gate.mean()),'sd':float(gate.std()),'multipliers_verified':True,'pooling_dimensions':768}
        else:
            assert p['Fusion S3']=='concat' and 'Hidden gate S3' not in p
            vectors=table(blobs[BOOKS[1]],'Ringkasan BiGRU Attention S3')
            assert len(vectors)==300 and all(len(r)==129 for r in vectors)
            assert [r['ID Jawaban'] for r in vectors]==[r['ID Jawaban'] for r in data]
            assert all(np.isfinite(v) for r in vectors for k,v in r.items() if k!='ID Jawaban')
            bigru_attention_audit={'answers':300,'summary_dimensions':128,'pooling_dimensions':768,'finite':True}
    if experiment in ('E23','E28'):
        assert p['Pooling jawaban S3']=='bigru' and p['Hidden pooling S3']==64
        vectors=table(blobs[BOOKS[1]],'BiGRU S3 State Akhir')
        pooled=table(blobs[BOOKS[1]],'Pooling Jawaban S3')
        assert len(vectors)==len(pooled)==300
        assert {r['ID Jawaban'] for r in vectors}=={r['ID Jawaban'] for r in data}
        assert all(len(r)==129 for r in vectors) and all(len(r)==769 for r in pooled)
        assert all(np.isfinite(v) for r in vectors+pooled for k,v in r.items() if k!='ID Jawaban')
        bigru_audit={'answers':300,'final_state_dimensions':128,'pooling_dimensions':768,'finite':True}
    assert p['Learning rate encoder RoBERTa S1']==p['Learning rate encoder RoBERTa S3']==5e-6
    if experiment in ('E27','E28'):
        assert p['Learning rate head kontrol S1 S2']==p['Learning rate proyeksi BGE']==p['LR pooling dan norm S3']==1e-4
        assert p['LR MLP S3 saja']==(5e-6 if experiment=='E28' else 1e-4)
        if experiment=='E27':
            assert p['Pooling jawaban S3']=='bigru_coattention' and p['Hidden pooling S3']==64
            coattention_audit=audit_coattention(blobs,data)
    else:assert p['Learning rate head']==p['Learning rate proyeksi BGE']==1e-4
    assert p['Skor minimum']==0 and p['Skor maksimum']==100
    by_id={r['ID Jawaban']:r for r in data};assert len(by_id)==300
    assert len({r['ID Jawaban'] for r in pairs})==60
    assert all(all(isinstance(r[k],str) and r[k].strip() for k in ('Soal','Jawaban Siswa','jawaban referensi')) for r in data)
    assert all(by_id[r['ID Jawaban']]['Nilai Guru']==r['Nilai Guru'] for r in pairs)
    split=table(blobs[BOOKS[0]],'Pembagian Data')
    test_ids={r['ID Jawaban'] for r in split if r['Penggunaan Data']=='Data uji'}
    assert len(split)==300 and test_ids=={r['ID Jawaban'] for r in pairs}
    assignments=table(blobs[BOOKS[0]],'Pembagian Lima Fold')
    validation_sets=[]
    for fold in range(1,6):
        valid={r['ID Jawaban'] for r in assignments if r['Fold']==fold and r['Penggunaan Data']=='Validasi'}
        train={r['ID Jawaban'] for r in assignments if r['Fold']==fold and r['Penggunaan Data']=='Latih'}
        assert valid and train and not valid&train and not (valid|train)&test_ids
        assert len(valid|train)==240;validation_sets.append(valid)
    assert set.union(*validation_sets)==set(by_id)-test_ids and sum(map(len,validation_sets))==240
    maps={};extra=[];cv=[];selected={};metric_deltas=[]
    wins={s:{'unique':0,'tied':0,'share':0.} for s in ('S1','S2','S3')}
    for r in pairs:
        errors={s:abs(r['Prediksi '+s]-r['Nilai Guru']) for s in wins};minimum=min(errors.values())
        best=[s for s in wins if abs(errors[s]-minimum)<=1e-5]
        for s in best:wins[s]['unique' if len(best)==1 else 'tied']+=1;wins[s]['share']+=1/len(best)
    for scenario in ('S1','S2','S3'):
        y=np.array([r['Nilai Guru'] for r in pairs]);pred=np.array([r['Prediksi '+scenario] for r in pairs])
        actual=metrics(y,pred,{'score_min':0,'score_max':100})
        report=next(r for r in reported if r['Skenario']==scenario)
        for k,label in [('RMSE','RMSE (poin)'),('Pearson','Pearson'),('QWK','QWK'),('AC2','Gwet AC2')]:
            delta=abs(actual[k]-report[label]);metric_deltas.append(delta)
            assert delta<1e-6,(scenario,label)
        matrix=[[0]*11 for _ in range(11)]
        for r in pairs:
            rounded=min(range(101),key=lambda n:abs(n-min(100,max(0,r['Prediksi '+scenario]))))
            assert r['Pembulatan '+scenario]==rounded
            matrix[min(10,int(r['Nilai Guru'])//10)][min(10,rounded//10)]+=1
        maps[scenario]=matrix
        rows=[r for r in folds if r['Skenario']==scenario];assert sorted(r['Fold'] for r in rows)==[1,2,3,4,5]
        for r in summary:
            if r['Skenario']!=scenario:continue
            trace=[h for h in history[scenario] if h['Fold']==r['Fold'] and h['Tahap']=='Validasi silang']
            chosen=next(h for h in trace if h['Epoch']==r['Epoch Terpilih'])
            assert len(trace)==r['Epoch Dijalankan']
            assert chosen['RMSE Validasi (poin)']==min(h['RMSE Validasi (poin)'] for h in trace)
            foldrow=next(h for h in rows if h['Fold']==r['Fold'])
            assert abs(chosen['RMSE Validasi (poin)']-foldrow['RMSE (poin)'])<1e-7
        selected[scenario]=int(np.ceil(np.median([r['Epoch Terpilih'] for r in summary if r['Skenario']==scenario])))
        refit=[r for r in history[scenario] if r['Tahap']=='Latihan final']
        assert len(refit)==selected[scenario]
        cv.append({'Skenario':scenario,'mean':statistics.mean(r['RMSE (poin)'] for r in rows),'sd':statistics.stdev(r['RMSE (poin)'] for r in rows)})
        extra.append({'Skenario':scenario,'MSE (poin²)':float(np.mean((pred-y)**2)),
            'MAE (poin)':float(np.mean(abs(pred-y))),'Bias (prediksi−guru)':float(np.mean(pred-y)),
            'Dalam ±5 poin (%)':float(np.mean(abs(pred-y)<=5)*100),'Dalam ±10 poin (%)':float(np.mean(abs(pred-y)<=10)*100)})
    coverage=table(blobs[BOOKS[0]],'Cakupan Token dan Konteks');assert len(coverage)==600
    coverage_summary=[]
    for role in ('roberta','bge'):
        rows=[r for r in coverage if r['model']==role+' tiga input'];assert len(rows)==300
        assert all(r['Format Input']==p['Format input tiga teks'] and r['Token Jawaban Tidak Tercakup']==0 for r in rows)
        coverage_summary.append({'Encoder':role,'Jawaban':len(rows),'Soal terpotong':sum(r['Soal Terpotong?']=='Ya' for r in rows),
            'Referensi terpotong':sum(r['Referensi Terpotong?']=='Ya' for r in rows),
            'Window total':sum(r['Jumlah Window'] for r in rows),'Token jawaban tak tercakup':0,
            'Kapasitas jawaban minimum':min(r['Kapasitas Jawaban per Window'] for r in rows)})
    public={}
    for i,names in {0:['Panduan'],1:['Panduan'],2:['Panduan'],3:['Panduan','Ringkasan Validasi','Arti Metrik'],4:['Panduan','Model yang Dibandingkan','Pengaturan Eksperimen']}.items():
        for name in names:public[str(i)+'|'+name]=table(blobs[BOOKS[i]],name)
    notebook_path={'E21':NOTEBOOK,'E22':NOTEBOOK.replace('E21_Tiga_Input','E22_Tiga_Input_ContextPair'),'E23':E23_NOTEBOOK,'E24':E24_NOTEBOOK,'E25':E25_NOTEBOOK,'E26':E26_NOTEBOOK,'E27':E27_NOTEBOOK,'E28':E28_NOTEBOOK}[experiment]
    pin=E27_PIN if experiment in ('E27','E28') else E25_PIN if experiment in ('E25','E26') else E24_PIN if experiment in ('E23','E24') else PIN
    snapshot={'id':experiment,'scope':'UPLOADED '+experiment+' RESULTS; NO NEW TRAINING','rows':300,'development':240,'test':60,
        'input_format':p['Format input tiga teks'],'s3_pooling':p.get('Pooling jawaban S3','cls'),'metric_recompute_max_delta':max(metric_deltas),'metric_recompute_tolerance':1e-6,'zip_sha256':hashlib.sha256(raw).hexdigest(),'source_commit':pin,'source_path':notebook_path,
        'runtime_source_matches_commit':False,'source_note':'Source rujukan repo cocok pengaturan workbook; notebook eksekusi tidak disertakan sehingga kecocokan byte runtime/perangkat belum dibuktikan.',
        'metrics':reported,'folds':folds,'history':history,'foldSummary':summary,'cvSummary':cv,'selected_epochs':selected,
        'matrices':maps,'native':{},'publicSheets':public,'parameters':parameters,'coverage':coverage_summary,
        'errorSummary':extra,'nearest':wins,'books':[{'file':n,'sha256':hashlib.sha256(blobs[n]).hexdigest(),'sheets':workbook_inventory(blobs[n])} for n in BOOKS]}
    if experiment in ('E24','E25','E26'):snapshot['attention_audit']=attention_audit
    if experiment in ('E23','E28'):snapshot['bigru_audit']=bigru_audit
    if experiment=='E27':snapshot['coattention_audit']=coattention_audit
    if experiment in ('E25','E26'):snapshot['s3_fusion']=p['Fusion S3']
    if experiment=='E25':snapshot['gate_audit']=gate_audit
    if experiment=='E26':snapshot['bigru_attention_audit']=bigru_attention_audit
    return snapshot


def content(snapshot):
    experiment=snapshot.get('id','E21');notebook_path=snapshot.get('source_path',NOTEBOOK)
    raw=subprocess.check_output(['git','show',snapshot.get('source_commit',PIN)+':'+notebook_path],cwd=ROOT);notebook=json.loads(raw)
    chapters,glossary=curriculum(snapshot);cells=[];missing=[]
    for i,cell in enumerate(notebook['cells']):
        if cell['cell_type']!='code':continue
        text=source(cell);lines=text.splitlines();notes=[]
        for index,anchor,title,body in PINS:
            if index!=i:continue
            if experiment=='E25' and anchor=="features['concat'] =":
                anchor="features['concat'], gate ="
                title='Fusion berbobot gate';body='Gate menghasilkan g per jawaban. Concat aktual menyambung 2g×RoBERTa768 dengan 2(1−g)×BGE256, lalu masuk MLP.'
            if experiment=='E22':
                if 'build_inputs_with_special_tokens(context_ids+second)' in anchor:
                    anchor=anchor.replace('context_ids+second','context_ids, second')
                    title='Pasangan konteks dan jawaban';body='E22 memasukkan konteks sebagai teks A dan penanda/jawaban sebagai teks B. Tokenizer menambahkan separator pasangan; label guru tetap di luar encoder.'
                if title=='Single sequence E21':
                    title='Context pair E22';body='Tiga kolom tetap sama. E22 memakai pasangan (soal + referensi, jawaban), bukan menambah informasi baru di luar E21.'
            line=next((j+1 for j,l in enumerate(lines) if anchor in l),None)
            if line is None:missing.append((i,anchor));continue
            notes.append({'line':line,'title':title,'body':body,'term':None,'kind':'fundamental'})
        if experiment=='E24' and i==30:
            for anchor,title,body in [
                ('def answer_token_mask','Mask posisi jawaban','Memakai answer_positions, bukan kecocokan ID token. Konteks, prefix, special dan padding tidak ikut pooling; konteks tetap tersedia melalui encoder.'),
                ('self.hidden = nn.Linear','Attention 768 → 64','Lapisan kecil belajar menilai tiap token jawaban. Sumbu urutan adalah token, bukan768koordinat.'),
                ('weights = torch.softmax','Softmax bertopeng','Logit non-jawaban dibuat minus tak hingga; bobotnya nol. Bobot token jawaban dalam setiap window berjumlah satu, dihitung FP32.'),
                ('weights.unsqueeze(-1)','Jumlah berbobot768','Kalikan setiap vektor token dengan bobotnya dan jumlahkan pada sumbu token. Keluaran tetap768, bukan satu skor.'),
                ('self.sequence_pool =','Pooling tambahan hanya S3','S1 tetap CLS dan linear, S2 tetap BGE+MLP. S3 mengganti ringkasan CLS dengan pooling token jawaban sebelum LayerNorm/fusion.'),
                ("features['answer_pool768']",'Fitur pooling aktual','Vektor768 yang masuk fusion disimpan untuk inspeksi setelah training; bobot attention terpisah di Excel.'),
            ]:
                line=next(j+1 for j,l in enumerate(lines) if anchor in l)
                notes.append({'line':line,'title':title,'body':body,'term':None,'kind':'fundamental'})
        tree=ast.parse(text.removeprefix('%%aes\n'));offset=int(text.startswith('%%aes\n'))
        defs=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
        for n in defs:
            line=n.lineno+offset
            if any(abs(x['line']-line)<=1 for x in notes):continue
            doc=ast.get_docstring(n)
            body=HINTS.get(n.name) or (doc.split('\n')[0] if doc else 'Definisi '+n.name+' menyiapkan bagian proses; pemanggilan fungsi di cell ini atau berikutnya menjalankannya. Baca bersama alur dan hasil Excel tahap ini.')
            notes.append({'line':line,'title':n.name+'()','body':body,'term':None,'kind':'function'})
        notes.sort(key=lambda n:n['line'])
        for j,n in enumerate(notes):n['id']=f'c{i}-n{j}'
        chapter=next((j for j,c in enumerate(chapters) if i in c['cells'] and j!=0),0)
        cells.append({'index':i,'ordinal':len(cells)+1,'source':text,'htmlLines':highlighted_lines(text),
            'title':defs[0].name if defs else 'Pengaturan & eksekusi','chapter':chapter,'notes':notes,'sha256':hashlib.sha256(text.encode()).hexdigest()})
    if missing:raise ValueError('Anchor E21 tidak ditemukan: '+str(missing))
    mappings,catalog=enrich_learning_notes(cells,experiment)
    return {'chapters':chapters,'glossary':glossary,'cells':cells,'studySnapshot':snapshot,'snapshot':snapshot,
        'screenshotMappings':mappings,'learningCatalog':catalog,'learningLayout':'margin',
        'markdown':[{'index':i,'source':source(c)} for i,c in enumerate(notebook['cells']) if c['cell_type']=='markdown'],
        'studySource':{'commit':snapshot.get('source_commit',PIN),'path':notebook_path,'experiment':experiment,'runtime_match':False},
        'coverage':{'code_cells':len(cells),'code_lines':sum(len(c['htmlLines']) for c in cells),'annotations':sum(len(c['notes']) for c in cells)}}


def reader_script():
    js=(ROOT/'scripts/templates/mlp_study.js').read_text()
    js=js.replace('/*__THREE_INPUT_NOTES__*/',(ROOT/'scripts/templates/three_input_learning.js').read_text())
    js=js.replace('/*__EXCEL_EXPLORER__*/',(ROOT/'scripts/templates/mlp_excel.js').read_text())
    js=js.replace('E6','E21').replace('MLP awal','E21')
    js=js.replace('readWorkbookSheet};','readWorkbookSheet,historyChart};')
    js=js.replace('l.append(s);selects.append(l);return s','s.setAttribute("aria-label",label);l.append(s);selects.append(l);return s')
    js=js.replace('Buku lengkap sudah menyertakan seluruh Excel; tidak perlu unggah ZIP.','Buku lengkap sudah menyertakan seluruh Excel E21; tidak perlu unggah ZIP.')
    js=js.replace('E21 · tabel aktual dari Excel pada tahap ini. Notebook eksekusi E21 tidak disertakan; log E21 tidak ditampilkan sebagai output E21.', 'E21 · hasil tahap ini berasal dari Excel unggahan. Notebook eksekusi/log mentah tidak disertakan; tidak ada output cell yang dikarang.')
    js=js.replace('Bandingkan peta semua eksperimen →','Buka peta S1/S2/S3 E21 →')
    js=js.replace("MLPCompare.setMode('dashboard',true);MLPDashboard.setTab('predictions');MLPPredictions.setView('heatmap')", "E21Book.setMode('results');E21Book.setTab('maps')")
    a=js.index('function windowDemo(){');b=js.index('// Pembaca XLSX',a)
    js=js[:a]+"function windowDemo(){return E21Book.windowDemo()}\n\n"+js[b:]
    a=js.index('function modelDiagram(){');b=js.index('function populateCells',a)
    js=js[:a]+"function modelDiagram(){return E21Book.modelDiagram()}\n"+js[b:]
    js=js.replace('Notebook sumber E21 tidak cocok arsip terverifikasi.','Notebook sumber E21 tidak cocok arsip terverifikasi.')
    a=js.index('(async()=>{try{if(!embeddedPayload)')
    js=js[:a]+'''(async()=>{try{if(!embeddedPayload)return;status('Membuka lima Excel E21…');const bytes=Uint8Array.from(atob(embeddedPayload),c=>c.charCodeAt(0));embeddedPayload=null;await loadResults(bytes);status('E21 siap: lima Excel asli, 60 jawaban test dan seluruh vektor.')}catch(e){error(e)}})();
})();
'''
    return js


def build(results=None, experiment='E21', family=False):
    if results:
        snapshot=audit(results,experiment)
        (ROOT/('docs/snapshots/'+experiment.lower()+'_study_snapshot.json')).write_text(json.dumps(snapshot,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    else:snapshot=json.loads((ROOT/('docs/snapshots/'+experiment.lower()+'_study_snapshot.json')).read_text())
    data=content(snapshot)
    pin=snapshot.get('source_commit',PIN)
    template=(ROOT/'scripts/templates/mlp_study.html').read_text()
    a=template.index('<nav class="workspace-switch"');b=template.index('<section id="reader"',a)
    template=template[:a]+(ROOT/'scripts/templates/e21_results.html').read_text()+template[b:]
    template=template.replace('E6','E21').replace('EKSPERIMEN 6','EKSPERIMEN 21').replace('eksperimen 6','eksperimen 21')
    template=template.replace('Dari satu esai.<br>Jadi <em>satu skor.</em>','Tiga teks masuk.<br>Satu <em>skor keluar.</em>')
    template=template.replace('Bongkar RoBERTa, BGE-M3 dan fusion MLP pelan-pelan. Kode aslinya ada. Hasilnya ada. Catatan pinggirnya bantu lo menyambungkan keduanya.',
        'Soal, kunci jawaban dan respons siswa dibaca oleh seluruh skenario. Pelajari kode E21, telusuri hasil aktual, lalu bandingkan prediksi dengan skor guru.')
    template=re.sub(r'<p class="source-caption">.*?</p>', '<p class="source-caption">Source E21 pada commit '+pin[:7]+'. Hasil dibaca dari lima Excel unggahan; notebook eksekusi tidak disertakan.</p>',template)
    template=template.replace('E21 / LR S3 5e-6','E21 / LR S1/S3 5e-6')
    template=template.replace('Ringkasan semua eksperimen →','Hasil E21 →').replace('__EXPERIMENT_COUNT__ eksperimen','1 eksperimen').replace('hasil tersimpan yang dibandingkan','E21 · tiga skenario')
    template=re.sub(r'https://github.com/MalikHariyanto/Automated_Essay_Scoring/blob/[^" ]+\.ipynb',f'https://github.com/MalikHariyanto/Automated_Essay_Scoring/blob/{pin}/'+snapshot['source_path'],template)
    template=template.replace('<section id="reader" class="study-layout"','<section id="reader" hidden class="study-layout"')
    css='\n'.join((ROOT/'scripts/templates'/p).read_text() for p in ['mlp_study.css','mlp_excel.css','mlp_comparison.css','mlp_metrics.css','mlp_frozen.css','mlp_scroll.css','mlp_compare_fullscreen.css','mlp_dashboard.css','mlp_predictions.css','mlp_score_maps.css','mlp_ui_refinement.css','mlp_e6_refresh.css','e21_study.css','three_input_learning.css'])
    for name,marker in [('paper_splash.svg','__SPLASH__'),('paper_city.svg','__CITY__')]:
        css=css.replace(marker,'data:image/svg+xml;base64,'+base64.b64encode((ROOT/'scripts/assets'/name).read_bytes()).decode())
    script=reader_script()+'\n'+(ROOT/'scripts/templates/e21_study.js').read_text()+'\n'+(ROOT/'scripts/templates/mlp_scroll_boundary.js').read_text()
    if experiment!='E21':
        template=template.replace('E21',experiment).replace('EKSPERIMEN 21','EKSPERIMEN '+experiment[1:]).replace('eksperimen 21','eksperimen '+experiment[1:])
        script=script.replace('E21',experiment)
    if experiment in ('E23','E24','E25','E26','E27','E28'):
        script=script.replace('cls = self.roberta','hidden = self.roberta')
    if experiment=='E24':
        script=script.replace("['LayerNorm RoBERTa 768','self.roberta_norm =']", "['Attention token jawaban','self.sequence_pool ='],['LayerNorm RoBERTa 768','self.roberta_norm =']")
    if experiment=='E23':
        script=script.replace("['LayerNorm RoBERTa 768','self.roberta_norm =']", "['BiGRU token jawaban','self.sequence_pool ='],['LayerNorm RoBERTa 768','self.roberta_norm =']")
    if experiment=='E25':
        script=script.replace("['LayerNorm RoBERTa 768','self.roberta_norm =']", "['Attention token jawaban','self.sequence_pool ='],['LayerNorm RoBERTa 768','self.roberta_norm =']")
        script=script.replace("['Concat 1024',\"features['concat'] =\"]", "['Gate dua cabang','self.fusion_gate ='],['Concat berbobot 1024',\"features['concat'], gate =\"]")
    if experiment=='E26':
        script=script.replace("['LayerNorm RoBERTa 768','self.roberta_norm =']", "['BiGRU + attention token','self.sequence_pool ='],['LayerNorm RoBERTa 768','self.roberta_norm =']")
    if experiment in ('E27','E28'):
        script=script.replace("['LayerNorm RoBERTa 768','self.roberta_norm =']", "['"+('BiGRU + co-attention referensi' if experiment=='E27' else 'BiGRU state akhir E23')+"','self.sequence_pool ='],['LayerNorm RoBERTa 768','self.roberta_norm =']")
    document=(template.replace('__STYLE__',css).replace('__SYMBOLS__',(ROOT/'scripts/assets/paper_spider_symbols.svg').read_text())
        .replace('__DATA__',json.dumps(data,ensure_ascii=False,allow_nan=False).replace('<','\\u003c'))
        .replace('__VENDOR__',(ROOT/'scripts/vendor/mlp_jszip.min.js').read_text()).replace('__SCRIPT__',script))
    if family:
        document=document.replace('<span class="e21-pending">E22 menunggu hasil</span>','<span class="e21-pending">E21 / E22 · hasil tersedia</span>')
    public=ROOT/('docs/belajar_'+experiment+'_Spider.html')
    if not family: public.write_text(document.replace('__PRIVATE_PAYLOAD__',''))
    if results:
        target=ROOT/('outputs/e21_study/belajar_'+experiment+'_Privat.html');target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(document.replace('__PRIVATE_PAYLOAD__',base64.b64encode(Path(results).read_bytes()).decode()))
    print(json.dumps({'public':None if family else str(public),'private':str(target) if results else None,'coverage':data['coverage'],'test_metrics':snapshot['metrics']}))
    return document.replace('__PRIVATE_PAYLOAD__',''),snapshot


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path);p.add_argument('--experiment',choices=['E21','E22','E23','E24','E25','E26','E27','E28'],default='E21');a=p.parse_args();build(a.results,a.experiment)
