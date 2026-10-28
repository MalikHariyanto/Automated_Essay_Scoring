"""Audit ZIP double-frozen terhadap MLP; snapshot publik hanya agregat aman."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import zipfile

from build_mlp_study import BOOKS, ROOT, table, workbook_inventory, source

TARGET = ROOT/'docs/snapshots/mlp_frozen_snapshot.json'
REFERENCE = '0c79d655898f967e855922be3c5fad1865efb1aa'
NOTEBOOK = 'notebooks/experiments/AES_300_S3_DoubleFrozen_MLP_Manual.ipynb'


def validate_predictions(metrics, pairs):
    import numpy as np
    from scipy.stats import pearsonr
    from sklearn.metrics import cohen_kappa_score
    sys.path.insert(0, str(ROOT/'src'))
    from aes.metrics import gwet_ac2
    y=np.asarray([r['Nilai Guru'] for r in pairs], dtype=float)
    if len(y)!=60 or not np.isfinite(y).all():
        raise ValueError('Perlu 60 pasangan test finite.')
    categories=np.arange(101)
    largest=0.0
    for s in ('S1','S2','S3'):
        p=np.asarray([r['Prediksi '+s] for r in pairs], dtype=float)
        if not np.isfinite(p).all() or not np.isin(y,categories).all():
            raise ValueError('Nilai test tidak sah.')
        ordinal=categories[np.abs(np.clip(p,0,100)[:,None]-categories).argmin(axis=1)]
        if not np.array_equal(ordinal,[r['Pembulatan '+s] for r in pairs]):
            raise ValueError('Pembulatan tidak cocok.')
        actual={'RMSE (poin)':float(np.sqrt(np.mean((p-y)**2))),
                'Pearson':float(pearsonr(y,p).statistic),
                'QWK':float(cohen_kappa_score(y,ordinal,labels=categories,weights='quadratic')),
                'Gwet AC2':gwet_ac2(y,ordinal,categories)}
        row=next(r for r in metrics if r['Skenario']==s)
        for k,v in actual.items():
            largest=max(largest,abs(v-row[k]))
            if not math.isclose(v,row[k],rel_tol=0,abs_tol=1e-6):
                raise ValueError('Metrik tidak cocok dengan pasangan: '+s+'/'+k)
    return {'max_absolute_difference':largest,'absolute_tolerance':1e-6,'category_metrics_match':True}


def identity(rows, keys):
    values={tuple(r[k] for k in keys):r for r in rows}
    if len(values)!=len(rows):raise ValueError('Identitas baris duplikat.')
    return values


def extract(frozen_path, mlp_path):
    raw=Path(frozen_path).read_bytes()
    expected=json.loads((ROOT/'docs/snapshots/mlp_study_snapshot.json').read_text())['zip_sha256']
    if hashlib.sha256(Path(mlp_path).read_bytes()).hexdigest()!=expected:
        raise ValueError('ZIP MLP acuan berbeda dari sumber buku.')
    with zipfile.ZipFile(io.BytesIO(raw)) as z, zipfile.ZipFile(mlp_path) as old:
        if set(z.namelist())!=set(BOOKS):raise ValueError('Lima workbook harus lengkap.')
        blobs={name:z.read(name) for name in BOOKS}
        metrics=table(blobs[BOOKS[3]],'Perbandingan Model')
        pairs=table(blobs[BOOKS[3]],'Nilai Guru dan Prediksi')
        previous=table(old.read(BOOKS[3]),'Nilai Guru dan Prediksi')
        recheck=validate_predictions(metrics,pairs)
        paired={}
        for sheet,keys in [('Data dan Nilai Guru',['ID Jawaban']),('Pembagian Data',['ID Jawaban']),
                           ('Pembagian Lima Fold',['Fold','ID Jawaban','Penggunaan Data'])]:
            a=table(blobs[BOOKS[0]],sheet);b=table(old.read(BOOKS[0]),sheet)
            if identity(a,keys)!=identity(b,keys):raise ValueError('Data/split berbeda: '+sheet)
            paired[sheet]={'identical':True,'rows':len(a)}
        a=identity(pairs,['ID Jawaban']);b=identity(previous,['ID Jawaban'])
        if set(a)!=set(b) or any(a[k]['Nilai Guru']!=b[k]['Nilai Guru'] for k in a):
            raise ValueError('ID atau nilai guru test berbeda.')
        models=table(blobs[BOOKS[4]],'Model yang Dibandingkan')
        s3=next(r for r in models if r['Skenario']=='S3')
        if 'frozen' not in s3['Encoder'] or 'MLP' not in s3['Lapisan Prediksi']:
            raise ValueError('Workbook bukan metode double frozen + MLP.')
        folds=table(blobs[BOOKS[3]],'Hasil Tiap Fold')
        summaries=table(blobs[BOOKS[4]],'Ringkasan Lima Fold')
        history={s:table(blobs[BOOKS[4]],name) for s,name in [('S1','RoBERTa (S1)'),('S2','BGE-M3 (S2)'),('S3','Fusion (S3)')]}
        if len(folds)!=15 or len(summaries)!=15:raise ValueError('Lima fold tidak lengkap.')
        cv=[];impact=[]
        for s in ('S1','S2','S3'):
            entries=[r for r in folds if r['Skenario']==s]
            if sorted(r['Fold'] for r in entries)!=[1,2,3,4,5]:raise ValueError('Fold duplikat/hilang.')
            xs=[r['RMSE (poin)'] for r in entries]
            for r in entries:
                selected=next(v for v in summaries if v['Skenario']==s and v['Fold']==r['Fold'])
                observed=[v for v in history[s] if v.get('Fold')==r['Fold'] and v.get('RMSE Validasi (poin)') is not None]
                best=next(v for v in observed if v['Epoch']==selected['Epoch Terpilih'])
                if not math.isclose(best['RMSE Validasi (poin)'],r['RMSE (poin)'],abs_tol=1e-8):raise ValueError('Epoch terpilih berbeda.')
            cv.append({'Skenario':s,'mean':statistics.mean(xs),'sd':statistics.stdev(xs),'n':5})
            errors=[(abs(a[k]['Prediksi '+s]-a[k]['Nilai Guru']),abs(b[k]['Prediksi '+s]-b[k]['Nilai Guru'])) for k in a]
            impact.append({'Skenario':s,'new_lower_error':sum(x<y for x,y in errors),'old_lower_error':sum(y<x for x,y in errors),'equal_error':sum(x==y for x,y in errors)})
        before=table(blobs[BOOKS[1]],'RoBERTa Sebelum Latihan')
        after=table(blobs[BOOKS[1]],'RoBERTa S3 Setelah Latihan')
        before=identity(before,['ID Jawaban']);after=identity(after,['ID Jawaban'])
        if set(before)!=set(after):raise ValueError('Vektor jawaban berbeda identitas.')
        max_delta=max(abs(before[k][col]-after[k][col]) for k in before for col in before[k] if col.startswith('Komponen '))
        # Tidak mengekspor teks, ID, pasangan prediksi, atau komponen vektor individual.
        nb=json.loads(subprocess.check_output(['git','show',REFERENCE+':'+NOTEBOOK],cwd=ROOT))
        evidence={}
        for label,anchor in [('model','class SingleScoreModel'),('config','ROBERTA_CHECKPOINT='),('fit','def fit(')]:
            c=next((i,c) for i,c in enumerate(nb['cells']) if anchor in source(c))
            evidence[label]={'cell':c[0],'source':source(c[1])}
        return {'id':'E3','scope':'SAVED DOUBLE FROZEN MLP RESULTS; NO NEW TRAINING',
                'label':'Double frozen + MLP','zip_sha256':hashlib.sha256(raw).hexdigest(),
                'rows':300,'development':240,'test':60,'metrics_recheck':recheck,'metrics':metrics,'folds':folds,
                'history':history,'foldSummary':summaries,'cvSummary':cv,
                'models':models,'parameters':table(blobs[BOOKS[4]],'Pengaturan Eksperimen'),
                'pairing':paired,'teacher_test_pairs_identical':True,'predictionImpact':impact,
                'vector_check':{'rows':300,'dimensions':768,'max_abs_difference_S3_vs_pretrained':max_delta,'allclose_1e_6':max_delta<=1e-6},
                'source_reference':{'commit':REFERENCE,'path':NOTEBOOK,'evidence':evidence,
                  'status':'Kode rujukan versi manual di repo; workbook menyatakan metode frozen. Notebook eksekusi run ini belum diunggah, sehingga source runtime tidak diklaim cocok byte demi byte.'},
                'books':[{'file':name,'sha256':hashlib.sha256(blobs[name]).hexdigest(),'sheets':workbook_inventory(blobs[name])} for name in BOOKS],
                'limits':['Hasil satu run per versi; belum ada pengulangan seed atau uji signifikansi.',
                  'S1/S2 juga bergeser meskipun arsitekturnya tetap; selisih antar-run tidak membuktikan efek pembekuan saja.',
                  'Test telah dilihat pada eksperimen terdahulu. Pengembangan berikutnya bersifat eksploratif; pemilihan parameter tetap melalui CV.']}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--frozen',required=True);p.add_argument('--mlp',required=True);a=p.parse_args()
    data=extract(a.frozen,a.mlp);TARGET.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'target':str(TARGET),'metrics':data['metrics'],'cv':data['cvSummary'],'vectors':data['vector_check']},ensure_ascii=False))
