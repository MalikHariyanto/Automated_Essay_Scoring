"""Audit E5 LoRA; publikasikan agregat tanpa jawaban/ID/nilai individual."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
from zipfile import ZipFile

from build_mlp_study import ROOT, BOOKS, table, workbook_inventory
from build_mlp_frozen import identity, validate_predictions


def extract(path, mlp_path):
    import numpy as np
    sys.path.insert(0,str(ROOT/'src'))
    from aes.teacher_score.models import metrics as evaluate
    raw=Path(path).read_bytes()
    reference=json.loads((ROOT/'docs/mlp_study_snapshot.json').read_text())
    if hashlib.sha256(Path(mlp_path).read_bytes()).hexdigest()!=reference['zip_sha256']:
        raise ValueError('ZIP acuan berbeda dari buku MLP.')
    with ZipFile(path) as z,ZipFile(mlp_path) as old:
        if set(z.namelist())!=set(BOOKS):raise ValueError('Lima workbook diperlukan.')
        blobs={b:z.read(b) for b in BOOKS}
        pairing={}
        for sheet,keys in [('Data dan Nilai Guru',['ID Jawaban']),('Pembagian Data',['ID Jawaban']),
                           ('Pembagian Lima Fold',['Fold','ID Jawaban','Penggunaan Data'])]:
            rows=table(blobs[BOOKS[0]],sheet)
            if identity(rows,keys)!=identity(table(old.read(BOOKS[0]),sheet),keys):
                raise ValueError('Data/split berbeda: '+sheet)
            pairing[sheet]={'identical':True,'rows':len(rows)}
        pairs=table(blobs[BOOKS[3]],'Nilai Guru dan Prediksi')
        previous=table(old.read(BOOKS[3]),'Nilai Guru dan Prediksi')
        a,b=identity(pairs,['ID Jawaban']),identity(previous,['ID Jawaban'])
        if set(a)!=set(b) or any(a[k]['Nilai Guru']!=b[k]['Nilai Guru'] for k in a):
            raise ValueError('ID/nilai guru test berbeda.')
        models=table(blobs[BOOKS[4]],'Model yang Dibandingkan')
        if 'LoRA' not in next(r for r in models if r['Skenario']=='S3')['Encoder']:
            raise ValueError('Metode S3 bukan LoRA.')
        history={s:table(blobs[BOOKS[4]],name) for s,name in [('S1','RoBERTa (S1)'),('S2','BGE-M3 (S2)'),('S3','Fusion (S3)')]}
        selected=table(blobs[BOOKS[4]],'Ringkasan Lima Fold')
        if len(selected)!=15:raise ValueError('Ringkasan harus 15 fold.')
        test=[];folds=[];cv=[];impact=[]
        for s in ('S1','S2','S3'):
            result=evaluate(np.array([r['Nilai Guru'] for r in pairs]),np.array([r['Prediksi '+s] for r in pairs]),{'score_min':0,'score_max':100})
            test.append({'Skenario':s,**{name:result[key] for key,name in [('RMSE','RMSE (poin)'),('Pearson','Pearson'),('QWK','QWK'),('AC2','Gwet AC2')]}})
            rows=[r for r in selected if r['Skenario']==s]
            if sorted(r['Fold'] for r in rows)!=[1,2,3,4,5]:raise ValueError('Fold hilang/duplikat.')
            for row in rows:
                trace=[h for h in history[s] if h.get('Fold')==row['Fold']]
                if len(trace)!=row['Epoch Dijalankan']:raise ValueError('Riwayat epoch tidak lengkap.')
                chosen=next(h for h in trace if h['Epoch']==row['Epoch Terpilih'])
                if chosen['RMSE Validasi (poin)']!=min(h['RMSE Validasi (poin)'] for h in trace):
                    raise ValueError('Epoch terpilih bukan RMSE terbaik.')
                folds.append({'Skenario':s,'Fold':row['Fold'],**{name:chosen[key] for key,name in [('RMSE Validasi (poin)','RMSE (poin)'),('Pearson Validasi','Pearson'),('QWK Validasi','QWK'),('Gwet AC2 Validasi','Gwet AC2')]}})
            xs=[r['RMSE (poin)'] for r in folds if r['Skenario']==s]
            cv.append({'Skenario':s,'mean':statistics.mean(xs),'sd':statistics.stdev(xs),'n':5})
            changes=[(abs(a[k]['Prediksi '+s]-a[k]['Nilai Guru']),abs(b[k]['Prediksi '+s]-b[k]['Nilai Guru'])) for k in a]
            impact.append({'Skenario':s,'new_lower_error':sum(x<y-1e-5 for x,y in changes),'old_lower_error':sum(y<x-1e-5 for x,y in changes),'equal_error':sum(abs(x-y)<=1e-5 for x,y in changes),'numeric_tolerance':1e-5})
        validate_predictions(test,pairs)
        return {'id':'E5','label':'LoRA + MLP','scope':'SAVED E5 WORKBOOKS; NO NEW TRAINING',
            'zip_sha256':hashlib.sha256(raw).hexdigest(),'rows':300,'development':240,'test':60,
            'metrics':test,'folds':folds,'cvSummary':cv,'history':history,'foldSummary':selected,
            'models':models,'parameters':table(blobs[BOOKS[4]],'Pengaturan Eksperimen'),
            'loraParameters':table(blobs[BOOKS[4]],'Parameter LoRA dan Fusion'),
            'control_prediction_delta':{s:max(abs(a[k]['Prediksi '+s]-b[k]['Prediksi '+s]) for k in a) for s in ('S1','S2')},
            'pairing':pairing,'teacher_test_pairs_identical':True,'predictionImpact':impact,
            'metrics_origin':'Dihitung ulang dari 60 pasangan guru/prediksi mentah Excel; QWK/AC2 memakai kategori 0–100, tie .5 ke bawah. ZIP tidak memiliki sheet Perbandingan Model/Hasil Tiap Fold.',
            'folds_origin':'Empat metrik dari epoch terpilih pada riwayat validasi; bukan dihitung ulang dari prediksi fold karena pasangan fold tidak disertakan.',
            'source_reference':'Workbook menyatakan LoRA query/value. Notebook eksekusi E5 tidak disertakan; kecocokan source byte demi byte tidak diklaim.',
            'books':[{'file':name,'sha256':hashlib.sha256(blobs[name]).hexdigest(),'sheets':workbook_inventory(blobs[name])} for name in BOOKS],
            'limits':['Satu run per metode, belum ada pengulangan seed atau uji signifikansi.',
                'S1/S2 memiliki prediksi test praktis sama dengan MLP acuan (selisih maksimum di bawah 0,000004 poin). Notebook eksekusi E5 tidak disertakan; asal proses training tidak disimpulkan hanya dari kesamaan hasil.',
                'Data uji sudah dilihat dalam pengembangan; pemilihan metode melalui CV, konfirmasi independen memerlukan data baru.']}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--mlp',required=True);args=p.parse_args()
    result=extract(args.results,args.mlp)
    (ROOT/'docs/mlp_lora_snapshot.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'metrics':result['metrics'],'cv':result['cvSummary'],'matched':result['pairing']},ensure_ascii=False))
