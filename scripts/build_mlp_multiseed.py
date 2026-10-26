"""Audit E6 2.0: tiga seed, agregat publik; semua nilai individual tetap privat."""
import argparse
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path
from zipfile import ZipFile,ZIP_STORED
import numpy as np
from build_mlp_study import ROOT,BOOKS,table,workbook_inventory


def identity(value):
    m=re.fullmatch(r'row_(\d+)',str(value))
    return f'J{int(m[1]):03d}' if m else str(value)


def build(manifest):
    sys.path.insert(0,str(ROOT/'src'))
    from aes.teacher_score.models import metrics
    assert set(manifest['seeds'])=={'42','123','2026'}
    raw=Path(manifest['summary']).read_bytes()
    sheets={s:table(raw,s) for s in ['Rerata CV 3 Seed','Rerata Uji 3 Seed','CV per Seed','Uji per Seed','Seluruh Fold','Guru dan Prediksi','Audit Model dan Seed']}
    pairs=sheets.pop('Guru dan Prediksi')
    old_sources=json.loads((ROOT/'outputs/mlp_study/e8_sources.json').read_text())
    with ZipFile(old_sources['E6']) as z:old={r['ID Jawaban']:r for r in table(z.read(BOOKS[3]),'Nilai Guru dan Prediksi')}
    assert len(pairs)==60 and {identity(r['ID Jawaban']) for r in pairs}==set(old)
    assert all(old[identity(r['ID Jawaban'])]['Nilai Guru']==r['Nilai Guru'] for r in pairs)
    assert len(sheets['Seluruh Fold'])==45 and len(sheets['Uji per Seed'])==9
    for seed in (42,123,2026):
        for scenario in ('S1','S2','S3'):
            actual=metrics(np.array([r['Nilai Guru'] for r in pairs]),np.array([r[f'{scenario} · seed {seed}'] for r in pairs]),{'score_min':0,'score_max':100})
            reported=next(r for r in sheets['Uji per Seed'] if r['Seed']==seed and r['Skenario']==scenario)
            fold=[r for r in sheets['Seluruh Fold'] if r['Seed']==seed and r['Skenario']==scenario]
            assert sorted(r['Fold'] for r in fold)==[1,2,3,4,5]
            cv=next(r for r in sheets['CV per Seed'] if r['Seed']==seed and r['Skenario']==scenario)
            for key,label in [('RMSE','RMSE (poin)'),('Pearson','Pearson'),('QWK','QWK'),('AC2','Gwet AC2')]:
                assert abs(actual[key]-reported[label])<1e-5,(seed,scenario,label)
                assert abs(statistics.mean(r[label] for r in fold)-cv[label])<1e-10
    for mode,source in [('Rerata CV 3 Seed','CV per Seed'),('Rerata Uji 3 Seed','Uji per Seed')]:
        for row in sheets[mode]:
            for metric in ('RMSE (poin)','Pearson','QWK','Gwet AC2'):
                values=[r[metric] for r in sheets[source] if r['Skenario']==row['Skenario']]
                assert abs(statistics.mean(values)-row[metric+' rata-rata'])<1e-10
                assert abs(statistics.stdev(values)-row['SD '+metric])<1e-10
    destination=ROOT/'outputs/mlp_study/e62_sources';destination.mkdir(exist_ok=True)
    inventories={};history={}
    for seed,files in manifest['seeds'].items():
        evaluation=table(Path(files[BOOKS[3]]).read_bytes(),'Nilai Guru dan Prediksi')
        assert len(evaluation)==60 and len({identity(r['ID Jawaban']) for r in evaluation})==60
        index={identity(r['ID Jawaban']):r for r in pairs}
        for row in evaluation:
            other=index[identity(row['ID Jawaban'])]
            assert row['Nilai Guru']==other['Nilai Guru']
            for s in ('S1','S2','S3'):assert abs(row['Prediksi '+s]-other[f'{s} · seed {seed}'])<1e-5
        with ZipFile(old_sources['E6']) as z:original=table(z.read(BOOKS[0]),'Data dan Nilai Guru')
        assert table(Path(files[BOOKS[0]]).read_bytes(),'Data dan Nilai Guru')==original
        with ZipFile(old_sources['E6']) as z:
            for name in ('Pembagian Data','Pembagian Lima Fold'):
                assert table(Path(files[BOOKS[0]]).read_bytes(),name)==table(z.read(BOOKS[0]),name)
        history[seed]={s:table(Path(files[BOOKS[4]]).read_bytes(),name) for s,name in [('S1','RoBERTa (S1)'),('S2','BGE-M3 (S2)'),('S3','Fusion (S3)')]}
        uploaded_folds=table(Path(files[BOOKS[3]]).read_bytes(),'Hasil Tiap Fold')
        assert len(uploaded_folds)==15 and len({(r['Skenario'],r['Fold']) for r in uploaded_folds})==15
        for row in uploaded_folds:
            other=next(r for r in sheets['Seluruh Fold'] if r['Seed']==int(seed) and r['Skenario']==row['Skenario'] and r['Fold']==row['Fold'])
            for key in ('RMSE (poin)','Pearson','QWK','Gwet AC2'):assert abs(row[key]-other[key])<1e-7
        for row in table(Path(files[BOOKS[4]]).read_bytes(),'Ringkasan Lima Fold'):
            trace=[r for r in history[seed][row['Skenario']] if r['Fold']==row['Fold']]
            chosen=next(r for r in trace if r['Epoch']==row['Epoch Terpilih'])
            assert len(trace)==row['Epoch Dijalankan'] and chosen['RMSE Validasi (poin)']==min(r['RMSE Validasi (poin)'] for r in trace)

        archive=destination/f'seed_{seed}.zip'
        with ZipFile(archive,'w',ZIP_STORED) as z:
            for name in BOOKS:z.write(files[name],name)
        inventories[seed]={'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'books':[{'file':name,'sha256':hashlib.sha256(Path(files[name]).read_bytes()).hexdigest(),'sheets':workbook_inventory(Path(files[name]).read_bytes())} for name in BOOKS]}
    rounded={identity(r['ID Jawaban']):r for r in table(raw,'Skor Dibulatkan')}
    maps={}
    for seed in (42,123,2026):
        maps[str(seed)]={}
        for scenario in ('S1','S2','S3'):
            matrix=[[0]*11 for _ in range(11)]
            for row in pairs:
                key=f'{scenario} · seed {seed}'
                value=rounded[identity(row['ID Jawaban'])][key]
                expected=min(range(101),key=lambda n:abs(n-min(100,max(0,row[key]))))
                assert value==expected
                matrix[min(10,int(row['Nilai Guru'])//10)][min(10,int(value)//10)]+=1
            maps[str(seed)][scenario]=matrix
    result={'id':'E62','label':'E6 2.0','seeds':[42,123,2026],'summary_sha256':hashlib.sha256(raw).hexdigest(),
            'maps':maps,'sheets':sheets,'history':history,'archives':inventories,'same_test_ids_labels':True,
            'notebook':'https://github.com/MalikHariyanto/Automated_Essay_Scoring/blob/d8c21accc6e0bbff7a9d95c02a553b8bfeed8340/notebooks/experiments/AES_300_E6_3Seed_CPU.ipynb',
            'input_policy':'Jawaban saja; semua S1/S2/S3 dilatih ulang per seed. Soal/kunci ditampilkan sebagai rujukan.',
            'limits':'Rerata dan SD dari tiga metrik per seed, bukan ensemble. Jangan memilih seed terbaik dari test. Source runtime byte demi byte belum dibuktikan dari workbook saja.'}
    (ROOT/'docs/mlp_multiseed_snapshot.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'seeds':result['seeds'],'folds':45,'models':9,'test_answers':60,'predictions':540,'all_uploaded_workbooks_matched':True}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);a=p.parse_args();build(json.loads(a.manifest.read_text()))
