"""Ekstrak pembanding awal dari notebook tersimpan; hanya agregat masuk publik.

Tidak menjalankan notebook/model. Source versi pertama dan revisi antara
diperiksa terhadap riwayat Git; prediksi individual tetap pada file privat.
"""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
import io
import zipfile
import math

ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'docs/snapshots/mlp_comparison_snapshot.json'
BASE='3afad79'
MID='312f394'
LATEST='15bc7bb'
NOTEBOOK='notebooks/03_AES_dataset_guru_300.ipynb'


def source(c):
    return c['source'] if isinstance(c['source'],str) else ''.join(c['source'])


def committed(ref):
    return json.loads(subprocess.check_output(['git','show',ref+':'+NOTEBOOK],cwd=ROOT))


class SavedTable(HTMLParser):
    def __init__(self,text):
        super().__init__(convert_charrefs=True)
        self.rows=[];self.row=None;self.value=None
        self.feed(text)

    def handle_starttag(self,tag,attrs):
        if tag=='tr':self.row=[]
        if tag in ('td','th'):self.value=[]

    def handle_data(self,data):
        if self.value is not None:self.value.append(data)

    def handle_endtag(self,tag):
        if tag in ('td','th') and self.row is not None and self.value is not None:
            self.row.append(''.join(self.value));self.value=None
        if tag=='tr' and self.row is not None:
            self.rows.append(self.row);self.row=None

    def records(self):
        headers,*rows=self.rows
        result=[]
        for row in rows:
            values=[]
            for value in row:
                try:value=float(value)
                except ValueError:pass
                values.append(value)
            result.append(dict(zip(headers,values)))
        return result


def history_from_log(text):
    history={s:[] for s in ('S1','S2','S3')};folds={}
    pattern=re.compile(r'(S[123]): validasi silang, fold (\d+) dari 5|'
                       r'(S[123]) · epoch (\d+)/(\d+) · loss latih \(MSE, target 0–1\): ([\d.eE+-]+)'
                       r'(?: \| RMSE validasi: ([\d.eE+-]+) poin| \| pelatihan final; data uji belum dibuka)')
    for m in pattern.finditer(text):
        if m[1]:folds[m[1]]=int(m[2]);continue
        scenario=m[3];cv=m[7] is not None
        history[scenario].append({'Tahap':'Validasi silang' if cv else 'Pelatihan final',
                                 'Fold':folds.get(scenario) if cv else None,'Epoch':int(m[4]),
                                 'Loss Training (MSE)':float(m[6]),
                                 'RMSE Validasi (poin)':float(m[7]) if cv else None})
    return history


def code_evidence(notebook):
    return {label:{'cell':index,'source':source(notebook['cells'][index])}
            for label,index in [('config',12),('model',30),('fit',36)]}


def extract(path):
    raw=Path(path).read_bytes();original=json.loads(raw);baseline=committed(BASE)
    if len(original['cells'])!=len(baseline['cells']) or any(
        a['cell_type']!=b['cell_type'] or source(a)!=source(b)
        for a,b in zip(original['cells'],baseline['cells']) if b['cell_type']=='code'
    ):raise ValueError('Source notebook awal tidak cocok dengan 3afad79.')
    outputs=original['cells'][39]['outputs']
    def parse(i):
        text=outputs[i]['data']['text/html']
        return SavedTable(''.join(text) if isinstance(text,list) else text).records()
    folds=parse(1);metrics=parse(2)
    if len(folds)!=15 or len(metrics)!=3:raise ValueError('Tabel CV/test awal tidak lengkap.')
    history=history_from_log(''.join(outputs[0]['text']))
    summary=[]
    for scenario in ('S1','S2','S3'):
        rows=history[scenario]
        for fold in range(1,6):
            local=[r for r in rows if r['Fold']==fold]
            if [r['Epoch'] for r in local]!=list(range(1,6)):raise ValueError('Riwayat CV awal tidak lengkap.')
            best=min(local,key=lambda r:r['RMSE Validasi (poin)'])
            reported=next(r for r in folds if r['Skenario']==scenario and r['Fold']==fold)
            if abs(best['RMSE Validasi (poin)']-reported['RMSE (poin)'])>0.0001:
                raise ValueError('Minimum riwayat tidak cocok tabel awal.')
            summary.append({'Skenario':scenario,'Fold':fold,'Epoch Terpilih':best['Epoch'],
                            'Epoch Dijalankan':len(local),'Loss Training Terpilih':best['Loss Training (MSE)']})
    baseline_commit=subprocess.check_output(['git','rev-parse',BASE],cwd=ROOT,text=True).strip()
    mid_commit=subprocess.check_output(['git','rev-parse',MID],cwd=ROOT,text=True).strip()
    return {'scope':'SAVED BASELINE AND SOURCE HISTORY; NO NEW TRAINING',
            'baseline':{'id':'E0','commit':baseline_commit,'notebook_sha256':hashlib.sha256(raw).hexdigest(),
                        'source_matches_commit':True,'metrics':metrics,'folds':folds,'history':history,
                        'foldSummary':summary,'evidence':code_evidence(baseline),
                        'precision':'Metrik awal disalin dari tampilan notebook (sekitar 8 digit); log loss 6 desimal dan RMSE 4 desimal.'},
            'intermediate':{'id':'E1','commit':mid_commit,'evidence':code_evidence(committed(MID)),
                            'metrics':None,'history':None,
                            'status':'Source revisi terverifikasi; tidak ada hasil akhir lengkap yang dipasangkan secara pasti dengan versi ini.'},
            'comparison_limits':[
                'Perubahan awal ke MLP mencakup learning rate, batas epoch, early stopping dan LayerNorm tambahan, selain head S3. Ini perkembangan eksperimen, bukan ablation yang membuktikan MLP sebagai penyebab tunggal.',
                'Nama dataset dan resep split sama; manifest keanggotaan test/fold versi awal belum tersedia untuk dicocokkan per jawaban. Nomor fold yang sama belum membuktikan anggotanya identik.',
                'Test versi awal sudah dilihat sebelum revisi. Perbandingan test lanjutan bersifat eksploratif; konfirmasi independen memerlukan holdout baru atau evaluasi tersarang.',
                'Angka yang tidak tersedia ditulis belum tersedia, tidak diganti nol. Prediksi satu jawaban hanya dibandingkan berpasangan jika identitas teks/keanggotaannya benar-benar cocok.'
            ]}


def enrich_from_excel(data, baseline_zip, mlp_zip):
    from openpyxl import load_workbook
    archives=[zipfile.ZipFile(p) for p in [baseline_zip,mlp_zip]]
    def rows(archive,file,sheet):
        w=load_workbook(io.BytesIO(archive.read(file)),read_only=True,data_only=True)
        result=list(w[sheet].values);w.close();return result
    def records(values):
        return [dict(zip(values[0],r)) for r in values[1:]]
    old,new=archives;first='01_Data_dan_Tokenisasi.xlsx';evaluation='04_Prediksi_dan_Evaluasi.xlsx';training='05_Riwayat_Pelatihan.xlsx'
    for sheet in ['Data dan Nilai Guru','Pembagian Data','Pembagian Lima Fold']:
        if rows(old,first,sheet)!=rows(new,first,sheet):raise ValueError('Manifest berbeda: '+sheet)
    teacher=records(rows(old,first,'Data dan Nilai Guru'))
    partition=records(rows(old,first,'Pembagian Data'))
    train_ids={r['ID Jawaban'] for r in partition if r['Penggunaan Data'].startswith('Data latih')}
    if len(teacher)!=300 or len(train_ids)!=240:raise ValueError('Ukuran dataset/development bukan 300/240.')
    mean=sum(r['Nilai Guru'] for r in teacher if r['ID Jawaban'] in train_ids)/len(train_ids)
    pairs=records(rows(old,evaluation,'Nilai Guru dan Prediksi'));latest=records(rows(new,evaluation,'Nilai Guru dan Prediksi'))
    left={r['ID Jawaban']:r for r in pairs};right={r['ID Jawaban']:r for r in latest}
    if len(left)!=60 or set(left)!=set(right) or any(left[k]['Nilai Guru']!=right[k]['Nilai Guru'] for k in left):raise ValueError('Pasangan test tidak cocok.')
    metrics=records(rows(old,evaluation,'Perbandingan Model'));folds=records(rows(old,evaluation,'Hasil Tiap Fold'))
    for original,values in [(data['baseline']['metrics'],metrics),(data['baseline']['folds'],folds)]:
        for a,b in zip(original,values):
            for key in ['RMSE (poin)','Pearson','QWK','Gwet AC2']:
                if abs(a[key]-b[key])>1e-5:raise ValueError('Excel tidak cocok dengan notebook awal.')
    for r in metrics:
        s=r['Skenario'];actual=math.sqrt(sum((p['Prediksi '+s]-p['Nilai Guru'])**2 for p in pairs)/60)
        if abs(actual-r['RMSE (poin)'])>1e-5:raise ValueError('RMSE tidak cocok pasangan test.')
    baseline=data['baseline'];baseline.update(metrics=metrics,folds=folds)
    baseline['history']={s:records(rows(old,training,sheet)) for s,sheet in [('S1','RoBERTa (S1)'),('S2','BGE-M3 (S2)'),('S3','Fusion (S3)')]}
    settings=dict(rows(old,training,'Pengaturan Eksperimen')[1:])
    if settings.get('Maksimum epoch per fold')!=5:raise ValueError('ZIP ini bukan eksperimen awal lima epoch.')
    baseline['zip_sha256']=hashlib.sha256(Path(baseline_zip).read_bytes()).hexdigest()
    baseline['eval_xlsx_sha256']=hashlib.sha256(old.read(evaluation)).hexdigest()
    baseline['precision']='Metrik dan riwayat memakai nilai numerik penuh dari Excel awal; tabel cocok dengan notebook tersimpan.'
    same_example=rows(old,evaluation,'Jawaban Contoh')==rows(new,evaluation,'Jawaban Contoh')
    data['pairing']={'dataset_rows':300,'development_rows':len(train_ids),'test_rows':60,'dataset_equal':True,'partition_equal':True,'fold_membership_equal':True,'teacher_values_equal':True,'same_example_text':same_example}
    data['naive_baseline']={'method':'Selalu memprediksi rata-rata nilai guru development; tidak belajar dari nilai guru test.','rmse':math.sqrt(sum((p['Nilai Guru']-mean)**2 for p in pairs)/60)}
    data['comparison_limits'][1]='Dataset, 240/60 train-test, anggota kelima fold, dan nilai guru telah dicocokkan persis pada kedua Excel. Perbandingan 60 prediksi dipasangkan berdasarkan ID jawaban, bukan posisi baris.'
    data['comparison_limits'][3]='Contoh teks pada kedua workbook cocok. Hasil individual dibaca lokal; data siswa tidak disertakan dalam HTML publik.'
    target=ROOT/'outputs/mlp_study/baseline_eval.xlsx';target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(old.read(evaluation))
    return data


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--baseline',required=True);parser.add_argument('--baseline-results',required=True);parser.add_argument('--mlp-results',required=True)
    args=parser.parse_args();data=extract(args.baseline)
    if bool(args.baseline_results)!=bool(args.mlp_results):raise ValueError('Sediakan kedua ZIP agar manifest bisa dicocokkan.')
    if args.baseline_results:data=enrich_from_excel(data,args.baseline_results,args.mlp_results)
    TARGET.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'target':str(TARGET),'CV':len(data['baseline']['folds']),
                      'history':sum(map(len,data['baseline']['history'].values())),'metrics':len(data['baseline']['metrics'])}))
