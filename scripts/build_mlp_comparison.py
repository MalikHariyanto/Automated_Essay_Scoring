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

ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'docs/mlp_comparison_snapshot.json'
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


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--baseline',required=True)
    args=parser.parse_args();data=extract(args.baseline)
    TARGET.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'target':str(TARGET),'CV':len(data['baseline']['folds']),
                      'history':sum(map(len,data['baseline']['history'].values())),'metrics':len(data['baseline']['metrics'])}))
