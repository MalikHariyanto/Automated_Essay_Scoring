"""Peringkat lintas eksperimen dari prediksi mentah pada jawaban uji yang sama."""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
from zipfile import ZipFile
from build_mlp_study import ROOT, table


def compare_predictions(predictions, expected_rows=60, tolerance=1e-5):
    if not predictions:raise ValueError('Eksperimen kosong.')
    indexed={}
    for experiment,rows in predictions.items():
        index={str(row['ID Jawaban']):row for row in rows}
        if len(rows)!=expected_rows or len(index)!=expected_rows:raise ValueError('Jumlah/ID jawaban harus lengkap dan unik.')
        indexed[experiment]=index
    reference=next(iter(indexed.values()))
    for index in indexed.values():
        if set(index)!=set(reference):raise ValueError('ID jawaban berbeda.')
        for key,row in index.items():
            if row['Nilai Guru']!=reference[key]['Nilai Guru']:raise ValueError('Nilai guru berbeda.')
            if not all(math.isfinite(row[k]) for k in ('Nilai Guru','Prediksi S1','Prediksi S2','Prediksi S3')):raise ValueError('Nilai prediksi tidak valid.')
    result={}
    for scenario in ('S1','S2','S3'):
        totals={run:{'id':run,'unique':0,'tied':0,'share':Fraction(0)} for run in indexed}
        tie_answers=0
        for key,row in reference.items():
            errors={run:abs(index[key]['Prediksi '+scenario]-row['Nilai Guru']) for run,index in indexed.items()}
            best=min(errors.values());winners=[run for run,error in errors.items() if abs(error-best)<=tolerance]
            tie_answers+=len(winners)>1
            for run in winners:
                totals[run]['unique' if len(winners)==1 else 'tied']+=1
                totals[run]['share']+=Fraction(1,len(winners))
        assert sum(r['share'] for r in totals.values())==expected_rows
        entries=[{**r,'share':float(r['share']),'closest':r['unique']+r['tied']} for r in totals.values()]
        result[scenario]={'answers':expected_rows,'tie_answers':tie_answers,'entries':entries}
    return {'scope':'AGGREGATE ONLY; SAME TEST ANSWERS; RAW PREDICTIONS','tolerance_points':tolerance,'scenarios':result,
            'rule':'Satu jawaban memberi total satu poin peringkat. Pemenang tunggal mendapat 1; seri k eksperimen masing-masing 1/k. Tidak memakai pembulatan kategori atau batas toleransi penilaian AES.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--sources',required=True,type=Path);a=p.parse_args()
    sources=json.loads(a.sources.read_text());predictions={}
    for run,path in sources.items():
        if run=='E0':raw=Path(path).read_bytes()
        else:
            with ZipFile(path) as z:raw=z.read('04_Prediksi_dan_Evaluasi.xlsx')
        predictions[run]=table(raw,'Nilai Guru dan Prediksi')
    ranking=compare_predictions(predictions)
    target=ROOT/'docs/snapshots/mlp_all_results_snapshot.json';data=json.loads(target.read_text());data['across_experiments']=ranking
    target.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(ranking,ensure_ascii=False))


if __name__=='__main__':main()
