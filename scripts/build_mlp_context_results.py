"""Audit hasil E4 context pair, agregat pemenang prediksi dan data pembanding."""
import argparse
import json
import sys
from pathlib import Path
from zipfile import ZipFile
from build_mlp_lora_results import extract
from build_mlp_study import ROOT,table,workbook_inventory


def winner_counts(rows):
    counts={s:0 for s in ('S1','S2','S3')};counts['Seri']=0
    for row in rows:
        errors={s:abs(row['Prediksi '+s]-row['Nilai Guru']) for s in ('S1','S2','S3')}
        best=min(errors.values());win=[s for s,x in errors.items() if abs(x-best)<=1e-5]
        counts[win[0] if len(win)==1 else 'Seri']+=1
    if sum(counts.values())!=60:raise ValueError('Prediksi harus 60 jawaban.')
    return counts


def audit(context, mlp, frozen, lora, baseline_eval):
    result=extract(context,mlp,method='context')
    (ROOT/'docs/snapshots/mlp_context_snapshot.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    entries={}
    for id,path in [('E2',mlp),('E3',frozen),('E4',context),('E5',lora)]:
        with ZipFile(path) as z:
            rows=table(z.read('04_Prediksi_dan_Evaluasi.xlsx'),'Nilai Guru dan Prediksi')
            entries[id]=winner_counts(rows)
    entries['E0']=winner_counts(table(Path(baseline_eval).read_bytes(),'Nilai Guru dan Prediksi'))
    data={'scope':'AGGREGATE WIN COUNTS ONLY; NO STUDENT ROWS','prediction_winners':entries,
          'baseline_eval_book':{'file':'04_Prediksi_dan_Evaluasi.xlsx','sheets':workbook_inventory(Path(baseline_eval).read_bytes())},
          'numeric_tie_tolerance_points':1e-5,
          'limits':'Tiap jawaban dihitung satu kali. Seri dipisahkan. Hitungan terdekat bukan RMSE atau bukti keunggulan statistik.'}
    (ROOT/'docs/snapshots/mlp_all_results_snapshot.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'metrics':result['metrics'],'CV':result['cvSummary'],'prediction_winners':entries},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('context','mlp','frozen','lora','baseline-eval'):p.add_argument('--'+name,required=True)
    a=p.parse_args();audit(a.context,a.mlp,a.frozen,a.lora,a.baseline_eval)
