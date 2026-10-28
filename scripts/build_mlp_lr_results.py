"""Audit hasil E6 LR dan tambahkan hitungan agregat tanpa data individual."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile
from build_mlp_lora_results import extract
from build_mlp_context_results import winner_counts
from build_mlp_study import ROOT, table


def audit(results, mlp):
    data=extract(results,mlp,method='lr5e6')
    (ROOT/'docs/snapshots/mlp_lr_snapshot.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    target=ROOT/'docs/snapshots/mlp_all_results_snapshot.json'
    all_results=json.loads(target.read_text())
    with ZipFile(results) as archive:
        all_results['prediction_winners']['E6']=winner_counts(table(archive.read('04_Prediksi_dan_Evaluasi.xlsx'),'Nilai Guru dan Prediksi'))
    target.write_text(json.dumps(all_results,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'metrics':data['metrics'],'CV':data['cvSummary'],'controls':data['control_prediction_delta']},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--mlp',required=True);a=p.parse_args();audit(a.results,a.mlp)
