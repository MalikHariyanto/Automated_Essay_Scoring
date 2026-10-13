"""Fixture metrik buatan; tidak dipakai sebagai dataset training."""
import argparse,csv,json,shutil,subprocess,sys,tempfile,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from aes.metrics import gwet_ac2
from aes.io import dump

FIXTURES=[([1,2,3,4,5],[1,2,3,4,5]),([1,1,5,5],[1,2,4,5]),
          ([2,2,2,4,4,4],[1,2,3,3,4,5]),([1,1,1,1],[1,1,1,1]),
          ([1,2,3,4,5],[5,4,3,2,1])]
def verify(rscript='Rscript'):
    rows=[]
    for a,b in FIXTURES:
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'ratings.csv'
            with path.open('w') as f:
                w=csv.writer(f);w.writerow(['teacher','prediction']);w.writerows(zip(a,b))
            p=subprocess.run([rscript,str(Path(__file__).with_name('ac2_reference.R')),str(path)],capture_output=True,text=True,check=True)
            ref=float(p.stdout.strip());python=gwet_ac2(a,b,[1,2,3,4,5])
            assert abs(python-ref)<=5.01e-6,(python,ref)
            rows.append({'ratings':list(zip(a,b)),'python':python,'irrCAC_AC2':ref,'absolute_error':abs(python-ref)})
    return rows
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--require-r',action='store_true');p.add_argument('--rscript',default=os.environ.get('AES_RSCRIPT','Rscript'));args=p.parse_args()
    if not shutil.which(args.rscript):
        if args.require_r:raise SystemExit('Rscript belum terpasang; AC2 belum diverifikasi terhadap R.')
        print('SKIPPED: Rscript tidak tersedia');raise SystemExit(0)
    result=verify(args.rscript);dump(Path(__file__).resolve().parents[1]/'outputs/ac2_verification.json',result);print(json.dumps(result,indent=2))
