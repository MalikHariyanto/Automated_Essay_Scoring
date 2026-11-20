"""Refresh the three-group catalog manifest without restoring historical active notebooks."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
GROUPS = ('01_satu_variabel', '02_tiga_input_single_trait', '03_tiga_input_multi_trait')
def build(result_paths=()):
    if result_paths:
        raise ValueError('Publikasi workbook memerlukan ekspor anonim; jangan impor workbook siswa mentah.')
    rows=[]
    for group in GROUPS:
        folder=ROOT/'eksperimen'/group
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                rows.append({'group':group,'path':path.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    target=ROOT/'eksperimen/manifest.json'
    target.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    print('Catalog files:',len(rows))
if __name__=='__main__':
    build()
