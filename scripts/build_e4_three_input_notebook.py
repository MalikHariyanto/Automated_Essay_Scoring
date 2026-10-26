"""E4 context pair tiga input seluruh skenario; label rater, runtime CPU."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
from build_context_pair_notebook import source,replace_once

ROOT=Path(__file__).resolve().parents[1]
PIN='8380355f0db53a1d258f29a2ab17152fd965d85d'
BASE='notebooks/experiments/AES_300_E6_Tiga_Input_CPU.ipynb'
TARGET=ROOT/'notebooks/experiments/AES_300_E4_Tiga_Input_CPU.ipynb'


def build():
    raw=subprocess.check_output(['git','show',PIN+':'+BASE],cwd=ROOT)
    n=copy.deepcopy(json.loads(raw));previous=n['metadata']['aes_standalone']['code_sha256']
    for cell in n['cells']:
        s=source(cell).replace(previous,'__NOTEBOOK_CODE_SHA256__')
        s=s.replace('AES_300_E6_Tiga_Input_CPU','AES_300_E4_Tiga_Input_CPU')
        s=s.replace('Hasil_AES_E6_Tiga_Input_CPU.zip','Hasil_AES_E4_Tiga_Input_CPU.zip')
        s=s.replace('e6_three_inputs_all_scenarios_cpu_v1','e4_three_inputs_all_scenarios_cpu_v1')
        s=s.replace('E6 tiga input','E4 tiga input').replace('Konfigurasi E6','Konfigurasi E4')
        s=s.replace('LR encoder S3 tetap 5e-6','E4 tiga input: LR encoder S3=5e-6; S1 tetap 1e-5')
        if cell['cell_type']=='code':
            cell['outputs']=[];cell['execution_count']=None
        cell['source']=s;cell['metadata']={};cell.pop('attachments',None)
    intro=source(n['cells'][0])
    intro=intro.replace('Varian baru resep E6:', 'Varian resep **E4 konteks pair**:')
    intro=intro.replace('Ini berbeda dari E8 yang menambahkan konteks hanya ke RoBERTa S3.',
                        'Pada E4 awal, tambahan konteks hanya dibaca RoBERTa S3. Pada varian ini, ketiga kolom dibaca oleh kedua encoder di seluruh skenario; S1/S2/S3 harus dilatih ulang.')
    intro=intro.replace('**Training mengikuti E6:** LR encoder S1=1e-5, S3=5e-6',
                        '**Konteks pair E4 + LR pilihan:** encoder S1=1e-5, S3=5e-6 (E4 awal S3=1e-5)')
    intro=intro.replace('# AES 300 · E4 tiga input · CPU','# AES 300 · E4 konteks pair · tiga input · LR S3 5e-6 · CPU')
    intro=intro.replace('Hasil E6/E8 lama tidak perlu dihapus.','Hasil E4/E6/E8 lama tidak perlu dihapus.')
    intro=intro.replace('daripada E6 yang hanya melatih S3.','daripada run yang memakai ulang model lama.')
    n['cells'][0]['source']=intro
    codes=[source(c).removeprefix('%%aes\n') for c in n['cells'] if c['cell_type']=='code']
    for code in codes:ast.parse(code)
    fingerprint=hashlib.sha256('\n'.join(codes).encode()).hexdigest()
    for cell in n['cells']:cell['source']=source(cell).replace('__NOTEBOOK_CODE_SHA256__',fingerprint)
    n['metadata']['aes_standalone']['code_sha256']=fingerprint
    n['metadata']['colab']['name']=TARGET.name
    n['metadata']['aes_experiment'].update(name='E4_THREE_INPUTS_LR5E6_CPU',base_commit=PIN,base_path=BASE,
        base_sha256=hashlib.sha256(raw).hexdigest(),s3_encoder_lr=5e-6,
        drive_root='AES_300_E4_Tiga_Input_CPU_Privat')
    TARGET.write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n')
    (ROOT/'aes-learning/notebooks/experiments'/TARGET.name).write_bytes(TARGET.read_bytes())
    print(json.dumps({'notebook':str(TARGET.relative_to(ROOT)),'code_sha256':fingerprint,'s3_encoder_lr':5e-6}))
    return TARGET


if __name__=='__main__':build()
