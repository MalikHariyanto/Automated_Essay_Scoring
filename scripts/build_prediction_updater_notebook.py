"""Build a small standalone Colab for completed result ZIPs; no retraining."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NAME='05_Perbarui_Prediksi_Hasil_Tanpa_Training.ipynb'


def build():
    cells=[]
    def add(kind,source):
        cell={'cell_type':kind,'metadata':{},'source':source}
        if kind=='code':
            ast.parse(source)
            cell.update(outputs=[],execution_count=None)
        cell['id']='prediction-'+str(len(cells))+'-'+hashlib.sha256(source.encode()).hexdigest()[:8]
        cells.append(cell)
    add('markdown','''# Perbarui rincian prediksi — tanpa training

Upload ZIP hasil **E46 atau E47 yang sudah selesai dan lengkap**. Tidak memerlukan dataset, GPU, RoBERTa, atau training ulang. ZIP sumber tetap utuh; salinan baru mengubah Excel04 saja. Empat Excel lainnya disalin identik.

Excel04 berisi **Nilai Guru dan Prediksi** (skor mentah 0–100, normalisasi 0–1, pembulatan, selisih), **Selisih Tepat** (=0/1/2/3/4/5/10 poin dan selisih lain), serta **Toleransi Selisih** (≤0/1/2/3/4/5/10). Rekap memuat jumlah dan persentase pada test, terpisah sebelum/sesudah pembulatan.

Normalisasi berbeda dari pembulatan: 1 poin = 0,01 pada skala 0–1. Pembulatan memakai aturan eksperimen: ke kategori terdekat dalam 0–100, tepat .5 ke bawah. Nilai mentah dan metrik lama tetap utuh. File dengan ringkasan kosong ditolak; notebook ini tidak memperbaiki bug ringkasan lama E44/E45.
''')
    add('markdown','## 1. Pustaka untuk membaca Excel dan ZIP')
    add('code','''import os
import sys
import json
import subprocess
import importlib.util
from pathlib import Path
if importlib.util.find_spec('openpyxl') is None:
    subprocess.check_call([sys.executable,'-m','pip','install','openpyxl==3.1.5'])
''')
    add('markdown','## 2. Lokasi file\n\nDi Colab, biarkan INPUT_ZIP_FILE kosong agar pemilih upload muncul. Di komputer sendiri isi path ZIP hasil lengkap.')
    add('code','''INPUT_ZIP_FILE = os.environ.get('AES_RESULT_ZIP', '')
OUTPUT_FOLDER = os.environ.get('AES_RESULT_OUTPUT', 'hasil_prediksi_lengkap')
output_folder=Path(OUTPUT_FOLDER).expanduser().resolve()
output_folder.mkdir(parents=True,exist_ok=True)
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
if INPUT_ZIP_FILE:
    input_path=Path(INPUT_ZIP_FILE).expanduser().resolve()
elif IN_COLAB:
    from google.colab import files
    uploaded=files.upload()
    if len(uploaded)!=1:
        raise ValueError('Upload tepat satu ZIP hasil lengkap E46/E47.')
    filename,payload=next(iter(uploaded.items()))
    input_path=output_folder/Path(filename).name
    input_path.write_bytes(payload)
    del uploaded,payload
else:
    raise ValueError('Isi INPUT_ZIP_FILE dengan path ZIP hasil lengkap.')
if not input_path.is_file() or input_path.suffix.lower()!='.zip':
    raise ValueError('File input harus ZIP hasil yang tersedia.')
print('Sumber:',input_path.name)
''')
    add('markdown','## 3. Fungsi pelaporan\n\nSeluruh fungsi terlihat di sini; tidak memanggil model atau menjalankan training.')
    add('code',(ROOT/'scripts/prediction_report_native.py').read_text())
    add('markdown','## 4. Buat dan periksa ZIP baru')
    add('code','''destination=output_folder/(input_path.stem+'_Prediksi_Lengkap.zip')
result=update_prediction_zip(input_path,destination)
print(json.dumps(result,ensure_ascii=False,indent=2))
print('Buka 04_Prediksi_dan_Evaluasi.xlsx → Nilai Guru dan Prediksi / Selisih Tepat / Toleransi Selisih.')
''')
    add('markdown','## 5. Unduh hasil')
    add('code','''if IN_COLAB:
    files.download(result['archive'])
else:
    print('ZIP tersimpan:',result['archive'])
''')
    code_hash=hashlib.sha256('\n'.join(c['source'] for c in cells if c['cell_type']=='code').encode()).hexdigest()
    notebook={'cells':cells,'nbformat':4,'nbformat_minor':5,'metadata':{
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},'colab':{'name':NAME},
        'prediction_report':{'code_sha256':code_hash,'training':False,'changes_excel04_only':True}}}
    raw=json.dumps(notebook,ensure_ascii=False,indent=1)+'\n'
    for folder in ['notebooks','aes-learning/notebooks']:
        (ROOT/folder/NAME).write_text(raw)
    print(json.dumps({'name':NAME,'cells':len(cells),'code_sha256':code_hash}))


if __name__=='__main__':build()
