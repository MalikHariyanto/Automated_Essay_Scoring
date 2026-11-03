"""Current E37: one encoder LR7e-6, five folds, no multi-seed/control sweep."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = '349f2e82e876333bf2270d47e7eb86b59e939a55'
BASE_PATH = 'notebooks/experiments/AES_300_E36_E30_S3_EncoderLR3e6_Native_Learnable_GPU.ipynb'
NAME = 'AES_300_E37_E30_S3_EncoderLR7e6_Native_Learnable_GPU'
PLACEHOLDER = '__NOTEBOOK_CODE_SHA256__'


def build():
    raw = subprocess.check_output(['git','show',BASE_COMMIT+':'+BASE_PATH],cwd=ROOT)
    notebook = copy.deepcopy(json.loads(raw))
    previous = notebook['metadata']['aes_standalone']['code_sha256']
    for c in notebook['cells']:
        role = c['metadata'].pop('aes36_role','')
        if role:c['metadata']['aes37_role'] = role
        s = ''.join(c['source']) if isinstance(c['source'],list) else c['source']
        s = s.replace(previous,PLACEHOLDER).replace(Path(BASE_PATH).stem,NAME)
        s = s.replace('E36','E37').replace('3e-6','7e-6').replace('lr3e6','lr7e6_single')
        if role == 'hyperparameters':
            s = s.replace('TRAIN_SEED = 42','TRAIN_SEED = 42  # Satu konstanta pengacakan; tidak ada loop beberapa seed.')
        c['source'] = s
    notebook['cells'][0]['source'] = '''# E37 · E30 S3 · LR encoder7e-6 · 5-fold CV

**Versi aktif sesuai koreksi pengguna: satu resep, lima fold CV.** Langsung menguji LR encoder RoBERTa S3 **7e-6** dibanding acuan historis E30 **5e-6**. Tidak menjalankan kontrol E30 lagi, tidak ada tiga seed atau pencarian LR otomatis. `TRAIN_SEED=42` hanya satu konstanta pengacakan, bukan eksperimen tambahan.

Arsitektur E30 tetap: tiga teks → RoBERTa fine-tuned → BiGRU → co-attention respons–referensi → attention pooling → concat proyeksi BGE frozen → MLP → satu skor0–100. MLP, proyeksi BGE dan seluruh pooling/norm/layer baru **1e-4**. BGE frozen, prediktor akhir MLP. S1/S2 tidak dijalankan atau dimuat dari hasil lama.

**5-fold CV sekali, maksimal25epoch/fold, early stopping patience5.** Setelah kelima fold selesai, refit sekali pada240development menggunakan ceil median best epoch5fold, kemudian test60jawaban sekali. Jadi total **5latihan fold +1refit**. Test tidak memilih LR atau epoch; perbandingan dengan test lama tetap eksploratif.

**Format native E35/E36:** Python per section langsung di kernel, tanpa magic, worker tersembunyi, pemeriksaan cosine, clone repo, atau Google Drive. X terdiri dari `soal`, `response_text`, `jawaban_referensi`; y=`rater` tetap diperlukan untuk training dan evaluasi, tidak masuk encoder. Output lima Excel standar.

Pilih runtime baru **T4 GPU → Run all → upload fix_dataset_300.xlsx**. Versi tiga seed terdahulu sudah digantikan oleh notebook ini. Tidak memakai checkpoint eksperimen lama. Run ulang pada runtime yang sama dapat melanjutkan checkpoint yang cocok; runtime dihapus berarti penyimpanan lokal ikut hilang. Unduh ZIP sebelum sesi berakhir.

Peta belajar: [nama model](#nama-model), [kolom X/y](#kolom-xy), [hyperparameter](#hyperparameter), [X/y aktual](#xy-aktual), [arsitektur](#arsitektur), [training](#training), [evaluasi](#evaluasi), [unduh hasil](#unduh).
'''
    fingerprint = hashlib.sha256('\n'.join(c['source'] for c in notebook['cells'] if c['cell_type']=='code').encode()).hexdigest()
    for i,c in enumerate(notebook['cells']):
        c['source'] = c['source'].replace(PLACEHOLDER,fingerprint)
        c['id'] = f'e37-single-{i:03d}-'+hashlib.sha256(c['source'].encode()).hexdigest()[:8]
    notebook['metadata']['colab']['name'] = NAME+'.ipynb'
    notebook['metadata']['aes_standalone']['code_sha256'] = fingerprint
    notebook['metadata']['aes_experiment'].update(
        name='E37',s3='E30 S3 only; encoder LR7e-6; single five-fold CV',
        encoder_lr=7e-6,s3_encoder_lr=7e-6,s1='not run',s2='not run',
        scenarios=['S3'],training_seed=42,multiple_seed_runs=False,control_retrained=False,
        cv_fits=5,refits=1,test_evaluations=1,ensemble=False,
        format_commit=BASE_COMMIT,format_path=BASE_PATH,format_sha256=hashlib.sha256(raw).hexdigest(),
        mlp_lr_scope='S3 model.head.* only',
        treatment='s3_encoder_lr only; one fixed training seed, five-fold CV once; supersedes paired design')
    output = json.dumps(notebook,ensure_ascii=False,indent=1)+'\n'
    for directory in ('notebooks/experiments','aes-learning/notebooks/experiments'):
        (ROOT/directory/(NAME+'.ipynb')).write_text(output)
    print(json.dumps({'name':NAME,'cells':len(notebook['cells']),'code_sha256':fingerprint,'cv_fits':5,'refits':1}))
    return notebook


if __name__ == '__main__':build()
