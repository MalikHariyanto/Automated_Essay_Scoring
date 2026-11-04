"""Complete E30 reference: original S1/S2/S3 recipe in plain native cells."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FORMAT_COMMIT = '02d4bf384421811fce77335e103cbb065a5bdfe2'
FORMAT_PATH = 'notebooks/experiments/AES_300_E35_E30_LayersLR5e5_Native_Learnable_GPU.ipynb'
NAME = 'AES_300_E30_Lengkap_S1_S2_S3_Native_Learnable_GPU'
PLACEHOLDER = '__NOTEBOOK_CODE_SHA256__'


def build():
    raw = subprocess.check_output(['git', 'show', FORMAT_COMMIT+':'+FORMAT_PATH], cwd=ROOT)
    n = copy.deepcopy(json.loads(raw))
    old_hash = n['metadata']['aes_standalone']['code_sha256']
    for cell in n['cells']:
        role = cell['metadata'].pop('aes35_role', '')
        if role:
            cell['metadata']['aes30full_role'] = role
        s = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
        s = s.replace(old_hash, PLACEHOLDER).replace(Path(FORMAT_PATH).stem, NAME)
        if role == 'hyperparameters':
            s = s.replace('resep E35, satu perubahan LR dari E30.', 'resep E30 asli, lengkap S1/S2/S3.')
            s = s.replace('S3_LAYERS_LR = 5e-5           # SATU perubahan: sequence_pool.* + roberta_norm.*.',
                          'S3_LAYERS_LR = 1e-4           # E30 asli: BiGRU/co-attention/pooling/norm S3.')
        if role == 'configuration_apply':
            s = s.replace("lesson_version='e35_native_layers_lr5e5_v1'", "lesson_version='e30_full_native_reference_v1'")
            s += "\nprint('Skenario yang dijalankan:', ', '.join(SCENARIOS))\nprint('Setiap skenario: 5 fold CV, satu refit, satu evaluasi test.')\n"
        if cell['cell_type'] == 'markdown':
            s = s.replace('E35', 'E30 acuan lengkap')
            if '## 5 · Hyperparameter' in s:
                s = '<a id="hyperparameter"></a>\n## 5 · Hyperparameter E30 lengkap\n\nResep asli E30: encoder RoBERTa S1/S3 seragam5e-6;linear S1,MLP S2/S3,proyeksi BGE dan BiGRU/co-attention/pooling/norm S3 semuanya1e-4. BiGRU64perarah,dropout0.1. Tiga skenario berjalan baru pada split dan fold yang sama.\n'
            if '## 22 · Optimizer' in s:
                s = '## 22 · Optimizer, loss MSE dan checkpoint\n\nEncoder RoBERTa S1/S3 memakai5e-6;head linear S1 dan MLP S2/S3 memakai1e-4. Proyeksi BGE serta pooling/norm S3 memakai1e-4seperti E30 asli. AdamW,weight decay0.01,warmup10%,scheduler linear,clipping1 dan MSE target0–1 mengikuti resep historis. Tidak ada LLRD atau Scalar Mix.\n'
            if '### `optimizer_groups`' in s:
                s = '### `optimizer_groups`\n\nSetiap parameter trainable diperiksa masuk tepat satu kelompok. Encoder5e-6,head/proyeksi/pooling_norm1e-4. S1 memakai linear,S2 memakai MLP,S3 memakai MLP setelah fusion.\n'
        cell['source'] = s
    n['cells'][0]['source'] = '''# E30 acuan lengkap · S1 + S2 + S3 · Native Learnable

**Kode acuan E30 lengkap dalam Python biasa per section.** Ketiga skenario dilatih dan dievaluasi baru pada data/split/fold yang sama. Versi ini melengkapi E30 native S3 saja yang dikirim sebelumnya; source dan hasil historis tetap utuh.

| Skenario | Encoder dan prediktor |
|---|---|
| S1 | Tiga teks → RoBERTa fine-tuned → CLS/mean window768 → dropout0.1 → Linear768→1. |
| S2 | Tiga teks → BGE-M3 frozen1024 → MLP1024→256→GELU→Dropout0.1→1. |
| S3 | Tiga teks → RoBERTa fine-tuned → BiGRU64perarah → co-attention respons–referensi → attention pooling → proyeksi768+LayerNorm; gabung BGE frozen1024→proyeksi256+LayerNorm; concat1024→MLP256→GELU→Dropout0.1→1. |

RoBERTa S1 dan S3 adalah model yang dilatih terpisah. BGE tetap frozen dan fiturnya dapat dipakai bersama S2/S3; head/proyeksi yang relevan tetap belajar. Semua skenario membaca X=`soal,jawaban_referensi,response_text`. Target y=`rater`0–100 hanya dipakai loss/evaluasi, tidak masuk encoder.

Resep E30 asli: encoder RoBERTa S1/S3 **5e-6**;head linear S1,MLP S2/S3,proyeksi BGE,BiGRU/co-attention/pooling/norm S3 **1e-4**;BiGRU64perarah,attention64,dropout0.1,concat dan MLP akhir. Tanpa perubahan Scalar Mix E41 atau LLRD E40. AdamW,weight decay0.01,warmup10%,scheduler linear,clipping1,batch efektif8/micro1tetap E30.

**Setiap skenario:5fold CV sekali,maksimal25epoch/fold,early stopping patience5,1refit+1test.** Total15fit CV dan3refit;test masing-masing skenario dinilai sekali. Ini tiga resep pembanding S1/S2/S3, bukan pengulangan tiga seed. TRAIN_SEED42satu konstanta; pola seed per fold mengikuti E30. Refit tiap skenario=ceil median best epoch lima fold. Semua pilihan dikunci sebelum test;test lama yang telah dilihat berulang kali tetap eksploratif.

**Cara menjalankan: T4 GPU → Run all → upload fix_dataset_300.xlsx.** Nama checkpoint,kolom X/y,hyperparameter,arsitektur,optimizer,training,evaluasi dan unduhan berada langsung di kernel Python. Tidak ada cell magic,worker tersembunyi,clone repo atau Drive. Lima Excel mencakup S1/S2/S3,seluruh vektor,prediksi,metrik,riwayat dan ringkasan15fold. Folder/hash/ZIP versi lengkap berbeda dari versi S3 saja.

Ini menjalankan ulang resep E30, bukan memuat prediksi/checkpoint historis. RMSE historis5.5247tidak dijamin terulang persis pada GPU/precision/dependency berbeda. Tidak ada hasil penelitian baru dalam notebook kosong ini. Unduh ZIP sebelum runtime dihapus.

Peta belajar: [nama model](#nama-model), [kolom X/y](#kolom-xy), [hyperparameter](#hyperparameter), [X/y aktual](#xy-aktual), [arsitektur](#arsitektur), [training](#training), [evaluasi](#evaluasi), [unduh hasil](#unduh).
'''
    fingerprint = hashlib.sha256('\n'.join(c['source'] for c in n['cells'] if c['cell_type'] == 'code').encode()).hexdigest()
    for index, cell in enumerate(n['cells']):
        cell['source'] = cell['source'].replace(PLACEHOLDER, fingerprint)
        cell['id'] = f'e30full-{index:03d}-'+hashlib.sha256(cell['source'].encode()).hexdigest()[:8]
        if cell['cell_type'] == 'code':
            ast.parse(cell['source'])
            cell['outputs'] = []
            cell['execution_count'] = None
    n['metadata']['colab']['name'] = NAME+'.ipynb'
    n['metadata']['aes_standalone']['code_sha256'] = fingerprint
    n['metadata']['aes_experiment'].update(name='E30_full_native', s3='E30 original MLP fusion; full S1/S2/S3 native reference',
        input_columns=['soal', 'jawaban_referensi', 'response_text'], scenarios=['S1', 'S2', 'S3'],
        s3_layers_lr=1e-4, new_layer_lr=1e-4, training_seed=42, multiple_seed_runs=False,
        cv_fits_per_scenario=5, cv_fits=15, refits=3, test_evaluations_per_scenario=1,
        scalar_mix=False, encoder_layer_decay=None, historical_e30_results_reproduced=False,
        format_commit=FORMAT_COMMIT, format_path=FORMAT_PATH, format_sha256=hashlib.sha256(raw).hexdigest(),
        treatment='Presentation-only complete E30 native reference; original S1/S2/S3 methods and LR')
    output = json.dumps(n, ensure_ascii=False, indent=1)+'\n'
    for directory in ('notebooks/experiments', 'aes-learning/notebooks/experiments'):
        (ROOT/directory/(NAME+'.ipynb')).write_text(output)
    print(json.dumps({'name': NAME, 'cells': len(n['cells']), 'code_sha256': fingerprint, 'scenarios': ['S1', 'S2', 'S3']}))
    return n


if __name__ == '__main__':
    build()
