"""E33: E30 asli, hanya LR MLP S3 menjadi5e-5; format47cell dipertahankan."""
import ast
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from build_context_pair_notebook import source, replace_once

ROOT = Path(__file__).resolve().parents[1]
PIN = '6b3c52569851b06720cb80a0e9534e4f46b70eaf'
BASE = 'notebooks/experiments/AES_300_E30_E26_BiGRU_CoAttention_Pooling_GPU.ipynb'
NAME = 'AES_300_E33_E30_BiGRU_CoAttention_MLPLR5e5_GPU'
PLACEHOLDER = '__NOTEBOOK_CODE_SHA256__'


def build():
    raw = subprocess.check_output(['git', 'show', PIN+':'+BASE], cwd=ROOT)
    n = copy.deepcopy(json.loads(raw))
    old_hash = n['metadata']['aes_standalone']['code_sha256']
    for index, cell in enumerate(n['cells']):
        s = source(cell).replace(old_hash, PLACEHOLDER).replace(Path(BASE).stem, NAME)
        if index == 10:
            s = replace_once(s, "'s3_mlp_lr': 0.0001", "'s3_mlp_lr': 5e-5")
        if index == 12:
            s = replace_once(s, 'S2_MLP_LR=S3_MLP_LR=0.0001  # Seluruh parameter head MLP S2/S3.',
                             'S2_MLP_LR=1e-4  # Kontrol S2 mengikuti E30.\nS3_MLP_LR=5e-5  # E33: hanya head.* MLP S3 turun2kali.')
            s = s.replace("lesson_version='e30_e26_gpu_v1'", "lesson_version='e33_e30_mlp_s3_lr5e5_gpu_v1'")
            s = replace_once(s, "print('E30 | E26 + Co-Attention + Attention Pooling | LR MLP S2/S3:'",
                             "print('E33 | E30 + LR MLP S3 5e-5 | LR MLP S2/S3:'")
        cell['source'] = s
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    n['cells'][0]['source'] = f'''# E33 · E30 dengan LR MLP S3 5e-5

**Satu perubahan dari E30 asli:** seluruh parameter `head.*` MLP **S3** memakai LR **5e-5**, turun2kali dari1e-4. S1 tetap linear1e-4; MLP S2 tetap1e-4. RoBERTa S1/S3 tetap5e-6; BiGRU, co-attention, attention pooling, LayerNorm dan proyeksi tetap1e-4. Ridge tidak dipakai.

Susunan S3 mengikuti E30: tiga input → RoBERTa fine-tuned → BiGRU → co-attention respons–referensi → attention pooling → concat proyeksi BGE frozen → MLP. Input seluruh skenario: **soal, jawaban_referensi, response_text**. Label tunggal **rater0–100**, tidak menjadi input encoder.

**Web Colab biasa → Runtime → Change runtime type → T4 GPU → Run all → unggah fix_dataset_300.xlsx.** Notebook mandiri dengan47cell/30codecell mengikuti format E30; tanpa clone, file Python pendamping, ekstensi atau Drive. Semua S1/S2/S3 dilatih baru; tidak mengambil hasil lama.

**5fold CV, maksimal25epoch, early stopping patience5** berdasarkan RMSE validation; checkpoint epoch terbaik dipulihkan. Refit model baru pada240development memakai ceil median best epoch lima fold, lalu evaluasi60test. MSE target0–1, seed42/split2026, microbatch1/efektif8, dropout0,1, warmup10%+linear scheduler, AdamW decay0,01 dan clipping1 mengikuti E30. LR adalah nilai awal/puncak sebelum scheduler.

Source E30 dipin pada commit `{PIN}`. Folder `{NAME}` terpisah; checkpoint dua slot hanya lokal, resume hanya resep/data/kode yang sama. Runtime hilang berarti progres hilang; ZIP lima Excel tidak mencakup bobot model. Jangan memindahkan checkpoint E30/Ridge ke E33.

Belum ada hasil penelitian E33. Kandidat5e-5 adalah hipotesis eksperimen, bukan jaminan peningkatan. Bandingkan kontrol E30 dan E33 pada fold/seed sama melalui development/CV; test historis sudah sering dilihat sehingga perbandingan lanjutan eksploratif.
'''
    n['cells'][8]['source'] = '## 3 Konfigurasi E33 · E30, LR MLP S3 5e-5\n\nS2 tetap1e-4. Hanya `S3_MLP_LR` berubah; encoder, layer S3, dropout, loss dan scheduler mengikuti E30.\n'
    n['cells'][29]['source'] = source(n['cells'][29]).replace('## 10 E30 ·', '## 10 E33 · E30:') + '\nE33 mempertahankan arsitektur ini. Hanya LR MLP `head.*` S3 menjadi5e-5; parameter pooling/norm/proyeksi tetap1e-4.\n'
    codes = [source(c).removeprefix('%%aes\n') for c in n['cells'] if c['cell_type'] == 'code']
    for code in codes:
        ast.parse(code)
    fingerprint = hashlib.sha256('\n'.join(codes).encode()).hexdigest()
    for cell in n['cells']:
        cell['source'] = source(cell).replace(PLACEHOLDER, fingerprint)
    n['metadata']['colab']['name'] = NAME+'.ipynb'
    n['metadata']['aes_standalone'].update(code_sha256=fingerprint, base_commit=PIN, base_path=BASE,
                                          base_sha256=hashlib.sha256(raw).hexdigest())
    n['metadata']['aes_experiment'].update(name='E33', base_commit=PIN, base_path=BASE,
        base_sha256=hashlib.sha256(raw).hexdigest(), s3='E30 + LR MLP S3 5e-5',
        s2_mlp_lr=1e-4, s3_mlp_lr=5e-5, mlp_lr_scope='S3 model.head.* only', reused_controls=[])
    output = json.dumps(n, ensure_ascii=False, indent=1)+'\n'
    for directory in ('notebooks/experiments', 'aes-learning/notebooks/experiments'):
        (ROOT/directory/(NAME+'.ipynb')).write_text(output)
    print(NAME, len(n['cells']), 'cells', fingerprint)
    return n


if __name__ == '__main__':
    build()
