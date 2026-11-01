"""E34: E33 with only pooling_norm learning rate reduced to5e-5."""
import ast
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from build_context_pair_notebook import source, replace_once

ROOT = Path(__file__).resolve().parents[1]
PIN = 'fa45c47a4d972a20a36445ea58d1fca8e8adcded'
BASE = 'notebooks/experiments/AES_300_E33_E30_BiGRU_CoAttention_MLPLR5e5_GPU.ipynb'
NAME = 'AES_300_E34_E33_BiGRU_CoAttention_LayersLR5e5_GPU'
PLACEHOLDER = '__NOTEBOOK_CODE_SHA256__'


def build():
    raw = subprocess.check_output(['git', 'show', PIN+':'+BASE], cwd=ROOT)
    n = copy.deepcopy(json.loads(raw))
    previous = n['metadata']['aes_standalone']['code_sha256']
    for index, cell in enumerate(n['cells']):
        s = source(cell).replace(previous, PLACEHOLDER).replace(Path(BASE).stem, NAME)
        if index == 10:
            s = replace_once(s, "'s3_layers_lr': 1e-4", "'s3_layers_lr': 5e-5")
        if index == 12:
            s = replace_once(s, 'S3_LAYERS_LR=1e-4  # BiGRU, attention pooling, co-attention jika ada, norm dan proyeksi ringkasan.',
                             'S3_LAYERS_LR=5e-5  # E34: kelompok pooling_norm S3 turun2kali; proyeksi BGE tetap1e-4.')
            s = s.replace("lesson_version='e33_e30_mlp_s3_lr5e5_gpu_v1'", "lesson_version='e34_e33_s3_layers_lr5e5_gpu_v1'")
            s = s.replace("print('E33 | E30 + LR MLP S3 5e-5 | LR MLP S2/S3:'", "print('E34 | E33 + LR layer S3 5e-5 | LR MLP S2/S3:'")
        cell['source'] = s
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    n['cells'][0]['source'] = f'''# E34 · E33 dengan LR layer S3 5e-5

**Satu perubahan dari E33:** `s3_layers_lr` turun **1e-4 → 5e-5**. Kelompok `pooling_norm` mencakup `sequence_pool.*` (BiGRU, co-attention, attention pooling dan proyeksi ringkasan128→768) serta `roberta_norm.*`. Ini satu kelompok optimizer, bukan ablasi satu layer saja.

**MLP S3 tetap5e-5 mengikuti E33.** Encoder RoBERTa S1/S3 tetap5e-6; linear S1, MLP S2 dan **proyeksi BGE `projection.*` tetap1e-4**. BGE encoder frozen. Susunan S3: tiga input → RoBERTa fine-tuned → BiGRU → co-attention respons–referensi → attention pooling → concat proyeksi BGE → MLP. Prediktor akhir MLP.

Input seluruh skenario: **soal, jawaban_referensi, response_text**. Label tunggal **rater0–100**, tidak masuk encoder. **T4 GPU → Run all → unggah fix_dataset_300.xlsx**. Notebook mandiri47cell/30codecell mengikuti E33/E30; tanpa clone, ekstensi, Drive atau file Python pendamping. Semua S1/S2/S3 dilatih baru; S1/S2 mempertahankan resep kontrol.

**5fold CV, maksimal25epoch, early stopping patience5** pada RMSE validasi. Bobot epoch terbaik dipulihkan. Refit model baru pada240development memakai ceil median best epoch lima fold, lalu evaluasi60test. Seed42/split2026, MSE target0–1, microbatch1/efektif8, dropout0,1, warmup10%+linear scheduler, AdamW decay0,01 dan clipping1 sama dengan E33. LR adalah nilai awal/puncak sebelum scheduler.

Source E33 dipin pada commit `{PIN}`. Folder `{NAME}` terpisah, checkpoint dua slot hanya lokal. Resume hanya resep/data/kode yang sama; jangan memakai checkpoint E33/E30. Runtime hilang berarti progres hilang; ZIP lima Excel tidak menyertakan bobot model.

Belum ada hasil penelitian E34. Kandidat ini menguji apakah pembaruan representasi lebih kecil membantu kedekatan prediksi. Bandingkan dengan kontrol E33 pada fold/seed sama melalui CV; jangan memilih konfigurasi/seed dari test historis yang sudah berulang kali dilihat. Tidak ada jaminan hasil lebih baik.
'''
    n['cells'][8]['source'] = '## 3 Konfigurasi E34 · hanya LR kelompok layer S3 berubah\n\nMLP S3 tetap5e-5; `S3_LAYERS_LR` menjadi5e-5. S1/S2 dan proyeksi BGE tetap resep E33.\n'
    architecture = source(n['cells'][29])
    marker = '\nE33 mempertahankan arsitektur ini.'
    assert marker in architecture
    n['cells'][29]['source'] = architecture.split(marker)[0].replace('## 10 E33 · E30:', '## 10 E34 · arsitektur E33/E30:') + '''

E34 mempertahankan seluruh arsitektur E33. `head.*` MLP S3 tetap5e-5. Kelompok `sequence_pool.*` dan `roberta_norm.*` kini5e-5. Proyeksi ringkasan di dalam sequence_pool ikut kelompok ini; **proyeksi BGE `projection.*` tetap1e-4**. Fungsi optimizer asli memastikan setiap parameter trainable tercakup tepat sekali.
'''
    codes = [source(c).removeprefix('%%aes\n') for c in n['cells'] if c['cell_type'] == 'code']
    for code in codes:
        ast.parse(code)
    fingerprint = hashlib.sha256('\n'.join(codes).encode()).hexdigest()
    for cell in n['cells']:
        cell['source'] = source(cell).replace(PLACEHOLDER, fingerprint)
    n['metadata']['colab']['name'] = NAME+'.ipynb'
    n['metadata']['aes_standalone'].update(code_sha256=fingerprint, base_commit=PIN, base_path=BASE,
                                          base_sha256=hashlib.sha256(raw).hexdigest())
    n['metadata']['aes_experiment'].update(name='E34', base_commit=PIN, base_path=BASE,
        base_sha256=hashlib.sha256(raw).hexdigest(), s3='E33 + LR kelompok pooling_norm S3 5e-5',
        s2_mlp_lr=1e-4, s3_mlp_lr=5e-5, s3_layers_lr=5e-5, new_layer_lr=5e-5,
        treatment='s3_layers_lr only: sequence_pool.* and roberta_norm.*', reused_controls=[])
    output = json.dumps(n, ensure_ascii=False, indent=1)+'\n'
    for directory in ('notebooks/experiments', 'aes-learning/notebooks/experiments'):
        (ROOT/directory/(NAME+'.ipynb')).write_text(output)
    print(NAME, len(n['cells']), 'cells', fingerprint)
    return n


if __name__ == '__main__':
    build()
