"""E36: E30 S3 with encoder LR 3e-6, presented as E35 native Python cells.

The transformation happens when building the file, never inside Colab.
Historical notebooks are read from pinned commits and are not overwritten.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FORMAT_COMMIT = '699aaa3be612400ccf264bd8a8b96984263c8c8c'
FORMAT_PATH = 'notebooks/experiments/AES_300_E35_E30_LayersLR5e5_Native_Learnable_GPU.ipynb'
NAME = 'AES_300_E36_E30_S3_EncoderLR3e6_Native_Learnable_GPU'
PLACEHOLDER = '__NOTEBOOK_CODE_SHA256__'


def pinned_file(commit, path):
    return subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=ROOT)


def build():
    raw = pinned_file(FORMAT_COMMIT, FORMAT_PATH)
    original = json.loads(raw)
    notebook = copy.deepcopy(original)
    previous = original['metadata']['aes_standalone']['code_sha256']
    old_name = Path(FORMAT_PATH).stem

    for cell in notebook['cells']:
        source = cell['source']
        source = source if isinstance(source, str) else ''.join(source)
        source = source.replace(previous, PLACEHOLDER).replace(old_name, NAME)
        role = cell['metadata'].pop('aes35_role', '')
        if role:
            cell['metadata']['aes36_role'] = role
        source = source.replace('E30/E35', 'E30/E36').replace('E35', 'E36')
        source = source.replace('e35_native_layers_lr5e5_v1', 'e36_native_s3_encoder_lr3e6_v1')

        if cell['cell_type'] == 'code':
            # Use the selected scenario in every training, evaluation and feature loop.
            # Shared E30 model branches are retained for source traceability.
            source = re.sub(r"\('S1',\s*'S2',\s*'S3'\)", 'SCENARIOS', source)
            source = re.sub(r"for scenario in \('S1',\s*'S3'\)", 'for scenario in SCENARIOS', source)
            if role == 'imports':
                source = source.replace('SCENARIOS=SCENARIOS', "SCENARIOS = ('S3',)  # Hanya model fusion yang dijalankan.")
            if role in ('hyperparameters', 'configuration_apply'):
                source = '\n'.join(line for line in source.splitlines()
                    if not any(key in line for key in ("S1_HEAD_LR =", "S2_MLP_LR =", "'Kelompok':'Linear S1'", "'Kelompok':'MLP S2'"))) + '\n'
            source = source.replace('S1_HEAD_LR', '1e-4').replace('S2_MLP_LR', '1e-4')
            if role == 'hyperparameters':
                source = source.replace('ENCODER_LR = 5e-6', 'ENCODER_LR = 3e-6')
                source = source.replace('# RoBERTa S1 dan S3.', '# SATU perlakuan: encoder S3; baseline E30 = 5e-6.')
                source = source.replace('S3_LAYERS_LR = 5e-5', 'S3_LAYERS_LR = 1e-4')
                source = source.replace('# SATU perubahan: sequence_pool.* + roberta_norm.*.', '# Kembali ke E30: BiGRU/co-attention/pooling/norm = 1e-4.')
            if role == 'configuration_defaults':
                source = source.replace("('pretrained', 'pretrained_pair', 'S1', 'S3')", "('pretrained', 'pretrained_pair', 'S3')")
                source = source.replace('vector_source harus pretrained, S1, atau S3', 'vector_source harus pretrained, pretrained_pair, atau S3')
                source = source.replace('# Batas dan aturan berhenti sama untuk S1, S2, dan S3; bukan wajib 25 epoch.', '# Maksimal 25 epoch/fold; berhenti bila RMSE validasi tidak membaik selama patience.')
            if role == 'configuration_apply':
                source = source.replace("'Kelompok':'RoBERTa S1/S3'", "'Kelompok':'Encoder RoBERTa S3'")
                source = source.replace("torch.set_num_threads(2)", "cfg['scenarios'] = list(SCENARIOS)\ntorch.set_num_threads(2)")
            if role == 'presentation_helpers':
                source = source.replace("MODELS = {'S1':'RoBERTa','S2':'BGE-M3','S3':'Fusion RoBERTa + BGE-M3'}", "MODELS = {'S3':'Fusion RoBERTa + BGE-M3'}")
                source = source.replace('S1 = RoBERTa; S2 = BGE-M3; S3 = fusion keduanya.', 'E36 menjalankan S3: fusion RoBERTa + BGE-M3.')
            if role == 'definition:make_model':
                source = source.replace('def make_model(scenario, cfg):', "def make_model(scenario, cfg):\n    if scenario not in SCENARIOS:\n        raise ValueError('E36 hanya menjalankan S3; S1/S2 tidak dibuat atau dilatih.')")
            if role == 'architecture_export':
                source = '\n'.join(line for line in source.splitlines() if "'scenario':'S1'" not in line and "'scenario':'S2'" not in line) + '\n'
            if role in ('definition:answer_vectors_report', 'definition:word_vectors_report'):
                source = '\n'.join(line for line in source.splitlines() if "('14_S1_" not in line and "('15_S1_" not in line) + '\n'
            if role == 'definition:training_report':
                source = source.replace("[('S1','RoBERTa (S1)'),('S2','BGE-M3 (S2)'),('S3','Fusion (S3)')]", "[('S3','Fusion (S3)')]")
                source = source.replace("'s2_mlp_lr':'LR MLP S2',", '').replace("'encoder_lr':'Learning rate encoder RoBERTa S1',", '').replace("'head_lr':'LR head linear S1',", '')
            if role == 'definition:evaluation_report':
                source = source.replace("['S1, S2, S3','S1 = RoBERTa; S2 = BGE-M3; S3 = fusion keduanya.']", "['S3 saja','Fusion RoBERTa fine-tuned + BGE-M3 frozen; tidak ada prediksi kosong S1/S2.']")
                source = source.replace("['ID Jawaban','Nilai Guru','Prediksi S1','Prediksi S2','Prediksi S3',\n                                                 'Pembulatan S1','Pembulatan S2','Pembulatan S3',\n                                                 'Selisih S1','Selisih S2','Selisih S3']", "['ID Jawaban','Nilai Guru','Prediksi S3','Pembulatan S3','Selisih S3']")
            # Internal journal names also describe the actual S3-only contents.
            source = source.replace('12_evaluasi_tiga_skenario', '12_evaluasi_s3')
            source = source.replace("'Tiga_skenario'", "'S3'")
            source = source.replace('ketiga model yang sudah dilatih', 'model S3 yang sudah dilatih')
            source = source.replace('Split dan5fold sama untukS1/S2/S3', 'Split dan 5 fold S3 sama dengan metode E30')
            source = source.replace('input sama untuk S1 dan S3', 'input tiga teks yang sama dengan S3')
            source = source.replace('768 angka: S1/S3 membaca', '768 angka: S3 membaca')
            source = source.replace('tidak perlu disalin berulang untuk S2 dan S3', 'dipakai langsung oleh cabang fusion S3')
            source = source.replace('Perbandingan model pada data uji', 'Hasil S3 pada data uji')
            ast.parse(source)
            cell['outputs'] = []
            cell['execution_count'] = None
        else:
            source = source.replace('## 19 · Arsitektur S1/S2/S3', '## 19 · Arsitektur S3')
            source = source.replace('S2/S3 memakai fitur yang sama;', 'S3 memakai fitur BGE langsung tanpa menjalankan S2;')
            source = source.replace('yang sama untuk S1/S2/S3', 'yang mengikuti metode E30 untuk S3')
            source = source.replace('ketiga model', 'model S3')
            source = source.replace('encoder5e-6', 'encoder3e-6').replace('sequence_pool/roberta_norm5e-5', 'sequence_pool/roberta_norm1e-4')
        cell['source'] = source

    # Replace the few explanations that described E35's different treatment.
    notebook['cells'][0]['source'] = '''# E36 · E30 S3 saja · LR RoBERTa 3e-6 · Native Learnable

**Metode mengikuti E30; format section mengikuti E35.** Kode Python berjalan langsung di kernel Colab, tanpa cell magic, worker training, clone repo, mount Drive, atau file Python pendamping.

**Satu perlakuan:** LR encoder RoBERTa S3 menjadi **3e-6** (E30: 5e-6). LR MLP, proyeksi BGE, BiGRU, co-attention, attention pooling dan LayerNorm tetap **1e-4**, persis E30. Jangan memakai LR layer E35 5e-5 di eksperimen ini. BGE-M3 frozen; prediktor akhir MLP, bukan Ridge.

**S3 saja:** tidak membuat/melatih S1 atau S2, tidak memakai checkpoint/hasil skenario tersebut. Cabang RoBERTa dan BGE di dalam S3 tetap diperlukan. BGE dipersiapkan di bagian17 langsung dari tiga input, tanpa target guru. Kelas dasar dari E30 masih memperlihatkan cabang historis untuk keterlacakan; SCENARIOS dan make_model membatasi eksekusi ke S3.

Pilih runtime baru **T4 GPU → Run all → upload fix_dataset_300.xlsx**. Tiga input: soal, jawaban_referensi, response_text. Satu target: rater 0–100; target tidak masuk encoder. **5-fold CV, maksimal25epoch/fold, early stopping patience5**, seed42, split_seed2026, 20% test. Refit memakai ceil median best epoch5fold. Ini hipotesis penelitian; tidak menjamin hasil lebih baik daripada E30. Test lama sudah sering dilihat sehingga perbandingan selanjutnya eksploratif.

Peta belajar: [nama model](#nama-model), [kolom X/y](#kolom-xy), [hyperparameter](#hyperparameter), [X/y aktual](#xy-aktual), [arsitektur](#arsitektur), [training](#training), [evaluasi](#evaluasi), [unduh hasil](#unduh). Hasil privat hanya disimpan lokal di runtime; unduh ZIP sebelum runtime dihapus.
'''
    for c in notebook['cells']:
        if c['cell_type'] != 'markdown':
            continue
        if '## 5 · Hyperparameter' in c['source']:
            c['source'] = '<a id="hyperparameter"></a>\n## 5 · Hyperparameter E36\n\nHanya encoder RoBERTa S3 memakai 3e-6. Semua kelompok lain kembali ke E30 1e-4. Nilai ini LR awal/puncak sebelum warmup dan scheduler linear; bukan tingkat kecerdasan model.\n'
        if '## 19 · Arsitektur S3' in c['source']:
            c['source'] = '<a id="arsitektur"></a>\n## 19 · Arsitektur S3\n\nTiga input → RoBERTa fine-tuned → respons BiGRU128 + referensi128 → co-attention → matching128 → attention pooling → proyeksi768 → LayerNorm. BGE frozen1024 → proyeksi256 + LayerNorm. Concat1024 → MLP256 → satu skor. Semua bobot awal dan rumus mengikuti E30.\n'
        if '## 22 · Optimizer' in c['source']:
            c['source'] = '## 22 · Optimizer, loss MSE, early stopping dan checkpoint\n\nRoBERTa S3:3e-6. MLP, proyeksi BGE dan pooling/norm:1e-4. Semua parameter trainable masuk satu kelompok optimizer tepat sekali. fit() adalah satu latihan fold/refit, bukan proses tersembunyi.\n'
        c['source'] = c['source'].replace('S3 head1e-4, sequence_pool/roberta_norm1e-4, proyeksi BGE1e-4, encoder3e-6. S1/S2 tetap resep kontrol E30.', 'S3: encoder3e-6; head, proyeksi BGE dan pooling/norm1e-4. S1/S2 tidak dijalankan.')
        c['source'] = c['source'].replace('S1: RoBERTa→linear. S2: BGE frozen→MLP. ', '')

    # Hash all visible code, keeping resume/cache identity distinct from E30/E35.
    fingerprint = hashlib.sha256('\n'.join(c['source'] for c in notebook['cells'] if c['cell_type'] == 'code').encode()).hexdigest()
    for i, cell in enumerate(notebook['cells']):
        cell['source'] = cell['source'].replace(PLACEHOLDER, fingerprint)
        cell['id'] = f'e36-{i:03d}-' + hashlib.sha256(cell['source'].encode()).hexdigest()[:8]
    notebook['metadata']['colab']['name'] = NAME + '.ipynb'
    notebook['metadata']['aes_standalone'].update(code_sha256=fingerprint, native_kernel=True)
    notebook['metadata']['aes_experiment'].update(
        name='E36', s3='E30 S3 only; encoder RoBERTa LR 3e-6',
        encoder_lr=3e-6, s3_encoder_lr=3e-6, s3_mlp_lr=1e-4,
        s3_layers_lr=1e-4, new_layer_lr=1e-4, projection_lr=1e-4,
        native_kernel=True, scenarios=['S3'], reused_controls=[],
        inspection_cosine=False, format_commit=FORMAT_COMMIT, format_path=FORMAT_PATH,
        format_sha256=hashlib.sha256(raw).hexdigest(),
        treatment='s3_encoder_lr only; S3-only execution and native presentation do not replace either fusion branch')
    output = json.dumps(notebook, ensure_ascii=False, indent=1) + '\n'
    for directory in ['notebooks/experiments', 'aes-learning/notebooks/experiments']:
        (ROOT / directory / (NAME + '.ipynb')).write_text(output)
    print(json.dumps({'name': NAME, 'code_sha256': fingerprint, 'cells': len(notebook['cells']),
                      'scenarios': ['S3'], 'encoder_lr': 3e-6, 's3_layers_lr': 1e-4}))
    return notebook


if __name__ == '__main__':
    build()
