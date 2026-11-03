"""E40: E30 MLP architecture, native S3 only, layer-wise encoder LR decay."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FORMAT_COMMIT = '5f1c993804560d12c06d7d11d4540e6d11307cf2'
FORMAT_PATH = 'notebooks/experiments/AES_300_E39_E30_S3_BiGRU128_Native_Learnable_GPU.ipynb'
NAME = 'AES_300_E40_E30_S3_LLRD095_Native_Learnable_GPU'
PLACEHOLDER = '__NOTEBOOK_CODE_SHA256__'

OPTIMIZER = '''def encoder_lr_schedule(number_of_layers, cfg):
    """LR puncak per kedalaman; scheduler waktu tetap warmup + linear E30."""
    if type(number_of_layers) is not int or number_of_layers < 1:
        raise ValueError('Jumlah layer encoder harus bilangan bulat positif.')
    peak = cfg['s3_encoder_lr']
    decay = cfg['encoder_layer_decay']
    if not math.isfinite(peak) or peak <= 0:
        raise ValueError('s3_encoder_lr harus positif dan finite.')
    if isinstance(decay, bool) or not math.isfinite(decay) or not 0 < decay <= 1:
        raise ValueError('encoder_layer_decay harus berada pada rentang (0,1].')
    # Layer dihitung dari bawah: indeks0 dekat embedding; indeksL-1 paling atas.
    rates = {'roberta.embeddings': peak * decay**number_of_layers}
    for index in range(number_of_layers):
        rates[f'roberta.encoder.layer.{index}'] = peak * decay**(number_of_layers-1-index)
    return rates


def optimizer_groups(model, cfg):
    """Setiap parameter trainable masuk tepat satu kelompok dengan LR yang jelas."""
    if model.scenario != 'S3':
        raise ValueError('E40 hanya membuat optimizer S3.')
    if not hasattr(model.roberta, 'embeddings') or not hasattr(model.roberta.encoder, 'layer'):
        raise ValueError('LLRD memerlukan struktur RoBERTa embeddings dan encoder.layer.')
    number_of_layers = len(model.roberta.encoder.layer)
    if number_of_layers != model.roberta.config.num_hidden_layers:
        raise ValueError('Jumlah layer aktual berbeda dari konfigurasi encoder.')
    rates = encoder_lr_schedule(number_of_layers, cfg)
    # Pooler hanya ditangani jika checkpoint menyediakannya; loader E30 menonaktifkannya.
    rates.update({'roberta.pooler': cfg['s3_encoder_lr'],
                  'projection': cfg['projection_lr'], 'head': cfg['s3_mlp_lr'],
                  'pooling_norm': cfg['s3_layers_lr']})
    buckets = {name: [] for name in rates}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith('roberta.embeddings.'):
            group = 'roberta.embeddings'
        elif name.startswith('roberta.encoder.layer.'):
            group = '.'.join(name.split('.')[:4])
        elif name.startswith('roberta.pooler.'):
            group = 'roberta.pooler'
        elif name.startswith('projection.'):
            group = 'projection'
        elif name.startswith('head.'):
            group = 'head'
        elif name.startswith(('sequence_pool.', 'roberta_norm.')):
            group = 'pooling_norm'
        else:
            raise ValueError('Parameter belum mempunyai kelompok LR: '+name)
        if group not in buckets:
            raise ValueError('Indeks layer tidak cocok dengan encoder aktual: '+name)
        buckets[group].append(parameter)
    result = [{'name': name, 'params': parameters, 'lr': rates[name]}
              for name, parameters in buckets.items() if parameters]
    ids = [id(parameter) for group in result for parameter in group['params']]
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError('Optimizer harus mencakup setiap parameter trainable tepat sekali.')
    return result
'''

INTRO = '''# E40 · E30 MLP S3 · Learning rate per layer RoBERTa · Native Learnable

**Satu perlakuan: layer-wise learning rate decay (LLRD) dengan faktor0.95.** Layer paling atas RoBERTa memakai LR puncak5e-6. Setiap turun satu layer, LR dikali0.95. Untuk12layer: indeks0 memakai2.844e-6 dan embeddings2.702e-6. Semua layer RoBERTa tetap dilatih. Jumlah layer diambil dari model aktual, bukan diasumsikan selalu12.

Arsitektur kembali ke **E30 MLP asli**: RoBERTa fine-tuned → BiGRU64perarah → co-attention respons–referensi → attention pooling128→64→1 → proyeksi768+LayerNorm; BGE-M3 frozen1024→proyeksi256+LayerNorm; concat1024→MLP256→GELU→Dropout0.1→1. MLP adalah prediktor akhir. Hidden128perarah E39 tidak dibawa.

MLP/proyeksi/BiGRU/co-attention/pooling/norm tetap1e-4; dropoutMLP0.1. Warmup10%,scheduler linear,AdamW,weight decay0.01,clipping1,batch efektif8/micro1 tetap E30. LR per layer menyatakan pembagian menurut kedalaman; warmup dan linear mengalikan seluruh kelompok menurut langkah training.

**S3 saja,5fold CV sekali,maksimal25epoch/fold,patience5,1refit+1test.** TRAIN_SEED42satu konstanta; seed tiap fold mengikuti E30, bukan pengulangan multi-seed. Refit=ceil median best epoch5fold. Test tidak memilih LR/epoch. Hasil test lama yang sudah dilihat berulang kali tetap eksploratif; tidak ada jaminan E40 mengalahkan E30.

Tiga input X=`soal,jawaban_referensi,response_text`; target y=`rater`0–100 tidak masuk encoder. Kode Python per section berada langsung dalam kernel, tanpa cell magic atau worker tersembunyi. Nama checkpoint,kolom,hyperparameter,arsitektur,optimizer,training,evaluasi dan unduhan mudah dicari.

**T4 GPU → Run all → upload fix_dataset_300.xlsx.** Output tetap lima Excel standar. Pada05→Pengaturan Eksperimen tersimpan faktor LLRD; sheet **Learning Rate per Layer** mencatat kelompok dan LR puncak aktual setiap fold/refit. Ringkasan co-attention kembali128komponen; fusion1024. Cache/checkpoint E40 terpisah; unduh ZIP sebelum runtime dihapus.

Peta belajar: [nama model](#nama-model), [kolom X/y](#kolom-xy), [hyperparameter](#hyperparameter), [X/y aktual](#xy-aktual), [arsitektur](#arsitektur), [optimizer LLRD](#optimizer-llrd), [training](#training), [evaluasi](#evaluasi), [unduh hasil](#unduh).
'''


def build():
    raw = subprocess.check_output(['git', 'show', FORMAT_COMMIT+':'+FORMAT_PATH], cwd=ROOT)
    notebook = copy.deepcopy(json.loads(raw))
    old_hash = notebook['metadata']['aes_standalone']['code_sha256']
    for cell in notebook['cells']:
        role = cell['metadata'].pop('aes39_role', '')
        if role:
            cell['metadata']['aes40_role'] = role
        source = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
        source = source.replace(old_hash, PLACEHOLDER).replace(Path(FORMAT_PATH).stem, NAME)
        source = source.replace('E39', 'E40').replace('bigru128_e30_mlp', 'llrd095_e30_mlp')
        if role == 'hyperparameters':
            source = source.replace('E30 MLP; hanya hidden BiGRU diperbesar.', 'E30 MLP; hanya pembagian LR encoder per layer berubah.')
            source = source.replace("ENCODER_LR = 5e-6             # Tetap E30; bukan LR E37 7e-6.",
                                    "ENCODER_LR = 5e-6             # LR puncak layer RoBERTa paling atas.\nENCODER_LAYER_DECAY = 0.95    # PERUBAHAN E40: setiap turun satu layer, LR dikali0.95.")
            source = source.replace('BIGRU_HIDDEN_PER_DIRECTION = 128  # PERUBAHAN: E30=64/arah; E40=128/arah.',
                                    'BIGRU_HIDDEN_PER_DIRECTION = 64   # Kembali ke E30: keluaran dua arah128fitur/token.')
            source = source.replace('tetap E30, tidak ikut diperbesar.', 'tetap64 seperti E30.')
        if role == 'configuration_defaults':
            source = source.replace("'s3_encoder_lr': ENCODER_LR,", "'s3_encoder_lr': ENCODER_LR, 'encoder_layer_decay': ENCODER_LAYER_DECAY,")
            anchor = "    if not 0 <= result['s3_mlp_dropout'] < 1:"
            assert source.count(anchor) == 1
            source = source.replace(anchor, "    decay = result['encoder_layer_decay']\n    if isinstance(decay, bool) or not math.isfinite(decay) or not 0 < decay <= 1:\n        raise ValueError('encoder_layer_decay harus berada pada rentang (0,1].')\n"+anchor)
        if role == 'configuration_apply':
            source = source.replace('encoder_lr=ENCODER_LR, s3_encoder_lr=ENCODER_LR,',
                                    'encoder_lr=ENCODER_LR, s3_encoder_lr=ENCODER_LR, encoder_layer_decay=ENCODER_LAYER_DECAY,')
            source = source.replace("lesson_version='e36_native_s3_encoder_llrd095_e30_mlp_v1'", "lesson_version='e40_native_s3_llrd095_e30_mlp_v1'")
            source = source.replace("'Encoder RoBERTa S3'", "'Layer RoBERTa paling atas (puncak)'")
            source += "\nprint('Faktor LR per layer:', cfg['encoder_layer_decay'], '| tabel kelompok aktual ditampilkan saat fit dan disimpan pada Excel05.')\n"
        if role in ('definition:SingleScoreModel', 'definition:AnswerBiGRUAttention', 'definition:AnswerBiGRUCoAttentionPool'):
            source = source.replace('sequence_hidden=128', 'sequence_hidden=64').replace('hidden_size=128', 'hidden_size=64')
            source = source.replace('BiGRU128/arah menghasilkan256fitur', 'BiGRU64/arah menghasilkan128fitur')
            source = source.replace('respons256 ↔ referensi256, lalu attention pooling256→64→1', 'respons128 ↔ referensi128, lalu attention pooling128→64→1')
        if role == 'definition:optimizer_groups':
            source = OPTIMIZER
        if role == 'definition:fit':
            anchor = "    learning_rates = {group['name']: group['lr'] for group in groups}\n"
            assert source.count(anchor) == 1
            source = source.replace(anchor, anchor+"    learning_rate_table = [{'Kelompok': group['name'], 'LR Puncak': group['lr'],\n                            'Jumlah Parameter': sum(p.numel() for p in group['params'])} for group in groups]\n    print('LLRD: LR puncak per kelompok, sebelum warmup + linear scheduler.', flush=True)\n    print(pd.DataFrame(learning_rate_table).to_string(index=False), flush=True)\n")
            source = source.replace("'optimizer_learning_rates': learning_rates,", "'optimizer_learning_rates': learning_rates, 'optimizer_lr_table': learning_rate_table,\n            'encoder_layer_decay': cfg['encoder_layer_decay'], 'encoder_num_layers': len(model.roberta.encoder.layer),")
        if role == 'definition:load_model':
            source = source.replace("'s3_encoder_lr',", "'s3_encoder_lr', 'encoder_layer_decay',")
        if role == 'definition:export_training_event':
            source = source.replace("'Prediksi_validation': predictions,", "'Learning_rate_groups': pd.DataFrame(training['optimizer_lr_table']),\n                                                           'Prediksi_validation': predictions,")
            source = source.replace("'Bukti_parameter': pd.DataFrame([result['parameter_samples_changed']])", "'Bukti_parameter': pd.DataFrame([result['parameter_samples_changed']]),\n                                            'Learning_rate_groups': pd.DataFrame(result['optimizer_lr_table'])")
        if role == 'definition:training_report':
            source = source.replace("settings={'pooling_attention_hidden'", "settings={'encoder_layer_decay':'Faktor LR per layer RoBERTa','pooling_attention_hidden'")
            source = source.replace("'s3_encoder_lr':'Learning rate encoder RoBERTa S3'", "'s3_encoder_lr':'Learning rate encoder RoBERTa S3'")
            anchor = '    return sheets\n'
            assert source.count(anchor) == 1
            source = source.replace(anchor, "    lr_rows = []\n    for file in sorted(source.files):\n        match = re.fullmatch(r'09_S3_fold_(\\d+)\\.xlsx', file)\n        if not match and file != '10_S3_refit.xlsx':\n            continue\n        if source.has(file, 'Learning_rate_groups'):\n            for row in source.records(file, 'Learning_rate_groups'):\n                lr_rows.append(['Validasi silang' if match else 'Latihan final',\n                                int(match[1]) if match else None, row['Kelompok'],\n                                row['LR Puncak'], row['Jumlah Parameter']])\n    if lr_rows:\n        sheets['Learning Rate per Layer'] = Table(\n            ['Tahap','Fold','Kelompok','LR Puncak','Jumlah Parameter'], lr_rows)\n"+anchor)
            source = source.replace("['Epoch','Satu putaran seluruh data latih pada fold/tahap itu.']", "['Epoch','Satu putaran seluruh data latih pada fold/tahap itu.'],\n        ['Learning Rate per Layer','LR Puncak adalah LR sebelum warmup dan linear scheduler; bukan LR konstan sepanjang training. Embeddings lebih kecil daripada layer0; layer atas memakai5e-6.']")
        if role == 'definition:answer_vectors_report':
            source = source.replace('BiGRU256', 'BiGRU128').replace('referensi256', 'referensi128').replace('matching256', 'matching128')
        if cell['cell_type'] == 'markdown':
            source = source.replace('BiGRU128/arah', 'BiGRU64/arah').replace('respons256', 'respons128').replace('referensi256', 'referensi128')
            source = source.replace('matching1280→256', 'matching640→128').replace('pooling256', 'pooling128').replace('/√256', '/√128')
            source = source.replace('ringkasan256', 'ringkasan128').replace('Linear256', 'Linear128').replace('256→64→1', '128→64→1')
            if '## 5 · Hyperparameter' in source:
                source = '<a id="hyperparameter"></a>\n## 5 · Hyperparameter E40\n\nSatu perlakuan: ENCODER_LAYER_DECAY=0.95. ENCODER_LR=5e-6 adalah LR puncak layer paling atas. Untuk layer indeks i dari L layer: LR=5e-6×0.95**(L-1-i). Embeddings memakai5e-6×0.95**L. BiGRU kembali64perarah,attention64,dropoutMLP0.1,semua LR di luar encoder1e-4. Jumlah layer dibaca dari model aktual.\n'
            if '### `SingleScoreModel`' in source:
                source = '### `SingleScoreModel`\n\nArsitektur E30 MLP: BiGRU64perarah menghasilkan128fitur/token,proyeksi akhir RoBERTa768,concat1024,MLP256→1. Perlakuan E40 berada pada optimizer, sehingga bentuk tensor dan jumlah parameter model kembali sama dengan E30.\n'
            if '### `AnswerBiGRUAttention`' in source:
                source = '### `AnswerBiGRUAttention`\n\nBiGRU64perarah menghasilkan128fitur/token. Additive attention128→64→1 merangkum token dengan bobot yang dipelajari. Ukuran dan operasi mengikuti E30 MLP.\n'
            if '## 22 · Optimizer' in source:
                source = '<a id="optimizer-llrd"></a>\n## 22 · Optimizer LLRD, loss MSE dan checkpoint\n\nLearning rate berbeda menurut kedalaman RoBERTa. Layer paling atas5e-6; setiap turun satu layer dikali0.95; embeddings satu tingkat di bawah layer0. Semua layer tetap dilatih. BiGRU/co-attention/pooling/norm/proyeksi/MLP tetap1e-4. AdamW,weight decay0.01,warmup10% dan scheduler linear tetap E30.\n\nContoh12layer: layer11=5e-6,layer8=4.286875e-6,layer5=3.675459e-6,layer2=3.151247e-6,layer0=2.844000e-6,embeddings=2.701800e-6. Ini LR puncak; scheduler mengalikan tiap kelompok dengan faktor waktu yang sama. Penurunan per layer berbeda dari penurunan sepanjang langkah training.\n'
            if '### `optimizer_groups`' in source:
                source = '### `encoder_lr_schedule` dan `optimizer_groups`\n\nFungsi pertama menghitung LR menurut kedalaman. Fungsi kedua menempatkan setiap parameter trainable pada tepat satu kelompok dan menolak parameter yang belum dikenali. Jumlah layer diambil dari model yang dimuat. Tabel LR puncak serta jumlah parameter dicetak setiap fit dan disimpan di Excel05→Learning Rate per Layer.\n'
            if '## Rujukan dan batas kesimpulan' in source:
                source = source.split('**Dasar eksperimen kapasitas:**')[0]
                source += '\n**Dasar perlakuan LLRD:** Sun, Qiu, Xu & Huang (2019), [How to Fine-Tune BERT for Text Classification?](https://arxiv.org/abs/1905.05583),Section5.3.4. [README repositori penulis](https://github.com/xuyige/BERT4doc-Classification) menyebut layer-wise decreasing learning rate. Paper memberi dasar metode pada klasifikasi teks dengan BERT; adaptasi ke RoBERTa fusion AES dan faktor0.95 di E40 adalah hipotesis eksperimen, bukan optimum AES yang sudah dibuktikan.\n\nE40 tidak menguji pembesaran BiGRU. Arsitektur kembali64perarah seperti E30; hanya pembagian LR encoder menurut kedalaman yang berubah. Jangan memilih konfigurasi atau epoch dari hasil test.\n'
        cell['source'] = source
    notebook['cells'][0]['source'] = INTRO
    fingerprint = hashlib.sha256('\n'.join(c['source'] for c in notebook['cells'] if c['cell_type'] == 'code').encode()).hexdigest()
    for index, cell in enumerate(notebook['cells']):
        cell['source'] = cell['source'].replace(PLACEHOLDER, fingerprint)
        cell['id'] = f'e40-{index:03d}-'+hashlib.sha256(cell['source'].encode()).hexdigest()[:8]
        if cell['cell_type'] == 'code':
            ast.parse(cell['source'])
            cell['outputs'] = []
            cell['execution_count'] = None
    notebook['metadata']['colab']['name'] = NAME+'.ipynb'
    notebook['metadata']['aes_standalone']['code_sha256'] = fingerprint
    meta = notebook['metadata']['aes_experiment']
    for key in ('s1_encoder_lr', 's2_mlp_lr'):
        meta.pop(key, None)
    meta.update(name='E40', s3='E30 MLP S3; layer-wise LR decay0.95; BiGRU64 per direction',
                s3_sequence_hidden=64, sequence_output_dimensions=128, pooling_attention_hidden=64,
                input_columns=['soal', 'jawaban_referensi', 'response_text'],
                encoder_layer_decay=.95, s3_encoder_lr=5e-6, encoder_lr=5e-6,
                format_commit=FORMAT_COMMIT, format_path=FORMAT_PATH, format_sha256=hashlib.sha256(raw).hexdigest(),
                treatment='E30 MLP; encoder LR top5e-6 multiplied by0.95 per lower layer; embeddings below layer0')
    output = json.dumps(notebook, ensure_ascii=False, indent=1)+'\n'
    for directory in ('notebooks/experiments', 'aes-learning/notebooks/experiments'):
        (ROOT/directory/(NAME+'.ipynb')).write_text(output)
    print(json.dumps({'name': NAME, 'cells': len(notebook['cells']), 'code_sha256': fingerprint,
                      'hidden_per_direction': 64, 'encoder_layer_decay': .95}))
    return notebook


if __name__ == '__main__':
    build()
