"""Native E45 from pinned E43: mix all twelve Transformer layers, original MSE."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PIN = '7241a21dfec37c942948b754da3960fba4c7731c'
BASE = 'notebooks/experiments/AES_300_E43_E42_S3_MaskedMeanPooling_Native_Learnable_GPU.ipynb'
NAME = 'AES_300_E45_E43_S3_ScalarMix12_Native_Learnable_GPU'
PLACEHOLDER = '__NOTEBOOK_CODE_SHA256__'

OOF_EXPORT = r'''    # Prediksi VALIDASI development, terpisah dari test; tidak mengubah training.
    questions = {r['row_id']:r.get('soal') for r in source.records('02_dataset_dan_cleaning.xlsx','Sebelum_sesudah')}
    oof_rows, seen = [], set()
    for file in sorted(source.files):
        match = re.fullmatch(r'09_S3_fold_(\d+)\.xlsx', file)
        if not match:
            continue
        for row in source.records(file, 'Prediksi_validation'):
            identifier = row['row_id']
            if identifier in seen or identifier in predicted:
                raise ValueError('ID OOF duplikat atau bercampur dengan data test.')
            seen.add(identifier)
            error = row['prediksi_mentah']-row['skor_guru']
            oof_rows.append([int(match[1]), friendly_value('row_id',identifier), questions.get(identifier),
                row['skor_guru'], row['prediksi_mentah'], row['kategori_QWK_AC2'], error, abs(error), error**2])
    if oof_rows:
        sheets['Prediksi Validasi OOF'] = Table(
            ['Fold','ID Jawaban','Soal','Nilai Guru','Prediksi S3','Pembulatan S3',
             'Selisih S3','Selisih Absolut','Selisih Kuadrat'], oof_rows)
'''

INTRO = '''# E45 · E43 + Scalar Mix seluruh 12 layer · S3 Native Learnable

**Satu perlakuan dari E43: campuran 4 layer terakhir diperluas menjadi seluruh 12 layer Transformer RoBERTa.** Loss tetap MSE. E45 langsung memakai E43 sebagai dasar; E44 menjadi eksperimen terpisah untuk MSE + ListNet.

RoBERTa → Scalar Mix layer 1–12 + gamma → BiGRU 64 per arah + co-attention → masked mean → proyeksi 768; gabung proyeksi BGE 256 → concat 1024 → MLP → satu nilai. Tidak menambah encoder atau forward RoBERTa baru. Delapan logit campuran ditambahkan, sehingga total 12 logit + satu gamma; bagian model lainnya tetap E43.

| Pengaturan | Nilai |
|---|---|
| X / y | `soal`, `jawaban_referensi`, `response_text` / `rater` 0–100 |
| Scalar Mix | Layer Transformer 1–12; embedding awal tidak dicampur langsung |
| Bobot awal | 12 logit 0 → masing-masing 1/12 ≈ 0,083333; gamma awal 1 |
| LR encoder / komponen lain | 5e-6 / 1e-4 |
| BiGRU / dropout MLP / loss | 64 per arah / 0,1 / MSE target 0–1 |
| Protokol | S3 saja, satu 5-fold, maksimal 25 epoch, patience 5 |
| Refit / test | Ceil median best epoch berdasarkan RMSE validasi; satu refit dan satu test |

**Embedding awal:** keluaran embedding pada `hidden_states[0]` tidak masuk penjumlahan Scalar Mix. Parameter embedding tetap dilatih melalui encoder. Model memeriksa bahwa jumlah layer campuran sama dengan jumlah layer Transformer aktual; checkpoint penelitian ini mempunyai 12 layer.

Nama checkpoint, kolom input dan hyperparameter terlihat di section awal. `SCALAR_MIX_LAYERS = 12` ada pada section 5; `ScalarMix` dan alur fusion ada pada section arsitektur. Python biasa per section, tanpa magic, worker tersembunyi, cosine, S1/S2, sweep seed atau pelatihan ulang kontrol otomatis. Gamma, optimizer AdamW, decay 0,01, warmup 10%, scheduler linear, clipping 1, microbatch 1 dan batch efektif 8 tetap E43.

**Dasar metode:** [Peters dkk. (2018), ELMo](https://aclanthology.org/N18-1202/) mendukung campuran layer berbobot. [Implementasi resmi AllenNLP](https://github.com/allenai/allennlp/blob/main/allennlp/modules/token_embedders/pretrained_transformer_embedder.py) menyediakan Scalar Mix seluruh layer Transformer dengan embedding awal dikecualikan. Penerapan pada fusion AES Indonesia ini tetap hipotesis; sumber tersebut tidak membuktikan 12 layer akan mengalahkan E43.

**Output:** lima Excel dalam satu ZIP. Excel 05 → **Bobot Scalar Mix** menyimpan 12 bobot dan gamma pada checkpoint terpilih tiap fold/refit: total 72 baris. **MAE dan Toleransi CV** tetap tersedia. Excel 04 → **Prediksi Validasi OOF** menambah pasangan nilai guru/prediksi per jawaban development, terpisah dari test. Excel 02 tetap menyimpan norma campuran, vektor fusion, bobot mean pooling dan co-attention.

Pilih **GPU T4 → Run all → unggah `fix_dataset_300.xlsx`**. Folder/hash/checkpoint E45 terpisah dari E43 dan E44. Belum ada hasil penelitian E45. Bandingkan dari CV terhadap acuan E43 (mean RMSE 5,8876); test historis yang sudah dilihat berulang kali tetap bersifat eksploratif. E44 dan E45 dijalankan sebagai dua kandidat terpisah.
'''


def build():
    raw = subprocess.check_output(['git','show',PIN+':'+BASE],cwd=ROOT)
    n = copy.deepcopy(json.loads(raw))
    old_hash = n['metadata']['aes_standalone']['code_sha256']
    for cell in n['cells']:
        role = cell['metadata'].pop('aes43_role','')
        if role:
            cell['metadata']['aes45_role'] = role
        s = cell['source'] if isinstance(cell['source'],str) else ''.join(cell['source'])
        s = s.replace(old_hash,PLACEHOLDER).replace(Path(BASE).stem,'__E45_NAME__')
        s = s.replace('E43','E45').replace('__E45_NAME__',NAME)
        s = s.replace('SCALAR_MIX_LAST_LAYERS','SCALAR_MIX_LAYERS')
        if role == 'hyperparameters':
            s = s.replace('SCALAR_MIX_LAYERS = 4','SCALAR_MIX_LAYERS = 12')
            s = s.replace('Scalar Mix empat layer terakhir.','Scalar Mix seluruh 12 layer Transformer.')
            s = s.replace('Empat layer Transformer terakhir, tidak termasuk embeddings.',
                          'Seluruh 12 layer Transformer; embedding awal tidak dicampur langsung.')
            s = s.replace('empat logit','12 logit')
        if role == 'configuration_apply':
            s = s.replace('e43_native_masked_mean_v1','e45_native_scalar_mix_all12_v1')
            s = s.replace("'layer terakhir | LR:'", "'layer Transformer (seluruhnya) | LR:'")
        if role == 'definition:SingleScoreModel':
            s = s.replace('number_of_layers=4','number_of_layers=12').replace('scalar_mix_layers=4','scalar_mix_layers=12')
            s = s.replace('# Empat logit nol memberi bobot0.25; gamma awal1 mempertahankan keluaran awal E41.',
                          '# Dua belas logit nol memberi bobot 1/12; gamma awal 1 tetap E43.')
            s = s.replace('# Penjumlahan bertahap menghindari stack tambahan [4,batch,token,768].',
                          '# Penjumlahan bertahap menghindari stack tambahan [12,batch,token,768].')
            s = s.replace('if roberta.config.num_hidden_layers < scalar_mix_layers:',
                          'if roberta.config.num_hidden_layers != scalar_mix_layers:')
            s = s.replace('RoBERTa mempunyai lebih sedikit layer daripada scalar_mix_layers.',
                          'E45 memerlukan seluruh layer Transformer: scalar_mix_layers harus sama dengan num_hidden_layers model.')
        if role == 'architecture_export':
            s = s.replace('Scalar Mix4 + gamma','Scalar Mix12 seluruh layer + gamma')
        if role == 'definition:answer_vectors_report':
            s = s.replace('Empat layer terakhir RoBERTa dicampur dengan bobot global yang dipelajari,',
                          'Seluruh 12 layer Transformer RoBERTa dicampur dengan bobot global yang dipelajari dan dikalikan gamma,')
        if role == 'definition:training_report':
            s = s.replace('Empat bobot global pada checkpoint terpilih tiap fold/refit;',
                          'Dua belas bobot global pada checkpoint terpilih tiap fold/refit;')
        if role == 'definition:evaluation_report':
            assert s.count('    return sheets\n') == 1
            s = s.replace('    return sheets\n',OOF_EXPORT+'    return sheets\n')
            s = s.replace("['Ringkasan Validasi','", "['Prediksi Validasi OOF','Setiap jawaban development diprediksi oleh model fold yang tidak melatih jawaban tersebut; terpisah dari test.'],\n        ['Ringkasan Validasi','")
        if cell['cell_type'] == 'markdown':
            s = s.replace('empat layer','seluruh 12 layer Transformer').replace('Empat layer','Seluruh 12 layer Transformer')
            s = s.replace('empat keluaran Transformer terakhir','seluruh 12 keluaran Transformer')
            s = s.replace('Empat logit','Dua belas logit').replace('empat logit','dua belas logit')
            s = s.replace('SCALAR_MIX_LAYERS=4','SCALAR_MIX_LAYERS=12')
            s = s.replace('seragam0.25','seragam 1/12 ≈ 0,083333')
            if '## Rujukan dan batas kesimpulan' in s:
                s = s.replace('**Perlakuan E45:** Taghipour','**Pooling acuan E43:** Taghipour')
                s += '\n**Perlakuan E45:** memperluas campuran empat layer terakhir menjadi seluruh 12 layer Transformer. [Peters dkk. (2018), ELMo](https://aclanthology.org/N18-1202/) memberi dasar campuran layer berbobot. [AllenNLP PretrainedTransformerEmbedder](https://github.com/allenai/allennlp/blob/main/allennlp/modules/token_embedders/pretrained_transformer_embedder.py) memakai `ScalarMix(config.num_hidden_layers)` dan `hidden_states[1:]` untuk seluruh layer Transformer tanpa embedding awal. Ini landasan metode/implementasi, bukan bukti 12 layer optimal untuk AES fusion Indonesia. Loss tetap MSE E43.\n'
        cell['source'] = s
    n['cells'][0]['source'] = INTRO
    fp = hashlib.sha256('\n'.join(c['source'] for c in n['cells'] if c['cell_type']=='code').encode()).hexdigest()
    for i,c in enumerate(n['cells']):
        c['source'] = c['source'].replace(PLACEHOLDER,fp)
        c['id'] = f'e45-{i:03d}-'+hashlib.sha256(c['source'].encode()).hexdigest()[:8]
        if c['cell_type']=='code':
            ast.parse(c['source']);c['outputs']=[];c['execution_count']=None
    n['metadata']['colab']['name'] = NAME+'.ipynb'
    n['metadata']['aes_standalone']['code_sha256'] = fp
    m = n['metadata']['aes_experiment']
    m.pop('removed_parameters_vs_e42',None)
    m.update(name='E45',s3='E43 with Scalar Mix all12 Transformer layers; original MSE',
        scalar_mix_layers=12,scalar_mix_all_layers=True,embedding_layer_included=False,
        scalar_mix_trainable_parameters=13,added_parameters_vs_e43=8,
        pooling_parameters_removed_vs_attention=8320,net_parameters_vs_e42=-8312,net_parameters_vs_e30=-8307,
        format_commit=PIN,format_path=BASE,format_sha256=hashlib.sha256(raw).hexdigest(),
        loss='MSE',ranking=False,
        treatment='Scalar Mix last4 to all12 Transformer layers; gamma, model components, MSE and protocol retained')
    output = json.dumps(n,ensure_ascii=False,indent=1)+'\n'
    for folder in ['notebooks/experiments','aes-learning/notebooks/experiments']:
        (ROOT/folder/(NAME+'.ipynb')).write_text(output)
    print(json.dumps({'name':NAME,'cells':len(n['cells']),'code_sha256':fp}))
    return n


if __name__ == '__main__':
    build()
