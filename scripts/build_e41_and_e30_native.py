"""Build E30 MLP native reference and E41 four-layer Scalar Mix, S3 only."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FORMAT_COMMIT = '5f1c993804560d12c06d7d11d4540e6d11307cf2'
FORMAT_PATH = 'notebooks/experiments/AES_300_E39_E30_S3_BiGRU128_Native_Learnable_GPU.ipynb'
BASE_NAME = 'AES_300_E30_S3_MLP_Native_Learnable_GPU'
MIX_NAME = 'AES_300_E41_E30_S3_ScalarMix4_Native_Learnable_GPU'
PLACEHOLDER = '__NOTEBOOK_CODE_SHA256__'

SCALAR_MIX = '''class ScalarMix(nn.Module):
    """Campurkan k layer Transformer terakhir dengan bobot global yang dipelajari."""
    def __init__(self, number_of_layers=4):
        super().__init__()
        if type(number_of_layers) is not int or number_of_layers < 1:
            raise ValueError('Jumlah layer Scalar Mix harus bilangan bulat positif.')
        self.number_of_layers = number_of_layers
        # Empat logit nol menghasilkan empat bobot awal0.25; tidak memakai skala gamma.
        self.logits = nn.Parameter(torch.zeros(number_of_layers))

    def weights(self):
        return torch.softmax(self.logits, dim=0)

    def forward(self, hidden_states):
        # hidden_states[0] adalah embeddings; hanya keluaran layer Transformer dipilih.
        if hidden_states is None or len(hidden_states)-1 < self.number_of_layers:
            raise ValueError('Encoder tidak menyediakan cukup layer Transformer untuk Scalar Mix.')
        selected = hidden_states[-self.number_of_layers:]
        if any(value.shape != selected[0].shape for value in selected):
            raise ValueError('Semua tensor layer Scalar Mix harus mempunyai shape yang sama.')
        weights = self.weights().to(dtype=selected[0].dtype)
        # Penjumlahan bertahap menghindari stack tambahan [4,batch,token,768].
        mixed = torch.zeros_like(selected[0])
        for weight, value in zip(weights, selected):
            mixed = mixed + weight * value
        return mixed

    def report(self, encoder_layers):
        weights = self.weights().detach().float().cpu().tolist()
        logits = self.logits.detach().float().cpu().tolist()
        start = encoder_layers-self.number_of_layers
        return [{'Indeks Layer RoBERTa': start+i, 'Layer ke-': start+i+1,
                 'Logit': logits[i], 'Bobot Softmax': weights[i],
                 'Bobot Awal': 1/self.number_of_layers}
                for i in range(self.number_of_layers)]
'''


def intro(mix):
    if mix:
        opening = '''# E41 · E30 MLP S3 · Scalar Mix empat layer RoBERTa · Native Learnable

**Satu perlakuan: campuran empat layer Transformer terakhir dari satu RoBERTa yang sama.** Empat logit global dipelajari lewat loss nilai guru. Softmax menghasilkan bobot positif berjumlah1; logit awal nol memberi bobot0.25masing-masing. Tidak ada cabang encoder tambahan, skala gamma atau normalisasi tambahan. Setiap posisi token dicampur menjadi768dimensi sebelum BiGRU dan co-attention. Empat layer terakhir tidak mencakup embeddings.

Empat parameter campuran memakai LR1e-4. Seluruh layer RoBERTa tetap dilatih dengan LR seragam5e-6. LLRD E40 tidak dibawa. Bobot global sama untuk semua jawaban pada checkpoint yang sama; tidak dihitung ulang sebagai attention khusus setiap siswa. Pada encoder12layer, yang dipilih adalah indeks8–11 atau layer ke9–12.
'''
    else:
        opening = '''# E30 acuan · MLP S3 · Native Learnable

**Versi belajar E30 MLP, dengan kode Python biasa per section.** Resep S3 historis dipertahankan: token memakai keluaran layer RoBERTa terakhir saja. Seluruh layer RoBERTa tetap dilatih dengan LR seragam5e-6. Notebook ini menjalankan satu resep acuan secara mandiri; tidak otomatis memanggil eksperimen lain.

Ini adalah format native untuk menjalankan ulang resep E30, bukan salinan prediksi/checkpoint historis. Hasil run baru dapat berbeda karena GPU,precision atau versi dependency; RMSE historis5.5247tidak dijamin terulang persis. Source E30 lama tetap tersedia.
'''
    return opening+'''
Arsitektur: RoBERTa768 → BiGRU64perarah → respons128; referensi768→128; co-attention respons–referensi → matching640→128 → attention pooling128→64→1 → proyeksi768+LayerNorm. BGE-M3 frozen1024→proyeksi256+LayerNorm; concat1024→MLP256→GELU→Dropout0.1→1. MLP adalah prediktor akhir. LR BiGRU/co-attention/pooling/norm/proyeksi/MLP tetap1e-4. Hidden128perarah E39 tidak dibawa.

**S3 saja,5fold CV sekali,maksimal25epoch/fold,patience5,1refit+1test.** TRAIN_SEED42adalah satu konstanta; pola seed per fold mengikuti E30. Tidak ada loop multi-seed atau pengulangan kontrol otomatis. Refit=ceil median best epoch5fold. Test tidak memilih LR/epoch; test lama yang telah berulang kali dilihat tetap eksploratif.

Tiga input X=`soal,jawaban_referensi,response_text`; target y=`rater`0–100tidak masuk encoder. Nama checkpoint,kolom X/y,hyperparameter,arsitektur,optimizer,training,evaluasi dan unduhan berada langsung dalam kernel Python. Tanpa cell magic,worker tersembunyi,clone repo atau Drive.

**T4 GPU → Run all → upload fix_dataset_300.xlsx.** Hasil lima Excel standar,folder dan ZIP terpisah untuk setiap notebook. Cache/checkpoint hanya digunakan bila hash data/config/kode sesuai. Unduh ZIP sebelum runtime dihapus.

Peta belajar: [nama model](#nama-model), [kolom X/y](#kolom-xy), [hyperparameter](#hyperparameter), [X/y aktual](#xy-aktual), [arsitektur](#arsitektur), [training](#training), [evaluasi](#evaluasi), [unduh hasil](#unduh).
'''+('''
Excel05→**Bobot Scalar Mix** menyimpan empat bobot pada checkpoint terpilih setiap fold dan model refit. Bobot tersebut diekspor setelah pemulihan epoch terbaik, bukan dari epoch terakhir yang mungkin lebih buruk. Tidak ada jaminan E41 mengalahkan E30.
''' if mix else '')


def build_one(raw, mix):
    n = copy.deepcopy(json.loads(raw))
    old_hash = n['metadata']['aes_standalone']['code_sha256']
    name, label, role_key = (MIX_NAME, 'E41', 'aes41_role') if mix else (BASE_NAME, 'E30', 'aes30native_role')
    for cell in n['cells']:
        role = cell['metadata'].pop('aes39_role', '')
        if role:
            cell['metadata'][role_key] = role
        s = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
        s = s.replace(old_hash, PLACEHOLDER).replace(Path(FORMAT_PATH).stem, name)
        s = s.replace('E39', label).replace('bigru128_e30_mlp', 'scalar_mix4_e30' if mix else 'e30_mlp_reference')
        if role == 'hyperparameters':
            s = s.replace('E30 MLP; hanya hidden BiGRU diperbesar.',
                          'E30 MLP; Scalar Mix empat layer terakhir.' if mix else 'E30 MLP asli; format native.')
            s = s.replace(f'BIGRU_HIDDEN_PER_DIRECTION = 128  # PERUBAHAN: E30=64/arah; {label}=128/arah.',
                          'BIGRU_HIDDEN_PER_DIRECTION = 64   # E30: dua arah menghasilkan128fitur/token.')
            if mix:
                s += '\nSCALAR_MIX_LAST_LAYERS = 4      # Empat layer Transformer terakhir, tidak termasuk embeddings.\nSCALAR_MIX_LR = 1e-4            # LR untuk empat logit campuran yang dipelajari.\n'
        if role == 'configuration_apply':
            s = s.replace("lesson_version='e36_native_s3_encoder_"+('scalar_mix4_e30' if mix else 'e30_mlp_reference')+"_v1'",
                          "lesson_version='"+('e41_native_scalar_mix4_v1' if mix else 'e30_native_mlp_reference_v1')+"'")
            if mix:
                s = s.replace('head_lr=1e-4, s3_mlp_lr=S3_MLP_LR,',
                              'head_lr=1e-4, s3_mlp_lr=S3_MLP_LR, scalar_mix_layers=SCALAR_MIX_LAST_LAYERS, scalar_mix_lr=SCALAR_MIX_LR,')
                s += "\nprint('Scalar Mix:', cfg['scalar_mix_layers'], 'layer terakhir | LR:', cfg['scalar_mix_lr'], '| bobot awal seragam')\n"
        if role in ('definition:SingleScoreModel', 'definition:AnswerBiGRUAttention', 'definition:AnswerBiGRUCoAttentionPool'):
            s = s.replace('sequence_hidden=128', 'sequence_hidden=64').replace('hidden_size=128', 'hidden_size=64')
            s = s.replace('BiGRU128/arah menghasilkan256fitur', 'BiGRU64/arah menghasilkan128fitur')
            s = s.replace('respons256 ↔ referensi256, lalu attention pooling256→64→1', 'respons128 ↔ referensi128, lalu attention pooling128→64→1')
        if mix:
            if role == 'configuration_defaults':
                s = s.replace("'s3_mlp_lr': S3_MLP_LR,", "'scalar_mix_layers': SCALAR_MIX_LAST_LAYERS, 'scalar_mix_lr': SCALAR_MIX_LR,\n    's3_mlp_lr': S3_MLP_LR,")
                s = s.replace("    if result['s3_fusion'] != 'concat':", "    if type(result['scalar_mix_layers']) is not int or result['scalar_mix_layers'] < 1:\n        raise ValueError('scalar_mix_layers harus bilangan bulat positif.')\n    if not math.isfinite(result['scalar_mix_lr']) or result['scalar_mix_lr'] <= 0:\n        raise ValueError('scalar_mix_lr harus positif dan finite.')\n    if result['s3_fusion'] != 'concat':")
            if role == 'definition:SingleScoreModel':
                s = s.replace('pooling_attention_hidden=64):', 'pooling_attention_hidden=64, scalar_mix_layers=4):')
                anchor = '        self.sequence_pool = AnswerBiGRUCoAttentionPool(sequence_hidden, pooling_attention_hidden)\n'
                assert s.count(anchor) == 1
                s = s.replace(anchor, anchor+"        if roberta.config.num_hidden_layers < scalar_mix_layers:\n            raise ValueError('RoBERTa mempunyai lebih sedikit layer daripada scalar_mix_layers.')\n        # Inisialisasi nol tidak mengubah RNG layer E30 yang sudah dibentuk.\n        self.scalar_mix = ScalarMix(scalar_mix_layers)\n")
                s = s.replace('        hidden = self.roberta(**windows).last_hidden_state\n        cls = hidden[:, 0, :]',
                              '        encoded = self.roberta(**windows, output_hidden_states=True, return_dict=True)\n        hidden = self.scalar_mix(encoded.hidden_states)\n        cls = encoded.last_hidden_state[:, 0, :]  # CLS referensi inspeksi tetap layer terakhir.')
                s = SCALAR_MIX+'\n\n'+s
            if role == 'definition:make_model':
                s = s.replace("pooling_attention_hidden=cfg['pooling_attention_hidden'])", "pooling_attention_hidden=cfg['pooling_attention_hidden'],\n                            scalar_mix_layers=cfg['scalar_mix_layers'])")
            if role == 'definition:optimizer_groups':
                s = s.replace('Empat kelompok optimizer S3;', 'Lima kelompok optimizer S3;')
                s = s.replace("'pooling_norm': []}", "'pooling_norm': [], 'scalar_mix': []}")
                s = s.replace("        elif name.startswith('projection.'):", "        elif name.startswith('scalar_mix.'):\n            group = 'scalar_mix'\n        elif name.startswith('projection.'):")
                s = s.replace("'pooling_norm': cfg['s3_layers_lr']}", "'pooling_norm': cfg['s3_layers_lr'], 'scalar_mix': cfg['scalar_mix_lr']}")
            if role == 'definition:fit':
                s = s.replace('    return model, engine, info\n', "    # Bobot diambil SESUDAH best_state dipulihkan; sesuai model yang dievaluasi.\n    info['scalar_mix_weights'] = model.scalar_mix.report(model.roberta.config.num_hidden_layers)\n    info['scalar_mix_layers'] = cfg['scalar_mix_layers']\n    print('Bobot Scalar Mix pada checkpoint terpilih:', flush=True)\n    print(pd.DataFrame(info['scalar_mix_weights']).to_string(index=False), flush=True)\n    return model, engine, info\n")
            if role == 'definition:load_model':
                s = s.replace("'s3_encoder_lr',", "'s3_encoder_lr', 'scalar_mix_layers', 'scalar_mix_lr',")
                s = s.replace("if payload['config'][key] != cfg[key]:", "if payload['config'].get(key) != cfg[key]:")
            if role == 'definition:export_training_event':
                s = s.replace("'Prediksi_validation': predictions,", "'Bobot_scalar_mix': pd.DataFrame(training['scalar_mix_weights']),\n                                                           'Prediksi_validation': predictions,")
                s = s.replace("'Bukti_parameter': pd.DataFrame([result['parameter_samples_changed']])", "'Bukti_parameter': pd.DataFrame([result['parameter_samples_changed']]),\n                                            'Bobot_scalar_mix': pd.DataFrame(result['scalar_mix_weights'])")
            if role == 'definition:training_report':
                s = s.replace("settings={'pooling_attention_hidden'", "settings={'scalar_mix_layers':'Jumlah layer Scalar Mix','scalar_mix_lr':'LR Scalar Mix','pooling_attention_hidden'")
                anchor = '    return sheets\n'
                assert s.count(anchor) == 1
                s = s.replace(anchor, "    mix_rows = []\n    for file in sorted(source.files):\n        match = re.fullmatch(r'09_S3_fold_(\\d+)\\.xlsx', file)\n        if not match and file != '10_S3_refit.xlsx':\n            continue\n        if source.has(file, 'Bobot_scalar_mix'):\n            for row in source.records(file, 'Bobot_scalar_mix'):\n                mix_rows.append(['Validasi silang' if match else 'Latihan final',\n                                 int(match[1]) if match else None, row['Indeks Layer RoBERTa'],\n                                 row['Layer ke-'], row['Logit'], row['Bobot Softmax'], row['Bobot Awal']])\n    if mix_rows:\n        sheets['Bobot Scalar Mix'] = Table(\n            ['Tahap','Fold','Indeks Layer RoBERTa','Layer ke-','Logit','Bobot Softmax','Bobot Awal'], mix_rows)\n"+anchor)
                s = s.replace("['Epoch','Satu putaran seluruh data latih pada fold/tahap itu.']", "['Epoch','Satu putaran seluruh data latih pada fold/tahap itu.'],\n        ['Bobot Scalar Mix','Empat bobot global pada checkpoint terpilih tiap fold/refit; berjumlah1. Indeks layer mulai0; layer ke- mulai1. Bukan skor siswa atau ukuran kepentingan kausal.']")
            if role == 'architecture_export':
                s = s.replace('RoBERTa → BiGRU', 'RoBERTa → Scalar Mix4 → BiGRU')
        if role == 'definition:answer_vectors_report':
            s = s.replace('BiGRU256', 'BiGRU128').replace('referensi256', 'referensi128').replace('matching256', 'matching128')
            if mix:
                s = s.replace('Output token BiGRU128', 'Empat layer terakhir RoBERTa dicampur dengan bobot global yang dipelajari, lalu output token BiGRU128')
        if cell['cell_type'] == 'markdown':
            s = s.replace('BiGRU128/arah', 'BiGRU64/arah').replace('respons256', 'respons128').replace('referensi256', 'referensi128')
            s = s.replace('matching1280→256', 'matching640→128').replace('pooling256', 'pooling128').replace('/√256', '/√128')
            s = s.replace('ringkasan256', 'ringkasan128').replace('Linear256', 'Linear128').replace('256→64→1', '128→64→1')
            if '## 5 · Hyperparameter' in s:
                s = '<a id="hyperparameter"></a>\n## 5 · Hyperparameter '+label+'\n\nEncoder5e-6seragam,MLP/proyeksi/BiGRU/co-attention/pooling/norm1e-4,BiGRU64perarah,attention64,dropoutMLP0.1. '+('E41 menambah SCALAR_MIX_LAST_LAYERS=4 dan SCALAR_MIX_LR=1e-4; bobot awal seragam0.25.\n' if mix else 'Ini resep E30 MLP; perubahan hanya struktur penyajian notebook.\n')
            if '### `SingleScoreModel`' in s:
                s = '### '+('`ScalarMix` dan ' if mix else '')+'`SingleScoreModel`\n\n'+('Satu encoder RoBERTa mengembalikan hidden_states. ScalarMix memilih empat keluaran Transformer terakhir, mengalikan bobot softmax yang dipelajari, lalu menjumlahkannya pada setiap posisi token. Tidak ada gamma/normalisasi tambahan. Tiga teks masuk encoder bersama; rater hanya menjadi target loss. Campuran768dimensi masuk BiGRU64perarah serta proyeksi referensi sebelum co-attention. CLS referensi untuk inspeksi tetap berasal dari layer terakhir.\n' if mix else 'Satu encoder RoBERTa memakai last_hidden_state seperti E30. BiGRU64perarah menghasilkan128fitur/token; proyeksi akhir768,concat1024,MLP256→1. MLP tetap prediktor akhir.\n')
            if mix and '## 19 · Arsitektur S3' in s:
                s = s.replace('Respons RoBERTa768', 'Satu RoBERTa → empat layer Transformer terakhir → Scalar Mix berbobot →768dimensi/token.\n\nRespons hasil campuran768')
                s += '\nToken referensi juga memakai campuran yang sama sebelum proyeksi referensi. Empat bobot global dipelajari dari loss; BGE tetap frozen dan MLP tetap prediktor akhir.\n'
            if '### `AnswerBiGRUAttention`' in s:
                s = '### `AnswerBiGRUAttention`\n\nBiGRU64perarah menghasilkan128fitur/token. Additive attention128→64→1 merangkum token; ukuran serta operasi mengikuti E30 MLP.\n'
            if '## 22 · Optimizer' in s:
                s = '## 22 · Optimizer, loss MSE dan checkpoint\n\nLR encoder seragam5e-6;MLP/proyeksi/pooling_norm1e-4. AdamW,weight decay0.01,warmup10%,scheduler linear dan clipping1 mengikuti E30. '+('Empat logit Scalar Mix mempunyai kelompok optimizer tersendiri dengan LR1e-4.\n' if mix else 'Tidak ada pembagian LR menurut layer encoder.\n')
            if '### `optimizer_groups`' in s:
                s = '### `optimizer_groups`\n\nSetiap parameter trainable diperiksa masuk tepat satu kelompok. RoBERTa5e-6;proyeksi BGE,MLP,pooling/norm1e-4. '+('Empat parameter scalar_mix.logits mempunyai kelompok scalar_mix dengan LR1e-4.\n' if mix else 'Hanya model S3 yang dijalankan.\n')
            if '## Rujukan dan batas kesimpulan' in s:
                s = s.split('**Dasar eksperimen kapasitas:**')[0]
                s += '\nSchuster&Paliwal(1997), [Bidirectional Recurrent Neural Networks](https://doi.org/10.1109/78.650093), menjadi dasar konteks dua arah. Hidden64perarah mengikuti resep E30; bukan optimum yang ditetapkan paper.\n'
                if mix:
                    s += '\n**Scalar Mix:** Peters dkk.(2018), [Deep Contextualized Word Representations / ELMo](https://aclanthology.org/N18-1202/), menggabungkan representasi beberapa layer dengan bobot yang dipelajari. ELMo memakai arsitektur berbeda serta skala gamma; E41 hanya mengadaptasi campuran softmax empat layer RoBERTa terakhir tanpa gamma atau normalisasi tambahan. [Tenney dkk.(2019), BERT Rediscovers the Classical NLP Pipeline](https://aclanthology.org/P19-1452/), memberi dasar bahwa informasi bahasa tersebar pada kedalaman layer berbeda. Pemilihan empat layer dan penerapan pada AES adalah hipotesis E41; hasil lebih baik tidak dijamin. Bobot global tidak boleh ditafsirkan sebagai penjelasan kausal atau penilaian siswa secara terpisah.\n'
        cell['source'] = s
    n['cells'][0]['source'] = intro(mix)
    fingerprint = hashlib.sha256('\n'.join(c['source'] for c in n['cells'] if c['cell_type'] == 'code').encode()).hexdigest()
    for index, cell in enumerate(n['cells']):
        cell['source'] = cell['source'].replace(PLACEHOLDER, fingerprint)
        cell['id'] = ('e41' if mix else 'e30native')+f'-{index:03d}-'+hashlib.sha256(cell['source'].encode()).hexdigest()[:8]
        if cell['cell_type'] == 'code':
            ast.parse(cell['source'])
            cell['execution_count'] = None
            cell['outputs'] = []
    n['metadata']['colab']['name'] = name+'.ipynb'
    n['metadata']['aes_standalone']['code_sha256'] = fingerprint
    meta = n['metadata']['aes_experiment']
    for key in ('s1_encoder_lr', 's2_mlp_lr'):
        meta.pop(key, None)
    meta.update(name=label, s3='E30 MLP S3'+('; Scalar Mix last four layers' if mix else '; native reference'),
                input_columns=['soal', 'jawaban_referensi', 'response_text'], s3_sequence_hidden=64,
                sequence_output_dimensions=128, pooling_attention_hidden=64, encoder_lr=5e-6, s3_encoder_lr=5e-6,
                scalar_mix=mix, scalar_mix_layers=4 if mix else 0, added_trainable_parameters=4 if mix else 0,
                format_commit=FORMAT_COMMIT, format_path=FORMAT_PATH, format_sha256=hashlib.sha256(raw).hexdigest(),
                historical_e30_results_reproduced=False,
                treatment='Learnable softmax mixture of last4 encoder layers; no gamma' if mix else 'Presentation-only native E30 MLP S3 reference')
    if mix:
        meta['scalar_mix_lr'] = 1e-4
    output = json.dumps(n, ensure_ascii=False, indent=1)+'\n'
    for directory in ('notebooks/experiments', 'aes-learning/notebooks/experiments'):
        (ROOT/directory/(name+'.ipynb')).write_text(output)
    print(json.dumps({'name': name, 'cells': len(n['cells']), 'code_sha256': fingerprint, 'scalar_mix': mix}))
    return n


def build():
    raw = subprocess.check_output(['git', 'show', FORMAT_COMMIT+':'+FORMAT_PATH], cwd=ROOT)
    return build_one(raw, False), build_one(raw, True)


if __name__ == '__main__':
    build()
