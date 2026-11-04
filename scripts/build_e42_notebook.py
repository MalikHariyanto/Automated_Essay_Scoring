"""E42: E41 plus one learnable Scalar Mix gamma; native S3 only."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PIN = '204720501f124d3fa7eb7091eb543baeba8ec21f'
BASE = 'notebooks/experiments/AES_300_E41_E30_S3_ScalarMix4_Native_Learnable_GPU.ipynb'
NAME = 'AES_300_E42_E41_S3_ScalarMix4_Gamma_Native_Learnable_GPU'
PLACEHOLDER = '__NOTEBOOK_CODE_SHA256__'


def build():
    raw = subprocess.check_output(['git', 'show', PIN+':'+BASE], cwd=ROOT)
    n = copy.deepcopy(json.loads(raw))
    previous = n['metadata']['aes_standalone']['code_sha256']
    for c in n['cells']:
        role = c['metadata'].pop('aes41_role', '')
        if role:
            c['metadata']['aes42_role'] = role
        s = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
        s = s.replace(previous, PLACEHOLDER).replace(Path(BASE).stem, '__E42_NAME__').replace('E41', 'E42').replace('__E42_NAME__', NAME)
        if role == 'hyperparameters':
            s += '\nSCALAR_MIX_GAMMA_INITIAL = 1.0  # Satu parameter baru; LR mengikuti SCALAR_MIX_LR=1e-4.\n'
        if role == 'configuration_defaults':
            s = s.replace("'scalar_mix_layers': SCALAR_MIX_LAST_LAYERS,", "'scalar_mix_gamma_init': SCALAR_MIX_GAMMA_INITIAL, 'scalar_mix_layers': SCALAR_MIX_LAST_LAYERS,")
            s = s.replace("    if result['s3_fusion'] != 'concat':", "    if isinstance(result['scalar_mix_gamma_init'], bool) or not math.isfinite(result['scalar_mix_gamma_init']):\n        raise ValueError('scalar_mix_gamma_init harus finite.')\n    if result['s3_fusion'] != 'concat':")
        if role == 'configuration_apply':
            s = s.replace('scalar_mix_layers=SCALAR_MIX_LAST_LAYERS,', 'scalar_mix_gamma_init=SCALAR_MIX_GAMMA_INITIAL, scalar_mix_layers=SCALAR_MIX_LAST_LAYERS,')
            s = s.replace('e41_native_scalar_mix4_v1', 'e42_native_scalar_mix4_gamma_v1')
        if role == 'definition:SingleScoreModel':
            s = s.replace('def __init__(self, number_of_layers=4):', 'def __init__(self, number_of_layers=4, gamma_initial=1.0):')
            s = s.replace('        # Empat logit nol menghasilkan empat bobot awal0.25; tidak memakai skala gamma.', '        # Empat logit nol memberi bobot0.25; gamma awal1 mempertahankan keluaran awal E41.')
            s = s.replace('        self.logits = nn.Parameter(torch.zeros(number_of_layers))', '        self.logits = nn.Parameter(torch.zeros(number_of_layers))\n        self.gamma = nn.Parameter(torch.full((1,), float(gamma_initial)))\n        self.capture_stats = False')
            s = s.replace('def forward(self, hidden_states):', 'def forward(self, hidden_states, attention_mask=None):')
            s = s.replace('        return mixed\n', '''        scaled = self.gamma.to(dtype=mixed.dtype) * mixed
        # Statistik hanya saat inspeksi fitur, tanpa graph gradien atau sinkronisasi CPU tiap train step.
        if self.capture_stats:
            with torch.no_grad():
                valid = attention_mask.bool() if attention_mask is not None else torch.ones_like(mixed[..., 0], dtype=torch.bool)
                counts = valid.sum(1).clamp_min(1)
                def average_norm(value):
                    return ((value.float().norm(dim=-1)*valid).sum(1)/counts).unsqueeze(1)
                self.norm_before = average_norm(mixed)
                self.norm_after = average_norm(scaled)
                self.norm_last_layer = average_norm(hidden_states[-1])
        return scaled
''')
            s = s.replace("'Bobot Awal': 1/self.number_of_layers}", "'Bobot Awal': 1/self.number_of_layers, 'Gamma': float(self.gamma.detach().cpu().item())}")
            s = s.replace('pooling_attention_hidden=64, scalar_mix_layers=4):', 'pooling_attention_hidden=64, scalar_mix_layers=4, gamma_initial=1.0):')
            s = s.replace('self.scalar_mix = ScalarMix(scalar_mix_layers)', 'self.scalar_mix = ScalarMix(scalar_mix_layers, gamma_initial)')
            s = s.replace('        hidden = self.scalar_mix(encoded.hidden_states)', "        self.scalar_mix.capture_stats = return_features\n        hidden = self.scalar_mix(encoded.hidden_states, windows['attention_mask'])")
            s = s.replace("        features = {}\n", "        features = {}\n        if return_features:\n            for name, value in [('mix_norm_before', self.scalar_mix.norm_before),\n                                ('mix_norm_after', self.scalar_mix.norm_after),\n                                ('mix_norm_last', self.scalar_mix.norm_last_layer)]:\n                features[name] = mean_by_essay(value, mapping, n_essays)\n")
        if role == 'definition:make_model':
            s = s.replace("scalar_mix_layers=cfg['scalar_mix_layers'])", "scalar_mix_layers=cfg['scalar_mix_layers'], gamma_initial=cfg['scalar_mix_gamma_init'])")
        if role == 'definition:load_model':
            s = s.replace("'scalar_mix_layers',", "'scalar_mix_gamma_init', 'scalar_mix_layers',")
        if role == 'definition:training_report':
            s = s.replace("settings={'scalar_mix_layers'", "settings={'scalar_mix_gamma_init':'Gamma awal Scalar Mix','scalar_mix_layers'")
            s = s.replace("row['Bobot Softmax'], row['Bobot Awal']])", "row['Bobot Softmax'], row['Bobot Awal'], row['Gamma']])")
            s = s.replace("'Bobot Softmax','Bobot Awal'], mix_rows)", "'Bobot Softmax','Bobot Awal','Gamma'], mix_rows)")
        if role == 'definition:answer_vectors_report':
            s = s.replace("mappings=[", "mappings=[('14_S3_fitur_setelah_training.xlsx','mix_norm_before','Norma Campuran Sebelum'),\n              ('14_S3_fitur_setelah_training.xlsx','mix_norm_after','Norma Campuran Sesudah'),\n              ('14_S3_fitur_setelah_training.xlsx','mix_norm_last','Norma Layer Terakhir'),")
        if c['cell_type'] == 'markdown':
            s = s.replace('Tidak ada gamma/normalisasi tambahan.', 'Gamma mengalikan campuran sebelum BiGRU/proyeksi referensi; tidak ada normalisasi tambahan.')
            s = s.replace('tanpa gamma atau normalisasi tambahan', 'dengan gamma dan tanpa normalisasi tambahan')
            s = s.replace('tanpa gamma/normalisasi tambahan', 'dengan gamma, tanpa normalisasi tambahan')
            s = s.replace('Empat parameter scalar_mix.logits', 'Empat logit dan satu gamma pada scalar_mix')
            s = s.replace('Empat logit Scalar Mix mempunyai kelompok optimizer tersendiri', 'Empat logit dan satu gamma Scalar Mix mempunyai kelompok optimizer yang sama')
            if '## 5 · Hyperparameter' in s:
                s += '\nPerlakuan E42: SCALAR_MIX_GAMMA_INITIAL=1.0. Gamma ikut dipelajari dengan LR SCALAR_MIX_LR=1e-4. Semua pengaturan lain mengikuti E41.\n'
            if '## Rujukan dan batas kesimpulan' in s:
                s += '\n**Gamma:** formulasi Peters dkk.(2018), [ELMo](https://aclanthology.org/N18-1202/), memakai skala gamma pada campuran layer. [AllenNLP ScalarMix](https://github.com/allenai/allennlp/blob/main/allennlp/modules/scalar_mix.py) menginisialisasi gamma1.0 dan [PretrainedTransformerEmbedder](https://github.com/allenai/allennlp/blob/main/allennlp/modules/token_embedders/pretrained_transformer_embedder.py) memakai modul tersebut untuk keluaran Transformer. Ini landasan metode dan implementasi; belum ada paper AES dengan susunan persis penelitian ini yang diverifikasi membuktikan gamma memperbaiki RMSE. LR1e-4 adalah pilihan eksperimen yang dipertahankan, bukan optimum yang ditetapkan paper.\n'
        c['source'] = s
    n['cells'][0]['source'] = '''# E42 · E41 + learnable gamma · S3 Native Learnable

**Satu perubahan dari E41: satu skalar gamma yang dipelajari, awal1.0, LR1e-4.** Representasi=gamma×jumlah berbobot empat layer RoBERTa terakhir. Empat logit tetap mulai nol (bobot0.25);gamma dan logit berada di kelompok scalar_mix dengan LR yang sama. Tambahan tepat satu parameter dari E41. Tidak ada normalisasi tambahan atau cabang encoder baru.

Gamma diterapkan pada representasi768dimensi sebelum respons masuk BiGRU dan referensi diproyeksikan untuk co-attention. Nilai gamma tidak dibatasi positif secara eksplisit, mengikuti parameter skalar bebas pada implementasi ScalarMix; norma sesudah penskalaan bergantung pada |gamma|. Gamma tidak ditambahkan ke skor akhir dan bukan koreksi bias berdasarkan test.

Sisa resep tetap E41: RoBERTa fine-tuned,LR seragam5e-6;BiGRU64perarah,co-attention,pooling,proyeksi/MLP1e-4;BGE frozen;concat1024→MLP256→GELU→Dropout0.1→1. **S3 saja,5fold sekali,maks25epoch,patience5,1refit+1test**,tanpa loop multi-seed atau kontrol otomatis. Split/seed/windowing/loss/optimizer/scheduler tetap. Refit=ceil median best epoch5fold;test tidak memilih pengaturan.

X=`soal,jawaban_referensi,response_text`;y=`rater`0–100hanya target loss/evaluasi,tidak masuk encoder. Python biasa per section tanpa magic atau worker tersembunyi. Nama model,kolom,hyperparameter,arsitektur,training dan unduhan terlihat langsung dalam kernel.

T4 GPU → Run all → upload fix_dataset_300.xlsx. Lima Excel standar;Excel05→Bobot Scalar Mix mencatat bobot dan gamma checkpoint terpilih tiap fold/refit. Excel02menambahkan Norma Campuran Sebelum,Norma Campuran Sesudah dan Norma Layer Terakhir:rerata L2token valid per window,kemudian rerata window per jawaban. Token khusus/konteks yang valid termasuk;padding tidak dihitung. Statistik hanya diambil saat inspeksi fitur setelah test,tanpa ikut update bobot.

Landasan gamma adalah adaptasi ELMo/ScalarMix dan implementasi Transformer AllenNLP,bukan bukti langsung peningkatan AES. Belum ada hasil penelitian E42;tidak dijamin mengalahkan E41/E30. Evaluasi test lama tetap eksploratif. Folder/hash/checkpoint terpisah;unduh ZIP sebelum runtime dihapus.

Peta belajar: [nama model](#nama-model),[kolom X/y](#kolom-xy),[hyperparameter](#hyperparameter),[arsitektur](#arsitektur),[training](#training),[evaluasi](#evaluasi),[unduh](#unduh).
'''
    fp = hashlib.sha256('\n'.join(c['source'] for c in n['cells'] if c['cell_type']=='code').encode()).hexdigest()
    for i,c in enumerate(n['cells']):
        c['source'] = c['source'].replace(PLACEHOLDER, fp)
        c['id'] = f'e42-{i:03d}-'+hashlib.sha256(c['source'].encode()).hexdigest()[:8]
        if c['cell_type']=='code':
            ast.parse(c['source']); c['outputs']=[]; c['execution_count']=None
    n['metadata']['colab']['name']=NAME+'.ipynb'
    n['metadata']['aes_standalone']['code_sha256']=fp
    n['metadata']['aes_experiment'].update(name='E42',s3='E41 Scalar Mix4 plus learnable gamma',scalar_mix_gamma_init=1.0,
        added_trainable_parameters=5,added_parameters_vs_e41=1,format_commit=PIN,format_path=BASE,format_sha256=hashlib.sha256(raw).hexdigest(),
        treatment='One learnable gamma multiplying E41 layer mixture; initial1 and LR1e-4; all other E41 settings retained')
    output=json.dumps(n,ensure_ascii=False,indent=1)+'\n'
    for directory in ['notebooks/experiments','aes-learning/notebooks/experiments']:
        (ROOT/directory/(NAME+'.ipynb')).write_text(output)
    print(json.dumps({'name':NAME,'cells':len(n['cells']),'code_sha256':fp}))
    return n


if __name__=='__main__':build()
