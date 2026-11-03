"""E39: E30 MLP with BiGRU128 per direction; native S3-only presentation."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
FORMAT_COMMIT='076a57ce1f6295951abd3f5b785ea4dc5b70095f'
FORMAT_PATH='notebooks/experiments/AES_300_E38_E30_S3_MLPDropout02_Native_Learnable_GPU.ipynb'
NAME='AES_300_E39_E30_S3_BiGRU128_Native_Learnable_GPU'
PLACEHOLDER='__NOTEBOOK_CODE_SHA256__'

MODEL='''class SingleScoreModel(nn.Module):
    """S3 saja: representasi dua encoder digabung dan dinilai oleh MLP."""
    def __init__(self, scenario, roberta=None, dropout=.1, sequence_hidden=128,
                 s3_mlp_dropout=.1, pooling_attention_hidden=64):
        super().__init__()
        if scenario != 'S3' or roberta is None:
            raise ValueError('E39 memerlukan S3 dan encoder RoBERTa; tidak membuat skenario lain.')
        self.scenario, self.roberta = scenario, roberta
        roberta.requires_grad_(True)
        self.dropout = nn.Dropout(dropout)  # Atribut historis; forward S3 memakai dropout di head.
        # Urutan inisialisasi layer yang tidak berubah tetap mengikuti E30.
        self.roberta_norm = nn.LayerNorm(768)
        self.projection = nn.Sequential(nn.Linear(1024, 256), nn.LayerNorm(256))
        self.head = nn.Sequential(nn.Linear(1024, 256), nn.GELU(),
                                  nn.Dropout(s3_mlp_dropout), nn.Linear(256, 1))
        self.sequence_pool = AnswerBiGRUCoAttentionPool(sequence_hidden, pooling_attention_hidden)

    def forward(self, windows=None, mapping=None, n_essays=None, bge=None,
                return_features=False, answer_mask=None, reference_mask=None):
        hidden = self.roberta(**windows).last_hidden_state
        cls = hidden[:, 0, :]
        # BiGRU128/arah menghasilkan256fitur per token; pooling diproyeksikan ke768.
        pooled = self.sequence_pool(hidden, answer_mask, reference_mask)
        essay = mean_by_essay(pooled, mapping, n_essays)
        features = {}
        features['roberta_cls_reference'] = mean_by_essay(cls, mapping, n_essays)
        features['answer_pool768'] = essay
        features['bigru_summary'] = mean_by_essay(self.sequence_pool.last_summary, mapping, n_essays)
        features.update(roberta=essay, roberta_window_cls=cls)
        if bge is None or bge.shape != (n_essays, 1024):
            raise ValueError('BGE frozen memerlukan satu vektor1024 untuk setiap jawaban.')
        features['bge'] = bge
        features['roberta_layernorm'] = self.roberta_norm(essay)
        features['projection'] = self.projection(bge)
        features['concat'] = torch.cat([features['roberta_layernorm'], features['projection']], dim=1)
        prediction = self.head(features['concat'])
        if prediction.ndim != 2 or prediction.shape[1] != 1:
            raise ValueError('S3 harus menghasilkan satu nilai untuk setiap jawaban.')
        return (prediction, features) if return_features else prediction
'''

ENGINE_PREFIX='''class Engine:
    """Jalankan tiga teks melalui model fusion S3 yang sama."""
    def __init__(self, model, texts, bge, cfg, device, roberta_tokenizer=None, contexts=None):
        if model.scenario != 'S3':
            raise ValueError('Engine E39 hanya menjalankan S3.')
        self.model, self.texts, self.bge, self.cfg, self.device = model, texts, bge, cfg, device
        self.contexts = contexts
        if contexts is None or len(contexts) != len(texts):
            raise ValueError('Setiap jawaban memerlukan satu pasangan soal dan referensi.')
        self.tokenizer = roberta_tokenizer or tokenizer(cfg, 'roberta')

    def forward(self, indices, return_features=False):
        ids = [int(i) for i in indices]
        features = self.bge[ids].to(self.device)
        windows, mapping, details = pair_batch_windows(
            [self.texts[i] for i in ids], [self.contexts[i] for i in ids],
            self.tokenizer, self.device, self.cfg['roberta_max_length'],
            self.cfg['roberta_overlap'], self.cfg)
        mask = answer_token_mask(windows, mapping, details, self.tokenizer.padding_side)
        ref_mask = reference_token_mask(windows, mapping, details, self.tokenizer.padding_side)
        self.model.sequence_pool.capture_alignment = return_features
        if return_features:
            self.pooling_batch = (windows, mask)
            self.coattention_batch = (windows, mask, ref_mask)
        return self.model(windows=windows, mapping=mapping, n_essays=len(ids), bge=features,
                          return_features=return_features, answer_mask=mask, reference_mask=ref_mask)

'''


def build():
    raw=subprocess.check_output(['git','show',FORMAT_COMMIT+':'+FORMAT_PATH],cwd=ROOT)
    n=copy.deepcopy(json.loads(raw));old_hash=n['metadata']['aes_standalone']['code_sha256']
    for c in n['cells']:
        role=c['metadata'].pop('aes38_role','')
        if role:c['metadata']['aes39_role']=role
        s=c['source'] if isinstance(c['source'],str) else ''.join(c['source'])
        s=s.replace(old_hash,PLACEHOLDER).replace(Path(FORMAT_PATH).stem,NAME)
        s=s.replace('E38','E39').replace('mlp_dropout02','bigru128_e30_mlp')
        if role=='hyperparameters':
            s=s.replace('# HIPERPARAMETER PENELITIAN — E30; hanya dropout MLP S3 berubah.',
                        '# HIPERPARAMETER PENELITIAN — E30 MLP; hanya hidden BiGRU diperbesar.')
            s=s.replace('BIGRU_HIDDEN_PER_DIRECTION = 64   # Dua arah menghasilkan128fitur/token.',
                        'BIGRU_HIDDEN_PER_DIRECTION = 128  # PERUBAHAN: E30=64/arah; E39=128/arah.\nPOOLING_ATTENTION_HIDDEN = 64    # Bottleneck attention pooling tetap E30, tidak ikut diperbesar.')
            s=s.replace('S3_MLP_DROPOUT = 0.2             # SATU PERUBAHAN: dropout setelah GELU pada MLP S3 (E30=0.1).',
                        'S3_MLP_DROPOUT = 0.1             # Kembali ke E30; bukan dropout0.2 E38.')
        if role=='configuration_defaults':
            s=s.replace("'s3_sequence_hidden': BIGRU_HIDDEN_PER_DIRECTION,",
                        "'s3_sequence_hidden': BIGRU_HIDDEN_PER_DIRECTION, 'pooling_attention_hidden': POOLING_ATTENTION_HIDDEN,")
            s=s.replace("'s2_mlp_lr': 1e-4, ",'').replace("'s2_mlp_lr', ",'')
            anchor="    if result['s3_fusion'] != 'concat':"
            assert s.count(anchor)==1
            s=s.replace(anchor,"    if type(result['pooling_attention_hidden']) is not int or result['pooling_attention_hidden'] < 1:\n        raise ValueError('pooling_attention_hidden harus bilangan bulat positif.')\n"+anchor)
        if role=='configuration_apply':
            s=s.replace('head_lr=1e-4, s2_mlp_lr=1e-4,','head_lr=1e-4,')
            s=s.replace("s3_sequence_hidden=BIGRU_HIDDEN_PER_DIRECTION,","s3_sequence_hidden=BIGRU_HIDDEN_PER_DIRECTION, pooling_attention_hidden=POOLING_ATTENTION_HIDDEN,")
            s+="\ndisplay(pd.DataFrame([{'Tahap':'Hidden BiGRU per arah','Dimensi':cfg['s3_sequence_hidden']}, {'Tahap':'Output dua arah / ringkasan co-attention','Dimensi':2*cfg['s3_sequence_hidden']}, {'Tahap':'Bottleneck attention pooling','Dimensi':cfg['pooling_attention_hidden']}, {'Tahap':'Proyeksi RoBERTa sebelum fusion','Dimensi':768}, {'Tahap':'Concat ke MLP','Dimensi':1024}]))\n"
        if role=='definition:AnswerBiGRUAttention':
            s=s.replace('def __init__(self, hidden_size=64):','def __init__(self, hidden_size=128, attention_hidden_size=64):')
            s=s.replace('nn.Linear(2*hidden_size, hidden_size)','nn.Linear(2*hidden_size, attention_hidden_size)')
            s=s.replace('nn.Linear(hidden_size, 1, bias=False)','nn.Linear(attention_hidden_size, 1, bias=False)')
        if role=='definition:AnswerBiGRUCoAttentionPool':
            s=s.replace('def __init__(self, hidden_size=64):','def __init__(self, hidden_size=128, attention_hidden_size=64):')
            s=s.replace('super().__init__(hidden_size)','super().__init__(hidden_size, attention_hidden_size)')
            s=s.replace('respons128 ↔ referensi128, lalu attention pooling128','respons256 ↔ referensi256, lalu attention pooling256→64→1')
        if role=='definition:SingleScoreModel':s=MODEL
        if role=='definition:Engine':
            tree=ast.parse(s).body[0];predict=next(v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name=='predict')
            predict_source='\n'.join(s.splitlines()[predict.lineno-1:predict.end_lineno])
            s=ENGINE_PREFIX+predict_source+'\n'
        if role=='definition:make_model':
            s='''def make_model(scenario, cfg):
    if scenario != 'S3':
        raise ValueError('E39 hanya membuat model S3.')
    return SingleScoreModel('S3', encoder(cfg, 'roberta'), dropout=cfg['dropout'],
                            sequence_hidden=cfg['s3_sequence_hidden'],
                            s3_mlp_dropout=cfg['s3_mlp_dropout'],
                            pooling_attention_hidden=cfg['pooling_attention_hidden'])
'''
        if role=='definition:optimizer_groups':
            s=s.replace('MLP S2/S3 mempunyai LR terpisah dari linear S1 dan pooling; semua parameter tercakup tepat sekali.',
                        'Empat kelompok optimizer S3; semua parameter trainable tercakup tepat sekali.')
            s=s.replace("cfg['s3_encoder_lr'] if model.scenario == 'S3' else cfg['encoder_lr']","cfg['s3_encoder_lr']")
            s=s.replace("cfg['head_lr'] if model.scenario == 'S1' else cfg['s2_mlp_lr'] if model.scenario == 'S2' else cfg['s3_mlp_lr']","cfg['s3_mlp_lr']")
        if role=='definition:load_model':
            s=s.replace("'s3_encoder_lr'):","'s3_encoder_lr', 's3_sequence_hidden', 'pooling_attention_hidden', 's3_pooling', 's3_fusion'):")
        if role=='architecture_export':
            s='''specifications=pd.DataFrame([{
    'scenario':'S3', 'encoder':'RoBERTa fine-tuned; BGE-M3 frozen', 'vector':1024,
    'head':f"RoBERTa → BiGRU{cfg['s3_sequence_hidden']}/arah → co-attention{2*cfg['s3_sequence_hidden']} → attention pooling → proyeksi768 + BGE proyeksi256 → concat1024 → MLP256 →1",
    'output':1}])
journal.save('08_arsitektur',{'Spesifikasi':specifications},'Arsitektur S3 E39; belum ada prediksi',status='DESIGN / CONFIGURATION')
show(specifications,'Satu model S3 dan dimensi yang dipakai')
'''
        if role=='definition:answer_vectors_report':
            s=s.replace('bigru_hidden128','bigru_summary')
            s=s.replace('BiGRU128','BiGRU256').replace('referensi128','referensi256').replace('matching128','matching256')
        if role=='definition:training_report':
            s=s.replace("settings={'s3_mlp_dropout'","settings={'pooling_attention_hidden':'Hidden attention pooling','s3_mlp_dropout'")
        if c['cell_type']=='markdown':
            s=s.replace('p=0.2 di E39 adalah hipotesis ablasi, bukan nilai optimal yang diwajibkan paper.',
                        'dropout0.1 dipertahankan dari E30, tidak menjadi perlakuan E39.')
            s=s.replace('BiGRU64/arah','BiGRU128/arah').replace('respons128','respons256').replace('referensi128','referensi256')
            s=s.replace('matching640→128','matching1280→256').replace('pooling128','pooling256').replace('/√128','/√256')
            s=s.replace('S3_MLP_DROPOUT menjadi0.2(E30=0.1)','S3_MLP_DROPOUT tetap0.1')
            if '## 5 · Hyperparameter' in s:
                s='<a id="hyperparameter"></a>\n## 5 · Hyperparameter E39\n\nSatu perlakuan: BIGRU_HIDDEN_PER_DIRECTION=128(E30=64). Dua arah menghasilkan256fitur/token. Bottleneck attention pooling tetap64. Encoder5e-6,semua LR lain1e-4,dropoutMLP0.1. Semua pilihan dapat dicari langsung di cell ini.\n'
            if '## 19 · Arsitektur S3' in s:
                s='<a id="arsitektur"></a>\n## 19 · Arsitektur S3 saja\n\nRespons RoBERTa768 → BiGRU128/arah → respons256; referensi RoBERTa768 → Linear256. Co-attention dua arah → matching1280→256 → attention pooling256→64→1 → ringkasan256 → Linear768 → LayerNorm. BGE frozen1024 → proyeksi256 + LayerNorm. Concat1024 → MLP256→GELU→Dropout0.1→1. Teks asli tidak dibalik; keluaran kedua arah sejajar dengan posisi token asli.\n'
            if '### `SingleScoreModel`' in s:
                s='### `SingleScoreModel`\n\nKelas ini hanya berisi model S3. Hidden BiGRU128perarah berbeda dari keluarannya256fitur/token. Parameter128tidak mengubah keluaran akhir768RoBERTa/1024fusion atau jumlah skor. MLP tetap prediktor akhir dengan dropout0.1.\n'
            if '### `AnswerBiGRUCoAttentionPool`' in s:
                s=s.replace('Komponen E30/E39:', 'Komponen E39:')
            if '### `AnswerBiGRUAttention`' in s:
                s+='\nE39 memisahkan ukuran hidden BiGRU128perarah dari bottleneck attention pooling64. Hanya kapasitas recurrent yang dipilih lebih besar; dimensi masuk proyeksi/matching mengikuti keluaran256secara konsisten.\n'
            if '## Rujukan dan batas kesimpulan' in s:
                s+='\n**Dasar eksperimen kapasitas:** Tan,Lee&Lim(2023), [RoBERTa-GRU](https://doi.org/10.3390/app13063915), menguji kapasitas GRU dan melaporkan pilihan256unit pada tugas sentimen. BiGRU128perarah di E39 menghasilkan256fitur gabungan, tetapi tidak identik dengan GRU256searah pada paper. Angka128perarah adalah hipotesis kapasitas untuk AES, bukan optimum yang sudah terbukti.\n\n**Dua arah:** Schuster&Paliwal(1997), [Bidirectional Recurrent Neural Networks](https://doi.org/10.1109/78.650093). E39 menguji kapasitas BiGRU, bukan ablation GRU satu arah.\n'
        c['source']=s
    n['cells'][0]['source']='''# E39 · E30 MLP S3 · BiGRU128 per arah · Native Learnable

**Satu perlakuan: hidden BiGRU64→128unit per arah.** Keluaran gabungan menjadi256fitur untuk setiap token. E39 menguji kapasitas BiGRU, bukan menggantinya dengan GRU satu arah. Bentuk kalimat tetap asli; keluaran dua arah disejajarkan pada posisi token asli.

**Acuan adalah E30 MLP.** RoBERTa fine-tuned → BiGRU → co-attention respons–referensi → attention pooling → proyeksi768+LayerNorm; BGE-M3 frozen1024→proyeksi256+LayerNorm; concat1024→MLP256→GELU→Dropout0.1→1. MLP tetap prediktor akhir. Tidak ada Ridge atau pengali fusion baru. Bottleneck attention pooling tetap64; hanya dimensi masukan matching/pooling/proyeksi yang mengikuti keluaran BiGRU256.

Encoder kembali5e-6; MLP/proyeksi/BiGRU/co-attention/pooling/norm1e-4; dropout MLP kembali0.1. **S3 saja,5fold CV sekali,maksimal25epoch/fold,early stopping patience5,1refit+1test**. TRAIN_SEED42satu konstanta; pola seed per fold mengikuti E30 asli, bukan pengulangan tiga seed. Epoch refit=ceil median best epoch5fold. Test tidak memilih hidden/LR/epoch; hasil pada test lama tetap eksploratif.

Tiga input X=`soal,response_text,jawaban_referensi`; target y=`rater`0–100 tidak masuk encoder. Section nama model,X/y,hyperparameter,arsitektur,training,evaluasi dan unduhan berada langsung di kernel. Tidak memakai cell magic,worker tersembunyi,pemeriksaan cosine,clone repo atau Drive. Kelas model/Engine hanya menjalankan S3.

Pilih runtime baru **T4 GPU → Run all → upload fix_dataset_300.xlsx**. Output lima Excel standar. Pada02,Ringkasan CoAttn Pooling S3 berisi256komponen dan Fusion S3 tetap1024. Pada05,Pengaturan Eksperimen mencatat hidden128perarah,attention64,dropout0.1 dan LR aktual. Cache/checkpoint E39 terpisah; unduh ZIP sebelum runtime dihapus. Tidak ada jaminan E39 mengalahkan E30.

Peta belajar: [nama model](#nama-model), [kolom X/y](#kolom-xy), [hyperparameter](#hyperparameter), [X/y aktual](#xy-aktual), [arsitektur](#arsitektur), [training](#training), [evaluasi](#evaluasi), [unduh hasil](#unduh).
'''
    fingerprint=hashlib.sha256('\n'.join(c['source'] for c in n['cells'] if c['cell_type']=='code').encode()).hexdigest()
    for i,c in enumerate(n['cells']):
        c['source']=c['source'].replace(PLACEHOLDER,fingerprint)
        c['id']=f'e39-{i:03d}-'+hashlib.sha256(c['source'].encode()).hexdigest()[:8]
        if c['cell_type']=='code':ast.parse(c['source']);c['outputs']=[];c['execution_count']=None
    n['metadata']['colab']['name']=NAME+'.ipynb';n['metadata']['aes_standalone']['code_sha256']=fingerprint
    n['metadata']['aes_experiment'].update(name='E39',s3='E30 MLP S3; BiGRU128 per direction, output256',
        s3_sequence_hidden=128,sequence_output_dimensions=256,pooling_attention_hidden=64,
        s3_mlp_dropout=.1,encoder_lr=5e-6,s3_encoder_lr=5e-6,s1='not run',s2='not run',
        scenarios=['S3'],training_seed=42,multiple_seed_runs=False,control_retrained=False,
        cv_fits=5,refits=1,test_evaluations=1,ensemble=False,
        format_commit=FORMAT_COMMIT,format_path=FORMAT_PATH,format_sha256=hashlib.sha256(raw).hexdigest(),
        treatment='BiGRU hidden64 to128 per direction; pooling attention bottleneck64; E30 MLP and all LR/dropout restored')
    out=json.dumps(n,ensure_ascii=False,indent=1)+'\n'
    for directory in ('notebooks/experiments','aes-learning/notebooks/experiments'):(ROOT/directory/(NAME+'.ipynb')).write_text(out)
    print(json.dumps({'name':NAME,'cells':len(n['cells']),'code_sha256':fingerprint,'hidden_per_direction':128,'summary_dimensions':256,'fusion_dimensions':1024}))
    return n


if __name__=='__main__':build()
