"""E38: E30 S3 with only MLP dropout0.1→0.2, using pinned native E36 format."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
BASE_COMMIT='349f2e82e876333bf2270d47e7eb86b59e939a55'
BASE_PATH='notebooks/experiments/AES_300_E36_E30_S3_EncoderLR3e6_Native_Learnable_GPU.ipynb'
NAME='AES_300_E38_E30_S3_MLPDropout02_Native_Learnable_GPU'
PLACEHOLDER='__NOTEBOOK_CODE_SHA256__'


def build():
    raw=subprocess.check_output(['git','show',BASE_COMMIT+':'+BASE_PATH],cwd=ROOT)
    n=copy.deepcopy(json.loads(raw));previous=n['metadata']['aes_standalone']['code_sha256']
    for c in n['cells']:
        role=c['metadata'].pop('aes36_role','')
        if role:c['metadata']['aes38_role']=role
        s=c['source'] if isinstance(c['source'],str) else ''.join(c['source'])
        s=s.replace(previous,PLACEHOLDER).replace(Path(BASE_PATH).stem,NAME)
        s=s.replace('E36','E38').replace('3e-6','5e-6').replace('lr3e6','mlp_dropout02')
        if role=='hyperparameters':
            s=s.replace('# HIPERPARAMETER PENELITIAN — resep E38, satu perubahan LR dari E30.',
                        '# HIPERPARAMETER PENELITIAN — E30; hanya dropout MLP S3 berubah.')
            s=s.replace('ENCODER_LR = 5e-6             # SATU perlakuan: encoder S3; baseline E30 = 5e-6.',
                        'ENCODER_LR = 5e-6             # Tetap E30; bukan LR E37 7e-6.')
            s=s.replace('DROPOUT = 0.1                   # Dropout hidden MLP, mengikuti E30.',
                        'DROPOUT = 0.1                   # Parameter lama untuk layer lain; tetap E30.\nS3_MLP_DROPOUT = 0.2             # SATU PERUBAHAN: dropout setelah GELU pada MLP S3 (E30=0.1).')
            s=s.replace('TRAIN_SEED = 42','TRAIN_SEED = 42  # Satu konstanta pengacakan, tanpa loop multi-seed.')
        if role=='configuration_defaults':
            s=s.replace("'dropout': DROPOUT,","'dropout': DROPOUT, 's3_mlp_dropout': S3_MLP_DROPOUT,")
            anchor="    if not 0 <= result['dropout'] < 1 or result['weight_decay'] < 0:"
            assert s.count(anchor)==1
            s=s.replace(anchor,"    if not 0 <= result['s3_mlp_dropout'] < 1:\n        raise ValueError('s3_mlp_dropout harus berada pada rentang [0,1).')\n"+anchor)
        if role=='configuration_apply':
            s=s.replace('dropout=DROPOUT, weight_decay=WEIGHT_DECAY,','dropout=DROPOUT, s3_mlp_dropout=S3_MLP_DROPOUT, weight_decay=WEIGHT_DECAY,')
            s+="\ndisplay(pd.DataFrame([{'Bagian':'MLP S3 setelah GELU','Dropout p':cfg['s3_mlp_dropout']}, {'Bagian':'Parameter dropout lama/layer lain','Dropout p':cfg['dropout']}]))\n"
        if role=='definition:SingleScoreModel':
            s=s.replace('dropout=.1, sequence_hidden=64):','dropout=.1, sequence_hidden=64, s3_mlp_dropout=.2):')
            anchor='            # MLP S3: linear 1024 ke 256, GELU, dropout, lalu satu skor.\n'
            before,after=s.split(anchor)
            assert after.count('nn.Dropout(dropout)')==1
            after=after.replace('nn.Dropout(dropout)','nn.Dropout(s3_mlp_dropout)')
            s=before+anchor+'            # E38: hanya head[2] memakai p khusus; dropout encoder/layer lain tetap.\n'+after
        if role=='definition:make_model':
            s=s.replace("sequence_hidden=cfg['s3_sequence_hidden'])","sequence_hidden=cfg['s3_sequence_hidden'],\n                            s3_mlp_dropout=cfg['s3_mlp_dropout'])")
        if role=='definition:load_model':
            s=s.replace("'bge_max_length'):","'bge_max_length', 'dropout', 's3_mlp_dropout', 's3_encoder_lr'):")
        if role=='architecture_export':
            s=s.replace("'head':'RoBERTa", "'head':f'RoBERTa")
            s=s.replace('GELU → dropout → Linear 1',"GELU → dropout(p={cfg[\"s3_mlp_dropout\"]}) → Linear 1")
        if role=='definition:training_report':
            s=s.replace("settings={'s3_mlp_lr'","settings={'s3_mlp_dropout':'Dropout MLP S3','dropout':'Dropout parameter lama','seed':'Seed training tetap','s3_mlp_lr'")
        if c['cell_type']=='markdown':
            if '## 5 · Hyperparameter' in s:
                s='<a id="hyperparameter"></a>\n## 5 · Hyperparameter E38\n\nEncoder RoBERTa kembali ke E30 5e-6; semua LR lain1e-4. Hanya S3_MLP_DROPOUT menjadi0.2(E30=0.1). DROPOUT lama tetap0.1; dropout pretrained encoder tidak diubah. Lima fold sekali,25epoch maksimum,patience5.\n'
            if '## 22 · Optimizer' in s:
                s='## 22 · Optimizer, loss MSE dan checkpoint\n\nSemua LR tetap E30: encoder5e-6; MLP/proyeksi/pooling_norm1e-4. Dropout MLP bukan learning rate: sebagian aktivasi dinolkan acak hanya saat training, sedangkan evaluasi memakai model.eval().\n'
            if '### `SingleScoreModel`' in s:
                s+='\nE38 memisahkan s3_mlp_dropout dari dropout lama. Urutan MLP S3: Linear1024→256, GELU, Dropout(p=0.2), Linear256→1. RoBERTa tetap fine-tuned; BGE frozen.\n'
            if '## Rujukan dan batas kesimpulan' in s:
                s+='\n**Dropout:** Srivastava dkk.(2014), [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html). Paper mendukung metode regularisasi dropout; p=0.2 di E38 adalah hipotesis ablasi, bukan nilai optimal yang diwajibkan paper.\n'
        c['source']=s
    n['cells'][0]['source']='''# E38 · E30 S3 · Dropout MLP0.2 · Native Learnable

**Satu perlakuan dari E30: dropout hidden MLP S3 naik0.1→0.2.** Encoder RoBERTa tetap5e-6; MLP,proyeksi BGE,BiGRU/co-attention/pooling/norm tetap1e-4. Dropout RoBERTa dan komponen lain tetap. Parameter yang diuji mudah dicari: `S3_MLP_DROPOUT = 0.2` di section5.

Tiga teks → RoBERTa fine-tuned → BiGRU → co-attention respons–referensi → attention pooling → gabung proyeksi BGE frozen dengan concat → MLP: Linear1024→256 → GELU → Dropout0.2 → Linear256→1. Hanya S3 dijalankan; tidak memuat atau melatih S1/S2. Prediktor akhir MLP, bukan Ridge. X=`soal,response_text,jawaban_referensi`; y=`rater`0–100 membimbing loss dan tidak masuk encoder.

**5-fold CV sekali, maksimal25epoch/fold, early stopping patience5**, lalu **1refit +1test**. Epoch refit =ceil median best epoch5fold. TRAIN_SEED42satu konstanta, tanpa loop tiga seed atau pengulangan kontrol E30. Split/kelompok/fold mengikuti E30. Test tidak memilih dropout/LR/epoch; karena test lama telah sering dilihat, perbandingan berikutnya tetap eksploratif.

Format native E35/E36: Python biasa per section, fungsi serta pemanggilan terlihat langsung di kernel, tanpa magic/worker tersembunyi, pemeriksaan cosine, clone repo atau Drive. Dropout aktif saat training dan dinonaktifkan saat evaluasi/prediksi. Tidak ada jaminan E38 mengalahkan E30.

Pilih runtime baru **T4 GPU → Run all → upload fix_dataset_300.xlsx**. Output lima Excel standar;05_Riwayat_Pelatihan.xlsx→Pengaturan Eksperimen mencatat dropout MLP S3 yang benar-benar dipakai. Checkpoint E38 terpisah dari eksperimen lama. Resume berlaku pada runtime lokal yang sama; unduh ZIP sebelum runtime dihapus.

Peta belajar: [nama model](#nama-model), [kolom X/y](#kolom-xy), [hyperparameter](#hyperparameter), [X/y aktual](#xy-aktual), [arsitektur](#arsitektur), [training](#training), [evaluasi](#evaluasi), [unduh hasil](#unduh).
'''
    fingerprint=hashlib.sha256('\n'.join(c['source'] for c in n['cells'] if c['cell_type']=='code').encode()).hexdigest()
    for i,c in enumerate(n['cells']):
        c['source']=c['source'].replace(PLACEHOLDER,fingerprint)
        c['id']=f'e38-{i:03d}-'+hashlib.sha256(c['source'].encode()).hexdigest()[:8]
        if c['cell_type']=='code':
            ast.parse(c['source']);c['outputs']=[];c['execution_count']=None
    n['metadata']['colab']['name']=NAME+'.ipynb'
    n['metadata']['aes_standalone']['code_sha256']=fingerprint
    n['metadata']['aes_experiment'].update(name='E38',s3='E30 S3; only MLP hidden dropout0.1 to0.2',
        s1='not run',s2='not run',encoder_lr=5e-6,s3_encoder_lr=5e-6,s3_mlp_dropout=.2,
        other_dropout_unchanged=True,scenarios=['S3'],training_seed=42,multiple_seed_runs=False,
        control_retrained=False,cv_fits=5,refits=1,test_evaluations=1,ensemble=False,
        mlp_lr_scope='S3 model.head.* only',format_commit=BASE_COMMIT,format_path=BASE_PATH,
        format_sha256=hashlib.sha256(raw).hexdigest(),
        treatment='only S3 MLP dropout0.1 to0.2 compared with E30; encoder5e-6 and every other setting unchanged')
    output=json.dumps(n,ensure_ascii=False,indent=1)+'\n'
    for d in ('notebooks/experiments','aes-learning/notebooks/experiments'):
        (ROOT/d/(NAME+'.ipynb')).write_text(output)
    print(json.dumps({'name':NAME,'code_sha256':fingerprint,'cells':len(n['cells']),'cv_fits':5,'refits':1}))
    return n


if __name__=='__main__':build()
