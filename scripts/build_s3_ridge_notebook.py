"""Build the standalone S3 Ridge notebook, preserving all prior experiments."""
import hashlib
import json
from pathlib import Path
import subprocess

from build_bge_readout_notebook import definitions

ROOT=Path(__file__).resolve().parents[1]
PIN='ce46a528a250d3fa43699e0e61235aa01e40f789'
BASE='notebooks/experiments/AES_300_BGE_M3_Ridge_Tiga_Input_Colab.ipynb'
E21_PIN='ef712b4c035d96178d9baa67caa7453ac9fd8738'
E21_PATH='notebooks/experiments/AES_300_E21_Tiga_Input_GPU.ipynb'
NAME='AES_300_S3_Fusion_Ridge_Tiga_Input_Colab'
PLACEHOLDER='__S3_RIDGE_CODE_HASH__'


def build():
    original=subprocess.check_output(['git','show',PIN+':'+BASE],cwd=ROOT);base=json.loads(original)
    old_hash=base['metadata']['aes_standalone']['code_sha256']
    e21raw=subprocess.check_output(['git','show',E21_PIN+':'+E21_PATH],cwd=ROOT);e21=json.loads(e21raw)
    def prior(i):
        return ''.join(base['cells'][i]['source']).removeprefix('%%aes\n').replace(old_hash,PLACEHOLDER).replace('AES_300_BGE_M3_Ridge_Tiga_Input_Colab',NAME)
    def source(i):return ''.join(e21['cells'][i]['source']).removeprefix('%%aes\n')
    cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s))
    def code(s,magic=True):cells.append(dict(cell_type='code',metadata={},source=('%%aes\n' if magic else '')+s,execution_count=None,outputs=[]))
    md('''# S3 · RoBERTa + BGE-M3 frozen → fitur gabungan → Ridge

**Tiga input tetap:** soal, jawaban referensi, respons siswa. Target satu `rater` 0–100.
Ini **feature fusion**, bukan penggabungan dua skor. Resep vektor mengikuti E21:

**RoBERTa CLS768 → LayerNorm** + **BGE frozen1024 → proyeksi256 → LayerNorm** → **concat1024 → Ridge**.

Representasi dilatih dengan MLP bantu hanya pada train, kemudian dikunci dan head diganti Ridge. MLP dari checkpoint yang sama tetap dievaluasi sebagai kontrol. S2 BGE frozen→Ridge alpha0,01 menjadi baseline tambahan, bukan hasil lama yang disalin.

Pilih **T4 GPU → Run all → unggah `fix_dataset_300.xlsx`**. Tanpa Drive atau clone repo. Ada16pelatihan representasi pada konfigurasi5fold:5outer+10pasangan inner+1final. Ini lebih berat dari notebook S2; waktu sebenarnya bergantung GPU/panjang respons. Cache tahap selesai dan progres epoch bersifat lokal; unduh ZIP sebelum runtime hilang.

**Belum ada hasil penelitian dari notebook ini.** Arsitektur ini adalah hipotesis yang diuji, bukan kombinasi yang sudah dinyatakan optimal oleh paper.''')
    md('''## 1 · Setup dan upload

Semua kode diembed. Worker numerik terpisah mempertahankan Torch CUDA Colab. Data, cache dan checkpoint berada di disk runtime lokal. Hasil tidak diunggah otomatis ke GitHub.''')
    for i in (2,3,4,5):
        s=prior(i).replace('BGE Ridge tiga input','S3 fusion Ridge').replace('BGE: GPU jika tersedia; Ridge: CPU float64','RoBERTa/BGE: GPU; Ridge: CPU float64')
        code(s,False)
    md('''## 2 · Konfigurasi sebelum melihat hasil

LR encoder5e-6; proyeksi/norm/MLP1e-4; AdamW decay0,01; batch efektif8/micro1; dropout0,1; warmup10%; clipping1. Resep diwarisi E21 sebagai kontrol, bukan parameter optimum dari paper.

**25epoch tetap**, tanpa early stopping memakai outer validation. Ini berbeda dari run historis E21, sehingga kontrol yang paling sepadan adalah MLP yang dilatih di notebook ini. Ridge tidak memiliki LR/epoch; grid alpha pada fitur concat mentah adalah0,01/0,1/1/10/100. Skala fitur fusion berbeda dari BGE L2, jadi alpha S2 tidak langsung dipindahkan sebagai alpha optimal S3.''')
    code(prior(7).replace("SCOPE='bge_m3_ridge_three_input_cv_and_test'","SCOPE='s3_fusion_ridge_nested_cv'").replace('from transformers import AutoModel, AutoTokenizer','from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup'))
    code('''# Ubah parameter sebelum run; jangan memilih dari test historis.
EPOCHS, FOLDS = 25, 5
S3_RIDGE_ALPHAS = [.01, .1, 1., 10., 100.]
S2_RIDGE_ALPHA = .01
DEVICE = 'cuda'
EXPECTED_ROWS = 300
cfg = dict(dataset=os.environ['AES300_DATASET'], output=str(ROOT/'outputs/s3_fusion_ridge'),
    sheet='Sheet1', expected_rows=EXPECTED_ROWS, score_min=0, score_max=100,
    test_fraction=.2, folds=FOLDS, split_seed=2026, split_candidates=128, seed=42,
    epochs=EPOCHS, early_stopping_patience=5, effective_batch=8, micro_batch=1,
    s3_encoder_lr=5e-6, encoder_lr=5e-6, projection_lr=1e-4, head_lr=1e-4,
    weight_decay=.01, warmup_ratio=.1, gradient_clip=1., dropout=.1,
    roberta_max_length=512, roberta_overlap=128, bge_max_length=1024, bge_overlap=128, bge_batch=4,
    device=DEVICE, context_max_tokens=192, question_max_tokens=64, input_format='single_sequence',
    s3_ridge_alphas=S3_RIDGE_ALPHAS, s2_ridge_alpha=S2_RIDGE_ALPHA,
    representation_policy='E21_fusion_fixed_epochs_train_only', code_sha256=NOTEBOOK_CODE_SHA256,
    roberta=dict(checkpoint='flax-community/indonesian-roberta-base', revision='cc5ecee7de8532880ad08d802292a6f42865bc83'),
    bge=dict(checkpoint='BAAI/bge-m3', revision='5617a9f61b028005a4858fdac845db406aefb181'))
if FOLDS < 3 or EPOCHS < 1:
    raise ValueError('Minimal3fold dan epoch positif.')
if DEVICE=='cuda' and not torch.cuda.is_available():
    raise RuntimeError('Pilih Runtime → Change runtime type → T4 GPU, lalu jalankan ulang.')
torch.set_num_threads(2)
print('Pelatihan representasi:', FOLDS+FOLDS*(FOLDS-1)//2+1, '| epoch setiap fit:', EPOCHS, flush=True)
print('Semua3teks → feature fusion → Ridge; kontrol MLP pada fitur sama.', flush=True)''')
    md('''## 3 · Dataset asli

Header: `Email address`, `response_number`, `response_text`, `soal`, `jawaban_referensi`, `rater`. Email/duplikat menjadi kelompok pencegah kebocoran dan tidak menjadi fitur model. Label tidak diisi, dibulatkan atau diganti otomatis.''')
    for i in (10,11,12):code(prior(i))
    md('''## 4 · Development/test dan fold kelompok

Resep split sama: GroupShuffleSplit seed2026 dengan128kandidat, pilih proporsi jumlah baris terdekat80:20 tanpa label; GroupKFold5 pada development. Untuk data300 yang sama, target240/60; jumlah aktual ditampilkan. Seluruh kelompok dipisahkan.

Test lama sudah dilihat dalam penelitian: evaluasi ini bersifat retrospektif. Keanggotaan dan teks harus dicocokkan sebelum dibandingkan dengan ZIP historis.''')
    code(prior(14));code(prior(15))
    md('''## 5 · Tiga teks dan BGE-M3 frozen

Soal/referensi membentuk konteks maksimal192token (soal64), lalu window respons. RoBERTa maksimal512, BGE1024, overlap128. Semua token respons dicakup mengikuti helper E21. BGE memakai eval/no_grad/requires_grad=False; mean CLS antar-window lalu L2. BGE tidak masuk optimizer S3.''')
    code(prior(17));code(definitions(source(21)))
    code(prior(19).replace('Fitur tiga input untuk Ridge','BGE frozen bersama untuk S3 dan baseline S2').replace('Selanjutnya Ridge CPU float64.','Selanjutnya fine-tuning representasi S3 per split.'))
    md('''## 6 · Fusi E21 dan training hanya pada train

MLP bantu1024→256→1 mempelajari representasi bersama RoBERTa, LayerNorm dan proyeksi BGE. Setelah25epoch tetap, vektor concat1024 dipakai untuk fit Ridge. Ridge tidak melakukan backpropagation ke encoder.

Seluruh target di luar train ditutupi NaN saat fungsi training dipanggil. Tidak ada early stopping memakai outer validation atau test. Checkpoint epoch menyimpan optimizer/RNG agar dapat dilanjutkan selama disk runtime masih ada.''')
    code(definitions(source(30))+'\n\n'+definitions(source(31)))
    code(prior(21))
    fitting=definitions(source(36))
    fitting=fitting.replace("'Jumlah epoch final yang dipilih dari CV selesai'","'Jumlah epoch tetap yang ditentukan sebelum run selesai'").replace("'fixed refit epochs selected from CV'","'fixed epoch budget declared before run'")
    code(fitting)
    md('''## 7 · Nested CV untuk encoder supervised dan Ridge

Untuk setiap outer fold, inner validation juga tidak boleh pernah menjadi label training representasi. Karena itu encoder/proyeksi dilatih pada3blok fold, Ridge dipasang di3blok tersebut, lalu alpha dibandingkan pada blok inner validation. Pasangan dua fold yang dikeluarkan menghasilkan train yang sama dalam dua arah; cache dipakai bersama sehingga10fit cukup untuk20validasi inner.

5fit outer memberi prediksi OOF yang tidak dilatih pada kelompoknya. Alpha final dipilih melalui5fold development menggunakan representasi outer masing-masing. Model final baru dilatih pada seluruh development, lalu test dievaluasi.

Cache berisi fitur dan identitas training. Dua snapshot optimizer sementara milik tahap ini dihapus hanya setelah cache selesai dan checksum tersimpan; tahap terputus mempertahankan snapshot resume. Checkpoint encoder final tetap disimpan. Folder eksperimen lama tidak disentuh.''')
    code(prior(23))
    helper=subprocess.check_output(['git','show',PIN+':scripts/bge_ridge_three_input.py'],cwd=ROOT).decode()
    code(definitions(helper,{'validate_ridge_inputs','ridge_metric_row'}))
    runtime=(ROOT/'scripts/s3_ridge_experiment.py').read_text();preamble=runtime[:runtime.index('def validate_s3_protocol')]
    groups=[{'validate_s3_protocol','representation_contract','collect_s3_features','fit_s3_representation','stage_values'},
            {'choose_s3_alpha','s3_predictions','s3_evaluation'}, {'run_s3_ridge'}, {'s3_duels','export_s3_ridge'}]
    for i,g in enumerate(groups):code((preamble if i==0 else '')+definitions(runtime,g))
    md('''## 8 · Jalankan S3 Ridge dan kontrol

Satu seed42. Ini bukan ensemble atau bukti kestabilan multi-seed. Head MLP kontrol dilatih bersama representasi; Ridge adalah penggantian readout setelah representasi dikunci. Perbandingan ini belum mengisolasi regularisasi, kelas model dan proses fitting satu per satu.

Rerun dengan data/kode/config/perangkat yang sama memakai ulang tahap lengkap dan melanjutkan epoch yang terputus. Cache yang rusak menghentikan run; tidak diabaikan diam-diam.''')
    code("result = run_s3_ridge(bundle, X, manifest, cfg, output)\nprint('S3 alpha final:',result['final_contract']['selected_alpha'],'| Test dievaluasi:',result['test_evaluated'],flush=True)")
    md('''## 9 · Evaluasi berpasangan

RMSE/MAE/bias dan toleransi memakai prediksi kontinu asli tanpa clipping. QWK/AC2 memakai salinan kategori0–100; nilai undefined diberi alasan. Jumlah dalam±1/±2/±5/±10 dan duel per jawaban dilaporkan agar penurunan RMSE tidak disamakan dengan semua ukuran presisi membaik.

OOF S3 Ridge memakai nested selection, sedangkan nilai grid final hanya untuk memilih alpha dan bukan skor independen. Bandingkan test dengan test pada data yang sama. Jangan menganggap MLP kontrol ini identik dengan run lama yang memakai early stopping.''')
    code('''show(pd.DataFrame([r for r in result['evaluation'] if r['partition']=='DEVELOPMENT_POOLED_OOF']), 'OOF development; S3 Ridge memakai nested selection')
show(pd.DataFrame([r for r in result['evaluation'] if r['partition']=='TEST_HISTORICAL_HOLDOUT']), 'Test historis pada60jawaban jika split240/60terpenuhi')
show(pd.DataFrame(s3_duels(result)), 'Duel pada jawaban sama')
show(pd.DataFrame(result['alpha_search']), 'Seleksi alpha; bukan pemilihan berdasarkan test')
show(pd.DataFrame([r for r in result['predictions'] if r['partition']=='TEST_HISTORICAL_HOLDOUT']), 'Prediksi test per jawaban')''')
    md('''## 10 · Lima Excel dan ZIP privat

Data/split; fitur BGE/RoBERTa/fusion final dan koefisien; prediksi OOF/test; evaluasi/duel/alpha; riwayat semua representasi dan referensi. ZIP juga memuat NPZ Ridge, kontrak model, split dan checksum.

**Checkpoint encoder/proyeksi/MLP final berukuran besar tetap lokal**, path ditampilkan. NPZ Ridge saja tidak cukup untuk menilai teks baru; perlu checkpoint representasi final dan resep token yang sama. Unduh checkpoint final juga jika ingin menyimpan model, sebelum runtime berakhir.

Referensi: [Ridge, Hoerl & Kennard1970](https://doi.org/10.1080/00401706.1970.10488634); [seleksi/evaluasi, Cawley & Talbot2010](https://www.jmlr.org/papers/v11/cawley10a.html); [SentEval2018](https://aclanthology.org/L18-1269/); [BGE-M32024](https://arxiv.org/abs/2402.03216); [RoBERTa2019](https://arxiv.org/abs/1907.11692); [AdamW2019](https://arxiv.org/abs/1711.05101).

Sumber tersebut mendukung komponen/prinsip. Fusi E21, grid alpha, LR dan25epoch merupakan rancangan eksperimen terkontrol; belum ada paper yang diverifikasi membuktikan resep persis ini optimal untuk AES ini.''')
    code("download_info = export_s3_ridge(bundle, X, manifest, cfg, result)\ndump(ROOT/'outputs/s3_ridge_download.json',download_info)\nshow(pd.DataFrame([download_info]), 'Paket hasil dan path model final')")
    code('''DOWNLOAD_FULL_ENCODER = False  # True: unduh checkpoint besar sesudah ZIP hasil.
info=json.loads((ROOT/'outputs/s3_ridge_download.json').read_text())
archive=Path(info['archive']);checkpoint=Path(info['final_representation_checkpoint'])
if not archive.is_file() or not checkpoint.is_file():
    raise FileNotFoundError('Paket/checkpoint final belum lengkap; periksa cell sebelumnya.')
print('ZIP hasil:',archive)
print('Checkpoint representasi final:',checkpoint)
print('NPZ Ridge saja tidak cukup untuk teks baru; simpan checkpoint jika diperlukan.')
if IN_COLAB:
    from google.colab import files
    files.download(str(archive))
    if DOWNLOAD_FULL_ENCODER:files.download(str(checkpoint))
else:
    print('Ambil file lokal pada path di atas.')''',False)
    identity=hashlib.sha256('\n'.join(c['source'].removeprefix('%%aes\n') for c in cells if c['cell_type']=='code').encode()).hexdigest()
    for i,c in enumerate(cells):
        c['source']=c['source'].replace(PLACEHOLDER,identity);c['id']=hashlib.sha256((str(i)+c['source']).encode()).hexdigest()[:12]
    n=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python'),colab=dict(provenance=[]),accelerator='GPU',
        aes_standalone=dict(code_sha256=identity,base_commit=PIN,base_path=BASE,base_sha256=hashlib.sha256(original).hexdigest(),
                            architecture_commit=E21_PIN,architecture_sha256=hashlib.sha256(e21raw).hexdigest()),
        aes_experiment=dict(id='S3_FEATURE_FUSION_RIDGE',input_columns=['soal','jawaban_referensi','response_text'],target='rater',
            fusion='E21 feature concat1024',models=['S3_RIDGE','S3_MLP_CONTROL','S2_RIDGE_FIXED'],BGE_frozen=True,
            RoBERTa_finetuned=True,supervised_representation_refit_inside_inner_cv=True,
            test_evaluation_enabled=True,research_results_included=False,drive_mount=False)),cells=cells)
    raw=json.dumps(n,ensure_ascii=False,indent=1)+'\n'
    for directory in ('notebooks/experiments','aes-learning/notebooks/experiments'):
        (ROOT/directory/(NAME+'.ipynb')).write_text(raw)
    print(NAME,len(cells),'cells',identity);return n


if __name__=='__main__':build()
