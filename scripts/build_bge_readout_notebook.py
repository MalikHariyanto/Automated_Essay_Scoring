"""Build standalone frozen-BGE readout diagnostic with the existing Colab structure."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PIN = 'ef712b4c035d96178d9baa67caa7453ac9fd8738'
BASE = 'notebooks/experiments/AES_300_E21_Tiga_Input_GPU.ipynb'
NAME = 'AES_300_BGE_Frozen_Readout_Colab'


def definitions(text, names=None):
    text = text.removeprefix('%%aes\n'); lines = text.splitlines()
    blocks = []
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and (names is None or node.name in names):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            blocks.append('\n'.join(lines[start:node.end_lineno]))
    return '\n\n\n'.join(blocks)


def build():
    original = subprocess.check_output(['git','show',PIN+':'+BASE], cwd=ROOT)
    base = json.loads(original)
    src = lambda i: ''.join(base['cells'][i]['source'])
    cells = []
    def md(s): cells.append(dict(cell_type='markdown',metadata={},source=s))
    def code(s, magic=True):
        cells.append(dict(cell_type='code',metadata={},source=('%%aes\n' if magic else '')+s,
                          execution_count=None,outputs=[]))
    md('''# Diagnosis BGE-M3 frozen · empat cara membaca fitur

Tiga teks tetap: **soal + jawaban referensi + respons siswa**. Target hanya **rater 0–100**.
BGE tidak dilatih. Vektor dihitung sekali, kemudian dipakai bersama oleh empat kondisi.
Ini diagnosis cabang BGE/readout, bukan pengganti eksperimen S1/S2/S3 atau bukti perbaikan S3.

1. **MLP asli** sebagai kontrol.
2. **MLP target-center**: kurangi rerata target **train fold**, tambahkan kembali pada prediksi.
3. **MLP StandardScaler**: skala tiap komponen fitur memakai statistik **train fold**, tanpa target-centering.
4. **Ridge**: alpha dipilih melalui inner CV, tidak memakai outer validation/test.

Struktur mengikuti notebook sebelumnya: setup → upload → audit data → split → fitur → CV → evaluasi → lima Excel + ZIP.
Pilih **T4 GPU** untuk mempercepat ekstraksi BGE. Head dijalankan **CPU float32** secara sengaja untuk perbandingan yang konsisten; CPU juga dapat mengekstrak BGE, tetapi lebih lambat.
Tanpa clone dan tanpa Google Drive. Jalankan dari atas dengan **Run all**, lalu pilih `fix_dataset_300.xlsx` saat diminta.''')
    md('''## 1 · Ruang kerja dan setup

Semua pustaka dan kode ada di notebook. Worker numerik terpisah menghindari benturan NumPy kernel Colab. Torch bawaan runtime dipertahankan. Data, cache dan checkpoint hanya pada disk lokal; unduh hasil sebelum runtime berakhir.''')
    setup=src(3)
    setup=setup.replace('AES_300_E21_Tiga_Input_GPU',NAME).replace('E21/E22', 'diagnostik BGE')
    code(setup,False);code(src(5),False);code(src(6),False)
    launch=src(7).replace('GPU Colab — pilih T4 GPU; CUDA diperiksa sebelum model dimuat',
                         'BGE: GPU jika tersedia; semua head: CPU float32').replace("'AES300_REQUIRE_GPU':'1'", "'AES300_REQUIRE_GPU':'0'")
    code(launch,False)
    md('''## 2 · Konfigurasi yang dikunci sebelum perbandingan

LR, dropout, jumlah update dan aturan early stopping MLP sama untuk tiga kondisi. Head selalu baru dengan inisialisasi yang sama pada seed/fold yang sama. Tiga seed adalah pengulangan, bukan ensemble. Ridge deterministik dihitung sekali; hasilnya tidak disalin seolah tiga pengulangan independen.

`PRIMARY_TOLERANCE=2` hanya memilih sumbu presisi untuk tabel Pareto. Tetap dilaporkan ±1, ±2, ±5 dan ±10; tidak ada klaim satu model menang semua ukuran. **Test disisihkan dan tidak dievaluasi dalam notebook diagnosis ini.**''')
    imports=src(9).removeprefix('%%aes\n')
    imports=imports[:imports.index("SCENARIOS=")]
    code(imports+"\nNOTEBOOK_CODE_SHA256='__READOUT_CODE_HASH__'\nSCOPE='bge_frozen_readout_diagnostic'\nREQUIRED=['Email address','response_number','response_text','soal','jawaban_referensi','rater']")
    code('''# Ubah hanya di sini sebelum mulai; jangan memilih parameter berdasarkan test.
EPOCHS, FOLDS = 25, 5
SEEDS = [42, 123, 2026]
HEAD_LR = 1e-4
RIDGE_ALPHAS = [.01, .1, 1., 10., 100.]
PRIMARY_TOLERANCE = 2
BGE_DEVICE = 'auto'
EXPECTED_ROWS = 300
cfg = dict(dataset=os.environ['AES300_DATASET'], output=str(ROOT/'outputs/bge_readout'),
    sheet='Sheet1', expected_rows=EXPECTED_ROWS, score_min=0, score_max=100,
    test_fraction=.2, folds=FOLDS, split_seed=2026, split_candidates=128,
    epochs=EPOCHS, early_stopping_patience=5, effective_batch=8, micro_batch=1,
    warmup_ratio=.1, gradient_clip=1., dropout=.1, weight_decay=.01,
    bge_max_length=1024, bge_overlap=128, bge_batch=4, device=BGE_DEVICE,
    context_max_tokens=192, question_max_tokens=64, input_format='single_sequence',
    bge=dict(checkpoint='BAAI/bge-m3', revision='5617a9f61b028005a4858fdac845db406aefb181'))
diagnostic = dict(seeds=SEEDS, lr=HEAD_LR, alphas=RIDGE_ALPHAS, primary_tolerance=PRIMARY_TOLERANCE)
if EPOCHS < 1 or FOLDS < 3:
    raise ValueError('Perlu epoch positif dan sedikitnya3fold untuk inner CV Ridge.')
torch.set_num_threads(2)
print('Ekstraksi BGE:',BGE_DEVICE,'| Head: CPU float32 | Data test dikunci',flush=True)''')
    md('''## 3 · Baca dan audit dataset

Enam header seperti eksperimen sebelumnya. Email/duplikat hanya membentuk kelompok agar tidak menyeberang partisi, bukan input model. Nilai kosong atau tidak sah menghentikan proses; tidak diisi otomatis.''')
    code(definitions(src(14)))
    code(src(11).removeprefix('%%aes\n'))
    code('''bundle = read_dataset(cfg)
output = run_folder(cfg, bundle)
show(pd.DataFrame([bundle.summary]), 'Audit dataset asli')
show(pd.DataFrame([dict(row_id=r['row_id'], soal=r['question'], referensi=r['reference'],
                       respons=r['text'], rater=r['score']) for r in bundle.rows]), 'Tiga input dan satu label')
print('Sumber dataset SHA256:',bundle.source_sha256,flush=True)''')
    md('''## 4 · Development, test terkunci dan lima fold

Sama seperti sumber acuan: pilih split kelompok berdasarkan proporsi jumlah baris, tanpa label. Lima fold hanya di development. Test tidak memiliki prediksi pada hasil diagnosis ini. Inner CV Ridge memakai blok fold lain, dengan kelompok tetap terpisah.''')
    code(definitions(src(18)))
    code('''manifest = make_split(bundle, cfg)
dump(output/'split_manifest.json', manifest)
show(pd.DataFrame([dict(development=len(manifest['development']), test_locked=len(manifest['test']),
                       folds=len(manifest['folds']), development_groups=manifest['development_groups'],
                       test_groups=manifest['test_groups'])]), 'Partisi aktual')
print('Identitas split:',manifest['split_sha256'],flush=True)''')
    md('''## 5 · Tokenisasi dan BGE-M3 frozen

Resep fitur mengikuti single-sequence E21: soal+referensi+respons, mean CLS antar-window, lalu L2. Nilai guru tidak masuk tokenizer. BGE memakai revision terkunci, `eval()`, `requires_grad=False` dan `no_grad()`. Tidak ada RoBERTa yang dimuat dan tidak ada optimizer untuk encoder.

Cache diverifikasi dengan hash data/resep dan checksum fitur. Menjalankan ulang dengan data/resep sama memakai vektor yang sama.''')
    code(definitions(src(20), {'mean_by_essay','context_token_ids','pair_content_windows','pair_batch_windows','paired_pretrained_values'}))
    encoder=definitions(src(21))
    encoder=encoder.replace("    spec = cfg[role]", "    if role != 'bge':\n        raise ValueError('Diagnosis ini hanya memuat BGE frozen.')\n    spec = cfg[role]")
    encoder=encoder.replace("result.requires_grad_(role == 'roberta')", "result.requires_grad_(False)")
    code(encoder)
    code(definitions(src(26))+'''
bge = bge_features(bundle, cfg, output)
X = bge.numpy().astype(np.float32)
np.save(output/'bge_frozen.npy', X)
show(pd.DataFrame([dict(shape=str(X.shape), norm_min=float(np.linalg.norm(X,axis=1).min()),
    norm_max=float(np.linalg.norm(X,axis=1).max()), encoder_trained=False)]), 'Fitur yang dipakai semua kondisi')
print('BGE selesai; model encoder sudah dilepas dari memori. Selanjutnya hanya head CPU.',flush=True)''')
    md('''## 6 · Metrik prediksi asli

RMSE/MAE memakai skor kontinu asli, tanpa clipping untuk mempercantik error. QWK/Gwet AC2 memakai salinan kategori0–100 dengan tie.5ke bawah. Pearson atau agreement yang tidak terdefinisi ditandai, bukan diganti0. Jumlah dalam toleransi memiliki arti berbeda dari RMSE.''')
    code(definitions(src(34))+'\n\n'+definitions(src(35)))
    md('''## 7 · Empat kondisi pembanding

StandardScaler memakai rerata dan standar deviasi tiap komponen pada train fold; ini berbeda dari L2 per jawaban yang sudah dipakai BGE. Target-center hanya mengubah offset target dan mengembalikannya saat membaca skor. Keduanya diuji terpisah; tidak ditambahkan fitur coverage atau diganti LR di tengah eksperimen.''')
    runtime=(ROOT/'scripts/bge_readout_diagnostic.py').read_text()
    tree=ast.parse(runtime)
    groups=[{'readout_config','prepare_transform','transform_features','readout_metrics','metric_row','make_readout_head'},
            {'read_progress','write_progress','fit_readout'}, {'ridge_fold','run_readout_cv'},
            {'summarize_readouts','readout_duels','write_readout_workbook','export_readout'}]
    preamble=runtime[:runtime.index('def readout_config')]
    for i,names in enumerate(groups):
        code((preamble if i==0 else '')+definitions(runtime,names))
    code('''diagnostic = readout_config(diagnostic)
show(pd.DataFrame([dict(kondisi=k, penjelasan=DESCRIPTIONS[k]) for k in CONDITIONS]), 'Perlakuan yang terpisah')
print('Seed MLP:',diagnostic['seeds'],'| LR:',diagnostic['lr'],'| Ridge alpha:',diagnostic['alphas'],flush=True)''')
    md('''## 8 · Jalankan validasi berpasangan

Semua kondisi memakai jawaban dan fold yang sama. MLP1024→256→1 memakai AdamW, MSE target0–1, warmup10%, clipping1 dan patience5. Checkpoint dua slot menyimpan progres epoch dan RNG; saat folder masih ada, proses dapat dilanjutkan. Perubahan kode/data/config membuat folder hasil berbeda.

Untuk setiap outer fold Ridge, alpha dipilih tanpa outer validation. Tidak ada pencarian parameter dari test. Riwayat training MSE mencakup dropout dan perubahan bobot; jangan dianggap sama dengan train RMSE saat evaluasi.''')
    code("result = run_readout_cv(bundle, X, manifest, cfg, diagnostic, output/'readout_runs')\nprint('Seluruh CV selesai. Test dievaluasi:',result['test_evaluated'],flush=True)")
    md('''## 9 · Baca hasil dan perpindahan prediksi

Tabel pooled OOF mencakup seluruh development: setiap jawaban dibaca oleh head yang tidak dilatih pada jawaban/kelompoknya. Ringkasan antar-seed bukan penggabungan prediksi menjadi ensemble. Pareto menandai kandidat yang tidak dikalahkan sekaligus pada RMSE dan jumlah dekat; bukan memilih satu pemenang berdasarkan test.''')
    code('''summary = summarize_readouts(result, diagnostic['primary_tolerance'])
show(summary, 'Ringkasan empat kondisi; jumlah dekat per development, bukan persentase')
show(pd.DataFrame([r for r in result['evaluation'] if r['fold']=='POOLED_OOF']), 'OOF setiap seed')
show(readout_duels(result, diagnostic), 'Duel pada jawaban development yang sama')
show(pd.DataFrame(result['history']), 'Riwayat MLP; rincian lengkap di Excel')
print('Jika error turun tetapi jumlah dekat turun juga, catat sebagai tradeoff. Jangan menyimpulkan perbaikan S3.',flush=True)''')
    md('''## 10 · Lima Excel asli dan ZIP unduhan

Laporan mencakup dataset/split, fitur BGE/transformasi train fold, prediksi OOF, evaluasi/duel, serta riwayat. File berisi data privat. ZIP berisi lima Excel dan manifest checksum; bobot/cache/checkpoint tetap lokal dan tidak dimasukkan ke repo publik.

**Tidak ada hasil test atau hasil S1/S3 yang dibuat.** Ini menyelesaikan rencana diagnosis pertama. Refit/test dan integrasi S3 merupakan pengujian berikutnya setelah resep dipilih dari development.''')
    code("download_info = export_readout(bundle, X, manifest, cfg, diagnostic, result)\ndump(ROOT/'outputs/bge_readout_download.json', download_info)\nshow(pd.DataFrame([download_info]), 'Paket laporan privat siap')")
    code('''info=json.loads((ROOT/'outputs/bge_readout_download.json').read_text())
archive=Path(info['archive'])
if not archive.is_file():raise FileNotFoundError('Paket belum selesai; periksa cell sebelumnya.')
print('ZIP hasil:',archive)
print('Test tidak dievaluasi. Tidak ada klaim checkpoint pretrained sudah dilatih.')
if IN_COLAB:
    from google.colab import files
    files.download(str(archive))
else:
    print('Jalankan di Colab untuk unduhan browser, atau ambil file lokal di path di atas.')''',False)
    # All code is embedded, with a verifiable source identity; no runtime imports from the checkout.
    for c in cells:
        if c['cell_type']=='code':
            c['source']=c['source'].replace("NOTEBOOK_CODE_SHA256='"+base['metadata']['aes_standalone']['code_sha256']+"'", "NOTEBOOK_CODE_SHA256='__READOUT_CODE_HASH__'")
    identity=hashlib.sha256('\n'.join(c['source'].removeprefix('%%aes\n') for c in cells if c['cell_type']=='code').encode()).hexdigest()
    for c in cells:c['source']=c['source'].replace('__READOUT_CODE_HASH__',identity)
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(
        kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python'), colab=dict(provenance=[]), accelerator='GPU',
        aes_standalone=dict(code_sha256=identity, base_commit=PIN, base_path=BASE, base_sha256=hashlib.sha256(original).hexdigest()),
        aes_experiment=dict(id='BGE_FROZEN_READOUT',scope='development-only diagnostic',conditions=list(__import__('ast').literal_eval(next(n for n in ast.parse(runtime).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CONDITIONS' for t in n.targets)).value)),
                            target='rater',input_columns=['soal','jawaban_referensi','response_text'],
                            BGE_frozen=True,test_evaluated=False,drive_mount=False)),cells=cells)
    for i,c in enumerate(cells):c['id']=hashlib.sha256((str(i)+c['source']).encode()).hexdigest()[:12]
    data=json.dumps(notebook,ensure_ascii=False,indent=1)+'\n'
    for directory in ('notebooks/experiments','aes-learning/notebooks/experiments'):
        path=ROOT/directory/(NAME+'.ipynb');path.parent.mkdir(parents=True,exist_ok=True);path.write_text(data)
    print(NAME,len(cells),'cells',identity)
    return notebook

if __name__=='__main__':build()
