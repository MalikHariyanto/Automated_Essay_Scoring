"""Build a separate standalone Ridge experiment without modifying the diagnostic."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

from build_bge_readout_notebook import definitions

ROOT = Path(__file__).resolve().parents[1]
PIN = '3201b70590d4838e3b94ab4a8021c616e9a1f3c6'
BASE = 'notebooks/experiments/AES_300_BGE_Frozen_Readout_Colab.ipynb'
NAME = 'AES_300_BGE_M3_Ridge_Tiga_Input_Colab'
PLACEHOLDER = '__RIDGE_CODE_HASH__'


def build():
    original = subprocess.check_output(['git', 'show', PIN + ':' + BASE], cwd=ROOT)
    base = json.loads(original)
    old_hash = base['metadata']['aes_standalone']['code_sha256']
    cells = []
    def md(source):
        cells.append(dict(cell_type='markdown', metadata={}, source=source))
    def code(source, magic=True):
        cells.append(dict(cell_type='code', metadata={}, source=('%%aes\n' if magic else '') + source,
                          execution_count=None, outputs=[]))
    def prior(i):
        source = ''.join(base['cells'][i]['source']).removeprefix('%%aes\n')
        return source.replace(old_hash, PLACEHOLDER).replace('AES_300_BGE_Frozen_Readout_Colab', NAME).replace('diagnostik BGE', 'BGE Ridge tiga input')
    md('''# BGE-M3 frozen + Ridge · tiga input · CV dan test

**Input:** soal, jawaban referensi, respons siswa. **Target:** satu skor guru `rater`0–100.
Eksperimen khusus BGE→Ridge: tidak menjalankan S1/S3 atau MLP. Ini lanjutan diagnosis readout, sekarang mencakup pemilihan alpha development, final refit dan evaluasi test.

Setup → upload Excel → audit → split → fitur frozen → nested CV → final fit → test → lima Excel dan ZIP.
Pilih **T4 GPU** untuk ekstraksi. Ridge dijalankan di CPU. Tanpa Google Drive, clone repo, atau file kode tambahan.
Jalankan dari atas dengan **Run all**, pilih `fix_dataset_300.xlsx`, lalu tunggu unduhan hasil.

**Belum ada klaim hasil penelitian dari notebook ini.** Angka baru tersedia setelah dijalankan pada dataset asli.''')
    md('''## 1 · Setup, ruang kerja dan upload

Worker numerik terpisah mempertahankan Torch CUDA Colab dan mencegah konflik NumPy kernel. Data/cache/model hanya berada di disk runtime lokal; unduh ZIP sebelum runtime berakhir.''')
    for i in (2, 3, 4, 5):
        source = prior(i).replace('semua head: CPU float32', 'Ridge: CPU float64')
        code(source, False)
    md('''## 2 · Parameter sebelum melihat hasil

Resep BGE mengikuti single-sequence E21: tiga teks, budget konteks192token (soal maksimal64), window1024 dengan overlap128, mean CLS antar-window lalu L2. Revision model dikunci.
Tidak memakai StandardScaler/target-centering. Ridge mempunyai intercept yang dipelajari hanya dari train dan alpha penalti L2. Tidak memiliki learning rate atau epoch.

Grid alpha adalah kandidat eksperimen yang diumumkan sebelum run, **bukan angka optimal dari paper**. Pemilihan meminimalkan pooled RMSE development; hasil ±1/±2/±5 tetap dilaporkan terpisah.''')
    source = prior(7).replace("SCOPE='bge_frozen_readout_diagnostic'", "SCOPE='bge_m3_ridge_three_input_cv_and_test'")
    source = source.replace('AutoModel, AutoTokenizer, get_linear_schedule_with_warmup', 'AutoModel, AutoTokenizer')
    code(source)
    code('''# Tentukan sebelum run; jangan ubah berdasarkan skor test historis.
FOLDS = 5
RIDGE_ALPHAS = [.01, .1, 1., 10., 100.]
BGE_DEVICE = 'auto'
EXPECTED_ROWS = 300
cfg = dict(dataset=os.environ['AES300_DATASET'], output=str(ROOT/'outputs/bge_ridge_three_input'),
    sheet='Sheet1', expected_rows=EXPECTED_ROWS, score_min=0, score_max=100,
    test_fraction=.2, folds=FOLDS, split_seed=2026, split_candidates=128,
    ridge_alphas=RIDGE_ALPHAS, bge_max_length=1024, bge_overlap=128, bge_batch=4,
    device=BGE_DEVICE, context_max_tokens=192, question_max_tokens=64, input_format='single_sequence',
    bge=dict(checkpoint='BAAI/bge-m3', revision='5617a9f61b028005a4858fdac845db406aefb181'))
if FOLDS < 3:
    raise ValueError('Nested CV memerlukan sedikitnya3fold.')
torch.set_num_threads(2)
print('Tiga input: soal + referensi + respons | BGE frozen | Ridge float64 CPU', flush=True)
print('Alpha:', RIDGE_ALPHAS, '| Seleksi: development saja', flush=True)''')
    md('''## 3 · Dataset asli dan satu label

Header wajib: `Email address`, `response_number`, `response_text`, `soal`, `jawaban_referensi`, `rater`.
Email dan duplikat membentuk kelompok pencegah kebocoran, tidak menjadi fitur. Label harus lengkap dan integer0–100; data tidak dipotong atau diisi otomatis.''')
    source = prior(10)
    # Remove obsolete four-trait audit prose; the actual single-label reader stays unchanged.
    source = source.replace("'source_sha256': source_hash, 'four_trait_labels_available': False,\n               'four_rubric_report': 'Arsitektur empat output memungkinkan, tetapi empat skor tidak dapat ditentukan dari satu skor guru. Perlu anotasi manusia per dimensi; skor tidak disalin/dipecah otomatis.',", "'source_sha256': source_hash,")
    code(source)
    code(prior(11))
    code(prior(12))
    md('''## 4 · Split dan lima fold yang dapat dicocokkan

GroupShuffleSplit seed2026,128kandidat; pilih proporsi jumlah baris terdekat80:20 tanpa melihat label. GroupKFold5 hanya di development. Resep ini sama dengan keluarga E21–E32; keanggotaan aktual wajib diverifikasi ketika membandingkan ZIP, terutama jika file dataset berubah.

Pada dataset300 yang sama, targetnya development240 dan test60. Notebook menampilkan jumlah **aktual**. Test ini dipakai ulang untuk perbandingan historis dan bukan holdout baru yang belum pernah dilihat dalam penelitian.''')
    code(prior(14))
    code(prior(15).replace('test_locked=', 'test='))
    md('''## 5 · Ekstraksi tiga teks dengan encoder frozen

Soal dan referensi membentuk konteks, disambung dengan window respons. BGE memakai `eval()`, `requires_grad=False`, `no_grad()`; skor guru tidak masuk encoder. Mean CLS antar-window→L2 menghasilkan1024komponen.

Cache mempunyai identitas data/resep/checksum. Budget konteks merupakan resep acuan yang dikendalikan untuk perbandingan, bukan klaim bahwa paper menetapkannya optimal untuk AES. Tidak menambah fitur baru.''')
    code(prior(17))
    code(prior(18).replace('Diagnosis ini hanya memuat BGE frozen.', 'Eksperimen ini hanya memuat BGE frozen.'))
    code(prior(19).replace('Fitur yang dipakai semua kondisi', 'Fitur tiga input untuk Ridge').replace('Selanjutnya hanya head CPU.', 'Selanjutnya Ridge CPU float64.'))
    md('''## 6 · Metrik tanpa menyembunyikan error

RMSE, MAE, bias, median/p90/p95/max absolute error serta jumlah prediksi dalam±1,±2,±5,±10poin memakai skor kontinu asli tanpa clipping. QWK/Gwet AC2 menggunakan salinan kategori0–100, tie.5ke bawah; Pearson atau agreement undefined diberi alasan.

RMSE menekankan error besar. Hitungan toleransi menghitung jumlah jawaban dekat. Keduanya diperlukan dan tidak otomatis memilih model yang sama.''')
    code(prior(21))
    md('''## 7 · Ridge dan pemilihan alpha yang terpisah

Untuk setiap outer fold, alpha dipilih pada empat blok fold lain; outer validation tidak ikut seleksi. Hasil outer digabung menjadi nested OOF development. Setelah itu, alpha final dipilih melalui lima fold development dan model baru dipasang pada seluruh development.

Kontrak alpha final dan koefisien disimpan **sebelum** menghitung skor test. Nilai CV yang digunakan memilih alpha bukan skor evaluasi independen. Matriks fitur disimpan float32 seperti acuan, perhitungan Ridge float64 untuk kestabilan numerik.''')
    old_runtime = subprocess.check_output(['git', 'show', PIN + ':scripts/bge_readout_diagnostic.py'], cwd=ROOT).decode()
    code(definitions(old_runtime, {'write_readout_workbook'}))
    runtime = (ROOT/'scripts/bge_ridge_three_input.py').read_text()
    preamble = runtime[:runtime.index('def validate_ridge_inputs')]
    for i, names in enumerate(({'validate_ridge_inputs','select_ridge_alpha'},
                               {'ridge_metric_row','ridge_prediction_rows'},
                               {'run_ridge_three_input'}, {'export_ridge_three_input'})):
        code((preamble if i == 0 else '') + definitions(runtime, names))
    md('''## 8 · Jalankan nested CV, final refit dan test

Tidak ada pemilihan seed terbaik, epoch terbaik, atau alpha terbaik berdasarkan test. Skor train ditandai **optimistis**; baseline test memakai rerata skor development saja.
Notebook tidak mengubah kode/hasil MLP lama dan tidak mengasumsikan Ridge pasti menang.''')
    code("result = run_ridge_three_input(bundle, X, manifest, cfg, output)\nprint('Alpha final terkunci:', result['contract']['selected_alpha'], '| Test dievaluasi:', result['test_evaluated'], flush=True)")
    md('''## 9 · Hasil untuk dibandingkan dengan S2

Pisahkan tabel OOF development dan test. Perbandingan dengan S2 E21–E32 memakai **test lawan test pada jawaban yang sama**. Periksa ketiga teks, label, keanggotaan test dan fold; kesamaan ID saja tidak cukup.

Catat pengecualian: E22 menggunakan format konteks berbeda; E29/E31 memakai LR MLP S2 yang berbeda. S2 dengan prediksi identik merupakan kontrol berulang, bukan replikasi independen. Selisih terhadap E22 belum mengisolasi pengaruh mengganti MLP menjadi Ridge.

Unggah ZIP hasil baru untuk audit berpasangan dengan hasil lama. Notebook ini tidak mengarang angka pembanding dari ringkasan ingatan.''')
    code('''show(pd.DataFrame([r for r in result['evaluation'] if r['partition']=='DEVELOPMENT_POOLED_NESTED_OOF']), 'Development: nested pooled OOF')
show(pd.DataFrame([r for r in result['evaluation'] if r['partition']=='TEST_HISTORICAL_HOLDOUT']), 'Test: final Ridge tiga input')
show(pd.DataFrame(result['evaluation']), 'Evaluasi lengkap; train optimistis dan baseline diberi label terpisah')
show(pd.DataFrame(result['alpha_search']), 'Seleksi alpha; bukan evaluasi test')
show(pd.DataFrame([r for r in result['predictions'] if r['partition']=='TEST_HISTORICAL_HOLDOUT']), 'Prediksi test per jawaban; tidak dibulatkan untuk menghitung error')''')
    md('''## 10 · Lima Excel, model final dan unduhan ZIP

Paket berisi dataset/split, fitur/koefisien, prediksi OOF dan test dalam sheet terpisah, evaluasi/seleksi alpha, pengaturan/referensi. Juga disertakan model NPZ tanpa pickle, split JSON, kontrak alpha sebelum test dan manifest checksum. **File hasil privat.**

Dasar metode: [BGE-M3, Chen et al.2024](https://arxiv.org/abs/2402.03216), [Ridge, Hoerl & Kennard1970](https://doi.org/10.1080/00401706.1970.10488634), [bias seleksi model, Cawley & Talbot2010](https://www.jmlr.org/papers/v11/cawley10a.html).
Paper mendukung komponen metode dan pemisahan evaluasi; tidak menjamin error di bawah5poin atau menetapkan alpha/budget token optimal untuk data ini.''')
    code("download_info = export_ridge_three_input(bundle, X, manifest, cfg, result)\ndump(ROOT/'outputs/bge_ridge_download.json', download_info)\nshow(pd.DataFrame([download_info]), 'Paket hasil siap diunduh')")
    code('''info=json.loads((ROOT/'outputs/bge_ridge_download.json').read_text())
archive=Path(info['archive'])
if not archive.is_file():
    raise FileNotFoundError('Paket belum selesai; periksa cell sebelumnya.')
print('ZIP hasil:', archive)
if IN_COLAB:
    from google.colab import files
    files.download(str(archive))
else:
    print('Ambil file lokal di path di atas.')''', False)
    identity = hashlib.sha256('\n'.join(c['source'].removeprefix('%%aes\n') for c in cells if c['cell_type']=='code').encode()).hexdigest()
    for i, cell in enumerate(cells):
        cell['source'] = cell['source'].replace(PLACEHOLDER, identity)
        cell['id'] = hashlib.sha256((str(i) + cell['source']).encode()).hexdigest()[:12]
    notebook = dict(nbformat=4, nbformat_minor=5, metadata=dict(
        kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        language_info=dict(name='python'), colab=dict(provenance=[]), accelerator='GPU',
        aes_standalone=dict(code_sha256=identity, base_commit=PIN, base_path=BASE,
                            base_sha256=hashlib.sha256(original).hexdigest()),
        aes_experiment=dict(id='BGE_M3_RIDGE_3INPUT', scope='nested development CV, final refit, historical test',
                            target='rater', input_columns=['soal','jawaban_referensi','response_text'],
                            BGE_frozen=True, test_evaluation_enabled=True,
                            research_results_included=False, drive_mount=False)), cells=cells)
    data = json.dumps(notebook, ensure_ascii=False, indent=1) + '\n'
    for directory in ('notebooks/experiments', 'aes-learning/notebooks/experiments'):
        (ROOT/directory/(NAME+'.ipynb')).write_text(data)
    print(NAME, len(cells), 'cells', identity)
    return notebook


if __name__ == '__main__':
    build()
