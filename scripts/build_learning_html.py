"""Perbarui source explorer HTML dari whitelist source publik; tanpa input siswa."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'docs/panduan_AES_interaktif.html'
MODULES = {
 'src/aes/io.py': ('Membaca sumber', 'Membaca CSV/JSON, menjaga urutan target, menghitung hash, dan menyimpan JSON tanpa NaN.'),
 'src/aes/audit.py': ('Audit dan kelayakan', 'Memisahkan inspeksi teks draf dari syarat training dengan label guru yang sah.'),
 'src/aes/cleaning.py': ('Cleaning minimal', 'Membersihkan artefak teknis sambil mempertahankan informasi bahasa dan mekanik.'),
 'src/aes/windows.py': ('Token dan window', 'Memotong token isi, membuat mapping chunk ke esai, lalu mengagregasi CLS secara differentiable.'),
 'src/aes/encoders.py': ('Encoder asli', 'Mengunci revision, mengekstrak dense BGE frozen, memeriksa cache, dan memuat RoBERTa trainable.'),
 'src/aes/model.py': ('Arsitektur dan loss', 'Membentuk S1/S2/S3 dengan tepat empat output dan menolak broadcasting label.'),
 'src/aes/inspection.py': ('Alur inspeksi', 'Audit sumber dan forward checkpoint asli dengan head UNTRAINED.'),
 'src/aes/trainer.py': ('Loop training', 'Micro-batch, gradien, AdamW, scheduler, validasi, early stopping, checkpoint, dan OOM.'),
 'src/aes/split.py': ('Split dan manifest', 'Holdout/fold pada esai atau group yang diverifikasi, sebelum chunking.'),
 'src/aes/experiment.py': ('Eksperimen proposal', 'Tuning, 25 fold, refit baru, penguncian seluruh skenario, lalu final test.'),
 'src/aes/metrics.py': ('Empat metrik', 'RMSE/Pearson kontinu, QWK/AC2 kategori, kondisi undefined, dan monitor validasi.'),
 'src/aes/ai_learning.py': ('Latihan acuan AI', 'Training nyata S1/S2/S3 pada usulan AI yang terpisah dari guru, holdout group 50/50, fixed epoch dan evaluasi validation.'),
 'src/aes/reporting.py': ('Tabel hasil final', '12 baris skenario/trait dengan verifikasi hash, revision, review dan ID test; nilai kosong dijelaskan.'),
 'src/aes/checkpoint.py': ('Simpan dan reload', 'Menyimpan state_dict bersama JSON asli, hash, urutan trait, preprocessing, dan revision.'),
 'src/aes/inference.py': ('Prediksi sesudah training', 'Memuat checkpoint trained dan menghasilkan empat angka serta empat kategori tanpa label guru sebagai input.'),
 'src/aes/cli.py': ('Pintu masuk CLI', 'Memilih inspect/train/evaluate/predict serta memeriksa path dan kelayakan sebelum aksi.'),
 'src/aes/script_entry.py': ('Pintasan perintah', 'Merapikan argumen path sehingga script pendek memakai CLI yang sama.'),
 'scripts/setup_notebook.py': ('Setup Python Colab', 'Lingkungan tanpa system-site-packages/ensurepip; bootstrap pip, lock CPU, pip check, dan import nyata.'),
 'scripts/notebook_step.py': ('Worker notebook', 'Menjalankan perhitungan AES di proses Python baru, terpisah dari memori kernel Colab.'),
 'scripts/verify_ac2.py': ('Referensi AC2', 'Membandingkan point estimate AC2 Python dan R pada lima fixture unit test.'),
 'scripts/smoke_real.py': ('Smoke checkpoint asli', 'Memeriksa forward/backward/reload dengan teks asli dan target teknis, bukan training penelitian.'),
 'scripts/inspect_data.py': ('CLI inspeksi', 'Pintasan menuju aes.cli inspect; modul audit dan inspection tetap sama.'),
 'scripts/train.py': ('CLI training', 'Pintasan menuju aes.cli train; guard data tetap wajib.'),
 'scripts/evaluate.py': ('CLI evaluasi', 'Pintasan menuju aes.cli evaluate dengan checkpoint dan manifest.'),
 'scripts/predict.py': ('CLI inference', 'Pintasan menuju aes.cli predict dari teks/file dan checkpoint trained.'),
 'scripts/setup_colab.sh': ('Backend R Colab', 'Setup Python AES terpisah, R jika belum tersedia, irrCAC, lalu verifikasi AC2. Khusus Colab; bukan model training.'),
 'scripts/ac2_reference.R': ('Kode referensi R', 'Membaca N×2 ratings fixture, memakai weights quadratic/categ.labels 1–5, dan memeriksa coefficient bernama AC2.'),
 'scripts/setup_r_local.sh': ('Backend R cloud Debian', 'Menggunakan indeks APT bertanda tangan dan checksum source irrCAC, dengan wrapper lokal; tidak menulis ke /usr. Berbeda dari setup Colab.'),
 'scripts/install_cloud.sh': ('Setup lingkungan tugas', 'Script untuk /workspace cloud, bukan cell Colab. Memasang lock, backend R lokal, test, dan inspeksi; tidak mengubah data sumber.'),
 'configs/default.json': ('Konfigurasi pipeline', 'Path relatif, checkpoint/revision, urutan kebijakan preprocessing/model/training/split yang benar-benar dipakai.'),
 'configs/data_review.json': ('Template review manusia', 'Reviewer/tanggal, hash CSV, keputusan independensi/duplikasi dan group. Nilai awal belum disahkan; tidak boleh diisi untuk memaksa training.'),
}
DESCRIPTIONS = {
 'sha256':'Menghasilkan sidik jari byte file. File yang berubah menghasilkan hash berbeda; ini mengikat data/rubrik/review pada run yang benar.',
 'text_hash':'Mengikat cache embedding pada isi teks yang dipakai model, bukan hanya pada essay_id.',
 'dump':'Membuat folder hasil dan menyimpan JSON UTF-8. allow_nan=False mencegah nilai tidak sah terselip sebagai angka hasil.',
 'load_config':'Mengubah path relatif menjadi path terhadap root proyek. Konfigurasi menentukan file input dan seluruh kebijakan pipeline.',
 'load_rubric':'Memeriksa urutan empat trait, kategori 1–5, deskriptor lengkap, version, confirmed, scoring_rule dan missing_rule. Tidak memberikan skor otomatis.',
 'load_data':'Membaca UTF-8 BOM, mempertahankan seluruh kolom sebagai string, dan menjaga skor kosong tetap kosong. Pemeriksaan kolom gagal dengan pesan konkret.',
 'targets':'Mengambil score_<trait> sesuai urutan JSON menjadi [N,4]. Missing dikonversi ke NaN agar eligibility menolaknya, bukan menjadi nol.',
 'duplicate_candidates':'Mendeteksi pasangan exact duplicate dan kandidat near-duplicate melalui SequenceMatcher, threshold 0.8. Hasilnya daftar untuk review, bukan bukti otomatis identitas siswa.',
 'validate_inputs_for_inspection':'Memeriksa CSV dapat diinspeksi, misalnya tidak kosong dan ID tersedia, lalu menjalankan audit. Status draf tidak otomatis menggagalkan inspeksi.',
 'audit':'Menghitung label missing/invalid/noninteger/out-of-range, versi rubrik, status, duplikasi dan EDA. Tidak mengubah sumber atau mengisi label.',
 'training_blockers':'Mengumpulkan syarat yang belum terpenuhi: empat skor, pengesahan, transkripsi/kelengkapan/identitas, rater, dan review independensi/group terhadap hash sumber.',
 'validate_inputs_for_training':'Menghentikan supervised dengan EligibilityError bila blockers ada. Jika sah, mengembalikan target [N,4] yang bisa dipakai training.',
 'clean':'Unicode NFC, newline dan karakter kendali teknis. Normalisasi kutip/dash/elipsis hanya bila konfigurasi mengaktifkan. Tanda baca dan salah ejaan default dipertahankan.',
 'content_windows':'Tokenisasi tanpa special tokens, hitung kapasitas, buat starts sampai token terakhir tercakup, lalu tambahkan special tokens sekali per window.',
 'batch_windows':'Menggabungkan window dari beberapa esai, melakukan padding, dan membuat chunk_to_essay. B label tetap jumlah esai, bukan jumlah window.',
 'mean_by_essay':'index_add menjumlahkan CLS berdasarkan mapping lalu membagi jumlah window tiap esai. Operasi tensor menjaga gradien menuju semua window.',
 'resolve_models':'Memeriksa hidden_size dan special-token accounting checkpoint/revision asli. Metadata tokenizer dan SHA dipakai kembali saat cache/reload.',
 'BGEEncoder.__init__':'Mengunduh snapshot revision BGE, memuat FlagEmbedding resmi, menetapkan eval dan requires_grad=False pada semua parameter encoder.',
 'BGEEncoder.encode':'Mengambil dense_vecs resmi tanpa graph, panjang maksimum 1024, tanpa sparse/ColBERT. Validasi [N,1024] finite lalu normalisasi L2.',
 'BGEEncoder.stats':'Menghitung token lengkap termasuk special tokens, apakah melebihi 1024, dan jumlah token yang dipotong.',
 'cached_bge':'Menyimpan satu dense per esai bersama ID/hash teks, revision, tokenizer, pooling, normalisasi dan preprocessing. Cache tidak memuat proyeksi/head trainable.',
 'load_roberta':'Memuat AutoModel revision asli, tanpa pooler karena kita memakai hidden token awal. Memeriksa hidden size 768 dan mengaktifkan gradien encoder.',
 'AESModel.__init__':'Memilih head sesuai S1/S2/S3. S1 Linear(768,4); S2 MLP 1024→256→4; S3 proyeksi 1024→256, LayerNorm, concat dan Linear(1024,4).',
 'AESModel.forward':'Mengambil CLS tiap window, merata-ratakan per esai, lalu menjalankan head skenario. return_features membuka representasi untuk belajar shapes; output wajib [B,4].',
 'loss_mse':'Memastikan prediksi dan label sama-sama [B,4] serta finite. Menghitung rata-rata kuadrat selisih mentah tanpa clipping/sigmoid/normalisasi target.',
 'inspect_data':'Memakai loader, audit dan cleaning yang sama dengan CLI, lalu menyimpan audit.json dan turunan cleaning_review.csv secara lokal.',
 'inspect_models':'Memuat encoder asli dan menjalankan ketiga head baru pada satu esai. Seed 42 untuk diagnostik; angka diberi label UNTRAINED, bukan penilaian.',
 'seed_everything':'Mengatur seed Python/NumPy/PyTorch dan kebijakan deterministik. Membantu reproduksibilitas, bukan jaminan semua hardware menghasilkan bit identik.',
 'device_name':'Memilih cuda jika tersedia, selain itu cpu. Wheel PyTorch CPU di notebook tetap memakai CPU walaupun jenis runtime T4.',
 'BatchEngine.__init__':'Mengikat model, teks, dense BGE, tokenizer, konfigurasi dan device untuk loop training/predict.',
 'BatchEngine.forward':'Memilih esai sesuai indeks. S2 memakai dense BGE; S1/S3 membuat semua window esai yang dipilih dan meneruskan mapping ke model.',
 'BatchEngine.predict':'Menetapkan eval dan no_grad, menjalankan micro-batch esai lalu menggabungkan [N,4] prediksi. Digunakan validasi/test/inference, bukan forward training.',
 'train_run':'Menginisialisasi parameter baru, membagi LR encoder/head, mengakumulasi gradien per effective batch 8 esai, melakukan step, validasi dan pemilihan checkpoint.',
 'train_with_oom_retry':'Pada CUDA OOM, log peristiwa dan kecilkan micro-batch; run diulang dengan seed/init sama. Tidak mengurangi cakupan teks atau membekukan RoBERTa.',
 'ordinal_prediction':'Clip salinan angka ke min/max, pilih kategori terdekat dengan argmin. Kategori lebih rendah dipilih pada midpoint karena urutan kategori naik.',
 'gwet_ac2':'Menghitung bobot agreement kuadratik, agreement teramati, chance agreement dari marginal gabungan, lalu AC2. Seluruh kategori dipertahankan.',
 'evaluate':'Memeriksa [N,4] dan finite; tiap trait dihitung terpisah. Menyimpan alasan Pearson/QWK undefined dan proporsi prediksi di luar rentang.',
 'selection_monitor':'Mean QWK empat trait hanya ketika semuanya finite/defined; selain itu gunakan validation MSE. Tidak menghapus trait undefined dari mean.',
 'make_manifest':'Memvalidasi data/review, melakukan holdout 80/20 dan lima repeat lima fold pada development, dengan group terkait tetap utuh. Simpan ID/hash untuk semua skenario.',
 'distribution':'Fungsi bantu menghitung distribusi kategori per trait pada development dan fold. Tidak menciptakan target holistik untuk stratifikasi.',
 'summarize':'Mengagregasi hasil CV per skenario/trait/metrik: mean, sample SD ddof=1, jumlah run selesai dan n_valid. Undefined tidak menjadi nol.',
 'prepare':'Menegakkan eligibility, manifest yang konsisten, revision asli, teks bersih dan cache BGE sebelum run supervised.',
 'run':'Learning satu epoch pada train/validation fold pertama, atau proposal: tuning A/B, 25 run, refit fresh, seluruh skenario locked, lalu held-out test.',
 'positions':'Mengubah essay_id dalam manifest menjadi posisi array teks/target/embedding agar subset tiap fold sesuai.',
 'prepare_ai_learning':'Memerlukan pilihan mode eksplisit, workbook/hash rubrik yang cocok dan usulan AI sah; menolak teks berubah serta duplikasi lintas kelompok, mengecualikan teks terpotong/rekonstruksi dan membuat split kelompok latihan.',
 'collect_report':'Menghitung empat metrik per trait serta langkah perhitungan dari acuan AI dan prediksi validation. Semua output diberi scope latihan AI, bukan guru.',
 'run_ai_learning':'Memuat checkpoint asli, melatih ketiga skenario dari awal selama epoch tetap tanpa validation tuning, memeriksa perubahan parameter, reload, memprediksi validation dan menyimpan laporan privat. Hasil lama hanya dipakai ulang bila hash cocok.',
 'final_report':'Menampilkan 12 baris empat metrik. Bila data belum layak, semua nilai null. Hasil test hanya dibaca bila hash/review/revision, ID test dan penguncian/refit ketiga skenario sah.',
 'save_checkpoint':'Menyimpan bobot dan metadata: JSON rubrik asli byte/hash, trait order, config, model revision dan keadaan training. Cache BGE berbeda dari checkpoint trainable.',
 'load_checkpoint':'Memeriksa metadata rubrik/tokenizer/preprocessing/revision, merekonstruksi arsitektur, memuat bobot secara strict dan menetapkan eval.',
 'predict':'Menolak checkpoint untrained, checkpoint latihan AI dan teks kosong; membersihkan teks, menjalankan jalur skenario, mengembalikan empat continuous dan empat categories. Label guru tidak diperlukan sebagai input.',
 'main':'Pintu masuk command line: membaca argumen dan menjalankan aksi sesuai pilihan. Error data atau dependency ditampilkan, tidak disembunyikan sebagai keberhasilan.',
 'run_command':'Memindahkan opsi path ke posisi argumen global agar script singkat menggunakan aes.cli yang sama.',
 'check_runtime':'Menolak Python selain 3.12 sebelum unduhan setup. Di Colab pilih runtime 2026.07.',
 'create_environment':'Membuat venv tanpa ensurepip dan tanpa paket sistem. Folder parsial dengan pyvenv.cfg dipertahankan; folder lain tidak ditimpa.',
 'ensure_pip':'Jika pip target belum tersedia, pip induk --python memasang pip 25.0.1 ke target. Kegagalan bootstrap tetap menghentikan setup.',
 'setup':'Memasang lock CPU dalam lingkungan terpisah, editable source, pip check dan import nyata. Marker menyimpan hash lock dan versi yang terverifikasi.',
 'audit_step':'Worker untuk audit dan preview satu esai/rubrik/cleaning. Hasil privat hanya ditulis di outputs lokal.',
 'supervised_step':'Memeriksa eligibility; data draf menghasilkan pesan skip. Learning/proposal tetap memerlukan guru; ai_learning memanggil modul latihan terpisah setelah pilihan eksplisit.',
 'models_step':'Worker untuk forward asli. Gangguan akses checkpoint dicatat sebagai status; error implementasi tetap gagal.',
 'verify':'Lima fixture unit test AC2 ditulis ke direktori sementara, dikirim ke R/irrCAC, lalu point estimate dibandingkan dengan Python dalam toleransi pembulatan.',
}

def functions(tree):
    result=[]
    def walk(node, parent=''):
        for item in ast.iter_child_nodes(node):
            if isinstance(item,(ast.FunctionDef,ast.AsyncFunctionDef)):
                name=(parent+'.' if parent else '')+item.name
                description=DESCRIPTIONS.get(name,DESCRIPTIONS.get(item.name))
                if not description:raise ValueError('Penjelasan fungsi belum tersedia: '+name)
                result.append({'name':name,'line':item.lineno,'end_line':item.end_lineno,'description':description})
                walk(item,name)
            elif isinstance(item,ast.ClassDef):walk(item,item.name)
            else:walk(item,parent)
    walk(tree)
    return result


def build():
    version=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    files=[]
    for path,(title,purpose) in MODULES.items():
        source=(ROOT/path).read_text()
        files.append({'path':path,'title':title,'purpose':purpose,'source':source,
                      'functions':functions(ast.parse(source)) if path.endswith('.py') else [], 'sha256':hashlib.sha256(source.encode()).hexdigest()})
    nb=json.loads((ROOT/'notebooks/01_belajar_alur_AES.ipynb').read_text())
    cells=[];title=''
    for cell in nb['cells']:
        source=''.join(cell['source'])
        if cell['cell_type']=='markdown':title=source.splitlines()[0].lstrip('# ').strip()
        elif cell['cell_type']=='code':cells.append({'number':len(cells)+1,'title':title,'source':source})
    payload={'version':version,'files':files,'cells':cells,
             'config':json.loads((ROOT/'configs/default.json').read_text()),
             'evidence':json.loads((ROOT/'docs/verifikasi/full_learning.json').read_text())['notebook']}
    encoded=json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')
    html=TARGET.read_text()
    html=re.sub(r'<!-- SOURCE_PAYLOAD_BEGIN -->.*?<!-- SOURCE_PAYLOAD_END -->',
                lambda _: '<!-- SOURCE_PAYLOAD_BEGIN -->\n<script id="source-data" type="application/json">'+encoded+'</script>\n<!-- SOURCE_PAYLOAD_END -->',html,flags=re.S)
    TARGET.write_text(html,encoding='utf-8')
    print('HTML source explorer:',len(files),'files;',sum(len(f['functions']) for f in files),'functions;',len(cells),'notebook cells')

if __name__=='__main__':build()
