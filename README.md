# Belajar AES: dari satu esai ke empat output

Automated Essay Scoring (AES) mencoba memprediksi skor tulisan dari teks. Proyek Python ini memisahkan empat aspek (**trait**) dengan urutan **isi, organisasi, kebahasaan, mekanik**, masing-masing kategori 1–5. Tujuannya membantu memahami alur model sekaligus menyiapkan pipeline yang bisa menerima skor guru nanti.

Untuk memperdalam kode setelah menjalankan notebook, unduh [panduan HTML interaktif](docs/panduan_AES_interaktif.html), lalu buka file tersebut di browser. Satu file ini bisa dipakai offline: 10 bab, penjelasan 11 cell, jelajah 30 file/59 fungsi source, diagram S1/S2/S3, latihan posisi window, MSE/gradien, dan empat metrik. Latihan angka hanya ilustrasi matematika; tidak menilai esai atau mengisi CSV. Tabel penelitian tetap dibedakan dari UNTRAINED dan fixture teknis. Tema terang/gelap, pencarian fungsi, kemajuan membaca dan tampilan cetak tersedia. Source explorer dapat diperbarui dengan `python scripts/build_learning_html.py`; script memakai whitelist source publik dan tidak membaca dataset siswa.

Saat audit publikasi, input lokal masih berisi 10 teks draf tanpa skor guru lengkap dan rubrik belum disahkan. Kita bisa melihat token, representasi dan empat angka dari head yang **UNTRAINED** (belum belajar menilai). Angka tersebut belum layak menjadi nilai siswa. Test yang lulus memeriksa mekanisme kode, bukan akurasi penilaian.

Mulai dari [alur satu esai](docs/alur_model.md), [panduan mentoring: model → test → Git](docs/belajar_dan_bukti.md), dan [notebook utama](notebooks/01_belajar_alur_AES.ipynb). Bukti audit ada di [laporan verifikasi](docs/verification_report.md).

## Apa yang ada di repo publik

Repo memuat source, konfigurasi, skema input, notebook **tanpa output**, test dan dokumentasi. CSV/JSON penelitian, teks/identitas siswa, dokumen unggahan, executed notebook, output privat, cache dan bobot model **tetap lokal dan tidak ikut commit/push**. Repo publik tidak menyertakan dataset pengganti.

Path default pada `configs/default.json` menunjuk input lokal:

- `data/source/dataset_aes_multitrait_DRAF.csv`
- `data/source/rubrik_multitrait_DRAF.json`

Sediakan file asli di lokasi tersebut atau ubah path. [Skema record](schemas/essay_record.schema.json) dan [skema rubrik](schemas/rubric.schema.json) menjelaskan struktur tanpa membagikan teks/deskriptor sumber. Skema bukan rubrik pengganti; indikator dan deskriptor penelitian tetap dibaca dari JSON asli. Folder data dan hasil diabaikan Git, dan staged files diperiksa sebelum publikasi.

## Pasang dan uji kode tanpa data siswa

Dari root checkout, gunakan Python 3.12 (lingkungan yang diverifikasi: Linux CPU):

```bash
python -m venv .venv
source .venv/bin/activate
# CPU: wheel resmi agar tidak mengunduh runtime CUDA.
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e . --no-deps
export HF_HOME="$PWD/outputs/cache/huggingface"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python -m pytest -q -rA
```

**Unit test** memeriksa perilaku bagian kode dengan input kecil yang disebut **fixture**. Unit test publik memakai fixture teknis di direktori sementara, bukan teks siswa atau rubrik asli. Misalnya, test mengecek apakah token terakhir masih masuk window dan loss menolak label `[B,1]` ketika output seharusnya `[B,4]`. Test model tiruan memeriksa shapes/gradien dengan encoder kecil; itu tidak menggantikan checkpoint asli untuk hasil penelitian.

Test AC2 memanggil R `irrCAC`. Jika R belum tersedia, test referensi tersebut ditandai **skipped**, bukan diam-diam diklaim lulus. Pada Colab jalankan `bash scripts/setup_colab.sh`; pada Debian cloud tersedia `bash scripts/setup_r_local.sh`. Setelah R tersedia:

```bash
AES_RSCRIPT=/path/ke/Rscript python -m pytest -q -rA
python scripts/verify_ac2.py --require-r --rscript /path/ke/Rscript
```

Di mesin tugas yang diaudit, interpreter ada di `/workspace/aes-venv/bin/python` dan R di `/workspace/aes-r-local/bin/Rscript`. `requirements.txt` berisi versi inti yang diuji, sedangkan `requirements.lock.txt` mencatat dependency lengkap Python 3.12/Linux CPU. GPU/AMP belum diverifikasi pada mesin ini; membutuhkan PyTorch CUDA yang sesuai. Jangan menafsirkan fixture training sebagai training penelitian.

## Periksa input asli secara lokal

Setelah CSV/JSON asli tersedia, jalankan:

```bash
python -m aes.cli --csv /path/dataset_aes_multitrait_DRAF.csv \
  --rubric /path/rubrik_multitrait_DRAF.json inspect
python -m aes.cli inspect --models
```

Perintah pertama memeriksa missing label, versi/status, duplikasi dan cleaning. Semua kolom dipertahankan pada file turunan lokal `outputs/cleaning_review.csv`; `text_raw` sumber tidak ditimpa. Perintah kedua memuat checkpoint asli BGE/RoBERTa, memeriksa config/revision, menghitung token/window, dan memperlihatkan shapes. Head baru diberi label UNTRAINED. Unduhan model sekitar 2,8 GB; cache/bobot tidak masuk repo.

**Checkpoint** adalah bobot model yang disimpan. Smoke check terpisah menguji bahwa checkpoint asli dapat dimuat, forward (teks → output), backward (gradien), dan save/reload bekerja:

```bash
python scripts/smoke_real.py
```

Script ini membutuhkan CSV/JSON asli dan `outputs/model_revisions.json` dari inspeksi encoder. Backward memakai target fixture teknis 3 untuk memeriksa mekanisme; nilai itu **tidak ditulis ke CSV**, tidak menjadi penilaian siswa, dan checkpoint teknisnya hanya sementara. Bukti smoke terpisah dari jumlah unit test.

Notebook lokal/Colab memakai modul `src/aes` yang sama. Letakkan checkout lengkap pada runtime, atur ROOT dan path CSV/JSON pada cell konfigurasi. Data draf tetap bisa menjalankan cell inspeksi; cell supervised berhenti/skip dengan alasan khusus. Jangan commit notebook yang sudah berisi teks/output siswa; versi publik harus tanpa output.

Untuk Colab, buka [notebook berurutan](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/01_belajar_alur_AES.ipynb). Pilih **CPU, Runtime version 2026.07 / Python 3.12**, lalu jalankan dari atas. Clone/update, setup, pengecekan import, dan pemilih CSV/JSON sudah ada di notebook. Sesuai permintaan belajar lengkap, default `MODE="inspect"` menjalankan encoder asli S1/S2/S3, unit test, dan referensi AC2, tanpa training supervised. Unduhan encoder sekitar 2,8 GB. Untuk inspeksi ringan, ubah `RUN_ENCODERS`, `RUN_TESTS`, dan `RUN_AC2_REFERENCE` menjadi False. [Panduan cell](docs/colab_berurutan.md) menjelaskan urutan dan batas bukti.

Notebook menjalankan modul AES melalui `scripts/notebook_step.py` dalam proses baru di `outputs/notebook_env`. `scripts/setup_notebook.py` membuat lingkungan tanpa paket bawaan Colab, memasang lock CPU, menjalankan `pip check`, dan benar-benar mengimpor pustaka model. Ini menghindari konflik NumPy yang sudah dimuat di memori notebook. Log ada di `outputs/notebook_setup.log`; data sumber tidak diubah. Salinan `aes-learning/notebooks/` disamakan karena sebelumnya tersimpan dari Colab ke path tersebut.

Setup tidak bergantung pada `ensurepip`: lingkungan dibuat tanpa pip, lalu pip induk (22.3+) memakai `--python` untuk memasang pip target jika belum ada. Percobaan yang gagal setelah membuat lingkungan parsial dapat dilanjutkan dari cell 1 lalu cell 2 tanpa menghapus folder.

Notebook menampilkan tabel diagnostik empat output per skenario dengan label UNTRAINED, tabel pemeriksaan AC2 Python/R pada fixture unit test, dan tabel penelitian RMSE/Pearson/QWK/AC2 untuk 12 kombinasi skenario/trait. `src/aes/reporting.py` mengisi tabel penelitian hanya dari test proposal berlabel dengan manifest/hash dan konfigurasi ketiga skenario yang cocok serta refit selesai. Dengan data draf, nilainya `null`/— beserta alasan, bukan nol. Setelah eksperimen selesai pada data yang sama, tabel membaca hasil final tanpa merancang ulang notebook.

Untuk workbook review Excel, pada cell pertama pilih `INPUT_FORMAT="xlsx"`, isi `EXCEL_FILENAME` dengan nama file sebenarnya, dan pertahankan `EXCEL_SHEET="Untuk_CSV_Guru"`. Upload Excel dan JSON rubrik saat diminta. Nama dataset boleh berbeda; pipeline membaca tab dengan teks, seluruh metadata dan `score_*`. `Ringkasan_AI` ditampilkan terpisah sebagai usulan, tanpa dipindahkan ke target guru. Workbook asli tetap utuh; impor menghasilkan CSV turunan dan catatan hash di `outputs/inputs/`. Skor di tab `Keputusan_guru` perlu dipindahkan oleh reviewer ke `Untuk_CSV_Guru` sesuai essay_id; kedua tab tidak tersambung otomatis.

CLI memakai impor yang sama:

```bash
python scripts/import_workbook.py --workbook /path/Review_Esai_AI.xlsx \
  --sheet Untuk_CSV_Guru --output outputs/inputs/teacher_data.csv
python -m aes.cli --csv outputs/inputs/teacher_data.csv \
  --rubric /path/rubrik_multitrait_DRAF.json inspect
```

Jika teacher `score_*` masih kosong, Excel belum menyediakan label guru meskipun usulan AI berbentuk angka. Setelah penilaian dan verifikasi manusia selesai, review independensi serta manifest harus merujuk hash CSV turunan terkini. Mengimpor Excel tidak mengesahkan rubrik, mengubah status, atau membuat training/evaluasi penelitian otomatis tersedia.

Notebook juga menampilkan target guru aktual per esai secara terpisah dari usulan AI. Untuk belajar algoritme, inspeksi encoder memperlihatkan 16 potongan token/ID awal, input window dan `chunk_to_essay`, CLS tiap window, mean per esai, BGE L2, MLP S2, proyeksi S3 serta kedua bagian concat. Angka vektor berasal dari checkpoint asli; head pada inspeksi masih UNTRAINED. Penambahan fitur diagnostik tidak mengubah parameter/arsitektur/loss.

Sesudah hasil test proposal yang sah tersedia, `TRACE_SCENARIO` dan `TRACE_TRAIT` memilih contoh perhitungan pada pasangan guru/prediksi test yang sama dengan tabel akhir. `src/aes/metric_trace.py` memperlihatkan selisih dan kuadratnya untuk RMSE, deviasi mean untuk Pearson, matriks hitungan/harapan dan disagreement untuk QWK, serta bobot agreement, Pa dan Pe untuk AC2. Kategori yang tidak teramati tetap disertakan. Trace tidak memakai skor AI, tidak menciptakan label dan tidak mengevaluasi head UNTRAINED. Pada data draf, bagian ini menjelaskan pasangan apa yang masih dibutuhkan tanpa mengganti hasil penelitian dengan angka ilustrasi.

## Training dan evaluasi setelah guru memberi skor

**Training** memperbarui bobot berdasarkan pasangan teks dan skor guru. Sebelum menjalankannya, verifikasi teks lengkap, transkripsi/identitas/relasi duplikasi, empat integer guru 1–5, `rubric_version`, rater/tanggal, dan pengesahan rubrik. Isi `configs/data_review.json` terhadap hash CSV terkini. Ini catatan hasil pemeriksaan manusia, bukan flag untuk memaksa training.

```bash
python -m aes.cli train --mode learning --scenarios S1
python -m aes.cli train --mode proposal
```

Learning ialah latihan satu epoch pada bagian train fold pertama. Mode proposal menguji S1/S2/S3 dengan manifest sama, tuning A/B pada fold pertama repeat pertama, 25 fold/run per skenario, refit fresh seed42, lalu test setelah seluruh konfigurasi dan refit selesai. Semua output/log penelitian tetap lokal. Group usulan tidak membuktikan identitas; pasangan terkait harus dipetakan dan tidak boleh bocor antarbagian.

S1 memakai RoBERTa fine-tuned dan head linear. S2 memakai BGE frozen + MLP256. S3 menggabungkan RoBERTa768 dengan projection BGE256, lalu head linear empat output. BGE truncation1024 tanpa window; RoBERTa window512 termasuk special/overlap128 token isi. Loss MSE raw 1–5 berbobot sama; monitor mean QWK keempat trait jika defined, fallback validation MSE dicatat. Rincian, asumsi dan perbedaan proposal ada di [kontrak](docs/implementation_contract.md).

Evaluasi melaporkan **QWK, Gwet AC2 kuadratik, RMSE, Pearson per trait**, bukan skor total. RMSE/Pearson memakai kontinu mentah; QWK/AC2 memakai salinan kategori setelah clipping/nearest category, midpoint memilih kategori rendah. Undefined tidak diganti nol. Tidak ada output holistik atau metrik error ternormalisasi.

Sesudah checkpoint sungguh dilatih:

```bash
python -m aes.cli predict --checkpoint outputs/runs/proposal/S3/final/best.pt --file esai.txt
python -m aes.cli evaluate --checkpoint outputs/runs/proposal/S3/final/best.pt \
  --manifest outputs/manifest.json --partition test
```

Inference (memprediksi esai baru) memerlukan teks dan checkpoint, bukan skor guru. Rubrik/hash/order/tokenizer/preprocessing/revision disimpan dan diperiksa saat reload. Jika JSON lokal tidak ada, inference memakai rubrik yang tertanam dalam checkpoint; `--rubric` dapat dipakai untuk pemeriksaan tambahan. Checkpoint UNTRAINED ditolak untuk penilaian. Evaluasi tetap memerlukan label guru dan split yang sah.

## Menemukan pekerjaan dan buktinya

**Branch** adalah jalur perubahan; **commit** rekaman versi lokal; **push** mengirim rekaman itu ke GitHub. Audit awal menemukan seluruh kode masih untracked pada branch unborn `work` dan remote kosong. Karena belum ada commit/push, log test lama tidak pernah membuktikan bahwa GitHub diperbarui. Menyimpan draft lingkungan juga tidak mengirim kode ke GitHub.

Laporan [verifikasi](docs/verification_report.md) memisahkan bukti test, smoke asli, notebook, data, dan publikasi. Branch hasil yang dituju `codex/aes-learning`; tidak melakukan merge ke main. Repo kosong belum memiliki base untuk PR biasa. Keberhasilan push harus dicek dengan SHA remote dan isi file, bukan diasumsikan dari commit lokal.

Proposal/paper pernah diperiksa pada salinan lokal yang **tidak diterbitkan**. Catatan rujukannya ada di [pencocokan proposal](docs/source_reconciliation.md) dan [pemeriksaan paper](docs/paper_review.md). Path sumber tetap configurable; tidak perlu dokumen privat untuk menjalankan unit test. Kebijakan mentoring dan bukti kerja tersimpan di [AGENTS.md](AGENTS.md).

Workbook review privat dapat memuat usulan AI empat dimensi dan rekonstruksi berlabel. `scripts/export_review_excel.py` membuka `Ringkasan_AI` agar kandidat skor terlihat, menyimpan bukti/deskriptor, memisahkan tambahan di `Rekonstruksi_AI`, dan mempertahankan semua score_* guru/status sumber. Skor mekanik pada transkripsi draf memerlukan label simulasi eksplisit. Teks tidak lengkap hanya boleh mendapat usulan berangka pada versi rekonstruksi yang ditandai, bukan sebagai penilaian teks asli. Workbook dan payload review tidak dibagikan di repo publik. Notebook default terbaru memakai nama `Review_Esai_AI_Lengkap.xlsx`; ganti EXCEL_FILENAME bila nama file lokal berbeda. Usulan ini tidak menghasilkan metrik penelitian atau membuat data siap training.

## Latihan berangka dengan acuan AI

Untuk belajar sebelum skor guru tersedia, buka [02_latihan_acuan_AI.ipynb](notebooks/02_latihan_acuan_AI.ipynb). Default mode sudah `ai_learning`. Upload workbook **Review_Esai_AI_Lengkap.xlsx** dan JSON rubrik yang sama; jalankan dari atas. S1/S2/S3 benar-benar dilatih dua epoch, lalu dievaluasi terhadap usulan AI pada kelompok validation yang terpisah. Tabel berisi RMSE, Pearson, QWK, AC2 per trait, dengan sumber acuan AI yang eksplisit; bukan hasil penelitian terhadap guru. Dua teks terpotong/rekonstruksi tidak masuk latihan. Kolom skor guru dan status sumber tidak diubah.

Baca [petunjuk dan batas latihan AI](docs/latihan_acuan_AI.md), termasuk arti setiap output, split latihan 50/50 yang terpisah dari proposal, bukti pembaruan parameter dan penggunaan ulang hasil yang cocok hash.

## Buku HTML untuk membedah hasil Colab

Unduh [belajar_colab_AI_interaktif.html](docs/belajar_colab_AI_interaktif.html) lewat **Download raw file** di GitHub, lalu buka di browser. Buku standalone memiliki tab per proses, bahasa Santai/Teknis, source code penting, hasil aktual latihan AI, dan simulator window/LR/empat metrik. [Petunjuk membuka dan membaca output pribadi](docs/cara_membuka_buku_Colab.md) menjelaskan cara memilih notebook lokal tanpa mempublikasikan data siswa. Tidak ada training tambahan atau perubahan split pada tugas buku ini.

## Meja belajar paper

[Code belajar paper — HTML interaktif](docs/code%20belajar%20paper.html) memetakan 42 sumber unik ke tujuh topik dan subtopik berdasarkan tugas model. Ruang baca memiliki scroll sendiri, identitas publikasi, proses input → tindakan → output dengan sticky note, rumus beserta arti simbol, kamus kontekstual, hasil dan batas bukti, serta rute membaca halaman sumber. Tema Spider-Man full colour memakai splash cat, kota komik, pose SVG yang menimpa border kartu, serta adegan Andrew–Gwen yang mengikuti scroll. Delapan tab ruang baca kini mencakup 82 visual/cuplik PDF dengan sticky note dan perbesaran; semua animasi dapat dijeda. [Preview beranda](docs/previews/paper_library/01_beranda_desktop.png) dan [panduan penggunaan](docs/cara_membaca_paper.md) tersedia untuk review. Buku/cuplikan, tesis, laporan, dan kebijakan diberi jenis serta batas cakupan; indeks tidak ditebak. PDF/ZIP lengkap tidak diterbitkan; hanya cuplikan visual akademik terpilih dengan atribusi sesuai permintaan pengguna.

[Panduan hosting statis dengan akses email](docs/hosting_belajar_paper.md) menyediakan builder dan konfigurasi Cloudflare Workers Free. Paket sudah diperiksa lokal; deployment dan login email belum aktif karena perlu akun Cloudflare serta kebijakan Access.
