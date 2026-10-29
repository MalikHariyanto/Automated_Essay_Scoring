# Panduan AI: konteks proyek, keputusan, dan urutan eksperimen

Dokumen serah-terima untuk AI pada akun atau sesi baru, disusun dari keputusan penting percakapan proyek sampai **28 Oktober 2026**. Ini sintesis, bukan transkrip. Pertanyaan belajar dicatat jika memengaruhi pemahaman metode atau implementasi; salah ketik, percakapan sementara, dan permintaan status berulang tidak dijadikan keputusan penelitian.

**Baca berurutan:** dokumen ini → [status numerik/config aktif](STATUS_PROYEK.md) → [katalog notebook](notebooks/experiments/README.md) → [review terbaru](docs/review/review_e25_e26_study.md). Untuk meneruskan pekerjaan, gunakan instruksi pengguna terbaru; jangan menganggap semua rencana lama masih berlaku.

## 1. Tujuan dan cara bekerja dengan pengguna

Pengguna mengembangkan Automated Essay Scoring Bahasa Indonesia untuk penelitian dan presentasi ke dosen. Ia membutuhkan source Colab yang bisa dijalankan, analisis hasil yang jujur, dan buku HTML untuk mempelajari hubungan kode–fitur–prediksi. Ia sedang belajar coding: jelaskan dengan bahasa Indonesia sederhana, boleh memakai gw/lo, dan hubungkan istilah dengan bagian source yang nyata.

Utamakan hasil konkret. Jangan berhenti pada “bisa”, janji mengerjakan, atau pertanyaan konfirmasi berulang ketika scope sudah jelas. Jawab ringkas dahulu; rincian teknis ditempatkan pada dokumen/kode yang ditautkan. Jangan mengarang nama fungsi, cell, metrik, isi lampiran, hasil test, atau paper. Jika kode/ZIP belum dibaca, katakan batasnya.

Untuk menjelaskan kode, pengguna menyukai kalimat singkat sekitar 5–10 kata disertai rincian opsional. Padanan screenshot kode temannya harus ditautkan ke cell/baris notebook miliknya, bukan diterangkan sebagai arsitektur yang sama. Screenshot teman memakai model lain seperti mDeBERTa; proyek ini memakai RoBERTa/BGE.

## 2. Kontrak yang berlaku sekarang

Keluarga aktif **E21–E30** memakai tiga fitur teks pada **seluruh S1/S2/S3**; hasil penelitian tersedia sampai E28, sementara E29/E30 adalah notebook lanjutan:

- `soal`: pertanyaan;
- `jawaban_referensi`: kunci/acuan jawaban;
- `response_text`: jawaban siswa;
- **`rater`: satu label guru 0–100**, bukan input ke tokenizer.

Tiga input tidak berarti tiga encoder atau tiga label. S1 = RoBERTa full fine-tuning + head linear; S2 = BGE frozen + MLP; S3 = RoBERTa fine-tuned + BGE frozen + representational fusion + MLP. Detail hyperparameter dan angka ada di [status](STATUS_PROYEK.md).

Ada workflow empat trait/rubrik 1–5 serta latihan dengan acuan AI dalam arsip repo. Itu bukan alasan mengganti label tunggal `rater` pada E21–E26. Jangan membagi satu skor menjadi label trait, menganggap label AI sebagai nilai guru, atau mencampur input lama dengan keluarga tiga input. Perubahan kontrak memerlukan instruksi pengguna baru.

## 3. Perjalanan eksperimen sebelum tiga input

Urutan ini menjelaskan mengapa percobaan berikutnya muncul, bukan klaim bahwa semua perubahan memperbaiki hasil. Angka di bawah adalah RMSE test S3 pada kelompok hasil lama; jangan langsung menjadikannya peringkat lintas kelompok input/label tanpa audit pairing.

| Tahap | Perubahan dan keputusan | Hasil/batas penting |
| --- | --- | --- |
| E0 · awal | Head regresi linear, anggaran 5 epoch, 5 fold | RMSE S3 11,7224; ini bukan 25 fold |
| E1 · revisi training | Source revisi 25 epoch tersedia pada perbandingan | Tidak diisi hasil penelitian jika hanya source yang tersedia |
| E2 · MLP acuan | S1 tetap linear; S2/S3 MLP; maksimal25epoch, patience5 | RMSE S3 6,7085; S1 6,3805. Source penting commit `15bc7bb` |
| E3 · double frozen | Bekukan RoBERTa dan BGE pada S3; MLP/proyeksi tetap belajar | RMSE S3 7,8611; pembekuan belum memperbaiki run tersebut |
| E4 · konteks pair | Uji pasangan soal/referensi dengan respons sesuai versi lama | RMSE S3 6,7315. Format pair tidak otomatis memberi hasil lebih baik |
| E5 · LoRA + MLP | Fine-tuning parameter efisien pada RoBERTa S3 | RMSE S3 6,5305, tetapi mean CV S3 6,2509 lebih tinggi daripada acuan 6,0516 |
| E6 · LR encoder | S3 encoder5e-6; S1 tetap1e-5, head/proyeksi1e-4 | RMSE S3 6,3744; mean CV5,9915. Pada varian reuse, S1/S2 memakai hasil lama |
| E6 2.0 / E7 tiga seed | Ulang seluruh S1/S2/S3 pada seed42,123,2026 | Menunjukkan variasi hasil. Mean metrik tiga seed bukan otomatis ensemble |
| E8 | Context pair + encoder S3 LR5e-6 pada versi sebelum E21 | RMSE S3 6,3818, mean CV5,8836; input tidak otomatis sama dengan E21 |

Sumber: [awal/revisi](docs/snapshots/mlp_comparison_snapshot.json), [MLP acuan](docs/snapshots/mlp_study_snapshot.json), [frozen](docs/snapshots/mlp_frozen_snapshot.json), [pair](docs/snapshots/mlp_context_snapshot.json), [LoRA](docs/snapshots/mlp_lora_snapshot.json), [E6](docs/snapshots/mlp_lr_snapshot.json), [tiga seed](docs/snapshots/mlp_multiseed_snapshot.json), [E8](docs/snapshots/mlp_e8_snapshot.json).

Penomoran dalam diskusi tidak selalu berurutan. E6 2.0 dan E7 muncul untuk pembahasan tiga seed; versi tiga input CPU E6/E4 juga pernah dibuat sebelum E21. Jangan mengarang hasil E9–E20 atau menyamakan nomor tanpa memeriksa nama source dan snapshot. [Katalog](notebooks/experiments/README.md) membedakan varian CPU/GPU, fresh training, dan reuse.

MLP acuan sempat ditetapkan sebagai bahan semi-final presentasi. Eksperimen lanjutan dibuat terpisah; jangan menimpa source acuan untuk menyamakan hasil baru. Pengguna menerima LoRA sebagai salah satu bentuk fine-tuning, tetapi E21–E26 tetap memakai full fine-tuning RoBERTa.

Selisih RMSE test E6 S3 terhadap S1 acuan hanya sekitar0,0061poin. Jangan menyebut selisih satu run sekecil itu sebagai keunggulan statistik yang sudah terbukti.

### Alasan perubahan selanjutnya

Pengguna mempertanyakan mengapa konteks belum menjadi input pada seluruh skenario. Permintaan akhirnya menetapkan tiga teks dan `rater` secara eksplisit, lalu lahir E21/E22. Perbandingan keluarga baru ditempatkan dalam buku tersendiri agar perubahan informasi input tidak tersembunyi. Jangan mengambil kesimpulan bahwa LR/LoRA/layer tunggal menyebabkan seluruh perbedaan ketika kondisi run juga berubah.

## 4. Keluarga tiga input dan hasil terbaru

| Eksperimen | Susunan dan layer S3 | Hubungan dengan sebelumnya |
| --- | --- | --- |
| E21 | Single sequence tiga teks; CLS → norm/proyeksi → concat → MLP | Acuan tiga input |
| E22 | Teks A=soal+referensi; teks B=respons, separator pasangan | Menguji format dari acuan, bukan menambah label |
| E23 | Token respons → BiGRU64/arah → state akhir128 → proyeksi768 | Menguji ringkasan urutan |
| E24 | Token respons768 → attention768→64→1 → weighted sum768 | Menguji bobot token tanpa BiGRU |
| E25 | Attention E24 + gate skalar dua cabang sebelum concat | Lanjutan arsitektur E24, bukan resume checkpoint E24 |
| E26 | Token respons768 → BiGRU → urutan128 → attention → ringkasan128 → proyeksi768 | Menggabungkan ide BiGRU dan attention; bukan attention atas satu state akhir |

Pada E23–E26, perubahan layer khusus S3; S1/S2 tetap kontrol. BGE frozen menghasilkan1024 fitur, diproyeksikan menjadi256 pada S3; ringkasan RoBERTa768 digabung menjadi1024 lalu MLP1024→256→1.

E25 menggunakan `g=σ(MLP([RoBERTa768,BGE256]))`, concat `[2g×RoBERTa,2(1−g)×BGE]`. Inisialisasi g=0,5 menjaga kedua pengali=1. Gate final pada300jawaban berkisar0,495721–0,496997: pada run ini perubahan skala kecil, bukan pembuktian bahwa gating selalu gagal atau encoder tertentu tidak berguna.

**Hasil saat ini:** E26 RMSE/MAE test S3 terendah (5,7435/4,4797) dan mean CV terendah (5,9245). E23 paling sering terdekat dalam persaingan enam eksperimen (13/60) dan terbanyak selisih≤5poin (39/60). Lihat [tabel lengkap](STATUS_PROYEK.md#eksperimen-dan-hasil).

Kode utama meja belajar **E23**, mengikuti aturan jumlah terdekat lalu RMSE bila seri. E26 mempunyai tombol tersendiri sebagai RMSE terbaik. Jangan mengganti penjelasan menjadi “satu model menang semua ukuran”. E24 vs E26 sama-sama menang30jawaban dalam duel; E25–E26 adalah31–29. Jumlah kemenangan tidak memperhitungkan besarnya kesalahan dengan cara yang sama seperti RMSE.

## 5. Interpretasi yang harus konsisten

- Prediksi adalah regresi kontinu. Tidak ada skor mentah yang persis sama dengan guru pada suatu run bukan alasan otomatis menyatakan model gagal. Baca besarnya residual, toleransi, dan kebutuhan penilaian guru.
- MSE training memakai target0–1. MSE0,5 bukan5%: akarnya sekitar0,7071; pada rentang100 setara70,71poin. MSE test memakai poin², RMSE/MAE memakai poin.
- RMSE10poin bukan jaminan seluruh prediksi berada dalam ±10poin. RMSE lebih peka pada kesalahan besar; MAE dan jumlah prediksi terdekat dapat memberi urutan berbeda.
- Pearson mengukur hubungan linear, bukan agreement atau akurasi. Korelasi tinggi dapat disertai bias skor.
- QWK/Gwet AC2 memakai salinan kategori lengkap0–100, dengan clipping dan kategori terdekat, tie .5 ke bawah. RMSE/Pearson/MAE memakai skor mentah kontinu.
- AC2 tinggi bukan persentase jawaban tepat; jangan memberi ambang kualitas universal tanpa konteks/rubrik.
- Heatmap: guru di baris, prediksi di kolom. Diagonal berarti rentang sama, bukan skor persis. Nol/kotak kosong bukan data hilang.
- Hitungan terdekat bergantung pada kandidat. Menambah eksperimen dapat mengubah ranking tanpa mengubah satu prediksi lama pun. Untuk seri, satu bagian kemenangan dibagi rata; total tetap60.
- SD fold bukan variasi seed. Hasil tiga seed yang melemah tidak cukup untuk menyebut run sebelumnya “hanya hoki”; itu bukti perlunya mengukur variasi.

Model/arsitektur sama dapat memberi hasil berbeda karena inisialisasi head/layer baru, shuffle, dropout, seed, versi numerik, presisi dan perangkat. Angka seed42/123/2026 adalah pengatur RNG, bukan tingkat kualitas atau persentase data. Seed training dan seed pembagian data berbeda peran; split harus tetap saat menguji variasi training.

## 6. Desain training dan inspeksi fitur

Gunakan split esai/kelompok untuk menghindari kelompok email/duplikat terpecah secara tidak semestinya. Email, ID dan skor bukan fitur teks. Lima fold berjalan pada development; tiap fold membuat model baru. Test tidak memilih epoch, LR atau checkpoint.

Epoch refit ditentukan dari **ceil median epoch terpilih lima fold**, lalu model baru dilatih pada240development dan dievaluasi pada60test. Early stopping memakai rekor RMSE validasi dengan patience5. Jangan menambahkan epoch dari test atau menyebut lima fold sebagai25fold.

Metrik validation yang bagus dapat diikuti test lebih buruk karena sampel/variasi/overfitting; tidak ada jaminan urutannya selalu sama. Test telah dilihat berulang kali selama eksplorasi: perlakukan hasil lanjutan secara eksploratif, pilih konfigurasi melalui CV, dan jangan memilih seed terbaik dari test.

### Pemetaan kode penting

| Konsep yang pernah ditanyakan | Lokasi/peran pada notebook |
| --- | --- |
| Hyperparameter | Cell12; checkpoint, LR encoder/proyeksi/head, epoch, fold, input format |
| X/y dan cleaning | Cell14; `read_dataset`, `Bundle`, `clean`, `bundle.y` |
| Split | Cell18; development/test dan lima fold |
| Susunan konteks/window | Cell20; `context_token_ids`, `windows_for_texts`, posisi respons |
| Pretrained | Cell21; `AutoTokenizer/AutoModel.from_pretrained` |
| Fitur sebelum latihan | Cell24 RoBERTa; cell26 BGE |
| Kata/kalimat | Cell28; vektor subtoken dengan konteks |
| Arsitektur | Cell30; `SingleScoreModel`, `Engine`, pooling, norm, proyeksi, head |
| Seed/model baru | Cell31 |
| Loss/optimizer/fit | Cell36 |
| CV/refit/test | Cell38; `train_all` |
| Perbandingan skor | Cell39; `test_score_comparison` |
| Prediksi tiga teks | Cell41; `score_answer` |
| Fitur model final | Cell43; `export_final_features(bundle,bge,cfg,output,journal)` |
| Lima Excel | Cell45; penggabungan ekspor |
| ZIP unduhan | Cell46 |

Nomor ini indeks cell yang dipakai buku/source E21–E26; nomor eksekusi `[n]` Colab dapat berbeda. Saat mengubah notebook, cari anchor dan verifikasi lagi, jangan mengandalkan nomor saja.

`bundle` adalah wadah data; `cfg` konfigurasi; `output` direktori; `journal` pencatat Excel. MLP adalah **head regresi**, bukan encoder pengganti RoBERTa. Layer terakhir MLP tetap linear. LayerNorm menormalkan statistik komponen per vektor dan mempelajari skala/geser, bukan menormalkan skor guru.

Jumlah token berubah menurut teks; dimensi768RoBERTa/1024BGE berasal dari hidden size, bukan panjang jawaban atau jumlah kata. Proyeksi BGE1024→256 khusus fusion S3. E23/E26 juga mempunyai proyeksi pada cabang ringkasan BiGRU128→768: jangan mengatakan semua proyeksi hanya berada pada BGE.

`clean()` merapikan Unicode/newline/karakter kontrol sesuai source; tidak otomatis stemming, stopword removal, koreksi ejaan, atau penulisan ulang jawaban siswa. Soal/referensi menggunakan anggaran konteks; coverage/truncation harus dibaca dari laporan.

Attention/BiGRU pooling hanya posisi respons yang sah. RoBERTa tetap membaca konteks melalui self-attention. Mask memakai posisi token sebenarnya, bukan ID yang kebetulan sama. BGE frozen tetap eval/no_grad; RoBERTa pada jalur training tidak boleh terputus oleh detach/no_grad/cache stale.

Fitur final diekspor sesudah final test, eval/no_grad. Tabel300jawaban untuk inspeksi representasi, bukan hasil penelitian300test. Vektor kata cabang RoBERTa S3 bukan otomatis fusion per kata. Perbandingan vektor awal/akhir harus memakai sumber yang jelas.

## 7. Notebook dan penyimpanan: keputusan akhir

Pengguna sempat memakai CPU karena kuota GPU dan mencoba IDE/ekstensi Colab. Keputusan terakhir untuk E21–E26: **web Colab biasa, GPU, notebook mandiri, tanpa clone atau ekstensi, tanpa Drive**. Sediakan CPU jika diminta; jangan diam-diam beralih dari GPU pada notebook yang meminta CUDA.

Format notebook mengikuti MLP acuan: source lengkap, cell berurutan, worker `%%aes`, setup pustaka terpisah, upload dataset, training, inspeksi, lima Excel. E25/E26 mempunyai47cell/30codecell. Worker menjaga namespace perhitungan terpisah dari kernel Colab; gunakan jalur lingkungan Python yang benar untuk import/diagnosis.

Lima workbook standar:

1. `01_Data_dan_Tokenisasi.xlsx`
2. `02_Vektor_Jawaban.xlsx`
3. `03_Vektor_Kata_dan_Kalimat.xlsx`
4. `04_Prediksi_dan_Evaluasi.xlsx`
5. `05_Riwayat_Pelatihan.xlsx`

Pengguna menolak kompleksitas auto-delete/sinkronisasi Drive. Pertahankan pilihan penyimpanan manual/lokal terbaru. Checkpoint lokal dua slot per tahap merupakan desain resume notebook; jangan memperkenalkan API hapus Drive tanpa permintaan baru.

### Kesalahan operasional yang sudah pernah muncul

- Drive mengembalikan file lokal tetapi daftar API belum sinkron; cleanup menghentikan training demi verifikasi. Itu alasan beralih dari auto-cleanup.
- `OSError 107: Transport endpoint is not connected` saat ekspor adalah masalah mount/penyimpanan; belum tentu training gagal. Periksa report/checkpoint dan kelengkapan ekspor sebelum menyuruh mengulang semua.
- Export fitur terhenti/cancel tidak selalu kegagalan training. ZIP yang terunduh juga harus diaudit isinya; jangan menyatakan benar hanya karena file ada.
- `torchvision::nms does not exist` muncul pada lingkungan LoRA/PEFT karena import pustaka; jangan otomatis menyalahkan data/model. Telusuri setup/import yang benar-benar dipakai.
- Reuse S1/S2 pernah ditolak karena versi Torch berbeda. Checkpoint bisa dipetakan CPU/GPU bila kompatibel, tetapi guard data/config/source/versi numerik tetap harus dihormati. Jangan melewati pemeriksaan hanya karena nama file sama.
- Widget upload perlu cell dieksekusi pada sesi browser aktif. Untuk web Colab, ikuti pemilih file notebook, bukan instruksi OneDrive/IDE yang berbeda.

Checkpoint S2 lebih ringan karena model yang disimpan berisi head (beserta state training yang relevan), sementara BGE pretrained tetap eksternal/frozen. Ukuran file bukan ukuran akurasi.

File checkpoint yang dibahas pengguna sebagai “py” biasanya `.pt`; pastikan ekstensi sebenarnya. Jangan menyarankan hapus seluruh `.pt` tanpa melihat tahap. `final.pt` diperlukan untuk memuat model; JSON penunjuk tanpa checkpoint yang dirujuk tidak cukup untuk resume. [Panduan pindah dan backup](PINDAH_AKUN.md#berkas-yang-perlu-dibawa).

## 8. HTML belajar: kebutuhan yang sudah disepakati

**Hanya dua aplikasi hasil aktif:** buku eksperimen lama dan buku tiga input E21–E26. Jangan menambah salinan HTML per eksperimen atau mengganti hasil lama dengan versi baru tanpa provenance. Paket lengkap terenkripsi, kunci tetap lama; source/agregat publik terpisah dari teks/ID/nilai individual.

Pengguna meminta pengalaman membaca sebagai berikut:

- Meja belajar berisi **seluruh kode**, bukan cuplikan; banyak sticky note di samping ruas penting, gaya belajar seperti `Belajar_Kode_E6.html` yang kini digabung ke buku.
- Ruang kosong antar-ruas boleh agar penjelasan sejajar dan tidak menumpuk. Pada HP catatan mengikuti ruasnya.
- Fullscreen harus memenuhi viewport **tanpa heading induk**. Kode dan catatan satu kanvas/satu scroll, tidak dua panel scroll terpisah.
- Mode **semua blok menyatu dengan pembatas nama** dan **per blok** sama-sama tersedia; Esc/keluar mengembalikan posisi halaman.
- Sepuluh padanan screenshot→kode tetap bisa diklik; tombol harus menuju anchor yang nyata.
- Tabel perbandingan menjadikan **eksperimen sebagai kolom**. Angka terbaik disorot, hasil sama/seri diberi penjelasan; semua0tidak disorot sebagai keberhasilan.
- Semua jawaban/prediksi dapat dibaca bersamaan; model di kiri, guru di kanan, warna konsisten per eksperimen. Mode sorotan dan peringkat tersedia, bukan pemilihan jawaban satu per satu saja.
- Peringkat memisahkan terdekat, RMSE, MAE. Duel menyebut nama dua eksperimen, bukan hanya “kiri/kanan”.
- Kurva mempunyai label sumbu/unit, legenda, pemeriksa epoch dan penanda epoch terpilih. Peta skor guru×prediksi tetap tersedia.
- Seluruh sheet dan nilai asli Excel dipertahankan. Paging/rentang komponen hanya membatasi layar; XLSX/ZIP unduhan byte-identik.
- Navigasi disederhanakan. Buang kontrol mati/duplikat; jangan kembalikan panel upload ZIP/simpan private HTML yang tidak dipakai atau tombol E2 yang tidak berfungsi.
- Scroll mengikuti area di bawah pointer dan meneruskan guliran saat batas tercapai. Fullscreen menahan halaman belakang. Pertahankan keyboard/mobile.
- Tema komik Spider-Man dipertahankan. Pengguna lebih membutuhkan keterbacaan daripada banyak panel atau kuis.

Buku membaca hasil tersimpan, tidak melatih model di browser. “Output per kode” berasal dari tabel tahap pada Excel bila log notebook tidak disertakan; jangan mengarang stdout/traceback atau label executed output. Source tiap eksperimen mengikuti pin masing-masing, bukan menempel kode E26 pada hasil E21.

## 9. Paper dan rekomendasi: tingkat bukti

Pengguna mengutamakan rekomendasi berdasarkan paper. Catat apa yang dibaca: full text, abstrak, metadata atau README penulis. Sumber tidak dapat diakses bukan berarti paper tidak ada; jangan mengarang hasil untuk mengisi kekosongan.

| Sumber | Peran dan batas |
| --- | --- |
| Liu dkk.2019, RoBERTa | Dasar encoder, bukan pembuktian resep AES ini |
| Chen dkk.2024, BGE-M3 | Dasar embedding; hasil retrieval tidak otomatis bukti AES |
| Dong dkk.2017, [attention-based recurrent CNN untuk AES](https://aclanthology.org/K17-1017/) | Inspirasi attention; metadata/abstrak resmi diperiksa pada penelusuran awal |
| Alhothali2026, [PaSTO-GNN](https://doi.org/10.3389/frai.2026.1842542) | Full text dari koleksi pengguna; BiGRU+attention adalah bagian sistem lebih besar, bukan jaminan adaptasi token-level |
| Arevalo dkk.2017, [GMU](https://openreview.net/forum?id=Hy-2G6ile) | README implementasi resmi diperiksa; gate skalar dua encoder teks adalah adaptasi, bukan replikasi tugas multimodal |
| Chen dkk.2017, [ESIM](https://aclanthology.org/P17-1152/) | Metadata/abstrak dan kode penulis diperiksa; interaksi fitur `[u,v,u*v,u-v]` adalah alternatif usulan, belum hasil AES proyek |
| Zhang & Litman2018, [co-attention AES](https://aclanthology.org/W18-0549/) | Full text penulis diperiksa; hasil dan batas rinci pada rencana berikutnya |
| Sun, Zhou & Fan2020, [Adaptive Huber Regression](https://doi.org/10.1080/01621459.2018.1543124) | Citation/paket penulis diverifikasi; full journal text belum berhasil diakses pada penelusuran |
| Pan, Sun & Zhou2021, [robust regression](https://doi.org/10.1214/21-EJS1862) | Repositori penulis memuat simulasi noise berat; error koefisien bukan RMSE prediksi esai |

Contoh ILAMM resmi membandingkan error koefisien L2 SCAD0,741→Huber-SCAD0,402 dan MCP0,795→Huber-MCP0,354. Itu contoh simulasi README, bukan tabel AES atau bukti Huber selalu lebih akurat/cepat daripada MSE. Buku Géron di koleksi pengguna menyebut konvergensi Huber dibanding **MAE**, bukan klaim universal lebih cepat daripada MSE.

Tidak menjanjikan layer, LoRA, konteks pair, LR lebih kecil, atau Huber pasti menurunkan RMSE. Jangan menyebut kombinasi novel atau signifikan tanpa bukti yang tepat. [Rancangan berikutnya dan sumber](RENCANA_EKSPERIMEN.md).

## 10. Titik lanjut setelah hasil E25/E26

Minat terbaru pengguna adalah **LR head MLP `5e-6`**, bukan LR encoder. Encoder sudah menggunakan5e-6; MLP masih1e-4. Uji MLP saja memerlukan pemisahan optimizer: kelompok `head` sekarang juga mengandung GRU/attention/proyeksi ringkasan/norm. Jangan mengubah semuanya lalu menamakannya uji LR MLP.

Dua jalur awal diusulkan dari E26: perubahan LR MLP dan co-attention respons–referensi. **Pilihan pengguna berikutnya mengganti dasar keduanya ke E23**, seperti dijelaskan di bawah. Uji terpisah; kombinasi keduanya belum otomatis lebih baik. Tiga seed untuk acuan dan kandidat terpilih diperlukan sebelum klaim konsistensi. Gate tambahan belum prioritas berdasarkan hasil E25. Huber tetap uji loss terpisah bersyarat analisis residual.

**Pembaruan sesudah serah-terima:** pengguna meminta kode dua jalur tersebut, lalu secara eksplisit mengganti dasar ke **E23**, karena prioritasnya banyaknya prediksi mendekati guru. Draf turunan E26 belum dipublikasikan dan digantikan; keputusan ini bukan klaim E26 hanya bagus saat training—RMSE/MAE test E26 tetap lebih rendah. **E27 = E23 + co-attention respons–referensi**, **E28 = E23 + LR MLP S3 saja5e-6**. Keduanya dibuat sebagai notebook GPU biasa, tanpa Drive, seluruh S1/S2/S3 dilatih baru, dengan sumber E23 dipin. [Metode dan source](docs/eksperimen/eksperimen_e27_e28_gpu.md). E28 tidak menambahkan co-attention; arsitektur E23 tetap. E27 MLP/layer baru tetap1e-4. Kedua optimizer memisahkan `head.*` dari pooling/norm; S1/S2 tetap kontrol. **Pembaruan29Oktober2026:** kedua ZIP E27/E28 telah diaudit dan ditambahkan ke buku. RMSE S3 E27=6,3277 dan E28=7,0447. Enam buku lama dipertahankan identik. Pada delapan peserta E23/E26 seri terdekat10/60; E23 tetap kode utama sesuai pilihan pengguna. [Review hasil dan preservasi isi](docs/review/review_e27_e28_study.md). Jangan menyebut E29 sudah dibuat atau training pretrained sudah dijalankan oleh tes fixture.

## 11. Berkas, akun baru, dan sumber kebenaran

**Keputusan terbaru setelah hasil E27/E28:** pengguna memilih **E26** untuk percobaan lanjutan. E29 menurunkan seluruh parameter MLP `head.*` **S2 dan S3** ke5e-6, sementara linear S1 tetap1e-4. E30 menambah **co-attention respons–referensi kemudian attention pooling E26**, bukan ringkasan ujung E27/E23. MLP E30 tetap1e-4. Encoder RoBERTa5e-6; BiGRU/attention/proyeksi/norm/layer baru1e-4. Dua notebook GPU tanpa Drive sudah dibuat terpisah; hasil penelitian belum tersedia. [Source/panduan](docs/eksperimen/eksperimen_e29_e30_gpu.md). Usulan ensemble tiga seed/E31 belum menjadi kode. HTML delapan eksperimen tidak diubah oleh permintaan kode ini.

| Kebutuhan | Lokasi |
| --- | --- |
| Jalur masuk | [README](README.md) |
| Config/hasil aktif ringkas | [STATUS_PROYEK.md](STATUS_PROYEK.md) |
| Notebook utama | [notebooks/experiments](notebooks/experiments/README.md) |
| Kedua pengunduh | [downloads](downloads/README.md) |
| Review terbaru | [review E25/E26](docs/review/review_e25_e26_study.md) |
| Snapshot agregat | [docs/snapshots](docs/snapshots/README.md) |
| Cara rebuild dan batas data | [scripts](scripts/README.md) |
| Sumber lama yang berbeda | [arsip notebook](notebooks/archive/README.md) dan [arsip dokumen](docs/arsip/README.md) |
| Pindah akun dan prompt singkat | [PINDAH_AKUN.md](PINDAH_AKUN.md) |

Repo publik tidak otomatis memberikan akses ke dataset, teks siswa, nilai individual, bobot, kunci, folder `outputs/`, atau lampiran chat. Buku lengkap menyimpan laporan dalam paket terenkripsi; ia tidak menyimpan bobot training. Untuk review individual di akun baru, pengguna perlu menyediakan berkas privat yang sesuai. Jangan meminta ulang berkas yang sebenarnya sudah tersedia pada sesi aktif; periksa dahulu.

Source notebook utama dipertahankan saat penataan; output eksekusi dibersihkan. Empat mirror lama yang berbeda disimpan sebagai varian arsip sebelum mirror disamakan. Riwayat Git tetap; pembersihan versi terbaru bukan penghapusan data dari seluruh riwayat.

Saat bekerja: periksa branch/status/sumber aktual, hormati edit pengguna, pisahkan data privat, uji sesuai perubahan, dan verifikasi commit/isi remote sebelum mengatakan GitHub sudah diperbarui. Bedakan **ditulis**, **diuji fungsi/fixture**, **smoke pretrained**, **dijalankan pada data penelitian**, dan **dipublikasikan**. Jangan menyamakan jumlah tes kode dengan kualitas prediksi.

## Identitas tanpa hasil · pembaruan antarmuka terakhir

Pengguna meminta HTML tetap menyediakan tempat untuk E29/E30 sebelum output tersedia. Katalog `familyData.identities` kini memuat E21–E30, sementara `comparison.experiments`, `documents`, dan arsip privat tetap E21–E28. Jangan menghitung placeholder sebagai eksperimen berangka. Ruang kode/output E29/E30 kosong; notebook sumber ditautkan secara eksternal, bukan diembed. Kode utama belajar tetap E23 dan seluruh delapan buku anak utuh. Tab identitas menyediakan acuan eksperimen, alur, kelompok LR dan sitasi dengan batas dukungan. LR5e-6/pair merupakan ablasi proyek, bukan jaminan keunggulan dari paper. [Review](docs/review/review_identitas_eksperimen.md).

## Hasil E29 dan pembatasan buku kode terbaru

ZIP E29 sudah diterima dan diaudit. S3 RMSE5,4293,MAE4,1713,≤5poin40/60; S2 RMSE75,9702 dengan seluruh fold memilih25epoch. Jangan menyembunyikan S2 buruk atau menyebut40/60 sebagai skor persis. E29 LR MLP S2/S3 5e-6; arsitektur E26, tanpa co-attention. Hasil E30 belum tersedia.

Pengguna mempertahankan HTML dan meminta kode hanya tiga eksperimen terbaik. Kriteria implementasi dinyatakan eksplisit: tiga RMSE test S3 terendah **E29,E26,E24**, utamaE29. Ini menggantikan defaultE23 dan kebijakan delapan buku kode sebelumnya. `documents` hanya3, `comparison.experiments`9, `identities`10. Semua45Excel/hasil tetap tersedia melalui explorer mandiri tab08; tombol Excel tidak boleh bergantung pada buku kode yang telah dihilangkan. Kartu eksperimen lain tetap menautkan notebook eksternal. E30 tetap dua ruang kosong dan tidak masuk perbandingan. [Review](docs/review/review_e29_study.md).
