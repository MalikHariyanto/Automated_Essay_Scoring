# E37 · LR encoder5e-6 vs7e-6 · tiga seed pada fold tetap

**ARSIP: pengguna membatalkan desain tiga seed. Pakai [E37 satu resep/lima fold](eksperimen_e37_single_lr_native_gpu.md). Dokumen ini hanya merekam desain sebelumnya.**

E37 menguji satu faktor pada metode E30: LR encoder RoBERTa. Penurunan ke3e-6 pada E36 belum mengalahkan E30; peningkatan ke7e-6 adalah hipotesis baru, bukan nilai optimal yang sudah dibuktikan paper.

[Buka Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E37_E30_S3_EncoderLR5e6_vs7e6_3Seed_Native_Learnable_GPU.ipynb) · [Source notebook](../../notebooks/experiments/AES_300_E37_E30_S3_EncoderLR5e6_vs7e6_3Seed_Native_Learnable_GPU.ipynb).

## Resep dan kontrol

| Komponen | Kontrol E30 dijalankan ulang | Kandidat E37 |
|---|---|---|
| Encoder RoBERTa S3 |5e-6|7e-6|
| MLP/proyeksi BGE/BiGRU/co-attention/pooling/norm |1e-4|1e-4|
| Model |Tiga teks → RoBERTa fine-tuned → BiGRU → co-attention respons–referensi → attention pooling → concat proyeksi BGE frozen → MLP|Sama|
| Seed training |42,142,242|42,142,242|
| Split/kelompok/fold |split_seed2026,holdout20%,5fold development|Sama persis|
| Anggaran training |maks25epoch/fold,patience5|Sama|

Input X adalah tiga kolom teks `soal`, `response_text`, `jawaban_referensi`. Label y `rater`0–100 tetap diperlukan untuk MSE dan evaluasi; tidak masuk encoder. Tidak ada S1/S2 terpisah, Ridge, pemeriksaan cosine, ensemble, worker/magic tersembunyi, clone repo atau Google Drive.

Warmup10%,scheduler linear,AdamW,weight decay0,01,dropout0,1,microbatch1/batch efektif8,hidden BiGRU64perarah serta token/window mengikuti E30. `fit`, optimizer grouping, arsitektur, rumus metrik dan pemisahan kelompok tetap sama secara AST dengan source E30. Kontrol ini merupakan run ulang resep E30 dalam protokol baru, bukan pemuatan angka lama; hasil identik dengan run historis tidak dijamin.

## Dua tahap yang terlihat pada notebook

1. **Development CV:** dua LR × tiga seed × lima fold =30latihan fold. BGE frozen dihitung sekali. Tidak ada refit/test pada tahap ini. Fold dan holdout dibentuk sekali; perubahan seed hanya memengaruhi training. Seed aktual tiap fold =seed dasar+nomor fold, sesuai E30.
2. **Seleksi LR:** mean dari tiga rerata RMSE lima fold. Rerata sama mempertahankan kontrol. Selisih kandidat−kontrol tiap seed dilaporkan; negatif berarti kandidat lebih baik. SD antar-fold dan SD antar-seed dilaporkan terpisah, bukan dianggap15fold independen atau interval kepercayaan.
3. **Kunci seleksi sebelum test:** simpan source/split/code/design dan SHA keenam laporan CV. Tolak perubahan hasil setelah seleksi terkunci.
4. **Refit/test:** tiga seed dari resep terpilih saja. Epoch setiap seed =ceil median best epoch lima fold miliknya. Laporan mencakup semua tiga seed dan mean/SD, tanpa memilih seed test terbaik. Tabel utama, vektor dan contoh prediksi memakai seed pertama42yang ditetapkan sebelum training. Ini model tunggal, bukan rata-rata prediksi ensemble.

Test lama sudah sering dilihat pada seri sebelumnya. Pemilihan LR dilakukan tanpa test di notebook ini, tetapi hasil akhirnya tetap eksploratif; konfirmasi kuat membutuhkan data baru. Satu resep yang menang rerata CV belum otomatis unggul pada setiap seed atau data baru.

## Cara menjalankan dan membaca hasil

Runtime Colab baru → T4 GPU → Run all → upload `fix_dataset_300.xlsx`. Notebook memiliki section model, kolom X/y, hyperparameter dan definisi/pemanggilan fungsi yang terlihat di kernel. Semua notebook yang dipublikasikan kosong outputnya; dataset asli tidak disertakan.

Enam run CV memerlukan sekitar enam kali jumlah latihan fold dibanding E36, ditambah tiga refit; waktu nyata bergantung early stopping dan perangkat. Tidak ada estimasi durasi GPU yang telah diukur untuk E37. Dua slot checkpoint fold yang **sudah selesai** dibuang hanya setelah JSON hasil fold tersimpan; fold terputus tetap dapat dilanjutkan pada runtime yang sama. Model final disimpan. Jika runtime dihapus, penyimpanan lokal ikut hilang.

ZIP keluaran tetap lima workbook standar. Tabel lama pada04/05 menjelaskan resep terpilih, seed42. Tab tambahan:

- **04_Prediksi_dan_Evaluasi.xlsx:** `Test per Seed`, `Test Mean SD Seed`, `Prediksi Semua Seed`.
- **05_Riwayat_Pelatihan.xlsx:** `CV Semua LR Seed`(30baris), `CV per Seed`(6baris), `CV Ringkasan LR`(2baris), `Selisih CV Berpasangan`, `Seleksi LR Terkunci`, `Riwayat Semua LR Seed`.

Jika kontrol5e-6menang, output tetap bernama studi E37 tetapi konfigurasi aktual/tabel terkunci menyebut5e-6. Jangan menyebut seluruh tabel sebagai hasil kandidat7e-6tanpa memeriksa resep yang dipilih. Seluruh hasil kedua LR tersedia pada tab CV, sedangkan test hanya resep terpilih.

## Rujukan dan bukti

Referensi komponen tetap terlihat pada section akhir notebook, termasuk RoBERTa,Liu dkk.2019; BGE-M3,Chen dkk.2024; GRU,Cho dkk.2014; AdamW,Loshchilov&Hutter2019; serta sumber co-attention/attention pooling/metrik dari E30. Rujukan tidak menjamin angka LR7e-6atau25epoch merupakan pilihan optimal.

[Bukti verifikasi teknis](../verifikasi/verification_e37_native.md). Tidak ada hasil penelitian E37 sebelum ZIP pengguna yang benar-benar dieksekusi tersedia; jangan menambahkan skor penelitian atau memperbarui ranking HTML berdasarkan fixture teknis.
