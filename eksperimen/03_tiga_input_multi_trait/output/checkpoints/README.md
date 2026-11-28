# Checkpoint eksperimen AES multi-trait

Folder ini menyimpan konfigurasi efektif dan bukti agregat yang bisa dipakai kembali saat merancang eksperimen atau menulis skripsi. **E21-B** adalah acuan test S3. **E23 seed 43** adalah kandidat pembobotan isi di development. Seed 44 mengulang deteksi 2/17 pada dua jawaban yang sama. Status run selanjutnya tercatat di [register bukti](../../../../docs/skripsi/Register_Bukti_AES_MultiTrait.xlsx).

| Artefak | Yang dicatat |
|---|---|
| JSON per run | Model dan revisi, hyperparameter, split dan kode, hash ZIP/Excel, epoch A/B, metrik tiap trait, deteksi skor rendah, bobot dan riwayat training agregat |
| E17E | Trial pencarian, lima-fold konfirmasi, cek stabilitas, dan konfigurasi akhir. ID 52 dan 60 adalah kandidat dari tahap berbeda. |
| E14 / E14-S1S2 | Workbook dari unggahan terpisah, dipadankan dengan checksum pada manifest asli dan dikemas sementara untuk proses ekspor. Kolom skenario pada E14-S1S2 membedakan S1 dan S2. |
| E5 | Pasangan workbook unggahan dengan metadata dataset, split, kode, dan konfigurasi identik. Checksum dibuat dari pasangan tersebut karena manifest run asli belum tersedia. |
| E1–E4 dan E17D | Hasil yang sudah ditautkan di [katalog eksperimen](../../README.md) dan [arsip E17D](../E17D_2026-11-25/README.md). |

JSON di sini adalah checkpoint **bukti dan resep**. Bobot model `.pt`, studi SQLite asli, kutipan jawaban, skor per individu, serta ZIP privat tetap di penyimpanan penelitian. Untuk menjalankan ulang, pakai notebook yang ditautkan, workbook semi-final yang sesuai hash, dan cek seed serta versi runtime. Kesamaan seed membantu replikasi, tetapi nilai numerik dapat berubah antar-runtime.

## Pembaruan register

1. Tambah atau revisi entri [paper](../../../../docs/skripsi/bukti/paper.tsv), [keputusan komponen](../../../../docs/skripsi/bukti/keputusan.tsv), dan [keputusan checkpoint](../../../../docs/skripsi/bukti/checkpoint.tsv). Catat tahun terbit, sistem dan kategori indeks, relevansi, asal nilai, dan bukti lokal. Jika yang ada baru rancangan, tandai status itu.
2. Jalankan `python scripts/build_aes_evidence_register.py` dari root ketika notebook baru masuk. Kalau hasil final privat sudah tersedia, jalankan `python scripts/build_aes_evidence_register.py --archive E23-Seed44 /path/ke/hasil.zip` dengan ID dan path hasil yang sesuai. Perintah ini memeriksa checksum sumber, mengekspor kolom agregat yang diizinkan, dan membangun ulang Excel. Pakai ID run unik saat menyimpan hasil tambahan.
3. Periksa isi perubahan, perbarui katalog eksperimen, dan commit register bersama sumber TSV dan arsip agregat. Daftar paper di register membedakan hasil verifikasi SJR tingkat venue dengan bukti relevansi multi-trait. Catat kandidat tanpa bukti indeks sebagai kandidat.

Rujukan indeks yang dipakai pada rilis awal direkam dari [snapshot sjrdata f7ed827](https://github.com/ikashnitsky/sjrdata/tree/f7ed827/data-raw/sjr-journal). Halaman bibliografi paper dan sumber indeks masing-masing ada di tab **Paper dan indeks**.
