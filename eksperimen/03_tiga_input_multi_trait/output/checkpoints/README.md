# Checkpoint eksperimen AES multi-trait

Folder ini menyimpan konfigurasi efektif dan bukti agregat yang bisa dipakai kembali saat merancang eksperimen atau menulis skripsi. **E21-B** adalah acuan test S3. **E23 seed 43** adalah kandidat pembobotan isi di development. Seed 44 mengulang deteksi 2/17 pada dua jawaban yang sama. Status run selanjutnya tercatat di [register bukti](../../../../docs/skripsi/Register_Bukti_AES_MultiTrait.xlsx).

| Artefak | Yang dicatat |
|---|---|
| JSON per run | Model dan revisi, hyperparameter, split dan kode, hash ZIP/Excel, epoch A/B, metrik tiap trait, deteksi skor rendah, bobot dan riwayat training agregat |
| E24 | [Ringkasan pilot NLI](E24/README.md) dan checksum terpisah dari hasil CV. |
| E24B | [Ringkasan pilot empat unsur](E24B/README.md), dibandingkan dengan kontrol lokasi E24. |
| E24C | [Ringkasan pilot ekstraksi bukti](E24C/README.md) dan keputusan akhir jalur pilot. |
| E25 | [Kontrol bobot isi 1×](E25.json) dengan seed 44 yang sama dengan E23. Bobot 2× memperbaiki isi, dengan kompromi kebahasaan dan mekanik. |
| S1/S2/S3 | [Audit test sepadan](S1S2S3/README.md) pada 270 jawaban dan 27 kelompok. |
| E26 | [Ensemble prediksi Isi](E26/README.md) dan perbandingan eksploratif dengan S1 serta E21-B. |
| E27 | [Paket representasi E43 untuk S3](E27.json). Validasi sepadan dengan E25. RMSE Isi membaik; macro QWK dan tiga trait lain turun. |
| E28 | [ScalarMix-4 pada E25](E28.json). Isi dan organisasi membaik pada empat metrik OOF; kebahasaan turun tipis dan mekanik beragam. |
| E29 | [ScalarMix per trait](E29.json), sedikit di atas E28 pada RMSE mean fold dan macro QWK. Selisih kecil dan hasil antartrait beragam. |
| E30 | [Transfer Trial 60 ke E29](E30.json). RMSE Isi turun, tetapi QWK empat trait turun. E29 tetap kandidat development yang diprioritaskan. |
| E31 | [Hasil EMA pada E29](E31.json), perbaikan RMSE semua trait OOF dan fold 4, dengan trade-off QWK Organisasi/Mekanik. |
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
