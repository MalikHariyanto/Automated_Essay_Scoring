# Verifikasi buku hasil MLP

## Penambahan menu pembanding — 22 Oktober 2026

Pekerjaan lanjutan dimulai dari HEAD `2a67b89641ae50e5fad8a19a6341acdacdea4c98`, branch `codex/aes-learning`. Menu **Bandingkan eksperimen** memakai snapshot agregat terpisah `docs/mlp_comparison_snapshot.json`; buku MLP 16 bab dan notebook training tetap dipertahankan.

Notebook awal privat yang ditemukan di arsip kerja mempunyai SHA-256 `d4f00d143e9389841d8741586417c787dc5e03b77b254124ce15e9df2e9e171e`. Seluruh code cell cocok commit `3afad79`; output tersimpan memuat 15 fold, 75 epoch CV, 14 epoch refit, dan tabel test tiga skenario dengan keempat metrik RMSE/Pearson/QWK/AC2. Riwayat refit aktual berjumlah 5/4/5 epoch untuk S1/S2/S3. Metrik awal berasal dari tampilan sekitar delapan digit; log loss enam desimal dan RMSE empat desimal, sehingga tidak diklaim mempunyai presisi file Excel aslinya.

Perkembangan source antara diambil dari `312f394`: LR lebih kecil, maksimum 25 epoch, patience 5, LayerNorm tambahan RoBERTa S3, tetapi head S3 masih linear. Hasil akhir yang dapat dipasangkan secara pasti dengan versi antara itu belum tersedia; snapshot menyimpan `null`, bukan nol atau angka dari log run lain. Run MLP tetap mengikuti `15bc7bb` dan lima Excel pengguna.

Perintah pengujian terakhir dari cwd `/workspace/Automated_Essay_Scoring`:

```bash
python scripts/build_mlp_comparison.py --baseline outputs/revision25_before/user_saved_original.ipynb
python -m unittest discover -s tests -p 'test_mlp_*.py' -v
node --check scripts/templates/mlp_study.js
node --check scripts/templates/mlp_comparison.js
node scripts/check_mlp_comparison.mjs
```

Unit test gabungan: **12 passed, 0 failed, 0 skipped, 0 xfailed**, exit 0. Tambahan pemeriksaan menguji source tiap versi terhadap commit-nya sendiri, kelengkapan 15 fold awal dan seluruh epoch, kecocokan minimum riwayat dengan tabel, parser metrik negatif, pemisahan CV/refit tanpa newline, penolakan source berbeda, dan ketiadaan hasil antara maupun prediksi individual pada snapshot publik.

Pemeriksaan browser pembanding: **8 kelompok lolos, 0 gagal**, exit 0. Cakupannya 120 kombinasi dua versi pembanding × tiga skenario × lima tab × empat lebar layar (1440/900/390/320), serta 60 konfigurasi grafik aktual (dua versi × tiga skenario × lima fold × dua jenis kurva). Tidak ada page error, angka NaN/undefined, overflow halaman utama, atau permintaan jaringan eksternal. Navigasi keyboard, sticky note layer, source berdampingan dengan baris berubah, dan pergantian kembali ke buku MLP bekerja. Pengecekan regresi buku lama tetap mencakup 34 output dan seluruh 41 sheet asli.

HTML pribadi memuat notebook awal selain notebook/ZIP MLP. Tiga prediksi contoh awal dan tiga prediksi contoh MLP dirender terpisah. Skor guru contoh awal tidak tersedia dan tidak diisi dari contoh MLP. Identitas contoh maupun manifest anggota test/fold awal belum dapat dipasangkan lintas run. Ekspor/reopen browser mempertahankan ketiga file pribadi. Tabel lengkap 60 pasangan awal tidak tersedia pada sumber yang dapat dibuka; ZIP lama melebihi batas transfer 32 MiB alat unggahan. Buku menandai pasangan awal belum tersedia, tanpa membuat prediksi pengganti.

Angka RMSE S3 test berubah 11,722398 → 6,708539824280181; rata-rata CV 10,95416436 → 6,051639733368358. Perubahan bukan ablation satu komponen: head, LR, batas epoch, aturan berhenti dan LayerNorm juga berubah. Test awal sudah pernah dibaca. Buku menjelaskan batas kausalitas, ketidakpastian keanggotaan split, dan status eksploratif pemakaian ulang test. Double-frozen/MLP, Ridge dan pemeriksaan kontribusi head dibahas sebagai pertanyaan lanjutan dengan status hasil yang jelas.

Preview aman tambahan: [peta perbandingan](previews/mlp_study/04_perbandingan.png), [susunan layer](previews/mlp_study/05_layer.png), [training/CV](previews/mlp_study/06_training.png), dan [HP](previews/mlp_study/07_perbandingan_mobile.png). Diambil sebelum file pribadi dimuat. Output privat, contoh prediksi individual, notebook terisi dan bundel HTML lengkap tetap tidak dipublikasikan.

Pemeriksaan ini tidak mengeksekusi model, training atau evaluasi penelitian baru. Publikasi hanya memperbarui buku pembelajaran dan alat pembangunnya.

## Verifikasi awal — 21 Oktober 2026

Dikerjakan pada 21 Oktober 2026, di `/workspace/Automated_Essay_Scoring`, branch `codex/aes-learning`, dari HEAD awal `0c79d655898f967e855922be3c5fad1865efb1aa`. Buku mengikuti source MLP **15bc7bbb2aee91ecaf44fb91a0139a8c21f64a20**; notebook utama maupun eksperimen double-frozen tidak diubah.

## Identitas sumber

| Sumber | SHA-256 |
| --- | --- |
| Notebook source bersih dari commit 15bc7bb | `b06735cb45d632b92ad8ed30990e2e34d0fb66c46c612c528b812dd0d090440b` |
| Notebook unggahan dengan output | `76ab7f0fb9fb6d1b8b67da45bbf78e495b30b496c640e28ae9d3b026a0cc6026` |

Hash ZIP dan setiap workbook dicatat pada `docs/mlp_study_snapshot.json`. Seluruh 30 code cell unggahan cocok source commit tersebut. Ada 17 markdown cell, 2.488 baris kode, 120 catatan, dan 52/52 anchor fundamental ditemukan pada baris aslinya. Sumber data adalah lima workbook asli dari ZIP pengguna, dengan 41 sheet dan 34 objek output notebook tersimpan.

## Pemeriksaan yang dijalankan

Semua perintah menggunakan cwd di atas. Exit code pemeriksaan terakhir adalah **0**.

```bash
python -m unittest discover -s tests -p test_mlp_study.py -v
node --check scripts/templates/mlp_study.js
python scripts/build_mlp_study.py
```

Unit test: **6 passed, 0 failed, 0 skipped, 0 xfailed**. Pemeriksaan mencakup source lengkap, reversibilitas syntax highlight dan escaping markup, ketepatan baris sticky note, whitelist output publik, inventaris workbook, jumlah observasi matriks agregat, dan epoch CV terpilih terhadap riwayat aktual.

Build ulang publik dari snapshot aman menghasilkan byte HTML yang identik. Build privat dilakukan dengan argumen `--notebook` dan `--results` menunjuk kedua unggahan lokal. Python pengujian unit: 3.12.14; ekstraksi workbook: Python 3.13.5, openpyxl 3.1.5. Versi ini adalah lingkungan pembaca buku, bukan klaim versi training pengguna.

Untuk browser, server HTTP lokal hanya dipakai sebagai sarana pengujian karena lingkungan pengujian membatasi navigasi `file://`:

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory /workspace/Automated_Essay_Scoring
node scripts/check_mlp_study.mjs \
  http://127.0.0.1:8765/docs/belajar_MLP_Spider.html \
  '/path/AES_300_S3_MLP (1).ipynb' \
  '/path/Hasil_AES_Ringkas (2).zip' \
  outputs/mlp_study/expected_workbooks.json
```

JSON pembanding privat disusun melalui openpyxl: seluruh header, jumlah baris, serta sampel baris pertama/tengah/akhir dan kolom pertama/kedua/tengah/terakhir di setiap sheet. JSON dan hasil unduhan browser tidak dipublikasikan.

Playwright dengan Chromium **154.0.8037.57**: **12 kelompok pemeriksaan lolos, 0 gagal**, tanpa page error atau permintaan jaringan eksternal:

- 16 bab × 4 tab × 2 ukuran layar = 128 kombinasi, tanpa overflow mendatar halaman utama.
- Seluruh 30 blok kode tampil dengan jumlah baris asli; sticky note menggulir kode tanpa memindahkan halaman utama.
- Navigasi tab menggunakan keyboard dan dialog istilah bekerja.
- Kedua unggahan dapat dibuka lokal; seluruh 34 output asli dirender pada bab terkait.
- Seluruh 41 sheet cocok pembanding openpyxl pada header, jumlah baris dan sampel nilai asli, termasuk nilai numerik dengan toleransi relatif `1e-12`.
- Komponen fusion ke-1024, pergantian halaman baris, tampilan angka lengkap dan inspeksi sel bekerja.
- Unduhan XLSX dari browser byte-identik dengan workbook asli.
- Ekspor satu HTML pribadi dari browser berhasil dibuka kembali dengan notebook dan lima workbook otomatis terisi.
- Output notebook yang sengaja disisipi script, event handler dan iframe tidak mengeksekusi kode atau memuat sumber eksternal; ZIP yang hash-nya berbeda ditolak.

[Preview desktop](previews/mlp_study/01_beranda.png), [kode dan catatan](previews/mlp_study/02_kode_catatan.png), serta [tampilan HP](previews/mlp_study/03_mobile.png) diambil dari versi publik sebelum file pribadi dimuat. Tampilan mobile juga diperiksa langsung pada viewport ruang baca; screenshot full-page pada perpindahan viewport dapat melewatkan lapisan scroll Chromium sehingga tidak dipakai sebagai preview HP.

## Privasi dan batas bukti

Audit membandingkan 300 jawaban asli terhadap HTML/snapshot dan source baru: **0 kecocokan teks jawaban lengkap**. Payload privat pada HTML publik kosong. Native output publik hanya memakai whitelist cell 12, 18, 32 dan bagian agregat cell 39 (indeks cell berbasis nol). Tabel individual tidak dimasukkan ke snapshot. Matriks publik hanya berisi jumlah agregat per rentang sepuluh poin, bukan pasangan nilai per siswa.

Seluruh file pribadi, executed notebook, ZIP asli, sampel pembanding dan HTML lengkap disimpan di `outputs/` atau direktori unggahan, tidak di-stage. HTML pribadi menyertakan file asli lengkap sesuai permintaan pengguna dan harus disimpan secara pribadi.

Pekerjaan ini **tidak menjalankan encoder, forward/backward, training, atau evaluasi AES baru**. Angka model berasal dari run tersimpan. Pemeriksaan browser dan unit test mengukur integritas buku serta pembacaan file, bukan peningkatan akurasi model. Perubahan hanya menambahkan buku dan alat pembangunnya; publikasi branch tidak melakukan merge ke main.
