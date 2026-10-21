# Verifikasi buku hasil MLP

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
