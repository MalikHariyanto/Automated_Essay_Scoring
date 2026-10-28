# Verifikasi meja hasil Excel dan output MLP

23 Oktober 2026. Pembaruan dari commit `45c6b5483115b6f0300209b441fbc74f4b321566`, branch `codex/aes-learning`. Lingkup pekerjaan: pembaca hasil HTML, hubungan kode/output/Excel dan kemudahan navigasi. Tidak menjalankan training baru, mengganti notebook atau mengubah hasil penelitian.

## Sumber hasil

Notebook MLP unggahan cocok source commit `15bc7bb` dan checksum snapshot. ZIP MLP, workbook evaluasi awal, notebook awal dan ZIP double frozen yang sebelumnya diunggah disertakan pada HTML pribadi di `outputs/mlp_study/belajar_MLP_Spider_Privat.html`. Semua 34 output notebook MLP dan 41 sheet MLP dimuat otomatis. Pembaca double frozen tetap memuat 41 sheet run tersebut; dua eksperimen tidak dicampur.

HTML publik menyimpan agregat serta sembilan sheet panduan/pengaturan/arti metrik dari Excel asli. Daftar ini memakai whitelist di builder. Jawaban, ID, nilai individual, vektor lengkap, file unggahan dan bundel pribadi tidak dipublikasikan. Pembandingan JSON memastikan semua source cell, catatan, output publik semula, metrik, riwayat, matriks dan snapshot pembanding identik; satu-satunya penambahan data adalah sembilan sheet penjelasan tersebut.

## Perubahan yang diperiksa

- Lima pintasan menuju jenis hasil, pilihan laporan/tabel yang terbaca dan penjelasan yang dapat dibuka sesuai kebutuhan.
- Tombol dari tabel menuju kode tahap dan output Colab terkait. Tab output mempertahankan semua output asli, memberi nama blok, menyembunyikan log panjang di dalam details serta menambahkan pratinjau Excel berlabel sumber.
- Seluruh 11 kolom nilai guru/prediksi/pembulatan tampil bersama. Pengatur komponen hanya tersedia pada vektor; komponen terakhir 1024 tetap dapat dijangkau.
- Pencarian seluruh kolom tabel biasa, pilihan 25/50/100 baris, pemeriksaan nilai sel dengan klik/Enter dan tampilan empat desimal tanpa mengubah nilai aslinya.
- Tombol CSV dihapus. Unduhan yang belum mempunyai file asli, navigasi baris satu halaman dan kontrol komponen pada tabel biasa disembunyikan. Pilihan publik hanya menawarkan sheet berisi data yang tersedia, tanpa placeholder tabel siswa.
- Tab aktif tetap masuk area terlihat pada HP. Halaman tidak meluber horizontal pada 320/390/900/1440 px; tabel lebar memakai scroll di dalam panel.
- Unduhan workbook dan ZIP mempunyai SHA-256 identik dengan unggahan. Ekspor HTML pribadi dibuka ulang dengan seluruh sumber tetap tersedia. Output notebook tidak dapat menjalankan script/event/iframe; ZIP berbeda ditolak.

## Perintah dan hasil

Cwd `/workspace/Automated_Essay_Scoring`. Builder dijalankan dengan `outputs/dataset300_env_py313/bin/python scripts/build_mlp_study.py`, disertai argumen notebook, ZIP MLP, notebook awal, workbook evaluasi awal dan ZIP double frozen yang cocok checksum. File bukti dengan argumen/manifest aktual disimpan privat di `outputs/mlp_study/excel_review/`.

```bash
outputs/dataset300_env_py313/bin/python -m pytest \
  tests/test_mlp_study.py tests/test_mlp_comparison.py tests/test_mlp_frozen.py -q \
  --junitxml=outputs/mlp_study/excel_review/integrity.xml
node scripts/check_mlp_excel.mjs
node scripts/check_mlp_study.mjs http://127.0.0.1:8765/docs/belajar_MLP_Spider.html \
  '<notebook MLP asli>' '<ZIP MLP asli>' outputs/mlp_study/expected_workbooks.json
node scripts/check_mlp_fullscroll.mjs
node scripts/check_mlp_frozen.mjs
node scripts/check_mlp_metrics.mjs
node scripts/check_mlp_comparison.mjs
```

Semua perintah exit **0**. Tes integritas: **17 passed, 0 failed, 0 skipped, 0 xfailed**. Pemeriksaan browser: **7 kelompok** meja Excel, **12 kelompok** buku/pembaca lokal, **5 kelompok** kanvas semua blok, **8 kelompok** double frozen, **6 kelompok** metrik dan **8 kelompok** perbandingan. Termasuk 128 kombinasi bab/tab/viewport serta pembandingan semua 41 sheet dengan openpyxl. Tidak ada error JavaScript atau permintaan jaringan eksternal dalam pembaca.

Lingkungan integritas memakai Python 3.13.5, pytest 8.3.5, openpyxl 3.1.5. Browser Chromium tersedia lokal melalui Playwright. Ini pengujian pembaca hasil tersimpan, bukan bukti training encoder atau peningkatan RMSE.

[Preview desktop](../previews/mlp_study/17_hasil_excel_desktop.png) · [Preview HP](../previews/mlp_study/18_hasil_excel_mobile.png).
