# Verifikasi perpindahan scroll buku MLP

23 Oktober 2026. Pembaruan dari `4b5e7fdee5d3d6b1f9a3f8a1d11d5c8c8ae535a3`, branch `codex/aes-learning`. Tidak ada training, penggantian notebook atau perhitungan ulang hasil penelitian.

## Penyebab dan perubahan

Panel sebelumnya memakai `overscroll-behavior: contain`, sehingga wheel/swipe berhenti di batas panel, termasuk tabel pendek yang tidak mempunyai rentang scroll vertikal. CSS khusus sekarang memakai `overscroll-behavior-y: auto` pada daftar bab, panel buku/kode biasa, output, tabel, perbandingan, metrik dan double frozen. Browser meneruskan gestur ke kontainer induk secara native. Tidak ada penangkap wheel/touch global, pencegahan pinch zoom atau pengguliran sintetis pada aplikasi.

Modal kode dan kanvas fullscreen mempertahankan containment agar halaman belakang tidak bergerak. Satu kanvas kode/catatan, kedua mode blok, pengembalian fokus dan pelepasan kunci scroll saat keluar tetap berlaku. Pencarian baris aktif pada seluruh kode memakai pencarian biner untuk mengurangi pembacaan layout; posisi catatan dan source asli tetap diperiksa.

## Bukti sebelum dan sesudah

`scripts/check_mlp_scroll.mjs` menjalankan wheel mouse dan input sentuhan melalui Chromium. Sebelum perubahan: **9 pemeriksaan lulus, 7 gagal**. Kegagalan mencakup batas atas/bawah buku, batas tabel panjang, tabel pendek, kode biasa, sidebar dan swipe HP. Sesudah perubahan: **16 lulus, 0 gagal**, termasuk perpindahan kursor halaman/panel, isolasi fullscreen dan scroll setelah fullscreen ditutup. Hasil tersimpan di `outputs/mlp_study/scroll_review/baseline.json` dan `verified.json`.

JSON buku dibandingkan dengan versi awal: seluruh kode notebook, sticky note, output, metrik, riwayat, matriks dan snapshot pembanding identik. Perbandingan screenshot desktop statis serta koordinat/ukuran elemen sebelum–sesudah juga identik piksel. Bukti privat: `visual.json`. Pengujian HP memakai simulasi input sentuhan Chromium, bukan perangkat fisik atau klaim kompatibilitas seluruh browser.

## Pemeriksaan ulang

Cwd `/workspace/Automated_Essay_Scoring`:

```bash
AES_SCROLL_TEST_URL=http://127.0.0.1:8765/outputs/mlp_study/scroll_review/before.html \
  node scripts/check_mlp_scroll.mjs --record-baseline
node scripts/check_mlp_scroll.mjs
node scripts/check_mlp_excel.mjs
node scripts/check_mlp_fullscroll.mjs
node scripts/check_mlp_fullscreen.mjs
node scripts/check_mlp_comparison.mjs
node scripts/check_mlp_metrics.mjs
node scripts/check_mlp_frozen.mjs
outputs/dataset300_env_py313/bin/python -m pytest \
  tests/test_mlp_study.py tests/test_mlp_comparison.py tests/test_mlp_frozen.py -q
```

Semua pemeriksaan sesudah perubahan exit **0**. Tes integritas: **17 passed, 0 failed, 0 skipped, 0 xfailed**. Pemeriksa buku lengkap juga dijalankan dengan notebook/ZIP asli dan sampel pembanding openpyxl; seluruh 34 output dan 41 sheet tetap cocok, unduhan XLSX tetap identik, ekspor/buka ulang berhasil, dan script dari output tidak dijalankan. Fullscreen API asli serta fallback viewport diuji; tidak ada error JavaScript atau permintaan jaringan eksternal pada pembaca.

HTML pribadi dibangun ulang dari sumber unggahan yang sama. Unduhan lengkap di `downloads/Belajar_AES_MLP_Lengkap.html` diperbarui dengan kunci pembuka sebelumnya; payload tetap terenkripsi, tidak ada kunci pada repository. Pembukaan di browser memverifikasi seluruh output, pembaca Excel dan kebijakan scroll baru. Dataset dan hasil pribadi tetap berada di dalam payload terenkripsi; perbaikan ini tidak mengubah angka.
