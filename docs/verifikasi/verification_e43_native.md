# Verifikasi E43 masked mean native — 4 November 2026

E43 mengganti attention pooling setelah matching E42 dengan masked mean. Verifikasi berikut memakai data sintetis dan Transformer kecil berbobot acak di CPU. Belum ada training penelitian GPU atau hasil AES E43.

## Identitas sumber

Cwd `/workspace/Automated_Essay_Scoring`, baseline `5a5cc596059bb27f020a3a87dc49abbdecc80103`. Builder `scripts/build_e43_notebook.py` membaca E42 yang dipin ke commit tersebut. Notebook utama dan mirror identik, 184 cell/89 cell kode, seluruh output kosong. Raw SHA256 `7955c87e97982d5c2c4f85471f954e13bf14cb941236aa6a64cafab1aa4c45b4`; visible-code SHA256 `d2b5ed1bed09d902429b312d39c5153aa3c321bb91dc386b46605583734f31ee`.

S3 saja, satu 5-fold, maksimal 25 epoch, patience 5, satu refit dan satu test. Encoder 5e-6, komponen lain 1e-4, Scalar Mix empat layer + gamma, BiGRU64 per arah, BGE frozen dan MLP tetap. Tidak ada sweep seed atau pelatihan ulang kontrol otomatis. Berkas E42/E30 dan HTML lama tidak berubah.

## Perintah dan hasil akhir

```bash
/tmp/e39-verify/bin/python -m pytest tests/test_e43_notebook.py \
 -q -s --tb=short --basetemp=outputs/e43_study/final_fixtures
```

Exit 0; **4 passed, 0 failed, 0 skipped, 0 xfailed**, 85,59 detik menurut pytest (87,90 detik termasuk startup proses). Satu warning deprecation plugin pytest lingkungan. Bukti privat: `outputs/e43_study/final_tests.log`, `test_command.json`, `verification.json`, dan fixture di `final_fixtures`. Percobaan awal menemukan assertion nama kolom `Window`, sedangkan Excel menggunakan `Bagian Teks`; pemeriksaan diperbaiki sebelum suite akhir ini. Eksekusi notebook awal sudah selesai; kesalahan tersebut ada pada nama kolom dalam tes, bukan pada hasil ekspor.

Empat pemeriksaan mencakup:

1. Schema, hash, mirror, rebuild deterministik dan source E42 utuh. Output kosong, Python biasa tanpa magic/exec/eval/worker tersembunyi/cosine/LLRD/loop multi-seed. Optimizer, fit, train_all, split, metrik, ScalarMix, encoder, BGE, mask/windowing identik dengan E42 secara AST; nama versi pada pesan error Engine dinormalisasi untuk perbandingan.
2. Masked mean sama dengan kontrol E42 yang skor pooling-nya dibuat nol sehingga softmax seragam. Token valid mendapat 1/N, padding 0, jumlah bobot 1. Mengubah token di luar respons/referensi menjadi angka besar tidak mengubah keluaran modul. Bobot co-attention tetap finite dan ternormalisasi; gradien BiGRU, proyeksi referensi, matching dan output nonzero. Respons dan referensi menerima gradien, padding tidak.
3. Semua bobot bersama model dan keadaan RNG awal identik dengan E42 pada seed sama; tepat 8.320 parameter skor pooling hilang. Tidak ada `attention_hidden`/`attention_score` di optimizer/checkpoint. Scalar Mix logit/gamma, encoder dan MLP menerima gradien; dimensi 128/768/1024 terjaga. Semua parameter masuk tepat sekali ke optimizer. Save/load menjaga gamma dan loader menolak konfigurasi pooling lama.
4. Seluruh 89 cell kode dijalankan dalam kernel terisolasi dengan 24 baris sintetis, random RoBERTa/XLM-R empat layer berdimensi 768/1024, CPU, satu epoch dan token pendek. Source publik tetap pretrained/GPU/300 baris/25 epoch. Lima fold, satu refit, satu test, lima Excel dan ZIP bernama benar selesai. Kernel baru memakai checkpoint final tanpa perubahan hash. X/y terlihat cocok dengan masukan training, tanpa import proyek atau worker tersembunyi. Excel 02 menamai bobot mean dengan benar dan setiap window berbobot 1/N; sheet co-attention tetap tersedia. Excel 05 memuat 24 baris Scalar Mix/gamma. MAE dan jumlah toleransi validasi per-fold serta OOF dihitung ulang dari prediksi mentah dan cocok; setiap ID OOF unik dan terpisah dari test. Norma sesudah gamma cocok dengan |gamma|×norma sebelum.

## Lingkungan dan batas bukti

Python 3.12.14, torch 2.6.0+cpu, transformers 4.44.2, pytest 9.2.0, nbformat 5.11.1, nbclient 0.11.0, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0, openpyxl 3.1.5. Kernel Jupyter memerlukan socket localhost. Tes ini tidak mengukur kualitas AES, menjalankan checkpoint pretrained asli pada GPU, atau membuktikan kompatibilitas seluruh runtime Colab.

Mean-over-time bersumber dari Taghipour & Ng (2016) dan implementasi penulis; pemakaiannya setelah matching fusion ini adalah adaptasi. Belum ada bukti E43 mengungguli E42/E30. Hanya notebook kosong, builder, tes dan dokumentasi aman dipublikasikan. Dataset, unggahan, executed notebook, Excel, cache dan bobot tetap privat. Publikasi diperiksa dengan SHA branch remote dan hash byte notebook utama, mirror serta dokumen metode; bukti dicatat privat setelah push.
