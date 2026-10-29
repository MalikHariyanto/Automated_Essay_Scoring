# Verifikasi E29/E30 · dua turunan E26

**29 Oktober2026.** [Metode dan source](../eksperimen/eksperimen_e29_e30_gpu.md). Hasil penelitian belum tersedia; tidak ada training ulang pada dataset pengguna atau perubahan buku hasil.

Working directory `/workspace/Automated_Essay_Scoring`:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
outputs/dataset300_env_py313/bin/python -m pytest -q \
  tests/test_e29_e30_notebooks.py \
  --basetemp=outputs/e29_e30_verification/fixtures \
  --junitxml=outputs/e29_e30_verification/final.xml
```

**Exit0;15passed,0failed/errors/skipped/xfail/xpass**,144,28detik. Satu warning deprecation jalur Jupyter. Python3.13.5, PyTorch2.6.0+cpu, Transformers4.48.3, NumPy2.2.6. CUDA tidak tersedia pada lingkungan pemeriksaan. Jupyter membutuhkan socket lokal sehingga eksekusi fixture menggunakan izin yang sesuai di luar pembatasan socket sandbox.

## Cakupan bukti

- Builder deterministik;47cell/30cellkode per notebook; checksum kode valid; source tanpa output/count eksekusi; mirror identik; tanpa impor proyek atau mount Drive. Source publik tetap meminta GPU; fixture secara eksplisit mengganti perangkat ke CPU.
- E29 mempertahankan source arsitektur/tokenisasi/fitur/evaluator E26. Seluruh parameter awal E26 identik pada dua resep; S1/S2 dan S3 E29 memberi prediksi awal sama dengan E26 pada fixture. Tidak mengklaim prediksi awal S3 E30 identik karena forward berubah.
- Semua skenario: output `[B,1]`, concat S3 `[B,1024]`, ringkasan128, BGE frozen, gradien RoBERTa/GRU/pooling/proyeksi/head, cakupan optimizer tepat sekali, dan state_dict reload mempertahankan prediksi.
- E29: LR `head.*` S2 dan S3 tepat5e-6; linear S1, proyeksi/BiGRU/pooling/norm1e-4, encoder5e-6. Pada langkah AdamW pertama, update non-MLP identik dengan kontrol1e-4; norm update MLP sekitar20×lebih kecil, diuji pada S2 dan S3.
- E30: perubahan parameter attention pooling mengubah bobot/ringkasan tanpa mengubah matriks co-attention pada input yang sama. Kedua tahap mempunyai gradien; gradien referensi/matching, GRU dan encoder tetap mengalir. Mask, padding, panjang berbeda, urutan batch dan pengaruh referensi diuji; softmax finite dan jumlah1. AMP bfloat16 CPU diuji, bukan AMP GPU.
- Posisi referensi/respons berasal dari layout; padding kiri/kanan, kata berulang pada prefix/soal/respons, truncation referensi kosong, dan penolakan ID offset tidak cocok diuji. Slow/fast RoBERTa API diuji memakai kosakata Byte-BPE buatan lokal, termasuk Unicode; bukan smoke tokenizer pretrained Indonesia.

## Seluruh cell dan keluaran

Setiap notebook dijalankan dengan workbook **24baris sintetis**, Transformer satu layer berbobot acak (dimensi768/1024 tetap), CPU, satu epoch. Worker melarang impor modul proyek untuk memeriksa kemandirian source. Masing-masing menyelesaikan30cellkode,15CVfit+3refit, final test teknis dan lima Excel.

Notebook dijalankan lagi pada kernel baru dengan folder lokal sama: hash ketiga checkpoint final tetap; training/test tidak diulang. Durasi eksekusi dua kernel sekitar70,78detik E30 dan65,45detik E29. LR nyata pada config, checkpoint S1/S2/S3 dan Excel diperiksa.

E30 mengekspor tiga bobot berbeda dengan jumlah1 per jawaban/window: pooling respons setelah matching, rata-rata co-attention referensi→respons, rata-rata respons→referensi. Sheet pasangan top1 dan ringkasan128 juga diperiksa. Semua output fixture tetap lokal dan bukan hasil akurasi penelitian.

| Notebook | SHA256 file | Fingerprint kode |
| --- | --- | --- |
| E29 | `d1791866de266fbbca8b8293d99f3fbc585460312fc8f1a870e8f4221cfbc04a` | `e04c2780419d08e4b625163c8133b1437b2ac9e52899a839e9dc4bbed521c8b1` |
| E30 | `19aec0a642338457265444fe9d16864810ec5f17eddbe4dbc1d75924bd2b2921` | `0c3529817eab5600fb3a3416caedb33ec276a0522abddcdb61d297bf0cc7a7d1` |

Seluruh58notebook terdahulu dan10berkas aplikasi/pengunduh/paket tidak berubah. Source E26 dipin `bc4713872c7a7ce23f7dad8ae3ea73a2b75d9b86`. Log/JUnit/versi/bukti eksekusi dan manifest sumber tersimpan lokal pada `outputs/e29_e30_verification/`.

Ini bukti implementasi dan fixture. Pretrained RoBERTa/BGE pada300jawaban, CUDA/AMP GPU asli, serta hasil penelitian E29/E30 belum dijalankan dalam pemeriksaan ini. Tidak menyatakan LR lebih kecil atau dua attention pasti meningkatkan prediksi.
