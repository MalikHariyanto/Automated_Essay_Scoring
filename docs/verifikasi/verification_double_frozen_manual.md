# Verifikasi notebook double frozen + MLP dengan checkpoint manual

Notebook baru: `notebooks/experiments/AES_300_S3_DoubleFrozen_MLP_Manual.ipynb`. Sumber alur adalah notebook MLP utama; kelas model dan penjelasan encoder frozen mengikuti eksperimen double frozen. Penyimpanan progres hanya menulis checkpoint dan metadata, tanpa penghapusan atau broker/API pembersihan.

## Identitas

- Notebook utama dan mirror tetap SHA-256 `b06735cb45d632b92ad8ed30990e2e34d0fb66c46c612c528b812dd0d090440b`.
- Notebook eksperimen auto-delete sebelumnya juga tetap identik dengan versi Git sebelum perubahan ini.
- File manual: SHA-256 `3e9af012d9bfe53794a3f6ce7a9c466c799286b25cea436fd650c5bf93198e45`.
- Fingerprint runtime manual: `119151fc00b4e03a447b3817026569a5556cdc54b3101362613902e4c8b882c6`.
- 47 cell, termasuk 30 cell kode, tanpa output eksekusi atau lampiran pribadi. Folder Drive dan lesson version baru memisahkan hasil manual dari run sebelumnya.

## Pengujian

Dari root repository:

```bash
PYTHONPATH="$PWD/src:$PWD/tests" \
HF_HOME="$PWD/outputs/cache/huggingface" \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 JUPYTER_PLATFORM_DIRS=1 \
outputs/dataset300_env_py313/bin/python -m pytest -q -x \
  tests/test_double_frozen_manual.py \
  --junitxml=outputs/double_frozen_manual/final_tests.xml
```

Hasil final: **9 passed, 0 failed, 0 errors, 0 skipped**, exit code 0, durasi pytest 68,39 detik. Lingkungan: Python 3.13.5, PyTorch 2.6.0+cpu, Transformers 4.48.3, NumPy 2.2.6, pandas 2.2.3, openpyxl 3.1.5, pytest 8.3.5, nbclient 0.10.2.

Pengujian memeriksa seluruh checkpoint lima epoch tetap ada, penunjuk memakai dua terbaru, cadangan dapat dimuat bila file terbaru rusak, dan kegagalan penulisan tidak menghapus progres lama. Source notebook tidak membawa kelas/API/broker cleanup, antrean pembersihan atau pemanggilan penghapusan progres setelah tahap selesai.

Definisi kelas model identik dengan eksperimen double frozen: encoder RoBERTa S3 tidak berubah/tidak menerima gradien dan tetap eval ketika model dilatih. LayerNorm tambahan, proyeksi dan MLP tetap memperoleh gradien. Konfigurasi CPU dan deteksi GPU diuji; training yang diputus lalu dilanjutkan menghasilkan bobot, riwayat dan prediksi identik dengan training tanpa putus pada fixture teknis.

Eksekusi notebook lengkap memakai data sintetis 24 baris berlabel teknis dan Transformer kecil berbobot acak. Seluruh 15 fold, refit, prediksi, evaluasi dan lima Excel dijalankan. Semua checkpoint dan metadata kelima fold serta refit ketiga skenario tetap tersedia sesudah selesai. Vektor RoBERTa S3 tetap sama dengan pretrained fixture, head MLP menghasilkan prediksi yang dapat direkonstruksi dari Excel, dan tabel perbandingan guru/prediksi/pembulatan tetap tersedia. Kernel baru memakai hasil tersimpan tanpa training ulang; checkpoint yang dipertahankan tetap ada. Import proyek diblokir dan Git tidak tersedia selama eksekusi mandiri.

Ini bukti perilaku kode, bukan training penelitian baru, akses ke akun Drive pengguna, pengukuran kecepatan CPU/T4 fisik atau bukti peningkatan RMSE. Penyimpanan Drive biasa masih bergantung pada layanan/sinkronisasi Drive; versi ini tidak meminta verifikasi server melalui API cleanup sebelum melanjutkan epoch.

Log, XML, manifest sumber dan bukti privat disimpan di `outputs/double_frozen_manual`; tidak dipublikasikan. Dataset, notebook terisi, bobot dan output individual siswa tidak menjadi bagian publikasi.
