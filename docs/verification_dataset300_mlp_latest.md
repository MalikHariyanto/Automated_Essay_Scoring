# Verifikasi Colab utama S3 MLP

Pada 21 Oktober 2026, notebook S3 MLP unggahan pengguna dipromosikan ke `notebooks/03_AES_dataset_guru_300.ipynb` dan salinan identik `aes-learning/notebooks/03_AES_dataset_guru_300.ipynb`. Versi sebelum MLP disimpan di `notebooks/experiments/AES_300_S3_Linear.ipynb`; file eksperimen S3 MLP sebelumnya tetap tersedia.

## Identitas dan privasi

- Seluruh 30 cell kode runtime sama persis dengan unggahan. Konfigurasi dan fingerprint `49f80d493aa531197b6a01e57fcb80e51ff18fb4f2ecda09abfc1f37814f1da7` dipertahankan.
- Output eksekusi dari 14 cell, penghitung eksekusi dan metadata runtime pribadi tidak disertakan. Notebook asli unggahan disimpan privat.
- Source runtime sama dengan eksperimen S3 MLP yang sebelumnya sudah dipublikasikan. Sumber runtime arsip linear sama dengan notebook utama pada commit `883b45740c7379456d0d4fd506020a54e5cd68cd`.
- Pemindaian 300 teks jawaban terhadap tujuh file calon publikasi menghasilkan 2.100 pemeriksaan tanpa kecocokan. Dataset, ZIP hasil, nilai individual dan bobot tidak menjadi bagian publikasi.
- Tidak menambahkan MLP pada S1, double frozen, Ridge, atau penghapusan permanen Sampah Drive. Rencana tersebut ditulis terpisah dan belum dieksekusi.

## Pengujian

Command dari root repository:

```bash
PYTHONPATH="$PWD/src:$PWD/tests" \
HF_HOME="$PWD/outputs/cache/huggingface" \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 JUPYTER_PLATFORM_DIRS=1 \
outputs/dataset300_env_py313/bin/python -m pytest -q -x \
  tests/test_standalone_notebook.py \
  tests/test_teacher_resume.py \
  tests/test_teacher_presentation.py \
  --basetemp=/tmp/aes_promote_latest_tests_fixed \
  --junitxml=outputs/promote_mlp_latest/test_results.xml
```

Hasil akhir: **23 passed, 0 failed, 0 errors, 0 skipped**, exit code 0, durasi pytest 100,68 detik. Lingkungan: Python 3.13.5, PyTorch 2.6.0+cpu, Transformers 4.48.3, NumPy 2.2.6, pandas 2.2.3, openpyxl 3.1.5, pytest 8.3.5, nbclient 0.10.2.

Pengujian menjalankan notebook lengkap tanpa repo, dengan import proyek dan Git diblokir, memakai data sintetis berlabel teknis serta encoder Transformer kecil dengan bobot acak. Kelima fold ketiga skenario, refit, prediksi, evaluasi dan lima Excel dijalankan. Bentuk bobot head S3 MLP diperiksa, keluaran lapisan MLP diekspor, dan skor direkonstruksi dari lapisan terakhir untuk dibandingkan dengan Excel. S1 diverifikasi tetap memakai head linear.

Notebook kemudian dijalankan pada kernel dan ruang kerja lokal baru dengan penyimpanan permanen simulasi yang sama. Hasil dipakai kembali tanpa pelatihan ulang, dengan hash checkpoint/prediksi tetap. Pengujian pemulihan mencakup putus/lanjut, cadangan checkpoint, penulisan gagal, early stopping, dan pemeriksaan GPU yang disimulasikan.

Ini bukti eksekusi teknis, bukan pelatihan ulang penelitian, pengujian GPU T4 fisik, atau akses langsung ke Drive pengguna. Bukti lokal, log dan checksum disimpan di direktori privat `outputs/promote_mlp_latest`. Kesamaan runtime dan fingerprint mendukung kompatibilitas dengan run MLP sebelumnya, tetapi tidak memulihkan checkpoint yang sudah hilang dan tidak menjamin sinkronisasi Google Drive.
