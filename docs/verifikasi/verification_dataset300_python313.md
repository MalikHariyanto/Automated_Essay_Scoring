# Perbaikan setup Colab Python3.13

17 Oktober2026. Basis perubahan `00e9264de2fac645ab14fe95f60d69ad7d91223a`, branch `codex/aes-learning`. Traceback pengguna menunjukkan kernel `/usr/lib/python3.13`; setup versi awal menolak semua versi selain3.12 sebelum instalasi berjalan. `CalledProcessError` hanyalah pembungkus kegagalan subprocess tersebut.

Perbaikan menerima3.12/3.13, memisahkan environment antarversi, dan memilih dependency sesuai versi Python. Transformers4.44.2/tokenizers0.19 tetap digunakan pada3.12;3.13 menggunakan Transformers4.48.3/tokenizers0.21.4 yang mempunyai wheel kompatibel. Torch runtime yang sudah tersedia tidak diganti. Notebook membaca path worker terverifikasi dari metadata setup dan menyertakan tail log jika pemasangan gagal.

## Bukti yang dijalankan

Semua command dari root `/workspace/Automated_Essay_Scoring`. Log/manifest rinci di `outputs/teacher_dataset300`, tidak dipublikasikan.

| Pemeriksaan | Hasil |
| --- | --- |
| Setup nyata Python3.13.5 | Exit0; `scripts/setup_dataset300.py` selesai memasang dan mengimpor torch2.6.0+cpu, Transformers4.48.3, NumPy2.2.6, Pandas2.2.3, SciPy1.15.3, sklearn1.6.1, openpyxl3.1.5 |
| Pengulangan setup di kernel3.13 | Exit0; pemasangan ulang idempoten, metadata worker cocok runtime |
| Workflow baru + regresi setup pada3.13 | **17 passed,0 failed,0 skipped,0 xfailed**; exit0 |
| Workflow baru + regresi setup pada3.12 | **17 passed**, exit0;1 FutureWarning tokenizer fixture |
| Seluruh suite pada3.12 | **78 passed,0 failed,0 skipped,0 xfailed**, exit0;1 FutureWarning tokenizer fixture |
| Eksekusi notebook memakai kernel3.13 sungguhan | **16 code cell selesai tanpa error**, termasuk setup tanpa `AES300_PYTHON` override, audit Excel asli dan split240/60; exit0 |
| Bentuk publik notebook | 33 cell; semua output kosong, sintaks valid, mirror identik |

Command test terkait pada kedua interpreter:

```bash
PYTHONPATH="$PWD/src" HF_HOME="$PWD/outputs/cache/huggingface" \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
outputs/dataset300_env_py313/bin/python -m pytest -q \
tests/test_dataset300_setup.py tests/test_teacher_score.py
```

Untuk3.12, path interpreter adalah `outputs/dataset300_env/bin/python`. Seluruh suite3.12 menggunakan `AES_RSCRIPT=/workspace/aes-r-local/bin/Rscript` dan `python -m pytest -q`.

Suite penuh juga dicoba pada3.13: **76 passed,2 failed**, exit1. Dua kegagalan berada pada test bootstrap notebook lama01/02 yang memang mensyaratkan Python3.12; guard runtime lama berjalan sebelum test timeout/clone. File/persyaratan notebook lama tidak diubah atau dinonaktifkan. Dukungan3.13 yang ditambahkan berlaku untuk notebook03, yang seluruh17 test terkaitnya lulus. Laporan ini tidak menyatakan semua notebook lama mendukung3.13.

Notebook diuji dengan `nbclient` dan kernelspec privat yang menunjuk interpreter3.13, `AES300_AUDIT_ONLY=1`, serta path Excel unggahan privat. Setup benar-benar dieksekusi dari cell notebook, bukan hanya dilewati menggunakan environment override. Executed notebook disimpan privat. SHA256 notebook publik baru dan mirror: `cb0e4460c0d684544b810e2b1dcad0df728618bee2c9abd0c82f1164ae2612a9`.

Tidak ada checkpoint pretrained yang dimuat atau training penelitian yang dijalankan pada verifikasi ini. CPU berhasil diuji; preservasi torch bawaan runtime dicakup test routing setup, tetapi GPU/AMP Colab belum diuji pada hardware nyata. Tidak ada data siswa atau output terisi yang masuk commit. Parameter/model/split penelitian tidak diubah oleh perbaikan ini.
