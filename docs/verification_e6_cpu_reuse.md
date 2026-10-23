# Verifikasi E6 CPU reuse S1/S2

**Catatan versi:** bukti di bawah mencatat versi awal dengan pemeriksaan PyTorch identik. Perbaikan portabilitas model final dan hasil pengujian terbaru ada di [verifikasi lintas PyTorch](verification_e6_reuse_torch_portability.md).

Notebook: `notebooks/experiments/AES_300_E6_S3_LR5e6_CPU_Reuse.ipynb`, mirror byte-identik. SHA-256 file: `c932093180f261d4ae1ec0d60104a4cf6d208ea9dbe0624aed8fbd9497c8bab6`. Fingerprint kode: `4b1a64dd6c5a8340acdc8d9dca9311a7d8c89cc5a119f93c567ed5f8e35fad02`.

Dijalankan dari root repo:

```bash
HF_HOME=/workspace/Automated_Essay_Scoring/outputs/lora_hf_cache \
PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
outputs/dataset300_env_py313/bin/python -m pytest \
  tests/test_s3_cpu_reuse.py -q --disable-warnings \
  --basetemp=outputs/e6_cpu_reuse_tests \
  --junitxml=outputs/e6_cpu_reuse_tests.xml
```

Exit **0**; **6 passed, 0 failed, 0 skipped, 0 xfailed**, satu warning, 70,10 detik. Log privat `outputs/e6_cpu_reuse_tests.log`. Lingkungan Python 3.13.5, Torch 2.6.0+cpu, Transformers 4.48.3, NumPy 2.2.6, pandas 2.2.3.

- Builder deterministik, notebook valid tanpa output, mirror identik, sumber GPU tetap utuh. AST kelas/fungsi model, optimizer, training, reuse, split, metrik, scoring dan ekspor sama dengan versi GPU.
- Konfigurasi CPU tetap dipilih ketika CUDA tersedia maupun tidak, termasuk jika environment lama meminta GPU. LR S3 tetap 5e-6.
- Reuse hanya memanggil training S3: lima fold dan refit; kontrol S1/S2 serta seluruh file sumber tetap utuh.
- Metadata sumber CUDA disimulasikan pada checkpoint fixture: model final berhasil dimuat CPU setelah checksum/profil/config diperbarui secara konsisten. Bobot fixture dibuat di CPU; ini bukan bukti eksekusi GPU fisik.
- Semua code cell dijalankan hingga lima Excel dan ZIP tanpa mengganti konfigurasi perangkat CPU pada salinan test. S1/S2 tidak dilatih ulang, test baru hanya S3. Run kedua dari runtime baru memakai kembali hasil selesai dengan hash final sama.

Pengujian memakai 24 jawaban buatan dan Transformer satu layer acak dimensi 768/1024, dengan epoch/dataset/model fixture. Bukan training checkpoint pretrained pada dataset penelitian; tidak membuktikan RMSE E6. Tidak ada GPU fisik yang diuji. Notebook terisi, bobot, Excel dan log tetap pada `outputs/` yang diabaikan Git. Pemeriksaan negatif reuse yang sama sudah tercakup pada [verifikasi versi GPU](verification_e6_gpu_reuse.md).
