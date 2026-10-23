# Verifikasi kandidat B S3 MLP · CPU

23 Oktober 2026, dari HEAD `f313d08f2268ff4c61cc823038c16e7e4b96a856`, branch `codex/aes-learning`. Notebook eksperimen baru dibangun dari MLP acuan; hanya LR encoder S3 menjadi 5e-6 dan perangkat eksperimen menjadi CPU. Setup memakai perbaikan isolasi teks/installer yang sudah ada, tanpa paket atau model PEFT. Notebook acuan, LoRA, double frozen, context pair dan HTML tidak diubah.

Builder: `python scripts/build_s3_lr_cpu_notebook.py`. Notebook/mirror identik, 47 cell dengan 30 code cell, tanpa output privat. Fingerprint kode `cb641bc95316553b8a700dd2366cbd8bfa3ebb87c9c50544f25dbdc557fbce17`, SHA-256 notebook `adebe8b52f5c974d17b852637b8115c67f41e829a523c015a43ca14547d6c3b5`. Hash MLP sumber dan setup sumber tercatat di metadata. Semua kode runtime ditanam di cell, tanpa clone/import proyek GitHub.

Perintah, cwd `/workspace/Automated_Essay_Scoring`:

```bash
HF_HOME=/workspace/Automated_Essay_Scoring/outputs/lora_hf_cache \
PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
outputs/dataset300_env_py313/bin/python -m pytest \
  tests/test_s3_lr_cpu_notebook.py -q --disable-warnings \
  --basetemp=outputs/s3_lr_cpu_tests \
  --junitxml=outputs/s3_lr_cpu_tests.xml
```

Exit **0**, **12 passed, 0 failed, 0 skipped, 0 xfailed**, satu warning, 69,62 detik. Log `outputs/s3_lr_cpu_tests.log`. Lingkungan Python 3.13.5, PyTorch 2.6.0+cpu, Transformers 4.48.3, NumPy 2.2.6, pandas 2.2.3. Pengujian menggunakan 24 jawaban buatan, tokenizer fixture, dan Transformer satu layer acak dengan dimensi 768/1024; **bukan dataset siswa, model pretrained asli, atau hasil RMSE penelitian**. Notebook Colab tetap menggunakan checkpoint dan revision asli acuan.

Yang diperiksa:

- Builder deterministik; acuan tetap byte-identik, source publik bersih, notebook/mirror identik, tidak ada metadata accelerator GPU. AST model, Engine, make_model, split, metrik, windows, fit, scoring dan ekspor fitur identik dengan acuan. Perubahan hanya pengaturan/perangkat, pemilihan LR S3, penyimpanan manual dan setup.
- Optimizer S1 memakai encoder 1e-5, S2 hanya head 1e-4, S3 encoder 5e-6 + proyeksi/head 1e-4. Parameter muncul tepat sekali; RoBERTa S3 tetap fine-tuned penuh. LR nol/negatif/NaN ditolak.
- Cell konfigurasi mempertahankan CPU baik saat CUDA dilaporkan tersedia maupun tidak tersedia. Flag GPU dari lingkungan tidak memaksa CUDA. Pemeriksaan ini memakai informasi CUDA tiruan; semua komputasi tensor pada pengujian sebenarnya memakai CPU.
- Interupsi setelah epoch 1 dan resume menghasilkan bobot, riwayat dan prediksi identik dengan run tiga epoch tanpa interupsi. LR S3 tercatat 5e-6 dalam informasi optimizer. Semua file epoch tetap ada.
- Probe setup memakai Transformers teks dan lolos saat torchvision fixture memiliki operator NMS rusak. Requirements dan probe tidak memasang/mengimpor PEFT.
- Notebook dijalankan dari atas sampai selesai dengan import proyek **dan PEFT** diblokir. Seluruh 30 code cell, 15 hasil fold, tiga refit dan satu evaluasi test per skenario selesai. Epoch diturunkan menjadi satu khusus pada salinan fixture, tanpa mengubah perangkat CPU atau LR kandidat.
- Lima Excel dan ZIP terbentuk; perbandingan guru/prediksi S1/S2/S3 tersedia, LR S3 tampil terpisah di pengaturan laporan pelatihan, vektor S3 final tersedia tanpa sheet context pair. Checkpoint final ketiga skenario menyatakan device=cpu dan LR yang benar. Semua checkpoint epoch/fold dipertahankan.
- Pada workspace runtime baru yang memakai penyimpanan permanen sama, hasil final dipakai ulang tanpa menjalankan epoch baru dan hash seluruh final.pt tetap identik.

Notebook terisi, log, checkpoint fixture dan Excel tetap di `outputs/` yang diabaikan Git. Belum ada training checkpoint pretrained asli pada dataset pengguna. CPU float32 dan GPU mixed precision dapat menghasilkan perbedaan numerik; pembandingan efek LR yang terkontrol memerlukan perangkat/precision/seed/split yang sama.
