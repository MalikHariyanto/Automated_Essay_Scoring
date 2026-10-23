# Verifikasi notebook S3 LoRA + MLP GPU

23 Oktober 2026. Pekerjaan dimulai dari HEAD `9526ac472f5e71837b55db3092d729d37969b41d`, branch `codex/aes-learning`. Notebook diturunkan dari MLP jawaban-saja, bukan context pair. Acuan `notebooks/experiments/AES_300_S3_MLP.ipynb` tetap dengan SHA-256 `d310bef978bdca5fda39f903aaefa442331af5d689c9bb31b081f0bce8dc9e6e`. Notebook utama, double frozen, context pair dan HTML tidak diubah.

## Sumber dan lingkungan

Builder: `python scripts/build_lora_notebook.py`. Notebook dan mirror identik byte, 47 cell/30 code cell, output publik kosong, execution count null. Fingerprint kode: `5d0e387197ab58f55a96b330d327fbce1325afea518a7f6a04186dae698eb6d4`. SHA-256 notebook: `07abbbe7dd93dc33d4132292412f6f2fbf074aa2190c092fa1c821d55d947ad3`. Helper ditanam langsung di cell; tidak ada import proyek saat runtime. LoRA memakai PEFT resmi 0.14.0, Accelerate 1.3.0.

Lingkungan pengujian: Python 3.13.5, PyTorch 2.6.0+cpu, Transformers 4.48.3, PEFT 0.14.0, Accelerate 1.3.0. Tidak tersedia GPU fisik dalam lingkungan ini. Notebook pengguna secara eksplisit memakai metadata GPU, `REQUIRE_GPU=True`, dan `device='cuda'`. Penolakan CPU diuji; CUDA/T4 aktual belum dijalankan di sini. Jalur Python 3.12 tetap memakai Transformers 4.44.2 dari acuan; import API cache yang diperlukan PEFT tersedia pada wheel tersebut, tetapi eksekusi penuh pengujian ini hanya Python 3.13.

## Perintah dan hasil

Cwd `/workspace/Automated_Essay_Scoring`:

```bash
HF_HOME=/workspace/Automated_Essay_Scoring/outputs/lora_hf_cache \
PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
outputs/dataset300_env_py313/bin/python -m pytest \
  tests/test_lora_notebook.py -q --disable-warnings \
  --basetemp=outputs/lora_verification \
  --junitxml=outputs/lora_tests.xml
```

Exit **0**; **14 passed, 0 failed, 0 skipped, 0 xfailed**, satu warning, 67,07 detik. Log lokal `outputs/lora_tests.log`, JUnit `outputs/lora_tests.xml`. Pengujian memakai 24 jawaban buatan dan Transformer satu layer acak berdimensi 768/1024. Seluruh angka dari fixture adalah bukti teknis, **bukan hasil AES penelitian atau ukuran peningkatan akurasi**. Tidak ada training dataset pengguna atau smoke bobot pretrained asli dalam pengujian ini. Source Colab tetap memuat checkpoint asli dan revision terkunci dari acuan.

## Perilaku yang diperiksa

- Rank/alpha/dropout/LR/target LoRA tidak sah ditolak sebelum memuat model. Target query/value harus ditemukan pada seluruh layer attention RoBERTa.
- Inisialisasi B LoRA nol membuat fungsi S3 sama dengan MLP acuan sebelum training pada input fixture yang sama. Split, fungsi metrik, dan pembentukan window identik dengan acuan.
- S1/S2 mempertahankan arsitektur dan bobot inisialisasi acuan. S1 tetap melatih encoder, S2 tidak memuat RoBERTa. LoRA hanya S3.
- Pada backward/optimizer fixture, hanya A/B adapter encoder S3 yang mendapat pembaruan; bobot asli encoder tetap identik. Proyeksi BGE, LayerNorm tambahan dan MLP tetap dilatih. Kelompok optimizer memuat semua parameter trainable tepat sekali; LR adapter terpisah.
- State S3 hanya menyimpan seluruh parameter trainable, tanpa base pretrained frozen. Memuat checkpoint final ringkas menghasilkan prediksi identik. Adapter/head hilang serta alpha/config tidak cocok ditolak.
- Vektor contextual sesudah adapter aktif berbeda dari base pada fixture, sementara lookup embedding tetap sama. Vektor S3 menggunakan wrapper PEFT aktif saat forward.
- Interupsi buatan setelah epoch 1 dan resume menghasilkan state model penuh, riwayat, serta prediksi identik dengan training tiga epoch tanpa interupsi. Semua file epoch tetap ada; manifest merujuk dua terbaru.
- Eksekusi notebook aktual dari awal hingga akhir, dengan import proyek diblokir, menyelesaikan 30 code cell, 15 fold, refit dan test sekali per skenario. GPU/model/tokenizer diganti khusus pada salinan test menjadi CPU dan fixture lokal; file publik tetap GPU wajib.
- Lima Excel dan ZIP terbentuk. Pengaturan LoRA dan tabel trainable/frozen masuk laporan pelatihan; perbandingan guru dan prediksi S1/S2/S3 tersedia. Vektor jawaban S3 sesudah training berdimensi 24×768 dan berbeda dari pretrained; tidak terdapat sheet context pair.
- Notebook yang sama dijalankan lagi pada workspace runtime baru dengan penyimpanan permanen sama: hasil dipakai kembali, tidak ada epoch baru, dan seluruh hash final.pt sama. Tidak ada auto-delete maupun Drive API cleanup.

Notebook terisi, bobot fixture, Excel dan log tetap di `outputs/` yang diabaikan Git. Commit publik hanya source, notebook kosong dan dokumentasi. Hasil penelitian baru perlu dinilai setelah eksperimen pretrained di GPU Colab dijalankan dengan dataset pengguna.

## Perbaikan import torchvision dan preservasi torch runtime

Revisi setelah `e62668093b70be1bfb044e71941813f8fa2b6363` menangani error pengguna `operator torchvision::nms does not exist` pada import PEFT → BloomPreTrainedModel → Transformers → torchvision. Penyebab langsung adalah operator gambar yang tidak tersedia; log pengguna tidak cukup untuk menyimpulkan nomor versi PyTorch/torchvision mana yang berbeda. Audit setup menemukan `--ignore-installed` dengan resolusi dependensi otomatis dapat menarik ulang torch dari dependensi PEFT. Preservasi torch tidak cukup dijamin hanya dengan tidak mencantumkannya sebagai paket utama.

Perbaikan membatasi Transformers ke PyTorch teks, menonaktifkan flag ketersediaan torchvision sebelum import PEFT pada probe **dan worker**, serta menggunakan overlay baru. Flag internal yang dipakai ada pada versi Transformers terkunci 4.44.2/4.48.3; ketidakcocokan versi ditolak. Tidak membuat fake operator NMS dan tidak menghapus paket runtime. Instalasi `--no-deps` dibarengi pemeriksaan dependensi rekursif dan pemasangan paket yang belum cocok dalam maksimum delapan putaran. Permintaan mengganti torch ditolak dan versinya diperiksa ulang setelah pemasangan. Import gagal menampilkan tail traceback langsung.

Perbandingan source notebook terhadap `e626680` setelah menormalkan fingerprint hanya menemukan perubahan pada **cell 5 (setup)** dan **cell 9 (impor worker)**, nomor zero-based. Seluruh konfigurasi LoRA, split, model, optimizer/training, evaluasi dan ekspor identik. Acuan MLP tetap byte-identik. Fingerprint baru `86bcfd7dff74cf2a47517244cadbc12b2c8e6b61e2345da27781f9bd2368b9f5`; SHA-256 notebook `f746fb6792d97b9cba2013d11d0da51a250b0503bbd06ad42dd5521fe22fd3ac`.

Perintah pengujian terbaru, cwd yang sama:

```bash
HF_HOME=/workspace/Automated_Essay_Scoring/outputs/lora_hf_cache \
PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
outputs/dataset300_env_py313/bin/python -m pytest \
  tests/test_lora_setup.py tests/test_lora_notebook.py -q --disable-warnings \
  --basetemp=outputs/lora_setup_fix \
  --junitxml=outputs/lora_setup_fix_tests.xml
```

Exit **0**; **20 passed, 0 failed, 0 skipped, 0 xfailed**, satu warning, 76,30 detik. Log `outputs/lora_setup_fix_tests.log`. Enam pemeriksaan baru mencakup:

- Reproduksi operator NMS hilang melalui paket torchvision fixture yang menjalankan registrasi fake pada dispatcher PyTorch nyata. Import PEFT sebelum perbaikan gagal dengan pesan yang sama; probe baru lolos, lalu forward/backward LoRA pada Transformer kecil acak berhasil tanpa mengimpor torchvision.
- Instalasi dependensi beberapa putaran tidak memasang torch, dengan pemeriksaan argumen pip dan simulasi informasi CUDA runtime. Runtime tanpa torch atau permintaan perubahan torch ditolak, tanpa fallback CPU.
- Pemasangan wheel lokal nyata dalam overlay baru secara offline, dengan dependensi `torch>=2.6.0`, menggunakan torch yang diwariskan. Tidak terbentuk `torch/` atau `torch-*.dist-info` di overlay; probe yang melihat torchvision fixture rusak tetap lolos melalui kebijakan teks.
- Penyebab import asli muncul di SetupError, dan metadata ready tidak diterbitkan ketika pemeriksaan gagal.

Seluruh 14 pemeriksaan sebelumnya, termasuk eksekusi 30 code cell hingga lima Excel dan reuse dari runtime baru, diulang dan lulus pada revisi ini. Tetap tidak ada GPU fisik atau training model pretrained penelitian di lingkungan pengujian; fixture ini membuktikan perbaikan jalur import/instalasi dan fungsi program, bukan RMSE penelitian.
