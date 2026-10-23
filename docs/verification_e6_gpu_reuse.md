# Verifikasi E6 GPU dengan reuse S1/S2

23 Oktober 2026. Dari HEAD `319088f1e649d642ca014b4048e73f79b475cb6f`, branch `codex/aes-learning`. Notebook GPU baru diturunkan dari E6 CPU tanpa menimpa notebook CPU, MLP acuan, LoRA, double frozen atau HTML. Builder `python scripts/build_s3_lr_gpu_reuse_notebook.py`; helper `src/aes/teacher_score/reuse_controls.py` ditanam langsung pada cell. Runtime tidak clone/import proyek.

Notebook publik/mirror identik, 47 cell/30 code cell, output kosong. Fingerprint kode `b82810010670f313a3ad84e610d230dd3b664613e7af643a805984fd1a0b564e`, SHA-256 notebook `48416a85da986efd751b0c8820b972e77999eb4e0306026d2a2f867ebcd8e34e`. Metadata meminta GPU, konfigurasi memakai `device='cuda'` dan `REQUIRE_GPU=True`. LR S3 5e-6; proyeksi/head 1e-4. S1/S2 tetap menggunakan kontrol selesai yang konfigurasi/metodenya sama.

## Perintah dan hasil

Cwd `/workspace/Automated_Essay_Scoring`:

```bash
HF_HOME=/workspace/Automated_Essay_Scoring/outputs/lora_hf_cache \
PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
outputs/dataset300_env_py313/bin/python -m pytest \
  tests/test_s3_gpu_reuse.py -q --disable-warnings \
  --basetemp=outputs/e6_gpu_reuse_verified \
  --junitxml=outputs/e6_gpu_reuse_verified.xml
```

Exit **0**, **12 passed, 0 failed, 0 skipped, 0 xfailed**, satu warning, 78,25 detik. Log lokal `outputs/e6_gpu_reuse_verified.log`. Lingkungan Python 3.13.5, Torch 2.6.0+cpu, Transformers 4.48.3, NumPy 2.2.6, pandas 2.2.3. **Tidak tersedia GPU fisik atau smoke checkpoint pretrained penelitian.** Pengujian memakai 24 jawaban buatan dan Transformer satu layer acak dimensi 768/1024; informasi identitas source profile disubstitusi secara eksplisit pada fixture metode S1/S2 yang sama. Semua angka adalah bukti teknis, bukan nilai penelitian.

## Perilaku yang dibuktikan

- Builder deterministik, source publik bersih, mirror identik dan notebook CPU sumber tetap byte-identik. Kode runtime tidak memerlukan PEFT atau import proyek; metadata/perangkat GPU tetap pada artefak publik.
- Reuse memeriksa profile kode yang dikenal, dataset, split, konfigurasi yang relevan, versi numerik, lima fold, pilihan epoch, checksum final/prediksi test, serta ID dan nilai guru yang sama. Model kontrol dimuat ke arsitektur saat ini sebelum reuse diterima.
- Source dataset/split/LR/versi numerik/checksum/profile yang berbeda ditolak sebelum training. Source yang hilang menghentikan proses tanpa melatih S1/S2 secara diam-diam.
- Dalam run tujuan, `fit` dipanggil **enam kali, seluruhnya S3**: lima fold dan refit. Wrapper test menggagalkan setiap percobaan melatih S1/S2. Model final/prediksi test S1/S2 sama byte dengan sumber, dan tidak ada checkpoint epoch S1/S2 baru. S3 memakai LR encoder 5e-6 dan menghasilkan bobot baru; S3 sumber tidak disalin.
- Seluruh hash file sumber, termasuk S3 sumber, identik sebelum dan sesudah reuse/training. Tidak ada penghapusan maupun penulisan sumber lama.
- Penyalinan yang terputus dipulihkan dari rencana sumber/checksum terkunci. File tujuan yang sudah berubah ditolak. Pencarian otomatis mengikuti urutan folder yang dinyatakan, bukan metrik performa.
- Eksekusi notebook lengkap memakai GPU guard/device override **khusus pada salinan fixture** menjadi CPU. Sumber kontrol disiapkan dari run fixture selesai, lalu import proyek/PEFT diblokir. Seluruh 30 code cell selesai, menghasilkan 15 baris CV (10 reuse + 5 baru), lima Excel dan ZIP. Hasil test S1/S2 dipakai ulang; report mencatat evaluasi test baru S1=0, S2=0, S3=1. Model asal tampil di sheet `Asal Model` dan perbandingan guru/prediksi tetap lengkap.
- Eksekusi kedua dari workspace runtime baru memakai hasil tersimpan dengan hash final.pt identik. Tidak ada kebutuhan menghapus S3 atau manifest sumber untuk melanjutkan eksperimen ini.

Checkpoint/Excel/notebook terisi/log fixture tetap pada `outputs/` yang diabaikan Git. Belum ada klaim RMSE penelitian E6 baru. Hasil S1/S2 pada laporan E6 harus dijelaskan sebagai reuse dari sumber, bukan training baru. Pemilihan LR melalui CV; test lama yang sudah dilihat tidak menjadi konfirmasi independen.
