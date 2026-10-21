# Penyimpanan progres dan pemeriksaan GPU

21 Oktober 2026. Basis perubahan `00f6c6c` pada branch `codex/aes-learning`. Notebook utama dan mirror tetap mandiri, dengan seluruh kode runtime tersedia di cell dan tanpa output privat di GitHub.

## Perilaku baru

Di Colab, Google Drive dihubungkan melalui `drive.mount`; otorisasi akun tetap dilakukan pengguna. Dataset, hasil dan checkpoint berada di `MyDrive/AES_300_Privat`. Lingkungan Python dan cache bobot Hugging Face tetap di penyimpanan lokal. Bila koneksi/izin/penulisan Drive gagal, proses berhenti tanpa fallback diam-diam ke `/content`. Lingkungan lokal dapat memakai `AES300_PERSISTENT_DIR` untuk penyimpanan permanen terpisah dari `AES300_WORKSPACE`.

Setiap epoch lengkap menyimpan keadaan model, optimizer, scheduler, GradScaler, bobot terbaik, riwayat, penghitung early stopping, generator pengacakan data, serta RNG Python/NumPy/torch/CUDA. Manifest hanya diperbarui setelah file selesai ditulis dan checksum dihitung; dua generasi terakhir dipertahankan selama tahap aktif. Resume memverifikasi data, fold, pengaturan dan perangkat/precision. Checksum dibaca bertahap untuk menghindari pemuatan file checkpoint besar sekaligus ke RAM. Fold selesai dilewati; pelatihan final yang sudah selesai juga dilewati ketika sesi putus di antara skenario. Checkpoint sementara dibersihkan setelah hasil tahap tersimpan.

Pemeriksaan GPU menampilkan nama hardware, status T4, CUDA, dan VRAM. Colab berhenti sebelum training jika GPU tidak aktif; hardware lain yang tersedia tetap dilaporkan dengan nama sebenarnya. Alokasi T4 tetap dipilih melalui menu Runtime. Batas 25 epoch, kandidat A, LayerNorm S3, split dan rumus metrik tidak berubah.

## Bukti pengujian

Suite final: **61 passed, 0 failed, 0 skipped, 0 xfailed**, exit 0, durasi 131,53 detik. Pengujian notebook menjalankan seluruh 30 cell kode produksi pada sesi pertama dan sesi baru; hasil lengkap digunakan kembali tanpa training ulang. Kedua notebook publik identik dan berisi output kosong. Bukti privat ada di `outputs/resume_tests*` dan folder pengujian `/tmp/aes_resume_final_suite`.

Perintah dari root repo:

```bash
PYTHONPATH="$PWD/src" HF_HOME="$PWD/outputs/cache/huggingface" \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 JUPYTER_PLATFORM_DIRS=1 \
outputs/dataset300_env_py313/bin/python -m pytest -q -rA \
--basetemp=/tmp/aes_resume_final_suite --junitxml=outputs/resume_tests.xml \
tests/test_teacher_resume.py tests/test_teacher_training_revision.py \
tests/test_standalone_notebook.py tests/test_dataset300_setup.py \
tests/test_teacher_presentation.py tests/test_teacher_excel.py tests/test_teacher_score.py
```

Lingkungan: Python 3.13.5, torch 2.6.0+cpu, Transformers 4.48.3, NumPy 2.2.6, Pandas 2.2.3, openpyxl 3.1.5, pytest 8.3.5, nbclient 0.10.2. Manifest hash source memastikan file kode/notebook yang dipublikasikan sama dengan yang diuji. SHA branch remote dan byte file utama diperiksa setelah push.

Uji putus/lanjut menggunakan data sintetis dan transformer kecil berbobot acak dengan dimensi 768/1024. Untuk S1/S2/S3, training terputus setelah checkpoint epoch kedua selesai; bobot, riwayat dan prediksi setelah resume dibandingkan persis dengan training tanpa putus, termasuk dropout aktif. Ada pemeriksaan terpisah untuk pemulihan penghitung early stopping dan bobot terbaik, checkpoint terbaru rusak dengan cadangan utuh, penulisan gagal, perubahan konfigurasi, dan putus saat refit S2 setelah refit S1 selesai.

Notebook dieksekusi di folder tanpa repo, dengan import proyek dan Git diblokir. Penyimpanan permanen disimulasikan dengan folder privat terpisah. Notebook kemudian dijalankan ulang dalam kernel dan workspace lokal baru, menggunakan penyimpanan permanen yang sama: hasil training lengkap dipakai kembali tanpa fit ulang, checkpoint/prediksi tidak berubah, dan lima Excel tetap terbentuk. Kegagalan otorisasi Drive serta identifikasi GPU T4/L4/CPU diuji dengan stub terkontrol.

Pengujian ini tidak mengakses Drive pengguna atau GPU T4 fisik dan bukan eksperimen akurasi pada dataset guru. Sinkronisasi layanan Drive/Colab tidak diklaim menjamin nol kehilangan data. Epoch yang belum selesai tersimpan perlu diulang; progres yang sudah hilang sebelum fitur ini tidak bisa dipulihkan. Simpan notebook, dataset, konfigurasi, versi pustaka dan perangkat konsisten untuk melanjutkan.
