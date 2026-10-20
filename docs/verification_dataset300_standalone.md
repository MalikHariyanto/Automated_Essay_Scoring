# Verifikasi Colab utama mandiri

20 Oktober 2026. CWD `/workspace/Automated_Essay_Scoring`, branch `codex/aes-learning`, basis implementasi `24fbe6ee4e4de05aa49ab08219bb8dda284cf12d`.

Notebook03 sekarang memuat semua kode runtime di cell. Langkah clone/pull, import modul `aes`, pemanggilan script repo, pembacaan daftar pustaka JSON, serta penggunaan Git/source repo untuk identitas hasil dihapus. Bagian pertama menyiapkan ruang kerja. Rujukan ditulis sebagai Markdown, sedangkan identitas hasil menggunakan fingerprint kode notebook. Builder membaca source lokal hanya saat menghasilkan file publik; tidak dijalankan oleh pengguna notebook.

Setup pustaka, protokol proses Python, pemeriksaan data, fungsi model, pembuat tabel, dan pembuat lima Excel semuanya ada di notebook. Proses numerik terpisah tetap digunakan untuk menghindari pustaka lama yang sudah dimuat kernel. Tidak ada source proyek yang diunduh atau ditulis sebagai modul tersembunyi. Daftar kebutuhan runtime berisi NumPy, Pandas, SciPy, scikit-learn, Transformers, dan openpyxl; torch dipertahankan dari runtime atau dipasang bila tidak ada. Pytest/nbformat/nbclient tidak dipasang untuk menjalankan penelitian. Pustaka pihak ketiga dan bobot pretrained tetap diperlukan.

## Bukti yang dijalankan

| Pemeriksaan | Hasil dan batas bukti |
| --- | --- |
| Suite terkait | **35 passed, 0 failed, 0 skipped, 0 xfailed**, exit 0. Mencakup setup, ekspor, tampilan, model, protokol notebook, dan eksekusi mandiri. |
| Notebook mandiri di folder `/tmp` di luar repo | Semua **30 cell kode produksi** selesai, ditambah satu cell penjaga khusus pengujian. Folder kerja awal hanya mempunyai notebook dan dataset fixture; pustaka numerik sudah tersedia pada interpreter pengujian. |
| Blokir ketergantungan repo | Kernel memakai mode isolated dan tanpa `PYTHONPATH`; import `aes` ditolak di kernel dan worker. Perintah Git diganti program yang gagal. Notebook tetap selesai. Tidak membuat file Python proyek, folder `src`, `docs`, atau `.git` di workspace. |
| Keluaran lengkap | Tiga skenario, 15 hasil validasi fold, satu evaluasi test per skenario, dan lima Excel pada ZIP akhir. Pemanggilan download Colab juga diperiksa oleh pengujian cell terakhir. |
| Kesamaan angka | Semua metrik CV/test dan seluruh CSV prediksi test S1/S2/S3 identik persis dengan fixture sebelum pemindahan kode. Tidak mengubah metode model atau metrik. |
| Audit Excel pengguna | Semua 30 cell kode selesai dalam mode audit; 300 jawaban, 240 development, 60 test, lima fold. Hanya satu laporan data yang tersedia; tidak dibuat prediksi/evaluasi pengganti. Tidak memuat checkpoint atau melatih model pada audit ini. |
| Jalur setup default | Audit juga selesai tanpa `AES300_PYTHON` override. Worker berada di workspace baru; pustaka runtime disimulasikan tersedia melalui `.pth`, seperti pustaka bawaan Colab. Torch dipakai ulang, 36 dependensi aktif sesuai, tanpa instalasi ulang. |
| Bentuk publik | 47 cell total, 30 kode; sintaks valid, seluruh output kosong, dan mirror notebook identik. |

Command suite:

```bash
PYTHONPATH="$PWD/src" HF_HOME="$PWD/outputs/cache/huggingface" \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 JUPYTER_PLATFORM_DIRS=1 \
outputs/dataset300_env_py313/bin/python -m pytest -q -rA \
--basetemp=/tmp/aes_standalone_final_suite \
tests/test_standalone_notebook.py tests/test_dataset300_setup.py \
tests/test_teacher_presentation.py tests/test_teacher_excel.py tests/test_teacher_score.py
```

`PYTHONPATH` pada command di atas hanya untuk controller pengujian modul repo. Kernel notebook menggunakan `-I`, `PYTHONPATH` dibuang, dan worker menolak import proyek pada uji mandiri. Fixture memakai 24 jawaban sintetis, satu epoch, tokenizer fixture, dan transformer kecil berbobot acak dengan dimensi 768/1024. Label TECHNICAL diteruskan ke laporan; ini bukan ukuran akurasi AES atau hasil checkpoint pretrained asli.

Lingkungan: Python 3.13.5, torch 2.6.0+cpu, Transformers 4.48.3, NumPy 2.2.6, Pandas 2.2.3, openpyxl 3.1.5, pytest 8.3.5, nbclient 0.10.2, ipykernel 6.29.5. Hardware GPU Colab dan eksekusi checkpoint pretrained asli tidak diklaim diuji pada perubahan struktur ini. Pemasangan pip nyata dan preservasi torch dicakup pengujian setup yang sama dengan versi sebelumnya.

Bukti privat: `outputs/standalone_tests_proof.json`, XML/log pengujian, `standalone_parity_proof.json`, `standalone_real_audit_proof.json`, `standalone_default_setup_proof.json`, serta notebook terisi di folder pengujian `/tmp`. Dataset, notebook terisi, log, checkpoint dan hasil tidak diterbitkan. SHA branch serta byte file GitHub diverifikasi setelah push.
