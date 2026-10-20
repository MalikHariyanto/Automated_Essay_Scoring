# Verifikasi keterbacaan output Colab

20 Oktober 2026. CWD `/workspace/Automated_Essay_Scoring`, branch `codex/aes-learning`, basis `c7bf3301b00be70d4557081455ebfcbb187335fb`.

Perubahan ini menyajikan ulang hasil. Tidak mengganti arsitektur, split, optimizer, rumus metrik, pemilihan epoch, atau prediksi. Perubahan `experiment.py` dan `lesson.py` hanya teks pesan progres. Notebook03 dan mirror diperbarui dari builder yang sama; notebook04 juga mempunyai mirror tanpa output privat.

## Pemeriksaan

| Pemeriksaan | Hasil | Batas bukti |
| --- | --- | --- |
| Suite terkait | 59 passed, 0 failed, 0 skipped, 0 xfailed; exit 0 | Pengujian fungsi dan fixture teknis, bukan ukuran akurasi AES. |
| Pemeriksaan ulang helper tampilan dan worker | 11 passed, exit 0 setelah penyesuaian label persen dan catatan kesamaan vektor | Mencakup alur worker yang benar-benar mengirim tabel HTML. Bukan tambahan 11 kasus unik di luar suite. |
| Notebook03 | Seluruh 23 cell kode selesai; 15 tabel HTML, 89 judul kolom tanpa nama internal berformat underscore; lima Excel di paket akhir | 24 jawaban sintetis, satu epoch, tokenizer dan transformer kecil berbobot acak. |
| Notebook04 | Seluruh tiga cell kode selesai; 38 Excel lama menjadi lima laporan dan lima tabel berpenjelasan di Colab | Membaca arsip fixture hasil lama; tidak memuat model, melatih ulang, atau menghitung ulang metrik. |
| Kesamaan proses | Semua metrik lima fold dan test, serta seluruh CSV prediksi test S1/S2/S3 identik dengan fixture sebelum perapian ini | Model dan dataset fixture yang sama; bukan klaim telah menjalankan checkpoint asli atau training penelitian siswa. |
| Perbandingan kata | Posisi nol di sumber ditampilkan mulai satu; kemiripan cosine dibedakan dari kesamaan seluruh komponen | Nilai sumber tetap sama. Hilangnya informasi tidak disimpulkan sebagai kesamaan atau perbedaan. |
| Data pribadi | Teks HTML di-escape, formula Excel tetap teks, output notebook publik kosong | Eksekusi notebook, Excel, arsip, screenshot dan log lengkap disimpan privat di `outputs/`. |

Nama seperti `occurrence_indices` dan `contextual_allclose_1e-6` diganti dengan posisi kata dan pertanyaan kesamaan vektor. Tabel menjelaskan vektor sebelum/sesudah konteks, cosine, dan batas numerik 0,000001. Batas itu tidak disebut tolerance skor AES. Dictionary `sheets_and_shapes` tidak ditampilkan mentah. Nilai kosong bukan nol, dan tabel tanpa pasangan kata berulang menjelaskan kondisinya.

Command suite:

```bash
PYTHONPATH="$PWD/src" HF_HOME="$PWD/outputs/cache/huggingface" \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
outputs/dataset300_env_py313/bin/python -m pytest -q -rA \
tests/test_teacher_presentation.py tests/test_teacher_excel.py tests/test_teacher_score.py \
tests/test_dataset300_setup.py tests/test_experiment_protocol.py tests/test_trainer.py \
tests/test_pipeline.py tests/test_reporting.py
```

Lingkungan: Python 3.13.5, torch 2.6.0+cpu, Transformers 4.48.3, NumPy 2.2.6, Pandas 2.2.3, openpyxl 3.1.5, pytest 8.3.5. Bukti privat mencatat command, cwd, exit code, versi dan hash source: `colab_readability_tests_proof.json`, `colab_readability_tests.xml`, `colab_readability_process_parity.json`, `colab_readability_04_proof.json`, dan `colab_readability_headers_proof.json`.

ZIP pengguna yang melampaui batas transfer tool chat tidak diklaim telah dibuka. Notebook04 memproses ZIP itu langsung di runtime Colab. Kode dan output di tab Colab yang sudah terbuka tidak otomatis berubah; buka notebook terbaru atau gunakan notebook04 untuk membaca hasil lama tanpa training ulang. Verifikasi SHA remote dan byte file dilakukan sesudah push, terpisah dari pemeriksaan di atas.
