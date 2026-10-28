# Unduhan langsung dari Colab utama

20 Oktober 2026. CWD `/workspace/Automated_Essay_Scoring`, branch `codex/aes-learning`, basis `38ebd6b5b3a5413a62614c2e07880886e72399b2`.

Bagian 15 notebook03 sekarang mempunyai satu cell terakhir: membaca hasil sesi yang aktif, menyusun lima laporan dengan sheet Panduan dan kolom yang sudah dirapikan, memeriksa anggota ZIP, menampilkan daftar isi, lalu memanggil `google.colab.files.download` untuk ZIP yang baru dibuat. Notebook utama tidak mengarahkan pengguna ke notebook perapian lain. Mengunduh ulang pada sesi yang sama tidak menjalankan training/evaluasi kembali.

Pemeriksaan yang dijalankan:

| Pemeriksaan | Hasil |
| --- | --- |
| Suite ekspor, tampilan, worker dan teacher-score | 26 passed, 0 failed, 0 skipped, 0 xfailed; exit 0. |
| Cell unduh utama | Source cell terakhir asli dieksekusi terhadap hasil fixture tersimpan. Fungsi download browser ditiru untuk memeriksa argumennya: tepat ZIP lima laporan dari sesi aktif. Pointer ZIP lama sengaja dipasang sebelumnya dan berhasil diganti. |
| Notebook utama utuh | Semua 22 cell kode selesai, tanpa error, sekitar 29 detik pada fixture teknis. |
| Arsip hasil notebook utama | Lima XLSX yang namanya cocok dengan daftar unduhan; pemeriksaan CRC ZIP lulus; setiap workbook diawali Panduan. |
| Kesamaan angka | Seluruh metrik CV/test dan semua CSV prediksi test S1/S2/S3 identik dengan fixture sebelum perubahan alur unduh ini. |
| Sumber dan publikasi | Workbook sumber tidak ditimpa; notebook publik tanpa output; mirror notebook utama identik. |

Command suite:

```bash
PYTHONPATH="$PWD/src" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
outputs/dataset300_env_py313/bin/python -m pytest -q \
tests/test_teacher_presentation.py tests/test_teacher_excel.py tests/test_teacher_score.py
```

Lingkungan yang digunakan tetap Python 3.13.5, torch 2.6.0+cpu, Transformers 4.48.3, NumPy 2.2.6, Pandas 2.2.3, openpyxl 3.1.5, pytest 8.3.5. Bukti privat `outputs/main_download_proof.json` mencatat command, cwd, exit code, hash source, hasil notebook, daftar anggota ZIP dan perbandingan angka.

Eksekusi utuh memakai 24 jawaban sintetis, satu epoch, tokenizer fixture dan transformer kecil berbobot acak, berlabel TECHNICAL. Ini bukti alur kode, bukan hasil training penelitian atau pengukuran akurasi checkpoint asli. Browser Colab pengguna tidak dikendalikan dalam verifikasi; pemanggilan download diuji melalui pengganti yang merekam path ZIP. Tidak ada perubahan pada perhitungan model, split, pemilihan epoch, atau rumus metrik. SHA branch dan byte file remote diperiksa tersendiri sesudah push.
