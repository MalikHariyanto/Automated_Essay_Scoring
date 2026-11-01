# E29 + Ridge · notebook Colab

Tiga input `soal`, `jawaban_referensi`, `response_text`; target `rater` 0–100.
RoBERTa fine-tuned → token respons → BiGRU64 per arah → attention pooling128 → proyeksi768 → LayerNorm; concat BGE frozen1024 → proyeksi256 → LayerNorm menghasilkan1024 fitur. Representasi dilatih menggunakan MLP bantu1024→256→1 dengan LR5e-6. Encoder5e-6; BiGRU, attention, proyeksi/norm1e-4. Ridge menjadi prediktor akhir; checkpoint yang sama mempertahankan MLP pembanding.

Source arsitektur, optimizer dan tokenisasi berasal dari E29 pada commit `47ddb8280592231fbb682f2e6a8b3c5735d185fb`. File E29 asli tidak diubah. Builder: `scripts/build_e29_ridge_notebook.py`. Runtime nested CV diembed dari `scripts/s3_ridge_experiment.py` dengan identitas E29.

Notebook mandiri: pilih T4 GPU, Run all, unggah `fix_dataset_300.xlsx`. Tanpa clone/Drive. Lima Excel dan ZIP hasil tersedia di akhir. Checkpoint representasi final besar disimpan lokal; aktifkan `DOWNLOAD_FULL_ENCODER=True` untuk mengunduhnya. NPZ Ridge saja tidak cukup untuk menilai teks baru.

## Protokol dan batas perbandingan

Split kelompok development/test dan lima fold mengikuti keluarga asli. Grid alpha0,01/0,1/1/10/100 dipilih memakai nested CV; representasi supervised juga dilatih ulang di inner split. Lima outer + sepuluh pasangan inner + satu final =16training representasi, masing-masing25epoch tetap. Label selain train ditutup NaN pada setiap fit; test tidak memilih alpha/epoch. BGE frozen dapat dihitung bersama karena tidak belajar dari label.

Anggaran25epoch tetap menggantikan early stopping historis E29 untuk menjaga outer/inner validation di luar training representasi. Karena itu ini turunan arsitektur/LR E29, bukan reproduksi persis run historis. Pembanding langsung ialah MLP yang dilatih di notebook yang sama. Baseline tambahan BGE→Ridge alpha0,01 dilaporkan terpisah.

Belum ada hasil penelitian E29 Ridge. Angka E29 historis bukan hasil notebook baru. Test historis sudah berulang kali dilihat sehingga perbandingan berikutnya eksploratif. Pengujian teknis dengan data/model buatan tidak mengukur akurasi penelitian atau membuktikan keberhasilan pada GPU Colab.
