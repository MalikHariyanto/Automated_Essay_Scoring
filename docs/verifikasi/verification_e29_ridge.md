# Verifikasi E29 + Ridge · 1 November 2026

Cwd: `/workspace/Automated_Essay_Scoring`.
Source arsitektur E29: commit `47ddb8280592231fbb682f2e6a8b3c5735d185fb`.
Notebook code SHA256: `41590c848cf9c7467be1593fbd8ede873df455b9d2931853e52ca23668aa3753`.

Command: `PYTHONPATH=src:scripts /tmp/e29-verify/bin/python -m pytest tests/test_e29_ridge_notebook.py -q --tb=short`

Hasil: exit0; collected3, passed3, failed0, skipped0, xfailed0; 20,88detik.

- Struktur notebook valid, semua27codecell dapat diparse, tanpa output publik; builder deterministik, dua salinan identik. Arsitektur dan optimizer cocok source E29 yang dipin.
- Eksekusi seluruh cell menggunakan24baris fixture sintetis dan random XLM-R satu layer berdimensi768/1024, tokenizer teknis, CPU,1epoch per16fit. Parameter RoBERTa dan BiGRU/attention berubah; LR MLP5e-6 dan pooling/norm1e-4. MLP checkpoint cocok prediksi kontrol. Lima Excel/ZIP dan model final terbentuk; resume tahap selesai tidak training ulang.
- Fixture representasi tiruan untuk audit label: perubahan target outer validation tidak mengubah prediksi outer/alpha terkait; perubahan target test tidak mengubah alpha atau model final. Audit cache rusak gagal dengan jelas; reload koefisien Ridge cocok prediksi; checksum ZIP diverifikasi.

Ini bukti teknis, bukan training pretrained pada skor guru, bukan hasil akurasi penelitian dan bukan eksekusi GPU Colab. Tidak ada klaim E29 Ridge mengungguli E29 historis.

Percobaan awal dihentikan saat koneksi kernel dibatasi sandbox. Percobaan berikutnya:2passed/1failed pada guard versi lingkungan tes. Versi numerik kemudian disamakan dengan setup notebook; rerun final seluruh3test lulus. Guard dan source notebook tidak dilonggarkan.

## Hash berkas sumber yang diverifikasi

- `scripts/build_e29_ridge_notebook.py`: `fce9f7ee1efa221c729b39fe4d97a621edbd21647362419b784d799c87851319`
- `tests/test_e29_ridge_notebook.py`: `bb69250942e1db7750e6205ff08c82c1d1cf22c85aff442ea5d36c8fdf9305a8`
- `notebooks/experiments/AES_300_E29_BiGRU_Attention_Ridge_Colab.ipynb`: `e387a52669cae6bc639f297b720a70f758f05f91db69c3577b917ae27e9bca02`
- `aes-learning/notebooks/experiments/AES_300_E29_BiGRU_Attention_Ridge_Colab.ipynb`: `e387a52669cae6bc639f297b720a70f758f05f91db69c3577b917ae27e9bca02`

Versi: torch 2.6.0+cpu, transformers 4.44.2, numpy 2.2.6, pandas 2.2.3, scipy 1.15.3, scikit-learn 1.6.1, openpyxl 3.1.5, nbformat 5.11.1, nbclient 0.11.0, pytest 9.2.0.
