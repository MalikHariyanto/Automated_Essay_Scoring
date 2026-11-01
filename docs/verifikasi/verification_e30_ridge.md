# Verifikasi E30 + Ridge · 1 November 2026

Cwd: `/workspace/Automated_Essay_Scoring`.
Source E30 asli: commit `47ddb8280592231fbb682f2e6a8b3c5735d185fb`.

Command: `PYTHONPATH=src:scripts /tmp/e29-verify/bin/python -m pytest tests/test_e30_ridge_notebook.py -q --tb=short`

Exit0; collected4, passed4, failed0, skipped0, xfailed0;12,77detik.

1. Notebook schema/sintaks/hash/mirror/builder deterministik; arsitektur/optimizer identik source E30 yang dipin. Versi publik tanpa output.
2. Seluruh27codecell dieksekusi pada24baris sintetis dan XLM-R acak satu layer768/1024, tokenizer teknis, CPU,1epoch. Lima CV + satu final berjalan. RoBERTa/BiGRU-coattention belajar, LR MLP1e-4; label validation memilih epoch; lima Excel+ZIP, model Ridge dan checkpoint MLP final dihasilkan. Resume tahap selesai tidak training ulang. Prediksi MLP dari checkpoint cocok fitur final.
3. Plateau RMSE buatan: batas8epoch, patience5 menghasilkan berhenti pada epoch6, best epoch1; state keluaran cocok snapshot best_state. Resume checkpoint memulihkan prediksi dan riwayat persis.
4. Fixture representasi tiruan: best epoch1/4/5/2/7 menghasilkan refit4epoch. Hanya enam fit, alpha tetap sebelum CV. Test labels ditutupi NaN di semua fit; mengubah test labels tidak mengubah model Ridge/checkpoint final. Cache rusak ditolak.

Tes bukan training pretrained dengan data guru, bukan bukti akurasi penelitian dan bukan eksekusi CUDA/Colab. Belum ada angka penelitian E30 Ridge. Source/notebook E29 lama tidak berubah; E30 dipilih sebagai versi berikutnya.

Code SHA256: `08e4e3b4c7fd3e32bb99fe5d681d6717b5ecc7c0e3cfa2355dd7797673afdaa6`.

- `scripts/build_e30_ridge_notebook.py`: `c4f90eaf323f53c4709a03a6109bc05a4455b81d9b3be6549d8efa0bca03a5b3`
- `scripts/e30_ridge_experiment.py`: `33c7b26f824c0b5d3805f5dae45d28d82d57087085a794e98738261c58d7cdd6`
- `tests/test_e30_ridge_notebook.py`: `ce124db1c658b69e2fd1de8f9163e0b0e426622ba4ab16c18dcf279b478c43d1`
- `notebooks/experiments/AES_300_E30_BiGRU_CoAttention_Ridge_Colab.ipynb`: `3838802a991fa60a2f8c6dfe94af4c2c489f9e614adffe9ef7492d3e3dd73ca0`

Versi: torch 2.6.0+cpu, transformers 4.44.2, numpy 2.2.6, pandas 2.2.3, scipy 1.15.3, scikit-learn 1.6.1, openpyxl 3.1.5, nbformat 5.11.1, nbclient 0.11.0, pytest 9.2.0.
