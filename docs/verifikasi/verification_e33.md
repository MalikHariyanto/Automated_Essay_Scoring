# Verifikasi E33 · LR MLP S3 5e-5

Tanggal1November2026;cwd `/workspace/Automated_Essay_Scoring`.
Source E30 asli dipin ke commit `6b3c52569851b06720cb80a0e9534e4f46b70eaf`.

Command: `PYTHONPATH=src:scripts /tmp/e29-verify/bin/python -m pytest tests/test_e33_notebook.py -q --tb=short`

Exit0;collected5;passed5;failed0;skipped0;xfailed0;57,07detik.

- Format47cell/30codecell, GPU guard,5fold/maks25epoch/patience5, mirror, sintaks, hash dan rebuild deterministik diperiksa. Setelah normalisasi nama folder/hash, hanya codecell10dan12berbeda dari E30: defaultLR, konfigurasi pengguna, identitas/log. Seluruh arsitektur/optimizer/training/tokenisasi/ekspor identik acuan.
- Parameter trainable masuk optimizer tepat sekali. Pada S1/S2, forward dan pembaruan optimizer pertama identik kontrol E30. Pada S3, forward sebelum update identik; hanya head.* berubah dibanding kontrol, besar update head berbanding2:1sesuaiLR1e-4vs5e-5. Gradien BiGRU/co-attention tetap mengalir.
- Seluruh notebook dieksekusi pada24baris fixture sintetis dan Transformer acak satu layer768/1024, tokenizer teknis, CPU,1epoch:15CVfit+3refit,5Excel,ZIP,checkpoint dan laporan terbentuk. Notebook diulang dalam kernel baru dengan folder sama; checkpoint final tidak berubah karena resume. Config, Excel dan checkpoint mengonfirmasi LR MLP S3 5e-5, MLP S2/linear S1 1e-4, pooling/proyeksi1e-4.

Ini bukti teknis. Tidak ada training pretrained dengan data guru, hasil akurasi penelitian E33 atau eksekusi CUDA Colab dalam pemeriksaan ini. Source E30 asli tidak diubah. Pemilihan epoch/early stopping mempertahankan fungsi E30 secara identik.

Code SHA256: `c46df14c3afdd3c767ac99eece7496118c1bfae1fd5744f6a89805ad33ad991a`.

- `scripts/build_e33_notebook.py`: `7433e014a9ff2c452baa814cd1e6c6c6c2fc81be3d2e1d037198d5a9f20f5339`
- `tests/test_e33_notebook.py`: `93373f9ad00d7cfba3bbfcbd48869838ac8b3d0faa2b16573066bf0d3062a86d`
- `notebooks/experiments/AES_300_E33_E30_BiGRU_CoAttention_MLPLR5e5_GPU.ipynb`: `43c97865854c07f79b1f2a2576cb16b2756e6e984dcb907b7bf84d027f9bbdf8`
- `aes-learning/notebooks/experiments/AES_300_E33_E30_BiGRU_CoAttention_MLPLR5e5_GPU.ipynb`: `43c97865854c07f79b1f2a2576cb16b2756e6e984dcb907b7bf84d027f9bbdf8`

Versi: torch 2.6.0+cpu, transformers 4.44.2, numpy 2.2.6, pandas 2.2.3, scipy 1.15.3, scikit-learn 1.6.1, openpyxl 3.1.5, nbformat 5.11.1, nbclient 0.11.0, pytest 9.2.0.
