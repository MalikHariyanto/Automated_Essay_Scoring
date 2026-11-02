# E36 · bukti teknis notebook S3-only native

Source induk saat pengerjaan: `f9638969d29f5f961d84dc5111e96ffd4f03b6af`. Metode E30 dan format E35 dipin di [builder](../../scripts/build_e36_notebook.py); source historis dan hasil penelitian tidak diubah. [Notebook](../../notebooks/experiments/AES_300_E36_E30_S3_EncoderLR3e6_Native_Learnable_GPU.ipynb) · [metode](../eksperimen/eksperimen_e36_s3_native_gpu.md).

## Perintah dan hasil

CWD `/workspace/Automated_Essay_Scoring`:

```sh
/tmp/e36-verify/bin/python -m pytest tests/test_e36_notebook.py -k 'not full_native_notebook' -q --basetemp=outputs/e36_core_verify -rA
# Exit 0: 3 passed, 1 deselected (tes notebook penuh dijalankan terpisah).

/tmp/e36-verify/bin/python -m pytest tests/test_e36_notebook.py::test_full_native_notebook_and_fresh_kernel_resume -q --basetemp=outputs/e36_kernel_verify -rA
# Exit 0: 1 passed.
```

Empat tes unik lulus. Percobaan awal sempat gagal karena folder induk basetemp belum ada; setelah diperbaiki, kernel Jupyter memerlukan socket localhost di luar pembatasan sandbox shell. Eksekusi kernel diulang dengan akses localhost yang diizinkan. Ini masalah infrastruktur tes, bukan hasil training penelitian.

## Bukti yang dicakup

- Validasi nbformat, 184cell/89cell Python, 28section bernomor, tanpa output tersimpan, magic, exec/eval, worker training atau pemeriksaan cosine. Builder deterministik dan mirror identik; notebook E30/E35 tetap byte-identik.
- Nilai default E30 cocok kecuali encoder/s3_encoder_lr5e-6→3e-6. State awal dan prediksi S3 pada fixture identik; update pertama hanya berbeda pada encoder, dengan norma langkah sekitar0,6×kontrol. Gradien mencapai encoder, GRU dan proyeksi referensi. Seluruh parameter trainable masuk optimizer tepat sekali.
- AST rumus BiGRU/co-attention, Engine, optimizer, fit, CV/refit/test, metrik, split dan BGE frozen sama dengan E30. Pembatasan skenario dan ekspor adalah perubahan orkestrasi.
- S1/S2 ditolak oleh make_model; tidak ada folder/checkpoint S1/S2 dan report hanya S3.
- Notebook benar-benar dijalankan dari atas di kernel baru, memakai24baris sintetis berlabel fixture dan Transformer acak satu layer; dimensi768/1024 dipertahankan. Hanya untuk tes, epoch dibatasi1 dan token/window diperkecil; notebook publik tetap25epoch/GPU.5CVfit dan1refit selesai, lima Excel dan ZIP dihasilkan.
- Kernel baru kedua menjalankan ulang semua cell, memverifikasi resume tanpa perubahan hash checkpoint. X/y yang terlihat sama dengan input/target training; target tidak masuk X. Workbook04 hanya mempunyai prediksi/pembulatan/selisih S3, seluruh prediksi terisi. Workbook05 mencatat LR encoder3e-6 dan LR pooling/norm1e-4.
- Guard GPU menolak training Colab ketika CUDA tidak tersedia.

Lingkungan teknis: Python3.12.14, Torch2.6.0+cpu, Transformers4.44.2, NumPy2.3.5, pandas2.2.3, scikit-learn1.8.0, SciPy1.17.0, openpyxl3.1.5, nbclient0.11.0, pytest9.2.0. Log dan bukti lokal berada di `outputs/e36_study/`; executed notebook dan hasil fixture tidak dipublikasikan.

**Belum menjalankan checkpoint pretrained asli pada GPU/300jawaban guru untuk E36.** Tidak ada klaim akurasi, RMSE baru, atau jaminan mengalahkan E30. Pengujian teknis bukan hasil penelitian. Keberadaan remote harus diverifikasi setelah push; tes lokal sendiri bukan bukti publikasi.
