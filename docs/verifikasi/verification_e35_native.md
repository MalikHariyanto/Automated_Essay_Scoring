# Verifikasi E35 native Colab — 2 November 2026

Cwd `/workspace/Automated_Essay_Scoring`. Source E30 dipin kecommit7f3363643e6bf402a293de069bcfc23d85b19c89.

Command: `PYTHONPATH=src:scripts /tmp/e29-verify/bin/python -m pytest tests/test_e35_notebook.py -q --tb=short --basetemp outputs/e35_verification/final_run`

Exit0;8passed,0failed,0skipped,0xfailed;56,32detik. Ini fixture teknis sintetis, bukan hasil akurasi pretrained atau GPU Colab.

- Notebook valid;184cell/89codecell,28bagian bernomor plus rujukan;mirror byte-identik,rebuild deterministik,output/count kosong. Codecell memakai Python biasa,tanpa magic,worker,exec/eval sumber dinamis atau import projectAES.
- AST model/BiGRU/co-attention/tokenisasi/window/optimizer/fit/CV/metrics/split/save/reload sama dengan E30. Input kolom terlihat dan diterapkan pada pembacaan Excel;rename header diuji menghasilkan teks/label/kelompok yang sama. Hanya LR kelompok sequence_pool/roberta_norm S3 yang memengaruhi pembaruan training E35 berbeda dari E30.
- Forward S1/S2/S3 dari bobot awal sama identik E30. Satu updateAdamW mempertahankan semua parameter di luar kelompok layerS3 identik;S1/S2 seluruhnya identik. Perubahan proyeksi referensi berbanding2:1sesuai LR;gradien BiGRU/co-attention mengalir. Seluruh parameter trainable tercakup optimizer tepat sekali. MLP1e-4,proyeksiBGE1e-4,encoder5e-6,pooling_norm5e-5.
- Notebook native dijalankan lengkap dua kali dalam kernel baru:24baris sintetis,Transformer acak1layer768/1024,tokenizer fixture,CPU,1epoch,5fold.15CVfit+3refit menghasilkan5Excel/ZIP. Resume mempertahankan SHA checkpoint final. X/y berada pada globals kernel yang sama dengan fit/make_model dan cocok data aktual yang diteruskan ke encoder/target. Tidak ada filePythonpendamping.
- Config,Excel dan checkpoint merekam LR yang tepat. Kata Berulang memiliki8kolom tanpa cosine. GuardGPU menolak CPU pada resep publik;setup menolak pustaka numerik tidak kompatibel. Pada fixture setup asli berjalan tanpa pemasangan karena seluruh dependensi sudah cocok. Ini tidak menguji pemasangan pada semua versi Colab atau eksekusi CUDA.
- Setup native mempertahankan Torch/CUDA/pustaka numerik yang cocok;versi aktual dicatat dalam hasil. E35 tetap RoBERTa fine-tuned,BGE frozen,MLP,5fold,maks25epoch,patience5. Belum ada hasil penelitian atau jaminan akurasi membaik.
- Source/hasil E30/E33/E34 dan paket HTML tidak diubah. Publikasi hanya notebook,builder,tes fixture dan dokumentasi;commit paket hasil siswa yang ditolak tidak masuk ancestry/perubahan publikasi E35.

Python 3.12.14. Versions: torch 2.6.0+cpu, transformers 4.44.2, numpy 2.2.6, pandas 2.2.3, scipy 1.15.3, scikit-learn 1.6.1, openpyxl 3.1.5, nbformat 5.11.1, nbclient 0.11.0, pytest 9.2.0.

Code SHA256: `718902342bf3aff97986939c14f241a16305736bd0236da2d10df577a0ec9047`.

- `scripts/build_e35_notebook.py`: `c1dfcc51304bdeeec436e04a39eb0b40e7980ff5fd1da42320d19c9ebd192452`
- `tests/test_e35_notebook.py`: `22c07bd70dcc3303319874cb5e91015936e1ee4ed93d732c056e432094183f30`
- `notebooks/experiments/AES_300_E35_E30_LayersLR5e5_Native_Learnable_GPU.ipynb`: `c833b03435424892f91b6557a5fd9be1719e422bd7b950815112e43f96134fd0`
- `aes-learning/notebooks/experiments/AES_300_E35_E30_LayersLR5e5_Native_Learnable_GPU.ipynb`: `c833b03435424892f91b6557a5fd9be1719e422bd7b950815112e43f96134fd0`
- `docs/eksperimen/eksperimen_e35_native_gpu.md`: `aaf9c12919805a493c8fe3bf4c5e8460ebd22ab121c1089b5c351983411e0f65`
