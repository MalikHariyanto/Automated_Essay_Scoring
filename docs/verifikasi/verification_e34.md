# Verifikasi E34 · 2 November 2026

Cwd `/workspace/Automated_Essay_Scoring`. Source E33 dipin ke commit `fa45c47a4d972a20a36445ea58d1fca8e8adcded`.

Command: `PYTHONPATH=src:scripts /tmp/e29-verify/bin/python -m pytest tests/test_e34_notebook.py -q --tb=short`

Exit0;collected5;passed5;failed0;skipped0;xfailed0;57,17detik.

- Schema/sintaks/fingerprint/rebuild deterministik dan dua salinan notebook identik.47cell/30codecell;5fold,maks25epoch,patience5 dan guard GPU dipertahankan. Setelah normalisasi nama folder/hash, hanya codecell10dan12berbeda dari E33:defaultLR layer S3,konfigurasi pengguna,identitas/log. Model/optimizer/loss/CV/refit/tokenisasi/exporter identik.
- Cakupan optimizer tepat sekali untuk semua parameter trainable. S1/S2 memiliki forward dan update pertama identik E33. Pada S3, forward sebelum update identik; hanya sequence_pool.* dan roberta_norm.* berbeda setelah update. Update proyeksi referensi co-attention berbanding2:1sesuai LR1e-4vs5e-5. MLP S3 tetap5e-5;proyeksi BGE1e-4;gradien GRU/co-attention tetap mengalir.
- Notebook lengkap dijalankan dua kali dengan kernel baru dan folder lokal sama:fixture24baris sintetis,Transformer acak satu layer768/1024,tokenizer teknis,CPU,1epoch.15CVfit+3refit,5Excel,ZIP,laporan dan checkpoint terbentuk. Resume mempertahankan hash checkpoint final. Config,Excel dan checkpoint mengonfirmasi LR MLP S3 5e-5 dan pooling_norm S3 5e-5;S1/S2/proyeksi BGE1e-4.

Ini bukti teknis, bukan training pretrained pada skor guru, hasil akurasi penelitian E34 atau eksekusi CUDA Colab. Tidak ada hasil penelitian E34. Source E30/E33 tidak diubah.

Publikasi ini hanya source notebook,builder,tes fixture dan dokumentasi. Branch publikasi berasal dari source publik fa45c47;commit paket hasil privat E33 yang masih menunggu persetujuan tidak masuk ancestry/perubahan publikasi source E34.

Code SHA256: `9f75943b57c77dbb734340ab806035395c08690029e4132920f4e3050fff82e2`.

- `scripts/build_e34_notebook.py`: `a2e4d4d0c939940123fe1469670ed1085998227417b474a8eebc9afe87c508c7`
- `tests/test_e34_notebook.py`: `e0e9872f7cbfbdd0ecf4e0a04070fe01f4f342a6483b24a6812e0297d2b06d55`
- `notebooks/experiments/AES_300_E34_E33_BiGRU_CoAttention_LayersLR5e5_GPU.ipynb`: `980acea9aa107262dad29f2615c558c693f94aaad175697621610ce89922b648`
- `aes-learning/notebooks/experiments/AES_300_E34_E33_BiGRU_CoAttention_LayersLR5e5_GPU.ipynb`: `980acea9aa107262dad29f2615c558c693f94aaad175697621610ce89922b648`

Versi: torch 2.6.0+cpu, transformers 4.44.2, numpy 2.2.6, pandas 2.2.3, scipy 1.15.3, scikit-learn 1.6.1, openpyxl 3.1.5, nbformat 5.11.1, nbclient 0.11.0, pytest 9.2.0.
