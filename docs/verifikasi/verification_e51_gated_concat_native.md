# Verifikasi E51 · 6 November 2026

E51 diturunkan dari E50 pada commit `c36003670ad9476951b8b607a6018e5a24ceb7e2`, juga HEAD awal pekerjaan. Cwd: `/workspace/Automated_Essay_Scoring`. Perubahan metode hanya gate per jawaban sebelum concat; ini bukan hasil penelitian baru.

Notebook memiliki 190 cell, termasuk 92 cell kode, tanpa output. Notebook utama dan mirror byte-identik. SHA256 notebook akhir: `44e556b213d6d6795de0830cae397511a84e2cb03a833266846f0cc001eec708`. SHA256 kode terlihat: `edf929becaae3e83b4e939de9351c69a6d4560d8d97755d8e9408f28e0229a02`.

```bash
/tmp/e39-verify/bin/python -m pytest tests/test_e51_notebook.py \
  -q -s --tb=short --basetemp=outputs/e51_study/fixtures
```

Exit 0: **6 passed, 0 failed, 0 skipped, 0 xfailed**, 92,51 detik. Ada satu deprecation warning dari plugin lingkungan. Setelah perapian teks Markdown, pemeriksaan source/build dijalankan kembali: **1 passed**, 3,15 detik; kode training dan hash kode tidak berubah. Manifest dan log berada di `outputs/e51_study` dan tidak dipublikasikan.

Enam kelompok pengujian:

1. **Source dan protokol.** Schema notebook, hash, cell native, output kosong, mirror dan rebuild deterministik. AST ScalarMix, BiGRU, co-attention multihead, metrik, split, encoder, BGE dan windowing tetap E50. Loop epoch training tetap sama. Lima CV baru dan satu refit/test; tidak ada LDS/ListNet, cosine, magic, worker, ZIP acuan, loop banyak seed atau scheduler refit E46/E47. Nama ZIP tepat dan source E50 tidak ditimpa.
2. **Persamaan gate.** Tambahan tepat 2.050 parameter. Kondisi awal menghasilkan concat identik pada FP32, FP16 dan BF16. Gradien mencapai bobot/bias gate serta kedua cabang. Optimizer step mengubah gate; contoh sintetis menunjukkan alpha dapat berbeda antarjawaban. Alpha berjumlah 1 dan cache inspeksi terlepas dari graph, sedangkan keluaran alpha untuk training tetap terhubung. Bentuk fitur salah ditolak.
3. **Model dan pembelajaran.** Parameter bersama serta RNG bit-identik E50 pada seed sama; prediksi awal persis sama. Split identik. Gate masuk kelompok head dengan LR 1e-4; gradien mencapai gate, proyeksi BGE dan multihead. Pengambilan alpha dalam `Engine.predict` mempertahankan urutan indeks pada microbatch 1/2 dan berasal dari forward prediksi yang sama.
4. **Pemetaan ID.** Indeks dataset dipetakan ke ID aktual dengan celah nomor baris Excel. Ketidakcocokan indeks, urutan ID atau jumlah alpha ditolak; alpha tidak berjumlah 1 juga ditolak.
5. **Resume.** Fit dua epoch tanpa interupsi dibandingkan dengan interupsi setelah penyimpanan epoch pertama lalu resume. History, parameter model, metadata gate, prediksi dan alpha bit-identik.
6. **Notebook lengkap dan hasil ZIP.** Eksekusi notebook mandiri dua kali dalam kernel Jupyter baru, tanpa impor source repo pada notebook. Fixture berisi 24 baris sintetis, Transformer acak empat layer, dimensi RoBERTa 768/BGE 1.024, token pendek dan CPU. Untuk waktu pengujian, CV dibatasi satu epoch. Lima CV, satu refit dan test menghasilkan lima Excel/ZIP, 18 OOF dan 6 test, tujuh ringkasan gate, enam konfigurasi checkpoint, serta 24 baris proyeksi multihead. Run kedua memulihkan hasil tanpa mengubah checkpoint final. Alpha CSV test cocok dengan Excel; cache CV sesuai indeks validasi; vektor sesudah gate cocok dengan pengali 2×alpha. Mengubah alpha, pengali, ID OOF, statistik ringkasan atau mengosongkan ringkasan test/CV memblokir unduhan.

Source publik tetap memakai 300 data, maksimal 25 epoch dan patience 5. Pengujian fixture tidak mengganti protokol publik. Metrik fixture bukan akurasi penelitian. Gate Final Semua Jawaban adalah inspeksi checkpoint refit, bukan OOF atau evaluasi 300 data. Alpha dan norma parameter tidak diklaim sebagai kontribusi kausal terhadap skor.

Lingkungan: Python 3.12.14, torch 2.6.0+cpu, transformers 4.44.2, pytest 9.2.0, nbformat 5.11.1, nbclient 0.11.0, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0, openpyxl 3.1.5. Jupyter memakai socket lokal. Tidak ada smoke pretrained GPU atau training penelitian E51 pada dataset asli. Pengujian teknis tidak menjamin bebas seluruh bug, kompatibilitas setiap runtime GPU, peningkatan RMSE, atau target 60/60.

Hanya source, notebook kosong, builder/helper, tests dan dokumentasi aman dipublikasikan. Dataset, hasil, executed notebook, Excel, checkpoint dan log tetap privat. Verifikasi SHA branch remote serta byte notebook/mirror/helper/builder/dokumentasi dicatat setelah push di `outputs/e51_study/publication.json`.

[Metode dan referensi paper](../eksperimen/eksperimen_e51_gated_concat_native_gpu.md).
