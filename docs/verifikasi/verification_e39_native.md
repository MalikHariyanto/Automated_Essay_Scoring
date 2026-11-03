# Verifikasi E39 native — 3 November 2026

E39 adalah uji kapasitas BiGRU128perarah pada konsep E30 MLP. Pengguna mengonfirmasi pilihan ini, bukan GRU satu arah. Tidak ada hasil training penelitian E39, klaim akurasi baru, atau eksekusi pretrained asli di GPU pada verifikasi ini.

## Identitas source

- Cwd `/workspace/Automated_Essay_Scoring`, baseline kerja `076a57ce1f6295951abd3f5b785ea4dc5b70095f`.
- Format dipin ke source E38 pada commit tersebut; builder `scripts/build_e39_notebook.py` mengembalikan dropout ke0,1 dan mempertahankan LR E30 encoder5e-6/layer lain1e-4.
- Notebook utama/mirror byte-identik,184cell/89codecell,semua output kosong. Visible-code SHA256 `020f2ccfac83685b351a49ce2c23b792e76cd66e477565932822ed4015e0797e`.
- Model,Engine dan optimizer hanya memakai jalur S3. BGE frozen,MLP akhir,fusion1024. Checkpoint guard menolak hidden/pooling/config yang berbeda. Tidak ada Ridge,magic/worker tersembunyi,pemeriksaan cosine atau loop multi-seed.

## Tes yang dijalankan

```bash
PYTHONPATH=src:scripts /tmp/e39-verify/bin/python -m pytest \
 tests/test_e39_notebook.py -q -s --tb=short \
 --basetemp=outputs/e39_study/fixtures
```

Exit0, **3passed,0failed,0skipped,0xfailed** dalam78,38detik pada fixture CPU. Satu warning deprecation plugin lingkungan. Log privat `outputs/e39_study/tests.log`. Durasi bukan estimasi waktu training penelitian GPU.

1. **Struktur/protokol:** schema,hash,builder deterministik,mirror dan source lama utuh.89cell adalah Python biasa tanpa exec/eval/magic. AST fit,train_all,split,metrik,BGE,mask dan windowing sama dengan E30. Seed per fold tetap seed dasar+nomor fold; maksimal25epoch/patience5,5fold+1refit/1test.
2. **Kesetaraan pada hidden64 dan kapasitas128:** model S3 yang disederhanakan dengan hidden64 menghasilkan state awal,prediksi eval,gradient/update AdamW dan state setelah satu langkah identik dengan E30 asli. Pada hidden128, parameter di luar sequence_pool tetap sama pada inisialisasi yang dikendalikan; tambahan820.224parameter seluruhnya berada pada sequence_pool. Reference projection256,matching1280→256,attention256→64→1,output256→768 serta MLP1024→256→1 diverifikasi. Tensor ringkasan [N,256],pooling [N,768],concat [N,1024] dan skor [N,1] finite; bobot pooling berjumlah1dan posisi di luar mask bernilai0. Gradient kedua arah GRU,proyeksi referensi,matching,pooling dan RoBERTa nonzero. LR semua kelompok diperiksa. Config pooling tidak sah dan pemuatan checkpoint hidden berbeda ditolak; S1/S2 ditolak oleh make_model.
3. **Notebook lengkap/resume:** semua section dijalankan dalam kernel terisolasi memakai24baris sintetis,tokenizer teknis dan random tiny RoBERTa/XLM-R satu layer dengan keluaran768/1024. Hanya salinan fixture memakai CPU,1epoch dan token pendek; source publik tetap GPU,300baris dan25epoch. Lima fold,satu refit,satu test,lima workbook dan ZIP akhir dihasilkan. Kolom X/y sama dengan masukan training aktual; tanpa import proyek atau worker .py. Kernel baru memakai hasil cocok hash; checkpoint final tidak berubah. Workbook02memuat24×257kolom ringkasan,24×769pooling dan24×1025fusion. Workbook05/config/checkpoint mencatat hidden128,attention64,dropout0,1 dan LR yang benar. Nama ZIP E39 diperiksa.

Sesudah memperjelas dua paragraf Markdown tentang dropout dan pembagi co-attention, struktur/builder diuji ulang:

```bash
PYTHONPATH=src:scripts /tmp/e39-verify/bin/python -m pytest \
 tests/test_e39_notebook.py -k native_s3_only -q --tb=short
```

Exit0,1passed,2deselected,0failed dalam3,63detik; satu warning deprecation. Ini pengulangan satu dari tiga kasus tadi, bukan test keempat. Kode numerik/fingerprint tidak berubah; hanya penjelasan Markdown diperbaiki. Tes tidak mengukur kualitas AES penelitian atau kompatibilitas GPU Colab terbaru.

## Lingkungan dan publikasi

Venv `/tmp/e39-verify`;Python3.12.14,torch2.6.0+cpu,transformers4.44.2,pytest9.2,nbformat5.11.1,nbclient0.11,NumPy2.3.5,pandas2.2.3,SciPy1.17,scikit-learn1.8,openpyxl3.1.5. Random Transformer/tokenizer merupakan fixture, bukan checkpoint pretrained yang diklaim akurat.

Source notebook lama dan HTML hasil tidak berubah. Hanya notebook tanpa output,builder,tes dan dokumentasi aman masuk commit; dataset,log unggahan,executed notebook,Excel,bobot,kunci dan paket siswa tetap lokal. Manifest SHA source/perintah disimpan privat di `outputs/e39_study/verification.json`; SHA branch dan byte notebook/mirror/metode diverifikasi setelah push di `remote_verification.json`.
