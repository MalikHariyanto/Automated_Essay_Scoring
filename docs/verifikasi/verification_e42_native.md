# Verifikasi E42 gamma native — 4 November 2026

E42 menambahkan satu gamma global pada campuran empat layer E41. Verifikasi memakai data sintetis dan model Transformer kecil berbobot acak di CPU. Tidak ada training penelitian GPU,hasil AES E42,atau klaim gamma sudah meningkatkan RMSE.

## Identitas source

Cwd `/workspace/Automated_Essay_Scoring`,baseline kerja/format `204720501f124d3fa7eb7091eb543baeba8ec21f`. Builder `scripts/build_e42_notebook.py` dipin ke E41 pada commit tersebut. Notebook utama/mirror byte-identik,184cell/89codecell,output kosong. Raw SHA256 `c632af4fc1d2b56f0b1cd4302dfb45a17137cd0bc2bff34136f1ae7c81b86d19`;visible-code SHA256 `c5d6a093be3c5dc44e3289d6806cfdb0e3fe9adccda110b913791f16fe665195`.

Gamma mulai1,0dan memakai LR1e-4bersama empat logit. Gamma bebas,tidak ditambahkan ke skor akhir. Tidak ada normalisasi baru. Statistik norma hanya diambil dalam inspeksi fitur dengan no_grad,memakai attention_maskuntuk mengecualikan padding. Sisa resep E41,S3saja,5CV+1refit+1test,maks25epoch/patience5,encoder5e-6dan komponen lain1e-4tetap.

## Perintah dan hasil akhir

```bash
PYTHONPATH=src:scripts /tmp/e39-verify/bin/python -m pytest \
 tests/test_e42_notebook.py -q -s --tb=short \
 --basetemp=outputs/e42_study/final_fixtures
```

Exit0, **4passed,0failed,0skipped,0xfailed**,91,56detik. Satu warning deprecation plugin pytest lingkungan. Log privat `outputs/e42_study/final_tests.log`. Percobaan awal menemukan pemeriksaan string versi Engine dan penamaan ZIP yang ikut terganti;label pemeriksaan dan urutan penggantian nama diperbaiki sebelum pengujian akhir ini. Hasil akhir memakai hash di atas.

1. Schema,hash,mirror,rebuild deterministik dan source E41utuh. Tanpa magic,exec/eval,cosine,worker tersembunyi,LLRD atau loop multi-seed. Optimizer,fit,train_all,split,metrik,encoder,BGE,mask/windowing dan pooling identik secara AST dengan E41;hanya nama E41→E42pada pesan error Engine dinormalisasi dalam perbandingan. Config mencatat gamma awal dan protokol yang benar.
2. Gamma1menghasilkan campuran awal yang sama;modul memiliki5parameter skalar. Gradien gamma dan logit nonzero. Gamma2melipatgandakan keluaran dan norma sesuai hitungan. Padding bermagnitudo besar tidak memengaruhi rerata norma;config gamma nonfinite ditolak.
3. Model E42berbobot awal sama dengan E41,hanya satu parameter lebih banyak. Dengan gamma dibekukan pada1,prediksi serta seluruh state komponen E41setelah satu update AdamW bit-identik. Ketika dibuka kembali,gamma memperoleh gradien;kelompok scalar_mix mencakup4logit+1gamma pada1e-4. Fitur norma berbentuk[N,1]. Load checkpoint mempertahankan gamma dan menolak gamma awal berbeda.
4. Seluruh89codecelldijalankan dalam kernel terisolasi dengan24baris sintetis,random RoBERTa/XLM-R4layer berdimensi768/1024,CPU,1epoch dan token pendek. Source publik tetap pretrained/GPU/300baris/25epoch. Lima fold,satu refit,satu test,lima Excel dan ZIP bernama benar selesai. Kernel baru memakai cache sesuai hash tanpa perubahan checkpoint final. Excel05memuat24baris bobot dan gamma checkpoint terpilih;gamma final berubah dari1. Bobot Excel cocok dengan softmax logit checkpoint. Ketiga sheet norma pada02berbentuk24×2dan norma sesudah≈|gamma|×sebelum. X/y terlihat cocok masukan training;tidak ada import proyek atau worker .py tersembunyi.

## Lingkungan dan batas bukti

Venv `/tmp/e39-verify`,Python3.12.14,torch2.6.0+cpu,transformers4.44.2,pytest9.2.0,nbformat5.11.1,nbclient0.11.0,NumPy2.3.5,pandas2.2.3,SciPy1.17.0,scikit-learn1.8.0,openpyxl3.1.5. Kernel Jupyter memerlukan izin socketlocalhost. Durasi tes bukan estimasi training GPU. Fixture tidak membuktikan kualitas AES atau kompatibilitas setiap runtime Colab.

Hanya notebook kosong,builder,tes dan dokumentasi aman dipublikasikan. Dataset,unggahan siswa,executed notebook,Excel,log,cache,bobot dan kunci tetap lokal. Manifest source/perintah/hasil ada privat di `outputs/e42_study/verification.json`. Setelah push,SHAbranch dan byte notebook/mirror/metode diperiksa. Landasan gamma adalah ELMo/ScalarMix/implementasi Transformer,bukan bukti empiris langsung pada arsitektur AES ini.
