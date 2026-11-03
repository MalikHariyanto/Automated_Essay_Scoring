# Verifikasi E40 native — 3 November 2026

E40 menguji layer-wise learning rate decay0,95 pada E30 MLP. Arsitektur kembali BiGRU64perarah dan dropoutMLP0,1. Tidak ada hasil training penelitian E40, klaim akurasi baru, atau pengujian checkpoint pretrained asli pada GPU dalam verifikasi ini.

## Identitas source

- Cwd `/workspace/Automated_Essay_Scoring`, baseline kerja/format `5f1c993804560d12c06d7d11d4540e6d11307cf2`.
- Builder `scripts/build_e40_notebook.py` dipin ke notebook E39 untuk format native; mengembalikan hidden64dan mengubah optimizer menjadi LLRD. Source E30/E36–E39 dan HTML hasil tidak ditimpa.
- Notebook utama/mirror byte-identik,184cell/89codecell,semua output kosong. Raw SHA256 `5ac16575c729deb49b75869a07852e8912f57bb00e699b80cfc421c9597eba9c`. Visible-code SHA256 `497ea5856776137cf2b0b8fcb62744ab445aa82044e8506840fb7a626722bcde`.
- S3 saja,5fold sekali,1refit,1test; tidak ada sweep seed/kontrol ulang. X/y,split,checkpoint encoder,tokenisasi,mask,loss,optimizer AdamW/decay,warmup/linear scheduler dan pilihan epoch mengikuti E30. Perubahan numerik yang diuji adalah LR per kedalaman encoder. Tambahan pada fit hanya tabel dan metadata LR; loop epoch sama secara AST dengan E30.

## Tes E40

```bash
PYTHONPATH=src:scripts /tmp/e39-verify/bin/python -m pytest \
 tests/test_e40_notebook.py -q -s --tb=short \
 --basetemp=outputs/e40_study/verified_fixtures
```

Exit0, **4passed,0failed,0skipped,0xfailed**,85,26detik. Satu warning deprecation plugin pytest lingkungan. Kernel Jupyter memerlukan socket localhost; percobaan sandbox tidak dapat membukanya dan dihentikan, kemudian perintah berhasil dijalankan dengan izin kernel lokal. Ini bukan kegagalan numerik atau hasil penelitian GPU.

1. **Struktur/protokol/builder:** validasi schema,fingerprint,mirror,rebuild deterministik dan source historis utuh. Kode native tanpa exec/eval/magic/worker tersembunyi atau pemeriksaan cosine. Input tiga teks,seed tunggal,5CV+1refit+1test benar. Definisi split,train_all,metrik,encoder,BGE,mask,windowing serta loop epoch identik dengan E30 secara AST.
2. **Pembagian dua belas layer:** container teknis kecil dengan12blok berbeda memverifikasi nama kelompok,LR setiap parameter,embeddings0,95^12dan layer11=5e-6. Seluruh parameter trainable tercakup tepat sekali. Scheduler warmup+linear mempertahankan rasio saat LR aktif; tiap parameter mendapat update. Faktor tidak sah dan parameter tak dikenal ditolak. Ini pemeriksaan optimizer dengan tensor teknis, bukan forward checkpoint12layer pretrained.
3. **Kesetaraan E30 dan update yang berubah:** random tiny RoBERTa2layer/hidden768ditambah model fusion asli,tokenizer teknis dan24baris sintetis. State awal serta prediksi identik E30. Pada decay1,seluruh state setelah satu langkah AdamW identik E30. Pada decay0,95,parameter di luar encoder dan layer atas tetap identik; update embeddings berbanding0,95^2dengan kontrol. Gradien RoBERTa/BiGRU nonzero; load checkpoint dengan faktor berbeda ditolak. S1/S2 ditolak.
4. **Notebook lengkap dan pemulihan:**89codecelldijalankan di kernel terisolasi memakai random Transformer2layer,CPU,1epoch,token pendek dan24baris sintetis. Source publik tetap GPU,25epoch dan300baris. Lima fold,satu refit,satu test dan lima workbook/ZIP terhasilkan. Kernel baru memakai cache yang sesuai tanpa perubahan checkpoint akhir. Excel05mencatat faktor0,95dan36baris LR aktual:6kelompok×(5fold+1refit). Ringkasan co-attention kembali128komponen. X/y aktual yang ditampilkan sama dengan masukan training. Tidak ada import proyek atau worker tersembunyi.

Regresi untuk default helper notebook sebelumnya:

```bash
PYTHONPATH=src:scripts /tmp/e39-verify/bin/python -m pytest \
 tests/test_e39_notebook.py -k full_e39 -q -s --tb=short \
 --basetemp=outputs/e40_study/e39_regression
```

Exit0, **1passed,2deselected,0failed,0skipped,0xfailed**,84,37detik. Notebook E39 fixture dijalankan sampai lima Excel dan pemulihan kernel baru; default kelompok LR encoder tunggal tetap kompatibel. Satu warning deprecation plugin lingkungan. Log privat `outputs/e40_study/e39_regression.log`. Ini satu kasus regresi tambahan, bukan hasil penelitian E39/E40.

## Lingkungan dan batas bukti

Venv `/tmp/e39-verify`,Python3.12.14,torch2.6.0+cpu,transformers4.44.2,pytest9.2.0,nbformat5.11.1,nbclient0.11.0. Fixture menggunakan model Transformer berbobot acak; tidak mengukur kualitas AES atau kompatibilitas runtime GPU Colab terbaru. Durasi tes bukan perkiraan lama training penelitian.

Helper `execute_full_native_fixture` pada tes E36 diperluas dengan argumen opsional jumlah layer dan kelompok LR encoder; default notebook lama dipertahankan. Bukti fixture,manifest source dan log regresi tetap lokal di `outputs/e40_study`, tidak dipublikasikan. Hanya notebook kosong,builder,tes dan dokumentasi aman dipublikasikan; dataset,unggahan siswa,Excel,executed notebook,cache,bobot dan kunci tetap lokal. SHA branch dan byte notebook utama/mirror/metode diverifikasi setelah push; publikasi tidak membuktikan training GPU.
