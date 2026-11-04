# Verifikasi HTML panduan E30 — 4 November 2026

Pengguna secara eksplisit meminta HTML baru yang sederhana untuk menjelaskan kode E30 MLP acuan lengkap tanpa cell magic. Artefak `downloads/Panduan_E30_MLP.html` dan mirror `docs/panduan_e30_mlp.html` berisi penjelasan,source notebook publik dan UI lokal. Buku hasil terenkripsi sebelumnya tidak diganti. Tidak ada training,model inference,hasil penelitian baru,teks siswa atau nilai individual yang dimasukkan ke panduan.

## Ikatan dengan source

Source notebook dipin kecommit `16670531f76bf27d07fe87c3c73614ae76d7c451`,path `notebooks/experiments/AES_300_E30_Lengkap_S1_S2_S3_Native_Learnable_GPU.ipynb`,SHA256 `3d4f035d5e89989c5c7bd506e10bc06cc7f38d2726137568e204ca6163bfce4a`. Builder menolak source yang berbeda sebelum penjelasan diperiksa ulang. Seluruh89cellkode disertakan dengan nomor cell fisik,penjelasan tugas/mekanisme dan source lengkap. Cuplikan utama menunjuk nomor baris asli. Nomor cell menghitung semua kotak notebook,termasuk Markdown.

Nama checkpoint,kolom X/y,LR,hidden,BiGRU/co-attention/pooling,concat dan head mengikuti source. Penjelasan menegaskan: tiga kolom teks tetap tiga fitur,disusun menjadi satu urutan token perwindow;label tidak masuk encoder tetapi memengaruhi pembelajaran melalui loss/gradien,validasi memilih epoch,dan evaluasi. Pemilihan split E30 tidak memakai label. Dimensi768+256bukan bobot pengaruh75:25. S1linear,S2MLP,S3fusionMLP dan model RoBERTa S1/S3terpisah dijelaskan. Tidak ada klaim ketepatan model dari halaman ini.

## Perintah dan pemeriksaan

Cwd `/workspace/Automated_Essay_Scoring`,baseline kerja `16670531f76bf27d07fe87c3c73614ae76d7c451`.

```bash
python scripts/build_e30_simple_guide.py
node --check scripts/templates/e30_simple_guide.js
node scripts/check_e30_simple_guide.mjs
```

Perintah akhir exit0. **6kelompok pemeriksaan lolos,0gagal** di Chromium154.0.8037.57,dengan Node24.19.0. Screenshot desktop,mobile dan bagian fusion diperiksa visual. Bukti browser disimpan privat di `outputs/e30_simple_guide/browser_verification.json`.

1. Rebuild deterministik,duaHTMLbyte-identik,source notebook tidak berubah. Notebook yang tertanam sama persis dengan source publik. Semua89cell,seluruh cuplikan serta target tautan diperiksa terhadap nomor/baris sumber. Judul dan penjelasan per cell wajib tersedia. Tidak ada output notebook yang disertakan.
2. Pilihan S1/S2/S3serta mode label saat belajar/prediksi bekerja dengan klik dan tombol panah/Home/End. Hanya panel yang dipilih tampil. Penjelasan tidak menghasilkan skor simulasi.
3. Pencarian berdasarkan source/variabel/fungsi bekerja,termasuk hasil kosong dan Escapeuntuk reset. Tautan cell membuka katalog dan cell yang benar serta mereset filter yang dapat menyembunyikannya.
4. Teks yang disalin sama dengan cuplikan source. Unduhan notebook dari HTML byte-identik dengan source asli,termasuk SHA256. HTML yang disimpan dari tombol unduh dapat dibaca ulang dan semua interaksi lokal tetap bekerja;toast lama disembunyikan saat dibuka kembali.
5. Tidak ada overflow horizontal halaman pada320/390/768/1440px di ketujuh bagian utama. Blok kode memiliki scroll lokal;navHPdapat digeser. Header/isi/code mobile diperiksa melalui screenshot.
6. Tidak ada script/stylesheet eksternal,CDN,iframe atau errorJavaScript. Seluruh permintaan eksternal diblokir dalam pengujian dan jumlahnya0. Hanya dokumen HTML awal dan salinan unduhan disajikan oleh serverlocalhostterbatas. Kebijakan Chromium lingkungan memblokir file://,sehingga protokolfile://tidak diuji langsung;halaman tidak mempunyai ketergantungan jaringan setelah dokumen tersedia.

HTML sekitar0,98MB,dengan font sistem,CSS/JSinline,highlight kode statis dan payload notebook publik sebagai JSON inert. Tombol salin memakai aksi pengguna;tidak membaca clipboard pengguna. Tidak ada fetch,analytics,unggahan data,persistensi localStorage,login atau kunci. Penjelasan semua kode dapat dibuka lewat details;kode inti tersedia di bagian utama. Akses keyboard,focusindikator dan reduced-motion disediakan.

Source/model notebook,hasiI penelitian lama dan HTML buku terenkripsi tidak diubah. Hanya builder,catatan source,template,pemeriksa browser,HTMLsource-only dan dokumentasi aman dipublikasikan. Setelah push,SHAbranch dan isi HTML/mirror/builder diverifikasi;hasilnya tersimpan privat di `outputs/e30_simple_guide/remote_verification.json`.
