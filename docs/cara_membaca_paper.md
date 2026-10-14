# Benang Merah — meja belajar paper

[Buka file HTML](paper_library_interaktif.html), lalu pilih **Download raw file** di GitHub dan buka file yang tersimpan di Chrome, Edge, atau Firefox. Semua isi, ilustrasi, dan rumus tersedia dalam satu file; tidak membutuhkan instalasi atau koneksi internet setelah diunduh. GitHub menampilkan source HTML, sehingga tampilan bisa diperiksa langsung melalui [preview beranda](previews/paper_library/01_beranda_desktop.png), [proses BGE](previews/paper_library/02_proses_bge.png), [rumus](previews/paper_library/03_rumus_bge.png), [tampilan HP](previews/paper_library/04_beranda_mobile.png), [cara membaca matriks hasil](previews/paper_library/05_membaca_matriks.png), dan [katalog full colour](previews/paper_library/06_katalog_full_colour.png).

Tema terbaru memakai latar full colour bergaya komik dan splash cat: merah, cyan, biru, magenta, serta kuning. Ilustrasi kota, pola halftone, dan Spider-Man menjadi latar visual; panel penjelasan memakai permukaan jelas agar teks panjang tetap mudah dibaca. Tema berlaku pada beranda, filter, kartu sumber, ruang baca, sticky note, rumus, kamus, dan perbandingan. Ilustrasi SVG dan CSS ikut tersimpan dalam HTML offline.

Panduan memetakan **42 sumber unik** dari 21 PDF terpisah dan PDF gabungan 4.051 halaman. Duplikasi menjadi satu kartu. Koleksi berisi artikel, preprint, buku/cuplikan buku, tesis, laporan, dan dokumen kebijakan; jenis sumber serta cakupan pembacaan dijelaskan pada awal setiap kartu. Dokumen asli tetap lokal; repository memuat ringkasan belajar yang diparafrasakan dan angka agregat yang dirujuk ke halaman sumber.

Mulai dari jalur belajar, lalu pilih salah satu dari tujuh topik: fondasi AI dan alur riset, pendidikan dan asesmen, embedding dan pencarian, RAG, penilaian esai, evaluasi dan kesepakatan, serta sentimen. Subtopik mempersempit tugas yang sebenarnya dikerjakan. Dalam topik, kartu mengikuti urutan pengantar sebelum metode lanjutan. Sorotan BGE membantu menemukan penggunaan BGE; bagian **Siapa mengerjakan apa?** membedakan model yang dipakai dalam metode dari model yang hanya menjadi rujukan. Skor relevansi pencarian tidak otomatis menjadi skor kualitas esai.

Setiap sumber dibaca dalam box dengan scroll sendiri melalui tujuh bagian berurutan: **Identitas**, **Inti paper**, **Proses**, **Rumus**, **Istilah**, **Hasil & batas**, dan **Baca terarah**. Proses menguraikan input → tindakan → output dengan sticky note pada langkah penting. Klik istilah bergaris titik untuk membaca artinya tanpa meninggalkan bagian itu. Rumus menjelaskan simbol dan cara membaca hasilnya; rumus panjang dapat digeser di dalam box. Bagian hasil memisahkan klaim, bukti, batas, dan catatan ketelitian sumber. Contoh confusion matrix IndoBERT menyertakan arah baris/kolom dan penjelasan tiap jenis kesalahan.

Pilih **Bandingkan** pada dua atau tiga kartu untuk membaca perbedaan tugas, model, input, output, dan bukti hasil. Kamus istilah juga menampilkan konteks tiap sumber, karena istilah yang sama bisa punya peran berbeda. Penanda selesai, bagian yang telah dibuka, serta pilihan jeda animasi disimpan di browser lokal. Navigasi tab mendukung tombol panah, Home/End, serta Escape untuk menutup ruang baca. Pengaturan perangkat untuk mengurangi animasi dihormati.

Identitas mengikuti PDF yang tersedia: tahun asli dan tanggal versi dibedakan; metadata yang tidak tampak tetap ditandai belum teridentifikasi. **Indeks, Scopus/SINTA, dan kuartil belum diverifikasi dari pengindeks resmi.** DOI, nama jurnal, logo, atau angka metrik pada sampul tidak dijadikan bukti indeks. Situs resmi pengindeks/metadata tidak dapat diakses melalui jaringan sesi penyusunan ini. Tautan terkait dapat mengarah ke DOI, preprint, penerbit, atau repository kode; cakupannya dijelaskan pada catatan sumber.

Nomor halaman mengacu pada posisi halaman PDF terpisah atau halaman global PDF gabungan, sesuai label. Buku panjang dirangkum pada bab pilihan. Cuplikan *Deep Learning* hanya memuat identitas, daftar isi, dan bibliografi: panduan tidak mengarang proses atau rumus dari bab yang tidak tersedia. Panduan ini tidak menjalankan eksperimen model, menilai mahasiswa, atau mengubah pipeline AES yang sudah ada.

Untuk membangun ulang dari catatan publik:

```bash
python scripts/build_paper_library.py
```

Data berada pada `docs/paper_library_data.json`; tampilan berada pada `scripts/templates/paper_library.html` dengan tema `scripts/templates/paper_library_full_colour.css` dan ilustrasi asli pada `scripts/assets/paper_splash.svg` serta `scripts/assets/paper_city.svg`. Renderer rumus KaTeX 0.16.22 beserta font dan lisensinya disertakan lokal dalam `scripts/vendor`. Bukti pemeriksaan browser versi awal berada pada [paper_library.json](verification/paper_library.json); pemeriksaan tema terbaru berada pada [paper_library_full_colour.json](verification/paper_library_full_colour.json). Pemeriksaan UI berbeda dari evaluasi akurasi penelitian AES.
