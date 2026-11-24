# Catatan perubahan draf skripsi Malik Hariyanto

Tanggal revisi: 24 November 2026. Status: **draf Bab I–III dengan kerangka Bab IV–V**. Konfigurasi akhir S3, hasil penelitian, pembahasan hasil, serta kesimpulan belum diisi.

Dokumen kerja: [SKRIPSI-MALIK-HARIYANTO-DRAF.docx](SKRIPSI-MALIK-HARIYANTO-DRAF.docx).

## Dasar penyuntingan

Penyuntingan menggunakan proposal `PROPOSAL-MALIK-HARIYANTO-TANPA-NRMSE.docx`, `PANDUAN SKRIPSI REAL.pdf`, workbook dataset yang dikirim, serta kode dan metadata eksperimen yang tersedia dalam proyek. Contoh skripsi pada `Screenshot 2026-11-24 210615.pdf` menjadi acuan susunan penyajian pengolahan data, implementasi, dan evaluasi. Metode serta hasil penelitian contoh tidak dipindahkan ke draf ini.

Judul pada dua halaman awal dipertahankan persis sesuai permintaan. Ejaan **UNIVERSITAS TRUNODJOYO MADURA** juga dipertahankan sesuai arahan pemilik naskah. Status sampul diubah dari “PROPOSAL SKRIPSI” menjadi “SKRIPSI”.

## Bagian yang diubah

| Bagian | Perubahan |
| --- | --- |
| Abstrak dan kata kunci | Menyelaraskan objek, dataset, tiga input, empat target, pembagian kelompok, dan status S3 yang masih dikembangkan. Abstrak belum memuat hasil yang belum tersedia. |
| Bab I: latar belakang dan batasan | Mengganti narasi tugas argumentasi ketahanan pangan, satu sekolah, dan rencana 300–400 sampel dengan karakteristik workbook yang tersedia. Unit analisis menjadi satu jawaban pada satu soal. |
| Bab I: istilah target penelitian | Menyesuaikan penyebutan skor menjadi **skor acuan** pada bagian khusus penelitian, karena identitas dan prosedur penilai belum tercantum dalam workbook. Arah pertanyaan penelitian serta tujuan perbandingan S1–S3 dipertahankan. |
| Bab I: struktur | Menambahkan **1.5 Sistematika Penulisan** yang menjelaskan susunan Bab I–V. |
| Bab II: objek penilaian | Menyesuaikan uraian teks dan rubrik dengan jawaban poster serta berita. Rentang kategori dinyatakan mengikuti data dan eksperimen; deskriptor rubrik yang belum tersedia tidak dibuat. |
| Bab II: representasi dan concat | Mengganti penjelasan dan Persamaan 2.3 tentang cosine dengan operasi konkatenasi representasi RoBERTa dan proyeksi BGE. Normalisasi L2 tetap dijelaskan sebagai pengolahan vektor BGE. |
| Bab II: penelitian terdahulu | Memisahkan uraian Attali–Burstein dan Taghipour–Ng serta Pack dan García-Varela menjadi paragraf masing-masing. Daftar pustaka dan sumber sitasi yang sudah ada dipertahankan. |
| Bab III: dataset dan skema | Menjelaskan 13 kolom asli, fungsi metadata, tiga teks input, empat label, serta kode baris/kelompok hasil pipeline. Menambahkan jumlah respons per sekolah, cakupan soal, distribusi label, kelengkapan, panjang jawaban, dan temuan audit yang dapat diperiksa. |
| Bab III: judul subbab data | Mengubah 3.2.2 menjadi **Struktur Respons dan Pengelompokan Akun** serta 3.2.4 menjadi **Label dan Dokumentasi Rubrik**, sesuai bukti yang tersedia. |
| Bab III: tokenisasi | Menjelaskan tokenizer bawaan setiap model, rangkaian pertanyaan–kunci jawaban–jawaban siswa, batas konteks, alokasi token pertanyaan/kunci, overlap jawaban, serta agregasi representasi window. |
| Bab III: pembagian dan pelatihan | Menyelaraskan pembagian development/test dan lima fold kelompok dengan metadata run. Memperjelas seed aktual, pooled RMSE untuk pemilihan checkpoint, penentuan epoch refit, serta status pengulangan antar-seed yang masih bersyarat. |
| Bab III: S1–S3 | S1 menggunakan RoBERTa, S2 menggunakan BGE-M3 frozen, dan S3 menggunakan concat sebagai pembanding dasar. Proyeksi, dimensi, head, serta hyperparameter pembanding dijelaskan sebagai konfigurasi yang dapat ditelusuri. Variasi S3 tetap berstatus kandidat. |
| Bab III: evaluasi | Memisahkan arah interpretasi RMSE dan Pearson. Menjelaskan metrik kontinu/kategori, keterbatasan kategori timpang, dan rencana bootstrap berpasangan pada kelompok akun. Riwayat penggunaan test selama pengembangan dicatat sebagai keterbatasan. |
| Gambar 3.1–3.3 | Mengganti tiga diagram metode: tahapan penelitian, alur model dengan tiga input, dan arsitektur pembanding concat. Diagram terakhir menyatakan konfigurasi akhir S3 masih ditentukan. |
| Tabel 3.1–3.6 | Menyesuaikan distribusi skor, skema kolom, skenario, konfigurasi pembanding/kandidat, pelaporan metrik, serta metadata eksperimen. Jadwal pada Tabel 3.7 dipertahankan sebagai rencana kerja. |
| Bab IV | Menambahkan judul **HASIL DAN PEMBAHASAN** dengan heading 4.1–4.6. Seluruh isi hasil masih kosong. |
| Bab V | Menambahkan judul **PENUTUP**, 5.1 Kesimpulan, dan 5.2 Saran. Seluruh isi masih kosong. |
| Format dan navigasi | Memperbarui daftar isi, daftar gambar, daftar tabel, caption, dan rujukan halaman; memperbaiki penomoran pertanyaan penelitian menggunakan fitur penomoran Word; menjaga tabel jadwal agar utuh pada satu halaman. Format A4, margin kiri 4 cm dan sisi lain 3 cm, Times New Roman, serta struktur heading naskah dipertahankan. |

## Fakta dataset yang dimasukkan

Workbook: `data_cleaned_3sekolah.xlsx`, sheet `Sheet1`.

SHA-256: `dfbd97128b09a4eb488dd890ccc3efd9b9fe232e74fcc00ee5504a03acc58226`.

| Komponen | Hasil pemeriksaan |
| --- | --- |
| Jumlah baris dan kolom | 1.340 jawaban; 13 kolom. |
| Akun | 134 kode akun anonim pada kolom `Email Address` setelah normalisasi. Setiap akun memiliki 10 jawaban. Jumlah ini ditulis sebagai akun, bukan hasil verifikasi identitas 134 individu. |
| Sekolah | SMAN 3 Pamekasan: 1.030 jawaban/103 akun; SMAN 1 Porong: 210/21; SMKN 55 Jakarta: 100/10. |
| Soal | Nomor 1–10, masing-masing 134 jawaban. Nomor 1–5 membahas poster, nomor 6–10 berita. |
| Teks input | `Pertanyaan`, `Kunci_Jawaban`, `Jawaban`. |
| Target | `Skor_Isi` 1–4; `Skor_Organisasi`, `Skor_Tata_Bahasa`, dan `Skor_Mekanik` 1–2. |
| Kelengkapan | Seluruh kolom terisi; semua status `Sudah Dinilai`; seluruh empat label berupa integer dalam rentang kategori eksperimen. |
| Distribusi isi | Skor 1: 20; 2: 182; 3: 466; 4: 672. |
| Distribusi organisasi | Skor 1: 193; 2: 1.147. |
| Distribusi kebahasaan | Skor 1: 296; 2: 1.044. |
| Distribusi mekanik | Skor 1: 291; 2: 1.049. |
| Panjang jawaban | Berdasarkan pemisahan spasi: minimum 1, median 63, maksimum 1.126 kata. Angka ini bukan jumlah token model. |
| Duplikat | Satu kelompok duplikat identik berisi dua jawaban setelah normalisasi untuk audit. |
| Skor total | Satu baris `Skor_Total` berbeda dari penjumlahan empat dimensi. Label dan file asli tidak dikoreksi otomatis; model menggunakan empat target dimensi. |
| Metadata run | Pembagian yang diperiksa memuat 1.070 jawaban development dan 270 test. Uraian ini bersumber dari indeks/metadata eksperimen, bukan kolom pembagian asli dalam workbook. |

## Bagian yang masih menunggu bukti atau hasil

1. **Konfigurasi final S3**: kandidat concat, dropout, head, serta cakupan fine-tuning belum dinyatakan sebagai pemenang. Tabel konfigurasi menyebut resep pembanding dan kandidat E17C; tidak mengklaim E17C sudah menghasilkan peningkatan.
2. **Kelas XI**: judul dipertahankan sesuai permintaan. Workbook tidak memuat kolom tingkat kelas, sehingga kesesuaian peserta dengan kelas XI perlu didukung dokumentasi pengumpulan data.
3. **Penilai dan rubrik**: workbook tidak memuat identitas penilai, deskriptor kategori, proses penilaian ulang, atau bukti konsistensi penilai. Status `Sudah Dinilai` menunjukkan isian status pada file dan tidak membuktikan prosedur penskoran tertentu.
4. **Bab IV–V**: hasil, angka metrik final, kesimpulan, dan saran belum ditulis. Bootstrap serta pengulangan antar-seed tetap berupa rencana bila belum dijalankan.
5. **Pemakaian data test**: test eksperimen terdahulu sudah diperiksa selama pengembangan. Keterbatasan ini dicatat; pemilihan kandidat selanjutnya diarahkan ke development.
6. **Kelengkapan skripsi final**: draf ini belum menyatakan seluruh syarat akhir pedoman telah terpenuhi. Bab IV–V, bagian awal yang memerlukan pengesahan, serta kelengkapan referensi akhir diperiksa kembali saat naskah dan hasil final tersedia. Jumlah halaman tidak ditambah dengan isi buatan.

## Lingkup berkas dan pemeriksaan

Dokumen DOCX tetap dapat diedit. Naskah telah dirender untuk memeriksa daftar, heading, tabel, diagram, dan perpindahan bab. Dua versi judul serta nama universitas diverifikasi terhadap sumber. Seluruh 30 entri daftar pustaka dipertahankan; 18 persamaan lain dipertahankan dan satu persamaan cosine diganti dengan concat. Ilustrasi teori dan logo di luar tiga diagram metode dipertahankan.

Folder ini memuat draf dan catatan revisi. Data individual, alamat email peserta, serta workbook mentah tidak dimasukkan ke unggahan revisi ini. Tidak ada perubahan kode eksperimen pada revisi naskah ini.


## Revisi lanjutan: abstrak dua bahasa dan tata letak persamaan

Perbaikan berdasarkan permintaan 24 November 2026:

- Menambahkan halaman **ABSTRACT** berbahasa Inggris setelah ABSTRAK Indonesia, beserta lima kata kunci terjemahan. Isi setara dengan abstrak Indonesia; status konfigurasi S3 tetap dalam pengembangan. Abstrak Indonesia dipertahankan. Kedua abstrak berupa satu paragraf, spasi 1,0, dan masing-masing di bawah 200 kata; versi Inggris berjumlah 150 kata berdasarkan pemisahan spasi.
- Menata ulang **19 persamaan** agar rumus berada di tengah area teks dan nomor dalam kurung rata kanan pada baris yang sama. Posisi tengah mengikuti arahan pengguna pada revisi ini. Rumus tetap berupa equation Word yang dapat diedit; seluruh isi 19 persamaan sama dengan draf sebelumnya.
- Menambahkan ABSTRACT ke daftar isi, menyesuaikan nomor Romawi bagian awal, dan memperbarui rujukan halaman daftar isi, gambar, serta tabel berdasarkan hasil render.
- Judul, nama universitas, uraian penelitian lainnya, sembilan tabel isi, tiga diagram metode, serta daftar pustaka tetap sama dengan draf sebelum perbaikan ini.
