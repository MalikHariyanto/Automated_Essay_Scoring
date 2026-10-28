# E24 dalam buku tiga input

> Dokumen tahap terdahulu. Angka dan keputusan dibaca dalam konteks versi ini; untuk kondisi aktif gunakan [status proyek](../../STATUS_PROYEK.md) dan [panduan AI](../../PANDUAN_AI_PROYEK.md).

HTML tiga input kini membandingkan **E21, E22 dan E24**, beserta tiga meja belajar kode, seluruh sheet dari **15 Excel asli**, dan seluruh prediksi pada60jawaban test yang sama. E23 belum mempunyai hasil unggahan; tidak dibuatkan nilai atau tombol hasil pengganti. Paket unduhan tetap `Belajar_AES_Tiga_Input_Lengkap`, dengan kunci sebelumnya.

## Audit kesetaraan dan sumber

Ketiga ZIP mempunyai data300baris, soal/referensi/respons, label `rater`, pembagian240development/60test dan anggota lima fold yang sama. E24 mencatat `single_sequence`, attention pooling S3, hidden64, encoder LR5e-6 dan proyeksi/head1e-4. S1/S2 pada output E24 identik dengan E21; tambahan attention berlaku pada S3. Kesamaan angka tidak diklaim sebagai bukti bahwa runtime menggunakan checkpoint yang sama atau melatih ulang pada hardware yang identik.

Source E24 dipin pada [notebook yang dikirim sebelumnya, commit f3d25c4](https://github.com/MalikHariyanto/Automated_Essay_Scoring/blob/f3d25c4/notebooks/experiments/AES_300_E24_Tiga_Input_AttentionPooling_GPU.ipynb). Kode di meja belajar dicocokkan dengan source notebook tersebut, tanpa mengganti kode saat memberi label E24. Pengaturan workbook cocok, tetapi notebook eksekusi/log mentah tidak disertakan. Buku tidak mengarang output cell, menjalankan model atau mengulang training.

Empat metrik test dihitung ulang dari prediksi tersimpan dalam toleransi1e-6; nilai laporan asli dipertahankan. CV dicocokkan dengan epoch minimum pada riwayat, lima fold dan ceil median epoch refit. CV tidak diklaim dihitung ulang dari pasangan prediksi individual fold yang tidak disertakan di ZIP.

## Hasil S3

| Ukuran | E21 | E22 | E24 |
| --- | ---: | ---: | ---: |
| RMSE test ↓ | 6,7231 | 7,1943 | **5,9818** |
| MAE test ↓ | 5,2515 | 5,8108 | **4,5286** |
| Pearson test ↑ | 0,5187 | 0,5270 | **0,6801** |
| QWK test ↑ | 0,4340 | 0,3181 | **0,6131** |
| Gwet AC2 test ↑ | 0,9789 | 0,9764 | **0,9831** |
| Rerata RMSE CV ↓ | 6,6246 | 6,4911 | **6,0148** |
| SD RMSE antar-fold ↓ | 1,1658 | **0,2988** | 1,4999 |
| Sama setelah pembulatan | 5/60 | 5/60 | **7/60** |
| Selisih maksimal5poin | 32/60 | 27/60 | **36/60** |
| Selisih maksimal10poin | **55/60** | 50/60 | **55/60** |

RMSE test E24 sekitar11,03% lebih rendah daripada E21. Pada perbandingan S3 berpasangan, E24 lebih dekat pada37jawaban vs23untuk E21, dan39vs21terhadap E22. Persaingan bersama tiga atau sembilan model menghitung kemenangan dengan kandidat berbeda, sehingga ranking dan jumlah kemenangan dapat berubah tanpa perubahan prediksi lama. Hasil satu run dan test yang sudah dilihat bersifat eksploratif; rerata CV membaik tetapi SD antar-fold E24 lebih besar, sehingga tidak disimpulkan lebih stabil antar-seed.

## Isi dan sorotan tabel

- Eksperimen menjadi kolom pada metrik dan ketepatan. Seluruh540prediksi (60jawaban × sembilan model) tersedia bersama, termasuk kartu tiga teks dan ranking. Hasil E21 vs E22 tetap bisa dibaca pada tabel pasangan yang terpisah.
- Angka terbaik diberi latar hijau dan font tebal. Jika beberapa angka terbaik sama tetapi ada pembanding yang lebih buruk, diberi label **Seri**. Jika semua angka sama, termasuk0/60, tampil netral. Hijau adalah perbandingan relatif, bukan penetapan ambang layak dipakai guru.
- Rerata dan SD CV dibandingkan pada baris terpisah. RMSE tiap fold juga disorot. Epoch refit/epoch pilihan tetap netral karena jumlah epoch bukan indikator mutu. Heatmap peta skor menyatakan kepadatan pasangan, bukan peringkat metrik.
- Ketiga riwayat training menggunakan sumbu grafik bersama dan titik epoch pilihan. Peta skor memakai skala warna bersama. Kontrol hasil dan kode mempertahankan fullscreen, navigasi keyboard serta scroll desktop/HP.
- Meja belajar E24 berisi16bab,30blok/2780baris kode dan118catatan, termasuk mask jawaban, jaringan attention, softmax bertopeng, weighted sum768 dan posisi pooling sebelum fusion. Kode/sticky note berada dalam satu kanvas fullscreen tanpa heading luar.
- Penjelajah Excel menyertakan **Pooling Jawaban S3** dan **Bobot Attention S3**. Ada16.946baris bobot pada300jawaban/window; seluruhnya finite/nonnegatif dan berjumlah satu per window. Posisi/ID token bobot dicocokkan dengan token jawaban pada tabel tokenisasi, bukan konteks atau padding. Bobot attention tidak diperlakukan sebagai bukti sebab-akibat skor.

## Privasi dan pemeriksaan

Versi publik hanya memuat agregat/kode/inventaris. ZIP, jawaban, ID, nilai individual, token/vektor dan bobot token berada dalam buku terenkripsi; kunci tidak diterbitkan. Payload mempertahankan ketiga ZIP serta15workbook byte demi byte. Prefix teks privat diperiksa terhadap HTML publik tanpa kecocokan.

14pengujian integritas/unit lulus (0failed/skipped/xfailed). Delapan kelompok browser memeriksa metrik dan highlight, delapan ukuran ketepatan, tiga pasangan,540prediksi,60label/teks, ranking,15grafik, peta, seluruh sheet dan lima unduhan XLSX E24 byte-identik,16bab/30blok, catatan baru dan fullscreen pada390/768/1440. Dua pemeriksaan paket memverifikasi pengunduh merekonstruksi ZIP identik dan kunci lama membuka perbandingan/kode/fullscreen E24 setelah jaringan diblokir. Pembukaan diuji melalui HTTP localhost; jalur `file://` tidak diklaim teruji di lingkungan browser ini. Ini audit hasil tersimpan dan pengujian HTML, bukan pengulangan eksperimen penelitian.

Bukti command, versi, hash, XML, audit privasi dan pemeriksaan remote disimpan lokal di `outputs/e24_study/`.
