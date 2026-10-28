# E23 dan meja belajar dengan catatan sejajar

> Dokumen tahap terdahulu. Angka dan keputusan dibaca dalam konteks versi ini; untuk kondisi aktif gunakan [status proyek](../../STATUS_PROYEK.md) dan [panduan AI](../../PANDUAN_AI_PROYEK.md).

Hasil E23 BiGRU ditambahkan pada buku tiga variabel yang sama, bersama E21, E22 dan E24. Tetap dua aplikasi utama: buku eksperimen sebelum tiga variabel dan buku tiga variabel terbaru. Tidak diterbitkan HTML E23 terpisah.

## Empat eksperimen, tiga skenario, data sama

Data300baris, soal/referensi/respons, label `rater`,240development/60test dan anggota lima fold dicocokkan antar-keempat ZIP. E23 memakai single_sequence, encoder RoBERTa S1/S3 LR5e-6, proyeksi/head1e-4 dan BiGRU hidden64 per arah hanya pada S3. S1/S2 E23 menghasilkan angka yang sama dengan E21/E24; kesamaan angka tidak membuktikan identitas runtime atau hardware.

Source E23 dipin pada [notebook asli, f3d25c4](https://github.com/MalikHariyanto/Automated_Essay_Scoring/blob/f3d25c4/notebooks/experiments/AES_300_E23_Tiga_Input_BiGRU_GPU.ipynb). Seluruh code cell di buku dicocokkan byte demi byte dengan notebook rujukan masing-masing. Notebook eksekusi tidak diunggah, sehingga buku tidak mengarang log cell atau mengklaim hardware runtime telah dibuktikan.

## Kode utama ditentukan dari ukuran yang eksplisit

| Ukuran S3 | E23 · BiGRU | E24 · attention |
| --- | ---: | ---: |
| RMSE test ↓ | 6,0439 | **5,9818** |
| MAE test ↓ | 4,8576 | **4,5286** |
| Pearson ↑ | **0,6923** | 0,6801 |
| QWK ↑ | **0,6254** | 0,6131 |
| Gwet AC2 ↑ | 0,9825 | **0,9831** |
| Terdekat dalam persaingan empat eksperimen | **18/60** | **18/60** |
| Terdekat dalam duel E23 vs E24 | 28/60 | **32/60** |
| Sama sesudah pembulatan | 2/60 | **7/60** |
| Selisih maksimal5poin | **39/60** | 36/60 |
| Selisih maksimal10poin | 53/60 | **55/60** |
| Rerata RMSE CV ↓ | 6,4143 | **6,0148** |
| SD RMSE antar-fold | **0,7442** | 1,4999 |

E23 tidak dinyatakan mengalahkan E24 pada semua ukuran. Pilihan default meja belajar adalah **E24**: seri jumlah terdekat, lalu RMSE lebih rendah. E23 mempunyai tombol belajar tersendiri karena unggul pada jumlah jawaban dalam toleransi lima poin serta Pearson/QWK. Pilihan default ini adalah ringkasan deskriptif untuk belajar dari hasil yang sudah dilihat, bukan prosedur baru memilih model penelitian dengan test.

Jumlah terdekat sekarang E21=12, E22=12, E23=18, E24=18 dari60jawaban S3. Jumlah lama berubah karena E23 menjadi peserta baru, tanpa mengubah prediksi lama. Pada S1/S2 ada prediksi identik antar-eksperimen; hasil seri dibagi sesuai jumlah peserta. Total bagian kemenangan pada setiap cakupan tetap60. Seluruh enam duel berpasangan tersedia, sehingga hasil E21 vs E22 atau E21 vs E24 sebelumnya tetap dapat diperiksa tanpa perubahan kandidat lain.

## Isi belajar lengkap

Buku memuat20Excel/176sheet,720prediksi mentah (60jawaban ×12model), empat riwayat kurva dan peta skor. Setiap eksperimen memiliki kode penuh,16bab dan30blok. E23 mempunyai2.786baris/136catatan; E24 mempunyai2.780baris/135catatan. Jumlah tersebut bukan jumlah parameter atau ukuran mutu model.

Catatan sekarang tampil **bersamaan di samping ruas kode yang dijelaskan**, mengikuti konsep `Belajar_Kode_E6.html`. Tinggi ruas mengikuti kode atau catatan yang lebih panjang; ruang kosong pada kode sengaja tersedia agar penjelasan tidak bertabrakan. Nomor baris dan isi kode tetap asli. Pada HP, catatan mengikuti ruasnya di bawah kode. Fullscreen mempunyai satu scroll vertikal kode/catatan, tanpa heading induk, dan tetap menyediakan mode semua blok, per blok serta satu catatan bila diperlukan.

Peta **Padanan screenshot → kode** menghubungkan sepuluh bagian: hyperparameter, susunan konteks, Dataset/X/y, inisialisasi pretrained, pooling, classifier/head, ekstraksi fitur awal, lima fold, loss/optimizer serta prediksi/evaluasi. Setiap tombol melompat ke cell/baris dan anchor asli. Penjelasan menyesuaikan eksperimen: E6 jawaban-saja tidak disalin sebagai penjelasan tiga input. S3 E23 memakai state BiGRU, S3 E24 attention; CLS S1 tetap dibedakan.

Catatan tambahan mencakup `bundle`, ketiga input, `rater` sebagai label, encoder versus head, MLP versus linear, LayerNorm, proyeksi BGE, proyeksi BiGRU128→768, vektor768/1024, mask token jawaban, gradien/frozen, LR, MSE/RMSE, CV/refit/test, dan `export_final_features` beserta argumennya.

## Audit dan privasi

Metrik test E23 dihitung ulang dari prediksi dalam toleransi1e-6. Pemilihan epoch, panjang riwayat dan refit diverifikasi dari workbook. Fitur `BiGRU S3 State Akhir` memuat300vektor128; `Pooling Jawaban S3` memuat300vektor768; nilai finite dan ID sejajar dengan data. Ini inspeksi fitur300jawaban, bukan evaluasi300test.

ZIP/XLSX asli tetap byte-identik dan disimpan hanya dalam paket buku terenkripsi. Kunci lama dipertahankan. HTML publik memuat agregat, source dan catatan; teks, ID, nilai individual serta vektor siswa tidak dipublikasikan terbuka. Pemeriksaan unit, browser, seluruh kode/catatan, pemetaan screenshot, fullscreen/scroll, unduhan dan paket disimpan lokal di `outputs/e23_study/`. Tidak dilakukan training model baru pada pekerjaan buku ini.

16 pengujian integritas/unit, delapan kelompok pemeriksaan browser dan dua pemeriksaan paket lulus (0 gagal). Pemeriksaan kode/catatan memakai source publik identik setelah pemeriksaan data privat; paket lengkap juga dibuka offline menggunakan kunci lama. Browser diuji melalui HTTP localhost lalu akses jaringan diblokir; jalur file:// tidak diklaim teruji di lingkungan ini.
