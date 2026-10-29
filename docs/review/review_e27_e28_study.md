# E27/E28 ditambahkan · isi lama dipertahankan

**29 Oktober2026.** Buku tiga input kini memuat E21–E28: **8eksperimen,24model,40Excel,363sheet**,60jawaban uji yang sama. Buku sebelum tiga input tetap terpisah dan tidak diperbarui. [Pengunduh lengkap](../../downloads/Belajar_AES_Tiga_Input_Lengkap.html) · [preview publik](../belajar_Tiga_Input_Spider.html).

## Hasil unggahan baru

| S3 | RMSE test ↓ | Mean RMSE CV ↓ | MAE test ↓ | Selisih≤5poin ↑ | Refit pilihan CV |
| --- | ---: | ---: | ---: | ---: | ---: |
| E23 · pembanding | 6,0439 | 6,4143 | 4,8576 | 39/60 | Sesuai workbook E23 |
| E27 · E23 + co-attention | 6,3277 | 6,5067 | 5,0281 | 34/60 | 5epoch |
| E28 · E23 + LR MLP S3 5e-6 | 7,0447 | 6,4386 | 5,8803 | 27/60 | 9epoch |

Angka disalin dari lima Excel masing-masing dan diaudit, bukan hasil training ulang untuk HTML. RMSE/Pearson/QWK/AC2 dihitung ulang dari prediksi mentah tersimpan dengan toleransi1e-6; perbedaan kecil timbul karena presisi prediksi dalam Excel. Nilai reported asli tetap digunakan pada tabel metrik. Kedua hasil belum memperbaiki RMSE S3 E23 pada run unggahan; ini bukan bukti metode selalu gagal atau kesimpulan signifikansi statistik.

## Audit sumber dan pairing

- 300teks/label,60ID/label test, dan anggota lima fold cocok dengan E21–E26. Semua prediksi S1/S2 E27/E28 identik dengan E23 per baris; kesamaan angka bukan bukti perangkat atau independensi run.
- E27 source [co-attention](../../notebooks/experiments/AES_300_E27_E23_BiGRU_CoAttention_GPU.ipynb), E28 source [MLP LR](../../notebooks/experiments/AES_300_E28_E23_BiGRU_MLPLR5e6_GPU.ipynb), keduanya dipin ke `daa745de0aee5c7bd26090af8fbf7ad99b0cb49c`. Notebook eksekusi/bobot/runtime lengkap tidak disertakan dalam ZIP; byte runtime dan perangkat aktual tidak diklaim sudah terbukti.
- E27:16946baris bobot respons,17600baris bobot referensi,16946pasangan token top1;300jawaban/window. Bobot finite/nonnegatif dan jumlah1 per window, error jumlah terbesar4,47e-8. Posisi/token cocok dengan tokenisasi aktual. Vektor ringkasan128 dan pooling768 finite, berurutan300jawaban.
- Top1 merupakan ringkasan yang diunggah. Matriks lengkap tidak tersedia untuk menghitung ulang apakah maksimum memang top1; tidak diperlakukan sebagai bukti kebenaran jawaban atau penjelasan kausal.
- E28: ringkasan state akhir128 dan proyeksi768 mengikuti E23. Config Excel mencatat MLP S3 saja5e-6, BiGRU/norm1e-4, proyeksi BGE1e-4, kontrol S1/S2 head1e-4. Tidak ditampilkan seolah-olah seluruh layer baru diturunkan LR-nya.

## Yang dipertahankan

Enam dokumen HTML anak E21–E26 dibandingkan utuh dengan commit sebelum pembaruan: **identik**, termasuk kode, catatan pinggir, penjelasan, curriculum, sumber dan metadata hasil. Snapshot, empat metrik, MSE/MAE/toleransi, riwayat/fold, hash40workbook, dan15duel lama juga diperiksa. Sepuluh workbook baru tetap byte sumbernya; enam ZIP lama tidak dimodifikasi.

Penambahan peserta mengubah hitungan terdekat. Pada delapan peserta S3, urutan E21/E22/E23/E24/E25/E26/E27/E28 adalah **8/8/10/6/8/10/8/2**; jumlah60. E23 dahulu13/60 pada enam peserta, sekarang seri dengan E26 pada10/60. **Prediksi lama tidak berubah.** E26 tetap RMSE/MAE terendah; E23 tetap terbanyak≤5poin39/60. Kode utama meja belajar tetap **E23** sesuai pilihan pengguna, terpisah dari urutan statistik yang dihitung ulang.

## Optimasi UI tanpa menyunting materi lama

- Warna E27/E28 konsisten pada tabel, prediksi, kurva dan pilihan kode; label baris tabel tetap terlihat saat digeser mendatar. Ukuran dan fokus kontrol diperjelas pada desktop/HP.
- Posisi scroll vertikal panel dan horizontal tabel disimpan per tab/skenario. Kembali ke tab memulihkan posisi; fullscreen tetap menutupi viewport dengan mode semua blok/per blok.
- Permintaan membuka kode diantrekan; pilihan terakhir menang sehingga perpindahan cepat tidak menampilkan kode eksperimen lain. Cache base64 dibatasi ke buku aktif dan arsip ke dua entri; isi tetap tersedia dari paket.
- Payload besar dipecah menjadi node transport1MiB dan dibaca bertahap. ZIP sumber yang didekode tetap sama; ini perubahan cara membuka paket, bukan pemotongan data, kode, atau Excel.

Materi baru E27/E28 mengikuti notebook masing-masing, termasuk sticky note,10padanan screenshot kode, output tahap dari Excel, co-attention dan kelompok optimizer. Tidak mengarang stdout cell yang tidak disertakan pada unggahan. Paket lengkap terenkripsi memakai kunci yang sama; preview publik hanya source/agregat.

Hasil akhir:21tes integritas/unit,10kelompok browser,2uji paket lulus; [bukti dan batas verifikasi](../verifikasi/verification_e27_e28_study.md). Tes memeriksa isi, navigasi, scroll/fullscreen dan unduhan byte-identik. Bukti lokal berada pada `outputs/e27_e28_study/`; hasil verifikasi final dicatat bersama manifest sumber sebelum publikasi. Tes HTML bukan ukuran akurasi AES dan tidak menjalankan training penelitian baru.
