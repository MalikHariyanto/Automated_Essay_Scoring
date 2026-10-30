# E31/E32 · hasil unggahan dari acuan E25

Dua ZIP unggahan diaudit dari lima workbook asli masing-masing, tanpa training baru. E31 mempertahankan attention dan gated fusion E25, menurunkan LR MLP S2/S3 menjadi5e-6. E32 menambahkan co-attention respons–referensi sebelum attention pooling768 dan gate E25; MLP tetap1e-4. Keduanya tanpa BiGRU. Linear S1, proyeksi, gate, pooling/norm tetap1e-4; encoder RoBERTa5e-6 dan BGE frozen.

| S3 | E25 | E31 | E32 |
| --- | ---: | ---: | ---: |
| RMSE test · poin | 6,1053 | 6,5735 | 5,8059 |
| MAE test · poin | 4,7006 | 5,0837 | 4,7006 |
| Mean RMSE lima fold | 6,0096 | 6,1740 | 5,8725 |
| Selisih maksimal5poin | 35/60 | 34/60 | 37/60 |
| Selisih maksimal10poin | 54/60 | 54/60 | 53/60 |

E32 menurunkan RMSE dibanding E25 pada unggahan ini, tetapi tidak memperbaiki semua ukuran. E31 justru memburuk; S2 E31 memiliki RMSE75,9702 dan hasil tersebut tetap ditampilkan. MAE E32=4,7005644667 hampir sama dengan E25=4,7006412500; jangan membesar-besarkan selisih kecil ini. Test telah berulang kali dilihat sehingga perbandingan bersifat eksploratif, bukan konfirmasi baru.

Source rujukan kedua resep dipin ke `d72a6e6f4f5271aa7939f4ffc93c28bc94a6b0be`. ZIP hanya memuat Excel; notebook eksekusi, checkpoint dan perangkat aktual tidak dibuktikan. Metrik dihitung ulang dari60pasangan test dengan selisih maksimum2,41e-7. Split240/60, lima fold, epoch refit, cakupan token, gate serta pengalinya diperiksa. Pada E32 bobot pooling dan kedua arah co-attention terpisah, finite, nonnegatif, jumlah1 per window; posisi token cocok sumber. Ringkasan pooling tetap768, tanpa ringkasan BiGRU128. Ekspor top1 bukan matriks alignment penuh dan tidak membuktikan sebab-akibat prediksi.

Buku yang sama memuat12hasil/36model,60Excel/557sheet dan66duel pasangan. Peringkat dihitung ulang untuk12peserta; penambahan kandidat dapat mengubah jumlah terdekat tanpa mengubah prediksi lama. Sepuluh snapshot lama,45duel lama dan tiga buku kode dipertahankan. Meja belajar tetap E29/E30/E26 menurut tiga RMSE test S3 terendah, utama E29. Identitas E31/E32 menyediakan source, sitasi dan tombol Excel meskipun bukan tiga kode yang diembed. Paket lengkap mempertahankan kunci lama dan ZIP/Excel byte-identik; preview publik hanya agregat/source aman. Buku sebelum tiga input tidak diubah.

Pada12peserta, jumlah terdekat S3 E21–E32 menjadi7/6/5/2/5/3/4/2/7/11/3/5. E30 tetap paling sering terdekat11/60. Ini perubahan kandidat, bukan perubahan prediksi lama.
