# Audit E34 — 2 November 2026

**E34 membaik sedikit dari E33 pada mean RMSE CV, RMSE/MAE test dan agreement, tetapi belum mengalahkan E30 pada test.** E29 masih memiliki mean RMSE CV dan RMSE test S3 terendah dalam keluarga E21–E34. E30 tetap acuan kuat untuk MAE dan jumlah prediksi dalam ±5 poin. Perbedaan ini berasal dari satu seed dan test yang telah diamati berulang kali; belum membuktikan keunggulan yang stabil.

## Data dan resep yang diaudit

Sumber: ZIP unggahan `Hasil_AES_300_E34_E33_BiGRU_CoAttention_LayersLR5e5_GPU.zip`, lima workbook asli. Terdapat 300 jawaban, 240 development, 60 test, lima fold pada development, maksimal 25 epoch, early stopping patience 5. Teks, label, keanggotaan test dan fold dicocokkan dengan hasil sebelumnya ketika membangun HTML. Tidak ada training ulang dalam audit ini.

Source rujukan: commit `7f3363643e6bf402a293de069bcfc23d85b19c89`, notebook E34. Workbook cocok dengan resep: satu-satunya parameter tercatat yang berubah dari E33 adalah **LR pooling/norm S3 1e-4 → 5e-5**. Kelompok ini mencakup `sequence_pool.*` dan `roberta_norm.*`: BiGRU, proyeksi referensi, matching/co-attention, attention pooling, proyeksi ringkasan128→768 dan norm. MLP S3 tetap5e-5; proyeksi BGE tetap1e-4; encoder RoBERTa5e-6. Ini bukan eksperimen yang mengisolasi satu layer individual. Resep mempertahankan RoBERTa fine-tuned, BGE frozen dan prediksi akhir MLP; tidak memakai Ridge.

Notebook eksekusi, log perangkat dan checkpoint tidak ada dalam ZIP. Kecocokan byte source runtime, perangkat dan status optimizer ketika training tidak dapat dibuktikan dari workbook saja. Identitas prediksi kontrol juga tidak membuktikan apakah kontrol dilatih ulang atau dipakai kembali.

## Perbandingan S3

| Ukuran | E29 | E30 | E33 | E34 |
|---|---:|---:|---:|---:|
| Mean RMSE CV ↓ | **5,7667** | 5,9491 | 5,9118 | 5,8931 |
| SD antar-fold | 1,4699 | 1,0737 | 1,0356 | 1,1105 |
| RMSE test ↓ | **5,4293** | 5,5247 | 6,0388 | 5,9610 |
| MAE test ↓ | 4,1713 | **4,0215** | 4,4769 | 4,4007 |
| Pearson ↑ | 0,7496 | 0,7133 | 0,6537 | 0,6591 |
| QWK ↑ | 0,6717 | 0,6655 | 0,5780 | 0,5922 |
| Gwet AC2 ↑ | 0,9857 | 0,9854 | 0,9826 | 0,9833 |
| Bias prediksi−guru, poin | −1,4286 | +0,4361 | −1,0356 | −0,7507 |
| Refit S3, epoch | 12 | 18 | 11 | 11 |

Mean RMSE CV E34 turun0,0187poin (sekitar0,32%) dari E33, urutan ketiga dari14eksperimen, setelah E29 dan E32. RMSE test turun0,0777poin (sekitar1,29%); MAE turun0,0763poin (sekitar1,70%). Perbaikannya konsisten pada beberapa ringkasan tetapi kecil. SD antar-fold justru meningkat, sehingga tidak tepat menyebut E34 lebih stabil dari bukti ini. SD fold juga bukan SD antar-seed.

| Ketepatan pada60jawaban test | E30 | E33 | E34 |
|---|---:|---:|---:|
| Selisih≤1poin | 6 | **14** | 13 |
| Selisih≤2poin | **22** | 20 | 19 |
| Selisih≤5poin | **44** | 37 | 39 |
| Selisih≤10poin | **56** | 55 | 54 |
| Sama setelah pembulatan | 3 | 5 | **6** |

E34 lebih baik pada RMSE/MAE dan toleransi5poin dibanding E33, namun lebih sedikit yang berada dalam1,2dan10poin. Ada6jawaban dengan error di atas10poin, dibanding5pada E33. Enam hasil sama setelah pembulatan bukan enam prediksi mentah persis sama; skor mentah yang persis sama tetap0. Kategori untuk agreement adalah salinan terpisah; prediksi kontinu tidak diubah.

Duel per jawaban S3: E34 lebih dekat daripada E33 pada32jawaban, E33 pada28, tanpa seri. Melawan E30, E34 menang27dan E30 menang33. Jumlah kemenangan tidak memberi bobot tambahan pada besarnya error seperti RMSE. Bias negatif E34 berarti secara rata-rata masih memberi skor0,7507poin lebih rendah daripada guru.

## Lima fold dan early stopping

| Fold S3 | E33 RMSE | E34 RMSE | Epoch terbaik E34 | Epoch dijalankan E34 |
|---|---:|---:|---:|---:|
| 1 | **4,6887** | 4,7434 | 15 | 20 |
| 2 | 5,5060 | **5,4794** | 11 | 16 |
| 3 | 6,1304 | **6,0640** | 15 | 20 |
| 4 | **7,5079** | 7,6943 | 7 | 12 |
| 5 | 5,7261 | **5,4846** | 10 | 15 |

E34 menang3dari5fold terhadap E33. Fold4 tetap paling sulit dan memburuk0,1864poin; Fold5 memberi penurunan terbesar0,2415poin. Epoch pada tabel ini mengikuti ringkasan asli yang diverifikasi terhadap riwayat, bukan menebak dari jumlah epoch final.

Refit menggunakan ceil median epoch terbaik lima fold. S3 E33 dan E34 sama-sama11epoch; perbedaan hasilnya tidak dapat dijelaskan oleh jumlah epoch refit yang berbeda. Early stopping memilih RMSE validasi terbaik, bukan epoch berdasarkan test.

## Kontrol dan kelengkapan ekspor

Prediksi S1 dan S2 E34 identik per jawaban dengan E33 dan E30. S1 RMSE8,3369; S2 RMSE7,8672. LR baru ini khusus kelompok layer S3 sehingga tidak memperbaiki S2. Jika masalah BGE S2 ingin diperbaiki, jadikan eksperimen terpisah agar penyebab perubahan S3 tetap dapat dibandingkan.

Seluruh RMSE, Pearson, QWK dan AC2 test dihitung ulang dari60pasangan guru/prediksi dan cocok dengan ringkasan dalam toleransi1e-6; deviasi maksimal sekitar2,40e-7. Audit juga memeriksa pemisahan development/test, lima validation fold, pemilihan epoch, jumlah refit, pembulatan, cakupan token, finite vector dan posisi token attention. Bobot respons16.946baris dan referensi17.600baris cocok dengan token sumber; jumlah bobot per window mendekati1, dengan deviasi maksimal sekitar3,54e-8. Ringkasan co-attention128dimensi dan pooling768dimensi lengkap untuk300jawaban. Matriks alignment penuh tidak diekspor; sifat maksimum top1 tidak dihitung ulang dari matriks tersebut.

HTML memperluas perbandingan menjadi14eksperimen/42model,91duel dan70Excel asli. Semua13snapshot hasil lama, audit prediksi,78duel lama, serta tiga buku kode E29/E30/E26 dipertahankan. Peringkat paling sering terdekat dihitung ulang ketika peserta bertambah; perubahan ranking bukan perubahan prediksi lama. Seluruh workbook dan ZIP hasil tetap byte-identik dalam paket privat. Buku lama keluarga MLP tidak diubah. Paket terbaru menggunakan kunci pembuka lama. Karena HTML lengkap melewati batas satu string besar pada V8/Chromium, pembuka kini menulis HTML secara bertahap dengan decoder UTF-8 streaming setelah verifikasi hash; isi ZIP/XLSX tidak diubah.

## Langkah penelitian berikutnya

E34 layak menjadi kandidat lanjutan E33, tetapi belum cukup kuat untuk menggantikan E30 sebagai acuan praktis. Langkah paling informatif adalah melengkapi E30/E33/E34 dengan seed123dan2026 pada split dan fold yang sama, lalu membandingkan mean dan SD antar-seed melalui development/CV. Jangan memilih seed dari skor test terbaik. Pertahankan E29 pada tabel sebagai acuan RMSE historis, dengan catatan S2 E29 berbeda LR dan hasilnya buruk.

Setelah mengukur kestabilan, baru tentukan apakah LR layer5e-5 layak dipertahankan atau perlu ablasi yang lebih terarah. Test60jawaban ini telah diamati berulang kali selama eksplorasi; klaim akhir memerlukan evaluasi yang benar-benar tidak dipakai untuk memilih resep, misalnya holdout baru yang disisihkan sebelum tuning. Audit ini tidak membuat hasil E35 atau mengklaim bahwa LR lebih kecil selalu lebih baik.
