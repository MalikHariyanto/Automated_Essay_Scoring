# E30 · co-attention lalu attention pooling

ZIP E30 berisi lima Excel asli. Hasil ditambahkan ke buku tiga input yang sama; tidak membuat aplikasi baru. Data300jawaban, label `rater`0–100,60test dan anggota lima fold cocok dengan sembilan unggahan sebelumnya. Snapshot/prediksi lama dan36duel lama tidak diubah.

E30 mengikuti **E26**, menambahkan pencocokan respons–referensi sebelum additive attention pooling. Urutan S3: RoBERTa token → BiGRU respons128 ↔ proyeksi referensi128 → co-attention dua arah → matching640→128+tanh → attention pooling128 → proyeksi768 → LayerNorm → concat BGE256 → MLP. E27 berbeda: memakai ringkasan ujung setelah matching, tanpa additive pooling ini.

RoBERTa S1/S3 memakai LR5e-6; linear S1, MLP S2/S3 dan seluruh proyeksi/BiGRU/attention/norm1e-4. BGE frozen. E30 tidak memakai LR MLP5e-6 seperti E29. Rujukan source `7a76191be20604482fa53f770699563b380120ba`; ZIP tidak menyertakan executed notebook/checkpoint untuk mengesahkan identitas byte kode runtime atau perangkat aktual.

## Hasil dan cara membandingkan

| S3 | E26 | E29 | E30 |
| --- | ---: | ---: | ---: |
| RMSE test · poin ↓ | 5,7435 | **5,4293** | 5,5247 |
| MAE test · poin ↓ | 4,4797 | 4,1713 | **4,0215** |
| Mean RMSE CV ↓ | 5,9245 | **5,7667** | 5,9491 |
| Dalam≤5poin ↑ | 38/60 | 40/60 | **44/60** |
| Dalam≤10poin ↑ | 54/60 | 55/60 | **56/60** |
| Terdekat di antara10peserta ↑ | 4/60 | 10/60 | **13/60** |

E30 lebih dekat pada lebih banyak jawaban menurut toleransi5/10poin dan memiliki MAE lebih rendah. E29 tetap RMSE terendah; kuadrat error membuat kesalahan besar lebih berpengaruh pada RMSE. E30 tidak otomatis paling baik di semua ukuran: CV-nya sedikit lebih tinggi daripada E26, dan E29 lebih baik pada toleransi1/2poin. Pada E30:0/60 skor mentah persis,3/60 sama setelah pembulatan,6/60 dalam1poin,22/60 dalam2poin. Angka44/60 dan13/60 bukan exact accuracy.

E30 S1 RMSE8,3369 dan S2 7,8672; prediksi keduanya identik per baris dengan kontrol E26. S3 Pearson0,7133,QWK0,6655,AC2 0,9854. Mean RMSE CV5,9491 dengan SD fold1,0737. Refit S1/S2/S3 memilih9/19/18epoch dari ceil median hasil lima fold.

Pada sepuluh peserta, jumlah terdekat S3 E21–E30 menjadi **7/7/5/3/5/4/4/2/10/13**. Menambah peserta mengubah hitungan kemenangan; tidak mengubah satu pun prediksi lama. Data test sudah sering dibandingkan, sehingga pembacaan lanjutan bersifat eksploratif; belum pengukuhan pada seed/test baru.

## Kode, bobot dan file asli

Meja belajar tetap hanya tiga kode, dipilih dengan aturan RMSE test S3 terendah: **E29, E30, E26**. E29 tetap kode utama; E30 menjadi pemimpin MAE/toleransi/jumlah terdekat. Label kartu kode utama menyebut kriteria RMSE agar tidak menyebut E29 pemimpin jumlah terdekat. Kode E24 tidak lagi diembed, tetapi hasil/Excel dan tautan sumbernya tetap tersedia.

E30 menyediakan30blok/2955baris,146catatan dan10padanan screenshot. Catatan menjelaskan input/label, pretrained, head linear/MLP, proyeksi, mask, BiGRU, co-attention dua arah, additive pooling, optimizer dan ekspor. E29/E26 yang tetap dimuat identik dengan buku sebelumnya.

Semua50Excel/458sheet tersedia melalui tab08SemuaExcel. Bobot pooling, respons→referensi dan referensi→respons dipisahkan: finite, nonnegatif, berjumlah1 per jawaban/window dan posisi token cocok dengan tabel tokenisasi. Audit memeriksa16946baris respons/pooling,17600referensi,16946pasangan top1 dan dimensi128/768. Matriks alignment penuh tidak tersedia; top1 tidak dapat digunakan untuk menghitung ulang seluruh matriks atau membuktikan maksimum secara independen. Bobot bukan nilai guru atau penjelasan kausal prediksi.

Sitasi pada identitas E30: [RoBERTa](https://arxiv.org/abs/1907.11692), [BGE-M3](https://arxiv.org/abs/2402.03216), [PaSTO-GNN/BiGRU](https://doi.org/10.3389/frai.2026.1842542), [Dong et al.2017/attention AES](https://aclanthology.org/K17-1017/), dan [Zhang & Litman2018/co-attention AES](https://aclanthology.org/W18-0549/). Komponen diadaptasi pada token RoBERTa/BGE; bukan klaim mereplikasi seluruh paper atau jaminan performa. Pembaruan ini tidak menjalankan penelitian baru.

Buku lengkap menggunakan kunci lama, arsip terenkripsi dan nama pengunduh yang sama. Preview publik hanya berisi agregat/source aman. Aplikasi MLP lama tidak diubah.
