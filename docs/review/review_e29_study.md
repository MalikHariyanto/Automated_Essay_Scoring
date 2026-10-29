# E29 · hasil aktual dan tiga kode terpilih

E29 berasal dari E26: arsitektur S3 tetap **BiGRU → attention pooling → proyeksi → fusion → MLP**. LR MLP **S2 dan S3** turun dari1e-4 menjadi5e-6. Linear S1, proyeksi BGE dan pooling/norm S3 tetap1e-4; encoder RoBERTa5e-6 dan BGE frozen. Tiga input dan label `rater`0–100 tidak berubah. Source rujukan dipin ke `7a76191be20604482fa53f770699563b380120ba`; ZIP tidak menyertakan executed notebook/checkpoint untuk mengesahkan identitas byte kode runtime atau perangkat aktual.

## Hasil yang diverifikasi

| Skenario E29 | RMSE test · poin | Pearson | QWK | Gwet AC2 | Mean RMSE CV |
| --- | ---: | ---: | ---: | ---: | ---: |
| S1 | 8,3369 | 0,4487 | 0,2515 | 0,9681 | 7,3645 |
| S2 | 75,9702 | −0,2237 | −0,0003 | −1,2044 | 75,0373 |
| S3 | **5,4293** | **0,7496** | **0,6717** | **0,9857** | **5,7667** |

Metrik dihitung ulang dari pasangan guru/prediksi asli, selisih maksimum terhadap laporan<1e-6. Nilai AC2 S2 yang negatif tidak dipotong. Tiga skenario tetap ditampilkan, termasuk hasil yang buruk.

S3 E29 memiliki MAE4,1713;0/60 persis pada skor mentah,4/60 setelah pembulatan,9/60 dalam1poin,23/60 dalam2poin,40/60 dalam5poin,55/60 dalam10poin. Angka40/60 bukan exact accuracy. RMSE S3 turun dari E26 5,7435 ke5,4293, sekitar5,47%. Mean CV E29 5,7667 membaik dari E26 5,9245, tetapi SD fold E29 1,4699 menunjukkan variasi yang masih besar; bukan bukti kestabilan seed.

S2 memprediksi sekitar11,4404–13,9018, dengan mean12,8410, sedangkan mean skor guru88,3833. Semua fold memilih epoch25, tepat batas maksimal; RMSE validasi masih menurun antara epoch1 dan25. Ini konsisten dengan underfitting/belum cukup belajar pada anggaran ini, bukan bukti tunggal implementasi salah atau jaminan menambah epoch akan menyelesaikannya. LR MLP S2 yang diturunkan20× tidak menguntungkan dalam run ini.

## Perbandingan sembilan eksperimen

Data300,60test, label dan anggota lima fold cocok dengan delapan unggahan sebelumnya. Seluruh snapshot, pasangan prediksi dan28duel lama dipertahankan. Menambah E29 menghasilkan36duel dan27model. Jumlah terdekat S3 pada sembilan peserta:

| E21 | E22 | E23 | E24 | E25 | E26 | E27 | E28 | E29 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 7 | 8 | 5 | 7 | 6 | 7 | 2 | **10** |

Ini jumlah prediksi dengan selisih terkecil di antara peserta, bukan exact accuracy. Prediksi lama tidak berubah; jumlah kemenangan bergantung pada peserta. E29 kini terendah RMSE/MAE S3 dan terbanyak dalam5poin pada run yang tersedia. Test telah dilihat berulang kali; hasil lanjutan bersifat eksploratif, bukan konfirmasi pada test baru.

## Meja belajar lebih ringan

Atas instruksi pengguna terbaru, kode lengkap/catatan hanya dimuat untuk **E29, E26, E24**, tiga RMSE test S3 terendah. E29 menjadi kode utama. Ini kriteria tampilan yang dinyatakan jelas; bukan klaim satu eksperimen unggul pada semua skenario atau semua ukuran.

Kode lama E26/E24 tetap identik dengan buku sebelumnya. Kode E29 mencakup30blok/2815baris dan143catatan, termasuk10padanan screenshot: hyperparameter, konteks/X/y, backbone, pooling, head linear/MLP, ekstraksi fitur, training, loss dan prediksi. Kode eksperimen lain tidak diembed, tetapi sumber notebook eksternal tetap tersedia pada kartu identitas.

Seluruh hasil sembilan eksperimen tetap ada. Tab **08 · Semua Excel** membaca semua sheet dari45workbook asli tanpa bergantung pada buku kode. Ada pencarian, halaman25baris, rentang8komponen vektor, inspector nilai lengkap, unduhan XLSX/ZIP asli. Tab kurva dapat difokuskan ke satu eksperimen agar skala S2 E29 yang jauh lebih besar tidak menyamarkan kurva lain.

E30 tetap identitas/sitasi dengan ruang kode/output kosong; tidak ikut ranking. Tidak menambah aplikasi HTML ketiga atau mengubah aplikasi MLP lama. Buku lengkap tetap terenkripsi dengan kunci lama. Mengurangi kode mengecilkan bagian source/DOM; ukuran paket terbesar masih berasal dari Excel vektor, sehingga tidak mengklaim seluruh unduhan menjadi kecil.
