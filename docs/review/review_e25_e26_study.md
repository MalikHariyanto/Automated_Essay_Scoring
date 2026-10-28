# E25/E26 ditambahkan pada buku tiga input

Buku yang sama sekarang memuat **E21–E26, 18 model, 30 Excel dan270sheet**. Tidak membuat aplikasi HTML baru. Buku eksperimen sebelum tiga input tetap byte-identik. Seluruh angka berasal dari ZIP unggahan; tidak ada training baru pada pekerjaan HTML ini.

## Hasil S3 dan pilihan kode belajar

| Ukuran | E21 | E22 | E23 | E24 | E25 | E26 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE test ↓ | 6,7231 | 7,1943 | 6,0439 | 5,9818 | 6,1053 | **5,7435** |
| MAE test ↓ | 5,2515 | 5,8108 | 4,8576 | 4,5286 | 4,7006 | **4,4797** |
| Rerata RMSE lima fold ↓ | 6,6246 | 6,4911 | 6,4143 | 6,0148 | 6,0096 | **5,9245** |
| SD antar-fold | 1,1658 | **0,2988** | 0,7442 | 1,4999 | 1,4086 | 0,9941 |
| Paling dekat di antara enam eksperimen | 11/60 | 9/60 | **13/60** | 9/60 | 8/60 | 10/60 |
| Selisih maksimal5poin | 32/60 | 27/60 | **39/60** | 36/60 | 35/60 | 38/60 |

E26 unggul pada RMSE/MSE, MAE, Pearson, QWK dan AC2 test S3. Itu tidak berarti E26 paling dekat pada setiap jawaban. Dalam duel E24–E26, masing-masing lebih dekat pada30jawaban. Dalam duel E25–E26, E25 lebih dekat pada31jawaban dan E26 pada29jawaban. Besar kesalahan dan jumlah kemenangan mengukur hal berbeda.

Aturan default meja belajar dipertahankan: jumlah terdekat S3 terbesar; jika seri, pilih RMSE lebih rendah. Dengan enam peserta, **E23 menjadi kode utama**. Tombol tersendiri membuka **E26 sebagai RMSE terkecil**. Ini pilihan isi belajar secara deskriptif dari hasil yang sudah dilihat, bukan pemilihan model penelitian baru menggunakan test.

Jumlah terdekat lama berubah karena kandidat baru ditambahkan. Seluruh snapshot, metrik, ZIP dan prediksi E21–E24 tetap; keenam duel lama tidak berubah. Semua15duel tersedia. Untuk hasil seri pada S1/S2 atau cakupan18model, bagian kemenangan dibagi rata; jumlah bagian tetap60.

## Apa yang berubah pada arsitektur

E25/E26 sama-sama membaca `soal`, `jawaban_referensi`, `response_text` pada seluruh skenario dengan label tunggal `rater` 0–100. Format single_sequence, RoBERTa S1/S3 LR5e-6; head/proyeksi/layer baru1e-4. S1 tetap linear; S2/S3 MLP; BGE tetap frozen. Maksimal25epoch, patience5, lima fold. Seluruh skenario pada notebook dilatih baru. Angka kontrol S1/S2 pada dua ZIP baru sama dengan E21/E23/E24; kesamaan hasil tidak membuktikan identitas perangkat atau independensi replikasi.

- **E25:** attention token jawaban E24 → LayerNorm768; BGE1024→256; gate1024→32→1 menghasilkan g per jawaban. Concat aktual `[2g×RoBERTa768, 2(1−g)×BGE256]` tetap1024 sebelum MLP. Awal g=0,5 mempertahankan concat acuan.
- **E26:** RoBERTa token768 → BiGRU64 per arah → seluruh output token128 → attention128→64→1 → weighted sum128 → proyeksi768 → LayerNorm → concat BGE256 → MLP. Berbeda dari E23 yang memakai state akhir; tidak memakai gate E25.

Source keduanya dipin ke [commit bc47138](https://github.com/MalikHariyanto/Automated_Essay_Scoring/commit/bc4713872c7a7ce23f7dad8ae3ea73a2b75d9b86). Setiap code cell yang diembed cocok byte demi byte dengan notebook rujukannya. ZIP tidak menyertakan executed notebook atau metadata runtime lengkap; kecocokan byte kode yang benar-benar dieksekusi dan perangkat aktual belum dibuktikan.

## Audit fitur aktual

Bobot attention masing-masing memuat16.946baris untuk300jawaban/window; semua finite, nonnegatif dan berjumlah satu per window dalam toleransi1e-5. Pooling hasil akhir masing-masing300vektor768. Mask respons, packing dan pemetaan kembali bobot dijelaskan dengan catatan pada source yang tepat.

E25 mempunyai300gate skalar. Sheet pengali RoBERTa dan BGE diperiksa terhadap2g dan2(1−g), urutan ID sejajar. Gate final berkisar0,495721–0,496997 dengan rerata0,496190: pengalinya masih dekat1. Ini inspeksi fitur seluruh dataset, bukan300test atau bukti kausal kualitas encoder.

E26 mempunyai300ringkasan BiGRU-attention128 yang finite, terpisah dari vektor pooling768. Ringkasan tersebut bukan state akhir E23 atau skor regresi. Semua sheet baru dapat dijelajahi dan diunduh sebagai bagian Excel asli.

## Kode dan pengalaman membaca

E25 memiliki30blok/2.810baris/142catatan. E26 memiliki30blok/2.798baris/141catatan. Jumlah catatan bukan ukuran mutu model. Setiap eksperimen mempertahankan sepuluh padanan screenshot→cell/baris: hyperparameter, konteks, Dataset/X/y, pretrained, pooling, head, fitur awal, lima fold, optimizer/loss, prediksi.

Catatan serentak berada di samping ruas kode yang dijelaskan; tinggi ruas dapat bertambah agar penjelasan tidak bertabrakan. Fullscreen memenuhi viewport, tanpa heading induk. Mode semua blok dan per blok tetap tersedia dengan satu scroll kode/catatan. Warna enam eksperimen konsisten pada tabel, kurva, duel dan peringkat. Peringkat dapat diurutkan menurut jumlah terdekat, RMSE atau MAE.

## Verifikasi dan batas

Data300baris, ketiga teks, label,60IDtest dan pembagian lima fold dicocokkan antar-enam ZIP. Empat metrik test dihitung ulang dari skor mentah dalam toleransi1e-6; epoch terpilih, panjang riwayat dan refit dicocokkan dengan workbook. Kategori dibulatkan hanya untuk QWK/AC2 dan peta skor; RMSE/MAE memakai skor kontinu.

Satu run per eksperimen dan test yang telah berulang kali dilihat tidak membuktikan kestabilan lintas seed, signifikansi atau generalisasi ke soal baru. SD fold bukan SD seed. Paper mendukung konsep, bukan menjamin kombinasi ini mengungguli semua model.

ZIP/XLSX asli tetap byte-identik di buku lengkap terenkripsi. Kunci lama dipertahankan. HTML publik berisi source, catatan dan agregat; teks/ID/nilai individual serta vektor siswa tidak dipublikasikan terbuka. Bukti unit, browser, paket dan privasi tersimpan lokal pada `outputs/e25_e26_study/`. Browser diuji melalui HTTP localhost, kemudian jaringan diblokir; jalur file:// tidak diklaim diuji di lingkungan ini.

18pengujian integritas/unit,8kelompok pemeriksaan browser dan2pemeriksaan paket lulus (0gagal). Pemeriksaan browser mencocokkan1.080prediksi dengan sumber,15duel,15grafik, keenam peta, sepuluh unduhan Excel E25/E26, source/catatan dan fullscreen desktop/tablet/HP. Pengunduh menghasilkan ZIP byte-identik; kunci lama membuka buku lengkap tanpa akses jaringan.
