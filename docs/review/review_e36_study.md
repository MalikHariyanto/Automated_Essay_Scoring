# Analisis E36 — 3 November 2026

**E36 belum mengalahkan E30: mean RMSE CV dan RMSE test keduanya sedikit lebih tinggi.** E36 memperbaiki RMSE test dibanding E33–E35 dan E26. Pada16eksperimen, urutan RMSE test S3 terendah adalah E29, E30, lalu E36. Ini peringkat deskriptif pada test yang telah sering dilihat, bukan pemilihan model yang terkonfirmasi pada data baru.

## Resep yang benar-benar tercatat

ZIP unggahan berisi lima workbook asli,300jawaban,240development,60test. Hanya S3 dijalankan. RoBERTa fine-tuned + BiGRU + co-attention respons–referensi + attention pooling → concat proyeksi BGE frozen → MLP. Dibanding E30, encoder S3 berubah5e-6→3e-6. MLP, proyeksi BGE, BiGRU/co-attention/pooling/norm tetap1e-4. Format single_sequence, panjang token512/1024, konteks192,soal64,microbatch1/batch efektif8,5fold,maks25epoch,patience5 tercatat sesuai resep.

S1/S2 tidak dilatih, bukan hasil hilang dan bukan skor nol. Kedua cabang encoder dalam S3 tetap dipakai. Jangan menyalin hasil S1/S2 E30 sebagai hasil E36. Notebook source rujukan dipin ke349f2e82e876333bf2270d47e7eb86b59e939a55; format Python native mengikuti E35. ZIP hanya berisi Excel, tanpa notebook eksekusi/checkpoint/log perangkat, sehingga kecocokan byte source runtime dan GPU tidak dibuktikan.

## Perbandingan S3

| Ukuran | E29 | E30 | E26 | E35 | E36 |
|---|---:|---:|---:|---:|---:|
| RMSE test ↓ | 5,4293 | **5,5247** | 5,7435 | 5,9452 | 5,6738 |
| MAE test ↓ | 4,1713 | **4,0215** | 4,4797 | 4,4253 | 4,2308 |
| Mean RMSE CV ↓ | 5,7667 | **5,9491** | 5,9245 | 5,8550 | 5,9629 |
| SD antar-fold | 1,4699 | 1,0737 | 0,9941 | 0,9042 | 1,0342 |
| Pearson ↑ | 0,7496 | **0,7133** | 0,7050 | 0,6645 | 0,6957 |
| QWK ↑ | 0,6717 | **0,6655** | 0,6302 | 0,5913 | 0,6379 |
| Gwet AC2 ↑ | 0,9857 | **0,9854** | 0,9842 | 0,9832 | 0,9852 |
| Bias prediksi−guru | -1,4286 | 0,4361 | -1,3156 | -0,9135 | -0,4011 |

Terhadap E30, RMSE test E36 naik0,1491poin (sekitar2,70%),MAE naik0,2093poin (5,20%),dan mean CV naik0,0138poin (0,23%). Selisih CV kecil dan belum membuktikan keunggulan stabil salah satu resep lintas seed. SD antar-fold sedikit lebih kecil tidak sama dengan bukti kestabilan antar-seed.

Terhadap E35, RMSE test turun0,2714poin, tetapi mean CV memburuk0,1079poin. Jadi E36 tidak memperbaiki kedua ukuran sekaligus. Penurunan LR tidak otomatis meningkatkan generalisasi.

| Ketepatan pada60test | E30 | E35 | E36 |
|---|---:|---:|---:|
| Error≤1poin | 6 | 12 | 12 |
| Error≤2poin | 22 | 18 | 18 |
| Error≤5poin | 44 | 38 | 40 |
| Error≤10poin | 56 | 56 | 57 |
| Sama setelah pembulatan | 3 | 4 | 3 |

E36 memiliki3jawaban dengan error>10poin, dibanding4padaE30. Namun MAE/RMSE E30 tetap lebih rendah. Dalam duel kedekatan per jawaban,E36 kalah27–33dariE30 dan26–34dariE29;menang33–27atasE26,36–24atasE33,serta33–27atasE34/E35. Semua duel ini hanya S3 dan60pasangan, bukan180perbandingan tiga skenario.

## Fold, early stopping dan refit

| Fold | RMSE E30 | RMSE E36 | Epoch terbaik E36 | Epoch dijalankan |
|---|---:|---:|---:|---:|
| 1 | 4,5132 | 4,8090 | 13 | 18 |
| 2 | 6,1815 | 5,5934 | 11 | 16 |
| 3 | 5,9209 | 6,0928 | 18 | 23 |
| 4 | 7,4967 | 7,6143 | 7 | 12 |
| 5 | 5,6334 | 5,7051 | 20 | 25 |

E36 unggul hanya padaFold2. Fold4 tetap paling sulit. Keempat fold pertama berhenti setelah5epoch tanpa perbaikan. Fold5 mencapai batas25epoch dengan bobot terbaik epoch20; riwayat ini sendiri tidak membuktikan bahwa menambah epoch akan memperbaiki hasil. Refit13epoch adalah ceil median13/11/18/7/20, berbeda dari E30 yang memakai18epoch. Ini konsekuensi pemilihan melalui CV, bukan perubahan hyperparameter refit secara manual.

Riwayat yang diekspor menunjukkan loss training turun pada seluruh fold: epoch pertama sekitar0,162–0,435 menjadi sekitar0,00224–0,00419pada epoch pilihan. Loss akhir dapat turun lagi sementara RMSE validasi tidak membaik. Jadi hasil yang kalah dari E30 bukan alasan menyimpulkan model tidak belajar. Yang belum membaik adalah kemampuan generalisasi; penyebab tunggal tidak dapat dipastikan dari satu seed dan Excel tanpa checkpoint.

## Audit dan pembaruan buku

Metrik test dihitung ulang dari60pasangan dengan deviasi maksimum1,70e-8terhadap angka laporan. Teks, label, anggota240/60 dan seluruh5fold dicocokkan dengan keluarga lama. Epoch pilihan diverifikasi minimum RMSE riwayat; bobot attention finite/nonnegatif/jumlah1, posisi respons/referensi cocok token asli, serta ringkasan128danpooling768 lengkap300jawaban. Top1 hanya ringkasan ekspor, bukan matriks alignment penuh. Audit tidak melatih model baru.

Buku yang sama kini memuat16eksperimen,46model aktual,80Excel/748sheet,120duel dan2760prediksi individual. Seluruh15snapshot/audit lama dan105duel lama dipertahankan. Buku kode E29/E30 byte-identik; E36 menggantikan E26 pada slot ketiga sesuai aturan tiga RMSE test S3 terendah. Hasil/Excel/source E26 tetap tersedia; tidak dihapus. Kode E36 lengkap89cell native dengan anotasi per bagian, tanpa magic/worker/pemeriksaan cosine. Paket lokal mempertahankan nama HTML/ZIP/pengunduh dan kunci lama.

Paket lengkap berisi jawaban serta nilai individual. Publikasi paket terenkripsi sebelumnya ditolak review persetujuan otomatis; pembaruan paket GitHub tidak boleh diklaim sebelum publikasinya diotorisasi dan benar-benar terverifikasi. Source/agregat publik terpisah dari payload privat.

## Arah berikutnya

Pertahankan E30 sebagai acuan keluarga co-attention ini; E29 tetap terbaik pada RMSE test seluruh keluarga. E36 tidak mendukung menurunkan LR encoder ke3e-6sebagai peningkatan yang sudah terbukti. Langkah berikut yang informatif adalah mengukur ulang acuan pada seed tambahan dengan fold tetap sebelum menumpuk perubahan metode. Laporkan semua hasil dan gunakan development untuk keputusan; pengujian konfirmasi memerlukan data yang belum dipakai memilih resep.
