# Analisis E35 — 2 November 2026

**E35 membaik pada mean RMSE CV dibanding E30/E33/E34, tetapi belum mengungguli E30 pada test.** Pada15eksperimen, mean CV S3 E35 berada di urutan kedua setelah E29. RMSE test E35 lebih rendah sedikit dari E34, sementara MAE dan sebagian toleransi justru memburuk. Hasil satu seed ini belum menunjukkan pemenang universal.

## Resep dan bukti yang tersedia

ZIP unggahan berisi lima workbook asli,300jawaban,240development,60test. E35 memakai RoBERTa fine-tuned + BiGRU + co-attention respons–referensi + attention pooling,concat proyeksi BGE frozen,lalu MLP. Hanya satu parameter training tercatat yang berubah dari E30: LR kelompok pooling/norm S3 1e-4→5e-5;MLP S3 tetap1e-4,proyeksi BGE1e-4,encoder5e-6. Kelompok LR mencakup sequence_pool.* dan roberta_norm.*,bukan satu layer individual.

Notebook E35 native menggunakan Python biasa; fungsi/kolom inspeksi cosine dihapus. Workbook Kata Berulang kini8kolom. Penataan notebook bukan tambahan metode scoring. Source rujukan dipin pada699aaa3be612400ccf264bd8a8b96984263c8c8c. ZIP tidak memuat notebook eksekusi,checkpoint atau bukti perangkat lengkap;identitas byte source runtime/GPU tidak diverifikasi. Audit ini tidak melatih model baru.

## Perbandingan S3

| Ukuran | E29 | E30 | E33 | E34 | E35 |
|---|---:|---:|---:|---:|---:|
| Mean RMSE CV ↓ | 5,7667 | 5,9491 | 5,9118 | 5,8931 | 5,8550 |
| SD antar-fold | 1,4699 | 1,0737 | 1,0356 | 1,1105 | 0,9042 |
| RMSE test ↓ | 5,4293 | 5,5247 | 6,0388 | 5,9610 | 5,9452 |
| MAE test ↓ | 4,1713 | 4,0215 | 4,4769 | 4,4007 | 4,4253 |
| Pearson ↑ | 0,7496 | 0,7133 | 0,6537 | 0,6591 | 0,6645 |
| QWK ↑ | 0,6717 | 0,6655 | 0,5780 | 0,5922 | 0,5913 |
| Gwet AC2 ↑ | 0,9857 | 0,9854 | 0,9826 | 0,9833 | 0,9832 |
| Bias prediksi−guru | -1,4286 | 0,4361 | -1,0356 | -0,7507 | -0,9135 |

Terhadap E30, mean RMSE CV turun0,0941poin (sekitar1,58%),tetapi RMSE test naik0,4205poin (sekitar7,61%) dan MAE naik0,4038poin (sekitar10,04%). Terhadap E34, mean CV turun0,0381poin dan RMSE test turun0,0158poin,sedangkan MAE naik0,0247poin. Jangan menyebut perbaikan kecil RMSE tersebut mengalahkan E34 pada semua ukuran.

SD fold E35 0,9042 lebih kecil dibanding E30 1,0737,E33 1,0356,E34 1,1105. Ini sebaran antar-fold yang lebih kecil pada run tersebut,bukan bukti kestabilan antar-seed. Refit S3 E35 memakai11epoch,berbeda dari E30 18epoch;E33/E34 juga11epoch. Jumlah refit mengikuti median epoch terpilih CV,bukan dihentikan karena melihat hasil test.

| Ketepatan S3 pada60test | E30 | E33 | E34 | E35 |
|---|---:|---:|---:|---:|
| Selisih≤1poin | 6 | 14 | 13 | 12 |
| Selisih≤2poin | 22 | 20 | 19 | 18 |
| Selisih≤5poin | 44 | 37 | 39 | 38 |
| Selisih≤10poin | 56 | 55 | 54 | 56 |
| Sama setelah pembulatan | 3 | 5 | 6 | 4 |

E35 mempunyai4jawaban dengan error>10poin,lebih sedikit dari E34(6). E35 menang kedekatan per jawaban atas E34 dan E33 masing-masing32–28;melawan E30 hanya29–31. Banyak kemenangan tidak memperhitungkan besarnya error seperti RMSE/MAE. Tidak ada prediksi mentah yang persis sama dengan guru;4hasil sama setelah pembulatan merupakan ukuran kategori yang terpisah.

## Tiap fold dan early stopping

| Fold S3 | E30 RMSE | E34 RMSE | E35 RMSE | Epoch terbaik E35 | Epoch dijalankan E35 |
|---|---:|---:|---:|---:|---:|
| 1 | 4,5132 | 4,7434 | 4,8310 | 15 | 20 |
| 2 | 6,1815 | 5,4794 | 5,6006 | 11 | 16 |
| 3 | 5,9209 | 6,0640 | 5,9643 | 18 | 23 |
| 4 | 7,4967 | 7,6943 | 7,2946 | 7 | 12 |
| 5 | 5,6334 | 5,4846 | 5,5847 | 10 | 15 |

E35 menang RMSE3/5fold dari E30 (Fold2,4,5),tetapi hanya2/5fold dari E34 (Fold3,4). Fold4 masih paling sulit,namun membaik dari7,4967(E30)/7,6943(E34) menjadi7,2946. Seluruh fold S3 berhenti setelah5epoch tanpa perbaikan;tidak ada yang melewati25epoch. Epoch terbaik diverifikasi minimum riwayat RMSE validasi,dan refit11sesuai ceil median15/11/18/7/10.

## Kontrol dan audit ekspor

Prediksi S1/S2 identik per jawaban dengan E30/E33/E34. RMSE S1 tetap8,3369;S2 tetap7,8672. Ini mendukung bahwa perubahan hasil utama berada pada S3 dan tidak memberi bukti bahwa notebook native memperbaiki/memperburuk kontrol. Kesamaan hasil tidak membuktikan independensi run atau perangkat aktual.

Metrik RMSE,Pearson,QWK dan AC2 test dihitung ulang dari60pasangan guru/prediksi;cocok dalam toleransi1e-6,deviasi maksimal2,40e-7. Audit mencocokkan teks/label/split/fold dengan keluarga lama,aturan pembulatan,epoch/riwayat/refit,cakupan token,vektor finite,dan posisi token bobot attention. Bobot respons16.946baris,referensi17.600baris,ringkasan matching128danpooling768lengkap pada300jawaban. Audit top1 hanya ringkasan ekspor;matriks alignment penuh tidak tersedia untuk menghitung ulang maksimum.

Buku diperluas menjadi15eksperimen/45model,105duel dan75Excel/704sheet. Seluruh14hasil lama,91duel lama,dan tiga buku kode E29/E30/E26 dipertahankan. Kode meja belajar tetap tiga RMSE test S3 terendah;E35 urutan kelima pada RMSE test,belum menggantikannya. Semua hasil/Excel E35 tersedia melalui tab hasil dan semua Excel. Nama pengunduh/ZIP/HTML buku serta kunci lama dipertahankan. Publikasi GitHub paket terenkripsi tidak boleh diklaim sebelum bagian paket benar-benar terunggah;blok review sebelumnya tetap dicatat terpisah.

## Implikasi eksperimen berikutnya

E35 melengkapi empat kombinasi LR MLP/layer E30,E33,E34,E35. Di development satu seed,E35 paling menjanjikan di antara keempatnya;di test historis E30 tetap lebih baik. E29 masih memiliki mean CV dan RMSE test terendah seluruh keluarga. Langkah paling informatif adalah membandingkan E30 dan E35 pada seed123dan2026,dengan data/fold tetap,lalu menilai mean/SD antar-seed melalui development. Jangan menambah beberapa layer/loss/LR sekaligus atau memilih seed dari test yang telah sering dilihat. Evaluasi akhir yang kuat memerlukan data yang tidak dipakai memilih resep.
