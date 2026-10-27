# Buku tiga input — audit versi E21 dan E22

Versi terbaru juga memuat E24; lihat [audit E24 dan sorotan tabel](review_e24_study.md). Dokumen ini mencatat versi awal dua eksperimen.

`docs/belajar_Tiga_Input_Spider.html` memuat agregat aman dan dua meja belajar kode. Paket `Belajar_AES_Tiga_Input_Lengkap` menambahkan seluruh isi kedua ZIP unggahan secara terenkripsi, dengan kunci buku yang sudah ada. Buku eksperimen lama dan paket E21 terpisah tetap tersedia.

## Kesetaraan perbandingan

Seluruh 300 baris, tiga teks input, label `rater`, pembagian 240 development/60 test, dan anggota lima fold dicocokkan antara E21 dan E22. ID/label 60 jawaban test sama. Pengaturan yang dicatat di workbook sama kecuali format input. E21 memakai `single_sequence`; E22 memakai `context_pair`: soal/referensi sebagai teks A dan penanda/respons sebagai teks B. Anggaran konteks serta window jawaban dibuat sama pada source rujukan, dengan cadangan empat special token; special aktual E21 dua dan E22 empat.

Pada seluruh S1/S2/S3, X terdiri dari `soal`, `jawaban_referensi`, dan `response_text`. Label tetap satu `rater` pada skala 0–100, bukan label baru atau tiga target. S1 RoBERTa fine-tuned + linear, S2 BGE-M3 frozen + MLP, S3 fusion + MLP. LR encoder S1/S3 5e-6; proyeksi/head 1e-4. Tidak ada training baru untuk membuat buku ini.

## Hasil test tersimpan

| RMSE test, poin ↓ | E21 | E22 |
| --- | ---: | ---: |
| S1 | 8,3369 | 7,9830 |
| S2 | 7,8672 | 7,8759 |
| S3 | 6,7231 | 7,1943 |

S3 E21 memiliki RMSE/MSE, QWK dan Gwet AC2 test lebih baik. Pearson S3 E22 sedikit lebih tinggi, 0,5270 dibanding 0,5187. Masing-masing lebih dekat pada guru untuk 30 dari 60 jawaban S3. Hitungan kemenangan tidak mempertimbangkan besar seluruh error seperti RMSE; seri hitungan tidak menyatakan metrik sama.

Mean RMSE CV S3 E22 lebih rendah (6,4911 vs 6,6246) dengan SD antar-fold lebih kecil (0,2988 vs 1,1658), tetapi RMSE test E22 lebih besar. Ini bukti bahwa hasil development dan test dapat berbeda, bukan alasan memilih parameter memakai test. Satu run per format tidak membuktikan stabilitas antar-seed atau keunggulan statistik.

Empat metrik test dihitung ulang dari prediksi tersimpan, dalam toleransi 1e-6 terhadap nilai laporan. Angka laporan dipertahankan. MSE dan analisis error diturunkan dari pasangan yang sama. CV dicocokkan dengan minimum riwayat epoch dan tabel fold; ZIP tidak menyertakan pasangan individual tiap fold, sehingga CV tidak diklaim dihitung ulang dari prediksi fold. Peta rentang 10 poin adalah visualisasi; kategori QWK/AC2 tetap integer 0–100.

## Isi dan UI

- Panel **Ketepatan prediksi** memisahkan kecocokan mentah, kecocokan setelah pembulatan, toleransi ±1/±2/±5/±10, MAE dan RMSE untuk masing-masing skenario atau enam model. Pembulatan mengikuti nearest integer 0–100 dengan tie ke bawah; prediksi mentah tidak diubah.
- Kartu ringkasan menyebut ukuran dan cakupannya: CV S3 E22 lebih rendah, RMSE test S3 E21 lebih rendah, dan hitungan berpasangan seluruh skenario E22 92 vs E21 88. Angka 180 adalah 60 jawaban × tiga skenario. Pada S3 saja hitungannya seri 30–30. Persaingan enam model sekaligus memberi total bagian E21 34/60 dan E22 26/60; ukuran ini berbeda dari 180 pasangan dan tidak menyatakan E22 unggul jauh pada semua prediksi.
- Perbandingan memakai eksperimen sebagai kolom: tiga skenario, lima metrik, CV, epoch refit, dan jumlah prediksi terdekat.
- Semua 60 jawaban tersedia bersama, dengan dua prediksi per skenario atau enam model sekaligus. Nilai guru di kanan; penanda terdekat, pencarian, dialog tiga teks, buka semua teks, dan ranking tersedia.
- Riwayat E21/E22 berbagi sumbu grafik; 15 pasangan fold dan epoch pilihan tersedia. Peta skor memakai skala warna bersama.
- Kedua meja belajar mempertahankan tema Spider-Man, 16 bab, 30 blok/2699 baris kode, catatan pada baris penting, dan fullscreen per blok/semua blok. Pergantian eksperimen mempertahankan bab/tab.
- Seluruh 86 sheet dari sepuluh Excel tersedia, dengan pagination, rentang komponen, inspeksi sel dan unduhan XLSX asli. Output tahap berasal dari Excel aktual.
- Fullscreen hasil menyembunyikan header besar dan memberi ruang scroll pada isi. Navigasi keyboard, HP/tablet/desktop, dan pembacaan offline diperiksa.
- Fullscreen kode memperbesar iframe ke seluruh viewport induk, menyembunyikan heading/pemilih eksperimen dan mengunci scroll induk. Kode dan sticky note tetap dalam satu kanvas. Fallback ini juga bekerja jika Fullscreen API tidak tersedia; tombol ×/Esc memulihkan ukuran dan posisi halaman.

## Asal kode dan batas bukti

Source kedua notebook dipin pada commit `ef712b4c035d96178d9baa67caa7453ac9fd8738`:

- [E21 GPU](https://github.com/MalikHariyanto/Automated_Essay_Scoring/blob/ef712b4c035d96178d9baa67caa7453ac9fd8738/notebooks/experiments/AES_300_E21_Tiga_Input_GPU.ipynb).
- [E22 GPU context pair](https://github.com/MalikHariyanto/Automated_Essay_Scoring/blob/ef712b4c035d96178d9baa67caa7453ac9fd8738/notebooks/experiments/AES_300_E22_Tiga_Input_ContextPair_GPU.ipynb).

Semua blok kode di buku dicocokkan byte demi byte dengan source rujukan masing-masing. Notebook eksekusi/log mentah tidak disertakan dalam ZIP; pengaturan workbook cocok, tetapi kecocokan source runtime/perangkat GPU tidak diklaim terbukti. Paper primer RoBERTa, BGE-M3, GELU dan LayerNorm ditautkan sebagai sumber komponen, bukan bukti langsung bahwa resep ini unggul pada data pengguna. Hasil test yang telah dilihat dalam pengembangan diperlakukan eksploratif.

## Privasi dan pemeriksaan

HTML/snapshot/preview publik hanya menyimpan agregat, kode dan inventaris. Teks, ID, nilai individu, vektor dan ZIP asli ada dalam payload terenkripsi; kunci tidak diterbitkan. Dua ZIP dan sepuluh workbook dalam buku lengkap byte-identik dengan unggahan. Pemeriksaan prefix teks privat terhadap HTML publik tidak menemukan kecocokan.

Validasi: 13 pengujian integritas/unit lulus (0 failed/skipped/xfailed); sembilan kelompok browser meliputi 360 prediksi, semua teks, ranking, 15 kurva, peta, unduhan sepuluh XLSX, kode/notes/fullscreen, pergantian eksperimen dan viewport 390/768/1440. Dua pemeriksaan paket menguji rekonstruksi ZIP dan pembukaan buku/iframe E22 dengan kunci lama setelah jaringan diblokir. Bukti lokal di `outputs/e21_study/family_*`. Ini audit hasil tersimpan serta pengujian buku, bukan eksekusi ulang eksperimen.

Pemeriksaan pembukaan dilakukan pada HTTP localhost, lalu seluruh permintaan jaringan diblokir. Navigasi langsung `file://` ditolak kebijakan browser lingkungan pengujian (`ERR_BLOCKED_BY_ADMINISTRATOR`); jalur tersebut tidak diklaim telah diuji di lingkungan ini.

Pembaruan ketepatan/fullscreen: empat kelompok browser memeriksa delapan ukuran pada enam model, denominasi perbandingan, seluruh 360 prediksi yang tetap sama, ranking S3 seri 30–30, fullscreen dan viewport. Dua kasus tambahan memeriksa kode menutupi viewport induk pada desktop E21 dan mobile E22 dengan Fullscreen API tidak tersedia, 30 blok/2699 baris, notes dalam satu scroll, serta pemulihan lewat ×/Esc. Pemeriksaan paket juga membuka/menutup fullscreen kode setelah dekripsi.
