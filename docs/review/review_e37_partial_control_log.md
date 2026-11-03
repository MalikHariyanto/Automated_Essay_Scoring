# Analisis log parsial E37 paired — 3 November2026

**Log belum menguji kandidat7e-6.** Header awal menyatakan `CV1/6,E30_control,encoder LR5e-6,seed42`; setelah lima fold selesai berlanjut ke `CV2/6,E30_control,seed142`. Bagian kedua berhenti pada epoch10Fold1. Tidak ada hasil test atau kandidat E37 dalam kutipan ini. Jangan mencatatnya sebagai eksperimen kandidat selesai atau memasukkannya ke ranking HTML.

| Fold kontrol seed42 | Epoch dipilih | Epoch dijalankan | RMSE validasi |
|---|---:|---:|---:|
|1|22|25|4,5132|
|2|7|12|6,1815|
|3|18|23|5,9209|
|4|7|12|7,4967|
|5|20|25|5,6334|

Rerata dari angka log yang dibulatkan5,94914; SD sampel1,07370. Keduanya sama dengan ringkasan E30 lama pada presisi laporan.97pasangan loss training/RMSE validasi per epoch cocok dengan histori E30 seed42yang tersimpan, pada presisi log6/4desimal. Ini mendukung bahwa run tersebut mengulang kontrol yang direncanakan; tidak membuktikan byte notebook/runtime atau identitas perangkat. Log tidak menyatakan pemakaian cache untuk kelima fold itu.

Loss training turun, misalnya Fold1 dari0,259819menjadi0,001214. Namun RMSE validasi terbaik berada pada epoch22(4,5132),sedangkan epoch25memburuk menjadi4,8481meski loss training lebih kecil. Jadi pemilihan checkpoint berdasarkan RMSE validasi bekerja dan MSE training paling kecil tidak otomatis memberi generalisasi terbaik. Fold4tetap paling sulit. Early stopping menghentikan Fold2/3/4setelah5epoch tanpa perbaikan; Fold1/5mencapai batas25. Median epoch pilihan kontrol adalah18; ini bukan epoch yang boleh dipaksakan pada kandidat baru.

MSE loss memakai target0–1dan RMSE memakai poin0–100 pada data berbeda. Untuk pasangan prediksi yang sama, RMSE4,5132poin setara MSE≈20,369poin² atau≈0,0020369pada skala0–1; angka ini bukan loss training0,001503di epoch22.

Seed142masih parsial: terbaik sementara RMSE5,2412padaepoch6Fold1; epoch10RMSE8,4806dengan4/5tanpa perbaikan. Belum sah mengambil mean5fold atau hasil akhir seed ini.

Atas koreksi pengguna, desain paired dibatalkan. [E37 aktif](../eksperimen/eksperimen_e37_single_lr_native_gpu.md) langsung7e-6,S3,5fold CV sekali,25epoch,patience5,1refit/1test. Batas publikasi data privat tetap berlaku; log unggahan tidak di-commit. SHA log `bfa9007c731baccbd437af831f6db85abab745b6f25c22b3baeda73f173a43c0`; hasil parsing disimpan privat `outputs/e37_single/log_analysis.json`.
