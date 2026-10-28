# Review hasil double frozen + MLP

> Dokumen tahap terdahulu. Angka dan keputusan dibaca dalam konteks versi ini; untuk kondisi aktif gunakan [status proyek](../../STATUS_PROYEK.md) dan [panduan AI](../../PANDUAN_AI_PROYEK.md).

ZIP hasil terbaru memuat lima workbook, 41 sheet, 15 fold dan 60 jawaban test. Review membaca hasil tersimpan; tidak menjalankan training atau memilih parameter dari test.

## Kesimpulan

Pada run ini, double frozen belum memperbaiki S3 dibanding MLP acuan. RMSE test S3 bertambah dari 6,7085 menjadi 7,8611; mean RMSE CV juga bertambah dari 6,0516 menjadi 6,8363. Pearson/QWK/AC2 test S3 lebih rendah. S1 MLP acuan tetap RMSE test terkecil dari versi hasil yang tersedia.

| Skenario | RMSE test MLP acuan | RMSE test run baru | Perubahan error, poin | RMSE CV run baru · mean ± SD |
| --- | ---: | ---: | ---: | ---: |
| S1 | 6,3805 | 6,8829 | 0,5024 | 7,7850 ± 1,4802 |
| S2 | 6,9481 | 6,8435 | -0,1046 | 6,4335 ± 0,6036 |
| S3 | 6,7085 | 7,8611 | 1,1525 | 6,8363 ± 1,1352 |

RMSE lebih kecil berarti error lebih kecil. Di run baru, S2 sedikit lebih kecil daripada S1; selisih sekitar 0,0394 poin belum membuktikan keunggulan yang stabil. S2 juga mempunyai sebaran RMSE lima fold paling kecil pada run baru. SD menggambarkan sebaran fold, bukan confidence interval atau bukti signifikansi.

## Empat metrik test run baru

| Skenario | RMSE ↓ | Pearson | QWK | Gwet AC2 |
| --- | ---: | ---: | ---: | ---: |
| S1 | 6,8829 | 0,5238 | 0,4750 | 0,9776 |
| S2 | 6,8435 | 0,5523 | 0,5046 | 0,9788 |
| S3 | 7,8611 | 0,3992 | 0,3831 | 0,9706 |

AC2 sekitar 0,97–0,98 bukan persentase prediksi tepat. Pada kategori 0–100, pembobotan kuadratik dapat tetap tinggi ketika selisih skor beberapa poin. Karena itu RMSE, korelasi dan kedua agreement dibaca bersama.

Loss refit S3 terakhir 0,002006 pada target 0–1 menunjukkan pemetaan head sudah belajar pada development. Angka itu bukan RMSE test dan tidak membuktikan generalisasi lebih baik. Hipotesis yang dapat diuji ialah kesesuaian fitur pretrained untuk penilaian esai, kapasitas/regularisasi head, dan variasi antar-run; tabel ini belum membuktikan penyebab tunggal.

## Proses yang benar-benar dibandingkan

S1 tetap RoBERTa fine-tuned dengan head linear. S2 tetap BGE-M3 frozen dengan MLP. Perubahan arsitektur utama S3 adalah membekukan encoder RoBERTa juga. LayerNorm tambahan RoBERTa, proyeksi BGE 1024→256 + LayerNorm, concat 768+256=1024 dan head MLP 1024→256→1 tetap dilatih. Frozen tidak berarti semua parameter S3 berhenti belajar.

Seluruh 300 × 768 komponen RoBERTa mentah S3 pada workbook baru identik dengan RoBERTa sebelum latihan: selisih maksimum 0. Sheet RoBERTa S3 Setelah LayerNorm menyimpan transformasi tambahan yang dilatih; bukan perubahan bobot encoder. Fusion berada pada tingkat jawaban dan tidak menciptakan vektor fusion per kata.

Epoch refit berasal dari ceil median epoch terbaik lima fold: S1 18, S2 23, S3 11. Batas CV tetap 25 epoch dan patience 5. Early stopping tidak diterapkan pada refit yang epoch-nya sudah dipilih sebelumnya.

## Kekuatan dan batas bukti

Dataset 300 baris, pembagian 240/60, 1.200 baris keanggotaan fold, ID dan nilai guru 60 test cocok dengan MLP acuan. Keempat metrik dihitung ulang dari pasangan workbook: selisih maksimum di bawah 1e-6; QWK/AC2 cocok pada presisi numerik. Perbedaan sangat kecil RMSE/Pearson konsisten dengan presisi prediksi yang tersimpan, sehingga nilai asli tabel metrik tetap dipertahankan.

Arsitektur S1/S2 tetap tetapi angkanya juga bergeser. Rincian perangkat/versi numerik dan pengulangan seed belum lengkap dalam workbook ini, sehingga hasil antar-run tidak boleh dianggap efek pembekuan semata. Hasil test sebelumnya sudah terlihat selama pengembangan; perbandingan lanjutan bersifat eksploratif. Tidak ada uji signifikansi atau klaim superioritas universal.

Kode di HTML untuk E3 merujuk notebook manual commit `0c79d655898f967e855922be3c5fad1865efb1aa`; penjelasan metode didukung tabel Model yang Dibandingkan dan pemeriksaan vektor. Notebook eksekusi double frozen belum dikirim, sehingga runtime source tidak diklaim identik byte demi byte.

## Langkah yang masuk akal

Pertahankan MLP acuan sebagai hasil presentasi, dan tampilkan double frozen sebagai percobaan yang hasil S3-nya belum membaik. Untuk penelitian lanjutan, pilih resep melalui CV dan ulangi beberapa seed. Ridge pada fitur frozen dapat menguji pemetaan yang lebih sederhana sebagai pembanding MLP, dengan scaler dipasang hanya pada training fold. Jangan menyatakan metode itu pasti memperkecil RMSE sebelum pengujian.

Semua skor individual, teks dan vektor tetap di HTML/ZIP pribadi. Halaman publik memuat agregat dan kode rujukan. [Buka buku belajar](../belajar_MLP_Spider.html).
