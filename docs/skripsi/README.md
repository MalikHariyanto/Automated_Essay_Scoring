# Konteks penelitian AES multi-trait

[Naskah skripsi terbaru yang diserahkan penulis](PROPOSAL-MALIK-HARIYANTO-TANPA-NRMSE.docx) menjadi rujukan untuk rumusan masalah, hipotesis, dan rancangan awal penelitian. Naskah disimpan sesuai berkas kiriman. Gunakan catatan berikut saat membaca bagian metode yang masih menggambarkan rencana sebelumnya.

## Rumusan penelitian

- §1.2–1.3 menguji hipotesis bahwa gabungan RoBERTa Indonesia dan BGE-M3 memperbaiki prediksi skor esai argumentasi Bahasa Indonesia dibanding masing-masing encoder tunggal. Analisis juga membahas pola kesalahan terhadap skor guru pada isi, organisasi, kebahasaan, dan mekanik.
- §3.5–3.6 mendefinisikan S1 sebagai RoBERTa yang di-fine-tune, S2 sebagai BGE-M3 frozen, dan S3 sebagai gabungan keduanya. Rancangan S3 memproyeksikan fitur BGE 1.024→256 lalu menerapkan LayerNorm dan concat dengan 768 fitur RoBERTa. Naskah merancang MLP 256 unit, GELU, dan dropout yang sama untuk ketiga skenario.
- §3.7–3.8 merancang MSE pada target yang dinormalisasi ke 0–1, lima fold dalam data pengembangan, pemilihan hyperparameter sebelum evaluasi test, dan konfirmasi lintas seed. RMSE dan Pearson dihitung dari prediksi kontinu pada skala rubrik asli. QWK dan Gwet AC2 berbobot kuadratik memakai kategori rubrik.

## Data dan eksperimen aktif

- Naskah masih menyebut rencana 300–400 esai dari satu sekolah, satu esai per siswa, dengan teks esai sebagai masukan. Untuk E13 reset, metadata evaluasi mencatat dataset **semi-final** dengan **1.340 jawaban**, **1.070 development**, **270 test**, dan **134 kelompok**. Masukannya adalah pertanyaan, kunci jawaban, dan jawaban siswa. Gunakan metadata run untuk mendeskripsikan data eksperimen. Identitas dan cakupan sekolah perlu dikonfirmasi saat merevisi naskah.
- E13 melaporkan skala isi 1–4 dan skala organisasi, kebahasaan, serta mekanik 1–2. Gunakan rubrik dan metadata eksperimen saat membaca empat metrik. Skor guru menjadi target.
- Gunakan konfigurasi notebook dan workbook evaluasi untuk mendeskripsikan resep yang benar-benar dijalankan. Misalnya, E13 memakai satu seed pelatihan dan head per trait berukuran 64, sedangkan naskah merancang konfirmasi tiga seed dan MLP 256. [Katalog eksperimen multi-trait](../../eksperimen/03_tiga_input_multi_trait/README.md)

## Implikasi untuk pengembangan S3

Pertahankan S1 dan S2 sebagai pembanding pada split yang sama. Gunakan S3 concat sebagai pembanding untuk perubahan fusion dan hyperparameter. Pilih konfigurasi melalui data development, laporkan empat metrik per trait, dan bedah kesalahan pada skor rendah. Untuk teknologi baru, gunakan rujukan yang langsung membahas AES multi-trait, terbit dalam lima tahun terakhir, dengan status Scopus Q1–Q3 atau SINTA 1–2 yang sudah diverifikasi. Catat setiap perubahan dari rancangan naskah agar hasil bisa ditelusuri ke konfigurasi run.
