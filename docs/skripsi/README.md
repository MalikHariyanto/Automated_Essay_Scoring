# Konteks penelitian AES multi-trait

[Naskah skripsi terbaru yang diserahkan penulis](PROPOSAL-MALIK-HARIYANTO-TANPA-NRMSE.docx) menjadi rujukan untuk rumusan masalah, hipotesis, dan rancangan awal penelitian. Naskah disimpan sesuai berkas kiriman. Gunakan catatan berikut saat membaca bagian metode yang masih menggambarkan rencana sebelumnya.

## Rumusan penelitian

- §1.2–1.3 menguji hipotesis bahwa gabungan RoBERTa Indonesia dan BGE-M3 memperbaiki prediksi skor esai argumentasi Bahasa Indonesia dibanding masing-masing encoder tunggal. Analisis juga membahas pola kesalahan terhadap skor guru pada isi, organisasi, kebahasaan, dan mekanik.
- §3.5–3.6 mendefinisikan S1 sebagai RoBERTa yang di-fine-tune, S2 sebagai BGE-M3 frozen, dan S3 sebagai gabungan keduanya. Rancangan S3 memproyeksikan fitur BGE 1.024→256 lalu menerapkan LayerNorm dan concat dengan 768 fitur RoBERTa. Naskah merancang MLP 256 unit, GELU, dan dropout yang sama untuk ketiga skenario.
- §3.7–3.8 merancang MSE pada target yang dinormalisasi ke 0–1, lima fold dalam data pengembangan, pemilihan hyperparameter sebelum evaluasi test, dan konfirmasi lintas seed. RMSE dan Pearson dihitung dari prediksi kontinu pada skala rubrik asli. QWK dan Gwet AC2 berbobot kuadratik memakai kategori rubrik.

## Data dan eksperimen

- Naskah masih menyebut rencana 300–400 esai dari satu sekolah, satu esai per siswa, dengan teks esai sebagai masukan. Untuk E13 reset, metadata evaluasi mencatat dataset semi-final dengan 1.340 jawaban, 1.070 development, 270 test, dan 134 kelompok. Masukannya adalah pertanyaan, kunci jawaban, dan jawaban siswa. Gunakan metadata run untuk mendeskripsikan data eksperimen. Identitas dan cakupan sekolah perlu dikonfirmasi saat merevisi naskah.
- E13 melaporkan skala isi 1–4 dan skala organisasi, kebahasaan, serta mekanik 1–2. Gunakan rubrik dan metadata eksperimen saat membaca empat metrik. Skor guru menjadi target.
- Gunakan konfigurasi notebook dan workbook evaluasi untuk mendeskripsikan resep yang benar-benar dijalankan. Misalnya, E13 memakai satu seed pelatihan dan head per trait berukuran 64, sedangkan naskah merancang konfirmasi tiga seed dan MLP 256. [Katalog eksperimen multi-trait](../../eksperimen/03_tiga_input_multi_trait/README.md)

## Resep historis per 24 November 2026

- **E17C adalah pembanding historis.** S3 memakai RoBERTa 768 dimensi full fine-tuning dan BGE-M3 1.024 dimensi frozen. Fitur BGE diproyeksikan ke 256 dimensi, lalu di-concat dengan RoBERTa menjadi 1.024 dimensi. Head bersama memakai Linear 256, GELU, dropout 0,1, dan empat output. Kontrol E14 S3 menggunakan dropout 0,2. E17C tidak memakai cosine similarity.
- **E17-A** merupakan eksperimen partial fine-tuning: embedding dan delapan layer bawah RoBERTa frozen, empat layer atas trainable.
- **E17-B** mempertahankan partial fine-tuning E17-A dan menguji tempered LDS khusus loss Isi. PR #1 masih draft; belum ada klaim full GPU training atau peningkatan akurasi.
- **E18A** dinonaktifkan. Jalur BGE relevance/cosine tidak digunakan untuk eksperimen aktif.
- Status notebook/kode tidak sama dengan hasil. Hasil E17C perlu diverifikasi melalui training GPU, CV, refit, test, dan ekspor workbook/ZIP sebelum dilaporkan.

## Acuan eksperimen terbaru

[Dashboard progres](progres/Progress_AES_MultiTrait.html) · [Pratinjau flowchart](progres/Progress_AES_MultiTrait.png)

E21-B menjadi acuan test S3. E23 seed 43 adalah kandidat bobot loss isi 2× yang masih berada di tahap development. Status konfirmasi seed 44 dan konfigurasi efektif ada di [register bukti](Register_Bukti_AES_MultiTrait.xlsx).

## Implikasi untuk pengembangan S3

Konfigurasi Colab seri reset menjadi acuan eksperimen. E14 mengikuti resep E13, termasuk maksimal 25 epoch, patience 5, dan satu refit per skenario. Tabel hyperparameter di naskah yang masih berkembang digunakan untuk konteks penelitian. Catatan LR A/B di workbook brainstorming mendokumentasikan rancangan E14 sebelum koreksi ini.

Pertahankan S1 dan S2 sebagai pembanding pada split yang sama. Gunakan S3 concat sebagai pembanding untuk perubahan fusion dan hyperparameter. Pilih konfigurasi melalui data development, laporkan empat metrik per trait, dan bedah kesalahan pada skor rendah. Untuk teknologi baru, gunakan rujukan yang langsung membahas AES multi-trait, terbit dalam lima tahun terakhir, dengan status Scopus Q1–Q3 atau SINTA 1–2 yang sudah diverifikasi. Catat setiap perubahan dari rancangan naskah agar hasil bisa ditelusuri ke konfigurasi run.

## Acuan referensi

- [Register bukti AES multi-trait](Register_Bukti_AES_MultiTrait.xlsx) menghubungkan paper, bukti indeks, angka lokal, versi notebook, dan keluaran terverifikasi. [Arsip checkpoint agregat](../../eksperimen/03_tiga_input_multi_trait/output/checkpoints/README.md)

- [Acuan Referensi Pertama: Brainstorming AES Multi-Trait](Brainstorming_Referensi_AES_MultiTrait.xlsx)
- [Acuan Referensi Kedua: evaluasi E17-A/E17-B/E17C, status E18A, roadmap, dan daftar pustaka](Acuan_Referensi_2_AES_E17A_E17B_E17C.md)
