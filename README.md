# Automated Essay Scoring

[Konteks penelitian dan naskah skripsi terbaru](docs/skripsi/README.md)

Eksperimen dipisahkan menjadi tiga kelompok. Buka folder kelompok untuk notebook Colab, source code, dan hasilnya.

| Kelompok | Buka |
| --- | --- |
| 1 variabel · eksperimen awal | [Notebook, source dan HTML](eksperimen/01_satu_variabel/README.md) |
| 3 variabel · single-trait | [Notebook, source dan HTML](eksperimen/02_tiga_input_single_trait/README.md) |
| 3 variabel · multi-trait | [Eksperimen reset, E16–E17C, source dan progres](eksperimen/03_tiga_input_multi_trait/README.md) |

## Status eksperimen multi-trait · 24 November 2026

**Jalur aktif: E17C concat dengan dropout MLP 0,1.** Notebook E17C tersedia pada branch default ini. E17C menggunakan RoBERTa full fine-tuning, BGE-M3 frozen, proyeksi BGE 1.024→256, concat dengan RoBERTa 768 dimensi, lalu MLP bersama 256 unit. Cosine similarity tidak digunakan.

| Eksperimen | Posisi | Ringkasan |
| --- | --- | --- |
| E17C | Jalur aktif | S3 concat; dropout MLP 0,1; RoBERTa full fine-tuning; BGE-M3 frozen. Hasil training/test E17C belum tercatat di repository. |
| E17A | Pembanding terpisah | Partial fine-tuning RoBERTa: embedding dan 8 layer bawah frozen, 4 layer atas trainable. |
| E17B | Alternatif terpisah | E17A + tempered LDS khusus loss Isi. PR [#1](https://github.com/MalikHariyanto/Automated_Essay_Scoring/pull/1) masih draft; belum ada klaim full GPU run atau kenaikan akurasi. |
| E18A | Dinonaktifkan | Jalur BGE relevance/cosine tidak digunakan; pengembangan diarahkan kembali ke E17C. |

[Notebook E17C](eksperimen/03_tiga_input_multi_trait/notebooks/AES_Reset_E17C_S3_Concat_Dropout01_Native_Learnable_GPU.ipynb) · [Katalog eksperimen](eksperimen/03_tiga_input_multi_trait/README.md) · [Progres dan batas bukti](eksperimen/03_tiga_input_multi_trait/PROGRES.md) · [Acuan Referensi Kedua](docs/skripsi/Acuan_Referensi_2_AES_E17A_E17B_E17C.md)

## Hasil historis

E1–E4 reset sudah selesai training dan evaluasi. E4 terbaik secara nominal pada RMSE, tetapi selisih test terhadap E2 sangat kecil (0,000101 poin atau 0,022%); hasil satu run belum membuktikan keunggulan yang meyakinkan. E3 membantu kebahasaan/mekanik, tetapi memburuk secara gabungan.

[Empat Excel hasil historis](eksperimen/03_tiga_input_multi_trait/hasil/) · [HTML perbandingan](eksperimen/03_tiga_input_multi_trait/html/Hasil_E1_E4_Reset.html)

Excel publik mempertahankan skor dan prediksi anonim, metrik, konfigurasi, serta riwayat training. Identitas/teks siswa dan vektor besar tidak dipublikasikan; workbook asli tetap privat. Notebook tetap menghasilkan keluaran privat lengkap.

## Pengembangan dan arsip

Folder notebooks/scripts/src pada root mempertahankan jalur dependensi builder dan tautan sumber lama. Salinan yang dikelompokkan tersedia di tiga folder eksperimen. Notebook Colab mandiri tidak memerlukan folder src terpisah. [Eksperimen multi-trait sebelum reset](arsip/multitrait_sebelum_reset/) dipisahkan dari kelompok aktif. Jangan mengubah salinan source tanpa menyinkronkan sumber builder.
