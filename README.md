# Automated Essay Scoring

[Konteks penelitian dan naskah skripsi terbaru](docs/skripsi/README.md)

Eksperimen dipisahkan menjadi tiga kelompok. Buka folder kelompok untuk notebook Colab, source code, dan hasilnya.

| Kelompok | Buka |
| --- | --- |
| 1 variabel · eksperimen awal | [Notebook, source dan HTML](eksperimen/01_satu_variabel/README.md) |
| 3 variabel · single-trait | [Notebook, source dan HTML](eksperimen/02_tiga_input_single_trait/README.md) |
| 3 variabel · multi-trait | [E1–E4 reset, source, Excel hasil dan progres](eksperimen/03_tiga_input_multi_trait/README.md) |

## Progres multi-trait

E1–E4 reset sudah selesai training dan evaluasi. E4 terbaik nominal pada RMSE; perbedaannya dengan E2 sangat kecil. E3 membantu kebahasaan/mekanik, tetapi memburuk secara gabungan.

[Empat Excel hasil](eksperimen/03_tiga_input_multi_trait/hasil/) · [HTML perbandingan](eksperimen/03_tiga_input_multi_trait/html/Hasil_E1_E4_Reset.html) · [Analisis progres](eksperimen/03_tiga_input_multi_trait/PROGRES.md)

Excel publik memuat hasil run dengan prediksi anonim, metrik, konfigurasi dan riwayat training. Identitas/teks siswa dan vektor besar tidak dipublikasikan; workbook asli tetap privat. Notebook tetap menghasilkan keluaran privat lengkap.

## Pengembangan dan arsip

Folder notebooks/scripts/src pada root mempertahankan jalur dependensi builder dan tautan sumber lama. Salinan yang dikelompokkan tersedia di tiga folder eksperimen. Notebook Colab mandiri tidak memerlukan folder src terpisah. [Eksperimen multi-trait sebelum reset](arsip/multitrait_sebelum_reset/) dipisahkan dari kelompok aktif. Jangan mengubah salinan source tanpa menyinkronkan sumber builder.
