# Automated Essay Scoring

**Rangkaian terbaru — [S3 multi-trait reset](eksperimen/03_tiga_input_multi_trait/reset/README.md)** memakai dataset semi-final dan satu workbook keluaran lengkap. [E3 reset](eksperimen/03_tiga_input_multi_trait/reset/notebooks/AES_Reset_E3_S3_MultiTrait_TraitAttention.ipynb) melanjutkan empat head E2 dengan attention token jawaban khusus tiap trait.

Eksperimen penilaian esai Bahasa Indonesia dengan RoBERTa, BGE-M3, dan fusion S3. Pilih kelompok untuk membuka notebook Colab, HTML hasil, dan panduannya.

| Kelompok eksperimen | Input dan target | Buka |
| --- | --- | --- |
| **Dua input dan eksperimen awal** | Varian awal, jawaban saja, dan context-pair. Satu skor. | [Notebook dan HTML](eksperimen/01_dua_input/README.md) |
| **Tiga input · single-trait** | Soal + kunci jawaban + jawaban siswa → satu skor. | [Notebook dan HTML](eksperimen/02_tiga_input_single_trait/README.md) |
| **Tiga input · multi-trait** | Soal + kunci jawaban + jawaban siswa → isi, organisasi, kebahasaan, mekanik. | [Notebook dan HTML](eksperimen/03_tiga_input_multi_trait/README.md) |

## Hasil rangkaian multi-trait sebelumnya

**Acuan rangkaian sebelumnya E3** memakai dropout MLP 0,2. E4 menguji sigmoid, dan E5 menguji attention antartait. Keduanya berangkat dari E3.

| Eksperimen | Perubahan | RMSE CV ↓ | RMSE test ↓ | Notebook |
| --- | --- | --- | --- | --- |
| E1 | Baseline multi-trait dari E43 | 0,4249 | 0,4715 | [Buka](eksperimen/03_tiga_input_multi_trait/notebooks/AES_E1_S3_MultiTrait_3Sekolah.ipynb) |
| E2 | Empat head khusus trait | 0,4291 | 0,4814 | [Buka](eksperimen/03_tiga_input_multi_trait/notebooks/AES_E2_S3_MultiTrait_TraitHeads_3Sekolah.ipynb) |
| **E3** | Dropout MLP 0,2 | **0,4242** | **0,4696** | [Buka](eksperimen/03_tiga_input_multi_trait/notebooks/AES_E3_S3_MultiTrait_Dropout02_3Sekolah.ipynb) |
| E4 | E3 + sigmoid | Menunggu hasil | Menunggu hasil | [Buka](eksperimen/03_tiga_input_multi_trait/notebooks/AES_E4_S3_MultiTrait_Dropout02_Sigmoid_3Sekolah.ipynb) |
| E5 | E3 + attention antartait | Menunggu hasil | Menunggu hasil | [Buka](eksperimen/03_tiga_input_multi_trait/notebooks/AES_E5_S3_MultiTrait_InterTraitAttention_3Sekolah.ipynb) |

[HTML hasil E1–E3 per trait](eksperimen/03_tiga_input_multi_trait/html/Hasil_AES_MultiTrait.html) · [Katalog semua eksperimen](eksperimen/README.md)

Hasil E1–E3 memakai dataset dan split siswa yang sama, dengan 1.070 jawaban development dan 270 jawaban test. RMSE memakai skala rubrik asli. Selisih berasal dari satu run tiap eksperimen. E1 pada workbook awal bernama E43 multi-trait.

## Menjalankan notebook

1. Buka kelompok eksperimen dan pilih **Colab** pada baris notebook.
2. Pilih perangkat sesuai notebook, lalu jalankan cell berurutan.
3. Untuk multi-trait, hubungkan Google Drive. Dataset semi-final dipakai dari folder privat MyDrive/AES/datasets/semi_final/.

Untuk HTML, pilih **Download raw file** di GitHub lalu buka file yang tersimpan. Buku lengkap eksperimen awal dan tiga input memakai pengunduh serta kunci yang sebelumnya dibagikan.

## Struktur repository

| Folder | Isi |
| --- | --- |
| [eksperimen/01_dua_input](eksperimen/01_dua_input/README.md) | Notebook dan HTML kelompok awal |
| [eksperimen/02_tiga_input_single_trait](eksperimen/02_tiga_input_single_trait/README.md) | Notebook dan HTML tiga input, satu target |
| [eksperimen/03_tiga_input_multi_trait](eksperimen/03_tiga_input_multi_trait/README.md) | Notebook dan HTML tiga input, empat target |
| notebooks/ | Notebook sumber builder, alat bantu, dan arsip |
| aes-learning/ | Salinan untuk tautan Colab yang sudah dibagikan |
| downloads/ | Pengunduh dan bagian buku terenkripsi |
| docs/ | Metode, review, dokumentasi penelitian, serta arsip |
| scripts/ | Builder dan helper |
| src/ dan tests/ | Implementasi Python dan pemeriksaan |

Jalankan **python scripts/build_experiment_catalog.py** setelah memperbarui notebook atau HTML sumber agar katalog kelompok ikut tersinkron. Builder multi-trait menjalankan sinkronisasi ini otomatis.

Dataset siswa, workbook individual, checkpoint, dan kunci disimpan secara privat. HTML multi-trait di katalog hanya memuat metrik agregat.

[Dokumentasi](docs/README.md) · [Panduan proyek](PANDUAN_AI_PROYEK.md) · [Pindah akun](PINDAH_AKUN.md) · [Arsip halaman eksperimen sebelumnya](docs/arsip/README_sebelum_pengelompokan.md)
