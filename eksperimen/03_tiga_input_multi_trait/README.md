# Tiga input · empat trait

Pertanyaan, kunci jawaban, dan jawaban siswa menjadi input. Target isi, organisasi, kebahasaan, dan mekanik. Penomoran E1–E5 berlaku di kelompok ini.

[Semua kelompok](../README.md)

## Notebook

| Eksperimen | Konfigurasi | Notebook | Jalankan |
| --- | --- | --- | --- |
| E1 | Baseline E43, head bersama | [File](notebooks/AES_E1_S3_MultiTrait_3Sekolah.ipynb) | [Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex%2Faes-learning/eksperimen/03_tiga_input_multi_trait/notebooks/AES_E1_S3_MultiTrait_3Sekolah.ipynb) |
| E2 | Empat head khusus trait | [File](notebooks/AES_E2_S3_MultiTrait_TraitHeads_3Sekolah.ipynb) | [Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex%2Faes-learning/eksperimen/03_tiga_input_multi_trait/notebooks/AES_E2_S3_MultiTrait_TraitHeads_3Sekolah.ipynb) |
| E3 | E1 dengan dropout MLP 0,2 | [File](notebooks/AES_E3_S3_MultiTrait_Dropout02_3Sekolah.ipynb) | [Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex%2Faes-learning/eksperimen/03_tiga_input_multi_trait/notebooks/AES_E3_S3_MultiTrait_Dropout02_3Sekolah.ipynb) |
| E4 | E3 dengan sigmoid pada keluaran | [File](notebooks/AES_E4_S3_MultiTrait_Dropout02_Sigmoid_3Sekolah.ipynb) | [Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex%2Faes-learning/eksperimen/03_tiga_input_multi_trait/notebooks/AES_E4_S3_MultiTrait_Dropout02_Sigmoid_3Sekolah.ipynb) |
| E5 | E3 dengan attention antartait | [File](notebooks/AES_E5_S3_MultiTrait_InterTraitAttention_3Sekolah.ipynb) | [Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex%2Faes-learning/eksperimen/03_tiga_input_multi_trait/notebooks/AES_E5_S3_MultiTrait_InterTraitAttention_3Sekolah.ipynb) |

## HTML hasil dan panduan

- [Ringkasan hasil E1–E3](html/Hasil_AES_MultiTrait.html)
- [Agregat metrik](html/hasil_agregat.json)

E3 adalah acuan aktif. Workbook E4/E5 belum tersedia pada pembaruan ini. E1 pada tabel hasil sama dengan E43 multi-trait pada workbook awal.

## Lokasi file

Folder ini menjadi pintu masuk notebook dan HTML yang sudah dikelompokkan. Salinan di notebooks/experiments dan aes-learning/notebooks/experiments menyediakan jalur sumber untuk builder dan kompatibilitas tautan Colab yang sudah dibagikan.
