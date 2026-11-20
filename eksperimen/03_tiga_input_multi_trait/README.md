# 3 variabel · multi-trait · E1–E4 reset

Input: soal, referensi, jawaban siswa. Target: isi, organisasi, kebahasaan, mekanik. Folder aktif ini hanya memuat empat eksperimen reset terakhir.

| Eksperimen | Teknologi | RMSE CV | RMSE test | Notebook | Hasil Excel |
| --- | --- | ---: | ---: | --- | --- |
| E1 | Fusion sederhana/head bersama | 0.426831 | 0.468598 | [File](notebooks/AES_Reset_E1_S3_MultiTrait.ipynb) · [Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex%2Faes-learning/eksperimen/03_tiga_input_multi_trait/notebooks/AES_Reset_E1_S3_MultiTrait.ipynb) | [Download](hasil/Hasil_AES_Reset_E1_S3_MultiTrait_Publik.xlsx) |
| E2 | Empat head independen | 0.425449 | 0.466544 | [File](notebooks/AES_Reset_E2_S3_MultiTrait_TraitHeads.ipynb) · [Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex%2Faes-learning/eksperimen/03_tiga_input_multi_trait/notebooks/AES_Reset_E2_S3_MultiTrait_TraitHeads.ipynb) | [Download](hasil/Hasil_AES_Reset_E2_S3_MultiTrait_Publik.xlsx) |
| E3 | Attention token per trait | 0.425778 | 0.472422 | [File](notebooks/AES_Reset_E3_S3_MultiTrait_TraitAttention.ipynb) · [Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex%2Faes-learning/eksperimen/03_tiga_input_multi_trait/notebooks/AES_Reset_E3_S3_MultiTrait_TraitAttention.ipynb) | [Download](hasil/Hasil_AES_Reset_E3_S3_MultiTrait_Publik.xlsx) |
| E4 | Attention antartrait | 0.425284 | 0.466443 | [File](notebooks/AES_Reset_E4_S3_MultiTrait_InterTraitAttention.ipynb) · [Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex%2Faes-learning/eksperimen/03_tiga_input_multi_trait/notebooks/AES_Reset_E4_S3_MultiTrait_InterTraitAttention.ipynb) | [Download](hasil/Hasil_AES_Reset_E4_S3_MultiTrait_Publik.xlsx) |

[HTML hasil](html/Hasil_E1_E4_Reset.html) · [Progres dan evaluasi](PROGRES.md) · [Source code](src/) · [Semua kelompok](../README.md)

Excel publik mempertahankan skor, prediksi anonim, empat metrik evaluasi dan riwayat training dari hasil run asli. Teks/identitas siswa dan vektor besar tidak disertakan; workbook asli tetap privat. File ini bukan workbook lengkap sebelum anonimisasi. Notebook Colab tetap mengekspor workbook privat lengkap.

Notebook mandiri dapat dijalankan langsung di Colab T4. Source di src merupakan salinan helper untuk dibaca; builder pengembangan di scripts pada root mempertahankan jalur dependensi. Eksperimen multi-trait sebelum reset berada di [arsip](../../arsip/multitrait_sebelum_reset/).
