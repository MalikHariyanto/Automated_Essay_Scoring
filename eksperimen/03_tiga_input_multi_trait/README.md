> **Jalur aktif per 24 November 2026:** [E17C concat](https://github.com/MalikHariyanto/Automated_Essay_Scoring/blob/codex/e17c-concat-dropout01/eksperimen/03_tiga_input_multi_trait/notebooks/AES_Reset_E17C_S3_Concat_Dropout01_Native_Learnable_GPU.ipynb). S3 memakai representasi RoBERTa dan BGE-M3 melalui concat serta regresi empat skor. Cosine similarity tidak digunakan pada jalur ini. E18A telah dinonaktifkan atas permintaan pengguna. Daftar eksperimen dan hasil terdahulu di bawah merupakan catatan riwayat.

# 3 variabel · multi-trait · eksperimen reset

Input: soal, referensi, jawaban siswa. Target: isi, organisasi, kebahasaan, mekanik.

| Eksperimen | Teknologi | RMSE CV | RMSE test | Notebook | Hasil Excel |
| --- | --- | ---: | ---: | --- | --- |
| E16 | E6 dynamic gate + E10 prompt/inter-trait attention + E14 S1/S2/S3 controls | Menunggu run | Menunggu run | [File](notebooks/AES_Reset_E16_E6_E10_E14_MultiTrait.ipynb) · [Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex%2Fe16-e6-e10-e14-multitrait/eksperimen/03_tiga_input_multi_trait/notebooks/AES_Reset_E16_E6_E10_E14_MultiTrait.ipynb) | — |
| E1 | Fusion sederhana/head bersama | 0.426831 | 0.468598 | [File](notebooks/AES_Reset_E1_S3_MultiTrait.ipynb) · [Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex%2Faes-learning/eksperimen/03_tiga_input_multi_trait/notebooks/AES_Reset_E1_S3_MultiTrait.ipynb) | [Download](hasil/Hasil_AES_Reset_E1_S3_MultiTrait_Publik.xlsx) |
| E2 | Empat head independen | 0.425449 | 0.466544 | [File](notebooks/AES_Reset_E2_S3_MultiTrait_TraitHeads.ipynb) · [Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex%2Faes-learning/eksperimen/03_tiga_input_multi_trait/notebooks/AES_Reset_E2_S3_MultiTrait_TraitHeads.ipynb) | [Download](hasil/Hasil_AES_Reset_E2_S3_MultiTrait_Publik.xlsx) |
| E3 | Attention token per trait | 0.425778 | 0.472422 | [File](notebooks/AES_Reset_E3_S3_MultiTrait_TraitAttention.ipynb) · [Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex%2Faes-learning/eksperimen/03_tiga_input_multi_trait/notebooks/AES_Reset_E3_S3_MultiTrait_TraitAttention.ipynb) | [Download](hasil/Hasil_AES_Reset_E3_S3_MultiTrait_Publik.xlsx) |
| E4 | Attention antartrait | 0.425284 | 0.466443 | [File](notebooks/AES_Reset_E4_S3_MultiTrait_InterTraitAttention.ipynb) · [Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex%2Faes-learning/eksperimen/03_tiga_input_multi_trait/notebooks/AES_Reset_E4_S3_MultiTrait_InterTraitAttention.ipynb) | [Download](hasil/Hasil_AES_Reset_E4_S3_MultiTrait_Publik.xlsx) |
| E5 | E2 + MSE berbobot untuk skor langka | Menunggu run | Menunggu run | [File](notebooks/AES_Reset_E5_S3_MultiTrait_WeightedLoss.ipynb) | — |
| E6 | E2 + gate RoBERTa/BGE per trait | Menunggu run | Menunggu run | [File](notebooks/AES_Reset_E6_S3_MultiTrait_TraitGatedConcat.ipynb) · [Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex%2Faes-learning/eksperimen/03_tiga_input_multi_trait/notebooks/AES_Reset_E6_S3_MultiTrait_TraitGatedConcat.ipynb) | — |
| E7 | E2 + campuran RoBERTa/BGE tetap per trait | Menunggu run | Menunggu run | [File](notebooks/AES_Reset_E7_S3_MultiTrait_StaticTraitMix.ipynb) · [Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex%2Faes-learning/eksperimen/03_tiga_input_multi_trait/notebooks/AES_Reset_E7_S3_MultiTrait_StaticTraitMix.ipynb) | — |
| E8 | E2 + jarak BGE jawaban–soal dan jawaban–kunci untuk isi | Menunggu run | Menunggu run | [File](notebooks/AES_Reset_E8_S3_MultiTrait_BGERelevance.ipynb) · [Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex%2Faes-learning/eksperimen/03_tiga_input_multi_trait/notebooks/AES_Reset_E8_S3_MultiTrait_BGERelevance.ipynb) | — |
| E9 | Adaptasi ProTACT dengan attention soal–jawaban, attention antartrait, dan loss relasi | Menunggu run | Menunggu run | [File](notebooks/AES_Reset_E9_S3_MultiTrait_ProTACTAdaptation.ipynb) · [Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex%2Faes-learning/eksperimen/03_tiga_input_multi_trait/notebooks/AES_Reset_E9_S3_MultiTrait_ProTACTAdaptation.ipynb) | — |
| E10 | Ablasi E9 dengan bobot loss relasi 0 | Menunggu run | Menunggu run | [File](notebooks/AES_Reset_E10_S3_MultiTrait_NoRelationLoss.ipynb) · [Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex%2Faes-learning/eksperimen/03_tiga_input_multi_trait/notebooks/AES_Reset_E10_S3_MultiTrait_NoRelationLoss.ipynb) | — |
| E11 | Ablasi E10 dengan attention soal–jawaban saja | Menunggu run | Menunggu run | [File](notebooks/AES_Reset_E11_S3_MultiTrait_PromptAttentionOnly.ipynb) · [Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex%2Faes-learning/eksperimen/03_tiga_input_multi_trait/notebooks/AES_Reset_E11_S3_MultiTrait_PromptAttentionOnly.ipynb) | — |
| E12 | Ablasi E10 dengan attention antartrait saja | Menunggu run | Menunggu run | [File](notebooks/AES_Reset_E12_S3_MultiTrait_InterTraitAttentionOnly.ipynb) · [Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex%2Faes-learning/eksperimen/03_tiga_input_multi_trait/notebooks/AES_Reset_E12_S3_MultiTrait_InterTraitAttentionOnly.ipynb) | — |
| E13 | Ablasi sinyal attention soal–jawaban langsung untuk organisasi | Menunggu run | Menunggu run | [File](notebooks/AES_Reset_E13_S3_MultiTrait_OrganisasiPromptAblation.ipynb) · [Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex%2Faes-learning/eksperimen/03_tiga_input_multi_trait/notebooks/AES_Reset_E13_S3_MultiTrait_OrganisasiPromptAblation.ipynb) | — |
| E14 | S1, S2, S3; concat S3 dan MLP bersama 256 dengan resep Colab E13 | Menunggu run | Menunggu run | [File](notebooks/AES_Reset_E14_S1_S2_S3_MultiTrait.ipynb) | — |
| E15 | Gate per trait untuk S3 dengan MLP bersama 256 dan pembanding S1/S2 | Menunggu run | Menunggu run | [File](notebooks/AES_Reset_E15_S1_S2_S3_TraitGatedConcat.ipynb) · [Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex%2Faes-learning/eksperimen/03_tiga_input_multi_trait/notebooks/AES_Reset_E15_S1_S2_S3_TraitGatedConcat.ipynb) | ZIP evaluasi per skenario |

E15 mengekspor workbook 04–05 secara terpisah untuk S1, S2, dan S3. Workbook
04 memuat validasi dan test per trait. Inspeksi S3 masuk ke ZIP 01–03.
Setiap ZIP dapat dibuka sendiri dan ukurannya diverifikasi di bawah 30 MB.

[HTML hasil](html/Hasil_E1_E4_Reset.html) · [Progres dan evaluasi](PROGRES.md) · [Source code](src/) · [Semua kelompok](../README.md)

Excel publik mempertahankan skor, prediksi anonim, empat metrik evaluasi dan riwayat training dari hasil run asli. Teks/identitas siswa dan vektor besar tidak disertakan; workbook asli tetap privat. File ini bukan workbook lengkap sebelum anonimisasi. Notebook Colab tetap mengekspor workbook privat lengkap.

Notebook mandiri dapat dijalankan langsung di Colab T4. Source di src merupakan salinan helper untuk dibaca; builder pengembangan di scripts pada root mempertahankan jalur dependensi. Eksperimen multi-trait sebelum reset berada di [arsip](../../arsip/multitrait_sebelum_reset/).

E5 menghitung bobot per kategori dan trait dari training fold, dengan rasio maksimum 2
dan rerata bobot 1 per trait. Refit memakai seluruh development. Evaluasi tetap tanpa bobot.
ZIP privat E5 mengelompokkan hasil ke workbook 01–05. Bagian yang melampaui batas
dipecah menjadi workbook Excel terpisah dan setiap file diperiksa agar di bawah 30.000.000 byte.
Mulai dari 04 untuk prediksi dan evaluasi, lalu 05 untuk bobot dan riwayat training.

E6 memakai empat gate agar setiap trait dapat belajar campuran RoBERTa dan BGE sendiri.
Gate mulai dengan pengali 1, memakai learning rate 1e-4, dan mengikuti loss MSE biasa E2.
ZIP ringkas 04–05 tersedia sesudah test. ZIP lengkap 01–05 mencakup inspeksi vektor tambahan.
Workbook 04 mencatat bobot gate OOF dan test serta ringkasan per trait.
