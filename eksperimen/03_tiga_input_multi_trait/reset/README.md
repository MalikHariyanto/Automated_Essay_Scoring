# Rangkaian reset · S3 multi-trait

| Eksperimen | Head penilaian | Notebook | Jalankan |
| --- | --- | --- | --- |
| E1 reset | MLP bersama 1024→256→4 | [File](notebooks/AES_Reset_E1_S3_MultiTrait.ipynb) | [Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex%2Faes-learning/eksperimen/03_tiga_input_multi_trait/reset/notebooks/AES_Reset_E1_S3_MultiTrait.ipynb) |
| E2 reset | Empat MLP 1024→64→1 | [File](notebooks/AES_Reset_E2_S3_MultiTrait_TraitHeads.ipynb) | [Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex%2Faes-learning/eksperimen/03_tiga_input_multi_trait/reset/notebooks/AES_Reset_E2_S3_MultiTrait_TraitHeads.ipynb) |

Keduanya memakai fusion sederhana, dataset semi-final, split siswa yang sama, dan dropout0,2. E2 adalah adaptasi ide representasi khusus trait dari [CTS](https://github.com/robert1ridley/cross-prompt-trait-scoring/blob/HEAD/models/CTS.py). Dimensi64 dipilih untuk menjaga anggaran parameter head mendekati E1.

E1 dan E2 dapat dijalankan di runtime GPU Colab terpisah. Hasil E2 diunduh sebagai Hasil_AES_Reset_E2_S3_MultiTrait.xlsx, dengan cakupan sheet yang sama seperti E1. Setiap konfigurasi mempunyai folder run dan checkpoint sendiri. Hasil penelitian keduanya menunggu run GPU pada data asli.

## E1 · Baseline multi-trait reset

Rangkaian ini dimulai kembali dari E1 dengan fusion sederhana. RoBERTa memakai CLS layer terakhir, BGE-M3 frozen diproyeksikan ke 256 dimensi, concat1024 masuk MLP256 dan menghasilkan empat skor.

| Bagian | Rancangan |
| --- | --- |
| Input | Pertanyaan, kunci jawaban, jawaban siswa |
| Target | Isi 1–4, organisasi 1–2, kebahasaan 1–2, mekanik 1–2 |
| RoBERTa | Fine-tuned, LR 5e-6, mean CLS antarwindow |
| BGE | Frozen, proyeksi1024→256 + LayerNorm |
| Head | Concat1024→256→GELU→Dropout0,2→4 |
| Optimizer | AdamW, weight decay0,01, warmup10%, linear scheduler |
| Batch dan training | Micro1/efektif8, maksimal25epoch, patience5 |
| Evaluasi | Lima fold development, refit median epoch, test sekali |
| Seleksi epoch | RMSE skala asli |
| Hasil penelitian | Menunggu run GPU pada data asli |

[Notebook](notebooks/AES_Reset_E1_S3_MultiTrait.ipynb) · [Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex%2Faes-learning/eksperimen/03_tiga_input_multi_trait/reset/notebooks/AES_Reset_E1_S3_MultiTrait.ipynb)

## Satu file Excel

Hasil diunduh sebagai Hasil_AES_Reset_E1_S3_MultiTrait.xlsx. Workbook memuat 57 sheet pada uji teknis, dengan penambahan bagian sheet otomatis jika melewati kapasitas Excel.

- Data asli dan hasil cleaning, empat target, kelompok development/test, serta anggota lima fold.
- Token, ID, mask, window, dan cakupan tiga input untuk kedua encoder.
- Vektor RoBERTa sebelum training dan BGE frozen untuk seluruh jawaban.
- Vektor token awal dan kontekstual, kata, kalimat, serta kata berulang untuk satu contoh jawaban development.
- Fitur final RoBERTa, proyeksi BGE, dan concat untuk seluruh jawaban.
- Inspeksi token/kata/kalimat pada checkpoint RoBERTa final untuk contoh development yang sama.
- Riwayat training, konfigurasi, arsitektur, RMSE CV, OOF, dan hasil test per trait.

Fitur seluruh jawaban adalah inspeksi checkpoint refit. Hasil penelitian memakai prediksi OOF dan test tersendiri. Workbook memuat teks siswa dan disimpan secara privat.

Dataset dibaca dari Drive semi-final yang sama. Pilih GPU, jalankan dari atas, lalu hubungkan Drive. Rangkaian E1–E5 sebelumnya dapat dibuka melalui [katalog historis](../README.md).
