> **Acuan test S3 per 28 November 2026:** [E21-B](output/checkpoints/E21.json). [E23 seed 43](output/checkpoints/E23.json) adalah kandidat pembobotan isi pada development. [E23 seed 44](notebooks/AES_Reset_E23_Konfirmasi_Seed44.ipynb) sudah selesai, dengan deteksi isi 2/17 pada kasus yang sama.

# 3 variabel · multi-trait · eksperimen reset

Input: soal, referensi/kunci jawaban, jawaban siswa. Target: isi, organisasi, kebahasaan, mekanik.

## Hasil historis E1–E4

| Eksperimen | Teknologi | RMSE CV | RMSE test | Notebook | Hasil Excel |
| --- | --- | ---: | ---: | --- | --- |
| E1 | Fusion sederhana/head bersama | 0.426831 | 0.468598 | [File](notebooks/AES_Reset_E1_S3_MultiTrait.ipynb) | [Download](hasil/Hasil_AES_Reset_E1_S3_MultiTrait_Publik.xlsx) |
| E2 | Empat head independen | 0.425449 | 0.466544 | [File](notebooks/AES_Reset_E2_S3_MultiTrait_TraitHeads.ipynb) | [Download](hasil/Hasil_AES_Reset_E2_S3_MultiTrait_Publik.xlsx) |
| E3 | Attention token per trait | 0.425778 | 0.472422 | [File](notebooks/AES_Reset_E3_S3_MultiTrait_TraitAttention.ipynb) | [Download](hasil/Hasil_AES_Reset_E3_S3_MultiTrait_Publik.xlsx) |
| E4 | Attention antartrait | 0.425284 | 0.466443 | [File](notebooks/AES_Reset_E4_S3_MultiTrait_InterTraitAttention.ipynb) | [Download](hasil/Hasil_AES_Reset_E4_S3_MultiTrait_Publik.xlsx) |

E4 terbaik nominal, tetapi peningkatan test terhadap E2 hanya 0,000101 poin (0,022%). Satu run belum cukup untuk menyatakan keunggulan statistik.

## Eksperimen reset lanjutan

| Eksperimen | Perubahan utama | Status / catatan | Notebook |
| --- | --- | --- | --- |
| E16 | Gabungan E6 dynamic gate, E10 prompt/inter-trait attention, dan fondasi E14; independent trait heads 64. | Notebook tersedia; gunakan workbook hasil run untuk klaim metrik. | [E16](notebooks/AES_Reset_E16_E6_E10_E14_MultiTrait.ipynb) |
| E17-A | Partial fine-tuning RoBERTa; embedding + 8 layer bawah frozen, 4 layer atas trainable. | Branch khusus; menjadi dasar E17-B. | [E17-A](notebooks/AES_Reset_E17A_PartialFineTuning.ipynb) |
| E17-B | E17-A dengan tempered LDS inverse-square-root (power 0,5) hanya pada loss Isi. | Branch khusus; [PR #1 masih draft](https://github.com/MalikHariyanto/Automated_Essay_Scoring/pull/1). Belum ada klaim full GPU training atau peningkatan akurasi. | [E17-B](notebooks/AES_Reset_E17B_Isi_TemperedLDS.ipynb) |
| E17C | Kontrol E14 S3 dengan dropout MLP 0,2 → 0,1; RoBERTa full fine-tuning, BGE-M3 frozen, concat. | Pembanding historis. Notebook tersedia; hasil training/test belum tercatat di repository. | [E17C](notebooks/AES_Reset_E17C_S3_Concat_Dropout01_Native_Learnable_GPU.ipynb) |
| E18B | S3 E13 dengan paket Trial 60 E17E, lima fold development | 0,419344 | — | [File](notebooks/AES_Reset_E18B_E13_Trial60_Validasi.ipynb) | [Audit hasil validasi](output/E18B_2026-11-27/README.md) |
| E19 | S3 E13 dengan rasio bobot loss organisasi 1,2× untuk skor 1, lima fold development | 0,424268 | — | [File](notebooks/AES_Reset_E19_E13_Bobot_Organisasi_Validasi.ipynb) | [Arsip](output/checkpoints/E19.json) |
| E20 | Resep E19 dengan perbandingan dua aturan pemilihan checkpoint dari lima fit yang sama | 0,427930 (B) | — | [File](notebooks/AES_Reset_E20_E19_Pilih_Checkpoint_AB.ipynb) | [Arsip](output/checkpoints/E20.json) |
| E20 konfirmasi | Ulang perbandingan checkpoint A/B pada lima fold development dengan seed dasar 43 | 0,427265 (B) | — | [File](notebooks/AES_Reset_E20_Konfirmasi_Seed43.ipynb) | [Arsip](output/checkpoints/E20-Seed43.json) |
| E21 | S3 pada test holdout, ensemble lima fold untuk masing-masing aturan checkpoint A dan B | 0,427513 (B) | 0,446588 (B) | [File](notebooks/AES_Reset_E21_S3_Test_Checkpoint_AB.ipynb) | [Arsip](output/checkpoints/E21.json) |
| E22 | Pilot fitur konteks soal 10 khusus prediksi isi pada S3, lima fold development dengan resep E21 | 0,427536 (B) | — | [File](notebooks/AES_Reset_E22_S3_Bukti_Konteks_Isi.ipynb) | [Arsip](output/checkpoints/E22.json) |
| E23 | Bobot loss isi skor 1 sebesar 2×, lima fold development dengan resep E21 | 0,431853 (B) | — | [File](notebooks/AES_Reset_E23_S3_Bobot_Isi_2x.ipynb) | [Arsip](output/checkpoints/E23.json) |
| E23 konfirmasi | Bobot isi 2× pada lima fold yang sama dengan seed dasar 44 | 0,428296 (B) | — | [File](notebooks/AES_Reset_E23_Konfirmasi_Seed44.ipynb) | [Arsip](output/checkpoints/E23-Seed44.json) |
| E24 | Pilot NLI frozen untuk cek konteks sekolah/rumah di soal 10 | Pilot selesai | — | [File](notebooks/AES_Reset_E24_Pilot_Bukti_Konteks_NLI.ipynb) | [Hasil pilot](output/checkpoints/E24/README.md) |
| E18A | BGE relevance/cosine pada trait Isi. | Dinonaktifkan; jangan diperlakukan sebagai jalur aktif. | [Branch E18A](https://github.com/MalikHariyanto/Automated_Essay_Scoring/tree/codex/e18a-isi-bge-relevance) |

[Register paper dan konfigurasi](../../docs/skripsi/Register_Bukti_AES_MultiTrait.xlsx) · [Arsip checkpoint](output/checkpoints/README.md)

[Notebook E14 khusus S1 dan S2](notebooks/AES_Reset_E14_S1_S2_Banding_MultiTrait.ipynb) menghasilkan satu ZIP 01–05. Evaluasi dan riwayat training dibedakan lewat kolom `skenario` pada workbook 04–05.

## Resep E17C

- Input tiga teks: soal, kunci jawaban, dan jawaban siswa.
- RoBERTa 768 dimensi di-fine-tune penuh; BGE-M3 1.024 dimensi tetap frozen.
- Proyeksi BGE 1.024→256 + LayerNorm; concat dengan RoBERTa menjadi 1.024 dimensi.
- Head bersama: Linear 1.024→256, GELU, dropout 0,1, Linear 256→4.
- MSE pada target rubrik ternormalisasi 0–1; loss relasi antartrait 0.
- Lima fold; maksimum 25 epoch; patience 5; seed training 42; split seed 2026.
- Effective batch 8, microbatch 1; RoBERTa LR 5e-6; head/projection/LayerNorm LR 1e-4; weight decay 0,01; warmup 10%; gradient clipping 1,0.
- E14 S3 menjadi kontrol dengan resep yang sama kecuali dropout MLP 0,2.

Notebook E17C mengimplementasikan CV, refit, evaluasi test, inspeksi, dan ekspor workbook 01–05. Status kode tidak sama dengan hasil eksperimen: sebelum Run all pada GPU dan verifikasi workbook/ZIP, jangan mengklaim training, test, atau ekspor berhasil.

[Progres dan batas bukti](PROGRES.md) · [Acuan Referensi Kedua](../../docs/skripsi/Acuan_Referensi_2_AES_E17A_E17B_E17C.md) · [Semua kelompok](../README.md)

Excel publik mempertahankan skor dan prediksi anonim, metrik, konfigurasi, serta riwayat training. Identitas/teks siswa dan vektor besar tidak dipublikasikan; workbook asli tetap privat. Notebook Colab tetap mengekspor keluaran privat lengkap.
