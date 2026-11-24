# Acuan Referensi Kedua · Automated Essay Scoring

**Versi:** 24 November 2026  
**Cakupan:** E17-A, E17-B, E17C, status E18A, benchmark E43–E53, dan referensi metode.  
**Relasi dokumen:** Acuan Referensi Pertama tetap tersedia sebagai `Brainstorming_Referensi_AES_MultiTrait.xlsx`; file pertama tidak diubah.

## Ringkasan eksekutif

- Jalur aktif repository per 24 November 2026 adalah **E17C concat/dropout 0,1**.
- E17-A tetap baseline historis untuk eksperimen partial fine-tuning. E17-B merupakan alternatif yang mempertahankan partial fine-tuning E17-A dan menambahkan tempered LDS hanya pada loss Isi; PR #1 masih draft.
- E18A BGE relevance/cosine dinonaktifkan. Pengembangan aktif diarahkan ke E17C tanpa cosine similarity.
- Belum ada hasil training/test E17C yang dapat diverifikasi dari repository pada saat laporan ini diperbarui. Status notebook tidak boleh disamakan dengan hasil eksperimen.

## Snapshot metrik E17-A

Angka berikut adalah snapshot hasil E17-A yang tersedia dalam analisis proyek. Workbook hasil mentah E17-A belum ditemukan pada branch publik yang diperiksa; cocokkan kembali angka dengan workbook asli sebelum dimasukkan ke naskah final.

| Trait | QWK ↑ | Gwet AC2 ↑ | Pearson ↑ | RMSE ↓ |
| --- | ---: | ---: | ---: | ---: |
| Isi | 0,404 | 0,853 | 0,420 | 0,688 |
| Organisasi | 0,327 | 0,845 | 0,358 | 0,325 |
| Kebahasaan | 0,388 | 0,706 | 0,460 | 0,395 |
| Mekanik | 0,562 | 0,816 | 0,588 | 0,331 |
| Macro average | 0,420 | 0,805 | 0,457 | 0,435 |

### Diagnosis skor Isi rendah

- Dataset berisi 1.340 respons; skor Isi 1 tercatat pada 20 respons (1,49%).
- Recall skor Isi 1 pada OOF tercatat 5,9%; pada test 0%.
- RMSE Isi 0,688; baseline konstan 0,726. Model mengungguli baseline sederhana, tetapi peningkatannya terbatas.
- Temuan tersebut menjadi alasan menguji LDS khusus Isi, bukan bukti bahwa LDS pasti memperbaiki hasil.

## E17-B · rancangan alternatif

E17-B mempertahankan E17-A partial fine-tuning: embedding dan delapan layer bawah RoBERTa frozen, empat layer atas trainable. Perubahan utama hanya pada loss Isi.

| Parameter | Nilai |
| --- | --- |
| Trait yang dibobot | Isi saja |
| LDS power | 0,5 (inverse-square-root) |
| Gaussian smoothing | radius 1, sigma 1,0, kernel diskret panjang 3 |
| Normalisasi bobot | Mean sample weight = 1 pada training fit |
| Loss Isi | Weighted MSE |
| Loss Organisasi/Kebahasaan/Mekanik | MSE tanpa bobot |
| Evaluasi validation/test | Tidak berbobot |
| Sumber bobot | Hanya label pada indeks training tiap fit |

Branch: `codex/e17b-isi-tempered-lds`. PR #1 masih draft. Notebook berhasil diambil ulang dan diparse sebagai JSON; pemeriksaan kode memeriksa LDS train-only, loss Isi-only, metrik validation/test unweighted, dan output/cache terpisah. Synthetic preflight belum dijalankan pada runtime GPU; full training belum diklaim.

## E17C · jalur aktif

E17C menguji perubahan dropout MLP 0,2→0,1 terhadap kontrol E14 S3.

| Komponen | Konfigurasi |
| --- | --- |
| Input | Soal, kunci jawaban, jawaban siswa |
| RoBERTa | 768 dimensi; full fine-tuning |
| BGE-M3 | 1.024 dimensi; frozen |
| Proyeksi BGE | 1.024→256 + LayerNorm |
| Fusion | Concat RoBERTa 768 + proyeksi BGE 256 = 1.024 |
| Head | Linear 1.024→256, GELU, dropout 0,1, Linear 256→4 |
| Loss | MSE pada target rubrik 0–1; relation loss 0 |
| Training | 5 fold; maksimum 25 epoch; patience 5; seed 42; split seed 2026 |
| Optimizer | AdamW; RoBERTa LR 5e-6; head/projection/LayerNorm LR 1e-4; weight decay 0,01 |
| Batch dan scheduler | Effective batch 8; microbatch 1; warmup 10%; clipping 1,0 |
| Token | RoBERTa max 512; BGE max 1.024; overlap 128 |

**Batas:** E17C menggunakan full fine-tuning RoBERTa, sedangkan E17-A/E17-B memakai partial fine-tuning. Karena itu, perbandingan E17C terhadap E17-A/B bukan ablation satu faktor. Kontrol yang paling tepat untuk perubahan dropout adalah E14 S3 dengan dropout MLP 0,2 dan split yang sama.

## E18A · status

E18A BGE relevance/cosine telah dinonaktifkan. README repository menyatakan jalur aktif adalah E17C concat tanpa cosine similarity. Branch E18A dipertahankan sebagai riwayat; jangan menganggapnya eksperimen aktif.

## Benchmark single-trait E43–E53

Angka E43/E48 berikut adalah hasil historis yang dirujuk dokumentasi eksperimen lanjutan. E49/E50/E53 belum memiliki hasil prediksi yang terdokumentasi.

| Eksperimen | Intervensi | Mean CV RMSE | Test RMSE | Test MAE | Status |
| --- | --- | ---: | ---: | ---: | --- |
| E43 | Masked mean pooling setelah co-attention | 5,8876 | 5,1265 | 3,7817 | Hasil historis tercatat |
| E48 | LDS inverse penuh, power 1,0 | 6,3908 | 6,4777 | 4,7292 | Memburuk terhadap E43 |
| E49 | LDS inverse-square-root, power 0,5 | — | — | — | 5 test synthetic CPU; belum ada hasil prediksi |
| E50 | Multi-head co-attention, 4 × 32 | — | — | — | 5 test synthetic CPU; belum ada hasil prediksi |
| E53 | EMA, decay maksimum 0,99 | — | — | — | 7 test synthetic CPU; belum ada hasil prediksi |

**Jangan bandingkan RMSE E43–E53 secara langsung dengan E17-A/E17-B/E17C.** E43–E53 memakai single-trait dan dataset `fix_dataset_300.xlsx` (300 respons, target 0–100), sedangkan E17 menggunakan multi-trait, 1.340 respons, dan skala rubrik per trait.

## Roadmap keputusan

1. Jalankan E17C pada T4; verifikasi lima fold, refit, test, manifest, workbook, dan ZIP.
2. Bandingkan E17C dengan E14 S3 pada split yang sama. Laporkan RMSE OOF gabungan/per trait, Pearson, QWK, Gwet AC2, confusion matrix, dan deteksi skor rendah.
3. Evaluasi E17-B sebagai jalur alternatif untuk kelas Isi langka; jangan membandingkannya dengan E17C seolah hanya satu faktor yang berubah.
4. Setelah baseline aktif stabil, uji satu perubahan tambahan per eksperimen. Co-attention dan EMA tetap kandidat, bukan klaim peningkatan.
5. Gunakan data development untuk pemilihan model; test hanya untuk evaluasi akhir. Laporkan variasi lintas seed sebelum menyimpulkan keunggulan.

## Referensi akademik

1. Taghipour, K. & Ng, H. T. (2016). A Neural Approach to Automated Essay Scoring. *Proceedings of EMNLP 2016*, 1882–1891. https://aclanthology.org/D16-1193/
2. Zhang, H. & Litman, D. (2018). Co-Attention Based Neural Network for Source-Dependent Essay Scoring. *BEA 2018*, 399–409. https://aclanthology.org/W18-0549/
3. Yang, Y., Zha, K., Chen, Y., Wang, H. & Katabi, D. (2021). Delving into Deep Imbalanced Regression. *ICML 2021*, PMLR 139, 11842–11851. https://proceedings.mlr.press/v139/yang21m.html
4. Morales-Brotons, D., Vogels, T. & Hendrikx, H. (2024). Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits. *Transactions on Machine Learning Research*. https://openreview.net/forum?id=2M9CUnYnBA
5. Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS 2017*. https://arxiv.org/abs/1706.03762
6. Peters, M. E. et al. (2018). Deep Contextualized Word Representations. *NAACL-HLT 2018*, 2227–2237. https://aclanthology.org/N18-1202/
7. Liu, Y. et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. https://arxiv.org/abs/1907.11692
8. Chen, J. et al. (2024). BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation. https://arxiv.org/abs/2402.03216

Paper di atas mendukung dasar metode umum. Paper tersebut tidak membuktikan bahwa konfigurasi atau model pada dataset proyek ini pasti lebih baik.

## Referensi internal

- Acuan Referensi Pertama: [Brainstorming_Referensi_AES_MultiTrait.xlsx](Brainstorming_Referensi_AES_MultiTrait.xlsx)
- Notebook E17-A: [AES_Reset_E17A_PartialFineTuning.ipynb](../../eksperimen/03_tiga_input_multi_trait/notebooks/AES_Reset_E17A_PartialFineTuning.ipynb)
- Notebook E17-B: [AES_Reset_E17B_Isi_TemperedLDS.ipynb](../../eksperimen/03_tiga_input_multi_trait/notebooks/AES_Reset_E17B_Isi_TemperedLDS.ipynb)
- Notebook E17C: [AES_Reset_E17C_S3_Concat_Dropout01_Native_Learnable_GPU.ipynb](../../eksperimen/03_tiga_input_multi_trait/notebooks/AES_Reset_E17C_S3_Concat_Dropout01_Native_Learnable_GPU.ipynb)
- PR E17-B: https://github.com/MalikHariyanto/Automated_Essay_Scoring/pull/1
