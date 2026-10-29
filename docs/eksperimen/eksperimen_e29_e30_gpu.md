# E29 dan E30 · dua resep dari E26, Colab GPU

Pengguna memilih E26 sebagai dasar kedua notebook. **E29 mengubah LR MLP S2/S3; E30 menambah co-attention sebelum attention pooling.** Keduanya terpisah dan belum memiliki hasil penelitian. Buku hasil E21–E28 dan notebook E26/E27/E28 tidak diganti.

| Eksperimen | Source | Perubahan |
| --- | --- | --- |
| E29 | [Notebook](../../notebooks/experiments/AES_300_E29_E26_BiGRU_Attention_MLPLR5e6_GPU.ipynb) | Seluruh parameter MLP `head.*` pada S2 dan S3 memakai5e-6 |
| E30 | [Notebook](../../notebooks/experiments/AES_300_E30_E26_BiGRU_CoAttention_Pooling_GPU.ipynb) | E26 + pencocokan respons–referensi, lalu attention pooling E26 |

Dasar [E26](../../notebooks/experiments/AES_300_E26_Tiga_Input_BiGRU_Attention_GPU.ipynb) dipin ke commit `bc4713872c7a7ce23f7dad8ae3ea73a2b75d9b86`; bukan turunan arsitektur E23/E27/E28. Source E27 dipakai sebagai rujukan implementasi mask/pencocokan, bukan untuk mengganti pooling E26 dengan ringkasan ujung dua arah.

## Pengaturan tetap

Tiga input pada seluruh skenario: `soal`, `jawaban_referensi`, `response_text`. Label tunggal **`rater`0–100** tidak masuk tokenizer. Format `single_sequence` E26, dengan anggaran soal64/konteks192 dan window/overlap yang sama.

S1 RoBERTa full fine-tuning + linear; S2 BGE frozen + MLP; S3 RoBERTa full fine-tuning + BGE frozen + BiGRU/attention/fusion MLP. Tidak menambah LoRA, trait, label baru, atau format context pair. Email/ID hanya metadata pemetaan/kelompok.

| Kelompok | E29 | E30 |
| --- | ---: | ---: |
| Encoder RoBERTa S1/S3 | 5e-6 | 5e-6 |
| Head linear S1 | 1e-4 | 1e-4 |
| **MLP S2** | **5e-6** | 1e-4 |
| **MLP S3** | **5e-6** | 1e-4 |
| Proyeksi BGE | 1e-4 | 1e-4 |
| BiGRU, attention pooling, norm, proyeksi ringkasan | 1e-4 | 1e-4 |
| Proyeksi co-attention/matching | Tidak ada | 1e-4 |
| Encoder BGE | Frozen | Frozen |

Nilai tabel ialah LR awal/puncak sebelum warmup+linear scheduler. E29 menurunkan LR MLP20×; tidak membekukan MLP. S2 kini juga perlakuan eksperimen, sementara S1 tetap kontrol. Parameter trainable lain tetap ikut belajar pada LR masing-masing; bobotnya dapat berubah secara tidak langsung karena gradien berubah setelah head belajar.

`optimizer_groups()` memisahkan `head.*` dari `sequence_pool.*`/`roberta_norm.*`, memastikan setiap parameter trainable masuk tepat satu kelompok. Config/Excel/checkpoint mencatat `s2_mlp_lr`, `s3_mlp_lr`, `s3_layers_lr`, dan head linear S1 secara terpisah.

Loss MSE target0–1, maksimal25epoch, patience5,5fold pada240development,60test, microbatch1/efektif8, seed42/split2026. Refit memakai ceil median epoch pilihan CV. Seluruh S1/S2/S3 dilatih baru; tidak memakai ulang checkpoint eksperimen lama.

## E30: dua tahap attention tambahan

```text
RoBERTa membaca soal + referensi + respons
            │                    │
       respons768          isi referensi768
            │                    │
     BiGRU64/arah             Linear128
            │                    │
        urutan128 ← co-attention dua arah → urutan128
            │
      matching640 → Linear128 + tanh
            │
      attention pooling128→64→1
            │
      weighted sum128 → Linear768 → LayerNorm
            │
      concat proyeksi BGE256 → MLP1024→256→1
```

**Co-attention** mencocokkan respons dengan kunci. Untuk urutan respons `R` dan referensi `K`, skor `R Kᵀ/√128` menghasilkan `A=softmax(S)` sepanjang referensi dan `B=softmax(Sᵀ)` sepanjang respons. Fitur `[R, AK, R−AK, R⊙AK, A(BR)]`640 diproyeksikan128+tanh.

**Attention pooling** mengikuti layer E26: jaringan128→64→1 memberi bobot pada urutan hasil matching, lalu weighted sum menghasilkan128 fitur. Berbeda dari E27 yang mengambil komponen ujung setelah matching. Self-attention di dalam RoBERTa tetap ada; “dua attention” di sini berarti dua tahap tambahan setelah backbone/BiGRU.

Mask berasal dari posisi layout dan offset tokenizer dengan ID fast/slow wajib identik. Prefix, soal, special dan padding bukan query/key/pooling langsung, tetapi RoBERTa tetap membaca ketiga teks bersama. Referensi mengikuti truncation E26; isi referensi yang habis terpotong menyebabkan error jelas. Softmax/pencocokan dihitung FP32 saat AMP; gradien tetap mencapai encoder, GRU, kedua proyeksi matching dan attention pooling.

Inisialisasi parameter E26 yang sudah ada dipertahankan; modul co-attention baru dibuat setelah layer acuan. Ini tidak berarti prediksi awal E30 sama dengan E26, karena jalur forward bertambah.

Rujukan: Zhang & Litman2018, [Co-Attention Based Neural Network for Source-Dependent Essay Scoring](https://aclanthology.org/W18-0549/), DOI10.18653/v1/W18-0549 dan [kode penulis](https://github.com/Rokeer/co-attention). Paper memakai CNN/LSTM/co-attention tingkat kalimat dengan teks sumber; kombinasi token RoBERTa–BiGRU–BGE ini adaptasi, bukan reproduksi atau jaminan peningkatan RMSE.

## Menjalankan dan membaca keluaran

**Web Colab → Runtime → Change runtime type → T4 GPU → Run all → unggah `fix_dataset_300.xlsx`.** GPU wajib; AMP bf16/fp16 otomatis. Notebook mandiri47cell/30cellkode, tanpa clone repo, ekstensi, file Python pendamping, mount atau autosave Drive. Dataset/schema mengikuti E26, termasuk metadata yang dipakai untuk kelompok.

Folder runtime terpisah sesuai nama notebook. Progres memakai dua slot per tahap dan checkpoint final lokal; seluruh fold/model masih memerlukan ruang disk. Tidak menghapus Drive atau file eksperimen lain. Runtime hilang berarti progres lokal hilang. Resume hanya untuk resep/config dan folder yang sama. ZIP ringkas berisi lima Excel, bukan bobot model.

E29 mempertahankan seluruh ekspor E26 dan menambahkan pencatatan dua LR MLP. E30 menambahkan sheet pada `02_Vektor_Jawaban.xlsx`:

- **Bobot Attention S3**: bobot additive attention pooling pada token hasil matching.
- **Bobot CoAttention Respons S3**: rata-rata referensi→respons, untuk inspeksi pencocokan.
- **Bobot Referensi S3**: rata-rata respons→referensi.
- **Pasangan Token Top1 S3**: satu kecocokan referensi dengan bobot tertinggi per token respons; bukan seluruh matriks.
- **Ringkasan CoAttn Pooling S3**:128fitur setelah matching dan pooling, sebelum proyeksi768.

Ketiga kelompok bobot masing-masing berjumlah1 per window; maknanya berbeda dan bukan skor guru atau penjelasan kausal. Vektor seluruh300jawaban untuk inspeksi; evaluasi generalisasi tetap pada60test. Prediksi mentah tidak dibulatkan untuk RMSE/Pearson.

Uji kedua resep terpisah terhadap E26. LR lebih kecil dan layer tambahan tidak dijamin lebih baik. Pilih melalui development/CV, laporkan biaya dan seluruh hasil; test lama sudah sering dilihat sehingga hasil lanjutan bersifat eksploratif. Ensemble tiga seed/E31 belum dibuat dalam permintaan ini.

[Bukti pengujian](../verifikasi/verification_e29_e30.md).
