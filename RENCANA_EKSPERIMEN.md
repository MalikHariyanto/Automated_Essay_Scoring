# Combo berikutnya setelah E26

**Usulan; belum dibuat notebook baru atau dijalankan.** Acuan E26: mean RMSE CV S3 **5,9245**, test **5,7435** pada hasil yang tersedia. Belum ada pengukuhan lintas seed. Tetap tiga input, label `rater`, RoBERTa fine-tuned, BGE frozen, serta kontrol S1/S2.

## Jalur yang dapat dibandingkan

| Jalur | Combo | Yang diuji | Yang dipertahankan |
| --- | --- | --- | --- |
| A · parameter | **E26 + LR MLP saja `5e-6`** | Apakah langkah MLP lebih kecil memperbaiki validasi? | Encoder `5e-6`; proyeksi/pooling/norm `1e-4`, MSE |
| B · layer | **E26 + co-attention respons–referensi** | Apakah pencocokan eksplisit dengan kunci membantu? | MLP/proyeksi/layer baru `1e-4`, encoder `5e-6`, MSE |
| C · pengukuhan | **Acuan dan kandidat × 3 seed** | Apakah peningkatan konsisten pada 42, 123, 2026? | Split/fold tetap; resep kandidat dikunci |

A dan B diuji terpisah dahulu. Gabungan keduanya baru layak setelah CV memberi dukungan masing-masing. Nomor eksperimen baru ditetapkan saat implementasi; tabel ini tidak berarti E27/E28 sudah tersedia.

## A: LR MLP yang dimaksud pengguna

Head MLP sampai E26 tetap `1e-4`. **`5e-6` turun 20 kali**, bukan kenaikan. Head baru diinisialisasi sehingga LR terlalu kecil dapat membuatnya kurang belajar dalam anggaran 25 epoch. Kandidat lebih moderat `5e-5` dapat direncanakan melalui CV sebelum test, bukan dipilih setelah melihat test.

Pada `optimizer_groups()` E26, seluruh parameter yang bukan `roberta.*` atau `projection.*` masuk kelompok `head`, termasuk:

- `head.*`: MLP regresi;
- `sequence_pool.*`: BiGRU, attention dan proyeksi ringkasan128→768;
- `roberta_norm.*`: LayerNorm tambahan.

Jangan sekadar mengubah `HEAD_LR` untuk uji **MLP saja**. Pisahkan `model.head.*` ke kelompok LR baru; pooling/norm tetap `1e-4`. Setiap parameter trainable masuk tepat satu kelompok. Bila perlakuan khusus S3, kontrol S1/S2 tetap memakai resep asal.

## B: combo layer yang paling relevan

**E26 + co-attention respons–referensi** menguji hubungan jawaban dengan kunci secara eksplisit. RoBERTa saat ini sudah membaca ketiga teks melalui self-attention; tambahan ini bukan memperkenalkan referensi yang sebelumnya tidak dibaca.

Rancangan yang perlu diimplementasikan dan diperiksa:

```text
RoBERTa membaca soal + referensi + respons
                 │
       ┌─────────┴──────────┐
 token respons768     token referensi768
       │                    │
 BiGRU → urutan128     proyeksi → urutan128
       └─────────┬──────────┘
       co-attention dua arah
                 │
 fitur pencocokan → proyeksi128
                 │
       attention pooling E26
                 │
 ringkasan128 →768 → LayerNorm → concat BGE256 → MLP
```

Ini adaptasi, bukan reproduksi paper. Mask referensi/respons harus memakai posisi token sebenarnya, bukan kesamaan ID token. Padding, prefix, special dan referensi terpotong ditangani. Attention membandingkan urutan token; satu key berupa vektor BGE global membuat softmax selalu satu.

Rujukan: **Zhang & Litman (2018), [Co-Attention Based Neural Network for Source-Dependent Essay Scoring](https://aclanthology.org/W18-0549/)**, DOI `10.18653/v1/W18-0549`. Full text diperiksa melalui [salinan pada repo situs penulis](https://raw.githubusercontent.com/Rokeer/academic/master/files/publications/W18-0549.pdf). Tabel4: QWK SELF-ATTN→CO-ATTN **0,677→0,697** pada ASAP3, **0,681→0,697** pada RTA MVP, **0,669→0,684** pada RTA Space. Ketiganya signifikan terhadap SELF-ATTN; peningkatan ASAP4–6 lebih kecil dan tidak signifikan terhadap SELF-ATTN.

Paper memakai **MSE**, CNN/LSTM dan co-attention tingkat kalimat dengan teks sumber. Model, unit, bahasa, label/skala dan dataset berbeda. Bukti ini mendukung alasan menguji pencocokan sumber, bukan jaminan RMSE proyek turun atau ablation terisolasi satu layer.

## Yang belum diprioritaskan

E26 + gate E25 mungkin dicoba kemudian. Namun gate E25 final hanya **0,495721–0,496997**; kedua pengali masih dekat1, dan RMSE E25 lebih tinggi dari E24 pada run itu. Bukti belum mendukung memprioritaskan penumpukan gate.

Huber menjadi uji loss terpisah bila residual development menunjukkan beberapa error besar mendominasi. Belum ada perbandingan AES MSE–Huber terkontrol yang berhasil diverifikasi pada penelusuran proyek. Tidak ada klaim Huber pasti lebih akurat/cepat. Pada target0–1, ambang5poin berarti delta0,05; default delta1 dapat hanya menjadi setengah MSE selama error≤1.

## Cara memutuskan

Gunakan split/fold sama. Catat konfigurasi sebelum membuka test, bandingkan RMSE lima fold, biaya/epoch dan residual. Kunci kandidat/refit melalui development; test yang telah berulang kali dilihat bersifat eksploratif. Tiga seed melatih acuan dan kandidat pada seed yang sama, bukan memilih seed paling bagus. Bedakan mean/SD metrik per seed dari metrik prediksi yang dirata-ratakan.

Acuan: [E26 GPU](notebooks/experiments/AES_300_E26_Tiga_Input_BiGRU_Attention_GPU.ipynb), [metode E25/E26](docs/eksperimen/eksperimen_e25_e26_gpu.md), [review hasil](docs/review/review_e25_e26_study.md).
