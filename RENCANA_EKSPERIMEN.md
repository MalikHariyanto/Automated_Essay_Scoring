# Eksperimen lanjutan · keputusan terbaru memakai E26

Pengguna sudah menjalankan E27/E28 berbasis E23, lalu menetapkan E26 sebagai acuan berikutnya. **E29 = E26 + LR MLP S2 dan S3 5e-6**; **E30 = E26 + co-attention respons–referensi lalu attention pooling**. MLP E30 tetap1e-4; kedua resep encoder5e-6 dan BiGRU/pooling/proyeksi/norm1e-4, linear S1 tetap1e-4. Kode GPU tersedia, hasil belum ada. [Notebook dan metode](docs/eksperimen/eksperimen_e29_e30_gpu.md).

Usulan berikutnya adalah menguji E26/kandidat dengan tiga seed dan, bila sesuai protokol CV, merata-ratakan prediksi setiap jawaban. **Mean metrik tiga seed bukan ensemble prediksi.** E31 belum dibuat; jangan menambahkan ensemble diam-diam ke E29/E30.

## Riwayat dua jalur E27/E28 dari E23

**E27/E28 dibuat dari E23 atas pilihan eksplisit pengguna; hasilnya kini tersedia.** RMSE S3 E27=6,3277, E28=7,0447; [review](docs/review/review_e27_e28_study.md). Angka frekuensi enam peserta di bawah menjadi riwayat; delapan peserta kini membuat E23/E26 seri10/60. Prioritas pengguna ialah banyaknya prediksi mendekati guru: E23 paling sering terdekat13/60 dan selisih≤5poin39/60 pada enam eksperimen. RMSE test E23 **6,0439**, MAE **4,8576**, mean CV **6,4143**. E26 tetap lebih rendah RMSE/MAE **test** (5,7435/4,4797), bukan hanya bagus saat training. Perbedaan ukuran menjelaskan pilihan dasar; satu run belum membuktikan konsistensi. Tetap tiga input, label `rater`, RoBERTa fine-tuned, BGE frozen, kontrol S1/S2. [Notebook dan metode](docs/eksperimen/eksperimen_e27_e28_gpu.md).

## Jalur yang dapat dibandingkan

| Jalur | Combo | Yang diuji | Yang dipertahankan |
| --- | --- | --- | --- |
| A · parameter | **E23 + LR MLP saja `5e-6`** | Apakah langkah MLP lebih kecil memperbaiki validasi dan kedekatan prediksi? | Encoder `5e-6`; BiGRU/proyeksi/norm `1e-4`, MSE |
| B · layer | **E23 + co-attention respons–referensi** | Apakah pencocokan eksplisit dengan kunci membantu? | MLP/proyeksi/layer baru `1e-4`, encoder `5e-6`, MSE |
| C · pengukuhan | **Acuan dan kandidat × 3 seed** | Apakah peningkatan konsisten pada 42, 123, 2026? | Split/fold tetap; resep kandidat dikunci |

A dan B diuji terpisah dahulu: **A menjadi E28**, **B menjadi E27**. Gabungan keduanya baru layak setelah CV memberi dukungan masing-masing. Jalur C masih rencana; tidak ada nomor baru atau hasil tambahan yang dibuat otomatis.

## A: LR MLP yang dimaksud pengguna

Head MLP sampai E26 tetap `1e-4`. **`5e-6` turun 20 kali**, bukan kenaikan. Head baru diinisialisasi sehingga LR terlalu kecil dapat membuatnya kurang belajar dalam anggaran 25 epoch. Kandidat lebih moderat `5e-5` dapat direncanakan melalui CV sebelum test, bukan dipilih setelah melihat test.

Pada `optimizer_groups()` E23 asal, seluruh parameter yang bukan `roberta.*` atau `projection.*` masuk kelompok `head`, termasuk:

- `head.*`: MLP regresi;
- `sequence_pool.*`: BiGRU dan proyeksi ringkasan128→768;
- `roberta_norm.*`: LayerNorm tambahan.

Jangan sekadar mengubah `HEAD_LR` untuk uji **MLP saja**. E28 sudah memisahkan `model.head.*` ke kelompok LR sendiri; pooling/norm tetap `1e-4`. Setiap parameter trainable masuk tepat satu kelompok. Perlakuan khusus S3; kontrol S1/S2 tetap memakai resep asal.

## B: combo layer yang paling relevan

**E23 + co-attention respons–referensi** menguji hubungan jawaban dengan kunci secara eksplisit. RoBERTa saat ini sudah membaca ketiga teks melalui self-attention; tambahan ini bukan memperkenalkan referensi yang sebelumnya tidak dibaca. Draf dasar E26 diganti sebelum publikasi; E27 tidak menambahkan additive attention pooling E26.

Rancangan yang diimplementasikan pada E27 dan diperiksa melalui fixture teknis:

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
  ujung maju64 + ujung mundur64
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

Acuan implementasi: [E23 GPU](notebooks/experiments/AES_300_E23_Tiga_Input_BiGRU_GPU.ipynb), [metode E27/E28](docs/eksperimen/eksperimen_e27_e28_gpu.md). Pembanding hasil: [review E25/E26](docs/review/review_e25_e26_study.md).
