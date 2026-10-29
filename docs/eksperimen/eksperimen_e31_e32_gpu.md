# E31/E32 · ulangi perlakuan E29/E30 dari acuan E25

Pengguna memilih E25 karena S3-nya paling banyak sama setelah pembulatan8/60 dan dalam1poin11/60 pada tabel sepuluh eksperimen. Ini bukan keunggulan pada semua ukuran: E29 masih RMSE terendah dan E30 unggul MAE/toleransi5poin. Kedua resep baru dibuat terpisah tanpa menimpa E25/E29/E30 atau mengubah isi HTML hasil.

| Komponen | E31 | E32 |
| --- | --- | --- |
| Acuan | E25 attention+gated fusion | E25 attention+gated fusion |
| Perlakuan | LR seluruh MLP S2/S3 menjadi5e-6 | Co-attention respons–referensi sebelum pooling |
| Encoder RoBERTa S1/S3 | 5e-6 | 5e-6 |
| Linear S1 | 1e-4 | 1e-4 |
| MLP S2/S3 | **5e-6** | 1e-4 |
| Attention/gate/proyeksi/norm/layer baru | 1e-4 | 1e-4 |
| Encoder BGE | Frozen | Frozen |
| BiGRU | Tidak ada | Tidak ada |

Source E25 dipin ke `bc4713872c7a7ce23f7dad8ae3ea73a2b75d9b86`. E31 mempertahankan forward/fitur E25. Optimizer memisahkan `head.*` dari gate/pooling/norm supaya hanya MLP S2/S3 yang diturunkan. Penurunan20× pernah membuat S2 E29 kurang belajar dalam25epoch; mengulang perlakuan bukan jaminan membaik.

## E32: adaptasi ke attention768 E25

```text
RoBERTa membaca soal + referensi + respons
      respons768                  referensi768
          ↓ Linear128                 ↓ Linear128
          └──────── co-attention dua arah ───────┘
                         ↓
           [R, AK, R−AK, R⊙AK, A(BR)]640
                         ↓ Linear768 + tanh
                 urutan matching768
                         ↓
           attention pooling E25:768→64→1
                         ↓ weighted sum768
                     LayerNorm768
                         ↓
        gate E25 dengan proyeksi BGE1024→256
                         ↓
           concat[2g×RoBERTa,2(1−g)×BGE]1024
                         ↓ MLP → satu skor
```

E32 mempertahankan ukuran pooling768 dan gate E25, sehingga tidak menyalin BiGRU128 E30. Dua proyeksi respons/referensi baru memadatkan pencocokan ke128; matching640→768 mengembalikannya sebelum pooling asli. Ini adaptasi desain yang dinyatakan eksplisit. Gate tetap skalar per jawaban, bukan bobot token; awalg=0,5 memberi pengali1 pada kedua cabang. RNG layer baru diisolasi agar inisialisasi parameter E25 yang sudah ada, termasuk gate, tidak bergeser.

Mask posisi/offset dan ID tokenizer diverifikasi; prefix/soal/special/padding bukan query/key/pooling langsung. RoBERTa tetap membaca ketiga teks. Softmax pencocokan/pooling FP32 saat AMP; jalur prediksi tetap differentiable. Salinan bobot inspeksi detach tidak memutus graph training.

Rujukan: [Zhang & Litman2018](https://aclanthology.org/W18-0549/) untuk co-attention source-dependent AES; paper memakai unit kalimat/CNN-LSTM, sehingga ini adaptasi token RoBERTa, bukan reproduksi. Gate mengikuti E25, terinspirasi [GMU2017](https://openreview.net/forum?id=Hy-2G6ile). Paper tidak membuktikan kombinasi ini atau LR tertentu pasti lebih baik.

## Menjalankan

Web Colab biasa → Runtime → Change runtime type → T4 GPU → Run all → unggah `fix_dataset_300.xlsx`. Tidak perlu clone, ekstensi atau file Python pendamping. GPU wajib; AMP otomatis. Tiga input semua skenario: `soal`,`jawaban_referensi`,`response_text`; satu label `rater`0–100. S1 linear, S2/S3 MLP. Semua skenario dilatih baru.

Maksimal25epoch,patience5,5fold pada240development,60test,microbatch1/efektif8,MSE target0–1,seed42/split2026. LR adalah nilai awal/puncak sebelum warmup+linear scheduler. Folder lokal terpisah per resep, checkpoint dua slot untuk resume jika filesystem runtime masih ada. Tanpa mount/autosave Drive. Jangan memindahkan checkpoint E25/eksperimen lain ke run ini. Runtime hilang berarti progres lokal hilang.

ZIP akhir berisi lima Excel, bukan bobot model. E32 menambahkan Bobot CoAttention Respons S3, Bobot Referensi S3 dan Pasangan Token Top1 S3; Bobot Attention S3 tetap pooling dan sheet Gate Fusion/Pengali tetap ada. Ketiga kelompok bobot masing-masing jumlah1 per window, berbeda makna dan bukan bukti sebab-akibat/kebenaran jawaban. Top1 bukan matriks alignment lengkap.

Belum ada hasil penelitian E31/E32. Pilih konfigurasi lewat development/CV; test yang sering dilihat tetap eksploratif. Tidak menambah ensemble/tiga seed atau mengganti label/metode lain diam-diam.
