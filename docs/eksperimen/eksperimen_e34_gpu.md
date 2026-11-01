# E34 · E33 dengan LR kelompok layer S3 5e-5

E34 mengikuti pilihan pengguna setelah analisis E33. Dasarnya notebook E33 pada commit `fa45c47a4d972a20a36445ea58d1fca8e8adcded`. Hanya `s3_layers_lr` berubah dari1e-4 menjadi5e-5; MLP S3 tetap5e-5.

| Kelompok | E33 | E34 |
|---|---:|---:|
| Encoder RoBERTa S1/S3 | 5e-6 | 5e-6 |
| Linear S1 | 1e-4 | 1e-4 |
| MLP S2 | 1e-4 | 1e-4 |
| MLP S3 `head.*` | 5e-5 | 5e-5 |
| **Layer S3 `pooling_norm`** | **1e-4** | **5e-5** |
| Proyeksi BGE `projection.*` | 1e-4 | 1e-4 |

Kelompok `pooling_norm` berisi `sequence_pool.*` dan `roberta_norm.*`: BiGRU, proyeksi referensi/co-attention, matching, additive attention pooling, proyeksi ringkasan128→768, dan LayerNorm RoBERTa tambahan. Karena beberapa modul berada dalam satu kelompok, ini bukan eksperimen yang mengisolasi satu layer. Proyeksi BGE1024→256 tetap dalam kelompok terpisah dengan LR1e-4.

Arsitektur tetap E33/E30: tiga input → RoBERTa fine-tuned → BiGRU → co-attention respons–referensi → attention pooling → concat proyeksi BGE frozen → MLP. Target satu `rater`0–100. Tidak ada Ridge. Notebook47cell/30codecell mempertahankan urutan setup, upload, inspeksi fitur awal, training S1/S2/S3, prediksi, fitur final dan lima Excel/ZIP. Tanpa clone/Drive/file Python pendamping; GPU wajib.

5fold CV, maksimal25epoch, patience5 berdasarkan RMSE validasi; pulihkan bobot terbaik. Refit model baru pada240development memakai ceil median epoch terbaik, lalu evaluasi60test. Seed42/split2026, microbatch1/efektif8, MSE target0–1, dropout0,1, AdamW decay0,01, warmup10%+linear scheduler dan clipping1 mengikuti E33. Semua skenario dilatih baru dengan resep S1/S2 sebagai kontrol.

Setelah normalisasi nama folder/hash, hanya codecell10dan12berbeda dari E33: defaultLR layer S3, nilai konfigurasi pengguna dan identitas/log resep. Fungsi model, optimizer, loss, CV/refit, tokenisasi dan exporter identik. Folder/ZIP baru, fingerprint dan lesson_version mencegah penggunaan checkpoint E33 sebagai resume E34. Jangan memindahkan checkpoint antar-resep.

Belum ada hasil penelitian E34. LR lebih kecil merupakan hipotesis untuk memperbaiki kedekatan prediksi, bukan jaminan peningkatan. Bandingkan E33 dan E34 pada fold/seed sama melalui development sebelum membuka test historis. Format ini tetap satu seed42; konfirmasi tiga seed merupakan langkah terpisah, bukan hasil yang sudah tersedia.
