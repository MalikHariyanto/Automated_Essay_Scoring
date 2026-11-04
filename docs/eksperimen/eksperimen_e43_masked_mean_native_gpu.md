# E43 · E42 dengan masked mean pooling · S3 native

[Buka Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E43_E42_S3_MaskedMeanPooling_Native_Learnable_GPU.ipynb) · [Source notebook](../../notebooks/experiments/AES_300_E43_E42_S3_MaskedMeanPooling_Native_Learnable_GPU.ipynb).

Pengguna menyetujui satu perubahan dari E42: mengganti attention pooling setelah matching co-attention dengan masked mean. Belum ada hasil penelitian E43. Notebook E42 dan E30 acuan lengkap dipertahankan.

## Mekanisme dan hipotesis

RoBERTa membaca soal, referensi dan jawaban dalam satu urutan. Scalar Mix empat layer terakhir dengan gamma menghasilkan token 768 dimensi. Token jawaban masuk BiGRU 64 per arah, sementara token referensi diproyeksikan menjadi 128 dimensi. Co-attention dua arah dan matching lima komponen tetap sama dengan E42. Setelah proyeksi matching 640→128 dan tanh, E43 merata-ratakan hanya token jawaban valid:

```python
pooling_weights = valid.float() / lengths.to(hidden.device).unsqueeze(1)
summary = (pooling_weights.unsqueeze(-1) * matched.float()).sum(1).to(response.dtype)
```

`lengths` adalah jumlah token jawaban valid pada setiap window. Bobot setiap token valid 1/N; padding 0. Mean tidak meratakan 128 komponen menjadi satu angka: hasilnya tetap vektor 128 dimensi, lalu Linear 128→768 dan LayerNorm. Mean antarwindow milik jawaban yang sama tetap seperti E42. Fitur RoBERTa 768 digabung dengan proyeksi BGE 256 menjadi 1024, kemudian MLP memprediksi satu skor.

Co-attention masih belajar memilih hubungan respons–referensi. Meskipun bobot mean tetap, keluaran setiap token tetap bergantung pada konteks, BiGRU, dan hasil matching yang dipelajari. Hipotesisnya, pengurangan parameter pembobotan akhir dapat membantu generalisasi pada development kecil; ini perlu diuji, bukan kesimpulan dari paper atau hasil E42.

Layer `attention_hidden` dan `attention_score` pooling dihapus sebelum optimizer dibuat. Keduanya terlebih dahulu diinisialisasi dalam urutan E42 agar seluruh bobot bersama dan keadaan RNG awal tetap identik untuk seed sama. Tidak ada parameter pooling attention yang dibiarkan masuk optimizer tanpa dipakai. Pengurangan tepat 8.320 parameter: 128×64+64 pada hidden dan 64 pada score. `POOLING_ATTENTION_HIDDEN=64` hanya ukuran inisialisasi acuan; E43 tidak memiliki bottleneck pooling yang dipelajari. Kelas dasar historis menyediakan inisialisasi, sedangkan forward subclass E43 menjalankan masked mean.

## Resep yang dipertahankan

| Komponen | E43, sama dengan E42 kecuali pooling |
|---|---|
| X / y | `soal`, `jawaban_referensi`, `response_text` / `rater` 0–100 |
| Encoder | `flax-community/indonesian-roberta-base`, fine-tuned; LR seragam 5e-6 |
| BGE | `BAAI/bge-m3`, frozen; fitur 1024→proyeksi 256 |
| Scalar Mix | Empat logit awal 0 → bobot 0,25; gamma awal 1; LR 1e-4 |
| BiGRU | Hidden 64 per arah, keluaran token 128 |
| Co-attention | Dot product terskala dan pencocokan dua arah seperti E42 |
| LR selain encoder | BiGRU, co-attention, norm, proyeksi, MLP, campuran/gamma: 1e-4 |
| Head / loss | MLP 1024→256→GELU→Dropout 0,1→1; MSE target 0–1 |
| Optimizer | AdamW, decay 0,01, warmup 10%, scheduler linear, clipping 1 |
| Batch | Microbatch 1, efektif 8 |
| Protokol | S3 saja, 5-fold sekali, maksimal 25 epoch, patience 5 |
| Refit / test | Ceil median best epoch lima fold; satu refit development dan satu test |

Satu konstanta TRAIN_SEED=42; seed per fold mengikuti E42/E30. Tidak ada sweep seed atau pengulangan kontrol E30. Pembagian data, fold, tokenizer/window, label, normalisasi, optimizer dan aturan pemilihan epoch tidak diubah. Label tidak masuk encoder tetapi mengarahkan loss, gradien, pemilihan epoch dan evaluasi.

## Pemakaian dan ekspor

Pilih GPU T4 di Colab, Run all, unggah `fix_dataset_300.xlsx`. Format 184 cell/89 cell kode Python biasa, tanpa magic, worker tersembunyi, clone repo, cosine atau Drive. Nama checkpoint, kolom X/y dan hyperparameter ada di section awal; model dan rumus pooling di section arsitektur.

Lima Excel standar tetap tersedia dalam `Hasil_AES_300_E43_E42_S3_MaskedMeanPooling_Native_Learnable_GPU.zip`:

- Excel 02 → **Bobot Mean Pooling S3**, kolom **Bobot Pooling**, menyimpan 1/N pada token valid. Sheet co-attention respons, referensi dan pasangan top-1 tetap terpisah. Bobot attention bukan penjelasan kausal atau skor kebenaran.
- Excel 02 tetap menyimpan ringkasan matching 128, pooling 768, fusion 1024, dan norma campuran sebelum/sesudah gamma.
- Excel 05 → **Bobot Scalar Mix**, empat baris bobot dan gamma untuk setiap checkpoint terpilih lima fold dan refit.
- Excel 05 → **MAE dan Toleransi CV**, per-fold dan gabungan OOF development. Dihitung dari prediksi validasi tersimpan pada skala desimal 0–100: MAE serta jumlah selisih ≤1, ≤5 dan ≤10 poin. Gabungan OOF menggunakan semua jawaban validasi, bukan rerata MAE fold tanpa pembobotan. Test tidak masuk ringkasan ini; empat metrik utama tetap.

Gunakan mean RMSE CV sebagai ukuran utama sesuai protokol, lalu periksa MAE/toleransi CV dan fold yang memburuk. Ringkasan tambahan tidak otomatis memilih resep berbeda atau mengubah loss. Test historis sudah dilihat berulang kali, sehingga perbandingan berikutnya bersifat eksploratif. E43 belum dapat disebut lebih baik sebelum hasil sah tersedia. Hash/config/folder E43 berbeda dan loader menolak pooling resep lama; resume lokal hanya berlaku selama file runtime tersedia.

## Dasar penelitian

**Taghipour & Ng (2016), [A Neural Approach to Automated Essay Scoring](https://aclanthology.org/D16-1193/)**, memberikan dasar mean-over-time pada AES. [Repositori penulis](https://github.com/nusnlp/nea), khususnya [`models.py`](https://github.com/nusnlp/nea/blob/master/nea/models.py) dan [`my_layers.py`](https://github.com/nusnlp/nea/blob/master/nea/my_layers.py), menyediakan agregasi `MeanOverTime` sebagai alternatif attention pada model recurrent, termasuk varian bidirectional. Paper dan kode ini mendukung metode agregasi; tidak membuktikan E43 akan unggul.

**Dong, Zhang & Yang (2017), [Attention-based Recurrent Convolutional Neural Network for Automatic Essay Scoring](https://aclanthology.org/K17-1017/)**, menjadi rujukan attention untuk AES. **Zhang & Litman (2018), [Co-Attention Based Neural Network for Source-Dependent Essay Scoring](https://aclanthology.org/W18-0549/)**, menjadi dasar pencocokan sumber–esai. Arsitektur dan tingkat representasi paper berbeda; E43 menguji mean setelah matching token pada fusion RoBERTa–BGE.

Scalar Mix dengan gamma tetap memakai dasar **Peters dkk. (2018), [ELMo](https://aclanthology.org/N18-1202/)** dan implementasi [AllenNLP ScalarMix](https://github.com/allenai/allennlp/blob/main/allennlp/modules/scalar_mix.py). Nilai LR/hidden/epoch dipertahankan dari kontrol, bukan optimum yang ditetapkan paper. [Verifikasi teknis](../verifikasi/verification_e43_native.md).
