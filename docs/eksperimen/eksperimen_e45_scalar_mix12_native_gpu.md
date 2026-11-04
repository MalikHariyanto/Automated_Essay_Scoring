# E45 · E43 + Scalar Mix seluruh 12 layer · S3 native

[Buka Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E45_E43_S3_ScalarMix12_Native_Learnable_GPU.ipynb) · [Source](../../notebooks/experiments/AES_300_E45_E43_S3_ScalarMix12_Native_Learnable_GPU.ipynb).

Pengguna memilih E45 sebagai kandidat kedua untuk dijalankan bersama E44. Keduanya langsung berangkat dari E43. E44 menguji objective MSE+ListNet, sedangkan E45 mempertahankan MSE dan memperluas Scalar Mix empat layer terakhir menjadi seluruh 12 layer Transformer RoBERTa. Belum ada hasil penelitian E45; E43 tetap acuan (mean RMSE CV5,8876, RMSEtest5,1265, MAEtest3,7817).

## Mekanisme

RoBERTa mempunyai 12 blok Transformer dan mengembalikan 13 tensor hidden states: embedding awal pada indeks0, kemudian layer1–12 pada indeks1–12. E43 memilih empat tensor terakhir, layer9–12. E45 memilih seluruh12tensor Transformer. Embedding awal tidak dijumlahkan langsung, tetapi parameter embedding tetap ikut fine-tuning melalui encoder.

Untuk setiap posisi token, rumusnya `gamma × sum(softmax(logit_layer) × hidden_layer)`. Dua belas logit mulai0, sehingga bobot awal masing-masing1/12≈0,083333. Gamma tetap mulai1 dan dipelajari bersama logit pada LR1e-4. Jumlah parameter campuran13:12logit+satu gamma, bertambah tepat8dari E43. Softmax tetap berjumlah1; gamma adalah skala terpisah dan tidak dibatasi positif. Tidak ada normalisasi baru.

Model memeriksa `roberta.config.num_hidden_layers == scalar_mix_layers`, sehingga campuran benar-benar mencakup seluruh layer model aktual. Checkpoint penelitian tetap `flax-community/indonesian-roberta-base` dengan12layer. Jika jumlah layer tidak cocok, notebook berhenti dengan pesan yang jelas. Keluaran campuran tetap768dimensi. Token respons kemudian masuk BiGRU64perarah→128, referensi diproyeksikan ke128, co-attention dan matching640→128, masked mean tokenvalid→proyeksi768/LayerNorm. Proyeksi BGE1024→256/LayerNorm digabung dengan RoBERTa menjadi1024, lalu MLP256/GELU/Dropout0,1→satu nilai.

Seluruh inisialisasi bersama mengikuti E43. Logit berupa tensor nol, sehingga memperbesar ukurannya tidak mengonsumsi RNG tambahan. Prediksi awal normal akan berbeda karena campurannya berbeda. Sebagai pemeriksaan mekanisme, ketika delapan bobot pertama dibuat nol dan empat terakhir0,25, model E45 menghasilkan prediksi/fitur yang sama dengan E43 pada bobot bersama yang identik.

## Resep yang dipertahankan

| Bagian | E45 |
|---|---|
| X / y | `soal`, `jawaban_referensi`, `response_text` / `rater`0–100 |
| RoBERTa | Fine-tuned, LR seragam5e-6 |
| BGE-M3 | `BAAI/bge-m3`, frozen; proyeksi256 tetap belajar |
| Scalar Mix | Seluruh12layer RoBERTa, gammaawal1, LR1e-4 |
| BiGRU/co-attention/pooling | Hidden64perarah; matching128; masked mean1/Nvalid,padding0 |
| LR di luar encoder | Seluruhnya1e-4 |
| Loss | MSE target0–1, sama persis E43 |
| Optimizer | AdamW, decay0,01, warmup10%, scheduler linear, clipping1 |
| Batch | Microbatch1, efektif8, sampler E43 |
| Protokol | S3 saja,satu5-fold,maks25epoch,patience5 |
| Refit/test | Ceil median best epoch berdasarkan RMSE validasi,satu refit dan satu test |

Input, label, tokenisasi/window, split/fold, TRAIN_SEED42, seed per-fold, optimizer dan aturan pemilihan epoch sama E43. Tidak ada ListNet, activation recompute E44, cosine, S1/S2, loop multi-seed atau pelatihan ulang kontrol otomatis. Pengujian kedua kandidat tidak menggabungkan perubahan sebelum hasilnya diketahui.

E43 sudah meminta seluruh hidden states pada satu forward RoBERTa, sehingga E45 tidak menambah encoder atau forward baru. Ada tambahan operasi pencampuran/gradien dari delapan layer; biaya waktu/memori aktual tetap perlu diukur pada GPU. Tidak ada klaim overhead nol atau kecepatan setara dari fixture CPU.

## Pemakaian dan output

Pilih GPU T4→Run all→unggah `fix_dataset_300.xlsx`. Notebook184cell/89cellkode Python biasa, tanpa magic atau worker tersembunyi. Nama checkpoint dan kolom X/y ada pada sectionawal, `SCALAR_MIX_LAYERS = 12` pada section5, rumus dan guard jumlah layer pada section arsitektur. MSE/backward/CV tetap terlihat dalam fungsi `fit` yang sama dengan E43.

Lima Excel dalam `Hasil_AES_300_E45_E43_S3_ScalarMix12_Native_Learnable_GPU.zip`:

- Excel05→**Bobot Scalar Mix** memuat12bobot+gamma pada setiap checkpoint terpilih lima fold dan refit, total72baris. Layer ke-1–12, indeks0–11; embedding awal tidak mempunyai baris bobot. Setiap tahap mempunyai12bobot softmax berjumlah1 dan satu gamma yang diulang untuk keterbacaan.
- Excel05 tetap menyimpan history **Loss Training (MSE)**, Pengaturan Eksperimen dan **MAE dan Toleransi CV**. Tidak ada loss ranking atau gabungan E44.
- Excel04→**Prediksi Validasi OOF** berisi fold, ID jawaban, soal, nilai guru, prediksi, pembulatan dan error. Setiap jawaban development muncul satu kali; ID duplikat atau bercampur test ditolak. Ekspor diagnostik ini tidak mengubah pelatihan. Nilai Guru dan Prediksi tetap khusus test.
- Excel02 tetap menyimpan norma campuran sebelum/sesudah gamma, layer terakhir, vektor128/768/1024, bobot mean1/N dan co-attention.

Folder/hash/checkpoint E45 terpisah; loader menolak konfigurasi jumlah layer berbeda. E43/E44 tidak ditimpa. Unduh ZIP sebelum runtime dihapus. Pilih berdasarkan mean RMSE CV dengan MAE/toleransi sebagai konteks tradeoff. Test historis yang sudah berulang kali dilihat tetap eksploratif. Untuk dua run bersamaan, runtime GPU terpisah dapat menghindari perebutan memori satu perangkat; ketersediaan runtime mengikuti layanan Colab.

## Dasar penelitian dan batas klaim

[Peters dkk.(2018), Deep Contextualized Word Representations/ELMo](https://aclanthology.org/N18-1202/) memberikan dasar campuran layer berbobot dengan gamma. [AllenNLP PretrainedTransformerEmbedder](https://github.com/allenai/allennlp/blob/main/allennlp/modules/token_embedders/pretrained_transformer_embedder.py) menyediakan `ScalarMix(self.config.num_hidden_layers)` ketika `last_layer_only=False`, lalu memasukkan `transformer_output.hidden_states[1:]` untuk mengecualikan embedding awal. Implementasi tersebut diperiksa sebelum pemilihan kandidat E45.

Ini landasan metode/implementasi seluruh layer, bukan bukti12layer optimal untuk AES Indonesia atau fusion ini. Pemilihan rentang layer menjadi hipotesis E45. BiGRU/co-attention/mean/fusion tetap memakai dasar E43, termasuk mean-over-time Taghipour&Ng(2016). Belum ada hasil E45 yang membuktikan perbaikan atas E43.

[Verifikasi teknis](../verifikasi/verification_e45_native.md).
