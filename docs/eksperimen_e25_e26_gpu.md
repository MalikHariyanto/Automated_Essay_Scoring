# E25 dan E26: eksperimen lanjutan S3, Colab GPU

Dua notebook mandiri diturunkan dari E24 pada commit `f3d25c4`. Source E23/E24 tetap tersedia. Hasil E25/E26 sudah ditambahkan ke buku tiga variabel; [audit hasil](review_e25_e26_study.md) memisahkan RMSE, jumlah prediksi terdekat dan variasi fold.

| Komponen | E25 | E26 |
| --- | --- | --- |
| Nama | E24 + gated fusion | BiGRU + attention pooling |
| Tiga input pada semua skenario | soal, jawaban_referensi, response_text | Sama |
| Label | rater0–100 | Sama |
| Format | single_sequence | Sama |
| S1 | RoBERTa full fine-tuning + linear768→1 | Sama |
| S2 | BGE frozen + MLP1024→256→1 | Sama |
| Pooling S3 | Attention token jawaban E24,768→64→1, ringkasan768 | Token768→BiGRU64/arah→urutan128→attention128→64→1→ringkasan128→proyeksi768 |
| Fusion S3 | Gate skalar per jawaban, concat tetap1024 | Concat768+256 mengikuti acuan, tanpa gate |
| Head S3 | MLP1024→256→1 | Sama |
| LR RoBERTa S1/S3 | 5e-6 | Sama |
| LR proyeksi/head/layer baru | 1e-4 | Sama |
| Validasi dan training | 5fold pada development; maksimum25epoch, patience5 | Sama |
| Batch | microbatch1, efektif8 melalui akumulasi gradien | Sama |
| Seed | training42, split2026 | Sama |
| Penyimpanan | Lokal runtime, dua slot checkpoint per fold/refit; tanpa Drive | Sama |

## E25: bobot kontribusi dua encoder

RoBERTa menghasilkan ringkasan jawaban768 dari attention E24, kemudian LayerNorm. Cabang BGE frozen1024 diproyeksikan menjadi256 dan dinormalisasi. Gate menerima concat kedua fitur: Linear1024→32, tanh, Linear32→1, sigmoid. Satu nilai g per jawaban mengatur kontribusi dua cabang:

`concat = [2g × RoBERTa768, 2(1−g) × BGE256]`

Faktor2 menjaga concat awal identik dengan E24 saat g=.5. Lapisan terakhir gate diinisialisasi nol agar kondisi awal ini berlaku. Setelah training, g dapat berbeda antarjawaban. Ini adaptasi gated concat dua encoder teks, bukan replikasi persamaan GMU dan bukan bukti kausal bahwa suatu encoder lebih benar. Head dan proyeksi awal E24 tetap; modul gate dibuat sesudahnya.

Gate dilatih pada kelompok head1e-4. Inisialisasi nol pada lapisan akhirnya berarti lapisan hidden gate mulai mendapat gradien setelah lapisan skor gate bergerak dari nol; ini disengaja, bukan encoder frozen tambahan.

## E26: attention setelah BiGRU

E23 lama mengambil dua state akhir GRU. E26 mengambil **keluaran di setiap langkah token** dari GRU dua arah, kemudian memberi bobot attention pada urutan tersebut. Ringkasan berbobot128 diproyeksikan ke768 agar LayerNorm, concat dengan BGE256 dan MLP berikutnya tetap mempunyai dimensi acuan.

Mask memilih posisi token jawaban dari `answer_positions`, bukan kesamaan ID kata. Konteks soal/referensi tetap dibaca self-attention RoBERTa. Packing mengecualikan padding dari GRU; attention bertopeng dihitung dengan softmax FP32. Bobot dipetakan kembali ke posisi token asli untuk Excel; konteks/special/padding bernilai nol. Ini bukan attention atas satu vektor akhir dan bukan urutan768koordinat fitur.

## Cara menjalankan

1. Buka notebook di **web Colab biasa**.
2. Pilih **Runtime → Change runtime type → T4 GPU**.
3. Klik **Run all**, lalu unggah `fix_dataset_300.xlsx` pada Choose Files.
4. Cell terakhir mengunduh ZIP lima Excel. Unduh sebelum runtime berakhir.

Tidak perlu clone repository, menginstal ekstensi editor atau menghubungkan Drive. Folder E25 dan E26 berbeda dan seluruh S1/S2/S3 dilatih baru. Resume hanya untuk notebook/config yang sama dan folder lokal yang masih ada. Jangan menyalin checkpoint E23/E24 ke folder baru; arsitektur dan fingerprint berbeda.

Lima Excel mengikuti acuan. E25 menambah `Gate Fusion S3`, `Pengali RoBERTa S3` dan `Pengali BGE S3`, bersama bobot attention E24. E26 menambah `Ringkasan BiGRU Attention S3` dan bobot attention token. Fitur dan concat yang diekspor berasal dari forward aktual. Tabel300jawaban adalah inspeksi representasi; evaluasi penelitian tetap pada60test. Bobot model tidak ikut ZIP ringkasan.

## Rujukan dan batas verifikasi

[GMU, Arevalo dkk.2017](https://openreview.net/forum?id=Hy-2G6ile), dengan [implementasi resmi penulis](https://github.com/johnarevalo/gmu-mmimdb), menjadi dasar konsep pembobotan fusion. Fullpaper GMU belum berhasil diakses dalam sesi telaah; jangan menganggap gate skalar ini replikasi eksperimen AES dari paper tersebut. [Dong dkk.2017](https://aclanthology.org/K17-1017/) dan [PaSTO-GNN §3.6](https://doi.org/10.3389/frai.2026.1842542) mendukung konsep peringkasan urutan berbobot; kombinasi pada token RoBERTa ini merupakan adaptasi.

Pengujian teknis memeriksa source mandiri, dimensi, mask, gradient/frozen, learning rate, kontrol S1/S2, state_dict reload dan eksekusi seluruh cell hingga ekspor/resume dengan Transformer kecil berbobot acak pada CPU. Guard CUDA pada source publik tetap aktif. Ini tidak sama dengan menjalankan pretrained RoBERTa/BGE pada GPU atau memperoleh hasil penelitian300jawaban. Bukti disimpan lokal di `outputs/e25_e26_verification/`.

Hasil verifikasi:12pengujian lulus,0failed/errors/skipped/xfailed. Kedua notebook menjalankan seluruh30codecell,15CV+3refit dan lima Excel, kemudian dijalankan pada kernel baru dengan hash checkpoint final tetap. Ekspor gate E25 dan attention/ringkasan128 E26 diperiksa. Satu warning deprecation Jupyter tidak memengaruhi hasil. Semua ini memakai fixture24baris dengan Transformer acak di CPU; GPU/CUDA/AMP asli belum dijalankan dalam verifikasi ini.
