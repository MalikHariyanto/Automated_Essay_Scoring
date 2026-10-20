# Hasil AES yang ringkas dan mudah dibaca

Keluaran akhir sekarang disatukan menjadi **lima Excel**. Proses S1/S2/S3, pembagian data, training, prediksi dan rumus metrik tetap sama. Perubahan ini mengatur cara menampilkan dan mengemas hasil.

## Unduh langsung dari Colab utama

Buka [Colab utama terbaru](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/03_AES_dataset_guru_300.ipynb) dan jalankan tahap-tahapnya berurutan. Pada bagian **15 · Unduh hasil Excel yang sudah rapi**, jalankan satu cell terakhir. Cell ini otomatis menyusun laporan dan mengunduh `Hasil_AES_Ringkas.zip` berisi lima Excel utama. Tidak perlu pindah notebook atau mengunggah ZIP untuk merapikannya.

Daftar file dan isinya ditampilkan sebelum unduhan; nama anggota ZIP diperiksa agar cocok dengan laporan yang baru dibuat dari sesi tersebut. Rincian teknis tidak ikut paket unduhan. Untuk mengunduh ulang pada sesi notebook yang sama, jalankan kembali cell terakhir saja; pengemasan tidak melatih atau mengevaluasi model ulang.

[Bukti uji unduhan Colab utama](verification_dataset300_main_download.md) mencakup cell unduh asli, isi ZIP, dan kesamaan prediksi/metrik pada fixture teknis.

## Isi paket utama

| File | Isi yang perlu dibaca |
| --- | --- |
| **01_Data_dan_Tokenisasi.xlsx** | Jawaban siswa dan nilai guru, pembagian data/fold, token RoBERTa dan BGE-M3. Padding tidak ditampilkan; special token tetap ada. |
| **02_Vektor_Jawaban.xlsx** | Seluruh komponen vektor tiap jawaban: RoBERTa sebelum/sesudah latihan, BGE-M3 dan fusion. |
| **03_Vektor_Kata_dan_Kalimat.xlsx** | Satu jawaban contoh, vektor kata/kalimat, lookup sebelum konteks dan perbandingan kata berulang. BGE frozen ditampilkan sekali. |
| **04_Prediksi_dan_Evaluasi.xlsx** | Mulai dari **Perbandingan Model**, lalu **Nilai Guru dan Prediksi**. Ada hasil tiap fold, ringkasan validasi dan prediksi satu jawaban. |
| **05_Riwayat_Pelatihan.xlsx** | Semua fold/epoch digabung menjadi satu sheet per model; pengaturan, rancangan model dan rujukan paper ikut dicatat. |

Setiap workbook diawali sheet **Panduan**. Nama kolom menjadi `ID Jawaban`, `Kata`, `Nilai Guru`, `Prediksi S1`, `RMSE (poin)`, dan `Komponen 1`, dst. `Komponen 1` adalah `v_0000` versi lama; hanya namanya berubah. J001 berarti jawaban baris pertama dan menjadi penghubung antarfile, bukan fitur model.

Semua koordinat vektor tetap tersimpan. Angka prediksi/metrik disalin dari hasil sumber, bukan dihitung ulang; format tampilan empat desimal tidak membuang presisi nilai sel. Angka kosong tetap kosong, bukan0. Pada tabel ringkasan validasi, rata-rata ± SD berasal dari sheet ringkasan lama; n menunjukkan jumlah fold yang metriknya dapat dihitung. SD antar-fold bukan interval kepercayaan.

Kolom lapisan internal seperti `dropout_eval`, `MLP_GELU256` dan `output_normalized` tidak dimasukkan ke paket utama. Rincian mentah tetap berada di folder privat `excel` run Colab sebagai catatan proses. Pesan Colab tidak lagi mencetak daftar panjang semua sheet/dimensi. Pemberitahuan dua bobot pooler BGE yang memang tidak digunakan disaring secara khusus; warning lain dan error tetap ditampilkan.

Jika baru sebagian tahap selesai, paket hanya berisi laporan yang datanya tersedia; tidak dibuat hasil evaluasi pengganti. ZIP dan Excel berisi jawaban/nilai pribadi, sehingga disimpan privat dan tidak diterbitkan ke GitHub.

## Membaca output Colab

Tabel Colab memakai judul, kolom, dan catatan cara membaca dalam Bahasa Indonesia. Contohnya:

| Nama internal lama | Yang ditampilkan |
| --- | --- |
| `occurrence_indices` | Posisi Kata A dan Posisi Kata B, dihitung mulai 1 |
| `same_token_ids` | ID Token Sama? — Ya/Tidak |
| `initial_max_abs_difference` | Disimpulkan menjadi Vektor Sebelum Konteks Sama? |
| `contextual_allclose_1e-6` | Vektor Setelah Konteks Sama? — Ya/Tidak |
| `contextual_cosine` | Kemiripan Setelah Konteks (Cosine), disertai cara membaca |
| `v_0000`, `v_0001`, ... | Komponen 1, Komponen 2, ... |

Sebelum konteks berarti vektor dari tabel embedding model; setelah konteks berarti model sudah memperhitungkan posisi dan kata di sekitarnya. Cosine mendekati 1 berarti arah vektornya sangat mirip, tetapi bukan bukti semua komponen identik. Kolom kesamaan memeriksa selisih per komponen dengan batas 0,000001. Batas numerik ini tidak berkaitan dengan tolerance skor AES. Tabel kosong menjelaskan bahwa belum ada pasangan kata berulang, bukan menyatakan vektornya sama atau berbeda.

Di tahap training, pesan menunjukkan skenario, epoch, loss latih, serta RMSE validasi. Loss MSE memakai target 0–1, sedangkan RMSE memakai poin skor asli 0–100; keduanya tidak dibandingkan langsung. Angka layar dibulatkan untuk dibaca; nilai lengkap tetap ada di Excel.

Notebook yang sudah terbuka di tab Colab tidak otomatis mengganti kode dan output lama. Buka notebook utama versi terbaru dari tautan di atas agar kode unduhnya ikut diperbarui.

[Bukti pengujian tampilan Colab](verification_dataset300_colab_readable.md) mencakup eksekusi kedua notebook dan perbandingan angka dengan versi sebelumnya.

## Cara menjelaskan alur ke dosen

“Teks jawaban diubah menjadi token, lalu direpresentasikan sebagai vektor. Saya membandingkan RoBERTa, BGE-M3, dan fusion keduanya. Model dilatih dengan validasi silang pada data latih, kemudian prediksinya dibandingkan dengan nilai guru pada data uji menggunakan QWK, Gwet AC2, RMSE dan Pearson.”

Tunjukkan file01untuk input/tokenisasi, file02–03untuk representasi, dan file04untuk hasil utama. File05adalah pendukung untuk menjelaskan proses latihan. Vektor bukan skor rubrik; komponennya tidak mempunyai arti tetap seperti “isi” atau “tata bahasa”.

## Metode yang dipertahankan

S1: RoBERTa Indonesia fine-tuned → mean CLS window768 → linear satu skor. S2: BGE-M3 frozen → dense CLS1024 + L2 → MLP1024→256→1. S3: RoBERTa768 + proyeksi BGE256 → concat1024 → linear satu skor. Tidak ada perubahan bobot, lapisan, optimizer atau cara model menghitung prediksi pada perapian laporan ini.

Dataset unggahan300respons mempunyai satu skor `rater`, bukan empat label rubrik; setiap skenario menghasilkan satu nilai. Rentang0–100 mengikuti notebook BGE-M3 unggahan. Hanya `response_text` masuk encoder; soal, kunci, email dan nilai tidak menjadi fitur. File bertopik metodologi penelitian ini tidak membuktikan populasi SMAkelasXI.

Email/duplikat tetap dikelompokkan. Pembagian yang telah dibuat adalah240development/60test, dengan5fold di development. Epoch dipilih dari RMSEvalidasi; refit memakai ceil median epoch pilihan. Semua refit selesai sebelum finaltest. Ekstraksi fitur seluruh jawaban dilakukan sesudah evaluasi final untuk inspeksi, bukan evaluasi penelitian baru.

Default5epoch/fold, microbatch1/effectivebatch8, AdamW, LRencoder2e−5/head1e−3, warmup10%, clipgrad1 tetap sama. RoBERTa memakai window512 termasuk2special, overlap128. BGE batas eksperimen1024 termasuk special. Loss memakai target0–1; RMSE/Pearson memakai skor mentah0–100. QWK/AC2 memakai salinan clip/nearestinteger dengan tie.5 ke bawah dan seluruh101kategori. Undefined tetap kosong dengan alasan.

Rujukan tetap berasal dari [koleksi pengguna](aes_dataset300_references.json), antara lain RoBERTa, M3-Embedding, AES hybrid dan evaluasi. Paper-paper tersebut tidak diklaim memakai arsitektur proyek yang identik. Bukti pengujian keluaran baru ada di [laporan verifikasi](verification_dataset300_readable.md).
