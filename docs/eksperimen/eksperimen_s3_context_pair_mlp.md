# S3 context pair + MLP

[Buka di Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_S3_ContextPair_MLP.ipynb) · [Notebook](../../notebooks/experiments/AES_300_S3_ContextPair_MLP.ipynb)

Eksperimen terpisah dari MLP acuan dan double frozen. Notebook mengikuti 30 blok kode dan 15 bagian utama yang sama: setup, dataset, split, tokenisasi, embedding, arsitektur, CV/refit/test, prediksi, vektor setelah training, lalu lima Excel. Tidak ada kode runtime yang diambil dari repo saat notebook dijalankan.

| Skenario | Input encoder | Bagian yang dilatih |
| --- | --- | --- |
| S1 | Jawaban siswa → RoBERTa | Encoder RoBERTa dan head linear |
| S2 | Jawaban siswa → BGE-M3 | MLP; encoder frozen |
| S3 | Soal/referensi + jawaban → RoBERTa; jawaban saja → BGE-M3 | RoBERTa, LayerNorm tambahan, proyeksi dan MLP; BGE frozen |

## Perubahan pada S3

Token pasangan dibentuk oleh tokenizer RoBERTa: `<s>konteks</s></s>jawaban</s>`. Teks konteks terdiri dari soal dan jawaban referensi, tanpa nilai guru, email atau identitas siswa. Satu pasangan konteks selalu mengikuti jawaban dari baris yang sama.

RoBERTa tetap menghasilkan 768 komponen. Vektor itu melewati LayerNorm, lalu disambung dengan 256 komponen proyeksi BGE. Head MLP tetap `1024 → 256 → GELU → Dropout → 1`. Learning rate encoder `1e-5`, proyeksi/head `1e-4`, maksimum 25 epoch, patience 5, microbatch 1 dan batch efektif 8 mengikuti acuan. Tidak menambahkan Ridge, gated fusion atau membekukan RoBERTa S3.

Kapasitas total RoBERTa tetap 512 token, termasuk special token pasangan. Batas konteks awal 192 token, dengan soal termasuk prefix maksimal 64 token; sisa anggaran konteks diberikan kepada referensi. Konteks yang panjang dipotong secara deterministik dan dilaporkan. Pada konteks penuh 192 token dan empat special token, satu window memuat 316 token jawaban; overlap 128 memberi langkah 188 token. Jika konteks lebih pendek, kapasitas jawaban bertambah. Seluruh token jawaban dicakup, dengan konteks yang sama diulang pada setiap window.

## Menjalankan

1. Buka notebook baru, pilih **Runtime → Change runtime type → T4 GPU**, lalu jalankan dari atas. Metadata notebook meminta GPU dan `REQUIRE_GPU=True` menghentikan model bila CUDA belum aktif. GPU NVIDIA lain juga diterima. Tabel pemeriksaan menampilkan nama GPU, status CUDA dan memori; embedding/training memakai CUDA, sedangkan tokenisasi/Excel tetap CPU. Mode audit tanpa model boleh CPU.
2. Sambungkan Drive ketika Colab meminta izin. Folder baru: `MyDrive/AES_300_S3_ContextPair_MLP_Privat`.
3. Pilih `fix_dataset_300.xlsx` jika diminta. Dataset pribadi tidak ditanam di repo. Enam header tetap: `Email address`, `response_number`, `response_text`, `soal`, `jawaban_referensi`, `rater`.
4. Untuk contoh prediksi, biarkan `NEW_ANSWER`, `NEW_QUESTION`, dan `NEW_REFERENCE` kosong. Ketiganya diambil dari satu baris data uji yang sama.
5. Untuk jawaban baru, isi ketiga variabel tersebut. Nilai guru tidak diperlukan untuk prediksi baru dan tidak diisi dengan angka pengganti.
6. Jalankan cell unduhan terakhir. `Hasil_AES_Ringkas.zip` berisi lima laporan Excel dengan nama yang sama seperti acuan.

Split tetap 80/20 dengan pengelompokan email/duplikat, dan lima fold hanya pada development. Bobot/epoch dipilih berdasarkan RMSE validasi; epoch refit berasal dari ceil median epoch terbaik lima fold. Model final dilatih pada development, lalu dinilai pada test. RMSE/Pearson memakai prediksi mentah, QWK/AC2 memakai salinan kategori; perbandingan nilai guru, prediksi dan pembulatan tetap disediakan.

Ini run baru. Folder dan fingerprint berbeda; checkpoint MLP/double frozen lama tidak dipilih untuk melanjutkan eksperimen baru. Untuk runtime yang terputus, jalankan notebook yang sama dari atas dengan Drive dan konfigurasi yang sama. Pemeriksaan kontrak checkpoint tetap mensyaratkan perangkat/versi numerik yang cocok; jangan menganggap berpindah GPU/CPU pasti bisa meneruskan checkpoint epoch.

## Hasil yang disimpan

| Laporan | Tambahan yang relevan |
| --- | --- |
| Data dan tokenisasi | Token pasangan S3, penanda konteks/jawaban/special, jumlah konteks yang terpotong dan cakupan token |
| Vektor jawaban | `RoBERTa S3 Pair Awal` sebagai pembanding dengan input sama sebelum fine-tuning; fitur final dan fusion tetap tersedia |
| Vektor kata dan kalimat | Vektor pretrained pasangan dan S3 setelah training; hanya token jawaban dipetakan ke kata, dengan attention ke konteks |
| Prediksi dan evaluasi | Nilai guru/prediksi/pembulatan, empat metrik test, seluruh 15 hasil fold |
| Riwayat pelatihan | Semua epoch, pilihan epoch dan konfigurasi, termasuk batas token konteks |

Vektor kalimat S3 diperoleh dengan memasangkan masing-masing kalimat dengan konteks yang sama. Ini analisis per kalimat, bukan vektor fusion per kata. BGE tetap memakai jawaban saja. Bandingkan fitur RoBERTa S3 final dengan **S3 Pair Awal**, agar perbedaan input tidak tertukar dengan efek training.

## Checkpoint: jangan hapus seluruh .pt

Notebook menyimpan progres ke Drive setiap epoch tanpa pembersihan otomatis atau akses Drive API tambahan. Seluruh file lama tetap ada dan perlu ditinjau manual. Penyimpanan dapat bertambah besar selama training.

| File | Aturan |
| --- | --- |
| `S1/final.pt`, `S2/final.pt`, `S3/final.pt` | Pertahankan. Ini bobot model final untuk prediksi/ekstraksi fitur. |
| Snapshot `.pt` yang dirujuk `progress_fold_N.json` atau `progress_refit.json` pada tahap aktif | Pertahankan kedua snapshot terbaru untuk resume dan cadangan. |
| `progress_fold_N_epoch_*.pt` untuk fold selesai | Kandidat hapus manual setelah `cv_fold_N.json` selesai, utuh dan dapat dibaca, serta fold itu tidak sedang berjalan. |
| `progress_refit_epoch_*.pt` untuk refit selesai | Kandidat hapus manual setelah `final.pt` dan catatan selesai cocok/utuh, serta model final berhasil dimuat. |
| Hasil `.json`, Excel dan arsip hasil | Pertahankan sebagai konfigurasi, manifest, hasil CV/evaluasi atau laporan. JSON bukan pengganti bobot. |

Hasil CV lengkap disimpan di JSON sehingga model fold tidak perlu dilatih lagi ketika run cocok diulang. Namun, JSON progres hanya menunjuk snapshot; tanpa `.pt` yang dirujuk, epoch aktif tidak dapat dipulihkan. Cocokkan **folder run dan skenario**, karena nama checkpoint yang sama bisa muncul di beberapa tempat. Memindahkan ke Sampah Drive belum tentu membebaskan kuota; hapus permanen hanya file yang sudah dipastikan tidak diperlukan.

## Batas pembandingan

S3 mendapat konteks tambahan sedangkan S1/S2 tetap menjadi kontrol jawaban-saja. Hasil menguji **S3 + konteks**, bukan mengisolasi keunggulan fusion saja. Untuk perbandingan jenis model dengan informasi setara, perlu eksperimen kontrol konteks yang sesuai. Hasil test lama sudah dilihat selama pengembangan; pilih kandidat melalui CV dan gunakan data baru yang belum disentuh untuk konfirmasi yang lebih kuat.

Belum ada RMSE penelitian baru sebelum pengguna menjalankan notebook pada dataset dan model pretrained sebenarnya. [Bukti pengujian](../verifikasi/verification_s3_context_pair_mlp.md).
