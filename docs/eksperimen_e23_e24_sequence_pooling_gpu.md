# E23 BiGRU dan E24 attention pooling — turunan E21, GPU lokal

Kedua notebook mandiri mengikuti E21 GPU pada commit `ef712b4c035d96178d9baa67caa7453ac9fd8738`. E21/E22 dan notebook sebelumnya tidak ditimpa. Perubahan arsitektur hanya pada S3; seluruh skenario tetap dilatih baru pada masing-masing eksperimen.

| Pengaturan | E23 | E24 |
| --- | --- | --- |
| Notebook | `AES_300_E23_Tiga_Input_BiGRU_GPU.ipynb` | `AES_300_E24_Tiga_Input_AttentionPooling_GPU.ipynb` |
| Input semua skenario | `soal`, `jawaban_referensi`, `response_text` | Sama |
| Label | `rater`, satu skor 0–100 | Sama |
| Format input | `single_sequence`, mengikuti E21 | Sama |
| S1 | RoBERTa full fine-tuning + linear | Sama |
| S2 | BGE-M3 frozen + MLP | Sama |
| S3 RoBERTa | Token jawaban → BiGRU, hidden64 per arah → state akhir128 → proyeksi768 | Token jawaban → additive attention768→64→1 → weighted sum768 |
| S3 BGE | Dense BGE1024 → proyeksi256 + LayerNorm | Sama |
| Fusion | RoBERTa768 + LayerNorm, concat BGE256 → MLP1024→256→1 | Sama |
| LR encoder S1/S3 | 5e-6 | Sama |
| LR proyeksi/head/layer baru | 1e-4 | Sama |
| CV/training | Lima fold, maksimum25epoch, patience5, seed42/split_seed2026 | Sama |
| Runtime | GPU CUDA, AMP sesuai E21 | Sama |
| Penyimpanan | Lokal runtime; tanpa mount/autosave Drive | Sama |

E23 tidak memakai attention pooling. E24 tidak memakai BiGRU. Jangan menyebut kedua notebook sebagai model gabungan BiGRU+attention. Pembedaan ini memungkinkan evaluasi tambahan layer secara terpisah. Kapasitas parameter tambahan berbeda: BiGRU+proyeksi menambah419.328 parameter pada hidden64; attention menambah49.280. Ini bukan perbandingan dengan kapasitas yang disamakan.

## Posisi token dan alur gradien

RoBERTa tetap membaca ketiga teks bersama, sehingga token jawaban dapat memperhatikan soal dan referensi. Layer baru hanya merangkum **posisi token jawaban**, menggunakan `answer_positions` hasil tokenizer. Token khusus, prefix, soal/referensi dan padding tidak ikut pooling. Penentuan posisi tidak memakai kesamaan ID token; kata yang sama di konteks dan jawaban tidak tertukar. Mask memperhitungkan padding kiri/kanan dan pemetaan beberapa window ke esai.

BiGRU membaca tensor `[window, token_jawaban, 768]`: 768 adalah jumlah fitur setiap token, bukan768langkah waktu. Urutan jawaban dipadatkan dengan `pack_padded_sequence`, lalu dua state akhir arah maju/mundur disambungkan menjadi128 dan diproyeksikan ke768 agar head E21 tetap dapat dipakai. Attention menormalkan logits hanya di sumbu token jawaban; bobot non-jawaban tepat nol dan total bobot tiap window satu. Softmax dihitung FP32. Kedua layer menerima gradien bersama encoder RoBERTa; BGE tetap frozen.

Jika jawaban membutuhkan beberapa window, hasil768 tiap window dirata-ratakan per esai mengikuti E21. Attention baru bekerja pada token, sehingga tetap bermakna untuk jawaban satu window. Tidak mengklaim bobot attention sebagai penjelasan kausal pemberian skor.

## Menjalankan dan membaca hasil

Buka notebook di web Colab biasa, pilih **Runtime → Change runtime type → T4 GPU**, lalu **Run all**. Unggah `fix_dataset_300.xlsx` melalui Choose Files. Guard CUDA berjalan sebelum training; notebook tidak beralih diam-diam ke CPU. Tidak memerlukan repository, ekstensi editor atau koneksi Drive.

Folder lokalnya terpisah: `/content/AES_300_E23_Tiga_Input_BiGRU_GPU` dan `/content/AES_300_E24_Tiga_Input_AttentionPooling_GPU`. Checkpoint memakai dua slot per fold/refit, tanpa auto-delete Drive atau satu file baru per epoch. Runtime yang dihapus dapat menghilangkan progres lokal; unduh sebelum sesi berakhir. Bila sesi yang sama masih menyimpan folder, notebook bisa dilanjutkan dari atas sesuai kontrak checkpoint. Jangan menyalin checkpoint E21 ke folder varian baru karena arsitektur/hash berbeda.

Cell terakhir mengunduh ZIP lima Excel seperti E21. `02_Vektor_Jawaban.xlsx` menambahkan **Pooling Jawaban S3**. E23 juga menyertakan **BiGRU S3 State Akhir**; E24 menyertakan **Bobot Attention S3** untuk setiap token jawaban/window. Pengaturan pooling tercatat di `05_Riwayat_Pelatihan.xlsx`; model S3 pada evaluasi menyebut nama varian. Bobot model tetap lokal, tidak ikut ZIP ringkasan lima Excel; unduh folder/checkpoint final terpisah bila diperlukan untuk inferensi berikutnya.

## Sumber dan batas kesimpulan

[PaSTO-GNN, Alhothali2026](https://doi.org/10.3389/frai.2026.1842542) menggabungkan RoBERTa yang diadaptasi dan embedding semantik frozen, BiGRU dan temporal attention pada kalimat (§3.6 dan Tabel6). [Dong, Zhang & Yang2017](https://aclanthology.org/K17-1017/) mendukung pembobotan kata/kalimat untuk AES. Penggunaan pada token jawaban S3, pilihan hidden64, serta proyeksi state BiGRU ke768 adalah adaptasi eksperimen ini. Tidak mengadopsi GNN, LoRA atau head CORAL dari PaSTO. Tidak ada jaminan tambahan layer menurunkan RMSE.

Pilih konfigurasi dari CV dengan seed/fold yang sama; nilai test lama yang sudah dilihat bersifat eksploratif. S1/S2 sama secara source/init/forward bila seed dan perangkat sama, tetapi identitas runtime/perangkat/numerik tetap perlu dicatat. Notebook bukan bukti bahwa hasil penelitian baru telah diperoleh.

## Validasi implementasi

15 pengujian lulus (0 failed, 0 errors, 0 skipped, 0 xfailed; satu warning deprecation Jupyter). Pemeriksaan mencakup builder deterministik/source mandiri, hash/dua salinan notebook, S1/S2 dengan state/forward identik E21 pada seed yang sama, dimensi/gradien/LR, frozen BGE, state_dict reload, urutan token BiGRU, mask/padding kanan-kiri, bobot attention bernilai nol pada posisi non-jawaban dan berjumlah satu.

Kedua notebook menjalankan seluruh30codecell,15fitCV+3refit, scoring, ekspor fitur baru dan lima Excel. Kernel baru menjalankan notebook lagi, memakai progres/hasil tersimpan dan mempertahankan hash checkpoint. Laporan memuat pooling768, stateBiGRU128 atau bobot token attention yang jumlahnya satu per jawaban/window.

Eksekusi tersebut memakai Transformer kecil berbobot acak pada CPU dengan guard/perangkat diubah hanya pada fixture. Ini bukan training300jawaban dengan pretrained RoBERTa/BGE, bukan pengujian GPU/CUDA/AMP asli dan bukan bukti peningkatan RMSE penelitian. Source publik tetap meminta GPU. Bukti command, versi, hash, XML dan eksekusi lokal ada di `outputs/e23_e24_verification/`.
