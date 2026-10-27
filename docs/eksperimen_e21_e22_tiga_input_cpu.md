# E21 dan E22: tiga input semua skenario, CPU, penyimpanan lokal

Dua notebook baru mengikuti E6 dengan tiga kolom input: `soal`, `jawaban_referensi`, dan `response_text`. `rater` adalah satu label skor guru 0–100; tidak masuk tokenisasi. Email dan hubungan duplikat digunakan untuk kelompok split, bukan fitur encoder. S1/S2/S3 dilatih ulang pada masing-masing eksperimen; tidak mengambil hasil kontrol eksperimen lama.

| Pengaturan | E21 | E22 |
| --- | --- | --- |
| Format input seluruh encoder | Satu urutan tiga teks | Pasangan (soal + referensi, jawaban) |
| LR encoder RoBERTa S1/S3 | 5e-6 | 5e-6 |
| LR head / proyeksi | 1e-4 / 1e-4 | 1e-4 / 1e-4 |
| S1 | RoBERTa full fine-tuning + linear | Sama |
| S2 | BGE-M3 frozen + MLP | Sama |
| S3 | RoBERTa fine-tuned + BGE frozen + fusion MLP | Sama |
| Perangkat | CPU float32, dua thread | Sama |
| Simpan otomatis Google Drive | Tidak | Tidak |

Head S1 tetap linear sesuai pilihan pengguna. S2 memakai MLP1024→256→1 (GELU/dropout). S3 memakai LayerNorm RoBERTa768, proyeksi BGE1024→256 + LayerNorm, concat1024 dan MLP1024→256→1. Tidak menggunakan LoRA atau fitur linguistik tambahan. Encoder BGE frozen tidak mempunyai kelompok LR; proyeksi/head tetap dilatih.

## Satu perubahan metode yang bisa dilacak

E21 menyusun token konteks `Soal: ... Jawaban referensi: ...`, penanda `Jawaban siswa:`, dan token jawaban dalam satu urutan. E22 memasukkan konteks sebagai teks A dan penanda/jawaban sebagai teks B melalui `build_inputs_with_special_tokens(context_ids, answer_ids)` milik tokenizer.

Kedua notebook berisi implementasi fungsi yang sama; pilihan `INPUT_FORMAT` menentukan jalurnya. Nama file/folder, identitas sumber dan pilihan format berbeda; hyperparameter, model, loss, split dan evaluasi sama. Isi, prefix, anggaran konteks192/soal64, panjang window RoBERTa512/BGE1024, overlap128 dan starts jawaban sama. Overhead terbesar pasangan dicadangkan untuk kedua format, sehingga E21 mempunyai dua slot kosong tambahan pada window penuh. Ini menjaga cakupan jawaban identik, alih-alih memberi salah satu eksperimen token jawaban lebih banyak.

Untuk RoBERTa dan XLM-R, pasangan tidak menambahkan token-type embedding A/B seperti BERT. Perbedaannya adalah separator khusus dan posisi; attention tetap membaca seluruh token yang tidak tertutup padding. Jadi E21 juga sudah dapat memperhatikan soal/kunci. E22 menguji format pemisahan, bukan menambahkan informasi yang tidak tersedia pada E21. Tidak ada jaminan konteks pair memperkecil RMSE.

## Prosedur yang sama

Seed training42, split_seed2026, development/test80/20, lima fold pada development, maksimal25epoch/fold, patience5, microbatch1/batch efektif8, AdamW, warmup10% dan scheduler linear. Refit ditetapkan dengan ceil median epoch terbaik CV. Pemilihan epoch memakai RMSE validasi; data uji baru dievaluasi sesudah ketiga checkpoint final dikunci. MSE memakai target0–1; RMSE/Pearson memakai prediksi mentah dalam poin0–100. QWK/AC2 memakai salinan kategori integer0–100, tie pembulatan ke kategori lebih rendah.

Tiga teks wajib tersedia pada training dan prediksi. Truncation soal/referensi dicatat; seluruh token jawaban mendapat window dengan konteks diulang. Aturan split mempertahankan kelompok email/duplikat; soal dapat muncul pada kedua bagian sehingga tidak menguji generalisasi ke soal baru. Test lama sudah pernah diperiksa selama pengembangan; hasil lanjutan bersifat eksploratif.

## Menjalankan dan menyimpan hasil

Buka `AES_300_E21_Tiga_Input_CPU.ipynb` atau `AES_300_E22_Tiga_Input_ContextPair_CPU.ipynb`, lalu jalankan dari atas. Unggah `fix_dataset_300.xlsx` saat diminta. Kode lengkap tertanam; tidak perlu clone/import repository. Di editor dengan runtime Colab, pilih kernel Colab; kode tetap memilih CPU secara eksplisit.

Tidak ada mount, autentikasi, API atau autosave Drive. Folder default di Colab adalah `/content/AES_300_E21_Tiga_Input_CPU` dan `/content/AES_300_E22_Tiga_Input_ContextPair_CPU`. Override `AES300_PERSISTENT_DIR` dari notebook lama diabaikan; workspace di bawah `/content/drive` ditolak sebelum penulisan. Dataset, cache, checkpoint, metadata dan Excel berada di disk runtime. Dua slot progres lokal dipakai ulang per fold/refit tanpa file baru setiap epoch atau penghapusan otomatis. Checkpoint lokal tetap memerlukan ruang disk runtime.

Notebook yang sama dapat melanjutkan progres selama folder lokal masih ada; kernel baru dengan filesystem yang sama dapat memakainya lagi. Runtime Colab dihapus/diganti berarti file lokal hilang. Cell terakhir mengunduh ZIP lima Excel. ZIP laporan tidak memuat bobot: jika model ingin dipakai lagi, unduh manual `S1/final.pt`, `S2/final.pt`, `S3/final.pt` dan metadata run melalui file browser sebelum mengakhiri sesi. Tidak ada backup Drive tersembunyi.

Lima laporan memuat data/tiga teks dan label, pembagian data/fold, token/segmen/cakupan dan format, vektor awal/akhir, vektor kata/kalimat, prediksi guru/model, empat metrik, pengaturan termasuk format input, serta seluruh riwayat fold/epoch. Pemanggilan ulang cell ekspor membaca hasil yang tersimpan tanpa training/test ulang.

## Provenance dan batas bukti

Builder mengambil source bersih E6 tiga input pada commit8380355, bukan notebook yang kemudian disimpan ulang oleh pengguna di Colab. Notebook lama, hasil penelitian dan buku HTML tidak diubah. Dua salinan notebook baru di direktori utama dan `aes-learning/notebooks/experiments/` harus byte-identik.

Rujukan encoder: [Liu dkk., RoBERTa](https://arxiv.org/abs/1907.11692) dan [Chen dkk., BGE-M3](https://arxiv.org/abs/2402.03216). Ini ablation format input proyek, bukan klaim kedua paper menguji konfigurasi E21/E22 pada dataset ini. BGE yang digunakan adalah dense CLS, bukan tiga keluaran retrieval sekaligus.

## Hasil pemeriksaan

17 pemeriksaan lulus (0 failed, 0 skipped, 0 xfailed; satu warning pustaka). Lima belas pemeriksaan mencakup identitas/determinisme source, kesetaraan implementasi E21/E22 selain format dan identitas, larangan Drive, isi dan cakupan token yang sama, perubahan fitur saat soal/referensi berubah, independensi fitur dari label, isolasi cache antarformat, forward/backward, head/LR/freeze/proyeksi, serta alignment token/vektor kata kedua encoder.

Dua pemeriksaan lainnya menjalankan seluruh30codecell masing-masing notebook dengan Transformer kecil berbobot acak:15fitCV+3refit, evaluasi, scoring, ekspor fitur/kata/kalimat dan lima Excel. Kernel baru kemudian membuka folder lokal yang sama; checkpoint final dan hasil test tetap sama. Ini fixture teknis dengan24jawaban buatan, bukan training/evaluasi pretrained pada300jawaban siswa. Akses tambahan ke tokenizer Hugging Face asli pada mesin pemeriksaan ditolak proxy403; tidak diklaim sebagai pemeriksaan tokenizer/bobot pretrained asli.

Dataset unggahan300baris diperiksa secara terpisah tanpa training: tiga input/label lengkap, development240/test60 dan anggota lima fold identik antara E21/E22. Tidak ada hasil RMSE baru yang diisi dari fixture. Command, versi pustaka, hash source, hasil terperinci dan bukti eksekusi disimpan privat di `outputs/e21_e22_*`.
