# Latihan berangka dengan acuan AI

> Dokumen tahap terdahulu. Angka dan keputusan dibaca dalam konteks versi ini; untuk kondisi aktif gunakan [status proyek](../../STATUS_PROYEK.md) dan [panduan AI](../../PANDUAN_AI_PROYEK.md).

Mode ini dibuat atas pilihan pengguna: **latihan berangka dengan acuan AI yang jelas sumbernya**. Model benar-benar menjalankan forward, backward dan pembaruan parameter. Empat metrik mengukur kecocokan dengan usulan AI pada transkripsi draf, bukan akurasi terhadap guru. Angka mekanik merupakan simulasi pada transkripsi yang belum diverifikasi.

## Mulai di Colab

1. Buka `notebooks/02_latihan_acuan_AI.ipynb` dari branch `codex/aes-learning`. Notebook ini sudah memilih `MODE="ai_learning"`; notebook 01 tetap default inspeksi.
2. Gunakan runtime Python 3.12 (runtime 2026.07), CPU. Lock notebook menggunakan PyTorch CPU; memilih T4 saja tidak mengubah paket menjadi versi CUDA.
3. Jalankan cell dari atas. Saat pemilih file muncul, upload **Review_Esai_AI_Lengkap.xlsx** dan **rubrik_multitrait_DRAF.json** yang menjadi sumber review. Jangan upload file berisi teks saja untuk mode ini. Jika nama workbook berbeda, ubah EXCEL_FILENAME sebelum menjalankan cell pertama.
4. Tunggu setup dan pengunduhan dua checkpoint pada penggunaan pertama. Progress training menunjukkan skenario dan epoch. RUN_ENCODERS menampilkan forward sebelum latihan; mematikannya tidak menonaktifkan model asli yang wajib dipakai untuk training.
5. Baca acuan AI, teks → token/window → vektor → proyeksi/concat → empat output, kemudian tabel 12 baris (tiga skenario × empat trait). Pilih TRACE_SCENARIO/TRACE_TRAIT untuk menelusuri perhitungan satu baris evaluasi.

Tab `Untuk_CSV_Guru` tetap diimpor untuk seluruh teks/provenance. Target latihan dibaca **terpisah** dari empat `ai_usul_*` di `Ringkasan_AI`, dalam memori. `score_*`, `confirmed`, `ready_for_training`, dan identitas tidak berubah. Workbook harus memiliki `Data_asli`, `Ringkasan_AI`, dan hash rubrik di `JSON_asli!B3`. Teks, ID, versi, kategori dan hash diperiksa; kesalahan pasangan tidak diabaikan.

## Apa yang dilatih dan dibandingkan

Delapan teks asli yang tidak diketahui terpotong dipakai; dua `continuation_unassigned`/`AI_RECONSTRUCTION_SIMULATION` dikecualikan dengan log. Teks rekonstruksi tidak dipakai. Kelompok kemiripan usulan dijaga utuh pada pembagian 50% train / 50% validation, seed 2026. Kelompok ini **belum bukti identitas siswa**, sehingga split latihan tidak dianggap manifest penelitian yang sudah sah. Kandidat duplikasi lintas kelompok menyebabkan latihan berhenti, tanpa mengganti split menjadi random.

S1, S2 dan S3 tetap arsitektur kontrak, dengan checkpoint/revision yang sama. Setiap skenario dimulai ulang dari checkpoint awal, seed 42, kandidat A tetap, dua epoch. Tidak ada tuning, early stopping, atau pemilihan bobot menggunakan validation pada latihan ini. Prediksi validation baru dibuat sesudah seluruh epoch tetap selesai. Empat esai train menghasilkan batch terakhir parsial (effective batch maksimum delapan esai), bukan delapan chunk. Dua epoch kecil ini hanya memiliki dua optimizer step; pemanasan LR memakai satu step sehingga satu step berikutnya benar-benar memperbarui bobot.

RMSE/Pearson memakai prediksi kontinu asli; QWK/AC2 memakai salinan kategori setelah clipping dan nearest category, midpoint ke kategori lebih rendah. Seluruh kategori 1–5 selalu dipertahankan. Nilai undefined tetap dijelaskan. Jangan menambahkan epoch atau mengganti split hanya agar angka terlihat bagus. Data sangat sedikit dan acuan AI tidak menjadi standar kebenaran siswa.

Contoh membaca angka: Pearson bisa tinggi saat prediksi mengikuti urutan acuan AI tetapi semua prediksi terlalu rendah. RMSE akan tetap besar. QWK nol dapat terjadi bila seluruh prediksi kategori menjadi satu kategori. AC2 berbeda karena menggunakan definisi peluang kesepakatan yang berbeda; membandingkan angkanya dengan QWK tidak membuktikan kualitas model.

## Bukti dan berkas lokal

`outputs/ai_learning/<hash>/manifest.json` mencatat sumber, split dan scope. Setiap skenario menyimpan checkpoint, history, result dan `validation_against_ai.csv`. `outputs/ai_learning_report.json` memuat tabel metrik, pasangan/perhitungan, shapes dan contoh elemen vektor setelah training. MSE train sebelum/sesudah, step dengan LR aktif, dan perubahan sampel parameter membantu membuktikan pembaruan bobot; bukan ukuran akurasi penelitian. Cache hanya embedding BGE frozen; RoBERTa tetap differentiable saat training.

Hasil lengkap dapat dipakai ulang bila hash data/workbook/rubrik, konfigurasi, device, dan checkpoint cocok; notebook menyatakan reuse tersebut, tidak mengklaim training baru. Ubah `ai_learning.reuse_completed=false` pada konfigurasi lokal untuk menjalankan ulang dari awal. Manifest latihan dan hasilnya terpisah dari `runs/proposal`; checkpoint AI ditolak oleh CLI penilaian siswa. CLI penelitian tetap membutuhkan skor/peninjauan guru serta protokol 80/20 dan repeated CV yang sah.

Perbaikan trainer juga menangani run hanya satu optimizer step: warmup dibatasi maksimum `total_steps - 1` agar LR tidak nol sepanjang latihan. Ini keputusan eksekusi yang dicatat, bukan klaim parameter tambahan dari proposal. Final refit tanpa validation kini menyimpan `best_validation_mse=null`, bukan mencari kolom yang tidak ada.

Berkas privat (teks, workbook, rubric sumber, hasil per esai, bobot, notebook terisi) tidak dipublikasikan. Repo menyediakan kode/notebook kosong dan fixture teknis saja.
