# E17D Concat dan horizon scheduler refit

[Notebook](../../eksperimen/03_tiga_input_multi_trait/notebooks/AES_Reset_E17D_S3_Concat_RefitHorizon25_Native_Learnable_GPU.ipynb) · [Buka di Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex%2Fe17d-refit-horizon25/eksperimen/03_tiga_input_multi_trait/notebooks/AES_Reset_E17D_S3_Concat_RefitHorizon25_Native_Learnable_GPU.ipynb)

Model, target, input, split utama, seed dan resep optimizer mengikuti E17C. Durasi latihan refit mengikuti ceil median best epoch CV baru. Horizon scheduler refit adalah 25 epoch, dengan warmup linear 10%. Refit tidak dipaksa menjadi 4 epoch.

## Menjalankan

1. Pilih GPU T4 pada runtime Colab baru.
2. Jalankan sel berurutan; pemilihan dataset dan verifikasi checksum tersedia dalam notebook.
3. Bawaan menjalankan 3 fit diagnostik development, 5 fit CV, lalu 1 refit utama dan evaluasi test. Diagnostik dapat dinonaktifkan melalui RUN_DEVELOPMENT_DIAGNOSTIC, dengan status dilaporkan.
4. Baca Excel 05: Riwayat Training memuat horizon, warmup, langkah scheduler, LR awal/akhir per kelompok dan epoch. Diagnostik Scheduler Dev memuat perbandingan kebijakan pada development. Excel 04 dan paket lengkap mengikuti struktur E17C.

## Diagnostik development

Satu pembagian kelompok development menahan outer holdout. Inner split di dalam outer training memilih epoch tanpa membaca outer holdout. Dua model diinisialisasi kembali dengan seed sama, dilatih pada outer training yang sama selama epoch yang sama, dan hanya berbeda horizon scheduler: durasi fit atau 25. Keduanya dievaluasi pada outer holdout. Ini diagnostik satu split, bukan OOF atau pengganti CV. JSON privat menyimpan indeks, prediksi dan riwayat kedua kebijakan. Test penelitian tidak ikut memilih epoch atau kebijakan.

CV utama tetap memakai horizon 25 epoch; perubahan refit tidak menguji peningkatan OOF. Perubahan dipilih sebelum evaluasi test. Jangan menjadikan test yang sudah berulang kali dibaca sebagai konfirmasi baru yang independen.

## Pemeriksaan publikasi

Struktur notebook, pembersihan output, preservasi blok arsitektur/optimizer/metrik E17C, tidak adanya pemanggilan cosine similarity, dan simulasi scheduler diperiksa. Dengan 134 langkah/epoch dan durasi refit 4 epoch: horizon 4 memiliki 536 langkah dan 53 langkah warmup; horizon 25 memiliki 3.350 langkah dan 335 langkah warmup. Pengali LR setelah langkah 536 adalah 0 vs 0,933333.

Runtime eksekusi tidak tersedia pada sesi penyusunan: sintaks melalui interpreter Python, eksekusi semua sel, forward/backward encoder pretrained, ekspor Excel aktual dan training GPU belum diverifikasi. Belum ada hasil penelitian E17D.

Data siswa, checkpoint dan hasil tetap privat di runtime pengguna. Notebook publik hanya memuat kode tanpa output.
