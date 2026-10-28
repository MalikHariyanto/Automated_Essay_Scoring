# E6 · learning rate S3 5e-6 dibanding MLP acuan

> Dokumen tahap terdahulu. Angka dan keputusan dibaca dalam konteks versi ini; untuk kondisi aktif gunakan [status proyek](../../STATUS_PROYEK.md) dan [panduan AI](../../PANDUAN_AI_PROYEK.md).

ZIP unggahan SHA-256 `06b898f882e0ec6e9de2896d7567fa9fecef5bb33c1b151afddff1b84d9a7af2` berisi lima workbook, 42 sheet. Ini **bukan LoRA**: S3 memakai full fine-tuning RoBERTa + BGE-M3 frozen + head MLP, masukan jawaban saja. LR encoder S3 5e-6, LR encoder S1 1e-5, LR proyeksi/head tetap 1e-4. Batas 25 epoch, patience 5. S1/S2 dipakai ulang dari run acuan; ini bukan tiga pelatihan baru.

| Ukuran S3 | MLP acuan | E6 LR 5e-6 |
| --- | ---: | ---: |
| RMSE test ↓ | 6.7085398243 | 6.3743658693 |
| Pearson ↑ | 0.5537789466 | 0.5970433758 |
| QWK ↑ | 0.5096391079 | 0.5596014343 |
| Gwet AC2 ↑ | 0.9785969889 | 0.9802455662 |
| Rata-rata RMSE lima fold ↓ | 6.0516397334 | 5.9915302002 |
| SD RMSE antar-fold | 0.8260348400 | 0.8345548927 |

E6 S3 lebih dekat ke guru pada 38/60 jawaban, acuan 22/60, tanpa seri menurut batas beda numerik 1e-5 poin. Seluruh 60 ID dan skor guru serta data/split 300 jawaban, development 240, test 60, dan anggota lima fold cocok dengan acuan. Keempat metrik test dihitung ulang dari prediksi mentah dan cocok dengan Excel. Metrik fold dicocokkan dengan riwayat epoch terpilih, bukan dihitung ulang dari pasangan prediksi fold yang tidak disertakan. Prediksi S1/S2 sama persis dengan acuan.

RMSE test terkecil numerik kini E6 S3 6.3744; selisih terhadap S1 acuan 6.3805 hanya sekitar 0.0061 poin. Pearson/QWK/AC2 S1 masih lebih tinggi. Hasil ini tidak membuktikan keunggulan signifikan. Rata-rata CV S3 E6 lebih kecil sekitar 0.0601 poin, tetapi SD sedikit lebih besar. CPU E6 berbeda dari GPU sumber acuan (tercatat di sheet Asal Model); perubahan ini belum mengisolasi pengaruh LR saja. Belum ada pengulangan seed. Data uji sudah dilihat selama pengembangan; hasil lanjutan bersifat eksploratif.

## Tampilan

Enam eksperimen menjadi kolom di dashboard serta matriks metrik lama. Ringkasan khusus E6 vs acuan memuat empat metrik, CV, perubahan LR, dan hitungan prediksi yang lebih dekat. Tab Fold membandingkan fold yang sama, mean dan SD. Tab Guru & prediksi membuka 18 kandidat prediksi untuk setiap jawaban, hanya dalam buku privat. Tab Excel menyediakan lima workbook E6 asli, termasuk seluruh 42 sheet, unduhan byte-identik, pencarian dan pemilih komponen vektor. Mode layar penuh, 30 blok kode MLP acuan serta sumber lama dipertahankan.

## Distribusi

Buku lengkap dengan seluruh sumber melewati batas 100 MiB per file GitHub. Arsip terenkripsi dibagi menjadi dua bagian di repository. File `downloads/Belajar_AES_MLP_Lengkap.html` adalah pengunduh kecil: simpan melalui Download raw file, buka, klik Unduh buku lengkap. Pengunduh memeriksa SHA-256 setiap bagian sebelum menyusun satu ZIP secara lokal. Ekstrak ZIP dan buka HTML dengan kunci sebelumnya; setelah itu offline. Tidak ada jawaban atau kunci plaintext di file publik. Versi agregat `docs/belajar_MLP_Spider.html` tetap satu HTML langsung dibuka tanpa kunci.

## Pemeriksaan

Dijalankan dari root repository dengan interpreter `outputs/dataset300_env_py313/bin/python`:

```bash
python -m pytest tests/test_mlp_dashboard.py tests/test_mlp_study.py tests/test_mlp_comparison.py tests/test_mlp_frozen.py -q --disable-warnings
node scripts/check_mlp_dashboard.mjs
node scripts/check_mlp_compare_fullscreen.mjs
node scripts/check_mlp_lr_package.mjs
```

- Unit/regresi: 21 passed, 0 failed/skipped/xfail.
- Browser dashboard: 8 kelompok lulus; mencakup enam kolom/empat metrik, 90 sel fold, 16 kombinasi viewport/tab, fullscreen/keyboard, 1.080 prediksi dan penandaan terdekat, seluruh 42 sheet E6 serta 47 sheet konteks pair, unduhan Excel byte-identik, dan 30 blok kode.
- Regresi fullscreen: 4 kelompok lulus, native/fallback dan perpindahan scroll/fokus.
- Paket: 2 kelompok lulus. Pengunduh diuji memakai berkas bagian asli melalui mirror HTTP lokal (bukan respons besar yang dimasukkan melalui protokol browser). ZIP yang diunduh byte-identik. HTML terenkripsi dibuka dengan kunci lama; seluruh sumber termuat dan 18 prediksi tampil setelah jaringan berikutnya diblokir. File-origin belum diuji karena kebijakan browser lingkungan; pengujian pembukaan menggunakan localhost.
- Semua sumber asli tetap di bundle privat; isi tiap sumber dibandingkan checksum. Tidak ada training baru. Bukti log/hash/versi berada di `outputs/mlp_study/e6_*`, tidak dipublikasikan bersama data individual.
