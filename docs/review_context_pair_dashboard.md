# Hasil konteks pair dan papan perbandingan semua eksperimen

24 Oktober 2026. ZIP unggahan konteks pair terverifikasi dengan SHA-256 `0473d3537a7766110bb1bee0be20b723a29c5777a3401b10faee4c556154e896`. Nama file unggahan sama dengan ZIP frozen terdahulu, tetapi isi/model berbeda: workbook ini menyatakan **RoBERTa context pair ikut dilatih; BGE-M3 jawaban saja tetap**, dengan batas konteks 192 dan soal 64 token. Head fusion tetap MLP, LR encoder 1e-5 dan proyeksi/head 1e-4, batas 25 epoch/patience 5. S1 linear dan S2 MLP tetap. Notebook eksekusi tidak disertakan bersama ZIP; kecocokan source runtime byte demi byte tidak diklaim.

## Hasil S3

| Eksperimen | Mean RMSE CV ↓ | SD antar-fold ↓ | RMSE test ↓ |
| --- | --- | --- | --- |
| Awal · 5 epoch | 10.95416448 | 1.98511746 | 11.72239770 |
| MLP acuan | 6.05163973 | 0.82603484 | 6.70853982 |
| Double frozen | 6.83634540 | 1.13522193 | 7.86108622 |
| Konteks pair | 6.09863492 | 1.00247796 | 6.73147456 |
| LoRA + MLP | 6.25086442 | 0.92787514 | 6.53049357 |

Konteks pair memiliki selisih RMSE test sekitar +0,0229 dan mean CV +0,0470 poin terhadap MLP acuan. Pada run ini RMSE belum membaik, tetapi selisih kecil tidak membuktikan konteks pair selalu merugikan. Secara keseluruhan S1 RMSE test terkecil 6,3805; hasil S1 MLP/konteks pair/LoRA sama dan tidak dihitung sebagai tiga pengulangan independen. Khusus fusion S3, LoRA test terkecil 6,5305, sedangkan mean CV S3 terkecil pada MLP acuan 6,0516.

## Pemeriksaan sumber

- Lima workbook dan **47 sheet** konteks pair terbaca. Empat metrik test diverifikasi ulang dari 60 pasangan guru/prediksi; pembulatan kategori, QWK/AC2 kuadratik 0–100 cocok. RMSE/Pearson memakai prediksi mentah.
- Sheet Hasil Tiap Fold dicocokkan dengan empat metrik pada epoch terpilih di riwayat; 15 fold lengkap, jumlah epoch dan minimum RMSE sesuai.
- Data 300 jawaban, split 240/60, 1.200 baris penggunaan lima fold, ID dan nilai guru test cocok dengan MLP acuan. Perbandingan versi lain bertumpu pada audit sebelumnya serta pembacaan workbook evaluasi yang sama.
- Hitungan skenario terdekat per jawaban dihitung satu kali; seri dipisahkan. Toleransi beda numerik 1e-5 poin dipakai untuk hitungan seri, bukan toleransi penilaian AES. Mean CV adalah rata-rata RMSE lima fold; SD memakai n−1, bukan confidence interval.

## Bentuk UI

Ruang **Semua eksperimen** menjadi tampilan awal. Tiga kartu memisahkan RMSE test terbaik, mean CV terbaik, dan fusion terbaik pada test. Pemilih metrik mengubah tabel utama tanpa menggabungkan empat metrik menjadi skor buatan. Eksperimen menjadi kolom; hasil terbaik/setara diberi warna sekaligus label teks. Empat metrik lengkap dapat dibuka melalui details.

Tab **Fold & kestabilan** membandingkan fold yang sama. Hijau berarti terbaik pada baris/fold itu, sedangkan bintang menandai fold terendah dalam eksperimen sendiri. Mean, SD dan minimum dijelaskan terpisah; satu fold tidak dijadikan model akhir. Untuk S3: acuan menang pada fold 1/3/4, konteks pair pada fold 2, frozen pada fold 5.

Tab **Guru & prediksi** memisahkan hitungan menang per jawaban dari RMSE. S3 frozen paling dekat pada 24 jawaban di run frozen meskipun RMSE S3 lebih besar; ini menjelaskan efek error besar pada RMSE. Pembaca dapat memilih salah satu dari 60 ID, membuka teks jawaban, melihat guru dan seluruh 15 kandidat prediksi dari lima eksperimen, serta label terdekat dan error absolut. Data individual hanya tersedia pada buku pribadi.

Tab **Excel eksperimen** menyediakan evaluasi awal, lima Excel MLP, lima frozen, lima konteks pair, dan lima LoRA. Nama file sama tetap dipisahkan menurut run. Paging, pencarian, delapan komponen per tampilan, serta unduhan XLSX asli dipertahankan. E1 source-only dan E6 yang belum disertai hasil final ditandai belum lengkap, tidak diberi peringkat final. Kode MLP 30 blok dan fullscreen seluruh ruang tetap tersedia.

## Bukti pengujian

Cwd root repo, interpreter `outputs/dataset300_env_py313/bin/python`:

```bash
python -m pytest tests/test_mlp_dashboard.py tests/test_mlp_study.py \
  tests/test_mlp_comparison.py tests/test_mlp_frozen.py -q --disable-warnings
node scripts/check_mlp_dashboard.mjs
node scripts/check_mlp_compare_fullscreen.mjs
```

- Unit/regresi: **21 passed, 0 failed, 0 skipped, 0 xfailed**, 0,27 detik. Empat kasus baru menguji seri, satu hitungan per jawaban, dan penolakan set test parsial.
- Browser dashboard: **8 kelompok lulus, 0 gagal**: nilai/arah metrik/seri, 75 sel fold dan kedua arti sorotan, 16 kombinasi viewport/tab, fullscreen/keyboard dan checksum, 900 prediksi individual, seluruh 47 sheet konteks pair serta lima pilihan sumber, komponen 1024 dan unduhan Excel byte-identik, serta 30 blok kode.
- Regresi LoRA: **8 kelompok lulus**, termasuk 60 metrik pada lima kolom, 48 kombinasi tampilan, 30 kurva dan 180 prediksi MLP/LoRA.
- Regresi fullscreen lama: **4 kelompok lulus, 0 gagal**, native/fallback, empat ruang lama, gulir, fokus, pemulihan posisi dan kode fullscreen. Ruang kelima diuji oleh dashboard.
- Ekspor HTML pribadi dibuka ulang; ZIP konteks pair dan seluruh sumber sebelumnya tetap termuat dan SHA-256 konteks sama. Output bukti ada di `outputs/mlp_study/dashboard_*.json/log`, tidak masuk Git.

Ini audit hasil tersimpan dan UI, tidak menjalankan training baru. Satu run per metode belum mengukur variasi seed/signifikansi. Test sudah dilihat selama pengembangan; pemilihan metode tetap lewat CV dan konfirmasi independen memerlukan data baru.

## Unduhan lengkap

Paket `downloads/Belajar_AES_MLP_Lengkap.zip` berisi **satu HTML terenkripsi**, termasuk hasil pribadi yang sudah ada dan ZIP konteks pair asli. Ekstrak, buka HTML, gunakan kunci yang sama. Pembungkus ZIP menjaga satu unduhan di bawah batas 100 MiB GitHub; HTML setelah ekstraksi lebih besar. Halaman `.html` pada path unduhan lama mengarahkan pembaca ke ZIP terbaru. Kunci dan jawaban tidak diterbitkan plaintext; paket dienkripsi AES-GCM dan diverifikasi roundtrip. Versi publik hanya memuat agregat.

Paket arsip diverifikasi lagi: satu HTML terenkripsi, di bawah 100 MiB, dekripsi cocok dengan HTML pribadi final dan seluruh tujuh sumber asli di bundle cocok SHA-256. Pembukaan HTML terenkripsi lewat localhost berhasil setelah semua jaringan berikutnya diblokir; kelima sumber tampil. Akses file:// pada Chromium lingkungan ini ditolak kebijakan administrator, sehingga pengujian file-origin tidak diklaim. Kode dan data tertanam, tidak memerlukan permintaan jaringan setelah file dimuat.
