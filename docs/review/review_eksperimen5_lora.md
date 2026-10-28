# Eksperimen 5 · LoRA + MLP sebagai pembanding

> Dokumen tahap terdahulu. Angka dan keputusan dibaca dalam konteks versi ini; untuk kondisi aktif gunakan [status proyek](../../STATUS_PROYEK.md) dan [panduan AI](../../PANDUAN_AI_PROYEK.md).

Hasil dari ZIP unggahan pengguna, SHA-256 `d2ed126f14f4dfba28d896ad48cd05371f85717f2118920262795086ba93c6af`. Lima workbook berisi 36 sheet. Tidak ada training baru untuk pembaruan buku ini.

S3 memakai bobot RoBERTa asli frozen + adapter LoRA query/value (rank 8, alpha 16, dropout 0, LR 1e-4), BGE-M3 frozen, LayerNorm/proyeksi/concat dan head MLP. Maksimal 25 epoch, patience 5, batch efektif 8. S1 tetap RoBERTa fine-tuned + linear; S2 BGE frozen + MLP. Metode ini dibuktikan oleh tabel model/pengaturan dalam workbook; notebook eksekusi E5 tidak disertakan, sehingga source runtime tidak diklaim byte-identik.

| Ukuran S3 | MLP acuan | E5 LoRA + MLP | Pembacaan |
| --- | --- | --- | --- |
| RMSE test | 6.70853982 | 6.53049357 | E5 lebih kecil 0.17804626 poin |
| Rata-rata RMSE CV | 6.05163973 | 6.25086442 | MLP acuan lebih kecil |
| SD RMSE CV | 0.82603484 | 0.92787514 | MLP acuan lebih rapat antar-fold |
| Pearson test | 0.55377895 | 0.56696068 | Baca bersama besar error |
| QWK test | 0.50963911 | 0.52712160 | Kesepakatan kategori |
| Gwet AC2 test | 0.97859699 | 0.97970211 | Bukan persen prediksi tepat |

S1 masih RMSE test terkecil, sekitar 6,3805. S1/S2 memiliki prediksi praktis sama dengan acuan: selisih maksimum masing-masing 0.0000038062 dan 0.0000038013 poin. Perubahan kecil ini tidak membuktikan proses training/disalin; asal eksekusinya tetap dibatasi bukti workbook. Pada S3, E5 memiliki error absolut lebih kecil pada 31 dari 60 jawaban, MLP acuan pada 29. Hitungan memakai toleransi beda numerik 1e-5 poin, bukan toleransi penilaian AES.

## Asal metrik dan pemeriksaan data

ZIP E5 tidak memuat sheet Perbandingan Model atau Hasil Tiap Fold. Empat metrik test dihitung ulang dari 60 pasangan guru/prediksi mentah pada Nilai Guru dan Prediksi; QWK/AC2 memakai kategori 0–100 dengan tie .5 ke bawah, sedangkan RMSE/Pearson memakai skor mentah. Pembulatan divalidasi terhadap Excel dan metrik diperiksa dengan dua jalur fungsi. Angka baru di HTML diberi keterangan dihitung ulang, bukan diklaim sebagai sheet asli ZIP.

Empat metrik CV berasal dari baris epoch terpilih pada riwayat setiap fold. Seluruh 15 fold lengkap dan pilihan epoch dicocokkan dengan minimum RMSE tersimpan; jumlah epoch dijalankan juga cocok. Karena prediksi individual validation fold tidak disertakan, metrik CV tidak diklaim dihitung ulang dari pasangan validation.

Seluruh 300 baris data, pembagian 240/60, 1.200 baris keanggotaan lima fold, 60 ID test dan nilai guru dicocokkan terhadap ZIP MLP. Angka agregat saja diterbitkan pada HTML publik/snapshot. Jawaban, nilai individual dan vektor disimpan dalam paket HTML pribadi terenkripsi, dengan kunci yang sama seperti versi sebelumnya. ZIP dan workbook asli disertakan tanpa perubahan.

## UI dan pengujian

- Menu **E5 · LoRA + MLP** dan tombol E5 pada daftar eksperimen membuka empat tab: hasil/metode, training/validasi, guru/prediksi dan Excel.
- Tabel metrik utama dan pembanding menempatkan awal, MLP, double frozen dan E5 sebagai kolom. E4 tidak diisi karena hasilnya tidak ada dalam unggahan ini.
- Seluruh 15 fold, mean/SD, pilihan epoch dan 30 kurva MSE/RMSE MLP–E5 dapat dibaca. Tabel individual memiliki paging dan seluruh komponen vektor tetap dapat dijangkau.
- Pemeriksaan browser `node scripts/check_mlp_lora.mjs`: 8 kelompok lulus, mencakup 48 nilai test, 48 kombinasi tab/skenario/viewport, 30 kurva, 180 prediksi individual, seluruh sheet, unduhan XLSX identik serta fullscreen 30 blok.
- `python -m pytest tests/test_mlp_study.py tests/test_mlp_comparison.py tests/test_mlp_frozen.py -q --disable-warnings`: 17 passed, 0 failed, 0 skipped, 0 xfailed (0,27 detik), memakai interpreter `outputs/dataset300_env_py313/bin/python`.
- Ekspor pribadi dibuka ulang dan checksum ZIP E5 tetap sama; notebook/ZIP acuan, workbook awal dan ZIP frozen tetap dimuat. Paket terenkripsi diverifikasi kembali terhadap HTML pribadi lengkap, memakai kunci yang sama dan berukuran 93,38 MiB.
- Regresi meja metrik `node scripts/check_mlp_metrics.mjs`: 6 kelompok lulus, termasuk 60 kombinasi tampilan, pencocokan ID, unduh/ekspor dan pembukaan ulang.
- Bukti browser dan ekspor tersimpan di outputs/mlp_study/e5_*.json/log yang diabaikan Git. Pengujian membaca hasil tersimpan; tidak melatih model atau membuktikan keunggulan statistik.

Satu run per metode belum mengukur variasi seed. Test telah dilihat sepanjang pengembangan; pemilihan metode tetap melalui CV dan konfirmasi independen memerlukan data baru. Hasil test E5 yang lebih kecil tidak menghapus hasil CV acuan yang lebih baik.
