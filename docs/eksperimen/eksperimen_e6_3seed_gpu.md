# E6 GPU · tiga seed · seluruh S1/S2/S3 dilatih ulang

[Buka Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E6_3Seed_GPU.ipynb) · [Notebook untuk diunduh](../../notebooks/experiments/AES_300_E6_3Seed_GPU.ipynb)

Pilih Runtime → Change runtime type → T4 GPU, lalu jalankan dari atas dan pilih `fix_dataset_300.xlsx`. Notebook mandiri; dataset tidak disimpan di GitHub dan tidak memerlukan source proyek untuk diunduh saat berjalan. GPU wajib untuk pelatihan, tanpa fallback CPU diam-diam.

| Bagian | Pengaturan |
| --- | --- |
| Seed training | 42, 123, 2026 |
| Split seed | 2026, dikunci untuk semua run |
| S1 | RoBERTa full fine-tuning, head linear, LR encoder 1e-5 |
| S2 | BGE-M3 frozen, head MLP baru untuk setiap seed |
| S3 | RoBERTa full fine-tuning + BGE frozen, LayerNorm/proyeksi/concat/MLP, LR encoder 5e-6 |
| LR proyeksi/head | 1e-4 |
| Latihan | Maksimum 25 epoch/fold, patience 5, microbatch 1, batch efektif 8 |
| Total | 45 latihan fold CV + 9 refit model final |
| Pemilihan epoch | Ceil median epoch validasi terbaik pada lima fold; refit development tanpa early stopping |
| Pemilihan seed | Tidak ada: semua tiga seed dilaporkan |

Seed run ditambah nomor fold, mengikuti E6. Run42 memakai seed43–47 untuk fold1–5 dan42 untuk refit; run123 memakai124–128 dan123; run2026 memakai2027–2031 dan2026. Group holdout 80/20 dan anggota lima fold tidak berganti; fungsi split diperiksa sama pada setiap seed. Dataset tetap satu skor guru 0–100 dengan fitur jawaban saja. Tidak ada LoRA atau konteks pair, tidak menggabungkan model menjadi ensemble.

## Mulai baru dan melanjutkan

Folder baru `MyDrive/AES_300_E6_3Seed_GPU_Privat`; setiap seed mempunyai subfolder `seed_42`, `seed_123`, `seed_2026`, masing-masing S1/S2/S3. Tidak membaca checkpoint atau hasil kontrol dari folder eksperimen sebelumnya. Vektor BGE/pretrained dan tokenisasi dipakai bersama sekali karena tidak dilatih; head serta RoBERTa yang trainable dimulai baru pada setiap fold/refit.

`RUN_TAG='ulangan_01'` digunakan untuk percobaan pertama. Untuk mengulang dari nol tanpa menghapus history, ganti menjadi `ulangan_02` sebelum Run all. Untuk resume sesi terputus, biarkan tag/pengaturan sama dan jalankan dari atas. Fold/seed yang selesai pada eksperimen ini dilewati setelah checksum diverifikasi; progres yang cocok dilanjutkan. Rencana config, dataset, split, hash bytecode/konstanta fungsi aktual, dan hash fitur BGE dikunci. Salinan Excel per seed diverifikasi terhadap checksum saat penyalinan pertama; timestamp XLSX yang dibuat ulang pada restart tidak menggagalkan resume. Perubahan fungsi runtime tidak boleh diam-diam memakai hasil lama.

Autosave menggunakan dua slot tetap A/B per fold/refit; slot lebih lama ditulis ulang langsung, tanpa rename/unlink file bobot. Bila penulisan terputus atau snapshot terbaru rusak, penunjuk checksum dapat memulihkan slot sebelumnya. Tidak ada Drive API cleanup/auto-delete atau pembersihan Sampah. Fold selesai tidak dibersihkan otomatis. Dua slot membatasi jumlah file per tahap, tetapi tiga seed tetap membutuhkan ruang besar dan penyimpanan Drive dapat mencakup versi/sampah; notebook tidak mengklaim mengosongkan kuota Drive.

## Kenapa hasil dapat berubah sementara kode/layer sama

Struktur model yang sama tidak berarti bobot awal head, urutan batch, dropout, optimizer state, atau perhitungan perangkat sama. Untuk mengukurnya, `Audit Model dan Seed` dan JSON per tahap menyimpan hash kode fungsi runtime, hash arsitektur, hash seluruh bobot awal head/proyeksi, sampel32 angka per parameter encoder, seed run/fold, GPU/versi/presisi serta selisih dua forward eval dari bobot yang sama. Hash encoder adalah sampel, bukan bukti checksum semua bobot pretrained. Revision checkpoint/tokenizer tetap dikunci.

Jika hash arsitektur sama tetapi hash awal head berbeda pada seed berbeda, variasi awal tercatat tanpa perubahan layer. Jika kode aktual berubah tetapi hasil lama dicoba dipakai, rencana eksperimen menolak. Seed/split/config/perangkat/versi yang sama diharapkan sama atau sangat dekat; beda besar perlu ditelusuri, bukan langsung dianggap wajar. Dua forward eval bukan pengulangan seluruh training.

Deterministic algorithms diaktifkan dengan warn_only, TF32 dimatikan, cuBLAS workspace dikunci dan kondisi acak disimpan untuk resume. Ini meningkatkan keterulangan, bukan jaminan identik lintas GPU, presisi atau versi pustaka. E6 terakhir berjalan CPU; T4 baru menggunakan AMP float16 (GPU yang mendukung dapat memakai bfloat16), sehingga seed42 tidak dijanjikan identik dengan angka CPU lama. Bandingkan tiga run baru di lingkungan GPU yang sama. Hasil S1/S2 lama yang dipakai ulang oleh E6 tidak dianggap tiga pengulangan independen.

## Paket hasil

`Hasil_E6_3_Seed_GPU.zip` secara default memuat16 Excel: `Ringkasan_3_Seed.xlsx` dan lima laporan per seed. Ringkasan memisahkan CV dan test, menampilkan mean serta SD **antar tiga seed** (n=3), seluruh45 fold, nilai guru/prediksi sembilan model final, salinan kategori dan audit. Mean CV dihitung per seed dari lima fold terlebih dahulu, kemudian dirata-ratakan antar-seed;15 fold tidak dianggap15 run independen. Metrik yang tidak terdefinisi tidak diubah menjadi nol. Tiga prediksi per skenario bukan ensemble.

Lima laporan tiap seed mengikuti format acuan: data/tokenisasi, vektor jawaban, vektor kata/kalimat, prediksi/evaluasi, riwayat pelatihan. Tahap sebelum training yang dibagi bersama bukan inisialisasi head khusus masing-masing seed. Model final dan vektor setelah training berasal dari seed masing-masing. Semua hasil tiga seed dilaporkan; data uji tidak memilih seed atau LR. Test yang sudah dilihat dalam pengembangan tetap eksploratif.

## Verifikasi teknis

Cwd root repository, interpreter `outputs/dataset300_env_py313/bin/python`:

```bash
python scripts/build_e6_three_seed_notebook.py
python -m pytest tests/test_three_seed_notebook.py -q --disable-warnings
```

10 passed, 0 failed, 0 skipped, 0 xfailed;1 warning. Pengujian memakai dataset sintetis24 baris, tokenizer/encoder kecil dan head S1/S2/S3 aktual. Seluruh45 CV dan9 refit dijalankan satu epoch untuk memeriksa alur, LR, split tetap, hash arsitektur/head, prediksi test, resume tanpa latihan ulang, perubahan config/fitur/kode yang ditolak, dan fallback snapshot. Pengulangan seed yang sama pada fixture CPU memberi prediksi identik. Dua proses Python baru menghasilkan fingerprint kode yang sama; perubahan fungsi runtime tetap ditolak. Cell scoring, ekspor fitur final, dan cell unduhan dieksekusi dengan fixture; paket16 Excel terbaca. Fixture kata memakai keluaran encoder kecil dan diberi label teknis, bukan embedding penelitian. Source/mirror/format/fingerprint notebook diperiksa, builder dapat diulang tanpa mengubah acuan.

Tidak ada GPU pada host pemeriksaan. Training T4 dengan dataset pengguna dan eksekusi seluruh notebook melalui Colab belum dijalankan di sini. Bukti output pengujian tidak mengukur kualitas AES penelitian. Log/hash/versi disimpan privat di `outputs/three_seed_*`.
