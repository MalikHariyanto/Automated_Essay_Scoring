# Status proyek untuk melanjutkan pekerjaan

Diperbarui **29 Oktober2026**. Baca bersama **[panduan konteks AI](PANDUAN_AI_PROYEK.md)**. Hasil E27/E28 kini ditambahkan; source keduanya dipin ke `daa745d`. Tabel enam peserta di bawah dipertahankan sebagai riwayat; prediksi aslinya tidak berubah.

## Kontrak aktif

Permintaan pengguna untuk **E21–E26**:

| Bagian | Konfigurasi |
| --- | --- |
| Dataset | `fix_dataset_300.xlsx`; 300 baris pada hasil yang diaudit |
| X, seluruh S1/S2/S3 | `soal`, `jawaban_referensi`, `response_text` |
| y | `rater`, satu skor guru 0–100; tidak masuk encoder |
| S1 | RoBERTa Indonesia full fine-tuning → head linear 768→1 |
| S2 | BGE-M3 frozen → MLP 1024→256→1 |
| S3 | RoBERTa fine-tuned + BGE frozen → pooling/fusion sesuai eksperimen → MLP |
| Pretrained | `flax-community/indonesian-roberta-base`, `BAAI/bge-m3` |
| LR RoBERTa S1/S3 | `5e-6` |
| LR proyeksi/head/layer tambahan | `1e-4` |
| Loss | MSE, target dinormalisasi 0–1 |
| Data | 240 development, 5 fold; 60 test disisihkan |
| Latihan | Maksimal 25 epoch/fold, patience 5; microbatch 1, efektif 8 |
| Seed GPU E21–E26 | Training 42, split 2026 |
| Penyimpanan | Disk runtime lokal, tanpa mount/autosave Drive |

Dokumen empat trait, rubrik 1–5, latihan berlabel AI, E6 jawaban-saja, dan E8 sebelum tiga input adalah workflow berbeda. Jangan menurunkan empat label dari `rater` atau mengganti S1 menjadi MLP tanpa instruksi baru. Kontrak eksplisit pengguna di atas mengatur keluarga aktif ini.

## Eksperimen dan hasil · riwayat enam peserta

| Eksperimen | Perubahan S3 | RMSE test | Mean RMSE CV | MAE test | Terdekat di antara enam |
| --- | --- | ---: | ---: | ---: | ---: |
| E21 | CLS, concat, MLP | 6,7231 | 6,6246 | 5,2515 | 11/60 |
| E22 | Context pair | 7,1943 | 6,4911 | 5,8108 | 9/60 |
| E23 | BiGRU → state akhir → proyeksi | 6,0439 | 6,4143 | 4,8576 | **13/60** |
| E24 | Attention pooling jawaban | 5,9818 | 6,0148 | 4,5286 | 9/60 |
| E25 | E24 + gated fusion | 6,1053 | 6,0096 | 4,7006 | 8/60 |
| E26 | Output token BiGRU → attention → proyeksi | **5,7435** | **5,9245** | **4,4797** | 10/60 |

Nilai asli ada pada [snapshot hasil](docs/snapshots/three_input_comparison_snapshot.json). Data, teks, label, test dan anggota fold dicocokkan antar-enam ZIP. S1/S2 tetap kontrol; kesamaan angka tidak membuktikan independensi run atau perangkat yang dipakai.

## Cara membaca hasil

- **RMSE/MAE terkecil:** E26. **Paling sering terdekat:** E23. Keduanya dapat memberi urutan berbeda.
- Kode utama HTML mengikuti jumlah terdekat S3 terbesar; seri dipecahkan dengan RMSE lebih rendah. Karena itu E23 menjadi kode utama, sementara E26 punya tombol tersendiri.
- Jumlah terdekat berubah ketika peserta berubah. Prediksi lama tetap; gunakan duel untuk pasangan tetap.
- E24 vs E26: masing-masing lebih dekat pada 30 jawaban. E25 vs E26: 31 vs 29, meskipun RMSE E26 lebih rendah.
- Test sudah berulang kali dilihat. Pilih konfigurasi berikutnya melalui development/CV; hasil test lanjutan bersifat eksploratif.
- SD fold berbeda dari SD seed. Belum ada pengukuhan tiga seed untuk E26.

## Kode dan batas bukti

Source hasil E21/E22 dipin ke `ef712b4c035d96178d9baa67caa7453ac9fd8738`; E23/E24 ke `f3d25c4`; E25/E26 ke `bc4713872c7a7ce23f7dad8ae3ea73a2b75d9b86`. [Katalog notebook](notebooks/experiments/README.md).

Kode dalam HTML cocok dengan notebook rujukan. ZIP hasil tidak menyertakan executed notebook, checkpoint dan metadata runtime lengkap: jangan mengklaim kecocokan byte kode runtime atau perangkat aktual terbukti hanya dari nama ZIP.

Pembaruan hasil terakhir lulus 18 tes integritas/unit, 8 kelompok browser dan 2 pemeriksaan paket. Ini bukti kode/HTML/unduhan, bukan ukuran akurasi model. Fixture CPU tidak membuktikan pretrained training GPU telah dijalankan oleh pemeriksaan tersebut.

## E27/E28 · hasil unggahan tersedia

Pengguna memilih **E23** sebagai acuan untuk memprioritaskan jumlah prediksi mendekati guru: **E27 = E23 + co-attention**, **E28 = E23 dengan LR MLP S3 saja5e-6**. Hasil dua ZIP sudah diaudit dan ditambahkan ke buku tiga input; kode utama tetap E23. [Review](docs/review/review_e27_e28_study.md).

| S3 | RMSE test | Mean RMSE CV | MAE test | Selisih≤5poin |
| --- | ---: | ---: | ---: | ---: |
| E27 | 6,3277 | 6,5067 | 5,0281 | 34/60 |
| E28 | 7,0447 | 6,4386 | 5,8803 | 27/60 |

Pada delapan peserta, jumlah terdekat S3 E21/E22/E23/E24/E25/E26/E27/E28 adalah **8/8/10/6/8/10/8/2**. E23/E26 seri10/60; E26 tetap RMSE/MAE terendah; E23 tetap terbanyak≤5poin39/60. Ini perubahan peserta, bukan perubahan prediksi lama. S1/S2 E27/E28 identik E23 per baris. Tidak ada executed notebook/runtime lengkap dalam ZIP, sehingga identitas kode runtime dan perangkat tidak diklaim terbukti.

Penting: kelompok `head` pada optimizer **E23 asal** juga menampung BiGRU, proyeksi ringkasan dan LayerNorm. **E27/E28 sudah memisahkan** `head.*` dari `pooling_norm`. Pada E28 hanya MLP S3 berubah5e-6; S1/S2 head serta pooling/norm/proyeksi tetap1e-4. `s3_mlp_lr` dan `s3_layers_lr` dicatat terpisah pada config, checkpoint dan Excel.

Untuk akun baru, ikuti [serah-terima](PINDAH_AKUN.md). Chat, lampiran privat, runtime Colab dan `outputs/` tidak otomatis tersedia di akun baru.

## E29/E30 · kode baru, hasil belum tersedia

Pengguna kembali memilih **E26** sebagai dasar. E29 mempertahankan arsitektur E26 dan menurunkan LR **MLP S2 dan S3** menjadi5e-6; linear S1 tetap1e-4. E30 menambah **co-attention lalu attention pooling E26**, MLP S2/S3 tetap1e-4. Encoder5e-6; BiGRU, pooling, proyeksi/norm/layer baru1e-4. Semua skenario dilatih baru, GPU biasa tanpa Drive. [Source dan metode](docs/eksperimen/eksperimen_e29_e30_gpu.md).

Co-attention mencocokkan dua urutan, sedangkan additive attention pooling merangkum hasil matching. Self-attention RoBERTa tetap ada. Keduanya tidak sama dengan E27 berbasis E23. Buku hasil delapan peserta tetap; belum ada hasil E29/E30 maupun kode E31 ensemble.

## Identitas E21–E30 dalam HTML

Tab **07 · Identitas & sitasi** menampilkan sepuluh kartu dengan acuan, alur, LR, label, notebook sumber dan sitasi per eksperimen. **E29/E30 hanya identitas**: ruang kode/output sengaja kosong dan tidak ikut perbandingan delapan hasil. Kode sumber eksternal tersedia; belum dimasukkan ke meja belajar. Seluruh hasil, delapan buku anak dan kode utama E23 dipertahankan. [Rincian pembaruan](docs/review/review_identitas_eksperimen.md).

## E29 · ZIP hasil sudah masuk, tiga kode di meja belajar

Hasil E29 diaudit dan dimasukkan ke HTML: S3 RMSE5,4293/MAE4,1713/≤5poin40dari60; S2 justru RMSE75,9702 dan seluruh fold memilih batas25epoch. E29 memakai LR MLP S2/S3 5e-6 dari arsitektur E26, bukan co-attention. [Review lengkap](docs/review/review_e29_study.md).

**Instruksi terbaru menggantikan default E23:** meja belajar hanya mengembed tiga kode dengan RMSE test S3 terendah, yaitu **E29, E26, E24**; utamaE29. Semua sembilan hasil/45Excel tetap tersedia lewat perbandingan dan tab08SemuaExcel. E30 masih pending. Jumlah terdekat S3 sembilan peserta E21–E29 adalah8/7/8/5/7/6/7/2/10; perubahan peserta tidak mengubah prediksi lama.
