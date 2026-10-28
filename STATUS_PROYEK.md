# Status proyek untuk melanjutkan pekerjaan

Diperbarui **28 Oktober 2026**. Baca bersama **[panduan konteks AI](PANDUAN_AI_PROYEK.md)** sebelum mengubah model atau menafsirkan dokumen lama. Titik acuan hasil E25/E26: commit **`6ee4ec5b20256dc284ecaec378b7474bce4c1826`**. Penataan folder sesudahnya tidak mengganti hasil tersebut.

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

## Eksperimen dan hasil

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

## Notebook lanjutan · hasil belum tersedia

Pengguna memilih **E23** sebagai acuan untuk memprioritaskan jumlah prediksi mendekati guru, lalu meminta dua resep: **E27 = E23 + co-attention respons–referensi**, **E28 = E23 dengan LR MLP S3 saja `5e-6`**. Keduanya tersedia sebagai notebook GPU mandiri tanpa Drive, seluruh S1/S2/S3 dilatih baru. [Source dan metode](docs/eksperimen/eksperimen_e27_e28_gpu.md). Belum ada hasil penelitian atau penambahan hasil E27/E28 ke HTML. Huber dan pengukuhan E26/kandidat tiga seed masih rencana.

Penting: kelompok `head` pada optimizer **E23 asal** juga menampung BiGRU, proyeksi ringkasan dan LayerNorm. **E27/E28 sudah memisahkan** `head.*` dari `pooling_norm`. Pada E28 hanya MLP S3 berubah5e-6; S1/S2 head serta pooling/norm/proyeksi tetap1e-4. `s3_mlp_lr` dan `s3_layers_lr` dicatat terpisah pada config, checkpoint dan Excel.

Untuk akun baru, ikuti [serah-terima](PINDAH_AKUN.md). Chat, lampiran privat, runtime Colab dan `outputs/` tidak otomatis tersedia di akun baru.
