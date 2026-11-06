# E52 · E43 + DoRA pada RoBERTa · S3 Native Learnable

[Buka E52 di Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E52_E43_S3_DoRA8_Native_Learnable_GPU.ipynb) · [Source notebook](../../notebooks/experiments/AES_300_E52_E43_S3_DoRA8_Native_Learnable_GPU.ipynb).

**GPU T4 → Run all → upload `fix_dataset_300.xlsx`.** Cukup dataset Excel, tanpa ZIP hasil lama. Seluruh kode Python terlihat per section, tanpa magic, worker, clone repo, atau Drive.

**Perubahan dari E43 adalah resep pelatihan RoBERTa:** full fine-tuning dengan LR `5e-6` diganti DoRA query/value dengan LR adapter `1e-4`. Bobot pretrained asli dibekukan; adapter arah A/B dan magnitude DoRA dipelajari dari skor guru. Ini perubahan metode **dan LR aktif encoder**, sehingga bukan ablasi LR yang seluruh komponennya identik. LR adapter adalah kandidat awal, bukan nilai optimum dari paper.

Arsitektur tetap E43: tiga teks → RoBERTa yang beradaptasi melalui DoRA → ScalarMix empat layer + gamma → BiGRU 64 per arah → co-attention satu head respons–referensi + roundtrip → matching → masked mean → fitur 768; BGE-M3 frozen → proyeksi 256; concat 1.024 → MLP 256/GELU/dropout 0,1 → satu skor. Tidak memakai multihead E50 atau fusion gate E51.

| Pengaturan | Nilai |
|---|---|
| Model RoBERTa / BGE | `flax-community/indonesian-roberta-base` / `BAAI/bge-m3`; revision sama E43 |
| Input X | `soal`, `jawaban_referensi`, `response_text` |
| Target y | `rater` 0–100; hanya loss/seleksi/evaluasi, tidak masuk encoder |
| Implementasi | PEFT `0.14.0`, `LoraConfig(..., use_dora=True)`; bukan LoRA biasa |
| DoRA | Rank 8, alpha 16, target query/value seluruh 12 layer, dropout adapter 0 |
| Parameter encoder yang belajar | 294.912 arah A/B + 18.432 magnitude = **313.344**; bobot dasar frozen |
| LR DoRA / komponen lain | `1e-4` / `1e-4`; bobot dasar frozen tidak masuk optimizer |
| Training | S3 saja; satu 5-fold CV baru, maksimal 25 epoch dan patience 5 |
| Refit / test | Ceil median epoch terbaik CV; satu refit development dan satu evaluasi test |
| Loss / seleksi | MSE biasa, target 0–1 / RMSE validasi tanpa bobot |

DoRA memisahkan besar bobot (*magnitude*) dan arah bobot. A/B mempelajari perubahan arah berperingkat rendah; magnitude juga dipelajari. Magnitude DoRA berbeda dari gamma ScalarMix: magnitude berada pada matriks query/value encoder, gamma mengalikan hasil campuran empat layer. Encoder tetap beradaptasi dari label melalui adapter meskipun bobot dasarnya frozen. BGE tetap frozen, sedangkan proyeksi BGE, BiGRU, co-attention/matching, normalisasi, ScalarMix/gamma dan MLP tetap dilatih.

Adapter dipasang setelah inisialisasi seluruh komponen E43. Inisialisasi memakai `fork_rng` CPU supaya tidak menggeser RNG/dropout bersama. B awal nol dan magnitude awal mengikuti norma bobot pretrained. Fungsi awal mempertahankan encoder pretrained dalam toleransi numerik; pengujian membandingkan keluaran dan gradien dengan implementasi resmi PEFT. Tidak ada quantization, QLoRA, merge adapter saat training, atau fallback diam-diam ke LoRA/full fine-tuning.

**Protokol tetap:** split/fold dan seed dasar 42 sama E43, tanpa loop banyak seed atau kontrol ulang. AdamW/weight decay 0,01, warmup 10%, linear scheduler sesuai durasi fit E43, clipping 1, microbatch 1/batch efektif 8. Loss, dropout MLP, tokenizer/window, pooling, fusion dan data tidak diubah. Tidak memakai LDS, ranking, cosine, S1/S2, atau scheduler refit E46/E47. Early stopping hanya membaca validasi.

## Laporan lengkap

- **Excel 04:** 60 prediksi test, skor mentah/normalisasi/pembulatan; selisih tepat 0/1/2/3/4/5/10 dan lainnya; toleransi ≤n; MSE/MAE/error maksimum; **240 prediksi validasi OOF** terpisah dari test, beserta fold dan error.
- **Excel 05 → Adapter DoRA:** status aktif, layer/target, rank/alpha/skala, jumlah parameter, status frozen/trainable, norma A/B/magnitude dari checkpoint terpilih lima fold serta refit. Untuk 12 layer: 24 modul × 6 checkpoint = **144 baris**. Norma bukan kontribusi kausal terhadap skor.
- **Excel 05 → Parameter Dilatih dan Frozen:** jumlah parameter aktual per kelompok dan checkpoint. History MSE, Bobot ScalarMix/gamma, MAE dan toleransi CV tetap tersedia.
- **Excel 02:** vektor BGE, fitur RoBERTa, masked mean dan concat S3 tetap tersedia. Inspeksi model refit atas 300 jawaban bukan evaluasi 300 data.

Pemeriksaan sebelum ZIP membaca ulang metrik, ringkasan, OOF, rekap selisih, adapter aktif, target/layer lengkap dan bobot asli frozen. Resume memakai hash/config/checkpoint E52 tersendiri dan memulihkan optimizer/scheduler/scaler/RNG. Checkpoint menyimpan state model lengkap termasuk bobot frozen, seperti alur native E43; optimizer hanya menyimpan parameter trainable. Jangan memakai checkpoint E5/E43/E50/E51. Report dapat dibangun ulang dari hasil tersimpan tanpa evaluasi test tambahan.

## Landasan dan batas bukti

Liu dkk., **[DoRA: Weight-Decomposed Low-Rank Adaptation, ICML 2024](https://arxiv.org/abs/2402.09353)** · [repositori penulis](https://github.com/NVlabs/DoRA) · [PEFT v0.14.0](https://github.com/huggingface/peft/tree/v0.14.0). Paper menguji tugas/model lain, bukan AES Indonesia dengan E43. Rank 8/alpha 16/query-value dan LR `1e-4` adalah resep awal proyek, mengikuti target LoRA E5 agar jelas; bukan kombinasi optimum yang dibuktikan paper.

E5 terdahulu memakai LoRA biasa dan input jawaban saja, sehingga bukan kontrol identik untuk E52. Acuan aktif E43: mean RMSE CV 5,8876, MAE OOF 4,6122, selisih mentah ≤5 sebanyak 151/240; RMSE test historis 5,1265. **Belum ada hasil penelitian E52.** Pilih melalui validasi; test historis bersifat eksploratif. Tidak menjamin peningkatan akurasi atau target 60/60 dalam ±3/±5 poin.

## Reproduksi dan verifikasi

Builder `scripts/build_e52_notebook.py` membaca arsitektur E43 pada commit `7241a21dfec37c942948b754da3960fba4c7731c`. Pelaporan lengkap dan pemeriksaan unduhan diturunkan dari E50 pada commit `c36003670ad9476951b8b607a6018e5a24ceb7e2`, tanpa mengambil model multihead-nya. Helper `scripts/dora_native.py` diembed sebagai kode Python biasa dalam notebook. Notebook utama dan mirror identik; hasil, data dan bobot tidak ditanam dalam source publik.

[Bukti pengujian teknis](../verifikasi/verification_e52_dora_native.md). Pengujian teknis bukan bukti peningkatan akurasi atau jaminan kompatibilitas semua runtime GPU.
