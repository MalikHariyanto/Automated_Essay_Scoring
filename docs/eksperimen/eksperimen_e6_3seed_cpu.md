# E6 CPU · tiga seed · seluruh S1/S2/S3 dilatih ulang

[Buka Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E6_3Seed_CPU.ipynb) · [Unduh notebook](../../notebooks/experiments/AES_300_E6_3Seed_CPU.ipynb)

Pilih Runtime → Change runtime type → CPU, lalu jalankan dari atas. Unggah `fix_dataset_300.xlsx` saat diminta. Semua kode tersedia dalam notebook; dataset tetap privat. Worker memakai `device='cpu'`, float32, tanpa AMP, dua thread; GPU tidak diwajibkan dan disembunyikan dari worker perhitungan. Versi GPU sebelumnya tetap tersedia.

Metode E6 tetap: seed training 42, 123, 2026; split seed 2026; pembagian development/test 80/20 dan lima fold yang sama pada seluruh seed. S1 RoBERTa full fine-tuning + linear, encoder LR 1e-5. S2 BGE frozen + MLP. S3 RoBERTa full fine-tuning + BGE frozen + fusion MLP, encoder LR 5e-6. LR proyeksi/head 1e-4, maksimal 25 epoch/fold, early stopping patience 5, microbatch 1 dan batch efektif 8. Input hanya jawaban, tanpa LoRA atau konteks pair.

Seluruh 45 pelatihan CV dan sembilan refit dimulai baru; S1/S2 tidak mengambil hasil eksperimen lain. Pelatihan CPU jauh lebih lama daripada GPU, tanpa jaminan selesai dalam satu sesi Colab. Vektor BGE frozen/tokenisasi dihitung bersama sekali. Audit inisialisasi, struktur, seed, perangkat dan versi tetap dicatat; hasil CPU tidak dijanjikan identik dengan GPU.

Folder baru: `MyDrive/AES_300_E6_3Seed_CPU_Privat`. Hasil eksperimen lama tidak perlu dihapus. Autosave memakai dua slot A/B per fold/refit, tanpa Drive API atau auto-delete. Tetap diperlukan ruang untuk seluruh seed/fold dan model final. `RUN_TAG='ulangan_01'` melanjutkan progres CPU yang cocok; ganti tag untuk ulang dari nol. Notebook ini tidak melanjutkan optimizer/progres versi GPU atau memakai ulang kontrol lama.

Paket `Hasil_E6_3_Seed_CPU.zip` berisi 16 Excel: ringkasan tiga seed dan lima laporan per seed. Ringkasan mencakup rerata/SD CV dan test, seluruh fold, nilai guru/prediksi dan audit. SD dihitung antar tiga seed, bukan memperlakukan 15 fold sebagai 15 eksperimen independen. Semua hasil seed dilaporkan; tidak memilih seed dari test.

Pengujian teknis:

```bash
AES_THREE_SEED_NOTEBOOK=AES_300_E6_3Seed_CPU.ipynb outputs/dataset300_env_py313/bin/python -m pytest tests/test_three_seed_notebook.py -q --disable-warnings
```

Hasil: **11 passed, 0 failed, 0 skipped, 0 xfailed**, satu warning. Fixture sintetis 24 baris dan encoder/tokenizer kecil memeriksa alur training seluruh skenario/seed, split tetap, learning rate, perangkat CPU, resume, fallback checkpoint, scoring, ekspor fitur dan paket 16 Excel. Tiga pemeriksaan konfigurasi/builder/fingerprint versi GPU juga lulus (delapan kasus lain tidak dipilih pada pemeriksaan GPU tambahan). Ini memeriksa perilaku program, bukan kualitas skor AES. Training pretrained dengan 300 jawaban pengguna dan eksekusi penuh di Colab belum dijalankan di sini.
