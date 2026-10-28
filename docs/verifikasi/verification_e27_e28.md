# Verifikasi E27 co-attention dan E28 LR MLP S3

**29 Oktober2026.** Implementasi sudah diuji; **belum ada hasil penelitian E27/E28**. [Metode dan notebook](../eksperimen/eksperimen_e27_e28_gpu.md). Dasar source **E23** commit `f3d25c475e372db90cda894efe447b9d8c41122a` sesuai pilihan eksplisit pengguna.

## Command dan hasil

Working directory: `/workspace/Automated_Essay_Scoring`.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
outputs/dataset300_env_py313/bin/python -m pytest -q \
  tests/test_e27_e28_notebooks.py \
  --basetemp=outputs/e27_e28_work/e23_final_fixtures \
  --junitxml=outputs/e27_e28_work/e23_final.xml
```

**Exit0;13passed,0failed,0errors,0skipped,0xfail/xpass**,166,19detik. Satu warning deprecation jalur Jupyter; bukan kegagalan notebook. Python3.13.5, PyTorch2.6.0+cpu, Transformers4.48.3, NumPy2.2.6, nbformat5.10.4, nbclient0.10.2, pytest8.3.5. CUDA tidak tersedia pada lingkungan pemeriksaan.

## Yang diperiksa

- Builder deterministik, source E23 tidak ditimpa,47cell/30cellkode per notebook, checksum kode valid, output/count eksekusi kosong, mirror identik, tanpa impor proyek/Drive. Source publik meminta CUDA; fixture secara eksplisit mengganti perangkat menjadi CPU.
- Seluruh S1/S2/S3: output `[B,1]`, concat S3 `[B,1024]`, ringkasan128, BGE frozen, gradien encoder/layer baru, optimizer mencakup semua parameter trainable tepat sekali, simpan–muat state mempertahankan prediksi.
- Bobot awal parameter E23 yang sudah ada identik pada E27/E28. Kontrol S1/S2 memberi prediksi identik dengan E23 pada fixture. Arsitektur, fitur, window dan evaluator E28 sama dengan E23; prediksi awal S3 identik.
- E27: gradien ke token respons dan referensi, GRU, proyeksi referensi dan proyeksi matching; tanpa layer additive attention pooling E26; mask terpisah; padding dan soal/prefix tidak menjadi query/key langsung; perubahan referensi mengubah hasil; batch panjang berbeda dan urutan asli konsisten; bobot finite/jumlah1, padding0. AMP bfloat16 **CPU** juga menghasilkan keluaran finite.
- Tokenisasi: ID input/window/start/cakupan respons tetap identik E26; mask posisi benar untuk padding kiri/kanan dan kata berulang pada soal/prefix/respons. Referensi tanpa isi akibat truncation gagal jelas. Ketidakcocokan ID fast/slow ditolak.
- Slow/fast RoBERTa API nyata diuji dengan Byte-BPE kosakata buatan lokal dan teks sintetis termasuk Unicode; offset dan cache fast fallback diperiksa. Ini bukan uji tokenizer pretrained Indonesia. Akses unduhan Hugging Face pada pemeriksaan ini terhalang proxy sehingga tidak diklaim sebagai smoke checkpoint asli.
- E28: langkah AdamW pertama menghasilkan pembaruan non-MLP yang identik dengan kontrol ber-LR1e-4, sementara norm pembaruan MLP sekitar20×lebih kecil. LR MLP S3 saja5e-6; GRU/proyeksi/norm1e-4, encoder5e-6, head S1/S2 tetap1e-4.

## Eksekusi seluruh notebook

Setiap notebook dijalankan pada workbook **24baris sintetis**, RoBERTa/XLM-R kecil satu layer berbobot acak tetapi dimensi768/1024dipertahankan, CPU, satu epoch. Impor modul proyek dilarang pada worker fixture untuk memeriksa kemandirian source.

Masing-masing menyelesaikan **30cellkode**, **15CVfits +3refit**, final test teknis dan **lima Excel**. Notebook dijalankan lagi pada kernel baru dengan folder runtime yang sama: hash tiga checkpoint final tetap; training/test tidak diulang. Durasi eksekusi dua kernel sekitar84,21detik E27 dan73,52detik E28.

Pengaturan LR aktual pada Excel/config/checkpoint diperiksa. E27 mengekspor bobot referensi yang berjumlah1 per window, pasangan token top1 dengan bobot0–1, serta posisi Excel mulai1. Output fixture tidak dipublikasikan dan tidak diperlakukan sebagai nilai penelitian.

## Identitas sumber

| Notebook | SHA256 file | Fingerprint kode |
| --- | --- | --- |
| E27 | `33fb26c4efa70d7a2d5892b476ae3c1ea528b7fc4dfcc954573771b39661ac1a` | `c0504849185fe62078de8321c0374b53f450d1a0a0beb2111d06a5a07f6f039a` |
| E28 | `026d771f04623e8eb244a73ee68ba4259667e725ee8d78dc041f0f9a3176b765` | `0e8dce06f2e544e63f78ef23860a85d6208819ec2db3168d95472abf9d7551f7` |

Seluruh54notebook terdahulu dan10berkas aplikasi/pengunduh/bagian paket dibandingkan dengan keadaan awal: tidak berubah. Dokumen status dan katalog diperbarui; hasil numerik/HTML E21–E26 tetap. Log, JUnitXML, versi, bukti eksekusi dan manifest sumber disimpan lokal pada `outputs/e27_e28_work/`.

Ini verifikasi implementasi dan fixture, **bukan** training pretrained pada300jawaban, uji CUDA/AMP GPU asli, atau bukti co-attention/LR baru meningkatkan akurasi. Hasil penelitian menunggu pengguna menjalankan resep dan mengirim keluaran.
