# Verifikasi revisi training kandidat A

20 Oktober 2026, branch `codex/aes-learning`. Basis sebelum revisi: `3afad79` (simpanan Colab pengguna). Notebook lama beserta output dipertahankan dalam riwayat; file notebook publik terbaru dibersihkan dari hasil privat.

## Perubahan yang diuji

- Learning rate RoBERTa `1e-5`, proyeksi BGE `1e-4`, dan head `1e-4`. Encoder BGE frozen; LayerNorm tambahan ikut kelompok parameter baru/head. Setiap parameter yang dilatih hanya masuk satu kelompok optimizer.
- Batas 25 epoch per fold untuk S1/S2/S3, patience 5 epoch berturut-turut tanpa penurunan RMSE validasi. Nilai yang sama tidak dianggap perbaikan. Bobot terbaik dipulihkan saat berhenti lebih awal maupun mencapai batas. Refit memakai jumlah epoch pilihan CV, tanpa early stopping atau pemantauan test.
- LayerNorm tambahan 768 komponen di cabang keluaran RoBERTa khusus S3 sebelum concat. Vektor asli tetap tersimpan; hasil LayerNorm dipakai sebagai 768 komponen pertama fusion, kemudian proyeksi BGE 256 komponen.
- Tabel Colab dan Excel membandingkan nilai guru, prediksi mentah skala 0–100, serta salinan kategori S1/S2/S3 pada jawaban test yang sama. Pembulatan menggunakan clip dan kategori integer terdekat, tepat .5 ke bawah; RMSE/Pearson tetap dari prediksi mentah. Jawaban baru tidak diberi nilai guru buatan.
- Tetap lima Excel utama. Workbook vektor menambah sheet hasil LayerNorm S3; riwayat latihan menambah ringkasan epoch maksimal/dijalankan/terpilih dan alasan berhenti. Fingerprint kode/config memberi folder run terpisah dari hasil lama.

## Pengujian

Suite final: **49 passed, 0 failed, 0 skipped, 0 xfailed**, exit 0, durasi 100,92 detik. Eksekusi notebook mandiri menyelesaikan seluruh **30 cell kode produksi**, ditambah satu cell penjaga khusus pengujian, dalam 36,53 detik. Hasilnya mencakup tiga skenario, 15 hasil fold, satu evaluasi test per skenario, tabel perbandingan tiga jenis skor di Colab, serta ZIP berisi lima Excel.

Perintah dari root repo:

```bash
PYTHONPATH="$PWD/src" HF_HOME="$PWD/outputs/cache/huggingface" \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 JUPYTER_PLATFORM_DIRS=1 \
outputs/dataset300_env_py313/bin/python -m pytest -q -rA \
--basetemp=/tmp/aes_candidate_a_suite --junitxml=outputs/candidate_a_tests.xml \
tests/test_teacher_training_revision.py tests/test_standalone_notebook.py \
tests/test_dataset300_setup.py tests/test_teacher_presentation.py \
tests/test_teacher_excel.py tests/test_teacher_score.py
```

Lingkungan pengujian: Python 3.13.5, torch 2.6.0+cpu, Transformers 4.48.3, NumPy 2.2.6, Pandas 2.2.3, openpyxl 3.1.5, pytest 8.3.5, nbclient 0.10.2. `PYTHONPATH` hanya dipakai controller pengujian modul; kernel notebook berjalan isolated tanpa import proyek.

Log, XML, manifest hash source, versi pustaka dan notebook terisi disimpan privat di `outputs/candidate_a_tests*` dan folder pengujian `/tmp`. Kedua notebook publik identik dan tidak berisi output atau execution count. SHA remote serta byte notebook dan source utama diverifikasi setelah push.

Pengujian mencakup pemisahan kelompok learning rate, checkpoint/reload, gradien LayerNorm/proyeksi dan BGE frozen, early stopping dengan urutan RMSE terkontrol yang menguji reset penghitung dan pemulihan bobot terbaik, refit tanpa membaca validasi, batas epoch, pemisahan folder hasil, serta pemetaan nilai guru/prediksi/pembulatan saat urutan baris tiap model berbeda. Ekspor Excel memeriksa bahwa concat benar-benar menggunakan vektor setelah LayerNorm dan ketiga jenis skor tetap cocok.

Pengujian notebook memakai 24 jawaban sintetis, satu epoch untuk pemeriksaan integrasi, tokenizer fixture, dan transformer kecil berbobot acak dengan dimensi 768/1024. Notebook dijalankan dari folder tanpa repo, import `aes` diblokir, dan perintah Git dibuat gagal. Uji early stopping terpisah menjalankan latihan beberapa epoch dengan hasil validasi terkontrol. Ini bukti perilaku kode, bukan hasil akurasi checkpoint pretrained atau pengganti training penelitian pada dataset guru.

Tidak diklaim bahwa RMSE S3 sudah menurun. Hasil tersebut baru diketahui setelah eksperimen asli berjalan. Pilihan konfigurasi berikutnya tetap didasarkan pada CV; test versi lama sudah dilihat dan bukan lagi konfirmasi yang sepenuhnya belum pernah disentuh selama pengembangan lanjutan.
