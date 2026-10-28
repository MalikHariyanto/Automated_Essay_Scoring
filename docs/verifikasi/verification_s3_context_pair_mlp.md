# Verifikasi S3 context pair + MLP

23 Oktober 2026. Pekerjaan dari HEAD `b537e32501af2a33e787ca327b37bcaf0f92641a`, branch `codex/aes-learning`. Notebook sumber MLP, double frozen, versi manual serta notebook utama tidak ditulis ulang. Perubahan menghasilkan notebook baru dan mirror di `aes-learning/notebooks/experiments/`.

## Identitas dan batas pengujian

Source dibangun dengan `python scripts/build_context_pair_notebook.py`. Fingerprint awal sebelum revisi GPU: `f2b8db3c043b81cd46cdf14df07c422771fd32debd5e55967daff5e2cc343276`; identitas terbaru ada pada bagian revisi GPU di bawah. Hash memakai source teks agar tidak bergantung pada perbedaan representasi AST Python 3.12/3.13; kedua interpreter menghasilkan notebook identik byte. Builder menanamkan definisi helper, tidak menambahkan import proyek atau clone GitHub pada runtime. Semua code cell publik mempunyai output kosong dan execution count null.

Pengujian utama memakai data buatan 24 baris dan Transformer kecil acak dengan dimensi fitur 768/1024. Ini **fixture teknis**, bukan skor siswa, bobot pretrained atau bukti peningkatan AES. Satu pengujian memakai kelas tokenizer RoBERTa slow/fast dari Transformers dengan kosakata byte-BPE lokal buatan, untuk memeriksa template pasangan sesungguhnya. Akses tokenizer checkpoint pretrained Hugging Face dari lingkungan cloud gagal melalui proxy dengan `403 Forbidden`; tidak ada klaim smoke checkpoint pretrained asli atau training penelitian pada lingkungan ini. Notebook Colab tetap memuat checkpoint asli yang terkunci pada revision acuan.

## Perintah dan hasil akhir

Cwd `/workspace/Automated_Essay_Scoring`:

```bash
HF_HOME=/workspace/Automated_Essay_Scoring/outputs/context_pair/hf_cache \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
outputs/dataset300_env_py313/bin/python -m pytest \
  tests/test_context_pair_notebook.py -q -s \
  --basetemp=outputs/context_pair/final_verified_tmp \
  --junitxml=outputs/context_pair/final_verified_tests.xml
```

Exit **0**; **13 passed, 0 failed, 0 skipped, 0 xfailed**, durasi 78,16 detik. Pengujian mencakup identitas builder lintas interpreter, pasangan token, forward/backward, resume epoch dan eksekusi notebook lengkap. Satu warning deprecation Jupyter mengenai platformdirs tidak memengaruhi notebook. Versi Python `3.13.5`, PyTorch `2.6.0+cpu`, Transformers `4.48.3`, pytest `8.3.5`. Notebook fixture terisi dan bobot teknis disimpan di `outputs/context_pair/`, tidak diterbitkan.

## Perilaku yang dibuktikan

- Pasangan menempatkan konteks dan jawaban dengan empat special token RoBERTa. Soal dan referensi mendapat anggaran terpisah; pemotongannya dicatat.
- Jawaban panjang seluruhnya tercakup, konteks diulang pada tiap window, padding dan attention mask cocok, serta pemetaan konteks tetap mengikuti baris jawaban.
- Konteks diambil dari soal/referensi, tidak membaca skor guru. Normalisasi Unicode/karakter kontrol konsisten. Prediksi baru mensyaratkan soal dan referensi.
- Pada fixture Transformer, mengubah konteks mengubah representasi RoBERTa S3, tanpa mengubah fitur BGE. Encoder RoBERTa S3, proyeksi dan head tetap menerima gradien.
- S1 tidak bergantung pada konteks; S2 memakai vektor BGE yang sama. AST kelas `SingleScoreModel`, fungsi metrik, split, kelompok optimizer dan ekstraksi BGE identik dengan MLP acuan. Split fixture dibandingkan dengan fungsi acuan dan cocok.
- Vektor kata S3 dipetakan hanya ke posisi token jawaban yang sebenarnya pada input berpasangan. Lookup token sama saat konteks berubah, sedangkan vektor contextual dapat berubah; seluruh token mendapat cakupan.
- Interupsi setelah epoch pertama dan pemulihan menghasilkan state model, riwayat dan prediksi yang identik dengan run utuh. Dua snapshot terbaru dirujuk metadata; file epoch lebih lama tidak dihapus.
- Notebook dijalankan dari atas dengan import proyek diblokir, dataset fixture dan workspace terpisah. Semua 30 code cell produksi selesai, 15 fold tersimpan, setiap skenario menjalani final test sekali, dan lima Excel terbentuk.
- Hasil menyertakan token pasangan dan cakupan, vektor pretrained pasangan, vektor S3 fine-tuned, kata/kalimat dengan konteks, serta nilai guru/prediksi/pembulatan. File `final.pt` dan progres setiap fold/refit tetap ada.
- Pada simulasi runtime baru yang memakai penyimpanan permanen sama, seluruh hasil final dipakai ulang; tidak ada epoch baru dan hash model final tetap sama.

Notebook sumber dan mirror dibandingkan byte. Source publik dan seluruh perubahan diperiksa terhadap teks jawaban unggahan; data pribadi, fixture terisi, Excel, cache dan checkpoint tidak masuk commit. Manifest sumber/checksum dan bukti publikasi disimpan lokal di `outputs/context_pair/`. Pengujian ini menunjukkan fungsi program; kualitas RMSE penelitian baru harus dinilai setelah pengguna menjalankan eksperimen sebenarnya.

## Revisi runtime GPU

Atas permintaan pengguna, default terbaru mewajibkan GPU untuk menjalankan model: `REQUIRE_GPU=True`, metadata Colab `accelerator=GPU`, serta pesan perangkat aktif sebelum ekstraksi/training. `device='auto'` yang sudah ada memilih CUDA setelah pemeriksaan ini berhasil. GPU NVIDIA selain T4 juga diterima. Kode arsitektur, optimizer, split, metrik, epoch dan checkpoint tetap identik; perbandingan AST dengan commit `f775fbb` hanya menemukan perubahan pada cell 8 (pengaturan lingkungan) dan 13 (pemeriksaan perangkat). Fingerprint baru: `51f52bfa5badbdd7e5dcb26a75db3999be8ccb1e813ecd62d54501efb38ff6ec`.

Perintah pengujian memakai environment yang sama seperti di atas, dengan `--basetemp=outputs/context_pair/gpu_revision/tests_tmp` dan `--junitxml=outputs/context_pair/gpu_revision/tests.xml`. Exit **0**; **14 passed, 0 failed, 0 skipped, 0 xfailed**, 74,27 detik. Log: `outputs/context_pair/gpu_revision/tests.log`. Pengujian baru membuktikan penolakan CPU dan penerimaan informasi T4/GPU lain dengan objek CUDA tiruan. Eksekusi notebook lengkap tetap memakai fixture Transformer acak di CPU, dengan kewajiban GPU dinonaktifkan khusus pada salinan test. Seluruh 30 code cell, hasil 15 fold, lima Excel dan pemakaian ulang hasil tetap diperiksa. Tidak tersedia GPU fisik di lingkungan pengujian; hasil ini bukan bukti pengukuran CUDA/T4 asli atau skor penelitian.

Mengganti source mengubah identitas run. Checkpoint eksperimen lama tidak ditimpa atau dipaksa dilanjutkan lintas CPU/GPU. Pengguna perlu memilih T4 pada menu Runtime Colab dan menjalankan notebook terbaru dari atas.
