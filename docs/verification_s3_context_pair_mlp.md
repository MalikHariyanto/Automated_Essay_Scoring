# Verifikasi S3 context pair + MLP

23 Oktober 2026. Pekerjaan dari HEAD `b537e32501af2a33e787ca327b37bcaf0f92641a`, branch `codex/aes-learning`. Notebook sumber MLP, double frozen, versi manual serta notebook utama tidak ditulis ulang. Perubahan menghasilkan notebook baru dan mirror di `aes-learning/notebooks/experiments/`.

## Identitas dan batas pengujian

Source dibangun dengan `python scripts/build_context_pair_notebook.py`. Fingerprint kode mandiri: `f2b8db3c043b81cd46cdf14df07c422771fd32debd5e55967daff5e2cc343276`. Hash memakai source teks agar tidak bergantung pada perbedaan representasi AST Python 3.12/3.13; kedua interpreter menghasilkan notebook identik byte. Builder menanamkan definisi helper, tidak menambahkan import proyek atau clone GitHub pada runtime. Semua code cell publik mempunyai output kosong dan execution count null.

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
