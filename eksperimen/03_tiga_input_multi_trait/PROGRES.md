# Progres eksperimen multi-trait

## Hasil historis E1–E4 reset

Keempat eksperimen telah selesai training dan evaluasi pada Tesla T4; hasil diambil dari workbook pengguna. Dataset sama: 1.340 jawaban, 134 kelompok, 10 soal; development 1.070/test 270. Keanggotaan split dan lima fold dicocokkan langsung, tidak hanya hash metadata. RMSE test dihitung ulang dari prediksi dan cocok. E3/E4 diverifikasi terhadap checksum seluruh pecahan dan manifest.

E4 terbaik nominal: CV 0,425284/test 0,466443. E2 CV 0,425449/test 0,466544 tetap kontrol; perbaikan test E4 hanya 0,000101 poin (0,022%), belum bukti keunggulan yang meyakinkan dari satu run. E3 CV 0,425778/test 0,472422; membantu kebahasaan/mekanik tetapi mengorbankan isi/organisasi. E1 CV 0,426831/test 0,468598.

| Trait | RMSE E2 | RMSE E3 | RMSE E4 |
| --- | ---: | ---: | ---: |
| Isi | 0,667898 | 0,693697 | 0,667114 |
| Organisasi | 0,328005 | 0,335481 | 0,328252 |
| Kebahasaan | 0,417080 | 0,406660 | 0,417389 |
| Mekanik | 0,378184 | 0,365508 | 0,378514 |

E3 best epoch 3/5/5/3/5; E4 2/10/5/3/5. Refit keduanya 5 epoch. Loss training turun sementara validasi memburuk setelah checkpoint terbaik, konsisten dengan overfitting. Menambah patience saja belum didukung hasil.

Attention E4 aktif: diagonal 0, baris berjumlah 1, norma regresor konteks dari 0 menjadi 0,0176–0,0189. Prediksi berubah rata-rata 0,00372 poin dari E2. Ini inspeksi checkpoint final, bukan penjelasan kausal atau OOF.

Skor rendah masih sulit dikenali. Organisasi skor 1 dikenali E2 11/35, E3 9/35, E4 11/35; kebahasaan 19/70, 22/70, 20/70; mekanik 19/56, 24/56, 19/56. Ini menjadi alasan menguji penanganan distribusi skor timpang dengan bobot yang dihitung hanya dari training fold.

## Status terbaru · 24 November 2026

### Jalur aktif: E17C concat + dropout 0,1

Notebook E17C ditambahkan ke branch default agar tautan jalur aktif tidak hanya menunjuk ke branch eksperimen. Konfigurasi yang tercantum di notebook:

- S3 saja; input soal, kunci jawaban, dan jawaban siswa.
- RoBERTa 768 dimensi di-fine-tune penuh; BGE-M3 1.024 dimensi frozen.
- BGE diproyeksikan 1.024→256 dan dinormalisasi; hasilnya di-concat dengan RoBERTa menjadi 1.024 dimensi.
- MLP bersama 256 unit, GELU, dropout 0,1, dan empat output.
- MSE pada target rubrik 0–1; loss relasi antartrait 0.
- Lima fold, maksimum 25 epoch, patience 5, train seed 42, split seed 2026, effective batch 8, microbatch 1.
- Kontrol E14 S3 memakai dropout MLP 0,2. Perubahan E17C yang dirancang adalah dropout 0,2→0,1; tetapi E17C dan E17-A/B tidak boleh dianggap satu rangkaian ablation yang identik karena strategi fine-tuning berbeda.

**Status bukti:** notebook/kode tersedia. Belum ada workbook hasil E17C yang dapat diverifikasi pada branch repository yang diperiksa. Jangan menyatakan E17C sudah lolos full training, test, atau ekspor.

### E17-A dan E17-B tetap eksperimen terpisah

- E17-A memakai partial fine-tuning: embedding dan delapan layer bawah RoBERTa frozen; empat layer atas trainable.
- E17-B mempertahankan konfigurasi E17-A dan mengubah loss Isi menjadi tempered LDS inverse-square-root (power 0,5). Tiga loss trait lain tetap MSE biasa. Bobot dihitung dari indeks training pada setiap fit; validasi dan test tidak berbobot.
- E17-B berada pada branch `codex/e17b-isi-tempered-lds`; PR #1 masih draft. Synthetic preflight dan inspeksi kode bukan bukti training GPU atau peningkatan akurasi.

### E18A dinonaktifkan

E18A BGE relevance/cosine tidak menjadi jalur aktif. README repository mengarahkan pengembangan ke E17C concat tanpa cosine similarity. Branch E18A dipertahankan sebagai riwayat, bukan digabungkan ke jalur aktif.

## Keputusan dan batas interpretasi

1. Jalankan E17C pada T4 dan verifikasi lima fold, refit, test, workbook, manifest, dan ZIP sebelum menulis hasil.
2. Bandingkan E17C dengan E14 S3 pada split yang sama. Fokus pada RMSE OOF gabungan/per trait, Pearson, QWK, Gwet AC2, confusion matrix, serta deteksi skor rendah.
3. E17-B adalah alternatif untuk masalah kelas Isi yang timpang. Perbandingannya terhadap E17C bukan ablation satu faktor karena E17-B memakai partial fine-tuning, sedangkan E17C full fine-tuning.
4. Jangan memakai test untuk memilih hyperparameter berulang kali. Hasil satu seed tidak membuktikan signifikansi statistik.
5. Identitas, teks jawaban siswa, dan vektor besar tetap privat; hanya hasil yang sudah dianonimkan yang boleh dipublikasikan.

[Notebook E17C](notebooks/AES_Reset_E17C_S3_Concat_Dropout01_Native_Learnable_GPU.ipynb) · [Katalog eksperimen](README.md) · [Acuan Referensi Kedua](../../docs/skripsi/Acuan_Referensi_2_AES_E17A_E17B_E17C.md)
