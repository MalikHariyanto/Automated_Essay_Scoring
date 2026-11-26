# Ringkasan E43 dan E17C

[Unduh Excel revisi](Ringkasan_E43_dan_E17C.xlsx).

Koreksi permintaan pengguna: tab pertama E43 tiga input–satu trait, tab kedua E17C tiga input–multi-trait. Nama/URL lama Ringkasan_E4_dan_E17A.xlsx dipertahankan sebagai alias, tetapi isinya sudah sama dengan file revisi.

E43: hasil ringkas audit run yang tercatat dalam repo; test RMSE5,1265457/Pearson0,7590896/QWK0,7201403/AC2 0,9873756, meanCV5,8875995. Teknologi dan hyperparameter dicocokkan dengan notebook E43: ScalarMix4+gamma, BiGRU64 per arah, co-attention, masked mean, BGE frozen/projection/concat/MLP. Pasangan prediksi perjawaban E43 tidak tersedia dalam sumber yang ditemukan dan ditandai belum tersedia; tidak diisi dari E4. Angka ringkas bukan hasil perhitungan ulang dari workbook asli pada sesi revisi.

E17C tidak menggunakan cosine similarity. Pemilihan sebelumnya E17A berdasarkan RMSE test yang lebih rendah, bukan karena E17C memakai cosine. E17C mempunyai meanCV lebih rendah daripada E17A, tetapi tidak unggul pada semua metrik. Tab kedua diperbarui dari workbook04/05 E17C, termasuk seluruh270test+1070OOF dalam format empat trait (5360baris). Semua nilai dan selisih diperiksa terhadap sumber. Tanpa teks, identitas pribadi, email, token, vektor atau kelompok siswa.

Skala E43 0–100 berbeda dari rubrik multi-trait. RMSE kedua tab tidak langsung dapat diperingkatkan. Tidak ada training baru.
