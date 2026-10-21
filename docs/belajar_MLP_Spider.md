# Buku belajar hasil MLP

Unduh [belajar_MLP_Spider.html](belajar_MLP_Spider.html) dengan **Download raw file**, lalu buka di Chrome, Edge, atau Firefox. Halaman berjalan offline tanpa instalasi, CDN, atau training. Tampilan memakai tema Spider-Man full colour dari buku paper: splash warna, kota komik, pose di tepi halaman dan sticky note yang menimpa border.

Source mengikuti [commit 15bc7bb](https://github.com/MalikHariyanto/Automated_Essay_Scoring/commit/15bc7bbb2aee91ecaf44fb91a0139a8c21f64a20). Seluruh 30 cell kode unggahan `AES_300_S3_MLP (1).ipynb` sama dengan source pada commit itu. Buku tidak mengikuti eksperimen double-frozen yang dibuat kemudian, dan tidak mengubah notebook utama.

## Cara membaca

Ada 16 bab. Mulai dari hasil, lalu ikuti setup → dataset → split → token → vektor → arsitektur → training → CV/refit/test → prediksi → lima laporan Excel.

Setiap bab mempunyai empat tab:

- **Alur & cara baca:** proses input sampai output, istilah kontekstual, diagram lapisan yang dapat diklik, serta catatan asli notebook.
- **Kode + sticky note:** seluruh kode, nomor baris, syntax highlight, pencarian dan salin kode. Klik baris bertanda atau daftar catatan untuk membaca fungsi baris tersebut. Ada 120 catatan, termasuk 52 titik penting seperti nama checkpoint, pemuat model, freeze/fine-tuning, LayerNorm, concat dan head MLP.
- **Output tersimpan:** output asli notebook, tanpa menjalankan ulang kode. Log dan tabel mempunyai scroll sendiri; catatan menjelaskan satuan angka serta kaitannya dengan proses. Fungsi yang baru didefinisikan dibedakan dari tahap yang benar-benar sudah dieksekusi.
- **Jelajah Excel:** pilih salah satu dari lima workbook dan 41 sheet. Tabel menampilkan 25 baris serta delapan kolom komponen per halaman. Semua baris/komponen tetap tersedia. Klik sel untuk nilai lengkap; XLSX asli dapat diunduh tanpa perubahan.

Daftar bab, kode, catatan, output, dan tabel mempunyai area scroll agar halaman utama tidak memanjang mengikuti seluruh hasil. Ada navigasi keyboard, dialog istilah, tampilan HP, dan tombol jeda animasi.

## Memuat seluruh output pribadi

Salinan GitHub memuat kode lengkap, hasil agregat aktual, riwayat epoch, grafik validasi, metadata seluruh sheet, dan matriks guru/prediksi yang dikelompokkan per rentang sepuluh poin. Matriks itu membantu membaca hasil; metrik QWK/AC2 aslinya tetap memakai kategori integer 0–100.

Teks siswa, ID, skor individual dan seluruh vektor tidak dipublikasikan. Untuk menampilkan semuanya:

1. Buka HTML yang sudah diunduh.
2. Pilih **Notebook .ipynb** dan buka `AES_300_S3_MLP (1).ipynb` yang diunggah dalam percakapan ini.
3. Pilih **Hasil_AES_Ringkas.zip** dan buka `Hasil_AES_Ringkas (2).zip` dari run yang sama.
4. Seluruh 34 objek output tersimpan dan 41 sheet akan tersedia. Pembaca memeriksa kesesuaian source notebook dan SHA-256 ZIP; hasil run lain tidak dicampurkan.
5. Klik **Simpan HTML lengkap** untuk membuat satu HTML pribadi dengan kedua file sudah tertanam. Buka kembali HTML itu tanpa memilih file lagi.

File diproses di browser, tidak dikirim ke server atau disimpan ke localStorage. Script/widget dari output notebook tidak dijalankan. Simpan HTML lengkap sebagai berkas pribadi karena di dalamnya ada data siswa.

Versi lengkap siap pakai juga dibuat secara lokal di `outputs/mlp_study/belajar_MLP_Spider_Privat.html`; berkas ini tidak masuk GitHub.

## Hasil dan batas penjelasan

Run ini memakai satu skor guru 0–100 untuk 300 jawaban, bukan empat target rubrik. Development berjumlah 240 dan test 60. Validasi dilakukan dengan lima fold di development.

| Skenario | Arsitektur pada run ini | RMSE test, poin |
| --- | --- | ---: |
| S1 | RoBERTa fine-tuned → dropout → linear 768→1 | 6.3805 |
| S2 | BGE-M3 frozen → MLP 1024→256→1 | 6.9481 |
| S3 | RoBERTa fine-tuned → LayerNorm 768; BGE frozen → proyeksi 256; concat 1024 → MLP 256→1 | 6.7085 |

S3 memiliki rata-rata RMSE CV terkecil pada run tersimpan, sedangkan S1 terbaik di test. Hasil ini tidak membuktikan satu arsitektur selalu unggul. AC2 mendekati satu bukan persentase prediksi tepat. Loss MSE memakai target 0–1; RMSE memakai poin skor asli. Pembahasan dan grafik memakai hasil tersimpan, bukan angka hasil training baru.

Vektor kata RoBERTa S3 berasal dari cabang RoBERTa sesudah latihan. Fusion berlangsung pada tingkat jawaban; buku tidak mengarang vektor fusion per kata. Simulator window diberi label ilustrasi dan tidak dianggap tokenisasi aktual.

## Membangun ulang

Perintah berikut hanya membaca source Git dan snapshot agregat aman:

```bash
python scripts/build_mlp_study.py
```

Untuk membuat ulang snapshot dan HTML pribadi dengan file asli, gunakan Python yang mempunyai `openpyxl`:

```bash
python scripts/build_mlp_study.py \
  --notebook '/path/AES_300_S3_MLP (1).ipynb' \
  --results '/path/Hasil_AES_Ringkas (2).zip'
```

Builder membaca source dari commit tetap, bukan dari notebook eksperimen terbaru. Template, isi catatan, dan JSZip berlisensi MIT/GPL disertakan di repository. Tidak ada bobot encoder yang diunduh atau dilatih oleh builder. Bukti pemeriksaan ada di [laporan verifikasi](verification_mlp_study.md).
