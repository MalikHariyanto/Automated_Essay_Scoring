# Buku AES: E6 sebagai acuan belajar

> Dokumen tahap terdahulu. Angka dan keputusan dibaca dalam konteks versi ini; untuk kondisi aktif gunakan [status proyek](../../STATUS_PROYEK.md) dan [panduan AI](../../PANDUAN_AI_PROYEK.md).

Meja belajar memuat seluruh 30 blok kode E6 `AES_300_E6_S3_LR5e6_CPU_Reuse.ipynb` dari commit `d8c21ac`. Source ditampilkan apa adanya, dengan 126 catatan termasuk LR S3 5e-6, optimizer per skenario, verifikasi serta pemakaian ulang S1/S2. Versi tiga seed belum mempunyai hasil dan tidak menggantikan E6 yang sudah menghasilkan Excel.

Hasil meja belajar berasal dari ZIP E6: metrik, riwayat, guru/prediksi, token dan vektor. Notebook eksekusi E6 tidak diunggah bersama ZIP, sehingga tidak ada klaim bahwa source runtime cocok byte demi byte. Output notebook MLP awal tidak ditempelkan pada source E6. Tab output menyajikan pratinjau Excel E6 dengan asal bukti yang jelas. Snapshot dan seluruh berkas sumber MLP awal tetap dipertahankan sebagai E2.

Halaman awal memiliki dua tujuan: ringkasan eksperimen dan meja belajar E6. Navigasi utama mengikuti fungsi: ringkasan, perbandingan1vs1, tabel/arti metrik, dan meja belajar. Halaman terpisah LoRA/frozen serta kontrol unggah notebook/ZIP dan ekspor ulang HTML di dalam buku dihapus. Seluruh hasil sudah diembed pada buku lengkap; unduhan XLSX asli tetap tersedia.

Perbandingan1vs1 menyediakan E0/E2/E3/E4/E5 di kiri dan E6 tetap di kanan, termasuk metode, source diff, kurva training/validasi, seluruh60pasangan prediksi dan empat metrik. Label E2 menjadi **MLP awal**. Ringkasan lintas eksperimen tetap menampilkan eksperimen sebagai kolom,90hasil fold,60jawaban/360prediksi dan peringkat kedekatan keseluruhan.

## Rujukan yang dapat ditelusuri

Menu Metode & sumber menghubungkan tiap eksperimen ke notebook pada commit tetap, paper, serta implementasi resmi jika tersedia. Enam URL notebook sudah diambil kembali dari GitHub dan cell source cocok. README resmi RoBERTa, BGE-M3, BERT, LoRA dan stabilitas fine-tuning berhasil dibaca. Beberapa laman arXiv/ACL ditolak jaringan; batas akses dicatat, tidak diklaim berhasil dibaca ulang.

Koleksi pengguna m04/p14/m14/p10 menjadi bukti tambahan. Paper p10 mendukung prinsip menggabungkan kelompok fitur, tetapi memakai fitur linguistik dan LwXGBoost. Buku tidak mengklaim paper tersebut memakai BGE–RoBERTa–MLP. LR5e-6, ukuran hidden256, patience5 dan lokasi normalisasi merupakan pengaturan eksperimen; sumber mekanisme tidak dianggap jaminan keunggulan atau bukti replikasi identik.

## Pemeriksaan

- 32 pengujian unit/integritas lulus; 0 failed/skipped/xfail. Source kode,55anchor penting, highlighting, snapshot awal, asal angka E6, pin source setiap eksperimen dan batas sitasi diperiksa.
- 11 kelompok pemeriksaan browser lulus: navigasi, E6 sebagai acuan1vs1, grafik, source diff, rujukan, tabel lintas eksperimen, viewport390/768/1440, kode/tabel fullscreen, pemisahan arsip E2/E6,60jawaban/360prediksi, serta unduhan XLSX byte-identik.
- 16 pemeriksaan scroll mouse/sentuhan lulus. Fallback meneruskan gestur hanya ketika seluruh panel di bawah pointer sudah mencapai batas; tidak menggerakkan halaman di belakang modal fullscreen.
- Dua pemeriksaan paket browser lulus: pengunduh menyusun ZIP byte-identik, dan buku terenkripsi dibuka dengan kunci sebelumnya serta menampilkan enam eksperimen/360prediksi setelah jaringan diputus.
- Tidak ada training/evaluasi model baru. Metrik penelitian yang ditampilkan adalah hasil tersimpan.

Command pemeriksaan, versi/hash sumber, hasil browser, log scroll dan verifikasi paket disimpan privat pada `outputs/mlp_study/e6_refresh_*`. Notebook, ZIP asli dan teks/nilai individual tidak ditambahkan ke source publik. Buku lengkap untuk GitHub dikemas terenkripsi dengan kunci sebelumnya.
