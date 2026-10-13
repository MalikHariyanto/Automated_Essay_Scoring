# Menjalankan notebook Colab dari atas

Buka [notebook utama](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/01_belajar_alur_AES.ipynb). Salinan lama yang tersimpan di `aes-learning/notebooks/` telah disamakan; gunakan link utama untuk belajar berikutnya. Perubahan di GitHub tidak otomatis mengganti cell pada tab Colab yang sudah terbuka.

1. Pilih **Runtime → Change runtime type → CPU**, dan **Runtime version 2026.07** (Python 3.12.13). Daftar versi berasal dari [repo resmi Colab](https://github.com/googlecolab/backend-info#colabs-past-runtime-versions).
2. Buka notebook terbaru melalui link di atas. Jika ingin menyimpan hasil, simpan salinan pribadi di Drive; jangan kirim notebook berisi teks/output siswa ke GitHub publik.
3. Jalankan dari cell paling atas atau **Runtime → Run all**. Tidak perlu menambahkan cell clone, pip install, reinstall NumPy, atau diagnosis dari chat sebelumnya.
4. Cell input membuka pemilih file jika CSV/JSON belum ada. Pilih `dataset_aes_multitrait_DRAF.csv` dan `rubrik_multitrait_DRAF.json` asli; tunggu sampai upload selesai. Bila sudah ada, file itu digunakan kembali. Path bisa diganti pada cell input.
5. Baca audit, satu esai/rubrik, cleaning, token/vektor dan output UNTRAINED S1/S2/S3, lalu referensi AC2. Bagian supervised menampilkan alasan skip bila label atau pengesahan belum lengkap. Tabel empat metrik ada di bagian 11.

## Isi cell

| Langkah | Yang dilakukan | Bukti yang ditampilkan |
| --- | --- | --- |
| 1 | Mengambil/memperbarui source yang bersih dan memilih mode | Python, lokasi proyek, pilihan mode |
| 2 | Memasang lock CPU pada lingkungan Python AES terpisah | Versi pustaka, import nyata; log pemasangan |
| 3 | Memilih CSV/JSON asli | Path dan hash sumber |
| 4 | Audit/EDA melalui modul AES | Jumlah esai, label kosong/invalid, status, duplikasi, syarat training |
| 5 | Membaca satu esai dan rubrik JSON | Teks, status, deskriptor asli |
| 6 | Membandingkan cleaning | Teks sebelum/sesudah dan log turunan |
| 7–8 | Tokenisasi dan forward encoder asli | Token/window, vektor, shapes, tabel keluaran UNTRAINED S1/S2/S3 |
| 9 | Unit test dan referensi AC2/R | Hasil runner dan tabel AC2 Python/R pada fixture, bukan akurasi AES |
| 10 | Pemeriksaan supervised | Kebutuhan konkret atau eksekusi mode yang dipilih setelah layak |
| 11 | Tabel empat metrik per skenario/trait | 12 baris; nilai tersedia hanya dari test proposal yang sah |

Biarkan `MODE="inspect"`; sesuai permintaan belajar lengkap, `RUN_ENCODERS`, `RUN_TESTS`, dan `RUN_AC2_REFERENCE` default True. Notebook menampilkan alur encoder dan referensi AC2 tanpa training dari label kosong. BGE/RoBERTa asli memerlukan sekitar 2,8 GB unduhan dan head tetap UNTRAINED. Setup memakai PyTorch CPU; memilih T4 saja tidak membuat wheel CPU menjadi GPU. Ketiga pilihan bisa dibuat False untuk inspeksi ringan.

Colab menyiapkan R/irrCAC sebelum suite jika belum tersedia. Lima fixture unit test membandingkan point estimate AC2 Python dan R, termasuk kategori tak teramati; toleransi 5,01×10⁻⁶ mengikuti pembulatan lima desimal backend R. Jika dependency gagal, error tetap dilaporkan. `AES_RSCRIPT` dapat menunjuk backend R lokal. Jangan memindahkan angka fixture ini ke tabel evaluasi penelitian.

Tabel bagian 11 berisi RMSE, Pearson, QWK, dan AC2 untuk masing-masing trait S1/S2/S3. Semua nilai —/null pada data draf berarti belum tersedia. Nilai berasal dari prediksi test proposal setelah seluruh konfigurasi/refit dikunci; label, manifest, review independensi, dan revision harus cocok. Output diagnostik head baru tidak dievaluasi sebagai hasil penelitian. Hasil learning satu epoch disimpan terpisah di `learning_summary.json`.

Penjelasan matematika setelah tabel memakai satu contoh angka buatan untuk memperlihatkan mengapa QWK/AC2 dapat bernilai 1 sementara RMSE masih positif. Contoh itu tidak berasal dari data siswa dan tidak ditulis sebagai CSV/label/prediksi penelitian. Head diagnostik memakai seed 42 agar keluaran yang belum terlatih dapat diulang; ini bukan tuning konfigurasi atau hasil training.

## Mengapa setup berubah

Log pengguna menunjukkan Python 3.13 mencoba membangun `tokenizers==0.19.1` dari sumber dan gagal. Paket tersebut memiliki wheel Python 3.12. Setelah pindah ke 3.12, import scikit-learn di kernel masih gagal saat NumPy memanggil `_center`, sedangkan proses Python baru berhasil memakai NumPy 2.2.6, SciPy 1.15.3, dan scikit-learn 1.6.1. Ini membuktikan import ketiganya dalam proses baru; belum membuktikan penyebab tepat keadaan kernel lama.

Notebook baru tidak mengimpor pustaka numerik ke kernel Colab. Setup membuat lingkungan tanpa system-site-packages, memeriksa dependency/import, dan setiap langkah AES memakai proses baru. Semua perhitungan tetap berasal dari modul `src/aes`, sama dengan CLI. File CSV/JSON tidak diubah, dan tidak ada label buatan untuk membuat training berhasil.

Pada Colab pengguna, pembuatan lingkungan lewat `venv(with_pip=True)` juga gagal di `ensurepip`, sebelum dependency AES dipasang. Setup kini membuat lingkungan dengan `with_pip=False` dan memakai [opsi resmi pip `--python`](https://pip.pypa.io/en/stable/topics/python-option/) untuk memasang pip 25.0.1 di target yang belum punya pip. Python sistem tetap dipertahankan. Lingkungan parsial dengan interpreter/pyvenv.cfg yang sudah ada dipulihkan tanpa menghapus folder; kegagalan bootstrap tetap menghentikan setup dan dicatat dalam log.

Jika sudah membuka notebook dengan cell setup ini, setelah perbaikan dipublikasikan cukup jalankan kembali **cell 1** (fetch/update kode), lalu **cell 2** (setup). Tidak perlu cell tambahan atau menghapus lingkungan yang gagal. Salinan notebook lama tanpa cell update sebaiknya diganti melalui link utama.

Folder `outputs/` menyimpan lingkungan, log, audit, preview, cache, dan konfigurasi efektif secara lokal. Restart session biasa mempertahankan file pada mesin yang sama. Mengganti/menghapus runtime dapat menghilangkan folder sementara; notebook dapat menyiapkan ulang dan meminta dua file asli kembali.
