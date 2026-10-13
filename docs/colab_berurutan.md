# Menjalankan notebook Colab dari atas

Buka [notebook utama](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/01_belajar_alur_AES.ipynb). Salinan lama yang tersimpan di `aes-learning/notebooks/` telah disamakan; gunakan link utama untuk belajar berikutnya. Perubahan di GitHub tidak otomatis mengganti cell pada tab Colab yang sudah terbuka.

1. Pilih **Runtime → Change runtime type → CPU**, dan **Runtime version 2026.07** (Python 3.12.13). Daftar versi berasal dari [repo resmi Colab](https://github.com/googlecolab/backend-info#colabs-past-runtime-versions).
2. Buka notebook terbaru melalui link di atas. Jika ingin menyimpan hasil, simpan salinan pribadi di Drive; jangan kirim notebook berisi teks/output siswa ke GitHub publik.
3. Jalankan dari cell paling atas atau **Runtime → Run all**. Tidak perlu menambahkan cell clone, pip install, reinstall NumPy, atau diagnosis dari chat sebelumnya.
4. Cell input membuka pemilih file jika CSV/JSON belum ada. Pilih `dataset_aes_multitrait_DRAF.csv` dan `rubrik_multitrait_DRAF.json` asli; tunggu sampai upload selesai. Bila sudah ada, file itu digunakan kembali. Path bisa diganti pada cell input.
5. Baca audit, satu esai/rubrik, dan cleaning. Bagian supervised menampilkan alasan skip bila label atau pengesahan belum lengkap.

## Isi cell

| Langkah | Yang dilakukan | Bukti yang ditampilkan |
| --- | --- | --- |
| 1 | Mengambil/memperbarui source yang bersih dan memilih mode | Python, lokasi proyek, pilihan mode |
| 2 | Memasang lock CPU pada lingkungan Python AES terpisah | Versi pustaka, import nyata; log pemasangan |
| 3 | Memilih CSV/JSON asli | Path dan hash sumber |
| 4 | Audit/EDA melalui modul AES | Jumlah esai, label kosong/invalid, status, duplikasi, syarat training |
| 5 | Membaca satu esai dan rubrik JSON | Teks, status, deskriptor asli |
| 6 | Membandingkan cleaning | Teks sebelum/sesudah dan log turunan |
| 7–8 | Tokenisasi dan forward encoder asli jika dipilih | Token/window, vektor, shapes, keluaran UNTRAINED |
| 9 | Unit test dan referensi AC2/R jika dipilih | Hasil runner dan backend, bukan akurasi AES |
| 10 | Pemeriksaan supervised | Kebutuhan konkret atau eksekusi mode yang dipilih setelah layak |

Biarkan `MODE="inspect"`, `RUN_ENCODERS=False`, `RUN_TESTS=False`, dan `RUN_AC2_REFERENCE=False` untuk putaran pertama. Notebook selesai dengan inspeksi dan pesan skip yang benar pada file draf. Untuk melihat alur encoder, ubah `RUN_ENCODERS=True` di cell 1 dan jalankan kembali dari atas; tidak perlu cell tambahan. Ini memakai BGE/RoBERTa asli, sekitar 2,8 GB unduhan, tetap head UNTRAINED. Setup ini memakai PyTorch CPU; memilih T4 saja tidak membuat wheel CPU menjadi GPU.

Untuk unit test, pilih `RUN_TESTS=True`. Di Colab, aktifkan `RUN_AC2_REFERENCE=True` juga untuk menyiapkan R/irrCAC sebelum suite. Tanpa R/irrCAC, verifikasi referensi belum terbukti; dependency yang gagal tetap dilaporkan. Jangan menyebut tahap yang belum dipilih sebagai selesai.

## Mengapa setup berubah

Log pengguna menunjukkan Python 3.13 mencoba membangun `tokenizers==0.19.1` dari sumber dan gagal. Paket tersebut memiliki wheel Python 3.12. Setelah pindah ke 3.12, import scikit-learn di kernel masih gagal saat NumPy memanggil `_center`, sedangkan proses Python baru berhasil memakai NumPy 2.2.6, SciPy 1.15.3, dan scikit-learn 1.6.1. Ini membuktikan import ketiganya dalam proses baru; belum membuktikan penyebab tepat keadaan kernel lama.

Notebook baru tidak mengimpor pustaka numerik ke kernel Colab. Setup membuat lingkungan tanpa system-site-packages, memeriksa dependency/import, dan setiap langkah AES memakai proses baru. Semua perhitungan tetap berasal dari modul `src/aes`, sama dengan CLI. File CSV/JSON tidak diubah, dan tidak ada label buatan untuk membuat training berhasil.

Folder `outputs/` menyimpan lingkungan, log, audit, preview, cache, dan konfigurasi efektif secara lokal. Restart session biasa mempertahankan file pada mesin yang sama. Mengganti/menghapus runtime dapat menghilangkan folder sementara; notebook dapat menyiapkan ulang dan meminta dua file asli kembali.
