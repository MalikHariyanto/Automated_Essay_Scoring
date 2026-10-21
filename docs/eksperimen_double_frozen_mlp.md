# Eksperimen S3 double frozen + MLP

[Buka di Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_S3_DoubleFrozen_MLP.ipynb) · [File notebook](../notebooks/experiments/AES_300_S3_DoubleFrozen_MLP.ipynb)

Eksperimen ini menguji apakah S3 lebih baik ketika kedua encoder hanya menghasilkan fitur tetap. Notebook utama S3 MLP untuk acuan/semi-final dan salinannya tidak diubah. Eksperimen baru tidak mengambil bobot S3 fine-tuned untuk melanjutkannya: model dan penyimpanan eksperimennya terpisah.

## Bagian model yang belajar

Encoder mengubah teks menjadi vektor. Head mengolah vektor tersebut menjadi prediksi skor.

| Skenario | Encoder | Sesudah encoder | Bagian yang dilatih |
| --- | --- | --- | --- |
| S1, kontrol | RoBERTa Indonesia fine-tuned | Mean CLS 768 → dropout → linear 1 | Encoder dan head, sama dengan acuan |
| S2, kontrol | BGE-M3 frozen | CLS + L2 1024 → MLP 256 → skor | MLP, sama dengan acuan |
| S3, eksperimen | RoBERTa dan BGE-M3 frozen | RoBERTa 768 → LayerNorm tambahan; BGE 1024 → proyeksi 256 + LayerNorm; concat 1024 → MLP 256 → skor | Proyeksi, LayerNorm tambahan dan MLP |

Pada S3, `requires_grad_(False)` mencegah bobot encoder RoBERTa diperbarui, forward encoder tidak membangun gradien, dan mode `eval()` menjaga dropout encoder tetap mati. BGE-M3 tetap menghasilkan fitur frozen seperti versi acuan. LayerNorm **tambahan di luar encoder** serta MLP tetap belajar. MLP memakai Linear → GELU → dropout → Linear. Tidak menambahkan Ridge atau MLP baru pada S1.

Vektor kata S3 yang ditampilkan adalah keluaran cabang RoBERTa frozen. Karena encoder tidak dilatih, vektor mentahnya tetap sama dengan pretrained; LayerNorm tambahan, proyeksi dan hasil fusion dapat berubah. Penggabungan terjadi pada tingkat jawaban, sehingga tidak dibuat vektor fusion per kata yang tidak dihasilkan model.

## Alur dan cara menjalankan

1. Buka notebook eksperimen lalu jalankan cell dari atas. Runtime **CPU didukung**; pilih **Runtime → Change runtime type → T4 GPU** bila tersedia untuk mempercepat training.
2. Sambungkan Drive dan izinkan akses API menggunakan akun yang memiliki folder penyimpanan. Otorisasi ini memungkinkan kode menghapus checkpoint usang secara permanen. Kredensial tidak ditulis ke notebook atau file hasil.
3. Pilih `fix_dataset_300.xlsx` jika diminta. Dataset tetap privat. Pembacaan data, tokenisasi, embedding, training, prediksi, evaluasi, dan ekspor berada langsung di cell, tanpa clone/import proyek GitHub.
4. Jalankan hingga cell terakhir untuk mengunduh `Hasil_AES_Ringkas.zip`, berisi lima Excel seperti acuan: data/tokenisasi, vektor jawaban, vektor kata/kalimat, prediksi/evaluasi, serta riwayat pelatihan.

Dataset menggunakan satu skor guru 0–100. Split tetap 80% development dan 20% test, dengan 5-fold CV di development. Maksimum 25 epoch per fold dan patience 5 berlaku sama untuk ketiga skenario. S1 tetap learning rate encoder `1e-5`; proyeksi/head `1e-4`. S3 tidak mempunyai kelompok optimizer encoder karena kedua encoder frozen. Empat metrik tetap RMSE, Pearson, QWK, dan Gwet AC2. Jumlah epoch refit final dipilih dari hasil CV sebelum evaluasi test.

`REQUIRE_GPU=False` pada cell konfigurasi mengizinkan training tanpa GPU. `device='auto'` memilih CUDA bila tersedia, atau CPU bila tidak; gunakan `device='cpu'` jika ingin memaksa CPU meskipun GPU tersedia. Batas epoch, model dan evaluasi tidak dikurangi otomatis. CPU memakai float32 dan tidak mengaktifkan AMP CUDA. Auto-save/resume dan pembersihan checkpoint Drive tetap berjalan, tetapi waktu training tiga skenario dan lima fold dapat jauh lebih lama daripada T4.

## Penyimpanan dan penghapusan otomatis

Folder baru adalah **`MyDrive/AES_300_DoubleFrozen_MLP_Privat`**. Dataset, konfigurasi dan fingerprint menentukan run tersendiri di dalamnya. Folder acuan `AES_300_MLP_Privat` tidak termasuk cakupan pembersihan.

| Saat | Yang dipertahankan | Yang dihapus permanen |
| --- | --- | --- |
| Epoch aktif selesai disimpan | Dua snapshot terbaru; saat penunjuk server tertinggal, tambahkan paling banyak dua snapshot yang masih dirujuk server | Snapshot epoch yang lebih lama dan tidak dirujuk kedua penunjuk, pada skenario, fold/refit dan run yang sama |
| Fold selesai | Hasil CV, prediksi validasi, riwayat dan split | Checkpoint sementara fold tersebut setelah hasilnya terverifikasi |
| Refit selesai | `final.pt`, penanda selesai dan pilihan epoch CV | Checkpoint sementara refit setelah checksum model final terverifikasi |

Snapshot memuat keadaan model, optimizer, scheduler, early stopping dan pengacak, sehingga pemulihan melanjutkan keadaan belajar. Bobot terbaik fold ikut tersimpan di dalam snapshot; mempertahankan dua generasi terbaru tidak membuang pilihan epoch terbaik.

Pembersih memeriksa folder yang tepat, isi metadata hasil, ukuran dan checksum checkpoint pengganti di Drive API. Ia juga dapat menghapus checkpoint usang yang sudah berada di Sampah **pada cakupan tersebut**; tidak mengosongkan Sampah Drive global. Dataset, Excel, hasil CV, model final, folder lain dan eksperimen lain tidak menjadi kandidat. File sementara `.pending` pada tahap aktif tidak dihapus.

Penunjuk progres lokal dan server tidak harus langsung identik selama training aktif. Jika server masih menunjuk versi lama yang sah, file-file yang dirujuknya tetap diverifikasi dan dipertahankan bersama dua snapshot terbaru. Ini menjaga jalur pemulihan dari kedua versi tanpa menghentikan training hanya karena penunjuk belum sinkron. Isi file rusak, checksum salah, penunjuk server lebih baru/berbeda tahap, atau perubahan penunjuk di tengah penghapusan tetap menghentikan pembersihan. Hasil fold/model final tetap memerlukan bukti server yang cocok. [Memasang perbaikan pada runtime yang sudah berjalan](perbaikan_sinkronisasi_checkpoint_drive.md).

Jika sinkronisasi Drive belum selesai, pembersihan mencoba kembali. Jika tetap gagal, proses berhenti dengan progres serta antrean pembersihan tersimpan, supaya training tidak terus menghasilkan file besar. Jalankan notebook yang sama dari atas dengan konfigurasi dan perangkat yang cocok: hasil fold yang lengkap dipakai kembali, lalu tahap terputus dilanjutkan. Notebook ini tidak membersihkan checkpoint lama milik acuan atau run lain.

Pengujian menggunakan API Drive simulasi dan encoder kecil; akun Drive pengguna serta T4 fisik belum diuji langsung di lingkungan pengembangan. Notebook ini belum menghasilkan skor penelitian baru. Frozen + MLP adalah percobaan yang perlu dinilai dari hasil CV; peningkatan RMSE tidak dijamin. [Bukti pengujian](verification_double_frozen_mlp.md).

## Memperbarui source eksperimen

Notebook ini membawa seluruh runtime sendiri. File pembangunnya adalah `scripts/build_double_frozen_notebook.py`; kode pembersih yang disalin ke cell berasal dari `src/aes/teacher_score/drive_checkpoint_cleanup.py` dan `experimental_checkpoints.py`. Pembangun hanya menulis notebook eksperimen, tidak menulis notebook utama. Gunakan Python 3.11 untuk membangun ulang fingerprint AST dengan lingkungan pembangun saat ini.
