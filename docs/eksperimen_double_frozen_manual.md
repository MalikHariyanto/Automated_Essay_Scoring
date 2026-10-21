# Double frozen + MLP tanpa penghapusan otomatis

[Buka Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_S3_DoubleFrozen_MLP_Manual.ipynb) · [File notebook](../notebooks/experiments/AES_300_S3_DoubleFrozen_MLP_Manual.ipynb)

Versi ini mengikuti alur notebook MLP acuan, dengan S3 double frozen dan checkpoint yang dihapus manual. Notebook utama/semi-final serta versi eksperimen sebelumnya tidak ditimpa.

| Skenario | Encoder | Bagian yang belajar |
| --- | --- | --- |
| S1 | RoBERTa Indonesia fine-tuned | Encoder dan head linear, sama dengan acuan |
| S2 | BGE-M3 frozen | MLP, sama dengan acuan |
| S3 | RoBERTa dan BGE-M3 frozen | LayerNorm tambahan, proyeksi BGE dan MLP setelah concat |

S3 tetap memakai RoBERTa 768 → LayerNorm tambahan, BGE 1024 → proyeksi 256 + LayerNorm, concat 1024 → Linear 256 → GELU → dropout → Linear 1. Encoder RoBERTa tetap eval dan tidak menerima gradien; head/proyeksi tetap dilatih. Tidak menambahkan Ridge atau MLP pada S1.

Jalankan notebook dari atas, sambungkan Drive seperti biasa, dan pilih `fix_dataset_300.xlsx` jika diminta. CPU didukung langsung; GPU/T4 bisa dipilih untuk mempercepat training. Konfigurasi, komentar kode dan tampilan hasil mengikuti notebook MLP. Tidak ada izin Google Drive API tambahan, permintaan cleanup, atau pemeriksaan pembersihan yang menahan training.

Progres tetap disimpan otomatis setiap epoch. Penunjuk progres mencatat dua snapshot terbaru untuk resume; **seluruh file checkpoint epoch yang lebih lama tetap disimpan**. Setelah fold/refit selesai, checkpoint dan penunjuknya juga tetap ada. Penghapusan dilakukan manual oleh pengguna. Hasil CV/model final menandai tahap selesai sehingga menjalankan ulang notebook yang sama dapat memakai hasil yang lengkap.

Folder penyimpanan baru: **`MyDrive/AES_300_DoubleFrozen_MLP_Manual_Privat`**. Versi ini memulai run baru yang terpisah, bukan kelanjutan otomatis checkpoint dari notebook auto-delete. Checkpoint versi lama tidak dipindahkan, dihapus atau diubah. Fingerprint berbeda agar hasil kedua versi tidak tercampur.

Dataset tetap satu skor guru 0–100; split 80/20, 5-fold CV development, maksimum 25 epoch per fold, patience 5, learning rate dan batch tetap. Prediksi, evaluasi RMSE/Pearson/QWK/Gwet AC2 dan perbandingan nilai guru/prediksi/pembulatan tetap tersedia. Cell terakhir menghasilkan `Hasil_AES_Ringkas.zip` berisi lima Excel: data/tokenisasi, vektor jawaban, vektor kata/kalimat, prediksi/evaluasi, dan riwayat pelatihan.

Notebook mandiri, tanpa clone/import proyek. Data/output siswa dan bobot tidak dipublikasikan. Source dibangun dengan `python scripts/build_double_frozen_manual_notebook.py`; perintah ini hanya menulis notebook manual baru. [Bukti pengujian](verification_double_frozen_manual.md).
