# Melanjutkan proyek dari akun lain

Mulai dari **[ringkasan percakapan untuk AI](PANDUAN_AI_PROYEK.md)**, [status proyek](STATUS_PROYEK.md), [katalog](notebooks/experiments/README.md), dan [dua HTML terbaru](downloads/README.md). Pindah akun chat, GitHub, dan Google/Colab membawa berkas yang berbeda.

## Akun ChatGPT atau Codex baru

1. Simpan alamat repo dan teks serah-terima di bawah.
2. Buka atau hubungkan repo pada branch **`codex/aes-learning`**; ini juga branch default saat penataan 28 Oktober 2026.
3. Minta asisten membaca status, aturan proyek, dan rencana berikutnya sebelum mengubah kode.
4. Untuk membedah jawaban individual, unggah lagi ZIP/Excel yang diperlukan. Lampiran dan chat lama tidak otomatis berpindah.

Teks yang dapat disalin:

```text
Lanjutkan proyek AES saya:
https://github.com/MalikHariyanto/Automated_Essay_Scoring
Branch: codex/aes-learning.

Baca PANDUAN_AI_PROYEK.md, STATUS_PROYEK.md, AGENTS.md, PINDAH_AKUN.md, dan
RENCANA_EKSPERIMEN.md dahulu. E21–E26 memakai soal +
jawaban_referensi + response_text pada seluruh S1/S2/S3,
dengan label tunggal rater 0–100. S1 linear; S2/S3 MLP.
RoBERTa fine-tuned, BGE frozen; GPU Colab biasa, tanpa Drive.

E26 saat ini RMSE S3 paling rendah; E23 paling sering terdekat
pada persaingan enam eksperimen. Jangan samakan dua ukuran itu.
Dua HTML lengkap tersedia di downloads; kunci dibawa terpisah.
Jangan mengaku dapat membaca lampiran privat dari chat lama.

Untuk MLP LR5e-6, pisahkan model.head dari pooling/BiGRU/
attention/LayerNorm: HEAD_LR sekarang mengontrol lebih dari MLP.
Rencana eksperimen belum berarti kode atau hasilnya tersedia.
Lanjutkan permintaan berikutnya dengan mempertahankan acuan
serta membedakan yang ditulis, diuji fixture, dan dijalankan nyata.
```

## Akun GitHub baru

**Fork repo ke akun baru** untuk mempertahankan asal dan riwayatnya. Pastikan branch `codex/aes-learning` ikut tersedia. Untuk salinan lokal:

```bash
git clone --branch codex/aes-learning https://github.com/MalikHariyanto/Automated_Essay_Scoring.git
cd Automated_Essay_Scoring
```

Clone membawa source, riwayat dan paket HTML terenkripsi yang masuk Git. **Download ZIP source** hanya membawa snapshot branch, bukan seluruh riwayat. Pemindahan kepemilikan melalui GitHub Settings adalah langkah terpisah oleh pemilik repo; penataan ini belum memindahkan kepemilikan.

Pengunduh HTML menunjuk bagian paket pada commit tetap di repo sumber. Notebook/HTML juga menyimpan tautan source asal. Fork tidak otomatis mengganti alamat tersebut. Sebelum menghapus atau membatasi repo sumber, sesuaikan manifest `base` kedua pengunduh ke lokasi bagian paket yang sama di repo tujuan, lalu periksa ukuran/SHA-256. Jangan mengganti checksum agar cocok dengan file berbeda. Selama repo sumber dapat diakses, pengunduh lama tetap bekerja.

Dataset dan kunci tetap dipindahkan secara privat. Repo baru dapat menjalankan notebook ketika dataset diunggah ke Colab.

## Akun Google atau Colab baru

1. Buka notebook dari katalog; simpan salinan di akun tujuan bila diperlukan.
2. Pilih **T4 GPU**, jalankan dari atas, dan unggah dataset asli yang sama.
3. E21–E26 menyimpan hasil di **disk runtime lokal**. Menyalin notebook tidak memindahkan runtime/cache/checkpoint.
4. Untuk resume, pindahkan folder run lengkap sebelum runtime lama hilang. Untuk training baru, jangan memakai checkpoint eksperimen lain.

## Berkas yang perlu dibawa

| Berkas | Kegunaan | Ikut clone? |
| --- | --- | --- |
| Notebook/source | Melanjutkan implementasi | Ya |
| Dua pengunduh dan `.part` | Mendapatkan buku lengkap | Ya |
| Kunci pembuka | Membuka paket terenkripsi | **Tidak; simpan terpisah** |
| Dataset asli yang dipakai | Training ulang dengan data/label sama | Tidak |
| ZIP hasil eksperimen | Review jawaban/Excel | Tersedia terenkripsi di buku; bawa salinan ZIP asli secara privat |
| Checkpoint/folder run | Resume atau memuat model terlatih | Tidak |
| Chat/lampiran lama | Konteks dan data privat | Tidak |

Buku lengkap memuat ZIP/Excel hasil. Setelah dibuka, pilih eksperimen → meja belajar → **Jelajah Excel**, lalu unduh Excel/ZIP asli. Ini memulihkan laporan, bukan bobot. Simpan dataset asli juga; ekspor laporan belum tentu identik dengan workbook sumber untuk resume.

Untuk model selesai, pertahankan `final.pt` bila ingin memakai bobotnya. Untuk resume, simpan konfigurasi, manifest/split, penunjuk JSON dan checkpoint yang dirujuk. **JSON tidak menggantikan bobot `.pt`.** Pindah akun tidak mengharuskan menghapus progres yang diperlukan.

## Tentang penataan ini

Dokumentasi dikelompokkan dan output notebook dibersihkan. Mirror lama yang berbeda disimpan pada [arsip varian](notebooks/archive/README.md) sebelum disamakan dengan jalur utama. Source notebook utama dipertahankan. Dua HTML, paket, hasil numerik dan commit rujukan tetap.

Pembersihan output berlaku pada versi terbaru; riwayat commit lama tetap ada. Tidak dilakukan penulisan ulang riwayat Git. Salinan privat sebelum pembersihan hanya lokal pada lingkungan kerja saat penataan, bukan layanan backup lintas akun.
