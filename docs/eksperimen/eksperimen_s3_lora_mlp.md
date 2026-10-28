# S3 RoBERTa–LoRA + BGE-M3 frozen + MLP

[Buka di Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_S3_LoRA_MLP_GPU.ipynb) · [Notebook](../../notebooks/experiments/AES_300_S3_LoRA_MLP_GPU.ipynb)

Notebook baru diturunkan langsung dari `AES_300_S3_MLP.ipynb`, tanpa konteks pair. Acuan dan eksperimen sebelumnya tetap utuh. Seluruh kode tersedia di 30 code cell; runtime tidak clone atau import proyek GitHub. Mirror identik berada di `aes-learning/notebooks/experiments/`.

| Skenario | Input | Bagian yang belajar |
| --- | --- | --- |
| S1 | Jawaban siswa → RoBERTa | Bobot RoBERTa dan head linear 768→1, seperti acuan |
| S2 | Jawaban siswa → BGE-M3 | Head MLP 1024→256→GELU→dropout→1; encoder frozen |
| S3 | Jawaban siswa → RoBERTa dan BGE-M3 | Adapter LoRA RoBERTa, LayerNorm tambahan, proyeksi BGE dan head MLP; kedua bobot pretrained asli frozen |

Pada S3, bobot asli RoBERTa frozen tetapi encoder tetap beradaptasi melalui LoRA; ini berbeda dari double frozen tanpa adapter. LoRA dipasang pada query/value attention setiap layer melalui **PEFT 0.14.0**, bukan implementasi penyesuaian bobot buatan. Tidak memakai kuantisasi atau QLoRA. BGE tetap frozen dan vektornya dihitung sekali. Fusion tetap: RoBERTa 768→LayerNorm; BGE 1024→Linear256→LayerNorm; concat 1024→Linear256→GELU→Dropout→Linear1.

## Parameter dan pembandingan

| Pengaturan | Nilai awal |
| --- | --- |
| Rank LoRA | 8 |
| Alpha LoRA | 16; skala alpha/r=2 |
| Target LoRA | Query dan value attention RoBERTa di seluruh layer |
| Dropout LoRA | 0; tidak menambah sumber dropout baru |
| LR LoRA S3 | 1e-4 |
| LR encoder S1 | 1e-5 |
| LR proyeksi/head | 1e-4 |
| Dropout head | 0,1 seperti acuan |
| Split | 80% development, 20% test; 5-fold CV dalam development |
| Batas epoch / patience | 25 / 5, sama untuk ketiga skenario |
| Microbatch / batch efektif | 1 / 8 |

Nilai LoRA merupakan **kandidat awal**, bukan hasil pencarian optimal. Selain jenis fine-tuning, LR parameter encoder S3 berubah dari 1e-5 menjadi LR adapter 1e-4; catat keduanya ketika membandingkan dengan acuan. Tidak ada pencarian hyperparameter otomatis. S1/S2 mempertahankan konfigurasi acuan tetapi hasil run ulang dapat berbeda akibat perangkat/numerik. Pemilihan epoch memakai RMSE validasi, final refit memakai ceil median epoch terbaik lima fold, lalu test akhir dilaporkan dengan RMSE, Pearson, QWK, Gwet AC2. MSE training memakai target 0–1; RMSE memakai satuan poin 0–100. Test lama yang berulang kali dilihat selama pengembangan tidak menjadi bukti konfirmasi independen untuk pemilihan metode baru.

## Menjalankan dengan GPU

1. Buka notebook, pilih **Runtime → Change runtime type → T4 GPU**. GPU NVIDIA lain juga diterima. `REQUIRE_GPU=True` dan `device='cuda'` mencegah training berjalan diam-diam pada CPU. Tokenisasi dan Excel tetap CPU.
2. Jalankan dari atas. Sambungkan Drive, lalu pilih `fix_dataset_300.xlsx` jika diminta. Dataset tidak ditanam di notebook atau repo publik. Setup mempertahankan PyTorch CUDA bawaan Colab dan memeriksa import PEFT.
3. Enam header dataset tetap: `Email address`, `response_number`, `response_text`, `soal`, `jawaban_referensi`, `rater`. **Hanya response_text menjadi masukan encoder**; rater adalah satu target skor guru 0–100. Soal/referensi tidak menjadi konteks pair.
4. Folder baru: `MyDrive/AES_300_S3_LoRA_MLP_Privat`. Checkpoint acuan/double frozen/context pair tidak dipakai untuk melanjutkan eksperimen baru. Jika runtime terputus, jalankan notebook yang sama dari atas dengan Drive/konfigurasi/perangkat yang cocok.
5. Jalankan cell unduhan terakhir untuk mendapatkan `Hasil_AES_Ringkas.zip` berisi lima Excel. Biarkan `NEW_ANSWER=''` untuk contoh held-out; isi jawaban baru bila ingin melakukan prediksi baru.

**Perbaikan import torchvision:** PEFT memuat model Transformers secara tidak langsung, yang dapat mencoba memuat torchvision dari runtime. Jika binary torchvision tidak cocok dengan PyTorch, import berhenti dengan `operator torchvision::nms does not exist`, sebelum training. Versi terbaru membatasi proses AES ke backend PyTorch teks dan menonaktifkan deteksi torchvision di Transformers **sebelum** import PEFT. Tidak menghapus torchvision, tidak membuat operator palsu, dan tidak mengubah model. Backend TensorFlow/Flax juga tidak digunakan oleh worker teks ini. Versi Transformers tetap dikunci seperti acuan.

Setup kini memakai overlay baru `outputs/dataset300_env_py313_text_lora` (atau `dataset300_env_text_lora` untuk Python 3.12). Overlay lama yang mungkin memuat salinan torch tidak dipakai otomatis. Pemasangan menggunakan `--ignore-installed --no-deps`; dependensi yang kurang/tidak cocok diperiksa dari metadata dan dipasang bertahap, dengan penolakan permintaan mengganti torch. Ini mencegah dependensi PEFT menarik ulang PyTorch. Tidak ada fallback pemasangan torch CPU. Error import berikutnya menampilkan bagian akhir traceback langsung, selain log lengkap.

Untuk kegagalan setup pada versi lama, buka notebook terbaru dan jalankan dari atas dengan T4 aktif. Error tersebut terjadi sebelum training; tidak perlu menghapus dataset atau hasil di Drive. Perubahan source menghasilkan identitas run baru; tidak memaksa checkpoint dari versi lama menjadi cocok.

## Penyimpanan dan keluaran

Tidak ada auto-delete, Google Drive API cleanup, atau pengosongan Sampah. Seluruh checkpoint epoch/fold tetap ada untuk pembersihan manual. S3 menyimpan **hanya LoRA + LayerNorm tambahan + proyeksi + MLP**, termasuk bobot terbaik untuk pemilihan epoch; progres tetap menyimpan optimizer, scheduler, scaler dan keadaan RNG. Bobot pretrained frozen dimuat kembali dari nama dan revision terkunci. S1 tetap menyimpan state penuh karena bobot aslinya dilatih. Saat memuat S3, semua nama parameter belajar harus cocok persis; checkpoint yang kehilangan adapter atau head ditolak.

Pertahankan `final.pt` setiap skenario dan JSON hasil. Untuk tahap aktif, pertahankan JSON progres serta dua `.pt` yang dirujuknya. Checkpoint epoch tahap yang sudah selesai menjadi kandidat hapus manual setelah hasil tahap/model final diperiksa. JSON progres bukan pengganti bobot.

Lima Excel mengikuti acuan: `01_Data_dan_Tokenisasi.xlsx`, `02_Vektor_Jawaban.xlsx`, `03_Vektor_Kata_dan_Kalimat.xlsx`, `04_Prediksi_dan_Evaluasi.xlsx`, `05_Riwayat_Pelatihan.xlsx`. Tabel parameter LoRA/frozen dan seluruh pengaturan LoRA masuk laporan pelatihan. Perbandingan nilai guru, prediksi mentah S1/S2/S3, dan salinan pembulatan tetap tersedia. Vektor S3 setelah training memakai adapter final aktif; lookup token tetap berasal dari embedding pretrained frozen. Ruang vektor RoBERTa 768 dan BGE 1024 berbeda, sehingga komponen antarmodel tidak dibandingkan langsung dengan cosine. Arsitektur tetap tidak menghasilkan vektor fusion per kata.

## Referensi dan batas bukti

Hu dkk., **LoRA: Low-Rank Adaptation of Large Language Models**, ICLR 2022: https://openreview.net/forum?id=nZeVKeeFYf9. Paper mendasari fine-tuning low-rank dengan bobot asli frozen; hasil NLU paper bukan janji RMSE AES Indonesia. Implementasi PEFT: https://github.com/huggingface/peft/tree/v0.14.0. Eksperimen ini mengadaptasi LoRA ke arsitektur S3 acuan; belum ada skor penelitian baru sebelum pengguna menjalankannya.

[Bukti pengujian teknis](../verifikasi/verification_s3_lora_mlp.md).
