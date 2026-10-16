# Colab terbaru: satu skor guru, tiga skenario

Buka [03_AES_dataset_guru_300.ipynb di Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/03_AES_dataset_guru_300.ipynb). Upload `fix_dataset_300.xlsx` pada cell input, lalu jalankan berurutan. Gunakan runtime GPU; Python **3.12 dan 3.13** didukung dengan dependency yang dipilih otomatis. Jangan menerbitkan notebook yang sudah berisi teks/nilai siswa.

Ini penyesuaian khusus permintaan dataset terbaru, dengan scope `teacher_single_score_dataset300`. Pipeline lama empat trait 1–5, notebook 01/02 dan HTML pembelajarannya tetap mempunyai cakupan semula. Kode baru tidak mengubah label/rubrik lama atau menyebut hasil latihan AI sebagai nilai guru.

## Mengapa satu output, bukan empat rubrik?

File unggahan berisi 300 respons, 25 kelompok email, dan 12 pertanyaan. Kolom yang tersedia adalah `Email address`, `response_number`, `response_text`, `soal`, `jawaban_referensi`, dan satu kolom nilai `rater`. Skor teramati 65–99. Tidak ada empat nilai terpisah maupun deskriptor rubrik empat dimensi. Kolom kosong tanpa header tidak menjadi fitur.

Secara pemrograman, head empat output bisa dibuat. Namun, satu nilai tidak menentukan empat komponen penilaian: nilai 80 bisa berasal dari berbagai kombinasi isi, organisasi, kebahasaan, dan mekanik. Menyalin 80 ke empat kolom atau membaginya rata tidak menghasilkan anotasi empat dimensi. Model seperti itu tidak dapat dievaluasi sebagai penilai empat rubrik guru.

Untuk studi empat dimensi, perlu rubrik dengan indikator/deskriptor dan empat label yang diberikan manusia pada setiap respons, idealnya disertai pemeriksaan kesepakatan antarpenilai. Sistem AI boleh membantu mengusulkan anotasi secara terpisah, tetapi usul tersebut bukan label guru. Workflow baru ini mempelajari satu skor guru yang memang tersedia, output `[N,1]`; tidak memaksakan empat label.

Isi pertanyaan membahas metodologi penelitian. File saja tidak membuktikan bahwa respondennya siswa SMA kelas XI atau bahwa tugasnya sama dengan rancangan esai penelitian sebelumnya. Kesesuaian populasi, tugas dan rubrik tetap perlu dipastikan sebelum menjadikan dataset ini bukti untuk klaim tersebut.

## Hubungan dengan notebook BGE-M3 unggahan

Urutan kerja dipertahankan: pemasangan pustaka, pembacaan Excel, konfigurasi/dataset, model, optimizer/metrik, training, evaluasi dan scoring. Label dibagi rentang skor saat menghitung MSE seperti notebook unggahan, lalu prediksi dikembalikan ke satuan poin asli.

Perubahan yang disengaja:

- Tiga skenario lengkap menggantikan satu model BGE. S2 BGE dibuat frozen sesuai rancangan skenario yang sudah dibahas, bukan fully fine-tuned seperti notebook unggahan.
- Maksimum default 5 epoch per fold, dapat diubah melalui `EPOCHS` sebelum eksperimen; notebook unggahan memakai 20 epoch. DataLoader mempunyai micro batch 1 dan effective batch 8 melalui akumulasi gradien.
- Split acak per baris diganti holdout kelompok dan 5-fold CV development. Set yang dipakai memilih epoch tidak lagi sekaligus disebut final test.
- Rentang kategori ditetapkan 0–100 mengikuti asumsi notebook unggahan. Tidak diturunkan dari min/max nilai test. Tidak ada deskriptor rubrik baru yang dikarang.
- Tabel evaluasi menggunakan tepat QWK, Gwet AC2 kuadratik, RMSE dan Pearson. Diagnostic out-of-range ditampilkan terpisah, bukan metrik pemilihan tambahan.

## Input, split, dan target eksperimen

Hanya teks `response_text` masuk encoder. `soal`, `jawaban_referensi`, email, nomor respons dan nilai tidak disisipkan ke teks atau vektor input. Model ini mempelajari kualitas yang berkaitan dengan skor pada kumpulan pertanyaan yang ada; model tidak secara eksplisit membandingkan jawaban dengan kunci. Menguji generalisasi ke prompt baru memerlukan split/protokol berbeda.

Ada tiga kelompok teks identik lintas email setelah normalisasi audit (casefold dan penyatuan whitespace). Agar tidak bocor, email yang mempunyai teks identik dihubungkan menjadi kelompok yang sama, menghasilkan 22 kelompok terhubung. Normalisasi audit ini tidak mengganti teks masukan encoder; cleaning masukan hanya NFC, kontrol karakter, dan normalisasi baris sesuai fungsi lama.

Targetnya 80% baris development dan 20% test dengan kelompok utuh. `GroupShuffleSplit(test_size=0.2, random_state=2026, n_splits=128)` menghasilkan kandidat pembagian kelompok; dipilih kandidat pertama yang paling dekat target proporsi baris, **berdasarkan jumlah baris saja**, tanpa melihat label guru/hasil model. Alasannya, 20% kelompok belum tentu berisi 20% baris. Jika rasio tepat tidak tersedia, kelompok tetap tidak dipecah dan proporsi aktual dilaporkan. `GroupKFold(n_splits=5)` membagi development menjadi lima fold; setiap baris development menjadi validasi sekali, tanpa kelompok yang menyeberang train/validation. Kelompok dan duplikat tidak menyeberang final test. Manifest yang sama digunakan S1/S2/S3.

Validasi sudah terjadi di lima fold tersebut. Tidak diperlukan lagi split validasi terpisah di luar CV untuk protokol ini. Epoch terbaik per fold dipilih berdasarkan RMSE validasi minimum. Epoch refit adalah pembulatan ke atas median lima epoch terbaik. Semua konfigurasi/epoch ketiga skenario dikunci, semua refit di seluruh development diselesaikan, baru final test dievaluasi. Perbandingan/pemilihan skenario menggunakan CV; angka test untuk pelaporan akhir, bukan tuning lanjutan.

Hasil CV fold tersimpan dan dapat dilanjutkan. Hash dataset, konfigurasi dan source membedakan folder eksperimen. Checkpoint final dan prediksi test diverifikasi dengan SHA256. Pengulangan run selesai memakai hasil tersimpan; hasil test per skenario juga tersimpan agar resume setelah gangguan tidak perlu memprediksi test ulang. Jangan sengaja mengganti konfigurasi setelah melihat test untuk mengejar hasil terbaik.

## Model dan bagian yang belajar

| Skenario | Alur | Parameter yang diperbarui |
| --- | --- | --- |
| S1 | RoBERTa Indonesia → CLS setiap window → mean 768 → dropout → linear 768→1 | Encoder RoBERTa dan head |
| S2 | BGE-M3 frozen → CLS 1024 → L2 → MLP 1024→256→1 | MLP saja |
| S3 | RoBERTa 768; BGE frozen 1024→proyeksi linear+LayerNorm 256; concat 1024→linear 1 | RoBERTa, proyeksi BGE, head |

RoBERTa: `flax-community/indonesian-roberta-base`, revision `cc5ecee7de8532880ad08d802292a6f42865bc83`. BGE: `BAAI/bge-m3`, revision `5617a9f61b028005a4858fdac845db406aefb181`. Sumber resmi: [RoBERTa Indonesia](https://huggingface.co/flax-community/indonesian-roberta-base), [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3), [BGE-M3 paper](https://arxiv.org/abs/2402.03216).

Metode BGE yang dipakai adalah dense CLS lalu normalisasi L2 melalui `AutoModel`; bukan sparse retrieval atau multi-vector ColBERT BGE-M3. Fitur BGE frozen dihitung tanpa gradien dalam mode `eval()` dan tidak menggunakan label. Karena checkpoint/normalisasi tersebut tidak di-fit pada dataset, fitur boleh diekstrak untuk semua baris terlebih dahulu, dengan seleksi/training tetap dibatasi indeks development.

RoBERTa memakai tokenizer slow karena artifact fast checkpoint tersebut mempunyai postprocessor special-token berbeda. Window 512 memuat 510 token isi + 2 special token; overlap 128 isi. Seluruh token isi tercakup, termasuk akhir jawaban. CLS window dirata-ratakan per jawaban. BGE memakai tokenizer sendiri, batas eksperimen 1024 termasuk special, dan jumlah jawaban terpotong dilaporkan. Batas ini pilihan eksperimen, bukan batas maksimum 8192 checkpoint.

## Training dan empat metrik

AdamW menggunakan learning rate encoder `2e-5`, head/proyeksi `1e-3`, weight decay 0.01, warmup 10%, gradient clipping 1. Micro batch 1 dikumpulkan ke effective batch 8; batch terakhir dibobot sesuai ukuran aktual. CPU memakai float32; GPU memilih autocast bf16 bila didukung atau fp16 dengan GradScaler. Setup mempertahankan PyTorch CUDA bawaan Colab. GPU/AMP perlu verifikasi pada runtime GPU pengguna; uji teknis lokal tidak menggantikannya.

Label MSE adalah `(rater−score_min)/(score_max−score_min)`. Sebelum evaluasi/prediksi, output dikembalikan dengan `prediksi × rentang + score_min`. Jadi RMSE dinyatakan dalam poin skor asli 0–100; nilai lebih dari 1 adalah sah dan tidak sama dengan MSE training berskala. Tidak ada metrik error ternormalisasi yang dilaporkan sebagai hasil.

| Metrik | Prediksi yang digunakan | Cara membaca |
| --- | --- | --- |
| RMSE | Mentah kontinu dalam poin asli | Lebih rendah berarti kesalahan kuadrat rata-rata lebih kecil; tidak dibatasi 0–1 |
| Pearson | Mentah kontinu | Arah/perubahan linear; tinggi tidak menjamin nilai tepat |
| QWK | Salinan prediksi di-clip ke 0–100 lalu integer terdekat | Kesepakatan dengan bobot jarak kuadratik dan koreksi peluang Cohen |
| Gwet AC2 | Salinan kategori yang sama | Agreement kuadratik dengan koreksi peluang Gwet |

Tepat setengah (`.5`) dibulatkan ke kategori yang lebih rendah. Daftar kategori selalu semua 101 integer 0…100, termasuk yang tidak teramati. Khusus AC2, daftar kategori dan bobot sangat memengaruhi expected agreement; skor AC2 bisa jauh lebih tinggi daripada QWK. Jangan menyimpulkan kualitas dari AC2 saja. Tidak ada pemetaan otomatis nilai total ke empat trait. Nilai tidak terdefinisi disimpan `null` disertai alasan, bukan dipaksa nol. Skor mentah dan persentase di luar rentang tetap tersedia, tidak disembunyikan oleh clipping.

## Satu jawaban dan vektor yang lengkap

Default satu jawaban held-out digunakan untuk scoring dan laboratorium vektor. `NEW_ANSWER` dapat diisi untuk prediksi teks baru tanpa nilai guru. RoBERTa dan BGE dijalankan dalam `eval()`/tanpa gradien pada laboratorium, terpisah dari training.

Empat jenis vektor diperlihatkan:

1. Token ID dan **input embedding**: nilai lookup berdasarkan ID sebelum penambahan posisi/konteks.
2. Vektor kata awal: rata-rata input embedding semua subword yang beririsan dengan span kata.
3. Vektor kata kontekstual: rata-rata hidden state terakhir subword kata ketika membaca jawaban. Untuk overlap window, vektor token dirata-ratakan antar-window dahulu.
4. Vektor kalimat: setiap kalimat dibaca sendiri, lalu mean CLS window RoBERTa atau CLS+L2 BGE. Bukan rata-rata vektor kata dari jawaban utuh. Segmentasi tanda baca sederhana bisa keliru pada singkatan.

Offset karakter berasal dari tokenizer fast dengan `add_special_tokens=False`. ID-nya wajib sama persis dengan tokenizer yang dipakai training. Special token dibangun manual menggunakan tokenizer authoritative agar bug postprocessor fast RoBERTa tidak menghilangkan `<s>`/`</s>`. Jika ID tidak cocok, proses gagal dengan alasan, tidak menebak pemetaan.

Kata ejaan sama dapat mempunyai ID subword berbeda karena spasi/case/pemecahan. **ID yang sama pada checkpoint yang sama menghasilkan lookup yang sama**, tetapi hidden state kontekstual dapat berubah karena konteks dan posisi. Tabel kata berulang menunjukkan kesamaan ID, cosine awal/kontekstual, selisih maksimum koordinat dan `allclose(atol=1e-6)`. Toleransi ini hanya pemeriksaan numerik vektor, bukan adjacent agreement skor.

Bandingkan pengulangan **di dalam encoder yang sama**. RoBERTa mempunyai 768 koordinat dan BGE 1024; cosine langsung antarmodel tidak sah karena dimensi/ruang berbeda. `VECTOR_SOURCE='pretrained'` mempelajari sifat encoder awal; pilihan `S1`/`S3` menggunakan RoBERTa fine-tuned final, dengan sumber/checksum ditandai. BGE selalu frozen pretrained.

NPZ menyimpan seluruh koordinat token, kata dan kalimat. CSV metadata menyimpan span, ID dan coverage. Preview notebook hanya lima koordinat agar terbaca; arsip unduhan berisi lengkap. Token/kata BGE yang melewati truncation ditandai tidak tercakup; vektornya `NaN`, tidak diganti nol. Semua file hasil tersebut privat.

## Menjalankan lokal dan batas bukti

```bash
python scripts/setup_dataset300.py
# Simpan config privat berisi dataset/output; gunakan contoh konfigurasi di notebook.
outputs/dataset300_env/bin/python scripts/dataset300_step.py audit --config outputs/dataset300/notebook_config.json
outputs/dataset300_env/bin/python scripts/dataset300_step.py encode --config outputs/dataset300/notebook_config.json
outputs/dataset300_env/bin/python scripts/dataset300_step.py train --config outputs/dataset300/notebook_config.json
outputs/dataset300_env/bin/python scripts/dataset300_step.py predict --config outputs/dataset300/notebook_config.json
outputs/dataset300_env/bin/python scripts/dataset300_step.py vectors --config outputs/dataset300/notebook_config.json
```

Konfigurasi efektif tercatat pada folder run. Setup menggunakan venv baru dengan akses site-packages runtime untuk mempertahankan torch GPU, lalu memasang versi NumPy/Pandas/Transformers worker yang dipatok. Python3.12 memakai Transformers4.44.2, sedangkan Python3.13 memakai4.48.3 karena tokenizers versi lama tidak menyediakan wheel3.13. Kedua environment dipisahkan: `outputs/dataset300_env` dan `outputs/dataset300_env_py313`. Notebook membaca path worker hasil setup dari `outputs/dataset300_setup.json`; perintah CLI contoh di atas memakai path3.12, ganti dengan path3.13 bila itu runtime aktif. Tidak memakai environment CPU notebook lama. Python kernel tetap hanya menjalankan stdlib/IPython dan subprocess.

Jika notebook lama berhenti pada `CalledProcessError` dan traceback menunjukkan `/usr/lib/python3.13`, buka ulang notebook terbaru dari link Colab di atas. Jalankan kembali dari cell pertama untuk mengambil source terbaru, lalu setup. Versi baru mendukung3.13 dan menyertakan rincian log jika instalasi gagal, sehingga penyebab tidak tertutup oleh traceback subprocess umum. Perubahan source di GitHub tidak otomatis mengganti cell pada tab lama yang sudah terbuka.

Pengujian publik `tests/test_teacher_score.py` menggunakan workbook sintetis dan transformer kecil **berbobot acak** dengan bentuk output yang sesuai. Test memeriksa mapping, split, gradien/freeze, perhitungan metrik, save/reload, CV/refit/resume dan offset vektor; angka test bukan performa penelitian. Eksekusi notebook audit-only membaca Excel unggahan asli tanpa mengunduh checkpoint. Lihat [bukti verifikasi](verification_dataset300.md) untuk yang benar-benar dijalankan; hasil penelitian membutuhkan training checkpoint asli di runtime dengan akses Hugging Face.
