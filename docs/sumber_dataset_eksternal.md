# Arsip repositori dataset eksternal

Kedua repositori sumber sudah disalin ke akun **MalikHariyanto** sebagai branch
khusus dalam `Automated_Essay_Scoring`. Buka tautan berikut untuk melihat seluruh
isi repositorinya di GitHub pengguna:

| Sumber | Salinan di GitHub pengguna | Kegunaan |
| --- | --- | --- |
| Cross-prompt Trait Scoring | [sources/asap-trait-scoring](https://github.com/MalikHariyanto/Automated_Essay_Scoring/tree/sources/asap-trait-scoring) | Kode penelitian AAAI 2021 untuk pengolahan dan prediksi dimensi esai ASAP. |
| ELLIPSE Corpus | [sources/ellipse-corpus](https://github.com/MalikHariyanto/Automated_Essay_Scoring/tree/sources/ellipse-corpus) | Dataset pembuat asli, skor dimensi, rubrik, dan keterangan publikasi. |

Setiap branch menyimpan isi dan riwayat commit asli. Ini arsip branch, sehingga
tidak menambah repositori terpisah berstatus fork pada profil GitHub. Arsip tidak
otomatis mengikuti perubahan pembuat sumber. Kode penelitian AES tetap berada
di branch `codex/aes-learning`.

## Pengolahan dimensi ASAP

- Sumber asli: [robert1ridley/cross-prompt-trait-scoring](https://github.com/robert1ridley/cross-prompt-trait-scoring).
- Commit yang disalin: `981ec606f039c5b3ee163b5b474e4329d9ebbd17` (7 commit dalam riwayatnya).
- [Kode pengolahan dimensi `arrange_data.py`](https://github.com/MalikHariyanto/Automated_Essay_Scoring/blob/sources/asap-trait-scoring/arrange_data.py).
- [Data asli `training_set_rel3.tsv`](https://github.com/MalikHariyanto/Automated_Essay_Scoring/blob/sources/asap-trait-scoring/data/training_set_rel3.tsv).
- Sumber dataset: [kompetisi ASAP AES di Kaggle](https://www.kaggle.com/c/asap-aes/data).

Pada data yang diperiksa, `essay_set = 7` memiliki 1.569 esai dengan empat
dimensi: content, organization, style, dan conventions. Setiap dimensi memiliki
skor dari dua penilai manusia, masing-masing 0–3. Fungsi
`combine_for_prompt_seven_eight` menjumlahkan kedua skor menjadi 0–6 per dimensi
untuk prompt 7. Ini bukan skor 1–5 dalam rubrik penelitian AES pengguna.

Atribusi penulis dan README sumber dipertahankan. Repo sumber tidak menyertakan
berkas lisensi terpisah; hak penggunaan tetap mengikuti ketentuan sumber dan
dataset Kaggle, tanpa pemberian lisensi baru dari arsip ini.

## Dataset dan rubrik ELLIPSE

- Sumber asli: [scrosseye/ELLIPSE-Corpus](https://github.com/scrosseye/ELLIPSE-Corpus).
- Commit yang disalin: `dc3b8f0b3b4332fc9f64302c4ccfc4ed582f4b43` (16 commit dalam riwayatnya).
- [Data train `ELLIPSE_Final_github_train.csv`](https://github.com/MalikHariyanto/Automated_Essay_Scoring/blob/sources/ellipse-corpus/ELLIPSE_Final_github_train.csv).
- [Rubrik asli `ELL_Rubrics.docx`](https://github.com/MalikHariyanto/Automated_Essay_Scoring/blob/sources/ellipse-corpus/ELL_Rubrics.docx).
- [README pembuat dan sitasi penelitian](https://github.com/MalikHariyanto/Automated_Essay_Scoring/blob/sources/ellipse-corpus/README.md).

File train yang diperiksa berisi 3.911 esai dengan enam dimensi lengkap:
Cohesion, Syntax, Vocabulary, Phraseology, Grammar, dan Conventions. Skor yang
diamati 1–5 termasuk nilai setengah poin. Corpus ini tidak memiliki skor
content terpisah yang dapat langsung menggantikan dimensi isi.

Lisensi yang dinyatakan pembuat: **CC BY-NC-SA 4.0**. Atribusi dan ketentuan
lisensi sumber tetap berlaku. File train, test, rubrik, dan arsip penilai mentah
di branch ini berasal dari repositori pembuat dan tidak dimodifikasi.

## Bukti salinan

[Laporan verifikasi](verification/dataset_source_archives.json) mencatat SHA
branch yang sama dengan commit sumber dan pemeriksaan byte enam berkas yang
diambil kembali dari GitHub pengguna. Tidak ada training model atau konversi
skor penelitian dalam proses pengarsipan ini.
