# Dataset esai dalam Excel

Data tersedia sebagai berkas Excel `.xlsx` yang sebenarnya. Klik tautan unduh
di bawah, simpan berkasnya, lalu buka memakai Microsoft Excel atau LibreOffice.

| File | Unduh | Isi |
| --- | --- | --- |
| ASAP_Train_dan_Prompt7.xlsx | [Download Excel ASAP](https://raw.githubusercontent.com/MalikHariyanto/Automated_Essay_Scoring/datasets/excel/ASAP_Train_dan_Prompt7.xlsx) | Seluruh 12.976 baris data asli, tab khusus 1.569 esai Prompt 7 dengan empat skor, dan skor kedua penilai. |
| ELLIPSE_Train.xlsx | [Download Excel ELLIPSE](https://raw.githubusercontent.com/MalikHariyanto/Automated_Essay_Scoring/datasets/excel/ELLIPSE_Train.xlsx) | Seluruh 3.911 esai train; tab enam skor analitik dan tab seluruh kolom sumber. |
| dataset_excel.zip | [Download kedua Excel sekaligus](https://raw.githubusercontent.com/MalikHariyanto/Automated_Essay_Scoring/datasets/excel/dataset_excel.zip) | Kedua workbook dan petunjuk ini dalam satu ZIP. |

## Cara membaca workbook

**ASAP** dibuka pada tab `Prompt7_4dimensi`. Empat kolom skor bernama
`content_0_6`, `organization_0_6`, `style_0_6`, dan `conventions_0_6`.
Masing-masing adalah jumlah skor dua penilai manusia 0–3, mengikuti kode
pengolahan sumber. Tab `Penilai_Prompt7` memuat skor tiap penilai. Tab
`ASAP_semua_asli` memuat semua kolom dan baris sumber TSV; tidak semua prompt
lain memiliki empat dimensi tersebut.

**ELLIPSE** dibuka pada tab `Skor_6dimensi`, berisi ID, teks, dan enam skor
manusia: Cohesion, Syntax, Vocabulary, Phraseology, Grammar, Conventions.
Tab `Train_asli` menyimpan semua 26 kolom CSV. Skor 1–5 dan nilai setengah poin
dipertahankan. File test dan arsip penilai mentah tidak digabungkan.

Setiap tab data memiliki filter dan header tetap. Teks esai lengkap berada
di sel; tinggi baris membatasi tampilan awal, bukan panjang teks yang disimpan.
Tidak ada pemotongan teks, pembuatan label kosong, atau normalisasi skala.

Kedua dataset berbahasa Inggris. Dimensi dan skala sumber tidak otomatis sama
dengan empat target bilangan bulat 1–5 pada penelitian AES pengguna. File ini
merupakan ekspor dataset eksternal; input, rubrik, dan model penelitian aktif
pengguna tidak diubah.

## Sumber dan atribusi

- ASAP: [robert1ridley/cross-prompt-trait-scoring](https://github.com/robert1ridley/cross-prompt-trait-scoring),
  kode paper *Automated Cross-prompt Scoring of Essay Traits* (AAAI 2021), commit
  `981ec606f039c5b3ee163b5b474e4329d9ebbd17`.
  Data: `data/training_set_rel3.tsv`; pengolahan empat dimensi Prompt 7:
  `arrange_data.py`, fungsi `combine_for_prompt_seven_eight`.
  [Dataset dan ketentuan asli di Kaggle](https://www.kaggle.com/c/asap-aes/data).
  Atribusi dan ketentuan sumber/Kaggle tetap berlaku; tidak ada lisensi baru
  yang diberikan untuk data ASAP.
- ELLIPSE: [scrosseye/ELLIPSE-Corpus](https://github.com/scrosseye/ELLIPSE-Corpus),
  commit `dc3b8f0b3b4332fc9f64302c4ccfc4ed582f4b43`, berkas
  `ELLIPSE_Final_github_train.csv`.
  Crossley et al. (2023), *Measuring second language proficiency using the
  English Language Learner Insight, Proficiency and Skills Evaluation
  (ELLIPSE) Corpus*. International Journal of Learner Corpus Research,
  9(2), 248–269.
  [Lisensi CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
  berlaku untuk sumber ELLIPSE dan adaptasi Excel-nya; perubahan berupa
  pengemasan CSV ke workbook, pemilihan kolom pada tab ringkas, dan format tabel.

Kedua repositori sumber lengkap tersedia pula di GitHub MalikHariyanto:
[ASAP](https://github.com/MalikHariyanto/Automated_Essay_Scoring/tree/sources/asap-trait-scoring)
dan [ELLIPSE](https://github.com/MalikHariyanto/Automated_Essay_Scoring/tree/sources/ellipse-corpus).

## Reproduksi dan pemeriksaan

`export_datasets_excel.py` memerlukan `xlsxwriter` dan `openpyxl`.
Jalankan dengan file sumber yang sama:

```sh
python export_datasets_excel.py --asap training_set_rel3.tsv --ellipse ELLIPSE_Final_github_train.csv --output hasil_excel
```

SHA-256 sumber dikunci dalam skrip. `verification.json` mencatat pemeriksaan
seluruh nilai dengan membuka kembali workbook memakai pembaca Excel terpisah,
termasuk teks/baris baru, skor, ID, dan jumlah baris. Bukti ini tentang
ketepatan ekspor file, bukan hasil training atau akurasi model AES.
