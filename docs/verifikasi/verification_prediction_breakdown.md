# Verifikasi rincian prediksi E46/E47 — 5 November 2026

Perubahan ini khusus pelaporan test: rincian mentah/normalisasi/pembulatan, kategori beda tepat 0/1/2/3/4/5/10 dan selisih lain, serta toleransi ≤n, masing-masing jumlah dan persentase. Tidak ada hasil penelitian baru atau perubahan model/training.

## Identitas source

HEAD awal `2181cfc29cb753b5505b28a7434ddbbe25b1bf75`; cwd `/workspace/Automated_Essay_Scoring`. Source E43/E44/E45 dan HTML lama tetap utuh. Notebook eksperimen masing-masing189cell/92cellkode, notebook pembaru11cell/5cellkode, semua output kosong. Mirror identik.

| Notebook | Raw SHA256 | Visible-code SHA256 |
|---|---|---|
| E46 | `c9a4ac3ecf3f4805debf572f86b3d010cc45f9f4de38ffe0dc3f991f5d55ec5b` | `767eb6e481b11c278d04e408671645ba6b6dfa00118313662822bfa3d5e3ccf2` |
| E47 | `3ad21a66224d519a6dcd3a3f302d22e5d321a298c11390214edb8d94bfc736e9` | `025696bf0f52d86a703a2640d409ad78e9bb28e47867cebecf072b42876ad862` |
| Pembaru ZIP | `591b9f3af7c922f59088dadf035f84be770e872e723e53e21de4c80b14681e9e` | `b581340e5fa388fefe19276478a013492d871b9192e7132dbc90ada217a16a77` |

Identitas training awal E46/E47 dipertahankan secara eksplisit; hash source laporan terbaru dicatat terpisah. Tes membandingkan setiap definisi lama dengan commit awal: hanya `run_folder`, `evaluation_report`, `validate_refit_download` yang berubah. Model/optimizer/fit/Engine/metric/split/train_all tidak berubah. Config identik dan pemanggilan `run_folder` dengan config/bundle yang sama menghasilkan lokasi sama sebelum/sesudah pembaruan. Identitas training harus berubah jika metode training diubah kelak.

## Pengujian otomatis

```bash
/tmp/e39-verify/bin/python -m pytest tests/test_prediction_report.py \
 tests/test_e46_e47_notebooks.py::test_structure_protocol_and_deterministic_builder \
 -q --basetemp=outputs/prediction_report_update/final_fixtures
```

Exit0; **15passed,0failed,0skipped,0xfailed**,6,36detik. Satu warning deprecation plugin lingkungan. Log privat `outputs/prediction_report_update/final_pytest.log`.

- Fixture teknis menguji seluruh kategori tepat, kategori lain dan kumulatif; jumlah tepat menjumlah ke N, persentase100%. Contoh raw50,5 dibulatkan50, raw50,51→51. Selisih tepat5 dibedakan dari ≤5.
- Normalisasi 1poin→0,01; prediksi mentah dan input tidak dimutasi. Rentang affine khusus diuji. Prediksi di luar0–100 tetap mentah, clip hanya salinan kategori. Normalisasi tidak mengubah jumlah kategori setara.
- ID duplikat, prediksiNaN, labelnoninteger, pembulatan/error sumber tidak cocok ditolak. Verifier menolak count atau normalisasi yang diubah.
- Pembaru ZIP diuji pada lima workbook sintetis lengkap: sumber SHA tetap, empat Excel lain byte-identik, metric/CV dan sheet lain Excel04 tetap, detail16kolom, ringkasan8/7baris. Pembaruan kedua tidak menggandakan catatan panduan dan mempertahankan tabel terserialisasi identik. Tabel metrik kosong dan overwrite sumber ditolak.
- Schema, hash, source native, mirror dan rebuild deterministik kedua notebook eksperimen lolos. Tidak ada magic/worker/exec/eval/sweepseed/cosine.
- Notebook pembaru dijalankan dari atas sampai akhir dalam kernel Jupyter baru dengan ZIP sintetis, menghasilkan ZIP baru dan tiga tabel yang diverifikasi ulang setelah serialisasi. Tidak memuat model, dataset atau menjalankan training.

## Integrasi ekspor E46/E47

Perintah privat `python outputs/prediction_report_update/reexport_check.py` memakai hasil fixture CPU yang sudah tersedia dari pengujian versi awal E46/E47. Masing-masing24baris sintetis,6test,CVlama5fold,refitfixture1epoch. Tidak melatih ulang fixture. Source publik tetap300baris/refit18atau7/horizon25.

`build_reports` terbaru menghasilkan lima Excel dan ZIP untuk kedua resep, dengan gate wajib statusPASSED. Pemanggilan model/fit/encoder dilarang selama ekspor. SHA checkpoint dan CSVprediksi tetap sama; lima kolom prediksi lama, metric test dan ringkasan/fold CV tetap identik dengan hasil tersimpan. Cell tampilan training dieksekusi dengan laporan tersimpan: detail6baris, kategori8baris, toleransi7baris tampil. Mengubah satu count pada Excel terserialisasi menyebabkan gate menolak unduhan; mengembalikannya membuat gate lolos. Bukti privat `reexport.log` dan `reexport_proof.json`.

Tidak menjalankan ulang training pretrained/GPU atau seluruh training fixture versi awal, karena metode training tidak berubah. [Bukti training versi awal](verification_e46_e47_refit_native.md) terpisah dari bukti pelaporan ini. Tidak menjamin semua kemungkinan bug atau kompatibilitas seluruh runtime.

Python3.12.14,openpyxl3.1.5,pytest9.2.0,nbformat5.11.1,nbclient0.11.0; pemuat source pengujian memakai torch2.6.0+cpu,transformers4.44.2,NumPy2.3.5,pandas2.2.3,SciPy1.17.0,scikit-learn1.8.0. Jupyter memerlukan socket lokal.

Hanya source notebook kosong, helper/builder, tes dan dokumentasi aman dipublikasikan. Dataset,ZIPhasil,executednotebook,bobot,cache,log dan bukti individual tetap privat. Verifikasi SHA remote/byte file sesudah push dicatat privat pada `outputs/prediction_report_update/publication.json`.
