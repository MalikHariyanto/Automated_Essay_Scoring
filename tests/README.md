# Pengujian: bedakan tingkat buktinya

Tes memastikan implementasi sesuai aturan, bukan membuktikan akurasi AES pada data penelitian. Hasil training pengguna dibaca dari ZIP/Excel dan [review hasil](../docs/review/review_e25_e26_study.md).

| Kelompok | Contoh |
| --- | --- |
| Buku tiga input: metrik, sumber, catatan, privasi | [test_three_input_study.py](test_three_input_study.py), [test_e21_study.py](test_e21_study.py) |
| Notebook tiga input/arsitektur | [test_e21_e22_notebooks.py](test_e21_e22_notebooks.py), [test_e23_e24_notebooks.py](test_e23_e24_notebooks.py), [test_e25_e26_notebooks.py](test_e25_e26_notebooks.py) |
| Buku dan eksperimen sebelumnya | `test_mlp_*`, `test_lora_*`, `test_context_pair_*`, `test_three_seed_*` |
| Workflow/modul Python terdahulu | `test_pipeline`, `test_metric_trace`, `test_trainer`, `test_experiment_protocol` |

Contoh pemeriksaan integritas buku aktif, dengan dependensi yang sudah tersedia:

```bash
python -m pytest -q tests/test_three_input_study.py tests/test_e21_study.py
```

Sebagian tes notebook menjalankan seluruh cell menggunakan model kecil/fixture CPU, bukan pretrained training GPU pada 300 jawaban. Browser memerlukan Playwright, Chromium, server lokal, serta berkas privat lokal untuk pemeriksaan paket penuh. Itu tidak otomatis tersedia setelah clone akun baru.

Pemeriksa historis dapat mengharapkan artefak versi lama yang kini ada di riwayat. Jangan menyebut semua tes repo lulus jika hanya subset dijalankan. Catat command, versi, hasil, dan batasnya. [Bukti terdokumentasi](../docs/verifikasi/README.md).
