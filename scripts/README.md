# Source pembuat notebook dan buku

Notebook Colab aktif bersifat mandiri; pengguna tidak perlu menjalankan seluruh folder ini. [Katalog notebook](../notebooks/experiments/README.md) · [panduan AI](../PANDUAN_AI_PROYEK.md).

| Pekerjaan | Source utama |
| --- | --- |
| Notebook GPU E21/E22 | [build_e21_e22_gpu_notebooks.py](build_e21_e22_gpu_notebooks.py), [build_e21_e22_notebooks.py](build_e21_e22_notebooks.py) |
| Notebook E23/E24 | [build_e23_e24_notebooks.py](build_e23_e24_notebooks.py), [helper pooling](s3_sequence_pooling_helpers.py) |
| Notebook E25/E26 | [build_e25_e26_notebooks.py](build_e25_e26_notebooks.py), [helper gate/GRU-attention](s3_gated_bigru_attention_helpers.py) |
| Notebook E27/E28 | [build_e27_e28_notebooks.py](build_e27_e28_notebooks.py), [helper co-attention/optimizer](s3_coattention_helpers.py) |
| Audit Excel/source setiap eksperimen | [build_e21_study.py](build_e21_study.py) |
| Gabungkan buku tiga input | [build_three_input_study.py](build_three_input_study.py) |
| Kode/catatan belajar | [e21_study_content.py](e21_study_content.py), [three_input_learning_notes.py](three_input_learning_notes.py) |
| Template UI | [templates](templates/) |
| Buku sebelum tiga input | [build_mlp_study.py](build_mlp_study.py) dan pembuat snapshot `build_mlp_*` |
| Enkripsi/paket buku | [package_private_study.py](package_private_study.py) |
| Pemeriksaan E27/E28 dan preservasi isi | [check_e27_e28_study.mjs](check_e27_e28_study.mjs), [test_e27_e28_study.py](../tests/test_e27_e28_study.py) |
| Pemeriksaan browser E25/E26 historis | [check_e25_e26_study.mjs](check_e25_e26_study.mjs), [check_e25_e26_package.mjs](check_e25_e26_package.mjs) |
| Siapkan hosting statis | [prepare_paper_site.mjs](prepare_paper_site.mjs), [konfigurasi hosting](../hosting/README.md) |

Snapshot kini berada di [docs/snapshots](../docs/snapshots/README.md); bukti tersimpan pada [docs/verifikasi](../docs/verifikasi/README.md). Sebagian script pemeriksaan lama mengharapkan tahap buku sebelumnya (misalnya hanya empat eksperimen). Gunakan pemeriksa sesuai versi, bukan mengklaim seluruh checker historis cocok dengan HEAD terbaru.

## Rebuild buku hasil

Pakai lingkungan Python yang sesuai dependensi [dataset300](../requirements.dataset300.txt). Builder tidak menjalankan training, tetapi memerlukan ZIP hasil asli untuk audit dan buku privat. ZIP/dataset/kunci tetap di luar Git. Contoh susunan argumen:

```bash
python scripts/build_three_input_study.py \
  --e21 /lokasi_privat/E21.zip --e22 /lokasi_privat/E22.zip \
  --e23 /lokasi_privat/E23.zip --e24 /lokasi_privat/E24.zip \
  --e25 /lokasi_privat/E25.zip --e26 /lokasi_privat/E26.zip
```

`--reuse-audit` memakai snapshot tersimpan setelah hash ZIP diperiksa. Source hasil dipin pada commit masing-masing; jangan mengganti kode hasil lama dengan notebook paling baru. Builder menghasilkan HTML publik agregat dan HTML lengkap privat di `outputs/`; langkah paket/enkripsi terpisah mempertahankan kunci lama.

Menjalankan builder notebook dapat mengganti file notebook yang dituju. Sebelum melakukannya, periksa edit pengguna; varian lama sudah didokumentasikan di [arsip notebook](../notebooks/archive/README.md). Jangan membangun ulang semua notebook hanya untuk merapikan folder.
