# Verifikasi buku E36 — 3 November 2026

Pekerjaan ini mengaudit hasil unggahan dan memperbarui buku tiga input yang sama. Tidak ada training penelitian baru atau klaim eksekusi GPU di workspace.

## Sumber dan audit

- Root: `/workspace/Automated_Essay_Scoring`; baseline source `349f2e82e876333bf2270d47e7eb86b59e939a55`.
- ZIP E36: `Hasil_AES_300_E36_E30_S3_EncoderLR3e6_Native_Learnable_GPU.zip`; SHA256 `6eca1930f290ea2cfcde87bcb0cf1acc6f55740ea070f78e013ce816dd7303e4`.
- `build_e21_study.audit(path, "E36")` menghitung ulang metrik60test, memeriksa parameter S3-only, riwayat5fold/25epoch/patience5, epoch refit13, bobot/mask/posisi token dan vektor hasil. Deviasi metrik maksimum1,70e-8. `runtime_source_matches_commit=false` karena notebook eksekusi/checkpoint/log perangkat tidak dilampirkan.
- Builder keluarga mencocokkan300teks/label,240development/60test dan5fold dengan keluarga lama. Semua15snapshot/audit prediksi serta105duel sebelumnya tetap sama.
- SHA16ZIP dan80Excel yang diembed cocok sumber;75Excel sebelumnya tidak berubah. Bukti lokal `outputs/e36_study/all_original_excel_verified.json` dan arsip payload verifikasi disimpan privat.
- Buku memuat16eksperimen,46model aktual,80Excel/748sheet dan2760prediksi. E36 tidak memiliki hasil S1/S2; keduanya tidak dibuat nol atau disalin dari E30.
- E29/E30document byte-identik; E36 menggantikan E26 pada slot kode ketiga. Notebook/source/hasil E26 tidak dihapus. Source E36 memuat89cell kode native,2667baris dan103anotasi.

## Tes source dan preservasi

Cwd root repo. Perintah berikut selesai exit0, **31passed,0failed,0skipped,0xfailed**; satu peringatan deprecation plugin lingkungan:

```bash
PYTHONPATH=src:scripts /tmp/e36-verify/bin/python -m pytest \
 tests/test_e36_study.py tests/test_e35_study.py tests/test_e34_study.py \
 tests/test_e33_study.py tests/test_e31_e32_study.py \
 tests/test_three_input_study.py tests/test_three_input_identities.py -q --tb=short
```

Lingkungan: Python3.12.14,pytest9.2,NumPy2.3.5,pandas2.2.3,scikit-learn1.8,openpyxl3.1.5,torch2.6.0+cpu,transformers4.44.2. Dependency torch tersedia bagi modul metrik; tidak ada forward/training baru pada audit ini.

## Browser dan paket

Paket lokal `outputs/e21_study/Belajar_AES_Tiga_Input_Lengkap.zip` berukuran467,82MiB, SHA256 `1420db8063e886ed3e32d61d12de4619439de2aadda7a145849186566fcfc414`. Isinya satu `Belajar_AES_Tiga_Input_Lengkap.html` terenkripsi (617,60MiB). Verifikasi dekripsi/dekompresi/hash plaintext, ZIP roundtrip dan gabungan6bagian pengunduh selesai exit0. Kunci lama tetap digunakan dan tidak ditanam pada file publik. Perintah pengemasan:

```bash
/tmp/e36-verify/bin/python scripts/package_private_study.py \
 --input outputs/e21_study/belajar_Tiga_Input_Privat.html \
 --output outputs/e21_study/Belajar_AES_Tiga_Input_Lengkap_terenkripsi.html \
 --key-file outputs/e21_study/family_download_key.json \
 --archive outputs/e21_study/Belajar_AES_Tiga_Input_Lengkap.zip \
 --parts-dir outputs/e36_study/publish_candidate/downloads \
 --artifact-name Belajar_AES_Tiga_Input_Lengkap \
 --download-template scripts/templates/three_input_download.html
```

Browser Chromium154.0.8037.57 dengan Playwright lokal, viewport1440×1000 dan390×844. Perintah berikut selesai exit0, dua kelompok pemeriksaan lolos tanpa pageerror:

```bash
NODE_PATH=/tmp/e36-browser/node_modules node scripts/check_e36_study.mjs
```

Diverifikasi:16identitas, metode/LR E36, pilihan S1/S2/S3 pada tabel/kurva/peta, tampilan mobile tanpa overflow body, toggle tampil/sembunyi kunci, penolakan kunci salah, dekripsi memakai kunci lama, akses offline960prediksi S3/2760seluruh model/900S1, unduhan workbook E36 sesuai SHA asli, sheet Kata Berulang8kolom tanpa cosine,89cell kode native, seluruh16bab dan diagram hanya S3. Screenshot mobile/desktop ditinjau. Bukti privat `outputs/e36_study/browser.json` dan `browser.log`; log tidak mencetak kunci atau teks siswa.

## Batas publikasi

Source, penjelasan dan agregat aman dapat dipublikasikan. Pemindaian300prefiks teks jawaban tidak menemukan kecocokan dalam HTML publik; `familyPrivate` publik kosong. ZIP asli, data individual, kunci, plaintext, paket terenkripsi dan bagian unduhan baru tetap lokal di outputs; tidak masuk staged source. Pengunduh GitHub yang lama belum diperbarui. Review persetujuan otomatis sebelumnya menolak publikasi paket data siswa; publikasi payload baru memerlukan otorisasi eksplisit baru.
