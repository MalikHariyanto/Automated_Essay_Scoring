# Verifikasi pembaruan hasil E31/E32

Sumber kerja awal: `d72a6e6f4f5271aa7939f4ffc93c28bc94a6b0be`. Python3.12, Torch2.6.0+cpu, Transformers4.44.2, NumPy2.2.6, openpyxl3.1.5. Pemeriksaan ini mengaudit unggahan dan HTML, tidak melatih model penelitian.

- `python` dari `/workspace/.venvs/aes` menjalankan audit kedua ZIP: metrik, split/fold, coverage, epoch, gate/pengali, bobot pooling/co-attention dan posisi token lulus.
- Dari root repo: `python -m pytest -q tests/test_e31_e32_study.py tests/test_three_input_study.py tests/test_e21_study.py`:18passed,2skipped,exit0. Dua skip pada pengujian paket historis bersifat terpisah; pengemasan aktual dijalankan memakai Python sistem dengan cryptography dan verifikasi roundtrip di bawah.
- Verifikasi preservasi lokal:10snapshot,10audit prediksi,45duel lama dan ketiga buku kode identik. Hash12ZIP serta60workbook sesuai sumber; total557sheet. Bukti tersimpan lokal di `outputs/e31_e32_study/preservation.json`.
- Chromium melalui Playwright: identitas12eksperimen, tabel metode E32 tanpa BiGRU, viewport1440/390 tanpa overflow halaman,2160prediksi pada60jawaban dan sheet gate E31/E32 melalui tombol Excel lulus, tanpa pageerror. Exit0; log lokal `outputs/e31_e32_study/browser.log`.
- `scripts/package_private_study.py` dengan kunci lama: enkripsi/dekripsi+gzip roundtrip, ZIP roundtrip dan gabungan5bagian terenkripsi lulus,exit0. Kunci dan HTML/ZIP lengkap berada di `outputs/` yang diabaikan Git; tidak disalin ke dokumen publik.

HTML dan paket diperbarui lokal. Tidak ada commit/push atau verifikasi publikasi GitHub pada tugas ini. Pengunduh dalam repo baru mencerminkan bagian paket baru setelah perubahan dipublikasikan; gunakan artefak ZIP/HTML lokal untuk hasil tugas ini.
