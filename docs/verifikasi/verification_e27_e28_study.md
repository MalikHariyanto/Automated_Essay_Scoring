# Verifikasi buku delapan eksperimen · 29 Oktober2026

[Review dan perubahan UI](../review/review_e27_e28_study.md). **Tidak ada training ulang.** Working directory seluruh command: `/workspace/Automated_Essay_Scoring`.

```bash
OMP_NUM_THREADS=2 outputs/dataset300_env_py313/bin/python -m pytest -q \
  tests/test_three_input_study.py tests/test_e21_study.py \
  tests/test_e27_e28_study.py \
  --junitxml=outputs/e27_e28_study/unit.xml
node scripts/check_e27_e28_study.mjs
node scripts/check_e27_e28_package.mjs
```

Hasil akhir: **21passed,0failed/errors/skipped** pada tes integritas/unit; **10kelompok browser lulus**; **2pemeriksaan paket lulus**. Chromium memerlukan socket lokal yang ditolak sandbox terbaru; pemeriksaan browser dijalankan melalui izin eksekusi yang sesuai, memakai server localhost. Sesudah halaman dibuka, jaringan diblokir untuk memeriksa operasi offline. `file://` tidak diuji secara langsung.

## Bukti preservasi isi

- Seluruh dokumen HTML anak E21–E26 **identik byte** dengan commit sebelum pembaruan `daa745d`: source kode, sticky note, curriculum, penjelasan dan snapshot hasil dipertahankan.
- Snapshot eksperimen, metrik, riwayat/fold, error/toleransi, hash workbook,15duel lama dan prediksi lama dipertahankan. Ranking delapan peserta dihitung ulang dari prediksi yang sama; bukan mengganti hasil lama.
- Dekode payload bertahap pada HTML privat mengembalikan **8ZIP yang byte-identik** dengan unggahan. Seluruh **40Excel asli** tersedia. Sepuluh Excel E27/E28 yang diunduh melalui UI dibandingkan hash-nya dengan ZIP sumber: identik.
- Semua58notebook terdahulu dan lima berkas buku/pengunduh/paket sebelum tiga input dibandingkan dengan sumber awal: tidak berubah. Kunci paket tiga input tetap sama.
- Preview publik tidak menyertakan payload individu. Arsip siswa dan kunci tetap lokal; yang dipublikasikan adalah source/agregat, screenshot source/agregat, dan bagian paket AES-GCM terenkripsi.

## Bukti fungsi

Browser mencocokkan1440prediksi dan60checksum teks/label terhadap Excel, menghitung ulang empat cakupan ranking, memeriksa28duel/84skenario,15kurva fold/skenario dengan8trace,8peta skor, sorotan angka, seluruh sheet baru dan kode30cell per notebook.

Scroll vertikal panel dan horizontal tabel pulih setelah pindah tab. Wheel di atas tabel tetap menggulir isi fullscreen. Pergantian kode cepat berakhir pada pilihan terakhir. Mode fullscreen kode menutupi viewport, tanpa heading induk, pada1440/768/390piksel; mode semua blok/per blok, salin kode, sticky note dan scroll tunggal berfungsi. Tidak ada error JavaScript.

Pengunduh menghasilkan ZIP yang hash-nya sama dengan arsip lokal. Kunci salah ditolak, percobaan ulang memakai kunci lama berhasil; buku delapan eksperimen dan kode dapat dipakai setelah jaringan diblokir. Enkripsi/dekripsi dan ZIP roundtrip diperiksa pula oleh pembuat paket.

HTML besar kini memakai node transport1MiB untuk payload dan ciphertext, dengan pembacaan bertahap. Ini tidak mengubah satu byte ZIP/Excel di dalamnya. Checksum dan manifest bagian unduhan tetap diperiksa.

Log, JUnitXML, hasil browser, checksum sumber dan bukti privasi tersimpan lokal pada `outputs/e27_e28_study/`. Hasil ini membuktikan integritas aplikasi, bukan signifikansi/kestabilan akurasi model atau kecocokan byte kode runtime pengguna.
