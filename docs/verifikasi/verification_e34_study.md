# Verifikasi audit E34 — 2 November 2026

Cwd `/workspace/Automated_Essay_Scoring`, base HEAD `7f3363643e6bf402a293de069bcfc23d85b19c89`. Source eksperimen E34 tidak diubah; ini audit hasil unggahan dan pembaruan buku, tanpa training baru.

- Audit lima workbook melalui `build_e21_study.audit(zip, 'E34')`:300data,240development,60test,15fold,riwayat dan epoch refit cocok. Seluruh metrik test dihitung ulang;toleransi1e-6,deviasi maksimal2,400250771472656e-7. Posisi token/bobot attention,finite vector,ringkasan128danpooling768diverifikasi. Notebook eksekusi/checkpoint tidak disertakan sehingga byte runtime/perangkat tidak dibuktikan.
- `PYTHONPATH=src:scripts /tmp/e29-verify/bin/python -m pytest tests/test_e34_study.py tests/test_e33_study.py tests/test_e31_e32_study.py tests/test_three_input_study.py -q --tb=short`:exit0;21passed,0failed,0skipped,0xfailed;1,83detik. Dua tes katalog tambahan `tests/test_three_input_identities.py`:exit0;2passed;0,66detik. Tidak ada pengujian akurasi model baru.
- Pengemas dipanggil oleh dua tes `tests/test_e21_study.py -k encrypted_packager` memakai pytest dari venv dan cryptography sistem di sys.path/PYTHONPATH:exit0;2passed,5deselected,0failed,0skipped;0,25detik. Python sistem tidak memiliki pytest; pemanggilan awal tersebut tidak menghasilkan hasil tes dan diganti dengan command venv. Peringatan deprecation anyio tidak memengaruhi hasil.
- `NODE_PATH=/workspace/Sistem-BTD/node_modules node scripts/check_e34_study.mjs`:exit0;2kelompok browserpassed,0failed. Chromium154.0.8037.57;14identitas,metode/LR,3buku kode,layar390/1440tanpa overflow. Kunci salah ditolak,kunci lama membuka buku;setelah jaringan diblokir2520prediksi terbaca,Excel riwayat E34 byte-identik,serta kode/output E30 tetap dapat dibuka.
- Pembuka awal gagal setelah ukuran plaintext565.027.738byte melampaui batas satu string V8. Pengemas kini mendecode UTF-8 bertahap1MiB setelah dekripsi dan pemeriksaanSHA256,sehingga tidak membentuk satu string seluruh buku. Pengujian browser yang lulus memakai paket akhir dengan perbaikan ini.
- Integritas paket privat:14ZIP dan70XLSX/655sheet tetap byte-identik melalui SHA256;13snapshot/audit prediksi lama,78duel lama,serta3dokumen kode identik. Seluruh91duel berjumlah60jawaban per skenario. Buku MLP lama tidak berubah.
- Pengemas memverifikasi roundtrip AES-GCM,gzip,HTML dalamZIP,dan rekonstruksi bagian lokal. Paket akhir408,65MiB;SHA256`f6d15ef172dd0b15edcda371898b0d298316b6e8c85280fc006bdba02b4ca3e1`. Kunci lama dipertahankan;tidak dicetak atau ditanam dalam source.
- File lengkap tersedia lokal di`outputs/e34_study/Belajar_AES_Tiga_Input_Lengkap_E34.zip`;tidak ada push GitHub atau publikasi paket privat pada tugas ini. Bagian lokal hanya hasil pengemasan,belum diunggah.

Manifest sumber aktual disimpan privat di`outputs/e34_study/source_manifest.json`. Hash file utama:

- `scripts/build_e21_study.py`: `ad5fdaac2f756cb17c283d70083acd7279000f51a5d6be7fdc7c455de9dcf41c`
- `scripts/build_three_input_study.py`: `577823415cc039da2f4f00e4911b606a7fdee8e8765421221f6959aaff03d7bb`
- `scripts/package_private_study.py`: `eeda74333a52f1b86881d629a520a9a353635d7895ed7e8886469b234be5b058`
- `scripts/templates/three_input_study.js`: `0cd0efa3bc7cdfcb10af8424f3aca1aa8d89a5327ff27c57431e9fe7526d4b82`
- `scripts/templates/three_input_study.css`: `88d8f28b9386c7937257df1ec4da722d14edac3ea1810365becb80e4c4f3e18a`
- `scripts/three_input_identities.py`: `f008229241fe6e0816f9c305767eab8dd74fbf011ea341d3d329cd94a2810b72`
- `scripts/check_e34_study.mjs`: `334290159a9370961e8c3b2aa1be31afd419fb558b4287bd9f517bd6216b777d`
- `tests/test_e34_study.py`: `a9195a843e038865cd37d02b46f864796b74fc3180e289602860bfb0ccba5ba2`
- `docs/review/review_e34_study.md`: `98cf56723b1f107ecbf2f7c8fe8e79ef3630ebb65db52bf3d21737a8ac8c08ba`
- `docs/snapshots/e34_study_snapshot.json`: `6c0cad89eea200595383d67209d087d82934b4b8b54c98c2326a2e13b246ab97`
- `docs/belajar_Tiga_Input_Spider.html`: `b01d53c1ccaecfe097fcc7c9bee0c1c47f61aee5b92d2f27effd2f5144ce21f5`
