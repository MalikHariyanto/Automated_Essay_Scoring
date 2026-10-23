# Reuse E6: perbaikan perbedaan PyTorch GPU/CPU

Versi awal menolak sumber hanya karena string versi Torch berbeda, sebelum model final dicoba dimuat. Ini menghalangi perpindahan GPU ke CPU yang seharusnya dapat memakai final state_dict. Tidak ada training S3 yang dimulai pada kegagalan pemeriksaan tersebut.

Sekarang reuse menerima perbedaan build CUDA/CPU, patch dan minor PyTorch stabil 2.x mulai 2.6. Model final wajib lolos checksum dan dimuat dengan `torch.load(map_location='cpu', weights_only=True)` lalu `load_state_dict(strict=True)`. Dataset, split, profil kode, metode, lima fold, pilihan epoch dan prediksi sumber tetap diperiksa. Transformers/NumPy tetap harus sama. Tidak ada perubahan ke resume optimizer atau kontrak checkpoint epoch aktif.

Versi sumber/aktif dicatat di `s1_s2_reuse.json`, `final_report.json` dan sheet **Asal Model**. Metrik S1/S2 adalah hasil tersimpan dari runtime sumber; pembacaan vektor/inferensi baru lintas runtime tidak dijamin identik bit demi bit. Model, prediksi dan konfigurasi sumber tidak diubah, dan S1/S2 tidak dilatih ulang.

## Identitas artefak

| Notebook | SHA-256 file | Fingerprint kode |
| --- | --- | --- |
| CPU | `ddaf09d78d962ea3a15dabb5d43fcaead5ca3c96e9ec4bc01c0b241d37dc8232` | `a7ad04454c273399dea45a54c7a695383e813affee36990e28b0c2d811c34f83` |
| GPU | `62b00191c6855ede9f02228f4bea7d25c9a584c6fc7697c33a8cb8205f2a94ba` | `953776a8c2125043e6660b334f0350f2533a803d46be9e8119c9df13a8f37ebe` |

Builder GPU dijalankan dahulu, lalu builder CPU; mirror `aes-learning/notebooks/experiments/` identik. Keduanya memakai helper reuse yang sama, sehingga fix tidak bergantung pada GPU tersedia. MLP acuan, LoRA dan E6 CPU non-reuse tetap tidak diubah.

## Pengujian

Cwd `/workspace/Automated_Essay_Scoring`:

```bash
HF_HOME=/workspace/Automated_Essay_Scoring/outputs/lora_hf_cache \
PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
outputs/dataset300_env_py313/bin/python -m pytest \
  tests/test_s3_cpu_reuse.py tests/test_s3_gpu_reuse.py -q --disable-warnings \
  --basetemp=/tmp/e6_reuse_torch_portability_verified \
  --junitxml=outputs/e6_reuse_torch_portability_verified.xml
```

Exit **0**; **28 passed, 0 failed, 0 skipped, 0 xfailed**, satu warning, 149,67 detik. Log privat: `outputs/e6_reuse_torch_portability_verified.log`.

Pengujian mencakup penerimaan metadata Torch beda build/patch/minor (`2.6.0+cu124`, `2.6.1+cpu`, `2.10.0+cu128`), pencatatan asal, pemuatan final dan penyalinan byte-identik, serta penolakan versi tak didukung, perubahan Transformers/NumPy dan kegagalan load_model. Kasus beda versi mensimulasikan metadata pada bobot fixture yang dibuat di CPU; tidak menjalankan tiga versi Torch tersebut secara nyata.

Regresi juga menjalankan seluruh cell notebook CPU dan salinan teknis GPU hingga lima Excel/ZIP, memulihkan run dari runtime baru, memastikan hanya S3 dilatih, dan menjaga checksum seluruh sumber. Salinan GPU diuji di CPU dengan override perangkat khusus fixture. Mesin tidak memiliki GPU fisik.

Lingkungan: Python 3.13.5, Torch 2.6.0+cpu, Transformers 4.48.3, NumPy 2.2.6, pandas 2.2.3. Data 24 jawaban buatan, encoder Transformer satu layer acak dimensi 768/1024. Ini bukti teknis; bukan pengujian bobot penelitian pengguna atau RMSE E6. Log, checkpoint dan Excel fixture berada di `outputs/` yang diabaikan Git.

Percobaan awal memakai ruang workspace dan berhenti karena disk penuh saat menyalin fixture: 23 passed, 1 failed, 4 errors (Errno 28). Folder fixture buatan percobaan tersebut dihapus, log disimpan; pengulangan di atas memakai /tmp. Ini bukan error pada Drive atau hasil penelitian pengguna.
