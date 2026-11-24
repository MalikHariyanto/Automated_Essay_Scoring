# E17D CPU lokal

[Notebook CPU](../../eksperimen/03_tiga_input_multi_trait/notebooks/AES_Reset_E17D_Local_CPU.ipynb) · [Requirements](../../eksperimen/03_tiga_input_multi_trait/requirements_e17d_cpu.txt)

Python 3.11/3.12 disarankan. Internet diperlukan untuk instalasi dan unduhan awal checkpoint. Sesudah model tersimpan pada cache Hugging Face, file cache dapat digunakan kembali. Tidak memerlukan GPU/CUDA atau Google Drive.

## Instalasi Windows PowerShell

Buat folder kerja dan letakkan notebook, requirements_e17d_cpu.txt, serta data_cleaned_3sekolah.xlsx di dalamnya. Buka PowerShell pada folder tersebut:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements_e17d_cpu.txt
.\.venv\Scripts\python.exe -m jupyterlab
```

Pada JupyterLab, buka notebook dan jalankan sel berurutan. Gunakan kernel lingkungan .venv yang menjalankan Jupyter. Tidak perlu mengaktifkan venv lewat Activate.ps1.

## Linux atau macOS

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install "torch>=2.6,<3"
.venv/bin/python -m pip install -r requirements_e17d_cpu.txt
.venv/bin/python -m jupyterlab
```

Notebook menetapkan DEVICE='cpu' sehingga GPU/MPS tidak dipakai, sekalipun tersedia. Untuk Linux yang khusus memakai wheel CPU, index PyTorch CPU pada contoh Windows dapat digunakan.

## Dataset dan keluaran

Jika dataset tidak berada pada folder kerja, notebook meminta path lengkap; path dengan spasi dapat dipakai. Variabel lingkungan AES_DATASET dan AES_OUTPUT bersifat opsional. Checksum dataset tetap dikunci. Hasil dan cache tersimpan pada outputs/aes_e17d_cpu secara bawaan, terpisah dari folder GPU. ZIP ditautkan dari notebook dan dapat dibuka langsung dari komputer.

Model tetap RoBERTa full fine-tuning + BGE-M3 frozen + proyeksi + concat + MLP empat skor. CPU memakai float32 tanpa CUDA mixed precision. Resep training, lima fold, scheduler refit, target dan evaluasi tidak diganti. Hasil numerik tidak dijamin identik dengan GPU.

Bawaan menjalankan 3 fit diagnostik development, 5 CV, dan 1 refit utama. RUN_DEVELOPMENT_DIAGNOSTIC=False melewati tiga fit diagnostik; catat jika digunakan. Fit utama yang telah selesai direuse sesuai identitas konfigurasi; fit terputus diulang dari awal. Tidak ada resume optimizer per epoch.

## Batas dan pemeriksaan

RoBERTa dan BGE-M3 tetap model besar. CPU dapat sangat lambat dan membutuhkan RAM memadai; waktu dan kebutuhan RAM puncak belum dibenchmark pada komputer pengguna. Beralih CPU tidak menjadikan encoder frozen atau menyederhanakan model.

Pemeriksaan publikasi: seluruh 62 sel kode lolos compile Python; pilihan CPU dan ketiadaan ketergantungan Colab/Drive diperiksa; fixture teknis menguji path lokal berspasi, pembuatan folder output, dan penolakan checksum salah. Arsitektur concat dipertahankan. Lingkungan pemeriksaan tidak menyediakan torch/transformers sehingga forward/backward pretrained, training penuh, dan instalasi semua dependensi belum diuji. Belum ada hasil penelitian CPU baru.
