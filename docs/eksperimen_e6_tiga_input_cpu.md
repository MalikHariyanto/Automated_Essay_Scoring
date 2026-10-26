# E6 tiga input untuk seluruh skenario

Notebook: `notebooks/experiments/AES_300_E6_Tiga_Input_CPU.ipynb` (salinan sama di `aes-learning/notebooks/experiments/`). File ini terpisah dari E6 jawaban-saja dan E8. Tidak mengubah notebook atau hasil lama.

| Peran | Kolom dataset |
| --- | --- |
| Input teks X | `soal`, `response_text`, `jawaban_referensi` |
| Label/target y | `rater` |
| Kelompok pencegah kebocoran | Email dan hubungan duplikat jawaban; bukan fitur encoder |

`bundle.texts` berisi jawaban siswa. `contexts_from_bundle(bundle)` menyelaraskan soal dan referensi dengan indeks jawaban. Keduanya membentuk satu contoh input berpasangan pada setiap encoder: teks A = soal/referensi; teks B = jawaban. `bundle.y` berasal dari `rater`, dipakai pada loss/evaluasi dan tidak masuk tokenizer. Dengan demikian tiga kolom input tidak berarti tiga target atau tiga encoder.

Semua skenario menerima ketiga teks: S1 RoBERTa fine-tuned + linear; S2 BGE-M3 frozen + MLP; S3 fusion keduanya + MLP. Head/dimensi mengikuti E6: RoBERTa768, BGE1024, proyeksi BGE256, concat1024, MLP hidden256. LR RoBERTa S1 1e-5, S3 5e-6, head/proyeksi1e-4. Karena input seluruh skenario berubah, S1/S2/S3 dilatih ulang; tidak menggunakan hasil kontrol E6 lama.

Tokenizer memasukkan special token pasangan sendiri. Konteks maksimal192token termasuk prefix; soal maksimal64token dan sisa untuk referensi. Window RoBERTa maksimal512, BGE1024, overlap128. Konteks diulang di setiap window; seluruh token jawaban dicakup. Pemotongan soal/referensi dicatat. CLS antar-window dirata-ratakan; BGE memakai L2 setelah agregasi. Pada jawaban satu window, BGE tetap memakai CLS lalu L2. Ini perubahan kebijakan input/panjang, bukan klaim resep asli BGE retrieval atau jaminan RMSE lebih rendah.

CPU float32 dua thread. Seed42, split_seed2026, development/test80/20, lima fold pada development, maksimal25epoch/fold, early stopping patience5, microbatch1/batch efektif8, warmup10% dan scheduler linear. Epoch refit dipilih dari ceil median epoch terbaik CV. Nilai guru0–100 dinormalisasi hanya untuk loss MSE; prediksi mentah dikembalikan ke poin0–100 untuk RMSE/Pearson, kategori salinan untuk QWK/AC2.

Drive: `MyDrive/AES_300_E6_Tiga_Input_CPU_Privat`. Autosave memakai dua slot tetap A/B per fold/refit tanpa Drive API atau auto-delete. Model final serta metadata disimpan. Jalankan notebook yang sama dari atas untuk melanjutkan progres yang cocok. Hasil eksperimen lama tidak perlu dihapus. Semua fold tetap memerlukan ruang Drive; CPU melatih ketiga skenario sehingga lebih lama daripada E6 reuse yang hanya melatih S3.

Jalankan dari atas dan unggah `fix_dataset_300.xlsx` jika diminta. Untuk prediksi baru, isi `NEW_QUESTION`, `NEW_REFERENCE`, `NEW_ANSWER`; skor guru baru tidak diperlukan untuk menghasilkan prediksi. Bila kosong, contoh held-out beserta konteks baris yang sama dipakai. Lima workbook utama diekspor ke `Hasil_AES_E6_Tiga_Input_CPU.zip`, meliputi data/tiga teks, token kedua encoder beserta segmen dan cakupan, vektor awal/akhir, kata/kalimat, riwayat, guru/prediksi/pembulatan dan empat metrik.

Dataset unggahan diperiksa lokal: 300 baris, tiga input dan label lengkap, satu label per jawaban, development240/test60 serta lima fold. Tidak dilakukan training penuh pretrained pada dataset tersebut. Soal boleh muncul pada train/test; protokol ini tidak menguji generalisasi ke soal baru. Test lama sudah digunakan selama pengembangan, jadi hasil lanjutan bersifat eksploratif dan pemilihan metode tetap melalui development/CV.

Verifikasi: delapan pengujian lulus, tanpa failed/skipped/xfail (satu peringatan pustaka notebook pada fixture). Pengujian memakai Transformer kecil berbobot acak, bukan bobot pretrained penelitian. Seluruh30codecell dijalankan: 15fitCV +3refit, scoring tiga model, ekspor fitur/kata/kalimat, lima Excel, lalu kernel baru membaca hasil yang sama tanpa mengubah model final. Diperiksa pula efek konteks pada kedua encoder, independensi fitur dari target, invalidasi cache saat soal/referensi berubah, kebijakan gradien/head/LR, cakupan token, fingerprint sumber dan builder deterministik. Bukti lokal: `outputs/three_input_tests_v2*` dan `outputs/three_input_design/`.

Builder memakai sumber bersih yang dipin pada commit2257cb0 dan membenamkan semua fungsi; Colab tidak perlu clone atau mengimpor proyek. Notebook eksekusi, dataset asli, identitas, jawaban dan skor individual tetap privat.
