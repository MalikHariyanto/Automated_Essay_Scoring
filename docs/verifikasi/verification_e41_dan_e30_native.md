# Verifikasi E41 dan E30 native — 4 November 2026

Pengguna meminta notebook E41 Scalar Mix dan E30 MLP acuan dalam format native tanpa cell magic. Verifikasi ini memakai data sintetis dan model Transformer kecil berbobot acak di CPU. Tidak ada hasil penelitian E41, reproduksi skor historis E30, atau eksekusi checkpoint pretrained asli di GPU.

## Identitas source

- Cwd `/workspace/Automated_Essay_Scoring`, baseline kerja `a16bfbbf76f774442c7d375d5d40b33fa0e5f122`.
- Builder `scripts/build_e41_and_e30_native.py` dipin ke format E39 pada `5f1c993804560d12c06d7d11d4540e6d11307cf2`; mengembalikan hidden64perarah dan membuat dua notebook mandiri. Source E30 historis,notebook E36–E40 dan HTML hasil tidak ditimpa.
- Masing-masing184cell/89codecell; output kosong dan mirror byte-identik.
- E30native raw SHA256 `e7f603667c0de8657b0a625804228f0522c7b74b86f308083d973882e8c98af6`; visible-code SHA256 `c89f75ffb17b0be01dc0c2658b287fa457b1bce3f28e590c8f91ca75016124e6`.
- E41 raw SHA256 `c1ac4579fc3c181a8c87f595edd10d947255ce7b2311a5f63eaf24155e3aa183`; visible-code SHA256 `5e299c4b0f70617f6a7726987b151d2419075fd688f2333d7d849df3e8906357`.

## Perintah dan hasil

```bash
PYTHONPATH=src:scripts /tmp/e39-verify/bin/python -m pytest \
 tests/test_e41_and_e30_native.py -q -s --tb=short \
 --basetemp=outputs/e41_study/verified_fixtures
```

Exit0, **6passed,0failed,0skipped,0xfailed**,134,43detik. Satu warning deprecation plugin pytest lingkungan. Log privat `outputs/e41_study/tests.log`; kernel Jupyter dijalankan dengan izin socket localhost. Pengujian awal empat kasus non-notebook juga lolos6,78detik/2deselected; empat kasus tersebut diulang dalam enam kasus akhir, bukan sepuluh kasus berbeda.

1. **Struktur dan protokol kedua notebook:** validasi schema/hash,mirror,rebuild deterministik,source historis utuh,Python biasa tanpa exec/eval/magic/worker tersembunyi,cosine atau LLRD. Masing-masing S3 saja,5CV+1refit+1test,tanpa loop multi-seed/ensemble/kontrol otomatis. AST split,train_all,metrik,encoder,BGE,mask,windowing dan save_model sama dengan E30. E30native memiliki fit identik secara AST; loop epoch E41 identik dengan E30.
2. **Matematika Scalar Mix:** empat parameter,softmax awal0,25,jumlah bobot1,pemilihan hanya empat layer Transformer terakhir dari urutan yang juga mempunyai embeddings. Nilai campuran pada tensor terkendali sesuai hitungan; gradien masuk ke empat logit dan empat tensor terpilih. Update AdamW mengubah bobot; jalur tensor float16 finite dan mempertahankan dtype. Shape/jumlah layer/config yang tidak sah ditolak.
3. **Kesetaraan E30native:** random RoBERTa4layer/hidden768,tokenizer teknis dan24baris sintetis. Semua setting E30 yang digunakan S3 sama,split identik,state awal,prediksi eval dan seluruh state setelah satu langkah backward/AdamW bit-identik dengan E30 historis. Notebook native tidak mempunyai parameter Scalar Mix atau LLRD.
4. **Gradien,optimizer,checkpoint dan bobot terpilih E41:** state awal komponen E30 tetap sama; tambahan parameter tepat4. LR encoder5e-6,head/proyeksi/pooling_norm/scalar_mix1e-4; semua parameter tercakup tepat sekali. Prediksi[N,1],ringkasan[N,128],concat[N,1024]. Gradien logit,embedding,layer encoder bawah/atas,BiGRU dan proyeksi referensi nonzero. Load checkpoint mempertahankan bobot; config campuran berbeda ditolak; S1/S2 dan jumlah layer tidak mencukupi ditolak. Subkasus teknis memaksa pilihan validasi epoch1dan early stop epoch2: bobot epoch2berbeda,namun model/info mengembalikan bobot epoch1yang telah dipulihkan. RMSE validasi1/2pada subkasus ini adalah nilai kontrol tes, bukan hasil AES.
5. **Notebook E30native lengkap:** seluruh89codecelldijalankan pada24baris sintetis,CPU,1epoch,token pendek dan random Transformer1layer/hidden768/1024. Lima fold,satu refit,satu test,lima Excel dan ZIP selesai. Tidak ada sheet/parameter Scalar Mix. X/y yang terlihat sama dengan data aktual training. Kernel baru memakai cache sesuai hash tanpa mengubah checkpoint akhir.
6. **Notebook E41 lengkap:** fixture sama dengan random Transformer4layeragar pemilihan empat layer dijalankan nyata; source publik tetap model pretrained,GPU,300baris dan25epoch. Lima fold,satu refit,satu test,lima Excel dan ZIP selesai. Excel05→Bobot Scalar Mix berisi24baris:empat bobot positif berjumlah1untuk setiap fold/refit. Bobot final Excel cocok dengan softmax logit checkpoint,berubah dari0,25dan tersimpan bersama LR1e-4. Ringkasan co-attention tetap128komponen. Kernel baru memakai cache sesuai hash tanpa perubahan checkpoint akhir. Import proyek dilarang dalam kernel fixture; tidak ada worker .py tersembunyi.

## Lingkungan dan publikasi

Venv `/tmp/e39-verify`,Python3.12.14,torch2.6.0+cpu,transformers4.44.2,pytest9.2.0,nbformat5.11.1,nbclient0.11.0,NumPy2.3.5,pandas2.2.3,SciPy1.17.0,scikit-learn1.8.0,openpyxl3.1.5. Durasi tes bukan estimasi training penelitian di GPU. Eksekusi fixture tidak membuktikan akurasi atau kompatibilitas GPU Colab terbaru.

Hanya source notebook kosong,builder,tes dan dokumentasi aman dipublikasikan. Dataset,teks siswa,unggahan,executed notebook,Excel,log,bobot dan kunci tetap lokal. Manifest source/perintah/hasil tersimpan privat pada `outputs/e41_study/verification.json`. Setelah push, SHA branch dan byte notebook utama/mirror/metode diperiksa; publikasi tidak membuktikan training penelitian.
