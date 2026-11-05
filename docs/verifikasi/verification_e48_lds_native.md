# Verifikasi E48 LDS-weighted MSE — 5 November2026

E48 dibuat dari E43 pada commit `7241a21dfec37c942948b754da3960fba4c7731c`. HEAD kerja awal `4bd822739c7ab80cf79a09ada1579442f5119596`, cwd `/workspace/Automated_Essay_Scoring`. Pengguna menyetujui penggantian loss, mempertahankan seluruh arsitektur E43. Ini bukan hasil penelitian E48.

## Source dan protokol

Notebook `AES_300_E48_E43_S3_LDS_WeightedMSE_Native_Learnable_GPU.ipynb`:186cell/90cellkode, semua output kosong; mirror utama/`aes-learning` identik. RawSHA256 `1c3a9926c06e8164fcf07b903ee34cf315b849ba6363865e65b5500b0af5ed87`; visible-codeSHA256 `728d3c885ce08d2b16d716885c9a3a569416327f95bde8581591d3ca48ddfd83`.

Tiga X/satu y;RoBERTafine-tuned,ScalarMix4+gamma,BiGRU64/arah,co-attention,maskedmean,BGEfrozen,fusion danMLP tetap. S3saja,satu5CVbaru,maks25epoch,patience5,saturefit/test,tanpasweepseed/kontrolulang. Upload cukupdatasetExcel. JadwalE43asli,durasi fit masing-masing;tidakmembawa horizonrefit25 atauZIPreuseE46/E47.

## Perintah dan hasil

```bash
/tmp/e39-verify/bin/python -m pytest tests/test_e48_notebook.py \
 -q -s --tb=short --basetemp=outputs/e48_study/final_fixtures
```

Exit0; **5passed,0failed,0skipped,0xfailed**,87,27detik. Satuwarningdeprecationpluginlingkungan. Log privat `outputs/e48_study/final_tests.log`;manifest `verification.json`.

1. **Source/protokol:** notebook schema/hash/mirror/rebuilddeterministik/outputs kosong. AST model,ScalarMix,BiGRU/co-attention/mean,Engine,optimizer,train_all,metrics,AC2,split,windowing,encoder danBGE tetapE43 setelah teks nama versi pada error dinormalisasi. Tidak ada magic,worker,exec/eval dalamnotebook,cosine,ListNet,loopseed,ZIPacuan atauhorizonrefittetap25. E43 tidakditimpa.
2. **Matematika dan batasdata:** histogram→Gaussian→inverse/mean1 cocok dengan `scipy.ndimage.convolve1d(mode=constant)` padafixtureteknis,termasuk batas100. Skorjarang mendapatbobotlebihbesar. Mengubah seluruh label di luartraining menjadi−999 danmengubahteksheld-out tidakmengubahbobot/audit. Indeks kosong/duplikat/di luar batas dankernelgenap ditolak.
3. **Loss/gradien:** weightedMSE cocok rumusanalitis;gradien=2w(p−y)/N. Akumulasi microbatch1/2/3/4termasukbatchtakmerata cocokloss/gradienbatchpenuh. Bobot1menghasilkanMSEbiasa. Shape broadcasting ditolak. Bobot tidakdinormalisasi ulang permicrobatch.
4. **Model/resume:** initialmodelstate bit-identikE43pada seedsama,splitidentik. Fit2epochdenganinterupsisetelahsaveepoch1lalu resume menghasilkanhistory,bobotauditLDSdanmodelstate bit-identikdenganfit2epochtanpainterupsi. LogMSEdanLDS-MSEterpisah. Train/validationtumpangtindihditolak.
5. **Eksekusinotebooklengkap:** kernelJupyterterisolasi menjalankanseluruhnotebook pada24datasintetis,randomTransformer4layer berdimensiRoBERTa768/BGE1024,tokenpendek,CPU,CV1epochuntukwaktu. LimaCV,saturefit,satutestmenghasilkan5Excel+ZIP;18OOFdan6testterpisah;90barisbobottrainfit lengkap. Kernelbarumenjalankannotebookulang tanpa mengubahcheckpointfinal. Sourcepubliktetap300data/25epoch/patience5. Gate memverifikasimetrik tanpa bobot,meanSDCV,OOFsesuaifold,rekapselisihdanLDSdaritrainfold. Mengosongkantabeltest/CV,mengubahcount,menggantiIDOOFataumerusakbobotLDSmasing-masingmemblokirunduhan.

Helperfixturebersama hanyamenambahparameteropsional`prediction_columns`untukmenerima16kolomrincianE48;default5kolomarsiptetap. Dataset,sourceZIP,executednotebook,Excel,bobot danlog tidakdipublikasikan.

## Batas bukti

Python3.12.14,torch2.6.0+cpu,transformers4.44.2,pytest9.2.0,nbformat5.11.1,nbclient0.11.0,NumPy2.3.5,pandas2.2.3,SciPy1.17.0,scikit-learn1.8.0,openpyxl3.1.5. Jupyter membutuhkan socketlokal. Tidak melakukan training/pretrainedsmokeGPU atau mengevaluasi300datapenelitian. Akses model asli HuggingFace dariworkspace ditolakHTTP403;perbandinganmodelteknis menggunakanrandomTransformer,sesuai scopefixture,bukanbuktiunduhcheckpointasli. Tidak mengklaim semua runtimeGPUkompatibel,semuabugmustahilterjadi,atauLDSmeningkatkanskorpenelitian.

Rujukanimplementasi/metode pada [rancanganE48](../eksperimen/eksperimen_e48_lds_native_gpu.md). Sourcekosong/helper/builder/tes/dokumentasiaman dipublikasikan;sesudahpush,SHAbranch danbyte notebookutama/mirror/helper/metode diperiksa. Buktiprivat `outputs/e48_study/publication.json`.
