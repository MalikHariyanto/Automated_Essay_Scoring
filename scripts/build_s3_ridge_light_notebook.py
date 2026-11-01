"""Build the budgeted variant without overwriting the nested experiment."""
import copy,hashlib,json,subprocess
from pathlib import Path
from build_bge_readout_notebook import definitions
ROOT=Path(__file__).resolve().parents[1]
NAME='AES_300_S3_Fusion_Ridge_Hemat_Colab'
PIN='0a5f6795e23809554767dbd0bbe5553a8b00b415'
BASE='notebooks/experiments/AES_300_S3_Fusion_Ridge_Tiga_Input_Colab.ipynb'
def build():
 raw=subprocess.check_output(['git','show',PIN+':'+BASE],cwd=ROOT);n=json.loads(raw);old=n['metadata']['aes_standalone']['code_sha256'];cells=copy.deepcopy(n['cells'][:31])
 for c in cells:c['source']=c['source'].replace(old,'__LIGHT_HASH__').replace('AES_300_S3_Fusion_Ridge_Tiga_Input_Colab',NAME)
 def md(i,s):cells[i]['source']=s
 def code(i,s):cells[i]['source']='%%aes\n'+s
 md(0,'''# S3 Ridge hemat · 5-fold CV · alpha tetap
Tiga input: soal + referensi + respons. Fusi E21 concat1024 → Ridge, dengan kontrol MLP dan S2 Ridge.
**6 pelatihan × 5 epoch = 30 epoch**, menggantikan anggaran lama400epoch. Ini ordinary grouped CV, bukan nested CV. Alpha S3 **1.0 tetap**, titik awal default sklearn Ridge; bukan optimum dari paper. Epoch5 merupakan batas sumber daya, bukan klaim optimum.
Pilih T4 GPU, Run all, unggah dataset asli. Kuota Colab tidak dijamin. Hasil penelitian belum tersedia.
**Setelah tiap fold, simpan unduhan Backup_S3_Ridge_Hemat_Fold_N.zip ke laptop.** Browser mungkin meminta izin beberapa unduhan. ZIP yang hanya berada di disk Colab belum melindungi dari runtime hilang.
Untuk restore, unggah backup terakhir dan dataset asli melalui panel Files sebelum Run all. Backup mencakup fold selesai dan fitur BGE; tahap yang terputus diulang paling banyak5epoch. Backup tidak membawa optimizer aktif atau encoder final. Tidak menggunakan Drive.''')
 md(6,'## 2 · Batas sumber daya\n5fold tetap,5epoch setiap fit. Alpha tetap1.0; tidak ada pencarian alpha. RoBERTa5e-6, proyeksi/norm/MLP1e-4 seperti acuan. Perubahan epoch/protokol harus dicatat ketika membandingkan hasil lama.')
 s=cells[8]['source'].removeprefix('%%aes\n').replace('EPOCHS, FOLDS = 25, 5','EPOCHS, FOLDS = 5, 5').replace('S3_RIDGE_ALPHAS = [.01, .1, 1., 10., 100.]','S3_FIXED_ALPHA = 1.0\nS3_RIDGE_ALPHAS = [S3_FIXED_ALPHA]').replace("outputs/s3_fusion_ridge'","outputs/s3_fusion_ridge_light'").replace('s3_ridge_alphas=S3_RIDGE_ALPHAS,','s3_fixed_alpha=S3_FIXED_ALPHA, s3_ridge_alphas=S3_RIDGE_ALPHAS,').replace('FOLDS+FOLDS*(FOLDS-1)//2+1','FOLDS+1')
 code(8,s)
 cells[7]['source']=cells[7]['source'].replace("SCOPE='s3_fusion_ridge_nested_cv'","SCOPE='s3_fusion_ridge_light_fixed'")
 md(20,'## 6 · Training representasi train-only\nMLP bantu melatih RoBERTa/norm/proyeksi selama5epoch tetap, lalu fitur dikunci untuk Ridge. Label di luar train ditutupi NaN. BGE encoder frozen. Snapshot epoch lokal hanya dapat dilanjutkan pada konfigurasi/perangkat yang cocok.')
 cells[23]['source']=cells[23]['source'].replace('pelatihan final; data uji belum dibuka','training representasi; label evaluasi tidak dipakai')
 md(24,'## 7 · Ordinary CV dengan parameter tetap\nLima outer fit + satu final; tidak ada inner training atau seleksi alpha. Semua prediksi OOF berasal dari kelompok yang tidak dipakai melatih representasinya. Satu seed; bukan pengukuhan multi-seed. Cache fold selesai dapat dipulihkan lintas perangkat; perbedaan perangkat dicatat pada asal cache.')
 runtime=(ROOT/'scripts/s3_ridge_experiment.py').read_text()
 pre=runtime[:runtime.index('def validate_s3_protocol')]
 representation=definitions(runtime,{'validate_s3_protocol','collect_s3_features','fit_s3_representation','stage_values'})
 representation=representation.replace("proof=dict(contract=contract,", "proof=dict(execution_environment=dict(torch=str(torch.__version__),device=device_name(cfg)),contract=contract,")
 code(27,pre+representation)
 code(28,definitions(runtime,{'s3_predictions','s3_evaluation'}))
 code(29,(ROOT/'scripts/s3_ridge_light.py').read_text())
 export=definitions(runtime,{'s3_duels','export_s3_ridge'})
 export=export.replace('S3_FEATURE_FUSION_RIDGE','S3_FUSION_RIDGE_LIGHT_FIXED_ALPHA').replace('Hasil_AES_300_S3_Fusion_Ridge_Tiga_Input.zip','Hasil_AES_300_S3_Fusion_Ridge_Hemat.zip').replace("selected_alpha=result['final_contract']['selected_alpha']","fixed_alpha=result['final_contract']['fixed_alpha']")
 export=export.replace("Alpha_Seleksi=result['alpha_search'],Inner_Validation=result['inner_validation'],Outer_Alpha=result['outer_alpha']","Alpha_Tetap=result['alpha_contract'],Outer_Fold=result['outer_summary']")
 export=export.replace('Nested OOF untuk S3 Ridge; final alpha dipilih development, test hanya setelah model dikunci','Ordinary grouped CV; alpha tetap sebelum run; tanpa tuning; test setelah model dikunci').replace('Grid pada fitur concat1024 mentah; alpha tidak disamakan skalanya dengan BGE L2','Alpha S3 tetap1.0 pada concat mentah; baseline S2 tetap0.01')
 code(30,export)
 def add(s,magic=False,markdown=False):cells.append(dict(cell_type='markdown' if markdown else 'code',metadata={},source=s if markdown or not magic else '%%aes\n'+s,**({} if markdown else dict(outputs=[],execution_count=None))))
 # Restore after the split but before BGE extraction. Functions are defined before these run cells.
 # Definitions above do not need BGE to have been extracted yet.
 code(19,definitions(cells[19]['source'],{'bge_features'}))
 add('## 8 · Restore backup opsional dan fitur BGE',markdown=True)
 add("""BACKUP_FILE = ''  # Kosong: cari otomatis satu backup terakhir yang diunggah di /content.
choices=list(Path('/content').glob('Backup_S3_Ridge_Hemat_Fold_*.zip'))+list(ROOT.glob('Backup_S3_Ridge_Hemat_Fold_*.zip'))
if BACKUP_FILE:backup=Path(BACKUP_FILE)
elif choices:backup=max(choices,key=lambda p:p.stat().st_mtime)
else:backup=None
print('Backup untuk restore:',backup if backup else 'Tidak ada; training baru')
aes_cell('', 'RESTORE_BACKUP_PATH='+repr(str(backup) if backup else ''))""")
 add("""protocol=initialize_s3_light(bundle,manifest,cfg,output)
if RESTORE_BACKUP_PATH:
    restore_light_backup(RESTORE_BACKUP_PATH,bundle,manifest,cfg,output)
if (output/'bge_frozen.npy').is_file():
    X=np.load(output/'bge_frozen.npy',allow_pickle=False)
    validate_s3_protocol(bundle,X,manifest,cfg)
    print('BGE dari backup lokal; encoder BGE tidak dihitung ulang.',flush=True)
else:
    X=bge_features(bundle,cfg,output).numpy().astype(np.float32)
    np.save(output/'bge_frozen.npy',X)
fold_results=[]""",True)
 for f in range(1,6):
  add(f'## Fold {f}/5 · 5epoch, kemudian unduh backup',markdown=True)
  add(f'fold_results.append(run_s3_light_fold({f},bundle,X,manifest,cfg,output))',True)
  add("""info=json.loads((ROOT/'outputs/s3_light_backup_download.json').read_text())
print('SIMPAN KE LAPTOP:',info['archive'])
print('Fold selesai dalam backup:',info['completed_outer_stages'])
if IN_COLAB:
    from google.colab import files
    files.download(info['archive'])
else:print('Backup tersedia lokal pada path di atas.')""")
 add('## 9 · Final fit5epoch dan evaluasi test\nTidak ada pemilihan parameter dari OOF atau test. Hasil ini ablation anggaran kecil dan belum tentu menyamai model25epoch.',markdown=True)
 add("result=finalize_s3_light(bundle,X,manifest,cfg,output,fold_results)\nshow(pd.DataFrame([r for r in result['evaluation'] if r['partition']=='TEST_HISTORICAL_HOLDOUT']),'Test: parameter tetap')\nshow(pd.DataFrame([r for r in result['evaluation'] if r['partition']=='DEVELOPMENT_POOLED_OOF']),'OOF ordinary grouped CV')\nshow(pd.DataFrame(s3_duels(result)),'Duel per jawaban')",True)
 add('## 10 · Lima Excel + ZIP\nReferensi komponen: [Ridge1970](https://doi.org/10.1080/00401706.1970.10488634), [Cawley/Talbot2010](https://www.jmlr.org/papers/v11/cawley10a.html), [SentEval2018](https://aclanthology.org/L18-1269/), [BGE-M3](https://arxiv.org/abs/2402.03216), [RoBERTa](https://arxiv.org/abs/1907.11692), [AdamW](https://arxiv.org/abs/1711.05101). Referensi lengkap di Excel. Tidak ada paper yang diklaim menetapkan5epoch/alpha1.0 optimal untuk AES ini.\nCheckpoint encoder final besar tetap lokal, unduh terpisah bila ingin menyimpan model teks baru.',markdown=True)
 add("download_info=export_s3_ridge(bundle,X,manifest,cfg,result)\ndump(ROOT/'outputs/s3_light_download.json',download_info)",True)
 add("""DOWNLOAD_FULL_ENCODER=False
info=json.loads((ROOT/'outputs/s3_light_download.json').read_text())
print('Hasil:',info['archive'])
print('Checkpoint final:',info['final_representation_checkpoint'])
if IN_COLAB:
    from google.colab import files
    files.download(info['archive'])
    if DOWNLOAD_FULL_ENCODER:files.download(info['final_representation_checkpoint'])""")
 identity=hashlib.sha256('\n'.join(c['source'].removeprefix('%%aes\n') for c in cells if c['cell_type']=='code').encode()).hexdigest()
 for i,c in enumerate(cells):c['source']=c['source'].replace('__LIGHT_HASH__',identity);c['id']=hashlib.sha256((str(i)+c['source']).encode()).hexdigest()[:12]
 n['cells']=cells;n['metadata']['aes_standalone']=dict(code_sha256=identity,base_commit=PIN,base_path=BASE,base_sha256=hashlib.sha256(raw).hexdigest())
 n['metadata']['aes_experiment'].update(id='S3_FUSION_RIDGE_LIGHT_FIXED_ALPHA',supervised_representation_refit_inside_inner_cv=False,evaluation_protocol='ordinary grouped CV; fixed alpha',default_epochs=5,representation_fits=6,completed_fold_backups=True)
 data=json.dumps(n,ensure_ascii=False,indent=1)+'\n'
 for d in ('notebooks/experiments','aes-learning/notebooks/experiments'):(ROOT/d/(NAME+'.ipynb')).write_text(data)
 print(NAME,identity)
if __name__=='__main__':build()
