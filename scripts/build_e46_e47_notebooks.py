"""Build native E46/E47: verified old CV, one fresh refit, aligned LR horizon."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER = '__NOTEBOOK_CODE_SHA256__'
TRAINING_HASH = {
    46:'162193a0b2ab2bbb10fcbff69ee9400515b726d4f33cd8eef410194e69f0de21',
    47:'e303f5375a5d11aeed78d315f8c53a34c2d81a988e149e6d6e9bd330111ca429'}
SPECS = [
    dict(number=46,base='E43',pin='7241a21dfec37c942948b754da3960fba4c7731c',
         base_name='AES_300_E43_E42_S3_MaskedMeanPooling_Native_Learnable_GPU',layers=4,refit=18,
         zip_sha='bc7b5bd88f574a478b40dc6cc5514c52e02655e19b4c33568524812511c5dd0a'),
    dict(number=47,base='E45',pin='6875780f709d1436f25a4706292d71c137f2fc55',
         base_name='AES_300_E45_E43_S3_ScalarMix12_Native_Learnable_GPU',layers=12,refit=7,
         zip_sha='6fea631fe1f86c3e37072506b8cbc4248ef41bf0b8d22290464a9c04d069ef64')]

REPORT_ADDITIONS = '''    old = '09_CV_acuan_dipakai_ulang.xlsx'
    for raw,title in [('Provenansi','Provenansi CV'),('Riwayat_CV_acuan','CV Acuan (dipakai ulang)'),
                      ('Ringkasan_fold_acuan','Ringkasan Fold CV Acuan'),('Bobot_mix_CV_acuan','Bobot Mix CV Acuan'),
                      ('MAE_toleransi_CV_acuan','MAE Toleransi CV Acuan')]:
        if source.has(old,raw):
            sheets[title] = source.copy(old,raw)
    if source.has('10_S3_refit.xlsx','Jadwal_LR'):
        sheets['Jadwal LR Refit'] = source.copy('10_S3_refit.xlsx','Jadwal_LR')
'''


def replace_once(text,old,new):
    assert text.count(old)==1,old[:110]
    return text.replace(old,new,1)


def intro(spec,name):
    e=f"E{spec['number']}";base=spec['base'];k=spec['layers'];epochs=spec['refit']
    return f'''# {e} · {base} refit · horizon LR 25 epoch · Native Learnable

**Diagnostik refit, tanpa menjalankan ulang CV.** Model mengikuti {base}, Scalar Mix {k} layer + gamma, loss MSE. Lima fold lama diverifikasi dan dipakai untuk memilih refit **{epochs} epoch**. Jadwal LR tetap memakai horizon **25 epoch**, sehingga pola warmup/decay mengikuti CV. Angka 25 bukan jumlah epoch training refit.

| Bagian | Nilai |
|---|---|
| CV baru / CV lama | 0 fit baru / 5 fold {base} dipakai ulang |
| Training baru | Satu refit {epochs} epoch dari awal pada 240 development |
| Scheduler | Horizon 25 epoch, warmup 10%, decay linear |
| Model | Scalar Mix{k}+gamma → BiGRU64/arah + co-attention + masked mean → fusion BGE → MLP |
| LR puncak | Encoder 5e-6; komponen lain 1e-4 |
| X / y | `soal`, `jawaban_referensi`, `response_text` / `rater` 0–100 |
| Evaluasi baru | Satu test; tidak digunakan memilih epoch |

**Upload dua file:** `fix_dataset_300.xlsx` dan **`Hasil_{spec['base_name']}.zip`**. Pilih T4 GPU → Run all. ZIP acuan dipilih pada cell upload setelah dataset. `DATASET_FILE` dan `CV_SOURCE_FILE` tersedia jika memakai path manual.

SHA256 ZIP acuan dikunci. Kode memeriksa seluruh teks/label, development/test, anggota lima fold, parameter yang tersimpan, history/epoch terpilih dan bobot campuran. Median CV harus menghasilkan tepat {epochs}. File acuan berbeda atau dataset berubah akan ditolak sebelum model dilatih. ZIP lama tidak mencatat seluruh revisi/runtime/parameter; verifikasi terbatas pada informasi yang tersedia dan tidak membuktikan semua lingkungan lama identik.

Fitur BGE frozen diambil dari ZIP yang sama setelah ID/dimensi/finite diperiksa. Ini menjaga input BGE identik dan menghindari ekstraksi ulang 300 jawaban. RoBERTa/BiGRU/proyeksi/MLP dilatih dari awal mengikuti {base}; tidak memakai bobot final lama. BGE masih dipakai di cabang fusion. Tidak ada ListNet, cosine, S1/S2, loop multi-seed atau kontrol tambahan.

**Output lima Excel:** hasil test dan history refit baru, CV acuan diberi label **dipakai ulang**. Excel05 → **Jadwal LR Refit** menunjukkan LR aktual setiap kelompok pada akhir epoch. **Bobot Scalar Mix** hanya checkpoint refit baru; bobot lima checkpoint CV lama dipisahkan. OOF acuan hanya disalin jika tersedia pada ZIP; tidak membuat OOF baru dari model yang sudah melatih seluruh development.

Bug ringkasan kosong diperbaiki dengan mengikat file sumber pada generator laporan. Sebelum ZIP dibuat, pemeriksaan wajib memastikan tabel test/CV terisi, prediksi sesuai split, metrik test cocok saat dihitung ulang, history/bobot/LR lengkap. Unduhan dihentikan bila gagal. Pengujian teknis tidak menjadi hasil penelitian atau jaminan bebas seluruh kemungkinan bug.

Ini menguji konsistensi optimisasi CV→refit, bukan menambah nilai CV baru. Rujukan stabilitas fine-tuning: [Mosbach dkk. (2021)](https://arxiv.org/abs/2006.04884). Horizon25 adalah hipotesis metode proyek, bukan optimum AES dari paper. Test historis yang sudah berulang kali dilihat tetap eksploratif. Belum ada hasil penelitian {e}.

Nama model, input dan hyperparameter terlihat per section Python biasa. Simpan ZIP hasil sebelum runtime dihapus. Folder dan checkpoint {e} terpisah dari acuan.
'''


def build_one(spec):
    e=f"E{spec['number']}";role_key='aes'+e[1:]+'_role';old_role='aes'+spec['base'][1:]+'_role'
    name=f"AES_300_{e}_{spec['base']}_S3_Refit{spec['refit']}_Schedule25_Native_Learnable_GPU"
    path='notebooks/experiments/'+spec['base_name']+'.ipynb'
    raw=subprocess.check_output(['git','show',spec['pin']+':'+path],cwd=ROOT)
    n=copy.deepcopy(json.loads(raw));old_hash=n['metadata']['aes_standalone']['code_sha256']
    for c in n['cells']:
        role=c['metadata'].pop(old_role,'')
        if role:c['metadata'][role_key]=role
        s=c['source'] if isinstance(c['source'],str) else ''.join(c['source'])
        s=s.replace(old_hash,PLACEHOLDER).replace(spec['base_name'],'__REFIT_NAME__')
        s=s.replace(spec['base'],e).replace('__REFIT_NAME__',name)
        if role=='imports':
            s=s.replace('from collections import Counter','from collections import Counter\nfrom io import BytesIO')
            s+=f"\n# Identitas training tetap; pembaruan rincian laporan tidak memulai training ulang.\nTRAINING_CODE_SHA256 = '{TRAINING_HASH[spec['number']]}'\n"
        if role=='dataset_definitions':
            s=replace_once(s,'code_hash = NOTEBOOK_CODE_SHA256','code_hash = TRAINING_CODE_SHA256')
        if role=='hyperparameters':
            s+=f'''\n# REUSE CV DAN REFIT — satu training baru, bukan lima fold baru.
CV_SOURCE_FILE = os.environ.get('AES300_CV_ZIP', '')
CV_SOURCE_EXPERIMENT = '{spec['base']}'
CV_SOURCE_ZIP_NAME = 'Hasil_{spec['base_name']}.zip'
CV_SOURCE_SHA256 = '{spec['zip_sha']}'
EXPECTED_REFIT_EPOCHS = {spec['refit']}
REFIT_SCHEDULER_EPOCHS = EPOCHS  # Horizon mengikuti 25 epoch pada CV acuan.
'''
        if role=='storage_and_upload':
            # Resolve helper is the only function needed before dataset helpers/metrics.
            helper=(ROOT/'scripts/refit_cv_reuse_native.py').read_text()
            tree=ast.parse(helper);fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='resolve_cv_archive')
            s+='\n'+ast.get_source_segment(helper,fn)+'\n'
            s+="\nCV_SOURCE_PATH = resolve_cv_archive(ROOT, CV_SOURCE_FILE, CV_SOURCE_ZIP_NAME, browser_upload=IN_COLAB)\n"
        if role=='configuration_defaults':
            s=s.replace("    's3_fusion': 'concat',", "    'cv_source_path': '', 'cv_source_experiment': CV_SOURCE_EXPERIMENT,\n    'cv_archive_sha256': CV_SOURCE_SHA256, 'expected_refit_epochs': EXPECTED_REFIT_EPOCHS,\n    'refit_scheduler_epochs': REFIT_SCHEDULER_EPOCHS,\n    's3_fusion': 'concat',")
            s=replace_once(s,'    return result\n', '''    if type(result['expected_refit_epochs']) is not int or not 1<=result['expected_refit_epochs']<=result['epochs']:
        raise ValueError('expected_refit_epochs harus positif dan tidak melebihi horizon CV.')
    if result['refit_scheduler_epochs']!=result['epochs']:
        raise ValueError('refit_scheduler_epochs harus mengikuti horizon epoch CV acuan.')
    if not re.fullmatch(r'[0-9a-f]{64}',result['cv_archive_sha256']):
        raise ValueError('SHA256 ZIP CV acuan tidak sah.')
    return result
''')
        if role=='configuration_apply':
            s=s.replace("    input_format='single_sequence', device=DEVICE,", "    cv_source_path=str(CV_SOURCE_PATH), cv_source_experiment=CV_SOURCE_EXPERIMENT,\n    cv_archive_sha256=CV_SOURCE_SHA256, expected_refit_epochs=EXPECTED_REFIT_EPOCHS,\n    refit_scheduler_epochs=REFIT_SCHEDULER_EPOCHS,\n    input_format='single_sequence', device=DEVICE,")
            s=s.replace('e43_native_masked_mean_v1',e.lower()+'_native_refit_schedule25_v1').replace('e45_native_scalar_mix_all12_v1',e.lower()+'_native_refit_schedule25_v1')
            s+=f"\nprint('CV dipakai ulang dari {spec['base']}; refit baru:', cfg['expected_refit_epochs'], 'epoch | horizon LR:', cfg['refit_scheduler_epochs'])\n"
        if role=='bge_execute':
            s=s.replace('bge=bge_features(bundle,cfg,output)','bge=load_reused_bge(bundle,cfg,output)')
            s=s.replace('BGE dense frozen; semua koordinat','BGE frozen dari ZIP acuan terverifikasi; semua koordinat')
        if role=='definition:fit':
            anchor='    seed_everything(seed)\n'
            s=replace_once(s,anchor,'''    if validation is not None:
        raise ValueError('Notebook ini hanya refit; lima CV lama dipakai ulang, tidak dilatih lagi.')
    if epochs!=cfg['expected_refit_epochs']:
        raise ValueError('Jumlah epoch refit berbeda dari median CV terverifikasi.')
    seed_everything(seed)
''')
            s=replace_once(s,'    total_steps = len(loader)*epochs\n', '''    # Tetap horizon CV25, meskipun refit berhenti pada median18/7.
    total_steps = len(loader)*cfg['refit_scheduler_epochs']
''')
            s=s.replace("'micro_batch': cfg['micro_batch']}", "'micro_batch': cfg['micro_batch'],\n               'scheduled_learning_rates_end': {g['name']:g['lr'] for g in optimizer.param_groups}}")
            s=s.replace("    info['scalar_mix_layers'] = cfg['scalar_mix_layers']", "    info['scalar_mix_layers'] = cfg['scalar_mix_layers']\n    info['scheduler_horizon_epochs'] = cfg['refit_scheduler_epochs']\n    info['scheduler_total_steps'] = total_steps\n    info['warmup_steps'] = warmup")
        if role=='definition:load_model':
            s=s.replace("for key in ('roberta',", "for key in ('refit_scheduler_epochs', 'expected_refit_epochs', 'cv_archive_sha256', 'cv_source_experiment', 'roberta',")
        if role=='definition:train_all':
            s=s.replace('output, observer=None):','output, observer=None, reused_cv=None):')
            s=s.replace('    output = Path(output)\n', '''    if reused_cv is None or reused_cv['dataset_sha256']!=bundle.source_sha256 or reused_cv['split_sha256']!=manifest['split_sha256'] or reused_cv['archive_sha256']!=cfg['cv_archive_sha256']:
        raise ValueError('CV acuan belum terverifikasi atau berbeda data/split/hash.')
    if digest(cfg['cv_source_path'])!=cfg['cv_archive_sha256']:
        raise ValueError('ZIP CV berubah setelah verifikasi.')
    output = Path(output)
''',1)
            begin=s.index('    cv_rows, selected_epochs, cv_folds =')
            end=s.index('    pd.DataFrame(cv_rows).to_csv',begin)
            s=s[:begin]+'''    # Lima metrik fold ini adalah HASIL LAMA; tidak ada pemanggilan fit CV.
    cv_rows = reused_cv['cv_rows']
    selected_epochs = {'S3':reused_cv['selected_epoch']}
    print('CV lama terverifikasi dipakai ulang; 0 fit CV baru. Refit dimulai dari bobot pretrained.',flush=True)
'''+s[end:]
            s=s.replace("'method': 'ceil median best validation-RMSE epoch across development folds',", "'method': 'ceil median best epoch from verified reused CV',\n                 'cv_source_experiment':cfg['cv_source_experiment'], 'cv_archive_sha256':cfg['cv_archive_sha256'],\n                 'new_cv_fits':0, 'refit_scheduler_epochs':cfg['refit_scheduler_epochs'],")
            s=s.replace("print('Validasi silang semua skenario selesai. Berikutnya pelatihan model final pada seluruh data latih; '", "print('Lima fold CV acuan dipakai ulang. Berikutnya satu refit pada seluruh development; '")
            s=s.replace("'cv': cv_rows, 'four_trait_labels_available': False}", "'cv': cv_rows, 'four_trait_labels_available': False,\n              'new_cv_fits':0, 'cv_folds_reused':5, 'cv_source_experiment':cfg['cv_source_experiment'],\n              'cv_archive_sha256':cfg['cv_archive_sha256'], 'refit_scheduler_epochs':cfg['refit_scheduler_epochs']}")
        if role=='definition:export_training_event':
            old="'Bobot_scalar_mix': pd.DataFrame(result['scalar_mix_weights'])}"
            new="""'Bobot_scalar_mix': pd.DataFrame(result['scalar_mix_weights']),
                                            'Jadwal_LR': pd.DataFrame([
                                                {'Epoch':r['epoch'],'Kelompok':group,'LR Puncak':result['optimizer_learning_rates'][group],
                                                 'LR Akhir Epoch':rate,'Optimizer Steps':r['optimizer_steps'],
                                                 'Horizon Jadwal (epoch)':result['scheduler_horizon_epochs'],
                                                 'Total Steps Jadwal':result['scheduler_total_steps'],'Warmup Steps':result['warmup_steps']}
                                                for r in result['history'] for group,rate in r['scheduled_learning_rates_end'].items()])}"""
            s=replace_once(s,old,new)
        if role=='training_execute':
            s=s.replace('observer=on_training_stage)','observer=on_training_stage,reused_cv=reused_cv)')
            s=s.replace("'Hasil validasi silang (lima fold)'","'CV acuan dipakai ulang (bukan lima fit baru)'")
            s=replace_once(s,"    show(comparison,'Nilai guru, prediksi mentah, dan pembulatan pada data uji',rows=len(comparison),columns=8)",
'''    prediction_rows=[{'ID Jawaban':friendly_value('row_id',r['row_id']),
                      'Nilai Guru':r['skor_guru'],'Prediksi S3':r['Prediksi S3'],
                      'Pembulatan S3':r['Pembulatan S3']} for r in comparison.to_dict('records')]
    prediction_tables=prediction_report_tables(prediction_rows,cfg['score_min'],cfg['score_max'])
    for title,data in prediction_tables.items():
        frame=pd.DataFrame(data)
        show(frame,title+' — TEST',rows=len(frame),columns=len(frame.columns))''')
        if role=='definition:evaluation_report':
            s=s.replace('def final_rows():','def final_rows(file=file):').replace('def summaries():','def summaries(file=file):')
            s=replace_once(s,"        sheets['Nilai Guru dan Prediksi']=Table(['ID Jawaban','Nilai Guru','Prediksi S3','Pembulatan S3','Selisih S3'],per_answer())",'''        parameters={r['name']:r['value'] for r in source.records('01_konfigurasi.xlsx','Parameter')}
        original_columns=['ID Jawaban','Nilai Guru','Prediksi S3','Pembulatan S3','Selisih S3']
        original_rows=[dict(zip(original_columns,r)) for r in per_answer()]
        detailed=prediction_report_tables(original_rows,parameters['score_min'],parameters['score_max'])
        for title,records in detailed.items():
            headers=list(records[0])
            sheets[title]=Table(headers,[[r[key] for key in headers] for r in records])''')
            s=s.replace("['Ringkasan Validasi','", "['Asal CV','Lima fold diimpor dari ZIP acuan terverifikasi. Tidak ada training/evaluasi CV baru.'],\n        ['Ringkasan Validasi','")
            s=s.replace("['Asal CV','", "['Rincian selisih','Selisih Tepat =n; Toleransi Selisih ≤n. Normalisasi 0–1 berbeda dari pembulatan: 1 poin = 0,01. Hanya data test; nilai mentah tidak diubah untuk metrik.'],\n        ['Asal CV','")
            # E47 inherits the old OOF export loop; its cv-fold filenames do not occur here.
            addition="""    old = '09_CV_acuan_dipakai_ulang.xlsx'
    sheets['Provenansi CV'] = source.copy(old,'Provenansi')
    if source.has(old,'OOF_acuan'):
        sheets['OOF Acuan (dipakai ulang)'] = source.copy(old,'OOF_acuan')
"""
            s=replace_once(s,'    return sheets\n',addition+'    return sheets\n')
        if role=='definition:training_report':
            s=s.replace('Riwayat latihan per skenario. Lima fold digabung ke satu sheet untuk setiap model.',
                        'Fusion (S3) berisi REFIT BARU. Lima fold CV lama ada pada sheet CV Acuan (dipakai ulang).')
            s=s.replace('    if len(sheets)==1:return None\n','')
            s=s.replace("settings={'scalar_mix_gamma_init'", "settings={'refit_scheduler_epochs':'Horizon LR refit (epoch)','expected_refit_epochs':'Epoch refit dikunci','cv_source_experiment':'Eksperimen sumber CV','cv_archive_sha256':'SHA256 sumber CV','scalar_mix_gamma_init'")
            s=replace_once(s,'    return sheets\n',REPORT_ADDITIONS+'    return sheets\n')
        if role=='definition:build_reports':
            s=s.replace('destination,verbose=True):','destination,verbose=True,bundle=None,manifest=None,cfg=None):')
            s=s.replace("    archive=destination/", "    if bundle is None or manifest is None or cfg is None:\n        raise RuntimeError('Konteks data/split/config diperlukan untuk memeriksa hasil sebelum download.')\n    verification=validate_refit_download(destination,bundle,manifest,cfg)\n    archive=destination/")
            s=s.replace("'metrics_recomputed':False}", "'metrics_recomputed':False,'download_verification':verification}")
        if role=='download_execute':
            s=s.replace("output/'hasil_ringkas',verbose=False)","output/'hasil_ringkas',verbose=False,bundle=bundle,manifest=manifest,cfg=cfg)")
            s=s.replace('Seluruh fold dan epoch digabung per model.','Refit baru, CV acuan terpisah, provenance dan jejak LR.')
        if c['cell_type']=='markdown' and '## Rujukan dan batas kesimpulan' in s:
            s+=f'\n**Diagnostik {e}:** jadwal LR refit memakai horizon25 yang sama dengan CV lama. [Mosbach, Andriushchenko & Klakow (2021)](https://arxiv.org/abs/2006.04884) membahas stabilitas optimisasi BERT/RoBERTa; bukan bukti horizon25 optimal untuk AES. CV lama tidak menjadi skor CV baru; test historis tetap eksploratif.\n'
        c['source']=s
    # Move pure metric functions early so verification finishes before any encoder runs.
    moved=[c for c in n['cells'] if c['metadata'].get(role_key) in ['definition:gwet_ac2','definition:metrics']]
    n['cells']=[c for c in n['cells'] if c not in moved]
    helper_source=(ROOT/'scripts/refit_cv_reuse_native.py').read_text()
    helper_tree=ast.parse(helper_source)
    helpers='\n\n'.join(ast.get_source_segment(helper_source,x) for x in helper_tree.body
                        if isinstance(x,ast.FunctionDef) and x.name!='resolve_cv_archive')+'\n'
    index=next(i for i,c in enumerate(n['cells']) if c['metadata'].get(role_key)=='split_execute')+1
    n['cells'][index:index]=[
        {'cell_type':'markdown','metadata':{},'source':'## Verifikasi CV acuan sebelum model dijalankan\n\nFungsi metrik dipakai hanya untuk audit OOF jika tersedia. Lima fold tidak dilatih ulang.\n'},
        *moved,
        {'cell_type':'code','metadata':{role_key:'definition:cv_reuse_helpers'},'source':helpers,'outputs':[],'execution_count':None},
        {'cell_type':'code','metadata':{role_key:'verify_cv_execute'},'source':'''reused_cv = verify_reused_cv(CV_SOURCE_PATH,bundle,manifest,cfg)
dump(output/'cv_reuse_verified.json',reused_cv)
export_reused_cv(reused_cv,journal)
print('CV acuan lolos verifikasi:',reused_cv['base_experiment'],'| 5 fold lama | 0 fit baru | refit',reused_cv['selected_epoch'],'epoch')
show(pd.DataFrame(reused_cv['cv_rows']),'CV acuan yang dipakai ulang',rows=5)
''','outputs':[],'execution_count':None}]
    prediction_source=(ROOT/'scripts/prediction_report_native.py').read_text()
    prediction_tree=ast.parse(prediction_source)
    prediction_helpers='\n\n'.join(ast.get_source_segment(prediction_source,x) for x in prediction_tree.body
                                 if isinstance(x,ast.FunctionDef) and x.name!='update_prediction_zip')+'\n'
    index=next(i for i,c in enumerate(n['cells']) if c['metadata'].get(role_key)=='definition:cv_reuse_helpers')+1
    n['cells'][index:index]=[
        {'cell_type':'markdown','metadata':{},'source':'## Rincian prediksi dan selisih — pelaporan saja\n\nSkor mentah 0–100, salinan normalisasi 0–1, dan pembulatan ditampilkan terpisah. Kategori tepat 0/1/2/3/4/5/10 berbeda dari toleransi ≤n. Fungsi ini tidak menjalankan model atau mengubah training.\n'},
        {'cell_type':'code','metadata':{role_key:'definition:prediction_report_helpers'},'source':prediction_helpers,'outputs':[],'execution_count':None}]
    n['cells'][0]['source']=intro(spec,name)+'\n**Rincian prediksi lengkap:** Excel04 → Nilai Guru dan Prediksi (mentah, normalisasi, pembulatan); Selisih Tepat (=0/1/2/3/4/5/10 poin); Toleransi Selisih (≤0/1/2/3/4/5/10). Jumlah dan persentase hanya test. Perubahan ini khusus laporan; identitas training/checkpoint versi awal E46/E47 dipertahankan.\n'
    fp=hashlib.sha256('\n'.join(c['source'] for c in n['cells'] if c['cell_type']=='code').encode()).hexdigest()
    for i,c in enumerate(n['cells']):
        c['source']=c['source'].replace(PLACEHOLDER,fp);c['id']=e.lower()+f'-{i:03d}-'+hashlib.sha256(c['source'].encode()).hexdigest()[:8]
        if c['cell_type']=='code':ast.parse(c['source']);c['outputs']=[];c['execution_count']=None
    n['metadata']['colab']['name']=name+'.ipynb';n['metadata']['aes_standalone']['code_sha256']=fp
    n['metadata']['aes_experiment'].update(name=e,format_commit=spec['pin'],format_path=path,format_sha256=hashlib.sha256(raw).hexdigest(),
        s3=spec['base']+' model; verified reused CV; one aligned-schedule refit',cv_fits=0,cv_fits_reused=5,refit_only=True,
        refit_epochs=spec['refit'],scheduler_horizon_epochs=25,cv_source_experiment=spec['base'],cv_source_zip_sha256=spec['zip_sha'],
        loss='MSE',ranking=False,new_cv_results=False,
        treatment='Keep parent model/selected epochs; align refit scheduler horizon25 with reused CV; no new CV fits',
        download_validation_required=True,export_summary_closure_fixed=True,prediction_breakdown=True,
        reporting_only_update=True,training_code_sha256=TRAINING_HASH[spec['number']])
    text=json.dumps(n,ensure_ascii=False,indent=1)+'\n'
    for folder in ['notebooks/experiments','aes-learning/notebooks/experiments']:(ROOT/folder/(name+'.ipynb')).write_text(text)
    print(json.dumps({'name':name,'cells':len(n['cells']),'code_sha256':fp}))
    return n


if __name__=='__main__':
    for spec in SPECS:build_one(spec)
