"""E48: pinned E43 architecture/protocol, training-only LDS-weighted MSE."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
from build_e45_notebook import OOF_EXPORT

ROOT=Path(__file__).resolve().parents[1]
PIN='7241a21dfec37c942948b754da3960fba4c7731c'
BASE='notebooks/experiments/AES_300_E43_E42_S3_MaskedMeanPooling_Native_Learnable_GPU.ipynb'
NAME='AES_300_E48_E43_S3_LDS_WeightedMSE_Native_Learnable_GPU'
PLACEHOLDER='__NOTEBOOK_CODE_SHA256__'

INTRO='''# E48 · E43 + LDS-weighted MSE · S3 Native Learnable

**Upload cukup `fix_dataset_300.xlsx`.** Tidak membutuhkan ZIP eksperimen lama. Pilih GPU T4 → Run all. Langsung S3: satu 5-fold CV baru, maksimal25epoch/patience5; satu refit dari ceil median best epoch CV, lalu satu test. Satu konstanta seed42, tanpa loop multi-seed/kontrol ulang.

**Satu perlakuan:** loss MSE menjadi LDS-weighted MSE. Arsitektur/input/tokenisasi/split/optimizer/LR/scheduler mengikuti E43 asli: tiga teks → RoBERTa fine-tuned → ScalarMix4+gamma → BiGRU64/arah + co-attention + masked mean → concat RoBERTa768+BGEproyeksi256 → MLP → satu skor. BGE encoder frozen. Bukan memakai horizon refit tetap25 dari E46/E47; refit mengikuti jadwal asli E43 dengan durasi pilihan CV.

| Bagian | Nilai |
|---|---|
| X / y | `soal`, `jawaban_referensi`, `response_text` / `rater`0–100 |
| LR encoder / komponen lain | 5e-6 / 1e-4 |
| LDS | Bin1poin; Gaussian5, sigma2poin; inverse frekuensi setelah smoothing |
| Bobot loss | Rata-rata bobot seluruh training split=1; tidak dinormalisasi ulang per microbatch |
| Target loss | Skor0–1, label guru asli tetap utuh |
| Seleksi checkpoint | RMSE validasi biasa, tanpa bobot LDS |
| Evaluasi | RMSE/Pearson dari mentah; QWK/AC2 dari kategori; semua tanpa bobot LDS |

LDS menghitung distribusi label **hanya pada train fold**. Validasi tidak ikut menghitung bobot. Refit menghitung ulang distribusi dari240development; test tidak pernah menentukan bobot. Bobot memengaruhi `backward()`/optimizer, bukan menambah angka pada prediksi. Saat inferensi tidak memerlukan rater atau bobot LDS.

**Rujukan:** Yang dkk. (ICML2021), [Delving into Deep Imbalanced Regression](https://arxiv.org/abs/2102.09554), [implementasi penulis STS-B-DIR](https://github.com/YyzHarry/imbalanced-regression/tree/main/sts-b-dir). LDS+inverse/Gaussian5/sigma2 tersedia pada contoh penulis untuk regresi skor teks. Bin1poin adalah adaptasi skala AES0–100. Normalisasi rata-rata bobot1 mengendalikan skala loss. Referensi tidak membuktikan kombinasi ini optimal untuk AES Indonesia. Belum ada hasil penelitian E48; target60/60 dalam±3atau±5 adalah sasaran, bukan jaminan.

**Output lima Excel:** 04→Nilai Guru dan Prediksi (mentah, normalisasi0–1, pembulatan), Selisih Tepat (=0/1/2/3/4/5/10), Toleransi Selisih (≤n), Prediksi Validasi OOF (240development terpisah dari60test). 05→Fusion(S3) mencatat MSE biasa dan LDS-MSE, Bobot LDS Training, Distribusi LDS Training, Audit LDS, Bobot Scalar Mix, serta MAE dan Toleransi CV. Rekap ≤3/≤5 dan kesalahan terbesar tersedia dari prediksi validasi. Normalisasi tidak sama dengan pembulatan.

Ringkasan ekspor memakai binding sumber yang tetap untuk menghindari bug E44/E45. Sebelum ZIP dibuat, gate wajib membaca ulang Excel dan memeriksa tabel metrik/OOF, rekap selisih serta bobot training tiap fold. Source historis/hasil/HTML E43 tidak diubah. Checkpoint/folder E48 terpisah. Python biasa per section, tanpa magic/worker/cosine/S1/S2.
'''


def once(s,old,new):
    assert s.count(old)==1,old[:130]
    return s.replace(old,new,1)


def build():
    raw=subprocess.check_output(['git','show',PIN+':'+BASE],cwd=ROOT)
    n=copy.deepcopy(json.loads(raw));old_hash=n['metadata']['aes_standalone']['code_sha256']
    for c in n['cells']:
        role=c['metadata'].pop('aes43_role','')
        if role:c['metadata']['aes48_role']=role
        s=c['source'] if isinstance(c['source'],str) else ''.join(c['source'])
        s=s.replace(old_hash,PLACEHOLDER).replace(Path(BASE).stem,'__E48_NAME__').replace('E43','E48').replace('__E48_NAME__',NAME)
        if role=='hyperparameters':
            s+='\n# LDS — hanya label pada bagian training; bukan pengaturan encoder atau fusion.\nLDS_BIN_WIDTH = 1\nLDS_KERNEL_SIZE = 5\nLDS_SIGMA = 2.0\nLDS_REWEIGHT = "inverse"\n'
        if role=='configuration_defaults':
            s=s.replace("    's3_fusion': 'concat',","    'lds_bin_width':LDS_BIN_WIDTH, 'lds_kernel_size':LDS_KERNEL_SIZE,\n    'lds_sigma':LDS_SIGMA, 'lds_reweight':LDS_REWEIGHT,\n    's3_fusion': 'concat',")
            s=once(s,'    return result\n', '''    if result['lds_bin_width']!=1 or result['lds_reweight']!='inverse':
        raise ValueError('E48 mengunci LDS bin1poin dan inverse.')
    size=result['lds_kernel_size'];sigma=result['lds_sigma']
    if type(size) is not int or size<1 or size%2!=1 or size>result['score_max']-result['score_min']+1 or not math.isfinite(sigma) or sigma<=0:
        raise ValueError('Kernel LDS harus ganjil dan sigma positif finite.')
    return result
''')
        if role=='configuration_apply':
            s=s.replace('e43_native_masked_mean_v1','e48_native_lds_inverse_v1')
            s+='\nprint("Loss training: LDS-weighted MSE | evaluasi dan early stopping: RMSE biasa tanpa bobot")\n'
        if role=='definition:fit':
            s=once(s,'    seed_everything(seed)\n', '''    if validation is not None and set(training)&set(validation):
        raise ValueError('Train/validation tumpang tindih.')
    weights_by_index,lds_audit=lds_training_weights(bundle,training,cfg)
    print('LDS training-only:',lds_audit['training_count'],'jawaban | mean/min/max bobot:',
          lds_audit['weight_mean'],lds_audit['weight_min'],lds_audit['weight_max'],flush=True)
    seed_everything(seed)
''')
            s=once(s,"'gpu': torch.cuda.get_device_name() if use_cuda else None}","'gpu': torch.cuda.get_device_name() if use_cuda else None, 'lds_weights_sha256':lds_audit['weights_sha256']}")
            s=once(s,'        loss_total = 0.\n','        loss_total, raw_mse_total = 0., 0.\n')
            s=once(s,'                    loss = torch.mean((prediction-targets)**2)', '''                    weights=torch.tensor([weights_by_index[i] for i in ids],dtype=torch.float32,device=device).reshape(-1,1)
                    loss,raw_mse=lds_weighted_mse(prediction,targets,weights)''')
            s=s.replace('# MSE mengukur selisih kuadrat antara prediksi dan target; optimizer berusaha memperkecilnya.',
                        '# Bobot LDS sudah mean1 pada training split. Jangan dibagi jumlah bobot microbatch (microbatch1 akan meniadakan LDS).')
            s=once(s,'                loss_total += loss.item()*len(ids)\n','                loss_total += loss.item()*len(ids)\n                raw_mse_total += raw_mse.item()*len(ids)\n')
            s=once(s,"'train_mse_scaled': loss_total/len(training),","'train_mse_scaled': raw_mse_total/len(training), 'train_lds_mse_scaled':loss_total/len(training),")
            s=s.replace('loss latih (MSE, target 0–1): {row["train_mse_scaled"]:.6f}',
                        'loss LDS-MSE: {row["train_lds_mse_scaled"]:.6f} | MSE biasa: {row["train_mse_scaled"]:.6f}')
            s=once(s,"    info['scalar_mix_layers'] = cfg['scalar_mix_layers']","    info['scalar_mix_layers'] = cfg['scalar_mix_layers']\n    info['loss_name']='LDS-weighted MSE'\n    info['lds']=lds_audit")
        if role=='definition:load_model':
            s=s.replace("for key in ('roberta',","for key in ('lds_bin_width','lds_kernel_size','lds_sigma','lds_reweight','roberta',")
        if role=='definition:export_training_event':
            s=once(s,"'Bobot_scalar_mix': pd.DataFrame(training['scalar_mix_weights']),","'Bobot_scalar_mix': pd.DataFrame(training['scalar_mix_weights']),\n                                                           **lds_journal_tables(training),")
            s=once(s,"'Bobot_scalar_mix': pd.DataFrame(result['scalar_mix_weights'])}","'Bobot_scalar_mix': pd.DataFrame(result['scalar_mix_weights']),\n                                            **lds_journal_tables(result)}")
        if role=='definition:training_report':
            s=once(s,"row.get('epoch'),row.get('train_mse_scaled'),","row.get('epoch'),row.get('train_mse_scaled'),row.get('train_lds_mse_scaled'),")
            s=once(s,"'Loss Training (MSE)','RMSE Validasi (poin)'","'Loss Training (MSE)','Loss Training (LDS-MSE)','RMSE Validasi (poin)'")
            s=s.replace("settings={'scalar_mix_gamma_init'","settings={'lds_bin_width':'LDS bin (poin)','lds_kernel_size':'LDS kernel Gaussian','lds_sigma':'LDS sigma (poin)','lds_reweight':'LDS pembobotan','scalar_mix_gamma_init'")
            s=s.replace('int((errors<=1).sum()), int((errors<=5).sum()), int((errors<=10).sum())',
                        'int((errors<=1).sum()), int((errors<=3).sum()), int((errors<=5).sum()), int((errors<=10).sum()), float(errors.max())')
            s=s.replace("'Dalam ±1 poin','Dalam ±5 poin','Dalam ±10 poin'","'Dalam ±1 poin','Dalam ±3 poin','Dalam ±5 poin','Dalam ±10 poin','Selisih Maksimal (poin)'")
            s=once(s,'    return sheets\n','    sheets.update(lds_report_tables(source))\n    return sheets\n')
            s=s.replace("['Loss Training (MSE)','", "['Loss Training (LDS-MSE)','Objective training dengan bobot LDS, target0–1; MSE biasa dicatat terpisah. Bobot hanya berasal dari training fold.'],\n        ['Loss Training (MSE)','")
        if role=='definition:evaluation_report':
            s=s.replace('def final_rows():','def final_rows(file=file):').replace('def summaries():','def summaries(file=file):')
            old="        sheets['Nilai Guru dan Prediksi']=Table(['ID Jawaban','Nilai Guru','Prediksi S3','Pembulatan S3','Selisih S3'],per_answer())"
            new='''        parameters={r['name']:r['value'] for r in source.records('01_konfigurasi.xlsx','Parameter')}
        keys=['ID Jawaban','Nilai Guru','Prediksi S3','Pembulatan S3','Selisih S3']
        detailed=prediction_report_tables([dict(zip(keys,r)) for r in per_answer()],parameters['score_min'],parameters['score_max'])
        for title,rows in detailed.items():
            headers=list(rows[0]);sheets[title]=Table(headers,[[r[k] for k in headers] for r in rows])
        errors=np.asarray([r['Selisih S3'] for r in detailed['Nilai Guru dan Prediksi']])
        sheets['Ringkasan Error Test']=Table(['Jumlah Test','MSE (poin²)','MAE (poin)','Selisih Maksimal (poin)'],
            [[len(errors),float(np.mean(errors**2)),float(np.mean(abs(errors))),float(np.max(abs(errors)))]])'''
            s=once(s,old,new)
            s=once(s,'    return sheets\n',OOF_EXPORT+'    return sheets\n')
            s=s.replace("['Ringkasan Validasi','","['LDS','Bobot loss hanya saat training; semua metrik/rekap test dan validasi tanpa bobot LDS.'],\n        ['Ringkasan Validasi','")
        if role=='definition:build_reports':
            s=s.replace('destination,verbose=True):','destination,verbose=True,bundle=None,manifest=None,cfg=None):')
            s=once(s,'    archive=destination/',"    if bundle is None or manifest is None or cfg is None:\n        raise RuntimeError('Konteks data/split/config wajib untuk memeriksa download.')\n    verification=validate_e48_download(destination,bundle,manifest,cfg)\n    archive=destination/")
            s=s.replace("'metrics_recomputed':False}","'metrics_recomputed':False,'download_verification':verification}")
        if role=='download_execute':
            s=s.replace("output/'hasil_ringkas',verbose=False)","output/'hasil_ringkas',verbose=False,bundle=bundle,manifest=manifest,cfg=cfg)")
        if role=='training_execute':
            s=once(s,"    show(comparison,'Nilai guru, prediksi mentah, dan pembulatan pada data uji',rows=len(comparison),columns=8)",'''    rows=[{'ID Jawaban':friendly_value('row_id',r['row_id']),'Nilai Guru':r['skor_guru'],
           'Prediksi S3':r['Prediksi S3'],'Pembulatan S3':r['Pembulatan S3']} for r in comparison.to_dict('records')]
    for title,data in prediction_report_tables(rows,cfg['score_min'],cfg['score_max']).items():
        frame=pd.DataFrame(data);show(frame,title+' — TEST',rows=len(frame),columns=len(frame.columns))''')
        c['source']=s
    helper=(ROOT/'scripts/lds_native.py').read_text()
    pred=(ROOT/'scripts/prediction_report_native.py').read_text()
    helpers='\n\n'.join(ast.get_source_segment(source,x) for source in [helper,pred]
        for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name!='update_prediction_zip')+'\n'
    index=next(i for i,c in enumerate(n['cells']) if c['metadata'].get('aes48_role')=='definition:fit')
    n['cells'][index:index]=[
        {'cell_type':'markdown','metadata':{},'source':'## LDS: distribusi label training → bobot loss\n\nFungsi terlihat di bawah. Train fold saja menentukan bobot. Tidak ada normalisasi bobot per microbatch. Helper laporan hanya membaca hasil tersimpan.\n'},
        {'cell_type':'code','metadata':{'aes48_role':'definition:lds_helpers'},'source':helpers,'outputs':[],'execution_count':None}]
    n['cells'][0]['source']=INTRO
    fp=hashlib.sha256('\n'.join(c['source'] for c in n['cells'] if c['cell_type']=='code').encode()).hexdigest()
    for i,c in enumerate(n['cells']):
        c['source']=c['source'].replace(PLACEHOLDER,fp);c['id']=f'e48-{i:03d}-'+hashlib.sha256(c['source'].encode()).hexdigest()[:8]
        if c['cell_type']=='code':ast.parse(c['source']);c['outputs']=[];c['execution_count']=None
    n['metadata']['colab']['name']=NAME+'.ipynb';n['metadata']['aes_standalone']['code_sha256']=fp
    m=n['metadata']['aes_experiment']
    m.update(name='E48',format_commit=PIN,format_path=BASE,format_sha256=hashlib.sha256(raw).hexdigest(),
        loss='LDS-weighted MSE',ranking=False,lds_training_only=True,lds_bin_width=1,lds_kernel_size=5,lds_sigma=2.,
        lds_reweight='inverse',download_validation_required=True,export_summary_closure_fixed=True,
        treatment='E43 unchanged architecture/protocol; training-only LDS inverse-weighted MSE',
        prediction_breakdown=True,oof_predictions=True,requires_previous_zip=False)
    raw=json.dumps(n,ensure_ascii=False,indent=1)+'\n'
    for folder in ['notebooks/experiments','aes-learning/notebooks/experiments']:(ROOT/folder/(NAME+'.ipynb')).write_text(raw)
    print(json.dumps({'name':NAME,'cells':len(n['cells']),'code_sha256':fp}))
    return n


if __name__=='__main__':build()
