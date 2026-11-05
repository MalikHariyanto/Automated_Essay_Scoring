"""E49 from pinned E48: only LDS inverse exponent 1 -> 0.5; visible native code."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
PIN='437b6ff15808c2122a92cfc32874c815d33af633'
BASE='notebooks/experiments/AES_300_E48_E43_S3_LDS_WeightedMSE_Native_Learnable_GPU.ipynb'
NAME='AES_300_E49_E43_S3_LDS_InverseSqrt_Native_Learnable_GPU'
PLACEHOLDER='__NOTEBOOK_CODE_SHA256__'

INTRO='''# E49 · E43 + LDS inverse akar · S3 Native Learnable

**Cukup upload `fix_dataset_300.xlsx` → GPU T4 → Run all.** Tidak memerlukan ZIP hasil lama. S3 saja; satu5foldCVbaru, maksimal25epoch/patience5; satu refit dari ceil median best epochCV, lalu satu test. Satu konstanta seed42, tanpa loop multi-seed atau kontrol ulang.

**Satu perubahan dari E48:** kekuatan inverse frekuensi setelah LDS dikurangi dari pangkat1 menjadi0,5. `a_i = frekuensi_efektif(skor_i)**(-0.5)`, kemudian `w_i = a_i / mean(a seluruh training split)`. Tidak menambah cap, tidak mengubah Gaussian5/sigma2/bin1poin, sampler, learning rate, dropout, atau model. Parameter terlihat pada section hyperparameter: `LDS_POWER=0.5`.

Arsitektur E43 tetap: tiga teks → RoBERTa fine-tuned → ScalarMix4+gamma → BiGRU64/arah + co-attention dua arah/matching + masked mean → concat RoBERTa768+BGEproyeksi256 → MLP → satu nilai. BGE encoder frozen. LRencoder5e-6,seluruh komponen lain1e-4. Scheduler/refit mengikuti E43 asli, bukan horizon refit tetap25 dari E46/E47.

LDS dihitung **hanya dari label train fold** (~192jawaban). Saat refit dihitung ulang240development. Validation/test tidak menentukan bobot. Label guru asli tidak diganti. Histogram menggunakan0–100; target loss0–1. Objective=`mean(w_i*(prediksi_i-target_i)**2)`. **Bobot tidak dinormalisasi ulang per microbatch1**, agar tidak membatalkan pembobotan. MSE biasa dan weightedMSE dicatat terpisah. Earlystop dan semua metrik validasi/test tetap tanpa bobot LDS. Inferensi tidak membutuhkan rater/bobot LDS.

**Alasan:** E48 full inverse memberi satu jawaban training berskor65 bobot27,7485; pada distribusi training yang sama, E49 diperkirakan5,7106. Ini perhitungan bobot, bukan estimasi hasil model. MeanRMSECV E48=6,3908/test6,4777, sedangkan acuanE43=5,8876/5,1265. E49 belum mempunyai hasil penelitian; belum menjamin60/60dalam±3atau±5poin.

**Rujukan:** Yangdkk.(ICML2021), [Delving into Deep Imbalanced Regression](https://arxiv.org/abs/2102.09554), [kode penulis](https://github.com/YyzHarry/imbalanced-regression). Inverse dan inverse-square-root adalah pilihan pembobotan di repositori tersebut. **Rancangan E49 mengambil akar SETELAH smoothing**, sebagai adaptasi satu perubahan dariE48; beberapa pipeline resmi mengambil akar histogram SEBELUM smoothing. Keduanya bukan rumus identik. Bin1poin adalah adaptasi skala AES. Paper tidak membuktikan resep ini optimum untuk AES Indonesia.

**Output lima Excel:** 04→prediksi test mentah/normalisasi/pembulatan, Selisih Tepat0/1/2/3/4/5/10danlainnya, Toleransi≤n, MSE/MAE/error maksimum, Prediksi Validasi OOF240terpisahdari60test. 05→historyMSEdanLDS-MSE, Bobot LDS Training1200baris, Distribusi LDS Training606baris, Audit LDS dengan`weight_power=0.5`, Bobot ScalarMix24baris, MAE/toleransi±3/±5 CV. Mean bobot perfit1, min/max dan fingerprint disimpan. Normalisasi berbeda dari pembulatan.

Gate wajib membaca ulang Excel sebelum ZIP: ringkasan nonempty,metrik tanpa bobot cocok dengan test/OOF,ID/split/label benar,rekap selisih benar,bobot LDS dari training fold cocok. Bobot E48 tidak boleh lolos sebagai E49. Folder/lesson_version/hash/checkpoint baru; resume memeriksa config, fingerprint bobot+power dan optimizer/scheduler/scaler/RNG/loader. E43/E48/HTML lama tidak diubah. Python biasa persection,tanpamagic/worker/cosine/S1/S2.
'''


def once(s,old,new):
    assert s.count(old)==1,old[:140]
    return s.replace(old,new,1)


def build():
    raw=subprocess.check_output(['git','show',PIN+':'+BASE],cwd=ROOT)
    n=copy.deepcopy(json.loads(raw));old_hash=n['metadata']['aes_standalone']['code_sha256']
    for c in n['cells']:
        role=c['metadata'].pop('aes48_role','')
        if role:c['metadata']['aes49_role']=role
        s=c['source'] if isinstance(c['source'],str) else ''.join(c['source'])
        s=s.replace(old_hash,PLACEHOLDER).replace(Path(BASE).stem,'__E49_NAME__').replace('E48','E49').replace('__E49_NAME__',NAME)
        if role=='hyperparameters':
            s=once(s,'LDS_REWEIGHT = "inverse"','LDS_REWEIGHT = "inverse_sqrt_after_smoothing"\nLDS_POWER = 0.5  # E49: akar frekuensi SETELAH smoothing; tidak menambah cap.')
        if role=='configuration_defaults':
            s=once(s,"'lds_sigma':LDS_SIGMA, 'lds_reweight':LDS_REWEIGHT,","'lds_sigma':LDS_SIGMA, 'lds_reweight':LDS_REWEIGHT, 'lds_power':LDS_POWER,")
            s=once(s,"if result['lds_bin_width']!=1 or result['lds_reweight']!='inverse':","if result['lds_bin_width']!=1 or result['lds_reweight']!='inverse_sqrt_after_smoothing' or result['lds_power']!=0.5:")
            s=s.replace('E49 mengunci LDS bin1poin dan inverse.','E49 mengunci LDS bin1poin dan inverse akar setelah smoothing (power0.5).')
        if role=='configuration_apply':
            s=s.replace('e48_native_lds_inverse_v1','e49_native_lds_tempered05_v1')
            s+='\nprint("LDS power:",cfg["lds_power"],"| inverse akar SETELAH smoothing | mean bobot training=1")\n'
        if role=='definition:lds_helpers':
            s=once(s,"if cfg['lds_bin_width']!=1 or cfg['lds_reweight']!='inverse':","if cfg['lds_bin_width']!=1 or cfg['lds_reweight']!='inverse_sqrt_after_smoothing' or cfg['lds_power']!=0.5:")
            s=s.replace('E49 mengunci bin 1 poin dan inverse frekuensi LDS.','E49 mengunci bin1poin dan inverse akar setelah smoothing (power0.5).')
            s=once(s,'    inverse = 1./density','    inverse = np.power(density,-cfg["lds_power"])')
            s=once(s,"float(1./effective[j]/normalizer)","float(effective[j]**(-cfg['lds_power'])/normalizer)")
            s=once(s,"'weights':weights.tolist(),'kernel':kernel.tolist()}","'weights':weights.tolist(),'kernel':kernel.tolist(),'lds_power':cfg['lds_power']}")
            s=once(s,"'method':'LDS inverse Gaussian; mean-one on training only'","'method':'LDS inverse square root AFTER Gaussian smoothing; mean-one on training only',\n             'weight_power':cfg['lds_power']")
            s=s.replace('validate_e48_download','validate_e49_download')
            s=once(s,"    samples=read(names[4],'Bobot LDS Training')", "    samples=read(names[4],'Bobot LDS Training')\n    audits=read(names[4],'Audit LDS')\n    if len(audits)!=cfg['folds']+1:\n        raise RuntimeError('Unduhan dihentikan: audit LDS tidak lengkap.')")
            s=once(s,"        wanted={r['ID Jawaban']:r for r in audit['samples']}", '''        wanted={r['ID Jawaban']:r for r in audit['samples']}
        saved=[r for r in audits if r['Fold']==fold['fold'] and r['Tahap']==('Latihan final' if fold['fold'] is None else 'Validasi silang')]
        if len(saved)!=1 or saved[0].get('weight_power')!=0.5 or saved[0].get('weights_sha256')!=audit['weights_sha256'] or saved[0].get('training_count')!=len(expected):
            raise RuntimeError('Unduhan dihentikan: audit eksponen/fingerprint LDS bukan E49 training fold yang sesuai.')''')
            s=once(s,"'lds_training_rows':expected_total,","'lds_power':cfg['lds_power'],'lds_training_rows':expected_total,")
        if role=='definition:load_model':
            s=once(s,"for key in ('lds_bin_width',","for key in ('lds_power','lds_bin_width',")
        if role=='definition:training_report':
            s=once(s,"settings={'lds_bin_width'","settings={'lds_power':'LDS eksponen inverse setelah smoothing','lds_bin_width'")
        if role=='definition:build_reports':s=s.replace('validate_e48_download','validate_e49_download')
        c['source']=s
    n['cells'][0]['source']=INTRO
    fp=hashlib.sha256('\n'.join(c['source'] for c in n['cells'] if c['cell_type']=='code').encode()).hexdigest()
    for i,c in enumerate(n['cells']):
        c['source']=c['source'].replace(PLACEHOLDER,fp);c['id']=f'e49-{i:03d}-'+hashlib.sha256(c['source'].encode()).hexdigest()[:8]
        if c['cell_type']=='code':ast.parse(c['source']);c['outputs']=[];c['execution_count']=None
    n['metadata']['colab']['name']=NAME+'.ipynb';n['metadata']['aes_standalone']['code_sha256']=fp
    n['metadata']['aes_experiment'].update(name='E49',format_commit=PIN,format_path=BASE,format_sha256=hashlib.sha256(raw).hexdigest(),
        lds_reweight='inverse_sqrt_after_smoothing',lds_power=.5,
        treatment='E48 LDS inverse power1 to power0.5 AFTER smoothing; E43 model/protocol unchanged')
    text=json.dumps(n,ensure_ascii=False,indent=1)+'\n'
    for folder in ['notebooks/experiments','aes-learning/notebooks/experiments']:(ROOT/folder/(NAME+'.ipynb')).write_text(text)
    print(json.dumps({'name':NAME,'cells':len(n['cells']),'code_sha256':fp}))
    return n


if __name__=='__main__':build()
