"""E50: E43 model with 4-head co-attention, original MSE and verified reports."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
PIN='7241a21dfec37c942948b754da3960fba4c7731c'
BASE='notebooks/experiments/AES_300_E43_E42_S3_MaskedMeanPooling_Native_Learnable_GPU.ipynb'
REPORT_PIN='437b6ff15808c2122a92cfc32874c815d33af633'
REPORT_BASE='notebooks/experiments/AES_300_E48_E43_S3_LDS_WeightedMSE_Native_Learnable_GPU.ipynb'
NAME='AES_300_E50_E43_S3_MultiHeadCoAttention4_Native_Learnable_GPU'
PLACEHOLDER='__NOTEBOOK_CODE_SHA256__'

INTRO='''# E50 · E43 + multi-head co-attention4 · S3 Native Learnable

**Cukup upload `fix_dataset_300.xlsx`. GPU T4 → Run all.** Tidak membutuhkan ZIP hasil lama. S3 saja,satu5foldCVbaru,maksimal25epoch/patience5,saturefit dari ceil median best epochCV,lalu satutest. Seed42konstanta/foldseed+nomorfold,tanpa loopmulti-seed/kontrolulang.

**Satu perubahan modul:** co-attention E43 menjadi multi-head co-attention4 dengan proyeksi Q/K/V/output yang dipelajari. Dimensi total128tetap;4head×32dimensi. Ini bukan sekadar membagi skor E43 ke empat bagian: ada65.536parameter proyeksi baru,biasFalse,tanpa attentiondropout tambahan. Proyeksi yang sama dipakai pada kedua arah; softmax jawaban→referensi dan referensi→jawaban dihitung terpisah. Jalur round-trip A(BVjawaban) dipertahankan perhead; hasil aligned/roundtrip digabung dan diproyeksikan ke128. Matching tetap concat5×128→128,maskedmean1/N→Linear768.

Arsitektur: tiga teks → RoBERTa fine-tuned → ScalarMix4+gamma → BiGRU64/arah → multi-head co-attention dua arah+matching → maskedmean → concatRoBERTa768+BGEproyeksi256 → MLP → satu nilai. BGE encoder frozen. Soal tetap masuk konteks encoder; attention eksplisit mencocokkan token jawaban dengan referensi. Tidak menambah encoder/model lain atau Transformerblock pengganti BiGRU.

| Pengaturan | Nilai |
|---|---|
| X/y | `soal`,`jawaban_referensi`,`response_text` / `rater`0–100 |
| Head/dimensi | 4 / total128 /32perhead; `COATTENTION_HEADS=4` |
| LR encoder / komponen lain | 5e-6 /1e-4; proyeksi Q/K/V/output masuk pooling_norm1e-4 |
| Loss/dropout | MSE biasa target0–1; dropoutMLP0,1tetap |
| Jadwal | E43asli: warmup10%,linear decay sesuai durasi fit masing-masing |

Tidak memakai LDS/ranking/cosine/S1/S2/horizonrefit25 E46/E47. Semua label guru asli tetap. Earlystop berdasarkanRMSEvalidasi biasa. Test tidak memilih epoch. Padding jawaban/referensi dimask; token valid minimal1. Softmax dan produk attention FP32 saat encoderAMP. BiGRU dan semua layer bersama mempertahankan inisialisasiE43; inisialisasi proyeksi baru di dalamforkRNGCPU menjaga stateRNGumum tetap. Bobot baru dilatih dari awal.

**Rujukan:** [Vaswani dkk.(2017), Attention Is All You Need](https://arxiv.org/abs/1706.03762), [implementasi Annotated Transformer](https://github.com/harvardnlp/annotated-transformer); [Zhang & Litman(2018), Co-Attention Based Neural Network for Source-Dependent Essay Scoring](https://aclanthology.org/W18-0549/) sebagai dasar matching sumber–esai. E50 mengadaptasi multihead ke co-attention dua arah+roundtrip milikproyek. Empathead/dim128 danbiasFalseadalah pilihanrancangan,bukanoptimumAESdaripaper. Head tidak diberi tugas semantik tetap. Belum ada hasil penelitianE50;target60/60dalam±3atau±5belumdijamin. E43tetapacuanCV5,8876/test5,1265.

**Output limaExcel:**04→testmentah/normalisasi/bulat,SelisihTepat0/1/2/3/4/5/10danlainnya,Toleransi≤n,MSE/MAE/maxerror,PrediksiValidasiOOF240terpisahdari60test. 05→historyMSE,CV±3/±5/maxerror,BobotScalarMix,**Proyeksi Multihead**4baris×6checkpoint=24baris(Q/K/V/outputnormperhead,dimensi,parametertotal). Norma parameter bukan attentionimportance atau ukuran kontribusi kausal.

Excel02/trace mempertahankan antarmuka inspeksi: bobotco-attention yang ditampilkan adalah **rata-rata probabilitas empathead**. Matriks perhead tersedia pada `sequence_pool.last_cross_attention_heads`/`last_reverse_attention_heads` hanya saat capture_alignmentaktif;tidakdisimpanuntukseluruhdataset. Maskedmean tetapmerangkum tokenvalid. Normalisasi0–1berbeda daripembulatan.

Gate wajib membacaulangExcel sebelumZIP: ringkasannonempty,metriktest/OOFcocok,ID/split/labelbenar,rekapselisihbenar,historyfinite,konfigurasihead4/dim128/32dan24barisproyeksicheckpointlengkap. ExportfixbindingfilemencegahbugringkasanE44/E45. Sourcehistoris/hasil/HTMLtidakdiubah. Folder/hash/lesson_version/checkpointE50terpisah. Pythonbiasa persection,tanpamagic/worker.
'''


def once(s,old,new):
    assert s.count(old)==1,old[:140]
    return s.replace(old,new,1)


def build():
    raw=subprocess.check_output(['git','show',PIN+':'+BASE],cwd=ROOT)
    report_raw=subprocess.check_output(['git','show',REPORT_PIN+':'+REPORT_BASE],cwd=ROOT)
    n=copy.deepcopy(json.loads(raw));report=json.loads(report_raw)
    reports={c['metadata'].get('aes48_role'):c for c in report['cells']}
    report_roles={'definition:evaluation_report','definition:training_report','definition:build_reports','download_execute','training_execute'}
    for c in n['cells']:
        role=c['metadata'].pop('aes43_role','')
        if role:c['metadata']['aes50_role']=role
        s=reports[role]['source'] if role in report_roles else c['source']
        s=s if isinstance(s,str) else ''.join(s)
        if role in report_roles:
            s=s.replace(report['metadata']['aes_standalone']['code_sha256'],PLACEHOLDER).replace(Path(REPORT_BASE).stem,NAME).replace('E48','E50')
        else:
            s=s.replace(n['metadata']['aes_standalone']['code_sha256'],PLACEHOLDER).replace(Path(BASE).stem,NAME).replace('E43','E50')
        s=s.replace('bigru_coattention_mean_pooling','bigru_multihead_coattention_mean_pooling')
        if role=='hyperparameters':s+='\n# Multi-head co-attention; BiGRU64/arah memberi dimensi128.\nCOATTENTION_HEADS = 4\n'
        if role=='configuration_defaults':
            s=s.replace("    's3_fusion': 'concat',","    'coattention_heads':COATTENTION_HEADS,\n    's3_fusion': 'concat',")
            s=once(s,'    return result\n',"    if type(result['coattention_heads']) is not int or result['coattention_heads']!=4 or result['s3_sequence_hidden']!=64:\n        raise ValueError('E50 mengunci empat head dan BiGRU64/arah, totaldim128.')\n    return result\n")
        if role=='configuration_apply':
            s=s.replace('e43_native_masked_mean_v1','e50_native_multihead_coattention4_v1')
            s+='\nprint("Multi-head co-attention:",cfg["coattention_heads"],"head |128total |32/head | LR",cfg["s3_layers_lr"])\n'
        if role=='definition:AnswerBiGRUCoAttentionMeanPool':
            s=once(s,'def __init__(self, hidden_size=64, attention_hidden_size=64):','def __init__(self, hidden_size=64, attention_hidden_size=64, coattention_heads=4):')
            s=once(s,'        self.matching_projection = nn.Linear(5 * dim, dim)', '''        self.matching_projection = nn.Linear(5 * dim, dim)
        # Layer bersama dan RNG/dropout umum tetap E43; model baru dibuat pada CPU.
        with torch.random.fork_rng(devices=[]):
            self.multihead = MultiHeadCoAttention(dim,coattention_heads)
        self.last_cross_attention_heads = None
        self.last_reverse_attention_heads = None''')
            start=s.index('        # Softmax/produk skor FP32')
            end=s.index('        matched =',start)
            s=s[:start]+'''        aligned_reference,round_trip,a_heads,r_heads=self.multihead(response,reference,valid,ref_valid)
        # Antarmuka ekspor E43 memakai matriks3D; tampilkan rerata probabilitas4head.
        a_to_r=a_heads.mean(1)
        r_to_a=r_heads.mean(1)
        with torch.autocast(device_type=hidden.device.type,enabled=False):
            response32=response.float()
            matching=torch.cat([response32,aligned_reference,response32-aligned_reference,
                                response32*aligned_reference,round_trip],dim=-1)
        self.last_cross_attention_heads=a_heads.detach() if self.capture_alignment else None
        self.last_reverse_attention_heads=r_heads.detach() if self.capture_alignment else None
'''+s[end:]
            s=s.replace('E50: BiGRU + co-attention respons–referensi, lalu masked mean token matching128.',
                        'E50: BiGRU + multi-head co-attention4 dua arah/round-trip, lalu maskedmean128.')
        if role=='definition:SingleScoreModel':
            s=once(s,'scalar_mix_layers=4, gamma_initial=1.0):','scalar_mix_layers=4, gamma_initial=1.0, coattention_heads=4):')
            s=once(s,'AnswerBiGRUCoAttentionMeanPool(sequence_hidden, pooling_attention_hidden)','AnswerBiGRUCoAttentionMeanPool(sequence_hidden, pooling_attention_hidden, coattention_heads)')
        if role=='definition:make_model':s=once(s,"gamma_initial=cfg['scalar_mix_gamma_init'])","gamma_initial=cfg['scalar_mix_gamma_init'], coattention_heads=cfg['coattention_heads'])")
        if role=='definition:load_model':s=s.replace("for key in ('roberta',","for key in ('coattention_heads','roberta',")
        if role=='definition:fit':
            s=once(s,"    info['scalar_mix_layers'] = cfg['scalar_mix_layers']","    info['scalar_mix_layers'] = cfg['scalar_mix_layers']\n    info['multihead_projections'] = model.sequence_pool.multihead.report()")
        if role=='definition:export_training_event':
            s=once(s,"'Bobot_scalar_mix': pd.DataFrame(training['scalar_mix_weights']),","'Bobot_scalar_mix': pd.DataFrame(training['scalar_mix_weights']),\n                                                           'Proyeksi_multihead':pd.DataFrame(training['multihead_projections']),")
            s=once(s,"'Bobot_scalar_mix': pd.DataFrame(result['scalar_mix_weights'])}","'Bobot_scalar_mix': pd.DataFrame(result['scalar_mix_weights']),\n                                            'Proyeksi_multihead':pd.DataFrame(result['multihead_projections'])}")
        if role=='definition:training_report':
            s='\n'.join(line for line in s.splitlines() if "['Loss Training (LDS-MSE)'" not in line)+'\n'
            s=s.replace("row.get('train_mse_scaled'),row.get('train_lds_mse_scaled'),","row.get('train_mse_scaled'),")
            s=s.replace("'Loss Training (MSE)','Loss Training (LDS-MSE)'","'Loss Training (MSE)'")
            s=s.replace("settings={'lds_bin_width':'LDS bin (poin)','lds_kernel_size':'LDS kernel Gaussian','lds_sigma':'LDS sigma (poin)','lds_reweight':'LDS pembobotan',","settings={'coattention_heads':'Jumlah Head Co-Attention',")
            s=s.replace('sheets.update(lds_report_tables(source))','sheets.update(multihead_report_tables(source))')
        if role=='definition:evaluation_report':
            s='\n'.join(line for line in s.splitlines() if "['LDS','" not in line)+'\n'
            s=s.replace(' + Co-Attention + Masked Mean Pooling',' + Multihead Co-Attention4 + Masked Mean Pooling')
        if role=='definition:build_reports':s=s.replace('validate_e48_download','validate_e50_download')
        if role=='architecture_export':s=s.replace('→ co-attention{','→ multihead4 co-attention{')
        c['source']=s
    # Embed plain Python definitions; no external project import required in Colab.
    module=(ROOT/'scripts/multihead_coattention_native.py').read_text()
    module_defs='\n\n'.join(ast.get_source_segment(module,node) for node in ast.parse(module).body if isinstance(node,(ast.FunctionDef,ast.ClassDef)))+'\n'
    index=next(i for i,c in enumerate(n['cells']) if c['metadata'].get('aes50_role')=='definition:AnswerBiGRUCoAttentionMeanPool')
    n['cells'][index:index]=[
        {'cell_type':'markdown','metadata':{},'source':'## Multi-head co-attention4: dua arah dan round-trip\n\nQ/K/V/output dilatih pada LR pooling_norm. Probabilitas perhead terpisah; tidak ada attentiondropout baru.\n'},
        {'cell_type':'code','metadata':{'aes50_role':'definition:MultiHeadCoAttention'},'source':module_defs,'outputs':[],'execution_count':None}]
    source=reports['definition:lds_helpers']['source']
    gate=next(node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name=='validate_e48_download')
    gate=ast.get_source_segment(source,gate).replace('validate_e48_download','validate_e50_download').replace('OOF/LDS','OOF/multihead')
    start=gate.index("    samples=read(names[4],'Bobot LDS Training')");end=gate.index("    history=read(names[4],'Fusion (S3)')",start)
    gate=gate[:start]+'''    projections=read(names[4],'Proyeksi Multihead')
    if len(projections)!=(cfg['folds']+1)*4:
        raise RuntimeError('Unduhan dihentikan: proyeksi multihead tidak lengkap.')
    for fold in [*manifest['folds'],{'fold':None}]:
        rows=[r for r in projections if r['Fold']==fold['fold'] and r['Tahap']==('Latihan final' if fold['fold'] is None else 'Validasi silang')]
        if len(rows)!=4 or {r['Head'] for r in rows}!={1,2,3,4} or any(
            r['Jumlah Head']!=4 or r['Dimensi Total']!=128 or r['Dimensi per Head']!=32 or r['Parameter Proyeksi Total']!=65536 or
            any(not isinstance(r.get(k),(int,float)) or not math.isfinite(r[k]) or r[k]<=0 for k in ['Norma Q','Norma K','Norma V','Norma Output per Head']) for r in rows):
            raise RuntimeError('Unduhan dihentikan: konfigurasi/norma proyeksi multihead tidak cocok.')
'''+gate[end:]
    gate=gate.replace(" or not math.isfinite(r['Loss Training (LDS-MSE)'])",'')
    gate=gate.replace("'training_only_lds_verified':True,","'multihead_projections_verified':True,")
    gate=gate.replace("'lds_training_rows':expected_total,","'coattention_heads':4,'multihead_projection_rows':len(projections),")
    pred='\n\n'.join(ast.get_source_segment(source,node) for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name in ['prediction_report_tables','verify_prediction_report_tables'])
    index=next(i for i,c in enumerate(n['cells']) if c['metadata'].get('aes50_role')=='definition:fit')
    n['cells'][index:index]=[
        {'cell_type':'markdown','metadata':{},'source':'## Rincian prediksi dan gate unduhan\n\nHanya menghitung laporan dari prediksi tersimpan. Metrik biasa, tanpa pembobotan loss.\n'},
        {'cell_type':'code','metadata':{'aes50_role':'definition:download_helpers'},'source':pred+'\n\n'+gate+'\n','outputs':[],'execution_count':None}]
    n['cells'][0]['source']=INTRO
    fp=hashlib.sha256('\n'.join(c['source'] for c in n['cells'] if c['cell_type']=='code').encode()).hexdigest()
    for i,c in enumerate(n['cells']):
        c['source']=c['source'].replace(PLACEHOLDER,fp);c['id']=f'e50-{i:03d}-'+hashlib.sha256(c['source'].encode()).hexdigest()[:8]
        if c['cell_type']=='code':ast.parse(c['source']);c['outputs']=[];c['execution_count']=None
    n['metadata']['colab']['name']=NAME+'.ipynb';n['metadata']['aes_standalone']['code_sha256']=fp
    n['metadata']['aes_experiment'].update(name='E50',format_commit=PIN,format_path=BASE,format_sha256=hashlib.sha256(raw).hexdigest(),
        reporting_commit=REPORT_PIN,loss='MSE',ranking=False,coattention_heads=4,coattention_head_dim=32,
        s3_pooling='bigru_multihead_coattention_mean_pooling',added_parameters_vs_e43=65536,
        prediction_breakdown=True,oof_predictions=True,requires_previous_zip=False,
        export_summary_closure_fixed=True,download_validation_required=True,
        treatment='E43 single-head matching to projected multihead4 bidirectional co-attention with per-head round trip; original MSE')
    text=json.dumps(n,ensure_ascii=False,indent=1)+'\n'
    for folder in ['notebooks/experiments','aes-learning/notebooks/experiments']:(ROOT/folder/(NAME+'.ipynb')).write_text(text)
    print(json.dumps({'name':NAME,'cells':len(n['cells']),'code_sha256':fp}))
    return n


if __name__=='__main__':build()
