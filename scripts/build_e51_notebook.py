"""E51 from pinned E50: identity-initialized per-answer gated concatenation."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
PIN='c36003670ad9476951b8b607a6018e5a24ceb7e2'
BASE='notebooks/experiments/AES_300_E50_E43_S3_MultiHeadCoAttention4_Native_Learnable_GPU.ipynb'
NAME='AES_300_E51_E50_S3_GatedConcat_Native_Learnable_GPU'
PLACEHOLDER='__NOTEBOOK_CODE_SHA256__'

INTRO='''# E51 · E50 + gated concatenation · S3 Native Learnable

**GPU T4 → Run all → upload `fix_dataset_300.xlsx`.** Cukup dataset Excel, tanpa ZIP hasil lama. Kode Python biasa disusun per section, tanpa magic atau worker tersembunyi.

E51 menguji **satu perubahan dari E50: gate per jawaban sebelum concatenation**. Gate membaca fitur RoBERTa 768 dimensi dan proyeksi BGE 256 dimensi. `Linear(1024, 2)` diikuti softmax menghasilkan `alpha_R` dan `alpha_B`, lalu:

```text
fusion = concat(2 × alpha_R × roberta, 2 × alpha_B × bge)
```

Fusion tetap 1.024 dimensi dan masuk ke MLP yang sama. Tambahan **2.050 parameter** dilatih bersama head dengan LR `1e-4`. Bobot dan bias gate dimulai dari nol: alpha awal `0,5 / 0,5`, sehingga pengali fitur `1 / 1`. Dengan demikian fitur dan prediksi awal identik dengan E50. Inisialisasi gate menjaga urutan pengacakan dan bobot komponen bersama tetap sama.

**Alur model:** tiga teks → RoBERTa fine-tuned → ScalarMix 4 layer + gamma → BiGRU 64 per arah → multihead co-attention 4 × 32 dengan dua arah dan roundtrip → matching + masked mean → proyeksi 768; BGE frozen → proyeksi 256; kedua cabang → gate → concat 1.024 → MLP 256/GELU/dropout 0,1 → satu skor. Encoder BGE tetap frozen, sedangkan proyeksinya ikut belajar.

| Pengaturan | Nilai |
|---|---|
| Input X | `soal`, `jawaban_referensi`, `response_text` |
| Target y | `rater`, skor guru 0–100; target loss dinormalisasi ke 0–1 |
| Gate | `FUSION_GATE_MODE='softmax_2x_identity'`; alpha 0–1, jumlah 1, pengali fitur 2 × alpha |
| Learning rate | Encoder RoBERTa `5e-6`; komponen lain dan gate `1e-4` |
| Loss / pemilihan epoch | MSE biasa / RMSE validasi tanpa bobot |
| Optimizer / scheduler | AdamW, decay 0,01; warmup 10%, linear decay sesuai durasi fit seperti E50 |
| Training | S3 saja; satu 5-fold CV baru; maksimal 25 epoch, patience 5; satu refit dan satu evaluasi test |
| Pengacakan | Seed dasar 42; seed fold mengikuti acuan; tanpa loop banyak seed atau menjalankan kontrol ulang |

Tiga teks saling terhubung melalui konteks encoder dan co-attention. **Label tidak masuk ke encoder atau gate**; label digunakan untuk loss, pemilihan epoch, dan evaluasi. Tidak ada LDS, ranking, cosine, S1/S2, atau perubahan jadwal refit seperti E46/E47. Gate tidak mengubah dimensi fusion, MLP, dropout, atau attention E50. Softmax gate dihitung dalam FP32 saat AMP; pengali mengikuti dtype fitur.

## Membaca hasil gate

- **Excel 04 → Gate Validasi OOF:** 240 jawaban development, masing-masing dari checkpoint fold terpilih yang tidak melatih jawaban tersebut. **Gate Test:** 60 jawaban test dari checkpoint refit. Keduanya menyimpan ID, nilai guru, prediksi, alpha, dan pengali fitur.
- **Excel 04 → Ringkasan Gate:** jumlah jawaban, mean, SD populasi, minimum, dan maksimum alpha; terpisah per fold, OOF gabungan, dan test.
- **Excel 05 → Konfigurasi Gate:** resep dan norma parameter dari lima checkpoint fold serta satu checkpoint refit. History, bobot ScalarMix, dan 24 baris proyeksi multihead tetap tersedia.
- **Excel 02 → Fusion S3 Sebelum Gate / Fusion S3:** vektor concat langsung dan vektor sesudah gate yang masuk MLP. **Gate Final Semua Jawaban** berisi inspeksi 300 jawaban memakai model refit; ini bukan prediksi OOF atau evaluasi 300 data.

Laporan prediksi lengkap tetap memuat skor mentah, normalisasi, pembulatan, selisih tepat 0/1/2/3/4/5/10 dan lainnya, toleransi ≤n, MSE, MAE, serta error maksimum. Normalisasi berbeda dari pembulatan.

Alpha disimpan **dari forward yang sama dengan prediksi**, dalam cache fold dan CSV test. Resume dapat membangun laporan dari hasil tersimpan tanpa mengevaluasi test lagi. Pemeriksaan sebelum ZIP memverifikasi metrik, OOF, rekap selisih, ID, nilai, alpha, pengali, statistik ringkasan, dan resep gate. Checkpoint E51 memakai identitas kode dan konfigurasi tersendiri; source E50 dan HTML lama tetap tersedia.

Alpha gate mengatur fitur, **bukan persentase kontribusi terhadap skor atau probabilitas jawaban benar**. Gate bisa mempelajari bobot hampir konstan atau saturasi. Variasi antarjawaban perlu diamati dari hasil; tidak dipaksakan. Bobot attention Excel 02 tetap rata-rata empat head.

## Landasan penelitian

Arevalo dkk. (2017), [Gated Multimodal Units for Information Fusion](https://openreview.net/forum?id=Hy-2G6ile), [kode penulis](https://github.com/johnarevalo/gmu-mmimdb). GMU asli memadukan representasi berdimensi sama dengan sigmoid gate dan penjumlahan tertimbang. **E51 merupakan adaptasi gated concatenation dua encoder teks**, memakai dua scalar softmax dan faktor 2 untuk menjaga kondisi awal identik. Ini bukan reproduksi persis GMU atau bukti bahwa resep ini optimal untuk AES. Rujukan multihead dan co-attention tetap mengikuti E50.

Acuan E50: mean RMSE CV 5,7954; MAE validasi 4,4512; selisih mentah ≤5 poin sebanyak 157/240. Test E50: RMSE 5,3059 dan MAE 3,8856. E43 dengan RMSE test historis 5,1265 tetap pembanding. **Belum ada hasil penelitian E51**. Pilih resep melalui validasi; perbandingan test historis bersifat eksploratif. Gate belum menjamin peningkatan RMSE atau target 60/60 dalam ±3/±5 poin.
'''


def once(s,old,new):
    assert s.count(old)==1,old[:140]
    return s.replace(old,new,1)


def build():
    raw=subprocess.check_output(['git','show',PIN+':'+BASE],cwd=ROOT)
    n=copy.deepcopy(json.loads(raw));old_hash=n['metadata']['aes_standalone']['code_sha256']
    for c in n['cells']:
        role=c['metadata'].pop('aes50_role','')
        if role:c['metadata']['aes51_role']=role
        s=c['source'] if isinstance(c['source'],str) else ''.join(c['source'])
        s=s.replace(old_hash,PLACEHOLDER).replace(Path(BASE).stem,'__E51_NAME__')
        s=s.replace('AES_300_E50_E50_S3_MultiHeadCoAttention4_Native_Learnable_GPU','__E51_NAME__')
        s=s.replace('E50','E51').replace('__E51_NAME__',NAME)
        if role=='hyperparameters':s+='\n# Gate perjawaban;zero logits memberi pengali1/1 pada kedua cabang.\nFUSION_GATE_MODE = "softmax_2x_identity"\n'
        if role=='configuration_defaults':
            s=s.replace("'s3_fusion': 'concat'","'s3_fusion': 'gated_concat', 'fusion_gate_mode':FUSION_GATE_MODE")
            s=s.replace("if result['s3_fusion'] != 'concat':","if result['s3_fusion'] != 'gated_concat' or result['fusion_gate_mode']!='softmax_2x_identity':")
        if role=='configuration_apply':
            s=s.replace('e50_native_multihead_coattention4_v1','e51_native_gated_concat_v1')
            s+='\nprint("Fusion gate:",cfg["fusion_gate_mode"],"|2.050parameter | LRhead",cfg["s3_mlp_lr"])\n'
        if role=='definition:SingleScoreModel':
            s=once(s,'coattention_heads=4):','coattention_heads=4, fusion_gate_mode="softmax_2x_identity"):')
            s=once(s,'        self.scalar_mix = ScalarMix(scalar_mix_layers, gamma_initial)','        self.scalar_mix = ScalarMix(scalar_mix_layers, gamma_initial)\n        self.fusion_gate = GatedConcatenation(fusion_gate_mode)')
            s=once(s,"        features['concat'] = torch.cat([features['roberta_layernorm'], features['projection']], dim=1)",'''        features['concat_before_gate'] = torch.cat([features['roberta_layernorm'], features['projection']], dim=1)
        features['concat'],features['gate_alpha']=self.fusion_gate(features['roberta_layernorm'],features['projection'])''')
        if role=='definition:make_model':s=once(s,"coattention_heads=cfg['coattention_heads'])","coattention_heads=cfg['coattention_heads'], fusion_gate_mode=cfg['fusion_gate_mode'])")
        if role=='definition:optimizer_groups':s=once(s,"elif name.startswith('head.'):","elif name.startswith(('head.','fusion_gate.')):")
        if role=='definition:load_model':s=s.replace("for key in ('coattention_heads',","for key in ('fusion_gate_mode','coattention_heads',")
        if role=='definition:fit':s=once(s,"    info['multihead_projections'] = model.sequence_pool.multihead.report()","    info['multihead_projections'] = model.sequence_pool.multihead.report()\n    info['fusion_gate'] = model.fusion_gate.report()")
        if role=='definition:Engine':
            s=once(s,'        values = []','        values = []\n        self.last_gate_rows=[]')
            s=once(s,'                values.append(self.forward(ids).reshape(-1).float().cpu().numpy())', '''                values.append(self.forward(ids).reshape(-1).float().cpu().numpy())
                alpha=self.model.fusion_gate.last_alpha.float().cpu().numpy()
                self.last_gate_rows.extend({'dataset_index':int(i), 'gate_roberta':float(a[0]),'gate_bge':float(a[1])} for i,a in zip(ids,alpha))''')
        if role=='definition:train_all':
            s=once(s,"'validation_indices': fold['validation'], 'prediction': prediction.tolist()}","'validation_indices': fold['validation'], 'prediction': prediction.tolist(),\n                          'gate_predictions':engine.last_gate_rows}")
            s=once(s,"'selisih': prediction-bundle.y[manifest['test']]}","'selisih': prediction-bundle.y[manifest['test']],\n                              'gate_roberta':[r['gate_roberta'] for r in engine.last_gate_rows],\n                              'gate_bge':[r['gate_bge'] for r in engine.last_gate_rows]}")
        if role=='definition:export_training_event':
            s=once(s,"'Proyeksi_multihead':pd.DataFrame(training['multihead_projections']),","'Proyeksi_multihead':pd.DataFrame(training['multihead_projections']),\n                                                           'Fusion_gate_config':pd.DataFrame([training['fusion_gate']]),\n                                                           'Gate_validation':gate_prediction_frame(bundle,result['validation_indices'],result['prediction'],result['gate_predictions']),")
            s=once(s,"'Proyeksi_multihead':pd.DataFrame(result['multihead_projections'])}","'Proyeksi_multihead':pd.DataFrame(result['multihead_projections']),\n                                            'Fusion_gate_config':pd.DataFrame([result['fusion_gate']])}")
            s=once(s,"'Prediksi_test': expanded,","'Prediksi_test': expanded,\n                                           'Gate_test':gate_prediction_frame(bundle,ids,raw['prediksi_mentah'].to_numpy(),raw.to_dict('records')),")
        if role=='definition:evaluation_report':
            s=once(s,'    return sheets\n',"    gate_tables=gate_report_tables(source)\n    for title in ['Gate Validasi OOF','Gate Test','Ringkasan Gate']:\n        if title in gate_tables:sheets[title]=gate_tables[title]\n    return sheets\n")
            s=s.replace(' + Multihead Co-Attention4 + Masked Mean Pooling',' + Multihead Co-Attention4 + Masked Mean Pooling + Gated Concat')
        if role=='definition:training_report':
            s=s.replace("settings={'coattention_heads'","settings={'fusion_gate_mode':'Mode Fusion Gate','coattention_heads'")
            s=once(s,'    return sheets\n',"    gate_tables=gate_report_tables(source)\n    if 'Konfigurasi Gate' in gate_tables:sheets['Konfigurasi Gate']=gate_tables['Konfigurasi Gate']\n    return sheets\n")
        if role=='definition:answer_vectors_report':
            s=s.replace("('14_S3_fitur_setelah_training.xlsx','concat','Fusion S3'),","('14_S3_fitur_setelah_training.xlsx','concat','Fusion S3'),\n              ('14_S3_fitur_setelah_training.xlsx','concat_before_gate','Fusion S3 Sebelum Gate'),\n              ('14_S3_fitur_setelah_training.xlsx','gate_alpha','Gate Final Semua Jawaban'),")
            s=s.replace('Hasilnya 1024 komponen.','Hasilnya1024komponen SETELAH gate2alpha;sheetSebelumGate menyimpanconcatlangsung. GateFinalSemuaJawaban adalahinspeksi refit, bukan OOF.')
            s=once(s,'    return sheets if len(sheets)>1 else None',"    if 'Gate Final Semua Jawaban' in sheets:sheets['Gate Final Semua Jawaban'].headers=['ID Jawaban','Alpha RoBERTa','Alpha BGE']\n    return sheets if len(sheets)>1 else None")
        if role=='definition:download_helpers':
            s=s.replace('validate_e50_download','validate_e51_download')
            s=once(s,"    history=read(names[4],'Fusion (S3)')",'''    try:
        gate_proof=verify_gate_exports(read(file,'Gate Validasi OOF'),read(file,'Gate Test'),read(file,'Ringkasan Gate'),
            read(names[4],'Konfigurasi Gate'),oof,detail,cfg)
    except ValueError as error:raise RuntimeError('Unduhan dihentikan: laporan fusion gate tidak cocok: '+str(error)) from error
    history=read(names[4],'Fusion (S3)')''')
            s=once(s,"'summaries_nonempty':True}","'summaries_nonempty':True,**gate_proof}")
        if role=='definition:build_reports':s=s.replace('validate_e50_download','validate_e51_download')
        if role=='architecture_export':s=s.replace('→ concat1024 → MLP256','→ gated concat1024 → MLP256')
        c['source']=s
    helper=(ROOT/'scripts/gated_concat_native.py').read_text()
    helper='\n\n'.join(ast.get_source_segment(helper,node) for node in ast.parse(helper).body if isinstance(node,(ast.FunctionDef,ast.ClassDef)))+'\n'
    index=next(i for i,c in enumerate(n['cells']) if c['metadata'].get('aes51_role')=='definition:SingleScoreModel')
    n['cells'][index:index]=[
        {'cell_type':'markdown','metadata':{},'source':'## Fusion gate: alpha per jawaban, lalu concat\n\nLogit awal 0 → alpha 0,5/0,5 → pengali 1/1. Gate membaca fitur, tidak membaca rater. Helper laporan merangkai hasil prediksi tersimpan.\n'},
        {'cell_type':'code','metadata':{'aes51_role':'definition:GatedConcatenation'},'source':helper,'outputs':[],'execution_count':None}]
    n['cells'][0]['source']=INTRO
    fp=hashlib.sha256('\n'.join(c['source'] for c in n['cells'] if c['cell_type']=='code').encode()).hexdigest()
    for i,c in enumerate(n['cells']):
        c['source']=c['source'].replace(PLACEHOLDER,fp);c['id']=f'e51-{i:03d}-'+hashlib.sha256(c['source'].encode()).hexdigest()[:8]
        if c['cell_type']=='code':ast.parse(c['source']);c['outputs']=[];c['execution_count']=None
    n['metadata']['colab']['name']=NAME+'.ipynb';n['metadata']['aes_standalone']['code_sha256']=fp
    n['metadata']['aes_experiment'].update(name='E51',format_commit=PIN,format_path=BASE,format_sha256=hashlib.sha256(raw).hexdigest(),
        s3_fusion='gated_concat',fusion_gate_mode='softmax_2x_identity',added_parameters_vs_e50=2050,
        gate_identity_initialization=True,gate_oof_and_test_export=True,
        treatment='E50 concat to per-answer gated concat; zero logits,2alpha scales,2050parameters; all other model/training settings retained')
    raw=json.dumps(n,ensure_ascii=False,indent=1)+'\n'
    for folder in ['notebooks/experiments','aes-learning/notebooks/experiments']:(ROOT/folder/(NAME+'.ipynb')).write_text(raw)
    print(json.dumps({'name':NAME,'cells':len(n['cells']),'code_sha256':fp}))
    return n


if __name__=='__main__':build()
