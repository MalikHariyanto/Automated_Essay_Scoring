"""E25/E26 dari E24: dua perubahan S3 terpisah, tetap Colab GPU mandiri."""
import ast
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from build_context_pair_notebook import source, replace_once, definitions
ROOT=Path(__file__).resolve().parents[1]
PIN='f3d25c4'
BASE='notebooks/experiments/AES_300_E24_Tiga_Input_AttentionPooling_GPU.ipynb'
VARIANTS={'E25':('AES_300_E25_Tiga_Input_Attention_GatedFusion_GPU','attention','Attention Pooling + Gated Fusion'),
          'E26':('AES_300_E26_Tiga_Input_BiGRU_Attention_GPU','bigru_attention','BiGRU + Attention Pooling')}


def build(experiment):
    name,mode,label=VARIANTS[experiment]
    raw=subprocess.check_output(['git','show',PIN+':'+BASE],cwd=ROOT)
    n=copy.deepcopy(json.loads(raw));old=n['metadata']['aes_standalone']['code_sha256']
    helpers=definitions((ROOT/'scripts/s3_gated_bigru_attention_helpers.py').read_text())
    description=('E24 tetap, ditambah gate skalar per jawaban sebelum concat. Gate menimbang kontribusi dua cabang tanpa mengubah dimensi1024.' if experiment=='E25' else
        'Output token BiGRU dua arah128 diberi attention, diringkas128 lalu diproyeksikan768. Attention bekerja atas urutan token, bukan satu state akhir.')
    for index,cell in enumerate(n['cells']):
        s=source(cell).replace(old,'__NOTEBOOK_CODE_SHA256__').replace('AES_300_E24_Tiga_Input_AttentionPooling_GPU',name)
        if cell['cell_type']=='code':
            if index==10:
                s=s.replace("'s3_pooling': 'attention'","'s3_pooling': '"+mode+"'").replace("result['s3_pooling'] != 'attention'","result['s3_pooling'] != '"+mode+"'")
                s=replace_once(s,"'vector_source': 'pretrained'", "'s3_fusion': '"+('gated_concat' if experiment=='E25' else 'concat')+"', "+("'s3_gate_hidden': 32, " if experiment=='E25' else '')+"\n    'vector_source': 'pretrained'")
                s=replace_once(s,'    return result',"    if result['s3_fusion'] != '"+('gated_concat' if experiment=='E25' else 'concat')+"'"+(" or type(result['s3_gate_hidden']) is not int or result['s3_gate_hidden']<1" if experiment=='E25' else '')+":\n        raise ValueError('Fusion/hidden gate tidak cocok dengan notebook ini.')\n    return result")
            if index==11:s+="\nLABELS.update({'fusion_gate':'Gate RoBERTa','roberta_multiplier':'Pengali RoBERTa','bge_multiplier':'Pengali BGE'})\n"
            if index==12:
                s=s.replace("s3_pooling='attention'","s3_pooling='"+mode+"'")
                s=s.replace("lesson_version='e24_s3_attention_e21_gpu_v1'", "lesson_version='"+experiment.lower()+"_s3_"+mode+"_gpu_v1'")
                s='\n'.join(line for line in s.splitlines() if not line.startswith("print('E24 |"))+'\n'
                s+="print('"+experiment+" | S3: "+description+" | LR layer baru:',cfg['head_lr'],flush=True)\n"
            if index==30:
                if experiment=='E25':
                    s=replace_once(s,'# Satu head satu nilai;',helpers['AnswerFusionGate']+'\n\n# Satu head satu nilai;')
                    s=replace_once(s,'sequence_hidden=64):','sequence_hidden=64, gate_hidden=32):')
                    s=replace_once(s,'            self.sequence_pool = AnswerAttentionPool(sequence_hidden)', '            self.sequence_pool = AnswerAttentionPool(sequence_hidden)\n            self.fusion_gate = AnswerFusionGate(gate_hidden)')
                    s=replace_once(s,"            features['concat'] = torch.cat([features['roberta_layernorm'], features['projection']], dim=1)","""            # Gate menerima fitur per jawaban setelah pooling dan normalisasi.
            features['concat'], gate = self.fusion_gate(features['roberta_layernorm'], features['projection'])
            features['fusion_gate'] = gate
            features['roberta_multiplier'] = 2*gate
            features['bge_multiplier'] = 2*(1-gate)""")
                else:
                    original=definitions(s)['AnswerAttentionPool']
                    s=replace_once(s,original,helpers['AnswerBiGRUAttention'])
                    s=s.replace('self.sequence_pool = AnswerAttentionPool(sequence_hidden)','self.sequence_pool = AnswerBiGRUAttention(sequence_hidden)')
            if index==31 and experiment=='E25':s=replace_once(s,"sequence_hidden=cfg['s3_sequence_hidden'])","sequence_hidden=cfg['s3_sequence_hidden'],gate_hidden=cfg['s3_gate_hidden'])")
            if index==32:
                s=s.replace('attention pooling768',label+' →768')
            if index==43:
                # Existing hooks export the actual weighted concat and actual head intermediates.
                s=s.replace('# Tangkap keluaran head/proyeksi aktual tanpa mengganti arsitektur.', '# Tangkap fitur aktual, termasuk gate atau attention BiGRU; tahap inspeksi eval/no_grad.')
            if index==45:
                extra=(",\n              ('14_S3_fitur_setelah_training.xlsx','fusion_gate','Gate Fusion S3'),\n              ('14_S3_fitur_setelah_training.xlsx','roberta_multiplier','Pengali RoBERTa S3'),\n              ('14_S3_fitur_setelah_training.xlsx','bge_multiplier','Pengali BGE S3')" if experiment=='E25' else ",\n              ('14_S3_fitur_setelah_training.xlsx','bigru_hidden128','Ringkasan BiGRU Attention S3')")
                s=replace_once(s,"('14_S3_fitur_setelah_training.xlsx','Bobot_token_jawaban','Bobot Attention S3')]", "('14_S3_fitur_setelah_training.xlsx','Bobot_token_jawaban','Bobot Attention S3')"+extra+']')
                s=s.replace("' + Attention Pooling'", "' + "+label+"'")
                s=replace_once(s,"settings={'s3_pooling'", "settings={'s3_fusion':'Fusion S3',"+("'s3_gate_hidden':'Hidden gate S3'," if experiment=='E25' else '')+"'s3_pooling'")
                s=s.replace('Additive attention768 →64 →1 pada posisi token jawaban; jumlah berbobot tetap768. Tidak memakai BiGRU.',description)
            cell['outputs']=[];cell['execution_count']=None
        cell['source']=s
    n['cells'][0]['source']=f'''# {experiment} · {label} pada S3

Tiga input semua skenario: **soal, jawaban_referensi, response_text**; label **rater 0–100**. Format single_sequence, mengikuti E24/E21.

**S3: {description}** S1 RoBERTa full fine-tuning + linear dan S2 BGE frozen + MLP tetap. BGE frozen pada S3; RoBERTa ikut fine-tuning. MLP tetap head regresi, bukan encoder.

**Jalankan di web Colab biasa:** Runtime → Change runtime type → T4 GPU → Run all. Pilih fix_dataset_300.xlsx. GPU wajib saat RUN_MODELS aktif. Tidak perlu clone repo, ekstensi, atau mount Drive.

LR encoder S1/S3 **5e-6**; head/proyeksi/pooling/gate baru **1e-4**. Batch efektif8, lima fold, maksimum25epoch, patience5; seed42/split_seed2026. Seluruh skenario dilatih baru. Folder lokal {name}; checkpoint dua slot per tahap. Runtime dihapus berarti progres lokal bisa hilang. Lima Excel diunduh pada cell terakhir; bobot model tidak ikut ZIP ringkasan.

Ini eksperimen baru; jangan memakai checkpoint E23/E24 untuk resume. Pilih metode dari CV, laporkan seluruh seed dan perlakukan test lama yang telah dilihat sebagai eksploratif. Tidak menjamin RMSE lebih rendah.

Rujukan konsep: [GMU, Arevalo dkk.2017](https://openreview.net/forum?id=Hy-2G6ile) dan [implementasi resmi](https://github.com/johnarevalo/gmu-mmimdb); [Dong dkk.2017](https://aclanthology.org/K17-1017/); [PaSTO-GNN, §3.6](https://doi.org/10.3389/frai.2026.1842542). Gate skalar dua encoder teks dan kombinasi BiGRU+attention token merupakan adaptasi, bukan replikasi hasil paper.
'''
    n['cells'][29]['source']=f'''## Arsitektur {experiment}: {label}

{description}

''' + ('''Gate g=σ(MLP([RoBERTa768, BGE256])). Concat aktual: [2g×RoBERTa768, 2(1−g)×BGE256] →MLP1024→256→1. Lapisan terakhir gate diinisialisasi nol: g=.5, sehingga fitur concat awal persis acuan E24. Bobot gate bukan bukti kausal kualitas jawaban.
''' if experiment=='E25' else '''RoBERTa token768 → BiGRU64 per arah → urutan128 → attention128→64→1 → weighted sum128 → Linear128→768 → LayerNorm → concat BGE256 → MLP. Berbeda dari E23 yang hanya mengambil state akhir. Mask mengecualikan konteks/special/padding dari pooling; encoder tetap membaca tiga teks.
''')
    codes=[source(c).removeprefix('%%aes\n') for c in n['cells'] if c['cell_type']=='code']
    for code in codes:ast.parse(code)
    fingerprint=hashlib.sha256('\n'.join(codes).encode()).hexdigest()
    for c in n['cells']:c['source']=source(c).replace('__NOTEBOOK_CODE_SHA256__',fingerprint)
    n['metadata']['colab']['name']=name+'.ipynb';n['metadata']['aes_standalone']['code_sha256']=fingerprint
    n['metadata']['aes_experiment'].update(name=experiment,base_commit=PIN,base_path=BASE,base_sha256=hashlib.sha256(raw).hexdigest(),s3=label,s3_pooling=mode,s3_fusion='gated_concat' if experiment=='E25' else 'concat',new_layer_lr=1e-4)
    p=ROOT/'notebooks/experiments'/(name+'.ipynb');p.write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n');(ROOT/'aes-learning/notebooks/experiments'/p.name).write_bytes(p.read_bytes());print(experiment,str(p.relative_to(ROOT)),fingerprint)
    return p

if __name__=='__main__':
    for e in VARIANTS:build(e)
