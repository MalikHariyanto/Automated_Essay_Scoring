"""E6 tiga input seluruh skenario; E6/E8 lama tetap utuh."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
from build_context_pair_notebook import source, replace_once, definitions

ROOT=Path(__file__).resolve().parents[1]
PIN='2257cb0f601148764d7844fc6ffca925a9b2ab95'
BASE='notebooks/experiments/AES_300_E8_ContextPair_LR5e6_CPU.ipynb'
DESTINATION=ROOT/'notebooks/experiments/AES_300_E6_Tiga_Input_CPU.ipynb'


def build():
    raw=subprocess.check_output(['git','show',PIN+':'+BASE],cwd=ROOT)
    n=copy.deepcopy(json.loads(raw));old=n['metadata']['aes_standalone']['code_sha256']
    helper=definitions((ROOT/'scripts/three_input_helpers.py').read_text())
    replacements={20:['pair_token_frames'],22:['export_tokens'],24:['roberta_pretrained_vectors'],
                  26:['bge_features'],28:['token_vectors']}
    for index,cell in enumerate(n['cells']):
        s=source(cell).replace(old,'__NOTEBOOK_CODE_SHA256__')
        s=s.replace('AES_300_E8_ContextPair_LR5e6_CPU','AES_300_E6_Tiga_Input_CPU')
        s=s.replace("lesson_version='e8_context_pair_lr5e6_cpu_v1'","lesson_version='e6_three_inputs_all_scenarios_cpu_v1'")
        s=s.replace('Hasil_AES_E8_ContextPair_LR5e6_GPU.zip','Hasil_AES_E6_Tiga_Input_CPU.zip')
        s=s.replace('Hasil_AES_E8_ContextPair_LR5e6_CPU.zip','Hasil_AES_E6_Tiga_Input_CPU.zip')
        if cell['cell_type']=='code':
            for name in replacements.get(index,[]):
                s=replace_once(s,definitions(s)[name],helper[name])
            if index==10:
                s=replace_once(s,"'bge_max_length': 1024, 'bge_batch': 4,","'bge_max_length': 1024, 'bge_overlap': 128, 'bge_batch': 4,")
                s=replace_once(s,"    'vector_source': 'pretrained',", "    'input_policy':'three_texts_all_scenarios_v1',\n    'vector_source': 'pretrained',")
                s=replace_once(s,'    return result',"""    if not (type(result['bge_overlap']) is int and 0 <= result['bge_overlap']
            < result['bge_max_length']-result['context_max_tokens']-4):
        raise ValueError('Anggaran konteks BGE harus menyisakan kapasitas jawaban di atas overlap.')
    if result['input_policy']!='three_texts_all_scenarios_v1':
        raise ValueError('Notebook ini khusus tiga input pada semua skenario.')
    return result""")
            if index==12:
                s=s.replace('# E8: langkah encoder S3 diperkecil, input tetap konteks pair.','# E6 tiga input: LR encoder S3 tetap 5e-6.')
                s=s.replace("'Jawaban saja'","'Soal + referensi + jawaban'")
                s=s.replace("'Tidak dipakai; BGE jawaban saja'","'BGE: soal + referensi + jawaban'")
                s=s.replace("'Soal/referensi + jawaban'","'Soal + referensi + jawaban'")
                s=s.replace("'E8: learning rate dasar; proyeksi S3 tetap 1e-4'","'E6 tiga input: S1/S2/S3 dilatih baru; proyeksi S3 tetap 1e-4'")
            if index==14:
                s=s.replace("raise ValueError(f'Baris {excel_row}: input/label wajib kosong; tidak diisi otomatis.')",
                            "raise ValueError(f'Baris {excel_row}: kolom wajib kosong: '+', '.join(k for k in REQUIRED if raw[k] is None)+'; tidak diisi otomatis.')")
                anchor="            if not isinstance(raw['response_text'], str):"
                s=replace_once(s,anchor,"""            for key in ('soal','jawaban_referensi'):
                if not isinstance(raw[key],str) or not raw[key].strip():
                    raise ValueError(f'Baris {excel_row}: {key} harus teks yang tidak kosong.')
"""+anchor)
                s=s.replace("'question': str(raw['soal']), 'reference': str(raw['jawaban_referensi'])","'question': clean(raw['soal']), 'reference': clean(raw['jawaban_referensi'])")
                s=s.replace("['response_text', 'soal (S3)', 'jawaban_referensi (S3)']","['soal','response_text','jawaban_referensi']")
                s=s.replace('S3 memakai soal/referensi dan jawaban; S1/S2 hanya jawaban. Email/nilai tidak masuk fitur encoder.',
                            'Semua skenario memakai soal/referensi dan jawaban. Email hanya kelompok; nilai guru hanya target, bukan fitur encoder.')
            if index==16:
                s=replace_once(s,"'berubah':r['raw_text']!=r['text'],'skor_guru':r['score'],", "'soal':r['question'],'jawaban_referensi':r['reference'],\n                       'berubah':r['raw_text']!=r['text'],'skor_guru':r['score'],")
            if index==20:
                s=replace_once(s,definitions(s)['roberta_pair_pretrained_vectors'],'')
                s+='\n\n'+helper['paired_pretrained_values']+'\n'
            if index==24:
                # Tidak mengekstrak dua kali: input pretrained S1 dan S3 sama.
                tail="\nif RUN_MODELS:\n    roberta_pair_before=roberta_pair_pretrained_vectors(bundle,cfg,journal)"
                assert tail in s
                s=s[:s.index(tail)]
            if index==28:
                s=s.replace("paired=role=='roberta' and cfg['vector_source'] in ('pretrained_pair','S3')","paired=True")
                s=s.replace("Vektor S3 context pair memerlukan soal dan referensi.","Semua vektor tiga input memerlukan soal dan referensi.")
                s=s.replace("item['sentence_method']='setiap kalimat dipasangkan dengan konteks yang sama; mean window CLS'", "item['sentence_method']='setiap kalimat dipasangkan dengan konteks yang sama; mean window CLS'+('; L2 setelah agregasi' if role=='bge' else '')")
                s=s.replace("before_words=vector_lab(trace_text,{**cfg,'vector_source':'pretrained'},output)","before_words=vector_lab(trace_text,{**cfg,'vector_source':'pretrained'},output,context=trace_context)")
                tail="\nif RUN_WORD_VECTORS:\n    before_pair_words=vector_lab("
                assert tail in s
                s=s[:s.index(tail)]
            if index==30:
                s=s.replace("if model.scenario=='S3' and (contexts is None or len(contexts)!=len(texts)):","if contexts is None or len(contexts)!=len(texts):")
                s=s.replace('S3 membutuhkan satu pasangan soal/referensi untuk setiap jawaban.','Semua skenario memerlukan satu pasangan soal/referensi per jawaban.')
                start=s.index("        if self.model.scenario=='S3':")
                end=s.index('        return self.model(windows=',start)
                s=s[:start]+"""        # S1 dan S3 membaca tiga teks yang sama; S2 memakai fitur BGE tiga input.
        windows,mapping,_=pair_batch_windows([self.texts[i] for i in ids],[self.contexts[i] for i in ids],
            self.tokenizer,self.device,self.cfg['roberta_max_length'],self.cfg['roberta_overlap'],self.cfg)
"""+s[end:]
            if index==32:
                s=s.replace('RoBERTa: bobot ikut dilatih','RoBERTa tiga input: bobot ikut dilatih')
                s=s.replace('BGE-M3: bobot tetap','BGE-M3 tiga input: bobot tetap')
                s=s.replace('RoBERTa context pair ikut dilatih; BGE-M3 jawaban saja tetap','RoBERTa tiga input ikut dilatih; BGE-M3 tiga input tetap')
            if index==45:
                s=s.replace("['row_id','teks_model','skor_guru']","['row_id','soal','teks_model','jawaban_referensi','skor_guru']")
                s=s.replace("keys=['row_id']+(['window'] if raw=='RoBERTa_input' else [])+['position','token','token_id']", "keys=['row_id','window','position','token','token_id','input_segment']")
                s=s.replace("mappings=[('05b_roberta_pair_pretrained.xlsx','Vektor_jawaban_768','RoBERTa S3 Pair Awal'),\n              ","mappings=[")
                s=s.replace("conditions=[('07b_vektor_pair_pretrained.xlsx','roberta','S3 Pair Sebelum'),(base", "conditions=[(base")
                s=s.replace('768 angka: S1 jawaban saja; S3 memakai pasangan. Bandingkan S3 sesudah latihan dengan S3 Pair Awal, agar input sama.',
                            '768 angka: S1/S3 sama-sama membaca pasangan soal/referensi + jawaban. Bandingkan sebelum/sesudah pada input tiga teks yang sama.')
                s=s.replace('Jawaban saja, encoder frozen. Vektor kata S3 RoBERTa memakai konteks soal/referensi; vektor kalimat memakai konteks yang sama.',
                            'BGE dan RoBERTa membaca soal/referensi + jawaban. Vektor kata merujuk kata jawaban, dengan konteks tetap tersedia; BGE frozen.')
                s=s.replace('Batas token konteks S3','Batas token konteks semua encoder').replace('Batas token soal S3','Batas token soal semua encoder')
            s=s.replace('Batas Token Konteks S3','Batas Token Konteks Semua Encoder').replace('Batas Token Soal S3','Batas Token Soal Semua Encoder')
            cell['outputs']=[];cell['execution_count']=None
        cell['source']=s;cell['metadata']={};cell.pop('attachments',None)
    n['cells'][0]['source']='''# AES 300 · E6 tiga input · CPU

Varian baru resep E6: **soal (`soal`), jawaban siswa (`response_text`), dan kunci jawaban (`jawaban_referensi`)** dibaca oleh seluruh S1/S2/S3. Ini berbeda dari E8 yang menambahkan konteks hanya ke RoBERTa S3. `rater` tetap satu target skor guru 0–100, tidak dimasukkan ke encoder. Tidak membuat empat label baru dari satu skor.

**Input:** teks A = soal + jawaban referensi, teks B = jawaban siswa. Kedua tokenizer membentuk special token pasangan sendiri. RoBERTa dan BGE membaca tiga kolom yang sama, dengan kosakata/batas token masing-masing. Konteks maksimal192token termasuk prefix, soal maksimal64token. Jawaban panjang dipecah menjadi window dan konteks diulang; seluruh token jawaban dicakup. Pemotongan soal/referensi dilaporkan. Soal dan referensi harus tersedia saat prediksi jawaban baru.

**Model:** S1 RoBERTa full fine-tuning → mean CLS window768 → dropout → linear1. S2 BGE-M3 frozen → mean CLS window1024 → L2 → MLP256/GELU/dropout/linear1. S3 RoBERTa768 + LayerNorm dan BGE1024 → proyeksi256 + LayerNorm → concat1024 → MLP256/GELU/dropout/linear1. Proyeksi/head belajar; encoder BGE tetap frozen. Agregasi BGE antar-window diperlukan bila input melebihi1024token; bila satu window tetap CLS lalu L2.

**Training mengikuti E6:** LR encoder S1=1e-5, S3=5e-6, head/proyeksi=1e-4; seed42, split_seed2026, development/test80/20,5fold, maksimal25epoch/fold, patience5, microbatch1/batch efektif8. S1/S2/S3 dilatih ulang karena seluruh input berubah; tidak reuse model jawaban-saja. CPU float32 dua thread, tanpa kuota GPU. Training penuh CPU akan lebih lama daripada E6 yang hanya melatih S3.

**Drive terpisah:** MyDrive/AES_300_E6_Tiga_Input_CPU_Privat. Autosave dua slot A/B per fold/refit, slot digunakan ulang tanpa Drive API atau auto-delete. Model final dan JSON disimpan. Jalankan notebook yang sama dari atas untuk memulihkan progres yang cocok setelah runtime berganti. Hasil E6/E8 lama tidak perlu dihapus.

**Jalankan dari atas**, unggah fix_dataset_300.xlsx jika diminta. Notebook berisi semua kode tanpa clone/import repository. Cell terakhir menyiapkan lima Excel dan Hasil_AES_E6_Tiga_Input_CPU.zip: data/tiga input, token dan cakupan kedua encoder, vektor awal/akhir, kata/kalimat, riwayat, nilai guru/prediksi/pembulatan, RMSE/Pearson/QWK/GwetAC2. Isi NEW_QUESTION, NEW_REFERENCE dan NEW_ANSWER untuk prediksi baru; bila kosong, gunakan contoh held-out beserta konteks baris yang sama.

**Batas interpretasi:** ini eksperimen input tiga teks, bukan bukti otomatis RMSE turun. Test lama sudah dilihat selama pengembangan; pilih metode melalui development/CV, bukan hasil test. Kelompok duplikat/email mencegah kebocoran jawaban; soal dapat muncul di kedua bagian sehingga hasil bukan uji generalisasi ke soal baru. Dataset/label asli tetap privat dan tidak disertakan notebook publik.
'''
    # Markdown lama E8 tidak boleh mengklaim jalur answer-only/reuse yang sudah berubah.
    chapters={1:'Ruang kerja & penyimpanan',2:'Setup lingkungan',3:'Konfigurasi E6 tiga input',
              4:'Baca soal, jawaban, referensi dan target',5:'Development, test dan lima fold',
              6:'Token pasangan untuk RoBERTa dan BGE',7:'RoBERTa tiga input sebelum latihan',
              8:'BGE tiga input frozen',9:'Vektor kata dan kalimat dengan konteks',
              10:'Arsitektur dan head tiga skenario',11:'MSE, optimizer dan checkpoint',
              12:'CV, refit dan evaluasi akhir',13:'Prediksi baru dengan tiga input',
              14:'Ekspor fitur checkpoint final',15:'Lima Excel dan ZIP hasil'}
    for cell in n['cells'][1:]:
        if cell['cell_type']!='markdown':continue
        s=source(cell)
        if s.startswith('## '):
            try:number=int(s.split()[1].rstrip('.'))
            except ValueError:continue
            cell['source']=f"## {number} {chapters[number]}\n\nSemua skenario membaca soal + jawaban referensi + jawaban siswa. Nilai guru hanya target. Definisi fungsi menyiapkan proses; pemanggilan di bawah cell menjalankannya.\n"
    codes=[source(c).removeprefix('%%aes\n') for c in n['cells'] if c['cell_type']=='code']
    for code in codes:ast.parse(code)
    fingerprint=hashlib.sha256('\n'.join(codes).encode()).hexdigest()
    for cell in n['cells']:cell['source']=source(cell).replace('__NOTEBOOK_CODE_SHA256__',fingerprint)
    n['metadata']['aes_standalone']['code_sha256']=fingerprint
    n['metadata']['colab']['name']=DESTINATION.name;n['metadata'].pop('accelerator',None)
    n['metadata']['aes_experiment']={'name':'E6_THREE_INPUTS_CPU','device':'cpu','precision':'float32',
        'base_commit':PIN,'base_path':BASE,'base_sha256':hashlib.sha256(raw).hexdigest(),
        'input_columns':['soal','response_text','jawaban_referensi'],'target':'rater',
        's1':'roberta_three_inputs_full_finetuning_linear','s2':'bge_three_inputs_frozen_mlp',
        's3':'roberta_and_bge_three_inputs_fusion_mlp','s1_encoder_lr':1e-5,'s3_encoder_lr':5e-6,
        'projection_lr':1e-4,'head_lr':1e-4,'reused_controls':[],
        'checkpoint_strategy':'two_slots_in_place','automatic_checkpoint_deletion':False,
        'drive_root':'AES_300_E6_Tiga_Input_CPU_Privat'}
    DESTINATION.write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n')
    (ROOT/'aes-learning/notebooks/experiments'/DESTINATION.name).write_bytes(DESTINATION.read_bytes())
    print(json.dumps({'notebook':str(DESTINATION.relative_to(ROOT)),'code_sha256':fingerprint,'blocks':len(codes)}))
    return DESTINATION


if __name__=='__main__':build()
