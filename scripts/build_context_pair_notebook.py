"""Buat eksperimen S3 context pair terpisah, tetap mandiri di Colab."""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'notebooks/experiments/AES_300_S3_MLP.ipynb'
DESTINATION = ROOT/'notebooks/experiments/AES_300_S3_ContextPair_MLP.ipynb'


def source(cell):
    return cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Anchor harus unik: '+old[:100])
    return text.replace(old, new)


def definitions(text):
    text = text.removeprefix('%%aes\n')
    return {n.name: ast.get_source_segment(text, n) for n in ast.parse(text).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def build():
    original = BASE.read_bytes()
    notebook = copy.deepcopy(json.loads(original))
    old_hash = notebook['metadata']['aes_standalone']['code_sha256']
    helper = definitions((ROOT/'src/aes/teacher_score/context_pair.py').read_text())
    manual = definitions((ROOT/'src/aes/teacher_score/manual_checkpoints.py').read_text())
    for cell in notebook['cells']:
        text = source(cell)
        if cell['cell_type'] == 'code':
            text = text.replace(old_hash, '__NOTEBOOK_CODE_SHA256__')
            text = text.replace('AES_300_MLP_Privat', 'AES_300_S3_ContextPair_MLP_Privat')
            text = text.replace('/content/AES_300_MLP', '/content/AES_300_S3_ContextPair_MLP')
            text = text.replace('outputs/AES_300_MLP', 'outputs/AES_300_S3_ContextPair_MLP')
            text = text.replace("lesson_version='s3_mlp_account2_v1'", "lesson_version='s3_context_pair_mlp_v1'")
            text = text.replace("'AES300_REQUIRE_GPU':'1' if IN_COLAB else os.environ.get('AES300_REQUIRE_GPU','0')", "'AES300_REQUIRE_GPU':'0'")
            text = text.replace("'new_answer', 'focus_word'", "'new_answer', 'focus_word', 'new_question', 'new_reference'")
            if 'DEFAULTS = {' in text:
                text = replace_once(text, "    'vector_source': 'pretrained', 'trace_position': 0,", """    # Batas konteks dihitung termasuk prefix; semua token jawaban tetap dicakup.
    'context_max_tokens': 192, 'question_max_tokens': 64,
    'new_question': '', 'new_reference': '',
    'vector_source': 'pretrained', 'trace_position': 0,""")
                text = text.replace("('pretrained', 'S1', 'S3')", "('pretrained', 'pretrained_pair', 'S1', 'S3')")
                text = replace_once(text, '    return result', """    if not (type(result['context_max_tokens']) is int and type(result['question_max_tokens']) is int
            and 1 <= result['question_max_tokens'] < result['context_max_tokens']
            and result['context_max_tokens']+4+result['roberta_overlap'] < result['roberta_max_length']):
        raise ValueError('Anggaran konteks harus menyisakan token jawaban lebih banyak daripada overlap.')
    if str(result.get('new_answer', '')).strip() and not all(str(result.get(k, '')).strip() for k in ('new_question', 'new_reference')):
        raise ValueError('Untuk NEW_ANSWER baru, isi juga NEW_QUESTION dan NEW_REFERENCE.')
    return result""")
            if 'def friendly_header(' in text:
                text = replace_once(text, 'def friendly_header(', """LABELS.update({'input_segment':'Bagian Input','question_tokens':'Token Soal Awal',
    'question_kept':'Token Soal Dibaca','reference_tokens':'Token Referensi Awal',
    'reference_kept':'Token Referensi Dibaca','question_truncated':'Soal Terpotong?',
    'reference_truncated':'Referensi Terpotong?','context_tokens_kept':'Token Konteks Dibaca',
    'content_tokens':'Token Jawaban','capacity':'Kapasitas Jawaban per Window',
    'stride':'Langkah Window','starts':'Posisi Awal Window','windows':'Jumlah Window',
    'pair_special_tokens':'Token Khusus Pasangan','answer_tokens_uncovered':'Token Jawaban Tidak Tercakup',
    'context_max_tokens':'Batas Token Konteks S3','question_max_tokens':'Batas Token Soal S3'})

def friendly_header(""")
            if 'ROBERTA_CHECKPOINT=' in text:
                text = replace_once(text, "NEW_ANSWER=''", """# Jawaban baru memerlukan soal dan referensi; kosong berarti contoh data uji.
NEW_QUESTION=''
NEW_REFERENCE=''
CONTEXT_MAX_TOKENS,QUESTION_MAX_TOKENS=192,64
NEW_ANSWER=''""")
                text = replace_once(text, "device='auto',new_answer=NEW_ANSWER,", """device='auto',new_answer=NEW_ANSWER,new_question=NEW_QUESTION,new_reference=NEW_REFERENCE,
                context_max_tokens=CONTEXT_MAX_TOKENS,question_max_tokens=QUESTION_MAX_TOKENS,""")
                text = replace_once(text, "gpu=check_gpu(require_gpu=RUN_MODELS and os.environ.get('AES300_REQUIRE_GPU')=='1')", """REQUIRE_GPU=False  # CPU boleh; pilih T4 untuk mempercepat training.
gpu=check_gpu(require_gpu=RUN_MODELS and REQUIRE_GPU)
if RUN_MODELS and not gpu['cuda_available']:
    print('Mode CPU aktif; training lebih lama. Progres tetap disimpan otomatis.', flush=True)""")
            if 'def read_dataset(' in text:
                text = text.replace("'input_features': ['response_text']", "'input_features': ['response_text', 'soal (S3)', 'jawaban_referensi (S3)']")
                text = text.replace('Pertanyaan/referensi/email/nilai tidak masuk fitur.', 'S3 memakai soal/referensi dan jawaban; S1/S2 hanya jawaban. Email/nilai tidak masuk fitur encoder.')
            if 'trace_text=cfg.get(' in text:
                text += """
# Konteks diselaraskan dengan jawaban contoh, bukan dipilih dari skor.
trace_context=(clean(cfg['new_question']),clean(cfg['new_reference'])) if cfg.get('new_answer','').strip() else contexts_from_bundle(bundle)[trace_index]
"""
                # Helper sudah harus tersedia saat menentukan contoh, sebelum cell window.
                text = text.replace('trace_context=', helper['contexts_from_bundle']+'\n\ntrace_context=', 1)
            if 'def content_windows(' in text:
                text += '\n\n'+'\n\n\n'.join(v for k, v in helper.items() if k != 'contexts_from_bundle')+'\n'
            if 'def export_tokens(' in text:
                text = replace_once(text, "    journal.save('04_tokenisasi',", """    pair_frame,pair_coverage=pair_token_frames(bundle,cfg,roberta_tok)
    statistics.extend(pair_coverage)
    journal.save('04_tokenisasi',""")
                text = replace_once(text, "'BGE_input': pd.DataFrame(rows),", "'BGE_input': pd.DataFrame(rows), 'RoBERTa_S3_pair_input': pair_frame,")
                text = replace_once(text, '    return roberta_frame, pd.DataFrame(rows)', "    show(pair_frame,'Token S3: konteks dan jawaban')\n    return roberta_frame, pd.DataFrame(rows)")
            if 'def roberta_pretrained_vectors(' in text:
                text += """
if RUN_MODELS:
    roberta_pair_before=roberta_pair_pretrained_vectors(bundle,cfg,journal)
    show(vector_frame(roberta_pair_before,[r['row_id'] for r in bundle.rows]),'RoBERTa S3: vektor pasangan sebelum training')
"""
            if 'def vector_lab(' in text:
                text = replace_once(text, '    # Artifact fast RoBERTa', """    if role=='roberta' and cfg.get('pair_context') is not None:
        return paired_token_vectors(text,model,slow,fast,cfg,device,cfg['pair_context'])
    # Artifact fast RoBERTa""")
                text = text.replace('def vector_lab(text, cfg, output):', 'def vector_lab(text, cfg, output, context=None):')
                text = text.replace("cfg['vector_source'] != 'pretrained'", "cfg['vector_source'] not in ('pretrained','pretrained_pair')")
                text = replace_once(text, '        trace = token_vectors(text, model, slow, fast, role, cfg, device)', """        paired=role=='roberta' and cfg['vector_source'] in ('pretrained_pair','S3')
        if paired and context is None:
            raise ValueError('Vektor S3 context pair memerlukan soal dan referensi.')
        trace_cfg={**cfg,'pair_context':context} if paired else cfg
        trace = token_vectors(text, model, slow, fast, role, trace_cfg, device)""")
                text = text.replace('token_vectors(text[start:end], model, slow, fast, role, cfg, device)', 'token_vectors(text[start:end], model, slow, fast, role, trace_cfg, device)')
                text = replace_once(text, "        dump(destination/f'{role}_metadata.json', item)", """        item['input_policy']='soal/referensi berpasangan dengan jawaban' if paired else 'jawaban siswa saja'
        if paired:
            item['sentence_method']='setiap kalimat dipasangkan dengan konteks yang sama; mean window CLS'
            item['source']+='; input context pair'
        dump(destination/f'{role}_metadata.json', item)""")
                text = text.replace("label += ' sebelum latihan' if cfg['vector_source']=='pretrained' else f' {cfg[\"vector_source\"]} setelah latihan'", "label += ' context pair sebelum latihan' if cfg['vector_source']=='pretrained_pair' else ' sebelum latihan' if cfg['vector_source']=='pretrained' else f' {cfg[\"vector_source\"]} setelah latihan'")
                text += """
if RUN_WORD_VECTORS:
    before_pair_words=vector_lab(trace_text,{**cfg,'vector_source':'pretrained_pair'},output,context=trace_context)
    export_word_sentence_vectors(before_pair_words,journal,'07b_vektor_pair_pretrained')
"""
            if 'class Engine:' in text:
                text = text.replace('roberta_tokenizer=None):', 'roberta_tokenizer=None, contexts=None):')
                text = replace_once(text, '        self.tokenizer = roberta_tokenizer', """        self.contexts = contexts
        if model.scenario=='S3' and (contexts is None or len(contexts)!=len(texts)):
            raise ValueError('S3 membutuhkan satu pasangan soal/referensi untuk setiap jawaban.')
        self.tokenizer = roberta_tokenizer""")
                text = replace_once(text, "        windows, mapping, _ = batch_windows([self.texts[i] for i in ids], self.tokenizer,\n                                             self.device, self.cfg['roberta_max_length'], self.cfg['roberta_overlap'])", """        if self.model.scenario=='S3':
            # Pasangan konteks-jawaban hanya masuk cabang RoBERTa S3.
            windows,mapping,_=pair_batch_windows([self.texts[i] for i in ids],[self.contexts[i] for i in ids],
                self.tokenizer,self.device,self.cfg['roberta_max_length'],self.cfg['roberta_overlap'],self.cfg)
        else:
            windows,mapping,_=batch_windows([self.texts[i] for i in ids],self.tokenizer,
                self.device,self.cfg['roberta_max_length'],self.cfg['roberta_overlap'])""")
            if 'specifications=pd.DataFrame' in text:
                text = text.replace("'RoBERTa ikut dilatih; BGE-M3 tetap'", "'RoBERTa context pair ikut dilatih; BGE-M3 jawaban saja tetap'")
            if 'def save_training_progress(' in text:
                parts = definitions(text)
                text = '%%aes\n# Progres disimpan setiap epoch; seluruh penghapusan checkpoint dilakukan manual.\n\n'+'\n\n\n'.join(manual[name] if name=='save_training_progress' else value for name,value in parts.items() if name!='clear_training_progress')+'\n'
            if 'def train_all(' in text:
                text = '\n'.join(line for line in text.splitlines() if not line.lstrip().startswith('clear_training_progress('))+'\n'
            if 'def score_answer(' in text:
                text = text.replace('def score_answer(text, cfg, output):', 'def score_answer(text, cfg, output, question, reference):')
                text = text.replace("sample = Bundle([{'text': text}]", "sample = Bundle([{'text': text, 'question': clean(question), 'reference': clean(reference)}]")
                text = text.replace('Engine(model, [text], features, cfg, device_name(cfg))', 'Engine(model, [text], features, cfg, device_name(cfg), contexts=contexts_from_bundle(sample))')
                text = text.replace('score_answer(trace_text,cfg,output)', 'score_answer(trace_text,cfg,output,*trace_context)')
                text = text.replace("'Input_privat':pd.DataFrame([{'text':trace_text}])", "'Input_privat':pd.DataFrame([{'text':trace_text}]),\n                                           'Konteks_privat':pd.DataFrame([{'soal':trace_context[0],'jawaban_referensi':trace_context[1]}])")
            text = text.replace('Engine(model, bundle.texts, bge, cfg, device)', 'Engine(model, bundle.texts, bge, cfg, device, contexts=contexts_from_bundle(bundle))')
            text = text.replace('Engine(model, bundle.texts, bge, cfg, device_name(cfg))', 'Engine(model, bundle.texts, bge, cfg, device_name(cfg), contexts=contexts_from_bundle(bundle))')
            text = text.replace('vector_lab(trace_text,{**cfg,\'vector_source\':scenario},output)', 'vector_lab(trace_text,{**cfg,\'vector_source\':scenario},output,context=trace_context)')
            if 'def data_report(' in text:
                old_data = definitions(text)['data_report']
                new_data = replace_once(old_data, "    return sheets if len(sheets)>1 else None", """    file='04_tokenisasi.xlsx'
    if source.has(file,'RoBERTa_S3_pair_input'):
        sheets['Token RoBERTa S3 Pair']=source.copy(file,'RoBERTa_S3_pair_input',['row_id','window','position','token','token_id','input_segment'])
    if source.has(file,'Panjang_dan_coverage'):
        sheets['Cakupan Token dan Konteks']=source.copy(file,'Panjang_dan_coverage')
    if source.has('13_prediksi_satu_jawaban.xlsx','Konteks_privat'):
        sheets['Konteks Jawaban Contoh']=source.copy('13_prediksi_satu_jawaban.xlsx','Konteks_privat')
    return sheets if len(sheets)>1 else None""")
                text = replace_once(text, old_data, new_data)
                text = text.replace("mappings=[('05_roberta_pretrained.xlsx'", "mappings=[('05b_roberta_pair_pretrained.xlsx','Vektor_jawaban_768','RoBERTa S3 Pair Awal'),\n              ('05_roberta_pretrained.xlsx'")
                text = text.replace("('14_S1_fitur_setelah_training.xlsx','roberta','RoBERTa S1 Setelah Latihan')", "('14_S1_fitur_setelah_training.xlsx','roberta','RoBERTa S1 Setelah Latihan')")
                text = text.replace("conditions=[(base,'roberta','RoBERTa Sebelum')", "conditions=[('07b_vektor_pair_pretrained.xlsx','roberta','S3 Pair Sebelum'),(base,'roberta','RoBERTa Sebelum')")
                text = text.replace("['RoBERTa','768 angka per jawaban. Bandingkan ID Jawaban yang sama sebelum dan sesudah latihan.']", "['RoBERTa','768 angka: S1 jawaban saja; S3 memakai pasangan. Bandingkan S3 sesudah latihan dengan S3 Pair Awal, agar input sama.']")
                text = text.replace("['BGE-M3','Ditampilkan sekali karena encodernya frozen. Rincian internal tetap tersimpan pada hasil sumber.']", "['BGE-M3','Jawaban saja, encoder frozen. Vektor kata S3 RoBERTa memakai konteks soal/referensi; vektor kalimat memakai konteks yang sama.']")
                text = text.replace("'projection_lr':'Learning rate proyeksi BGE'", "'context_max_tokens':'Batas token konteks S3','question_max_tokens':'Batas token soal S3',\n              'projection_lr':'Learning rate proyeksi BGE'")
            cell['source'] = text
            cell['outputs'], cell['execution_count'] = [], None
        else:
            if text.startswith('# AES 300'):
                text = '''# AES 300 · S3 context pair + MLP — eksperimen terpisah

S1 tetap RoBERTa fine-tuned + head linear, hanya membaca jawaban siswa. S2 tetap BGE-M3 frozen + MLP, hanya membaca jawaban siswa. **S3** memakai RoBERTa fine-tuned dengan teks A = soal + jawaban referensi dan teks B = jawaban siswa; cabang BGE tetap jawaban saja. LayerNorm, proyeksi BGE 1024→256, concat 768+256 dan MLP tetap seperti MLP acuan.

Alur dan lima laporan Excel mengikuti MLP acuan: setup → dataset → split → tokenisasi → embedding → model → CV/refit/test → prediksi → vektor → unduhan. Maksimal 25 epoch, patience 5, encoder LR 1e-5, proyeksi/head LR 1e-4. Tidak menambahkan gated fusion, Ridge atau perubahan learning rate lain.

**Drive terpisah:** MyDrive/AES_300_S3_ContextPair_MLP_Privat. Progres otomatis disimpan tiap epoch; checkpoint lama tidak dihapus otomatis dan tidak menggunakan Drive API cleanup. CPU didukung; T4 dianjurkan untuk training. Jalankan dari atas dan pilih dataset fix_dataset_300.xlsx bila diminta. Notebook tidak clone/import proyek.

**Anggaran input S3:** maksimal 512 token total, termasuk special token pasangan. Konteks dibatasi 192 token; soal termasuk prefix maksimal 64 token, sisa anggaran konteks untuk referensi. Pemotongan konteks dilaporkan. Konteks diulang di setiap window; seluruh token jawaban dicakup dengan overlap 128. Nilai guru tidak masuk encoder.

**Batas kesimpulan:** S3 mendapat informasi tambahan dibanding S1/S2. Hasil ini menguji S3 + konteks, bukan bukti bahwa fusion saja lebih unggul. Test lama sudah dilihat; pilih konfigurasi melalui CV. Skor baru/prediksi belum tersedia sebelum notebook dijalankan. Dataset pribadi dan bobot tidak disertakan.

**Jangan hapus seluruh .pt:** pertahankan final.pt setiap skenario dan dua snapshot yang dirujuk progress_*.json untuk tahap yang masih berjalan. JSON bukan pengganti bobot. Checkpoint tahap yang telah selesai dapat dibersihkan manual setelah artefak hasilnya diperiksa; jangan mengosongkan Sampah Drive secara membabi buta.
'''
            if text.startswith('## 4 '):
                text += '\n\nS3 juga membaca soal dan jawaban referensi sebagai konteks. S1/S2 tetap jawaban saja; rater hanya target.\n'
            if text.startswith('## 6 '):
                text += '\n\nS3: pasangan dibentuk oleh tokenizer, bukan menulis [SEP] sebagai teks biasa. Pada RoBERTa formatnya <s>konteks</s></s>jawaban</s>. Tabel menandai bagian konteks, jawaban dan special token. Pemotongan konteks serta cakupan jawaban ditampilkan di Excel.\n'
            if text.startswith('## 7 '):
                text += '\n\nDua pembanding awal disimpan: jawaban saja untuk S1 dan context pair untuk S3. Dimensi keduanya tetap 768.\n'
            if text.startswith('## 9 '):
                text += '\n\nTambahan S3 Pair Sebelum memakai bobot pretrained dengan pasangan input yang sama. Kata S3 sesudah training dipetakan hanya ke token jawaban, tetapi hidden state-nya membaca soal/referensi juga. Tidak dibuat vektor fusion per kata.\n'
            if text.startswith('## 10 '):
                text += '\n\nS3 RoBERTa tetap fine-tuned. Yang berubah dari acuan adalah input berpasangan, bukan jenis head.\n'
            if text.startswith('## 12 '):
                text += '\n\nCheckpoint setiap epoch dan fold tetap disimpan untuk dibersihkan manual. Jalankan ulang dari atas pada runtime baru untuk memulihkan konfigurasi dan melanjutkan progres yang cocok.\n'
            if text.startswith('## 13 '):
                text += '\n\nUntuk NEW_ANSWER, isi NEW_QUESTION dan NEW_REFERENCE di konfigurasi. Jika kosong, jawaban contoh dan konteksnya diambil dari baris held-out yang sama.\n'
            cell['source'] = text
        cell['metadata'] = {}
        cell.pop('attachments', None)
    # Hash source mentah agar Python 3.12/3.13 menghasilkan identitas yang sama.
    normalized = [source(c).removeprefix('%%aes\n')
                  for c in notebook['cells'] if c['cell_type'] == 'code']
    for code in normalized:
        ast.parse(code)
    fingerprint = hashlib.sha256('\n'.join(normalized).encode()).hexdigest()
    for cell in notebook['cells']:
        cell['source'] = source(cell).replace('__NOTEBOOK_CODE_SHA256__', fingerprint)
    notebook['metadata']['aes_standalone']['code_sha256'] = fingerprint
    notebook['metadata']['colab']['name'] = DESTINATION.name
    notebook['metadata'].pop('accelerator', None)
    notebook['metadata']['aes_experiment'] = {'name': 'S3_CONTEXT_PAIR_MLP', 'base_sha256': hashlib.sha256(original).hexdigest(),
        's1': 'answer_only_fine_tuned_linear', 's2': 'answer_only_frozen_mlp',
        's3': 'roberta_context_pair_fine_tuned_bge_answer_only_frozen_mlp',
        'automatic_checkpoint_deletion': False, 'context_max_tokens': 192, 'question_max_tokens': 64,
        'drive_root': 'AES_300_S3_ContextPair_MLP_Privat'}
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n')
    mirror = ROOT/'aes-learning/notebooks/experiments'/DESTINATION.name
    mirror.parent.mkdir(parents=True, exist_ok=True)
    mirror.write_bytes(DESTINATION.read_bytes())
    assert BASE.read_bytes() == original
    print(json.dumps({'notebook': str(DESTINATION.relative_to(ROOT)), 'code_sha256': fingerprint,
                      'code_cells': len(normalized), 'base_unchanged': True}))
    return DESTINATION


if __name__ == '__main__':
    build()
