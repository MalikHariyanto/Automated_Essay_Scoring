"""Map the native E36 source into the existing three-input study reader."""
import ast
import hashlib
from build_mlp_study import highlighted_lines, source
from e21_study_content import curriculum, HINTS


def native_content(snapshot, notebook):
    chapters, glossary = curriculum({**snapshot, 'id': 'E30'})
    for chapter in chapters:
        chapter['cells'] = []
        for key in ('title', 'goal'):
            chapter[key] = chapter[key].replace('E30', 'E36')
        for key in ('read', 'flow'):
            chapter[key] = [s.replace('E30', 'E36') for s in chapter[key]]
    chapters[0].update(goal='Baca hasil S3 saja pada run E36.', flow=['Tiga input', 'S3 saja', '5fold CV', '60test'])
    chapters[0]['read'] = ['E36 hanya menjalankan S3. S1/S2 tidak dilatih atau dievaluasi; kedua cabang RoBERTa/BGE di S3 tetap ada.',
        'LR encoder3e-6; MLP, proyeksi BGE dan kelompok BiGRU/co-attention/pooling/norm1e-4. Metode mengikuti E30; format Python native mengikuti E35.',
        'Hasil berasal dari ZIP unggahan. Mean CV dan RMSE test E36 belum mengalahkan E30; bukan bukti model tidak belajar.']
    chapters[2]['read'] = ['Setup dan fungsi berjalan langsung di kernel notebook Colab. Tidak ada magic command atau worker training tersembunyi.', 'Pilih GPU T4 lalu Run all. Identitas GPU/byte source runtime tidak dapat dibuktikan hanya dari lima Excel hasil.']
    chapters[3]['read'] = ['X adalah soal, jawaban_referensi, response_text. y adalah rater0–100; tidak masuk encoder.', 'Hanya LR encoder RoBERTa S3 berubah5e-6→3e-6. MLP, proyeksi BGE dan pooling/norm kembali ke E30 1e-4.', 'SCENARIOS=(S3,) mengatur seluruh run; tidak ada latihan atau checkpoint S1/S2.']
    chapters[3]['flow'] = ['Checkpoint', 'X dan y', 'LR encoder3e-6', 'Layer/head1e-4']
    chapters[9]['read'] = ['Token jawaban dirangkum menjadi kata memakai offset. Soal/referensi tetap dibaca sebagai konteks.', 'Inspeksi menampilkan selisih komponen dan kesamaan numerik dalam encoder yang sama, tanpa cosine. Vektor bukan skor guru.', 'Vektor kata setelah training berasal dari cabang RoBERTa S3; fusion bekerja pada ringkasan jawaban.']
    chapters[10]['title'] = 'Arsitektur S3 · dua cabang, satu nilai'
    chapters[10]['read'][0] = 'RoBERTa fine-tuned dan BGE frozen merupakan dua cabang S3. Tidak membutuhkan model S1/S2 yang sudah dilatih.'
    chapters[12]['flow'] = ['5hasil fold S3', 'Kunci epoch CV', 'Refit240jawaban', 'Test60jawaban']
    chapters[12]['read'] = ['Lima fold development memilih epoch melalui RMSE validasi. Refit menggunakan ceil median epoch terpilih.', 'Data test tidak memilih epoch atau LR. Perbandingan E21–E36 bersifat eksploratif karena test lama sudah sering dilihat.']
    for chapter in chapters:
        for field in ('read','flow'):
            chapter[field]=[v.replace('S1/S2/S3','S3').replace('S1/S3','cabang RoBERTa S3').replace('S2 melatih MLP.','S2 tidak dijalankan pada E36.') for v in chapter[field]]
        chapter['terms']=[t for t in chapter['terms'] if t.lower()!='cosine']
    glossary.pop('cosine',None)
    for chapter in chapters:
        if chapter.get('book'):
            available={s['name'] for s in snapshot['books'][chapter['book']-1]['sheets']}
            chapter['sheets']=[s for s in chapter['sheets'] if s in available]
    glossary['learning rate'] = ('Ukuran langkah pembaruan', 'Encoder S3 E36 memakai3e-6; E30 memakai5e-6. Semua kelompok lain1e-4. Langkah lebih kecil tidak otomatis memberi hasil lebih baik.')
    groups = {
        1: {'storage_and_upload'}, 2: {'dependency_setup','imports'},
        3: {'model_names','input_columns','hyperparameters','runtime_flags','configuration_defaults','configuration_apply','presentation_helpers','definition:show'},
        4: {'dataset_definitions','dataset_execute_xy','definition:excel_value','definition:write_excel','definition:vector_frame','definition:ExcelJournal'},
        5: {'definition:make_split','definition:contexts_from_bundle','split_execute'},
        6: {'definition:content_windows','definition:batch_windows','definition:mean_by_essay','definition:context_token_ids','definition:reference_context_positions','definition:pair_content_windows','definition:pair_batch_windows','definition:pair_token_frames','definition:paired_token_vectors','definition:paired_pretrained_values','definition:export_tokens','token_export_execute'},
        7: {'definition:device_name','definition:release','definition:ExpectedPoolerNotice','definition:encoder','definition:tokenizer','definition:roberta_pretrained_vectors','pretrained_roberta_execute'},
        8: {'definition:bge_features','bge_execute'},
        9: {'word_helpers','definition:sentence_spans','definition:token_vectors','definition:word_vectors','definition:compare_repetitions','definition:vector_lab','definition:export_word_sentence_vectors','word_execute'},
        10: {'definition:answer_token_mask','definition:check_answer_pool_inputs','definition:AnswerBiGRUAttention','definition:reference_token_mask','definition:AnswerBiGRUCoAttentionPool','definition:SingleScoreModel','definition:Engine','definition:seed_everything','definition:make_model','architecture_export'},
        11: {'definition:optimizer_groups','definition:save_training_progress','definition:load_training_progress','definition:fit','definition:save_model','definition:load_model'},
        12: {'definition:train_all','definition:export_training_event','definition:test_score_comparison','definition:export_completed_training','training_execute'},
        13: {'definition:gwet_ac2','definition:metrics','definition:ordinal_copy','definition:prediction_frame','definition:score_answer','prediction_execute'},
        14: {'definition:export_final_features','final_features_execute'},
    }
    special = {
        'imports': ('SCENARIOS =', 'Hanya S3', 'Loop training, evaluasi dan ekspor hanya menjalankan S3. RoBERTa dan BGE tetap dipakai dalam fusion.'),
        'model_names': ('ROBERTA_CHECKPOINT =', 'Checkpoint model', 'Alamat bobot pretrained berbeda dari objek model yang dimuat encoder(). Revision mengikuti E30.'),
        'input_columns': ('TARGET_COLUMN =', 'Label bukan input', 'Tiga kolom teks masuk X. rater menjadi y untuk menghitung loss; tidak ditokenisasi.'),
        'hyperparameters': ('ENCODER_LR =', 'Satu perlakuan LR', 'Encoder3e-6. MLP/proyeksi/BiGRU/co-attention/pooling/norm1e-4. Maksimal25epoch,5fold,patience5.'),
        'definition:optimizer_groups': ('rates =', 'LR per kelompok', 'Setiap parameter trainable masuk optimizer tepat sekali. Hanya kelompok roberta memakai3e-6.'),
        'definition:SingleScoreModel': ('self.roberta_norm =', 'Ringkasan masuk fusion', 'Jalur S3 memakai pooling RoBERTa768, LayerNorm, proyeksi BGE256 dan concat1024 sebelum MLP.'),
    }
    cells, catalog = [], []
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        role = cell['metadata']['aes36_role']; text = source(cell); lines = text.splitlines()
        chapter = next((n for n, roles in groups.items() if role in roles), 15)
        chapters[chapter]['cells'].append(index)
        if role=='training_execute':chapters[0]['cells'].append(index)
        defs = [n for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        notes = []
        for n in defs:
            doc = ast.get_docstring(n)
            body = HINTS.get(n.name) or (doc.split('\n')[0] if doc else 'Definisi ini menyiapkan proses; pemanggilannya menjalankan bagian notebook terkait.')
            notes.append(dict(line=n.lineno, title=n.name+'()', body=body, term=None, kind='function'))
        if role in special:
            anchor, title, body = special[role]
            line = next(i+1 for i, value in enumerate(lines) if anchor in value)
            notes.append(dict(line=line,title=title,body=body,term=None,kind='fundamental'))
        if not notes:
            notes.append(dict(line=1,title='Eksekusi bagian ini',body='Kode berjalan langsung di kernel. Baca bersama penjelasan bab dan hasil Excel tahap yang sama.',term=None,kind='fundamental'))
        for j, note in enumerate(sorted(notes,key=lambda n:n['line'])):
            note.update(id=f'c{index}-n{j}',anchor=lines[note['line']-1].strip(),short=note['body'])
            catalog.append(dict(cell=index,ordinal=len(cells)+1,**{k:note[k] for k in ('line','title','short','anchor')}))
        cells.append(dict(index=index,ordinal=len(cells)+1,source=text,htmlLines=highlighted_lines(text),chapter=chapter,
            title=defs[0].name if defs else role,notes=notes,sha256=hashlib.sha256(text.encode()).hexdigest()))
    return dict(chapters=chapters,glossary=glossary,cells=cells,studySnapshot=snapshot,snapshot=snapshot,
        screenshotMappings=None,learningCatalog=catalog,learningLayout='margin',
        markdown=[dict(index=i,source=source(c)) for i,c in enumerate(notebook['cells']) if c['cell_type']=='markdown'],
        studySource=dict(commit=snapshot['source_commit'],path=snapshot['source_path'],experiment='E36',runtime_match=False),
        coverage=dict(code_cells=len(cells),code_lines=sum(len(c['htmlLines']) for c in cells),annotations=sum(len(c['notes']) for c in cells)))
