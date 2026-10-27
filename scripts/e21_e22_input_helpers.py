"""Dua format tiga input; kode ini dibenamkan, tidak diimpor notebook."""


def resolve_dataset_upload(workspace, configured='', search_paths=None, browser_upload=False):
    """Web Colab: pakai file yang ada, atau tampilkan Choose Files pada cell ini."""
    from pathlib import Path
    import hashlib
    import os
    import shutil
    workspace=Path(workspace).expanduser().resolve()
    destination=workspace/'data/dataset300/fix_dataset_300.xlsx'
    if configured:
        candidates=[Path(configured).expanduser().resolve()]
        if not candidates[0].is_file():
            raise FileNotFoundError('DATASET_FILE/AES300_DATASET belum ditemukan: '+str(candidates[0]))
    else:
        places=search_paths if search_paths is not None else [Path('/content'),Path.cwd(),workspace]
        candidates=[]
        for place in places:
            path=(Path(place)/'fix_dataset_300.xlsx').resolve()
            if path.is_file() and path not in candidates and path!=destination:
                candidates.append(path)
        if not candidates and destination.is_file():
            candidates=[destination]
    if not candidates and browser_upload:
        from google.colab import files
        destination.parent.mkdir(parents=True,exist_ok=True)
        print('Klik Choose Files, lalu pilih fix_dataset_300.xlsx dari komputer.',flush=True)
        previous=Path.cwd()
        try:
            os.chdir(destination.parent)
            uploaded=files.upload()
        finally:
            os.chdir(previous)
        if 'fix_dataset_300.xlsx' not in uploaded:
            raise ValueError('Pilih fix_dataset_300.xlsx. Jika dibatalkan, jalankan ulang cell setup ini.')
        candidates=[destination]
        del uploaded
    if not candidates:
        raise FileNotFoundError(
            'Dataset belum ada. Di web Colab, jalankan cell setup ini dan pilih '
            'fix_dataset_300.xlsx melalui Choose Files. Alternatif: panel Files/Berkas kiri '
            '→ Upload, lalu jalankan ulang cell ini. Jika lokasinya berbeda, isi DATASET_FILE.')
    hashes={hashlib.sha256(p.read_bytes()).hexdigest() for p in candidates}
    if len(hashes)>1:
        raise ValueError('Ada beberapa dataset berbeda: '+', '.join(map(str,candidates))+
                         '. Isi DATASET_FILE dengan path file yang ingin dipakai.')
    source=candidates[0]
    if not source.stat().st_size:
        raise ValueError('File upload masih kosong; tunggu upload selesai lalu jalankan ulang cell ini.')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if source!=destination:
        shutil.copy2(source,destination)
    print('Dataset dari upload lokal:',source,'→',destination,flush=True)
    return destination


def pair_content_windows(text, context, tokenizer, maximum=512, overlap=128, cfg=None):
    """Informasi dan window sama; E22 menambah separator resmi pasangan."""
    if cfg is None or cfg.get('input_format') not in ('single_sequence', 'context_pair'):
        raise ValueError('Pilih format input E21 atau E22 sebelum tokenisasi.')
    context_ids, context_info = context_token_ids(context, tokenizer, cfg)
    ids = tokenizer.encode(text, add_special_tokens=False)
    answer_prefix = tokenizer.encode('\nJawaban siswa:\n', add_special_tokens=False)
    if not ids:
        raise ValueError('Jawaban tidak menghasilkan token.')
    paired = cfg['input_format'] == 'context_pair'
    specials = tokenizer.num_special_tokens_to_add(pair=paired)
    # Cadangkan overhead terbesar untuk KEDUA eksperimen. Starts/cakupan sama.
    reserved = max(tokenizer.num_special_tokens_to_add(pair=False),
                   tokenizer.num_special_tokens_to_add(pair=True))
    capacity = maximum-len(context_ids)-len(answer_prefix)-reserved
    if not 0 <= overlap < capacity:
        raise ValueError('Konteks dan penanda jawaban menyisakan kapasitas di bawah overlap.')
    starts = [0]
    while starts[-1]+capacity < len(ids):
        starts.append(starts[-1]+capacity-overlap)
    chunks, answer_positions, segments = [], [], []
    for start in starts:
        answer = ids[start:start+capacity]
        second = answer_prefix+answer
        if paired:
            prepared = tokenizer.build_inputs_with_special_tokens(context_ids, second)
            layout = tokenizer.build_inputs_with_special_tokens([-1]*len(context_ids),
                     [-3]*len(answer_prefix)+[-2]*len(answer))
        else:
            prepared = tokenizer.build_inputs_with_special_tokens(context_ids+second)
            layout = tokenizer.build_inputs_with_special_tokens(
                     [-1]*len(context_ids)+[-3]*len(answer_prefix)+[-2]*len(answer))
        positions = [i for i, value in enumerate(layout) if value == -2]
        if (len(prepared) != len(layout) or len(prepared) > maximum
                or len(prepared)-len(context_ids)-len(second) != specials
                or [prepared[i] for i in positions] != answer):
            raise ValueError('Template tokenizer tidak konsisten dengan format input.')
        chunks.append({'input_ids':prepared, 'attention_mask':[1]*len(prepared)})
        answer_positions.append(positions)
        segments.append(['Soal/referensi' if v == -1 else 'Jawaban siswa' if v == -2
                         else 'Penanda jawaban' if v == -3 else 'Token khusus' for v in layout])
    stats = {**context_info, 'input_format':cfg['input_format'], 'content_tokens':len(ids),
             'capacity':capacity, 'stride':capacity-overlap, 'starts':starts,
             'windows':len(chunks), 'pair_special_tokens':specials,
             'reserved_special_tokens':reserved, 'answer_prefix_tokens':len(answer_prefix),
             'answer_positions':answer_positions, 'segments':segments, 'answer_tokens_uncovered':0}
    return chunks, stats
