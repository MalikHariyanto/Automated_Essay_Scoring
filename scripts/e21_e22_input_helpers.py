"""Dua format tiga input; kode ini dibenamkan, tidak diimpor notebook."""


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
