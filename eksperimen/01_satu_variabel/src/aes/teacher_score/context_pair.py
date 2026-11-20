"""Pasangan konteks–jawaban khusus cabang RoBERTa S3.

Definisi fungsi disalin builder ke notebook mandiri. Data siswa tidak disertakan.
"""
import numpy as np
import torch

from aes.windows import mean_by_essay
from .models import device_name, encoder, release, tokenizer
from .excel import vector_frame
from .data import clean


def contexts_from_bundle(bundle):
    """Ambil soal/referensi tanpa membaca target skor guru."""
    contexts = []
    for row in bundle.rows:
        question, reference = row.get('question'), row.get('reference')
        if not isinstance(question, str) or not question.strip():
            raise ValueError('Context pair memerlukan soal yang tidak kosong.')
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError('Context pair memerlukan jawaban referensi yang tidak kosong.')
        contexts.append((clean(question), clean(reference)))
    return contexts


def context_token_ids(context, tokenizer, cfg):
    """Sisakan anggaran untuk soal dan referensi sekaligus."""
    if not isinstance(context, (tuple, list)) or len(context) != 2:
        raise ValueError('Konteks harus pasangan (soal, jawaban referensi).')
    question, reference = context
    if not all(isinstance(t, str) and t.strip() for t in context):
        raise ValueError('Isi soal dan jawaban referensi sebelum memprediksi.')
    limit, qlimit = cfg['context_max_tokens'], cfg['question_max_tokens']
    if not 1 <= qlimit < limit:
        raise ValueError('Anggaran token soal harus lebih kecil daripada konteks.')
    # Prefix menjadi bagian input; nilai guru tidak pernah dimasukkan.
    qids = tokenizer.encode('Soal: '+question, add_special_tokens=False)
    rids = tokenizer.encode('\nJawaban referensi: '+reference, add_special_tokens=False)
    question_kept = qids[:qlimit]
    reference_kept = rids[:limit-len(question_kept)]
    return question_kept+reference_kept, {
        'question_tokens': len(qids), 'question_kept': len(question_kept),
        'reference_tokens': len(rids), 'reference_kept': len(reference_kept),
        'question_truncated': len(question_kept) < len(qids),
        'reference_truncated': len(reference_kept) < len(rids),
        'context_tokens_kept': len(question_kept)+len(reference_kept),
    }


def pair_content_windows(text, context, tokenizer, maximum=512, overlap=128, cfg=None):
    """Ulangi konteks tetap; seluruh token jawaban mendapat window."""
    if cfg is None:
        raise ValueError('Konfigurasi anggaran konteks diperlukan.')
    context_ids, context_info = context_token_ids(context, tokenizer, cfg)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError('Jawaban tidak menghasilkan token untuk context pair.')
    specials = tokenizer.num_special_tokens_to_add(pair=True)
    capacity = maximum-len(context_ids)-specials
    if not 0 <= overlap < capacity:
        raise ValueError('Konteks menyisakan token jawaban terlalu sedikit untuk overlap.')
    starts = [0]
    while starts[-1]+capacity < len(ids):
        starts.append(starts[-1]+capacity-overlap)
    chunks, answer_positions, segments = [], [], []
    for start in starts:
        answer = ids[start:start+capacity]
        prepared = tokenizer.build_inputs_with_special_tokens(context_ids, answer)
        # Sentinel hanya memetakan posisi; tidak pernah dikirim ke encoder.
        layout = tokenizer.build_inputs_with_special_tokens([-1]*len(context_ids), [-2]*len(answer))
        positions = [i for i, value in enumerate(layout) if value == -2]
        if (len(prepared) != len(layout) or len(prepared) > maximum
                or len(prepared)-len(context_ids)-len(answer) != specials
                or [prepared[i] for i in positions] != answer):
            raise ValueError('Template special token pasangan tidak konsisten.')
        chunks.append({'input_ids': prepared, 'attention_mask': [1]*len(prepared)})
        answer_positions.append(positions)
        segments.append(['Soal/referensi' if v == -1 else 'Jawaban siswa' if v == -2 else 'Token khusus'
                         for v in layout])
    stats = {**context_info, 'content_tokens': len(ids), 'capacity': capacity,
             'stride': capacity-overlap, 'starts': starts, 'windows': len(chunks),
             'pair_special_tokens': specials, 'answer_positions': answer_positions,
             'segments': segments, 'answer_tokens_uncovered': 0}
    return chunks, stats


def pair_batch_windows(texts, contexts, tokenizer, device='cpu', maximum=512, overlap=128, cfg=None):
    """Susun batch pasangan tanpa menukar konteks antarjawaban."""
    if len(texts) != len(contexts) or not texts:
        raise ValueError('Jumlah konteks harus sama dengan jumlah jawaban.')
    chunks, mapping, stats = [], [], []
    for index, (text, context) in enumerate(zip(texts, contexts)):
        values, detail = pair_content_windows(text, context, tokenizer, maximum, overlap, cfg)
        chunks.extend(values)
        mapping.extend([index]*len(values))
        stats.append(detail)
    batch = tokenizer.pad(chunks, padding=True, return_tensors='pt')
    return {key: value.to(device) for key, value in batch.items()}, torch.tensor(mapping, device=device), stats


def pair_token_frames(bundle, cfg, tok):
    """Simpan token pasangan aktual dan cakupan setiap jawaban."""
    import pandas as pd
    rows, coverage = [], []
    for row, text, context in zip(bundle.rows, bundle.texts, contexts_from_bundle(bundle)):
        chunks, stats = pair_content_windows(text, context, tok, cfg['roberta_max_length'], cfg['roberta_overlap'], cfg)
        coverage.append({'row_id': row['row_id'], 'model': 'RoBERTa S3 context pair',
                         **{k: v for k, v in stats.items() if k not in ('answer_positions', 'segments')}})
        for window, chunk in enumerate(chunks):
            pieces = tok.convert_ids_to_tokens(chunk['input_ids'])
            for position, (identifier, token, segment) in enumerate(zip(chunk['input_ids'], pieces, stats['segments'][window])):
                rows.append({'row_id': row['row_id'], 'window': window, 'position': position,
                             'token': token, 'token_id': identifier, 'attention_mask': 1, 'input_segment': segment})
    return pd.DataFrame(rows), coverage


def roberta_pair_pretrained_vectors(bundle, cfg, journal):
    """Ekstrak pembanding pretrained dengan pasangan input yang sama."""
    device = device_name(cfg)
    model, tok = encoder(cfg, 'roberta').to(device), tokenizer(cfg, 'roberta')
    model.eval()
    vectors = []
    with torch.no_grad():
        for text, context in zip(bundle.texts, contexts_from_bundle(bundle)):
            windows, mapping, _ = pair_batch_windows([text], [context], tok, device,
                                                    cfg['roberta_max_length'], cfg['roberta_overlap'], cfg)
            cls = model(**windows).last_hidden_state[:, 0, :].float()
            vectors.append(mean_by_essay(cls, mapping, 1).cpu().numpy())
    values = np.concatenate(vectors)
    journal.save('05b_roberta_pair_pretrained',
                 {'Vektor_jawaban_768': vector_frame(values, [r['row_id'] for r in bundle.rows])},
                 'RoBERTa pretrained dengan pasangan konteks-jawaban; belum fine-tuning')
    del model, tok
    release()
    return values


def paired_token_vectors(text, model, slow, fast, cfg, device, context):
    """Vektor kata jawaban memakai konteks persis jalur training."""
    aligned = fast(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = slow.encode(text, add_special_tokens=False)
    if list(aligned['input_ids']) != ids:
        raise ValueError('ID fast/slow jawaban berbeda; offset tidak ditebak.')
    offsets = [tuple(value) for value in aligned['offset_mapping']]
    if len(offsets) != len(ids):
        raise ValueError('Offset jawaban tidak cocok dengan jumlah token.')
    chunks, stats = pair_content_windows(text, context, slow, cfg['roberta_max_length'], cfg['roberta_overlap'], cfg)
    dim = model.config.hidden_size
    sums = np.zeros((len(ids), dim), dtype=np.float32)
    counts = np.zeros(len(ids), dtype=np.int64)
    initial = np.full_like(sums, np.nan)
    cls_vectors = []
    model.eval()
    with torch.no_grad():
        for chunk, start, positions in zip(chunks, stats['starts'], stats['answer_positions']):
            tensor = torch.tensor([chunk['input_ids']], dtype=torch.long, device=device)
            hidden = model(input_ids=tensor, attention_mask=torch.ones_like(tensor)).last_hidden_state
            stop = start+len(positions)
            sums[start:stop] += hidden[0, positions].float().cpu().numpy()
            counts[start:stop] += 1
            initial[start:stop] = model.get_input_embeddings()(tensor[:, positions])[0].float().cpu().numpy()
            cls_vectors.append(hidden[0, 0].float().cpu().numpy())
    contextual = sums/counts[:, None]
    return {'ids': ids, 'offsets': offsets, 'initial': initial, 'contextual': contextual.astype(np.float32),
            'counts': counts, 'starts': stats['starts'], 'capacity': stats['capacity'],
            'cls': np.mean(cls_vectors, axis=0)}
