"""Laboratorium privat: lookup token, konteks kata, dan CLS kalimat kedua encoder."""
from collections import Counter
import itertools
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import normalize

from aes.cleaning import clean
from .data import digest, dump
from .models import device_name, encoder, release, tokenizer

WORD = re.compile(r"\b\w+(?:[-’']\w+)*\b", re.UNICODE)


def sentence_spans(text):
    """Segmentasi aturan sederhana, bukan model sintaksis; singkatan bisa terpisah."""
    result = []
    for match in re.finditer(r'[^.!?\n]+(?:[.!?]+|$)|[^.!?\n]+(?=\n)', text):
        start, end = match.span()
        while start < end and text[start].isspace():
            start += 1
        if start < end:
            result.append((start, end))
    return result


def token_vectors(text, model, slow, fast, role, cfg, device):
    # Artifact fast RoBERTa mempunyai postprocessor berbeda. Gunakan fast hanya
    # untuk offset TANPA specials, lalu builder slow yang dipakai training.
    aligned = fast(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = slow.encode(text, add_special_tokens=False)
    if list(aligned['input_ids']) != ids:
        raise ValueError('ID fast/slow berbeda; penyelarasan kata tidak boleh ditebak.')
    offsets = [tuple(x) for x in aligned['offset_mapping']]
    if len(offsets) != len(ids):
        raise ValueError('Jumlah offset tidak cocok token.')
    maximum = cfg['roberta_max_length'] if role == 'roberta' else cfg['bge_max_length']
    capacity = maximum-slow.num_special_tokens_to_add(False)
    starts = [0]
    if role == 'roberta':
        while starts[-1]+capacity < len(ids):
            starts.append(starts[-1]+capacity-cfg['roberta_overlap'])
    hidden = model.config.hidden_size
    sums = np.zeros((len(ids), hidden), dtype=np.float32)
    counts = np.zeros(len(ids), dtype=np.int64)
    initial = np.full_like(sums, np.nan)
    cls_vectors = []
    model.eval()
    with torch.no_grad():
        for start in starts:
            content = ids[start:start+capacity]
            prepared = slow.build_inputs_with_special_tokens(content)
            # Fast XLM-R menolak get_special_tokens_mask(...False). Tandai
            # posisi konten melalui builder yang sama; sentinel tidak masuk model.
            template = slow.build_inputs_with_special_tokens([-1]*len(content))
            positions = [i for i, v in enumerate(template) if v == -1]
            if len(positions) != len(content) or len(prepared) > maximum or [prepared[i] for i in positions] != content:
                raise ValueError('Builder special token tidak konsisten.')
            tensor = torch.tensor([prepared], dtype=torch.long, device=device)
            output = model(input_ids=tensor, attention_mask=torch.ones_like(tensor)).last_hidden_state
            cls_vectors.append(output[0, 0].float().cpu().numpy())
            stop = start+len(content)
            sums[start:stop] += output[0, positions].float().cpu().numpy()
            counts[start:stop] += 1
            lookup = model.get_input_embeddings()(tensor[:, positions])
            initial[start:stop] = lookup[0].float().cpu().numpy()
    contextual = np.full_like(sums, np.nan)
    np.divide(sums, counts[:, None], out=contextual, where=counts[:, None] > 0)
    sentence_cls = np.mean(cls_vectors, axis=0)
    if role == 'bge':
        sentence_cls = normalize(torch.from_numpy(sentence_cls)[None], dim=1)[0].numpy()
    return {'ids': ids, 'offsets': offsets, 'initial': initial, 'contextual': contextual,
            'counts': counts, 'starts': starts, 'capacity': capacity, 'cls': sentence_cls}


def word_vectors(text, trace):
    rows, initial, contextual = [], [], []
    for word in WORD.finditer(text):
        pieces = [i for i, (start, end) in enumerate(trace['offsets'])
                  if end > start and start < word.end() and end > word.start()]
        covered = bool(pieces) and all(trace['counts'][i] > 0 for i in pieces)
        dim = trace['initial'].shape[1]
        before = trace['initial'][pieces].mean(axis=0) if covered else np.full(dim, np.nan, dtype=np.float32)
        after = trace['contextual'][pieces].mean(axis=0) if covered else np.full(dim, np.nan, dtype=np.float32)
        rows.append({'word_index': len(rows), 'word': word.group(), 'start': word.start(), 'end': word.end(),
                     'token_indices': pieces, 'token_ids': [trace['ids'][i] for i in pieces],
                     'covered': covered, 'initial_first5': before[:5].tolist() if covered else None,
                     'contextual_first5': after[:5].tolist() if covered else None})
        initial.append(before)
        contextual.append(after)
    dim = trace['initial'].shape[1]
    return rows, np.asarray(initial, dtype=np.float32).reshape(-1, dim), np.asarray(contextual, dtype=np.float32).reshape(-1, dim)


def cosine(a, b):
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a, b)/denominator) if denominator > 0 else None


def compare_repetitions(rows, initial, contextual, focus=''):
    available = [r for r in rows if r['covered']]
    counts = Counter(r['word'].casefold() for r in available)
    target = focus.casefold().strip() if focus.strip() else next((w for w, n in counts.most_common() if n > 1), '')
    candidates = [r for r in available if r['word'].casefold() == target]
    results = []
    # Hindari tabel kuadratik pada kata sangat sering; semua vektor tetap disimpan.
    for left, right in itertools.islice(itertools.combinations(candidates, 2), 30):
        a, b = left['word_index'], right['word_index']
        results.append({'word': target, 'occurrence_indices': [a, b],
                        'same_token_ids': left['token_ids'] == right['token_ids'],
                        'initial_cosine': cosine(initial[a], initial[b]),
                        'initial_max_abs_difference': float(np.max(np.abs(initial[a]-initial[b]))),
                        'contextual_cosine': cosine(contextual[a], contextual[b]),
                        'contextual_max_abs_difference': float(np.max(np.abs(contextual[a]-contextual[b]))),
                        'contextual_allclose_1e-6': bool(np.allclose(contextual[a], contextual[b], rtol=0, atol=1e-6))})
    return results


def vector_lab(text, cfg, output):
    text = clean(text)
    if not text.strip():
        raise ValueError('Jawaban untuk analisis vektor kosong.')
    output = Path(output)
    destination = output/'vectors'/cfg['vector_source']
    destination.mkdir(parents=True, exist_ok=True)
    results = {}
    for role in ('roberta', 'bge'):
        source = cfg[role]['checkpoint']+'@'+cfg[role]['revision']
        if role == 'roberta' and cfg['vector_source'] != 'pretrained':
            from .experiment import load_model
            report = json.loads((output/'final_report.json').read_text())
            checkpoint = output/cfg['vector_source']/'final.pt'
            if digest(checkpoint) != report['checkpoints'][cfg['vector_source']]:
                raise ValueError('Checkpoint vektor berbeda dari model final.')
            wrapper, _ = load_model(checkpoint, cfg)
            model = wrapper.roberta
            del wrapper
            source = cfg['vector_source']+' fine-tuned; sha256='+digest(checkpoint)
        else:
            model = encoder(cfg, role)
        device = device_name(cfg)
        model.to(device).eval()
        slow, fast = tokenizer(cfg, role), tokenizer(cfg, role, fast=True)
        trace = token_vectors(text, model, slow, fast, role, cfg, device)
        words, before, after = word_vectors(text, trace)
        sentences, sentence_vectors = [], []
        for start, end in sentence_spans(text):
            part = token_vectors(text[start:end], model, slow, fast, role, cfg, device)
            sentence_vectors.append(part['cls'])
            sentences.append({'sentence_index': len(sentences), 'text': text[start:end], 'start': start, 'end': end,
                              'covered_tokens': int(np.count_nonzero(part['counts'])), 'total_tokens': len(part['ids']),
                              'first5': part['cls'][:5].tolist()})
        vectors = np.asarray(sentence_vectors, dtype=np.float32).reshape(-1, model.config.hidden_size)
        np.savez_compressed(destination/f'{role}_vectors.npz', token_initial=trace['initial'],
                            token_contextual=trace['contextual'], token_ids=np.asarray(trace['ids']),
                            token_offsets=np.asarray(trace['offsets']).reshape(-1, 2), token_coverage=trace['counts'],
                            word_initial=before, word_contextual=after, sentence_cls=vectors)
        token_rows = [{'token_index': i, 'token': slow.convert_ids_to_tokens([identifier])[0], 'token_id': identifier,
                       'start': trace['offsets'][i][0], 'end': trace['offsets'][i][1],
                       'window_coverage': int(trace['counts'][i])} for i, identifier in enumerate(trace['ids'])]
        pd.DataFrame(words).to_csv(destination/f'{role}_words.csv', index=False)
        pd.DataFrame(sentences).to_csv(destination/f'{role}_sentences.csv', index=False)
        pd.DataFrame(token_rows).to_csv(destination/f'{role}_tokens.csv', index=False)
        item = {'source': source, 'dimension': model.config.hidden_size, 'words': words, 'sentences': sentences,
                'tokens': token_rows, 'word_shape': list(after.shape), 'sentence_shape': list(vectors.shape),
                'uncovered_tokens': int(np.count_nonzero(trace['counts'] == 0)), 'window_starts': trace['starts'],
                'repeated_words': compare_repetitions(words, before, after, cfg.get('focus_word', '')),
                'word_method': 'mean subword vectors overlapping each word; contextual overlap windows averaged per token first',
                'sentence_method': 'encode each sentence independently; mean window CLS' if role == 'roberta' else 'encode each sentence independently; CLS then L2; truncate at configured maximum',
                'npz': str(destination/f'{role}_vectors.npz')}
        dump(destination/f'{role}_metadata.json', item)
        results[role] = item
        print(f'{role}: {len(words)} kata × {model.config.hidden_size}; {len(sentences)} kalimat; token tak tercakup={item["uncovered_tokens"]}', flush=True)
        del model, slow, fast
        release()
    results['interpretation'] = 'Bandingkan kemunculan dalam encoder yang sama. RoBERTa 768 dan BGE 1024 mempunyai ruang koordinat berbeda: jangan menghitung cosine langsung antarmodel. Same token ID => lookup sama pada checkpoint yang sama; konteks/posisi dapat mengubah hidden state. Kata ejaan sama bisa mempunyai ID subword berbeda (spasi/case).'
    dump(destination/'report.json', results)
    return results
