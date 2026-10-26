"""Fungsi source untuk notebook tiga input; disalin builder, bukan diimpor Colab."""


def pair_token_frames(bundle, cfg, tok, role='roberta'):
    rows, coverage = [], []
    maximum, overlap = cfg[role+'_max_length'], cfg[role+'_overlap']
    for row, text, context in zip(bundle.rows, bundle.texts, contexts_from_bundle(bundle)):
        batch, _, details = pair_batch_windows([text], [context], tok, 'cpu', maximum, overlap, cfg)
        stats = details[0]
        coverage.append({'row_id': row['row_id'], 'model': role+' tiga input',
                         **{k: v for k, v in stats.items() if k not in ('answer_positions','segments')}})
        for window, (ids, masks) in enumerate(zip(batch['input_ids'].tolist(), batch['attention_mask'].tolist())):
            segments = list(stats['segments'][window])
            padding = ['Padding']*(len(ids)-len(segments))
            segments = padding+segments if tok.padding_side=='left' else segments+padding
            for position, (identifier, piece, mask, segment) in enumerate(zip(ids, tok.convert_ids_to_tokens(ids), masks, segments)):
                rows.append({'row_id':row['row_id'], 'window':window, 'position':position,
                             'token_id':identifier, 'token':piece, 'attention_mask':mask,
                             'input_segment':segment})
    return pd.DataFrame(rows), coverage


def export_tokens(bundle, cfg, journal):
    roberta, rinfo = pair_token_frames(bundle,cfg,tokenizer(cfg,'roberta'),'roberta')
    bge, binfo = pair_token_frames(bundle,cfg,tokenizer(cfg,'bge'),'bge')
    journal.save('04_tokenisasi', {'RoBERTa_input':roberta, 'BGE_input':bge,
                 'Panjang_dan_coverage':pd.DataFrame(rinfo+binfo)},
                 'Token pasangan aktual: soal/referensi + jawaban untuk seluruh skenario',
                 source={'roberta':cfg['roberta'],'bge':cfg['bge']},
                 input_features=['soal','response_text','jawaban_referensi'])
    return roberta,bge


def paired_pretrained_values(bundle, cfg, role):
    """Mean CLS antar-window; BGE dinormalisasi L2 setelah agregasi."""
    device = device_name(cfg)
    model, tok = encoder(cfg,role).to(device), tokenizer(cfg,role)
    model.eval()
    vectors, window_vectors, ids, coverage = [], [], [], []
    with torch.no_grad():
        for index, (text,context) in enumerate(zip(bundle.texts,contexts_from_bundle(bundle))):
            windows,mapping,details = pair_batch_windows([text],[context],tok,device,
                cfg[role+'_max_length'],cfg[role+'_overlap'],cfg)
            # Satu jawaban diproses bertahap agar jumlah window tidak meledakkan memori.
            cls_parts=[]
            for start in range(0,len(mapping),cfg['bge_batch']):
                part={k:v[start:start+cfg['bge_batch']] for k,v in windows.items()}
                cls_parts.append(model(**part).last_hidden_state[:,0,:].float())
            cls=torch.cat(cls_parts,dim=0)
            value=mean_by_essay(cls,mapping,1)
            if role=='bge':value=torch.nn.functional.normalize(value,dim=1)
            vectors.append(value.cpu().numpy())
            window_vectors.append(cls.cpu().numpy())
            row_id=bundle.rows[index].get('row_id',f'inference_{index}')
            ids.extend(f'{row_id}_window_{j}' for j in range(len(cls)))
            coverage.append({k:v for k,v in details[0].items() if k not in ('answer_positions','segments')})
            if (index+1)%25==0 or index+1==len(bundle.rows):
                print(f'{role}: tiga input {index+1}/{len(bundle.rows)} jawaban selesai.',flush=True)
    values=np.concatenate(vectors).astype(np.float32)
    if role=='bge' and any(p.requires_grad for p in model.parameters()):
        raise ValueError('Encoder BGE harus frozen.')
    del model,tok
    release()
    return values,np.concatenate(window_vectors),ids,coverage


def roberta_pretrained_vectors(bundle, cfg, journal):
    values,windows,window_ids,coverage=paired_pretrained_values(bundle,cfg,'roberta')
    journal.save('05_roberta_pretrained',
                 {'Vektor_jawaban_768':vector_frame(values,[r['row_id'] for r in bundle.rows]),
                  'CLS_tiap_window':vector_frame(windows,window_ids)},
                 'RoBERTa tiga input SEBELUM fine-tuning; input sama untuk S1 dan S3',
                 source=cfg['roberta'],pooling='mean window CLS; konteks diulang per window')
    return values


def bge_features(bundle, cfg, output):
    """Cache berubah jika soal, referensi, jawaban, atau resep token berubah."""
    contexts=contexts_from_bundle(bundle)
    contract={'model':cfg['bge'], 'texts':bundle.texts, 'contexts':contexts,
              'maximum':cfg['bge_max_length'],'overlap':cfg['bge_overlap'],
              'context_max_tokens':cfg['context_max_tokens'],'question_max_tokens':cfg['question_max_tokens'],
              'policy':'three_inputs_pair_mean_CLS_then_L2_v1'}
    key=hashlib.sha256(json.dumps(contract,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:20]
    cache=Path(cfg['output'])/'cache'/f'bge_three_inputs_{key}.npz'
    metadata=cache.with_suffix('.json')
    if cache.exists() and metadata.exists():
        proof=json.loads(metadata.read_text())
        if proof['file_sha256']!=hashlib.sha256(cache.read_bytes()).hexdigest():
            raise ValueError('Cache BGE berubah; tidak dipakai.')
        with np.load(cache,allow_pickle=False) as data:values=data['features']
        reused=True
    else:
        values,_,_,coverage=paired_pretrained_values(bundle,cfg,'bge')
        cache.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(cache,features=values)
        proof={'model':cfg['bge'],'shape':list(values.shape),'frozen':True,
               'pooling':'mean paired-window CLS then L2','max_length':cfg['bge_max_length'],
               'feature_source':'soal + jawaban_referensi + response_text; no teacher score',
               'answer_tokens_uncovered':sum(s['answer_tokens_uncovered'] for s in coverage),
               'context_truncated_rows':sum(s['question_truncated'] or s['reference_truncated'] for s in coverage),
               'file_sha256':hashlib.sha256(cache.read_bytes()).hexdigest()}
        dump(metadata,proof)
        reused=False
    if values.shape!=(len(bundle.rows),1024) or not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values,axis=1),1,atol=1e-4):
        raise ValueError('Fitur BGE shape/normalisasi tidak sah.')
    dump(Path(output)/'bge_encoding.json',{**proof,'reused':reused})
    return torch.from_numpy(values)


def token_vectors(text, model, slow, fast, role, cfg, device):
    if cfg.get('pair_context') is None:
        raise ValueError('Vektor tiga input memerlukan soal dan jawaban referensi.')
    # Helper alignment yang sama dipakai untuk RoBERTa dan BGE; batas mengikuti peran.
    paired_cfg={**cfg,'roberta_max_length':cfg[role+'_max_length'],
                'roberta_overlap':cfg[role+'_overlap']}
    trace=paired_token_vectors(text,model,slow,fast,paired_cfg,device,cfg['pair_context'])
    if role=='bge':trace['cls']=torch.nn.functional.normalize(torch.from_numpy(trace['cls'])[None],dim=1)[0].numpy()
    return trace
