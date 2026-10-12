import torch

def content_windows(text, tokenizer, maximum=512, overlap=128):
    capacity = maximum - tokenizer.num_special_tokens_to_add(pair=False)
    if not 0 <= overlap < capacity: raise ValueError('Overlap harus lebih kecil dari content capacity')
    ids = tokenizer.encode(text, add_special_tokens=False)
    starts = [0]
    while starts[-1]+capacity < len(ids): starts.append(starts[-1]+capacity-overlap)
    chunks=[]
    for start in starts:
        content=ids[start:start+capacity]
        prepared=tokenizer.prepare_for_model(content, add_special_tokens=True, truncation=False,
                                             return_attention_mask=True, return_token_type_ids=False)
        assert len(prepared['input_ids']) <= maximum
        chunks.append(dict(prepared))
    return chunks, {'content_tokens':len(ids),'capacity':capacity,'stride':capacity-overlap,
                    'starts':starts,'windows':len(chunks)}

def batch_windows(texts, tokenizer, device='cpu', maximum=512, overlap=128):
    chunks=[]; mapping=[]; stats=[]
    for i,text in enumerate(texts):
        c,s=content_windows(text,tokenizer,maximum,overlap); chunks.extend(c); mapping.extend([i]*len(c));stats.append(s)
    batch=tokenizer.pad(chunks, padding=True, return_tensors='pt')
    return {k:v.to(device) for k,v in batch.items()}, torch.tensor(mapping,device=device), stats

def mean_by_essay(cls, mapping, n):
    sums=cls.new_zeros((n,cls.shape[-1])).index_add(0,mapping,cls)
    counts=cls.new_zeros(n).index_add(0,mapping,cls.new_ones(len(mapping)))
    if (counts==0).any(): raise ValueError('Esai tanpa window')
    return sums / counts[:,None]
