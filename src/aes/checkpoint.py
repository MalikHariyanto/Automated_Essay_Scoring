import copy
from pathlib import Path
import torch
from .io import sha256, load_rubric
from .cleaning import POLICY
from .model import AESModel
from .encoders import load_roberta


def save_checkpoint(path, model, cfg, resolved, rubric, training_metadata):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload={'state_dict':model.state_dict(),'scenario':model.scenario,'config':copy.deepcopy(cfg),
        'resolved_models':resolved,'rubric':rubric,'rubric_original_bytes':Path(cfg['rubric']).read_bytes(),
        'rubric_sha256':sha256(cfg['rubric']),'trait_order':[t['name'] for t in rubric['traits']],
        'preprocessing_policy':POLICY,'training':training_metadata}
    torch.save(payload,path)


def load_checkpoint(path,device='cpu', expected_rubric=None,expected_preprocessing=None):
    # Hanya checkpoint lokal yang dipercaya; weights_only mencegah pickle arbitrer.
    p=torch.load(path,map_location='cpu',weights_only=True)
    import hashlib,json
    if hashlib.sha256(p['rubric_original_bytes']).hexdigest()!=p['rubric_sha256']:raise ValueError('Hash rubrik checkpoint rusak')
    if json.loads(p['rubric_original_bytes'])!=p['rubric']:raise ValueError('Rubrik checkpoint tidak konsisten')
    if p['trait_order']!=[t['name'] for t in p['rubric']['traits']]:raise ValueError('Urutan trait berubah')
    if expected_rubric and sha256(expected_rubric)!=p['rubric_sha256']:raise ValueError('Rubrik reload berbeda')
    if expected_preprocessing is not None and expected_preprocessing!=p['config']['preprocessing']:raise ValueError('Preprocessing reload berbeda')
    if p['preprocessing_policy']!=POLICY:raise ValueError('Versi cleaning berbeda')
    for role, spec in p['resolved_models'].items():
        if not spec.get('revision'):raise ValueError('Revision model belum dikunci')
        if spec['checkpoint']!=p['config']['models'][role]['checkpoint']:raise ValueError('Identitas model berbeda')
    from transformers import AutoTokenizer
    for role,spec in p['resolved_models'].items():
        tok=AutoTokenizer.from_pretrained(spec['checkpoint'],revision=spec['revision'],use_fast=(role!='roberta'))
        if type(tok).__name__!=spec['tokenizer_class'] or tok.num_special_tokens_to_add(False)!=spec['special_tokens']:
            raise ValueError('Metadata tokenizer berubah saat reload')
    if p['config']['models']['bge']['max_length']!=1024 or p['config']['models']['roberta']['max_length']!=512 or p['config']['models']['roberta']['overlap']!=128:
        raise ValueError('Kebijakan tokenisasi checkpoint berbeda dengan kontrak')
    s=p['scenario'];roberta=load_roberta(p['resolved_models']['roberta']) if s!='S2' else None
    model=AESModel(s,roberta,**p['config']['scenarios'][s]);model.load_state_dict(p['state_dict'],strict=True);model.to(device).eval()
    return model,p
