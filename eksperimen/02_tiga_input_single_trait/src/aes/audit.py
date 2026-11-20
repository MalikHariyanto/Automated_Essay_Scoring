from difflib import SequenceMatcher
from functools import lru_cache
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .io import targets, text_hash, sha256

class EligibilityError(ValueError): pass

@lru_cache(maxsize=4)
def duplicate_candidates(samples):
    exact=[];near=[]
    for i,(a_id,a_text) in enumerate(samples):
        for b_id,b_text in samples[:i]:
            if a_text==b_text:exact.append([a_id,b_id])
            else:
                matcher=SequenceMatcher(None,a_text,b_text,autojunk=False)
                if matcher.real_quick_ratio()<.8 or matcher.quick_ratio()<.8:continue
                sim=matcher.ratio()
                if sim>=.8:near.append({'ids':[a_id,b_id],'similarity':sim})
    return exact,near

def validate_inputs_for_inspection(df, rubric):
    if not len(df): raise ValueError('CSV kosong')
    if df.essay_id.eq('').any(): raise ValueError('essay_id kosong')
    # Status draf ditampilkan; bukan alasan menghentikan inspeksi.
    return audit(df, rubric)

def audit(df, rubric):
    y = targets(df, rubric)
    report = {'rows':len(df), 'unique_essay_ids':int(df.essay_id.nunique()),
        'empty_text_ids':df.loc[df.text_raw.str.strip().eq(''),'essay_id'].tolist(),
        'duplicate_ids':df.loc[df.essay_id.duplicated(False),'essay_id'].tolist(),
        'version_mismatch_ids':df.loc[df.rubric_version.ne(rubric['version']),'essay_id'].tolist(),
        'complete_label_rows':int(np.isfinite(y).all(1).sum()), 'rubric_confirmed':rubric['confirmed'],
        'traits':{}, 'statuses':{}, 'exact_duplicates':[], 'near_duplicates':[]}
    for j,t in enumerate(rubric['traits']):
        col = df['score_'+t['name']]; v = y[:,j]; finite = np.isfinite(v)
        report['traits'][t['name']] = {
            'missing':int(col.str.strip().eq('').sum()),
            'invalid_or_nonfinite':int((~finite & col.str.strip().ne('')).sum()),
            'noninteger':int((finite & (v != np.floor(v))).sum()),
            'out_of_range':int((finite & ((v<t['min']) | (v>t['max']))).sum()),
            'distribution':{str(c):int((v==c).sum()) for c in t['categories']}}
    for k in ['scoring_status','transcription_status','text_completeness','identity_status','group_status','ready_for_training']:
        report['statuses'][k] = df[k].value_counts(dropna=False).to_dict() if k in df else {'MISSING_COLUMN':len(df)}
    report['exact_duplicates'],report['near_duplicates']=copy.deepcopy(duplicate_candidates(tuple(zip(df.essay_id,df.text_raw))))
    n_groups = int(df.group_id.replace('',np.nan).nunique())
    report['proposed_group_count'] = n_groups
    report['five_group_folds_possible_by_count'] = n_groups >= 5
    report['note'] = 'Kelompok usulan bukan bukti identitas; kelayakan fold memerlukan review dan label.'
    report['eda'] = {'characters':df.text_raw.str.len().describe().to_dict(),
                     'words':df.text_raw.str.split().str.len().describe().to_dict(),
                     'paragraphs':df.text_raw.map(lambda s:len([p for p in s.split('\n\n') if p.strip()])).tolist(),
                     'raw_equals_supplied_clean':int(df.text_raw.eq(df.text_clean).sum())}
    return report

def training_blockers(df, rubric, review=None, dataset_hash=None):
    a = audit(df,rubric); errors = []
    if not rubric['confirmed'] or not rubric.get('teacher_approved_by') or not rubric.get('teacher_approved_on'):
        errors.append('Rubrik perlu pengesahan guru, identitas, dan tanggal pengesahan.')
    if a['duplicate_ids'] or a['empty_text_ids']: errors.append('ID duplikat atau teks kosong perlu ditinjau.')
    if a['version_mismatch_ids']: errors.append('rubric_version CSV berbeda dengan JSON.')
    y=targets(df,rubric)
    if not np.isfinite(y).all() or not np.isin(y,[1,2,3,4,5]).all():
        errors.append('Seluruh empat skor guru harus integer sah 1–5; missing bukan nol.')
    allowed = {'transcription_status':{'verified'}, 'text_completeness':{'complete'},
               'identity_status':{'verified'}, 'scoring_status':{'scored','teacher_scored'},
               'ready_for_training':{'true'}}
    for k,valid in allowed.items():
        if k not in df or not df[k].str.lower().isin(valid).all():
            errors.append(f'{k} belum memenuhi {sorted(valid)} pada semua esai.')
    for k in ['rater_id','scoring_date']:
        if k not in df or df[k].str.strip().eq('').any(): errors.append(f'{k} penilaian guru perlu lengkap.')
    if not review or not review.get('independence_verified') or not review.get('near_duplicates_reviewed'):
        errors.append('Review independensi, identitas, dan kandidat duplikasi belum disahkan.')
    else:
        if not review.get('reviewer') or not review.get('reviewed_on'): errors.append('Identitas/tanggal reviewer belum lengkap.')
        if review.get('dataset_sha256') != dataset_hash: errors.append('Hash dataset pada review tidak cocok.')
        if review.get('mode') not in ['independent','grouped']: errors.append('Review mode harus independent/grouped.')
        decisions=review.get('pair_decisions',{})
        for pair in a['exact_duplicates']:
            if review.get('mode')!='grouped' or review.get('groups',{}).get(pair[0])!=review.get('groups',{}).get(pair[1]):
                errors.append(f'Exact duplicate {pair} harus berada dalam group utuh yang ditinjau.')
        for pair in a['near_duplicates']:
            ids=pair['ids'];key='|'.join(sorted(ids));decision=decisions.get(key,{})
            relation=decision.get('relation')
            if relation not in ['independent','related'] or not decision.get('evidence'):
                errors.append(f'Near-duplicate {key} memerlukan keputusan hubungan dan bukti review.')
            elif relation=='related' and (review.get('mode')!='grouped' or review.get('groups',{}).get(ids[0])!=review.get('groups',{}).get(ids[1])):
                errors.append(f'Pasangan terkait {key} tidak boleh terpisah group.')
        if review.get('mode') == 'grouped':
            groups=review.get('groups',{})
            if set(groups) != set(df.essay_id) or not all(isinstance(v,str) and v.strip() for v in groups.values()):
                errors.append('Mapping group terverifikasi harus lengkap per essay_id.')
    return errors

def validate_inputs_for_training(df, rubric, review=None, dataset_hash=None):
    errors=training_blockers(df,rubric,review,dataset_hash)
    if errors: raise EligibilityError('\n'.join(errors))
    return targets(df,rubric)
