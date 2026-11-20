import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit, KFold
from .io import sha256, dump, text_hash, targets
from .audit import validate_inputs_for_training, EligibilityError


def make_manifest(df,rubric,review,cfg):
    validate_inputs_for_training(df,rubric,review,sha256(cfg['csv']))
    ids=np.arange(len(df)); mode=review['mode']; s=cfg['split']
    groups=np.array([review['groups'][i] for i in df.essay_id]) if mode=='grouped' else None
    if mode=='grouped':
        dev,test=next(GroupShuffleSplit(n_splits=1,test_size=s['test_size'],random_state=s['seed']).split(ids,groups=groups))
        if len(set(groups[dev]))<s['folds']:raise EligibilityError('Development memiliki kurang dari lima group terverifikasi; tidak boleh fallback random.')
    else:dev,test=train_test_split(ids,test_size=s['test_size'],random_state=s['seed'])
    if len(dev)<s['folds']:raise EligibilityError('Development terlalu kecil untuk lima fold.')
    folds=[]
    for repeat,seed in enumerate(s['repeat_seeds']):
        if mode=='independent':
            pairs=[(dev[a],dev[b]) for a,b in KFold(n_splits=s['folds'],shuffle=True,random_state=seed).split(dev)]
        else:
            # Repeated group folds: acak group terverifikasi, lalu bagi kelompok utuh.
            unique=np.unique(groups[dev]);np.random.default_rng(seed).shuffle(unique)
            buckets=np.array_split(unique,s['folds']);pairs=[]
            for bucket in buckets:
                valid=dev[np.isin(groups[dev],bucket)];train=dev[~np.isin(groups[dev],bucket)];pairs.append((train,valid))
        for fold,(train,valid) in enumerate(pairs):
            if groups is not None:assert not set(groups[train]) & set(groups[valid])
            folds.append({'repeat':repeat+1,'fold':fold+1,'seed':seed,
                          'train':df.iloc[train].essay_id.tolist(),'validation':df.iloc[valid].essay_id.tolist()})
    manifest={'dataset_sha256':sha256(cfg['csv']),'rubric_sha256':sha256(cfg['rubric']),
        'review_sha256':sha256(cfg['split']['review_file']), 'mode':mode,'split_config':s,
        'essay_text_hashes':dict(zip(df.essay_id,map(text_hash,df.text_raw))),
        'development':df.iloc[dev].essay_id.tolist(),'test':df.iloc[test].essay_id.tolist(),'folds':folds}
    if groups is not None:assert not set(groups[dev]) & set(groups[test])
    def distribution(subset):
        y=targets(subset,rubric)
        return {t['name']:{str(c):int((y[:,j]==c).sum()) for c in t['categories']} for j,t in enumerate(rubric['traits'])}
    manifest['label_distribution']={'development':distribution(df.iloc[dev])}
    manifest['test_labels_sealed_until_final_evaluation']=True
    manifest['fold_distributions']=[{'repeat':f['repeat'],'fold':f['fold'],
        'train':distribution(df[df.essay_id.isin(f['train'])]),
        'validation':distribution(df[df.essay_id.isin(f['validation'])])} for f in folds]
    return manifest
