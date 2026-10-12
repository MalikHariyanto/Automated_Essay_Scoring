import hashlib
import json
from pathlib import Path
import pandas as pd
import numpy as np

TRAITS = ['isi', 'organisasi', 'kebahasaan', 'mekanik']

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def load_config(path='configs/default.json'):
    path = Path(path).resolve(); cfg = json.loads(path.read_text())
    root = path.parent.parent
    for key in ['csv','rubric','output']:
        p = Path(cfg[key]); cfg[key] = str(p if p.is_absolute() else root/p)
    for key,value in cfg.get('sources',{}).items():
        if value:
            source_path=Path(value)
            cfg['sources'][key]=str(source_path if source_path.is_absolute() else root/source_path)
    p = Path(cfg['split']['review_file'])
    cfg['split']['review_file'] = str(p if p.is_absolute() else root/p)
    return cfg

def load_rubric(path):
    raw = Path(path).read_bytes(); r = json.loads(raw)
    assert [t['name'] for t in r['traits']] == TRAITS, 'Urutan trait tidak sesuai kontrak'
    for t in r['traits']:
        for key in ['min','max','scale_type','categories','definition','indicators','exclude','levels']:
            if key not in t: raise ValueError(f"Rubrik kurang {t['name']}.{key}")
        if t['min'] != 1 or t['max'] != 5 or t['scale_type'] != 'ordinal' or t['categories'] != [1,2,3,4,5]:
            raise ValueError('Kontrak memerlukan ordinal 1–5')
        if set(t['levels']) != {'1','2','3','4','5'}: raise ValueError('Levels tidak lengkap')
    for k in ['version','confirmed','scoring_rule','missing_rule']: r[k]
    if not isinstance(r['confirmed'],bool):raise ValueError('confirmed rubrik harus boolean')
    return r

def load_data(path):
    # utf-8-sig menghapus BOM sebelum parser membaca tanda kutip header.
    df = pd.read_csv(path, encoding='utf-8-sig', dtype=str, keep_default_na=False)
    needed = ['essay_id','text_raw','text_clean','group_id','rubric_version','scoring_status',
              'transcription_status','ready_for_training'] + ['score_'+n for n in TRAITS]
    missing = sorted(set(needed)-set(df.columns))
    if missing: raise ValueError(f'Kolom wajib tidak tersedia: {missing}')
    return df

def targets(df, rubric):
    a = np.stack([pd.to_numeric(df['score_'+t['name']], errors='coerce').to_numpy(float)
                  for t in rubric['traits']], axis=1)
    assert a.shape == (len(df),4)
    return a
