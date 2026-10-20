"""Tiga skenario satu skor; BGE frozen, RoBERTa fine-tuned pada S1/S3."""
import gc
import hashlib
import json
import logging
from pathlib import Path
import warnings

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from aes.metrics import gwet_ac2
from aes.windows import batch_windows, mean_by_essay
from .data import dump


def device_name(cfg):
    requested = cfg['device']
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA diminta tetapi tidak tersedia; pilih GPU di Colab atau device=cpu.')
    return ('cuda' if torch.cuda.is_available() else 'cpu') if requested == 'auto' else requested


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class ExpectedPoolerNotice(logging.Filter):
    """Hanya saring pemberitahuan dua bobot pooler yang memang tidak dipakai."""
    def filter(self, record):
        import ast
        import re
        message = record.getMessage()
        if not message.startswith('Some weights of the model checkpoint at ') or 'were not used when initializing' not in message:
            return True
        match = re.search(r"\['[^\n]+?\]", message)
        if match is None:
            return True
        try:
            keys = ast.literal_eval(match.group())
        except (ValueError, SyntaxError):
            return True
        return set(keys) != {'pooler.dense.bias', 'pooler.dense.weight'}


def encoder(cfg, role):
    spec = cfg[role]
    logger = logging.getLogger('transformers.modeling_utils')
    notice = ExpectedPoolerNotice()
    logger.addFilter(notice)
    try:
        result = AutoModel.from_pretrained(spec['checkpoint'], revision=spec['revision'], add_pooling_layer=False)
    finally:
        logger.removeFilter(notice)
    expected = 768 if role == 'roberta' else 1024
    if result.config.hidden_size != expected:
        raise ValueError(f'{role}: dimensi checkpoint tidak cocok dengan {expected}.')
    # Parameter RoBERTa ikut diperbarui; parameter BGE tidak ikut diperbarui.
    result.requires_grad_(role == 'roberta')
    return result


def tokenizer(cfg, role, fast=False):
    result = AutoTokenizer.from_pretrained(cfg[role]['checkpoint'], revision=cfg[role]['revision'],
                                           use_fast=fast if role == 'roberta' else True)
    if role == 'roberta' and not fast and result.num_special_tokens_to_add(False) != 2:
        raise ValueError('RoBERTa slow tokenizer harus menambahkan dua special token.')
    return result


class SingleScoreModel(nn.Module):
    def __init__(self, scenario, roberta=None, dropout=.1):
        super().__init__()
        if scenario not in ('S1', 'S2', 'S3'):
            raise ValueError('Skenario harus S1, S2, atau S3')
        if scenario != 'S2' and roberta is None:
            raise ValueError('RoBERTa diperlukan oleh S1/S3.')
        self.scenario, self.roberta = scenario, roberta
        if roberta is not None:
            roberta.requires_grad_(True)
        self.dropout = nn.Dropout(dropout)
        if scenario == 'S1':
            self.head = nn.Linear(768, 1)
        elif scenario == 'S2':
            self.head = nn.Sequential(nn.Linear(1024, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 1))
        else:
            # LayerNorm tambahan pada vektor jawaban RoBERTa, khusus cabang fusion.
            self.roberta_norm = nn.LayerNorm(768)
            self.projection = nn.Sequential(nn.Linear(1024, 256), nn.LayerNorm(256))
            self.head = nn.Linear(1024, 1)

    def forward(self, windows=None, mapping=None, n_essays=None, bge=None, return_features=False):
        features = {}
        if self.scenario != 'S2':
            cls = self.roberta(**windows).last_hidden_state[:, 0, :]
            essay = mean_by_essay(cls, mapping, n_essays)
            features.update(roberta=essay, roberta_window_cls=cls)
        if self.scenario == 'S1':
            prediction = self.head(self.dropout(essay))
        elif self.scenario == 'S2':
            if bge is None or bge.ndim != 2 or bge.shape[1] != 1024:
                raise ValueError('BGE memerlukan [N,1024]')
            features['bge'] = bge
            prediction = self.head(bge)
        else:
            if bge is None or bge.shape != (n_essays, 1024):
                raise ValueError('BGE memerlukan [N,1024]')
            features['bge'] = bge
            # Simpan vektor asli dan hasil normalisasi agar setiap tahap bisa diperiksa.
            features['roberta_layernorm'] = self.roberta_norm(essay)
            features['projection'] = self.projection(bge)
            features['concat'] = torch.cat([features['roberta_layernorm'], features['projection']], dim=1)
            prediction = self.head(self.dropout(features['concat']))
        if prediction.ndim != 2 or prediction.shape[1] != 1:
            raise ValueError('Dataset satu skor membutuhkan output [N,1].')
        return (prediction, features) if return_features else prediction


class Engine:
    def __init__(self, model, texts, bge, cfg, device, roberta_tokenizer=None):
        self.model, self.texts, self.bge, self.cfg, self.device = model, texts, bge, cfg, device
        self.tokenizer = roberta_tokenizer or (tokenizer(cfg, 'roberta') if model.scenario != 'S2' else None)

    def forward(self, indices, return_features=False):
        ids = [int(i) for i in indices]
        features = self.bge[ids].to(self.device) if self.model.scenario != 'S1' else None
        if self.model.scenario == 'S2':
            return self.model(bge=features, return_features=return_features)
        windows, mapping, _ = batch_windows([self.texts[i] for i in ids], self.tokenizer,
                                             self.device, self.cfg['roberta_max_length'], self.cfg['roberta_overlap'])
        return self.model(windows=windows, mapping=mapping, n_essays=len(ids), bge=features,
                          return_features=return_features)

    def predict(self, indices):
        self.model.eval()
        values = []
        with torch.no_grad():
            for start in range(0, len(indices), self.cfg['micro_batch']):
                ids = indices[start:start+self.cfg['micro_batch']]
                values.append(self.forward(ids).reshape(-1).float().cpu().numpy())
        scaled = np.concatenate(values)
        # Kembalikan skala asli sebelum RMSE/Pearson maupun tampilan prediksi.
        return scaled*(self.cfg['score_max']-self.cfg['score_min'])+self.cfg['score_min']


def metrics(teacher, prediction, cfg):
    y, p = np.asarray(teacher, dtype=float), np.asarray(prediction, dtype=float)
    if y.ndim != 1 or y.shape != p.shape or len(y) == 0:
        raise ValueError('Metrik satu skor memerlukan dua vektor [N] yang sama dan tidak kosong.')
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Label/prediksi tidak finite.')
    cats = np.arange(cfg['score_min'], cfg['score_max']+1)
    if not np.isin(y, cats).all():
        raise ValueError('Skor guru harus kategori integer pada rentang yang dinyatakan.')
    clipped = np.clip(p, cfg['score_min'], cfg['score_max'])
    # Tie .5 memilih kategori lebih rendah, sama dengan kontrak konversi proyek.
    ordinal = cats[np.abs(clipped[:, None]-cats[None, :]).argmin(axis=1)]
    result = {'RMSE': float(np.sqrt(np.mean((y-p)**2))), 'Pearson': None,
              'QWK': None, 'AC2': None, 'undefined_reasons': {},
              'out_of_range_fraction': float(np.mean((p < cats[0]) | (p > cats[-1])))}
    if len(y) > 1 and np.ptp(y) > 0 and np.ptp(p) > 0:
        result['Pearson'] = float(pearsonr(y, p).statistic)
    else:
        result['undefined_reasons']['Pearson'] = 'Perlu minimal dua pasangan dan kedua vektor tidak konstan.'
    if len(y) > 1:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            qwk = cohen_kappa_score(y, ordinal, labels=cats, weights='quadratic')
        if np.isfinite(qwk):
            result['QWK'] = float(qwk)
        else:
            result['undefined_reasons']['QWK'] = 'Expected disagreement nol; undefined, bukan nol.'
        result['AC2'] = gwet_ac2(y, ordinal, cats)
    else:
        result['undefined_reasons'].update(QWK='Perlu minimal dua pasangan.', AC2='Perlu minimal dua pasangan.')
    return result


def bge_features(bundle, cfg, output):
    key = hashlib.sha256(json.dumps({'data': bundle.source_sha256, 'model': cfg['bge'],
                                     'texts': hashlib.sha256(json.dumps(bundle.texts, ensure_ascii=False).encode()).hexdigest(),
                                     'max_length': cfg['bge_max_length'], 'pooling': 'CLS_L2',
                                     'cleaning': 'NFC_controls_only'}, sort_keys=True).encode()).hexdigest()[:20]
    cache = Path(cfg['output'])/'cache'/f'bge_{key}.npz'
    metadata = cache.with_suffix('.json')
    if cache.exists() and metadata.exists():
        proof = json.loads(metadata.read_text())
        if proof['file_sha256'] != hashlib.sha256(cache.read_bytes()).hexdigest():
            raise ValueError('Cache BGE berubah; tidak dipakai.')
        with np.load(cache, allow_pickle=False) as data:
            values = data['features']
        if values.shape != (len(bundle.rows), 1024) or not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-4):
            raise ValueError('Cache BGE shape/normalisasi tidak sah.')
        dump(Path(output)/'bge_encoding.json', {**proof, 'reused': True})
        return torch.from_numpy(values)
    device = device_name(cfg)
    model = encoder(cfg, 'bge').to(device)
    model.eval()
    tok = tokenizer(cfg, 'bge')
    lengths = [len(tok.encode(t, add_special_tokens=True)) for t in bundle.texts]
    output_vectors = []
    with torch.no_grad():
        for start in range(0, len(bundle.rows), cfg['bge_batch']):
            batch = tok(bundle.texts[start:start+cfg['bge_batch']], padding=True, truncation=True,
                        max_length=cfg['bge_max_length'], return_tensors='pt')
            batch = {k: v.to(device) for k, v in batch.items() if k in ('input_ids', 'attention_mask')}
            cls = model(**batch).last_hidden_state[:, 0, :].float()
            output_vectors.append(torch.nn.functional.normalize(cls, dim=1).cpu().numpy())
            processed = min(start+cfg['bge_batch'], len(bundle.rows))
            if processed == len(bundle.rows) or processed % 25 < cfg['bge_batch']:
                print(f'BGE-M3: vektor {processed} dari {len(bundle.rows)} jawaban selesai.', flush=True)
    values = np.concatenate(output_vectors).astype(np.float32)
    if values.shape != (len(bundle.rows), 1024) or not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-4):
        raise RuntimeError('Encoder BGE menghasilkan fitur tidak sah; tidak disimpan.')
    assert not any(p.requires_grad for p in model.parameters())
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, features=values)
    proof = {'model': cfg['bge'], 'shape': list(values.shape), 'frozen': True, 'pooling': 'last_hidden_state CLS then L2',
             'truncated_answers': sum(n > cfg['bge_max_length'] for n in lengths),
             'max_length': cfg['bge_max_length'], 'feature_source': 'response_text only',
             'file_sha256': hashlib.sha256(cache.read_bytes()).hexdigest()}
    dump(metadata, proof)
    dump(Path(output)/'bge_encoding.json', {**proof, 'reused': False})
    del model, tok
    release()
    return torch.from_numpy(values)
