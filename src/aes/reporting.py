"""Tabel empat metrik final: hanya dari test proposal yang sah."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .audit import training_blockers
from .io import dump, sha256, targets
from .metrics import evaluate
from .metric_trace import metric_trace

SCENARIOS = ['S1', 'S2', 'S3']
METRICS = ['RMSE', 'Pearson', 'QWK', 'AC2']


def final_report(cfg, df, rubric):
    out = Path(cfg['output'])
    review_path = Path(cfg['split']['review_file'])
    review = json.loads(review_path.read_text()) if review_path.exists() else None
    blockers = training_blockers(df, rubric, review, sha256(cfg['csv']))
    result = {'scope': 'proposal_final_held_out_test', 'rows': [], 'metric_traces': {},
              'data_blockers': blockers,
              'note': 'Tidak mengevaluasi head UNTRAINED. Null berarti belum tersedia/undefined, bukan nol.'}
    manifest_path = out/'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if not blockers and manifest_path.exists() else None
    if manifest is not None:
        if manifest['dataset_sha256'] != sha256(cfg['csv']) or manifest['rubric_sha256'] != sha256(cfg['rubric']):
            raise ValueError('Laporan final: manifest tidak cocok sumber CSV/JSON saat ini.')
        if manifest.get('review_sha256') != sha256(review_path):
            raise ValueError('Laporan final: review independensi/group berbeda dari manifest.')
    for scenario in SCENARIOS:
        reason = 'Menunggu empat skor guru dan verifikasi data/rubrik.' if blockers else 'Training/refit/evaluasi final skenario belum selesai.'
        metrics = None
        base = out/'runs/proposal'/scenario
        files = [base/'locked_config.json', base/'final/test_metrics.json', base/'final/test_predictions.csv']
        if manifest is not None and all(p.exists() for p in files):
            resolved_path=out/'model_revisions.json'
            if not resolved_path.is_file():
                raise ValueError('Laporan final: metadata revision model belum tersedia.')
            resolved=json.loads(resolved_path.read_text())
            for role,spec in cfg['models'].items():
                if resolved[role]['checkpoint']!=spec['checkpoint'] or not resolved[role].get('revision') or (spec['revision'] and resolved[role]['revision']!=spec['revision']):
                    raise ValueError('Laporan final: checkpoint/revision berbeda dari konfigurasi.')
            lock, saved = [json.loads(p.read_text()) for p in files[:2]]
            # Tuning harus sudah dikunci dan seluruh tiga skenario refit selesai.
            locked = lock.get('final_refit_completed') and saved.get('configuration_locked')
            all_locked = set(saved.get('all_requested_scenarios_locked', [])) == set(SCENARIOS)
            for other in SCENARIOS:
                other_lock = out/'runs/proposal'/other/'locked_config.json'
                all_locked = all_locked and other_lock.exists() and json.loads(other_lock.read_text()).get('final_refit_completed', False)
            if not locked or not all_locked:
                raise ValueError('Laporan final: ketiga konfigurasi/refit harus dikunci sebelum membuka hasil test.')
            predictions = pd.read_csv(files[2], dtype={'essay_id':str})
            ids = manifest['test']
            if predictions.essay_id.duplicated().any() or set(predictions.essay_id) != set(ids):
                raise ValueError('Laporan final: ID prediksi berbeda dari held-out test.')
            predicted = predictions.set_index('essay_id').loc[ids, ['pred_'+t['name'] for t in rubric['traits']]].to_numpy(float)
            teacher = targets(df.set_index('essay_id').loc[ids].reset_index(), rubric)
            metrics = evaluate(teacher, predicted, rubric)
            if metrics != saved['metrics']:
                # CSV round-trip dapat memiliki beda float sangat kecil.
                for trait in metrics:
                    for metric in METRICS + ['out_of_range_fraction']:
                        a,b=metrics[trait][metric],saved['metrics'][trait][metric]
                        if (a is None) != (b is None) or (a is not None and not np.isclose(a,b,rtol=1e-9,atol=1e-12)):
                            raise ValueError('Laporan final: angka tersimpan tidak cocok dengan prediksi/label test.')
            result['metric_traces'][scenario] = metric_trace(teacher,predicted,rubric,ids)
        for trait in rubric['traits']:
            name = trait['name']
            row = {'scenario':scenario, 'trait':name,
                   **{m:metrics[name][m] if metrics is not None else None for m in METRICS},
                   'status':'available_final_test' if metrics is not None else 'not_available',
                   'reason':metrics[name]['undefined_reasons'] if metrics is not None else reason}
            if metrics is not None:
                row['out_of_range_fraction'] = metrics[name]['out_of_range_fraction']
            result['rows'].append(row)
    dump(out/'final_learning_report.json',result)
    return result
