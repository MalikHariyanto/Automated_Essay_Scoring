"""Langkah perhitungan dari pasangan label/prediksi; tidak membuat label baru."""
import numpy as np

from .metrics import evaluate, ordinal_prediction


def metric_trace(teacher, predicted, rubric, essay_ids):
    """Empat trait dihitung terpisah. Pemanggil harus memverifikasi sumber/split."""
    y, p = np.asarray(teacher, dtype=float), np.asarray(predicted, dtype=float)
    metrics = evaluate(y, p, rubric)
    if len(essay_ids) != len(y) or len(set(essay_ids)) != len(essay_ids):
        raise ValueError('ID pasangan evaluasi harus unik dan cocok jumlah baris.')
    traces = {}
    for j, trait in enumerate(rubric['traits']):
        a, b = y[:, j], p[:, j]
        cats = np.asarray(trait['categories'], dtype=float)
        ordinal = ordinal_prediction(b, trait)
        clipped = np.clip(b, trait['min'], trait['max'])
        error = b - a
        n, q = len(a), len(cats)
        pairs = [{'essay_id':str(essay_ids[i]), 'teacher':int(a[i]), 'raw_prediction':float(b[i]),
                  'error':float(error[i]), 'squared_error':float(error[i]**2),
                  'clipped_prediction':float(clipped[i]), 'ordinal_prediction':int(ordinal[i]),
                  'out_of_range':bool(b[i]<trait['min'] or b[i]>trait['max'])} for i in range(n)]
        observed = np.zeros((q,q), dtype=float)
        for teacher_category, model_category in zip(a, ordinal):
            observed[np.searchsorted(cats,teacher_category),np.searchsorted(cats,model_category)] += 1
        disagreement = ((cats[:,None] - cats[None,:]) / (cats[-1]-cats[0]))**2
        agreement = 1-disagreement
        expected = np.outer(observed.sum(1), observed.sum(0))/n if n else np.zeros_like(observed)
        observed_error = float(np.sum(disagreement*observed)/n) if n else None
        expected_error = float(np.sum(disagreement*expected)/n) if n else None
        pooled = (observed.sum(1)+observed.sum(0))/(2*n) if n else np.zeros(q)
        pa = float(np.sum(agreement*observed)/n) if n else None
        pe = float(agreement.sum()*np.sum(pooled*(1-pooled))/(q*(q-1))) if n else None
        centered_teacher = a-a.mean() if n else a
        centered_prediction = b-b.mean() if n else b
        covariance_sum = float(np.sum(centered_teacher*centered_prediction))
        teacher_ss = float(np.sum(centered_teacher**2))
        prediction_ss = float(np.sum(centered_prediction**2))
        value = metrics[trait['name']]
        if value['QWK'] is not None:
            assert np.isclose(1-observed_error/expected_error,value['QWK'])
        if value['AC2'] is not None:
            assert np.isclose((pa-pe)/(1-pe),value['AC2'])
        traces[trait['name']] = {
            'n_pairs':n, 'categories':trait['categories'], 'pairs':pairs,
            'midpoint_policy':'Nearest category; tie chooses lower category.',
            'metrics':value,
            'RMSE_steps':{'sum_squared_errors':float(np.sum(error**2)),
                          'mean_squared_error':float(np.mean(error**2)) if n else None,
                          'RMSE':value['RMSE']},
            'Pearson_steps':{'mean_teacher':float(a.mean()) if n else None,
                'mean_raw_prediction':float(b.mean()) if n else None,
                'centered_teacher':centered_teacher.tolist(),'centered_raw_prediction':centered_prediction.tolist(),
                'sum_centered_products':covariance_sum,'sum_squared_centered_teacher':teacher_ss,
                'sum_squared_centered_raw_prediction':prediction_ss,
                'denominator':float(np.sqrt(teacher_ss*prediction_ss)),'Pearson':value['Pearson']},
            'QWK_steps':{'observed_counts_teacher_rows_prediction_columns':observed.tolist(),
                'expected_counts_from_marginals':expected.tolist(),
                'quadratic_disagreement_weights':disagreement.tolist(),
                'observed_weighted_disagreement':observed_error,
                'expected_weighted_disagreement':expected_error,'QWK':value['QWK']},
            'AC2_steps':{'quadratic_agreement_weights':agreement.tolist(),
                'pooled_category_proportions':pooled.tolist(),
                'Pa_observed_agreement':pa,'Pe_chance_agreement':pe,'AC2':value['AC2']},
        }
    return traces
