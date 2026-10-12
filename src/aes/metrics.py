import warnings
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score

def ordinal_prediction(values, trait):
    cats=np.asarray(trait['categories'],dtype=float)
    raw=np.asarray(values,dtype=float)
    if not np.isfinite(raw).all():raise ValueError('Prediksi nonfinite')
    a=np.clip(raw,trait['min'],trait['max'])
    # argmin memilih kategori pertama (lebih rendah) ketika jarak sama.
    return cats[np.abs(a[...,None]-cats).argmin(-1)].astype(int)

def gwet_ac2(teacher, predicted, categories):
    a=np.asarray(teacher);b=np.asarray(predicted);c=np.asarray(categories,dtype=float)
    if len(a)!=len(b) or len(a)<2: raise ValueError('AC2 perlu minimal dua pasangan')
    if not np.isin(a,c).all() or not np.isin(b,c).all(): raise ValueError('Kategori AC2 tidak sah')
    q=len(c); weights=1-((c[:,None]-c[None,:])/(c.max()-c.min()))**2
    ai=np.searchsorted(c,a); bi=np.searchsorted(c,b)
    pa=weights[ai,bi].mean()
    pi=(np.bincount(ai,minlength=q)+np.bincount(bi,minlength=q))/(2*len(a))
    pe=weights.sum()*np.sum(pi*(1-pi))/(q*(q-1))
    return float((pa-pe)/(1-pe))

def evaluate(teacher, predicted, rubric):
    y=np.asarray(teacher,dtype=float);p=np.asarray(predicted,dtype=float)
    if y.shape!=p.shape or y.ndim!=2 or y.shape[1]!=4: raise ValueError('Metrik memerlukan [N,4] yang sama')
    if not np.isfinite(y).all() or not np.isfinite(p).all(): raise ValueError('Label/prediksi nonfinite')
    result={}
    for j,t in enumerate(rubric['traits']):
        a,b=y[:,j],p[:,j]; cats=t['categories']
        if not np.isin(a,cats).all(): raise ValueError('Label harus kategori guru yang sah')
        o=ordinal_prediction(b,t); reasons={}
        rmse=float(np.sqrt(np.mean((a-b)**2))) if len(a) else None
        pearson=None
        if len(a)>=2 and np.ptp(a)>0 and np.ptp(b)>0: pearson=float(pearsonr(a,b).statistic)
        else: reasons['Pearson']='Minimal dua pasangan dan kedua input tidak konstan.'
        qwk=None;ac2=None
        if len(a)>=2:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore'); v=cohen_kappa_score(a,o,weights='quadratic',labels=cats)
            if np.isfinite(v): qwk=float(v)
            else: reasons['QWK']='Expected disagreement nol; koefisien undefined.'
            ac2=gwet_ac2(a,o,cats)
        else: reasons.update({'QWK':'Minimal dua pasangan.','AC2':'Minimal dua pasangan.'})
        result[t['name']]={'RMSE':rmse,'Pearson':pearson,'QWK':qwk,'AC2':ac2,
             'out_of_range_fraction':float(((b<t['min']) | (b>t['max'])).mean()) if len(b) else None,
             'undefined_reasons':reasons}
    return result

def selection_monitor(metrics, validation_mse, force_fallback=False):
    values=[v['QWK'] for v in metrics.values()]
    if not force_fallback and len(values)==4 and all(v is not None and np.isfinite(v) for v in values):
        return 'mean_qwk_all_four',float(np.mean(values))
    return 'validation_mse',float(validation_mse)
