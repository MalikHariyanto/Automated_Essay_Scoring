"""Audit hasil E8 dan perbarui agregat buku tanpa menerbitkan data siswa."""
import argparse
import json
from pathlib import Path
import subprocess
from zipfile import ZipFile

from build_mlp_lora_results import extract
from build_mlp_context_results import winner_counts
from build_mlp_study import ROOT, BOOKS, table, source
from build_prediction_ranking import compare_predictions

COMMIT = '4f45615e374d102c5b88473ee559c280e51c2b0f'
NOTEBOOK = 'notebooks/experiments/AES_300_E8_ContextPair_LR5e6_GPU.ipynb'


def audit(sources):
    if set(sources) != {'E0', 'E2', 'E3', 'E4', 'E5', 'E6', 'E8'}:
        raise ValueError('Sediakan tujuh sumber lengkap; E7 belum memiliki hasil.')
    data = extract(sources['E8'], sources['E2'], method='context')
    parameters = {r['Pengaturan']: r['Nilai'] for r in data['parameters']}
    for key, value in [('Learning rate encoder RoBERTa S1', 1e-5),
                       ('Learning rate encoder RoBERTa S3', 5e-6),
                       ('Learning rate proyeksi BGE', 1e-4),
                       ('Learning rate head', 1e-4),
                       ('Batas token konteks S3', 192), ('Batas token soal S3', 64),
                       ('Jumlah fold', 5), ('Maksimum epoch per fold', 25),
                       ('Batas epoch tanpa perbaikan', 5)]:
        if parameters.get(key) != value:
            raise ValueError('Pengaturan E8 tidak sesuai: ' + key)
    data.update(id='E8', label='Konteks pair · LR S3 5e-6',
                scope='SAVED E8 CONTEXT PAIR LR5e6; NO NEW TRAINING',
                source_reference='Workbook mencatat konteks pair S3, LR encoder S3 5e-6, S1 1e-5, head/proyeksi 1e-4. Source rujukan adalah notebook E8 GPU pada commit tetap; notebook eksekusi tidak disertakan, sehingga perangkat runtime dan kecocokan source byte demi byte tidak dibuktikan oleh nama ZIP.',
                limits=[
                    'Satu run; hasil ini belum membuktikan keunggulan statistik atau stabilitas lintas seed.',
                    'S1/S2 E8 mempunyai prediksi berbeda dari kontrol E6/MLP awal. Source E8 melatih seluruh S1/S2/S3; bukan reuse kontrol. Perubahan perangkat dan proses run membatasi atribusi sebab-akibat pada LR/konteks saja.',
                    'Data uji sudah dilihat selama pengembangan. Metode dipilih melalui development/CV; konfirmasi independen memerlukan data baru.'
                ])
    predictions = {}
    for run, path in sources.items():
        if run == 'E0':
            raw = Path(path).read_bytes()
        else:
            with ZipFile(path) as archive:
                raw = archive.read(BOOKS[3])
        predictions[run] = table(raw, 'Nilai Guru dan Prediksi')
    # compare_predictions memeriksa seluruh ID, guru, nilai finite dan seri.
    ranking = compare_predictions(predictions)
    for run in ('E4', 'E6'):
        reference = {str(r['ID Jawaban']): r for r in predictions[run]}
        impact = []
        for scenario in ('S1', 'S2', 'S3'):
            delta = [abs(r['Prediksi '+scenario]-r['Nilai Guru']) -
                     abs(reference[str(r['ID Jawaban'])]['Prediksi '+scenario]-r['Nilai Guru'])
                     for r in predictions['E8']]
            impact.append({'Skenario': scenario, 'new_lower_error': sum(x < -1e-5 for x in delta),
                           'old_lower_error': sum(x > 1e-5 for x in delta),
                           'equal_error': sum(abs(x) <= 1e-5 for x in delta)})
        data.setdefault('comparisonImpact', {})[run] = impact
    (ROOT/'docs/mlp_e8_snapshot.json').write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    target = ROOT/'docs/mlp_all_results_snapshot.json'
    all_results = json.loads(target.read_text())
    all_results['prediction_winners']['E8'] = winner_counts(predictions['E8'])
    all_results['across_experiments'] = ranking
    target.write_text(json.dumps(all_results, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    target = ROOT/'docs/mlp_method_sources.json'
    catalog = json.loads(target.read_text())
    notebook = json.loads(subprocess.check_output(['git', 'show', COMMIT+':'+NOTEBOOK], cwd=ROOT))
    entry = dict(id='E8', name='Konteks pair · LR S3 5e-6', commit=COMMIT, path=NOTEBOOK,
                 notebook=f'https://github.com/MalikHariyanto/Automated_Essay_Scoring/blob/{COMMIT}/{NOTEBOOK}',
                 choice='Menggabungkan konteks soal/referensi pada RoBERTa S3 dengan LR 5e-6. BGE tetap membaca jawaban; head fusion tetap MLP. Seluruh S1/S2/S3 dilatih pada run E8; bukan reuse S1/S2 E6.',
                 source_status=data['source_reference'], papers=['bert', 'stability', 'roberta', 'bge', 'fusion'],
                 evidence={key: {'cell': index, 'source': source(notebook['cells'][index])}
                           for key, index in [('model', 30), ('config', 12), ('fit', 36)]})
    catalog['experiments'] = [e for e in catalog['experiments'] if e['id'] != 'E8'] + [entry]
    target.write_text(json.dumps(catalog, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'id': data['id'], 'metrics': data['metrics'], 'CV': data['cvSummary'],
                      'impact': data['comparisonImpact']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=Path, required=True, help='JSON privat berisi path E0/E2/E3/E4/E5/E6/E8.')
    args = parser.parse_args()
    audit(json.loads(args.sources.read_text()))
