"""Peta rentang guru/prediksi dari Excel asli; output hanya hitungan agregat."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from zipfile import ZipFile
from build_mlp_study import ROOT, table
from build_prediction_ranking import compare_predictions


def matrix_for(rows, scenario):
    matrix = [[0]*11 for _ in range(11)]
    for row in rows:
        teacher, prediction = row['Nilai Guru'], row['Prediksi '+scenario]
        if not isinstance(prediction, (int, float)) or not math.isfinite(prediction):
            raise ValueError('Prediksi tidak valid.')
        clipped = min(100, max(0, prediction))
        derived = min(range(101), key=lambda category: abs(category-clipped))
        # E0 tidak menyertakan kolom pembulatan. Kategori terdekat memakai
        # aturan evaluator: clip 0–100; tie .5 ke kategori lebih rendah.
        ordinal = row.get('Pembulatan '+scenario, derived)
        if not all(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 100
                   for x in (teacher, ordinal)) or ordinal != int(ordinal):
            raise ValueError('Skor/pembulatan tidak valid.')
        if ordinal != derived:
            raise ValueError('Pembulatan tersimpan berbeda dari aturan kategori.')
        # Bukan pembulatan tampilan browser atau pembulatan banker's rounding.
        matrix[min(10, int(teacher)//10)][min(10, int(ordinal)//10)] += 1
    return matrix


def build(sources):
    if set(sources) != {'E0','E2','E3','E4','E5','E6','E8'}:
        raise ValueError('Tujuh sumber diperlukan.')
    predictions, hashes = {}, {}
    for run, path in sources.items():
        raw = Path(path).read_bytes()
        hashes[run] = hashlib.sha256(raw).hexdigest()
        if run != 'E0':
            with ZipFile(path) as archive:
                raw = archive.read('04_Prediksi_dan_Evaluasi.xlsx')
        predictions[run] = table(raw, 'Nilai Guru dan Prediksi')
    compare_predictions(predictions)  # Verifikasi 60 ID unik, skor guru dan sumber sama.
    matrices = {run: {s: matrix_for(rows, s) for s in ('S1','S2','S3')}
                for run, rows in predictions.items()}
    for run, path in [('E2','docs/mlp_study_snapshot.json'),('E6','docs/mlp_e6_study_snapshot.json')]:
        old = json.loads((ROOT/path).read_text())
        if matrices[run] != old['matrices']:
            raise ValueError('Matriks lama berubah: '+run)
    result = dict(scope='AGGREGATE TEST COUNTS ONLY; NO NEW TRAINING', answers=60,
                  bands=[f'{i*10}–{i*10+9}' if i<10 else '100' for i in range(11)],
                  source_sha256=hashes, matrices=matrices,
                  category_origin='E0 dihitung dari prediksi mentah (nearest 0–100, tie .5 ke bawah); E2–E8 memakai kategori Excel, diverifikasi dengan aturan yang sama.',
                  rule='Baris = rentang nilai guru. Kolom = rentang salinan kategori prediksi dari Excel. Diagonal = rentang sama, bukan nilai persis sama. Kategori 0–100 tetap dipakai evaluator; kelompok 10 poin hanya untuk visualisasi.')
    (ROOT/'docs/mlp_score_maps_snapshot.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'experiments': len(matrices), 'matrices': 21, 'answers_each': 60,
                      'previous_E2_E6_matrices_unchanged': True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--sources',type=Path,required=True)
    args = parser.parse_args();build(json.loads(args.sources.read_text()))
