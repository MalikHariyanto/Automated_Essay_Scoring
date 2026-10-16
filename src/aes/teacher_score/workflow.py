"""Tahap notebook dalam proses Python baru; hasil dan data tetap privat."""
import json
from pathlib import Path

from .data import config, dump, make_split, read_dataset, run_folder


def run(step, values):
    cfg = config(values)
    bundle = read_dataset(cfg)
    manifest = make_split(bundle, cfg)
    output = run_folder(cfg, bundle)
    dump(output/'dataset_audit.json', bundle.summary)
    dump(output/'split_manifest.json', manifest)
    dump(Path(cfg['output'])/'active_run.json', {'folder': str(output.resolve())})
    position = cfg['trace_position']
    if not 0 <= position < len(manifest['test']):
        raise ValueError('trace_position harus indeks di holdout test.')
    index = manifest['test'][position]
    text = cfg.get('new_answer', '').strip() or bundle.rows[index]['text']
    if step == 'audit':
        dump(output/'preview.json', {'row_id': bundle.rows[index]['row_id'], 'text': bundle.rows[index]['text'],
                                    'teacher_score': float(bundle.y[index]), 'is_held_out': True})
        print(json.dumps({'audit': bundle.summary, 'split_counts': {k: manifest[k] for k in
                         ('development_rows', 'test_rows', 'development_groups', 'test_groups', 'actual_test_row_fraction')}}, indent=2), flush=True)
        return
    if step == 'encode':
        from .models import bge_features
        bge_features(bundle, cfg, output)
    elif step == 'train':
        from .models import bge_features
        from .experiment import train_all
        bge = bge_features(bundle, cfg, output)
        train_all(bundle, manifest, bge, cfg, output)
    elif step == 'report':
        file = output/'final_report.json'
        print(file.read_text() if file.exists() else 'BELUM DILATIH: tidak ada hasil evaluasi penelitian.', flush=True)
    elif step == 'predict':
        from .experiment import score_answer
        if not (output/'final_report.json').exists():
            raise RuntimeError('Selesaikan training tiga skenario sebelum scoring.')
        dump(output/'prediction_input.json', {'text': text, 'source': 'new_answer' if cfg.get('new_answer', '').strip() else 'held-out example',
                                            'added_to_training_in_prediction_step': False})
        score_answer(text, cfg, output)
    elif step == 'vectors':
        from .vectors import vector_lab
        vector_lab(text, cfg, output)
    else:
        raise ValueError('Tahap tidak dikenal: '+step)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('step', choices=['audit', 'encode', 'train', 'report', 'predict', 'vectors'])
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run(args.step, json.loads(Path(args.config).read_text()))
