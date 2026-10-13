"""Worker notebook: modul AES berjalan dalam proses Python bersih."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from aes.audit import EligibilityError, validate_inputs_for_training
from aes.cleaning import clean
from aes.inspection import inspect_data, inspect_models
from aes.io import dump, load_config, sha256


def audit_step(cfg):
    df, rubric, report = inspect_data(cfg)
    essay = df.iloc[0].to_dict() if len(df) else None
    preview = {
        'essay': essay, 'rubric': rubric,
        'text_model': clean(essay['text_raw'], **cfg['preprocessing']) if essay else None
    }
    dump(Path(cfg['output']) / 'notebook_preview.json', preview)
    print(f"Audit aktual selesai: {report['rows']} baris; "
          f"{report['complete_label_rows']} esai memiliki empat label sah.")
    print('Laporan audit dan preview disimpan lokal di', cfg['output'])


def supervised_step(cfg, mode):
    df, rubric, report = inspect_data(cfg)
    review_path = Path(cfg['split']['review_file'])
    review = json.loads(review_path.read_text()) if review_path.exists() else None
    try:
        validate_inputs_for_training(df, rubric, review, sha256(cfg['csv']))
    except EligibilityError as exc:
        dump(Path(cfg['output']) / 'notebook_supervised.json', {
            'mode': mode, 'status': 'skip_data_ineligible',
            'teacher_training_run': False, 'blockers': report['training_blockers']
        })
        print('SUPERVISED SKIP — kebutuhan data sebelum training:\n' + str(exc))
        return
    if mode == 'inspect':
        print('Mode inspect selesai. Training tidak dipilih.')
        dump(Path(cfg['output']) / 'notebook_supervised.json', {
            'mode': mode, 'status': 'inspect_only', 'teacher_training_run': False
        })
        return
    from aes.experiment import run
    dump(Path(cfg['output']) / 'notebook_supervised.json', {
        'mode': mode, 'status': 'running', 'training_requested': True
    })
    run(cfg, mode=mode)
    dump(Path(cfg['output']) / 'notebook_supervised.json', {
        'mode': mode, 'status': 'completed', 'teacher_training_run': True
    })


def models_step(cfg):
    from requests.exceptions import RequestException
    from huggingface_hub.errors import HfHubHTTPError
    path = Path(cfg['output']) / 'notebook_models_status.json'
    # Hanya gangguan akses jaringan yang menjadi status; error kode tetap gagal.
    try:
        inspect_models(cfg)
    except (RequestException, HfHubHTTPError) as exc:
        dump(path, {'status': 'checkpoint_access_blocked',
                    'reason': f'{type(exc).__name__}: {str(exc)[:500]}'})
        print('Akses checkpoint belum berhasil:', type(exc).__name__, str(exc)[:500])
        return
    dump(path, {'status': 'forward_completed', 'heads_trained': False})
    print('Forward checkpoint asli selesai; head UNTRAINED, bukan penilaian esai.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=['audit', 'models', 'supervised'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['inspect', 'learning', 'proposal'], default='inspect')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.step == 'audit':
        audit_step(cfg)
    elif args.step == 'models':
        models_step(cfg)
    else:
        supervised_step(cfg, args.mode)


if __name__ == '__main__':
    main()
