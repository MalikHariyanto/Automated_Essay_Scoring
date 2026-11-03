"""Build-time sources, embedded as individual visible Python cells in E37.

Globals are supplied by the preceding native notebook sections, never a worker.
"""


def select_lr_from_cv(fold_rows, seeds, control_lr, candidate_lr):
    """Pilih resep dari mean RMSE CV seluruh seed; tidak menerima hasil test."""
    if len(seeds) != 3 or len(set(seeds)) != 3 or control_lr == candidate_lr:
        raise ValueError('Gunakan tiga seed berbeda dan dua LR berbeda.')
    expected = {(lr, seed, fold) for lr in (control_lr, candidate_lr)
                for seed in seeds for fold in range(1, 6)}
    keys = [(r['encoder_lr'], r['seed'], r['fold']) for r in fold_rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('Seleksi membutuhkan tepat dua LR × tiga seed × lima fold lengkap.')
    if not all(math.isfinite(r['RMSE']) and r['RMSE'] >= 0 for r in fold_rows):
        raise ValueError('RMSE CV harus finite dan nonnegatif.')
    per_seed = []
    for lr in (control_lr, candidate_lr):
        for seed in seeds:
            values = [r['RMSE'] for r in fold_rows if r['encoder_lr'] == lr and r['seed'] == seed]
            per_seed.append({'encoder_lr': lr, 'seed': seed,
                             'mean_rmse_cv': float(np.mean(values)),
                             'sd_between_folds': float(np.std(values, ddof=1)), 'folds': len(values)})
    summaries = []
    for lr in (control_lr, candidate_lr):
        values = [r['mean_rmse_cv'] for r in per_seed if r['encoder_lr'] == lr]
        summaries.append({'encoder_lr': lr, 'mean_rmse_cv': float(np.mean(values)),
                          'sd_between_seed_means': float(np.std(values, ddof=1)), 'seeds': len(values)})
    # LR kontrol menang bila rerata sama; aturan ditentukan sebelum test.
    winner = min(summaries, key=lambda r: (r['mean_rmse_cv'], r['encoder_lr'] != control_lr))
    differences = []
    for seed in seeds:
        control = next(r['mean_rmse_cv'] for r in per_seed if r['seed'] == seed and r['encoder_lr'] == control_lr)
        candidate = next(r['mean_rmse_cv'] for r in per_seed if r['seed'] == seed and r['encoder_lr'] == candidate_lr)
        differences.append({'seed': seed, 'candidate_minus_control_cv': candidate-control})
    return {'selected_encoder_lr': winner['encoder_lr'], 'per_seed': per_seed,
            'summary': summaries, 'paired_differences': differences,
            'selection_metric': 'mean of three seed means of five validation-fold RMSEs',
            'tie_rule': 'keep E30 control', 'test_used_for_selection': False,
            'primary_seed': seeds[0], 'primary_seed_rule': 'first preregistered seed; never best test seed'}


def discard_completed_cv_progress(destination, fold_number):
    """Hapus dua slot bobot fold selesai; JSON hasil CV tetap untuk resume."""
    destination = Path(destination)
    if not (destination/f'cv_fold_{fold_number}.json').is_file():
        raise ValueError('Jangan menghapus checkpoint sebelum JSON hasil fold tersimpan.')
    for suffix in ('_slot_A.pt', '_slot_B.pt', '.json'):
        (destination/f'progress_fold_{fold_number}{suffix}').unlink(missing_ok=True)


def run_paired_cv(bundle, manifest, bge, cfg, study_output):
    """Enam run CV pada fold yang sama; tahap ini tidak refit atau membuka test."""
    study_output = Path(study_output)
    design = cfg['paired_study']
    seeds = design['training_seeds']
    if len(seeds) != 3 or len(set(seeds)) != 3 or cfg['folds'] != 5:
        raise ValueError('E37 memerlukan tiga seed berbeda dan lima fold tetap.')
    trials, fold_rows = [], []
    for label, lr in [('E30_control', design['control_encoder_lr']),
                      ('E37_candidate', design['candidate_encoder_lr'])]:
        for seed in seeds:
            print(f'CV {len(trials)+1}/6 · {label} · encoder LR={lr:g} · seed={seed}', flush=True)
            trial_cfg = config({**cfg, 'seed': seed, 'encoder_lr': lr, 's3_encoder_lr': lr,
                                'output': str(study_output/'paired_runs')})
            if make_split(bundle, trial_cfg) != manifest:
                raise ValueError('Seed training tidak boleh mengubah holdout atau fold.')
            folder = run_folder(trial_cfg, bundle)
            dump(folder/'split_manifest.json', manifest)
            report = train_all(bundle, manifest, bge, trial_cfg, folder, cv_only=True)
            trial = {'recipe': label, 'encoder_lr': lr, 'seed': seed,
                     'cfg': trial_cfg, 'folder': str(folder), 'cv_report_sha256': digest(folder/'cv_report.json')}
            trials.append(trial)
            for row in report['cv']:
                fold_rows.append({'recipe': label, 'encoder_lr': lr, 'seed': seed,
                                  'fold': row['Fold'], **{k: row[k] for k in ('RMSE', 'Pearson', 'QWK', 'AC2')}})
    selection = select_lr_from_cv(fold_rows, seeds, design['control_encoder_lr'], design['candidate_encoder_lr'])
    lock = {'source_sha256': bundle.source_sha256, 'split_sha256': manifest['split_sha256'],
            'notebook_code_sha256': NOTEBOOK_CODE_SHA256, 'design': design,
            'selection': selection, 'cv_report_hashes': [t['cv_report_sha256'] for t in trials]}
    path = study_output/'paired_selection_locked.json'
    if path.exists() and json.loads(path.read_text()) != lock:
        raise ValueError('Seleksi CV yang telah dikunci berubah; gunakan percobaan baru, jangan tuning dari test.')
    dump(path, lock)
    pd.DataFrame(fold_rows).to_csv(study_output/'paired_cv_all_folds.csv', index=False)
    print('Pilihan CV terkunci: encoder LR =', selection['selected_encoder_lr'], flush=True)
    return {'trials': trials, 'fold_rows': fold_rows, 'selection': selection,
            'study_output': str(study_output), 'lock': lock}


def evaluate_selected_recipe(study, bundle, manifest, bge):
    """Evaluasi tiga seed resep terpilih; tidak memilih seed atau ensemble dari test."""
    path = Path(study['study_output'])/'paired_selection_locked.json'
    if json.loads(path.read_text()) != study['lock']:
        raise ValueError('Kunci seleksi CV berubah sebelum final test.')
    if any(digest(Path(t['folder'])/'cv_report.json') != t['cv_report_sha256'] for t in study['trials']):
        raise ValueError('Hasil CV berubah setelah pemilihan LR.')
    lr = study['selection']['selected_encoder_lr']
    selected = [t for t in study['trials'] if t['encoder_lr'] == lr]
    rows, predictions, primary = [], [], None
    for trial in selected:
        print(f'Refit + test resep terpilih · LR={lr:g} · seed={trial["seed"]}', flush=True)
        report = train_all(bundle, manifest, bge, trial['cfg'], trial['folder'])
        for row in report['rows']:
            rows.append({'recipe': trial['recipe'], 'seed': trial['seed'], 'encoder_lr': lr, **row})
        frame = pd.read_csv(Path(trial['folder'])/'S3/test_predictions.csv')
        predictions.extend({'seed': trial['seed'], 'encoder_lr': lr, **r} for r in frame.to_dict('records'))
        if trial['seed'] == study['selection']['primary_seed']:
            primary = {**trial, 'report': report}
    if primary is None or len(selected) != 3:
        raise ValueError('Ketiga seed terpilih wajib dilaporkan; tidak boleh mengambil seed test terbaik.')
    summary = []
    for metric in ('RMSE', 'Pearson', 'QWK', 'AC2'):
        values = [r[metric] for r in rows]
        valid = [v for v in values if v is not None and math.isfinite(v)]
        summary.append({'metric': metric, 'mean': float(np.mean(valid)) if valid else None,
                        'sd_between_seeds': float(np.std(valid, ddof=1)) if len(valid) > 1 else None,
                        'defined_seeds': len(valid), 'total_seeds': len(selected)})
    result = {'primary': primary, 'test_rows': rows, 'test_summary': summary, 'test_predictions': predictions}
    dump(Path(study['study_output'])/'paired_final_summary.json', {
        'selection': study['selection'], 'test_rows': rows, 'test_summary': summary,
        'test_evaluations_per_selected_seed': 1, 'ensemble': False})
    return result


def export_paired_results(study, final, journal):
    """Masukkan semua hasil LR/seed ke laporan standar; tidak menimpa hasil lama."""
    histories = []
    for trial in study['trials']:
        for path in sorted((Path(trial['folder'])/'S3').glob('cv_fold_*.json')):
            result = json.loads(path.read_text())
            for entry in result['training']['history']:
                histories.append({'recipe': trial['recipe'], 'encoder_lr': trial['encoder_lr'],
                    'seed': trial['seed'], 'fold': result['fold'], 'epoch': entry['epoch'],
                    'train_mse_scaled': entry['train_mse_scaled'],
                    'selected_epoch': result['training']['best_epoch'],
                    **{k: entry.get('validation', {}).get(k) for k in ('RMSE', 'Pearson', 'QWK', 'AC2')}})
    journal.save('16_paired_lr_seed', {
        'CV_semua_LR_seed': pd.DataFrame(study['fold_rows']),
        'CV_per_seed': pd.DataFrame(study['selection']['per_seed']),
        'CV_ringkasan_LR': pd.DataFrame(study['selection']['summary']),
        'Selisih_CV_berpasangan': pd.DataFrame(study['selection']['paired_differences']),
        'Seleksi_terkunci': pd.DataFrame([{'field': k, 'value': v} for k, v in study['selection'].items()
                                       if k not in ('per_seed', 'summary', 'paired_differences')]),
        'Riwayat_semua_LR_seed': pd.DataFrame(histories),
        'Test_per_seed': pd.DataFrame(final['test_rows']),
        'Test_mean_SD_seed': pd.DataFrame(final['test_summary']),
        'Prediksi_semua_seed': pd.DataFrame(final['test_predictions']),
    }, 'E37 paired study; pemilihan LR hanya CV; semua seed test resep terpilih dilaporkan',
        primary_seed=study['selection']['primary_seed'], ensemble=False)
