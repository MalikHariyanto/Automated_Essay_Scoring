"""CV di development, refit, lalu test sekali untuk tiga skenario yang dikunci."""
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from . import SCOPE
from .data import digest, dump
from .models import Engine, SingleScoreModel, device_name, encoder, metrics, release

SCENARIOS = ('S1', 'S2', 'S3')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_model(scenario, cfg):
    return SingleScoreModel(scenario, encoder(cfg, 'roberta') if scenario != 'S2' else None,
                            dropout=cfg['dropout'])


def fit(scenario, training, validation, bundle, bge, cfg, epochs, seed):
    seed_everything(seed)
    device = device_name(cfg)
    model = make_model(scenario, cfg).to(device)
    engine = Engine(model, bundle.texts, bge, cfg, device)
    enc = [p for name, p in model.named_parameters() if name.startswith('roberta.') and p.requires_grad]
    head = [p for name, p in model.named_parameters() if not name.startswith('roberta.') and p.requires_grad]
    groups = ([{'params': enc, 'lr': cfg['encoder_lr']}] if enc else [])+[
        {'params': head, 'lr': cfg['head_lr']}]
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg['weight_decay'])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader([int(i) for i in training], batch_size=cfg['effective_batch'],
                        shuffle=True, generator=generator, num_workers=0)
    total_steps = len(loader)*epochs
    warmup = min(int(cfg['warmup_ratio']*total_steps), max(0, total_steps-1))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
    use_cuda = device == 'cuda'
    dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda and dtype == torch.float16)
    selected_parameters = {}
    for name, parameter in model.named_parameters():
        group = name.split('.')[0]
        if parameter.requires_grad and group not in selected_parameters:
            selected_parameters[group] = (name, parameter.detach().reshape(-1)[:32].cpu().clone())
    best_state, best_epoch, best_rmse = None, 0, float('inf')
    history, active_steps, optimizer_steps, skipped_steps = [], 0, 0, 0
    for epoch in range(1, epochs+1):
        model.train()
        loss_total = 0.
        for effective in loader:
            batch = effective.tolist()
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, len(batch), cfg['micro_batch']):
                ids = batch[start:start+cfg['micro_batch']]
                targets = torch.as_tensor((bundle.y[ids]-cfg['score_min'])/(cfg['score_max']-cfg['score_min']),
                                          dtype=torch.float32, device=device).reshape(-1, 1)
                with torch.autocast(device_type=device, dtype=dtype, enabled=use_cuda):
                    prediction = engine.forward(ids).float()
                    if prediction.shape != targets.shape:
                        raise ValueError('Shape prediksi/target harus [B,1], tanpa broadcasting.')
                    loss = torch.mean((prediction-targets)**2)
                if not torch.isfinite(loss):
                    raise RuntimeError('Loss nonfinite; run dihentikan, tidak diklaim berhasil.')
                scaler.scale(loss*(len(ids)/len(batch))).backward()
                loss_total += loss.item()*len(ids)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
            lr_active = any(g['lr'] > 0 for g in optimizer.param_groups)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.is_enabled() and scaler.get_scale() < previous_scale
            if skipped:
                skipped_steps += 1
            else:
                active_steps += int(lr_active)
                scheduler.step()
                optimizer_steps += 1
        row = {'epoch': epoch, 'train_mse_scaled': loss_total/len(training),
               'optimizer_steps': optimizer_steps, 'active_lr_steps': active_steps, 'amp_skipped_steps': skipped_steps,
               'effective_batch_target': cfg['effective_batch'], 'micro_batch': cfg['micro_batch']}
        if validation is not None:
            predicted = engine.predict(validation)
            row['validation'] = metrics(bundle.y[validation], predicted, cfg)
            if row['validation']['RMSE'] < best_rmse:
                best_rmse, best_epoch = row['validation']['RMSE'], epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            message = f" | RMSE validasi: {row['validation']['RMSE']:.4f} poin"
        else:
            best_epoch = epoch
            message = ' | pelatihan final; data uji belum dibuka'
        history.append(row)
        print(f'{scenario} · epoch {epoch}/{epochs} · loss latih (MSE, target 0–1): {row["train_mse_scaled"]:.6f}'+message, flush=True)
    if active_steps == 0:
        raise RuntimeError('Tidak ada pembaruan dengan LR aktif.')
    named = dict(model.named_parameters())
    changed = {group: not torch.equal(before, named[name].detach().reshape(-1)[:32].cpu())
               for group, (name, before) in selected_parameters.items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    info = {'scenario': scenario, 'best_epoch': best_epoch, 'epochs_completed': len(history),
            'history': history, 'optimizer_steps': optimizer_steps, 'active_lr_steps': active_steps, 'amp_skipped_steps': skipped_steps,
            'parameter_samples_changed': changed, 'BGE_encoder_trained': False,
            'selection': 'minimum validation RMSE in original score units' if validation is not None else 'fixed refit epochs selected from CV'}
    return model, engine, info


def save_model(path, model, cfg, bundle, manifest, info):
    payload = {'scope': SCOPE, 'trained': True, 'scenario': model.scenario,
               'source_sha256': bundle.source_sha256, 'split_sha256': manifest['split_sha256'],
               'config': cfg, 'training': info,
               'state': {k: v.detach().cpu() for k, v in model.state_dict().items()}}
    torch.save(payload, path)


def load_model(path, cfg):
    # Checkpoint hanya dari workflow ini; weights_only membatasi objek pickle.
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('scope') != SCOPE or payload.get('trained') is not True:
        raise ValueError('Checkpoint bukan model satu skor guru yang telah dilatih.')
    for key in ('roberta', 'bge', 'score_min', 'score_max', 'roberta_max_length', 'roberta_overlap', 'bge_max_length'):
        if payload['config'][key] != cfg[key]:
            raise ValueError('Metadata checkpoint berbeda: '+key)
    model = make_model(payload['scenario'], cfg)
    model.load_state_dict(payload['state'], strict=True)
    model.eval()
    return model, payload


def train_all(bundle, manifest, bge, cfg, output, observer=None):
    output = Path(output)
    final_path = output/'final_report.json'
    if final_path.exists():
        report = json.loads(final_path.read_text())
        if report['source_sha256'] != bundle.source_sha256 or report['split_sha256'] != manifest['split_sha256']:
            raise ValueError('Hasil lama memakai sumber/split berbeda.')
        for scenario in SCENARIOS:
            if digest(output/scenario/'final.pt') != report['checkpoints'][scenario]:
                raise ValueError('Checkpoint hasil lama berubah.')
            if digest(output/scenario/'test_predictions.csv') != report['prediction_files'][scenario]:
                raise ValueError('Prediksi test tersimpan berubah.')
        print('Memakai hasil tersimpan yang cocok dengan data, pengaturan, dan versi kode. Pelatihan serta evaluasi data uji tidak diulang.', flush=True)
        return {**report, 'reused_completed_run': True}
    cv_rows, selected_epochs, cv_folds = [], {}, {}
    for scenario in SCENARIOS:
        destination = output/scenario
        destination.mkdir(exist_ok=True)
        completed = []
        for fold in manifest['folds']:
            file = destination/f'cv_fold_{fold["fold"]}.json'
            if file.exists():
                result = json.loads(file.read_text())
                if result['split_sha256'] != manifest['split_sha256'] or result['fold'] != fold['fold']:
                    raise ValueError('Cache fold memakai split berbeda.')
                print(f'{scenario}: memakai hasil validasi silang fold {fold["fold"]} yang sudah selesai.', flush=True)
            else:
                print(f'{scenario}: validasi silang, fold {fold["fold"]} dari {cfg["folds"]}. Data uji disimpan untuk evaluasi akhir.', flush=True)
                model, engine, info = fit(scenario, fold['train'], fold['validation'], bundle, bge, cfg,
                                          cfg['epochs'], cfg['seed']+fold['fold'])
                prediction = engine.predict(fold['validation'])
                result = {'fold': fold['fold'], 'split_sha256': manifest['split_sha256'],
                          'training': info, 'metrics': metrics(bundle.y[fold['validation']], prediction, cfg),
                          'validation_indices': fold['validation'], 'prediction': prediction.tolist()}
                dump(file, result)
                del engine, model
                release()
            completed.append(result)
            if observer is not None:
                observer('cv_fold', scenario, result)
            cv_rows.append({'Skenario': scenario, 'Fold': fold['fold'],
                            **{k: result['metrics'][k] for k in ('QWK', 'AC2', 'RMSE', 'Pearson')}})
        cv_folds[scenario] = completed
        selected_epochs[scenario] = int(math.ceil(np.median([r['training']['best_epoch'] for r in completed])))
    pd.DataFrame(cv_rows).to_csv(output/'cv_metrics.csv', index=False)
    selection = {'scope': SCOPE, 'selected_epochs': selected_epochs,
                 'method': 'ceil median best validation-RMSE epoch across development folds',
                 'all_scenarios_locked_before_final_test': True, 'split_sha256': manifest['split_sha256']}
    dump(output/'selection_locked.json', selection)
    # Seluruh refit selesai sebelum membaca prediksi/skor test untuk pelaporan.
    lock = output/'refit_locked.json'
    if lock.exists():
        previous = json.loads(lock.read_text())
        checksums = previous['checkpoints']
        if previous['selection'] != selection or any(digest(output/s/'final.pt') != checksums[s] for s in SCENARIOS):
            raise ValueError('Model/config final yang sudah dikunci berubah; jangan mengulang test untuk tuning.')
    else:
        checksums = {}
        for scenario in SCENARIOS:
            model, engine, info = fit(scenario, manifest['development'], None, bundle, bge, cfg,
                                      selected_epochs[scenario], cfg['seed'])
            path = output/scenario/'final.pt'
            save_model(path, model, cfg, bundle, manifest, info)
            if observer is not None:
                observer('refit', scenario, info)
            checksums[scenario] = digest(path)
            del engine, model
            release()
        dump(lock, {'scope': SCOPE, 'checkpoints': checksums,
                    'selection': selection, 'test_evaluated': False})
    rows, prediction_files = [], {}
    for scenario in SCENARIOS:
        completed_test = output/scenario/'test_result.json'
        if completed_test.exists():
            cached = json.loads(completed_test.read_text())
            if cached['checkpoint_sha256'] != checksums[scenario] or cached['split_sha256'] != manifest['split_sha256'] or digest(output/scenario/'test_predictions.csv') != cached['prediction_sha256']:
                raise ValueError('Hasil test parsial berubah; tidak dievaluasi ulang.')
            rows.append(cached['row'])
            prediction_files[scenario] = cached['prediction_sha256']
            continue
        model, payload = load_model(output/scenario/'final.pt', cfg)
        if payload['source_sha256'] != bundle.source_sha256 or payload['split_sha256'] != manifest['split_sha256']:
            raise ValueError('Checkpoint final berbeda sumber/split.')
        model.to(device_name(cfg))
        engine = Engine(model, bundle.texts, bge, cfg, device_name(cfg))
        prediction = engine.predict(manifest['test'])
        score = metrics(bundle.y[manifest['test']], prediction, cfg)
        rows.append({'Skenario': scenario, **{k: score[k] for k in ('QWK', 'AC2', 'RMSE', 'Pearson')},
                     'Status': 'test held-out; satu skor guru', 'undefined_reasons': score['undefined_reasons'],
                     'out_of_range_fraction': score['out_of_range_fraction']})
        frame = pd.DataFrame({'row_id': [bundle.rows[i]['row_id'] for i in manifest['test']],
                              'skor_guru': bundle.y[manifest['test']], 'prediksi_mentah': prediction,
                              'selisih': prediction-bundle.y[manifest['test']]})
        path = output/scenario/'test_predictions.csv'
        frame.to_csv(path, index=False)
        prediction_files[scenario] = digest(path)
        dump(completed_test, {'row': rows[-1], 'checkpoint_sha256': checksums[scenario],
                              'split_sha256': manifest['split_sha256'], 'prediction_sha256': digest(path)})
        if observer is not None:
            observer('test', scenario, {'metrics': rows[-1], 'predictions': frame.to_dict('records')})
        del engine, model
        release()
    pd.DataFrame([{k: row[k] for k in ('Skenario', 'QWK', 'AC2', 'RMSE', 'Pearson', 'Status')} for row in rows]).to_csv(output/'test_metrics.csv', index=False)
    report = {'scope': SCOPE, 'source_sha256': bundle.source_sha256, 'split_sha256': manifest['split_sha256'],
              'rows': rows, 'checkpoints': checksums, 'prediction_files': prediction_files,
              'selected_epochs': selected_epochs, 'test_rows': manifest['test_rows'],
              'test_evaluations_per_scenario': 1, 'reused_completed_run': False,
              'cv': cv_rows, 'four_trait_labels_available': False}
    dump(final_path, report)
    return report


def score_answer(text, cfg, output):
    """Prediksi baru tanpa memerlukan nilai guru jawaban yang akan diprediksi."""
    from aes.cleaning import clean
    from .data import Bundle
    from .models import bge_features
    output = Path(output)
    text = clean(text)
    if not text.strip():
        raise ValueError('Jawaban prediksi kosong.')
    source = hashlib.sha256(text.encode()).hexdigest()
    # Inference tidak mempunyai target/kelompok training; jangan mengisi skor 0.
    sample = Bundle([{'text': text}], np.empty(0), np.empty(0, dtype=int), source, {})
    features = bge_features(sample, cfg, output/'prediction_encoding')
    rows = []
    report = json.loads((output/'final_report.json').read_text())
    for scenario in SCENARIOS:
        path = output/scenario/'final.pt'
        if digest(path) != report['checkpoints'][scenario]:
            raise ValueError('Checkpoint prediksi berbeda dari final yang dievaluasi.')
        model, payload = load_model(path, cfg)
        model.to(device_name(cfg))
        engine = Engine(model, [text], features, cfg, device_name(cfg))
        prediction = float(engine.predict([0])[0])
        cats = np.arange(cfg['score_min'], cfg['score_max']+1)
        rounded = int(cats[np.abs(cats-np.clip(prediction, cats[0], cats[-1])).argmin()])
        rows.append({'Skenario': scenario, 'prediksi_mentah': prediction, 'kategori_salinan': rounded,
                     'di_luar_rentang': not cfg['score_min'] <= prediction <= cfg['score_max'],
                     'target': 'satu skor guru; bukan empat rubrik'})
        del engine, model
        release()
    dump(output/'one_answer_prediction.json', {'rows': rows})
    return rows
