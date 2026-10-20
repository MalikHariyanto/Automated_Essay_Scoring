"""Tahapan yang dieksekusi inline di notebook dan dapat diuji sebagai modul."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aes.windows import batch_windows, content_windows, mean_by_essay
from .data import digest
from .excel import vector_frame
from .experiment import load_model
from .models import Engine, device_name, encoder, release, tokenizer


def export_tokens(bundle, cfg, journal):
    rows, statistics = [], []
    roberta_tok, bge_tok = tokenizer(cfg, 'roberta'), tokenizer(cfg, 'bge')
    # Rekam input window RoBERTa beserta padding/attention mask aktual per jawaban.
    for index, text in enumerate(bundle.texts):
        windows, mapping, stats = batch_windows([text], roberta_tok, 'cpu', cfg['roberta_max_length'], cfg['roberta_overlap'])
        row_id = bundle.rows[index]['row_id']
        statistics.append({'row_id': row_id, 'model': 'roberta', **stats[0]})
        for window, ids in enumerate(windows['input_ids'].tolist()):
            masks = windows['attention_mask'][window].tolist()
            pieces = roberta_tok.convert_ids_to_tokens(ids)
            for position, (identifier, token, mask) in enumerate(zip(ids, pieces, masks)):
                rows.append({'row_id': row_id, 'window': window, 'position': position,
                             'token_id': identifier, 'token': token, 'attention_mask': mask})
    roberta_frame = pd.DataFrame(rows)
    rows = []
    # Urutan/batch/padding BGE sama dengan ekstraksi fitur frozen sesudah tahap ini.
    for start in range(0, len(bundle.rows), cfg['bge_batch']):
        texts = bundle.texts[start:start+cfg['bge_batch']]
        encoded = bge_tok(texts, padding=True, truncation=True, max_length=cfg['bge_max_length'], return_tensors='pt')
        for local, ids in enumerate(encoded['input_ids'].tolist()):
            row_id = bundle.rows[start+local]['row_id']
            full_length = len(bge_tok.encode(texts[local], add_special_tokens=True))
            statistics.append({'row_id': row_id, 'model': 'bge', 'tokens_before_truncation': full_length,
                               'kept_including_special': int(encoded['attention_mask'][local].sum()),
                               'truncated': full_length > cfg['bge_max_length']})
            for position, (identifier, token, mask) in enumerate(zip(ids, bge_tok.convert_ids_to_tokens(ids), encoded['attention_mask'][local].tolist())):
                rows.append({'row_id': row_id, 'position': position, 'token_id': identifier,
                             'token': token, 'attention_mask': mask})
    journal.save('04_tokenisasi', {'RoBERTa_input': roberta_frame, 'BGE_input': pd.DataFrame(rows),
                                 'Panjang_dan_coverage': pd.DataFrame(statistics)}, 'Token/ID/window/padding aktual',
                 source={'roberta': cfg['roberta'], 'bge': cfg['bge']})
    return roberta_frame, pd.DataFrame(rows)


def roberta_pretrained_vectors(bundle, cfg, journal):
    device = device_name(cfg)
    model, tok = encoder(cfg, 'roberta').to(device), tokenizer(cfg, 'roberta')
    model.eval()
    vectors, cls_vectors, window_ids = [], [], []
    with torch.no_grad():
        for index, text in enumerate(bundle.texts):
            windows, mapping, stats = batch_windows([text], tok, device, cfg['roberta_max_length'], cfg['roberta_overlap'])
            # [:,0,:] adalah hidden state token CLS, bukan seluruh token dirata-rata.
            cls = model(**windows).last_hidden_state[:, 0, :].float()
            vectors.append(mean_by_essay(cls, mapping, 1).cpu().numpy())
            cls_vectors.append(cls.cpu().numpy())
            window_ids.extend(f'{bundle.rows[index]["row_id"]}_window_{j}' for j in range(len(cls)))
            if (index+1) % 25 == 0:
                print(f'RoBERTa sebelum pelatihan: vektor {index+1} dari {len(bundle.rows)} jawaban selesai.', flush=True)
    values = np.concatenate(vectors)
    ids = [row['row_id'] for row in bundle.rows]
    journal.save('05_roberta_pretrained', {'Vektor_jawaban_768': vector_frame(values, ids),
                                          'CLS_tiap_window': vector_frame(np.concatenate(cls_vectors), window_ids)},
                 'Embedding RoBERTa SEBELUM fine-tuning; seluruh koordinat', source=cfg['roberta'],
                 pooling='mean window CLS; bukan sentence similarity yang sudah dikalibrasi')
    del model, tok
    release()
    return values


def export_word_sentence_vectors(result, journal, name):
    sheets = {}
    for role in ('roberta', 'bge'):
        report = result[role]
        with np.load(report['npz'], allow_pickle=False) as arrays:
            for key, id_key, metadata in [('token_initial','token_index',report['tokens']),
                                          ('token_contextual','token_index',report['tokens']),
                                          ('word_initial','word_index',report['words']),
                                          ('word_contextual','word_index',report['words']),
                                          ('sentence_cls','sentence_index',report['sentences'])]:
                values = arrays[key]
                frame = vector_frame(values, [row[id_key] for row in metadata])
                label_key = 'token' if key.startswith('token') else 'word' if key.startswith('word') else 'text'
                frame.insert(1, label_key, [row[label_key] for row in metadata])
                if key.startswith('word'):
                    frame.insert(2, 'covered', [row['covered'] for row in metadata])
                sheets[role+'_'+key] = frame
        sheets[role+'_metadata_token'] = pd.DataFrame(report['tokens'])
        sheets[role+'_metadata_word'] = pd.DataFrame(report['words'])
        sheets[role+'_metadata_sentence'] = pd.DataFrame(report['sentences'])
        sheets[role+'_kata_berulang'] = pd.DataFrame(report['repeated_words'])
    journal.save(name, sheets, 'Vektor token/kata/kalimat LENGKAP untuk SATU jawaban',
                 source={r: result[r]['source'] for r in ('roberta','bge')},
                 methods={r:{k:result[r][k] for k in ('word_method','sentence_method','uncovered_tokens','window_starts')} for r in ('roberta','bge')},
                 interpretation=result['interpretation'],
                 missing_vectors='Sel kosong berarti NaN/tidak tercakup; lihat kolom covered/coverage, bukan vektor nol.')


def ordinal_copy(prediction, cfg):
    categories = np.arange(cfg['score_min'], cfg['score_max']+1)
    return categories[np.abs(np.clip(prediction, categories[0], categories[-1])[:, None]-categories).argmin(axis=1)]


def prediction_frame(bundle, indices, prediction, cfg):
    teacher = bundle.y[indices]
    ordinal = ordinal_copy(np.asarray(prediction), cfg)
    return pd.DataFrame({'row_id': [bundle.rows[i]['row_id'] for i in indices],
                         'skor_guru': teacher, 'prediksi_mentah': prediction,
                         'kategori_QWK_AC2': ordinal, 'selisih_mentah': prediction-teacher,
                         'selisih_kuadrat': (prediction-teacher)**2,
                         'bobot_agreement_kuadratik': 1-((ordinal-teacher)/(cfg['score_max']-cfg['score_min']))**2})


def export_training_event(kind, scenario, result, bundle, cfg, journal):
    if kind == 'cv_fold':
        history = []
        for entry in result['training']['history']:
            history.append({**{k: v for k, v in entry.items() if k != 'validation'},
                            **entry.get('validation', {})})
        predictions = prediction_frame(bundle, result['validation_indices'], np.asarray(result['prediction']), cfg)
        journal.save(f'09_{scenario}_fold_{result["fold"]}', {'Riwayat_epoch': pd.DataFrame(history),
                                                           'Prediksi_validation': predictions,
                                                           'Metrik_validation': pd.DataFrame([result['metrics']])},
                     'CV development; bukan final test', scenario=scenario,
                     selected_epoch=result['training']['best_epoch'], encoder_fine_tuned=scenario != 'S2')
    elif kind == 'refit':
        journal.save(f'10_{scenario}_refit', {'Riwayat_epoch': pd.DataFrame(result['history']),
                                            'Bukti_parameter': pd.DataFrame([result['parameter_samples_changed']])},
                     'Refit seluruh development; test belum dibuka', scenario=scenario,
                     epochs=result['epochs_completed'], BGE_encoder_trained=False)
    elif kind == 'test':
        raw = pd.DataFrame(result['predictions'])
        lookup = {r['row_id']: i for i, r in enumerate(bundle.rows)}
        ids = [lookup[row_id] for row_id in raw['row_id']]
        expanded = prediction_frame(bundle, ids, raw['prediksi_mentah'].to_numpy(), cfg)
        journal.save(f'11_{scenario}_test', {'Prediksi_test': expanded,
                                           'Metrik_test': pd.DataFrame([result['metrics']])},
                     'Final test satu skor guru', scenario=scenario)


def export_completed_training(report, bundle, cfg, output, journal):
    output = Path(output)
    # Juga membangun ulang Excel dari artefak cocok hash ketika run di-resume.
    for scenario in ('S1','S2','S3'):
        for path in sorted((output/scenario).glob('cv_fold_*.json')):
            export_training_event('cv_fold', scenario, json.loads(path.read_text()), bundle, cfg, journal)
        checkpoint = output/scenario/'final.pt'
        if digest(checkpoint) != report['checkpoints'][scenario]:
            raise ValueError('Checkpoint final berbeda dari yang dievaluasi.')
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        export_training_event('refit', scenario, payload['training'], bundle, cfg, journal)
        del payload
        predicted = pd.read_csv(output/scenario/'test_predictions.csv')
        row = next(x for x in report['rows'] if x['Skenario'] == scenario)
        export_training_event('test', scenario, {'metrics': row, 'predictions': predicted.to_dict('records')}, bundle, cfg, journal)
    cv = pd.DataFrame(report['cv'])
    summary = cv.groupby('Skenario')[['QWK','AC2','RMSE','Pearson']].agg(['mean','std','count'])
    summary.columns = [f'{metric}_{stat}' for metric,stat in summary.columns]
    journal.save('12_evaluasi_tiga_skenario', {'CV_semua_fold': cv, 'CV_mean_SD_n': summary.reset_index(),
                                             'TEST_final': pd.DataFrame(report['rows']),
                                             'Epoch_refit': pd.DataFrame([report['selected_epochs']])},
                 'Empat metrik QWK/AC2/RMSE/Pearson; CV dan test terpisah',
                 test_evaluations_per_scenario=report['test_evaluations_per_scenario'])


def export_final_features(bundle, bge, cfg, output, journal):
    output = Path(output)
    report = json.loads((output/'final_report.json').read_text())
    for scenario in ('S1','S2','S3'):
        checkpoint = output/scenario/'final.pt'
        if digest(checkpoint) != report['checkpoints'][scenario]:
            raise ValueError('Checkpoint fitur berubah.')
        model, payload = load_model(checkpoint, cfg)
        del payload
        model.to(device_name(cfg)).eval()
        engine = Engine(model, bundle.texts, bge, cfg, device_name(cfg))
        hooks, intermediates = [], {}
        def save_hidden(name):
            def callback(module, args, value):
                intermediates[name] = value.detach().float().cpu().numpy()
            return callback
        # Tangkap keluaran head/proyeksi aktual tanpa mengganti arsitektur.
        if scenario == 'S2':
            for name, layer in [('MLP_linear256',model.head[0]),('MLP_GELU256',model.head[1]),('MLP_dropout_eval',model.head[2])]:
                hooks.append(layer.register_forward_hook(save_hidden(name)))
        if scenario == 'S3':
            hooks.append(model.projection[0].register_forward_hook(save_hidden('proyeksi_linear256')))
        if scenario != 'S2':
            hooks.append(model.dropout.register_forward_hook(save_hidden('dropout_eval')))
        values, window_values, window_ids = {}, [], []
        try:
            with torch.no_grad():
                for index, row in enumerate(bundle.rows):
                    prediction, features = engine.forward([index], return_features=True)
                    for name, value in features.items():
                        array = value.float().cpu().numpy()
                        if name == 'roberta_window_cls':
                            window_values.append(array)
                            window_ids.extend(f'{row["row_id"]}_window_{j}' for j in range(len(array)))
                        else:
                            values.setdefault(name, []).append(array)
                    for name, array in intermediates.items():
                        values.setdefault(name, []).append(array)
                    values.setdefault('output_normalized', []).append(prediction.float().cpu().numpy())
        finally:
            for handle in hooks:
                handle.remove()
        ids = [r['row_id'] for r in bundle.rows]
        sheets = {name: vector_frame(np.concatenate(parts), ids) for name, parts in values.items()}
        if window_values:
            sheets['CLS_tiap_window'] = vector_frame(np.concatenate(window_values), window_ids)
        raw = np.concatenate(values['output_normalized']).reshape(-1)*(cfg['score_max']-cfg['score_min'])+cfg['score_min']
        sheets['Prediksi_semua_jawaban'] = pd.DataFrame({'row_id':ids, 'prediksi_mentah':raw})
        journal.save(f'14_{scenario}_fitur_setelah_training', sheets,
                     'Fitur aktual checkpoint final; diekstrak SESUDAH final test, mode eval',
                     source={'scenario':scenario,'checkpoint_sha256':digest(checkpoint)},
                     note='Seluruh jawaban untuk inspeksi fitur; evaluasi penelitian hanya pada TEST_final, bukan tabel semua jawaban.')
        del engine, model
        release()
