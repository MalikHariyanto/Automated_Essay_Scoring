"""Pengaturan kandidat A dan penghentian latihan; fixture teknis, bukan hasil siswa."""
import numpy as np
import pandas as pd
import pytest
import torch

from aes.teacher_score import data, experiment, lesson
from test_teacher_score import dataset, encoders


def test_candidate_a_defaults_and_separate_result_folder(dataset, tmp_path):
    cfg, bundle = dataset
    defaults = data.config({})
    assert defaults['epochs'] == 25 and defaults['early_stopping_patience'] == 5
    assert defaults['encoder_lr'] == 1e-5
    assert defaults['projection_lr'] == defaults['head_lr'] == 1e-4
    # Perubahan pengaturan harus membuat folder baru, bukan mengambil hasil lama.
    base = {**cfg, 'output': str(tmp_path/'runs')}
    old = data.run_folder({**base, 'epochs': 5, 'encoder_lr': 2e-5, 'head_lr': 1e-3}, bundle)
    marker = old/'old_result.txt'; marker.write_text('PRIVATE TECHNICAL OLD RESULT')
    revised = data.run_folder({**base, 'epochs': 25, 'encoder_lr': 1e-5, 'head_lr': 1e-4}, bundle)
    assert old != revised and marker.read_text() == 'PRIVATE TECHNICAL OLD RESULT'


@pytest.mark.parametrize('settings', [
    {'early_stopping_patience': 0}, {'early_stopping_patience': 1.5},
    {'early_stopping_patience': True}, {'projection_lr': 0}, {'encoder_lr': float('nan')},
])
def test_invalid_training_settings_rejected(settings):
    with pytest.raises(ValueError):
        data.config(settings)


@pytest.mark.parametrize('scenario', ['S1', 'S2', 'S3'])
def test_optimizer_covers_each_trainable_parameter_once(dataset, encoders, scenario):
    cfg, _ = dataset
    model = experiment.make_model(scenario, cfg)
    groups = experiment.optimizer_groups(model, cfg)
    actual = {g['name']: g for g in groups}
    expected_names = {'head'} | ({'roberta'} if scenario != 'S2' else set()) | ({'projection'} if scenario == 'S3' else set())
    assert set(actual) == expected_names
    ids = [id(p) for g in groups for p in g['params']]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in model.parameters() if p.requires_grad}
    if scenario != 'S2':
        assert actual['roberta']['lr'] == 1e-5
        assert {id(p) for p in actual['roberta']['params']} == {id(p) for p in model.roberta.parameters()}
    assert actual['head']['lr'] == 1e-4
    if scenario == 'S3':
        assert actual['projection']['lr'] == 1e-4
        assert {id(p) for p in model.roberta_norm.parameters()} <= {id(p) for p in actual['head']['params']}


def test_layernorm_fusion_uses_normalized_branch_and_receives_gradients(dataset, encoders):
    cfg, _ = dataset
    model = experiment.make_model('S3', cfg)
    class FixedEncoder(torch.nn.Module):
        def forward(self, **kwargs):
            from types import SimpleNamespace
            raw = torch.arange(768, dtype=torch.float32).reshape(1, 1, -1).repeat(2, 1, 1)
            return SimpleNamespace(last_hidden_state=raw)
    model.roberta = FixedEncoder()
    model.eval()
    bge = torch.randn(2, 1024)
    prediction, features = model(windows={}, mapping=torch.tensor([0, 1]), n_essays=2, bge=bge, return_features=True)
    raw = features['roberta']
    normalized = features['roberta_layernorm']
    torch.testing.assert_close(normalized, model.roberta_norm(raw))
    torch.testing.assert_close(features['concat'][:, :768], normalized)
    torch.testing.assert_close(features['concat'][:, 768:], features['projection'])
    assert raw.shape == normalized.shape == (2, 768)
    assert not torch.allclose(raw, normalized)
    prediction.square().mean().backward()
    assert model.roberta_norm.weight.grad.abs().sum() > 0
    assert model.projection[0].weight.grad.abs().sum() > 0
    # S1 tidak mendapat LayerNorm keluaran tambahan; perubahan hanya untuk fusion.
    assert not hasattr(experiment.make_model('S1', cfg), 'roberta_norm')


def test_early_stopping_counts_consecutive_epochs_and_restores_best_weights(dataset, encoders, monkeypatch):
    cfg, bundle = dataset
    cfg = {**cfg, 'epochs': 25, 'early_stopping_patience': 5}
    bge = torch.nn.functional.normalize(torch.randn(24, 1024), dim=1)
    make_model = experiment.make_model
    active = {}
    def tracked_model(*args):
        model = make_model(*args); active['model'] = model
        return model
    monkeypatch.setattr(experiment, 'make_model', tracked_model)
    # Perbaikan pada epoch 5 harus mereset penghitung; nilai sama tidak dianggap membaik.
    errors = iter([10., 9., 9., 10., 8., 8., 9., 9., 9., 9.])
    states = []
    def controlled_validation(y, p, settings):
        states.append({k: v.detach().clone() for k, v in active['model'].state_dict().items()})
        return {'RMSE': next(errors)}
    monkeypatch.setattr(experiment, 'metrics', controlled_validation)
    model, _, info = experiment.fit('S2', list(range(12)), list(range(12, 16)), bundle, bge, cfg, 25, 42)
    assert info['epochs_completed'] == 10 and info['best_epoch'] == 5
    assert info['early_stopped'] and info['best_validation_rmse'] == 8.
    assert [r['epochs_without_improvement'] for r in info['history']] == [0, 0, 1, 2, 0, 1, 2, 3, 4, 5]
    assert any(not torch.equal(states[4][k], states[-1][k]) for k in states[4])
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, states[4][key], rtol=0, atol=0)


def test_refit_does_not_early_stop_or_read_validation(dataset, encoders, monkeypatch):
    cfg, bundle = dataset
    bge = torch.nn.functional.normalize(torch.randn(24, 1024), dim=1)
    def forbidden(*args):
        raise AssertionError('Refit tidak boleh menghitung validasi atau mengintip data uji.')
    monkeypatch.setattr(experiment, 'metrics', forbidden)
    _, _, info = experiment.fit('S2', list(range(12)), None, bundle, bge,
                                {**cfg, 'early_stopping_patience': 1}, 4, 42)
    assert info['epochs_completed'] == info['best_epoch'] == 4
    assert info['early_stopping_patience'] is None and not info['early_stopped']
    assert info['best_validation_rmse'] is None


def test_cv_still_stops_at_maximum_when_validation_keeps_improving(dataset, encoders, monkeypatch):
    cfg, bundle = dataset
    bge = torch.nn.functional.normalize(torch.randn(24, 1024), dim=1)
    errors = iter([4., 3., 2., 1.])
    monkeypatch.setattr(experiment, 'metrics', lambda *args: {'RMSE': next(errors)})
    _, _, info = experiment.fit('S2', list(range(12)), list(range(12, 16)), bundle, bge, cfg, 4, 42)
    assert info['epochs_completed'] == info['best_epoch'] == 4
    assert not info['early_stopped'] and info['best_validation_rmse'] == 1.


def test_comparison_aligns_teacher_raw_and_rounded_scores_without_inference(dataset, tmp_path):
    cfg, bundle = dataset
    ids = [row['row_id'] for row in bundle.rows[:3]]
    teachers = bundle.y[:3]
    report = {'prediction_files': {}}
    raw = [-5., 80.5, 100.7]
    for scenario in ('S1', 'S2', 'S3'):
        path = tmp_path/scenario/'test_predictions.csv'; path.parent.mkdir()
        frame = pd.DataFrame({'row_id': ids, 'skor_guru': teachers, 'prediksi_mentah': raw})
        if scenario == 'S2':
            frame = frame.iloc[::-1]
        frame.to_csv(path, index=False)
        report['prediction_files'][scenario] = data.digest(path)
    comparison = lesson.test_score_comparison(report, bundle, cfg, tmp_path)
    assert comparison.row_id.tolist() == ids
    np.testing.assert_array_equal(comparison.skor_guru, teachers)
    for scenario in ('S1', 'S2', 'S3'):
        assert comparison[f'Prediksi {scenario}'].tolist() == raw
        assert comparison[f'Pembulatan {scenario}'].tolist() == [0, 80, 100]
    # Bahkan jika hash diperbarui, nilai guru palsu tetap harus ditolak.
    path = tmp_path/'S3/test_predictions.csv'
    frame = pd.read_csv(path); frame.loc[0, 'skor_guru'] += 1; frame.to_csv(path, index=False)
    report['prediction_files']['S3'] = data.digest(path)
    with pytest.raises(ValueError, match='Nilai guru'):
        lesson.test_score_comparison(report, bundle, cfg, tmp_path)
