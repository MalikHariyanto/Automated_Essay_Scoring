"""Fixture teknis sintetis; BUKAN skor siswa atau akurasi checkpoint pretrained."""
import ast
import hashlib
import json
from pathlib import Path
import re

import nbformat
import numpy as np
from openpyxl import Workbook
import pytest
import torch
from transformers import RobertaConfig, RobertaModel, XLMRobertaConfig, XLMRobertaModel
from transformers import XLMRobertaTokenizerFast

from aes.teacher_score import data, experiment, models, vectors


class TechnicalTokenizer:
    """Tokenisasi kata deterministik untuk menguji jalur offset/window; bukan HF."""
    def encode(self, text, add_special_tokens=True):
        ids = [3+int(hashlib.sha256(m.group().encode()).hexdigest()[:5], 16)%57
               for m in re.finditer(r'\w+|[^\w\s]', text)]
        return self.build_inputs_with_special_tokens(ids) if add_special_tokens else ids

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def build_inputs_with_special_tokens(self, ids):
        return [0]+list(ids)+[2]

    def get_special_tokens_mask(self, ids, already_has_special_tokens=False):
        return [1]+[0]*len(ids)+[1]

    def prepare_for_model(self, ids, **kwargs):
        values = self.build_inputs_with_special_tokens(ids)
        return {'input_ids': values, 'attention_mask': [1]*len(values)}

    def pad(self, rows, **kwargs):
        maximum = max(len(row['input_ids']) for row in rows)
        return {key: torch.tensor([r[key]+[1 if key == 'input_ids' else 0]*(maximum-len(r[key])) for r in rows])
                for key in ('input_ids', 'attention_mask')}

    def convert_ids_to_tokens(self, ids):
        return [f'technical_{i}' for i in ids]

    def __call__(self, texts, add_special_tokens=True, return_offsets_mapping=False, **kwargs):
        if isinstance(texts, str):
            result = {'input_ids': self.encode(texts, add_special_tokens)}
            if return_offsets_mapping:
                result['offset_mapping'] = [m.span() for m in re.finditer(r'\w+|[^\w\s]', texts)]
            return result
        rows = []
        for text in texts:
            ids = self.encode(text)[:kwargs.get('max_length', 10000)]
            rows.append({'input_ids': ids, 'attention_mask': [1]*len(ids)})
        return self.pad(rows)


def make_book(path, rows=24, missing=False, duplicate=False):
    book = Workbook()
    book.active.title = 'Sheet1'
    book.active.append(data.REQUIRED+[None, None])
    for i in range(rows):
        # Dua respons per kelompok; data buatan, tidak menyalin teks unggahan.
        text = f'alpha technical response {i}. alpha second sentence.'
        if duplicate and i == 4:
            text = 'alpha technical response 0. alpha second sentence.'
        book.active.append([f'technical_{i//2}@invalid.example', i%2, text, 'fixture question',
                            'fixture reference', None if missing and i == 0 else 65+i%30])
    book.save(path)


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path/'fixture.xlsx'
    make_book(path)
    cfg = data.config(dict(dataset=str(path), output=str(tmp_path/'private'), expected_rows=24,
                           epochs=1, roberta_max_length=16, roberta_overlap=4,
                           bge_max_length=16, effective_batch=4, micro_batch=1,
                           device='cpu', dropout=0.))
    return cfg, data.read_dataset(cfg)


@pytest.fixture
def encoders(monkeypatch):
    torch.set_num_threads(2)
    def technical_encoder(cfg, role):
        cls, model, dim, heads = (RobertaConfig, RobertaModel, 768, 12) if role == 'roberta' else (XLMRobertaConfig, XLMRobertaModel, 1024, 16)
        definition = cls(vocab_size=64, hidden_size=dim, num_hidden_layers=1,
                         num_attention_heads=heads, intermediate_size=16,
                         max_position_embeddings=64, pad_token_id=1, bos_token_id=0, eos_token_id=2,
                         hidden_dropout_prob=0., attention_probs_dropout_prob=0.)
        result = model(definition, add_pooling_layer=False)
        result.requires_grad_(role == 'roberta')
        return result
    for module in (models, experiment, vectors):
        monkeypatch.setattr(module, 'encoder', technical_encoder)
    for module in (models, vectors):
        monkeypatch.setattr(module, 'tokenizer', lambda *a, **k: TechnicalTokenizer())
    return technical_encoder


def test_audit_single_score_and_source_unchanged(dataset):
    cfg, bundle = dataset
    original = data.digest(cfg['dataset'])
    assert bundle.y.shape == (24,)
    assert bundle.summary['model_outputs'] == 1
    assert bundle.summary['four_trait_labels_available'] is False
    assert bundle.summary['input_features'] == ['response_text']
    assert all('fixture reference' not in t and '@invalid' not in t for t in bundle.texts)
    assert data.digest(cfg['dataset']) == original


def test_missing_grade_rejected_instead_of_imputed(tmp_path):
    path = tmp_path/'missing.xlsx'
    make_book(path, missing=True)
    with pytest.raises(ValueError, match='kosong'):
        data.read_dataset(data.config(dict(dataset=str(path), expected_rows=24)))


def test_groups_duplicates_and_five_fold_isolation(tmp_path):
    path = tmp_path/'duplicates.xlsx'
    make_book(path, duplicate=True)
    cfg = data.config(dict(dataset=str(path), expected_rows=24))
    bundle = data.read_dataset(cfg)
    assert bundle.groups[0] == bundle.groups[1] == bundle.groups[4] == bundle.groups[5]
    manifest = data.make_split(bundle, cfg)
    assert len(manifest['folds']) == 5
    assert not set(bundle.groups[manifest['development']]) & set(bundle.groups[manifest['test']])
    validation = []
    for fold in manifest['folds']:
        assert set(fold['train']) | set(fold['validation']) == set(manifest['development'])
        assert not set(bundle.groups[fold['train']]) & set(bundle.groups[fold['validation']])
        validation += fold['validation']
    assert sorted(validation) == sorted(manifest['development'])
    assert data.make_split(bundle, cfg) == manifest


def test_metrics_raw_units_and_full_category_reference():
    cfg = data.config({})
    y, p = np.array([0, 40, 80, 100]), np.array([-20, 40, 80, 130])
    result = models.metrics(y, p, cfg)
    assert result['RMSE'] == pytest.approx(np.sqrt((20**2+30**2)/4))
    assert result['QWK'] == pytest.approx(1)
    assert result['AC2'] == pytest.approx(1)
    assert result['out_of_range_fraction'] == .5
    assert np.array_equal(p, [-20, 40, 80, 130])
    # Rumus pembanding independen: seluruh101 kategori, pooled p, quadratic agreement.
    y, p = np.array([65, 78, 90, 99]), np.array([66, 76, 91, 97])
    categories = np.arange(101)
    weights = 1-((categories[:, None]-categories[None, :])/100)**2
    pooled = np.bincount(np.r_[y, p], minlength=101)/8
    pa = np.mean([weights[a,b] for a,b in zip(y,p)])
    pe = weights.sum()*(pooled*(1-pooled)).sum()/(101*100)
    assert models.metrics(y,p,cfg)['AC2'] == pytest.approx((pa-pe)/(1-pe))
    result = models.metrics([80,80], [80,80], cfg)
    assert result['Pearson'] is None and result['QWK'] is None
    assert 'Pearson' in result['undefined_reasons']


@pytest.mark.parametrize('scenario', ['S1','S2','S3'])
def test_forward_backward_freeze_and_reload(dataset, encoders, scenario, tmp_path):
    cfg, bundle = dataset
    bge_model = encoders(cfg, 'bge')
    before = {k:v.clone() for k,v in bge_model.state_dict().items()}
    bge_model.eval()
    tokens = TechnicalTokenizer()(bundle.texts, max_length=16)
    with torch.no_grad():
        bge = torch.nn.functional.normalize(bge_model(**tokens).last_hidden_state[:,0], dim=1).detach()
    model, engine, info = experiment.fit(scenario, list(range(12)), list(range(12,16)), bundle, bge, cfg, 1, 42)
    assert engine.forward([0,1]).shape == (2,1)
    assert info['active_lr_steps'] > 0 and info['BGE_encoder_trained'] is False
    assert any(info['parameter_samples_changed'].values())
    assert all(torch.equal(before[k], v) for k,v in bge_model.state_dict().items())
    assert not any(p.requires_grad for p in bge_model.parameters())
    if scenario != 'S2':
        assert any(p.grad is not None for p in model.roberta.parameters())
    if scenario == 'S3':
        assert model.projection[0].weight.grad.abs().sum() > 0
        _, features = engine.forward([0,1], return_features=True)
        assert features['concat'].shape == (2,1024)
    expected = engine.predict([16,17])
    path = tmp_path/f'{scenario}.pt'
    experiment.save_model(path, model, cfg, bundle, data.make_split(bundle,cfg), info)
    restored, _ = experiment.load_model(path, cfg)
    actual = models.Engine(restored,bundle.texts,bge,cfg,'cpu').predict([16,17])
    np.testing.assert_allclose(expected,actual,atol=1e-6)


def test_frozen_bge_cache_integrity_and_truncation(dataset, encoders, tmp_path):
    cfg,bundle = dataset
    cfg['bge_max_length'] = 8
    result = models.bge_features(bundle,cfg,tmp_path)
    assert result.shape == (24,1024) and not result.requires_grad
    metadata = json.loads((tmp_path/'bge_encoding.json').read_text())
    assert metadata['truncated_answers'] == 24
    np.testing.assert_array_equal(result,models.bge_features(bundle,cfg,tmp_path))
    cache = next((Path(cfg['output'])/'cache').glob('*.npz'))
    cache.write_bytes(b'tamper')
    with pytest.raises(ValueError, match='Cache BGE berubah'):
        models.bge_features(bundle,cfg,tmp_path)


def test_accumulation_matches_full_batch_including_short_last_batch(dataset, encoders):
    cfg,bundle = dataset
    bge = torch.nn.functional.normalize(torch.randn(24,1024),dim=1)
    cfg = {**cfg, 'warmup_ratio':0., 'micro_batch':1, 'effective_batch':4}
    first,_,one = experiment.fit('S2',list(range(13)),None,bundle,bge,cfg,1,42)
    second,_,two = experiment.fit('S2',list(range(13)),None,bundle,bge,{**cfg,'micro_batch':4},1,42)
    assert one['optimizer_steps'] == two['optimizer_steps'] == 4
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value,second.state_dict()[name],atol=1e-6,rtol=1e-5)


def test_vectors_same_lookup_context_differs_and_full_shapes(dataset, encoders, tmp_path):
    cfg,_ = dataset
    cfg['roberta_max_length'],cfg['roberta_overlap'],cfg['bge_max_length'] = 8,2,8
    text = 'alpha first context alpha. Different sentence has alpha and many other words to test windows.'
    result = vectors.vector_lab(text,cfg,tmp_path)
    assert result['roberta']['uncovered_tokens'] == 0
    assert len(result['roberta']['window_starts']) > 1
    assert result['bge']['uncovered_tokens'] > 0
    for role,dim in [('roberta',768),('bge',1024)]:
        item = result[role]
        pairs = item['repeated_words']
        assert pairs and pairs[0]['same_token_ids']
        assert pairs[0]['initial_max_abs_difference'] == 0
        assert pairs[0]['contextual_max_abs_difference'] > 1e-5
        assert item['sentence_shape'] == [2,dim]
        with np.load(item['npz'], allow_pickle=False) as full:
            assert full['word_contextual'].shape[1] == dim
            assert full['token_contextual'].shape == full['token_initial'].shape
            if role == 'bge':
                assert np.isnan(full['word_contextual']).any()
    assert not vectors.compare_repetitions([], np.empty((0,768)), np.empty((0,768)))


def test_vector_offset_mismatch_stops(dataset, encoders):
    cfg,_ = dataset
    class Wrong(TechnicalTokenizer):
        def __call__(self,*args,**kwargs):
            result = super().__call__(*args,**kwargs)
            result['input_ids'][0] += 1
            return result
    with pytest.raises(ValueError, match='ID fast/slow berbeda'):
        vectors.token_vectors('alpha beta',encoders(cfg,'roberta'),TechnicalTokenizer(),Wrong(),'roberta',cfg,'cpu')


def test_vector_real_fast_xlm_tokenizer_api(dataset, encoders):
    # API fast asli, vocabulary fixture lokal; tidak memerlukan checkpoint/network.
    from tokenizers import Tokenizer, models as backend_models, pre_tokenizers, processors
    backend = Tokenizer(backend_models.WordLevel({'<s>':0, '<pad>':1, '</s>':2, '<unk>':3,
                                                   'alpha':4, 'beta':5, '.':6, '<mask>':7}, unk_token='<unk>'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(single='<s> $A </s>', special_tokens=[('<s>',0),('</s>',2)])
    tok = XLMRobertaTokenizerFast(tokenizer_object=backend, bos_token='<s>',eos_token='</s>',
                                   unk_token='<unk>',pad_token='<pad>',mask_token='<mask>')
    cfg,_ = dataset
    result = vectors.token_vectors('alpha beta alpha.',encoders(cfg,'bge'),tok,tok,'bge',cfg,'cpu')
    assert result['ids'] == [4,5,4,6]
    assert result['contextual'].shape == (4,1024)
    assert np.isfinite(result['contextual']).all()


def test_complete_cv_refit_test_once_and_resume(dataset, encoders, tmp_path, monkeypatch):
    cfg,bundle = dataset
    manifest = data.make_split(bundle,cfg)
    bge = torch.nn.functional.normalize(torch.randn(24,1024),dim=1)
    original_predict = models.Engine.predict
    test_calls = []
    def guarded_predict(self, indices):
        if list(indices) == manifest['test']:
            assert (tmp_path/'selection_locked.json').exists()
            assert (tmp_path/'refit_locked.json').exists()
            assert all((tmp_path/s/'final.pt').exists() for s in ('S1','S2','S3'))
            test_calls.append(self.model.scenario)
        return original_predict(self,indices)
    monkeypatch.setattr(models.Engine,'predict',guarded_predict)
    report = experiment.train_all(bundle,manifest,bge,cfg,tmp_path)
    assert len(report['cv']) == 15 and len(report['rows']) == 3
    assert test_calls == ['S1','S2','S3']
    assert report['test_evaluations_per_scenario'] == 1
    assert (tmp_path/'selection_locked.json').exists() and (tmp_path/'refit_locked.json').exists()
    predicted = experiment.score_answer('alpha technical unseen response.',cfg,tmp_path)
    assert [r['Skenario'] for r in predicted] == ['S1','S2','S3']
    assert all(isinstance(r['prediksi_mentah'],float) and 0 <= r['kategori_salinan'] <= 100 for r in predicted)
    def forbidden(*args,**kwargs):
        raise AssertionError('Resume tidak boleh fit/predict test ulang')
    monkeypatch.setattr(experiment,'fit',forbidden)
    monkeypatch.setattr(models.Engine,'predict',forbidden)
    assert experiment.train_all(bundle,manifest,bge,cfg,tmp_path)['reused_completed_run']
    # Gangguan setelah test sudah tersimpan tetapi sebelum reportfinal: pakai cachetest.
    (tmp_path/'final_report.json').unlink()
    assert len(experiment.train_all(bundle,manifest,bge,cfg,tmp_path)['rows']) == 3
    (tmp_path/'S1/test_predictions.csv').write_text('changed')
    with pytest.raises(ValueError,match='Prediksi test tersimpan berubah'):
        experiment.train_all(bundle,manifest,bge,cfg,tmp_path)


def test_public_notebook_syntax_mirror_and_empty_outputs():
    root = Path(__file__).resolve().parents[1]
    first = root/'notebooks/03_AES_dataset_guru_300.ipynb'
    assert first.read_bytes() == (root/'aes-learning/notebooks'/first.name).read_bytes()
    notebook = nbformat.read(first,as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            ast.parse(cell.source)
            assert cell.outputs == [] and cell.execution_count is None
