"""Uji ablation E27/E28; data teknis dan Transformer acak, bukan hasil penelitian."""
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests'))
import test_context_pair_notebook as context
import test_e21_e22_notebooks as base
from test_e6_three_input_notebook import ThreeTechnicalTokenizer
from test_teacher_score import encoders, make_book

NAMES = {
    'E27': 'AES_300_E27_E23_BiGRU_CoAttention_GPU.ipynb',
    'E28': 'AES_300_E28_E23_BiGRU_MLPLR5e6_GPU.ipynb',
}
BASE = ROOT/'notebooks/experiments/AES_300_E23_Tiga_Input_BiGRU_GPU.ipynb'


def load_runtime(monkeypatch, experiment):
    monkeypatch.setattr(context, 'NOTEBOOK', ROOT/'notebooks/experiments'/NAMES[experiment])
    return context.runtime()


@pytest.fixture(params=NAMES)
def runtime(request, monkeypatch, tmp_path):
    ns = load_runtime(monkeypatch, request.param)
    path = tmp_path/'TECHNICAL_FIXTURE_ONLY.xlsx'
    make_book(path)
    cfg = ns['config']({'dataset': str(path), 'output': str(tmp_path/'results'), 'expected_rows': 24,
        'device': 'cpu', 'epochs': 1, 'micro_batch': 1, 'effective_batch': 4, 'dropout': 0.,
        'roberta_max_length': 32, 'roberta_overlap': 4, 'bge_max_length': 32, 'bge_overlap': 4,
        'context_max_tokens': 12, 'question_max_tokens': 4})
    ns['tokenizer'] = lambda *a, **k: ThreeTechnicalTokenizer()
    return request.param, ns, cfg, ns['read_dataset'](cfg)


def test_pinned_standalone_sources_gpu_guard_mirrors_and_builder():
    baseline = BASE.read_bytes()
    paths = [ROOT/'notebooks/experiments'/name for name in NAMES.values()]
    old = [p.read_bytes() for p in paths]
    subprocess.run([sys.executable, 'scripts/build_e27_e28_notebooks.py'], cwd=ROOT, check=True, capture_output=True)
    assert BASE.read_bytes() == baseline
    for path, raw in zip(paths, old):
        assert path.read_bytes() == raw == (ROOT/'aes-learning/notebooks/experiments'/path.name).read_bytes()
        n = json.loads(raw)
        meta = n['metadata']['aes_experiment']
        assert meta['target'] == 'rater' and meta['input_columns'] == ['soal', 'response_text', 'jawaban_referensi']
        assert meta['input_format'] == 'single_sequence' and not meta['reused_controls']
        assert not meta['drive_mount'] and not meta['drive_autosave']
        assert meta['base_sha256'] == hashlib.sha256(baseline).hexdigest()
        assert n['metadata']['accelerator'] == 'GPU'
        assert "device='cuda'" in n['cells'][12]['source'] and 'check_gpu(require_gpu=True)' in n['cells'][12]['source']
        fingerprint = n['metadata']['aes_standalone']['code_sha256']
        codes = []
        for cell in n['cells']:
            if cell['cell_type'] != 'code':continue
            assert cell['outputs'] == [] and cell['execution_count'] is None
            code = cell['source'].removeprefix('%%aes\n')
            ast.parse(code)
            assert not any(x in code for x in ('drive.mount', 'googleapiclient', 'from aes.'))
            codes.append(code.replace(fingerprint, '__NOTEBOOK_CODE_SHA256__'))
        assert len(codes) == 30
        assert hashlib.sha256('\n'.join(codes).encode()).hexdigest() == fingerprint
    e28 = json.loads(paths[1].read_text())
    baseline = json.loads(baseline)
    # E28 tidak mengubah fitur, window, arsitektur, atau evaluasi.
    for i in (14, 18, 20, 21, 26, 28, 30, 31, 34, 35, 38, 39, 41, 43):
        assert e28['cells'][i]['source'] == baseline['cells'][i]['source']


@pytest.mark.parametrize('scenario', ['S1', 'S2', 'S3'])
def test_optimizer_coverage_lr_initialization_controls_gradient_and_reload(runtime, encoders, scenario, tmp_path, monkeypatch):
    experiment, ns, cfg, bundle = runtime
    ns['encoder'] = encoders
    bge = ns['bge_features'](bundle, cfg, Path(cfg['output'])/'run')
    assert not bge.requires_grad
    torch.manual_seed(44)
    model = ns['make_model'](scenario, cfg).eval()
    engine = ns['Engine'](model, bundle.texts, bge, cfg, 'cpu', contexts=ns['contexts_from_bundle'](bundle))
    pred, features = engine.forward([0, 1], True)
    assert pred.shape == (2, 1) and torch.isfinite(pred).all()
    pred.square().mean().backward()
    groups = ns['optimizer_groups'](model, cfg)
    by_id = {id(p): g for g in groups for p in g['params']}
    assert len(by_id) == sum(len(g['params']) for g in groups)
    assert set(by_id) == {id(p) for p in model.parameters() if p.requires_grad}
    for name, parameter in model.named_parameters():
        expected = 5e-6 if name.startswith('roberta.') or (experiment == 'E28' and scenario == 'S3' and name.startswith('head.')) else 1e-4
        assert by_id[id(parameter)]['lr'] == expected, name
    if scenario == 'S3':
        assert features['concat'].shape == (2, 1024) and features['bigru_hidden128'].shape == (2, 128)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.roberta.parameters())
        assert model.sequence_pool.gru.weight_ih_l0.grad.abs().sum() > 0
        assert not hasattr(model.sequence_pool, 'attention_score')
        if experiment == 'E27':
            assert model.sequence_pool.reference_projection.weight.grad.abs().sum() > 0
            assert model.sequence_pool.matching_projection.weight.grad.abs().sum() > 0
    # Inisialisasi parameter E23 dipertahankan, termasuk saat menambah E27.
    monkeypatch.setattr(context, 'NOTEBOOK', BASE)
    old = context.runtime()
    torch.manual_seed(44)
    control = old['SingleScoreModel'](scenario, encoders(cfg, 'roberta') if scenario != 'S2' else None, dropout=0.).eval()
    assert all(torch.equal(model.state_dict()[k], v) for k, v in control.state_dict().items())
    if scenario != 'S3' or experiment == 'E28':
        previous = old['Engine'](control, bundle.texts, bge, cfg, 'cpu', roberta_tokenizer=ThreeTechnicalTokenizer(), contexts=ns['contexts_from_bundle'](bundle))
        torch.testing.assert_close(pred, previous.forward([0, 1]), rtol=0, atol=0)
    path = tmp_path/'technical_state.pt'
    torch.save(model.state_dict(), path)
    restored = ns['make_model'](scenario, cfg).eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    engine2 = ns['Engine'](restored, bundle.texts, bge, cfg, 'cpu', contexts=ns['contexts_from_bundle'](bundle))
    torch.testing.assert_close(pred, engine2.forward([0, 1]), rtol=0, atol=0)


def test_reference_offsets_layout_padding_windows_and_no_prefix_or_id_guess(monkeypatch):
    ns = load_runtime(monkeypatch, 'E27')
    tok = ThreeTechnicalTokenizer()
    cfg = ns['config']({'roberta_max_length': 32, 'roberta_overlap': 4, 'bge_max_length': 32, 'bge_overlap': 4,
                        'context_max_tokens': 12, 'question_max_tokens': 4})
    # Kata sama pada prefix/soal/respons tidak boleh mengacaukan mask referensi.
    text = 'Jawaban referensi ' * 24
    ctx = ('Jawaban referensi', 'Jawaban referensi isi khusus')
    monkeypatch.setattr(context, 'NOTEBOOK', BASE)
    old = context.runtime()
    values, stats = ns['pair_content_windows'](text, ctx, tok, 32, 4, cfg)
    before, old_stats = old['pair_content_windows'](text, ctx, tok, 32, 4, cfg)
    assert values == before and stats['starts'] == old_stats['starts'] and stats['answer_positions'] == old_stats['answer_positions']
    assert stats['reference_content_kept'] == 4 and stats['reference_truncated'] is False
    for side in ('left', 'right'):
        class PadTokenizer(ThreeTechnicalTokenizer):
            padding_side = side
            def pad(self, rows, **kwargs):
                maximum = max(len(row['input_ids']) for row in rows)
                return {key: torch.tensor([(r[key] + [1 if key == 'input_ids' else 0]*(maximum-len(r[key]))) if side == 'right'
                                          else ([1 if key == 'input_ids' else 0]*(maximum-len(r[key])) + r[key]) for r in rows])
                        for key in ('input_ids', 'attention_mask')}
        t = PadTokenizer()
        batch, mapping, details = ns['pair_batch_windows']([text, 'short answer'], [ctx, ctx], t, 'cpu', 32, 4, cfg)
        a = ns['answer_token_mask'](batch, mapping, details, side)
        r = ns['reference_token_mask'](batch, mapping, details, side)
        assert not (a & r).any() and not (r & ~batch['attention_mask'].bool()).any()
        assert r.sum(1).tolist() == [4]*len(mapping)
        expect = t.encode(ctx[1], add_special_tokens=False)
        assert all(batch['input_ids'][i, r[i]].tolist() == expect for i in range(len(mapping)))
    with pytest.raises(ValueError, match='truncation'):
        ns['pair_content_windows'](text, ctx, tok, 32, 4, {**cfg, 'context_max_tokens': 5})
    class BadOffsets(ThreeTechnicalTokenizer):
        def __call__(self, *args, **kwargs):
            result = super().__call__(*args, **kwargs)
            result['input_ids'] = [99]*len(result['input_ids'])
            return result
    with pytest.raises(ValueError, match='ID/offset'):
        ns['pair_content_windows'](text, ctx, BadOffsets(), 32, 4, cfg)


def test_coattention_masks_two_directions_batch_order_reference_effect_and_gradients(monkeypatch):
    ns = load_runtime(monkeypatch, 'E27')
    torch.manual_seed(7)
    pool = ns['AnswerReferenceCoAttention'](64).eval()
    pool.capture_alignment = True
    h = torch.randn(2, 12, 768, requires_grad=True)
    a = torch.tensor([[0,0,0,0,0,1,1,1,0,0,0,0], [0,0,0,0,1,1,1,1,1,1,0,0]], dtype=torch.bool)
    r = torch.tensor([[0,1,1,1,0,0,0,0,0,0,0,0], [0,1,1,0,0,0,0,0,0,0,0,0]], dtype=torch.bool)
    out = pool(h, a, r)
    cross = pool.last_cross_attention.clone()
    weights = pool.last_weights.clone()
    reference_weights = pool.last_reference_weights.clone()
    torch.testing.assert_close(weights.sum(1), torch.ones(2))
    torch.testing.assert_close(reference_weights.sum(1), torch.ones(2))
    assert not weights[~a].any() and not reference_weights[~r].any()
    assert (cross[0, :3, :3].sum(-1)-1).abs().max() < 1e-6
    assert cross[0, 3:].abs().sum() == 0 and cross[1, :, 2:].abs().sum() == 0
    changed = h.detach().clone();changed[~(a | r)] = 1e6
    torch.testing.assert_close(out, pool(changed, a, r), rtol=0, atol=0)
    changed = h.detach().clone();changed[r] += torch.randn_like(changed[r])
    assert not torch.allclose(out, pool(changed, a, r))
    single = torch.cat([pool(row[None], am[None], rm[None]) for row, am, rm in zip(h, a, r)])
    torch.testing.assert_close(out, single, rtol=1e-5, atol=1e-6)
    out.square().mean().backward()
    assert h.grad[a].abs().sum() > 0 and h.grad[r].abs().sum() > 0
    assert h.grad[~(a | r)].abs().sum() == 0
    assert pool.reference_projection.weight.grad.abs().sum() > 0
    assert pool.matching_projection.weight.grad.abs().sum() > 0
    pool.capture_alignment = False
    pool(h.detach(), a, r)
    assert pool.last_cross_attention is None
    with pytest.raises(ValueError, match='minimal satu'):pool(h, a, torch.zeros_like(r))
    with pytest.raises(ValueError, match='terpisah'):pool(h, a, a)
    # AMP CPU menguji cast/masked-softmax; bukan bukti eksekusi CUDA.
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        amp = pool(h.detach(), a, r)
    assert torch.isfinite(amp).all()


@pytest.mark.parametrize('experiment', NAMES)
def test_full_notebook_cv_refit_five_excels_export_actual_lr_and_resume(experiment, tmp_path, monkeypatch):
    monkeypatch.setattr(base, 'NAMES', {'single_sequence': NAMES[experiment]})
    monkeypatch.setattr(base, 'GPU', True)
    base.test_full_notebook_thirty_cells_and_resume_same_local_folder('single_sequence', tmp_path)
    import pandas as pd
    pointer = json.loads((tmp_path/'standalone/workspace/outputs/dataset300_download.json').read_text())
    folder = Path(pointer['run_folder'])
    config = json.loads((folder/'effective_config.json').read_text())
    assert config['s3_mlp_lr'] == (1e-4 if experiment == 'E27' else 5e-6)
    setting = pd.read_excel(folder/'hasil_ringkas/05_Riwayat_Pelatihan.xlsx', sheet_name='Pengaturan Eksperimen')
    settings = dict(zip(setting['Pengaturan'], setting['Nilai']))
    assert float(settings['LR MLP S3 saja']) == config['s3_mlp_lr']
    assert float(settings['LR pooling dan norm S3']) == 1e-4
    checkpoint = torch.load(folder/'S3/final.pt', map_location='cpu', weights_only=True)
    assert checkpoint['training']['optimizer_learning_rates']['head'] == config['s3_mlp_lr']
    assert checkpoint['training']['optimizer_learning_rates']['pooling_norm'] == 1e-4
    if experiment == 'E27':
        book = folder/'hasil_ringkas/02_Vektor_Jawaban.xlsx'
        weights = pd.read_excel(book, sheet_name='Bobot Referensi S3')
        np.testing.assert_allclose(weights.groupby(['ID Jawaban', 'Bagian Teks'])['Bobot Attention'].sum(), 1., atol=1e-6)
        top1 = pd.read_excel(book, sheet_name='Pasangan Token Top1 S3')
        assert len(top1) > 24 and top1['Bobot Kecocokan'].between(0, 1).all()
        assert top1['Posisi Token Jawaban'].ge(1).all() and top1['Posisi Token Referensi'].ge(1).all()
        assert not weights['Token'].isna().any()


def test_real_byte_bpe_slow_fast_offsets_unicode_and_cached_fallback(monkeypatch, tmp_path):
    """API tokenizer riil dengan kosakata buatan lokal; bukan tokenizer checkpoint Indonesia."""
    from tokenizers import ByteLevelBPETokenizer
    from transformers import RobertaTokenizer, RobertaTokenizerFast
    trained = ByteLevelBPETokenizer()
    trained.train_from_iterator(['Soal: apa itu metode? Jawaban referensi: penelitian terarah café Indonesia. Jawaban siswa: metode ilmiah.']*3,
                                vocab_size=300, special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])
    trained.save_model(str(tmp_path))
    params = dict(vocab_file=str(tmp_path/'vocab.json'), merges_file=str(tmp_path/'merges.txt'))
    slow = RobertaTokenizer(**params)
    fast = RobertaTokenizerFast(**params)
    ns = load_runtime(monkeypatch, 'E27')
    calls = []
    class Loader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.append((args, kwargs));return fast
    ns['AutoTokenizer'] = Loader
    cfg = ns['config']({'context_max_tokens': 100, 'question_max_tokens': 12})
    ctx = ('apa itu metode?', 'penelitian terarah café Indonesia.')
    for _ in range(2):
        chunks, info = ns['pair_content_windows']('metode ilmiah penelitian', ctx, slow, 512, 128, cfg)
        ids = chunks[0]['input_ids']
        ref = slow.decode([ids[p] for p in info['reference_positions'][0]])
        # BPE dapat punya token spasi berdiri sendiri; mask hanya mengambil token isi.
        assert ''.join(ref.split()) == ''.join(ctx[1].split()) and 'referensi:' not in ref
    assert len(calls) == 1 and calls[0][1]['revision'] == cfg['roberta']['revision']


@pytest.mark.parametrize('runtime', ['E28'], indirect=True)
def test_e28_first_optimizer_step_changes_only_mlp_learning_rate(runtime, encoders, monkeypatch):
    experiment, ns, cfg, bundle = runtime
    assert experiment == 'E28'
    ns['encoder'] = encoders
    bge = torch.randn(len(bundle.rows), 1024)
    torch.manual_seed(24)
    candidate = ns['make_model']('S3', cfg).eval()
    baseline = ns['make_model']('S3', cfg).eval()
    baseline.load_state_dict(candidate.state_dict())
    groups1 = ns['optimizer_groups'](candidate, cfg)
    groups2 = ns['optimizer_groups'](baseline, {**cfg, 's3_mlp_lr': 1e-4})
    before = {k: v.clone() for k, v in candidate.state_dict().items()}
    for model, groups in [(candidate, groups1), (baseline, groups2)]:
        engine = ns['Engine'](model, bundle.texts, bge, cfg, 'cpu', contexts=ns['contexts_from_bundle'](bundle))
        loss = (engine.forward([0, 1]) - .8).square().mean()
        loss.backward()
        torch.optim.AdamW(groups, weight_decay=.01).step()
    for name, value in candidate.state_dict().items():
        if not name.startswith('head.'):
            torch.testing.assert_close(value, baseline.state_dict()[name], rtol=0, atol=0)
    head = 'head.0.weight'
    small = (candidate.state_dict()[head]-before[head]).norm()
    large = (baseline.state_dict()[head]-before[head]).norm()
    assert float(large/small) == pytest.approx(20., rel=.002)
