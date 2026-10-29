"""Identitas metode dipisahkan dari hasil; tidak membuat angka untuk run yang belum masuk."""
PAPERS = {
    'roberta': ('Liu et al. (2019) · RoBERTa', 'https://arxiv.org/abs/1907.11692', 'Backbone RoBERTa; bukan bukti keunggulan eksperimen ini.'),
    'bge': ('Chen et al. (2024) · BGE-M3', 'https://arxiv.org/abs/2402.03216', 'Encoder embedding BGE-M3 yang dibekukan.'),
    'mlp': ('Hendrycks & Gimpel (2016) · GELU', 'https://arxiv.org/abs/1606.08415', 'Aktivasi GELU pada MLP; bukan pengujian konfigurasi AES ini.'),
    'norm': ('Ba et al. (2016) · Layer Normalization', 'https://arxiv.org/abs/1607.06450', 'Normalisasi representasi sebelum penggabungan.'),
    'bigru': ('Alhothali (2026) · PaSTO-GNN', 'https://doi.org/10.3389/frai.2026.1842542', 'Rujukan BiGRU dan attention pada AES. Paper memakai urutan kalimat dan komponen lain; proyek mengadaptasi urutan token, bukan mereplikasi seluruh metode.'),
    'attention': ('Dong, Zhang & Yang (2017) · Attention-based Recurrent CNN untuk AES', 'https://aclanthology.org/K17-1017/', 'Rujukan attention untuk merangkum informasi AES. Arsitektur dan data berbeda; tidak menjamin RMSE proyek turun.'),
    'gate': ('Arevalo et al. (2017) · Gated Multimodal Units', 'https://openreview.net/forum?id=Hy-2G6ile', 'Inspirasi gate yang mempelajari kontribusi representasi. Paper multimodal, bukan eksperimen RoBERTa–BGE AES ini.'),
    'coattention': ('Zhang & Litman (2018) · Co-attention untuk source-dependent essay scoring', 'https://aclanthology.org/W18-0549/', 'Paper mencocokkan esai dan sumber pada tingkat kalimat. Di sini diadaptasi menjadi respons–referensi pada tingkat token; bukan replikasi langsung.'),
}
RECIPES = {
 'E21': ('MLP · tiga input', None, 'Acuan tiga input untuk seluruh skenario.', 'S3: CLS RoBERTa768 + BGE1024 → proyeksi256 → concat1024 → MLP → skor', 'Single sequence: soal + referensi + respons', 'MLP S2/S3 1e-4', ['mlp','norm']),
 'E22': ('MLP · konteks pair', 'E21', 'Menguji format pasangan teks dengan informasi dan label yang sama.', 'S3: konteks–respons → RoBERTa CLS768 + BGE → concat → MLP → skor', 'Pair: (soal + referensi, respons)', 'MLP S2/S3 1e-4', ['mlp','norm']),
 'E23': ('BiGRU', 'E21', 'Menguji urutan token respons sebelum fusion.', 'S3: token respons768 → BiGRU64/arah → state128 → proyeksi768 → concat BGE256 → MLP', 'Single sequence; mask token respons', 'MLP S2/S3 1e-4', ['bigru']),
 'E24': ('Attention pooling', 'E21', 'Mempelajari bobot token respons saat membuat ringkasan.', 'S3: token respons768 → attention768→64→1 → weighted sum768 → concat BGE256 → MLP', 'Single sequence; mask token respons', 'MLP S2/S3 1e-4', ['attention']),
 'E25': ('Attention + gated fusion', 'E24', 'Menguji kontribusi relatif representasi RoBERTa dan BGE.', 'S3: attention pooling → gate → concat[2g×RoBERTa, 2(1−g)×BGE] → MLP', 'Single sequence; mask token respons', 'MLP S2/S3 1e-4', ['attention','gate']),
 'E26': ('BiGRU + attention pooling', 'E23', 'Meringkas urutan keluaran BiGRU dengan bobot attention.', 'S3: token respons → BiGRU → attention pooling128 → proyeksi768 → concat BGE256 → MLP', 'Single sequence; mask token respons', 'MLP S2/S3 1e-4', ['bigru','attention']),
 'E27': ('BiGRU + co-attention', 'E23', 'Mencocokkan token respons dan referensi sebelum ringkasan state ujung.', 'S3: BiGRU respons128 ↔ referensi128 → matching640→128 → state ujung128 → proyeksi768 → fusion → MLP', 'Single sequence; mask respons dan referensi', 'MLP S2/S3 1e-4', ['bigru','coattention']),
 'E28': ('BiGRU · LR MLP S3 5e-6', 'E23', 'Ablasi learning rate MLP S3; arsitektur E23 dipertahankan.', 'S3: BiGRU → state128 → proyeksi768 → fusion → MLP; tanpa attention tambahan', 'Single sequence; mask token respons', 'MLP S2 1e-4 · MLP S3 5e-6', ['bigru']),
 'E29': ('BiGRU + pooling · LR MLP S2/S3 5e-6', 'E26', 'Ablasi learning rate kedua MLP; layer lain mengikuti E26.', 'S3: BiGRU → attention pooling → proyeksi → fusion → MLP', 'Single sequence; mask token respons', 'MLP S2/S3 5e-6 · linear S1 1e-4', ['bigru','attention']),
 'E30': ('BiGRU + co-attention + pooling', 'E26', 'Mencocokkan respons–referensi, kemudian merangkum hasil matching dengan attention pooling.', 'S3: BiGRU respons ↔ referensi → co-attention → matching → attention pooling → proyeksi768 → fusion → MLP', 'Single sequence; mask respons dan referensi', 'MLP S2/S3 1e-4 · linear S1 1e-4', ['bigru','attention','coattention']),
}
PENDING_SOURCE = {
 'E29': 'notebooks/experiments/AES_300_E29_E26_BiGRU_Attention_MLPLR5e6_GPU.ipynb',
 'E30': 'notebooks/experiments/AES_300_E30_E26_BiGRU_CoAttention_Pooling_GPU.ipynb',
}
SOURCE_PIN = '7a76191be20604482fa53f770699563b380120ba'


def build_identities(snapshots):
    result = {}
    for experiment, (name, base, purpose, flow, text_format, head_lr, papers) in RECIPES.items():
        available = experiment in snapshots
        if not available and experiment not in PENDING_SOURCE:
            continue
        snapshot = snapshots.get(experiment, {})
        commit = snapshot.get('source_commit', SOURCE_PIN)
        path = snapshot.get('source_path', PENDING_SOURCE.get(experiment))
        citations = [dict(key=k, title=PAPERS[k][0], url=PAPERS[k][1], scope=PAPERS[k][2]) for k in ['roberta','bge',*papers]]
        result[experiment] = dict(id=experiment, name=name, base=base, purpose=purpose, flow=flow,
            text_format=text_format, head_lr=head_lr, encoder_lr='RoBERTa S1/S3 5e-6 · BGE frozen',
            other_lr='Layer trainable di luar encoder/head 1e-4',
            inputs=['soal','jawaban_referensi','response_text'], label='rater · 0–100',
            heads='S1 linear · S2/S3 MLP', citations=citations,
            evidence='Hasil unggahan tersedia' if available else 'Menunggu hasil unggahan',
            result_available=available, book_available=available,
            reserved_code=None, reserved_output=None,
            source_commit=commit, source_path=path,
            source_url='https://github.com/MalikHariyanto/Automated_Essay_Scoring/blob/'+commit+'/'+path,
            colab_url='https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/'+commit+'/'+path,
            adaptation=('Nilai LR adalah pilihan ablasi proyek; paper tidak menetapkan 5e-6 sebagai nilai terbaik.' if experiment in ('E28','E29') else
                        'Format pair adalah ablasi implementasi proyek; bukan klaim paper membuktikan format ini lebih unggul.' if experiment=='E22' else
                        'Komponen diadaptasi untuk proyek ini; keunggulan hanya dinilai dari hasil eksperimen yang tersedia.'))
    return result
