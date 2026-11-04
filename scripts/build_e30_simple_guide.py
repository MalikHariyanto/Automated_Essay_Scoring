"""Standalone, source-checked guide to complete native E30; no student data."""
import ast
from collections import OrderedDict
import hashlib
import html
import io
import json
import keyword
from pathlib import Path
import re
import tokenize

from e30_simple_guide_content import NOTES

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'notebooks/experiments/AES_300_E30_Lengkap_S1_S2_S3_Native_Learnable_GPU.ipynb'
EXPECTED_SHA = '3d4f035d5e89989c5c7bd506e10bc06cc7f38d2726137568e204ca6163bfce4a'
COMMIT = '16670531f76bf27d07fe87c3c73614ae76d7c451'
SOURCE_PATH = SOURCE.relative_to(ROOT).as_posix()
GITHUB = f'https://github.com/MalikHariyanto/Automated_Essay_Scoring/blob/{COMMIT}/{SOURCE_PATH}'
COLAB = f'https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/{COMMIT}/{SOURCE_PATH}'
RAW = f'https://raw.githubusercontent.com/MalikHariyanto/Automated_Essay_Scoring/{COMMIT}/{SOURCE_PATH}'


def readable_prose(markup):
    """Space prose without touching code, identifiers in <code>, or HTML attributes."""
    blocked = 0
    result = []
    for part in re.split(r'(<[^>]*>)', markup):
        if part.startswith('<'):
            if re.match(r'</(?:pre|code)>', part):
                blocked -= 1
            result.append(part)
            if re.match(r'<(?:pre|code)(?:\s|>)', part):
                blocked += 1
        elif blocked:
            result.append(part)
        else:
            for piece in re.split(r'(&(?:#\d+|#x[\da-fA-F]+|[A-Za-z][A-Za-z0-9]+);)', part):
                if piece.startswith('&') and piece.endswith(';'):
                    result.append(piece)
                    continue
                piece = re.sub(r'(BiGRU|BGE|MLP|RoBERTa|LayerNorm|CLS|CV|Cell)(?=\d)', r'\1 ', piece)
                piece = re.sub(r'(?<=[a-zà-ÿ])(?=\d)', ' ', piece)
                piece = re.sub(r'(?<=\d)(?=[a-df-zà-ÿ])', ' ', piece)
                piece = re.sub(r'(?<=[,;:])(?=[^\s\d])', ' ', piece)
                piece = re.sub(r'(?<=%)(?=[A-Za-z])', ' ', piece)
                result.append(piece.replace('tepat.5', 'tepat 0,5'))
    assert blocked == 0
    return ''.join(result)


def source(cell):
    return cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])


def highlighted_lines(text, first=1, last=None):
    lines = text.splitlines()
    intervals = [[] for _ in lines]
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        kind = None
        if token.type == tokenize.COMMENT:
            kind = 'comment'
        elif token.type == tokenize.STRING:
            kind = 'string'
        elif token.type == tokenize.NUMBER:
            kind = 'number'
        elif token.type == tokenize.NAME and keyword.iskeyword(token.string):
            kind = 'keyword'
        if kind:
            for row in range(token.start[0], token.end[0]+1):
                if row > len(lines):
                    continue
                start = token.start[1] if row == token.start[0] else 0
                end = token.end[1] if row == token.end[0] else len(lines[row-1])
                intervals[row-1].append((start, end, kind))
    rendered = []
    last = last or len(lines)
    for number in range(first, last+1):
        line = lines[number-1]
        cursor, pieces = 0, []
        for start, end, kind in sorted(intervals[number-1]):
            pieces.append(html.escape(line[cursor:start]))
            pieces.append(f'<span class="tok-{kind}">'+html.escape(line[start:end])+'</span>')
            cursor = end
        pieces.append(html.escape(line[cursor:]))
        rendered.append(f'<span class="code-line"><span class="ln" aria-hidden="true">{number}</span><span class="lt">'+''.join(pieces)+'</span></span>')
    return '<pre><code>'+''.join(rendered)+'</code></pre>'


def build():
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_SHA, 'Notebook source changed: review explanations before rebuilding.'
    notebook = json.loads(raw)
    cells, by_role, chapters = [], {}, OrderedDict()
    chapter = 'Mulai'
    for index, cell in enumerate(notebook['cells'], 1):
        text = source(cell)
        if cell['cell_type'] == 'markdown':
            heading = re.search(r'^## (\d+ · .+)$', text, re.M)
            if heading:
                chapter = heading[1]
            continue
        assert not cell['outputs'] and cell['execution_count'] is None
        role = cell['metadata']['aes30full_role']
        assert role in NOTES, role
        ast.parse(text)
        data = dict(number=index, role=role, text=text, chapter=chapter)
        cells.append(data)
        by_role[role] = data
        chapters.setdefault(chapter, []).append(data)
    assert len(cells) == 89 and set(by_role) == set(NOTES)
    constants = {v.targets[0].id: ast.literal_eval(v.value) for v in ast.parse(by_role['hyperparameters']['text']).body
                 if isinstance(v, ast.Assign) and isinstance(v.targets[0], ast.Name)}
    assert constants['ENCODER_LR'] == 5e-6 and constants['S3_LAYERS_LR'] == 1e-4
    assert constants['EPOCHS'] == 25 and constants['FOLDS'] == constants['EARLY_STOPPING_PATIENCE'] == 5

    def link(role, label='Lihat cell lengkap'):
        item = by_role[role]
        return f'<a href="#cell-{item["number"]}">{html.escape(label)} · Cell {item["number"]}</a>'

    def snippet(role, start, end=None, *, opened=False, title=None, caption='Cuplikan asli; nomor baris mengikuti cell notebook.'):
        item = by_role[role]
        lines = item['text'].splitlines()
        begin = next(i for i, line in enumerate(lines) if start in line)
        finish = next(i for i in range(begin, len(lines)) if (end or start) in lines[i])
        label = title or NOTES[role][0]
        return f'''<details class="snippet"{' open' if opened else ''}><summary><div>{html.escape(label)}<span>Cell {item['number']} · baris {begin+1}–{finish+1}</span></div></summary>
<div class="code-toolbar"><span>Python · cuplikan asli</span><button class="copy" type="button" data-copy-cell="{item['number']}" data-start="{begin+1}" data-end="{finish+1}">Salin kode</button></div>
{highlighted_lines(item['text'], begin+1, finish+1)}<div class="snippet-caption">{html.escape(caption)} {link(role)}</div></details>'''

    def section(identity, number, title, body):
        return f'<section id="{identity}"><div class="section-top"><span class="section-no">{number}</span><h2>{title}</h2></div>{body}</section>'

    content = []
    content.append(section('input', '01', 'Variabel input ada di sini.', f'''
<p class="section-intro"><strong>X memiliki tiga kolom teks.</strong> Satu baris berisi soal, jawaban referensi, dan jawaban siswa yang saling berkaitan untuk satu kasus penilaian.</p>
<div class="io-grid"><div class="io-item"><span class="type">X / konteks</span><strong>soal</strong><p>Pertanyaan yang harus dijawab siswa.</p></div><div class="io-item"><span class="type">X / acuan teks</span><strong>jawaban_referensi</strong><p>Isi jawaban acuan untuk pertanyaan tersebut.</p></div><div class="io-item"><span class="type">X / respons</span><strong>response_text</strong><p>Jawaban siswa yang akan dinilai.</p></div></div>
<div class="target-line"><strong>y / rater</strong><span>Satu nilai guru 0–100 untuk jawaban pada baris yang sama.</span></div>
{snippet('input_columns', 'QUESTION_COLUMN =', 'INPUT_COLUMNS =', opened=True, title='Kode yang menentukan nama kolom X dan y')}
<ul class="line-notes"><li><code>QUESTION_COLUMN</code>, <code>REFERENCE_COLUMN</code>, dan <code>ANSWER_COLUMN</code> menunjuk nama kolom Excel.</li><li><code>INPUT_COLUMNS</code> mengumpulkan ketiga nama kolom X. <code>TARGET_COLUMN</code> menunjuk nilai guru, terpisah dari daftar X.</li></ul>
{snippet('dataset_execute_xy', 'X = pd.DataFrame', 'assert TARGET_COLUMN not', title='Kode yang membentuk X dan y aktual')}
<p class="text-block"><code>X</code> tetap tabel tiga kolom. <code>y</code> adalah daftar nilai guru. Di dalam proses, nilai yang sama dibawa melalui <code>bundle.texts</code>, pasangan soal–referensi, dan <code>bundle.y</code>. Ini cara fungsi mengakses data yang sama.</p>
<h3 class="subheading">Dibaca satu-satu atau sebagai keterhubungan?</h3>
<p class="text-block">Saat membaca Excel, kolom diambil terpisah agar perannya jelas. Saat menyiapkan masukan encoder, token ketiganya <strong>disusun bersama dalam satu urutan untuk setiap window</strong>. RoBERTa dan BGE membaca urutan gabungan tersebut melalui tokenizer masing-masing.</p>
<div class="joined"><div class="label">Susunan konseptual input / bukan keluaran model</div><div class="joined-line"><span>Soal: …</span><span class="separator">→</span><span>Jawaban referensi: …</span><span class="separator">→</span><span>Jawaban siswa: …</span></div><p>Tiap model menghasilkan satu prediksi untuk jawaban itu. Tidak ada tiga nilai terpisah untuk soal, referensi, dan respons.</p></div>
{snippet('definition:context_token_ids', 'qids = tokenizer.encode', 'reference_kept = rids', title='Kode penanda dan token konteks')}
{snippet('definition:pair_content_windows', 'prepared = tokenizer.build_inputs_with_special_tokens(context_ids+second)', title='Di sini konteks dan respons masuk satu urutan')}
<div class="note"><strong>Bersama, dengan posisi tiap bagian tetap diketahui.</strong><p>Jika jawaban panjang, respons dipecah menjadi window yang saling overlap. Konteks soal–referensi disertakan pada tiap window. Mask kemudian memilih posisi respons dan referensi untuk S3; itu bukan memisahkan penilaian menjadi tiga skor.</p></div>
<details class="plain-details"><summary>Bagaimana dengan Email address dan response_number?</summary><div><p class="small">Keduanya metadata, bukan teks masukan encoder. Email dan duplikat respons membantu pembentukan kelompok train–test agar kelompok yang sama tidak menyeberang. <code>response_number</code> dibawa sebagai metadata respons. {link('dataset_definitions', 'Lihat pembacaan dataset')}.</p></div></details>
'''))

    content.append(section('model', '02', 'Model yang dipakai, dan tugasnya.', f'''
<p class="section-intro">Nama checkpoint memilih model pretrained. Kelas <code>SingleScoreModel</code> di notebook merakit encoder dan lapisan prediktornya.</p>
{snippet('model_names', 'ROBERTA_CHECKPOINT =', 'BGE_CHECKPOINT =', opened=True, title='Nama checkpoint yang benar-benar digunakan')}
<div class="table-wrap"><table><thead><tr><th>Encoder</th><th>Peran di E30</th><th>Saat training</th></tr></thead><tbody><tr><td>RoBERTa Indonesia</td><td><code>flax-community/indonesian-roberta-base</code><br>Representasi kontekstual,768angka per token.</td><td>Bobot diperbarui di S1 dan S3.</td></tr><tr><td>BGE-M3</td><td><code>BAAI/bge-m3</code><br>Satu vektor1024per jawaban dari tiga teks.</td><td>Encoder frozen;head/proyeksi di luarnya tetap belajar.</td></tr></tbody></table></div>
<p class="small">Revision checkpoint juga dikunci di Cell7. {link('definition:encoder', 'AutoModel.from_pretrained memuat encoder')} dan {link('definition:tokenizer', 'tokenizer mengubah teks ke ID')}.</p>
<div class="tablist" role="tablist" aria-label="Pilih skenario model" data-tabs><button type="button" id="tab-s1" role="tab" aria-controls="panel-s1" aria-selected="false" tabindex="-1">S1 · RoBERTa</button><button type="button" id="tab-s2" role="tab" aria-controls="panel-s2" aria-selected="false" tabindex="-1">S2 · BGE</button><button type="button" id="tab-s3" role="tab" aria-controls="panel-s3" aria-selected="true" tabindex="0">S3 · Fusion</button></div>
<div id="panel-s1" role="tabpanel" aria-labelledby="tab-s1" hidden><ol class="path"><li><span class="step-num">01 / INPUT</span><strong>Tiga teks</strong><p>Soal,referensi,respons disusun dalam window.</p></li><li><span class="step-num">02 / ENCODER</span><strong>RoBERTa</strong><p>CLS tiap window,dirata-ratakan per jawaban.</p></li><li><span class="step-num">03 / HEAD</span><strong>Dropout + linear</strong><p>768fitur dipetakan ke satu angka.</p></li><li><span class="step-num">04 / OUTPUT</span><strong>Prediksi S1</strong><p>Dikembalikan ke skala0–100.</p></li></ol><p class="scenario-note">S1 memakai linear. Bobot RoBERTa dan linear belajar dari label. {link('definition:SingleScoreModel', 'Lihat cabang S1')}.</p></div>
<div id="panel-s2" role="tabpanel" aria-labelledby="tab-s2" hidden><ol class="path"><li><span class="step-num">01 / INPUT</span><strong>Tiga teks</strong><p>Isi gabungan sama perannya;tokenizer milik BGE.</p></li><li><span class="step-num">02 / ENCODER</span><strong>BGE frozen</strong><p>CLS antar-window dirata-ratakan,lalu L2 normalize.</p></li><li><span class="step-num">03 / HEAD</span><strong>MLP 1024→256→1</strong><p>Linear,GELU,dropout,linear keluaran.</p></li><li><span class="step-num">04 / OUTPUT</span><strong>Prediksi S2</strong><p>Satu angka untuk jawaban tersebut.</p></li></ol><p class="scenario-note">Frozen berlaku pada encoder BGE. MLP S2 tetap dilatih terhadap label. {link('definition:bge_features', 'Lihat fitur BGE')}.</p></div>
<div id="panel-s3" role="tabpanel" aria-labelledby="tab-s3"><ol class="path"><li><span class="step-num">01 / KONTEKS</span><strong>RoBERTa</strong><p>Semua token tiga teks diproses bersama.</p></li><li><span class="step-num">02 / HUBUNGAN</span><strong>BiGRU + co-attention</strong><p>Respons dicocokkan dengan isi referensi.</p></li><li><span class="step-num">03 / RANGKUM</span><strong>Pooling + fusion BGE</strong><p>Ringkasan RoBERTa digabung fitur BGE.</p></li><li><span class="step-num">04 / HEAD</span><strong>MLP → satu nilai</strong><p>Belajar dari gabungan representasi.</p></li></ol><p class="scenario-note">S3 adalah jalur fusion E30 MLP. S1 dan S3 dilatih dengan instance RoBERTa terpisah;S3 tidak mengambil prediksi S1 atau S2 sebagai input.</p></div>
'''))

    content.append(section('fusion', '03', 'Di sini hubungan teks menjadi prediksi S3.', f'''
<p class="section-intro">Representasi token → BiGRU → co-attention → pooling → concat BGE → MLP. Setiap tahap punya tugas berbeda.</p>
<div class="table-wrap"><table><thead><tr><th>Tahap</th><th>Apa yang dilakukan?</th><th>Kode asli</th></tr></thead><tbody><tr><td>RoBERTa</td><td>Vektor token768yang sudah membaca konteks tiga teks.</td><td>{link('definition:SingleScoreModel', 'forward')}</td></tr><tr><td>BiGRU</td><td>Urutan respons diolah dua arah:hidden64per arah→keluaran128per token. Teks asli tidak dibalik.</td><td>{link('definition:AnswerBiGRUAttention', 'self.gru')}</td></tr><tr><td>Co-attention</td><td>Respons128dicocokkan dengan referensi yang diproyeksikan ke128. Skor dot product dibagi√128,lalu softmax memberi bobot pencocokan.</td><td>{link('definition:AnswerBiGRUCoAttentionPool', 'scores / a_to_r / r_to_a')}</td></tr><tr><td>Matching</td><td>Lima fitur pencocokan128digabung jadi640,lalu Linear+tanh memadatkannya kembali ke128.</td><td>{link('definition:AnswerBiGRUCoAttentionPool', 'matching_projection')}</td></tr><tr><td>Attention pooling</td><td>Token hasil matching diberi bobot,dirangkum menjadi128,lalu diproyeksikan ke768.</td><td>{link('definition:AnswerBiGRUCoAttentionPool', 'pooling_weights / summary')}</td></tr><tr><td>Proyeksi BGE</td><td>BGE frozen1024diubah oleh Linear+LayerNorm yang belajar menjadi256.</td><td>{link('definition:SingleScoreModel', 'self.projection')}</td></tr><tr><td>Concat + MLP</td><td>768+256menjadi1024fitur. MLP1024→256→1menghasilkan satu angka.</td><td>{link('definition:SingleScoreModel', "features['concat']")}</td></tr></tbody></table></div>
<details class="plain-details"><summary>Co-attention dan attention pooling, bedanya apa?</summary><div><p class="small"><strong>Co-attention</strong> membangun hubungan respons dengan referensi. <strong>Attention pooling</strong> merangkum token hasil hubungan itu menjadi satu vektor. Bobot attention bukan skor kebenaran setiap kata atau ukuran kepentingan kausal.</p>{snippet('definition:AnswerBiGRUCoAttentionPool', 'pooling_scores =', 'summary = (pooling_weights', title='Kode attention pooling setelah matching')}</div></details>
<h3 class="subheading">Concat fusion tepatnya ada di sini.</h3>
<div class="dimensions"><div class="dimension"><strong>768</strong><span>RoBERTa setelah LayerNorm</span></div><span class="operator">+</span><div class="dimension bge"><strong>256</strong><span>Proyeksi BGE</span></div><span class="operator">=</span><div class="dimension total"><strong>1024</strong><span>Input MLP S3</span></div></div>
{snippet('definition:SingleScoreModel', "features['roberta_layernorm'] =", "prediction = self.head(features['concat'])", opened=True, title='Kode proyeksi, concat, dan pemanggilan MLP')}
<ul class="line-notes"><li><code>self.roberta_norm(essay)</code> menormalisasi ringkasan RoBERTa.</li><li><code>self.projection(bge)</code> mengubah1024komponen BGE menjadi256. Layer proyeksi ini ikut belajar.</li><li><code>torch.cat(..., dim=1)</code> menyambung komponen fitur untuk setiap jawaban.</li><li><code>self.head(features['concat'])</code> meneruskan gabungan1024ke MLP,lalu menghasilkan satu prediksi.</li></ul>
{snippet('definition:SingleScoreModel', '# MLP S3:', 'self.sequence_pool =', title='Definisi MLP dan pooling S3')}
<div class="note warm"><strong>Jumlah dimensi bukan persentase pengaruh.</strong><p>768dan256menunjukkan banyaknya komponen vektor. Tidak ada aturan bobot pengaruh75:25dari ukuran itu;pemetaan ke nilai dipelajari model. Di dalam matching juga ada <code>torch.cat</code> lain,yaitu5×128→640. Itu tahap berbeda dari concat fusion1024di atas.</p></div>
'''))

    content.append(section('label', '04', 'Label tidak masuk encoder. Tapi model belajar darinya.', f'''
<p class="section-intro"><strong>rater adalah target nilai guru.</strong> Ia tidak ditambahkan ke teks soal,referensi,atau jawaban. Saat training,label membentuk loss yang menentukan pembaruan bobot.</p>
<div class="tablist" role="tablist" aria-label="Peran label pada tahap program" data-tabs><button type="button" id="tab-train" role="tab" aria-controls="mode-train" aria-selected="true" tabindex="0">Saat belajar</button><button type="button" id="tab-predict" role="tab" aria-controls="mode-predict" aria-selected="false" tabindex="-1">Saat prediksi baru</button></div>
<div class="mode-explanation" id="mode-train" role="tabpanel" aria-labelledby="tab-train"><h4>Label diperlukan untuk supervised training.</h4><p>Model membuat prediksi dari X. Prediksi dibandingkan dengan y melalui MSE. Gradien dari selisih tersebut memperbarui bobot yang trainable.</p><div class="mode-steps"><span>X → prediksi</span><span class="arr">+</span><span>y / rater</span><span class="arr">→</span><span>loss MSE</span><span class="arr">→</span><span>gradien → update bobot</span></div></div>
<div class="mode-explanation" id="mode-predict" role="tabpanel" aria-labelledby="tab-predict" hidden><h4>Jawaban baru tidak perlu rater untuk diprediksi.</h4><p>Soal,referensi,dan respons tetap dibutuhkan. Program memuat checkpoint terlatih lalu menghasilkan prediksi. Tidak ada loss,backward,atau update bobot pada langkah ini. Tanpa nilai guru,error jawaban baru belum bisa dihitung.</p><div class="mode-steps"><span>Tiga teks baru</span><span class="arr">→</span><span>checkpoint terlatih</span><span class="arr">→</span><span>prediksi</span></div><p class="small" style="margin-top:12px">{link('definition:score_answer', 'score_answer tidak meminta nilai guru')}. Target kosong tidak diisi skor0.</p></div>
{snippet('definition:fit', 'targets = torch.as_tensor', 'loss_total +=', title='Kode label → loss → backward')}
<ul class="line-notes"><li><code>bundle.y[ids]</code> mengambil label untuk baris dalam batch.</li><li>Label0–100diubah menjadi target0–1untuk loss;pada rentang ini artinya dibagi100.</li><li><code>engine.forward(ids)</code> membuat prediksi dari teks. Label tidak menjadi argumen forward.</li><li><code>(prediction-targets)**2</code> mengukur selisih kuadrat. <code>backward()</code> menghitung gradien;pembaruan dijalankan melalui <code>scaler.step(optimizer)</code>.</li></ul>
<div class="note"><strong>Kalimat yang tepat untuk menjelaskannya:</strong><p>“Nilai guru tidak menjadi fitur input encoder,tetapi menjadi target untuk menghitung loss,melatih bobot model,memilih epoch lewat validasi,dan mengukur hasil evaluasi.”</p></div>
<p class="small">Dalam E30 ini,pemilihan split tidak memakai label. {link('definition:make_split', 'Lihat selection_uses_labels = False')}. Prediksi0–1dikembalikan ke skala0–100di {link('definition:Engine', 'Engine.predict')}.</p>
'''))

    content.append(section('pengaturan', '05', 'Hyperparameter ada di satu tempat.', f'''
<p class="section-intro">Pengaturan utama ada di <strong>Cell11,bagian5</strong>. <code>optimizer_groups</code> memasangkan LR tersebut dengan bobot yang benar.</p>
<div class="table-wrap"><table><thead><tr><th>Yang diatur</th><th>Nilai E30</th><th>Artinya</th></tr></thead><tbody><tr><td>RoBERTa S1/S3</td><td class="lr-number">5e-6</td><td>0,000005. LR awal/puncak encoder yang ikut dilatih.</td></tr><tr><td>Linear S1</td><td class="lr-number">1e-4</td><td>0,0001untuk head linear.</td></tr><tr><td>MLP S2 dan S3</td><td class="lr-number">1e-4</td><td>Mempelajari pemetaan fitur ke nilai.</td></tr><tr><td>BiGRU,co-attention,pooling,norm S3</td><td class="lr-number">1e-4</td><td>Kelompok layer pengolah representasi S3.</td></tr><tr><td>Proyeksi BGE S3</td><td class="lr-number">1e-4</td><td>Proyeksi1024→256ikut belajar.</td></tr><tr><td>Encoder BGE</td><td>Frozen</td><td>Bobot encoder tidak diperbarui saat supervised training.</td></tr></tbody></table></div>
{snippet('hyperparameters', 'EPOCHS =', 'PROJECTION_LR =', title='Kode pengaturan training dan learning rate')}
{snippet('definition:optimizer_groups', 'rates =', "'pooling_norm': cfg", title='Kode kelompok bobot dan LR')}
<p class="text-block">Learning rate mengatur besar langkah pembaruan parameter. Angka di tabel merupakan LR awal/puncak,<strong>bukan LR konstan pada setiap langkah</strong>:E30 memakai warmup10%lalu scheduler linear. LR lebih besar belum tentu menghasilkan model lebih baik.</p>
<details class="plain-details"><summary>Epoch,fold,batch,dropout,dan batas token</summary><div><div class="table-wrap"><table><thead><tr><th>Parameter</th><th>Nilai</th><th>Fungsi</th></tr></thead><tbody><tr><td>EPOCHS</td><td>25</td><td>Batas maksimal putaran training tiap fold.</td></tr><tr><td>FOLDS</td><td>5</td><td>Lima pembagian latih–validasi di development.</td></tr><tr><td>EARLY_STOPPING_PATIENCE</td><td>5</td><td>Lima epoch tanpa rekor RMSE validasi baru.</td></tr><tr><td>MICRO_BATCH / EFFECTIVE_BATCH</td><td>1 / 8</td><td>Diproses bertahap;gradien dikumpulkan sebelum update batch efektif.</td></tr><tr><td>Hidden BiGRU</td><td>64per arah</td><td>Dua arah menghasilkan128fitur/token.</td></tr><tr><td>DROPOUT</td><td>0,1</td><td>Aktivasi dinolkan acak saat training;eval menonaktifkannya.</td></tr><tr><td>WEIGHT_DECAY / GRADIENT_CLIP</td><td>0,01 / 1</td><td>Regularisasi parameter dan batas norm gradien.</td></tr><tr><td>Token RoBERTa / BGE</td><td>512 / 1024</td><td>Batas tiap window,termasuk konteks dan token khusus.</td></tr><tr><td>Konteks / soal / overlap</td><td>192 / 64 / 128</td><td>Anggaran konteks,batas soal,tumpang tindih window.</td></tr><tr><td>TRAIN_SEED / SPLIT_SEED</td><td>42 / 2026</td><td>Konstanta pengacakan;bukan loop tiga seed.</td></tr></tbody></table></div>{link('hyperparameters', 'Lihat seluruh konstanta asli')}</div></details>
'''))

    content.append(section('training', '06', 'Dari training sampai lima Excel.', f'''
<p class="section-intro">Tiga skenario memakai split dan fold yang sama. Ketiganya dilatih baru,lalu dibandingkan pada data test yang disimpan terpisah.</p>
<div class="training-flow"><div><span class="count">5×</span><h4>CV per skenario</h4><p>Model baru tiap fold. Validasi memilih epoch dengan RMSE terendah.</p></div><div><span class="count">1×</span><h4>Refit per skenario</h4><p>Model baru pada seluruh development,selama ceil median epoch terbaik lima fold.</p></div><div><span class="count">1×</span><h4>Test per skenario</h4><p>Model final dikunci,lalu dievaluasi pada test held-out.</p></div></div>
<p class="text-block">S1/S2/S3 lengkap berarti <strong>15fit CV dan3refit</strong>. Ini tiga skenario pembanding,bukan pengulangan tiga seed. Target holdout sekitar20%baris;pembagian riil mengikuti kelompok. Kelompok email/duplikat dijaga agar tidak menyeberang train–test.</p>
<div class="note warm"><strong>Early stopping bukan berarti model tidak belajar.</strong><p>Training dapat membaik sementara validasi tidak mencetak rekor baru. Patience5mengacu pada perbaikan RMSE validasi terbaik;bobot epoch terbaik dipulihkan. Test tidak dipakai memilih epoch.</p></div>
<p class="small">{link('definition:fit', 'Loss,backward,early stopping')} · {link('definition:train_all', 'Urutan CV/refit/test')} · {link('training_execute', 'Pemanggilan training')}</p>
<h3 class="subheading">Metrik dan label dipakai di mana?</h3>
<p class="text-block">RMSE dan Pearson memakai prediksi mentah skala0–100. QWK dan Gwet AC2 memakai salinan kategori yang dibatasi dan dibulatkan;tepat.5memilih kategori lebih rendah. Metrik membutuhkan pasangan nilai guru dan prediksi. MSE training pada target0–1berbeda satuan dari RMSE dalam poin.</p>
<p class="small">{link('definition:metrics', 'Lihat rumus dan konversi')} · Evaluasi pada test lama yang sudah sering dilihat tetap eksploratif.</p>
<div class="table-wrap"><table><thead><tr><th>File hasil</th><th>Yang bisa diperiksa</th></tr></thead><tbody><tr><td>01 · Data dan Tokenisasi</td><td>Tiga teks,label,split/fold,token yang dibaca.</td></tr><tr><td>02 · Vektor Jawaban</td><td>RoBERTa768,BGE1024,ringkasan128,fusion1024,bobot attention.</td></tr><tr><td>03 · Vektor Kata dan Kalimat</td><td>Inspeksi representasi token/kata/kalimat.</td></tr><tr><td>04 · Prediksi dan Evaluasi</td><td>Prediksi S1/S2/S3 dan empat metrik test.</td></tr><tr><td>05 · Riwayat Pelatihan</td><td>Loss,RMSE validasi,epoch pilihan,15fold,pengaturan aktual.</td></tr></tbody></table></div>
<p class="small">{link('download_execute', 'Di sini ZIP lima Excel dibuat')}. Panduan ini memuat source dan penjelasan,tanpa hasil individual siswa.</p>
'''))

    cards = []
    for chapter_name, items in chapters.items():
        group = []
        for item in items:
            title, purpose, mechanism = NOTES[item['role']]
            names = [node.name for node in ast.parse(item['text']).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
            named = '<p class="function-label">Nama di kode: '+', '.join('<code>'+html.escape(name)+'</code>' for name in names)+'</p>' if names else ''
            group.append(f'''<details class="code-card" id="cell-{item['number']}" data-cell="{item['number']}" data-role="{html.escape(item['role'])}"><summary><span class="cell-badge">Cell {item['number']}</span><span><span class="code-title">{html.escape(title)}</span><span class="code-preview">{html.escape(purpose)}</span></span></summary><div class="code-body"><p class="code-purpose">{html.escape(mechanism)}</p>{named}<div class="code-toolbar"><span>Cell {item['number']} · kode lengkap,tanpa perubahan</span><button class="copy" type="button" data-copy-cell="{item['number']}">Salin cell</button></div>{highlighted_lines(item['text'])}</div></details>''')
        cards.append('<div class="code-chapter"><h3 class="chapter-title">Bagian '+html.escape(chapter_name)+'</h3>'+''.join(group)+'</div>')
    content.append(section('peta-kode', '07', 'Cari bagian kode yang ingin dipahami.', '''
<p class="section-intro">Semua <strong>89cell kode</strong> punya penjelasan fungsi dan mekanisme. Cari nama variabel,fungsi,atau nomor cell. Misalnya <code>rater</code>,<code>concat</code>,<code>fit</code>,atau <code>111</code>.</p>
<div class="catalog-head"><div class="search-wrap"><label class="search-label" for="code-search">Cari fungsi,variabel,atau cell</label><input id="code-search" type="search" placeholder="Contoh: concat,rater,optimizer…" autocomplete="off" spellcheck="false" aria-controls="code-catalog"></div><span id="code-count" class="small result-count" aria-live="polite">89dari89cell kode</span></div>
<p id="empty-search" class="empty-search" hidden>Tidak ada yang cocok. Coba nama fungsi atau variabel lain.</p>
<details class="plain-details" id="code-catalog"><summary>Buka daftar lengkap89cell kode</summary><div>'''+''.join(cards)+'</div></details>'))
    template = (ROOT/'scripts/templates/e30_simple_guide.html').read_text()
    # JSON string preserves the exact original notebook bytes after decoding.
    data = json.dumps(raw.decode(), ensure_ascii=False).replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    replacements = {'@@CSS@@': (ROOT/'scripts/templates/e30_simple_guide.css').read_text(),
                    '@@JS@@': (ROOT/'scripts/templates/e30_simple_guide.js').read_text(),
                    '@@CONTENT@@': readable_prose(''.join(content)), '@@NOTEBOOK_JSON@@': data,
                    '@@SOURCE_SHA@@': EXPECTED_SHA, '@@GITHUB_URL@@': GITHUB, '@@COLAB_URL@@': COLAB, '@@NOTEBOOK_URL@@': RAW}
    for marker, value in replacements.items():
        template = template.replace(marker, value)
    assert not re.search(r'@@[A-Z_]+@@', template)
    outputs = [ROOT/'docs/panduan_e30_mlp.html', ROOT/'downloads/Panduan_E30_MLP.html']
    for path in outputs:
        path.parent.mkdir(exist_ok=True)
        path.write_text(template)
    print(json.dumps({'files': [str(p.relative_to(ROOT)) for p in outputs], 'bytes': len(template.encode()),
                      'source_sha256': EXPECTED_SHA, 'explained_code_cells': len(cells), 'student_data': False}))
    return template


if __name__ == '__main__':
    build()
