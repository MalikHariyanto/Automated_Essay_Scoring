"""Buku kode E6 lengkap dengan catatan sebidang; hanya source publik."""
import hashlib
import json
from pathlib import Path
import subprocess

from build_e6_learning_excel import ROWS, PIN, URL
from build_mlp_study import ROOT, SNAPSHOT, content, source

TARGET = ROOT/'downloads/Belajar_Kode_E6.html'
TEMPLATES = ROOT/'scripts/templates'
NOTEBOOK = 'notebooks/experiments/AES_300_E6_S3_LR5e6_CPU_Reuse.ipynb'

# Penjelasan padanan mengikuti screenshot pengguna, bukan mengganti metode E6.
MAPPINGS = [
    ('Hyperparameter & optimisasi GPU', 'MODEL_NAME, EPOCHS, PATIENCE, LEARNING_RATE',
     'Pengaturan model dan latihan', 12, 'ROBERTA_CHECKPOINT=',
     'Di E6 pengaturan utama berada bersama dalam satu blok. RoBERTa S1 memakai LR 1e-5, S3 5e-6; head/proyeksi 1e-4. E6 yang dibahas memilih CPU.', 'Peran setara'),
    ('Konstruksi pasangan teks / context_text', 'Soal + kunci jawaban dipasangkan dengan jawaban siswa',
     'E6 membaca jawaban saja', 30, 'windows, mapping, _ = batch_windows',
     'Bagian konteks pair pada screenshot tidak diterapkan di E6. Perhatikan daftar self.texts: isinya teks jawaban. Jangan menjelaskan E6 sebagai eksperimen konteks pair E4/E8.', 'Metode berbeda'),
    ('Dataset PyTorch / X dan y', 'Dataset menyiapkan teks dan skor untuk model',
     'Bundle menyimpan data; Engine menyiapkan tensor', 14, 'class Bundle',
     'Perannya tersebar: bundle.texts adalah masukan teks, bundle.y adalah target skor guru. Engine mengambil indeks jawaban dan menyiapkan tensor. Tidak harus ada variabel bernama X.', 'Peran tersebar'),
    ('Inisialisasi model regresi pretrained', 'AutoModelForSequenceClassification(..., num_labels=1)',
     'Encoder dan head dimuat terpisah', 21, 'AutoModel.from_pretrained(',
     'E6 memuat backbone melalui fungsi encoder(), lalu menyusun head sendiri di SingleScoreModel. AutoModel memuat encoder; nn.Linear / nn.Sequential membentuk head.', 'Peran setara'),
    ('Context pooling / CLS token pooler', 'ContextPooler: Linear → GELU → Dropout',
     'CLS per window → rata-rata per jawaban', 20, 'def mean_by_essay(',
     'Pooling E6 berbeda dari ContextPooler mDeBERTa pada screenshot. E6 mengambil CLS dari setiap window, lalu merata-ratakannya untuk satu jawaban.', 'Metode berbeda'),
    ('Regression head / classifier', 'classifier: Linear 768 → 1',
     'S1 linear; S2 dan S3 memakai MLP', 30, 'self.head = nn.Linear(768, 1)',
     'Head berada setelah encoder/fitur. S1: 768 → 1. S2/S3: 1024 → 256 → GELU → dropout → 1. MLP tetap head regresi, bukan pengganti encoder.', 'Lihat tiap skenario'),
    ('Ekstraksi vektor embedding awal', 'last_hidden_state[:, 0, :] lalu (300, 768)',
     'Fitur sebelum dan sesudah training dipisahkan', 24, 'def roberta_pretrained_vectors(',
     'roberta_pretrained_vectors mengekstrak fitur awal. SingleScoreModel.forward menghasilkan fitur yang dipakai model. export_final_features membaca fitur checkpoint final. 768 adalah dimensi, bukan jumlah kata.', 'Peran setara'),
    ('Pelatihan 5-fold', 'for fold_idx ... model baru ... train/validation',
     'train_all mengatur; fit melatih satu model', 38, 'def train_all(',
     'Pembagian data dibuat lebih dahulu. train_all mengatur CV/refit/test, sedangkan fit berisi pembaruan bobot. E6 memakai ulang S1/S2 yang selesai dan melatih S3 baru.', 'Peran setara'),
    ('Loss dan optimizer', 'MSELoss / AdamW / backward / optimizer.step',
     'MSE → backward → AdamW → scheduler', 36, 'loss = torch.mean(',
     'Target guru dinormalisasi ke 0–1 untuk loss MSE. Backward menghitung gradien; optimizer memperbarui bobot. RMSE validasi memakai kembali poin 0–100.', 'Peran setara'),
    ('Prediksi & evaluasi', 'Nilai rater dibanding keluaran model',
     'Skor guru, prediksi mentah, dan pembulatan', 39, 'def test_score_comparison(',
     'Nilai guru adalah acuan. RMSE/Pearson memakai prediksi mentah. QWK/AC2 memakai salinan kategori; pembulatan tidak menggantikan prediksi mentah.', 'Peran setara'),
]

EXTRA = {
    'Wadah data: bundle': 'Bundle adalah wadah, bukan encoder. rows menyimpan baris data, y menyimpan skor guru, groups menjaga pemisahan kelompok. Properti texts menyiapkan daftar teks yang dibaca model.',
    'Target atau y': 'Kolom skor guru dibaca ke array angka. Pada pembelajaran supervised, angka inilah target yang dibandingkan dengan prediksi. Nilai guru tidak dimasukkan sebagai fitur teks.',
    'Head regresi linear S1': 'Inilah padanan classifier pada screenshot. Linear(768, 1) menerima 768 fitur dan menghasilkan satu angka. Baris ini membentuk head; head dijalankan di forward(). S1 E6 memakai model lama yang diverifikasi.',
    'Head regresi MLP S2': 'Ini head S2: fitur BGE 1024 → hidden 256 → GELU → dropout → satu angka. GELU membuat pemetaan nonlinier. Encoder BGE frozen, tetapi parameter head boleh belajar pada run sumber S2.',
    'Head regresi MLP S3': 'Ini head S3, setelah concat. MLP menerima 1024 fitur gabungan, membentuk hidden 256 lalu menghasilkan satu angka. Lapisan terakhirnya tetap linear; keseluruhan head adalah MLP regresi.',
    'LayerNorm RoBERTa S3': 'LayerNorm tambahan ini bekerja pada 768 komponen vektor jawaban sebelum fusion. Jumlah komponen tetap 768. Ini lapisan di luar encoder, berbeda dari LayerNorm yang sudah ada di dalam Transformer.',
    'Proyeksi BGE S3': 'Linear(1024, 256) memadatkan fitur BGE. LayerNorm berikutnya menata skala 256 komponen itu. Proyeksi ikut dilatih meskipun bobot encoder BGE frozen.',
    'Fusion yang sebenarnya': 'torch.cat menyambung koordinat fitur: 768 RoBERTa + 256 proyeksi BGE = 1024. Ini penggabungan vektor, bukan penjumlahan skor S1 dan S2.',
    'Prediksi fusion S3': 'features[concat] masuk ke self.head, yaitu MLP S3 yang dibentuk di __init__. Keluaran head masih berskala latihan; Engine.predict mengembalikannya ke skala skor.',
    'Fitur contextual RoBERTa': 'self.roberta(**windows) menjalankan encoder. last_hidden_state memuat fitur semua token. [:, 0, :] memilih token pertama setiap window: seluruh window, token posisi 0, seluruh 768 komponen.',
    'Fitur per jawaban': 'Satu jawaban bisa memiliki beberapa window. mean_by_essay menggabungkan CLS window milik jawaban yang sama menjadi satu vektor. Inilah pooling yang dipakai sebelum head.',
    'Input RoBERTa jawaban saja': 'E6 memasukkan teks jawaban melalui self.texts. Tidak ada pasangan soal/referensi pada pemanggilan ini. Jadi bagian context_text pada screenshot bukan padanan metode yang dipakai E6.',
    'Keluaran pembaca dataset': 'Fungsi mengembalikan satu objek Bundle yang mengikat teks, skor dan kelompok agar indeksnya tetap cocok. Bundle kemudian dipakai saat split, ekstraksi fitur, training dan ekspor.',
    'Learning rate S3': '5e-6 berarti 0,000005. Ini langkah pembaruan encoder RoBERTa S3, setengah dari 1e-5 pada MLP awal. Angka LR bukan nilai akurasi atau skor model.',
    'Learning rate head/proyeksi': 'Head dan proyeksi adalah lapisan yang belajar memetakan fitur. E6 memberi keduanya LR 1e-4. Encoder BGE tidak mendapat LR training karena frozen.',
    'Pooling antar-window': 'Mapping menandai asal tiap window. index_add menjumlahkan vektor milik jawaban yang sama, lalu hasilnya dibagi jumlah window. Ini rata-rata CLS antar-window, bukan mean seluruh token.',
    'Loss training MSE': 'Prediction dan targets sama-sama berbentuk [batch, 1]. Selisihnya dikuadratkan lalu dirata-ratakan. Target memakai skala 0–1; loss ini bukan persentase kesalahan.',
    'Backward dan akumulasi': 'backward menghitung bagaimana parameter berkontribusi pada loss. Gradien beberapa microbatch dikumpulkan sebelum optimizer melangkah. Belum ada pembaruan bobot hanya karena backward selesai.',
    'Pemilihan epoch terbaik': 'RMSE dibandingkan dengan rekor terbaik seluruh epoch sebelumnya. Jika membaik, simpan bobot dan reset penghitung; jika tidak, tambah penghitung. Dibanding epoch terbaik, bukan hanya epoch terakhir.',
    'Menetapkan epoch refit': 'Ambil median epoch terbaik lima fold lalu bulatkan ke atas. Angka itu menjadi durasi refit pada development. Hasil test tidak digunakan memilih durasi latihan.',
    'Menjalankan reuse kontrol': 'Baris pemanggilan ini benar-benar menjalankan verifikasi dan pemakaian ulang S1/S2. Adanya kode arsitektur S1/S2 di notebook tidak berarti keduanya dilatih ulang pada E6.',
    'Fitur setelah training': 'Fungsi ini memuat checkpoint final dan membaca fitur dalam mode eval tanpa pembaruan bobot. Vektor yang diekspor berbeda tahap dari fitur pretrained sebelum training.',
}


def build():
    raw = subprocess.check_output(['git', 'show', PIN+':'+NOTEBOOK], cwd=ROOT)
    notebook = json.loads(raw)
    data = content(json.loads(SNAPSHOT.read_text()))
    cells = data['cells']
    codes = {c['index']: c for c in cells}
    for cell in cells:
        assert cell['source'] == source(notebook['cells'][cell['index']])
        # Gunakan catatan yang sudah sesuai E6; jangan menanam snapshot nilai/hasil.
        cell['notes'] = [{**n, 'body': n['body'], 'short': '', 'anchor': cell['source'].splitlines()[n['line']-1].strip()}
                         for n in cell['notes']]
        cell['chapterTitle'] = data['chapters'][cell['chapter']]['title']
        cell.pop('chapter', None)
    entries = []
    for index, anchor, name, brief, length, occurrence in ROWS:
        cell = codes[index]
        lines = cell['source'].splitlines()
        hits = [i for i, line in enumerate(lines) if anchor in line]
        start = hits[occurrence-1]
        snippet = '\n'.join(lines[start:start+length])
        if anchor == 'WORKER_SOURCE':
            snippet = lines[start].split('=', 1)[0]+'='
        assert snippet in cell['source']
        entry = dict(cell=index, line=start+1, anchor=anchor, title=name, short=brief,
                     snippet=snippet, ordinal=cell['ordinal'], chapter=cell['chapterTitle'])
        entries.append(entry)
        existing = next((n for n in cell['notes'] if n['line'] == start+1), None)
        body = EXTRA.get(name) or (existing['body'] if existing else brief)
        note = dict(line=start+1, title=name, short=brief, body=body, anchor=anchor,
                    kind='core', id=f'n{index}-{start+1}')
        if existing:
            cell['notes'][cell['notes'].index(existing)] = note
        else:
            cell['notes'].append(note)
    mappings = []
    for title, example, equivalent, index, anchor, explanation, status in MAPPINGS:
        cell = codes[index]
        line = next(i+1 for i, text in enumerate(cell['source'].splitlines()) if anchor in text)
        mappings.append(dict(title=title, example=example, equivalent=equivalent,
                             cell=index, line=line, anchor=anchor, explanation=explanation,
                             status=status, ordinal=cell['ordinal']))
        note = next((n for n in cell['notes'] if n['line'] == line), None)
        if note:
            note['screenshot'] = title
    for cell in cells:
        cell['notes'].sort(key=lambda n: n['line'])
        # Setiap catatan dimulai tepat di samping barisnya. Tinggi ruas mengikuti
        # kode atau catatan yang lebih panjang; tidak ada scroll panel terpisah.
        cell['groups'] = [dict(line=note['line'], notes=[note]) for note in cell['notes']]
        cell.pop('notes')
    entries.sort(key=lambda e: (e['cell'], e['line']))
    payload = dict(cells=cells, entries=entries, mappings=mappings,
                   source=dict(commit=PIN, path=NOTEBOOK, url=URL,
                               colab=f'https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/{PIN}/{NOTEBOOK}',
                               sha256=hashlib.sha256(raw).hexdigest()),
                   counts=dict(blocks=len(cells), lines=sum(len(c['htmlLines']) for c in cells),
                               notes=sum(len(g['notes']) for c in cells for g in c['groups']), entries=len(entries)))
    document = (TEMPLATES/'e6_annotated_code.html').read_text()
    for marker, name in [('__CSS__', 'e6_annotated_code.css'), ('__JS__', 'e6_annotated_code.js')]:
        document = document.replace(marker, (TEMPLATES/name).read_text())
    document = document.replace('__DATA__', json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c'))
    TARGET.write_text(document)
    assert document.count('id="guideData"') == 1
    print(json.dumps({'file': str(TARGET.relative_to(ROOT)), 'bytes': TARGET.stat().st_size,
                      **payload['counts'], 'source_sha256': payload['source']['sha256']}))


if __name__ == '__main__':
    build()
