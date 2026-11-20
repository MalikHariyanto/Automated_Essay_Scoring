"""Tabel belajar di Colab; hanya mengatur penyajian hasil yang sudah dihitung."""
import html
import itertools
import math
import re
from pathlib import Path

from openpyxl import load_workbook

from .presentation import REPEATED_COLUMNS, friendly_header, friendly_value, repeated_word_rows


REPEATED_HEADERS = REPEATED_COLUMNS
REPEATED_NOTES = [
    'Satu baris membandingkan kata yang sama pada dua posisi di jawaban contoh. Posisi dihitung mulai 1.',
    'Sebelum konteks = vektor dari tabel kosakata model. Setelah konteks = vektor setelah model membaca posisi dan kata di sekitarnya.',
    'Cosine mengukur kemiripan arah vektor: mendekati 1 berarti sangat mirip. Nilai yang terlihat 1 setelah pembulatan belum membuktikan seluruh angka vektornya sama.',
    'Kolom Vektor ... Sama? memeriksa selisih setiap komponen, dengan batas 0,000001. Ini batas perbandingan angka komputer, bukan tolerance skor AES.',
]
METRIC_NOTES = [
    'S1 = RoBERTa; S2 = BGE-M3; S3 = fusion keduanya. RMSE memakai satuan poin skor: lebih kecil berarti kesalahan lebih kecil.',
    'Pearson menunjukkan hubungan perubahan skor; QWK dan Gwet AC2 menunjukkan kesepakatan skor kategori. Korelasi tinggi saja belum menjamin prediksi mendekati nilai guru.',
    'Tanda — berarti hasil tidak tersedia, bukan nol. Baca keempat metrik bersama-sama.',
]


def preview_table(headers, records, title='', row_count=None, max_rows=8, max_columns=8):
    """Bentuk preview tanpa mengubah DataFrame/Excel sumber atau menghitung ulang model."""
    headers=list(headers)
    records=list(itertools.islice(records,max_rows))
    count=len(records) if row_count is None else row_count
    notes=[]
    original_columns=len(headers)
    if 'occurrence_indices' in headers:
        values=list(repeated_word_rows(records,''))
        # Nama model hanya ditampilkan bila memang diberikan oleh pemanggil.
        headers=REPEATED_HEADERS[1:]
        values=[r[1:] for r in values]
        notes=REPEATED_NOTES
        empty='Belum ada pasangan kata berulang yang bisa dibandingkan. Ini bukan error; lihat tabel vektor kata di atas.'
    elif {'file','stage','sheets_and_shapes'}.issubset(headers):
        # Dictionary ukuran tabel adalah rincian internal, bukan isi laporan belajar.
        headers=['File Hasil','Isi File','Keadaan']
        values=[[r.get('file'),r.get('stage'),friendly_value('status',r.get('status'))] for r in records]
        notes=['Daftar ini menunjukkan file yang sudah dibuat. Pilih lima laporan utama pada bagian unduhan untuk membaca hasilnya.']
        empty='Belum ada file hasil yang tersimpan.'
    else:
        selected=headers[:max_columns]
        values=[[friendly_value(key,row.get(key)) for key in selected] for row in records]
        headers=[friendly_header(key) for key in selected]
        empty='Belum ada baris hasil pada tahap ini. Tidak ada angka pengganti yang ditampilkan.'
        if 'Vektor Setelah Konteks Sama?' in headers or 'Vektor Akhir Sama' in headers:
            notes=REPEATED_NOTES
            empty='Belum ada pasangan kata berulang yang bisa dibandingkan. Ini bukan error; lihat tabel vektor kata di atas.'
        elif {'RMSE','Pearson','QWK','AC2'}.issubset(selected) or 'RMSE (poin)' in headers:
            notes=METRIC_NOTES
        elif 'prediksi_mentah' in selected or 'Prediksi S1' in headers:
            notes=['Prediksi Skor adalah nilai yang diberikan model. Nilai Guru adalah acuan jika tersedia.',
                   'Skor Dibulatkan adalah salinan untuk metrik kategori; prediksi asli tetap disimpan. Di Luar Rentang berarti prediksi melewati batas skor yang ditetapkan.']
        elif 'token_id' in selected or 'ID Token' in headers:
            notes=['Token adalah potongan teks yang dibaca model; satu kata bisa terpecah menjadi beberapa token. ID Token adalah nomor dalam kosakata model, bukan nilai siswa.',
                   'Posisi Token dan Bagian Teks dihitung mulai 1. Token khusus seperti <s> dan </s> menandai batas input.']
        components=sum(bool(re.fullmatch(r'v_\d+|Komponen \d+',str(h))) for h in selected)
        if components:
            notes=['Satu baris berisi satu vektor. Komponen 1, 2, dan seterusnya adalah angka penyusun vektor, bukan skor penilaian.',
                   'Tabel ini hanya memperlihatkan beberapa komponen pertama. Seluruh komponen tersimpan di Excel. Komponen yang sama nomornya pada RoBERTa dan BGE-M3 tidak memiliki arti yang otomatis sama.']
    return {'title':title,'columns':headers,'rows':values,'shape':[count,original_columns],
            'notes':notes,'empty_message':empty}


def cell_text(value):
    if value is None:return '—'
    if isinstance(value,bool):return 'Ya' if value else 'Tidak'
    if isinstance(value,float):
        return format(value,'.8g') if math.isfinite(value) else '—'
    if isinstance(value,dict):
        return '; '.join(f'{friendly_header(k)}: {cell_text(v)}' for k,v in value.items()) or '—'
    if isinstance(value,(list,tuple)):
        return ', '.join(cell_text(x) for x in value) or '—'
    return str(value)


def render_table(event):
    """HTML escapé dengan warna eksplisit agar terbaca pada Colab terang maupun gelap."""
    esc=lambda value:html.escape(cell_text(value))
    parts=['<section style="background:#f8fafc!important;color:#172035!important;padding:18px;border:1px solid #94a3b8;border-radius:12px;margin:12px 0">',
           '<h3 style="color:#172035!important;margin:0 0 10px">'+esc(event.get('title',''))+'</h3>']
    for note in event.get('notes',[]):parts.append('<p style="color:#334155!important;margin:6px 0">'+esc(note)+'</p>')
    if not event['rows']:
        parts.append('<p>'+esc(event.get('empty_message','Belum ada hasil.'))+'</p>')
    else:
        parts.append(f'<p style="color:#475569!important">Ditampilkan {len(event["rows"])} dari {event["shape"][0]} baris. Angka pada layar diringkas; nilai lengkap ada di Excel.</p>')
        th='background:#172035!important;color:#fff!important;padding:10px;border:1px solid #64748b;text-align:left'
        td='background:#f8fafc!important;color:#172035!important;padding:10px;border:1px solid #cbd5e1;vertical-align:top;min-width:90px;max-width:460px;white-space:normal;overflow-wrap:anywhere'
        parts.append('<div style="max-height:420px;overflow:auto"><table style="border-collapse:collapse"><thead><tr>')
        parts.extend('<th style="'+th+'">'+esc(x)+'</th>' for x in event['columns'])
        parts.append('</tr></thead><tbody>')
        for row in event['rows']:
            parts.append('<tr>'+''.join('<td style="'+td+'">'+esc(x)+'</td>' for x in row)+'</tr>')
        parts.append('</tbody></table></div>')
    parts.append('</section>')
    return ''.join(parts)


def saved_report_previews(folder):
    """Baca cuplikan lima Excel yang telah dirapikan; tidak memerlukan pandas/torch/model."""
    choices=[('01_Data_dan_Tokenisasi.xlsx','Token RoBERTa','Contoh token RoBERTa',8,6),
             ('02_Vektor_Jawaban.xlsx','RoBERTa Sebelum Latihan','Contoh vektor jawaban',5,6),
             ('03_Vektor_Kata_dan_Kalimat.xlsx','Kata Berulang','Kata yang sama: apakah vektornya sama?',8,10),
             ('04_Prediksi_dan_Evaluasi.xlsx','Perbandingan Model','Hasil evaluasi tiga model',3,8),
             ('04_Prediksi_dan_Evaluasi.xlsx','Nilai Guru dan Prediksi','Nilai guru dibandingkan prediksi model',8,8)]
    for filename,sheet,title,nrows,ncols in choices:
        path=Path(folder)/filename
        if not path.is_file():continue
        book=load_workbook(path,read_only=True,data_only=True)
        try:
            if sheet not in book.sheetnames:continue
            iterator=book[sheet].iter_rows(values_only=True)
            headers=next(iterator,())
            values=[];count=0
            for row in iterator:
                count+=1
                if count<=nrows:values.append(dict(zip(headers,row)))
            event=preview_table(headers,values,title,row_count=count,max_rows=nrows,max_columns=ncols)
            event['report']=filename+' → '+sheet
            yield event
        finally:book.close()
