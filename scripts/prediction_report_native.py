"""Prediction detail and exact/cumulative counts; reporting only, no model calls."""
import math
from pathlib import Path
from io import BytesIO
import zipfile
from openpyxl import load_workbook


def prediction_report_tables(records, score_min=0, score_max=100):
    """Keep raw scores; normalize by declared range and round ties downward."""
    records = list(records)
    if not records or not math.isfinite(score_min) or not math.isfinite(score_max) or score_min>=score_max or int(score_min)!=score_min or int(score_max)!=score_max:
        raise ValueError('Prediksi test tidak boleh kosong dan rentang skor integer harus sah.')
    thresholds = (0,1,2,3,4,5,10)
    span = score_max-score_min
    details,ids = [],set()
    for row in records:
        identifier = row['ID Jawaban']
        y,p = float(row['Nilai Guru']),float(row['Prediksi S3'])
        if identifier in ids or not math.isfinite(y) or not math.isfinite(p):
            raise ValueError('ID test duplikat atau skor/prediksi tidak finite.')
        if int(y)!=y or not score_min<=y<=score_max:
            raise ValueError('Nilai guru harus kategori integer dalam rentang skor.')
        ids.add(identifier)
        clipped = min(score_max,max(score_min,p))
        lower = math.floor(clipped)
        rounded = lower+int(clipped-lower>0.5)  # Tepat .5 ke bawah, sama dengan kontrak QWK/AC2.
        if row.get('Pembulatan S3') is not None and float(row['Pembulatan S3'])!=rounded:
            raise ValueError('Pembulatan sumber berbeda dari aturan terdekat dengan tie .5 ke bawah.')
        error = p-y
        if row.get('Selisih S3') is not None and not math.isclose(float(row['Selisih S3']),error,rel_tol=1e-9,abs_tol=1e-7):
            raise ValueError('Selisih mentah pada sumber tidak cocok dengan prediksi dan nilai guru.')
        exact = abs(error)
        integer_error = rounded-y
        absolute_integer = abs(integer_error)
        def category(value):
            if value==0:return 'Sama persis'
            if value in thresholds:return f'Beda tepat {int(value)} poin'
            return 'Selisih lain'
        details.append({'ID Jawaban':identifier,'Nilai Guru':y,'Prediksi S3':p,
            'Pembulatan S3':rounded,'Selisih S3':error,
            'Nilai Guru Normalisasi (0–1)':(y-score_min)/span,
            'Prediksi Normalisasi (0–1)':(p-score_min)/span,
            'Prediksi Bulat Normalisasi (0–1)':(rounded-score_min)/span,
            'Prediksi Dibatasi (poin)':clipped,
            'Selisih Absolut Mentah (poin)':exact,
            'Selisih Normalisasi (0–1)':error/span,
            'Selisih Absolut Normalisasi (0–1)':exact/span,
            'Selisih Bulat (poin)':integer_error,'Selisih Absolut Bulat (poin)':absolute_integer,
            'Kategori Selisih Mentah':category(exact),'Kategori Selisih Bulat':category(absolute_integer)})
    total = len(details)
    raw = [r['Selisih Absolut Mentah (poin)'] for r in details]
    rounded = [r['Selisih Absolut Bulat (poin)'] for r in details]
    def count_row(label,threshold,a,b):
        return {'Kategori':label,'Selisih (poin)':threshold,
                'Setara normalisasi (0–1)':None if threshold is None else threshold/span,
                'Jumlah Mentah':a,'Persentase Mentah (%)':100*a/total,
                'Jumlah Setelah Pembulatan':b,'Persentase Setelah Pembulatan (%)':100*b/total,
                'Jumlah Test':total}
    exact_rows = [count_row('Sama persis' if n==0 else f'Beda tepat {n} poin',n,
                           sum(v==n for v in raw),sum(v==n for v in rounded)) for n in thresholds]
    exact_rows.append(count_row('Selisih lain',None,sum(v not in thresholds for v in raw),sum(v not in thresholds for v in rounded)))
    tolerance_rows = [count_row(f'Selisih maksimal {n} poin (≤{n})',n,
                               sum(v<=n for v in raw),sum(v<=n for v in rounded)) for n in thresholds]
    # Counts are evaluated in point units; normalization is the same affine scale.
    # This avoids binary floating-point boundary artifacts when, e.g., 0.01 is represented.
    return {'Nilai Guru dan Prediksi':details,'Selisih Tepat':exact_rows,'Toleransi Selisih':tolerance_rows}


def verify_prediction_report_tables(tables, score_min=0, score_max=100):
    """Recompute from raw test columns; reject missing, stale or inconsistent counts."""
    required = ('Nilai Guru dan Prediksi','Selisih Tepat','Toleransi Selisih')
    if any(name not in tables or not tables[name] for name in required):
        raise ValueError('Detail prediksi/ringkasan selisih kosong atau tidak tersedia.')
    expected = prediction_report_tables(tables[required[0]],score_min,score_max)
    for name in required:
        if len(tables[name])!=len(expected[name]):
            raise ValueError('Jumlah baris ringkasan/detail selisih tidak cocok: '+name)
        for actual,wanted in zip(tables[name],expected[name]):
            for key,value in wanted.items():
                if key not in actual:
                    raise ValueError('Kolom detail/selisih hilang: '+key)
                observed = actual[key]
                equal = math.isclose(float(observed),value,rel_tol=1e-9,abs_tol=1e-10) if isinstance(value,(int,float)) and isinstance(observed,(int,float)) else observed==value
                if not equal:
                    raise ValueError('Angka detail/ringkasan selisih tidak cocok: '+name+' / '+key)
    return {'test_rows':len(expected[required[0]]),'exact_categories':7,'tolerance_thresholds':7,
            'normalized_unit_matches_points':True,'counts_verified':True}


def update_prediction_zip(source_zip, destination=None):
    """Update a completed standard result ZIP without loading data/models or training."""
    source_zip = Path(source_zip).expanduser().resolve()
    destination = Path(destination).expanduser().resolve() if destination else source_zip.with_name(source_zip.stem+'_Prediksi_Lengkap.zip')
    if source_zip==destination:
        raise ValueError('Gunakan nama tujuan berbeda agar ZIP sumber tetap utuh.')
    expected_names = {'01_Data_dan_Tokenisasi.xlsx','02_Vektor_Jawaban.xlsx','03_Vektor_Kata_dan_Kalimat.xlsx',
                      '04_Prediksi_dan_Evaluasi.xlsx','05_Riwayat_Pelatihan.xlsx'}
    with zipfile.ZipFile(source_zip) as archive:
        if len(archive.namelist())!=5 or set(archive.namelist())!=expected_names:
            raise ValueError('Pilih ZIP hasil standar berisi tepat lima Excel.')
        settings_book = load_workbook(BytesIO(archive.read('05_Riwayat_Pelatihan.xlsx')),read_only=True,data_only=True)
        try:
            iterator=iter(settings_book['Pengaturan Eksperimen'].values);header=next(iterator)
            settings={r[0]:r[1] for r in iterator}
            score_min,score_max=settings['Skor minimum'],settings['Skor maksimum']
        finally:settings_book.close()
        file='04_Prediksi_dan_Evaluasi.xlsx'
        workbook=load_workbook(BytesIO(archive.read(file)))
        try:
            # Completed E46/E47 exports already have these valid tables; refuse partial exports.
            for name in ['Perbandingan Model','Ringkasan Validasi','Nilai Guru dan Prediksi']:
                if name not in workbook or workbook[name].max_row<2:
                    raise ValueError('Pilih hasil lengkap E46/E47; tabel '+name+' kosong/tidak tersedia.')
            iterator=iter(workbook['Nilai Guru dan Prediksi'].values);header=next(iterator)
            rows=[dict(zip(header,r)) for r in iterator]
            tables=prediction_report_tables(rows,score_min,score_max)
            proof=verify_prediction_report_tables(tables,score_min,score_max)
            from openpyxl.styles import Font,PatternFill,Alignment
            for name,data in tables.items():
                if name in workbook:
                    index=workbook.sheetnames.index(name);workbook.remove(workbook[name]);sheet=workbook.create_sheet(name,index)
                else:sheet=workbook.create_sheet(name)
                sheet.append(list(data[0]))
                for row in data:sheet.append(list(row.values()))
                sheet.freeze_panes='A2';sheet.auto_filter.ref=sheet.dimensions
                for cell in sheet[1]:
                    cell.font=Font(bold=True,color='FFFFFF');cell.fill=PatternFill('solid',fgColor='243C5A')
                    cell.alignment=Alignment(wrap_text=True,vertical='top')
                for column in sheet.columns:
                    sheet.column_dimensions[column[0].column_letter].width=26
                for row in sheet.iter_rows(min_row=2):
                    for cell in row:
                        if isinstance(cell.value,float):cell.number_format='0.000000'
            if 'Panduan' in workbook:
                guide=workbook['Panduan']
                note='Selisih Tepat memakai =n; Toleransi Selisih memakai ≤n. Normalisasi berbeda dari pembulatan; 1 poin=0,01 pada skala0–1. Ringkasan hanya data test.'
                existing=next((r[0].row for r in guide.iter_rows() if r[0].value=='Rincian selisih'),None)
                if existing:guide.cell(existing,2,note)
                else:guide.append(['Rincian selisih',note])
            buffer=BytesIO();workbook.save(buffer)
        finally:workbook.close()
        destination.parent.mkdir(parents=True,exist_ok=True)
        pending=destination.with_name(destination.name+'.pending')
        try:
            with zipfile.ZipFile(pending,'w',zipfile.ZIP_DEFLATED) as target:
                for name in archive.namelist():target.writestr(name,buffer.getvalue() if name==file else archive.read(name))
            # Verify serialized workbook contents, not only the values held before saving.
            with zipfile.ZipFile(pending) as check:
                saved=load_workbook(BytesIO(check.read(file)),read_only=True,data_only=True)
                try:
                    serialized={}
                    for name in tables:
                        iterator=iter(saved[name].values);header=next(iterator)
                        serialized[name]=[dict(zip(header,r)) for r in iterator]
                    verify_prediction_report_tables(serialized,score_min,score_max)
                finally:saved.close()
            pending.replace(destination)
        except BaseException:
            if pending.exists():pending.unlink()
            raise
    return {'archive':str(destination),'source_unchanged':True,'training_run':False,'files':5,
            'prediction_verification':proof}
