"""Reporting-only tests; all scores below are synthetic technical examples."""
import ast
import copy
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile,ZIP_DEFLATED

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from openpyxl import Workbook,load_workbook
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from scripts.prediction_report_native import prediction_report_tables,verify_prediction_report_tables,update_prediction_zip
from test_e46_e47_notebooks import SPECS
from test_e41_and_e30_native import runtime,setup,definitions

UPDATER=ROOT/'notebooks/05_Perbarui_Prediksi_Hasil_Tanpa_Training.ipynb'
ORIGINAL='2181cfc29cb753b5505b28a7434ddbbe25b1bf75'


def technical_records():
    return [{'ID Jawaban':f'TECHNICAL-{i}','Nilai Guru':50,'Prediksi S3':50+delta}
            for i,delta in enumerate([0,1,2,3,4,5,10,6,.49,.5,.51])]


def technical_zip(path):
    members={}
    for name in ['01_Data_dan_Tokenisasi.xlsx','02_Vektor_Jawaban.xlsx','03_Vektor_Kata_dan_Kalimat.xlsx',
                 '04_Prediksi_dan_Evaluasi.xlsx','05_Riwayat_Pelatihan.xlsx']:
        book=Workbook();sheet=book.active;sheet.title='Technical Sentinel';sheet.append(['Not research','unchanged'])
        if name.startswith('04'):
            for title in ['Perbandingan Model','Ringkasan Validasi']:
                sheet=book.create_sheet(title);sheet.append(['TECHNICAL','value']);sheet.append(['preserve',9.125])
            sheet=book.create_sheet('Nilai Guru dan Prediksi');data=prediction_report_tables(technical_records())['Nilai Guru dan Prediksi']
            keys=list(data[0])[:5];sheet.append(keys)
            for row in data:sheet.append([row[k] for k in keys])
            sheet=book.create_sheet('Panduan');sheet.append(['Topic','Description'])
        if name.startswith('05'):
            sheet=book.create_sheet('Pengaturan Eksperimen');sheet.append(['Setting','Value'])
            sheet.append(['Skor minimum',0]);sheet.append(['Skor maksimum',100])
        stream=BytesIO();book.save(stream);book.close();members[name]=stream.getvalue()
    with ZipFile(path,'w',ZIP_DEFLATED) as archive:
        for name,data in members.items():archive.writestr(name,data)
    return members


def workbook_tables(data):
    book=load_workbook(BytesIO(data),read_only=True,data_only=True)
    try:
        tables={}
        for title in ['Nilai Guru dan Prediksi','Selisih Tepat','Toleransi Selisih']:
            rows=iter(book[title].values);header=next(rows);tables[title]=[dict(zip(header,r)) for r in rows]
        return tables
    finally:book.close()


def test_exact_is_distinct_from_cumulative_and_normalization():
    tables=prediction_report_tables(technical_records());details=tables['Nilai Guru dan Prediksi']
    exact=tables['Selisih Tepat'];tolerance=tables['Toleransi Selisih']
    assert [r['Jumlah Mentah'] for r in exact]==[1,1,1,1,1,1,1,4]
    assert [r['Jumlah Setelah Pembulatan'] for r in exact]==[3,2,1,1,1,1,1,1]
    assert [r['Jumlah Mentah'] for r in tolerance]==[1,5,6,7,8,9,11]
    assert [r['Jumlah Setelah Pembulatan'] for r in tolerance]==[3,5,6,7,8,9,11]
    assert exact[5]['Jumlah Mentah']==1 and tolerance[5]['Jumlah Mentah']==9
    assert sum(r['Jumlah Mentah'] for r in exact)==sum(r['Jumlah Setelah Pembulatan'] for r in exact)==11
    assert sum(r['Persentase Mentah (%)'] for r in exact)==pytest.approx(100)
    assert details[1]['Prediksi Normalisasi (0–1)']==.51
    assert details[1]['Selisih Absolut Normalisasi (0–1)']==.01
    assert exact[1]['Setara normalisasi (0–1)']==.01
    assert details[9]['Prediksi S3']==50.5 and details[9]['Pembulatan S3']==50
    assert details[10]['Pembulatan S3']==51
    assert verify_prediction_report_tables(tables)['counts_verified']


def test_clipping_changes_only_rounded_copy_and_affine_range():
    source=[{'ID Jawaban':'TECH-A','Nilai Guru':0,'Prediksi S3':-4.2},
            {'ID Jawaban':'TECH-B','Nilai Guru':100,'Prediksi S3':105.5}]
    rows=prediction_report_tables(source)['Nilai Guru dan Prediksi']
    assert [r['Prediksi S3'] for r in rows]==[-4.2,105.5]
    assert [r['Pembulatan S3'] for r in rows]==[0,100]
    assert rows[1]['Prediksi Normalisasi (0–1)']==1.055
    assert rows[1]['Selisih Absolut Mentah (poin)']==5.5
    assert source[0]['Prediksi S3']==-4.2  # input never mutated
    custom=prediction_report_tables([{'ID Jawaban':'TECH-C','Nilai Guru':10,'Prediksi S3':11}],10,110)
    assert custom['Nilai Guru dan Prediksi'][0]['Prediksi Normalisasi (0–1)']==.01


@pytest.mark.parametrize('change',[
    lambda rows:rows.append(copy.deepcopy(rows[0])),
    lambda rows:rows[0].update({'Prediksi S3':float('nan')}),
    lambda rows:rows[0].update({'Nilai Guru':50.5}),
    lambda rows:rows[0].update({'Pembulatan S3':49}),
    lambda rows:rows[0].update({'Selisih S3':1}),
])
def test_bad_inputs_rejected(change):
    rows=technical_records();change(rows)
    with pytest.raises(ValueError):prediction_report_tables(rows)


@pytest.mark.parametrize('sheet,key',[
    ('Selisih Tepat','Jumlah Mentah'),('Toleransi Selisih','Jumlah Setelah Pembulatan'),
    ('Nilai Guru dan Prediksi','Prediksi Normalisasi (0–1)')])
def test_serialized_verifier_rejects_changed_values(sheet,key):
    tables=prediction_report_tables(technical_records());tables[sheet][0][key]+=1
    with pytest.raises(ValueError,match='tidak cocok'):verify_prediction_report_tables(tables)


def test_zip_update_preserves_source_metrics_and_four_other_files(tmp_path):
    source=tmp_path/'TECHNICAL_complete.zip';members=technical_zip(source);sha=hashlib.sha256(source.read_bytes()).hexdigest()
    result=update_prediction_zip(source);assert not result['training_run'] and result['files']==5
    assert hashlib.sha256(source.read_bytes()).hexdigest()==sha
    name='04_Prediksi_dan_Evaluasi.xlsx'
    with ZipFile(result['archive']) as archive:
        assert len(archive.namelist())==5
        assert all(archive.read(k)==v for k,v in members.items() if k!=name)
        first_tables=workbook_tables(archive.read(name))
        assert verify_prediction_report_tables(first_tables)['test_rows']==11
        old=load_workbook(BytesIO(members[name]));new=load_workbook(BytesIO(archive.read(name)))
        for title in ['Technical Sentinel','Perbandingan Model','Ringkasan Validasi']:
            assert list(old[title].values)==list(new[title].values)
        assert new['Nilai Guru dan Prediksi'].max_column==16
        old.close();new.close()
    second=update_prediction_zip(result['archive'])
    with ZipFile(second['archive']) as archive:
        assert workbook_tables(archive.read(name))==first_tables
        book=load_workbook(BytesIO(archive.read(name)))
        assert sum(r[0]=='Rincian selisih' for r in book['Panduan'].values)==1
        book.close()
    with pytest.raises(ValueError,match='nama tujuan berbeda'):update_prediction_zip(source,source)
    broken=tmp_path/'TECHNICAL_broken.zip'
    w=load_workbook(BytesIO(members[name]));w['Ringkasan Validasi'].delete_rows(2);out=BytesIO();w.save(out);w.close()
    with ZipFile(broken,'w') as archive:
        for key,data in members.items():archive.writestr(key,out.getvalue() if key==name else data)
    with pytest.raises(ValueError,match='kosong'):update_prediction_zip(broken)


@pytest.mark.parametrize('number',[46,47])
def test_training_definitions_and_cache_identity_unchanged(tmp_path,number):
    s=SPECS[number];relative=s['path'].relative_to(ROOT)
    old=tmp_path/'original.ipynb';old.write_bytes(subprocess.check_output(['git','show',ORIGINAL+':'+str(relative)],cwd=ROOT))
    before=definitions(old);after=definitions(s['path'])
    for name in before:
        if name not in ['run_folder','evaluation_report','validate_refit_download']:
            assert ast.dump(before[name])==ast.dump(after[name]),name
    first=runtime(old,s['role']);second=runtime(s['path'],s['role']);cfg,bundle=setup(first,tmp_path)
    assert second['config'](cfg)==cfg
    assert first['run_folder'](cfg,bundle)==second['run_folder'](cfg,bundle)
    assert second['TRAINING_CODE_SHA256']==first['NOTEBOOK_CODE_SHA256']


def test_updater_deterministic_and_fresh_kernel(tmp_path):
    before=UPDATER.read_bytes();subprocess.run([sys.executable,'scripts/build_prediction_updater_notebook.py'],cwd=ROOT,check=True,capture_output=True)
    assert UPDATER.read_bytes()==before==(ROOT/'aes-learning/notebooks'/UPDATER.name).read_bytes()
    notebook=nbformat.read(UPDATER,as_version=4);nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type=='code':
            assert not cell.outputs and cell.execution_count is None
            assert not any(line.lstrip().startswith(('%','!')) for line in cell.source.splitlines())
    source=tmp_path/'TECHNICAL_complete.zip';technical_zip(source)
    kernels=tmp_path/'kernels';folder=kernels/'prediction_fixture';folder.mkdir(parents=True)
    (folder/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-I','-m','ipykernel_launcher','-f','{connection_file}'],
        'display_name':'Prediction report technical fixture','language':'python'}))
    manager=KernelManager(kernel_name='prediction_fixture',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
    try:
        NotebookClient(notebook,timeout=120,km=manager,resources={'metadata':{'path':str(tmp_path)}}).execute(
            env={**os.environ,'AES_RESULT_ZIP':str(source),'AES_RESULT_OUTPUT':str(tmp_path/'new_output')})
    finally:
        if manager.has_kernel:manager.shutdown_kernel(now=True)
    output=tmp_path/'new_output/TECHNICAL_complete_Prediksi_Lengkap.zip'
    with ZipFile(output) as archive:
        assert verify_prediction_report_tables(workbook_tables(archive.read('04_Prediksi_dan_Evaluasi.xlsx')))['test_rows']==11
