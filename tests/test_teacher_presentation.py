"""Penyajian ulang fixture; angka penelitian tidak dibuat atau diubah di sini."""
import hashlib
import ast
import logging
import json
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
import zipfile
import copy

import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook

from aes.teacher_score.excel import write_excel,vector_frame
from aes.teacher_score.models import ExpectedPoolerNotice
from aes.teacher_score.presentation import build_reports,repack_archive
from aes.teacher_score.display import preview_table,render_table,saved_report_previews


@pytest.fixture
def saved_outputs(tmp_path):
    folder=tmp_path/'old';folder.mkdir()
    metadata={'evidence_scope':'TECHNICAL FIXTURE ONLY; NOT STUDENT SCORES'}
    def save(name,sheets):write_excel(folder/(name+'.xlsx'),sheets,metadata)
    ids=['row_0001','row_0002']
    save('01_konfigurasi',{'Parameter':pd.DataFrame([{'name':'epochs','value':5},{'name':'folds','value':5}])})
    save('02_dataset_dan_cleaning',{'Sebelum_sesudah':pd.DataFrame({'row_id':ids,'teks_model':['=literal fixture','synthetic second answer'],'skor_guru':[80,90]})})
    save('03_split_dan_cv',{'Split_setiap_baris':pd.DataFrame({'row_id':ids,'role':['test','development']}),
                           'Fold_setiap_baris':pd.DataFrame({'row_id':ids,'fold':[1,1],'role':['validation','train']})})
    token=pd.DataFrame({'row_id':['row_0001','row_0001'],'position':[0,1],'window':[0,0],
                        'token':['<s>','<pad>'],'token_id':[0,1],'attention_mask':[1,0]})
    save('04_tokenisasi',{'RoBERTa_input':token,'BGE_input':token.drop(columns='window')})
    roberta=np.arange(1536,dtype=float).reshape(2,768)/1000
    bge=np.arange(2048,dtype=float).reshape(2,1024)/1000
    save('05_roberta_pretrained',{'Vektor_jawaban_768':vector_frame(roberta,ids)})
    save('06_bge_frozen',{'Vektor_jawaban_1024':vector_frame(bge,ids)})
    for scenario in ['S1','S3']:
        save(f'14_{scenario}_fitur_setelah_training',{'roberta':vector_frame(roberta+.1,ids),
            'concat':vector_frame(bge+.2,ids),'dropout_eval':vector_frame(roberta,ids)})
    base={}
    for role,dim in [('roberta',768),('bge',1024)]:
        word=vector_frame(np.ones((2,dim)),[0,1]);word.insert(1,'word',['alpha','alpha']);word.insert(2,'covered',[True,True])
        sentence=vector_frame(np.zeros((1,dim)),[0]);sentence.insert(1,'text',['alpha alpha.'])
        for name in ['word_initial','word_contextual']:base[role+'_'+name]=word
        base[role+'_sentence_cls']=sentence
        base[role+'_kata_berulang']=pd.DataFrame([{'word':'alpha','occurrence_indices':[0,1],
            'same_token_ids':True,'initial_max_abs_difference':0.,'contextual_allclose_1e-6':False,'contextual_cosine':.9}])
    save('07_vektor_kata_kalimat_pretrained',base)
    for s in ['S1','S3']:save(f'15_{s}_vektor_kata_setelah_training',base)
    evaluations=[];cv=[]
    for index,scenario in enumerate(['S1','S2','S3']):
        predictions=np.array([81.125+index,88.375-index])
        save(f'11_{scenario}_test',{'Prediksi_test':pd.DataFrame({'row_id':ids,'skor_guru':[80,90],
            'prediksi_mentah':predictions,'selisih_mentah':predictions-np.array([80,90])})})
        evaluations.append({'Skenario':scenario,'QWK':.7+index/100,'AC2':.8,'RMSE':2.125+index,
                            'Pearson':.9,'out_of_range_fraction':.1,'undefined_reasons':{}})
        for fold in range(1,6):
            cv.append({'Skenario':scenario,'Fold':fold,'QWK':.75,'AC2':.8,'RMSE':2.5,'Pearson':.85})
            history=pd.DataFrame([{'epoch':1,'train_mse_scaled':.1,'RMSE':4.,'QWK':.4},
                                  {'epoch':2,'train_mse_scaled':.05,'RMSE':3.,'QWK':.6}])
            save(f'09_{scenario}_fold_{fold}',{'Riwayat_epoch':history})
        save(f'10_{scenario}_refit',{'Riwayat_epoch':pd.DataFrame([{'epoch':1,'train_mse_scaled':.04}])})
    means=pd.DataFrame([{'Skenario':s,**{k+'_'+a:v for k in ['QWK','AC2','RMSE','Pearson'] for a,v in [('mean',.75),('std',.03),('count',5)]}} for s in ['S1','S2','S3']])
    save('12_evaluasi_tiga_skenario',{'TEST_final':pd.DataFrame(evaluations),'CV_semua_fold':pd.DataFrame(cv),'CV_mean_SD_n':means})
    save('13_prediksi_satu_jawaban',{'Input_privat':pd.DataFrame([{'text':'alpha alpha.'}]),
                                   'Tiga_skenario':pd.DataFrame([{'Skenario':s,'prediksi_mentah':80.125,'kategori_salinan':80,'di_luar_rentang':False} for s in ['S1','S2','S3']])})
    return folder,roberta,bge


def test_five_reports_copy_numbers_and_combine_folds(saved_outputs,tmp_path):
    source,roberta,bge=saved_outputs
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.xlsx')}
    result=build_reports(source,tmp_path/'new')
    assert len(result['files'])==5 and not result['training_run'] and not result['metrics_recomputed']
    with zipfile.ZipFile(result['archive']) as bundle:assert len(bundle.namelist())==5
    folder=Path(result['archive']).parent
    for path in result['files']:
        book=load_workbook(path,read_only=True,data_only=False)
        assert book.sheetnames[0]=='Panduan'
        assert 'TECHNICAL FIXTURE' in str(list(book['Panduan'].values)[1][1])
        assert not any('dropout' in name or 'GELU' in name or 'normalized' in name for name in book.sheetnames)
        book.close()
    copied=pd.read_excel(folder/'02_Vektor_Jawaban.xlsx',sheet_name='RoBERTa Sebelum Latihan')
    assert copied.columns[0]=='ID Jawaban' and copied.iloc[0,0]=='J001'
    assert copied.columns[-1]=='Komponen 768'
    np.testing.assert_allclose(copied.iloc[:,1:].to_numpy(),roberta,atol=0,rtol=1e-14)
    final=pd.read_excel(folder/'04_Prediksi_dan_Evaluasi.xlsx',sheet_name='Perbandingan Model')
    assert final['RMSE (poin)'].tolist()==[2.125,3.125,4.125]
    assert final.QWK.tolist()==[.7,.71,.72]
    pairs=pd.read_excel(folder/'04_Prediksi_dan_Evaluasi.xlsx',sheet_name='Nilai Guru dan Prediksi')
    assert pairs['Prediksi S1'].tolist()==[81.125,88.375]
    assert pairs['Selisih S1'].tolist()==[1.125,-1.625]
    history=pd.read_excel(folder/'05_Riwayat_Pelatihan.xlsx',sheet_name='RoBERTa (S1)')
    assert len(history)==11 and set(history.Tahap)=={'Validasi silang','Latihan final'}
    words=pd.read_excel(folder/'03_Vektor_Kata_dan_Kalimat.xlsx',sheet_name='Kata RoBERTa Sebelum')
    assert words.columns[0]=='Urutan Kata' and words.iloc[:,0].tolist()==[1,2]
    assert 'Komponen 768' in words.columns
    tokens=pd.read_excel(folder/'01_Data_dan_Tokenisasi.xlsx',sheet_name='Token RoBERTa')
    assert len(tokens)==1 and tokens['Token'].tolist()==['<s>']
    data=load_workbook(folder/'01_Data_dan_Tokenisasi.xlsx',read_only=True,data_only=False)
    assert data['Data dan Nilai Guru']['B2'].data_type=='s'
    data.close()
    assert hashes=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.xlsx')}


def test_archive_converter_and_partial_results(saved_outputs,tmp_path):
    source,_,_=saved_outputs
    archive=tmp_path/'old.zip'
    with zipfile.ZipFile(archive,'w') as bundle:
        for p in source.glob('*.xlsx'):bundle.write(p,arcname='nested/'+p.name)
        bundle.writestr('../irrelevant.txt','must not be extracted')
    result=repack_archive(archive,tmp_path/'from_zip')
    assert len(result['files'])==5 and not (tmp_path/'irrelevant.txt').exists()
    partial=tmp_path/'partial';partial.mkdir()
    p=source/'02_dataset_dan_cleaning.xlsx';(partial/p.name).write_bytes(p.read_bytes())
    result=build_reports(partial,tmp_path/'partial_result')
    assert len(result['files'])==1 and Path(result['files'][0]).name=='01_Data_dan_Tokenisasi.xlsx'


def test_only_expected_pooler_notice_is_filtered():
    filter=ExpectedPoolerNotice()
    def record(message):return logging.LogRecord('transformers.modeling_utils',logging.WARNING,'fixture',1,message,(),None)
    expected="Some weights of the model checkpoint at BAAI/bge-m3 were not used when initializing XLMRobertaModel: ['pooler.dense.bias', 'pooler.dense.weight']"
    assert not filter.filter(record(expected))
    assert filter.filter(record(expected.replace("'pooler.dense.weight'","'encoder.layer.0.weight'")))
    assert filter.filter(record('Some weights of XLMRobertaModel were not initialized: pooler.dense.weight'))
    assert filter.filter(record('CUDA out of memory'))


def test_screen_comparison_explains_positions_and_similarity_without_changing_source():
    records=[{'word':'pangan','occurrence_indices':[2,10],'same_token_ids':True,
              'initial_cosine':1.,'initial_max_abs_difference':0.,'contextual_cosine':.999999999,
              'contextual_max_abs_difference':.05,'contextual_allclose_1e-6':False}]
    original=copy.deepcopy(records)
    table=preview_table(records[0],records,'Perbandingan kata')
    row=dict(zip(table['columns'],table['rows'][0]))
    assert row['Posisi Kata A']==3 and row['Posisi Kata B']==11
    assert row['Vektor Sebelum Konteks Sama?']=='Ya'
    assert row['Vektor Setelah Konteks Sama?']=='Tidak'
    assert row['Kemiripan Setelah Konteks (Cosine)']==.999999999
    assert records==original
    html=render_table(table)
    assert 'occurrence_indices' not in html and 'allclose' not in html
    assert 'belum membuktikan' in html and 'bukan tolerance skor AES' in html
    # Tidak mengarang perbandingan ketika informasi sumber hilang.
    partial=preview_table(['word','occurrence_indices'],[{'word':'x','occurrence_indices':[0,1]}])
    assert 'belum tersedia' in partial['rows'][0][-1]


def test_screen_registry_and_empty_comparison_do_not_dump_internal_metadata():
    record={'file':'01_konfigurasi.xlsx','stage':'Konfigurasi aktual; belum ada hasil training',
            'status':'ACTUAL EXECUTION','sheets_and_shapes':{'Info':{'rows':7,'columns':2}}}
    table=preview_table(record,[record])
    assert table['columns']==['File Hasil','Isi File','Keadaan']
    html=render_table(table)
    assert 'sheets_and_shapes' not in html and 'ACTUAL EXECUTION' not in html
    assert "&#x27;rows&#x27;" not in html
    empty=preview_table(['word','occurrence_indices'],[])
    assert 'bukan error' in render_table(empty) and 'Belum ada pasangan' in render_table(empty)


def test_screen_escapes_student_text_and_keeps_numbers():
    records=[{'row_id':'row_0001','word':'<script>private</script>','v_0000':.1234567890123,'v_0001':None}]
    original=copy.deepcopy(records)
    table=preview_table(records[0],records)
    assert table['rows'][0][0]=='J001' and table['rows'][0][2]==.1234567890123
    html=render_table(table)
    assert '<script>' not in html and '&lt;script&gt;' in html and '—' in html
    assert 'v_0000' not in html and 'Komponen 1' in html
    assert records==original
    percentage=preview_table(['out_of_range_fraction'],[{'out_of_range_fraction':.25}])
    assert percentage['columns']==['Di Luar Rentang (%)'] and percentage['rows']==[[25.]]


def test_old_results_can_be_read_in_colab_without_training(saved_outputs,tmp_path):
    source,_,_=saved_outputs
    result=build_reports(source,tmp_path/'readable')
    tables=list(saved_report_previews(Path(result['archive']).parent))
    assert len(tables)==5
    evaluations=next(t for t in tables if t['title']=='Hasil evaluasi tiga model')
    rmse=evaluations['columns'].index('RMSE (poin)')
    assert [r[rmse] for r in evaluations['rows']]==[2.125,3.125,4.125]
    repeats=next(t for t in tables if 'apakah vektornya' in t['title'])
    assert repeats['notes'] and 'Vektor Setelah Konteks Sama?' in repeats['columns']


def test_main_notebook_final_cell_downloads_current_optimized_reports(saved_outputs,tmp_path,monkeypatch):
    """Jalankan cell terakhir asli; hanya transport worker dan unduhan browser yang ditiru."""
    source,_,_=saved_outputs
    root=tmp_path/'runtime';(root/'outputs').mkdir(parents=True)
    pointer=root/'outputs/dataset300_download.json'
    pointer.write_text(json.dumps({'archive':'stale_unoptimized.zip'}))
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.xlsx')}
    worker={'journal':SimpleNamespace(folder=source),'output':root/'current_run','ROOT':root,'Path':Path,
            'dump':lambda path,value:path.write_text(json.dumps(value))}
    def aes_cell(line,cell):exec(compile(cell,'<main-export-worker>','exec'),worker)
    downloads=[]
    google=ModuleType('google');google.__path__=[]
    colab=ModuleType('google.colab');colab.files=SimpleNamespace(download=downloads.append)
    monkeypatch.setitem(sys.modules,'google',google)
    monkeypatch.setitem(sys.modules,'google.colab',colab)
    notebook=json.loads((Path(__file__).resolve().parents[1]/'notebooks/03_AES_dataset_guru_300.ipynb').read_text())
    # Muat definisi impor/tampilan/laporan dari notebook itu sendiri, tanpa import helper AES.
    for entry in notebook['cells']:
        if entry['cell_type']!='code':continue
        body=''.join(entry['source']).removeprefix('%%aes\n')
        nodes=ast.parse(body).body
        names={node.name for node in nodes if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
        imports={alias.name for node in nodes if isinstance(node,ast.Import) for alias in node.names}
        if {'numpy','torch','pandas'}<=imports or 'friendly_header' in names or 'Table' in names:
            exec(compile(body,'<standalone-report-definitions>','exec'),worker)
    cell=notebook['cells'][-1]
    kernel={'aes_cell':aes_cell,'ROOT':root,'Path':Path,'json':json,'IN_COLAB':True}
    exec(compile(''.join(cell['source']),'<main-notebook-final-cell>','exec'),kernel)
    current=json.loads(pointer.read_text())
    assert downloads==[current['archive']] and current['excel_count']==5
    assert Path(downloads[0]).parent==root/'current_run/hasil_ringkas'
    with zipfile.ZipFile(downloads[0]) as archive:
        assert sorted(archive.namelist())==sorted(current['files'])
        assert set(archive.namelist())=={'01_Data_dan_Tokenisasi.xlsx','02_Vektor_Jawaban.xlsx',
            '03_Vektor_Kata_dan_Kalimat.xlsx','04_Prediksi_dan_Evaluasi.xlsx','05_Riwayat_Pelatihan.xlsx'}
    assert before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.xlsx')}
