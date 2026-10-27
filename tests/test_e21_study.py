"""Integritas buku E21, provenance dan pemisahan data privat."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
from zipfile import ZipFile
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_e21_study import content,PIN,NOTEBOOK


@pytest.fixture
def snapshot():return json.loads((ROOT/'docs/e21_study_snapshot.json').read_text())


def test_three_inputs_one_label_and_matching_e21_recipe(snapshot):
    p={r['Pengaturan']:r['Nilai'] for r in snapshot['parameters']}
    assert snapshot['id']=='E21' and snapshot['test']==60 and snapshot['rows']==300
    assert p['Format input tiga teks']=='single_sequence'
    assert p['Learning rate encoder RoBERTa S1']==p['Learning rate encoder RoBERTa S3']==5e-6
    assert p['Skor minimum']==0 and p['Skor maksimum']==100
    assert snapshot['runtime_source_matches_commit'] is False
    assert snapshot['metric_recompute_max_delta']<snapshot['metric_recompute_tolerance']==1e-6


def test_cv_selected_epochs_come_from_actual_minima_and_refit_median(snapshot):
    for s in ('S1','S2','S3'):
        folds=[r for r in snapshot['folds'] if r['Skenario']==s]
        assert sorted(r['Fold'] for r in folds)==[1,2,3,4,5]
        chosen=[]
        for r in snapshot['foldSummary']:
            if r['Skenario']!=s:continue
            trace=[h for h in snapshot['history'][s] if h['Fold']==r['Fold'] and h['Tahap']=='Validasi silang']
            best=next(h for h in trace if h['Epoch']==r['Epoch Terpilih'])
            assert best['RMSE Validasi (poin)']==min(h['RMSE Validasi (poin)'] for h in trace)
            assert len(trace)==r['Epoch Dijalankan'];chosen.append(r['Epoch Terpilih'])
        assert snapshot['selected_epochs'][s]==int(__import__('math').ceil(statistics.median(chosen)))
        assert len([h for h in snapshot['history'][s] if h['Tahap']=='Latihan final'])==snapshot['selected_epochs'][s]


def test_aggregate_maps_rankings_and_error_units(snapshot):
    assert sum(r['share'] for r in snapshot['nearest'].values())==60
    for s in ('S1','S2','S3'):
        assert sum(map(sum,snapshot['matrices'][s]))==60
        err=next(r for r in snapshot['errorSummary'] if r['Skenario']==s)
        metric=next(r for r in snapshot['metrics'] if r['Skenario']==s)
        assert abs(err['MSE (poin²)']-metric['RMSE (poin)']**2)<1e-5
        assert err['MAE (poin)']<=metric['RMSE (poin)']
    assert all(r['Token jawaban tak tercakup']==0 for r in snapshot['coverage'])


def test_public_snapshot_has_no_individual_rows_and_no_older_results(snapshot):
    forbidden={'ID Jawaban','Jawaban Siswa','Soal','Nilai Guru','Prediksi S1','Prediksi S2','Prediksi S3','pairs'}
    def scan(v):
        if isinstance(v,dict):
            assert not forbidden.intersection(v)
            for x in v.values():scan(x)
        elif isinstance(v,list):
            for x in v:scan(x)
    scan(snapshot)
    d=content(snapshot)
    assert not {'comparison','multiSeed','lrExperiment','e8Experiment'}.intersection(d)
    raw=(ROOT/'docs/belajar_E21_Spider.html').read_text()
    assert '<script id="privatePayload" type="application/octet-stream"></script>' in raw


def test_all_code_blocks_and_fundamental_notes_match_pinned_source(snapshot):
    d=content(snapshot);n=json.loads(subprocess.check_output(['git','show',PIN+':'+NOTEBOOK],cwd=ROOT))
    assert d['coverage']['code_cells']==30 and d['coverage']['annotations']>=100
    for c in d['cells']:
        raw=n['cells'][c['index']]['source'];raw=raw if isinstance(raw,str) else ''.join(raw)
        assert c['source']==raw and c['sha256']==hashlib.sha256(raw.encode()).hexdigest()
        for note in c['notes']:assert 1<=note['line']<=len(c['htmlLines'])
    token=next(c for c in d['cells'] if c['index']==20)
    note=next(n for n in token['notes'] if n['title']=='Anggaran token jawaban')
    assert 'len(context_ids)' in token['source'].splitlines()[note['line']-1]


@pytest.mark.parametrize('name',['Belajar_AES_MLP_Lengkap','Belajar_AES_E21_Lengkap'])
def test_encrypted_packager_supports_separate_names_without_overwriting_other_books(tmp_path,name):
    pytest.importorskip('cryptography')
    source=tmp_path/'private.html';source.write_text('<html><p>TECHNICAL TEST ONLY</p></html>')
    output=tmp_path/'encrypted.html';archive=tmp_path/'book.zip';parts=tmp_path/'parts';key=tmp_path/'key.json'
    result=subprocess.run([sys.executable,str(ROOT/'scripts/package_private_study.py'),'--input',str(source),'--output',str(output),'--key-file',str(key),'--archive',str(archive),'--parts-dir',str(parts),'--artifact-name',name],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    with ZipFile(archive) as z:assert z.namelist()==[name+'.html']
    assert (parts/(name+'.html')).is_file() and (parts/(name+'.part1')).read_bytes()==archive.read_bytes()
    assert 'TECHNICAL TEST ONLY' not in output.read_text()
