"""Catalog availability must not alter research evidence or disclose private payloads."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE='7a76191be20604482fa53f770699563b380120ba'


def family(raw=None):
    raw=raw or (ROOT/'docs/belajar_Tiga_Input_Spider.html').read_text()
    return json.loads(re.search(r'<script id="familyData" type="application/json">(.*?)</script>',raw,re.S)[1])


def test_catalog_adds_two_pending_identities_without_changing_any_research_content():
    old=family(subprocess.check_output(['git','show',BASE+':docs/belajar_Tiga_Input_Spider.html'],cwd=ROOT,text=True))
    new=family()
    assert new['comparison']==old['comparison']
    assert new['documents']==old['documents']
    assert list(new['identities'])==['E'+str(i) for i in range(21,31)]
    assert sum(x['result_available'] for x in new['identities'].values())==8
    for key in ('E29','E30'):
        item=new['identities'][key]
        assert item['base']=='E26'
        assert item['reserved_code'] is None and item['reserved_output'] is None
        assert item['book_available'] is False and item['result_available'] is False
        assert key not in new['comparison']['experiments'] and key not in new['documents']
        assert item['source_commit']==BASE
        assert (ROOT/item['source_path']).is_file()
        assert subprocess.check_output(['git','show',BASE+':'+item['source_path']],cwd=ROOT)==(ROOT/item['source_path']).read_bytes()
    assert new['identities']['E29']['head_lr']=='MLP S2/S3 5e-6 · linear S1 1e-4'
    assert new['identities']['E30']['head_lr']=='MLP S2/S3 1e-4 · linear S1 1e-4'
    assert new['identities']['E27']['base']==new['identities']['E28']['base']=='E23'


def test_every_identity_has_specific_citations_traceable_source_and_same_input_contract():
    catalog=family()['identities']
    for key,item in catalog.items():
        assert item['inputs']==['soal','jawaban_referensi','response_text']
        assert item['label']=='rater · 0–100' and item['heads']=='S1 linear · S2/S3 MLP'
        assert 7<=len(item['source_commit'])<=40 and item['source_path'].endswith('.ipynb')
        assert all(c['url'].startswith('https://') and c['scope'] for c in item['citations'])
        assert {'roberta','bge'} <= {c['key'] for c in item['citations']}
        assert item['adaptation']
    assert {'coattention','attention','bigru'} <= {c['key'] for c in catalog['E30']['citations']}
    assert 'bukan' in catalog['E29']['adaptation'] or 'tidak' in catalog['E29']['adaptation']
    public=(ROOT/'docs/belajar_Tiga_Input_Spider.html').read_text()
    assert not re.search(r'data-family-chunk>',public)
    assert '<script id="familyPrivate" type="application/octet-stream"></script>' in public
    protected=['docs/belajar_MLP_Spider.html',*map(str,(ROOT/'downloads').glob('Belajar_AES_MLP_Lengkap*'))]
    for path in protected:
        path=str(Path(path).relative_to(ROOT)) if Path(path).is_absolute() else path
        assert subprocess.check_output(['git','show',BASE+':'+path],cwd=ROOT)==(ROOT/path).read_bytes()
