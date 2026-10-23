"""Notebook GPU LoRA S3, diturunkan dari MLP jawaban saja tanpa mengubah acuan."""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'notebooks/experiments/AES_300_S3_MLP.ipynb'
DESTINATION = ROOT/'notebooks/experiments/AES_300_S3_LoRA_MLP_GPU.ipynb'


def source(cell):
    return cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Anchor harus unik: '+old[:100])
    return text.replace(old, new)


def definitions(text):
    text = text.removeprefix('%%aes\n')
    return {n.name: ast.get_source_segment(text, n) for n in ast.parse(text).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def build():
    original = BASE.read_bytes()
    notebook = copy.deepcopy(json.loads(original))
    old_hash = notebook['metadata']['aes_standalone']['code_sha256']
    helpers = definitions((ROOT/'src/aes/teacher_score/lora_fusion.py').read_text())
    manual = definitions((ROOT/'src/aes/teacher_score/manual_checkpoints.py').read_text())
    text_backend = definitions((ROOT/'src/aes/teacher_score/lora_text_runtime.py').read_text())['configure_text_only_transformers']
    bootstrap = text_backend+'\n\nconfigure_text_only_transformers()\n'
    for cell in notebook['cells']:
        text = source(cell)
        text = text.replace('AES_300_MLP_Privat', 'AES_300_S3_LoRA_MLP_Privat')
        text = text.replace('/content/AES_300_MLP', '/content/AES_300_S3_LoRA_MLP')
        text = text.replace('outputs/AES_300_MLP', 'outputs/AES_300_S3_LoRA_MLP')
        if cell['cell_type'] == 'code':
            text = text.replace(old_hash, '__NOTEBOOK_CODE_SHA256__')
            text = text.replace("lesson_version='s3_mlp_account2_v1'", "lesson_version='s3_lora_mlp_gpu_v1'")
            if 'REQUIREMENTS_TEXT=' in text:
                # PEFT memasang adapter resmi; versi dikunci dan diperiksa oleh setup acuan.
                text = replace_once(text, 'openpyxl==3.1.5', 'openpyxl==3.1.5\npeft==0.14.0\naccelerate==1.3.0')
            if 'IMPORTS = ' in text:
                # Setiap probe dan worker baru menerapkan kebijakan sebelum PEFT
                # memuat model Transformers melalui import tidak langsung.
                text = replace_once(text, "IMPORTS = r'''\n", "IMPORTS = r'''\n"+bootstrap)
                text = replace_once(text, 'from transformers import AutoModel, AutoTokenizer\navailable=',
                                    'from transformers import AutoModel, AutoTokenizer\nfrom peft import LoraConfig, get_peft_model\navailable=')
                text = text.replace("'imports_verified':True", "'imports_verified':True,'transformers_backend':'pytorch_text_only'")
                # Pakai overlay baru agar torch yang terlanjur terpasang pada
                # versi sebelumnya tidak menutupi torch bawaan Colab.
                text = replace_once(text, "'dataset300_env' if version==(3,12) else 'dataset300_env_py313'", "'dataset300_env_text_lora' if version==(3,12) else 'dataset300_env_py313_text_lora'")
                text = replace_once(text, 'queue=[Requirement(raw) for raw in roots];seen=set();issues=[]',
                                    'queue=[Requirement(raw) for raw in roots];seen=set();issues=[];install_requirements=[]')
                text = replace_once(text, "issues.append(str(req)+' belum tersedia');continue", "issues.append(str(req)+' belum tersedia');install_requirements.append(str(req));continue")
                text = replace_once(text, "issues.append(str(req)+' diperlukan; tersedia '+dist.version)", "issues.append(str(req)+' diperlukan; tersedia '+dist.version);install_requirements.append(str(req))")
                text = replace_once(text, "'dependency_issues':sorted(set(issues)),", "'dependency_issues':sorted(set(issues)),'install_requirements':sorted(set(install_requirements)),")
                text = replace_once(text, "raise SetupError(stage+' gagal. Rincian lengkap: '+str(logfile))", """details='\\n'.join((process.stderr or process.stdout or 'Proses gagal tanpa rincian.').splitlines()[-18:])
                raise SetupError(stage+' gagal.\\n'+details+'\\nRincian lengkap: '+str(logfile))""")
                start=text.index('        installed=False\n')
                end=text.index("        if after['dependency_issues']:",start)
                text=text[:start]+'''        if not before['torch_present']:
            raise SetupError('PyTorch bawaan runtime belum tersedia. Pilih runtime Colab GPU yang baru; setup ini tidak memasang wheel CPU.')
        installed=False
        after=before
        if after['dependency_issues']:
            if args.verify_python:
                raise SetupError('Worker yang dipilih belum memenuhi kebutuhan AES. Rincian: '+str(logfile))
            print('2/3 · Menyiapkan pustaka AES tanpa mengganti PyTorch runtime...',flush=True)
            # --no-deps mencegah PEFT menarik ulang torch. Dependensi lain
            # diperiksa ulang dari metadata dan dipasang hanya jika diperlukan.
            pip=[sys.executable,'-m','pip','--python',python,'install','--ignore-installed','--no-deps','--no-input']
            for attempt in range(8):
                if not after['dependency_issues']:
                    break
                pending=after['install_requirements']
                if not pending or any(Requirement(raw).name.lower().replace('_','-')=='torch' for raw in pending):
                    raise SetupError('Dependensi meminta perubahan PyTorch. Gunakan runtime GPU yang kompatibel; torch bawaan tidak ditimpa.')
                execute(pip+pending,'Pemasangan pustaka AES privat, putaran '+str(attempt+1))
                installed=True
                after=inspect(python)
                if after['versions'].get('torch')!=torch_version:
                    raise SetupError('Versi PyTorch berubah selama setup; training tidak dimulai.')
        else:
            print('2/3 · Versi pustaka sudah cocok; tidak perlu instalasi ulang.',flush=True)
''' + text[end:]
            if 'from torch import nn' in text:
                text = replace_once(text, 'from torch import nn', 'from torch import nn\n'+bootstrap+'\nfrom peft import LoraConfig, get_peft_model')
            if 'DEFAULTS = {' in text:
                text = replace_once(text, "    'vector_source': 'pretrained', 'trace_position': 0,", """    # Nilai awal eksperimen, bukan nilai optimal yang sudah dibuktikan.
    'lora_rank': 8, 'lora_alpha': 16, 'lora_dropout': 0.,
    'lora_lr': 1e-4, 'lora_targets': ['query', 'value'],
    'vector_source': 'pretrained', 'trace_position': 0,""")
                text = replace_once(text, '    return result', '    validate_lora_config(result)\n    return result')
                text += '\n\n'+helpers['validate_lora_config']+'\n'
            if 'ROBERTA_CHECKPOINT=' in text:
                text = replace_once(text, 'PROJECTION_LR=HEAD_LR=1e-4', """PROJECTION_LR=HEAD_LR=1e-4
# S3: hanya LoRA di encoder yang belajar; bobot RoBERTa asli tetap.
LORA_RANK,LORA_ALPHA=8,16  # Rank = lebar matriks kecil; skala penyesuaian = alpha/rank.
LORA_LR=1e-4  # Titik awal terpisah dari LR encoder S1; pilih melalui CV, bukan data uji.
LORA_DROPOUT=0.0  # Tidak menambah dropout baru; dropout head tetap 0.1.""")
                text = replace_once(text, 'encoder_lr=ENCODER_LR,projection_lr=PROJECTION_LR,head_lr=HEAD_LR,',
                                    'encoder_lr=ENCODER_LR,projection_lr=PROJECTION_LR,head_lr=HEAD_LR,\n                lora_rank=LORA_RANK,lora_alpha=LORA_ALPHA,lora_lr=LORA_LR,lora_dropout=LORA_DROPOUT,')
                text = text.replace("('torch','transformers','numpy')", "('torch','transformers','numpy','peft','accelerate')")
                text = replace_once(text, "device='auto'", "device='cuda'")
                text = replace_once(text, "gpu=check_gpu(require_gpu=RUN_MODELS and os.environ.get('AES300_REQUIRE_GPU')=='1')", """REQUIRE_GPU=True  # Pilih Runtime → Change runtime type → T4 GPU sebelum menjalankan.
gpu=check_gpu(require_gpu=RUN_MODELS and REQUIRE_GPU)
if RUN_MODELS:
    print(f"GPU aktif: {gpu['name']}. Embedding dan training menggunakan CUDA.",flush=True)""")
            if "journal.save('01_konfigurasi'" in text:
                text=text.replace("'scikit-learn','openpyxl')", "'scikit-learn','openpyxl','peft','accelerate')")
            if 'def make_model(' in text:
                text = text.replace('def make_model(', '\n\n'.join(v for k,v in helpers.items() if k != 'validate_lora_config')+'\n\ndef make_model(', 1)
                text = replace_once(text, "    return SingleScoreModel(scenario, encoder(cfg, 'roberta') if scenario != 'S2' else None,\n                            dropout=cfg['dropout'])", """    model=SingleScoreModel(scenario, encoder(cfg,'roberta') if scenario!='S2' else None,
                           dropout=cfg['dropout'])
    # Pasang setelah konstruksi agar requires_grad dari acuan tidak membuka base lagi.
    return attach_s3_lora(model,cfg)""")
            if 'specifications=pd.DataFrame' in text:
                text = text.replace("'RoBERTa ikut dilatih; BGE-M3 tetap'", "'RoBERTa: LoRA dilatih, bobot asli frozen; BGE-M3 frozen'")
                text += """
if RUN_MODELS:
    # Hanya inspeksi parameter; bukan training. Seed model latihan diatur ulang setiap fold.
    inspection_model=make_model('S3',cfg)
    inspection=pd.DataFrame(parameter_report(inspection_model))
    journal.save('08b_parameter_lora',{'Parameter':inspection},'Bagian S3 yang dilatih dan dibekukan',status='DESIGN / CONFIGURATION')
    show(inspection,'S3: parameter LoRA, fusion dan head yang belajar')
    del inspection_model
    release()
"""
            if 'def save_training_progress(' in text:
                parts=definitions(text)
                text='%%aes\n# Checkpoint disimpan tiap epoch; penghapusan dilakukan manual.\n\n'+'\n\n\n'.join(manual[name] if name=='save_training_progress' else value for name,value in parts.items() if name!='clear_training_progress')+'\n'
                text = replace_once(text, "buckets = {'roberta': [], 'projection': [], 'head': []}", "buckets = {'roberta': [], 'lora': [], 'projection': [], 'head': []}")
                text = replace_once(text, "        buckets[group].append(parameter)", "        if group=='roberta' and model.scenario=='S3':\n            group='lora'\n        buckets[group].append(parameter)")
                text = replace_once(text, "'roberta': cfg['encoder_lr'],", "'roberta': cfg['encoder_lr'], 'lora': cfg['lora_lr'],")
                text = text.replace("model.load_state_dict(saved['model'])", "restore_training_state(model,saved['model'])")
                text = text.replace("best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}", "best_state = training_state(model)")
                text = text.replace("'model': model.state_dict()", "'model': training_state(model)")
                text = text.replace('model.load_state_dict(best_state)', 'restore_training_state(model,best_state)')
            if 'def train_all(' in text:
                text='\n'.join(line for line in text.splitlines() if not line.lstrip().startswith('clear_training_progress('))+'\n'
                text=replace_once(text, "'state': {k: v.detach().cpu() for k, v in model.state_dict().items()}", "'checkpoint_format': 'lora_trainable_only_v1' if model.scenario=='S3' else 'full_state_v1',\n               'state': training_state(model)")
                text=replace_once(text, "'roberta_overlap', 'bge_max_length'):", "'roberta_overlap', 'bge_max_length',\n                'lora_rank','lora_alpha','lora_dropout','lora_targets','dropout'):")
                text=replace_once(text, "    model.load_state_dict(payload['state'], strict=True)", """    expected_format='lora_trainable_only_v1' if payload['scenario']=='S3' else 'full_state_v1'
    if payload.get('checkpoint_format')!=expected_format:
        raise ValueError('Format checkpoint berbeda dari eksperimen LoRA ini.')
    restore_training_state(model,payload['state'])""")
            if 'def vector_lab(' in text:
                text=text.replace("source = cfg['vector_source']+' fine-tuned; sha256='+digest(checkpoint)", "source = cfg['vector_source']+(' fine-tuned dengan LoRA' if cfg['vector_source']=='S3' else ' fine-tuned')+'; sha256='+digest(checkpoint)")
            if 'def config_report(' in text or "settings={'epochs'" in text:
                text=text.replace("'micro_batch':'Micro batch',", "'micro_batch':'Micro batch','lora_rank':'Rank LoRA S3','lora_alpha':'Alpha LoRA S3',\n              'lora_dropout':'Dropout LoRA S3','lora_lr':'Learning rate LoRA S3','lora_targets':'Target attention LoRA S3',")
                text=text.replace("'encoder_lr':'Learning rate encoder'", "'encoder_lr':'Learning rate encoder RoBERTa S1'")
                text=replace_once(text, "    settings={'epochs'", """    if source.has('08b_parameter_lora.xlsx','Parameter'):
        sheets['Parameter LoRA dan Fusion']=source.copy('08b_parameter_lora.xlsx','Parameter')
    settings={'epochs'""")
            cell['outputs'],cell['execution_count']=[],None
        else:
            if text.startswith('# AES 300'):
                text='''# AES 300 · S3 RoBERTa–LoRA + BGE frozen + MLP · GPU

Eksperimen terpisah dari **MLP acuan**, dengan input **jawaban siswa saja**, tanpa konteks pair. S1 tetap RoBERTa fine-tuned + head linear. S2 tetap BGE-M3 frozen + MLP. Perubahan hanya S3: RoBERTa di-fine-tune menggunakan LoRA pada query/value attention; bobot aslinya dibekukan. BGE-M3 tetap frozen. LayerNorm, proyeksi BGE 1024→256, concat 768+256 dan MLP 1024→256→GELU→dropout→1 tetap dilatih seperti acuan.

**Jalankan dari atas dengan GPU:** Runtime → Change runtime type → T4 GPU (GPU NVIDIA lain juga diterima). Guard GPU dan device=cuda menghentikan latihan bila GPU belum aktif. Setup mempertahankan PyTorch CUDA bawaan Colab; PEFT dan Accelerate dipasang dengan versi terkunci. Pengerjaan aktual GPU berlangsung setelah pemeriksaan perangkat berhasil.

**Urutan:** setup/Drive → pilih fix_dataset_300.xlsx → audit/cleaning → split 80/20 → 5-fold CV pada 80% development → tokenisasi/vektor → model → training/refit/test → prediksi → vektor sesudah latihan → lima Excel. Dataset memiliki satu skor guru 0–100. Tidak membuat label empat rubrik atau memasukkan skor guru ke encoder.

**Parameter awal LoRA:** rank 8, alpha 16 (skala alpha/r=2), target query/value di seluruh layer attention, dropout LoRA 0, LR LoRA 1e-4. Ini kandidat eksperimen, bukan parameter optimal yang telah dibuktikan. LR encoder S1 tetap 1e-5; LR proyeksi/head 1e-4, maksimal 25 epoch/fold, patience 5, microbatch 1/batch efektif 8. Test tidak digunakan memilih hyperparameter.

**Drive terpisah:** MyDrive/AES_300_S3_LoRA_MLP_Privat. Simpan progres setiap epoch, termasuk optimizer dan keadaan acak, untuk melanjutkan notebook yang sama dari atas setelah sesi terputus. Tidak ada auto-delete atau Drive API cleanup. Checkpoint S3 menyimpan LoRA + LayerNorm/proyeksi/head; base frozen dipulihkan dari nama/revision pretrained terkunci. S1 tetap menyimpan bobot fine-tuning penuh. Jangan hapus final.pt; untuk tahap berjalan pertahankan JSON progres dan dua snapshot yang dirujuknya. JSON bukan pengganti bobot. Checkpoint eksperimen lama tidak dipakai otomatis.

**Hasil:** lima Excel seperti acuan, termasuk tokenisasi, seluruh komponen vektor, riwayat epoch, nilai guru/prediksi mentah/pembulatan, RMSE, Pearson, QWK dan Gwet AC2. Ada tabel parameter yang dilatih/frozen dan pengaturan LoRA. Vektor kata/kalimat S3 diambil dari encoder dengan adapter final aktif; tidak mengarang vektor fusion per kata.

Notebook mandiri: seluruh kode tersedia di cell, tidak clone/import proyek GitHub. Dataset dan output privat tidak ditanam di notebook publik. Belum ada hasil penelitian sebelum notebook dijalankan. Perbandingan dengan test lama yang sudah dilihat bersifat eksploratif; pilih metode melalui CV.
'''
            if text.startswith('<details><summary>Rujukan metode'):
                text += '\n\n**LoRA:** Hu dkk., ICLR 2022, https://openreview.net/forum?id=nZeVKeeFYf9. Bobot asli frozen + penyesuaian low-rank. Implementasi memakai PEFT 0.14.0: https://github.com/huggingface/peft/tree/v0.14.0. Rank/LR notebook merupakan pilihan awal eksperimen; paper tidak menjamin peningkatan RMSE pada dataset ini.\n'
            if text.startswith('## 10 '):
                text += '\n\nS3: make_model() membangun fusion acuan dahulu, lalu attach_s3_lora() memasang adapter pada RoBERTa. Head MLP dan proyeksi tetap dilatih. S1/S2 tidak dipasangi LoRA.\n'
            if text.startswith('## 11 '):
                text += '\n\nOptimizer memiliki kelompok LoRA S3 dengan LR tersendiri. Bobot asli S3 frozen tidak masuk optimizer. S1 tetap memakai encoder_lr; proyeksi/head tetap memakai LR acuan.\n'
            if text.startswith('## 12 '):
                text += '\n\nCheckpoint S3 ringkas memuat seluruh parameter belajar dan status optimizer; bobot pretrained frozen dimuat ulang dari revision terkunci. Tidak ada penghapusan checkpoint otomatis.\n'
        cell['source']=text
        cell['metadata']={}
        cell.pop('attachments',None)
    normalized=[source(c).removeprefix('%%aes\n') for c in notebook['cells'] if c['cell_type']=='code']
    for code in normalized:
        ast.parse(code)
    fingerprint=hashlib.sha256('\n'.join(normalized).encode()).hexdigest()
    for cell in notebook['cells']:
        cell['source']=source(cell).replace('__NOTEBOOK_CODE_SHA256__',fingerprint)
    notebook['metadata']['aes_standalone']['code_sha256']=fingerprint
    notebook['metadata']['colab']['name']=DESTINATION.name
    notebook['metadata']['accelerator']='GPU'
    notebook['metadata']['aes_experiment']={
        'name':'S3_LORA_MLP_GPU','base_sha256':hashlib.sha256(original).hexdigest(),
        'input':'answer_only','context_pair':False,'s1':'full_fine_tuned_linear',
        's2':'frozen_bge_mlp','s3':'lora_roberta_frozen_bge_mlp',
        'lora_rank':8,'lora_alpha':16,'lora_targets':['query','value'],
        'automatic_checkpoint_deletion':False,'drive_root':'AES_300_S3_LoRA_MLP_Privat'}
    DESTINATION.parent.mkdir(parents=True,exist_ok=True)
    DESTINATION.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
    mirror=ROOT/'aes-learning/notebooks/experiments'/DESTINATION.name
    mirror.parent.mkdir(parents=True,exist_ok=True)
    mirror.write_bytes(DESTINATION.read_bytes())
    assert BASE.read_bytes()==original
    print(json.dumps({'notebook':str(DESTINATION.relative_to(ROOT)),'code_sha256':fingerprint,'base_unchanged':True}))
    return DESTINATION


if __name__=='__main__':
    build()
