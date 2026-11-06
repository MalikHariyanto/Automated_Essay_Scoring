"""Build native E52 from pinned E43 architecture and verified modern reporting."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PIN = '7241a21dfec37c942948b754da3960fba4c7731c'
BASE = 'notebooks/experiments/AES_300_E43_E42_S3_MaskedMeanPooling_Native_Learnable_GPU.ipynb'
REPORT_PIN = 'c36003670ad9476951b8b607a6018e5a24ceb7e2'
REPORT_BASE = 'notebooks/experiments/AES_300_E50_E43_S3_MultiHeadCoAttention4_Native_Learnable_GPU.ipynb'
NAME = 'AES_300_E52_E43_S3_DoRA8_Native_Learnable_GPU'
PLACEHOLDER = '__NOTEBOOK_CODE_SHA256__'

INTRO = '''# E52 · E43 + DoRA pada RoBERTa · S3 Native Learnable

**GPU T4 → Run all → upload `fix_dataset_300.xlsx`.** Cukup dataset Excel, tanpa ZIP hasil lama. Seluruh kode Python terlihat per section, tanpa magic, worker, clone repo, atau Drive.

**Perubahan dari E43 adalah resep pelatihan RoBERTa:** full fine-tuning dengan LR `5e-6` diganti DoRA query/value dengan LR adapter `1e-4`. Bobot pretrained asli dibekukan; adapter arah A/B dan magnitude DoRA dipelajari dari skor guru. Ini perubahan metode **dan LR aktif encoder**, sehingga bukan ablasi LR yang seluruh komponennya identik. LR adapter adalah kandidat awal, bukan nilai optimum dari paper.

Arsitektur tetap E43: tiga teks → RoBERTa yang beradaptasi melalui DoRA → ScalarMix empat layer + gamma → BiGRU 64 per arah → co-attention satu head respons–referensi + roundtrip → matching → masked mean → fitur 768; BGE-M3 frozen → proyeksi 256; concat 1.024 → MLP 256/GELU/dropout 0,1 → satu skor. Tidak memakai multihead E50 atau fusion gate E51.

| Pengaturan | Nilai |
|---|---|
| Model RoBERTa / BGE | `flax-community/indonesian-roberta-base` / `BAAI/bge-m3`; revision sama E43 |
| Input X | `soal`, `jawaban_referensi`, `response_text` |
| Target y | `rater` 0–100; hanya loss/seleksi/evaluasi, tidak masuk encoder |
| Implementasi | PEFT `0.14.0`, `LoraConfig(..., use_dora=True)`; bukan LoRA biasa |
| DoRA | Rank 8, alpha 16, target query/value seluruh 12 layer, dropout adapter 0 |
| Parameter encoder yang belajar | 294.912 arah A/B + 18.432 magnitude = **313.344**; bobot dasar frozen |
| LR DoRA / komponen lain | `1e-4` / `1e-4`; bobot dasar frozen tidak masuk optimizer |
| Training | S3 saja; satu 5-fold CV baru, maksimal 25 epoch dan patience 5 |
| Refit / test | Ceil median epoch terbaik CV; satu refit development dan satu evaluasi test |
| Loss / seleksi | MSE biasa, target 0–1 / RMSE validasi tanpa bobot |

DoRA memisahkan besar bobot (*magnitude*) dan arah bobot. A/B mempelajari perubahan arah berperingkat rendah; magnitude juga dipelajari. Magnitude DoRA berbeda dari gamma ScalarMix: magnitude berada pada matriks query/value encoder, gamma mengalikan hasil campuran empat layer. Encoder tetap beradaptasi dari label melalui adapter meskipun bobot dasarnya frozen. BGE tetap frozen, sedangkan proyeksi BGE, BiGRU, co-attention/matching, normalisasi, ScalarMix/gamma dan MLP tetap dilatih.

Adapter dipasang setelah inisialisasi seluruh komponen E43. Inisialisasi memakai `fork_rng` CPU supaya tidak menggeser RNG/dropout bersama. B awal nol dan magnitude awal mengikuti norma bobot pretrained. Fungsi awal mempertahankan encoder pretrained dalam toleransi numerik; pengujian membandingkan keluaran dan gradien dengan implementasi resmi PEFT. Tidak ada quantization, QLoRA, merge adapter saat training, atau fallback diam-diam ke LoRA/full fine-tuning.

**Protokol tetap:** split/fold dan seed dasar 42 sama E43, tanpa loop banyak seed atau kontrol ulang. AdamW/weight decay 0,01, warmup 10%, linear scheduler sesuai durasi fit E43, clipping 1, microbatch 1/batch efektif 8. Loss, dropout MLP, tokenizer/window, pooling, fusion dan data tidak diubah. Tidak memakai LDS, ranking, cosine, S1/S2, atau scheduler refit E46/E47. Early stopping hanya membaca validasi.

## Laporan lengkap

- **Excel 04:** 60 prediksi test, skor mentah/normalisasi/pembulatan; selisih tepat 0/1/2/3/4/5/10 dan lainnya; toleransi ≤n; MSE/MAE/error maksimum; **240 prediksi validasi OOF** terpisah dari test, beserta fold dan error.
- **Excel 05 → Adapter DoRA:** status aktif, layer/target, rank/alpha/skala, jumlah parameter, status frozen/trainable, norma A/B/magnitude dari checkpoint terpilih lima fold serta refit. Untuk 12 layer: 24 modul × 6 checkpoint = **144 baris**. Norma bukan kontribusi kausal terhadap skor.
- **Excel 05 → Parameter Dilatih dan Frozen:** jumlah parameter aktual per kelompok dan checkpoint. History MSE, Bobot ScalarMix/gamma, MAE dan toleransi CV tetap tersedia.
- **Excel 02:** vektor BGE, fitur RoBERTa, masked mean dan concat S3 tetap tersedia. Inspeksi model refit atas 300 jawaban bukan evaluasi 300 data.

Pemeriksaan sebelum ZIP membaca ulang metrik, ringkasan, OOF, rekap selisih, adapter aktif, target/layer lengkap dan bobot asli frozen. Resume memakai hash/config/checkpoint E52 tersendiri dan memulihkan optimizer/scheduler/scaler/RNG. Checkpoint menyimpan state model lengkap termasuk bobot frozen, seperti alur native E43; optimizer hanya menyimpan parameter trainable. Jangan memakai checkpoint E5/E43/E50/E51. Report dapat dibangun ulang dari hasil tersimpan tanpa evaluasi test tambahan.

## Landasan dan batas bukti

Liu dkk., **[DoRA: Weight-Decomposed Low-Rank Adaptation, ICML 2024](https://arxiv.org/abs/2402.09353)** · [repositori penulis](https://github.com/NVlabs/DoRA) · [PEFT v0.14.0](https://github.com/huggingface/peft/tree/v0.14.0). Paper menguji tugas/model lain, bukan AES Indonesia dengan E43. Rank 8/alpha 16/query-value dan LR `1e-4` adalah resep awal proyek, mengikuti target LoRA E5 agar jelas; bukan kombinasi optimum yang dibuktikan paper.

E5 terdahulu memakai LoRA biasa dan input jawaban saja, sehingga bukan kontrol identik untuk E52. Acuan aktif E43: mean RMSE CV 5,8876, MAE OOF 4,6122, selisih mentah ≤5 sebanyak 151/240; RMSE test historis 5,1265. **Belum ada hasil penelitian E52.** Pilih melalui validasi; test historis bersifat eksploratif. Tidak menjamin peningkatan akurasi atau target 60/60 dalam ±3/±5 poin.
'''


def once(source, old, new):
    assert source.count(old) == 1, old[:150]
    return source.replace(old, new, 1)


def build():
    raw = subprocess.check_output(['git', 'show', PIN+':'+BASE], cwd=ROOT)
    report_raw = subprocess.check_output(['git', 'show', REPORT_PIN+':'+REPORT_BASE], cwd=ROOT)
    n = copy.deepcopy(json.loads(raw)); report = json.loads(report_raw)
    reports = {c['metadata'].get('aes50_role'): c for c in report['cells']}
    report_roles = {'definition:evaluation_report', 'definition:training_report', 'definition:build_reports',
                    'download_execute', 'training_execute', 'definition:export_training_event'}
    for cell in n['cells']:
        role = cell['metadata'].pop('aes43_role', '')
        if role:
            cell['metadata']['aes52_role'] = role
        origin = report if role in report_roles else json.loads(raw)
        source = reports[role]['source'] if role in report_roles else cell['source']
        source = source if isinstance(source, str) else ''.join(source)
        source = source.replace(origin['metadata']['aes_standalone']['code_sha256'], PLACEHOLDER)
        for old in [Path(BASE).stem, Path(REPORT_BASE).stem,
                    'AES_300_E50_E50_S3_MultiHeadCoAttention4_Native_Learnable_GPU']:
            source = source.replace(old, '__E52_NAME__')
        source = source.replace('E43', 'E52').replace('E50', 'E52').replace('__E52_NAME__', NAME)
        if role == 'dependency_setup':
            source = source.replace("'openpyxl>=3.1,<4']", "'openpyxl>=3.1,<4', 'peft==0.14.0', 'accelerate==0.34.2']")
            source = source.replace('/tmp/aes35_native_constraints.txt', '/tmp/aes52_native_constraints.txt')
        if role == 'imports':
            source = source.replace('from scipy.stats import pearsonr', 'from peft import LoraConfig, get_peft_model\nfrom scipy.stats import pearsonr')
        if role == 'presentation_helpers':
            source = '\n'.join(line for line in source.splitlines() if "LABELS.update({'fusion_gate':" not in line)+'\n'
        if role == 'model_names':
            source = source.replace('| encoder ikut fine-tuning', '| bobot dasar frozen; adapter DoRA arah/magnitude belajar')
        if role == 'hyperparameters':
            source = once(source, 'ENCODER_LR = 5e-6             # Tetap E30; bukan LR E37 7e-6.', '''# LR encoder di E52 berarti LR adapter; seluruh bobot pretrained asli frozen.
DORA_LR = 1e-4
ENCODER_LR = DORA_LR
DORA_RANK = 8
DORA_ALPHA = 16
DORA_DROPOUT = 0.0
DORA_TARGET_MODULES = ['query', 'value']
DORA_USE_DORA = True''')
            source = source.replace('# HIPERPARAMETER PENELITIAN — E30 MLP; Scalar Mix empat layer terakhir.', '# HIPERPARAMETER E52 — E43 dengan DoRA pada encoder RoBERTa.')
        if role == 'configuration_defaults':
            source = once(source, "    's3_fusion': 'concat',", "    'encoder_adaptation':'dora', 'dora_rank':DORA_RANK, 'dora_alpha':DORA_ALPHA,\n    'dora_dropout':DORA_DROPOUT, 'dora_targets':DORA_TARGET_MODULES,\n    'dora_use_dora':DORA_USE_DORA, 'dora_lr':DORA_LR,\n    's3_fusion': 'concat',")
            source = once(source, '    return result\n', '    validate_dora_config(result)\n    return result\n')
        if role == 'configuration_apply':
            source = source.replace('e43_native_masked_mean_v1', 'e52_native_dora8_v1')
            source = source.replace("('torch','transformers','numpy','pandas','scikit-learn')", "('torch','transformers','peft','accelerate','numpy','pandas','scikit-learn')")
            source = source.replace("'Encoder RoBERTa S3'", "'Adapter DoRA RoBERTa S3 (bobot dasar frozen)'")
            source += "\nprint('DoRA aktif:',cfg['dora_use_dora'],'| rank:',cfg['dora_rank'],'| alpha:',cfg['dora_alpha'],'| LR:',cfg['dora_lr'])\n"
        if role == 'definition:make_model':
            source = once(source, "    return SingleScoreModel('S3',", "    model = SingleScoreModel('S3',")
            source += '\n    return attach_dora(model, cfg)\n'
        if role == 'definition:optimizer_groups':
            source = source.replace('Lima kelompok optimizer S3;', 'Lima kelompok optimizer S3; kelompok roberta hanya adapter DoRA;')
        if role == 'definition:fit':
            source = once(source, "    info['scalar_mix_layers'] = cfg['scalar_mix_layers']", "    info['scalar_mix_layers'] = cfg['scalar_mix_layers']\n    info['dora_adapters'] = dora_adapter_report(model)\n    info['parameter_counts'] = dora_parameter_counts(model)\n    info['encoder_adaptation'] = 'dora'")
        if role == 'definition:load_model':
            source = source.replace("for key in ('roberta',", "for key in ('encoder_adaptation','dora_use_dora','dora_rank','dora_alpha','dora_dropout','dora_targets','dora_lr','lesson_version','roberta',")
        if role == 'architecture_export':
            source = source.replace('RoBERTa fine-tuned; BGE-M3 frozen', 'RoBERTa DoRA; bobot dasar dan BGE-M3 frozen')
            source = source.replace('RoBERTa → Scalar', 'RoBERTa + DoRA → Scalar')
        if role == 'definition:export_training_event':
            source = once(source, "'Proyeksi_multihead':pd.DataFrame(training['multihead_projections']),", "'Adapter_DoRA':pd.DataFrame(training['dora_adapters']),\n                                                           'Parameter_DoRA':pd.DataFrame(training['parameter_counts']),")
            source = once(source, "'Proyeksi_multihead':pd.DataFrame(result['multihead_projections'])", "'Adapter_DoRA':pd.DataFrame(result['dora_adapters']),\n                                            'Parameter_DoRA':pd.DataFrame(result['parameter_counts'])")
        if role == 'definition:training_report':
            source = once(source, "settings={'coattention_heads':'Jumlah Head Co-Attention',", "settings={'encoder_adaptation':'Adaptasi Encoder','dora_use_dora':'DoRA Aktif','dora_rank':'Rank DoRA','dora_alpha':'Alpha DoRA','dora_dropout':'Dropout Adapter DoRA','dora_lr':'Learning Rate DoRA','dora_targets':'Target DoRA',")
            source = source.replace('sheets.update(multihead_report_tables(source))', 'sheets.update(dora_report_tables(source))')
        if role == 'definition:evaluation_report':
            source = source.replace(' + Multihead Co-Attention4', ' + DoRA + Co-Attention')
            source = source.replace('Fusion RoBERTa fine-tuned + BGE-M3 frozen', 'Fusion RoBERTa dengan DoRA + BGE-M3 frozen')
        if role == 'definition:build_reports':
            source = source.replace('validate_e50_download', 'validate_e52_download')
        cell['source'] = source
    helper = (ROOT/'scripts/dora_native.py').read_text()
    helper = '\n\n'.join(ast.get_source_segment(helper, node) for node in ast.parse(helper).body if isinstance(node, (ast.FunctionDef, ast.ClassDef)))+'\n'
    index = next(i for i,c in enumerate(n['cells']) if c['metadata'].get('aes52_role') == 'configuration_defaults')
    n['cells'][index:index] = [
        {'cell_type':'markdown','metadata':{},'source':'## DoRA resmi: konfigurasi, pemasangan, dan audit parameter\n\n`use_dora=True` mengaktifkan A/B serta magnitude. Fungsi di bawah dipanggil saat model dibuat; label tidak menjadi input adapter. Helper laporan memeriksa checkpoint terpilih.\n'},
        {'cell_type':'code','metadata':{'aes52_role':'definition:dora_helpers'},'source':helper,'outputs':[],'execution_count':None}]
    source = reports['definition:download_helpers']['source']
    source = source.replace('validate_e50_download', 'validate_e52_download').replace('OOF/multihead', 'OOF/DoRA')
    start = source.index("    projections=read(names[4],'Proyeksi Multihead')")
    end = source.index("    history=read(names[4],'Fusion (S3)')", start)
    source = source[:start]+'''    try:
        dora_proof=verify_dora_exports(read(names[4],'Adapter DoRA'),read(names[4],'Parameter Dilatih dan Frozen'),cfg)
    except ValueError as error:
        raise RuntimeError('Unduhan dihentikan: laporan DoRA tidak cocok: '+str(error)) from error
'''+source[end:]
    source = source.replace("'multihead_projections_verified':True,", "**dora_proof,")
    source = source.replace("'coattention_heads':4,'multihead_projection_rows':len(projections),", '')
    index = next(i for i,c in enumerate(n['cells']) if c['metadata'].get('aes52_role') == 'definition:fit')
    n['cells'][index:index] = [
        {'cell_type':'markdown','metadata':{},'source':'## Rincian prediksi dan pemeriksaan unduhan\n\nLaporan menghitung ulang skor mentah/normalisasi/pembulatan dan memisahkan OOF dari test. ZIP dibuat setelah metrik dan adapter DoRA lolos pemeriksaan.\n'},
        {'cell_type':'code','metadata':{'aes52_role':'definition:download_helpers'},'source':source,'outputs':[],'execution_count':None}]
    n['cells'][0]['source'] = INTRO
    fingerprint = hashlib.sha256('\n'.join(c['source'] for c in n['cells'] if c['cell_type']=='code').encode()).hexdigest()
    for i, cell in enumerate(n['cells']):
        cell['source'] = cell['source'].replace(PLACEHOLDER, fingerprint)
        cell['id'] = f'e52-{i:03d}-'+hashlib.sha256(cell['source'].encode()).hexdigest()[:8]
        if cell['cell_type'] == 'code':
            ast.parse(cell['source']); cell['outputs']=[]; cell['execution_count']=None
    n['metadata']['colab']['name'] = NAME+'.ipynb'
    n['metadata']['aes_standalone']['code_sha256'] = fingerprint
    n['metadata']['aes_experiment'].update(name='E52',format_commit=PIN,format_path=BASE,format_sha256=hashlib.sha256(raw).hexdigest(),
        reporting_commit=REPORT_PIN,loss='MSE',ranking=False,encoder_adaptation='dora',dora_use_dora=True,dora_rank=8,dora_alpha=16,
        dora_lr=1e-4,s3_encoder_lr=1e-4,original_roberta_frozen=True,adapter_targets=['query','value'],
        prediction_breakdown=True,oof_predictions=True,requires_previous_zip=False,export_summary_closure_fixed=True,
        download_validation_required=True,treatment='E43 full encoder fine-tuning LR5e-6 to official PEFT DoRA rank8/alpha16 query-value LR1e-4; E43 architecture and MSE retained')
    text = json.dumps(n,ensure_ascii=False,indent=1)+'\n'
    for folder in ['notebooks/experiments','aes-learning/notebooks/experiments']:
        (ROOT/folder/(NAME+'.ipynb')).write_text(text)
    print(json.dumps({'name':NAME,'cells':len(n['cells']),'code_sha256':fingerprint}))
    return n


if __name__ == '__main__':
    build()
