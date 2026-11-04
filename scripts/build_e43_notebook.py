"""E43: E42 with masked mean over matched response tokens; native S3 only."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
PIN='5a5cc596059bb27f020a3a87dc49abbdecc80103'
BASE='notebooks/experiments/AES_300_E42_E41_S3_ScalarMix4_Gamma_Native_Learnable_GPU.ipynb'
NAME='AES_300_E43_E42_S3_MaskedMeanPooling_Native_Learnable_GPU'
PLACEHOLDER='__NOTEBOOK_CODE_SHA256__'

CV_ERRORS=r'''    # Ringkasan tambahan hanya dari prediksi VALIDASI yang sudah tersimpan.
    error_rows = []
    all_errors = []
    for file in sorted(source.files):
        match = re.fullmatch(r'09_S3_fold_(\d+)\.xlsx', file)
        if not match or not source.has(file, 'Prediksi_validation'):
            continue
        errors = np.asarray([abs(float(row['prediksi_mentah'])-float(row['skor_guru']))
                             for row in source.records(file, 'Prediksi_validation')])
        if not len(errors):
            continue
        all_errors.extend(errors.tolist())
        error_rows.append([int(match[1]), len(errors), float(errors.mean()),
                           int((errors<=1).sum()), int((errors<=5).sum()), int((errors<=10).sum())])
    if all_errors:
        errors = np.asarray(all_errors)
        error_rows.append(['OOF development', len(errors), float(errors.mean()),
                           int((errors<=1).sum()), int((errors<=5).sum()), int((errors<=10).sum())])
        sheets['MAE dan Toleransi CV'] = Table(
            ['Fold','Jumlah Jawaban','MAE Validasi (poin)','Dalam ±1 poin','Dalam ±5 poin','Dalam ±10 poin'], error_rows)
'''


def build():
    raw=subprocess.check_output(['git','show',PIN+':'+BASE],cwd=ROOT)
    n=copy.deepcopy(json.loads(raw));old_fp=n['metadata']['aes_standalone']['code_sha256']
    for c in n['cells']:
        role=c['metadata'].pop('aes42_role','')
        if role:c['metadata']['aes43_role']=role
        s=c['source'] if isinstance(c['source'],str) else ''.join(c['source'])
        s=s.replace(old_fp,PLACEHOLDER).replace(Path(BASE).stem,'__E43_NAME__').replace('E42','E43').replace('__E43_NAME__',NAME)
        s=s.replace('bigru_coattention_pooling','bigru_coattention_mean_pooling')
        if role=='configuration_apply':
            s=s.replace('e42_native_scalar_mix4_gamma_v1','e43_native_masked_mean_v1')
            s=s.replace("{'Tahap':'Bottleneck attention pooling','Dimensi':cfg['pooling_attention_hidden']}",
                        "{'Tahap':'Masked mean token valid (tanpa bottleneck)','Dimensi':2*cfg['s3_sequence_hidden']}")
        if role=='hyperparameters':
            lines=s.splitlines()
            lines=[line.split('#')[0].rstrip()+'  # Ukuran init acuan; layer skor attention dihapus sebelum training.' if line.startswith('POOLING_ATTENTION_HIDDEN =') else line for line in lines]
            s='\n'.join(lines)+'\n'
        if role=='definition:AnswerBiGRUCoAttentionPool':
            c['metadata']['aes43_role']='definition:AnswerBiGRUCoAttentionMeanPool'
            s=s.replace('class AnswerBiGRUCoAttentionPool(', 'class AnswerBiGRUCoAttentionMeanPool(')
            s=s.replace('E30: E26 + co-attention respons128 ↔ referensi128, lalu attention pooling128→64→1.',
                        'E43: BiGRU + co-attention respons–referensi, lalu masked mean token matching128.')
            anchor='        self.last_response_coattention_weights = None\n'
            assert s.count(anchor)==1
            s=s.replace(anchor,anchor+'''        # Semua bobot bersama dibuat dengan urutan RNG E42 yang sama.
        # Dua layer skor pooling acuan kemudian dihapus; tidak ada parameter attention pooling yang dilatih.
        del self.attention_hidden
        del self.attention_score
''')
            old='''        # Attention pooling E26 sekarang merangkum keluaran matching co-attention.
        pooling_scores = self.attention_score(torch.tanh(self.attention_hidden(matched))).squeeze(-1).float()
        pooling_weights = torch.softmax(pooling_scores.masked_fill(~valid, -torch.inf), dim=1)
'''
            new='''        # Setiap token matching yang valid mendapat bobot 1/N.
        # Padding berbobot 0; mean ini merangkum token dalam satu window.
        pooling_weights = valid.float() / lengths.to(hidden.device).unsqueeze(1)
'''
            assert s.count(old)==1;s=s.replace(old,new)
        if role=='definition:SingleScoreModel':
            s=s.replace('AnswerBiGRUCoAttentionPool(', 'AnswerBiGRUCoAttentionMeanPool(')
        if role=='definition:export_final_features':
            s=s.replace("'attention_weight':float(weights[window, position])", "'pooling_weight':float(weights[window, position])")
            s=s.replace("sheets['Bobot_token_jawaban']", "sheets['Bobot_mean_jawaban']")
        if role=='presentation_helpers':
            s=s.replace("LABELS.update({'attention_weight':'Bobot Attention'})", "LABELS.update({'attention_weight':'Bobot Attention','pooling_weight':'Bobot Pooling'})")
        if role=='definition:answer_vectors_report':
            s=s.replace("'Bobot_token_jawaban','Bobot Attention S3'", "'Bobot_mean_jawaban','Bobot Mean Pooling S3'")
            s=s.replace('additive attention pooling E26 merangkum hasil matching128', 'masked mean merangkum semua token matching128yang valid')
            s=s.replace('Bobot Attention S3 adalah pooling respons SETELAH matching.', 'Bobot Mean Pooling S3 adalah1/Npada token respons valid SETELAH matching,padding0;tidak dipelajari.')
            s=s.replace("['Dua tahap attention'", "['Co-attention dan mean pooling'")
        if role=='definition:training_report':
            s=s.replace("'pooling_attention_hidden':'Hidden attention pooling'", "'pooling_attention_hidden':'Ukuran init attention acuan (dihapus)'")
            assert s.count('    return sheets\n')==1;s=s.replace('    return sheets\n',CV_ERRORS+'    return sheets\n')
        if role=='architecture_export':
            s=s.replace('attention pooling','masked mean pooling').replace('Scalar Mix4 →','Scalar Mix4 + gamma →')
        if role=='definition:evaluation_report':s=s.replace('Co-Attention + Attention Pooling', 'Co-Attention + Masked Mean Pooling')
        if c['cell_type']=='markdown':
            s=s.replace('berbeda dari attention pooling token.', 'berbeda dari masked mean pooling antartoken setelah matching.')
            if '## 19 · Arsitektur S3' in s:
                s='<a id="arsitektur"></a>\n## 19 · Arsitektur S3 E43\n\nRoBERTa → Scalar Mix empat layer + gamma → token 768. Respons → BiGRU 64 per arah → 128; referensi → Linear 128. Co-attention dua arah → matching 640→128 → masked mean token valid 128 → Linear 768 + LayerNorm. BGE frozen 1024 → proyeksi 256 + LayerNorm. Concat 1024 → MLP 256 → GELU → Dropout 0,1 → satu skor.\n\nCo-attention tetap dipelajari. Pooling akhir memakai bobot seragam 1/N untuk token valid dan 0 untuk padding.\n'
            if '### `AnswerBiGRUCoAttentionPool`' in s:
                s='### `AnswerBiGRUCoAttentionMeanPool`\n\nSesudah BiGRU dan matching co-attention, semua token respons valid dirata-ratakan. Layer `attention_hidden` dan `attention_score` pooling dihapus setelah inisialisasi acuan agar bobot bersama serta keadaan pengacakan tetap sama untuk seed yang sama. Seluruh parameter yang tersisa ikut jalur prediksi; tidak ada skor pooling yang dipelajari.\n'
            if '### `AnswerBiGRUAttention`' in s:
                s='### `AnswerBiGRUAttention` · dasar inisialisasi acuan\n\nKelas dasar ini menyediakan inisialisasi BiGRU/proyeksi yang sama dengan E42. Subclass E43 menghapus dua layer skor attention pooling dan mendefinisikan forward masked mean sendiri. Forward attention pada kelas dasar tidak dipanggil oleh E43.\n'
            if '## Rujukan dan batas kesimpulan' in s:
                s+='\n**Perlakuan E43:** Taghipour&Ng(2016), [A Neural Approach to Automated Essay Scoring](https://aclanthology.org/D16-1193/), dan [repositori penulis](https://github.com/nusnlp/nea) memakai MeanOverTime untuk merangkum urutan recurrent pada AES. [Dong,Zhang&Yang(2017)](https://aclanthology.org/K17-1017/) menjadi rujukan attention untuk AES. E43 mengadaptasi mean pooling pada keluaran matching co-attention,berbeda dari arsitektur paper;paper tidak membuktikan mean akan unggul pada fusion ini.\n'
        c['source']=s
    n['cells'][0]['source']='''# E43 · E42 + masked mean pooling · S3 Native Learnable

**Perubahan dari E42: attention pooling terakhir diganti masked mean atas token hasil matching co-attention.** Setiap token valid berbobot 1/N, padding 0. Ringkasan tetap 128 dimensi sebelum proyeksi 768. Mean ini merangkum token dalam satu window; mean antarwindow tetap seperti E42.

Alur S3: **RoBERTa → Scalar Mix empat layer + gamma → BiGRU + co-attention → masked mean → gabung BGE → MLP**. Bobot respons–referensi pada co-attention tetap dipelajari. Yang menjadi seragam hanya bobot pooling setelah matching.

Layer `attention_hidden` dan `attention_score` pooling dihapus sebelum optimizer dibuat. Urutan inisialisasi acuan dipertahankan agar semua bobot bersama dan keadaan pengacakan dimulai identik dengan E42 pada seed yang sama. Jumlah parameter berkurang 8.320 sebagai konsekuensi penggantian ini. Parameter `POOLING_ATTENTION_HIDDEN` hanya menyatakan ukuran inisialisasi acuan, bukan bottleneck aktif E43.

| Pengaturan | Nilai |
|---|---|
| LR seluruh encoder RoBERTa | 5e-6 |
| LR BiGRU, co-attention, proyeksi, norm, MLP, logit dan gamma | 1e-4 |
| BiGRU / dropout MLP | 64 per arah / 0,1 |
| Scalar Mix | Empat logit awal 0 → bobot 0,25; gamma awal 1 |
| Pelatihan | S3 saja; 5 fold sekali; maksimal 25 epoch; patience 5 |
| Setelah CV | Satu refit dengan ceil(median best epoch); satu evaluasi test |

AdamW, weight decay 0,01, warmup 10%, scheduler linear, clipping 1, batch efektif 8/microbatch 1 tetap seperti E42. Tidak ada loop multi-seed atau pelatihan ulang kontrol otomatis. `TRAIN_SEED` tetap satu konstanta untuk pengacakan.

**Input X:** `soal`, `jawaban_referensi`, `response_text`. **Target y:** `rater` 0–100, digunakan untuk loss dan evaluasi, tidak masuk tokenizer. RoBERTa tetap dilatih dan BGE-M3 tetap frozen. Python biasa per section, tanpa magic atau worker tersembunyi.

**Jalankan:** pilih T4 GPU → Run all → unggah `fix_dataset_300.xlsx`. Output tetap lima Excel dalam satu ZIP. Excel 02 → **Bobot Mean Pooling S3** menyimpan 1/N. **Bobot CoAttention Respons S3** dan **Bobot Referensi S3** tetap menyimpan bobot co-attention.

Excel 05 → **Bobot Scalar Mix** menyimpan empat bobot dan gamma. Sheet baru **MAE dan Toleransi CV** merangkum lima fold dan seluruh prediksi validasi development (OOF). MAE dan jumlah selisih ≤1/5/10 poin dihitung dari prediksi desimal 0–100 tanpa pembulatan. Ringkasan ini tidak membaca test dan tidak mengubah empat metrik utama.

Landasan mean-over-time: Taghipour & Ng (2016), AES. Penerapan sesudah matching pada fusion ini tetap hipotesis; tidak dijamin mengalahkan E42/E30. Belum ada hasil penelitian E43. Test historis yang sudah dilihat berulang kali tetap bersifat eksploratif. Folder dan checkpoint E43 terpisah; unduh ZIP sebelum runtime dihapus.

Peta belajar: [nama model](#nama-model), [kolom X/y](#kolom-xy), [hyperparameter](#hyperparameter), [arsitektur](#arsitektur), [training](#training), [evaluasi](#evaluasi), [unduh](#unduh).
'''
    fp=hashlib.sha256('\n'.join(c['source'] for c in n['cells'] if c['cell_type']=='code').encode()).hexdigest()
    for i,c in enumerate(n['cells']):
        c['source']=c['source'].replace(PLACEHOLDER,fp);c['id']=f'e43-{i:03d}-'+hashlib.sha256(c['source'].encode()).hexdigest()[:8]
        if c['cell_type']=='code':ast.parse(c['source']);c['outputs']=[];c['execution_count']=None
    n['metadata']['colab']['name']=NAME+'.ipynb';n['metadata']['aes_standalone']['code_sha256']=fp
    n['metadata']['aes_experiment'].update(name='E43',s3='E42 Scalar Mix gamma + BiGRU co-attention masked mean pooling',
        s3_pooling='bigru_coattention_mean_pooling',attention_pooling=False,pooling_rule='uniform over valid matched response tokens',
        removed_parameters_vs_e42=8320,format_commit=PIN,format_path=BASE,format_sha256=hashlib.sha256(raw).hexdigest(),
        treatment='Replace learned attention pooling after matching with masked mean; all shared initialization and E42 settings retained')
    # These inherited counts described adding gamma to E41, not the E43 treatment.
    n['metadata']['aes_experiment'].pop('added_trainable_parameters',None)
    n['metadata']['aes_experiment'].pop('added_parameters_vs_e41',None)
    n['metadata']['aes_experiment'].update(scalar_mix_trainable_parameters=5,net_parameters_vs_e30=-8315)
    output=json.dumps(n,ensure_ascii=False,indent=1)+'\n'
    for folder in ['notebooks/experiments','aes-learning/notebooks/experiments']:(ROOT/folder/(NAME+'.ipynb')).write_text(output)
    print(json.dumps({'name':NAME,'cells':len(n['cells']),'code_sha256':fp}));return n


if __name__=='__main__':build()
