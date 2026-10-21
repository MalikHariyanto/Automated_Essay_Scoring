"""Bangun eksperimen terpisah; tidak menulis notebook acuan/semi-final."""
import ast
import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT/'notebooks/experiments/AES_300_S3_DoubleFrozen_MLP.ipynb'


def functions(path, names):
    source = (ROOT/path).read_text()
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise ValueError('Definisi belum lengkap: '+str(names))
    return '\n\n\n'.join(ast.get_source_segment(source, node) for node in nodes)


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Anchor notebook tidak unik: '+old[:100])
    return text.replace(old, new)


def build():
    base = ROOT/'notebooks/03_AES_dataset_guru_300.ipynb'
    original = base.read_bytes()
    notebook = copy.deepcopy(json.loads(original))
    old_hash = notebook['metadata']['aes_standalone']['code_sha256']
    cleaner = (ROOT/'src/aes/teacher_score/drive_checkpoint_cleanup.py').read_text()
    checkpoint_functions = functions('src/aes/teacher_score/experimental_checkpoints.py', [
        'checkpoint_file_proof', 'drive_cleanup_enabled', 'request_checkpoint_cleanup',
        'retry_pending_checkpoint_cleanup', 'sweep_completed_checkpoint_cleanup',
        'save_training_progress', 'clear_training_progress'])

    for cell in notebook['cells']:
        text = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
        text = text.replace('AES_300_MLP_Privat', 'AES_300_DoubleFrozen_MLP_Privat')
        text = text.replace('/content/AES_300_MLP', '/content/AES_300_DoubleFrozen_MLP')
        text = text.replace("outputs/AES_300_MLP", "outputs/AES_300_DoubleFrozen_MLP")
        if cell['cell_type'] == 'code':
            text = text.replace(old_hash, '__NOTEBOOK_CODE_SHA256__')
            text = text.replace("lesson_version='s3_mlp_account2_v1'", "lesson_version='s3_double_frozen_mlp_autocleanup_v1'")
            if 'WORKER_SOURCE=' in text:
                worker_node = next(n for n in ast.parse(text).body if isinstance(n, ast.Assign))
                worker = ast.literal_eval(worker_node.value)
                worker = replace_once(worker, "namespace['show'] = show", '''
def request_drive_cleanup(request):
    # Kredensial tetap di kernel. Worker hanya mengirim identitas run/tahap.
    import uuid
    request_id = uuid.uuid4().hex
    event('drive_cleanup', request_id=request_id, request=request)
    response = sys.stdin.readline()
    if not response:
        raise RuntimeError('Koneksi pembersihan terputus; checkpoint dipertahankan.')
    reply = json.loads(response)
    if reply.get('command') != 'drive_cleanup_response' or reply.get('request_id') != request_id:
        raise RuntimeError('Respons pembersihan tidak cocok; checkpoint dipertahankan.')
    return reply

namespace['_drive_cleanup'] = request_drive_cleanup
namespace['show'] = show'''.strip())
                text = '# Penghubung proses Python: perhitungan dan permintaan pembersihan memakai worker yang sama.\nWORKER_SOURCE=' + repr(worker) + '\n'
            if text.startswith("PYTHON = os.environ.get('AES300_PYTHON'"):
                text = cleaner + '\n\n' + text
                text = replace_once(text, "print('Penyimpanan data, checkpoint dan Excel:',PERSISTENT_ROOT)", '''
print('Penyimpanan data, checkpoint dan Excel:',PERSISTENT_ROOT)
_aes_cleaner = None
if IN_COLAB:
    # Drive API hanya di kernel Colab, tidak di worker numerik dan tidak ditulis ke file.
    from google.colab import auth
    import google.auth
    from googleapiclient.discovery import build
    from google_auth_httplib2 import AuthorizedHttp
    import httplib2
    import time
    auth.authenticate_user()
    credentials, _ = google.auth.default()
    authorized_http = AuthorizedHttp(credentials, http=httplib2.Http(timeout=120))
    drive_api = build('drive', 'v3', http=authorized_http, cache_discovery=False)
    _aes_cleaner = DriveCheckpointCleaner(drive_api, PERSISTENT_ROOT)
    for attempt in range(5):
        try:
            _aes_cleaner.preflight()
            break
        except CleanupDeferred:
            if attempt == 4:
                raise
            print('Menunggu folder eksperimen terlihat di Drive...', flush=True)
            time.sleep(2 ** (attempt + 1))
    print('Pembersihan otomatis siap: hanya checkpoint usang eksperimen ini yang dihapus permanen.')
else:
    print('Mode lokal: checkpoint usang dibersihkan di filesystem lokal; tidak mengakses Drive API.')
'''.strip())
                text = replace_once(text, "'AES300_REQUIRE_GPU':'1' if IN_COLAB else os.environ.get('AES300_REQUIRE_GPU','0'),",
                                    "'AES300_REQUIRE_GPU':'0',\n               'AES300_DRIVE_CLEANUP':'1' if IN_COLAB else '0',")
                text = replace_once(text, "        if event['kind']=='table':display_aes_table(event)", '''
        if event['kind']=='table':display_aes_table(event)
        if event['kind']=='drive_cleanup':
            # Selalu balas, termasuk saat API gagal, agar worker tidak menunggu selamanya.
            response = {'command':'drive_cleanup_response','request_id':event['request_id']}
            try:
                if _aes_cleaner is None:
                    raise RuntimeError('Drive cleaner belum diinisialisasi.')
                response.update(ok=True, result=_aes_cleaner(event['request']))
            except Exception as error:
                status = getattr(getattr(error, 'resp', None), 'status', None)
                retryable = isinstance(error, CleanupDeferred) or status in (429, 500, 502, 503, 504)
                response.update(ok=False, retryable=retryable, error=str(error))
            finally:
                _aes_process.stdin.write(json.dumps(response)+'\\n')
                _aes_process.stdin.flush()
'''.rstrip())
            if 'class SingleScoreModel' in text:
                text = replace_once(text, '''            # Fine-tuning: bobot RoBERTa boleh berubah ketika optimizer menjalankan langkah latihan.
            roberta.requires_grad_(True)''', '''            # S1 tetap fine-tuned sebagai kontrol; S3 membekukan seluruh encoder RoBERTa.
            roberta.requires_grad_(scenario == 'S1')
            if scenario == 'S3':
                roberta.eval()''')
                text = replace_once(text, '    def forward(self, windows=None, mapping=None, n_essays=None, bge=None, return_features=False):', '''    def train(self, mode=True):
        # model.train() tidak boleh mengaktifkan kembali dropout encoder S3 yang frozen.
        super().train(mode)
        if self.scenario == 'S3' and self.roberta is not None:
            self.roberta.eval()
        return self

    def forward(self, windows=None, mapping=None, n_essays=None, bge=None, return_features=False):''')
                text = replace_once(text, '            cls = self.roberta(**windows).last_hidden_state[:, 0, :]', '''            # Gradien hanya melalui encoder S1; proyeksi/LN/MLP S3 tetap belajar di luar blok ini.
            with torch.set_grad_enabled(torch.is_grad_enabled() and self.scenario == 'S1'):
                cls = self.roberta(**windows).last_hidden_state[:, 0, :]''')
            if 'specifications=pd.DataFrame' in text:
                text = text.replace("'RoBERTa ikut dilatih; BGE-M3 tetap'", "'RoBERTa dan BGE-M3 frozen; proyeksi, LayerNorm tambahan dan MLP dilatih'")
            if "gpu=check_gpu(require_gpu=RUN_MODELS" in text:
                text = replace_once(text,
                    "# Colab harus mendeteksi GPU sebelum latihan panjang dimulai. Pengujian lokal tetap boleh CPU.\ngpu=check_gpu(require_gpu=RUN_MODELS and os.environ.get('AES300_REQUIRE_GPU')=='1')",
                    "# False: CPU boleh dipakai. True: hentikan sebelum training jika GPU belum aktif.\nREQUIRE_GPU=False\n# device='auto' di atas memilih CUDA jika tersedia; jika tidak, memakai CPU.\ngpu=check_gpu(require_gpu=RUN_MODELS and REQUIRE_GPU)\nif RUN_MODELS and not gpu['cuda_available']:\n    print('Mode CPU aktif. Training tetap berjalan, tetapi lebih lama dibanding GPU; progres tetap disimpan otomatis.', flush=True)")
            if 'def vector_lab(' in text:
                text = text.replace("source = cfg['vector_source']+' fine-tuned; sha256='+digest(checkpoint)",
                                    "source = cfg['vector_source']+(' encoder frozen; head/proyeksi dilatih; sha256=' if cfg['vector_source']=='S3' else ' fine-tuned; sha256=')+digest(checkpoint)")
                text = text.replace("label += ' sebelum latihan' if cfg['vector_source']=='pretrained' else f' {cfg[\"vector_source\"]} setelah latihan'",
                                    "label += ' sebelum latihan' if cfg['vector_source']=='pretrained' else ' S3 encoder frozen; head/proyeksi setelah latihan' if cfg['vector_source']=='S3' else ' S1 setelah fine-tuning'")
            if 'def save_training_progress(' in text:
                tree = ast.parse(text.removeprefix('%%aes\n'))
                blocks = []
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef) and node.name not in ('save_training_progress', 'clear_training_progress'):
                        blocks.append(ast.get_source_segment(text.removeprefix('%%aes\n'), node))
                text = '%%aes\n# Checkpoint lengkap, dua generasi, dan pembersihan permanen yang dibatasi pada eksperimen ini.\nimport time\n\n' + checkpoint_functions + '\n\n' + '\n\n\n'.join(blocks) + '\n'
                text = text.replace("if parameter.requires_grad and group not in selected_parameters:",
                                    "if (parameter.requires_grad or group == 'roberta') and group not in selected_parameters:")
                text = text.replace("'parameter_samples_changed': changed, 'BGE_encoder_trained': False,",
                                    "'parameter_samples_changed': changed, 'BGE_encoder_trained': False,\n            'RoBERTa_encoder_trained': scenario == 'S1',\n            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),")
            if 'def train_all(' in text:
                text = replace_once(text, "    final_path = output/'final_report.json'", "    retry_pending_checkpoint_cleanup(output)\n    final_path = output/'final_report.json'")
                text = replace_once(text, "        return {**report, 'reused_completed_run': True}",
                                    "        sweep_completed_checkpoint_cleanup(output)\n        return {**report, 'reused_completed_run': True}")
            if 'def export_training_event(' in text:
                text = text.replace("encoder_fine_tuned=scenario != 'S2'", "encoder_fine_tuned=scenario == 'S1'")
            if 'def answer_vectors_report(' in text:
                text = text.replace("['RoBERTa','768 angka per jawaban. Bandingkan ID Jawaban yang sama sebelum dan sesudah latihan.']",
                                    "['RoBERTa','768 angka per jawaban. S1 fine-tuned; encoder S3 frozen sehingga vektor mentahnya tetap pretrained. LayerNorm tambahan, proyeksi dan head S3 tetap dilatih.']")
                text = text.replace("['Sebelum Konteks','Lookup embedding kata sebelum penambahan konteks; hanya perlu disajikan sekali untuk tiap encoder pretrained.']",
                                    "['Sebelum Konteks','Lookup embedding kata sebelum penambahan konteks; hanya perlu disajikan sekali untuk tiap encoder pretrained. Encoder S3 frozen; vektor kata RoBERTa S3 tidak menjadi vektor fusion per kata.']")
            cell['outputs'] = []
            cell['execution_count'] = None
        else:
            if text.startswith('# AES guru 300'):
                text = '''# AES 300 · Eksperimen S3 double frozen + MLP

Notebook ini terpisah dari acuan/semi-final S3 MLP. **S1** tetap RoBERTa fine-tuned + head linear. **S2** tetap BGE-M3 frozen + MLP. **S3 eksperimen** membekukan kedua encoder; proyeksi BGE, LayerNorm tambahan dan MLP setelah concat tetap dilatih. Ridge dan MLP tambahan pada S1 belum diterapkan.

Alur sama dengan acuan: setup → dataset → split/tokenisasi → embedding → model → training → prediksi/evaluasi → lima Excel. Dataset hanya satu skor guru; tidak dibuat empat label rubrik baru. Batas 25 epoch, patience 5, learning rate, batch, split 80/20 dan 5-fold CV tetap. S1 dan S2 dijalankan sebagai pembanding; ini run eksperimen baru, bukan kelanjutan bobot S3 fine-tuned.

**Cara menjalankan:** runtime CPU maupun GPU didukung; T4 disarankan untuk mempercepat training. Jalankan dari atas, sambungkan Drive dan otorisasi API pada akun yang memiliki folder eksperimen. `REQUIRE_GPU=False` mengizinkan CPU; `device='auto'` memilih GPU jika tersedia dan CPU jika tidak. Upload `fix_dataset_300.xlsx` jika diminta. Data, checkpoint dan hasil berada di `MyDrive/AES_300_DoubleFrozen_MLP_Privat`; acuan `AES_300_MLP_Privat` tidak dibersihkan atau ditimpa.

**Pembersihan otomatis:** setelah checkpoint lengkap dan cadangannya terverifikasi melalui Drive API, hanya epoch usang dari tahap/run ini yang dihapus permanen, termasuk yang sudah di Sampah. Jika penunjuk progres di server tertinggal, checkpoint yang masih dirujuknya juga dilindungi: maksimal dua snapshot terbaru ditambah dua rujukan server. Setelah sinkron, versi usang bisa dilepas. Saat fold selesai, checkpoint sementaranya dilepas setelah hasil CV terverifikasi. Saat refit selesai, checkpoint sementara dilepas setelah model final lolos checksum. Model final, hasil CV, dataset, Excel dan file eksperimen lain tetap disimpan. Sampah Drive global tidak dikosongkan. Jika bukti belum lengkap, pembersihan mencoba kembali; jika tetap gagal, training berhenti dengan progres tersimpan agar disk tidak terus penuh. Jalankan notebook yang sama dari atas untuk mencoba pembersihan dan melanjutkan. Untuk run yang sudah berjalan dengan versi sebelumnya, gunakan petunjuk perbaikan kernel di GitHub agar fingerprint/run lama tetap dipakai.

Notebook mandiri: semua kode runtime ada di cell, tanpa clone/pull atau import proyek. Internet tetap diperlukan untuk pustaka dan model Hugging Face. Notebook publik tidak menyertakan data/output siswa. Hasil utama tetap lima Excel; semua vektor utama dipertahankan. S3 fusion terjadi pada tingkat jawaban; vektor kata S3 adalah keluaran cabang RoBERTa frozen, bukan fusion per kata.
'''
            if '## 10 · Model' in text:
                text = '''## 10 · Model S1/S2/S3 dan bagian yang belajar

S1 tetap RoBERTa fine-tuned → mean CLS 768 → dropout → linear satu skor. S2 tetap BGE-M3 frozen → CLS + L2 1024 → MLP. S3 memakai RoBERTa frozen 768 → LayerNorm tambahan, serta BGE frozen 1024 → proyeksi 256 + LayerNorm; concat 1024 → Linear 256 → GELU → dropout → Linear 1.

Pada S3, kedua encoder tidak menerima gradien. Encoder RoBERTa tetap mode eval meskipun `model.train()` dipanggil; dropout MLP tetap aktif saat training. Proyeksi, LayerNorm tambahan dan MLP menerima gradien. Tidak ada Ridge, GRU atau perubahan pooling.
'''
            if '## 12 · Pelatihan' in text:
                text += '\n\nPembersihan checkpoint otomatis berlaku untuk seluruh tahap S1/S2/S3 di folder eksperimen ini. Dua snapshot terbaru dipertahankan pada tahap aktif. Bila pembersihan tidak dapat diverifikasi, proses berhenti sebelum training berikutnya; hasil tahap dan antrean pembersihan tetap disimpan untuk resume.\n'
            if '## 14 · Lihat vektor' in text:
                text += '\n\nEncoder RoBERTa S3 frozen: sheet “RoBERTa S3 Setelah Latihan” berarti sesudah head/proyeksi dilatih; vektor mentah encoder tetap sama dengan pretrained. Hasil LayerNorm tambahan dan concat tetap dapat berubah.\n'
        cell['metadata'] = {}
        cell.pop('attachments', None)
        cell['source'] = text

    normalized = [ast.dump(ast.parse(c['source'].removeprefix('%%aes\n')), include_attributes=False)
                  for c in notebook['cells'] if c['cell_type'] == 'code']
    fingerprint = hashlib.sha256('\n'.join(normalized).encode()).hexdigest()
    for cell in notebook['cells']:
        cell['source'] = cell['source'].replace('__NOTEBOOK_CODE_SHA256__', fingerprint)
    notebook['metadata']['aes_standalone']['code_sha256'] = fingerprint
    notebook['metadata']['colab']['name'] = DESTINATION.name
    notebook['metadata']['aes_experiment'] = {
        'name': 'S3_DOUBLE_FROZEN_MLP', 's1_roberta': 'fine_tuned', 's3_roberta': 'frozen',
        'bge': 'frozen', 's3_head': '1024 -> 256 -> GELU -> dropout -> 1',
        'ridge': False, 'permanent_checkpoint_cleanup': True,
        'drive_root': 'AES_300_DoubleFrozen_MLP_Privat'}
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n')
    assert base.read_bytes() == original
    print('Notebook eksperimen:', DESTINATION.relative_to(ROOT))
    print('Fingerprint:', fingerprint)
    return DESTINATION


if __name__ == '__main__':
    build()
