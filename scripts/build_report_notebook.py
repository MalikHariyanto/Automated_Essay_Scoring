"""Notebook kecil khusus mengemas ulang hasil yang sudah ada, tanpa training."""
import json
from pathlib import Path
import textwrap

ROOT=Path(__file__).resolve().parents[1]


def build():
    cells=[]
    def md(text):cells.append(dict(cell_type='markdown',metadata={},source=textwrap.dedent(text).strip()+'\n'))
    def code(text):cells.append(dict(cell_type='code',metadata={},outputs=[],execution_count=None,source=textwrap.dedent(text).strip()+'\n'))
    md('''
    # Rapikan hasil AES yang sudah ada

    Upload **ZIP hasil lama yang berisi banyak Excel**. Notebook ini merangkumnya menjadi **lima file utama** dengan nama sheet dan kolom yang lebih jelas. Tidak memerlukan dataset, checkpoint atau GPU; **tidak melakukan training, prediksi, atau perhitungan metrik ulang**. Angka disalin dari hasil lama, bukan dibuat ulang.

    Jika proses lama belum selesai, paket hanya memuat bagian yang sudah tersedia. Output berisi data/nilai pribadi; simpan privat. Jalankan tiga cell kode berurutan.
    ''')
    code('''
    import os,sys,subprocess,json,importlib.util
    from pathlib import Path
    IN_COLAB='google.colab' in sys.modules or Path('/content').is_dir()
    ROOT=Path('/content/AES_Report_Tools') if IN_COLAB else Path.cwd()
    if not IN_COLAB:
        while not (ROOT/'src/aes').is_dir() and ROOT!=ROOT.parent:ROOT=ROOT.parent
    if not (ROOT/'.git').exists():
        if ROOT.exists() and any(ROOT.iterdir()):raise RuntimeError('Folder alat sudah berisi file lain.')
        subprocess.run(['git','clone','--branch','codex/aes-learning','--single-branch',
                        'https://github.com/MalikHariyanto/Automated_Essay_Scoring.git',str(ROOT)],check=True)
    elif IN_COLAB:
        if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
            raise RuntimeError('Folder alat mempunyai perubahan lokal; simpan dahulu.')
        subprocess.run(['git','pull','--ff-only'],cwd=ROOT,check=True)
    if importlib.util.find_spec('openpyxl') is None:
        subprocess.run([sys.executable,'-m','pip','install','openpyxl==3.1.5'],check=True)
    print('Alat siap. Tidak perlu GPU atau model bahasa.')
    ''')
    md('''
    ## Pilih hasil lama

    Cara biasa: upload ZIP `AES_300_semua_tahapan_Excel_PRIVAT.zip` yang sudah diunduh. File besar diproses langsung di runtime Colab, sehingga tidak perlu mengunggahnya ke chat.

    Jika hasil lama masih ada di runtime/Drive, isi `FOLDER_HASIL_LAMA` dengan folder run atau folder `excel`; tidak perlu upload ZIP. Folder asli tidak diubah.
    ''')
    code('''
    FOLDER_HASIL_LAMA=os.environ.get('AES_REPACK_FOLDER','')
    ZIP_HASIL_LAMA=os.environ.get('AES_REPACK_ARCHIVE','')
    DESTINATION=Path(os.environ.get('AES_REPACK_OUTPUT',str(ROOT/'outputs/hasil_ringkas'))).resolve()
    if not FOLDER_HASIL_LAMA and not ZIP_HASIL_LAMA:
        if not IN_COLAB:raise FileNotFoundError('Isi FOLDER_HASIL_LAMA atau ZIP_HASIL_LAMA.')
        from google.colab import files
        private=ROOT/'outputs/zip_lama';private.mkdir(parents=True,exist_ok=True)
        previous=Path.cwd()
        try:
            os.chdir(private);uploaded=files.upload()
        finally:os.chdir(previous)
        names=[name for name in uploaded if name.lower().endswith('.zip')]
        if len(names)!=1 or Path(names[0]).name!=names[0]:raise ValueError('Upload satu ZIP hasil AES.')
        ZIP_HASIL_LAMA=str(private/names[0])
    arguments=['--folder',FOLDER_HASIL_LAMA] if FOLDER_HASIL_LAMA else ['--archive',ZIP_HASIL_LAMA]
    command=[sys.executable,str(ROOT/'scripts/rapikan_hasil_aes.py'),*arguments,'--output',str(DESTINATION)]
    process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    log=[]
    for line in process.stdout:
        print(line,end='',flush=True);log.append(line)
    if process.wait():raise RuntimeError('Hasil belum berhasil dirapikan: '+''.join(log[-12:]))
    print('Angka asli tetap sama. Hanya susunan file, nama tabel dan tampilannya yang dirapikan.')
    ''')
    md('''
    ## Unduh hasil ringkas

    Lima file pada run lengkap:

    | File | Untuk apa? |
    | --- | --- |
    | 01 Data dan Tokenisasi | Melihat jawaban, nilai guru, pembagian data, dan token |
    | 02 Vektor Jawaban | Membandingkan embedding setiap jawaban sebelum/sesudah latihan dan fusion |
    | 03 Vektor Kata dan Kalimat | Melihat satu contoh dan perbandingan kata berulang |
    | 04 Prediksi dan Evaluasi | Buka **Perbandingan Model** untuk hasil utama, lalu **Nilai Guru dan Prediksi** |
    | 05 Riwayat Pelatihan | Riwayat epoch, hasil validasi dan pengaturan eksperimen |

    Setiap workbook diawali **Panduan**. Vektor tetap lengkap; `Komponen 1`, `Komponen 2`, dst. adalah angka dalam vektor, bukan skor rubrik.
    ''')
    code('''
    summary=json.loads((DESTINATION/'ringkasan.json').read_text())
    print('Paket siap:',len(summary['files']),'Excel.')
    for file in summary['files']:print('•',Path(file).name)
    if IN_COLAB:
        from google.colab import files
        files.download(summary['archive'])
    else:print('ZIP:',summary['archive'])
    ''')
    nb=dict(cells=cells,metadata={'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python'}},nbformat=4,nbformat_minor=4)
    for directory in ['notebooks','aes-learning/notebooks']:
        (ROOT/directory/'04_Rapikan_Hasil_AES.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
    return nb


if __name__=='__main__':print('Notebook hasil lama:',len(build()['cells']),'cell.')
