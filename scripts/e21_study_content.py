"""Materi E21: tiga input, satu label; tanpa angka/teks siswa individual."""
import copy
from mlp_study_content import CHAPTERS as BASE_CHAPTERS, GLOSSARY as BASE_GLOSSARY


def curriculum(snapshot):
    c=copy.deepcopy(BASE_CHAPTERS);g=copy.deepcopy(BASE_GLOSSARY)
    lessons=[
        ('Hasil E21', 'Baca hasil satu eksperimen dengan tiga skenario.',
         ['3 input X','Satu label y','15 hasil CV','60 jawaban test'],
         ['E21 memakai soal, jawaban referensi dan jawaban siswa pada seluruh skenario. Label tetap rater, satu skor guru 0–100.',
          'S1/S2/S3 dilatih ulang pada run E21. Bandingkan empat metrik dan pola error; satu nilai terbaik belum membuktikan keunggulan statistik.',
          'Hasil E22 belum tersedia. Buku ini hanya berisi bukti E21 dan tidak mengisi angka untuk E22.']),
        ('Folder lokal & kode lengkap', 'Pisahkan notebook, bobot pretrained, checkpoint dan laporan.',
         ['Kode notebook','Disk runtime lokal','Checkpoint dua slot','ZIP lima Excel'],
         ['Kode tersedia lengkap pada notebook rujukan. Nama checkpoint Hugging Face adalah alamat bobot, bukan objek model yang sudah dimuat.',
          'Tidak ada mount atau autosave Drive. File lokal dapat hilang bila runtime dihapus; ini memengaruhi penyimpanan, bukan definisi input/label.']),
        ('Setup Colab & GPU', 'Pahami setup lingkungan dan pemeriksaan CUDA.',
         ['Python Colab','Pustaka terpisah','Choose Files','Worker perhitungan'],
         ['Pilih T4 GPU di web Colab sebelum Run all. Worker terpisah menjaga import numerik konsisten antar-cell %%aes.',
          'Upload berjalan pada proses notebook utama. GPU dipilih oleh konfigurasi dan diperiksa sebelum encoder dimuat. Nama ZIP saja tidak membuktikan perangkat aktual runtime.']),
        ('Model, LR & tiga input', 'Lacak X, y dan parameter yang benar-benar dilatih.',
         ['X: tiga kolom','y: rater','LR encoder 5e-6','LR head/proyeksi 1e-4'],
         ['X berisi soal, jawaban_referensi dan response_text; y berasal dari rater. Nilai guru tidak masuk tokenizer atau cache fitur.',
          'S1/S3 memakai LR encoder RoBERTa 5e-6. Head dan proyeksi 1e-4. Encoder BGE frozen; MLP/proyeksi sesudah BGE tetap belajar.',
          'E21 memakai single_sequence: konteks dan jawaban disambung dalam satu urutan. Ini belum context pair E22.']),
        ('Baca data & label', 'Ikuti baris Excel menjadi contoh model dan target.',
         ['Soal','Referensi','Jawaban siswa','rater → bundle.y'],
         ['read_dataset menyimpan tiga teks terpisah pada bundle.rows. bundle.texts mengambil jawaban; contexts_from_bundle mengambil soal dan referensi pada urutan yang sama.',
          'Cleaning menormalkan Unicode/newline dan membuang karakter kontrol. Ejaan siswa dan skor guru tidak ditulis ulang.',
          'Email dan duplikat dipakai untuk kelompok pencegah kebocoran; bukan input encoder. Tiga input tidak berarti tiga label atau tiga encoder.']),
        ('240 development / 60 test', 'Bedakan train fold, validation fold dan test.',
         ['300 jawaban','240 development','5 fold pada development','60 test'],
         ['Setiap fold memulai model baru. Anggota train dan validasi berbeda pada fold itu; kelompok yang berkaitan tidak dipisah sembarangan.',
          'Lima fold berbeda dari maksimal 25 epoch per fold. Data test tidak memilih epoch, LR atau checkpoint. Soal dapat muncul pada kedua bagian; ini bukan uji soal baru.']),
        ('Token tiga teks & window', 'Pahami isi yang diterima setiap encoder.',
         ['Soal + referensi','Penanda jawaban','Token jawaban','Special + window'],
         ['E21 menyambung token konteks dan jawaban sebelum menambahkan special token satu urutan. RoBERTa dan BGE memakai kosakata sendiri.',
          'Anggaran soal 64 dan konteks 192 token termasuk prefix. Kapasitas jawaban = batas model − konteks dibaca − penanda jawaban − overhead cadangan 4.',
          'Overhead pasangan dicadangkan juga untuk E21 agar window/cakupan sebanding dengan E22 kelak. E21 memakai dua special token aktual; seluruh token jawaban tetap dicakup lewat overlap.']),
        ('RoBERTa sebelum training', 'Bedakan jumlah token dari dimensi embedding.',
         ['Window tiga teks','RoBERTa pretrained','CLS per window','Mean → 768 fitur'],
         ['Panjang jawaban mengubah jumlah token/window. Panjang fitur RoBERTa tetap 768 karena hidden_size model, bukan karena setiap jawaban mempunyai 768 kata.',
          'Ekstraksi awal memakai eval/no_grad. RoBERTa S1/S3 tetap ikut fine-tuning saat training supervised.']),
        ('BGE frozen membaca tiga teks', 'Bedakan encoder tetap dari lapisan prediksi yang belajar.',
         ['Tiga teks','BGE frozen','Mean CLS + L2','1024 fitur'],
         ['BGE E21 juga menerima soal dan referensi. Frozen hanya berarti bobot encodernya tidak diperbarui.',
          'Cache bergantung pada tiga teks, model, batas token dan format input. Mengubah rater tidak mengubah fitur encoder.',
          'S2 melatih MLP. S3 melatih proyeksi BGE dan MLP walaupun encoder BGE tetap frozen.']),
        ('Token → kata → kalimat', 'Baca vektor aktual tanpa menganggap koordinat sebagai skor.',
         ['Offset subtoken','Mean fitur kata','Kalimat dengan konteks','Vektor awal/akhir'],
         ['Vektor kata berasal dari subtoken jawaban dengan konteks soal/referensi tersedia. Vektor sebelum konteks adalah lookup embedding.',
          'ID token sama tidak menjamin vektor contextual sama. Cosine mendekati satu berarti arah mirip; periksa juga selisih komponen.',
          'Fusion dilakukan pada tingkat jawaban. Vektor kata RoBERTa S3 menunjukkan cabang RoBERTa, bukan fusion per kata.']),
        ('Arsitektur S1, S2 dan S3', 'Temukan encoder, proyeksi, normalisasi dan head.',
         ['S1: 768 → linear1','S2: 1024 → MLP','S3: 768 + 256','MLP → satu skor'],
         ['S1 tetap memakai head linear. S2/S3 memakai Linear → GELU → Dropout → Linear. MLP adalah head, bukan pengganti encoder.',
          'S3 menormalkan fitur RoBERTa768 dan memproyeksikan BGE1024 menjadi256. Concat menghasilkan1024 fitur, bukan penjumlahan skor.',
          'Seluruh skenario menerima informasi tiga kolom yang sama. Label tetap satu nilai; keluaran berbentuk [N,1].']),
        ('MSE, optimizer & checkpoint', 'Hubungkan target, gradien dan pemilihan epoch.',
         ['rater/100','MSE','Backward + AdamW','RMSE validasi'],
         ['MSE training memakai target0–1. RMSE validasi memakai poin0–100. MSE0,5 pada skala0–1 bukan error0,5% atau5%.',
          'Gradien diakumulasi ke batch efektif8 lalu dipotong sebelum optimizer memperbarui bobot. LR encoder dan head/proyeksi berada pada kelompok berbeda.',
          'Patience5 menghitung epoch yang tidak mengalahkan rekor RMSE terbaik. Dua slot checkpoint lokal dipakai untuk resume, tidak ditulis ke Drive.']),
        ('CV → refit → test E21', 'Pisahkan pemilihan model dari pelaporan akhir.',
         ['15 hasil fold','Kunci epoch CV','Refit 240 jawaban','Test 60 jawaban'],
         ['Semua S1/S2/S3 dilatih pada E21. Jumlah refit ditetapkan dari ceil median epoch terbaik lima fold; tidak menggunakan hasil test.',
          'Grafik menampilkan riwayat aktual dari Excel. Titik kuning adalah epoch terpilih pada fold yang dipilih.',
          'Hasil E21 hanya satu run. Perbedaan metrik tidak otomatis membuktikan signifikansi atau stabilitas lintas seed.']),
        ('Soal, referensi & prediksi', 'Baca nilai guru di sebelah skor model.',
         ['Soal + referensi + jawaban','Checkpoint final','S1/S2/S3','Bandingkan dengan guru'],
         ['Prediksi baru memerlukan ketiga teks. rater baru tidak diperlukan; nilai guru hanya tersedia jika jawaban sudah dinilai.',
          'RMSE/Pearson memakai skor mentah. QWK/AC2 memakai salinan kategori terdekat, tie .5 ke kategori lebih rendah.',
          'Jumlah prediksi terdekat dan RMSE mengukur hal berbeda. Kesalahan besar mempunyai pengaruh lebih kuat pada RMSE.']),
        ('Fitur setelah training', 'Lihat representasi yang berubah dan bagian yang tetap.',
         ['Model final','Fitur encoder','LayerNorm/proyeksi','Concat dan head'],
         ['RoBERTa S1 dan S3 dilatih terpisah sehingga fitur setelah training dapat berbeda. Encoder BGE tetap, sementara proyeksi/head S3 berubah.',
          'Ekstraksi sesudah training memakai eval/no_grad. Tabel semua300jawaban untuk inspeksi fitur; evaluasi penelitian tetap pada60test.']),
        ('Lima Excel E21', 'Cari bukti dan unduh file asli.',
         ['Data + token','Vektor jawaban','Kata/kalimat','Prediksi + riwayat'],
         ['Semua sheet dan komponen tetap disertakan pada buku lengkap. Tabel web dibagi per halaman dan rentang komponen agar terbaca.',
          'Klik sel untuk nilai/teks lengkap. Unduhan XLSX dan ZIP mengembalikan byte sumber asli, tanpa menghitung ulang prediksi.',
          'Output per tahap berasal dari Excel yang tersedia. Notebook eksekusi/log mentah tidak disertakan; buku tidak mengarang output cell atau training baru.'])]
    for chapter,(title,goal,flow,read) in zip(c,lessons):chapter.update(title=title,goal=goal,flow=flow,read=read)
    c[0]['read'][2]='Baca hasil run eksperimen yang dipilih. Jumlah skenario berbeda dari jumlah eksperimen; setiap run memiliki S1/S2/S3.'
    if snapshot.get('id')=='E22':
        for chapter in c:
            for key in ('title','goal'):chapter[key]=chapter[key].replace('E21','E22')
            for key in ('flow','read'):chapter[key]=[v.replace('E21','E22') for v in chapter[key]]
        c[3]['read'][2]='E22 memakai context_pair: teks A = soal + referensi, teks B = penanda jawaban + respons siswa. E21 memakai satu urutan teks; jumlah informasi input tetap sama.'
        c[6]['read'][0]='E22 memasangkan token konteks dan jawaban dengan separator resmi tokenizer. RoBERTa/XLM-R tidak menambahkan token-type embedding A/B seperti BERT.'
        c[6]['read'][2]='Overhead pasangan empat special token dicadangkan pada kedua format. E22 memakai empat special aktual; konteks dan starts/cakupan jawaban tetap sebanding dengan single sequence E21.'
    c[4]['sheets']=['Data dan Nilai Guru'];c[6]['sheets']=['Token RoBERTa','Token BGE-M3','Cakupan Token dan Konteks']
    c[3]['sheets']=['Pengaturan Eksperimen'];c[3]['book']=5
    g.update({'learning rate':('Ukuran langkah pembaruan','E21: encoder RoBERTa S1/S3 5e-6; proyeksi/head 1e-4. BGE frozen tidak mempunyai LR encoder.'),
              'special':('Penanda tokenizer','E21 memakai template satu urutan dengan dua special token. Cadangan empat posisi dipakai untuk menyamakan window dengan format pasangan E22.'),
              'window':('Potongan input tiga teks','Konteks soal/referensi diulang; kapasitas jawaban memperhitungkan konteks, penanda jawaban dan overhead. Jumlah window tidak mengubah hidden_size768/1024.'),
              'X':('Fitur input','Soal, jawaban referensi dan jawaban siswa masuk kedua encoder pada seluruh skenario.'),
              'y':('Target supervised','rater adalah satu skor guru0–100. Label ini tidak dimasukkan ke tokenizer atau cache BGE.')})
    if snapshot.get('id')=='E22':
        g['learning rate']=(g['learning rate'][0],g['learning rate'][1].replace('E21','E22'))
        g['special']=('Separator pasangan','E22 memakai empat special token pada pasangan konteks/jawaban. Keduanya tetap saling memperhatikan melalui self-attention; label guru tidak disisipkan.')
    c[3]['terms']=['X','y','learning rate','head'];c[4]['terms']=['X','y','tokenizer']
    if snapshot.get('id')=='E23':
        for chapter in c:
            for key in ('title','goal'):chapter[key]=chapter[key].replace('E21','E23')
            for key in ('flow','read'):chapter[key]=[v.replace('E21','E23') for v in chapter[key]]
        c[3]['read'][2]='E23 memakai single_sequence dari E21. Hanya S3 mendapat BiGRU token jawaban; S1/S2 mengikuti acuan E21.'
        c[6]['read'][2]='Dua special aktual pada single sequence; cadangan empat posisi menjaga anggaran token sama. answer_positions menentukan posisi respons yang dibaca BiGRU.'
        c[10]['flow']=['Token jawaban768','BiGRU dua arah','State128 →768','Concat BGE256 → MLP']
        c[10]['read'].append('AnswerBiGRU membaca urutan token jawaban dengan hidden64 per arah. State akhir dua arah disambung128 lalu diproyeksikan ke768. S3 ini tidak memakai attention pooling.')
        c[14]['flow']=['Model final','State BiGRU128','Pooling768','Fusion1024 dan skor']
        c[14]['read'].append('BiGRU S3 State Akhir berisi128 komponen; Pooling Jawaban S3 berisi768 hasil proyeksi. Kedua tabel adalah fitur, bukan nilai guru atau prediksi akhir.')
        c[14]['sheets']=['Pooling Jawaban S3','BiGRU S3 State Akhir','Fusion S3']
        g['BiGRU']=('Ringkasan urutan dua arah','E23 menjalankan GRU maju dan mundur pada token respons. Dimensi768 adalah fitur tiap token; gabungan state akhir128 diproyeksikan kembali ke768 sebelum fusion.')
        g['learning rate']=(g['learning rate'][0],g['learning rate'][1].replace('E21','E23')+' BiGRU/proyeksi barunya ikut kelompok head1e-4.')
        c[10]['terms'].append('BiGRU')
    if snapshot.get('id')=='E24':
        for chapter in c:
            for key in ('title','goal'):chapter[key]=chapter[key].replace('E21','E24')
            for key in ('flow','read'):chapter[key]=[v.replace('E21','E24') for v in chapter[key]]
        c[3]['read'][2]='E24 memakai single_sequence E21. Hanya S3 mendapat attention pooling token jawaban; S1/S2 mengikuti acuan E21.'
        c[6]['read'][2]='Dua special aktual pada single sequence; cadangan empat posisi menjaga anggaran token sama. answer_positions mencatat posisi respons yang boleh dipool.'
        c[10]['flow']=['Token RoBERTa','Mask token jawaban','Attention →768','Concat BGE256 → MLP']
        c[10]['read'].append('AnswerAttentionPool mempelajari bobot token jawaban dengan jaringan768→64→1; weighted sum tetap768. Token konteks/special/padding dikecualikan dari pooling, bukan dihilangkan dari self-attention encoder.')
        c[14]['flow']=['Model final','Pooling768','Bobot per token','Fusion1024 dan prediksi']
        c[14]['read'].append('Pooling Jawaban S3 menyimpan vektor aktual. Bobot Attention S3 memuat bobot token tiap jawaban/window; totalnya satu. Bobot besar bukan bukti sebab-akibat skor.')
        c[14]['sheets']=['Pooling Jawaban S3','Bobot Attention S3','Fusion S3']
        g['attention pooling']=('Ringkasan token berbobot','E24 memberi bobot hanya pada token respons. RoBERTa tetap membaca soal/referensi; bobot pooling dinormalisasi pada sumbu token, bukan768fitur.')
        g['learning rate']=(g['learning rate'][0],g['learning rate'][1].replace('E21','E24')+' Layer pooling baru ikut kelompok head1e-4.')
        c[10]['terms'].append('attention pooling')
    if snapshot.get('id') in ('E25','E26'):
        experiment=snapshot['id']
        for chapter in c:
            for key in ('title','goal'):chapter[key]=chapter[key].replace('E21',experiment)
            for key in ('flow','read'):chapter[key]=[v.replace('E21',experiment) for v in chapter[key]]
        c[3]['read'][2]=experiment+' memakai single_sequence, mengikuti E24. Perubahan layer hanya pada S3; seluruh S1/S2/S3 tetap dilatih baru. Head1e-4, bukan5e-6; angka5e-6 milik encoder RoBERTa.'
        c[6]['read'][2]='Dua special aktual dan cadangan empat posisi menjaga anggaran token. answer_positions menentukan token respons; konteks tetap tersedia melalui self-attention RoBERTa.'
        if experiment=='E25':
            c[10]['flow']=['Attention jawaban768','BGE1024 →256','Gate1024→32→1','Concat berbobot →MLP']
            c[10]['read'].extend(['AnswerAttentionPool tetap mengikuti E24. AnswerFusionGate menghasilkan g skalar per jawaban dari RoBERTa768 dan BGE256 setelah normalisasi/proyeksi.',
                'Concat aktual=[2g×RoBERTa768, 2(1−g)×BGE256]. Lapisan akhir gate diinisialisasi nol sehingga g=0,5 dan kedua pengali=1 saat awal; dimensi concat tetap1024.',
                'Gate berbeda dari attention token dan berbeda dari head regresi. Bobot gate bukan bukti encoder mana yang benar atau penyebab nilai guru.'])
            c[14]['flow']=['Pooling768','Bobot token','Gate + pengali','Concat1024 dan skor']
            c[14]['read'].append('Gate Fusion S3 menyimpan g; dua sheet pengali menyimpan2g dan2(1−g). Semuanya fitur aktual checkpoint final pada300jawaban; test tetap60jawaban.')
            c[14]['sheets']=['Pooling Jawaban S3','Bobot Attention S3','Gate Fusion S3','Pengali RoBERTa S3','Pengali BGE S3','Fusion S3']
            g['gated fusion']=('Koefisien dua cabang per jawaban','E25 mempelajari satu g melalui sigmoid. Concat berbobot tetap1024; saat awal g=0,5 mempertahankan concat E24. Gate bukan ukuran akurasi atau pembuktian kausal.')
            c[10]['terms'].append('gated fusion')
        else:
            c[10]['flow']=['Token jawaban768','BiGRU64/arah →128','Attention →ringkasan128','Proyeksi768 →fusion →MLP']
            c[10]['read'].extend(['AnswerBiGRUAttention mempertahankan semua output token GRU128 sebelum attention128→64→1. Ini berbeda dari E23 yang mengambil state akhir dan dari E24 yang melakukan attention langsung atas token768.',
                'Weighted sum128 diproyeksikan ke768 sebelum LayerNorm dan concat BGE256. Tidak ada gated fusion E25 pada E26.',
                'Mask dan packing mengecualikan konteks/special/padding dari pooling. Bobot attention dipetakan kembali ke posisi token asli untuk ekspor Excel.'])
            c[14]['flow']=['Output token GRU128','Bobot attention','Ringkasan128 →768','Concat1024 dan skor']
            c[14]['read'].append('Ringkasan BiGRU Attention S3 menyimpan128 komponen weighted sum, bukan state akhir E23. Pooling Jawaban S3 menyimpan768 komponen setelah proyeksi; bobot token ada di sheet tersendiri.')
            c[14]['sheets']=['Pooling Jawaban S3','Bobot Attention S3','Ringkasan BiGRU Attention S3','Fusion S3']
            g['BiGRU']=('Urutan token dua arah','E26 menghasilkan128 fitur pada setiap token melalui GRU64 per arah. Seluruh output token diteruskan ke attention; state akhir tidak menjadi ringkasan utama.')
            c[10]['terms'].append('BiGRU')
        g['attention pooling']=('Ringkasan token berbobot',experiment+' memberi bobot token respons yang berjumlah satu per window. Konteks tetap tersedia melalui encoder. Bobot bukan penjelasan sebab-akibat nilai.')
        g['learning rate']=(g['learning rate'][0],g['learning rate'][1].replace('E21',experiment)+' Layer tambahan ikut kelompok head1e-4.')
        c[10]['terms'].append('attention pooling')
    return c,g


PINS=[
 (3,'def prepare_local_storage','Hasil hanya lokal','Folder data/model/Excel berada di disk runtime; fungsi menolak workspace Drive. Ini bukan layer model.'),
 (3,'def resolve_dataset_upload','Upload pada web Colab','Memakai dataset yang sudah ada, atau Choose Files pada proses utama. Tidak melakukan training.'),
 (7,'browser_upload=IN_COLAB','Baca file unggahan','DATA_PATH menunjuk Excel asli. Pengecekan kolom/label dilakukan read_dataset setelahnya.'),
 (12,'ROBERTA_CHECKPOINT=','Checkpoint RoBERTa','Nama model pretrained Indonesia untuk S1/S3; encoder() memuat bobotnya.'),
 (12,'BGE_CHECKPOINT=','Checkpoint BGE-M3','BGE-M3 menghasilkan fitur dense1024. Encoder tetap frozen pada E21.'),
 (12,'ENCODER_LR=','LR RoBERTa S1/S3','Kedua encoder RoBERTa memakai5e-6. Head/proyeksi tetap1e-4; LR bukan angka kualitas.'),
 (12,"INPUT_FORMAT=",'Single sequence E21','Tiga teks disambung sebagai satu urutan. Ini pilihan format, bukan tiga encoder atau tiga target.'),
 (12,"device='cuda'",'GPU CUDA','Notebook rujukan memilih GPU; nama ZIP saja tidak membuktikan perangkat pada run unggahan.'),
 (14,"score = raw['rater']",'y berasal dari rater','Skor guru divalidasi sebagai angka kategori0–100. Tidak disalin menjadi fitur teks.'),
 (14,"'question': clean",'X tetap tiga kolom','Soal dan referensi disimpan sejajar dengan jawaban. Cleaning tidak memperbaiki ejaan siswa.'),
 (14,'y = np.asarray','bundle.y','Array target mengikuti urutan bundle.rows. Dipakai loss/evaluasi, bukan tokenizer.'),
 (18,'def make_split','Development dan test','Kelompok email/duplikat dipertahankan. Test tidak digunakan untuk memilih epoch.'),
 (20,'def context_token_ids','Buat token konteks','Soal dan referensi memiliki anggaran sendiri termasuk prefix. Pemotongannya dicatat.'),
 (20,"paired = cfg['input_format']",'Pilih jalur input','E21 membuat single sequence; cabang pasangan disediakan fungsi bersama untuk E22.'),
 (20,'capacity = maximum-len(context_ids)','Anggaran token jawaban','Batas model dikurangi konteks, prefix jawaban dan overhead cadangan. Kapasitas bukan selalu510.'),
 (20,'prepared = tokenizer.build_inputs_with_special_tokens(context_ids+second)','Satu urutan tiga teks','E21 menambahkan special token setelah menyambung konteks dan jawaban. Tensor tidak memuat nilai guru.'),
 (21,'result.requires_grad_','Kebijakan fine-tuning','RoBERTa trainable; BGE frozen. MLP/proyeksi sesudah fitur BGE tetap dapat belajar.'),
 (26,"'input_format':cfg['input_format']",'Cache mengikuti format','Format, soal, referensi dan jawaban ikut menentukan identitas cache. rater tidak termasuk.'),
 (30,'self.head = nn.Linear(768, 1)','Head linear S1','Satu pemetaan linear dari768fitur menjadi satu skor. Bukan MLP dan bukan encoder.'),
 (30,'self.head = nn.Sequential','MLP S2','Hidden256 + GELU + dropout + linear1. Fitur BGE tetap; parameter head diperbarui.'),
 (30,'self.roberta_norm =','LayerNorm S3','Normalisasi tambahan per vektor768, terpisah dari LayerNorm internal Transformer.'),
 (30,'self.projection =','Proyeksi BGE S3','Linear1024→256 + LayerNorm. Parameternya belajar walaupun encoder BGE frozen.'),
 (30,"features['concat'] =",'Fusion representasi','Menyambung768fitur RoBERTa dan256proyeksi BGE. Tidak menjumlahkan skor model.'),
 (30,"prediction = self.head(features['concat'])",'MLP S3 menghasilkan skor','Concat masuk head MLP. Keluaran masih skala latihan sebelum inverse transform.'),
 (31,'def seed_everything','Kendalikan keacakan','Mengatur RNG sebelum model/fold baru. Satu seed belum membuktikan kestabilan metode.'),
 (36,'def optimizer_groups','Tiga kelompok parameter','RoBERTa5e-6, proyeksi/head1e-4. Encoder BGE tidak masuk optimizer.'),
 (36,'targets = torch.as_tensor','Label untuk loss','rater0–100 diubah menjadi0–1. Bentuk target [B,1] harus sama dengan prediksi.'),
 (36,'loss = torch.mean','MSE target0–1','Rata-rata selisih kuadrat pada target ternormalisasi. Tidak ditafsirkan sebagai persentase.'),
 (36,'.backward()','Hitung gradien','Backward menghitung arah perubahan parameter; optimizer.step memperbarui bobot.'),
 (36,'save_training_progress(checkpoint','Checkpoint lokal','Menyimpan model/optimizer/RNG ke dua slot untuk resume pada filesystem yang masih ada.'),
 (38,'selected_epochs[scenario] =','Epoch refit dari CV','Ceil median epoch terbaik lima fold; data test tidak ikut memilih jumlah epoch.'),
 (38,"prediction = engine.predict(manifest['test'])",'Evaluasi akhir','Checkpoint final dibaca sesudah pemilihan model dikunci. Ini bukan langkah training.'),
 (41,'def score_answer','Prediksi tiga teks','Jawaban baru memerlukan soal dan referensi; tidak memerlukan rater baru.'),
 (43,'def export_final_features','Inspeksi model final','Mengekspor fitur dengan eval/no_grad. Tabel300jawaban bukan evaluasi300test.'),
 (46,'files.download','Unduh laporan','ZIP berisi lima Excel, bukan bobot model. Buku membaca laporan tanpa menjalankan training.')]

HINTS={
 'train_all':'Menjalankan15foldCV, menetapkan epoch, refit ketiga skenario, lalu mengevaluasi60test. Seluruh S1/S2/S3 dilatih baru pada eksperimen ini.',
 'fit':'Melatih satu fold/refit memakai MSE target0–1 dan optimizer sesuai kelompok LR. Test tidak dipakai early stopping.',
 'Engine':'Menyelaraskan tiga teks dengan indeks jawaban dan fitur BGE, lalu mengembalikan skor ke poin0–100 saat predict.',
 'contexts_from_bundle':'Mengambil soal dan jawaban referensi dalam urutan yang sama dengan jawaban siswa.',
 'bge_features':'Menghitung/cache fitur BGE tiga teks tanpa label guru. Format dan tokenisasi ikut menentukan identitas cache.',
 'token_vectors':'Mengambil lookup dan fitur contextual subtoken jawaban dengan konteks soal/referensi sesuai format input eksperimen.'}
