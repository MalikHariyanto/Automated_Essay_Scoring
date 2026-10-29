"""Catatan belajar tiga input: padanan screenshot ditautkan ke source yang tepat."""
# Cell notebook, anchor asli, judul, penjelasan singkat, rincian, kemunculan opsional.
CORE = [
(3,'def prepare_local_storage','Folder dan penyimpanan lokal','Menyiapkan direktori kerja tanpa menyambungkan Google Drive.','Kode, dataset, cache dan hasil adalah benda berbeda. Fungsi ini mengatur lokasi file pada disk runtime; tidak membentuk layer atau melatih model.'),
(3,'def resolve_dataset_upload','Unggah dataset','Memilih Excel asli untuk seluruh proses eksperimen.','Upload dilakukan oleh notebook utama. Worker menerima alamat file yang sama. Mengunggah data belum berarti encoder sudah membaca atau mempelajarinya.'),
(5,'def prepare_environment','Lingkungan pustaka','Menyiapkan versi pustaka sebelum menjalankan model AES.','Lingkungan Python terpisah membantu mencegah konflik NumPy, Torch dan Transformers bawaan Colab. Tahap ini belum memperbarui bobot model.'),
(6,'WORKER_SOURCE','Worker dan %%aes','Menghubungkan cell Colab dengan proses perhitungan terpisah.','Cell %%aes dikirim ke worker Python yang mempertahankan namespace antar-cell. Jadi bundle, cfg dan model yang dibuat sebelumnya dapat dipakai pada cell berikutnya.'),
(7,'def aes_cell','Menjalankan cell %%aes','Mengirim isi cell ke worker yang sama.','Definisi magic ini menjembatani notebook dengan worker. Traceback worker dikirim kembali ke tampilan Colab; nomor baris error perlu dibaca pada fungsi yang benar.'),
(12,'ROBERTA_CHECKPOINT=','Nama model RoBERTa','Menentukan alamat pretrained RoBERTa bahasa Indonesia.','String flax-community/indonesian-roberta-base adalah nama checkpoint di Hugging Face. Objek encoder baru dimuat oleh encoder() melalui AutoModel.from_pretrained; variabel nama ini bukan model yang sedang berjalan.'),
(12,'BGE_CHECKPOINT=','Nama model BGE-M3','Menentukan pretrained encoder semantik pada S2/S3.','BAAI/bge-m3 menyediakan representasi dense 1024 komponen. Encoder BGE dibekukan, sedangkan proyeksi dan head setelahnya tetap boleh dilatih.'),
(12,'ROBERTA_REVISION=','Versi pretrained','Mengunci versi bobot dan tokenizer sumber model.','Revision membantu eksperimen menggunakan sumber pretrained yang sama. Ini berbeda dari seed yang mengatur keacakan dan berbeda dari checkpoint final hasil training.'),
(12,'EPOCHS,FOLDS=','Epoch dan fold','Membatasi latihan dan jumlah pembagian validasi silang.','Lima fold adalah lima pembagian development. Maksimal 25 epoch adalah jumlah putaran membaca data latih pada setiap fold, bukan 25 fold.'),
(12,'EARLY_STOPPING_PATIENCE=','Early stopping','Berhenti setelah lima epoch tanpa rekor validasi.','Yang dipantau adalah RMSE validasi terbaik. Lima epoch tanpa perbaikan bukan lima kali memperbaiki prediksi siswa dan tidak menggunakan nilai test.'),
(12,'ENCODER_LR=','Learning rate encoder','Mengatur langkah pembaruan RoBERTa S1 dan S3.','Pada keluarga tiga input ini S1 dan S3 memakai 5e-6. Nilai kecil membatasi besar langkah tiap update; bukan batas skor, kualitas model, atau persentase jawaban yang benar.'),
(12,'PROJECTION_LR=HEAD_LR=','LR proyeksi dan head','Mengatur learning rate lapisan baru sebesar 1e-4.','Proyeksi BGE, MLP, LayerNorm tambahan dan pooling baru S3 memakai kelompok 1e-4 sesuai optimizer. BGE frozen tidak memiliki langkah update encoder.'),
(12,'MICRO_BATCH,EFFECTIVE_BATCH=','Batch dan akumulasi gradien','Mengolah satu jawaban hingga batch efektif delapan.','Microbatch 1 membatasi penggunaan memori. Gradien dikumpulkan sebelum optimizer.step sehingga satu update dapat mewakili delapan jawaban; ini tidak menambah jumlah dataset.'),
(12,'INPUT_FORMAT=','Format tiga input','Menentukan susunan soal, referensi, dan respons siswa.','E21/E23/E24 memakai single_sequence. E22 memakai context_pair. Tiga kolom input tetap sama; format adalah cara token disusun, bukan jumlah label.'),
(12,"device='cuda'",'Perangkat GPU','Memilih CUDA untuk perhitungan model eksperimen ini.','Bobot dan tensor perlu berada pada perangkat yang sesuai. Nama ZIP berakhiran GPU saja bukan bukti perangkat runtime; source notebook ini meminta CUDA sebelum training.'),
(14,'def clean(','Pembersihan teks','Merapikan Unicode tanpa memperbaiki ejaan jawaban siswa.','Cleaning menyamakan penulisan Unicode/newline dan membuang karakter kontrol sesuai fungsi. Isinya tidak ditulis ulang menjadi jawaban ideal dan nilai guru tidak diubah.'),
(14,'class Bundle','Bundle: wadah data','Menyatukan teks, target, kelompok, dan metadata sumber.','bundle adalah objek Bundle. rows menyimpan tiap baris, texts menyediakan respons, y menyimpan skor guru dan groups membantu mencegah kelompok terkait bocor antar-split. Bundle bukan encoder dan bukan layer.'),
(14,"'question': clean",'Input X: tiga variabel','Menyimpan soal, referensi, dan respons secara sejajar.','Ketiga teks menjadi fitur masukan pada S1/S2/S3. Variabel bernama X tidak wajib ada: rows, texts dan contexts bersama-sama menjalankan perannya.'),
(14,"score = raw['rater']",'Label dari rater','Mengambil nilai guru sebagai acuan supervised learning.','rater adalah target, bukan input teks ke model. Nilai ini diperiksa sebelum disimpan; model memprediksinya dari tiga teks tanpa diberi tahu skor saat inferensi.'),
(14,'y = np.asarray','Array target y','Menyusun skor guru mengikuti urutan jawaban dataset.','bundle.y adalah padanan y pada contoh teman lo. Saat training nilainya dinormalisasi ke 0–1 untuk loss; saat evaluasi skor kembali dibaca dalam poin 0–100.'),
(14,'return Bundle(','Hasil pembacaan data','Mengembalikan satu wadah berisi seluruh data terurut.','return mengirim objek ke pemanggil read_dataset. Dengan urutan ini, indeks satu jawaban menunjuk teks, label dan kelompok yang sama.'),
(15,'class ExcelJournal','Pencatat output','Menyimpan keluaran tahap beserta asal datanya.','journal mencatat tabel dan sumbernya ke workbook. Ia bukan model prediksi. Parameter journal pada export_final_features menunjuk objek pencatat yang sudah dibuat sebelumnya.'),
(16,'bundle=read_dataset(','Membaca dataset sungguhan','Memanggil pembaca Excel dan membentuk bundle aktual.','Berbeda dari def read_dataset yang hanya menyiapkan fungsi, baris pemanggilan ini menjalankannya. Ini awal alur data aktual yang kemudian ditokenisasi dan dibagi.'),
(18,'GroupShuffleSplit(','Development dan test','Memisahkan kelompok sebelum memilih model melalui validasi.','Development digunakan untuk latihan dan lima fold. Test disimpan untuk evaluasi akhir. Kelompok terkait dijaga agar tidak terpecah secara sembarangan.'),
(18,'GroupKFold(','Lima fold validasi','Menggilir bagian development menjadi latih dan validasi.','Setiap fold membuat model baru. Skor validasi membantu memilih epoch; siswa pada validasi fold itu tidak ikut memperbarui parameter model fold tersebut.'),
(20,'def context_token_ids','Membentuk konteks soal dan kunci','Menyiapkan token konteks dengan anggaran yang ditentukan.','Soal dan jawaban referensi tidak langsung menjadi angka skor. Tokenizer mengubahnya menjadi ID token, memakai batas konteks/soal yang dicatat di konfigurasi.'),
(20,'capacity = maximum-len(context_ids)','Kapasitas respons per window','Menyisihkan ruang bagi konteks, prefix, dan token khusus.','Batas 512 RoBERTa bukan selalu 512 token jawaban. Konteks menggunakan sebagian ruang; jawaban panjang dibagi menjadi window bertumpang tindih agar cakupannya tetap tercatat.'),
(20,'def mean_by_essay(','Pooling antar-window','Menggabungkan vektor window milik jawaban yang sama.','Sumbu window berbeda dari sumbu token dan dari 768 komponen fitur. S3 E23/E24 merangkum token jawaban dahulu, kemudian merata-ratakan ringkasan window per jawaban.'),
(21,'AutoTokenizer.from_pretrained(','Memuat tokenizer','Membaca kosakata dan aturan tokenisasi pretrained model.','Tokenizer RoBERTa dan BGE memiliki kosakata sendiri. Satu kata dapat menjadi beberapa subtoken; ID token merupakan indeks kosakata, bukan nilai kualitas jawaban.'),
(21,'AutoModel.from_pretrained(','Memuat backbone encoder','Membentuk objek encoder dari bobot pretrained terpilih.','Padanan inisialisasi model teman lo ada di sini, tetapi head regresi kita disusun terpisah di SingleScoreModel. AutoModel sendiri belum menjadi sistem pemberi skor lengkap.'),
(21,'result.requires_grad_','Fine-tuning dan frozen','Menentukan parameter encoder yang boleh diperbarui training.','RoBERTa S1/S3 trainable. BGE frozen: bobot encodernya tidak diperbarui. Proyeksi atau MLP di luar encoder BGE tetap memiliki parameter belajar sendiri.'),
(24,'def roberta_pretrained_vectors(','Fitur sebelum latihan','Mengekstrak representasi RoBERTa pretrained sebagai bahan inspeksi.','Bagian ini setara dengan ekstraksi embedding awal pada screenshot. Representasi sebelum fine-tuning berbeda dari fitur checkpoint final yang diekspor sesudah training.'),
(20,'last_hidden_state[:,0,:]','CLS sebelum latihan','Mengambil seluruh komponen token pertama setiap window.','Tanda [:, 0, :] berarti seluruh window, token posisi pertama, seluruh dimensi fitur. 768 adalah lebar vektor RoBERTa; jawaban pendek tetap menghasilkan vektor selebar itu.'),
(26,'def bge_features(','Fitur BGE-M3 frozen','Menghitung dan menyimpan cache representasi tiga teks.','Cabang ini menghasilkan fitur 1024 komponen tanpa melatih encoder BGE. Cache harus cocok dengan teks, tokenizer, model dan format input; label guru tidak menjadi embedding.'),
(20,'functional.normalize(','Normalisasi L2 BGE','Membagi vektor dengan panjangnya sebelum dipakai berikutnya.','L2 normalization mengubah skala panjang vektor, bukan jumlah komponennya. Ini berbeda dari LayerNorm yang menata statistik komponen dalam satu representasi.'),
(28,'def word_vectors(','Vektor per kata','Menggabungkan representasi subtoken menjadi vektor kata.','Jumlah kata dan jumlah komponen berbeda. Kata RoBERTa tetap diwakili 768 angka; jumlah baris token/kata berubah mengikuti panjang teks.'),
(30,'class SingleScoreModel','Arsitektur tiga skenario','Menyusun encoder, pooling, proyeksi, dan head prediksi.','Kelas ini setara dengan bagian arsitektur model teman lo. __init__ membentuk layer; forward menentukan urutan pemakaiannya. Periksa cabang if scenario agar S1/S2/S3 tidak tertukar.'),
(30,'self.head = nn.Linear(768, 1)','Head regresi linear S1','Memetakan 768 fitur menjadi satu prediksi kontinu.','Inilah padanan classifier linear pada screenshot. S1 memakai Linear(768,1), bukan MLP. Baris ini membuat layer; self.head(...) pada forward menjalankannya.'),
(30,'self.head = nn.Sequential(','Head regresi MLP S2','Mengolah fitur BGE dengan hidden nonlinier lalu skor.','S2: Linear 1024→256, GELU, Dropout, Linear 256→1. Keseluruhan rangkaian adalah MLP regresi; layer terakhirnya tetap linear. Encoder BGE tetap menjadi encoder.'),
(30,'self.roberta_norm =','LayerNorm sebelum fusion','Menata skala 768 komponen representasi jawaban RoBERTa.','LayerNorm menghitung rata-rata dan varians komponen per vektor, menormalkannya, lalu memakai parameter skala/geser yang dipelajari. Jumlah fitur tetap 768; ini berbeda dari normalisasi nilai guru.'),
(30,'self.projection =','Proyeksi BGE pada S3','Memadatkan fitur BGE dari 1024 menjadi 256.','Proyeksi ini milik cabang BGE di S3. Linear dan LayerNorm di dalamnya dilatih meskipun encoder BGE frozen. S1 tidak memakai proyeksi BGE ini.'),
(30,'self.head = nn.Sequential(','Head regresi MLP S3','Mengubah fitur fusion menjadi satu skor regresi.','S3 menerima concat 1024 fitur: 768 dari ringkasan RoBERTa dan 256 dari proyeksi BGE. MLP 1024→256→1 berada setelah fusion; ia bukan encoder RoBERTa atau BGE.',2),
(30,"features['concat'] =",'Fusion representasi sebenarnya','Menyambung fitur dua encoder sebelum menghasilkan skor.','torch.cat menyusun 768 dan 256 komponen menjadi 1024. Ini bukan menjumlahkan skor S1 dan S2. Kedua cabang S3 masuk head yang dilatih untuk skenario S3 sendiri.'),
(30,"prediction = self.head(features['concat'])",'Prediksi akhir dari head S3','Menjalankan MLP pada vektor gabungan dua cabang.','Keluaran ini satu angka pada skala latihan. Engine.predict kemudian mengembalikannya ke poin skor guru. MLP tetap head regresi, bukan pengganti encoder.'),
(30,'class Engine','Engine: penghubung data dan model','Menyiapkan tensor lalu memanggil forward model yang sesuai.','Engine mengambil indeks jawaban, menyelaraskan konteks, membuat batch/window dan mengambil fitur BGE bila diperlukan. Bundle menyimpan data; Engine menyiapkannya untuk model.'),
(30,'return scaled*','Mengembalikan skala nilai guru','Mengubah prediksi target nol–satu menjadi poin skor.','Keluaran dinormalisasi saat loss dihitung. Perkalian rentang skor dan penambahan minimum mengembalikannya ke 0–100 sebelum RMSE/Pearson serta tabel prediksi.'),
(31,'def seed_everything(','Seed dan keacakan','Mengatur pengacakan sebelum memulai model dan latihan.','Seed memengaruhi inisialisasi layer baru, dropout dan urutan sampel. Source yang sama dapat memberi hasil berbeda jika seed, perangkat atau pustaka berbeda. Satu seed tidak membuktikan kestabilan metode.'),
(31,'def make_model(','Model baru setiap fold','Membentuk model segar sesuai skenario dan konfigurasi.','Setiap fold memulai model baru dari pretrained yang sama dan head yang diinisialisasi. Bobot hasil fold sebelumnya tidak diteruskan menjadi awal fold berikutnya.'),
(36,'def optimizer_groups','Kelompok learning rate','Memisahkan langkah encoder dari proyeksi dan head.','RoBERTa memakai 5e-6. Head, proyeksi, LayerNorm tambahan dan pooling baru menggunakan 1e-4. Parameter encoder BGE frozen tidak ikut di-update optimizer.'),
(36,'def fit(','Loop training satu model','Melatih satu fold atau refit sampai aturan berhenti.','Padanan fungsi pelatihan teman lo ada di sini. fit berisi forward, loss, backward, update parameter, validasi dan checkpoint; train_all mengatur kapan fit dipanggil.'),
(36,'targets = torch.as_tensor','Label untuk MSE','Menyiapkan rater pada skala nol–satu sesuai prediksi.','Target dan prediksi harus berbentuk sama, [batch,1]. Nilai guru tidak dimasukkan ke encoder; ia hanya dibandingkan dengan keluaran model melalui loss.'),
(36,'loss = torch.mean','Loss MSE training','Merata-ratakan kuadrat selisih prediksi terhadap target ternormalisasi.','MSE memakai kuadrat satuan target. MSE 0,5 bukan error 5%. RMSE validasi di laporan dihitung setelah prediksi kembali ke poin 0–100.'),
(36,'.backward()','Menghitung gradien','Menentukan arah perubahan bagi parameter model yang belajar.','backward menghitung gradien melalui encoder/head yang trainable. Bobot belum diperbarui sampai optimizer.step dipanggil, termasuk ketika gradien sedang diakumulasi.'),
(38,'def train_all(','CV, refit, dan test','Mengatur urutan eksperimen sampai laporan evaluasi akhir.','Padanan loop for fold pada screenshot tersebar antara make_split, train_all dan fit. Lima fold memilih epoch, refit melatih development, lalu test mengevaluasi checkpoint final.'),
(38,'selected_epochs[scenario] =','Epoch refit dari CV','Mengunci jumlah epoch tanpa melihat hasil data test.','Jumlah refit ditetapkan dari pembulatan ke atas median lima epoch terbaik. Karena itu S1/S2/S3 dapat memiliki jumlah epoch berbeda walaupun batas maksimalnya sama.'),
(38,"prediction = engine.predict(manifest['test'])",'Prediksi data uji','Membaca checkpoint final untuk jawaban yang ditahan.','Prediksi test tidak dipakai untuk memperbarui bobot atau memilih epoch. Membandingkan banyak eksperimen pada test yang sama tetap perlu dilaporkan sebagai eksplorasi.'),
(39,'def prediction_frame(','Tabel guru dan prediksi','Menyandingkan skor asli dengan keluaran mentah model.','RMSE/Pearson menggunakan skor kontinu. Pembulatan adalah salinan untuk kategori QWK/AC2; model tidak diubah agar selalu mengeluarkan bilangan bulat.'),
(41,'def score_answer(','Menilai tiga teks baru','Memerlukan soal, referensi, dan jawaban tanpa label baru.','Saat inferensi, model hanya membutuhkan tiga input. Rater hanya diperlukan bila kita hendak mengevaluasi kesalahannya terhadap penilaian guru.'),
(43,'def export_final_features','Fitur setelah training','Membaca representasi aktual checkpoint final tiap skenario.','bundle berisi data, bge berisi fitur BGE, cfg berisi pengaturan, output alamat folder, journal pencatat Excel. Mode eval/no_grad membaca fitur; tidak melatih ulang model. Seluruh 300 baris untuk inspeksi, bukan 300 test.'),
(46,'files.download','Mengunduh laporan hasil','Mengemas lima Excel aktual menjadi satu ZIP.','ZIP ringkasan berisi tabel output, bukan seluruh bobot model. Buku HTML membaca file ini dan tidak menjalankan notebook atau training di browser.'),
]


def enrich_learning_notes(cells, experiment):
    by_index={c['index']:c for c in cells}
    required=[]
    for spec in CORE:
        index,anchor,title,short,body,*occ=spec
        if experiment in ('E27','E28') and anchor in ('PROJECTION_LR=HEAD_LR=','def optimizer_groups'):
            body='Head kontrol S1/S2, proyeksi BGE, BiGRU dan norm tetap1e-4. Hanya head.* S3 memakai '+('5e-6' if experiment=='E28' else '1e-4')+'. s3_mlp_lr dipisah dari s3_layers_lr; satu parameter trainable masuk tepat satu kelompok.'
        if experiment=='E29' and anchor in ('PROJECTION_LR=HEAD_LR=','def optimizer_groups'):
            body='E29: linear S1, proyeksi BGE dan pooling/norm S3 tetap1e-4; seluruh head.* MLP S2/S3 memakai5e-6. Encoder RoBERTa5e-6, BGE frozen. Setiap parameter trainable masuk tepat satu kelompok optimizer.'
        if experiment=='E25' and anchor=="features['concat'] =":
            anchor="features['concat'], gate ="
            body='Gate menghasilkan g dari kedua fitur per jawaban. Concat aktual [2g×RoBERTa768, 2(1−g)×BGE256] tetap1024. Ini bukan penjumlahan skor S1 dan S2; kedua cabang masuk head S3 yang dilatih bersama.'
        c=by_index[index];hits=[j+1 for j,line in enumerate(c['source'].splitlines()) if anchor in line]
        if not hits:
            # Some runtime/export anchors differ between historical notebook variants.
            if index in (3,7,39):continue
            raise ValueError('Anchor catatan belajar tidak ditemukan: '+str((experiment,index,anchor)))
        line=hits[(occ or [1])[0]-1]
        note=next((n for n in c['notes'] if n['line']==line),None)
        data={'line':line,'title':title,'short':short,'body':body,'anchor':anchor,'kind':'fundamental','term':None}
        if note:note.update(data)
        else:c['notes'].append(data)
        required.append((index,line,title))
    if experiment in ('E23','E24','E25','E26','E27','E28','E29'):
        extra=[('def answer_token_mask','Mask token jawaban','Memilih posisi respons tanpa memasukkan konteks atau padding.','RoBERTa tetap membaca ketiga teks. Pooling tambahan S3 hanya memakai posisi token jawaban dari answer_positions. Mask bukan pemeriksaan apakah ID kata sama dengan kata pada konteks.'),
          ('hidden = self.roberta','Fitur token contextual','Mengambil representasi semua token dari backbone RoBERTa.','Tensor hidden berbentuk [window,token,768]. Pada S3 ringkasan jawaban berasal dari pooling tambahan. CLS tetap dihitung untuk kontrol/referensi; jangan menyebut CLS sebagai input prediksi S3 varian ini.'),
          ('self.sequence_pool =','Pooling tambahan khusus S3','Membentuk peringkas token sebelum normalisasi dan fusion.','S1 tetap CLS dan head linear. S2 tetap BGE frozen dan MLP. Layer tambahan hanya dijalankan pada S3; encoder RoBERTa tetap ikut fine-tuning.'),
          ("features['answer_pool768']",'Representasi jawaban S3','Menyimpan 768 fitur hasil pooling untuk inspeksi.','Vektor ini belum menjadi skor. Setelah LayerNorm, ia digabung dengan proyeksi BGE lalu melewati head MLP.'),
        ]
        if experiment in ('E23','E28'):extra += [
          ('class AnswerBiGRU','BiGRU membaca urutan jawaban','Meringkas urutan token jawaban dari dua arah.','GRU maju dan mundur masing-masing mempunyai hidden 64. Sumbu urutan adalah token jawaban, bukan 768 koordinat fitur. Ini tambahan di luar RoBERTa.'),
          ('self.gru = nn.GRU','GRU dua arah','Membentuk dua pembaca urutan dengan hidden enam puluh empat.','Input setiap langkah adalah 768 fitur satu token. bidirectional=True membentuk arah maju dan mundur; bukan dua encoder pretrained baru.'),
          ('packed = torch.nn.utils.rnn.pack_padded_sequence','Panjang token yang sebenarnya','Mencegah state akhir GRU membaca padding tambahan.','Jumlah token tiap jawaban dapat berbeda. Packing memakai panjang aktual, sehingga token kosong penyama batch tidak menjadi isi jawaban.'),
          ('summary = torch.cat([states[-2]','State akhir dua arah','Menyambung dua hidden 64 menjadi 128 fitur.','states[-2] dan states[-1] adalah state akhir arah maju dan mundur. Ini ringkasan BiGRU, berbeda dari CLS dan berbeda dari fusion RoBERTa–BGE.'),
          ('self.output = nn.Linear','Proyeksi ringkasan BiGRU','Mengembalikan 128 fitur ringkasan menjadi 768 komponen.','Linear 128→768 memetakan gabungan state maju/mundur. Ini proyeksi tambahan pada cabang RoBERTa E23, berbeda dari proyeksi BGE 1024→256. Dimensi 768 menjaga alur LayerNorm dan fusion berikutnya sesuai acuan.'),
        ]
        elif experiment in ('E24','E25'):extra += [
          ('self.hidden = nn.Linear','Jaringan attention token','Menghitung penilaian token melalui fitur hidden enam puluh empat.','Jaringan 768→64→1 menghasilkan satu logit per token. Attention bekerja di sumbu token, bukan memilih satu dari 768 koordinat sebagai jawaban.'),
          ('weights = torch.softmax','Bobot attention bertopeng','Membagi bobot hanya kepada token respons yang sah.','Logit non-jawaban dibuat minus tak hingga, lalu softmax FP32 menghasilkan bobot nonnegatif berjumlah satu per window. Bobot ini tidak sama dengan nilai guru.'),
          ('weights.unsqueeze(-1)','Ringkasan berbobot attention','Menjumlahkan vektor token setelah dikalikan bobotnya.','Setiap token mempunyai vektor 768 komponen. Weighted sum menghasilkan satu vektor 768, yang kemudian masuk normalisasi dan fusion. Bobot attention bukan bukti sebab-akibat skor.'),
        ]
        if experiment=='E25':extra += [
          ('class AnswerFusionGate','Gate dua encoder','Mempelajari kontribusi kedua cabang untuk setiap jawaban.','Gate menerima RoBERTa768 setelah LayerNorm dan proyeksi BGE256. Jaringan1024→32→1 menghasilkan satu g per jawaban; bukan satu bobot per token atau per koordinat.'),
          ('self.hidden = nn.Linear(1024','Jaringan gate 1024→32','Meringkas dua cabang sebelum menentukan koefisien fusion.','Hidden32 memakai tanh, lalu satu keluaran sigmoid. Gate berbeda dari head MLP1024→256→1 yang memprediksi nilai guru.'),
          ('nn.init.zeros_(self.score.weight)','Awal sama dengan concat E24','Mengawali gate setengah sehingga kedua pengali bernilai satu.','Bobot/bias akhir gate nol menghasilkan sigmoid(0)=0,5. Faktor2 membuat 2g=1 dan 2(1−g)=1; concat awal sama dengan acuan, bukan membagi dua fitur secara tidak sengaja.'),
          ('gate = torch.sigmoid','Koefisien gate per jawaban','Membatasi koefisien antara nol dan satu.','g mengatur pengali cabang RoBERTa; 1−g mengatur pengali BGE. g tinggi tidak membuktikan encoder tertentu lebih benar atau menjelaskan sebab skor.'),
          ('fused = torch.cat','Concat aktual berbobot','Menyambung dua cabang yang telah dikalikan gate.','[2g×RoBERTa768, 2(1−g)×BGE256] menghasilkan1024 fitur. g=0,5 identik acuan; ukuran g bukan persentase akurasi.'),
          ('self.fusion_gate =','Gate khusus fusion S3','Membentuk gate hanya pada cabang skenario tiga.','S1/S2 tidak menjalankan gate. Parameter gate masuk kelompok head1e-4; bobot BGE pretrained tetap frozen.'),
          ("features['fusion_gate'] =",'Ekspor gate final','Menyimpan koefisien dan pengali aktual untuk inspeksi.','Excel Gate Fusion S3, Pengali RoBERTa S3 dan Pengali BGE S3 berasal dari forward checkpoint final. Baris300jawaban untuk inspeksi fitur, bukan300test.'),
        ]
        if experiment in ('E26','E29'):extra += [
          ('class AnswerBiGRUAttention','BiGRU dilanjutkan attention','Membaca urutan token lalu memilih ringkasan berbobot.','Ini kombinasi E23/E24 dengan implementasi berbeda: attention menerima semua output token BiGRU128, bukan satu state akhir. Tidak memakai gated fusion E25.'),
          ('self.gru = nn.GRU','GRU dua arah pada respons','Menghasilkan seratus dua puluh delapan fitur per token.','Input token768 dibaca GRU hidden64 per arah. Keluarannya urutan [window,token,128], sehingga attention masih mempunyai beberapa token untuk dibandingkan.'),
          ('packed = torch.nn.utils.rnn.pack_padded_sequence','Packing token aktual','Menghindari token padding masuk urutan GRU.','Mask memilih posisi jawaban saja dan packing memakai panjang sebenarnya. Konteks/special/padding tidak dipool; RoBERTa tetap membaca konteks.'),
          ('packed_outputs, _ = self.gru','Semua output BiGRU','Mempertahankan urutan keluaran untuk tahap attention.','Berbeda dari E23 yang menyambung state akhir dua arah. E26 mempertahankan representasi128 setiap token; underscore berarti state akhir tidak dipakai sebagai ringkasan utama.'),
          ('self.attention_hidden =','Attention atas output GRU','Menghitung logit token melalui jaringan128→64→1.','Attention bekerja setelah BiGRU. Lebar masukan attention128, bukan768 pada E24. Keluaran akhirnya masih bobot token, belum skor guru.'),
          ('weights = torch.softmax','Bobot attention bertopeng','Membagi bobot kepada token respons yang sebenarnya.','Softmax FP32 pada logit nonpadding menghasilkan bobot nonnegatif yang berjumlah satu per window. Padding hasil unpacking mendapat bobot nol.'),
          ('summary = (weights.unsqueeze','Ringkasan BiGRU attention128','Menjumlahkan output token128 memakai bobot attention.','Ringkasan ini mempunyai128 komponen. Ini berbeda dari state akhir E23 dan berbeda dari prediksi regresi1komponen.'),
          ('self.output = nn.Linear','Proyeksi ringkasan attention','Mengubah128 fitur ringkasan menjadi768 sebelum fusion.','Linear128→768 menjaga ukuran LayerNorm RoBERTa dan concat berikutnya. Proyeksi ini terpisah dari proyeksi BGE1024→256.'),
          ('self.last_weights = torch.zeros_like','Bobot kembali ke posisi asli','Menyelaraskan bobot GRU dengan token pada Excel.','masked_scatter menempatkan bobot pada posisi respons asli. Konteks/special/padding tetap nol; tanpa pemetaan ini penjelasan token bisa bergeser.'),
          ('self.last_summary = summary.detach','Salinan ringkasan untuk inspeksi','Menyimpan fitur tanpa memutus jalur prediksi model.','detach dipakai pada salinan last_summary untuk ekspor. Keluaran aktual tetap self.output(summary), sehingga gradien mengalir melalui attention dan BiGRU saat training.'),
        ]
        if experiment=='E27':extra += [
          ('class AnswerReferenceCoAttention','Co-attention dari dasar E23','Mencocokkan urutan respons dengan token isi referensi.','Kelas ini mewarisi GRU dan output projection AnswerBiGRU E23, lalu mengganti forward untuk pencocokan. Definisi forward induk tetap terlihat sebagai acuan; jalur aktif S3 memakai subclass ini.'),
          ('self.reference_projection =','Proyeksi referensi768→128','Menyamakan lebar fitur referensi dengan keluaran BiGRU.','Token referensi tetap berasal dari hidden RoBERTa pada input yang sama. Layer Linear baru dilatih1e-4; ini bukan proyeksi BGE1024→256.'),
          ('self.matching_projection =','Pencocokan640→128','Memadatkan lima jenis hubungan menjadi fitur pencocokan.','Respons, referensi selaras, selisih, produk dan respons jalur balik masing-masing128komponen disambung640, lalu Linear128+tanh. Fitur ini belum skor guru.'),
          ('packed_outputs, _ = self.gru','Urutan token respons128','Mempertahankan keluaran setiap token untuk pencocokan referensi.','E23 asli memakai state akhir langsung. E27 perlu urutan128 agar softmax dapat membandingkan beberapa token respons dan referensi, bukan satu vektor global.'),
          ('scores = torch.bmm','Skor pencocokan dua urutan','Menghitung dot product dengan skala akar dimensi.','Skor respons×referensi dibagi √128. Perhitungan skor/softmax FP32 tetap digunakan saat encoder memakai AMP; ini bukan RMSE atau nilai guru.'),
          ('a_to_r = torch.softmax','Respons menuju referensi','Membagi bobot hanya pada token isi referensi.','Softmax berjalan sepanjang referensi. Mask mencegah prefix, soal, special dan padding mendapat bobot; jumlah setiap query respons valid adalah1.'),
          ('r_to_a = torch.softmax','Referensi menuju respons','Membentuk pencocokan arah balik pada token respons.','Softmax transpose berjalan sepanjang respons. Kedua arah berbeda; bobot inspeksi yang dirata-ratakan tidak menjadi attention pooling.'),
          ('round_trip =','Jalur balik co-attention','Membentuk respons melalui hubungan referensi dan respons.','A(BR) menghasilkan fitur respons yang dipengaruhi pencocokan dua arah. Gradien tetap mengalir ke encoder dan layer baru; BGE tetap frozen.'),
          ('summary = torch.cat([matched','Ringkasan ujung setelah matching','Menyambung komponen ujung dua arah menjadi128 fitur.','Ambil64komponen pertama dari token akhir dan64komponen terakhir dari token awal setelah matching. Tidak ada weighted attention pooling E26. Linear output128→768 diwarisi dari E23.'),
          ('self.last_cross_attention =','Salinan matriks untuk inspeksi','Menyimpan bobot tanpa memutus jalur training aktual.','Matriks dilepas dari graph hanya untuk ekspor ketika return_features aktif. Workbook memuat top1 per token, bukan seluruh matriks.'),
        ]
        if experiment=='E29':extra += [
          ('class AnswerBiGRUAttention','Arsitektur tetap E26','Mempertahankan BiGRU dan attention pooling pada S3.','E29 tidak menambah co-attention. Yang berubah adalah LR head MLP S2 dan S3 menjadi5e-6; pooling, norm dan proyeksi tetap1e-4.'),
        ]
        c=by_index[30]
        for anchor,title,short,body in extra:
            line=next(i+1 for i,t in enumerate(c['source'].splitlines()) if anchor in t)
            old=next((n for n in c['notes'] if n['line']==line),None)
            data=dict(line=line,title=title,short=short,body=body,anchor=anchor,kind='fundamental',term=None)
            if old:old.update(data)
            else:c['notes'].append(data)
    if experiment=='E29':
        c=by_index[12]
        for anchor,title,body in [('S2_MLP_LR=S3_MLP_LR=','LR kedua MLP5e-6','Seluruh parameter head MLP pada S2 dan S3 belajar dengan5e-6, turun20kali dari acuan E26. BGE tetap frozen; linear S1 tidak berubah.'),('S3_LAYERS_LR=','LR BiGRU dan attention tetap','BiGRU, attention pooling, LayerNorm dan proyeksi ringkasan memakai1e-4, terpisah dari MLP. Nama kelompok head pada versi lama tidak berarti semua layer baru ikut diturunkan.')]:
            line=next(i+1 for i,t in enumerate(c['source'].splitlines()) if anchor in t)
            c['notes'].append(dict(line=line,title=title,short=body,body=body,anchor=anchor,kind='fundamental',term=None))
    mappings=[]
    specs=[
      ('Hyperparameter dan optimisasi','MODEL_NAME, EPOCHS, PATIENCE, LEARNING_RATE',12,'ROBERTA_CHECKPOINT=','Nama pretrained, batas latihan, batch dan LR ada pada blok pengaturan. Encoder S1/S3 5e-6; head/proyeksi/layer baru 1e-4.'),
      ('Konstruksi konteks / text pair','Soal + kunci jawaban dipasangkan dengan jawaban siswa',20,'def context_token_ids','Ketiga teks masuk semua skenario. E22 memakai context_pair; E21/E23/E24 memakai single_sequence. Informasi sama, susunan token berbeda.'),
      ('Dataset PyTorch / X dan y','Dataset berisi teks dan skor guru',14,'class Bundle','Peran Dataset tersebar: Bundle menyimpan tiga teks dan target rater, Engine menyiapkan batch tensor. Tidak harus ada variabel bernama X.'),
      ('Inisialisasi model pretrained','AutoModelForSequenceClassification(..., num_labels=1)',21,'AutoModel.from_pretrained(','Backbone dimuat melalui AutoModel; head regresi dibuat sendiri dalam SingleScoreModel, bukan classifier bawaan AutoModelForSequenceClassification.'),
      ('Pooling / ContextPooler','CLS → Dense → GELU → Dropout',30,'self.sequence_pool =' if experiment in ('E23','E24','E25','E26','E27','E28','E29') else 'essay = mean_by_essay','S1 tetap CLS. S3 '+('memakai BiGRU token jawaban, state128 → proyeksi768.' if experiment in ('E23','E28') else 'memakai attention berbobot pada token jawaban.' if experiment in ('E24','E25') else 'memakai urutan output BiGRU128 → attention → ringkasan128 → proyeksi768.' if experiment in ('E26','E29') else 'memakai BiGRU128 → co-attention referensi → ujung128 → proyeksi768.' if experiment=='E27' else 'acuan memakai CLS lalu mean antar-window.')+' Ini berbeda dari ContextPooler pada screenshot.'),
      ('Regression head / classifier','classifier: Linear(768, 1)',30,'self.head = nn.Linear(768, 1)','S1 memakai head linear. S2/S3 memakai MLP 1024→256→1. Layer terakhir MLP tetap linear; keseluruhan head mempunyai hidden nonlinier.'),
      ('Ekstraksi fitur sebelum training','last_hidden_state[:, 0, :] menghasilkan vektor768',24,'def roberta_pretrained_vectors(','Fitur awal dibaca di roberta_pretrained_vectors. Fitur aktual saat model berjalan ada pada forward; fitur checkpoint final diekspor sesudah training.'),
      ('Pelatihan lima fold','for fold_idx ... model baru ... train/validation',38,'def train_all(','make_split menentukan anggota; train_all mengatur fold/refit/test; fit memperbarui bobot satu model. Semua skenario memakai development/test yang sama.'),
      ('Loss dan optimizer','MSELoss, AdamW, backward, optimizer.step',36,'loss = torch.mean','MSE training memakai target0–1. Backward menghitung gradien; optimizer memperbarui parameter. RMSE validasi memakai poin0–100.'),
      ('Prediksi dan label guru','rater dibanding hasil model',39,'def test_score_comparison(','Bagian ini membaca prediksi test tersimpan dan menyandingkan skor guru, prediksi mentah, serta salinan pembulatan. RMSE/Pearson memakai skor kontinu; QWK/AC2 memakai kategori.'),
    ]
    for title,example,index,anchor,body in specs:
        if experiment in ('E27','E28') and title=='Hyperparameter dan optimisasi':
            body='Encoder S1/S3 5e-6; head kontrol S1/S2 dan proyeksi BGE1e-4. Hanya MLP S3 '+('5e-6' if experiment=='E28' else '1e-4')+'; BiGRU/norm tetap1e-4. Semua LR dan rentang latihan ada di blok pengaturan.'
        if experiment=='E29' and title=='Hyperparameter dan optimisasi':
            body='E29: encoder S1/S3 dan MLP S2/S3 5e-6. Linear S1, proyeksi BGE, BiGRU, attention pooling dan norm1e-4. Layer selain MLP tidak ikut diturunkan.'
        c=by_index[index];line=next(i+1 for i,t in enumerate(c['source'].splitlines()) if anchor in t)
        mappings.append(dict(title=title,example=example,explanation=body,cell=index,line=line,ordinal=c['ordinal'],anchor=anchor))
        n=next((n for n in c['notes'] if n['line']==line),None)
        if n is None:
            n=dict(line=line,title=title,short='Menghubungkan bagian screenshot dengan implementasi pada notebook ini.',body=body,anchor=anchor,kind='fundamental',term=None)
            c['notes'].append(n)
        n['screenshot']=title
    catalog=[]
    for c in cells:
        c['notes'].sort(key=lambda n:n['line'])
        for i,n in enumerate(c['notes']):
            n['id']=f'c{c["index"]}-n{i}'
            n.setdefault('anchor',c['source'].splitlines()[n['line']-1].strip())
            n.setdefault('short','Menjelaskan bagian ini dalam alur eksperimen AES.')
            catalog.append(dict(cell=c['index'],ordinal=c['ordinal'],line=n['line'],title=n['title'],short=n['short'],anchor=n['anchor']))
    return mappings,catalog
