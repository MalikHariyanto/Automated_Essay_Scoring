"""Materi meja belajar E6; sumber kode dan bukti hasil tetap dibedakan."""
import copy

STUDY_COMMIT = 'd8c21accc6e0bbff7a9d95c02a553b8bfeed8340'
STUDY_PATH = 'notebooks/experiments/AES_300_E6_S3_LR5e6_CPU_Reuse.ipynb'
FUNCTION_HINTS_E6 = {
    'optimizer_groups': 'Mengelompokkan parameter yang belajar menjadi encoder RoBERTa, proyeksi, dan head. S3 memilih LR5e-6; S1 mengikuti1e-5. Parameter frozen tidak masuk optimizer.',
    'inspect_reuse_runtime': 'Memeriksa kompatibilitas versi pustaka sumber kontrol. Memakai model final untuk prediksi berbeda dari melanjutkan optimizer training lintas versi.',
    'inspect_control_source': 'Memeriksa folder kontrol lama: identitas data, split, pengaturan, metrik, dan checksum model S1/S2 harus sesuai sebelum dipakai ulang.',
    'prepare_reused_controls': 'Mencari sumber S1/S2 selesai yang kompatibel, memverifikasi dan menyalin hasil kontrol ke run E6. Tidak melatih kedua kontrol lagi.',
    'train_all': 'Alur E6: verifikasi atau pulihkan hasil, pakai kontrol S1/S2, jalankan CV S3, kunci epoch, refit S3, lalu laporkan test dengan asal model yang jelas.',
    'fit': 'Melatih satu model pada satu fold atau refit. MSE target0–1 memperbarui bobot; minimum RMSE validasi menentukan bobot CV terpilih. Test tidak dipakai early stopping.',
    'save_training_progress': 'Source E6 ini menyimpan checkpoint per epoch dengan dua rujukan terbaru pada JSON. File lama tetap ada untuk pembersihan manual; bukan mekanisme dua slot dari notebook tiga seed.',
}


def curriculum(chapters, glossary, pins):
    chapters, glossary = copy.deepcopy(chapters), copy.deepcopy(glossary)
    chapters[0]['title'] = 'Hasil eksperimen 6'
    chapters[0]['read'] = [
        'E6 S3 memiliki RMSE test 6,3744; S1 6,3805 dan S2 6,9481. Selisih S3–S1 hanya sekitar 0,0061 poin; ini minimum numerik, bukan bukti keunggulan statistik.',
        'Rata-rata RMSE CV S3 E6 adalah 5,9915. Pearson, QWK dan AC2 test S1 tetap lebih tinggi daripada S3; jangan menyebut E6 unggul pada semua metrik.',
        'E6 mempertahankan fusion MLP dan memperkecil LR RoBERTa S3 menjadi 5e-6. S1/S2 memakai hasil MLP awal yang diverifikasi. Eksperimen tiga seed berikutnya belum mempunyai hasil di buku ini.'
    ]
    chapters[2]['read'][1] = 'Source E6 memilih CPU float32. Buku membaca source repo dan hasil Excel; log instalasi notebook MLP lama tidak ditempelkan pada E6.'
    chapters[3]['read'].append('S1 encoder LR 1e-5; S3 encoder LR 5e-6; proyeksi/head 1e-4. REUSE_RUN menunjuk hasil S1/S2 yang sudah selesai; ini bukan melatih ulang tiga skenario.')
    chapters[11]['read'][1] = 'Optimizer memilih LR encoder menurut skenario: S1 1e-5, S3 5e-6. Head/proyeksi 1e-4. BGE tetap frozen; hanya fitur dan head/proyeksi yang terkait dipakai.'
    chapters[12]['read'] = [
        'E6 memeriksa kontrol S1/S2 yang sudah selesai, menyalin bukti hasil yang cocok, lalu menjalankan lima fold dan refit S3 baru. Reuse bukan pengulangan independen.',
        'Epoch terpilih S3 pada fold 1–5: 14, 10, 8, 16, 10. Ceil median menghasilkan refit 10 epoch. S1/S2 mempertahankan hasil sumbernya.',
        'RMSE/Pearson memakai prediksi mentah; QWK/AC2 memakai salinan kategori. Tabel dan riwayat E6 berasal dari ZIP hasil E6, bukan dari log MLP awal.'
    ]
    chapters[15]['read'][-1] = 'Buku lengkap sudah memuat semua sumber hasil. Tidak perlu mengunggah ZIP atau notebook. Unduhan Excel tetap mengembalikan berkas asli; buku tidak menjalankan training.'
    glossary['learning rate'] = ('Ukuran langkah pembaruan bobot', 'E6 memakai LR RoBERTa S1 1e-5, RoBERTa S3 5e-6, dan head/proyeksi 1e-4. BGE frozen tidak diperbarui. Besar LR bukan skor kualitas model.')
    revised = []
    for cell, anchor, title, body, term in pins:
        if cell == 12 and anchor == "device='auto'":
            anchor, title, body = "device='cpu'", 'CPU dipilih eksplisit', 'E6 ini memakai CPU float32. Bukan mode otomatis dan bukan notebook tiga seed; source mengulang S3 serta memakai kontrol S1/S2 selesai.'
        if cell == 12 and anchor == 'ENCODER_LR=':
            body = 'ENCODER_LR=1e-5 berlaku pada S1. S3_ENCODER_LR=5e-6 dipilih khusus S3. Head/proyeksi tetap 1e-4; encoder BGE frozen.'
        if cell == 38 and anchor == 'for scenario in SCENARIOS:':
            body = 'E6 memeriksa serta memakai hasil kontrol S1/S2 yang cocok; hanya S3 dilatih baru. Tiga skenario tetap dilaporkan dengan asal model yang dicatat.'
        revised.append((cell, anchor, title, body, term))
    revised.extend([
        (12, 'S3_ENCODER_LR=', 'Perubahan utama E6', 'Learning rate encoder RoBERTa S3 dikurangi separuh dari MLP awal: 1e-5 menjadi 5e-6. Arsitektur fusion dan LR head/proyeksi tetap.', 'learning rate'),
        (12, 'REUSE_RUN=', 'Kontrol S1/S2 dipakai ulang', 'Folder sumber harus cocok data, split, konfigurasi dan model. Mengambil S1/S2 selesai menghemat latihan; hasil ini tidak dianggap run independen.', 'checkpoint'),
        (36, "model.scenario=='S3'", 'Optimizer membedakan skenario', 'Pemilihan LR encoder dilakukan pada kelompok RoBERTa: S3 memakai s3_encoder_lr, S1 memakai encoder_lr. Proyeksi dan head punya kelompok sendiri.', 'learning rate'),
    ])
    return chapters, glossary, revised
