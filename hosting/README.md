# Hosting statis opsional

Dua buku hasil aktif tetap berada di [downloads](../downloads/README.md), dan preview publik berada di `docs/`. Hosting adalah pilihan membaca, bukan layanan prediksi model.

```bash
node scripts/prepare_paper_site.mjs
```

Perintah menyiapkan preview tiga input terbaru pada `outputs/paper_site/public/`. Tambahkan `--earlier` untuk buku sebelumnya. [Script](../scripts/prepare_paper_site.mjs) dan [konfigurasi Cloudflare](code-belajar-paper/wrangler.jsonc) tidak otomatis menerbitkan situs atau memindahkan akun.

Dokumentasi hosting lama ada di [arsip](../docs/arsip/hosting_belajar_paper.md). Nama konfigurasi masih memakai `code-belajar-paper` untuk kompatibilitas. Preview publik tidak membuka jawaban individual dari paket terenkripsi.
