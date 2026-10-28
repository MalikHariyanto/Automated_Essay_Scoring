# Hosting statis code belajar paper

> Dokumen tahap terdahulu. Angka dan keputusan dibaca dalam konteks versi ini; untuk kondisi aktif gunakan [status proyek](../../STATUS_PROYEK.md) dan [panduan AI](../../PANDUAN_AI_PROYEK.md).

> Catatan historis: HTML terpisah pada dokumen ini telah digantikan oleh [dua buku terbaru](../../downloads/README.md). Status kode utama mengikuti [status terbaru](../../STATUS_PROYEK.md); file lama dapat ditelusuri pada commit versi asal.

Web menampilkan versi HTML yang sudah disetujui. Tab, navigasi, gambar, dan catatan berjalan di browser; tidak ada database, komentar, formulir unggahan, atau layanan penilaian model. Login email ditempatkan di depan web melalui Cloudflare Access.

Cloudflare Workers Static Assets menyediakan hosting gratis dan alamat `code-belajar-paper.<nama-akun>.workers.dev`, sehingga domain berbayar tidak diperlukan. Pilih Workers Free dan Zero Trust Free. Menurut panduan resmi Cloudflare, aktivasi Zero Trust Free tetap meminta data pembayaran, walaupun paket Free tidak dikenai biaya. Jika data pembayaran tidak boleh diminta, pilihan penyedia perlu diganti sebelum deployment.

## Berkas hosting

Dari direktori utama repository:

```bash
node scripts/prepare_paper_site.mjs
```

Perintah ini menyiapkan `outputs/paper_site/public/index.html` dan `_headers` dari `docs/code belajar paper.html`. Isi HTML tetap sama; tambahan hanya metadata `noindex` untuk mesin pencari. Header `X-Robots-Tag` juga meminta halaman tidak diindeks. Pengaturan ini tidak menggantikan pemeriksaan izin email.

Konfigurasi berada di `hosting/code-belajar-paper/wrangler.jsonc`. Semua hasil build berada di direktori `outputs` yang diabaikan Git. Daftar email, data pembayaran, dan kredensial tidak dimasukkan ke source.

## Menyiapkan akun dan deployment

1. Buat akun di [Cloudflare](https://dash.cloudflare.com/sign-up), kemudian aktifkan Zero Trust Free. Lengkapi data pembayaran hanya di dashboard resmi Cloudflare.
2. Hubungkan repository GitHub ke Workers Builds. Gunakan build command `node scripts/prepare_paper_site.mjs` dan deploy command `npx wrangler@4 deploy --config hosting/code-belajar-paper/wrangler.jsonc`. Direktori build adalah root repository dan branch adalah `codex/aes-learning`.
3. Deployment pertama sengaja tidak mengaktifkan URL publik atau preview: `workers_dev` dan `preview_urls` masih `false`.
4. Buka **Workers & Pages → code-belajar-paper → Access**, lalu lindungi Worker dengan **All traffic**. Gunakan perlindungan seluruh Worker supaya production dan preview tidak menjadi jalan masuk lain.
5. Di Zero Trust, tambahkan identity provider **One-time PIN**. Edit kebijakan Access menjadi **Allow → Include → Emails**, lalu masukkan email sendiri dan email teman yang boleh membaca. Gunakan alamat email lengkap, bukan izin untuk semua pengguna `gmail.com`.
6. Pilih One-time PIN sebagai metode login aplikasi dan cek bahwa kebijakan tersebut sudah tersimpan. Setelah perlindungan aktif, ubah `workers_dev` menjadi `true` pada konfigurasi dan deploy ulang. `preview_urls` tetap `false`.
7. Uji dari jendela incognito: tanpa login harus muncul pintu masuk Access; email yang diizinkan menerima kode dan dapat membaca; email lain tidak mendapat akses. Verifikasi juga URL `/index.html` dan URL langsung aset tidak membuka isi sebelum login.

Jika deployment dilakukan dari komputer sendiri, jalankan build di atas, lalu `npx wrangler@4 login` dan deploy command yang sama. Autentikasi dilakukan melalui browser resmi; jangan menaruh token atau password di chat maupun GitHub.

## Status dan batas privasi

Berkas siap untuk deployment, tetapi akun Cloudflare dan izin email belum dikonfigurasi oleh agent. Belum ada URL hosting aktif yang telah diverifikasi. Hosting baru boleh disebut privat setelah pemeriksaan akses anonim, email diizinkan, serta email ditolak selesai.

Repository GitHub saat ini publik dan salinan HTML sudah diterbitkan di sana. Pembatasan email melindungi akses website yang dihosting; jika isi HTML juga harus privat, salinan publik dan pengaturan repository perlu ditangani secara terpisah.

Referensi resmi: [Workers Static Assets — biaya](https://developers.cloudflare.com/workers/static-assets/billing-and-limitations/), [perlindungan Workers dengan Access](https://developers.cloudflare.com/workers/configuration/cloudflare-access/), [kode masuk melalui email](https://developers.cloudflare.com/cloudflare-one/integrations/identity-providers/one-time-pin/), dan [setup Zero Trust](https://developers.cloudflare.com/cloudflare-one/setup/).

Bukti pemeriksaan build, browser desktop/HP, dan Wrangler dry-run: [paper_site_preparation.json](../verifikasi/paper_site_preparation.json). Pemeriksaan ini tidak membuktikan login email atau deployment aktif.
