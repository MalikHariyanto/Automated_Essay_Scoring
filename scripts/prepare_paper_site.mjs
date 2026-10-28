#!/usr/bin/env node
// Prepare the approved, read-only paper guide for static hosting.
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
// Hanya dua aplikasi hasil terbaru; default buku tiga variabel E21–E26.
const sourceName = process.argv.includes('--earlier') ? 'docs/belajar_MLP_Spider.html' : 'docs/belajar_Tiga_Input_Spider.html';
const source = join(root, sourceName);
const output = join(root, 'outputs', 'paper_site', 'public');
const approved = await readFile(source, 'utf8');
const marker = '<meta name="viewport" content="width=device-width,initial-scale=1">';
if (approved.split(marker).length !== 2) {
  throw new Error('Expected exactly one viewport marker in the approved HTML.');
}
const noindex = '<meta name="robots" content="noindex, nofollow, noarchive">';
const html = approved.includes('name="robots"') ? approved : approved.replace(marker, `${marker}\n${noindex}`);
const headers = '/*\n  X-Robots-Tag: noindex, nofollow, noarchive\n  X-Content-Type-Options: nosniff\n  Referrer-Policy: no-referrer\n';
await mkdir(output, { recursive: true });
await writeFile(join(output, 'index.html'), html);
await writeFile(join(output, '_headers'), headers);
const hash = value => createHash('sha256').update(value).digest('hex');
const manifest = {
  source: sourceName,
  source_sha256: hash(approved),
  index_sha256: hash(html),
  index_bytes: Buffer.byteLength(html),
  source_preserved_except_noindex_meta: html.replace(`\n${noindex}`, '') === approved,
  files: ['index.html', '_headers'],
  access: 'Must be configured in Cloudflare Access before enabling a public URL.',
};
await writeFile(join(output, '..', 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
console.log(JSON.stringify(manifest, null, 2));
