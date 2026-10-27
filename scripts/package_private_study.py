"""Kemas HTML pribadi menjadi satu unduhan terenkripsi, tanpa menanam kunci."""
import argparse
import base64
import gzip
import hashlib
import json
import re
from pathlib import Path
import secrets
import zipfile

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

TEMPLATE = '''<!doctype html>
<html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'; font-src data:; object-src 'none'; base-uri 'none'; form-action 'none'">
<title>Buku AES lengkap · Buka dengan kunci</title>
<style>*{box-sizing:border-box}body{margin:0;background:#102041;color:#142344;font:16px/1.65 'Segoe UI',Arial,sans-serif;min-height:100vh;display:grid;place-items:center;padding:24px}main{width:min(580px,100%);background:#fffdf5;border:3px solid #15213e;border-radius:8px 24px 8px 8px;padding:32px;box-shadow:7px 7px 0 #13d6e9}small{font-weight:800;letter-spacing:2px;color:#ad1a3c}h1{font-size:32px;line-height:1.2;margin:14px 0}p{color:#435773}label{display:block;font-size:14px;font-weight:700;margin-top:22px}input{width:100%;padding:13px;border:1px solid #879fb7;border-radius:5px;font:16px monospace;margin-top:6px}button{font:700 14px 'Segoe UI',sans-serif;border:2px solid #15213e;border-radius:5px;padding:12px 15px;cursor:pointer;background:#ffe86b;color:#142344}button:disabled{opacity:.6;cursor:wait}.actions{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px}.actions button:first-child{background:#14d8eb}#status{font-size:14px;margin:18px 0 0}input:focus-visible,button:focus-visible{outline:3px solid #dd2470;outline-offset:3px}.note{font-size:12px;margin:22px 0 0;border-top:1px solid #cfdae3;padding-top:14px}</style></head>
<body><main><small>AES / BUKU BELAJAR LENGKAP</small><h1>Seluruh hasil.<br>Satu buku.</h1><p>File ini memuat kode, output notebook dan Excel asli. Masukkan kunci pembuka yang dikirim di chat untuk membaca buku secara offline.</p>
<form id="unlock"><label for="password">Kunci pembuka</label><input id="password" type="password" autocomplete="off" autocapitalize="off" spellcheck="false" required><div class="actions"><button id="open" type="submit">Buka buku →</button><button id="show" type="button" aria-pressed="false">Tampilkan kunci</button></div></form><p id="status" role="status" aria-live="polite">Tidak perlu mengunggah ulang ZIP atau notebook.</p><p class="note">Buku berisi jawaban dan nilai siswa. Isi file terenkripsi agar tidak terbuka langsung pada repository publik. Kunci diproses di browser, tanpa jaringan atau penyimpanan otomatis.</p></main>
<script id="encryptedBook" type="application/octet-stream">__PAYLOAD__</script>
<script>
(()=>{'use strict';
const form=document.getElementById('unlock'),password=document.getElementById('password'),button=document.getElementById('open'),status=document.getElementById('status');
const salt='__SALT__',iv='__IV__',expected='__SHA__';
function decode(value){const size=value.length*3/4-(value.endsWith('==')?2:value.endsWith('=')?1:0),result=new Uint8Array(size);let offset=0;for(let start=0;start<value.length;start+=1048576){const chunk=atob(value.slice(start,start+1048576));for(let i=0;i<chunk.length;i++)result[offset++]=chunk.charCodeAt(i)}return result}
document.getElementById('show').addEventListener('click',event=>{const show=password.type==='password';password.type=show?'text':'password';event.currentTarget.textContent=show?'Sembunyikan kunci':'Tampilkan kunci';event.currentTarget.setAttribute('aria-pressed',String(show))});
form.addEventListener('submit',async event=>{
 event.preventDefault();if(!password.value)return;
 button.disabled=true;status.textContent='Membuka dan memeriksa buku lengkap…';
 try{
  if(!crypto.subtle||!window.DecompressionStream)throw Error('Gunakan Chrome, Edge atau Firefox versi terbaru, lalu buka file yang sudah diunduh.');
  const material=await crypto.subtle.importKey('raw',new TextEncoder().encode(password.value),'PBKDF2',false,['deriveKey']);
  const key=await crypto.subtle.deriveKey({name:'PBKDF2',salt:decode(salt),iterations:600000,hash:'SHA-256'},material,{name:'AES-GCM',length:256},false,['decrypt']);
  const payload=document.getElementById('encryptedBook');let encrypted=decode(payload.textContent.trim());let compressed;try{compressed=await crypto.subtle.decrypt({name:'AES-GCM',iv:decode(iv)},key,encrypted);encrypted=null;payload.remove()}catch{throw Error('Kunci tidak cocok atau file unduhan tidak lengkap. Salin kunci dari chat, lalu coba lagi.')}
  const bytes=await new Response(new Blob([compressed]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  const actual=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))).map(v=>v.toString(16).padStart(2,'0')).join('');
  if(actual!==expected)throw Error('Isi buku tidak cocok dengan checksum. Unduh file kembali.');
  const html=new TextDecoder('utf-8',{fatal:true}).decode(bytes);password.value='';
  document.open();document.write(html);document.close();
 }catch(error){status.textContent=error.message;button.disabled=false}
});
})();
</script></body></html>
'''


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--key-file',type=Path,required=True)
    parser.add_argument('--archive',type=Path,help='ZIP satu HTML terenkripsi untuk unduhan besar.')
    parser.add_argument('--parts-dir',type=Path,help='Pecah arsip besar menjadi bagian aman untuk batas GitHub; buat HTML pengunduh.')
    parser.add_argument('--download-ref',default='codex/aes-learning',help='Commit/ref yang menyimpan bagian arsip agar pengunduh tetap cocok.')
    parser.add_argument('--artifact-name',default='Belajar_AES_MLP_Lengkap',help='Nama dasar paket/HTML; memungkinkan buku terpisah tanpa menimpa buku lama.')
    parser.add_argument('--download-template',type=Path,help='Template HTML pengunduh khusus buku ini.')
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_]+',args.artifact_name):raise ValueError('Nama artifact tidak sah.')
    if args.key_file.exists():
        password=json.loads(args.key_file.read_text())['password']
    else:
        password=secrets.token_urlsafe(12)
    original=args.input.read_bytes()
    salt,iv=secrets.token_bytes(16),secrets.token_bytes(12)
    key=PBKDF2HMAC(algorithm=hashes.SHA256(),length=32,salt=salt,iterations=600000).derive(password.encode())
    compressed=gzip.compress(original,compresslevel=9,mtime=0)
    ciphertext=AESGCM(key).encrypt(iv,compressed,None)
    assert gzip.decompress(AESGCM(key).decrypt(iv,ciphertext,None))==original
    encode=lambda value:base64.b64encode(value).decode('ascii')
    document=TEMPLATE.replace('__PAYLOAD__',encode(ciphertext)).replace('__SALT__',encode(salt)).replace('__IV__',encode(iv)).replace('__SHA__',hashlib.sha256(original).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(document,encoding='utf-8')
    if args.archive:
        args.archive.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(args.archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
            archive.writestr(args.artifact_name+'.html',document.encode('utf-8'))
        with zipfile.ZipFile(args.archive) as archive:
            assert archive.read(args.artifact_name+'.html')==document.encode('utf-8')
        if not args.parts_dir and args.archive.stat().st_size>=100*1024**2:
            raise ValueError('Arsip melebihi batas file GitHub 100 MiB; jangan dipublikasikan.')
    if args.parts_dir:
        if not args.archive:raise ValueError('--parts-dir memerlukan --archive.')
        args.parts_dir.mkdir(parents=True,exist_ok=True)
        raw=args.archive.read_bytes();parts=[]
        for i,start in enumerate(range(0,len(raw),80*1024**2),1):
            chunk=raw[start:start+80*1024**2];name=f'{args.artifact_name}.part{i}'
            (args.parts_dir/name).write_bytes(chunk)
            parts.append({'name':name,'size':len(chunk),'sha256':hashlib.sha256(chunk).hexdigest()})
        manifest={'base':f'https://raw.githubusercontent.com/MalikHariyanto/Automated_Essay_Scoring/{args.download_ref}/downloads/','size':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'parts':parts}
        template=(args.download_template or Path(__file__).parent/'templates/mlp_download.html').read_text()
        (args.parts_dir/(args.artifact_name+'.html')).write_text(template.replace('__MANIFEST__',json.dumps(manifest)))
        assert b''.join((args.parts_dir/p['name']).read_bytes() for p in parts)==raw
        print(json.dumps({'split_archive_verified':True,'parts':parts,'archive_sha256':manifest['sha256']}))
    args.key_file.parent.mkdir(parents=True,exist_ok=True)
    args.key_file.write_text(json.dumps({'password':password,'plaintext_sha256':hashlib.sha256(original).hexdigest(),
        'download_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest()},indent=2)+'\n')
    args.key_file.chmod(0o600)
    print(json.dumps({'file':str(args.output),'size_mib':round(args.output.stat().st_size/1024**2,2),'roundtrip_verified':True,'key_file':str(args.key_file)}))
    if args.archive:
        print(json.dumps({'archive':str(args.archive),'size_mib':round(args.archive.stat().st_size/1024**2,2),'sha256':hashlib.sha256(args.archive.read_bytes()).hexdigest(),'archive_roundtrip_verified':True}))


if __name__=='__main__':
    main()
