// Pembaca arsip pembanding; tidak menambahkan halaman khusus per eksperimen.
(()=>{'use strict';
const A=window.MLPStudy;
function archiveReader(snapshot){let bytes=null,archive=null;const cache=new Map();return {
 async loadResults(raw){const actual=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',raw))).map(x=>x.toString(16).padStart(2,'0')).join('');if(actual!==snapshot.zip_sha256)throw Error('Arsip eksperimen berbeda dari sumber terverifikasi.');archive=await JSZip.loadAsync(raw);bytes=raw;cache.clear();window.dispatchEvent(new Event('mlp-private-ready'))},
 async readSheet(index,name){if(!archive)throw Error('Tabel individual tersedia pada buku lengkap.');const key=index+'|'+name;if(!cache.has(key))cache.set(key,A.readWorkbookSheet(await archive.file(snapshot.books[index].file).async('uint8array'),name));return cache.get(key)},
 getBytes:()=>bytes,getState:()=>({loaded:Boolean(bytes)})
}}
window.MLPFrozen=archiveReader(A.data.frozen);window.MLPLora=archiveReader(A.data.lora);
})();
