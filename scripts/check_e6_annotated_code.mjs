// Uji buku kode: source utuh, padanan screenshot, satu scroll dan layar penuh.
import {createRequire} from 'node:module';
import fs from 'node:fs';
import assert from 'node:assert/strict';
const require=createRequire(import.meta.url),{chromium}=require('playwright');
const root='outputs/e6_annotated/',base='http://127.0.0.1:8765/downloads/Belajar_Kode_E6.html';
fs.mkdirSync(root,{recursive:true});
const b=await chromium.launch({executablePath:'/usr/bin/chromium',args:['--no-sandbox','--disable-dev-shm-usage']});
const checks=[],errors=[];const ok=text=>{checks.push(text);console.log('PASS',text)};
try{
 const context=await b.newContext({viewport:{width:1440,height:1000},permissions:['clipboard-read','clipboard-write']});
 const p=await context.newPage();p.on('pageerror',e=>errors.push(e.message));
 await p.goto(base);await p.waitForFunction(()=>window.E6Guide);
 await p.route('**/*',r=>r.abort());
 const source=await p.evaluate(()=>E6Guide.data.cells.map(c=>({index:c.index,source:c.source})));
 for(const c of source){const restored=await p.locator(`#block-${c.index} .code-text`).allTextContents();assert.equal(restored.join('\n'),c.source)}
 assert.equal(await p.locator('.code-block').count(),30);assert.equal(await p.locator('.sticky-note').count(),153);
 ok('Seluruh 30 blok / 2695 baris ditampilkan identik dengan source, bukan cuplikan.');
 await p.locator('[data-quick=head]').first().click();assert.match(await p.locator('.is-destination .code-text').innerText(),/self.head = nn.Linear\(768, 1\)/);
 const alignment=await p.locator('.is-destination').evaluate(e=>{const segment=e.closest('.code-segment'),code=segment.querySelector('.segment-code').getBoundingClientRect(),note=segment.querySelector('.margin-notes').getBoundingClientRect();return {sameTop:Math.abs(code.top-note.top)<2,beside:note.left>code.left}});assert(alignment.sameTop&&alignment.beside);const exact=await p.evaluate(()=>[...document.querySelectorAll('.sticky-note')].every(note=>{const segment=note.closest('.code-segment'),row=segment.querySelector('[data-line="'+note.dataset.noteLine+'"]');return row&&Math.abs(row.getBoundingClientRect().top-note.getBoundingClientRect().top)<2}));assert(exact);
 await p.screenshot({path:root+'desktop_head.png'});ok('Tombol head menuju kode tepat; sticky note dan ruas kode sejajar berdampingan.');
 const before=await p.evaluate(()=>scrollY);await p.locator('.is-destination').hover();await p.mouse.wheel(0,500);await p.waitForTimeout(150);assert(await p.evaluate(y=>scrollY>y,before));
 const nested=await p.locator('.segment-code,.margin-notes,.code-row').evaluateAll(es=>es.some(e=>['auto','scroll'].includes(getComputedStyle(e).overflowY)));assert.equal(nested,false);ok('Kode dan catatan bergulir melalui halaman yang sama tanpa scrollbar vertikal terpisah.');
 await p.locator('[data-view=map]').click();assert.equal(await p.locator('.mapping-card').count(),10);
 const jumps=await p.locator('.mapping-card .jump-button').count();for(let i=0;i<jumps;i++){await p.locator('[data-view=map]').click();await p.locator('.mapping-card .jump-button').nth(i).click();assert(await p.locator('.is-destination').isVisible())}
 await p.locator('[data-view=map]').click();assert.match(await p.locator('#mapView').innerText(),/tidak diterapkan di E6/);await p.screenshot({path:root+'mapping.png'});ok('Sepuluh padanan screenshot benar-benar membuka baris E6, termasuk perbedaan konteks pair/pooling.');
 await p.locator('[data-view=find]').click();assert.equal(await p.locator('.catalog-item').count(),99);await p.getByLabel('Nama bagian, fungsi, atau kode').fill('Head regresi MLP S3');assert.equal(await p.locator('.catalog-item').count(),1);await p.locator('.catalog-item summary').click();await p.locator('.catalog-item .jump-button').click();assert.match(await p.locator('.is-destination .code-text').innerText(),/self.head = nn.Sequential/);ok('Pencarian 99 bagian bisa difilter dan membuka lokasi MLP S3 yang tepat.');
 await p.getByLabel('Cari dalam kode').fill('S3_ENCODER_LR=');await p.getByLabel('Cari dalam kode').press('Enter');assert.equal(await p.locator('.is-current-match').count(),1);assert.match(await p.locator('.is-current-match').innerText(),/5e-6/);ok('Pencarian seluruh source menyorot dan menggulir ke kecocokan.');
 await p.getByLabel('Lompat ke blok kode').selectOption('30');await p.locator('#block-30 .copy-block').click();assert.equal(await p.evaluate(()=>navigator.clipboard.readText()),source.find(c=>c.index===30).source);ok('Salin blok menghasilkan source lengkap tanpa catatan atau nomor baris.');
 await p.locator('#fullscreen').click();await p.waitForFunction(()=>E6Guide.getState().full);assert.equal(await p.locator('#workspace').evaluate(e=>e.getBoundingClientRect().top),0);assert.equal(await p.evaluate(()=>getComputedStyle(document.body).overflow),'hidden');await p.locator('#exitFullscreen').click();await p.waitForFunction(()=>!E6Guide.getState().full);ok('Mode layar penuh mengisi viewport dan kembali tanpa mengganti kode.');
 for(const [width,height] of [[390,844],[768,1024],[1440,1000]]){await p.setViewportSize({width,height});await p.evaluate(()=>E6Guide.quick('features'));assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth>innerWidth+2),false);const nodes=await p.locator('.is-destination').evaluate(e=>{const c=e.closest('.code-segment');return {code:c.querySelector('.segment-code').getBoundingClientRect().width,notes:c.querySelector('.margin-notes').getBoundingClientRect().width}});assert(nodes.code>100&&nodes.notes>100);if(width===390)await p.screenshot({path:root+'mobile.png'})}
 ok('HP, tablet dan desktop bebas overflow halaman; catatan tetap dapat dibaca.');
 const fallbackContext=await b.newContext({viewport:{width:1440,height:900}});await fallbackContext.addInitScript(()=>Object.defineProperty(Element.prototype,'requestFullscreen',{value:undefined,configurable:true}));const f=await fallbackContext.newPage();await f.goto(base);await f.waitForFunction(()=>window.E6Guide);await f.locator('[data-quick=features]').click();await f.locator('#fullscreen').click();assert(await f.locator('#workspace').evaluate(e=>e.classList.contains('is-full')));await f.keyboard.press('Escape');assert.equal(await f.locator('#workspace').evaluate(e=>e.classList.contains('is-full')),false);ok('Fallback layar penuh bekerja ketika API tidak tersedia; Esc mengembalikan halaman.');
 assert.deepEqual(errors,[]);const report={passed:checks.length,failed:0,checks,browser:b.version(),scope:'Source reading/UI only; no model training'};fs.writeFileSync(root+'browser.json',JSON.stringify(report,null,2));console.log(JSON.stringify({passed:checks.length,failed:0}));
}finally{await b.close()}
