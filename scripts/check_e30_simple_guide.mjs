// Source fidelity and offline browser verification; no research data is loaded.
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {createRequire} from 'node:module';
const {chromium} = createRequire(import.meta.url)('playwright');
const root = process.cwd();
const sourcePath = 'notebooks/experiments/AES_300_E30_Lengkap_S1_S2_S3_Native_Learnable_GPU.ipynb';
const htmlPath = 'docs/panduan_e30_mlp.html';
const out = 'outputs/e30_simple_guide';
fs.mkdirSync(out, {recursive:true});
const hash = value => crypto.createHash('sha256').update(value).digest('hex');
const raw = fs.readFileSync(sourcePath), notebook = JSON.parse(raw);
const originalHtml = fs.readFileSync(htmlPath);
execFileSync('python', ['scripts/build_e30_simple_guide.py'], {stdio:'pipe'});
assert.equal(hash(fs.readFileSync(htmlPath)), hash(originalHtml));
assert.equal(hash(fs.readFileSync('downloads/Panduan_E30_MLP.html')), hash(originalHtml));
assert.equal(hash(fs.readFileSync(sourcePath)), hash(raw));
const checks = [], errors = [], requests = [];
const ok = message => {checks.push(message); console.log('PASS', message);};
// Chromium in this environment blocks file://. Serve only these two public files.
const server = http.createServer((request,response) => {
  const file = request.url === '/guide.html' ? htmlPath : request.url === '/offline.html' ? path.join(out,'downloaded_offline.html') : null;
  if (!file || !fs.existsSync(file)) {response.writeHead(404);response.end();return;}
  response.writeHead(200,{'content-type':'text/html;charset=utf-8'});response.end(fs.readFileSync(file));
});
await new Promise(resolve => server.listen(0,'127.0.0.1',resolve));
const origin = `http://127.0.0.1:${server.address().port}`;
const browser = await chromium.launch({executablePath:'/usr/bin/chromium', args:['--no-sandbox','--disable-dev-shm-usage']});
try {
 const context = await browser.newContext({viewport:{width:1440,height:1000},acceptDownloads:true});
 await context.route(/^https?:/, route => route.request().url().startsWith(origin+'/') ? route.continue() : route.abort());
 context.on('request', request => {if (/^https?:/.test(request.url()) && !request.url().startsWith(origin+'/')) requests.push(request.url());});
 const page = await context.newPage();
 page.on('pageerror', error => errors.push(error.message));
 await page.goto(origin+'/guide.html');
 await page.waitForFunction(() => window.E30Guide?.codeCellCount === 89);
 const embedded = await page.locator('#notebook-data').textContent();
 assert.equal(JSON.parse(embedded), raw.toString());
 assert.equal(await page.evaluate(() => E30Guide.sourceSha256), hash(raw));
 const allCells = await page.locator('.code-card').evaluateAll(elements => elements.map(element => ({
   cell:Number(element.dataset.cell), role:element.dataset.role,
   lines:[...element.querySelectorAll('.lt')].map(line => line.textContent),
   title:element.querySelector('.code-title').textContent,
   purpose:element.querySelector('.code-purpose').textContent
 })));
 const lines = source => {const result=source.split('\n');if(result.at(-1)==='')result.pop();return result;};
 for (const item of allCells) {
   const original = notebook.cells[item.cell-1];
   assert.equal(item.role, original.metadata.aes30full_role);
   assert.deepEqual(item.lines, lines(original.source));
   assert(item.title.length > 8 && item.purpose.length > 35);
 }
 for (const snippet of await page.locator('.snippet').evaluateAll(elements => elements.map(element => {
   const button = element.querySelector('[data-copy-cell]');
   return {cell:Number(button.dataset.copyCell),start:Number(button.dataset.start),end:Number(button.dataset.end),
     lines:[...element.querySelectorAll('.lt')].map(line => line.textContent)};
 }))) assert.deepEqual(snippet.lines, lines(notebook.cells[snippet.cell-1].source).slice(snippet.start-1,snippet.end));
 const ids = await page.locator('[id]').evaluateAll(elements => elements.map(element=>element.id));
 assert.equal(new Set(ids).size, ids.length);
 for (const target of await page.locator('a[href^="#"]').evaluateAll(elements => elements.map(element=>element.hash.slice(1)))) assert(ids.includes(target),target);
 assert.equal(await page.locator('#fusion a').filter({hasText:"features['concat'] · Cell 111"}).count(),1);
 ok('Notebook tersimpan persis sumber;89cell dan seluruh cuplikan/tautan sesuai baris kode asli.');

 await page.getByRole('tab',{name:'S1 · RoBERTa',exact:true}).click();
 assert(await page.locator('#panel-s1').isVisible()); assert(!(await page.locator('#panel-s3').isVisible()));
 await page.getByRole('tab',{name:'S2 · BGE',exact:true}).click(); assert(await page.locator('#panel-s2').isVisible());
 await page.getByRole('tab',{name:'S2 · BGE',exact:true}).press('ArrowRight'); assert(await page.locator('#panel-s3').isVisible());
 await page.getByRole('tab',{name:'Saat prediksi baru',exact:true}).click();
 assert(await page.locator('#mode-predict').isVisible()); assert(!(await page.locator('#mode-train').isVisible()));
 assert.match(await page.locator('#mode-predict').innerText(),/tidak perlu rater/);
 await page.getByRole('tab',{name:'Saat prediksi baru',exact:true}).press('Home'); assert(await page.locator('#mode-train').isVisible());
 ok('Tiga skenario dan dua peran label bekerja melalui klik serta navigasi keyboard.');

 const search = page.getByLabel('Cari fungsi, variabel, atau cell',{exact:true});
 await search.fill('concat');
 assert(await page.locator('#code-catalog').getAttribute('open') !== null);
 assert(await page.locator('.code-card:visible').count() > 0);
 assert(await page.locator('.code-card:visible').count() < 89);
 assert(await page.locator('#cell-111').isVisible());
 await search.fill('zzzz_no_function_zzzz'); assert(await page.locator('#empty-search').isVisible());
 await page.locator('#fusion a[href="#cell-111"]').first().click();
 assert.equal(await search.inputValue(),''); assert(await page.locator('#cell-111').getAttribute('open') !== null);
 assert(await page.locator('#code-catalog').getAttribute('open') !== null);
 await search.fill('rater'); await search.press('Escape'); assert.equal(await search.inputValue(),'');
 assert.equal(await page.locator('.code-card:not([hidden])').count(),89);
 ok('Pencarian,hasil kosong,reset,dan tautan langsung membuka cell yang tepat.');

 await page.evaluate(() => {
   Object.defineProperty(navigator,'clipboard',{configurable:true,value:{writeText:async text=>{window.copiedText=text;}}});
   document.execCommand = () => {window.copiedText=document.querySelector('textarea').value;return true;};
 });
 const copy = page.locator('#input .snippet').first().locator('[data-copy-cell]');
 await copy.click();
 assert.equal(await page.evaluate(()=>window.copiedText),lines(notebook.cells[8].source).slice(1,6).join('\n')+'\n');
 const notebookEvent = page.waitForEvent('download'); await page.locator('#download-notebook').click();
 const notebookDownload = await notebookEvent; const notebookFile=path.join(out,'downloaded_e30.ipynb');
 await notebookDownload.saveAs(notebookFile); assert.equal(hash(fs.readFileSync(notebookFile)),hash(raw));
 const htmlEvent=page.waitForEvent('download'); await page.locator('#download-html').click();
 const htmlDownload=await htmlEvent;const offlineFile=path.join(out,'downloaded_offline.html');await htmlDownload.saveAs(offlineFile);
 const offline=await context.newPage();offline.on('pageerror',error=>errors.push(error.message));
 await offline.goto(origin+'/offline.html');await offline.waitForFunction(()=>E30Guide.codeCellCount===89);
 assert(!(await offline.locator('#toast').isVisible()));
 await offline.getByRole('tab',{name:'S2 · BGE',exact:true}).click();assert(await offline.locator('#panel-s2').isVisible());
 await offline.close();
 ok('Salin kode benar;unduhan notebook byte-identik dan HTML unduhan berjalan offline.');

 await page.locator('#code-catalog').evaluate(element=>{element.open=false;});
 for (const width of [320,390,768,1440]) {
   await page.setViewportSize({width,height:width<500?844:1000});
   for (const section of ['input','model','fusion','label','pengaturan','training','peta-kode']) {
     await page.locator('#'+section).scrollIntoViewIfNeeded();
     assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+2),false,`${width}:${section}`);
   }
 }
 await page.setViewportSize({width:1440,height:1000});await page.evaluate(()=>scrollTo(0,0));
 await page.screenshot({path:path.join(out,'desktop.png')});
 await page.setViewportSize({width:390,height:844});await page.evaluate(()=>scrollTo(0,0));
 await page.screenshot({path:path.join(out,'mobile.png')});
 await page.locator('#fusion .dimensions').scrollIntoViewIfNeeded();await page.waitForTimeout(150);await page.screenshot({path:path.join(out,'mobile_fusion.png')});
 ok('Tidak ada overflow halaman pada320/390/768/1440px;preview desktop dan HP disimpan.');
 assert.deepEqual(requests,[]);assert.deepEqual(errors,[]);
 assert.equal(await page.locator('script[src],link[rel=stylesheet],iframe').count(),0);
 ok('Seluruh fitur bekerja dari HTML lokal,tanpa permintaan eksternal,CDN,iframe,atau error JavaScript.');
 const proof={scope:'PUBLIC SOURCE GUIDE ONLY; NO RESEARCH DATA OR TRAINING',passed:checks.length,failed:0,checks,
   source_sha256:hash(raw),html_sha256:hash(originalHtml),browser:browser.version(),viewports:[320,390,768,1440],external_requests:requests.length,errors,transport:'localhost documents only; all external requests blocked; file URLs blocked by environment policy'};
 fs.writeFileSync(path.join(out,'browser_verification.json'),JSON.stringify(proof,null,2)+'\n');
 console.log(JSON.stringify({passed:checks.length,failed:0}));
} finally {await browser.close();await new Promise(resolve=>server.close(resolve));}
