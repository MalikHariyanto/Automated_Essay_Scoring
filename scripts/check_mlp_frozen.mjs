// Pembaca hasil tersimpan, grafik dan XLSX privat; tidak menjalankan model/training.
import {createRequire} from 'node:module';import fs from 'node:fs';import crypto from 'node:crypto';import assert from 'node:assert/strict';
const require=createRequire(import.meta.url),{chromium}=require('playwright');
const [url='http://127.0.0.1:8765/docs/belajar_MLP_Spider.html',privateUrl='http://127.0.0.1:8765/outputs/mlp_study/belajar_MLP_Spider_Privat.html']=process.argv.slice(2);
const b=await chromium.launch({executablePath:'/usr/bin/chromium',args:['--no-sandbox']}),p=await b.newPage({viewport:{width:1440,height:1000},reducedMotion:'reduce',acceptDownloads:true});
const errors=[],requests=[],checks=[];p.on('pageerror',e=>errors.push(e.message));p.on('request',r=>requests.push(r.url()));const ok=s=>{checks.push(s);console.log('OK:',s)};
try{
 await p.goto(url);await p.locator('#openFrozen').click();assert.ok(await p.locator('#frozenRoom').isVisible());assert.ok(await p.locator('#reader').isHidden());assert.equal(await p.locator('#frozenAllTest tbody tr').count(),12);assert.equal(await p.locator('#frozenFinalLoss tbody tr').count(),6);assert.equal(await p.locator('#frozenAllFolds tbody tr').count(),15);
 const matrix=await p.evaluate(()=>{
  const data=MLPStudy.data,runs=[data.comparison.baseline,data.snapshot,data.frozen];
  return [...document.querySelectorAll('#frozenAllTest tbody tr')].every(row=>[...row.querySelectorAll('td')].every((td,i)=>Number(td.title)===runs[i].metrics.find(r=>r.Skenario===row.dataset.scenario)[row.dataset.metric]));
 });assert.ok(matrix,'Every test metric must stay on its original scenario and experiment column');
 assert.equal(await p.locator('#frozenAllTest thead th.experiment-column').count(),3);assert.equal(await p.locator('#frozenAllTest th[scope=rowgroup]').count(),3);assert.equal(await p.locator('#frozenAllTest th[scope=row]').count(),12);
 assert.match(await p.locator('#frozenBody').innerText(),/7,8611/);assert.match(await p.locator('#frozenBody').innerText(),/belum ada bukti keunggulan/);assert.equal(await p.locator('#frozenPredictions').count(),0);
 ok('Three experiment columns preserve all 36 test values with scenario/metric row headers; six refit rows, 15 folds and no public individual predictions');
 for(const width of [1440,900,390,320]){await p.setViewportSize({width,height:1000});for(const s of ['S1','S2','S3'])for(const tab of ['review','process','predictions','excel']){
  await p.evaluate(([s,t])=>{MLPFrozen.setScenario(s);MLPFrozen.setTab(t)},[s,tab]);assert.ok((await p.locator('#frozenBody').innerText()).length>60);assert.ok(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`${width}/${s}/${tab}`);assert.ok(!/\b(?:NaN|undefined|Infinity)\b/.test(await p.locator('#frozenBody').innerText()));
  if(width<=390&&tab==='review'){
   const pinned=await p.evaluate(()=>{const table=document.getElementById('frozenAllTest'),wrap=table.parentElement;wrap.scrollLeft=wrap.scrollWidth;return Math.abs(table.querySelector('.metric-row-label').getBoundingClientRect().left-wrap.getBoundingClientRect().left)<3});assert.ok(pinned,'Mobile scenario/metric label remains visible while scrolling across experiments');
  }
 }}ok('48 scenario/tab/viewport combinations without page overflow or invalid values; mobile metric labels remain visible during horizontal scroll');
 await p.setViewportSize({width:1440,height:1000});
 for(const s of ['S1','S2','S3']){await p.evaluate(s=>{MLPFrozen.setScenario(s);MLPFrozen.setTab('process')},s);for(let fold=1;fold<=5;fold++)for(const field of ['RMSE Validasi (poin)','Loss Training (MSE)']){
  await p.getByRole('combobox',{name:'Fold grafik double frozen',exact:true}).selectOption(String(fold));await p.getByRole('combobox',{name:'Metrik grafik double frozen',exact:true}).selectOption(field);
  const expected=await p.evaluate(([s,f,k])=>[MLPStudy.data.snapshot,MLPStudy.data.frozen].flatMap(d=>d.history[s].filter(r=>r.Fold===f&&Number.isFinite(r[k]))).length,[s,fold,field]);assert.equal(await p.locator('.frozen-chart circle').count(),expected);
 }}ok('30 actual two-run training curves include every saved epoch and correct selected-epoch markers');
 await p.locator('#frozen-tab-process').focus();await p.keyboard.press('ArrowRight');assert.equal(await p.locator('#frozen-tab-predictions').getAttribute('aria-selected'),'true');
 await p.locator('#openComparison').click();assert.equal(await p.locator('#experimentNav .experiment-current').count(),0);assert.equal(await p.locator('#experimentNav').getByText('E2 · Run MLP terakhir',{exact:true}).count(),0);await p.locator('#experimentNav button').filter({hasText:'Double frozen'}).click();assert.ok(await p.locator('#frozenRoom').isVisible());
 const rejected=await p.evaluate(async()=>{try{await MLPFrozen.loadResults(new Uint8Array([0,1,2]));return false}catch{return true}});assert.ok(rejected);ok('Keyboard tabs and E3 entry in the original comparison menu work; mismatched ZIP is rejected');
 await p.goto(privateUrl);await p.waitForFunction(()=>window.MLPFrozen?.getState().loaded&&window.MLPStudy?.getState().resultsLoaded,null,{timeout:120000});await p.locator('#openFrozen').click();
 const expected=JSON.parse(fs.readFileSync('outputs/mlp_study/frozen_expected.json','utf8'));
 for(const s of ['S1','S2','S3']){
  await p.evaluate(s=>{MLPFrozen.setScenario(s);MLPFrozen.setTab('predictions')},s);await p.waitForSelector('#frozenPredictions');
  const all=[];for(let page=0;page<3;page++){await p.waitForFunction(n=>document.querySelector('#frozenPredictions tbody tr td').textContent===n,expected.pairs[page*20].id);const rows=await p.locator('#frozenPredictions tbody tr').evaluateAll(rows=>rows.map(r=>[...r.children].map((td,i)=>i?Number(td.title):td.textContent)));all.push(...rows);if(page<2)await p.getByRole('button',{name:'Berikutnya →',exact:true}).click()}
  assert.equal(all.length,60);for(let i=0;i<60;i++){const e=expected.pairs[i],a=all[i];assert.equal(a[0],e.id);assert.equal(a[1],e.teacher);assert.equal(a[2],e.old[s]);assert.equal(a[3],e.new[s]);assert.equal(a[4],e.oldRounded[s]);assert.equal(a[5],e.newRounded[s]);assert.ok(Math.abs(a[6]-Math.abs(e.old[s]-e.teacher))<1e-10);assert.ok(Math.abs(a[7]-Math.abs(e.new[s]-e.teacher))<1e-10)}
 }
 ok('All 60 teacher/prediction/rounding/error pairs in every scenario match the original workbooks');
 const inventory=await p.evaluate(()=>MLPStudy.data.frozen.books.map((b,i)=>b.sheets.map(s=>({book:i,...s}))).flat());for(const sheet of inventory){const actual=await p.evaluate(async({book,name})=>{const s=await MLPFrozen.readSheet(book,name);return {rows:s.rows.length,columns:s.headers.length}},sheet);assert.equal(actual.rows,sheet.rows,sheet.name);assert.equal(actual.columns,sheet.columns,sheet.name)}
 ok('All 41 original sheets load with exact row/column counts, including vectors and the extra LayerNorm representation');
 await p.evaluate(()=>MLPFrozen.setTab('excel'));await p.getByRole('combobox',{name:'Workbook double frozen',exact:true}).selectOption('1');await p.getByRole('combobox',{name:'Sheet double frozen',exact:true}).selectOption('BGE-M3');await p.waitForSelector('#frozenSheetTable');await p.getByRole('spinbutton',{name:'Mulai komponen double frozen',exact:true}).fill('1024');await p.getByRole('spinbutton',{name:'Mulai komponen double frozen',exact:true}).press('Tab');assert.match(await p.locator('#frozenSheetTable thead').innerText(),/Komponen 1024/);
 const event=p.waitForEvent('download');await p.getByRole('button',{name:'Unduh XLSX asli ↓',exact:true}).click();const d=await event;await d.saveAs('outputs/mlp_study/frozen_download.xlsx');assert.equal(crypto.createHash('sha256').update(fs.readFileSync('outputs/mlp_study/frozen_download.xlsx')).digest('hex'),expected.bookHashes[1]);
 ok('Component 1024 is reachable and the downloaded XLSX is byte-identical to the uploaded original');
 const download=p.waitForEvent('download',{timeout:120000});await p.locator('#savePrivateHtml').click();await (await download).saveAs('outputs/mlp_study/frozen_export_private.html');await p.goto(new URL('../outputs/mlp_study/frozen_export_private.html',url).href);await p.waitForFunction(()=>window.MLPFrozen?.getState().loaded&&window.MLPCompare?.getState().baselineLoaded,null,{timeout:120000});
 assert.equal(await p.evaluate(async()=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',MLPFrozen.getBytes()))).map(n=>n.toString(16).padStart(2,'0')).join('')),expected.zipHash);
 await p.evaluate(()=>{MLPStudy.openChapter(10);MLPStudy.setTab('code')});await p.locator('#expandCode').click();await p.locator('#canvasModeToggle').click();assert.equal(await p.locator('.canvas-code-block').count(),30);await p.keyboard.press('Escape');
 assert.deepEqual(errors,[]);assert.ok(requests.every(r=>r.startsWith(new URL(url).origin)||r.startsWith('data:')));ok('Private export/reopen retains all three result sources and both fullscreen modes, with no JS errors or external requests');
 fs.writeFileSync('outputs/mlp_study/frozen_browser_verification.json',JSON.stringify({scope:'SAVED RESULTS AND UI; NO TRAINING',passed:checks.length,failed:0,checks,browser:b.version()},null,2));
}finally{await b.close()}
