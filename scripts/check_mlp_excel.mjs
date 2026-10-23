// UI dan nilai XLSX tersimpan; tidak menjalankan notebook atau training.
import {createRequire} from 'node:module';import fs from 'node:fs';import assert from 'node:assert/strict';import crypto from 'node:crypto';
const require=createRequire(import.meta.url),{chromium}=require('playwright');
const base='http://127.0.0.1:8765/',directory='outputs/mlp_study/excel_review';
const browser=await chromium.launch({executablePath:'/usr/bin/chromium',args:['--no-sandbox']});
const page=await browser.newPage({viewport:{width:1440,height:1000},reducedMotion:'reduce',acceptDownloads:true});
const errors=[],checks=[];page.on('pageerror',e=>errors.push(e.message));
const ok=s=>{checks.push(s);console.log('OK:',s)};
async function ready(){await page.waitForFunction(()=>document.querySelector('#sheetStatus').textContent.match(/^(XLSX asli|Output agregat aktual)/))}
async function select(book,sheet){await page.evaluate(([book,sheet])=>MLPStudy.selectExcel(book,sheet),[book,sheet]);await ready()}
try{
  await page.goto(base+'docs/belajar_MLP_Spider.html');await select(3,'Perbandingan Model');
  assert.equal(await page.locator('#downloadCsv').count(),0);assert.ok(await page.locator('#componentControl').isHidden());
  assert.ok(await page.locator('#downloadBook').isHidden());assert.ok(await page.locator('#rowPagination').isHidden());
  for(const [book,sheet] of [[0,'Panduan'],[1,'Panduan'],[2,'Panduan'],[3,'Ringkasan Validasi'],[3,'Arti Metrik'],[4,'Pengaturan Eksperimen'],[4,'Model yang Dibandingkan']]){
    await select(book,sheet);assert.ok(await page.locator('#sheetTable tbody tr').count()>0);assert.equal(await page.locator('.placeholder-row').count(),0);
  }
  assert.equal(await page.locator('#excelRoutes button').count(),5);
  ok('Public aggregate/explanatory tables contain actual rows; missing XLSX downloads, CSV and irrelevant component controls are absent');
  await page.goto(base+'outputs/mlp_study/belajar_MLP_Spider_Privat.html');
  await page.waitForFunction(()=>window.MLPStudy?.getState().resultsLoaded&&document.querySelector('#status').textContent.startsWith('Siap:'),{timeout:60000});
  assert.equal((await page.evaluate(()=>MLPStudy.getState())).nativeOutputs,34);
  const all=await page.evaluate(()=>MLPStudy.data.snapshot.books.map((b,book)=>b.sheets.map(s=>[book,s.name])).flat());
  assert.equal(all.length,41);
  for(const [book,sheet] of all){await select(book,sheet);assert.equal(await page.locator('#sheetSelect').inputValue(),sheet);assert.equal(await page.locator('.placeholder-row').count(),0)}
  ok('Private HTML loads the notebook and both ZIP runs automatically; all 41 MLP sheets are selectable and filled');
  await select(3,'Nilai Guru dan Prediksi');
  assert.equal(await page.locator('#sheetTable thead th').count(),11);assert.ok(await page.locator('#componentControl').isHidden());
  const query=await page.evaluate(async()=>{const s=await MLPStudy.readSheet(3,'Nilai Guru dan Prediksi');const value=String(s.rows[0][10]);return {value,count:s.rows.filter(r=>r.some(v=>String(v??'').toLocaleLowerCase('id').includes(value.toLocaleLowerCase('id')))).length}});
  await page.locator('#rowSearch').fill(query.value);assert.equal(await page.locator('#sheetTable tbody tr').count(),Math.min(25,query.count));
  await page.locator('#rowSearch').fill('NO_MATCH_TECHNICAL_SENTINEL');assert.match(await page.locator('#sheetTable tbody').innerText(),/Tidak ada baris/);await page.locator('#rowSearch').fill('');
  await page.locator('#rowsPerPage').selectOption('100');assert.equal(await page.locator('#sheetTable tbody tr').count(),60);assert.ok(await page.locator('#rowPagination').isHidden());
  await page.locator('#sheetTable tbody td').last().focus();await page.keyboard.press('Enter');assert.match(await page.locator('#cellInspector').innerText(),/Baris/);
  ok('All 11 prediction columns show together; search reaches later columns, no-match is clear, all 60 rows fit at 100/page, and Enter inspects a cell');
  await select(1,'Fusion S3');assert.ok(await page.locator('#componentControl').isVisible());
  await page.locator('#componentStart').fill('1024');await page.locator('#componentStart').dispatchEvent('change');assert.match(await page.locator('#sheetTable thead').innerText(),/Komponen 1024/);
  await page.locator('#nextRows').click();assert.match(await page.locator('#rowRange').innerText(),/^101/);
  await page.locator('#precisionToggle').click();assert.equal(await page.locator('#precisionToggle').getAttribute('aria-pressed'),'true');
  ok('Vector-only controls reach component 1024, row paging respects selected size, and full precision remains available');
  await select(0,'Token RoBERTa');await page.locator('#excelCode').click();assert.equal((await page.evaluate(()=>MLPStudy.getState())).cell,22);
  await select(0,'Token RoBERTa');await page.locator('#excelOutput').click();assert.equal((await page.evaluate(()=>MLPStudy.getState())).chapter,6);assert.equal((await page.evaluate(()=>MLPStudy.getState())).tab,'output');
  await page.waitForFunction(()=>document.querySelector('.output-evidence-preview table'));
  assert.ok(await page.locator('.native-card').count()>0);assert.ok(await page.locator('.output-evidence-actions button').count()>0);
  ok('Excel → producing code/output links land on the correct stage; actual notebook output and related Excel preview are both present');
  await select(3,'Perbandingan Model');
  const event=page.waitForEvent('download');await page.locator('#downloadBook').click();const download=await event;const file=directory+'/original.xlsx';await download.saveAs(file);
  assert.equal(crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex'),await page.evaluate(()=>MLPStudy.data.snapshot.books[3].sha256));
  const zipEvent=page.waitForEvent('download');await page.locator('#downloadResults').click();const zip=await zipEvent;await zip.saveAs(directory+'/original.zip');
  assert.equal(crypto.createHash('sha256').update(fs.readFileSync(directory+'/original.zip')).digest('hex'),await page.evaluate(()=>MLPStudy.data.snapshot.zip_sha256));
  ok('Excel and complete results ZIP downloads preserve original SHA-256 bytes');
  await page.evaluate(()=>MLPStudy.openChapter(0));
  if(await page.locator('#precisionToggle').getAttribute('aria-pressed')==='true')await page.locator('#precisionToggle').click();
  await page.locator('#rowsPerPage').selectOption('25');
  for(const width of [1440,900,390,320]){
    await page.setViewportSize({width,height:1000});await page.evaluate(()=>document.querySelector('#reader').scrollIntoView());
    await select(3,'Perbandingan Model');assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    assert.ok(await page.locator('#tab-excel').evaluate(e=>{const a=e.getBoundingClientRect(),b=e.parentElement.getBoundingClientRect();return a.left>=b.left-1&&a.right<=b.right+1}),'active Excel tab must remain visible');
    if(width===1440||width===390){await page.locator('#reader').scrollIntoViewIfNeeded();await page.screenshot({path:directory+`/excel_${width}.png`})}
    await select(1,'Fusion S3');assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  }
  ok('Desktop and 320/390px mobile retain table scrolling inside the panel without horizontal page overflow');
  assert.deepEqual(errors,[]);fs.writeFileSync(directory+'/browser_checks.json',JSON.stringify({scope:'SAVED OUTPUT READER; NO TRAINING',passed:checks.length,failed:0,checks,browser:browser.version()},null,2));
}finally{await browser.close()}
