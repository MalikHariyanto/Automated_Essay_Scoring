// Pemeriksaan browser buku statis. Tidak menjalankan notebook atau model AES.
// Argumen: URL publik, notebook privat, ZIP privat, JSON pembanding privat.
import {createRequire} from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
const require=createRequire(import.meta.url), {chromium}=require('playwright');
const [url='http://127.0.0.1:8765/docs/belajar_MLP_Spider.html',notebook,results,expectedFile]=process.argv.slice(2);
const directory=path.resolve('outputs/mlp_study');fs.mkdirSync(directory,{recursive:true});
const browser=await chromium.launch({executablePath:'/usr/bin/chromium',headless:true,args:['--no-sandbox']});
const context=await browser.newContext({viewport:{width:1440,height:1000},reducedMotion:'reduce',acceptDownloads:true});
const page=await context.newPage(),errors=[],requests=[];
page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
const report={scope:'STATIC HTML AND LOCAL FILE READER; NO TRAINING',checks:[],browser:browser.version()};
function passed(message){report.checks.push(message);console.log('OK:',message)}
try{
  await page.goto(url);await page.waitForFunction(()=>window.MLPStudy);
  const coverage=await page.evaluate(()=>MLPStudy.data.coverage);
  assert.equal(coverage.code_cells,30);assert.equal(coverage.manual_pins_found,coverage.manual_pins_total);
  assert.equal(await page.locator('#chapterNav button').count(),16);
  for(const width of [1440,390]){
    await page.setViewportSize({width,height:1000});
    for(let i=0;i<16;i++)for(const tab of ['guide','code','output','excel']){
      await page.evaluate(([i,tab])=>{MLPStudy.openChapter(i);MLPStudy.setTab(tab)},[i,tab]);
      assert.ok(await page.locator('#pane-'+tab).isVisible());
      assert.ok((await page.locator('#pane-'+tab).innerText()).length>20);
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`overflow ${width}/${i}/${tab}`);
    }
  }
  passed('128 chapter/tab/viewport combinations, no horizontal page overflow');
  await page.setViewportSize({width:1440,height:1000});
  await page.evaluate(()=>{MLPStudy.openChapter(0);MLPStudy.setTab('guide')});
  await page.screenshot({path:path.join(directory,'verified_desktop.png'),fullPage:true});
  await page.evaluate(()=>{MLPStudy.openChapter(10);MLPStudy.setTab('code')});
  const all=await page.evaluate(()=>MLPStudy.data.cells.map(c=>[c.index,c.htmlLines.length]));
  for(const [index,count]of all){await page.evaluate(i=>MLPStudy.jumpCode(i,1),index);assert.equal(await page.locator('.code-line').count(),count)}
  passed('All 30 full code blocks render with original line counts');
  await page.evaluate(()=>MLPStudy.jumpCode(30,58));
  await page.locator('#reader').scrollIntoViewIfNeeded();
  const bodyY=await page.evaluate(()=>scrollY);
  const note=page.locator('.note-links button').last();await note.click();
  assert.equal(await page.evaluate(()=>scrollY),bodyY);assert.ok((await page.locator('#notePane').innerText()).length>80);
  await page.screenshot({path:path.join(directory,'verified_code.png')});
  passed('Sticky note changes and code-only scroll preserve page position');
  await page.locator('#tab-code').focus();await page.keyboard.press('ArrowRight');assert.equal(await page.locator('#tab-output').getAttribute('aria-selected'),'true');
  await page.evaluate(()=>{MLPStudy.openChapter(10);MLPStudy.setTab('guide')});
  await page.locator('#chapterTerms .term-button').first().click();assert.ok(await page.locator('#termDialog').isVisible());await page.keyboard.press('Escape');
  passed('Keyboard tabs and glossary dialog');
  await page.setViewportSize({width:390,height:844});await page.evaluate(()=>{MLPStudy.openChapter(10);MLPStudy.setTab('guide')});await page.locator('#reader').scrollIntoViewIfNeeded();await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));await page.screenshot({path:path.join(directory,'verified_mobile.png')});
  if(notebook&&results){
    await page.setViewportSize({width:1440,height:1000});
    await page.locator('#notebookFile').setInputFiles(notebook);await page.waitForFunction(()=>MLPStudy.getState().notebookLoaded);
    await page.locator('#resultsFile').setInputFiles(results);await page.waitForFunction(()=>MLPStudy.getState().resultsLoaded);
    assert.equal((await page.evaluate(()=>MLPStudy.getState())).nativeOutputs,34);
    passed('Both original private files load locally; all 34 notebook outputs available');
    let renderedOutputs=0;
    for(let i=1;i<16;i++){await page.evaluate(i=>{MLPStudy.openChapter(i);MLPStudy.setTab('output')},i);renderedOutputs+=await page.locator('.native-card').count()}
    assert.equal(renderedOutputs,34);
    passed('All 34 original output objects render in their corresponding chapters');
    if(expectedFile){
      const expected=JSON.parse(fs.readFileSync(expectedFile,'utf8'));
      for(const item of expected){
        const actual=await page.evaluate(async({book,name,samples})=>{const s=await MLPStudy.readSheet(book,name);return {headers:s.headers,rows:s.rows.length,samples:samples.map(v=>s.rows[v.row]?.[v.col]??null)}},item);
        assert.deepEqual(actual.headers,item.headers,item.name+' headers');assert.equal(actual.rows,item.rows,item.name+' rows');
        item.samples.forEach((s,i)=>typeof s.value==='number'?assert.ok(Math.abs(s.value-actual.samples[i])<1e-12*Math.max(1,Math.abs(s.value)),item.name+' numeric cell'):assert.deepEqual(actual.samples[i],s.value,item.name+' text cell'));
      }
      passed('All '+expected.length+' XLSX sheets match openpyxl headers, row counts and representative original cell values');
    }
    await page.evaluate(()=>MLPStudy.selectExcel(1,'Fusion S3'));await page.waitForFunction(()=>document.querySelector('#sheetStatus').textContent.startsWith('XLSX asli'));
    await page.locator('#componentStart').fill('1024');await page.locator('#componentStart').dispatchEvent('change');
    assert.ok((await page.locator('#sheetTable thead').innerText()).includes('Komponen 1024'));
    await page.locator('#nextRows').click();assert.match(await page.locator('#rowRange').innerText(),/^26/);
    await page.locator('#precisionToggle').click();await page.locator('#sheetTable tbody td').last().click();assert.ok((await page.locator('#cellInspector').innerText()).includes('Komponen 1024'));
    passed('Final fusion component 1024, pagination and full-precision cell inspection');
    const xlsxEvent=page.waitForEvent('download');await page.locator('#downloadBook').click();const xlsx=await xlsxEvent;const saved=path.join(directory,'browser_download.xlsx');await xlsx.saveAs(saved);
    const expectedHash=await page.evaluate(()=>MLPStudy.data.snapshot.books[1].sha256);assert.equal(crypto.createHash('sha256').update(fs.readFileSync(saved)).digest('hex'),expectedHash);
    passed('Original XLSX download is byte-identical');
    const htmlEvent=page.waitForEvent('download',{timeout:60000});await page.locator('#savePrivateHtml').click();const htmlDownload=await htmlEvent;const exported=path.join(directory,'browser_export_private.html');await htmlDownload.saveAs(exported);
    const p2=await context.newPage();p2.on('pageerror',e=>errors.push(e.message));await p2.goto(new URL('../outputs/mlp_study/browser_export_private.html',url).href);await p2.waitForFunction(()=>window.MLPStudy?.getState().resultsLoaded,{timeout:60000});assert.equal((await p2.evaluate(()=>MLPStudy.getState())).nativeOutputs,34);await p2.close();
    passed('Browser-exported self-contained private HTML reopens with every native output and all five workbooks');
    const malicious=JSON.parse(fs.readFileSync(notebook,'utf8'));malicious.cells[39].outputs.push({output_type:'display_data',data:{'text/html':'<script>window.__injected=1</script><iframe src="https://invalid.example/"></iframe><img src="https://invalid.example/x" onerror="window.__injected=2"><p onclick="window.__injected=3">INERT_SENTINEL</p>'},metadata:{}});
    malicious.cells[39].outputs.push({output_type:'display_data',data:{'application/javascript':'window.__injected=4'},metadata:{}});
    await page.evaluate(async n=>{await MLPStudy.loadNotebook(new TextEncoder().encode(JSON.stringify(n)));MLPStudy.openChapter(12);MLPStudy.setTab('output')},malicious);
    assert.equal(await page.evaluate(()=>window.__injected),undefined);assert.equal(await page.locator('#outputContent iframe,#outputContent script,#outputContent [onerror],#outputContent [onclick]').count(),0);assert.ok((await page.locator('#outputContent').innerText()).includes('INERT_SENTINEL'));
    const refused=await page.evaluate(async()=>{try{await MLPStudy.loadResults(new Uint8Array([1,2,3]));return false}catch{return true}});assert.equal(refused,true);
    passed('Untrusted notebook output cannot execute script/events/iframe; wrong results ZIP rejected');
  }
  assert.deepEqual(errors,[]);assert.ok(requests.every(r=>r.startsWith(new URL(url).origin)),requests.join('\n'));
  passed('No page errors or external network requests');
  report.passed=report.checks.length;report.failed=0;report.coverage=coverage;fs.writeFileSync(path.join(directory,'browser_verification.json'),JSON.stringify(report,null,2));
}finally{await browser.close()}
