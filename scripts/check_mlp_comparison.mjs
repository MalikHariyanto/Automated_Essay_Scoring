// Browser regression untuk menu pembanding; tidak menjalankan training AES.
import {createRequire} from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
const require=createRequire(import.meta.url),{chromium}=require('playwright');
const [url='http://127.0.0.1:8765/docs/belajar_MLP_Spider.html',privateUrl='http://127.0.0.1:8765/outputs/mlp_study/belajar_MLP_Spider_Privat.html']=process.argv.slice(2);
const dir=path.resolve('outputs/mlp_study');fs.mkdirSync(dir,{recursive:true});
const browser=await chromium.launch({executablePath:'/usr/bin/chromium',headless:true,args:['--no-sandbox']});
const context=await browser.newContext({viewport:{width:1440,height:1000},reducedMotion:'reduce',acceptDownloads:true});
const page=await context.newPage(),errors=[],requests=[],checks=[];
page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
function pass(text){checks.push(text);console.log('OK:',text)}
try{
 await page.goto(url);await page.waitForFunction(()=>window.MLPCompare);
 await page.locator('#heroComparison').click();assert.ok(await page.locator('#comparison').isVisible());assert.ok(await page.locator('#reader').isHidden());
 const overview=await page.locator('#compare-overview').innerText();assert.ok(overview.includes('11,7224'));assert.ok(overview.includes('6,7085'));assert.ok(overview.includes('10,9542'));assert.ok(overview.includes('6,0516'));
 pass('Discoverable hero/menu entry and actual S3 baseline/MLP test and CV summary');
 for(const width of [1440,900,390,320]){await page.setViewportSize({width,height:1000});for(const experiment of ['E0','E1'])for(const scenario of ['S1','S2','S3'])for(const tab of ['overview','layers','training','results','why']){
  await page.evaluate(([e,s,t])=>{MLPCompare.setExperiment(e);MLPCompare.setScenario(s);MLPCompare.setTab(t)},[experiment,scenario,tab]);
  assert.ok(await page.locator('#compare-'+tab).isVisible());const text=await page.locator('#compare-'+tab).innerText();assert.ok(text.length>100);assert.ok(!/\b(?:NaN|undefined|Infinity)\b/.test(text));assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`overflow ${width} ${experiment} ${scenario} ${tab}`);
 }}
 pass('120 experiment/scenario/tab/viewport combinations without page overflow or missing values');
 await page.setViewportSize({width:1440,height:1000});
 for(const experiment of ['E0','E1'])for(const scenario of ['S1','S2','S3']){
  await page.evaluate(([e,s])=>{MLPCompare.setExperiment(e);MLPCompare.setScenario(s);MLPCompare.setTab('training')},[experiment,scenario]);
  const selects=page.locator('#compare-training select');
  for(let fold=1;fold<=5;fold++)for(const field of ['RMSE Validasi (poin)','Loss Training (MSE)']){
   await selects.nth(0).selectOption(String(fold));await selects.nth(1).selectOption(field);
   const expected=await page.evaluate(([e,s,f,k])=>{const d=MLPStudy.data,older=d.comparison[e==='E0'?'baseline':'intermediate'].history?.[s]||[],latest=d.snapshot.history[s];return [...older,...latest].filter(r=>r.Fold===f&&Number.isFinite(r[k])).length},[experiment,scenario,fold,field]);
   assert.equal(await page.locator('.comparison-chart circle').count(),expected);
  }
 }
 pass('60 actual curve configurations retain every original epoch, including unavailable intermediate history');
 await page.evaluate(()=>{MLPCompare.setExperiment('E0');MLPCompare.setScenario('S3');MLPCompare.setTab('layers')});
 await page.locator('#comparison').scrollIntoViewIfNeeded();const y=await page.evaluate(()=>scrollY);await page.locator('.layer-stack.new .layer-block').last().click();assert.equal(await page.evaluate(()=>scrollY),y);assert.ok((await page.locator('#compare-layers').innerText()).includes('262.657 parameter'));
 const original=await page.evaluate(()=>MLPStudy.data.comparison.baseline.evidence.model.source);assert.equal(await page.locator('.source-frame').first().locator('.compare-code-line code').allTextContents().then(a=>a.map(s=>s===' '?'':s).join('\n')),original);
 const changedText=await page.locator('.source-frame').last().locator('.changed code').allTextContents();assert.ok(changedText.some(t=>t.includes('self.head = nn.Sequential')));
 pass('Clickable layer notes, head parameter count and exact original source diff');
 await page.locator('#compare-tab-layers').focus();await page.keyboard.press('ArrowRight');assert.equal(await page.locator('#compare-tab-training').getAttribute('aria-selected'),'true');await page.locator('#openStudy').click();assert.ok(await page.locator('#reader').isVisible());assert.ok(await page.locator('#comparison').isHidden());await page.locator('#openComparison').click();
 pass('Keyboard comparison tabs and switching back to the unchanged MLP book');
 await page.evaluate(()=>{MLPCompare.setExperiment('E0');MLPCompare.setScenario('S3');MLPCompare.setTab('overview')});await page.locator('#comparison').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(dir,'comparison_overview_verified.png')});
 await page.evaluate(()=>MLPCompare.setTab('layers'));await page.screenshot({path:path.join(dir,'comparison_layers_verified.png')});
 await page.evaluate(()=>MLPCompare.setTab('training'));await page.screenshot({path:path.join(dir,'comparison_training_verified.png')});
 await page.setViewportSize({width:390,height:844});await page.evaluate(()=>MLPCompare.setTab('overview'));await page.locator('.comparison-book').scrollIntoViewIfNeeded();await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));await page.screenshot({path:path.join(dir,'comparison_mobile_verified.png')});
 await page.setViewportSize({width:1440,height:1000});await page.goto(privateUrl);await page.waitForFunction(()=>window.MLPCompare?.getState().baselineLoaded);assert.ok((await page.evaluate(()=>MLPStudy.getState())).resultsLoaded);
 await page.evaluate(()=>{MLPCompare.setMode('comparison');MLPCompare.setTab('results')});await page.waitForFunction(()=>document.querySelectorAll('.prediction-pair table').length===2);assert.equal(await page.locator('.prediction-pair article').first().locator('tbody tr').count(),3);assert.equal(await page.locator('.prediction-pair article').last().locator('tbody tr').count(),3);assert.ok((await page.locator('#compare-results').innerText()).includes('belum terverifikasi sebagai jawaban yang sama'));
 pass('Private HTML loads both original runs; prediction examples stay separate with no fabricated teacher label');
 const event=page.waitForEvent('download',{timeout:60000});await page.locator('#savePrivateHtml').click();const download=await event;await download.saveAs(path.join(dir,'comparison_export_private.html'));
 await page.goto(new URL('../outputs/mlp_study/comparison_export_private.html',url).href);await page.waitForFunction(()=>window.MLPCompare?.getState().baselineLoaded);assert.ok((await page.evaluate(()=>MLPStudy.getState())).resultsLoaded);
 pass('Export/reopen retains both private notebooks, original results ZIP and comparison menu');
 const refused=await page.evaluate(async()=>{try{await MLPCompare.loadBaselineNotebook(new Uint8Array([123,125]));return false}catch{return true}});assert.equal(refused,true);
 assert.deepEqual(errors,[]);assert.ok(requests.every(r=>r.startsWith(new URL(url).origin)));
 pass('Wrong baseline rejected; zero page errors and external network requests');
 fs.writeFileSync(path.join(dir,'comparison_browser_verification.json'),JSON.stringify({scope:'SAVED RUN COMPARISON UI; NO NEW TRAINING',passed:checks.length,failed:0,browser:browser.version(),checks},null,2));
}finally{await browser.close()}
