import {createRequire} from 'node:module';
import fs from 'node:fs';import path from 'node:path';import assert from 'node:assert/strict';
const require=createRequire(import.meta.url),{chromium}=require('playwright'),dir=path.resolve('outputs/mlp_study');
const browser=await chromium.launch({executablePath:'/usr/bin/chromium',headless:true,args:['--no-sandbox']});
const context=await browser.newContext({viewport:{width:1440,height:1000},reducedMotion:'reduce',acceptDownloads:true}),page=await context.newPage(),errors=[],requests=[],checks=[];
page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
function ok(s){checks.push(s);console.log('OK:',s)}
try{
 await page.goto('http://127.0.0.1:8765/docs/belajar_MLP_Spider.html');await page.locator('#openMetricGuide').click();assert.ok(await page.locator('#metricsRoom').isVisible());
 assert.equal(await page.locator('#completeTestMetrics tbody tr').count(),12);assert.equal(await page.locator('#completeTestMetrics thead .experiment-column').count(),2);assert.equal(await page.locator('#completeFoldMetrics tbody tr').count(),15);assert.equal(await page.locator('#completeTrainingMSE tbody tr').count(),6);
 assert.ok(await page.evaluate(()=>{
  const data=MLPStudy.data,runs=[data.comparison.baseline,data.snapshot];
  return [...document.querySelectorAll('#completeTestMetrics tbody tr')].every(row=>[...row.querySelectorAll('td')].every((td,i)=>Number(td.title)===runs[i].metrics.find(r=>r.Skenario===row.dataset.scenario)[row.dataset.metric]));
 }),'Every metric remains in the matching original experiment column');
 ok('Two experiment columns preserve all 24 test values, six refit metric rows and all 15 folds without private inputs');
 for(const width of [1440,900,390,320]){await page.setViewportSize({width,height:1000});for(const s of ['S1','S2','S3'])for(const k of ['RMSE (poin)','MSE','Pearson','QWK','Gwet AC2']){await page.evaluate(([s,k])=>{MLPResults.setScenario(s);MLPResults.setMetric(k)},[s,k]);assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),`${width}/${s}/${k}`);assert.ok(!/\b(?:NaN|undefined|Infinity)\b/.test(await page.locator('#metricsRoom').innerText()))}}
 ok('60 responsive metric/scenario combinations; no overflow or invalid values');
 await page.setViewportSize({width:1440,height:1000});await page.evaluate(()=>{MLPResults.setScenario('S3');MLPResults.setMetric('RMSE (poin)')});const criterion=page.locator('.criterion-demo select');assert.ok((await page.locator('.criterion-demo').innerText()).includes('belum memenuhi'));await criterion.selectOption('10');assert.ok(!(await page.locator('.criterion-demo p').innerText()).includes('belum memenuhi'));assert.ok((await page.locator('.criterion-demo').innerText()).includes('hanya ilustrasi'));
 await page.evaluate(()=>MLPResults.setMetric('MSE'));assert.ok((await page.locator('.metric-meaning').innerText()).includes('tidak dibatasi maksimum 1'));await page.evaluate(()=>MLPResults.setMetric('Gwet AC2'));assert.ok((await page.locator('.metric-meaning').innerText()).includes('tidak berarti 98%'));
 ok('Thresholds explicitly illustrative; correct MSE scale and AC2 interpretation');
 await page.evaluate(()=>{MLPResults.setMetric('RMSE (poin)');MLPCompare.setMode('metrics',true)});await page.screenshot({path:path.join(dir,'metrics_public_desktop.png')});await page.setViewportSize({width:390,height:844});await page.locator('#metricsRoom').scrollIntoViewIfNeeded();await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));await page.screenshot({path:path.join(dir,'metrics_public_mobile.png')});
 await page.goto('http://127.0.0.1:8765/outputs/mlp_study/belajar_MLP_Spider_Privat.html');await page.waitForFunction(()=>window.MLPCompare?.getBaselineEvaluation());await page.evaluate(()=>MLPCompare.setMode('metrics',true));await page.waitForFunction(()=>document.querySelectorAll('#pairedPredictions tbody tr').length===60);
 const comparison=await page.evaluate(async()=>{const a=MLPCompare.getBaselineEvaluation(),b=await MLPStudy.readSheet(3,'Nilai Guru dan Prediksi'),map=new Map(b.rows.map(r=>[String(r[0]),r]));return a.rows.every(r=>r[1]===map.get(String(r[0]))[1])});assert.ok(comparison);
 const ids=await page.locator('#pairedPredictions tbody tr td:first-child').allTextContents();await page.evaluate(()=>{MLPCompare.getBaselineEvaluation().rows.reverse();MLPResults.render()});await page.waitForFunction(()=>document.querySelectorAll('#pairedPredictions tbody tr').length===60);const reversed=await page.locator('#pairedPredictions tbody tr td:first-child').allTextContents();assert.deepEqual(reversed,ids.toReversed());
 ok('Private 60-pair table joins on IDs/teacher and remains correct when baseline row order reverses');
 const bad=await page.evaluate(async()=>{try{await MLPCompare.loadBaselineWorkbook(new Uint8Array([1,2,3]));return false}catch{return true}});assert.ok(bad);
 await page.setViewportSize({width:1440,height:1000});const event=page.waitForEvent('download',{timeout:60000});await page.locator('#savePrivateHtml').click();await(await event).saveAs(path.join(dir,'metrics_export_private.html'));await page.goto('http://127.0.0.1:8765/outputs/mlp_study/metrics_export_private.html');await page.waitForFunction(()=>window.MLPCompare?.getBaselineEvaluation());await page.evaluate(()=>MLPCompare.setMode('metrics',true));await page.waitForFunction(()=>document.querySelectorAll('#pairedPredictions tbody tr').length===60);
 ok('Private export/reopen retains original baseline workbook; tampered workbook rejected');
 assert.deepEqual(errors,[]);assert.ok(requests.every(r=>r.startsWith('http://127.0.0.1:8765/')));ok('No page errors or external requests');fs.writeFileSync(path.join(dir,'metrics_browser_verification.json'),JSON.stringify({passed:checks.length,failed:0,checks,browser:browser.version()},null,2));
}finally{await browser.close()}
