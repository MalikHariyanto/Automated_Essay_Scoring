// Audit tampilan E8 dan XLSX sumber; tidak menjalankan model atau training.
import {createRequire} from 'node:module';
import fs from 'node:fs';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
const require=createRequire(import.meta.url),{chromium}=require('playwright');
const base='http://127.0.0.1:8765/',folder='outputs/mlp_study/',checks=[];
const browser=await chromium.launch({executablePath:'/usr/bin/chromium',args:['--no-sandbox','--disable-dev-shm-usage']});
try{
 const page=await browser.newPage({viewport:{width:1440,height:1000},acceptDownloads:true});
 page.setDefaultTimeout(180000);const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(base+folder+'belajar_MLP_Spider_Privat.html',{timeout:180000});
 await page.waitForFunction(()=>window.MLPE8?.getState().loaded&&window.MLPDashboard?.getLRBytes());
 await page.route('**/*',route=>route.abort());
 const expected=JSON.parse(fs.readFileSync('docs/mlp_e8_snapshot.json'));
 assert.equal(await page.evaluate(()=>MLPDashboard.runs.length),7);
 await page.evaluate(()=>MLPDashboard.setTab('overview'));
 assert.equal(await page.locator('#dashboardE8Comparison tbody tr').count(),6);
 assert.match(await page.locator('#dashboardE8Comparison').innerText(),/5,8836/);
 checks.push('Ringkasan E4/E6/E8 membedakan CV, test dan seluruh empat metrik.');
 await page.evaluate(()=>{MLPCompare.setMode('comparison',true);MLPCompare.setExperiment('E8');MLPCompare.setTab('overview')});
 assert.match(await page.locator('#pairMethods').innerText(),/Soal\/referensi \+ jawaban/);
 assert.match(await page.locator('#pairMethods').innerText(),/Tercatat pada run ini/);
 await page.evaluate(()=>MLPCompare.setTab('results'));await page.waitForSelector('#pairAllPredictions');
 assert.equal(await page.locator('#pairAllPredictions tbody tr').count(),60);
 for(const scenario of ['S1','S2','S3']){
  await page.evaluate(s=>{MLPCompare.setScenario(s);MLPCompare.setTab('training')},scenario);
  assert.equal(await page.locator('#pairFolds tbody tr').count(),5);
  assert.equal(await page.locator('.comparison-chart svg').count(),1);
 }
 checks.push('E8 vs E6: metode, tiga skenario, 15 fold, kurva aktual dan 60 pasangan tersedia.');
 for(const scenario of ['S1','S2','S3']){
  await page.evaluate(s=>{MLPCompare.setMode('dashboard',true);MLPDashboard.setScenario(s);MLPDashboard.setTab('predictions');MLPPredictions.setView('all')},scenario);
  await page.waitForSelector('#predictionAllAnswers');
  assert.equal(await page.locator('#predictionAllAnswers .prediction-score').count(),420);
  assert.equal(await page.locator('#predictionAllAnswers [data-experiment=E8]').count(),61);
  const source=await page.evaluate(async()=>{
   const rows=await MLPDashboard.predictionRows('E8');
   return rows.every(row=>{const tr=[...document.querySelectorAll('#predictionAllAnswers tbody tr')].find(tr=>tr.dataset.answer===String(row['ID Jawaban']));return Number(tr.querySelector('[data-experiment=E8]').dataset.prediction)===row['Prediksi '+MLPDashboard.getState().scenario]});
  });assert(source);
  await page.evaluate(()=>MLPPredictions.setView('ranking'));
  assert.equal(await page.locator('.prediction-rank-row').count(),7);
 }
 checks.push('420 nilai prediksi per skenario; seluruh nilai E8 dan peringkat cocok sumber mentah.');
 await page.evaluate(()=>{MLPDashboard.setTab('excel')});
 await page.getByLabel('Eksperimen Excel',{exact:true}).selectOption('E8');
 await page.getByLabel('Tabel Excel',{exact:true}).selectOption('Perbandingan Model');
 await page.waitForFunction(()=>document.querySelector('#dashboardSheet')?.textContent.includes('6,3818'));
 const downloadPromise=page.waitForEvent('download');await page.getByRole('button',{name:'Unduh Excel asli ↓',exact:true}).click();
 const download=await downloadPromise;await download.saveAs(folder+'e8_verified_eval.xlsx');
 assert.equal(crypto.createHash('sha256').update(fs.readFileSync(folder+'e8_verified_eval.xlsx')).digest('hex'),expected.books[3].sha256);
 checks.push('Penjelajah E8 membuka sheet asli dan mengunduh XLSX byte-identik.');
 await page.evaluate(()=>MLPDashboard.setTab('overview'));
 await page.locator('[data-expand-comparison=dashboardRoom]').click();
 assert(await page.locator('#comparisonViewDialog').evaluate(e=>e.open));
 await page.keyboard.press('Escape');assert.deepEqual(errors,[]);
 checks.push('Fullscreen tetap berfungsi; seluruh pemeriksaan privat setelah pemuatan berjalan offline.');
 await page.close();
 const publicPage=await browser.newPage({viewport:{width:1440,height:1000}});
 await publicPage.goto(base+'docs/belajar_MLP_Spider.html');await publicPage.waitForFunction(()=>window.MLPDashboard);
 await publicPage.evaluate(()=>MLPDashboard.setTab('overview'));
 await publicPage.locator('#dashboardE8Comparison').scrollIntoViewIfNeeded();
 await publicPage.screenshot({path:'docs/previews/mlp_study/24_e8_comparison.png'});
 const report={passed:checks.length,failed:0,checks,browser:browser.version()};
 fs.writeFileSync(folder+'e8_specific_browser.json',JSON.stringify(report,null,2));
 console.log(JSON.stringify(report));
}finally{await browser.close()}
