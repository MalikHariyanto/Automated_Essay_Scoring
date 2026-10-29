// Semua data pemeriksaan individual tetap lokal di outputs/, bukan artefak publik.
import {createRequire} from 'node:module';
import fs from 'node:fs';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
const {chromium}=createRequire(import.meta.url)('playwright');
const base='http://127.0.0.1:8765/',out='outputs/e30_study/';
const meta=JSON.parse(fs.readFileSync('docs/snapshots/three_input_comparison_snapshot.json'));
const expected=JSON.parse(fs.readFileSync(out+'browser_expected.json'));
const ids=Object.keys(meta.experiments),checks=[],errors=[];
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const ok=text=>{checks.push(text);console.log('PASS',text)};
const browser=await chromium.launch({executablePath:'/usr/bin/chromium',args:['--no-sandbox','--disable-dev-shm-usage']});
const attach=p=>{p.setDefaultTimeout(120000);p.on('pageerror',e=>{errors.push(e.message);console.error('PAGE ERROR',e.message)});p.on('crash',()=>console.error('PAGE CRASH'))};
try{
 const p=await browser.newPage({viewport:{width:1440,height:1000},acceptDownloads:true});attach(p);
 await p.goto(base+'outputs/e21_study/belajar_Tiga_Input_Privat.html');
 await p.waitForFunction(()=>window.ThreeInputBook?.ready());await p.route('**/*',r=>r.abort());
 assert.equal(await p.locator('#familyExperiment option').count(),3);
 assert.equal(await p.locator('#familyExperiment').inputValue(),'E29');
 for(const s of ['S1','S2','S3'])assert.equal(await p.locator('#familyMetrics'+s+' th').count(),11);
 assert.match(await p.locator('#familyMetricsS3').innerText(),/6,3277/);
 assert.match(await p.locator('#familyMetricsS3').innerText(),/7,0447/);
 assert(await p.locator('#familyMetricsS3 tbody tr').first().locator('td').nth(ids.indexOf(meta.studyRecommendation.rmse_best)+1).evaluate(e=>e.classList.contains('best')));assert.match(await p.locator('#familyMetricsS3').innerText(),/5,4293/);
 ok('Sepuluh hasil tersedia; hanya tiga kode dimuat, utama E29 dengan RMSE5,4293.');

 await p.setViewportSize({width:1000,height:1000});
 await p.locator('#familyBody').evaluate(e=>{e.scrollTop=700;e.querySelector('.compare-table-wrap').scrollLeft=250});
 await p.locator('#family-method').click();await p.locator('#family-overview').click();
 await p.waitForFunction(()=>document.getElementById('familyBody').scrollTop>=690);
 assert(await p.locator('#familyBody .compare-table-wrap').first().evaluate(e=>e.scrollLeft>=240));
 await p.setViewportSize({width:1440,height:1000});
 await p.locator('#familyBody').evaluate(e=>e.scrollTop=0);
 const table=await p.locator('#familyMetricsS3').boundingBox();
 assert(await p.locator('#familyMetricsS3 th').first().evaluate(e=>getComputedStyle(e).position==='sticky'));
 ok('Posisi scroll panel dan horizontal tabel pulih saat kembali ke tab; label baris tetap terlihat.');

 await p.locator('#family-accuracy').click();
 for(const scenario of ['S1','S2','S3','all']){
  await p.getByLabel('Skenario perbandingan',{exact:true}).selectOption(scenario);
  assert(await p.evaluate(()=>{const c=ThreeInputBook.data,s=ThreeInputBook.getState().scenario,k=Object.keys(c.experiments).flatMap(e=>['S1','S2','S3'].filter(x=>s==='all'||x===s).map(x=>[e,x])),get=[r=>r.raw_exact,r=>r.rounded_exact,...['1','2','5','10'].map(t=>r=>r.within[t]),r=>r.mae,r=>r.rmse];return [...document.querySelectorAll('#familyAccuracy tbody tr')].every((tr,i)=>[...tr.children].slice(1).every((td,j)=>Number(td.dataset.value)===get[i](c.predictionAudit[k[j][0]][k[j][1]])))}));
 }
 assert.equal(await p.locator('.pair-card').count(),45);
 assert.equal(await p.locator('.pair-round').count(),135);
 assert(await p.evaluate(()=>ThreeInputBook.data.pairwise.every(d=>d.impact.every(r=>{const row=document.querySelector('[data-pair="'+d.left+'_'+d.right+'"] [data-scenario="'+r.scenario+'"]');return ['left','right','ties'].every(k=>Number(row.querySelector('[data-side="'+k+'"]').dataset.value)===r[k])}))));
 ok('Ketepatan, sorotan dan45duel/135skenario benar; tidak menukar jumlah dekat dengan RMSE.');

 await p.locator('#family-predictions').click();await p.waitForSelector('#familyPredictions');
 const actual=await p.evaluate(()=>ThreeInputBook.answers());
 assert.equal(actual.length,60);
 for(let i=0;i<60;i++){
  assert.equal(actual[i].id,expected[i].id);assert.equal(actual[i].teacher,expected[i].teacher);
  assert.deepEqual(actual[i].predictions,expected[i].predictions);
  assert.equal(sha([actual[i].question,actual[i].reference,actual[i].answer].join('\0')),expected[i].text_sha256);
 }
 assert.equal(await p.locator('#familyPredictions [data-prediction]').count(),1800);
 assert(await p.evaluate(async()=>{const rows=await ThreeInputBook.answers();return [...document.querySelectorAll('#familyPredictions tbody tr')].every((tr,i)=>[...tr.querySelectorAll('[data-model]')].every(td=>Number(td.dataset.prediction)===rows[i].predictions[td.dataset.model]))}));
 for(const scope of ['S1','S2','S3','all']){
  const keys=ids.flatMap(e=>['S1','S2','S3'].filter(s=>scope==='all'||scope===s).map(s=>e+'_'+s));
  const nearest=Object.fromEntries(keys.map(k=>[k,{unique:0,tied:0,share:0}]));
  for(const row of expected){const min=Math.min(...keys.map(k=>Math.abs(row.predictions[k]-row.teacher))),wins=keys.filter(k=>Math.abs(Math.abs(row.predictions[k]-row.teacher)-min)<=1e-5);for(const k of wins){nearest[k][wins.length===1?'unique':'tied']++;nearest[k].share+=1/wins.length}}
  assert.deepEqual(nearest,meta.nearest[scope]);
 }
 await p.getByRole('button',{name:'Peringkat keseluruhan',exact:true}).click();
 assert.equal(await p.locator('.rank-card').count(),30);
 await p.getByLabel('Skenario perbandingan',{exact:true}).selectOption('S3');
 for(const sort of ['nearest','rmse','mae']){
  await p.getByLabel('Urutkan peringkat').selectOption(sort);
  const order=ids.map(e=>({e,...meta.nearest.S3[e+'_S3'],...meta.predictionAudit[e].S3})).sort((a,b)=>sort==='nearest'?b.share-a.share||a.rmse-b.rmse:a[sort]-b[sort]).map(r=>r.e+'_S3');
  assert.deepEqual(await p.locator('.rank-card').evaluateAll(es=>es.map(e=>e.dataset.model)),order);
 }
 ok('1800prediksi dan60teks/checksum tepat sumber; semua peringkat dihitung ulang mandiri, E29 dihitung tanpa mengubah prediksi lama.');

 await p.locator('#family-training').click();
 for(const scenario of ['S1','S2','S3']){
  await p.getByLabel('Skenario perbandingan',{exact:true}).selectOption(scenario);
  for(const fold of ['1','2','3','4','5']){
   await p.getByLabel('Fold kurva').selectOption(fold);
   assert.equal(await p.locator('.family-curve polyline').count(),10);
   assert(await p.evaluate(()=>{const st=ThreeInputBook.getState(),x=ThreeInputBook.data.experiments;return [...document.querySelectorAll('.family-curve circle')].every(d=>{const r=x[d.dataset.experiment].history[st.scenario].find(r=>r.Tahap==='Validasi silang'&&Number(r.Fold)===Number(st.fold)&&r.Epoch===Number(d.dataset.epoch));return Number(d.dataset.value)===r[st.curve]})}));
  }
 }
 await p.getByLabel('Angka kurva').selectOption('Loss Training (MSE)');
 await p.getByLabel('Periksa epoch',{exact:true}).evaluate(e=>e.value='2');
 await p.getByLabel('Periksa epoch',{exact:true}).dispatchEvent('input');
 assert.equal(await p.locator('.curve-inspect-value').count(),10);
 await p.locator('#family-maps').click();
 for(const e of ids)assert.equal(await p.locator('#familyMap'+e+' [data-count]').evaluateAll(es=>es.reduce((a,e)=>a+Number(e.dataset.count),0)),60);
 ok('15kombinasi fold/skenario, sepuluh kurva, inspeksi epoch dan sepuluh peta skor sesuai nilai asli.');

 await p.locator('#family-training').click();await p.getByLabel('Skenario perbandingan',{exact:true}).selectOption('S2');await p.getByLabel('Eksperimen kurva',{exact:true}).selectOption('E29');assert.equal(await p.locator('.family-curve polyline').count(),1);assert.equal(await p.locator('.family-curve [data-trace="E29"]').count(),1);await p.getByLabel('Eksperimen kurva',{exact:true}).selectOption('all');assert.equal(await p.locator('.family-curve polyline').count(),10);
 await p.locator('#family-identities').click();assert.equal(await p.locator('.identity-card').count(),10);assert.equal(await p.locator('.identity-pending').count(),0);assert.equal(await p.locator('[data-reserved]').count(),0);assert.equal(await p.locator('.identity-card').getByRole('button',{name:'Pelajari kode',exact:true}).count(),3);await p.getByRole('button',{name:'Menunggu hasil',exact:true}).click();assert.equal(await p.locator('.identity-card:visible').count(),0);assert.equal(await p.locator('.identity-no-results').count(),1);await p.getByRole('button',{name:'Semua identitas',exact:true}).click();await p.locator('#identitySearch').fill('E30');assert.equal(await p.locator('.identity-card:visible').count(),1);assert.equal(await p.locator('#identity-E30').getByRole('button',{name:'Pelajari kode',exact:true}).count(),1);await p.locator('#identitySearch').fill('');
 ok('Fokus kurva S2 tidak memotong angka; sepuluh hasil aktual tanpa placeholder.');
 await p.evaluate(()=>{Object.defineProperty(document.documentElement,'requestFullscreen',{value:undefined,configurable:true})});
 await p.locator('#familyExpand').click();
 assert(await p.locator('#comparisonViewDialog').evaluate(e=>{const r=e.getBoundingClientRect();return r.x===0&&r.y===0&&Math.abs(r.width-innerWidth)<2&&Math.abs(r.height-innerHeight)<2}));
 assert.equal(await p.locator('#familyResults>.dashboard-heading').isVisible(),false);
 await p.locator('#family-overview').click();await p.locator('#familyBody').evaluate(e=>e.scrollTop=0);
 await p.locator('#familyMetricsS1 td').first().hover();await p.mouse.wheel(0,350);
 await p.waitForFunction(()=>document.getElementById('familyBody').scrollTop>100);
 await p.locator('#familyClose').click();
 ok('Fullscreen menutupi viewport; wheel pada tabel tetap menggerakkan panel isi.');

 await p.locator('#family-excel').click();
 for(const experiment of ['E30','E24']){
  await p.getByLabel('Eksperimen Excel',{exact:true}).selectOption(experiment);
  for(let i=0;i<5;i++){
   await p.getByLabel('Workbook hasil',{exact:true}).selectOption(String(i));await p.waitForFunction(()=>document.getElementById('familyBody').getAttribute('aria-busy')==='false'&&document.querySelector('#familyExcelTable tbody tr'));
   assert.equal(await p.getByLabel('Sheet hasil',{exact:true}).locator('option').count(),meta.experiments[experiment].books[i].sheets.length);
   const waiting=p.waitForEvent('download');await p.getByRole('button',{name:'Unduh XLSX asli ↓',exact:true}).click();const download=await waiting,path=out+'download_'+experiment+'_'+i+'.xlsx';await download.saveAs(path);assert.equal(sha(fs.readFileSync(path)),meta.experiments[experiment].books[i].sha256);
  }
  await p.getByLabel('Workbook hasil',{exact:true}).selectOption('1');await p.waitForFunction(()=>document.getElementById('familyBody').getAttribute('aria-busy')==='false');
  for(const name of experiment==='E30'?['Bobot Attention S3','Ringkasan CoAttn Pooling S3','Bobot Referensi S3','Bobot CoAttention Respons S3','Pasangan Token Top1 S3']:['Bobot Attention S3','Pooling Jawaban S3']){await p.getByLabel('Sheet hasil',{exact:true}).selectOption(name);await p.waitForFunction(()=>document.getElementById('familyBody').getAttribute('aria-busy')==='false'&&document.querySelector('#familyExcelTable tbody tr'));assert((await p.locator('#familyExcelTable tbody tr').count())>0)}
 }
 await p.getByLabel('Eksperimen Excel',{exact:true}).selectOption('E29');await p.getByLabel('Workbook hasil',{exact:true}).selectOption('1');await p.getByLabel('Sheet hasil',{exact:true}).selectOption('Ringkasan BiGRU Attention S3');await p.waitForFunction(()=>document.getElementById('familyBody').getAttribute('aria-busy')==='false');
 await p.getByLabel('Mulai komponen hasil',{exact:true}).fill('120');await p.getByLabel('Mulai komponen hasil',{exact:true}).dispatchEvent('change');assert.match(await p.locator('#familyExcelTable thead').innerText(),/Komponen 120/);await p.locator('#familyExcelTable td').nth(1).click();assert.match(await p.locator('.family-excel-inspector').innerText(),/Komponen 120/);await p.getByLabel('Cari baris hasil',{exact:true}).fill('tidak-ada-jawaban');assert.equal(await p.locator('#familyExcelTable tbody tr').count(),0);await p.getByLabel('Cari baris hasil',{exact:true}).fill('');
 ok('Sepuluh XLSX E30/E24 byte-identik; explorer mandiri bekerja untuk eksperimen tanpa kode, vektor, pencarian dan nilai lengkap.');
 await p.close();

 const pub=await browser.newPage({viewport:{width:1440,height:1000}});attach(pub);
 await pub.goto(base+'docs/belajar_Tiga_Input_Spider.html');await pub.waitForFunction(()=>window.ThreeInputBook);
 await pub.route('**/*',r=>r.abort());assert.equal(await pub.evaluate(()=>ThreeInputBook.ready()),false);
 await pub.evaluate(()=>Promise.all([ThreeInputBook.openBook('E26',10,'code'),ThreeInputBook.openBook('E29',10,'code'),ThreeInputBook.openBook('E30',10,'code')]));
 assert.equal(await pub.locator('#familyExperiment').inputValue(),'E30');
 assert.equal(await pub.locator('#familyFrame').getAttribute('title'),'Meja belajar E30');
 ok('Pergantian kode cepat tidak saling menimpa; pilihan terakhir tetap aktif.');
 for(const e of ['E29','E30','E26']){
  await pub.evaluate(e=>ThreeInputBook.openBook(e,10,'code'),e);
  const child=pub.frames().find(f=>f.parentFrame()===pub.mainFrame());await child.waitForSelector('.margin-note');
  assert.equal(await child.locator('#cellSelect option').count(),30);
  await child.evaluate(()=>{Object.defineProperty(document.documentElement,'requestFullscreen',{value:undefined,configurable:true});window.copied=null;Object.defineProperty(navigator,'clipboard',{value:{writeText:async t=>{window.copied=t}},configurable:true})});
  await child.locator('#copyCode').click();assert(await child.evaluate(()=>window.copied===MLPStudy.data.cells.find(c=>c.index===MLPStudy.getState().cell).source));
  await child.locator('#learningMapButton').click();assert.equal(await child.locator('#termBody .learning-map-card').count(),10);await child.locator('#closeTerm').click();
  for(const [width,height]of [[1440,1000],[768,1024],[390,844]]){
   await pub.setViewportSize({width,height});await child.locator('#expandCode').click();
   await pub.waitForFunction(()=>document.body.classList.contains('family-code-fullscreen'));
   assert(await pub.locator('#familyFrame').evaluate(e=>{const r=e.getBoundingClientRect();return r.x===0&&r.y===0&&Math.abs(r.width-innerWidth)<2&&Math.abs(r.height-innerHeight)<2}));
   assert(await child.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+2));
   assert.equal(await child.locator('#notePane').isVisible(),false);
   assert.equal(await child.locator('.canvas-code-block').count(),30);
   assert(await child.evaluate(()=>{const source=new Map(MLPStudy.data.cells.map(c=>[c.index,c.source.split('\n')]));return [...document.querySelectorAll('#codeLines .code-line')].every(r=>{const a=r.querySelector('code').textContent,b=source.get(Number(r.dataset.cell))[Number(r.dataset.line)-1];return b===''?a.trim()==='':a===b})}));
   await child.locator('#codeScroll').evaluate(e=>e.scrollTop=2000);assert(await child.locator('#codeScroll').evaluate(e=>e.scrollTop>=1900));
   await child.locator('#canvasModeToggle').click();assert.equal(await child.locator('.canvas-code-block').count(),0);
   await child.locator('#canvasModeToggle').click();assert.equal(await child.locator('.canvas-code-block').count(),30);
   if(width===1440&&e==='E30'){await child.evaluate(()=>MLPStudy.jumpCode(30,1));await pub.screenshot({path:'docs/previews/mlp_study/52_e30_selected_code.png'})}
   await child.locator('#closeCode').click();await pub.waitForFunction(()=>!document.body.classList.contains('family-code-fullscreen'));
  }
 }
 ok('Kode utuh dan dapat disalin; sticky note, dua mode fullscreen, scroll desktop/tablet/HP bekerja.');
 await pub.setViewportSize({width:1440,height:1000});await pub.evaluate(()=>ThreeInputBook.setMode('results'));
 await pub.locator('#family-overview').click();await pub.locator('#familyResults').scrollIntoViewIfNeeded();
 await pub.screenshot({path:'docs/previews/mlp_study/54_ten_experiments.png'});
 assert.deepEqual(errors,[]);ok('Tidak ada error JavaScript; preview publik tanpa payload individu.');
 fs.writeFileSync(out+'browser.json',JSON.stringify({passed:checks.length,failed:0,checks,browser:browser.version()},null,2));
 console.log(JSON.stringify({passed:checks.length,failed:0}));
}finally{await browser.close()}
