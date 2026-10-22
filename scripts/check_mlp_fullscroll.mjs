// Uji membaca semua blok pada satu kanvas; tidak mengeksekusi source notebook.
import {createRequire} from 'node:module';import fs from 'node:fs';import assert from 'node:assert/strict';
const require=createRequire(import.meta.url),{chromium}=require('playwright');
const url=process.argv[2]||'http://127.0.0.1:8765/docs/belajar_MLP_Spider.html';
const b=await chromium.launch({executablePath:'/usr/bin/chromium',args:['--no-sandbox']}),context=await b.newContext({viewport:{width:1440,height:1000},reducedMotion:'reduce',permissions:['clipboard-read','clipboard-write']});
await context.addInitScript(()=>Object.defineProperty(Element.prototype,'requestFullscreen',{value:undefined,configurable:true}));
const p=await context.newPage(),errors=[],checks=[];p.on('pageerror',e=>errors.push(e.message));function ok(s){checks.push(s);console.log('OK:',s)}const settle=()=>p.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
try{
 await p.goto(url);await p.evaluate(()=>{MLPStudy.openChapter(1);MLPStudy.setTab('code')});await p.locator('#expandCode').click();
 assert.equal((await p.evaluate(()=>MLPStudy.getState())).codeMode,'block');assert.equal(await p.locator('.canvas-code-block').count(),0);
 await p.locator('#canvasModeToggle').click();await settle();assert.equal(await p.locator('.canvas-code-block').count(),30);assert.equal(await p.locator('.code-line').count(),2488);
 const code=await p.evaluate(()=>[...document.querySelectorAll('.canvas-code-block')].map(block=>({cell:Number(block.dataset.cell),source:[...block.querySelectorAll('.code-line code')].map(c=>c.textContent===' '?'':c.textContent).join('\n')})));
 const original=await p.evaluate(()=>MLPStudy.data.cells.map(c=>({cell:c.index,source:c.source})));assert.deepEqual(code,original);
 assert.match(await p.locator('.canvas-block-heading').first().innerText(),/Environment & folder kerja/);assert.equal(await p.locator('.canvas-block-heading').count(),30);
 ok('Per-block mode retained; all mode renders every original line in 30 named blocks with exact reversible source');
 for(const width of [1440,900,390,320]){
  await p.setViewportSize({width,height:900});await settle();
  const d=await p.evaluate(()=>({scrollers:[...document.getElementById('codeDialog').querySelectorAll('*')].filter(e=>['auto','scroll'].includes(getComputedStyle(e).overflowY)&&e.scrollHeight>e.clientHeight+1).map(e=>e.id),x:document.getElementById('codeScroll').scrollWidth-document.getElementById('codeScroll').clientWidth}));
  assert.deepEqual(d.scrollers,['codeScroll']);assert.ok(d.x<=1);
 }
 ok('All-code mode shares one scroll, with no horizontal overflow across desktop and small mobile viewports');
 await p.setViewportSize({width:1440,height:1000});await p.locator('#codeTools summary').click();await p.locator('#cellSelect').selectOption('30');await p.locator('#canvasNoteSelect').selectOption({label:'Baris 58 · Titik fusion yang sebenarnya'});await settle();
 const highlight=await p.locator('.selected-line').evaluate(e=>({cell:e.dataset.cell,line:e.dataset.line}));assert.deepEqual(highlight,{cell:'30',line:'58'});
 assert.ok(await p.locator('#notePane .sticky-note').innerText().then(t=>t.includes('Titik fusion')));
 assert.ok(await p.evaluate(()=>Math.abs(document.querySelector('.selected-line').getBoundingClientRect().top-document.querySelector('.sticky-note').getBoundingClientRect().top)<2));
 ok('Sticky-note lookup includes block and line identity; repeated line numbers do not target the wrong block');
 await p.locator('#codeTools summary').click();await p.locator('#codeFind').fill('prepare_persistent_storage');assert.ok(await p.locator('.find-match[data-cell="3"]').count()>0);await p.locator('#codeFind').fill('');
 await p.locator('#cellSelect').selectOption('46');await settle();await p.locator('#copyCode').click();
 const active=await p.evaluate(()=>MLPStudy.getState().cell);assert.equal(await p.evaluate(()=>navigator.clipboard.readText()),await p.evaluate(i=>MLPStudy.data.cells.find(c=>c.index===i).source,active));
 await p.locator('#codeTools summary').click();await p.locator('#canvasModeToggle').click();await settle();assert.equal(await p.locator('.canvas-code-block').count(),0);assert.equal(await p.locator('.code-line').count(),await p.evaluate(()=>MLPStudy.data.cells.find(c=>c.index===MLPStudy.getState().cell).htmlLines.length));
 ok('Search spans all blocks, block picker jumps within the same canvas, copy uses the active original block, and per-block mode still works');
 await p.locator('#canvasModeToggle').click();await p.locator('#closeCode').click();await settle();assert.equal(await p.locator('.study-book > #pane-code').count(),1);assert.equal(await p.locator('.canvas-code-block').count(),0);
 await p.locator('#expandCode').click();await settle();assert.equal(await p.locator('.canvas-code-block').count(),30);await p.keyboard.press('Escape');await settle();assert.equal(await p.locator('#codeDialog').evaluate(e=>e.open),false);
 assert.deepEqual(errors,[]);ok('Exit restores the regular book, reopening retains the chosen mode, and Esc works with no page errors');
 fs.writeFileSync('outputs/mlp_study/fullscroll_browser_verification.json',JSON.stringify({scope:'ALL CODE CANVAS; NO TRAINING',passed:checks.length,failed:0,checks,browser:b.version()},null,2));
}finally{await b.close()}
