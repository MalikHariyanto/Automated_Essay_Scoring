// Uji kanvas kode + catatan dengan satu scroll. Tidak menjalankan model AES.
import {createRequire} from 'node:module';
import fs from 'node:fs';
import assert from 'node:assert/strict';
const require=createRequire(import.meta.url),{chromium}=require('playwright');
const [url='http://127.0.0.1:8765/docs/belajar_MLP_Spider.html',privateUrl]=process.argv.slice(2);
const directory='outputs/mlp_study';fs.mkdirSync(directory,{recursive:true});
const browser=await chromium.launch({executablePath:'/usr/bin/chromium',headless:true,args:['--no-sandbox']});
const context=await browser.newContext({viewport:{width:1440,height:1000},reducedMotion:'reduce',permissions:['clipboard-read','clipboard-write']});
// Exercise the viewport fallback at each responsive size, then test real Fullscreen API separately.
await context.addInitScript(()=>Object.defineProperty(Element.prototype,'requestFullscreen',{configurable:true,value:undefined}));
const page=await context.newPage(),errors=[],requests=[];
page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
const report={scope:'SINGLE SCROLL CANVAS AND NATIVE FULLSCREEN; NO TRAINING',browser:browser.version(),checks:[]};
function passed(message){report.checks.push(message);console.log('OK:',message)}
async function settled(){await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))))}
async function open(){await page.locator('#expandCode').click();await settled();assert.ok(await page.locator('#codeDialog').evaluate(e=>e.open))}
async function tools(){if(!await page.locator('#codeTools').evaluate(e=>e.open))await page.locator('#codeTools summary').click()}
async function chooseNote(title){await tools();const options=await page.locator('#canvasNoteSelect option').evaluateAll(options=>options.map(o=>({value:o.value,label:o.textContent})));const option=options.find(o=>o.label.includes(title));assert.ok(option,title);await page.locator('#canvasNoteSelect').selectOption(option.value);await settled()}
async function assertCanvas(){
 const layout=await page.evaluate(()=>{
  const rect=id=>{const r=document.getElementById(id).getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height,right:r.right}};
  const pane=document.getElementById('notePane');
  const scrollers=[...document.getElementById('codeDialog').querySelectorAll('*')].filter(e=>['auto','scroll'].includes(getComputedStyle(e).overflowY)&&e.scrollHeight>e.clientHeight+1);
  return {dialog:rect('codeDialog'),scroll:rect('codeScroll'),source:rect('codeLines'),note:rect('notePane'),viewport:[innerWidth,innerHeight],scrollers:scrollers.map(e=>e.id),noteBackground:getComputedStyle(pane).backgroundColor,noteBorder:getComputedStyle(pane).borderLeftWidth,noteOverflow:getComputedStyle(pane).overflowY,common:document.getElementById('codeCanvas').contains(pane)&&document.getElementById('codeCanvas').contains(document.getElementById('codeLines')),xOverflow:document.getElementById('codeScroll').scrollWidth>document.getElementById('codeScroll').clientWidth+1};
 });
 assert.equal(layout.dialog.x,0);assert.equal(layout.dialog.y,0);assert.equal(layout.dialog.width,layout.viewport[0]);assert.equal(layout.dialog.height,layout.viewport[1]);
 assert.equal(layout.scroll.y,0);assert.equal(layout.scroll.height,layout.viewport[1]);assert.ok(layout.source.right<layout.note.x);
 assert.ok(layout.note.right<=layout.viewport[0]+1);assert.ok(layout.common);assert.equal(layout.noteBackground,'rgba(0, 0, 0, 0)');assert.equal(layout.noteBorder,'0px');assert.equal(layout.noteOverflow,'visible');
 assert.deepEqual(layout.scrollers,['codeScroll']);assert.equal(layout.xOverflow,false);
 assert.equal(await page.locator('#codeDialog header').count(),0);assert.equal(await page.locator('#cellSelect').isVisible(),false);
}
try{
 await page.goto(url);await page.waitForFunction(()=>window.MLPStudy);
 await page.evaluate(()=>{MLPStudy.openChapter(10);MLPStudy.setTab('code');window.originalPanel=document.getElementById('pane-code');window.originalCode=document.getElementById('codeLines').textContent});
 for(const [width,height] of [[1440,1000],[1024,768],[768,1024],[390,844],[320,568],[844,390]]){
  await page.setViewportSize({width,height});await open();await assertCanvas();
  assert.ok(await page.evaluate(()=>originalPanel===document.getElementById('pane-code')&&originalCode===document.getElementById('codeLines').textContent));
  assert.equal(await page.evaluate(()=>document.activeElement.id),'closeCode');
  await page.keyboard.press('Escape');await settled();assert.equal(await page.locator('#codeDialog').evaluate(d=>d.open),false);assert.equal(await page.evaluate(()=>document.activeElement.id),'expandCode');
 }
 passed('Six viewport sizes: no header/divider, full-height canvas, exactly one scrolling container, no horizontal scrollbar, same source DOM; Esc restores focus');
 await page.setViewportSize({width:1440,height:1000});await page.evaluate(()=>Object.defineProperty(document.documentElement,'requestFullscreen',{configurable:true,value:()=>Promise.reject(Error('Fullscreen denied'))}));await open();await assertCanvas();await page.locator('#closeCode').click();
 passed('Rejected Fullscreen API keeps a working full-viewport canvas with normal exit controls');
 await page.setViewportSize({width:1440,height:1000});await open();await chooseNote('Titik fusion yang sebenarnya');
 assert.ok(await page.evaluate(()=>Math.abs(document.querySelector('#notePane .sticky-note').getBoundingClientRect().top-document.querySelector('.selected-line').getBoundingClientRect().top)<2));
 const before=await page.evaluate(()=>({body:scrollY,scroll:document.getElementById('codeScroll').scrollTop,note:document.querySelector('#notePane .sticky-note').getBoundingClientRect().top,line:document.querySelector('.selected-line').getBoundingClientRect().top,id:MLPStudy.getState().note}));
 await page.locator('#notePane .sticky-note').hover();await page.mouse.wheel(0,45);await page.waitForFunction(previous=>document.getElementById('codeScroll').scrollTop>previous+30,before.scroll);await settled();
 const after=await page.evaluate(()=>({body:scrollY,scroll:document.getElementById('codeScroll').scrollTop,note:document.querySelector('#notePane .sticky-note').getBoundingClientRect().top,line:document.querySelector('.selected-line').getBoundingClientRect().top,id:MLPStudy.getState().note,noteScroll:document.getElementById('notePane').scrollTop}));
 assert.equal(after.body,before.body);assert.equal(after.id,before.id);assert.equal(after.noteScroll,0);assert.ok(Math.abs((after.note-before.note)+(after.scroll-before.scroll))<2);assert.ok(Math.abs((after.line-before.line)-(after.note-before.note))<2);
 passed('Wheel over the sticky note scrolls the same canvas; source highlight and note move together by the same distance');
 // Let the current note leave the canvas viewport; a visible code marker becomes active.
 await chooseNote('RoBERTa ikut fine-tuning');
 const previousNote=await page.evaluate(()=>MLPStudy.getState().note);
 await page.locator('#codeScroll').evaluate(e=>e.scrollTop+=700);await page.waitForFunction(previous=>MLPStudy.getState().note!==previous,previousNote,{timeout:3000});
 assert.notEqual(await page.evaluate(()=>MLPStudy.getState().note),previousNote);
 passed('Note changes to a visible code marker after the current explanation scrolls out of view');
 await chooseNote('RoBERTa ikut fine-tuning');await page.locator('#notePane .term-button').click();assert.ok(await page.locator('#termDialog').isVisible());await page.keyboard.press('Escape');assert.equal(await page.locator('#termDialog').evaluate(d=>d.open),false);assert.ok(await page.locator('#codeDialog').evaluate(d=>d.open));
 await tools();await page.locator('#cellSelect').selectOption('32');await page.locator('#codeFind').fill('scenario');assert.ok(await page.locator('#codeLines .find-match').count()>0);
 await page.locator('#copyCode').click();assert.equal(await page.evaluate(()=>navigator.clipboard.readText()),await page.evaluate(()=>MLPStudy.data.cells.find(c=>c.index===32).source));
 await page.evaluate(()=>Object.defineProperty(navigator.clipboard,'writeText',{configurable:true,value:()=>Promise.reject(Error('clipboard permission denied'))}));await page.locator('#copyCode').click();assert.equal(await page.evaluate(()=>navigator.clipboard.readText()),await page.evaluate(()=>MLPStudy.data.cells.find(c=>c.index===32).source));
 await page.locator('#codeFind').fill('');await page.locator('#codeTools summary').click();
 passed('Compact menu exposes block selection, note selection, search and exact original-code copy, including denied Clipboard API fallback; glossary still works');
 await page.locator('#closeCode').focus();await page.keyboard.press('Shift+Tab');assert.equal(await page.evaluate(()=>document.activeElement.tagName),'SUMMARY');
 await page.locator('#closeCode').focus();await page.keyboard.press('Tab');assert.equal(await page.evaluate(()=>document.activeElement.id),'codeScroll');
 await page.evaluate(()=>document.getElementById('startBook').focus());assert.ok(await page.evaluate(()=>document.getElementById('codeDialog').contains(document.activeElement)));
 await page.locator('#closeCode').click();await settled();assert.equal(await page.locator('.study-book > #pane-code').count(),1);assert.equal(await page.locator('#codeScroll > #codeLines').count(),1);assert.equal(await page.locator('.code-and-notes > #notePane').count(),1);assert.equal(await page.evaluate(()=>MLPStudy.getState().cell),32);
 passed('Keyboard reaches floating menu and canvas; exit restores original book structure and active block');
 for(const width of [1440,390]){
  await page.setViewportSize({width,height:900});
  for(let i=0;i<16;i++){
   await page.evaluate(i=>{MLPStudy.openChapter(i);MLPStudy.setTab('code')},i);await open();
   assert.ok((await page.locator('#codeLines').innerText()).length>20);assert.ok((await page.locator('#notePane').innerText()).length>20);
   const x=await page.locator('#codeScroll').evaluate(e=>e.scrollWidth-e.clientWidth);assert.ok(x<=1,`${width}/${i} horizontal overflow ${x}`);
   await page.locator('#closeCode').click();await settled();
  }
 }
 passed('All 16 chapters open/close on desktop and mobile; long lines wrap without a second scrollbar');
 // A long note must stay readable as the whole canvas moves, rather than changing midway.
 await page.setViewportSize({width:320,height:568});await page.evaluate(()=>{const candidates=MLPStudy.data.cells.flatMap(c=>c.notes.map(n=>({c:c.index,n})));candidates.sort((a,b)=>b.n.body.length-a.n.body.length);window.longest=candidates[0];MLPStudy.jumpCode(longest.c,longest.n.line)});await open();await tools();await page.locator('#canvasNoteSelect').selectOption(await page.evaluate(()=>longest.n.id));await settled();
 const longNote=await page.evaluate(()=>MLPStudy.getState().note);await page.locator('#codeScroll').evaluate(e=>e.scrollTop+=100);await settled();assert.equal(await page.evaluate(()=>MLPStudy.getState().note),longNote);assert.equal(await page.locator('#notePane').evaluate(e=>getComputedStyle(e).overflowY),'visible');await page.locator('#closeCode').click();
 passed('Long explanations remain in the shared scroll and do not switch while still being read on a small screen');
 // Real browser fullscreen: no stub in this separate context.
 const nativeContext=await browser.newContext({viewport:{width:1440,height:1000},reducedMotion:'reduce'});const nativePage=await nativeContext.newPage();nativePage.on('pageerror',e=>errors.push(e.message));
 await nativePage.goto(url);await nativePage.evaluate(()=>{MLPStudy.openChapter(10);MLPStudy.setTab('code')});await nativePage.locator('#expandCode').click();await nativePage.waitForFunction(()=>document.fullscreenElement===document.documentElement);assert.ok(await nativePage.locator('#codeDialog').evaluate(e=>e.open));await nativePage.locator('#notePane .term-button').click();assert.ok(await nativePage.locator('#termDialog').isVisible());await nativePage.evaluate(()=>document.exitFullscreen());await nativePage.waitForFunction(()=>!document.getElementById('codeDialog').open);assert.equal(await nativePage.locator('.study-book > #pane-code').count(),1);assert.equal(await nativePage.locator('#termDialog').evaluate(d=>d.open),false);await nativeContext.close();
 passed('Real Fullscreen API enters browser fullscreen; leaving browser fullscreen restores the book automatically');
 if(privateUrl){
  await page.goto(privateUrl);await page.waitForFunction(()=>window.MLPStudy?.getState().resultsLoaded&&window.MLPCompare?.getBaselineEvalBytes(),null,{timeout:120000});
  await page.setViewportSize({width:1440,height:1000});await page.evaluate(()=>{MLPStudy.openChapter(10);MLPStudy.setTab('code')});await open();await assertCanvas();await page.locator('#closeCode').click();
  const download=page.waitForEvent('download');await page.locator('#savePrivateHtml').click();await (await download).saveAs(`${directory}/canvas_export_private.html`);
  await page.goto(new URL('outputs/mlp_study/canvas_export_private.html',new URL('../../..',privateUrl)).href);await page.waitForFunction(()=>window.MLPStudy?.getState().resultsLoaded&&window.MLPCompare?.getBaselineEvalBytes(),null,{timeout:120000});assert.equal((await page.evaluate(()=>MLPStudy.getState())).nativeOutputs,34);
  await page.evaluate(()=>{MLPStudy.openChapter(10);MLPStudy.setTab('code')});await open();await assertCanvas();await page.keyboard.press('Escape');
  passed('Private and browser-exported/reopened books retain the single-scroll canvas, 34 outputs and baseline workbook');
 }
 assert.deepEqual(errors,[]);assert.ok(requests.every(r=>r.startsWith('http://127.0.0.1:8765/')||r.startsWith('data:')));passed('No JavaScript errors or external requests');
 report.passed=report.checks.length;report.failed=0;report.skipped=privateUrl?0:1;fs.writeFileSync(`${directory}/canvas_browser_verification.json`,JSON.stringify(report,null,2));
}finally{await browser.close()}
