// Uji browser panel kode + catatan. Tidak menjalankan model atau training AES.
import {createRequire} from 'node:module';
import fs from 'node:fs';
import assert from 'node:assert/strict';
const require=createRequire(import.meta.url),{chromium}=require('playwright');
const [url='http://127.0.0.1:8765/docs/belajar_MLP_Spider.html',privateUrl]=process.argv.slice(2);
const directory='outputs/mlp_study';fs.mkdirSync(directory,{recursive:true});
const browser=await chromium.launch({executablePath:'/usr/bin/chromium',headless:true,args:['--no-sandbox']});
const context=await browser.newContext({viewport:{width:1440,height:1000},reducedMotion:'reduce',permissions:['clipboard-read','clipboard-write']});
const page=await context.newPage(),errors=[],requests=[];
page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
const report={scope:'STATIC FULLSCREEN READER; NO TRAINING',browser:browser.version(),checks:[]};
function passed(message){report.checks.push(message);console.log('OK:',message)}
async function open(){await page.locator('#expandCode').click();assert.ok(await page.locator('#codeDialog').isVisible())}
try{
 await page.goto(url);await page.waitForFunction(()=>window.MLPStudy);
 await page.evaluate(()=>{MLPStudy.openChapter(10);MLPStudy.setTab('code');window.originalPanel=document.querySelector('#pane-code');window.originalCode=document.querySelector('#codeLines').textContent});
 for(const [width,height] of [[1440,1000],[1024,768],[768,1024],[390,844],[320,568],[844,390]]){
  await page.setViewportSize({width,height});await open();
  const layout=await page.evaluate(()=>{
   const rect=id=>{const r=document.getElementById(id).getBoundingClientRect();return {x:r.x,y:r.y,right:r.right,bottom:r.bottom,width:r.width,height:r.height}};
   return {dialog:rect('codeDialog'),code:rect('codeScroll'),notes:rect('notePane'),viewport:[innerWidth,innerHeight],scroll:document.getElementById('codeDialog').scrollHeight,focus:document.activeElement.id,retained:originalPanel===document.getElementById('pane-code')&&originalCode===document.getElementById('codeLines').textContent};
  });
  assert.equal(layout.dialog.x,0);assert.equal(layout.dialog.y,0);assert.equal(layout.dialog.width,width);assert.equal(layout.dialog.height,height);
  assert.ok(layout.code.right<=layout.notes.x+1);assert.ok(Math.abs(layout.code.y-layout.notes.y)<1);
  assert.ok(layout.notes.right<=width+1&&layout.code.height>100&&layout.notes.height>100);
  assert.ok(layout.scroll<=height+1);assert.ok(layout.retained);assert.equal(layout.focus,'closeCode');
  if(width===1440||width===390){await page.screenshot({path:`${directory}/fullscreen_${width}.png`})}
  await page.keyboard.press('Escape');assert.equal(await page.locator('#codeDialog').evaluate(d=>d.open),false);
  assert.equal(await page.evaluate(()=>document.activeElement.id),'expandCode');
 }
 passed('Six viewport sizes: whole viewport, code and notes side by side, no clipped controls; same DOM/code, Esc and focus restoration');
 await page.setViewportSize({width:1440,height:1000});await open();
 const before=await page.evaluate(()=>({y:scrollY,notes:document.getElementById('notePane').scrollTop}));
 await page.locator('#codeScroll').hover();await page.mouse.wheel(0,450);await page.waitForFunction(()=>document.getElementById('codeScroll').scrollTop>100);
 assert.equal(await page.evaluate(()=>document.getElementById('notePane').scrollTop),before.notes);
 const codeY=await page.locator('#codeScroll').evaluate(e=>e.scrollTop);
 await page.locator('#notePane').hover();await page.mouse.wheel(0,300);await page.waitForFunction(()=>document.getElementById('notePane').scrollTop>100);
 assert.equal(await page.locator('#codeScroll').evaluate(e=>e.scrollTop),codeY);assert.equal(await page.evaluate(()=>scrollY),before.y);
 await page.setViewportSize({width:768,height:1000});await page.locator('#codeScroll').hover();await page.mouse.wheel(500,0);await page.waitForFunction(()=>document.getElementById('codeScroll').scrollLeft>10);
 await page.setViewportSize({width:1440,height:1000});
 passed('Independent vertical/horizontal code scrolling and note scrolling keep background page still');
 await page.locator('#closeCode').focus();await page.keyboard.press('Shift+Tab');
 assert.ok(await page.evaluate(()=>document.getElementById('codeDialog').contains(document.activeElement)));
 await page.locator('#closeCode').focus();await page.keyboard.press('Tab');assert.equal(await page.evaluate(()=>document.activeElement.id),'cellSelect');
 await page.evaluate(()=>document.getElementById('startBook').focus());assert.ok(await page.evaluate(()=>document.getElementById('codeDialog').contains(document.activeElement)));
 passed('Native modal traps keyboard focus and makes background controls inert');
 await page.locator('#notePane').evaluate(e=>e.scrollTop=e.scrollHeight);
 await page.locator('#notePane .note-link').last().click();
 assert.ok(await page.locator('#notePane .sticky-note').isVisible());
 assert.equal(await page.locator('#notePane .note-link.active').count(),1);
 assert.equal(await page.locator('#codeScroll .selected-line').count(),1);
 // A known note carries a glossary term; opening its nested dialog must keep the code modal intact.
 await page.locator('#notePane .note-link').filter({hasText:'RoBERTa ikut fine-tuning'}).click();
 await page.locator('#notePane .term-button').click();assert.ok(await page.locator('#termDialog').isVisible());
 await page.keyboard.press('Escape');assert.equal(await page.locator('#termDialog').evaluate(d=>d.open),false);assert.ok(await page.locator('#codeDialog').evaluate(d=>d.open));
 passed('Note selection highlights the matching line; glossary opens above fullscreen and Esc closes only the glossary');
 await page.locator('#notePane .note-link').filter({hasText:'Titik fusion yang sebenarnya'}).click();
 await page.waitForFunction(()=>Math.abs(document.querySelector('#notePane .sticky-note').getBoundingClientRect().top-document.querySelector('.selected-line').getBoundingClientRect().top)<2);
 await page.locator('#codeScroll').evaluate(e=>{const line=document.querySelector('.selected-line');e.scrollTop=line.offsetTop-document.getElementById('codeLines').offsetTop-240});
 await page.waitForFunction(()=>Math.abs(document.querySelector('#notePane .sticky-note').getBoundingClientRect().top-document.querySelector('.selected-line').getBoundingClientRect().top)<2);
 const tracking=await page.locator('#notePane .sticky-note').evaluate(e=>e.getBoundingClientRect().top);
 await page.locator('#codeScroll').hover();await page.mouse.wheel(0,60);
 await page.waitForFunction(y=>document.querySelector('#notePane .sticky-note').getBoundingClientRect().top<y-40,tracking);
 assert.ok(await page.evaluate(()=>Math.abs(document.querySelector('#notePane .sticky-note').getBoundingClientRect().top-document.querySelector('.selected-line').getBoundingClientRect().top)<2));
 const oldNote=await page.evaluate(()=>MLPStudy.getState().note);
 await page.locator('#codeScroll').evaluate(e=>e.scrollTop=e.scrollHeight);
 await page.waitForFunction(previous=>MLPStudy.getState().note!==previous,oldNote);
 assert.ok(await page.evaluate(()=>{const pane=document.getElementById('notePane').getBoundingClientRect(),note=document.querySelector('#notePane .sticky-note').getBoundingClientRect();return note.top>=pane.top&&note.bottom<=pane.bottom}));
 passed('Note aligns to highlighted line, tracks scrolling, changes to a visible code marker and stays within panel bounds');
 await page.locator('#cellSelect').selectOption('32');
 assert.equal(await page.evaluate(()=>MLPStudy.getState().cell),32);assert.ok(await page.locator('#codeDialog #notePane').isVisible());
 await page.locator('#codeFind').fill('scenario');assert.ok(await page.locator('#codeLines .find-match').count()>0);
 await page.locator('#copyCode').click();
 assert.equal(await page.evaluate(()=>navigator.clipboard.readText()),await page.evaluate(()=>MLPStudy.data.cells.find(c=>c.index===32).source));
 await page.locator('#codeFind').fill('');
 await page.locator('#codeScroll').evaluate(e=>e.scrollLeft=150);
 await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
 const position=await page.evaluate(()=>({y:scrollY,cell:MLPStudy.getState().cell,note:MLPStudy.getState().note,top:document.getElementById('codeScroll').scrollTop,left:document.getElementById('codeScroll').scrollLeft}));
 await page.locator('#closeCode').click();
 const restored=await page.evaluate(()=>({y:scrollY,cell:MLPStudy.getState().cell,note:MLPStudy.getState().note,top:document.getElementById('codeScroll').scrollTop,left:document.getElementById('codeScroll').scrollLeft,maxLeft:document.getElementById('codeScroll').scrollWidth-document.getElementById('codeScroll').clientWidth}));
 assert.deepEqual({...restored,maxLeft:undefined},{...position,left:Math.min(position.left,restored.maxLeft),maxLeft:undefined});
 assert.equal(await page.locator('.study-book > #pane-code').count(),1);
 passed('Block selection, search and exact original-code copy work fullscreen; exit restores cell/note/scroll and original location');
 // The clipboard fallback also has to work inside an inert-background modal.
 await open();await page.evaluate(()=>Object.defineProperty(navigator.clipboard,'writeText',{configurable:true,value:()=>Promise.reject(new Error('clipboard permission denied'))}));
 await page.locator('#copyCode').click();assert.equal(await page.evaluate(()=>navigator.clipboard.readText()),await page.evaluate(()=>MLPStudy.data.cells.find(c=>c.index===32).source));
 assert.equal(await page.evaluate(()=>document.activeElement.id),'copyCode');await page.keyboard.press('Escape');
 await open();await page.evaluate(()=>MLPStudy.setTab('output'));
 assert.equal(await page.locator('#codeDialog').evaluate(d=>d.open),false);assert.ok(await page.locator('#pane-output').isVisible());assert.equal(await page.evaluate(()=>document.body.classList.contains('reading-code-fullscreen')),false);
 passed('Denied Clipboard API fallback works inside modal; changing tabs releases fullscreen and scroll lock');
 for(const width of [1440,390]){
  await page.setViewportSize({width,height:900});
  for(let i=0;i<16;i++){
   await page.evaluate(i=>{MLPStudy.openChapter(i);MLPStudy.setTab('code')},i);
   await open();assert.ok((await page.locator('#codeDialog #codeLines').innerText()).length>20);assert.ok((await page.locator('#codeDialog #notePane').innerText()).length>20);
   await page.locator('#closeCode').click();assert.equal(await page.locator('.study-book > #pane-code').count(),1);
  }
 }
 passed('All 16 chapters open and close fullscreen on desktop and mobile');
 await page.setViewportSize({width:1440,height:1000});
 await page.evaluate(()=>{MLPStudy.openChapter(10);MLPStudy.setTab('code')});
 await page.locator('#notePane .note-link').filter({hasText:'Titik fusion yang sebenarnya'}).click();
 await page.waitForFunction(()=>Math.abs(document.querySelector('#notePane .sticky-note').getBoundingClientRect().top-document.querySelector('.selected-line').getBoundingClientRect().top)<2);
 assert.equal(await page.locator('#codeDialog').evaluate(d=>d.open),false);
 passed('Line-aligned notes also work in the regular side-by-side book view');
 if(privateUrl){
  await page.goto(privateUrl);await page.waitForFunction(()=>window.MLPStudy?.getState().resultsLoaded&&window.MLPCompare?.getBaselineEvalBytes(),null,{timeout:120000});
  await page.evaluate(()=>{MLPStudy.openChapter(10);MLPStudy.setTab('code')});await open();
  assert.ok(await page.locator('#codeDialog #notePane .sticky-note').isVisible());await page.keyboard.press('Escape');
  const download=page.waitForEvent('download');await page.locator('#savePrivateHtml').click();
  await (await download).saveAs(`${directory}/fullscreen_export_private.html`);
  const exported=new URL('outputs/mlp_study/fullscreen_export_private.html',new URL('../../..',privateUrl));
  await page.goto(exported.href);await page.waitForFunction(()=>window.MLPStudy?.getState().resultsLoaded&&window.MLPCompare?.getBaselineEvalBytes(),null,{timeout:120000});
  assert.equal((await page.evaluate(()=>MLPStudy.getState())).nativeOutputs,34);
  await page.evaluate(()=>{MLPStudy.openChapter(10);MLPStudy.setTab('code')});await open();assert.ok(await page.locator('#codeDialog #codeLines').isVisible());await page.keyboard.press('Escape');
  passed('Private book and browser-exported/reopened book retain fullscreen, all 34 outputs and baseline workbook');
 }
 assert.deepEqual(errors,[]);assert.ok(requests.every(r=>r.startsWith('http://127.0.0.1:8765/')||r.startsWith('data:')));
 passed('No JavaScript errors or external requests');
 report.passed=report.checks.length;report.failed=0;report.skipped=privateUrl?0:1;
 fs.writeFileSync(`${directory}/fullscreen_browser_verification.json`,JSON.stringify(report,null,2));
}finally{await browser.close()}
