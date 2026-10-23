// Gestur browser asli untuk perpindahan halaman/panel. Tidak menjalankan notebook.
import {createRequire} from 'node:module';import fs from 'node:fs';
const require=createRequire(import.meta.url),{chromium}=require('playwright');
const baseline=process.argv.includes('--record-baseline'),directory='outputs/mlp_study/scroll_review';fs.mkdirSync(directory,{recursive:true});
const browser=await chromium.launch({executablePath:'/usr/bin/chromium',args:['--no-sandbox']});
const context=await browser.newContext({viewport:{width:1440,height:1000},reducedMotion:'reduce'});
await context.addInitScript(()=>Object.defineProperty(Element.prototype,'requestFullscreen',{value:undefined,configurable:true}));
const page=await context.newPage(),errors=[],checks=[];page.on('pageerror',e=>errors.push(e.message));
const url=process.env.AES_SCROLL_TEST_URL||'http://127.0.0.1:8765/docs/belajar_MLP_Spider.html';
const pause=()=>page.waitForTimeout(180);
async function prepare(chapter,tab){
  await page.evaluate(([chapter,tab])=>{MLPStudy.openChapter(chapter);MLPStudy.setTab(tab);window.scrollTo({top:document.querySelector('#reader').offsetTop-70,behavior:'instant'})},[chapter,tab]);await pause();
}
async function point(selector){
  const pt=await page.locator(selector).evaluate(e=>{
    const rect=e.getBoundingClientRect();let left=Math.max(0,rect.left),right=Math.min(innerWidth,rect.right),top=Math.max(0,rect.top),bottom=Math.min(innerHeight,rect.bottom);
    for(let p=e.parentElement;p;p=p.parentElement){const s=getComputedStyle(p),r=p.getBoundingClientRect();if(/auto|scroll|hidden|clip/.test(s.overflowY)){top=Math.max(top,r.top);bottom=Math.min(bottom,r.bottom)}if(/auto|scroll|hidden|clip/.test(s.overflowX)){left=Math.max(left,r.left);right=Math.min(right,r.right)}}
    if(bottom-top<15||right-left<15)throw Error('Scroll target is outside visible panel');
    return {x:(left+right)/2,y:(top+bottom)/2};
  });await page.mouse.move(pt.x,pt.y);
}
async function measure(selector){return page.locator(selector).evaluate(e=>({top:e.scrollTop,max:e.scrollHeight-e.clientHeight,page:scrollY,left:e.scrollLeft}))}
async function wheel(selector,delta){await point(selector);await page.mouse.wheel(0,delta);await pause()}
function check(name,pass,evidence){checks.push({name,pass,evidence});console.log((pass?'PASS: ':'FAIL: ')+name)}
try{
  await page.goto(url);await page.waitForFunction(()=>window.MLPStudy);
  await prepare(12,'guide');await page.locator('#pane-guide').evaluate(e=>e.scrollTop=0);
  let before=await measure('#pane-guide');await wheel('#pane-guide',250);let after=await measure('#pane-guide');
  check('Wheel enters the guide panel while the outer page stays still',after.top>before.top+100&&Math.abs(after.page-before.page)<2,{before,after});
  await page.locator('#pane-guide').evaluate(e=>e.scrollTop=e.scrollHeight);before=await measure('#pane-guide');await wheel('#pane-guide',260);after=await measure('#pane-guide');
  check('Guide bottom hands continued downward wheel to the page',after.page>before.page+50,{before,after});
  await prepare(12,'guide');await page.locator('#pane-guide').evaluate(e=>e.scrollTop=0);before=await measure('#pane-guide');await wheel('#pane-guide',-240);after=await measure('#pane-guide');
  check('Guide top hands upward wheel to the page',after.page<before.page-50,{before,after});
  await prepare(12,'guide');await page.locator('#pane-guide').evaluate(e=>e.scrollTop=100);await wheel('#pane-guide',100);
  before=await measure('#pane-guide');await page.mouse.move(5,400);await page.mouse.wheel(0,120);await pause();after=await measure('#pane-guide');
  const pageMoved=after.page>before.page+30;before=after;await wheel('#pane-guide',100);after=await measure('#pane-guide');
  check('Moving the pointer page → panel → page → panel transfers scrolling without a mode switch',pageMoved&&after.top>before.top+30&&Math.abs(after.page-before.page)<2,{before,after});
  await prepare(0,'excel');await page.evaluate(()=>MLPStudy.selectExcel(4,'Fusion (S3)'));await page.waitForFunction(()=>document.querySelector('#sheetStatus').textContent.startsWith('Output agregat'));
  await page.locator('#pane-excel').evaluate(e=>e.scrollTop=Math.max(0,(e.scrollHeight-e.clientHeight)/2));
  await page.locator('#tableViewport').evaluate(e=>e.scrollTop=0);before=await measure('#tableViewport');await wheel('#tableViewport',220);after=await measure('#tableViewport');
  check('Wheel enters the tall Excel table',after.top>before.top+70&&Math.abs(after.page-before.page)<2,{before,after});
  await page.locator('#tableViewport').evaluate(e=>e.scrollTop=e.scrollHeight);const paneBefore=await measure('#pane-excel');before=await measure('#tableViewport');await wheel('#tableViewport',220);after=await measure('#tableViewport');const paneAfter=await measure('#pane-excel');
  check('Excel table bottom hands wheel to its parent panel or page',paneAfter.top>paneBefore.top+20||after.page>before.page+20,{paneBefore,paneAfter,before,after});
  await prepare(0,'excel');await page.evaluate(()=>MLPStudy.selectExcel(3,'Perbandingan Model'));await page.waitForFunction(()=>document.querySelector('#sheetStatus').textContent.startsWith('Output agregat'));await page.locator('#pane-excel').evaluate(e=>e.scrollTop=0);
  const shortBefore=await measure('#pane-excel');before=await measure('#tableViewport');await wheel('#tableViewport',200);const shortAfter=await measure('#pane-excel');after=await measure('#tableViewport');
  check('A short Excel table with no vertical scroll cannot trap downward wheel',shortAfter.top>shortBefore.top+20||after.page>before.page+20,{shortBefore,shortAfter,before,after});
  await prepare(11,'code');await page.evaluate(()=>MLPStudy.jumpCode(36,1));await page.locator('#codeScroll').evaluate(e=>e.scrollTop=0);before=await measure('#codeScroll');await wheel('#codeScroll',220);after=await measure('#codeScroll');
  check('Wheel switches from the page into the code panel',after.top>before.top+70&&Math.abs(after.page-before.page)<2,{before,after});
  await page.locator('#codeScroll').evaluate(e=>e.scrollTop=e.scrollHeight);before=await measure('#codeScroll');await wheel('#codeScroll',240);after=await measure('#codeScroll');
  check('Regular code panel releases wheel to the page at its bottom',after.page>before.page+30,{before,after});
  await prepare(0,'guide');await page.locator('#chapterNav').evaluate(e=>e.scrollTop=e.scrollHeight);before=await measure('#chapterNav');await wheel('#chapterNav',220);after=await measure('#chapterNav');
  check('Sidebar boundary releases wheel to the page',after.page>before.page+30,{before,after});
  await prepare(10,'code');await page.locator('#expandCode').click();await page.locator('#canvasModeToggle').click();before=await measure('#codeScroll');await wheel('#codeScroll',260);after=await measure('#codeScroll');
  check('Fullscreen scroll stays in the shared code canvas',after.top>before.top+70&&Math.abs(after.page-before.page)<2,{before,after});
  await page.locator('#codeScroll').evaluate(e=>e.scrollTop=e.scrollHeight);before=await measure('#codeScroll');await wheel('#codeScroll',260);after=await measure('#codeScroll');
  check('Fullscreen boundary never moves the hidden page',Math.abs(after.page-before.page)<2,{before,after});
  await page.keyboard.press('Escape');await pause();check('Closing fullscreen removes the body scroll lock',await page.evaluate(()=>!document.body.classList.contains('reading-code-fullscreen')));
  await page.mouse.move(5,500);before=await measure('html');await page.mouse.wheel(0,-250);await pause();after=await measure('html');check('Page wheel works immediately after closing fullscreen',after.page<before.page-50,{before,after});
  const mobile=await browser.newContext({viewport:{width:390,height:844},isMobile:true,hasTouch:true,reducedMotion:'reduce'}),touch=await mobile.newPage();touch.on('pageerror',e=>errors.push(e.message));await touch.goto(url);
  await touch.evaluate(()=>{MLPStudy.openChapter(12);MLPStudy.setTab('guide');const pane=document.querySelector('#pane-guide');pane.scrollTop=pane.scrollHeight;window.scrollTo({top:pane.getBoundingClientRect().top+scrollY-100,behavior:'instant'})});
  const pt=await touch.locator('#pane-guide').evaluate(e=>{const r=e.getBoundingClientRect();return {x:r.left+r.width/2,y:Math.min(innerHeight-80,r.bottom-30)}}),mobileBefore=await touch.evaluate(()=>scrollY),cdp=await mobile.newCDPSession(touch);
  await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[pt]});
  for(let i=1;i<=6;i++){await cdp.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:pt.x,y:pt.y-i*35}]});await touch.waitForTimeout(20)}
  await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});await touch.waitForTimeout(300);const mobileAfter=await touch.evaluate(()=>scrollY);
  check('Mobile touch swipe continues onto the page at a panel boundary',mobileAfter>mobileBefore+30,{before:mobileBefore,after:mobileAfter});await mobile.close();
  check('No JavaScript errors',errors.length===0,{errors});
  const report={scope:'REAL WHEEL AND MOBILE TOUCH; NO TRAINING',baseline,passed:checks.filter(c=>c.pass).length,failed:checks.filter(c=>!c.pass).length,checks,browser:browser.version()};
  fs.writeFileSync(directory+(baseline?'/baseline.json':'/verified.json'),JSON.stringify(report,null,2));
  if(report.failed&&!baseline)throw Error(report.failed+' scroll checks failed');
}finally{await browser.close()}
