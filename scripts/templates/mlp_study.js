(()=>{'use strict';
const publicTemplate=document.documentElement.outerHTML.replace(/<script id="privatePayload"[^>]*>[\s\S]*?<\/script>/,'<script id="privatePayload" type="application/octet-stream">'+'<'+ '/script>');
const D=JSON.parse(document.getElementById('studyData').textContent), S=D.snapshot;
const $=id=>document.getElementById(id), node=(tag,text,cls)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e};
const cells=new Map(D.cells.map(c=>[c.index,c])), state={chapter:0,tab:'guide',cell:39,note:null,book:3,sheet:'Perbandingan Model',page:0,col:0,full:false,pageSize:25,codeMode:'block',paused:matchMedia('(prefers-reduced-motion: reduce)').matches};
let baselineBundle=null;
let noteFrame=0,followReadingPending=false;
let native=S.native, notebookLoaded=false, originalNotebook=null, resultZip=null, originalResults=null, bookCache=new Map(), sheetCache=new Map(), activeSheet=null, tableTicket=0;
const status=(message,error=false)=>{$('status').textContent=message;$('status').classList.toggle('error',error)}, error=e=>status(e.message||String(e),true);
const num=(v,digits=4)=>v===null||v===undefined?'—':typeof v==='number'?(Number.isInteger(v)?String(v):v.toFixed(digits)):String(v);
const shown=v=>typeof v==='number'?(state.full?String(v):Number.isInteger(v)?String(v):v.toLocaleString('id-ID',{maximumFractionDigits:4})):v===null||v===undefined?'—':typeof v==='boolean'?(v?'Ya':'Tidak'):String(v);
const text=v=>Array.isArray(v)?v.join(''):String(v??'');
function termButton(term){const b=node('button',term,'term-button');b.addEventListener('click',()=>openTerm(term));return b}
function openTerm(term){const v=D.glossary[term];if(!v)return;$('termTitle').textContent=term+' · '+v[0];$('termBody').textContent=v[1];$('termDialog').showModal()}
// Satu kanvas dan satu scroll: catatan menjadi bagian dari ruang kode.
let codeHome=null,codePagePosition=null,codeCanvas=null,codeOwnsFullscreen=false;
function panelScroll(){return ['codeScroll','notePane'].map(id=>({id,top:$(id).scrollTop,left:$(id).scrollLeft}))}
function restorePanelScroll(positions){positions.forEach(p=>{$(p.id).scrollTop=p.top;$(p.id).scrollLeft=p.left})}
function readingAnchor(){
  const top=$('codeScroll').getBoundingClientRect().top;
  // Baris berurutan secara vertikal; pencarian biner menghindari ribuan baca layout per scroll.
  const lines=$('codeLines').querySelectorAll('.code-line');let left=0,right=lines.length;
  while(left<right){const mid=(left+right)>>>1;if(lines[mid].getBoundingClientRect().bottom>top+2)right=mid;else left=mid+1}
  const line=lines[left];
  return line?{cell:Number(line.dataset.cell),line:line.dataset.line,offset:line.getBoundingClientRect().top-top}:null;
}
function restoreReadingAnchor(anchor){
  if(!anchor)return;
  const line=codeLine(anchor.cell??state.cell,anchor.line);
  if(line)$('codeScroll').scrollTop+=line.getBoundingClientRect().top-$('codeScroll').getBoundingClientRect().top-anchor.offset;
}
function canvasAll(){return Boolean(codeCanvas&&state.codeMode==='all')}
function codeLine(cell,line){return $('codeLines').querySelector('[data-cell="'+cell+'"][data-line="'+line+'"]')}
function readingCells(){return canvasAll()?D.cells:[cells.get(state.cell)]}
function updateCanvasNotes(){
  if(!codeCanvas)return;
  const select=$('canvasNoteSelect'),list=readingCells();
  select.replaceChildren();
  let total=0;
  for(const c of list){
    const group=node('optgroup');group.label='Blok '+String(c.ordinal).padStart(2,'0')+' · '+blockTitle(c);
    for(const n of c.notes){const o=node('option','Baris '+n.line+' · '+n.title);o.value=n.id;group.append(o);total++}
    if(group.children.length)select.append(group);
  }
  select.disabled=!total;
  if(!total)select.append(node('option','Blok ini tidak memiliki penanda tambahan.'));
  else select.value=state.note;
  $('canvasModeToggle').textContent=canvasAll()?'Per blok':'Semua blok';
  $('canvasModeToggle').setAttribute('aria-pressed',String(canvasAll()));
  $('canvasModeToggle').title=canvasAll()?'Kembali membaca satu blok':'Baca semua 30 blok berurutan dalam satu scroll';
  $('copyCode').textContent=canvasAll()?'Salin blok aktif':'Salin kode';
}
function setCodeMode(mode){
  if(!codeCanvas||!['all','block'].includes(mode)||mode===state.codeMode)return;
  const anchor=readingAnchor();if(anchor)state.cell=anchor.cell;
  state.codeMode=mode;renderCode();restoreReadingAnchor(anchor);if(mode==='all'&&anchor&&Number(anchor.line)<=3){const block=$('codeLines').querySelector('.canvas-code-block[data-cell="'+anchor.cell+'"]');if(block)$('codeScroll').scrollTop+=block.getBoundingClientRect().top-$('codeScroll').getBoundingClientRect().top-20}followCodeNote();
}
$('canvasModeToggle').addEventListener('click',()=>setCodeMode(canvasAll()?'block':'all'));

function restoreCodeView(positions=panelScroll(),anchor=null){
  if(!codeHome)return;
  const panel=$('pane-code'),wasAll=canvasAll();
  $('codeScroll').append($('codeLines'));
  panel.querySelector('.code-and-notes').append($('notePane'));
  codeCanvas.remove();codeCanvas=null;
  panel.prepend($('codeToolsBody').querySelector('.code-toolbar'),$('codeToolsBody').querySelector('.code-meta'));
  $('codeTools').open=false;$('notePane').tabIndex=0;
  $('notePane').style.removeProperty('top');
  codeHome.replaceWith(panel);codeHome=null;
  panel.setAttribute('aria-labelledby','tab-code');
  document.body.classList.remove('reading-code-fullscreen');
  if(wasAll){if(anchor)state.cell=anchor.cell;renderCode()}
  $('copyCode').textContent='Salin kode';
  $('expandCode').focus({preventScroll:true});
  restorePanelScroll(positions);restoreReadingAnchor(anchor);followCodeNote();
  window.scrollTo({...codePagePosition,behavior:'instant'});codePagePosition=null;
}
function closeCodeView(){
  const positions=panelScroll(),anchor=readingAnchor();
  if(codeHome&&$('termDialog').open)$('termDialog').close();
  if($('codeDialog').open)$('codeDialog').close();
  restoreCodeView(positions,anchor);
  const owned=codeOwnsFullscreen;codeOwnsFullscreen=false;
  if(owned&&document.fullscreenElement===document.documentElement)document.exitFullscreen().catch(()=>{});
}
function openCodeView(){
  const dialog=$('codeDialog'),panel=$('pane-code');
  if(dialog.open||panel.hidden)return;
  const positions=panelScroll(),anchor=readingAnchor();
  codePagePosition={left:scrollX,top:scrollY};
  codeHome=node('div');codeHome.style.height=panel.getBoundingClientRect().height+'px';
  codeHome.setAttribute('aria-hidden','true');panel.before(codeHome);
  $('codeDialogTitle').textContent='Kode + sticky note · '+D.chapters[state.chapter].title;
  const tools=$('codeToolsBody');tools.prepend(panel.querySelector('.code-toolbar'),panel.querySelector('.code-meta'));
  codeCanvas=node('div');codeCanvas.id='codeCanvas';
  codeCanvas.append($('codeLines'),$('notePane'));$('codeScroll').append(codeCanvas);
  $('notePane').tabIndex=-1;
  dialog.append(panel);panel.setAttribute('aria-labelledby','codeDialogTitle');
  document.body.classList.add('reading-code-fullscreen');
  try{
    dialog.showModal();if(state.codeMode==='all')renderCode();updateCanvasNotes();restoreReadingAnchor(anchor);syncCodeNote();
    if(!document.fullscreenElement&&document.documentElement.requestFullscreen){
      codeOwnsFullscreen=true;
      document.documentElement.requestFullscreen().then(()=>{
        if(!dialog.open&&document.fullscreenElement===document.documentElement)document.exitFullscreen().catch(()=>{});
      }).catch(()=>{codeOwnsFullscreen=false});
    }
  }catch(e){restoreCodeView(positions,anchor);error(e)}
}
document.addEventListener('fullscreenchange',()=>{
  if(codeOwnsFullscreen&&!document.fullscreenElement){codeOwnsFullscreen=false;closeCodeView()}
});
$('canvasNoteSelect').addEventListener('change',()=>{selectNote($('canvasNoteSelect').value);$('codeTools').open=false;$('codeScroll').focus({preventScroll:true})});
$('expandCode').addEventListener('click',openCodeView);
$('closeCode').addEventListener('click',closeCodeView);
$('codeDialog').addEventListener('cancel',e=>{e.preventDefault();closeCodeView()});
$('codeDialog').addEventListener('close',()=>{if(!$('codeDialog').open)restoreCodeView()});
$('codeDialog').addEventListener('keydown',e=>{
  if(e.key!=='Tab'||$('termDialog').open)return;
  const focusable=Array.from($('codeDialog').querySelectorAll('button,input,select,textarea,summary,a[href],[tabindex]')).filter(el=>!el.disabled&&el.tabIndex>=0&&el.getClientRects().length);
  const first=focusable[0],last=focusable.at(-1);
  if(e.shiftKey&&document.activeElement===first){e.preventDefault();last?.focus()}
  else if(!e.shiftKey&&document.activeElement===last){e.preventDefault();first?.focus()}
});
// Catatan mengikuti baris yang dibaca, tanpa keluar dari panel kanan.
function syncCodeNote(followReading=false){
  const pane=$('notePane'),code=$('codeScroll');
  if(!pane.getClientRects().length||!code.getClientRects().length)return;
  const bounds=code.getBoundingClientRect(),right=pane.getBoundingClientRect();
  if(codeCanvas){
    pane.classList.remove('note-follows-code');pane.style.removeProperty('--note-offset');
    let line=$('codeLines').querySelector('.selected-line')||codeLine(state.cell,1);
    const paper=pane.querySelector('.sticky-note'),paperBounds=paper?.getBoundingClientRect();
    const lineVisible=line&&line.getBoundingClientRect().bottom>bounds.top+20&&line.getBoundingClientRect().top<bounds.bottom-20;
    // Catatan panjang tetap terbaca sampai selesai sebelum penjelasan berganti.
    if(followReading&&!lineVisible&&(!paperBounds||paperBounds.bottom<bounds.top+36||paperBounds.top>bounds.bottom)){
      const candidates=readingCells().flatMap(c=>c.notes.map(n=>({note:n,line:codeLine(c.index,n.line)}))).filter(v=>{const r=v.line.getBoundingClientRect();return r.top>bounds.top+60&&r.top<bounds.bottom-30});
      candidates.sort((a,b)=>Math.abs(a.line.getBoundingClientRect().top-bounds.top-100)-Math.abs(b.line.getBoundingClientRect().top-bounds.top-100));
      if(candidates.length){selectNote(candidates[0].note.id,false);line=candidates[0].line}
    }
    if(canvasAll()&&followReading){const anchor=readingAnchor();if(anchor&&state.cell!==anchor.cell){state.cell=anchor.cell;$('cellSelect').value=state.cell}}
    const canvasTop=codeCanvas.getBoundingClientRect().top;
    const target=Math.max(80,line?line.getBoundingClientRect().top-canvasTop:80);
    pane.style.top=target+'px';
    const sourceBottom=$('codeLines').getBoundingClientRect().bottom-canvasTop;
    codeCanvas.style.minHeight=Math.max(code.clientHeight,sourceBottom+96,target+pane.offsetHeight+64)+'px';
    return;
  }
  const beside=right.left>=bounds.right-2&&Math.abs(right.top-bounds.top)<2;
  pane.classList.toggle('note-follows-code',beside);
  if(!beside){pane.style.removeProperty('--note-offset');return}
  let line=$('codeLines').querySelector('.selected-line');
  if(!line)return;
  const visible=el=>{const r=el.getBoundingClientRect();return r.bottom>bounds.top+12&&r.top<bounds.bottom-12};
  if(followReading&&!visible(line)){
    const candidates=cells.get(state.cell).notes.map(n=>({note:n,line:$('codeLines').querySelector('[data-line="'+n.line+'"]')})).filter(v=>v.line&&visible(v.line));
    candidates.sort((a,b)=>Math.abs(a.line.getBoundingClientRect().top-bounds.top-80)-Math.abs(b.line.getBoundingClientRect().top-bounds.top-80));
    if(candidates.length){selectNote(candidates[0].note.id,false);line=candidates[0].line}
  }
  const note=pane.querySelector('.sticky-note');if(!note)return;
  const style=getComputedStyle(pane),top=parseFloat(style.paddingTop),bottom=parseFloat(style.paddingBottom);
  const maximum=Math.max(top,pane.clientHeight-note.offsetHeight-bottom);
  const target=Math.max(top,Math.min(maximum,line.getBoundingClientRect().top-bounds.top));
  pane.style.setProperty('--note-offset',Math.max(0,target-top)+'px');
  pane.scrollTop=0;
}
function followCodeNote(reading=false){
  followReadingPending=followReadingPending||reading;
  if(noteFrame)return;
  noteFrame=requestAnimationFrame(()=>{noteFrame=0;const reading=followReadingPending;followReadingPending=false;syncCodeNote(reading)});
}
$('codeScroll').addEventListener('scroll',()=>followCodeNote(true),{passive:true});
const noteResize=new ResizeObserver(()=>followCodeNote());
noteResize.observe($('codeScroll'));noteResize.observe($('notePane'));
function setMotion(){document.body.classList.toggle('paused',state.paused);$('motionToggle').textContent=state.paused?'▶ Aktifkan animasi':'Ⅱ Jeda animasi';$('motionToggle').setAttribute('aria-pressed',String(state.paused))}
function nav(){const q=$('chapterSearch').value.toLowerCase(), root=$('chapterNav');root.replaceChildren();D.chapters.forEach((c,i)=>{if(q&&!(c.title+' '+c.terms.join(' ')+' '+c.read.join(' ')).toLowerCase().includes(q))return;const b=node('button');b.append(node('span',String(i).padStart(2,'0')),node('b',c.title));if(i===state.chapter)b.setAttribute('aria-current','step');b.addEventListener('click',()=>openChapter(i));root.append(b)});if(!root.children.length)root.append(node('p','Bab tidak ditemukan. Coba istilah lain.'))}
function setTab(name){if(name!=='code')closeCodeView();state.tab=name;document.querySelectorAll('[data-tab]').forEach(b=>{const selected=b.dataset.tab===name;b.setAttribute('aria-selected',String(selected));b.tabIndex=selected?0:-1});for(const t of ['guide','code','output','excel'])$('pane-'+t).hidden=t!==name;if(name==='code')renderCode();if(name==='output')renderOutputs();if(name==='excel'){renderExcel();$('pane-excel').scrollTop=0}const selectedTab=$('tab-'+name),tabStrip=selectedTab.parentElement,buttonRect=selectedTab.getBoundingClientRect(),stripRect=tabStrip.getBoundingClientRect();if(buttonRect.left<stripRect.left)tabStrip.scrollLeft+=buttonRect.left-stripRect.left;else if(buttonRect.right>stripRect.right)tabStrip.scrollLeft+=buttonRect.right-stripRect.right;}
function openChapter(i){state.chapter=Math.max(0,Math.min(D.chapters.length-1,i));const c=D.chapters[state.chapter];$('chapterTitle').textContent=c.title;$('chapterGoal').textContent=c.goal;$('chapterEyebrow').textContent=String(state.chapter).padStart(2,'0')+' / '+(state.chapter===0?'PETA HASIL':'ALUR COLAB');$('pageNumber').textContent=String(state.chapter+1).padStart(2,'0')+' / '+D.chapters.length;$('prevChapter').disabled=state.chapter===0;$('nextChapter').disabled=state.chapter===D.chapters.length-1;$('nextReading').disabled=state.chapter===D.chapters.length-1;$('edgePose').setAttribute('href',['#spiderCrawl','#spiderHang','#spiderPeek','#outerPerch'][state.chapter%4]);state.cell=c.cells[0];state.note=null;state.page=state.col=0;if(c.book){state.book=c.book-1;state.sheet=c.sheets[0]||S.books[state.book].sheets[0].name}nav();renderGuide();populateCells();$('chapterTerms').replaceChildren(...c.terms.slice(0,4).map(termButton));for(const t of ['guide','output','excel'])$('pane-'+t).scrollTop=0;setTab(state.tab)}
function metricCards(){const wrap=node('div',undefined,'metric-grid');for(const m of S.metrics){const c=node('div',undefined,'metric-card'+(m.Skenario==='S1'?' best':''));c.append(node('small',m.Skenario+' · RMSE TEST'),node('strong',num(m['RMSE (poin)'])),node('span',m.Skenario==='S1'?'Terkecil pada run tersimpan':'Poin skor · lebih kecil lebih baik'));wrap.append(c)}return wrap}
function metricTable(){const wrap=node('div',undefined,'chart-box');wrap.append(node('span','OUTPUT AKTUAL · 60 TEST','actual-label'),node('h3','Empat cara membaca hasil'));const t=node('table',undefined,'metric-table'), head=node('thead'), tr=node('tr');for(const h of ['Model','RMSE ↓','Pearson','QWK','Gwet AC2'])tr.append(node('th',h));head.append(tr);t.append(head);const body=node('tbody');for(const m of S.metrics){const r=node('tr');for(const v of [m.Skenario,m['RMSE (poin)'],m.Pearson,m.QWK,m['Gwet AC2']])r.append(node('td',num(v)));body.append(r)}t.append(body);wrap.append(t,node('p','RMSE memakai prediksi mentah dalam poin 0–100. QWK/AC2 memakai salinan kategori. AC2 bukan persentase prediksi tepat.'));return wrap}
function readingCards(c){const list=node('div',undefined,'reading-list');c.read.forEach((p,i)=>{const card=node('div',undefined,'reading-card');card.append(node('b','CATATAN '+String(i+1).padStart(2,'0')),node('p',p));list.append(card)});return list}
function originalNotes(){const indices=[[0,1],[2],[4],[8],[13],[17],[19],[23],[25],[27],[29],[33],[37],[40],[42],[44]][state.chapter];const details=node('details',undefined,'original-notes');details.append(node('summary','Catatan asli notebook · source 15bc7bb'));for(const m of D.markdown.filter(m=>indices.includes(m.index)))details.append(node('pre',m.source));return details}
function renderGuide(){const c=D.chapters[state.chapter],root=$('guideContent');root.replaceChildren();root.append(node('p',c.goal,'chapter-summary'));const flow=node('div',undefined,'flow');c.flow.forEach((v,i)=>{const step=node('div',undefined,'flow-step');step.append(node('small','TAHAP '+(i+1)),node('span',v));flow.append(step)});root.append(flow);if(state.chapter===0)root.append(metricCards(),metricTable());if(state.chapter===10)root.append(modelDiagram());root.append(readingCards(c));if(state.chapter===6)root.append(windowDemo());if([0,11,12].includes(state.chapter))root.append(historyChart());if(state.chapter===12)root.append(matrixChart());const terms=node('div',undefined,'term-chips');c.terms.forEach(t=>terms.append(termButton(t)));root.append(terms,originalNotes(),node('p','Source: notebook MLP unggahan, dicocokkan dengan commit 15bc7bb. Tabel dan grafik berlabel aktual berasal dari lima Excel tersimpan.','privacy-note'))}
function modelDiagram(){const box=node('div',undefined,'diagram');box.append(node('h3','Klik lapisan. Lompat ke kode aslinya.'));const specs=[['S1',[['RoBERTa · 768',15],['Dropout → Linear 1',19]]],['S2',[['BGE frozen · 1024',22],['MLP 256 → 1',22]]],['S3',[['RoBERTa → LayerNorm 768',25],['BGE → proyeksi 256',27],['Concat 1024',58],['MLP 256 → 1',61]]]];for(const [name,parts] of specs){const row=node('div',undefined,'diagram-row');row.append(node('b',name));parts.forEach(([label,line],i)=>{if(i)row.append(node('span','→'));const b=node('button',label,name==='S3'?'fusion':'');b.addEventListener('click',()=>jumpCode(30,line));row.append(b)});box.append(row)}return box}
function populateCells(){const sel=$('cellSelect');sel.replaceChildren();const current=new Set(D.chapters[state.chapter].cells);for(const [label,list] of [['Di bab ini',D.cells.filter(c=>current.has(c.index))],['Semua blok kode lainnya',D.cells.filter(c=>!current.has(c.index))]]){const group=node('optgroup');group.label=label;for(const c of list){const o=node('option',String(c.ordinal).padStart(2,'0')+' · '+c.title+' · '+c.source.split('\n').length+' baris');o.value=c.index;group.append(o)}sel.append(group)}sel.value=state.cell}
const BLOCK_TITLES={3:'Environment & folder kerja',5:'Instalasi pustaka AES',6:'Penghubung Python Colab',7:'Setup runtime & dataset',9:'Impor pustaka',10:'Konfigurasi dan nilai bawaan',11:'Tampilan tabel dan istilah',12:'Nama model & parameter eksperimen',14:'Pembacaan dataset & cleaning',15:'Penulisan Excel',16:'Audit jawaban dan skor guru',18:'Split 80/20 & lima fold',20:'Tokenizer & window',21:'Pemuat encoder',22:'Tokenisasi aktual',24:'Embedding RoBERTa',26:'Embedding BGE-M3',28:'Vektor kata & kalimat',30:'Arsitektur S1, S2 & fusion MLP',31:'Model baru & seed',32:'Ringkasan arsitektur',34:'Rumus Gwet AC2',35:'Empat metrik evaluasi',36:'Loop training & checkpoint',38:'Alur CV, refit & test',39:'Evaluasi & ekspor hasil',41:'Prediksi satu jawaban',43:'Vektor setelah training',45:'Laporan Excel ringkas',46:'Unduh hasil'};
function blockTitle(c){return BLOCK_TITLES[c.index]||c.title}
function renderCode(){
  const c=cells.get(state.cell);if(!c)return;
  $('cellSelect').value=state.cell;
  $('codeMeta').textContent=canvasAll()?'Semua 30 blok · '+D.coverage.code_lines+' baris · '+D.coverage.annotations+' catatan': 'Blok '+c.ordinal+'/30 · cell '+(c.index+1)+' di file · '+c.htmlLines.length+' baris · '+c.notes.length+' catatan';
  const lines=$('codeLines');lines.replaceChildren();
  for(const block of readingCells()){
    let parent=lines;
    if(canvasAll()){
      parent=node('section',undefined,'canvas-code-block');parent.dataset.cell=block.index;
      const heading=node('header',undefined,'canvas-block-heading');
      heading.append(node('small','BLOK '+String(block.ordinal).padStart(2,'0')+' / 30'),node('h3',blockTitle(block)),node('p','Cell '+(block.index+1)+' · '+block.htmlLines.length+' baris · '+block.notes.length+' catatan'));
      parent.append(heading);lines.append(parent);
    }
    const marked=new Map(block.notes.map(n=>[n.line,n]));
    for(let i=0;i<block.htmlLines.length;i++){
      const line=node('div',undefined,'code-line'+(marked.has(i+1)?' has-note':''));line.dataset.line=i+1;line.dataset.cell=block.index;
      const n=node(marked.has(i+1)?'button':'span',String(i+1),'line-number'+(marked.has(i+1)?' pin':''));
      if(marked.has(i+1)){n.setAttribute('aria-label','Blok '+block.ordinal+', catatan baris '+(i+1)+': '+marked.get(i+1).title);n.addEventListener('click',()=>selectNote(marked.get(i+1).id))}
      const code=node('code');code.innerHTML=block.htmlLines[i]||' ';line.append(n,code);parent.append(line);
    }
  }
  const chosen=c.notes.find(n=>n.id===state.note)||c.notes.find(n=>n.kind==='fundamental')||c.notes[0];
  if(chosen)selectNote(chosen.id,false);else{state.note=null;$('notePane').replaceChildren(node('p','Kode ini menyiapkan atau menjalankan tahap yang dijelaskan pada nama blok. Seluruh baris tetap dapat dibaca dan disalin.'))}
  $('codeScroll').scrollTop=0;updateCanvasNotes();applyCodeFind();followCodeNote();
}

function selectNote(id,scroll=true){const c=D.cells.find(c=>c.notes.some(n=>n.id===id));if(!c)return;const n=c.notes.find(n=>n.id===id);if(canvasAll()){state.cell=c.index;$('cellSelect').value=state.cell}if(!n)return;state.note=id;document.querySelectorAll('.selected-line').forEach(e=>e.classList.remove('selected-line'));const line=codeLine(c.index,n.line);if(line){line.classList.add('selected-line');if(scroll)$('codeScroll').scrollTop+=line.getBoundingClientRect().top-$('codeScroll').getBoundingClientRect().top-96}const root=$('notePane');root.replaceChildren();const note=node('div',undefined,'sticky-note'),svg=document.createElementNS('http://www.w3.org/2000/svg','svg'),use=document.createElementNS('http://www.w3.org/2000/svg','use');use.setAttribute('href','#spiderHold');svg.setAttribute('aria-hidden','true');svg.append(use);note.append(svg,node('small','BARIS '+n.line+' / '+(n.kind==='fundamental'?'TITIK PENTING':'DEFINISI')),node('h3',n.title),node('p',n.body));if(n.term)note.append(termButton(n.term));root.append(note);const links=node('div',undefined,'note-links');for(const x of c.notes){const b=node('button',undefined,'note-link'+(x.id===id?' active':''));b.append(node('small','BARIS '+x.line),node('span',x.title));b.addEventListener('click',()=>selectNote(x.id));links.append(b)}root.append(links);root.scrollTop=0;updateCanvasNotes();followCodeNote()}
function jumpCode(cell,line){state.cell=cell;setTab('code');const c=cells.get(cell),near=c.notes.reduce((best,n)=>!best||Math.abs(n.line-line)<Math.abs(best.line-line)?n:best,null);if(near)selectNote(near.id);$('tab-code').focus({preventScroll:true})}
function applyCodeFind(){
  const q=$('codeFind').value.toLowerCase();let first=null;
  for(const el of $('codeLines').querySelectorAll('.code-line')){
    const c=cells.get(Number(el.dataset.cell)),hit=q&&c.source.split('\n')[Number(el.dataset.line)-1].toLowerCase().includes(q);
    el.classList.toggle('find-match',Boolean(hit));if(hit&&!first)first=el;
  }
  if(first)$('codeScroll').scrollTop+=first.getBoundingClientRect().top-$('codeScroll').getBoundingClientRect().top-80;
}
function chooseCodeBlock(index){
  state.cell=index;state.note=null;
  if(canvasAll()){
    const c=cells.get(index),chosen=c.notes.find(n=>n.kind==='fundamental')||c.notes[0];
    if(chosen)selectNote(chosen.id,false);else{document.querySelectorAll('.selected-line').forEach(e=>e.classList.remove('selected-line'));$('notePane').replaceChildren(node('p',blockTitle(c)+' · Seluruh kode blok ini dapat dibaca dan disalin.'));updateCanvasNotes()}
    const block=$('codeLines').querySelector('.canvas-code-block[data-cell="'+index+'"]');
    if(block)$('codeScroll').scrollTop+=block.getBoundingClientRect().top-$('codeScroll').getBoundingClientRect().top-20;
    followCodeNote();
  }else renderCode();
}

function safeHtml(raw){const doc=new DOMParser().parseFromString(raw,'text/html'),out=document.createDocumentFragment(),allowed=new Set('div p span table thead tbody tfoot tr td th h1 h2 h3 h4 h5 h6 pre code strong b em i small br hr details summary ul ol li blockquote'.split(' ')),drop=new Set(['SCRIPT','STYLE','IFRAME','OBJECT','EMBED','LINK','META','FORM','INPUT','BUTTON','SVG','MATH']);function walk(x,parent){if(x.nodeType===3){parent.append(document.createTextNode(x.textContent));return}if(x.nodeType!==1||drop.has(x.tagName))return;const tag=x.tagName.toLowerCase();if(tag==='img'){const src=x.getAttribute('src')||'';if(/^data:image\/(png|jpeg|webp);base64,/.test(src)){const img=node('img');img.src=src;img.alt=x.getAttribute('alt')||'Gambar output notebook';parent.append(img)}return}const target=allowed.has(tag)?node(tag):document.createDocumentFragment();if(tag==='td'||tag==='th'){for(const k of ['colspan','rowspan']){const n=Number(x.getAttribute(k));if(n>1&&n<200)target.setAttribute(k,String(n))}}for(const child of x.childNodes)walk(child,target);parent.append(target)}for(const child of doc.body.childNodes)walk(child,out);return out}
function outputNode(output){const wrap=node('div',undefined,'native-inner'),data=output.data||{};if(output.output_type==='stream'){wrap.append(node('pre',text(output.text).replace(/\x1b\[[0-9;]*m/g,'')))}else if(output.output_type==='error'){wrap.append(node('pre',text(output.traceback).replace(/\x1b\[[0-9;]*m/g,'')))}else if(data['text/html']){wrap.append(safeHtml(text(data['text/html'])))}else if(data['image/png']){const img=node('img');img.src='data:image/png;base64,'+text(data['image/png']);img.alt='Output gambar tersimpan';wrap.append(img)}else{wrap.append(node('pre',data['text/plain']?text(data['text/plain']):JSON.stringify(data,null,2)))}if(data['application/javascript']){const more=node('details');more.append(node('summary','JavaScript widget tersimpan · ditampilkan sebagai teks, tidak dijalankan'),node('pre',text(data['application/javascript'])));wrap.append(more)}return wrap}
function renderOutputs(){
  const root=$('outputContent'),chapter=D.chapters[state.chapter];root.replaceChildren();
  root.append(node('p',notebookLoaded?'Output asli notebook sudah dimuat. Log panjang bisa dibuka sesuai kebutuhan; tabel Excel terkait ada di bawah.':'Output agregat aktual tersedia. HTML pribadi memuat seluruh output notebook dan lima Excel.','output-intro'));
  if([0,11,12].includes(state.chapter))root.append(metricTable(),historyChart());
  let count=0;
  for(const id of chapter.cells){
    const outputs=native[String(id)]||[],block=cells.get(id);
    for(let j=0;j<outputs.length;j++){
      const card=node('section',undefined,'native-card'),header=node('header'),label=node('span','Output '+(j+1)+' · '+(block?blockTitle(block):'Cell '+(id+1)));
      const code=node('button','Lihat kode →','mini-button');code.addEventListener('click',()=>jumpCode(id,1));header.append(label,code);card.append(header);
      const output=outputs[j],content=outputNode(output);
      if(output.output_type==='stream'&&text(output.text).length>2000){const detail=node('details');detail.append(node('summary','Buka log lengkap · '+text(output.text).split('\n').length+' baris tersimpan'),content);card.append(detail)}else card.append(content);
      root.append(card);count++;
    }
  }
  if(!count&&!chapter.book){const box=node('div',undefined,'empty-output');box.append(node('strong','Blok ini menyiapkan fungsi atau lingkungan.'),node('p','Tidak ada output tersimpan untuk blok ini. Pemanggilan pada tahap berikutnya menjalankan prosesnya.'));const b=node('button','Lihat kode & catatan →','mini-button');b.addEventListener('click',()=>setTab('code'));box.append(b);root.append(box)}
  if(chapter.book)renderOutputEvidence(root,chapter);
  root.append(readingCards(chapter));
}
function historyChart(){const box=node('div',undefined,'chart-box');box.append(node('span','RIWAYAT AKTUAL · LIMA FOLD','actual-label'),node('h3','Apa yang terjadi setiap epoch?'));const selects=node('div',undefined,'chart-selects');function select(label,options){const l=node('label',label),s=node('select');for(const [v,t] of options){const o=node('option',t);o.value=v;s.append(o)}l.append(s);selects.append(l);return s}const scenario=select('Skenario',[['S1','S1 · RoBERTa'],['S2','S2 · BGE-M3'],['S3','S3 · Fusion MLP']]),fold=select('Fold',[1,2,3,4,5].map(v=>[v,'Fold '+v])),metric=select('Yang dibaca',[['RMSE Validasi (poin)','RMSE validation · poin'],['Loss Training (MSE)','MSE training · target 0–1']]);box.append(selects);const chart=node('div'),caption=node('p');box.append(chart,caption);function draw(){const rows=S.history[scenario.value].filter(r=>Number(r.Fold)===Number(fold.value)&&Number.isFinite(r[metric.value]));if(!rows.length){chart.replaceChildren(node('p','Tidak ada angka pada tahap ini.'));return}const points=rows.map(r=>({x:Number(r.Epoch),y:r[metric.value]})),maxY=Math.max(...points.map(p=>p.y))*1.1||1,maxX=Math.max(...points.map(p=>p.x)),W=660,H=240,L=57,R=18,T=17,B=38;const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('class','history-svg');svg.setAttribute('role','img');svg.setAttribute('aria-label',scenario.value+' fold '+fold.value+' '+metric.value);function el(tag,attrs,txt){const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(txt!==undefined)e.textContent=txt;svg.append(e);return e}const x=v=>L+(v-1)/Math.max(1,maxX-1)*(W-L-R),y=v=>H-B-v/maxY*(H-B-T);for(let i=0;i<=4;i++){const v=maxY*i/4;el('line',{x1:L,y1:y(v),x2:W-R,y2:y(v),stroke:'#dce7f0','stroke-width':1});el('text',{x:L-9,y:y(v)+4,'text-anchor':'end',fill:'#526985','font-size':10},num(v,metric.value.startsWith('Loss')?3:1))}el('polyline',{points:points.map(p=>x(p.x)+','+y(p.y)).join(' '),fill:'none',stroke:scenario.value==='S1'?'#e72036':scenario.value==='S2'?'#009bb8':'#a62a91','stroke-width':3});const selected=S.foldSummary.find(r=>r.Skenario===scenario.value&&Number(r.Fold)===Number(fold.value)),best=Number(selected?.['Epoch Terpilih']);for(const p of points){const dot=el('circle',{cx:x(p.x),cy:y(p.y),r:p.x===best?5.5:3,fill:p.x===best?'#ffe86b':'#14244d',stroke:'#14244d','stroke-width':1.3});const title=document.createElementNS('http://www.w3.org/2000/svg','title');title.textContent='Epoch '+p.x+' · '+p.y;dot.append(title)}for(const v of [1,maxX])el('text',{x:x(v),y:H-16,'text-anchor':'middle',fill:'#526985','font-size':10},'Epoch '+v);chart.replaceChildren(svg);caption.textContent='Titik kuning: epoch '+best+' yang dipilih berdasarkan RMSE validasi. '+(metric.value.startsWith('Loss')?'MSE ini memakai target 0–1; satuannya berbeda dari RMSE poin.':'Baca titik terhadap rekor RMSE terendah, bukan hanya epoch sebelumnya. Refit final dilakukan sesudah CV selesai.')}for(const s of [scenario,fold,metric])s.addEventListener('change',draw);draw();return box}
function matrixChart(){const box=node('div',undefined,'chart-box');box.append(node('h3','Membaca matriks guru × prediksi'));const label=node('label','Skenario '),sel=node('select');for(const s of ['S1','S2','S3']){const o=node('option',s);o.value=s;sel.append(o)}label.append(sel);box.append(label,node('p','Baris = rentang nilai guru. Kolom = rentang prediksi dibulatkan. Isi kotak = jumlah jawaban test. Diagonal berarti masuk rentang yang sama, belum tentu skornya persis sama. Pengelompokan 10 poin ini hanya untuk membaca; QWK/AC2 tetap memakai kategori integer 0–100.'));const root=node('div');box.append(root);function draw(){const m=S.matrices[sel.value],mx=Math.max(...m.flat(),1),grid=node('div',undefined,'matrix');const bands=Array.from({length:11},(_,i)=>i===10?'100':(i*10)+'–'+(i*10+9));grid.append(node('div','Guru ↓'));for(const b of bands)grid.append(node('div',b));m.forEach((row,i)=>{grid.append(node('div',bands[i]));row.forEach((v,j)=>{const cell=node('div',v||'·','heat');cell.style.backgroundColor=`rgba(16,173,205,${.05+.83*v/mx})`;cell.title='Guru '+bands[i]+' · prediksi '+bands[j]+': '+v+' jawaban';grid.append(cell)})});root.replaceChildren(node('div','Prediksi dibulatkan →','matrix-label'),grid)}sel.addEventListener('change',draw);draw();return box}
function windowDemo(){const box=node('div',undefined,'window-demo');box.append(node('span','ILUSTRASI INTERAKTIF · BUKAN TOKENISASI AKTUAL','illustration-label'),node('h3','Kenapa +2 special dan overlap?'));const l=node('label'),input=node('input');input.type='range';input.min='1';input.max='2000';input.value='900';input.step='1';input.setAttribute('aria-label','Jumlah token isi ilustrasi RoBERTa');l.append(node('span','Jumlah token isi RoBERTa: '),node('b','900'));box.append(l,input);const bars=node('div'),p=node('p');box.append(bars,p);function draw(){const n=Number(input.value),starts=[0];while(starts.at(-1)+510<n)starts.push(starts.at(-1)+382);l.querySelector('b').textContent=n;bars.replaceChildren();starts.forEach((start,i)=>{const row=node('div',undefined,'window-row'),b=node('div',undefined,'window-bar'),length=Math.min(510,n-start);b.style.width=(length/510*65)+'%';row.append(node('span','W'+(i+1)),b,node('span',length+' isi + 2 special'));bars.append(row)});p.textContent='Start isi: ['+starts.join(', ')+']. Kapasitas 510, overlap 128, langkah 382. BGE memakai tokenizer berbeda; jumlah tokennya tidak diturunkan dari slider RoBERTa ini.'}input.addEventListener('input',draw);draw();return box}

// Pembaca XLSX kecil: membaca nilai cache XML saja, tidak menjalankan formula/macro.
function xmlDocument(raw){if(/<!DOCTYPE|<!ENTITY/i.test(raw))throw Error('Deklarasi XML eksternal tidak diizinkan.');const d=new DOMParser().parseFromString(raw,'application/xml');if(d.getElementsByTagName('parsererror').length)throw Error('XML workbook tidak dapat dibaca.');return d}
const xmlDecode=s=>s.replace(/&(#x[0-9a-f]+|#\d+|amp|lt|gt|quot|apos);/gi,(all,v)=>v[0]==='#'?String.fromCodePoint(v[1].toLowerCase()==='x'?parseInt(v.slice(2),16):parseInt(v.slice(1),10)):({amp:'&',lt:'<',gt:'>',quot:'"',apos:"'"}[v.toLowerCase()]||all));
const attr=(s,name)=>new RegExp('(?:^|\\s)'+name+'="([^"]*)"').exec(s)?.[1];
function columnIndex(reference){let value=0;for(const c of reference.replace(/\d/g,''))value=value*26+c.charCodeAt(0)-64;return value-1}
async function zipText(zip,path){const f=zip.file(path);if(!f)throw Error('Bagian workbook tidak ditemukan: '+path);if((f._data?.uncompressedSize||0)>100*1024*1024)throw Error('Sheet terlalu besar untuk penjelajah lokal.');return f.async('string')}
async function workbook(bookIndex){if(bookCache.has(bookIndex))return bookCache.get(bookIndex);if(!resultZip)throw Error('Buka ZIP hasil pribadi terlebih dahulu.');const name=S.books[bookIndex].file,entry=resultZip.file(name);if(!entry)throw Error('Workbook tidak ada: '+name);const bytes=await entry.async('uint8array'),book=await workbookFromBytes(bytes);bookCache.set(bookIndex,book);return book}
async function workbookFromBytes(bytes){const zip=await JSZip.loadAsync(bytes),doc=xmlDocument(await zipText(zip,'xl/workbook.xml')),rels=xmlDocument(await zipText(zip,'xl/_rels/workbook.xml.rels')),targets=new Map(Array.from(rels.getElementsByTagNameNS('*','Relationship')).filter(r=>r.getAttribute('TargetMode')!=='External').map(r=>[r.getAttribute('Id'),r.getAttribute('Target')]));const sheets=new Map();for(const s of doc.getElementsByTagNameNS('*','sheet')){const id=s.getAttribute('r:id')||s.getAttributeNS('http://schemas.openxmlformats.org/officeDocument/2006/relationships','id'),target=targets.get(id);if(!target||target.includes('..'))throw Error('Path sheet tidak didukung.');sheets.set(s.getAttribute('name'),target[0]==='/'?target.slice(1):'xl/'+target)}let shared=[];if(zip.file('xl/sharedStrings.xml')){const strings=xmlDocument(await zipText(zip,'xl/sharedStrings.xml'));shared=Array.from(strings.getElementsByTagNameNS('*','si')).map(n=>n.textContent)}return {zip,sheets,shared,bytes}}
async function readWorkbookSheet(bytes,name){const b=await workbookFromBytes(bytes);const path=b.sheets.get(name);if(!path)throw Error('Sheet tidak ditemukan: '+name);return parseSheet(await zipText(b.zip,path),b.shared)}
function parseSheet(raw,shared){if(/<!DOCTYPE|<!ENTITY/i.test(raw))throw Error('XML eksternal tidak diizinkan.');const rows=[];let cellsRead=0;for(const match of raw.matchAll(/<row\b[^>]*>([\s\S]*?)<\/row>/g)){const row=[];for(const cell of match[1].matchAll(/<c\b([^>]*?)(?:\/>|>([\s\S]*?)<\/c>)/g)){if(++cellsRead>5000000)throw Error('Sheet melewati batas pembacaan lokal.');const ref=attr(cell[1],'r');if(!ref)continue;const index=columnIndex(ref);if(index>4095)throw Error('Terlalu banyak kolom.');const type=attr(cell[1],'t'),body=cell[2]||'',v=/<v\b[^>]*>([\s\S]*?)<\/v>/.exec(body)?.[1];let value=null;if(type==='inlineStr'){value=Array.from(body.matchAll(/<t\b[^>]*>([\s\S]*?)<\/t>/g)).map(m=>xmlDecode(m[1])).join('')}else if(type==='s'){value=shared[Number(v)]??''}else if(type==='b'){value=v==='1'}else if(type==='str'||type==='e'||type==='d'){value=xmlDecode(v||'')}else if(v!==undefined&&v!==''){const n=Number(v);value=Number.isFinite(n)?n:xmlDecode(v)}else if(/<f\b/.test(body)){value='='+xmlDecode(/<f\b[^>]*>([\s\S]*?)<\/f>/.exec(body)?.[1]||'')}row[index]=value===''?null:value}rows.push(row)}if(!rows.length)return {headers:[],rows:[]};const headers=rows.shift().map(v=>String(v??''));return {headers,rows}}
async function readSheet(bookIndex,sheetName){const key=bookIndex+'|'+sheetName;if(sheetCache.has(key))return sheetCache.get(key);const book=await workbook(bookIndex),path=book.sheets.get(sheetName);if(!path)throw Error('Sheet tidak ditemukan: '+sheetName);const sheet=parseSheet(await zipText(book.zip,path),book.shared);sheetCache.set(key,sheet);if(sheetCache.size>2)sheetCache.delete(sheetCache.keys().next().value);return sheet}
function publicSheet(bookIndex,name){let records=S.publicSheets?.[bookIndex+'|'+name]||null;if(bookIndex===3&&name==='Perbandingan Model')records=S.metrics;if(bookIndex===3&&name==='Hasil Tiap Fold')records=S.folds;if(bookIndex===4&&name==='Ringkasan Lima Fold')records=S.foldSummary;const historyNames={'RoBERTa (S1)':'S1','BGE-M3 (S2)':'S2','Fusion (S3)':'S3'};if(bookIndex===4&&historyNames[name])records=S.history[historyNames[name]];if(!records?.length)return null;const headers=Object.keys(records[0]);return {headers,rows:records.map(r=>headers.map(h=>r[h]))}}
function sheetExplanation(name){if(name.includes('Token'))return 'Satu baris adalah token, bukan selalu satu kata. ID Token adalah nomor kosakata. Ġ menandai spasi pada tokenizer RoBERTa; tokenizer BGE memiliki aturan sendiri. ID jawaban menghubungkan baris token dengan esai asal.';if(name.includes('Kata Berulang'))return 'Bandingkan kata yang muncul pada dua posisi. ID token sama, embedding awal sama, dan vektor contextual sama adalah tiga pemeriksaan berbeda. Cosine dekat satu belum berarti semua komponennya identik.';if(name.includes('Kalimat'))return 'Satu baris = satu kalimat contoh. Setelah kolom identitas/teks, seluruh koordinat vektor tetap tersedia. Pilih rentang komponen agar tidak menampilkan ratusan kolom sekaligus.';if(name.includes('Kata ')||name.includes('Konteks'))return 'Satu baris = satu kata contoh, yang dapat terdiri dari beberapa subtoken. Tercakup menunjukkan apakah token kata ikut terbaca. Komponen 1, 2, … adalah koordinat embedding, bukan skor rubrik.';if(name.includes('Prediksi')||name.includes('Perbandingan'))return 'Nilai Guru adalah acuan asli. Prediksi adalah skor kontinu dari model. Pembulatan adalah salinan kategori untuk QWK/AC2; RMSE/Pearson tetap memakai nilai mentah. Angka layar dapat diringkas, sedangkan XLSX asli mempertahankan nilai tersimpan.';if(name.includes('Fold')||name.includes('Pelatihan')||name.includes('(S'))return 'Pisahkan validation fold dari refit final. RMSE validasi memakai poin 0–100; loss training memakai target 0–1. Epoch Terpilih berdasarkan RMSE validation, bukan skor test.';if(name.includes('Fusion'))return 'Fusion S3 berisi 1024 fitur: 768 RoBERTa setelah LayerNorm + 256 proyeksi BGE. Penggabungan terjadi pada tingkat jawaban. Vektor ini bukan hasil penjumlahan skor S1 dan S2.';if(name.includes('RoBERTa')||name.includes('BGE'))return 'Satu baris = satu jawaban. Kolom ID adalah penghubung, bukan komponen embedding. RoBERTa mempunyai 768 komponen; BGE 1024. Cabang RoBERTa S1/S3 di run MLP ini ikut fine-tuning; encoder BGE tetap frozen.';return 'Tabel asli dari workbook hasil run MLP. Gunakan nama kolom dan catatan bab untuk menghubungkan tabel dengan kode. Penjelajah hanya membaca nilai tersimpan; tidak menjalankan training atau formula Excel.'}
/*__EXCEL_EXPLORER__*/
function download(bytes,name,type){const url=URL.createObjectURL(new Blob([bytes],{type})),a=node('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
async function digest(bytes){if(!crypto.subtle)throw Error('Browser ini belum mendukung pemeriksaan SHA-256 lokal. Gunakan Chrome/Edge/Firefox terbaru.');return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))).map(x=>x.toString(16).padStart(2,'0')).join('')}
async function loadNotebook(bytes){const n=JSON.parse(new TextDecoder().decode(bytes));if(!Array.isArray(n.cells)||n.cells.length!==47)throw Error('Pilih notebook MLP 47 cell yang mengikuti commit 15bc7bb.');for(const c of D.cells){const actual=n.cells[c.index],src=Array.isArray(actual.source)?actual.source.join(''):actual.source;if(actual.cell_type!=='code'||src!==c.source)throw Error('Kode notebook berbeda dari run MLP buku ini. Pilih AES_300_S3_MLP (1).ipynb yang sesuai.')}native=Object.fromEntries(n.cells.map((c,i)=>[String(i),c.outputs||[]]));notebookLoaded=true;originalNotebook=bytes;updateFileStatus();if(state.tab==='output')renderOutputs();window.dispatchEvent(new Event('mlp-private-ready'))}
async function loadResults(bytes){if(bytes.byteLength>64*1024*1024)throw Error('ZIP lebih besar dari batas 64 MiB penjelajah.');if(await digest(bytes)!==S.zip_sha256)throw Error('ZIP ini berbeda dari hasil run MLP yang dijelaskan. Pilih Hasil_AES_Ringkas (2).zip yang sesuai agar angka dan penjelasan tetap cocok.');const zip=await JSZip.loadAsync(bytes);if(!S.books.every(b=>zip.file(b.file)))throw Error('Lima workbook belum lengkap.');resultZip=zip;originalResults=bytes;bookCache=new Map();sheetCache=new Map();updateFileStatus();if(state.tab==='excel')renderExcel();window.dispatchEvent(new Event('mlp-private-ready'))}
function updateFileStatus(){$('savePrivateHtml').disabled=!(notebookLoaded&&resultZip);$('savePrivateHtml').hidden=$('savePrivateHtml').disabled;$('fileStatus').textContent=notebookLoaded&&resultZip?'Buku pribadi lengkap siap dibaca.':notebookLoaded?'Seluruh output notebook sudah dimuat.':resultZip?'Lima workbook pribadi sudah dimuat.':'Ringkasan aktual sudah siap.';$('fileDetail').textContent=notebookLoaded&&resultZip?S.saved_outputs+' output tersimpan + seluruh sheet dari lima Excel. Semua dibaca lokal; tidak dikirim ke server.':resultZip?'Seluruh nilai Excel tersedia. Pilih notebook untuk menambahkan output Colab aslinya.':'Buka file pribadi untuk semua output dan tabel individual. Pemrosesan berlangsung lokal di browser.'}
$('savePrivateHtml').addEventListener('click',async()=>{try{if(!originalNotebook||!originalResults)return;status('Menyusun satu HTML pribadi lengkap di browser…');const pack=new JSZip();pack.file('notebook.ipynb',originalNotebook,{compression:'DEFLATE'});pack.file('results.zip',originalResults,{compression:'STORE'});const baseline=window.MLPCompare?.getBaselineBytes()||baselineBundle;if(baseline)pack.file('baseline.ipynb',baseline,{compression:'DEFLATE'});const evaluation=window.MLPCompare?.getBaselineEvalBytes();if(evaluation)pack.file('baseline_eval.xlsx',evaluation,{compression:'STORE'});const frozen=window.MLPFrozen?.getBytes();if(frozen)pack.file('frozen_results.zip',frozen,{compression:'STORE'});const payload=await pack.generateAsync({type:'base64',compression:'DEFLATE',compressionOptions:{level:6}});const marker=/<script id="privatePayload"[^>]*>[\s\S]*?<\/script>/;const embedded='<script id="privatePayload" type="application/octet-stream">'+payload+'<'+ '/script>';const complete='<!doctype html>\n'+publicTemplate.replace(marker,()=>embedded);download(complete,'belajar_MLP_Spider_Privat.html','text/html;charset=utf-8');status('HTML pribadi lengkap siap disimpan. File berisi seluruh output dan tabel individual; simpan sebagai berkas pribadi.')}catch(e){error(e)}});
$('motionToggle').addEventListener('click',()=>{state.paused=!state.paused;setMotion()});$('startBook').addEventListener('click',()=>{$('reader').scrollIntoView({behavior:state.paused?'instant':'smooth'})});$('jumpResults').addEventListener('click',()=>{openChapter(12);setTab('guide');$('reader').scrollIntoView({behavior:state.paused?'instant':'smooth'})});$('chapterSearch').addEventListener('input',nav);$('navToggle').addEventListener('click',()=>{const expanded=$('navToggle').getAttribute('aria-expanded')==='true';$('navToggle').setAttribute('aria-expanded',String(!expanded));$('chapterNav').hidden=expanded});$('prevChapter').addEventListener('click',()=>openChapter(state.chapter-1));$('nextChapter').addEventListener('click',()=>openChapter(state.chapter+1));$('nextReading').addEventListener('click',()=>{state.tab='guide';openChapter(state.chapter+1)});document.querySelectorAll('[data-tab]').forEach(b=>{b.addEventListener('click',()=>setTab(b.dataset.tab));b.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();const tabs=Array.from(document.querySelectorAll('[data-tab]')),i=tabs.indexOf(b),next=e.key==='Home'?tabs[0]:e.key==='End'?tabs.at(-1):tabs[(i+(e.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length];setTab(next.dataset.tab);next.focus()})});$('cellSelect').addEventListener('change',()=>{chooseCodeBlock(Number($('cellSelect').value))});$('codeFind').addEventListener('input',applyCodeFind);$('copyCode').addEventListener('click',async()=>{const value=cells.get(state.cell).source;try{await navigator.clipboard.writeText(value)}catch(e){const t=node('textarea');t.value=value;t.style.cssText='position:fixed;left:0;top:0;width:1px;height:1px;opacity:0';const focus=document.activeElement;$('pane-code').append(t);t.select();document.execCommand('copy');t.remove();focus?.focus({preventScroll:true})}status('Kode asli disalin, tanpa sticky note atau nomor baris.')});$('bookSelect').addEventListener('change',()=>{state.book=Number($('bookSelect').value);state.sheet=EXCEL_ROUTES.find(r=>r.book===state.book).sheet;renderExcel()});$('sheetSelect').addEventListener('change',()=>{state.sheet=$('sheetSelect').value;renderExcel()});$('rowSearch').addEventListener('input',()=>{state.page=0;renderTable()});$('componentStart').addEventListener('change',()=>{state.col=Math.max(0,Number($('componentStart').value)-1);renderTable()});$('rowsPerPage').addEventListener('change',()=>{state.pageSize=Number($('rowsPerPage').value);state.page=0;renderTable()});$('downloadResults').addEventListener('click',()=>{if(originalResults)download(originalResults,'Hasil_AES_Ringkas_MLP.zip','application/zip')});$('excelCode').addEventListener('click',()=>openExcelSource('code'));$('excelOutput').addEventListener('click',()=>openExcelSource('output'));$('prevRows').addEventListener('click',()=>{state.page--;renderTable()});$('nextRows').addEventListener('click',()=>{state.page++;renderTable()});$('prevCols').addEventListener('click',()=>{state.col-=8;renderTable()});$('nextCols').addEventListener('click',()=>{state.col+=8;renderTable()});$('precisionToggle').addEventListener('click',()=>{state.full=!state.full;$('precisionToggle').textContent=state.full?'Tampilkan angka ringkas':'Tampilkan angka lengkap';$('precisionToggle').setAttribute('aria-pressed',String(state.full));renderTable()});$('downloadBook').addEventListener('click',async()=>{try{const b=await workbook(state.book);download(b.bytes,S.books[state.book].file,'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}catch(e){error(e)}});for(const id of ['closeTerm','closeTermBottom'])$(id).addEventListener('click',()=>$('termDialog').close());$('termDialog').addEventListener('click',e=>{if(e.target===$('termDialog')){const b=$('termDialog').getBoundingClientRect();if(e.clientX<b.left||e.clientX>b.right||e.clientY<b.top||e.clientY>b.bottom)$('termDialog').close()}});$('notebookFile').addEventListener('change',async()=>{try{const f=$('notebookFile').files[0];if(!f)return;if(f.size>8*1024*1024)throw Error('Notebook melewati batas 8 MiB.');await loadNotebook(await f.arrayBuffer());status('Notebook dibuka lokal. Seluruh output tersimpan siap dibaca.')}catch(e){error(e)}});$('resultsFile').addEventListener('change',async()=>{try{const f=$('resultsFile').files[0];if(!f)return;status('Membuka ZIP hasil secara lokal…');await loadResults(await f.arrayBuffer());status('Lima workbook siap dijelajahi.')}catch(e){error(e)}});
window.MLPStudy={data:D,setCodeMode,openChapter,setTab,jumpCode,readSheet,selectExcel:(book,sheet)=>{state.book=book;state.sheet=sheet;setTab('excel')},getState:()=>({...state,notebookLoaded,resultsLoaded:Boolean(resultZip),nativeOutputs:Object.values(native).reduce((n,o)=>n+o.length,0)}),loadNotebook,loadResults,readWorkbookSheet};
$('pinCount').textContent=D.coverage.annotations;setMotion();openChapter(0);updateFileStatus();
(async()=>{try{const element=$('privatePayload'),payload=element.textContent.trim();if(!payload)return;status('Membuka buku pribadi lengkap…');const binary=atob(payload),bytes=Uint8Array.from(binary,c=>c.charCodeAt(0));element.remove();const pack=await JSZip.loadAsync(bytes);await loadNotebook(await pack.file('notebook.ipynb').async('uint8array'));await loadResults(await pack.file('results.zip').async('uint8array'));if(pack.file('baseline.ipynb')){baselineBundle=await pack.file('baseline.ipynb').async('uint8array');await window.MLPCompare.loadBaselineNotebook(baselineBundle)}if(pack.file('baseline_eval.xlsx'))await window.MLPCompare.loadBaselineWorkbook(await pack.file('baseline_eval.xlsx').async('uint8array'));if(pack.file('frozen_results.zip'))await window.MLPFrozen.loadResults(await pack.file('frozen_results.zip').async('uint8array'));status('Siap: seluruh kode, output notebook dan lima Excel sudah dimuat.')}catch(e){error(e)}})();
})();
