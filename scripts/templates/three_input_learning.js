// Semua catatan sebidang dengan kode: tinggi ruas mengikuti catatan atau kode.
function prepareLearningUI(){
 document.body.classList.toggle('learning-margin',state.notesLayout==='margin');
 if($('marginNotesToggle'))return;
 const toolbar=$('pane-code').querySelector('.code-toolbar')||$('codeToolsBody').querySelector('.code-toolbar');
 const toggle=node('button','Satu catatan','mini-button');toggle.id='marginNotesToggle';toggle.title='Ganti antara semua sticky note sejajar dan satu catatan aktif';toggle.setAttribute('aria-pressed','true');
 toggle.addEventListener('click',()=>{const anchor=readingAnchor();state.notesLayout=state.notesLayout==='margin'?'single':'margin';document.body.classList.toggle('learning-margin',state.notesLayout==='margin');toggle.textContent=state.notesLayout==='margin'?'Satu catatan':'Semua catatan sejajar';toggle.setAttribute('aria-pressed',String(state.notesLayout==='margin'));$('notePane').style.removeProperty('top');if(codeCanvas)codeCanvas.style.removeProperty('min-height');renderCode();restoreReadingAnchor(anchor);followCodeNote()});
 const mapping=node('button','Padanan screenshot → kode','mini-button');mapping.id='learningMapButton';mapping.addEventListener('click',openLearningMap);toolbar.append(toggle,mapping);
}
function learningMapCards(){
 const root=node('section',undefined,'learning-map');root.append(node('h3','Bagian kode teman lo ada di mana di sini?'),node('p','Padanan peran sesuai source eksperimen ini. Nama fungsi dan metode dapat berbeda dari screenshot.'));
 for(const m of D.screenshotMappings){const card=node('article',undefined,'learning-map-card');card.append(node('small','DARI SCREENSHOT'),node('h4',m.title),node('code',m.example),node('p',m.explanation));const b=node('button','Buka blok '+m.ordinal+' · baris '+m.line,'mini-button');b.dataset.cell=m.cell;b.dataset.line=m.line;b.addEventListener('click',()=>{$('termDialog').close();jumpCode(m.cell,m.line)});card.append(b);root.append(card)}return root;
}
function openLearningMap(){
 $('termTitle').textContent='Padanan screenshot → '+D.studySource.experiment;
 $('termBody').replaceChildren(learningMapCards());$('termDialog').classList.add('learning-map-dialog');if(!$('termDialog').open)$('termDialog').showModal();
}
function marginCard(n,block){
 const card=node('aside',undefined,'sticky-note margin-note');card.dataset.noteId=n.id;card.dataset.cell=block.index;card.dataset.line=n.line;
 card.append(node('small','BARIS '+n.line+' · '+(n.kind==='fundamental'?'BAGIAN PENTING':'DEFINISI')),node('h3',n.title));
 if(n.short)card.append(node('p',n.short,'margin-short'));card.append(node('p',n.body));
 if(n.screenshot)card.append(node('p','Padanan screenshot: '+n.screenshot,'margin-screenshot'));
 if(n.anchor.startsWith('def ')&&!/pemanggilan|menyiapkan fungsi|dipanggil/.test(n.body))card.append(node('p','def menyiapkan fungsi; proses berjalan saat fungsi dipanggil.','margin-context'));
 const anchor=node('code',n.anchor,'margin-anchor'),copy=node('button','Salin kata pencarian','margin-copy');copy.type='button';copy.setAttribute('aria-label','Salin anchor '+n.title);copy.addEventListener('click',async()=>{try{await navigator.clipboard.writeText(n.anchor);copy.textContent='Tersalin · Ctrl+F di Colab'}catch{const area=node('textarea');area.value=n.anchor;area.style.cssText='position:fixed;width:1px;height:1px;opacity:0';card.append(area);area.select();const copied=document.execCommand('copy');area.remove();copy.textContent=copied?'Tersalin · Ctrl+F di Colab':'Pilih teks anchor, lalu salin manual'}});card.append(anchor,copy);return card;
}
function renderMarginCode(){
 const c=cells.get(state.cell),lines=$('codeLines');if(!c)return;$('cellSelect').value=state.cell;
 $('codeMeta').textContent=(canvasAll()?'Semua '+D.cells.length+' blok · '+D.coverage.code_lines+' baris': 'Blok '+c.ordinal+'/'+D.cells.length+' · cell '+(c.index+1)+' · '+c.htmlLines.length+' baris')+' · catatan sejajar · kode asli tidak diubah';
 lines.replaceChildren();
 for(const block of readingCells()){
  let parent=lines;
  if(canvasAll()){parent=node('section',undefined,'canvas-code-block');parent.dataset.cell=block.index;const heading=node('header',undefined,'canvas-block-heading');heading.append(node('small','BLOK '+String(block.ordinal).padStart(2,'0')+' / '+D.cells.length),node('h3',blockTitle(block)),node('p','Cell '+(block.index+1)+' · '+block.htmlLines.length+' baris · '+block.notes.length+' catatan'));parent.append(heading);lines.append(parent)}
  const groups=new Map();for(const n of block.notes){if(!groups.has(n.line))groups.set(n.line,[]);groups.get(n.line).push(n)}
  if(!groups.has(1))groups.set(1,[]);const starts=[...groups.keys()].sort((a,b)=>a-b);
  for(let j=0;j<starts.length;j++){
   const start=starts[j],end=j+1<starts.length?starts[j+1]-1:block.htmlLines.length,segment=node('div',undefined,'annotation-segment'),code=node('div',undefined,'annotation-code');segment.dataset.cell=block.index;segment.dataset.start=start;
   for(let i=start;i<=end;i++){const row=node('div',undefined,'code-line'+(groups.get(i)?.length?' has-note':''));row.dataset.cell=block.index;row.dataset.line=i;const n=node(groups.get(i)?.length?'button':'span',String(i),'line-number'+(groups.get(i)?.length?' pin':''));if(groups.get(i)?.length){n.setAttribute('aria-label','Blok '+block.ordinal+', catatan baris '+i);n.addEventListener('click',()=>selectNote(groups.get(i)[0].id))}const text=node('code');text.innerHTML=block.htmlLines[i-1]||' ';row.append(n,text);code.append(row)}
   segment.append(code);if(groups.get(start).length){const notes=node('div',undefined,'annotation-notes');for(const n of groups.get(start))notes.append(marginCard(n,block));segment.append(notes)}parent.append(segment);
  }
 }
 $('notePane').replaceChildren();const chosen=c.notes.find(n=>n.id===state.note)||c.notes.find(n=>n.kind==='fundamental')||c.notes[0];if(chosen)selectMarginNote(chosen.id,false);$('codeScroll').scrollTop=0;updateCanvasNotes();applyCodeFind();
}
function selectMarginNote(id,scroll=true){
 const c=D.cells.find(c=>c.notes.some(n=>n.id===id));if(!c)return;const n=c.notes.find(n=>n.id===id);state.note=id;if(canvasAll()){state.cell=c.index;$('cellSelect').value=c.index}
 $('codeLines').querySelectorAll('.selected-line,.margin-selected').forEach(e=>e.classList.remove('selected-line','margin-selected'));const line=codeLine(c.index,n.line),card=$('codeLines').querySelector('[data-note-id="'+id+'"]');line?.classList.add('selected-line');card?.classList.add('margin-selected');
 if(scroll&&line)$('codeScroll').scrollTop+=line.getBoundingClientRect().top-$('codeScroll').getBoundingClientRect().top-40;updateCanvasNotes();
}
