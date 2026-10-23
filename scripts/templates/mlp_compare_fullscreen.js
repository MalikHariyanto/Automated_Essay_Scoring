// Perbandingan memakai ruang layar penuh tanpa membangun ulang tabel atau data.
(()=>{'use strict';
const specs=[
 ['comparison','Perbandingan eksperimen','comparisonBody',()=>MLPCompare.getState().scenario,s=>MLPCompare.setScenario(s)],
 ['metricsRoom','Tabel hasil & arti angka','metricsContent',()=>MLPResults.getState().scenario,s=>MLPResults.setScenario(s)],
 ['frozenRoom','MLP vs double frozen','frozenBody',()=>MLPFrozen.getState().scenario,s=>MLPFrozen.setScenario(s)],
 ['loraRoom','E5 · LoRA + MLP','loraBody',()=>MLPLora.getState().scenario,s=>MLPLora.setScenario(s)]
];
const dialog=document.createElement('dialog');dialog.id='comparisonViewDialog';dialog.setAttribute('aria-label','Perbandingan layar penuh');
const bar=document.createElement('div');bar.className='comparison-view-bar';
const caption=document.createElement('span');caption.id='comparisonViewTitle';dialog.setAttribute('aria-labelledby',caption.id);
const label=document.createElement('label');label.textContent='Skenario ';const choice=document.createElement('select');choice.setAttribute('aria-label','Skenario layar penuh');
for(const s of ['S1','S2','S3']){const o=document.createElement('option');o.value=s;o.textContent=s;choice.append(o)}label.append(choice);
const closeButton=document.createElement('button');closeButton.type='button';closeButton.className='mini-button';closeButton.textContent='✕ Kembali';closeButton.setAttribute('aria-label','Keluar dari perbandingan layar penuh');
bar.append(caption,label,closeButton);dialog.append(bar);document.body.append(dialog);
let active=null,ownsFullscreen=false;
function close(){
 if(!active)return;const saved=active;active=null;const body=document.getElementById(saved.spec[2]);const position={top:body.scrollTop,left:body.scrollLeft};
 if(dialog.open)dialog.close();saved.room.classList.remove('comparison-expanded');saved.placeholder.replaceWith(saved.room);document.body.classList.remove('reading-comparison-fullscreen');
 body.scrollTop=position.top;body.scrollLeft=position.left;window.scrollTo({...saved.page,behavior:'instant'});saved.opener.focus({preventScroll:true});
 const owned=ownsFullscreen;ownsFullscreen=false;if(owned&&document.fullscreenElement===document.documentElement)document.exitFullscreen().then(()=>{if(!active&&!saved.room.hidden)window.scrollTo({...saved.page,behavior:'instant'})}).catch(()=>{});
}
function open(id){
 const spec=specs.find(s=>s[0]===id),room=document.getElementById(id);if(!spec||!room||room.hidden||active||document.querySelector('#codeDialog[open]'))return;
 const page={left:scrollX,top:scrollY};
 const body=document.getElementById(spec[2]),position={top:body.scrollTop,left:body.scrollLeft},placeholder=document.createElement('div');placeholder.style.height=room.getBoundingClientRect().height+'px';placeholder.setAttribute('aria-hidden','true');room.before(placeholder);
 active={spec,room,placeholder,page,opener:room.querySelector('[data-expand-comparison]')};caption.textContent=spec[1];choice.value=spec[3]();room.classList.add('comparison-expanded');dialog.append(room);document.body.classList.add('reading-comparison-fullscreen');
 const session=active;
 try{dialog.showModal();body.scrollTop=position.top;body.scrollLeft=position.left;closeButton.focus({preventScroll:true});
  if(!document.fullscreenElement&&document.documentElement.requestFullscreen){ownsFullscreen=true;document.documentElement.requestFullscreen().then(()=>{if(!active&&document.fullscreenElement===document.documentElement)document.exitFullscreen().catch(()=>{})}).catch(()=>{if(active===session)ownsFullscreen=false})}
 }catch(error){close();throw error}
}
choice.addEventListener('change',()=>{if(active)active.spec[4](choice.value)});closeButton.addEventListener('click',close);
dialog.addEventListener('cancel',event=>{event.preventDefault();close()});dialog.addEventListener('close',()=>{if(active)close()});
document.addEventListener('fullscreenchange',()=>{if(ownsFullscreen&&!document.fullscreenElement){ownsFullscreen=false;close()}});
for(const spec of specs){const room=document.getElementById(spec[0]),entry=document.createElement('div'),button=document.createElement('button');entry.className='comparison-view-actions';button.type='button';button.className='mini-button';button.dataset.expandComparison=spec[0];button.textContent='⛶ Layar penuh';button.setAttribute('aria-haspopup','dialog');button.setAttribute('aria-controls',dialog.id);button.setAttribute('aria-label','Layar penuh '+spec[1]);button.addEventListener('click',()=>open(spec[0]));entry.append(button);
 const heading=room.querySelector('.comparison-heading,.metrics-heading,.frozen-heading');heading.after(entry);
}
window.MLPCompareView={open,close,getState:()=>({room:active?.spec[0]||null,nativeFullscreen:Boolean(document.fullscreenElement),open:dialog.open})};
})();
