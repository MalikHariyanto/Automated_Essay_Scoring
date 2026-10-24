// Sumber mendukung mekanisme; pengaturan eksperimen tetap diberi batas klaim.
(()=>{'use strict';const A=window.MLPStudy,S=A.data.methodSources;
const el=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n};
function link(label,url){const a=el('a',label,'source-link');a.href=url;a.target='_blank';a.rel='noopener noreferrer';return a}
function render(root,id){const sources=id?S.experiments.filter(e=>e.id===id):[S.experiments.find(e=>e.id==='E6'),...S.experiments.filter(e=>e.id!=='E6')];
 if(!id){const intro=el('aside',undefined,'dashboard-note');intro.append(el('b','Paper → mekanisme → adaptasi → hasil'),el('p','Setiap kartu menghubungkan source notebook, sumber penelitian, bagian yang diterapkan, dan batasnya. Angka LR, dimensi MLP serta patience merupakan pengaturan yang diuji; tidak disajikan sebagai resep optimal dari paper.'));root.append(intro)}
 for(const exp of sources){const card=el(id?'article':'details',undefined,'method-source-card');card.dataset.sourceExperiment=exp.id;if(!id){card.open=exp.id==='E6';card.append(el('summary',exp.id+' · '+exp.name+(exp.id==='E6'?' · ACUAN MEJA BELAJAR':'')))}card.append(el('small',exp.id+' / JEJAK IMPLEMENTASI'),el('h3',exp.name),link('Source notebook '+exp.id+' ↗',exp.notebook),el('p',exp.source_status,'source-status'),el('p',exp.choice,'source-adaptation'));
  for(const key of exp.papers){const p=S.papers[key],row=el('section',undefined,'method-paper');row.append(el('h4',p.title));const links=el('div',undefined,'source-links');links.append(link('Paper ↗',p.paper));if(p.code)links.append(link('Implementasi penulis / resmi ↗',p.code));row.append(links,el('p','Yang digunakan: '+p.supports),el('p','Batas penerapan: '+p.limit,'source-limit'));const detail=el('details');detail.append(el('summary','Bukti yang diperiksa'),el('p',p.evidence));row.append(detail);card.append(row)}root.append(card)
 }}
window.MLPSources={render,link,getExperiment:id=>S.experiments.find(e=>e.id===id)};
})();
