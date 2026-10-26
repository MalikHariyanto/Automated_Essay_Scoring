// Jawaban lengkap E6; konteks privat terikat pada ID/teks/label yang diverifikasi.
(()=>{'use strict';const A=window.MLPStudy,state={query:'',openAll:false};let promise=null,ticket=0;
const el=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n};
const number=n=>n.toLocaleString('id-ID',{maximumFractionDigits:3});
function note(root,text){root.append(el('p',text,'answer-provenance'))}
async function rows(){
 if(!promise){
  promise=(async()=>{
   const [sheet,predictions]=await Promise.all([A.readSheet(0,'Data dan Nilai Guru'),MLPDashboard.predictionRows('E6')]);
   const all=new Map(sheet.rows.map(r=>{const x=Object.fromEntries(sheet.headers.map((h,i)=>[h,r[i]]));return [String(x['ID Jawaban']),x]}));
   const context=A.getAnswerContext();
   if(!context)throw Error('Konteks jawaban tersedia pada buku lengkap terbaru.');
   return predictions.map(p=>{
    const id=String(p['ID Jawaban']),answer=all.get(id),prompt=context.prompts[context.links[id]];
    if(!answer||!prompt||answer['Nilai Guru']!==p['Nilai Guru'])throw Error('ID, konteks atau label E6 tidak cocok.');
    return {id,teacher:p['Nilai Guru'],answer:answer['Jawaban Siswa'],question:prompt.question,reference:prompt.reference,
     predictions:Object.fromEntries(['S1','S2','S3'].map(s=>[s,p['Prediksi '+s]]))};
   });
  })().catch(e=>{promise=null;throw e});
 }
 return promise;
}

function texts(row){const content=el('div',undefined,'answer-texts');for(const [label,value,cls]of [['Soal',row.question,'question'],['Kunci jawaban / jawaban_referensi',row.reference,'reference'],['Jawaban siswa / response_text',row.answer,'response']]){const section=el('section',undefined,'answer-'+cls);section.append(el('h4',label),el('p',value,'student-answer'));content.append(section)}return content}
function scoreStrip(row){const strip=el('div',undefined,'answer-score-strip'),minimum=Math.min(...Object.values(row.predictions).map(v=>Math.abs(v-row.teacher)));for(const [s,value]of Object.entries(row.predictions)){const closest=Math.abs(Math.abs(value-row.teacher)-minimum)<=1e-5;const score=el('span',undefined,'answer-model-score'+(closest?' closest':''));score.dataset.scenario=s;score.dataset.prediction=value;score.append(el('small',s),el('b',number(value)),el('small','Selisih '+number(Math.abs(value-row.teacher))));score.title='Prediksi mentah: '+value;if(closest)score.append(el('em','Terdekat di E6'));strip.append(score)}const teacher=el('span',undefined,'answer-teacher');teacher.append(el('small','Nilai guru / rater'),el('strong',String(row.teacher)));strip.append(teacher);return strip}
async function render(root){const serial=++ticket;note(root,'Konteks ditampilkan sebagai rujukan. Model E6 lama membaca jawaban saja; tampilan soal/kunci ini tidak mengubah metode atau prediksinya.');const area=el('div');root.append(area);
 try{const data=await rows();if(serial!==ticket||!root.isConnected||MLPPredictions.getState().view!=='answers')return;const toolbar=el('div',undefined,'dashboard-tools'),label=el('label','Cari ID atau isi jawaban E6'),search=el('input');search.type='search';search.value=state.query;search.placeholder='Semua 60 jawaban';search.setAttribute('aria-label','Cari ID atau isi jawaban E6');label.append(search);const expand=el('button',state.openAll?'Tutup semua teks':'Buka semua teks','mini-button');toolbar.append(label,expand);area.append(toolbar);const count=el('p',undefined,'prediction-count');count.setAttribute('role','status');const cards=el('div',undefined,'answer-cards');area.append(count,cards);
 function draw(){cards.replaceChildren();const query=state.query.toLocaleLowerCase('id-ID'),filtered=data.filter(r=>[r.id,r.answer,r.question,r.reference].some(v=>v.toLocaleLowerCase('id-ID').includes(query)));count.textContent=filtered.length+' dari 60 jawaban ditampilkan. Sorotan hanya membandingkan S1/S2/S3 E6 pada jawaban yang sama.';for(const row of filtered){const card=el('details',undefined,'answer-card');card.dataset.answer=row.id;card.open=state.openAll;const summary=el('summary'),identity=el('span',row.id,'answer-id');summary.append(identity,scoreStrip(row));card.append(summary,texts(row));cards.append(card)}}
 search.addEventListener('input',()=>{state.query=search.value;draw()});expand.addEventListener('click',()=>{state.openAll=!state.openAll;expand.textContent=state.openAll?'Tutup semua teks':'Buka semua teks';cards.querySelectorAll('details').forEach(d=>d.open=state.openAll)});draw();
 }catch(e){area.replaceChildren(el('p','Jawaban lengkap tersedia di buku privat terbaru. '+e.message,'dashboard-note'))}}
async function openDialog(row,runs,scenario){let dialog=document.getElementById('predictionAnswerDialog');if(!dialog){dialog=el('dialog',undefined,'prediction-answer-dialog');dialog.id='predictionAnswerDialog';dialog.setAttribute('aria-labelledby','predictionAnswerTitle');document.body.append(dialog);dialog.addEventListener('click',e=>{if(e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close()}})}dialog.replaceChildren();const close=el('button','Tutup ×','mini-button');close.addEventListener('click',()=>dialog.close());dialog.append(close,el('h3','Jawaban '+row.id,'dashboard-title'));dialog.querySelector('h3').id='predictionAnswerTitle';dialog.append(el('p','Nilai guru / rater: '+row.teacher+' · '+scenario));dialog.showModal();try{const answer=(await rows()).find(x=>x.id===row.id);if(!answer)throw Error('ID jawaban tidak ditemukan.');dialog.append(texts(answer));note(dialog,'Soal/kunci merupakan rujukan data. Penggunaan sebagai input mengikuti metode masing-masing eksperimen.')}catch(e){dialog.append(el('p',e.message))}}
window.MLPAnswers={render,rows,openDialog,textFields:texts,getState:()=>({...state})};
})();
