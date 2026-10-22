(()=>{'use strict';
const A=window.MLPStudy,F=A.data.frozen,S=A.data.snapshot,B=A.data.comparison.baseline,$=id=>document.getElementById(id);
const el=(tag,text,cls)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e};
const state={tab:'review',scenario:'S3',fold:1,curve:'RMSE Validasi (poin)',book:3,sheet:'Perbandingan Model',page:0,component:0,query:''};
const tabs=['review','process','predictions','excel'],versions=[{label:'Awal · 5 epoch',data:B},{label:'MLP acuan',data:S},{label:'Double frozen + MLP',data:F}];
let original=null,zip=null,sheets=new Map(),ticket=0;
const fmt=(v,d=4)=>v===undefined||v===null?'—':typeof v==='number'?Number.isInteger(v)?String(v):v.toLocaleString('id-ID',{minimumFractionDigits:d,maximumFractionDigits:d}):String(v);
function section(root,kicker,title,body){const h=el('header',undefined,'compare-section-title');h.append(el('small',kicker),el('h3',title));if(body)h.append(el('p',body));root.append(h)}
function note(root,title,text,kind='yellow'){const n=el('aside',undefined,'compare-note '+kind);n.append(el('b',title),el('p',text));root.append(n);return n}
function table(headers,rows,id){const box=el('div',undefined,'compare-table-wrap');box.tabIndex=0;const t=el('table');if(id)t.id=id;const head=el('thead'),tr=el('tr');headers.forEach(h=>tr.append(el('th',h)));head.append(tr);t.append(head);const body=el('tbody');rows.forEach(row=>{const r=el('tr');row.forEach(v=>{const c=el('td',typeof v==='number'?fmt(v):v??'—');if(typeof v==='number')c.title=String(v);r.append(c)});body.append(r)});t.append(body);box.append(t);return box}
function stats(d,s=state.scenario){const xs=d.folds.filter(r=>r.Skenario===s).map(r=>r['RMSE (poin)']),mean=xs.reduce((a,b)=>a+b,0)/xs.length;return {mean,sd:Math.sqrt(xs.reduce((a,b)=>a+(b-mean)**2,0)/(xs.length-1))}}
function metric(d,s=state.scenario){return d.metrics.find(r=>r.Skenario===s)}
function last(d,s){return d.history[s].filter(r=>!r.Fold&&['Latihan final','Pelatihan final'].includes(r.Tahap)).at(-1)}
function controls(){const root=$('frozenScenarios');root.replaceChildren();for(const s of ['S1','S2','S3']){const b=el('button',s+(s==='S1'?' · RoBERTa':s==='S2'?' · BGE-M3':' · Fusion'));b.setAttribute('aria-pressed',String(s===state.scenario));b.addEventListener('click',()=>{state.scenario=s;state.page=0;render()});root.append(b)}}
function render(){
 controls();document.querySelectorAll('[data-frozen-tab]').forEach(b=>{const on=b.dataset.frozenTab===state.tab;b.setAttribute('aria-selected',String(on));b.tabIndex=on?0:-1});
 const root=$('frozenBody');root.setAttribute('aria-labelledby','frozen-tab-'+state.tab);root.replaceChildren();root.scrollTop=0;ticket++;
 ({review,process,predictions,excel})[state.tab](root);
}
function review(root){
 section(root,'01 / HASIL RUN BARU','Pembekuan belum memperbaiki S3.','ZIP terbaru lengkap dengan lima Excel, 300 jawaban dan 60 data uji. Anggota development/test, seluruh fold, ID dan nilai guru sudah dicocokkan dengan MLP acuan.');
 const old=metric(S),now=metric(F),a=stats(S),b=stats(F),cards=el('div',undefined,'frozen-scorecards');
 for(const [label,before,after] of [['RMSE test',old['RMSE (poin)'],now['RMSE (poin)']],['Rata-rata RMSE CV',a.mean,b.mean],['Pearson test',old.Pearson,now.Pearson]]){
  const card=el('article');card.append(el('small',state.scenario+' / '+label),el('div',undefined,'frozen-values'));card.lastChild.append(el('span',fmt(before)),el('i','→'),el('strong',fmt(after)));card.append(el('p','MLP acuan → run double frozen'));cards.append(card);
 }root.append(cards);
 const delta=now['RMSE (poin)']-old['RMSE (poin)'];
 note(root,state.scenario+' · cara membaca hasil test','RMSE '+(delta>=0?'bertambah ':'berkurang ')+fmt(Math.abs(delta))+' poin. '+(delta>=0?'Kesalahan prediksi run baru lebih besar.':'Kesalahan prediksi run baru lebih kecil.')+' Baca Pearson, QWK dan AC2 bersama-sama; ini hasil run, bukan ukuran efek satu lapisan.');
 root.append(window.MLPCompare.experimentTable(versions,[['RMSE (poin)','RMSE ↓ · poin'],['Pearson','Pearson ↑'],['QWK','QWK ↑'],['Gwet AC2','Gwet AC2 ↑']].map(([key,label])=>({key,label,get:(d,s)=>metric(d,s)[key]})),'frozenAllTest',fmt));
 note(root,'Nama head pada kolom merujuk perubahan S3.','S1 tetap memakai head linear dan S2 tetap memakai MLP pada ketiga eksperimen. Pada eksperimen ketiga, double frozen berlaku pada kedua encoder S3. Semua head tetap menghasilkan prediksi skor regresi.');
 note(root,'Pemenang run baru berbeda dari pemenang keseluruhan.','Run double frozen: S2 RMSE 6,8435, S1 6,8829 dan S3 7,8611. Selisih S2–S1 hanya sekitar 0,039 poin; belum ada bukti keunggulan yang konsisten dari pengulangan seed. Dari hasil yang tersedia, S1 MLP acuan masih RMSE test terkecil: 6,3805.');
 section(root,'VALIDATION / LIMA FOLD',state.scenario+' · RMSE per pembagian yang sama.','CV dilakukan pada 240 development; tabel test di atas memakai 60 jawaban held-out. Keduanya dibaca terpisah.');
 root.append(table(['Fold',...versions.map(v=>v.label)],Array.from({length:5},(_,i)=>[i+1,...versions.map(v=>v.data.folds.find(r=>r.Skenario===state.scenario&&r.Fold===i+1)['RMSE (poin)'])]),'frozenFoldComparison'));
 root.append(table(['Skenario',...versions.map(v=>v.label+' · mean ± SD')],['S1','S2','S3'].map(s=>[s,...versions.map(v=>{const x=stats(v.data,s);return fmt(x.mean)+' ± '+fmt(x.sd)})]),'frozenCVSummary'));
 const detail=el('details',undefined,'compare-details');detail.append(el('summary','Keempat metrik semua 15 fold double frozen'));detail.append(table(['Skenario','Fold','RMSE ↓','Pearson','QWK','Gwet AC2'],F.folds.map(r=>[r.Skenario,r.Fold,r['RMSE (poin)'],r.Pearson,r.QWK,r['Gwet AC2']]),'frozenAllFolds'));root.append(detail);
 section(root,'LOSS / LATIHAN FINAL','Loss rendah tetap perlu diuji pada jawaban baru.','MSE training memakai target 0–1 dan dicatat selama pembaruan bobot/dropout. RMSE test memakai poin 0–100. Angka ini tidak dipertukarkan.');
 root.append(window.MLPCompare.experimentTable(versions,[{key:'Epoch',label:'Epoch refit',get:(d,s)=>last(d,s)?.Epoch},{key:'MSE',label:'MSE training ↓',digits:6,get:(d,s)=>last(d,s)?.['Loss Training (MSE)']}],'frozenFinalLoss',fmt));
 note(root,'S3 baru: MSE 0,002006 belum menjamin prediksi test paling dekat.','Training S3 sudah mempelajari pemetaan fitur tetap, tetapi RMSE test masih 7,8611 dan Pearson 0,3992. MLP acuan S3 mempunyai RMSE 6,7085. Fitur frozen mungkin kurang sesuai untuk tugas penilaian atau resep head masih perlu diuji; ini hipotesis, bukan penyebab yang dibuktikan oleh tabel.');
 for(const text of F.limits)note(root,'Batas kesimpulan',text,'blue');
}
function process(root){
 section(root,'02 / METODE','Yang dibekukan ialah encoder S3.','S1 tetap RoBERTa fine-tuned + head linear. S2 tetap BGE-M3 frozen + MLP. Pada S3, kedua encoder frozen, sedangkan LayerNorm tambahan, proyeksi dan head MLP tetap dilatih.');
 root.append(table(['Bagian','MLP acuan','Double frozen'],[['Encoder RoBERTa S3','Fine-tuned · ikut diperbarui','Frozen · bobot tetap'],['Encoder BGE-M3 S3','Frozen','Frozen'],['LayerNorm RoBERTa tambahan','Dilatih · 768 komponen','Dilatih · 768 komponen'],['Proyeksi BGE + LayerNorm','Dilatih · 1024 → 256','Dilatih · 1024 → 256'],['Concat','768 + 256 = 1024','768 + 256 = 1024'],['Head S3','MLP 1024 → 256 → 1','MLP 1024 → 256 → 1'],['S1 / S2','Head linear / MLP','Arsitektur tetap']],'frozenLayerTable'));
 note(root,'Bukti pada vektor hasil','Seluruh 300 × 768 komponen RoBERTa mentah S3 identik dengan RoBERTa sebelum latihan: selisih maksimum '+fmt(F.vector_check.max_abs_difference_S3_vs_pretrained,8)+'. Ini mendukung fitur encoder tetap. Sheet setelah LayerNorm menyimpan fitur yang sudah melewati lapisan tambahan; vektornya dapat berubah walaupun encoder frozen.');
 const explanation=note(root,'Kenapa dua encoder tidak otomatis lebih baik?','Menambah informasi tidak menjamin head dapat memakainya. Fitur RoBERTa frozen tetap berasal dari pretraining; ketika fine-tuned, cabangnya dapat menyesuaikan diri terhadap skor guru. Pembandingan satu run belum mengisolasi faktor ini.');
 const buttons=el('div',undefined,'frozen-term-buttons');
 for(const [title,text] of [['Frozen','requires_grad=False menghentikan pembaruan bobot encoder. Layer lain tetap dapat belajar. Ini berbeda dari mematikan seluruh training.'],['LayerNorm','LayerNorm tambahan menata fitur sebelum concat. Parameter tambahan ini tetap dilatih meskipun encoder RoBERTa frozen.'],['Concat','torch.cat menyambung fitur jawaban 768 RoBERTa dan 256 proyeksi BGE menjadi 1024. Ini bukan penggabungan skor dan tidak membuat vektor fusion per kata.'],['MLP','MLP memetakan fitur ke satu skor dengan Linear → GELU → Dropout → Linear. Head ini tetap belajar pada double frozen.'],['Resep berikutnya','Pertahankan MLP acuan untuk presentasi, simpan double frozen sebagai hasil eksperimen yang sah. Jika bereksperimen lagi, pilih kandidat memakai CV dan ulangi seed. Ridge pada fitur frozen dapat menjadi pembanding head; peningkatannya perlu diuji.']]){const b=el('button',title,'mini-button');b.addEventListener('click',()=>explanation.replaceChildren(el('b',title),el('p',text)));buttons.append(b)}root.append(buttons);
 section(root,'SOURCE / KODE RUJUKAN','Lihat perubahan kebijakan bobot.','Sumber MLP acuan cocok notebook unggahan. Untuk double frozen, kode di bawah adalah versi manual di repo yang sesuai deskripsi workbook; notebook eksekusinya belum dikirim.');
 const controls=el('div',undefined,'compare-inline-tools'),select=el('select');select.setAttribute('aria-label','Kode rujukan double frozen');for(const [value,label]of [['model','Arsitektur model'],['config','Konfigurasi'],['fit','Loop training']]){const o=el('option',label);o.value=value;select.append(o)}controls.append(select);const sources=el('div',undefined,'source-pair');
 function drawSource(){sources.replaceChildren();const evidence=F.source_reference.evidence[select.value],old=A.data.cells.find(c=>c.index===evidence.cell).source,newer=evidence.source,changes=window.MLPCompare.changedLines(old,newer);for(const [i,src] of [old,newer].entries()){const frame=el('section',undefined,'source-frame');frame.append(el('header',i?'DOUBLE FROZEN · kode rujukan '+F.source_reference.commit.slice(0,7):'MLP ACUAN · source 15bc7bb'));const code=el('div',undefined,'compare-source');code.tabIndex=0;src.split('\n').forEach((line,j)=>{const row=el('div',undefined,'compare-code-line'+(changes[i].has(j)?' changed':''));row.append(el('span',j+1),el('code',line||' '));code.append(row)});frame.append(code);sources.append(frame)}}select.addEventListener('change',drawSource);drawSource();root.append(controls,sources);
 section(root,'RIWAYAT / ANGKA AKTUAL',state.scenario+' · proses belajar per epoch.','Garis memakai seluruh epoch yang tersimpan untuk fold terpilih. Titik kuning adalah epoch terbaik berdasarkan RMSE validation.');
 const tools=el('div',undefined,'compare-inline-tools'),fold=el('select'),field=el('select');fold.setAttribute('aria-label','Fold grafik double frozen');field.setAttribute('aria-label','Metrik grafik double frozen');for(let i=1;i<=5;i++){const o=el('option','Fold '+i);o.value=i;fold.append(o)}fold.value=state.fold;for(const k of ['RMSE Validasi (poin)','Loss Training (MSE)']){const o=el('option',k);o.value=k;field.append(o)}field.value=state.curve;tools.append(fold,field);const chart=el('div',undefined,'frozen-chart');const draw=()=>{state.fold=Number(fold.value);state.curve=field.value;drawChart(chart)};fold.addEventListener('change',draw);field.addEventListener('change',draw);root.append(tools,chart);draw();
 root.append(table(['Pengaturan','Nilai'],F.parameters.map(r=>[r.Pengaturan,r.Nilai]),'frozenParameters'));
}
function drawChart(root){
 root.replaceChildren();const series=[{label:'MLP acuan',data:S,color:'#00879e'},{label:'Double frozen',data:F,color:'#cf2458'}].map(v=>({...v,rows:v.data.history[state.scenario].filter(r=>r.Fold===state.fold&&Number.isFinite(r[state.curve]))}));
 const points=series.flatMap(s=>s.rows),maxX=Math.max(...points.map(r=>r.Epoch),1),maxY=Math.max(...points.map(r=>r[state.curve]),.001)*1.1,W=760,H=280,L=65,R=20,T=20,Bot=40;
 const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('role','img');svg.setAttribute('aria-label',state.scenario+' fold '+state.fold+' '+state.curve+' MLP vs double frozen');
 const add=(name,attrs,text)=>{const n=document.createElementNS(svg.namespaceURI,name);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text!==undefined)n.textContent=text;svg.append(n);return n},x=v=>L+(v-1)/Math.max(1,maxX-1)*(W-L-R),y=v=>H-Bot-v/maxY*(H-Bot-T);
 for(let i=0;i<=4;i++){const v=i/4*maxY;add('line',{x1:L,y1:y(v),x2:W-R,y2:y(v),stroke:'#d0dfec'});add('text',{x:L-8,y:y(v)+4,'text-anchor':'end','font-size':10,fill:'#4a6480'},fmt(v,state.curve.startsWith('Loss')?3:1))}
 for(const s of series){add('polyline',{points:s.rows.map(r=>x(r.Epoch)+','+y(r[state.curve])).join(' '),fill:'none',stroke:s.color,'stroke-width':3});const best=s.data.foldSummary.find(r=>r.Skenario===state.scenario&&r.Fold===state.fold)['Epoch Terpilih'];for(const r of s.rows){const dot=add('circle',{cx:x(r.Epoch),cy:y(r[state.curve]),r:r.Epoch===best?5:3,fill:r.Epoch===best?'#ffe76b':s.color,stroke:s.color});const title=document.createElementNS(svg.namespaceURI,'title');title.textContent=s.label+' · epoch '+r.Epoch+' · '+r[state.curve];dot.append(title)}}
 for(const v of new Set([1,maxX]))add('text',{x:x(v),y:H-12,'font-size':11,'text-anchor':'middle',fill:'#4a6480'},'Epoch '+v);
 root.append(svg,el('p','Biru: MLP acuan. Merah: double frozen. MSE memakai target 0–1; RMSE memakai poin 0–100. Tidak ada prediksi kurva di luar epoch yang tersimpan.'));
}
function asRecords(sheet){return sheet.rows.map(r=>Object.fromEntries(sheet.headers.map((h,i)=>[h,r[i]])))}
async function readSheet(book,name){
 if(!zip)throw Error('Buka ZIP double frozen untuk membaca nilai individual.');const key=book+'|'+name;
 if(sheets.has(key))return sheets.get(key);
 const bytes=await zip.file(F.books[book].file).async('uint8array'),sheet=await A.readWorkbookSheet(bytes,name);sheets.set(key,sheet);if(sheets.size>2)sheets.delete(sheets.keys().next().value);return sheet;
}
async function loadResults(bytes){
 if(bytes.byteLength>64*1024*1024)throw Error('ZIP melampaui batas 64 MiB.');
 const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))).map(n=>n.toString(16).padStart(2,'0')).join('');if(hash!==F.zip_sha256)throw Error('Pilih Hasil_AES_Ringkas (3).zip yang cocok dengan hasil double frozen ini.');
 const loaded=await JSZip.loadAsync(bytes);if(!F.books.every(b=>loaded.file(b.file)))throw Error('Lima workbook belum lengkap.');
 zip=loaded;original=bytes;sheets.clear();$('frozenStatus').textContent='Lima Excel double frozen lengkap dimuat lokal.';
 if(!$('frozenRoom').hidden)render();window.dispatchEvent(new Event('mlp-private-ready'));
}
async function predictions(root){
 const serial=ticket;section(root,'03 / PASANGAN TEST',state.scenario+' · nilai guru dan dua prediksi.','Pasangan dicocokkan melalui ID unik dan nilai guru, bukan urutan baris. Prediksi mentah untuk RMSE/Pearson; salinan pembulatan untuk QWK/AC2.');
 const impact=F.predictionImpact.find(r=>r.Skenario===state.scenario);note(root,'Error absolut per jawaban',impact.new_lower_error+' dari 60 jawaban mempunyai error lebih kecil pada run double frozen; '+impact.old_lower_error+' lebih kecil pada MLP acuan; '+impact.equal_error+' sama. Jumlah ini dibaca bersama RMSE, sebab besarnya kesalahan juga berpengaruh.');
 if(!original||!A.getState().resultsLoaded){root.append(el('p','Buka ZIP double frozen di atas dan ZIP MLP acuan pada pemilih file di bagian atas halaman. HTML pribadi terbaru sudah memuat keduanya.'));return}
 const target=el('div');target.append(el('p','Mencocokkan pasangan…'));root.append(target);
 try{
  const [old,current]=await Promise.all([A.readSheet(3,'Nilai Guru dan Prediksi'),readSheet(3,'Nilai Guru dan Prediksi')]);if(serial!==ticket)return;
  const left=asRecords(old),right=new Map(asRecords(current).map(r=>[String(r['ID Jawaban']),r]));if(left.length!==60||right.size!==60)throw Error('Test harus memuat 60 ID unik.');
  const pairs=left.map(a=>{const b=right.get(String(a['ID Jawaban']));if(!b||a['Nilai Guru']!==b['Nilai Guru'])throw Error('ID atau nilai guru tidak cocok.');const s=state.scenario,y=a['Nilai Guru'];return [a['ID Jawaban'],y,a['Prediksi '+s],b['Prediksi '+s],a['Pembulatan '+s],b['Pembulatan '+s],Math.abs(a['Prediksi '+s]-y),Math.abs(b['Prediksi '+s]-y)]});
  target.replaceChildren();const search=el('input');search.type='search';search.placeholder='Cari ID jawaban…';search.setAttribute('aria-label','Cari pasangan double frozen');search.value=state.query;const content=el('div');
  const prev=el('button','← Sebelumnya','mini-button'),next=el('button','Berikutnya →','mini-button'),caption=el('span'),nav=el('div',undefined,'frozen-pagination');nav.append(prev,caption,next);
  function draw(){const filtered=pairs.filter(r=>String(r[0]).toLowerCase().includes(state.query.toLowerCase()));state.page=Math.min(state.page,Math.max(0,Math.ceil(filtered.length/20)-1));const start=state.page*20;content.replaceChildren(table(['ID Jawaban','Nilai guru','Prediksi MLP','Prediksi double frozen','Pembulatan MLP','Pembulatan frozen','Error absolut MLP','Error absolut frozen'],filtered.slice(start,start+20),'frozenPredictions'));caption.textContent=(filtered.length?start+1:0)+'–'+Math.min(filtered.length,start+20)+' / '+filtered.length;prev.disabled=state.page===0;next.disabled=start+20>=filtered.length}
  search.addEventListener('input',()=>{state.query=search.value;state.page=0;draw()});prev.addEventListener('click',()=>{state.page--;draw()});next.addEventListener('click',()=>{state.page++;draw()});target.append(search,content,nav);draw();
 }catch(error){target.replaceChildren();note(target,'Tabel belum dapat dibuka',error.message)}
}
function publicSheet(book,name){
 let records=null;if(book===3&&name==='Perbandingan Model')records=F.metrics;if(book===3&&name==='Hasil Tiap Fold')records=F.folds;if(book===4&&name==='Ringkasan Lima Fold')records=F.foldSummary;if(book===4&&name==='Pengaturan Eksperimen')records=F.parameters;if(book===4&&name==='Model yang Dibandingkan')records=F.models;
 const names={'RoBERTa (S1)':'S1','BGE-M3 (S2)':'S2','Fusion (S3)':'S3'};if(book===4&&names[name])records=F.history[names[name]];
 if(!records)return null;const headers=Object.keys(records[0]);return {headers,rows:records.map(r=>headers.map(h=>r[h]))};
}
async function excel(root){
 const serial=ticket;section(root,'04 / LIMA EXCEL RUN BARU','Data lengkap, dibaca per bagian.','Tabel mempertahankan seluruh baris dan komponen. Satu halaman berisi 25 baris dan delapan komponen; nama sheet dan angka berasal dari workbook baru.');
 const controls=el('div',undefined,'excel-selects'),book=el('select'),sheet=el('select');book.setAttribute('aria-label','Workbook double frozen');sheet.setAttribute('aria-label','Sheet double frozen');
 F.books.forEach((b,i)=>{const o=el('option',b.file);o.value=i;book.append(o)});book.value=state.book;
 if(!F.books[state.book].sheets.some(s=>s.name===state.sheet))state.sheet=F.books[state.book].sheets[0].name;
 F.books[state.book].sheets.forEach(s=>{const o=el('option',s.name+' · '+s.rows+' × '+s.columns);o.value=s.name;sheet.append(o)});sheet.value=state.sheet;controls.append(book,sheet);root.append(controls);
 book.addEventListener('change',()=>{state.book=Number(book.value);state.sheet=F.books[state.book].sheets[0].name;state.page=state.component=0;render()});sheet.addEventListener('change',()=>{state.sheet=sheet.value;state.page=state.component=0;render()});
 const meta=F.books[state.book].sheets.find(s=>s.name===state.sheet);root.append(el('p',meta.rows+' baris data · '+meta.columns+' kolom','table-caption'));
 if(state.sheet.includes('LayerNorm'))note(root,'Sesudah lapisan tambahan','Vektor encoder S3 tetap pretrained. Sheet ini menyimpan hasil LayerNorm yang parameternya dilatih, sehingga dibedakan dari vektor mentah RoBERTa.');
 if(state.sheet.includes('Fusion'))note(root,'Fusion pada tingkat jawaban','768 komponen RoBERTa setelah LayerNorm disambung dengan 256 proyeksi BGE. Vektor 1024 komponen ini bukan vektor fusion setiap kata.');
 const target=el('div');root.append(target);
 try{
  const data=original?await readSheet(state.book,state.sheet):publicSheet(state.book,state.sheet);if(serial!==ticket)return;
  if(!data){target.append(el('p','Isi individual tersedia setelah ZIP double frozen dibuka secara lokal. Kolom awal: '+meta.headers.join(' · ')));return}
  const first=data.headers.findIndex(h=>/^Komponen \d+$/.test(h)),pinned=first<0?0:first,count=first<0?data.headers.length:data.headers.length-first;state.component=Math.min(state.component,Math.max(0,count-1));
  const start=el('input');start.type='number';start.min=1;start.max=count;start.value=state.component+1;start.setAttribute('aria-label','Mulai komponen double frozen');const tools=el('div',undefined,'frozen-pagination'),prev=el('button','← Baris','mini-button'),next=el('button','Baris →','mini-button'),caption=el('span'),view=el('div');tools.append(prev,caption,next);
  const compPrev=el('button','← Komponen','mini-button'),compNext=el('button','Komponen →','mini-button'),cols=el('div',undefined,'frozen-pagination');if(first>=0)cols.append(compPrev,el('label','Mulai komponen'),start,compNext);
  function draw(){const columns=first<0?data.headers.map((_,i)=>i):[...Array.from({length:pinned},(_,i)=>i),...Array.from({length:Math.min(8,count-state.component)},(_,i)=>pinned+state.component+i)];const begin=state.page*25;view.replaceChildren(table(columns.map(i=>data.headers[i]),data.rows.slice(begin,begin+25).map(r=>columns.map(i=>r[i])),'frozenSheetTable'));caption.textContent=(data.rows.length?begin+1:0)+'–'+Math.min(data.rows.length,begin+25)+' / '+data.rows.length;prev.disabled=state.page===0;next.disabled=begin+25>=data.rows.length;compPrev.disabled=state.component===0;compNext.disabled=state.component+8>=count;start.value=state.component+1}
  prev.addEventListener('click',()=>{state.page--;draw()});next.addEventListener('click',()=>{state.page++;draw()});start.addEventListener('change',()=>{state.component=Math.max(0,Math.min(count-1,Number(start.value)-1));draw()});compPrev.addEventListener('click',()=>{state.component=Math.max(0,state.component-8);draw()});compNext.addEventListener('click',()=>{state.component=Math.min(count-1,state.component+8);draw()});target.append(cols,view,tools);draw();
  if(original){const download=el('button','Unduh XLSX asli ↓','mini-button');download.addEventListener('click',async()=>{const bytes=await zip.file(F.books[state.book].file).async('uint8array'),url=URL.createObjectURL(new Blob([bytes],{type:'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'})),a=el('a');a.href=url;a.download='DoubleFrozen_'+F.books[state.book].file;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)});target.append(download)}
 }catch(error){note(target,'Sheet belum dapat dibuka',error.message)}
}
$('metricToFrozen').addEventListener('click',()=>window.MLPCompare.setMode('frozen',true));
$('openFrozen').addEventListener('click',()=>window.MLPCompare.setMode('frozen',true));
$('frozenResultsFile').addEventListener('change',async()=>{try{if($('frozenResultsFile').files[0])await loadResults(await $('frozenResultsFile').files[0].arrayBuffer())}catch(e){$('frozenStatus').textContent=e.message}});
for(const b of document.querySelectorAll('[data-frozen-tab]')){
 b.addEventListener('click',()=>{state.tab=b.dataset.frozenTab;state.page=state.component=0;render()});
 b.addEventListener('keydown',event=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;event.preventDefault();const i=tabs.indexOf(state.tab),next=event.key==='Home'?tabs[0]:event.key==='End'?tabs.at(-1):tabs[(i+(event.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length];state.tab=next;render();$('frozen-tab-'+next).focus()});
}
window.addEventListener('mlp-private-ready',()=>{if(!$('frozenRoom').hidden&&state.tab==='predictions')render()});
window.MLPFrozen={render,loadResults,readSheet,getBytes:()=>original,getState:()=>({...state,loaded:Boolean(original)}),setTab:tab=>{if(tabs.includes(tab)){state.tab=tab;state.page=state.component=0;render()}},setScenario:s=>{if(['S1','S2','S3'].includes(s)){state.scenario=s;render()}}};
})();
