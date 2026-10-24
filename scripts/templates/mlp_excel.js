// Pembaca hasil tersimpan. Tidak menghitung ulang model atau mengubah isi workbook.
const EXCEL_ROUTES=[
  {book:3,title:'Prediksi & evaluasi',detail:'Nilai guru, skor model, empat metrik',sheet:'Perbandingan Model'},
  {book:0,title:'Data & tokenisasi',detail:'Jawaban, token dan pembagian data',sheet:'Token RoBERTa'},
  {book:1,title:'Vektor jawaban',detail:'RoBERTa, BGE-M3 dan fusion',sheet:'Fusion S3'},
  {book:2,title:'Vektor kata & kalimat',detail:'Kata sama, konteks berbeda',sheet:'Kata Berulang'},
  {book:4,title:'Riwayat training',detail:'Epoch, loss dan validasi tiap fold',sheet:'Ringkasan Lima Fold'}
];
function excelOrigin(book,name){
  if(name==='Panduan'||name==='Arti Metrik')return {chapter:15,cell:45,label:'Penyusunan lima laporan Excel'};
  if(book===0){
    if(name.includes('Token'))return {chapter:6,cell:22,label:'Tokenisasi aktual RoBERTa dan BGE-M3'};
    if(name.includes('Pembagian'))return {chapter:5,cell:18,label:'Pembagian 80/20 dan lima fold'};
    return {chapter:4,cell:16,label:'Pembacaan jawaban dan nilai guru'};
  }
  if(book===1){
    if(name==='BGE-M3')return {chapter:8,cell:26,label:'Embedding BGE-M3 frozen'};
    if(name.includes('Sebelum'))return {chapter:7,cell:24,label:'Embedding RoBERTa sebelum training'};
    return {chapter:14,cell:43,label:'Ekstraksi fitur dari model final'};
  }
  if(book===2)return name==='Jawaban Contoh'?{chapter:13,cell:41,label:'Jawaban contoh untuk prediksi'}:
    /S1|S3/.test(name)?{chapter:14,cell:43,label:'Vektor kata dan kalimat sesudah training'}:
    {chapter:9,cell:28,label:'Vektor kata, kalimat dan kata berulang'};
  if(book===3&&/Satu Jawaban|Jawaban Contoh/.test(name))return {chapter:13,cell:41,label:'Prediksi satu jawaban'};
  if(book===4&&name==='Pengaturan Eksperimen')return {chapter:3,cell:12,label:'Pengaturan model dan training'};
  if(book===4&&name==='Model yang Dibandingkan')return {chapter:10,cell:32,label:'Arsitektur S1, S2 dan S3'};
  return {chapter:12,cell:39,label:'Validasi silang, pelatihan final dan evaluasi test'};
}
function openExcelSource(tab){
  const origin=excelOrigin(state.book,state.sheet);
  openChapter(origin.chapter);
  if(tab==='code')jumpCode(origin.cell,1);else setTab('output');
}
function sheetAvailable(book,name){return Boolean(resultZip||publicSheet(book,name))}
function renderExcel(){
  const routes=$('excelRoutes');routes.replaceChildren();
  for(const route of EXCEL_ROUTES){
    const b=node('button');b.type='button';b.setAttribute('aria-pressed',String(state.book===route.book));
    b.append(node('strong',route.title),node('span',route.detail));
    b.addEventListener('click',()=>{state.book=route.book;state.sheet=route.sheet;renderExcel();$('pane-excel').scrollTop=0});routes.append(b);
  }
  const availability=$('excelAvailability');availability.replaceChildren();availability.hidden=Boolean(resultZip);
  if(!resultZip){
    availability.append(node('p','Salinan agregat memuat tabel hasil dan riwayat E6. Buku lengkap sudah menyertakan seluruh Excel; tidak perlu unggah ZIP.'));
  }
  const bs=$('bookSelect');bs.replaceChildren();
  S.books.forEach((book,i)=>{const label=EXCEL_ROUTES.find(r=>r.book===i).title,o=node('option',String(i+1).padStart(2,'0')+' · '+label);o.value=i;bs.append(o)});bs.value=state.book;
  const available=S.books[state.book].sheets.filter(s=>sheetAvailable(state.book,s.name));
  if(!available.some(s=>s.name===state.sheet))state.sheet=available[0]?.name||S.books[state.book].sheets[0].name;
  const ss=$('sheetSelect');ss.replaceChildren();
  for(const meta of available){const o=node('option',meta.name+' · '+meta.rows.toLocaleString('id-ID')+' baris');o.value=meta.name;ss.append(o)}ss.value=state.sheet;
  $('sheetExplanation').textContent=sheetExplanation(state.sheet);
  $('excelOrigin').textContent='Asal hasil: '+excelOrigin(state.book,state.sheet).label+'.';
  $('downloadBook').hidden=$('downloadResults').hidden=!resultZip;
  $('cellInspector').textContent='Pilih sel dengan klik atau Enter untuk membaca isi lengkapnya.';
  loadActiveSheet();
}
async function loadActiveSheet(){
  const ticket=++tableTicket;activeSheet=null;
  $('sheetStatus').textContent='Membuka tabel…';$('sheetTable').replaceChildren();
  try{
    const sheet=resultZip?await readSheet(state.book,state.sheet):publicSheet(state.book,state.sheet);
    if(ticket!==tableTicket)return;
    if(!sheet)throw Error('Tabel ini memerlukan buku lengkap yang sudah memuat hasil E6.');
    activeSheet=sheet;state.page=state.col=0;$('componentStart').value=1;$('rowSearch').value='';renderTable();
    $('tableViewport').scrollTop=$('tableViewport').scrollLeft=0;
  }catch(e){if(ticket===tableTicket){$('sheetStatus').textContent=e.message;error(e)}}
}
function renderTable(){
  if(!activeSheet)return;
  const headers=activeSheet.headers,first=headers.findIndex(h=>/^Komponen \d+$/.test(h)),hasComponents=first>=0;
  const q=$('rowSearch').value.trim().toLocaleLowerCase('id');
  const rows=q?activeSheet.rows.filter(row=>row.slice(0,hasComponents?first:headers.length).some(v=>String(v??'').toLocaleLowerCase('id').includes(q))):activeSheet.rows;
  const componentCount=hasComponents?headers.length-first:0,pageSize=state.pageSize;
  state.col=Math.max(0,Math.min(Math.max(0,componentCount-1),Number.isFinite(state.col)?state.col:0));
  state.page=Math.max(0,Math.min(Math.max(0,Math.ceil(rows.length/pageSize)-1),state.page));
  const columns=hasComponents?[...Array.from({length:first},(_,i)=>i),...Array.from({length:Math.min(8,componentCount-state.col)},(_,i)=>first+state.col+i)]:headers.map((_,i)=>i);
  const start=state.page*pageSize,end=Math.min(rows.length,start+pageSize),table=$('sheetTable');table.replaceChildren();
  const caption=node('caption',state.sheet+' — '+(resultZip?'nilai dari Excel asli':'hasil agregat tersimpan'));caption.className='screen-reader-only';table.append(caption);
  const head=node('thead'),hr=node('tr');for(const i of columns){const th=node('th',headers[i]);th.scope='col';hr.append(th)}head.append(hr);table.append(head);
  const body=node('tbody');
  for(let index=start;index<end;index++){
    const row=node('tr');
    for(const i of columns){
      const value=rows[index][i],td=node('td',shown(value));if(typeof value==='number')td.classList.add('cell-number');td.tabIndex=0;td.title='Lihat isi lengkap: '+String(value??'—');
      function inspect(){const identity=rows[index][0];$('cellInspector').replaceChildren(node('strong',headers[i]),node('p','Baris '+(index+1)+(identity==null?'':' · '+String(identity))),node('pre',String(value??'—')))}
      td.addEventListener('click',inspect);td.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();inspect()}});row.append(td);
    }body.append(row);
  }
  if(!rows.length){const row=node('tr'),td=node('td','Tidak ada baris yang cocok. Ubah atau kosongkan pencarian.');td.colSpan=Math.max(1,columns.length);row.append(td);body.append(row)}table.append(body);
  $('sheetStatus').textContent=(resultZip?'XLSX asli · ':'Output agregat aktual · ')+(q?rows.length+' baris cocok dari ':'')+activeSheet.rows.length+' baris × '+headers.length+' kolom. '+(state.full?'Angka lengkap ditampilkan.':'Angka layar dibulatkan maksimal 4 desimal; pilih sel untuk nilai asli.');
  $('rowRange').textContent=rows.length?(start+1)+'–'+end+' dari '+rows.length:'0 baris';
  $('componentControl').hidden=$('componentPagination').hidden=!hasComponents;
  $('rowPagination').hidden=rows.length<=pageSize;
  $('componentStart').max=componentCount||1;$('componentStart').value=state.col+1;
  $('colRange').textContent='Komponen '+(state.col+1)+'–'+Math.min(componentCount,state.col+8)+' dari '+componentCount;
  $('prevRows').disabled=state.page===0;$('nextRows').disabled=end>=rows.length;
  $('prevCols').disabled=state.col===0;$('nextCols').disabled=state.col+8>=componentCount;
  $('downloadBook').disabled=!resultZip;
}

let outputEvidenceTicket=0;
async function renderOutputEvidence(root,chapter){
  const ticket=++outputEvidenceTicket,book=chapter.book-1,box=node('section',undefined,'output-evidence');
  box.append(node('h4','Hasil tahap ini di Excel'),node('p','Nilai berikut berasal dari laporan tersimpan, terpisah dari output Colab asli di atas.'));
  const actions=node('div',undefined,'output-evidence-actions');
  for(const name of chapter.sheets){
    if(!S.books[book].sheets.some(s=>s.name===name))continue;
    const button=node('button',name+' →','mini-button');
    button.addEventListener('click',()=>{state.book=book;state.sheet=name;setTab('excel')});actions.append(button);
  }
  box.append(actions);root.append(box);
  const name=chapter.sheets.find(name=>sheetAvailable(book,name));
  if(!name){box.append(node('p','Buka ZIP hasil E6 untuk membaca tabel token, jawaban dan vektor. Pada HTML pribadi semuanya sudah dimuat.'));return}
  try{
    const sheet=resultZip?await readSheet(book,name):publicSheet(book,name);
    if(ticket!==outputEvidenceTicket||!box.isConnected||!sheet)return;
    box.append(node('p',name+' · '+sheet.rows.length.toLocaleString('id-ID')+' baris. Pratinjau maksimal lima baris; buka tabel untuk seluruh isi.'));
    const view=node('div',undefined,'output-evidence-preview'),table=node('table'),head=node('thead'),hr=node('tr'),indices=sheet.headers.map((_,i)=>i).slice(0,6);
    for(const i of indices){const th=node('th',sheet.headers[i]);th.scope='col';hr.append(th)}head.append(hr);table.append(head);
    const body=node('tbody');for(const row of sheet.rows.slice(0,5)){const tr=node('tr');for(const i of indices)tr.append(node('td',shown(row[i])));body.append(tr)}table.append(body);view.append(table);box.append(view);
  }catch(e){if(box.isConnected)box.append(node('p','Pratinjau belum tersedia: '+e.message))}
}
