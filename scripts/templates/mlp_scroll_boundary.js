// Browser tertentu menahan gestur di ujung panel. Teruskan hanya sisa gestur ke halaman.
(()=>{'use strict';
const selector='.panel-scroll,.code-scroll,.note-pane,.table-viewport,.comparison-scroll,.compare-table-wrap,.frozen-scroll,.metrics-scroll,#chapterNav,.compare-source,.native-inner';
function blocked(){return Boolean(document.querySelector('dialog[open]'))}
function boundary(target,delta){
 if(!(target instanceof Element)||!target.closest(selector)||blocked())return false;
 for(let node=target;node&&node!==document.body;node=node.parentElement){
  const style=getComputedStyle(node);
  if(!/auto|scroll/.test(style.overflowY))continue;
  const max=node.scrollHeight-node.clientHeight;
  if(max>1&&(delta>0?node.scrollTop<max-1:node.scrollTop>1))return false;
 }
 const page=document.scrollingElement,max=page.scrollHeight-innerHeight;
 return delta>0?page.scrollTop<max-1:page.scrollTop>1;
}
function transfer(event,delta){
 if(!event.cancelable||event.defaultPrevented||!boundary(event.target,delta))return;
 event.preventDefault();window.scrollBy({top:delta,behavior:'instant'});
}
document.addEventListener('wheel',event=>{
 if(event.ctrlKey||event.shiftKey||Math.abs(event.deltaX)>Math.abs(event.deltaY)||!event.deltaY)return;
 const scale=event.deltaMode===1?16:event.deltaMode===2?innerHeight:1;
 transfer(event,event.deltaY*scale);
},{passive:false});
let touch=null;
document.addEventListener('touchstart',event=>{touch=event.touches.length===1?{x:event.touches[0].clientX,y:event.touches[0].clientY}:null},{passive:true});
document.addEventListener('touchmove',event=>{
 if(!touch||event.touches.length!==1)return;
 const point=event.touches[0],delta=touch.y-point.clientY,horizontal=touch.x-point.clientX;
 touch={x:point.clientX,y:point.clientY};
 if(Math.abs(delta)>Math.abs(horizontal)&&Math.abs(delta)>1)transfer(event,delta);
},{passive:false});
document.addEventListener('touchend',()=>{touch=null},{passive:true});
})();
