import {CLASS_NAMES,COLORS,letterboxGeometry} from './detection.js';
const $=id=>document.getElementById(id);
const state={image:null,name:'',detections:null,busy:false,elapsedMs:0,backend:'',worker:null};
const confidence=()=>Number($('confidence').value)/100;
const visibleDetections=()=>state.detections?.filter(box=>box.confidence>=confidence())||[];
function status(message,error=false){$('status').textContent=message;$('status').classList.toggle('error',error);}
function setBusy(busy){state.busy=busy;$('loading').hidden=!busy;$('detect').disabled=busy||!state.image;$('replace').disabled=busy;$('file').disabled=busy;document.querySelectorAll('.sample').forEach(button=>button.disabled=busy);$('dropzone').setAttribute('aria-busy',String(busy));}
function updateResults(){
  const boxes=visibleDetections(),counts=new Map();for(const box of boxes)counts.set(box.classId,(counts.get(box.classId)||0)+1);
  $('object-count').textContent=state.detections===null?'—':boxes.length;$('class-count').textContent=state.detections===null?'—':counts.size;
  $('results-empty').hidden=state.detections!==null;$('class-summary').hidden=state.detections===null;$('detections').hidden=state.detections===null;
  $('class-summary').replaceChildren();$('detections').replaceChildren();
  for(const [id,count]of [...counts].sort((a,b)=>b[1]-a[1])){const chip=document.createElement('span');chip.className='class-chip';chip.append(CLASS_NAMES[id]);const number=document.createElement('b');number.textContent=count;chip.append(number);$('class-summary').append(chip);}
  if(state.detections!==null&&!boxes.length){const note=document.createElement('p');note.className='helper';note.textContent='No objects above this threshold. Try lowering it or using another aerial image.';$('detections').append(note);}
  for(const [index,box]of boxes.entries()){
    const row=document.createElement('div');row.className='detection-row';const label=document.createElement('span'),swatch=document.createElement('i'),score=document.createElement('strong');swatch.className='swatch';swatch.style.background=COLORS[box.classId];label.append(swatch,`${String(index+1).padStart(2,'0')}  ${box.className}`);score.textContent=`${(box.confidence*100).toFixed(1)}%`;row.append(label,score);$('detections').append(row);
  }
  $('download-image').disabled=state.detections===null||state.busy;$('download-json').disabled=state.detections===null||state.busy;
  draw();
}
function draw(){
  if(!state.image)return;
  const canvas=$('canvas'),context=canvas.getContext('2d');canvas.width=state.image.width;canvas.height=state.image.height;context.drawImage(state.image,0,0);
  const scale=Math.max(canvas.width,canvas.height)/1100,line=Math.max(2,scale*2),font=Math.max(12,scale*13);context.lineWidth=line;context.font=`600 ${font}px sans-serif`;
  for(const box of visibleDetections()){
    context.strokeStyle=COLORS[box.classId];context.strokeRect(box.x1,box.y1,box.x2-box.x1,box.y2-box.y1);
    if($('show-labels').checked){const label=`${box.className} ${(box.confidence*100).toFixed(0)}%`,width=context.measureText(label).width+8*scale,height=font+7*scale;const x=Math.max(0,Math.min(box.x1,canvas.width-width)),y=Math.max(height,box.y1);context.fillStyle=COLORS[box.classId];context.fillRect(x,y-height,width,height);context.fillStyle='#101419';context.fillText(label,x+4*scale,y-5*scale);}
  }
}
async function loadImage(blob,name){
  if(state.busy)return;
  if(!['image/jpeg','image/png','image/webp'].includes(blob.type)){status('Please choose a JPG, PNG, or WEBP image.',true);return;}
  if(blob.size>20*1024*1024){status('This image is too large. Choose a file under 20 MB.',true);return;}
  try{
    const image=await createImageBitmap(blob);
    if(image.width*image.height>32000000){image.close();throw new Error('This image has too many pixels. Resize it to under 32 megapixels.');}
    state.image?.close();state.image=image;state.name=name;state.detections=null;
    $('canvas').hidden=false;$('empty-state').hidden=true;$('stage-label').hidden=false;$('stage-label-text').textContent='IMAGE LOADED';$('filename').textContent=name;
    $('file-detail').textContent=`${(blob.size/1024/1024).toFixed(2)} MB · Ready to analyze`;$('image-meta').textContent=`${image.width.toLocaleString()} × ${image.height.toLocaleString()} px`;
    $('result-badge').textContent='WAITING';$('detect').disabled=false;status('Image ready. Let’s find what’s in it.');updateResults();
  }catch(error){status(error.message||'We couldn’t read this image. Try another file.',true);}
}
function worker(){
  if(state.worker)return state.worker;
  const inference=new Worker(new URL('./inference-worker.js',import.meta.url),{type:'module'});
  inference.onmessage=event=>{
    const data=event.data;
    if(data.type==='progress'){$('loading-title').textContent=data.phase;$('loading-detail').textContent=data.detail;status(data.phase+'…');}
    if(data.type==='result'){state.detections=data.detections;state.elapsedMs=data.elapsedMs;state.backend=data.backend;setBusy(false);updateResults();$('result-badge').textContent='COMPLETE';$('stage-label-text').textContent='DETECTION COMPLETE';$('file-detail').textContent=`${state.image.width} × ${state.image.height} px · ${data.backend==='webgpu'?'GPU':'CPU'} inference`;status(`Done in ${(data.elapsedMs/1000).toFixed(1)}s · Adjust confidence to explore.`);}
    if(data.type==='error'){setBusy(false);status(data.message,true);}
  };
  inference.onerror=()=>{setBusy(false);status('The browser inference engine couldn’t start. Refresh the page or try a current Chrome or Edge browser.',true);inference.terminate();state.worker=null;};
  state.worker=inference;return inference;
}
$('detect').addEventListener('click',()=>{
  if(!state.image||state.busy)return;
  setBusy(true);$('download-image').disabled=true;$('download-json').disabled=true;$('loading-title').textContent='Preparing your image';$('loading-detail').textContent='Keeping the full frame in view.';
  try{
    const geometry=letterboxGeometry(state.image.width,state.image.height),canvas=document.createElement('canvas');canvas.width=1280;canvas.height=1280;const context=canvas.getContext('2d',{willReadFrequently:true});context.fillStyle='rgb(114,114,114)';context.fillRect(0,0,1280,1280);context.drawImage(state.image,geometry.left,geometry.top,geometry.resizedWidth,geometry.resizedHeight);
    const rgba=context.getImageData(0,0,1280,1280).data,pixels=1280*1280,input=new Float32Array(pixels*3);
    for(let i=0;i<pixels;i++){input[i]=rgba[i*4]/255;input[pixels+i]=rgba[i*4+1]/255;input[2*pixels+i]=rgba[i*4+2]/255;}
    worker().postMessage({type:'detect',input,geometry},[input.buffer]);
  }catch(error){setBusy(false);status(error.message,true);}
});
function chooseFile(){if(!state.busy)$('file').click();}
$('replace').addEventListener('click',chooseFile);$('dropzone').addEventListener('click',chooseFile);$('dropzone').addEventListener('keydown',event=>{if(event.target===$('dropzone')&&['Enter',' '].includes(event.key)){event.preventDefault();chooseFile();}});
$('file').addEventListener('change',event=>{const file=event.target.files[0];if(file)loadImage(file,file.name);event.target.value='';});
for(const eventName of ['dragenter','dragover'])$('dropzone').addEventListener(eventName,event=>{event.preventDefault();if(!state.busy)$('dropzone').classList.add('dragging');});
for(const eventName of ['dragleave','drop'])$('dropzone').addEventListener(eventName,event=>{event.preventDefault();$('dropzone').classList.remove('dragging');});
$('dropzone').addEventListener('drop',event=>{if(!state.busy&&event.dataTransfer.files[0])loadImage(event.dataTransfer.files[0],event.dataTransfer.files[0].name);});
document.querySelectorAll('.sample').forEach(button=>button.addEventListener('click',async()=>{if(state.busy)return;try{const response=await fetch(`/assets/sample-${button.dataset.sample}.jpg`);if(!response.ok)throw new Error('Sample unavailable. Please upload an image.');await loadImage(await response.blob(),button.dataset.sample==='city'?'sample-city.jpg':'sample-traffic.jpg');}catch(error){status(error.message,true);}}));
$('confidence').addEventListener('input',()=>{$('confidence-value').value=`${$('confidence').value}%`;updateResults();});$('show-labels').addEventListener('change',draw);
function download(blob,filename){const url=URL.createObjectURL(blob),anchor=document.createElement('a');anchor.href=url;anchor.download=filename;anchor.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
const basename=()=>state.name.replace(/\.[^.]+$/,'')||'aerial-image';
$('download-image').addEventListener('click',()=>$('canvas').toBlob(blob=>{if(blob)download(blob,`${basename()}-detections.png`);else status('Could not export this image. Try again.',true);},'image/png'));
$('download-json').addEventListener('click',()=>download(new Blob([JSON.stringify({model:'VisDrone YOLOv8s',image:state.name,width:state.image.width,height:state.image.height,confidenceThreshold:confidence(),coordinateFormat:'xyxy pixels in original image',inferenceMs:state.elapsedMs,backend:state.backend,detections:visibleDetections()},null,2)],{type:'application/json'}),`${basename()}-detections.json`));
$('about').addEventListener('click',()=>$('model-dialog').showModal());$('close-dialog').addEventListener('click',()=>$('model-dialog').close());$('model-dialog').addEventListener('click',event=>{if(event.target===$('model-dialog')){const rect=event.target.getBoundingClientRect();if(event.clientX<rect.left||event.clientX>rect.right||event.clientY<rect.top||event.clientY>rect.bottom)event.target.close();}});
