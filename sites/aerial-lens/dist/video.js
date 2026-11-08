import {COLORS,CLASS_NAMES,letterboxGeometry} from './detection.js';
import {sampleTimes,nearestFrame} from './video-utils.js';
const $=id=>document.getElementById(id),video=$('video'),canvas=$('video-canvas'),context=canvas.getContext('2d');
const state={url:null,name:'',frames:[],rate:5,busy:false,cancel:false,complete:false,worker:null,pending:null,exporting:false};
const confidence=()=>Number($('video-confidence').value)/100;
const visible=frame=>frame?.detections.filter(box=>box.confidence>=confidence())||[];
const formatTime=time=>`${Math.floor(time/60)}:${String(Math.floor(time%60)).padStart(2,'0')}`;
function status(text,error=false){$('video-status').textContent=text;$('video-status').classList.toggle('error',error);}
function setBusy(value){state.busy=value;for(const id of ['upload-video','sample-video','video-file','analysis-rate','play','timeline'])$(id).disabled=value||(!state.url&&['play','timeline'].includes(id));$('analyze-video').disabled=value||!state.url;$('cancel-analysis').hidden=!value||state.exporting;$('export-video-json').disabled=value||!state.frames.length;$('export-video').disabled=value||!state.frames.length||!window.MediaRecorder;$('video-drop').setAttribute('aria-busy',String(value));}
function waitEvent(element,name,action){return new Promise((resolve,reject)=>{const timer=setTimeout(()=>finish(new Error('Video decoding timed out. Try an MP4 encoded with H.264.')),20000);function finish(error){clearTimeout(timer);element.removeEventListener(name,success);element.removeEventListener('error',failure);error?reject(error):resolve();}const success=()=>finish(),failure=()=>finish(new Error('This browser cannot decode the video. Try an H.264 MP4 or a WebM file.'));element.addEventListener(name,success,{once:true});element.addEventListener('error',failure,{once:true});try{action();}catch(error){finish(error);}});}
async function seek(time){if(Math.abs(video.currentTime-time)<.001&&video.readyState>=2)return;await waitEvent(video,'seeked',()=>{video.currentTime=Math.min(time,Math.max(0,video.duration-.001));});}
async function loadVideo(blob,name){
  if(state.busy)return;
  if(blob.size>150*1024*1024){status('Choose a video smaller than 150 MB.',true);return;}
  if(!/\.(mp4|webm|mov)$/i.test(name)){status('Choose an MP4, WEBM, or MOV video.',true);return;}
  video.pause();state.frames=[];state.complete=false;state.name=name;if(state.url)URL.revokeObjectURL(state.url);state.url=URL.createObjectURL(blob);setBusy(true);$('video-overlay').hidden=false;$('progress-title').textContent='Opening your video';$('progress-detail').textContent='Reading duration and dimensions.';
  $('cancel-analysis').hidden=true;
  video.hidden=false;
  try{
    await waitEvent(video,'loadeddata',()=>{video.src=state.url;video.load();});sampleTimes(video.duration,5);
    if(video.videoWidth*video.videoHeight>8300000)throw new Error('Please resize this clip to 4K or below.');
    canvas.width=video.videoWidth;canvas.height=video.videoHeight;canvas.hidden=false;$('empty-state').hidden=true;$('video-meta').textContent=`${video.videoWidth} × ${video.videoHeight} · ${video.duration.toFixed(1)}s`;$('video-name').textContent=name;$('video-file-detail').textContent=`${(blob.size/1024/1024).toFixed(1)} MB · Ready to analyze`;$('timeline').max=video.duration;$('timeline').value=0;$('progress').value=0;$('frame-progress').textContent='No analyzed frames yet';$('video-result-badge').textContent='WAITING';status('Video ready. Choose a sampling rate and start analysis.');
  }catch(error){URL.revokeObjectURL(state.url);state.url=null;video.removeAttribute('src');canvas.hidden=true;$('empty-state').hidden=false;$('video-name').textContent='No video selected';status(error.message,true);}
  finally{setBusy(false);$('video-overlay').hidden=true;draw();}
}
function draw(update=true){
  if(!state.url||video.readyState<2)return;
  context.drawImage(video,0,0,canvas.width,canvas.height);const frame=nearestFrame(state.frames,video.currentTime,state.rate),boxes=visible(frame),scale=Math.max(canvas.width,canvas.height)/1100,font=Math.max(12,scale*13);context.lineWidth=Math.max(2,scale*2);context.font=`600 ${font}px sans-serif`;
  for(const box of boxes){context.strokeStyle=COLORS[box.classId];context.strokeRect(box.x1,box.y1,box.x2-box.x1,box.y2-box.y1);if($('video-labels').checked){const text=`${box.className} ${(box.confidence*100).toFixed(0)}%`,width=context.measureText(text).width+8*scale,height=font+7*scale,x=Math.max(0,Math.min(box.x1,canvas.width-width)),y=Math.max(height,box.y1);context.fillStyle=COLORS[box.classId];context.fillRect(x,y-height,width,height);context.fillStyle='#101419';context.fillText(text,x+4*scale,y-5*scale);}}
  $('timeline').value=video.currentTime;$('timecode').textContent=`${formatTime(video.currentTime)} / ${formatTime(video.duration||0)}`;
  if(!update)return;
  $('video-object-count').textContent=frame?boxes.length:'—';$('video-frame-count').textContent=state.frames.length;const counts=new Map();for(const box of boxes)counts.set(box.classId,(counts.get(box.classId)||0)+1);$('video-summary').replaceChildren();$('video-detections').replaceChildren();
  for(const [cls,count]of [...counts].sort((a,b)=>b[1]-a[1])){const chip=document.createElement('span');chip.className='class-chip';chip.append(CLASS_NAMES[cls]);const number=document.createElement('b');number.textContent=count;chip.append(number);$('video-summary').append(chip);}
  if(!boxes.length){const note=document.createElement('p');note.className='helper';note.textContent=frame?'No objects above this confidence threshold.':'This moment has not been analyzed yet.';$('video-summary').append(note);}
  for(const box of boxes.slice(0,100)){const row=document.createElement('div');row.className='detection-row';const name=document.createElement('span'),score=document.createElement('strong');name.textContent=box.className;score.textContent=`${(box.confidence*100).toFixed(1)}%`;row.append(name,score);$('video-detections').append(row);}
}
function detector(){
  if(state.worker)return state.worker;
  const worker=new Worker(new URL('./inference-worker.js',import.meta.url),{type:'module'});
  worker.onmessage=event=>{const data=event.data;if(data.type==='progress'){if(['Loading your model','Preparing the detector','Switching to CPU'].includes(data.phase)){$('video-overlay').hidden=false;$('progress-title').textContent=data.phase;$('progress-detail').textContent=data.detail;}else $('video-overlay').hidden=true;return;}if(data.type==='result'){state.pending?.resolve(data);}else if(data.type==='error')state.pending?.reject(new Error(data.message));state.pending=null;};
  worker.onerror=()=>{state.pending?.reject(new Error('The inference engine could not start. Try a current Chrome or Edge browser.'));state.pending=null;worker.terminate();state.worker=null;};state.worker=worker;return worker;
}
function infer(){
  const geometry=letterboxGeometry(video.videoWidth,video.videoHeight),inputCanvas=document.createElement('canvas');inputCanvas.width=1280;inputCanvas.height=1280;const ctx=inputCanvas.getContext('2d',{willReadFrequently:true});ctx.fillStyle='rgb(114,114,114)';ctx.fillRect(0,0,1280,1280);ctx.drawImage(video,geometry.left,geometry.top,geometry.resizedWidth,geometry.resizedHeight);const rgba=ctx.getImageData(0,0,1280,1280).data,pixels=1280*1280,input=new Float32Array(pixels*3);for(let i=0;i<pixels;i++){input[i]=rgba[i*4]/255;input[pixels+i]=rgba[i*4+1]/255;input[2*pixels+i]=rgba[i*4+2]/255;}
  return new Promise((resolve,reject)=>{state.pending={resolve,reject};try{detector().postMessage({type:'detect',input,geometry},[input.buffer]);}catch(error){state.pending=null;reject(error);}});
}
$('analyze-video').addEventListener('click',async()=>{
  if(state.busy||!state.url)return;video.pause();state.frames=[];state.cancel=false;state.complete=false;state.rate=Number($('analysis-rate').value);setBusy(true);$('cancel-analysis').disabled=false;$('cancel-analysis').textContent='Stop analysis';$('video-result-badge').textContent='ANALYZING';const times=sampleTimes(video.duration,state.rate),start=performance.now();
  try{for(const time of times){if(state.cancel)break;await seek(time);const result=await infer();state.frames.push({time,detections:result.detections,inferenceMs:result.elapsedMs,backend:result.backend});$('progress').value=state.frames.length/times.length;$('frame-progress').textContent=`${state.frames.length} / ${times.length} frames`;status(`Analyzing ${state.frames.length} of ${times.length} frames · ${Math.round($('progress').value*100)}%`);draw();}
    state.complete=state.frames.length===times.length;$('video-result-badge').textContent=state.complete?'COMPLETE':'PARTIAL';status(`${state.complete?'Complete':'Stopped'}: ${state.frames.length} frames analyzed in ${((performance.now()-start)/1000).toFixed(1)}s. Play or scrub to inspect detections.`);await seek(0);
  }catch(error){$('video-result-badge').textContent='PARTIAL';status(error.message,true);}finally{setBusy(false);$('video-overlay').hidden=true;draw();}
});
$('cancel-analysis').addEventListener('click',()=>{state.cancel=true;$('cancel-analysis').disabled=true;$('cancel-analysis').textContent='Stopping after this frame…';});
$('upload-video').addEventListener('click',()=>{if(!state.busy)$('video-file').click();});$('video-drop').addEventListener('click',()=>{if(!state.busy)$('video-file').click();});$('video-drop').addEventListener('keydown',event=>{if(['Enter',' '].includes(event.key)&&!state.busy){event.preventDefault();$('video-file').click();}});
$('video-file').addEventListener('change',event=>{const file=event.target.files[0];if(file)loadVideo(file,file.name);event.target.value='';});
for(const name of ['dragenter','dragover'])$('video-drop').addEventListener(name,event=>{event.preventDefault();if(!state.busy)$('video-drop').classList.add('dragging');});for(const name of ['dragleave','drop'])$('video-drop').addEventListener(name,event=>{event.preventDefault();$('video-drop').classList.remove('dragging');});$('video-drop').addEventListener('drop',event=>{if(event.dataTransfer.files[0])loadVideo(event.dataTransfer.files[0],event.dataTransfer.files[0].name);});
$('sample-video').addEventListener('click',async()=>{if(state.busy)return;try{status('Loading the aerial sample…');const response=await fetch('/assets/sample-aerial.mp4');if(!response.ok)throw new Error('Sample unavailable. Please upload a video.');await loadVideo(await response.blob(),'visdrone-sample.mp4');}catch(error){status(error.message,true);}});
$('play').addEventListener('click',async()=>{try{if(video.paused){if(video.ended)await seek(0);await video.play();}else video.pause();}catch(error){status(error.message,true);}});video.addEventListener('play',()=>{$('play').textContent='Ⅱ Pause';$('play').setAttribute('aria-label','Pause video');});video.addEventListener('pause',()=>{$('play').textContent='▷ Play';$('play').setAttribute('aria-label','Play video');});video.addEventListener('ended',()=>{$('play').textContent='▷ Play';});video.addEventListener('seeked',()=>draw());video.addEventListener('timeupdate',()=>draw());
$('timeline').addEventListener('input',()=>{if(!state.busy){video.pause();video.currentTime=Number($('timeline').value);}});$('video-confidence').addEventListener('input',()=>{$('video-confidence-value').value=`${$('video-confidence').value}%`;draw();});$('video-labels').addEventListener('change',()=>draw());
function playbackFrame(){if(!video.paused)draw(false);if(video.requestVideoFrameCallback)video.requestVideoFrameCallback(playbackFrame);}
if(video.requestVideoFrameCallback)video.requestVideoFrameCallback(playbackFrame);
function download(blob,name){const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
const basename=()=>state.name.replace(/\.[^.]+$/,'')||'aerial-video';
$('export-video-json').addEventListener('click',()=>download(new Blob([JSON.stringify({model:'VisDrone YOLOv8s',video:state.name,width:video.videoWidth,height:video.videoHeight,durationSeconds:video.duration,analysisRate:state.rate,complete:state.complete,confidenceThreshold:confidence(),coordinateFormat:'xyxy pixels',note:'Sampled-frame detections, no tracking IDs. Repeated detections are not unique objects.',frames:state.frames.map(frame=>({...frame,detections:visible(frame)}))},null,2)],{type:'application/json'}),`${basename()}-video-detections.json`));
$('export-video').addEventListener('click',async()=>{
  if(state.busy||!state.frames.length)return;const mime=['video/webm;codecs=vp9','video/webm;codecs=vp8','video/webm'].find(type=>MediaRecorder.isTypeSupported(type));if(!mime){status('This browser cannot export WebM. Export detections JSON instead.',true);return;}
  video.pause();state.exporting=true;setBusy(true);$('video-overlay').hidden=false;$('progress-title').textContent='Exporting annotated clip';$('progress-detail').textContent='Creating a silent video from analyzed frames. Keep this tab open.';let stream,recorder;
  try{stream=canvas.captureStream(0);const track=stream.getVideoTracks()[0];if(!track.requestFrame)throw new Error('Your browser does not support sampled-frame recording. Export JSON instead.');const chunks=[];recorder=new MediaRecorder(stream,{mimeType:mime,videoBitsPerSecond:5000000});recorder.ondataavailable=event=>{if(event.data.size)chunks.push(event.data);};const stopped=new Promise((resolve,reject)=>{recorder.onstop=resolve;recorder.onerror=()=>reject(new Error('Video recording failed.'));});recorder.start();
    for(let i=0;i<state.frames.length;i++){await seek(state.frames[i].time);draw(false);track.requestFrame();$('progress-detail').textContent=`Exporting frame ${i+1} of ${state.frames.length}…`;await new Promise(resolve=>setTimeout(resolve,1000/state.rate));}
    recorder.stop();await stopped;download(new Blob(chunks,{type:'video/webm'}),`${basename()}-annotated.webm`);status('Annotated sampled-frame clip saved. The export has no audio.');await seek(0);
  }catch(error){if(recorder?.state==='recording')recorder.stop();status(error.message,true);}finally{stream?.getTracks().forEach(track=>track.stop());state.exporting=false;setBusy(false);$('video-overlay').hidden=true;draw();}
});
window.addEventListener('beforeunload',()=>{if(state.url)URL.revokeObjectURL(state.url);state.worker?.terminate();});
