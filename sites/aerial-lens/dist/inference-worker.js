import * as ort from './vendor/ort.webgpu.min.mjs';
import {decodeDetections} from './detection.js';
ort.env.wasm.numThreads=1;
ort.env.wasm.wasmPaths=new URL('./vendor/',import.meta.url).href;
let session=null,modelBytes=null,backend='wasm';
const progress=(phase,detail)=>self.postMessage({type:'progress',phase,detail});
async function loadModel(){
  if(session)return;
  progress('Loading your model','Downloading the trained detector. This is only needed on first use.');
  const metadataResponse=await fetch('./assets/model.json');
  if(!metadataResponse.ok)throw new Error('Could not load the model manifest. Please refresh and retry.');
  const metadata=await metadataResponse.json();
  let completed=0;
  const parts=await Promise.all(metadata.parts.map(async path=>{
    const response=await fetch(path);if(!response.ok)throw new Error('Model download failed. Check your connection and retry.');
    const bytes=new Uint8Array(await response.arrayBuffer());completed++;
    progress('Loading your model',`Downloaded ${completed} of ${metadata.parts.length} model parts.`);return bytes;
  }));
  modelBytes=new Uint8Array(parts.reduce((sum,p)=>sum+p.length,0));let offset=0;
  for(const part of parts){modelBytes.set(part,offset);offset+=part.length;}
  if(modelBytes.length!==metadata.bytes)throw new Error('The model download was incomplete. Refresh and retry.');
  const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',modelBytes))).map(n=>n.toString(16).padStart(2,'0')).join('');
  if(hash!==metadata.sha256)throw new Error('The model integrity check failed. Refresh and retry.');
  progress('Preparing the detector','Initializing browser inference…');
  let gpuError=null;
  if(self.navigator.gpu){
    try{session=await ort.InferenceSession.create(modelBytes,{executionProviders:['webgpu','wasm'],graphOptimizationLevel:'all'});backend='webgpu';}catch(error){gpuError=error;session=null;}
  }
  if(!session){try{session=await ort.InferenceSession.create(modelBytes,{executionProviders:['wasm'],graphOptimizationLevel:'all'});backend='wasm';}catch(error){throw new Error('Model initialization failed: '+(gpuError?.message||error.message));}}
}
self.onmessage=async event=>{
  if(event.data.type!=='detect')return;
  try{
    await loadModel();progress('Finding objects',backend==='webgpu'?'Running on your browser GPU.':'Running on your device. CPU inference can take a little longer.');
    const started=performance.now(),input=new ort.Tensor('float32',event.data.input,[1,3,1280,1280]);
    let output;
    try{output=await session.run({[session.inputNames[0]]:input});}
    catch(error){
      if(backend!=='webgpu')throw error;
      progress('Switching to CPU','Your browser GPU could not run this model. Trying the CPU fallback.');
      await session.release();session=await ort.InferenceSession.create(modelBytes,{executionProviders:['wasm'],graphOptimizationLevel:'all'});backend='wasm';
      output=await session.run({[session.inputNames[0]]:input});
    }
    const tensor=output[session.outputNames[0]],detections=decodeDetections(tensor.data,tensor.dims,event.data.geometry);
    input.dispose();for(const tensor of Object.values(output))tensor.dispose();
    self.postMessage({type:'result',detections,elapsedMs:performance.now()-started,backend});
  }catch(error){self.postMessage({type:'error',message:error.message||'Detection failed. Please try a smaller image or refresh the page.'});}
};
