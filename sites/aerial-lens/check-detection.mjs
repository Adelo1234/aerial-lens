import assert from 'node:assert/strict';
import fs from 'node:fs';
import {decodeDetections,letterboxGeometry} from './dist/detection.js';
const geometry=letterboxGeometry(1920,1080);
assert.equal(geometry.gain,2/3);assert.equal(geometry.left,0);assert.equal(geometry.top,280);
// Overlapping predictions of one class collapse; a different class remains.
const count=3,data=new Float32Array(14*count);
for(let i=0;i<count;i++){data[i]=640;data[count+i]=640;data[2*count+i]=200;data[3*count+i]=100;}
data[(4+3)*count]=.9;data[(4+3)*count+1]=.8;data[(4+8)*count+2]=.85;
const boxes=decodeDetections(data,[1,14,count],geometry);
assert.equal(boxes.length,2);assert.deepEqual(boxes.map(b=>b.className),['car','bus']);
assert.ok(Math.abs(boxes[0].x1-810)<1e-6);assert.ok(Math.abs(boxes[0].y1-465)<1e-6);
assert.equal(decodeDetections(new Float32Array(14),[1,14,1],geometry).length,0);
const metadata=JSON.parse(fs.readFileSync(new URL('./dist/assets/model.json',import.meta.url)));
assert.equal(metadata.parts.reduce((sum,part)=>sum+fs.statSync(new URL('./dist/'+part,import.meta.url)).size,0),metadata.bytes);
console.log('Passed: coordinate restoration, class-aware NMS, empty detections, model parts.');
