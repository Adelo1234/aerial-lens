export const CLASS_NAMES = ['pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor'];
export const COLORS = ['#bdeb83','#6edac6','#d7a5ff','#83bef9','#f3cb79','#ffab84','#f4a5cd','#b5b4ff','#70d9a0','#e4db8e'];
export function letterboxGeometry(width,height,size=1280){
  const gain=Math.min(size/width,size/height);
  const resizedWidth=Math.round(width*gain),resizedHeight=Math.round(height*gain);
  return {width,height,gain,resizedWidth,resizedHeight,left:Math.max(0,Math.round((size-resizedWidth)/2-.1)),top:Math.max(0,Math.round((size-resizedHeight)/2-.1))};
}
export function iou(a,b){
  const area=Math.max(0,Math.min(a.x2,b.x2)-Math.max(a.x1,b.x1))*Math.max(0,Math.min(a.y2,b.y2)-Math.max(a.y1,b.y1));
  return area/((a.x2-a.x1)*(a.y2-a.y1)+(b.x2-b.x1)*(b.y2-b.y1)-area+1e-9);
}
export function decodeDetections(data,dims,geometry,floor=.05,nms=.7){
  if(dims.length!==3||dims[0]!==1||dims[1]!==14) throw new Error('The model returned an unexpected output shape.');
  const count=dims[2],all=[];
  for(let i=0;i<count;i++){
    let score=0,classId=0;
    for(let c=0;c<10;c++){const value=data[(c+4)*count+i];if(value>score){score=value;classId=c;}}
    if(score<floor||!Number.isFinite(score))continue;
    const cx=data[i],cy=data[count+i],w=data[2*count+i],h=data[3*count+i];
    const clip=(v,max)=>Math.max(0,Math.min(max,v));
    const x1=clip((cx-w/2-geometry.left)/geometry.gain,geometry.width),y1=clip((cy-h/2-geometry.top)/geometry.gain,geometry.height);
    const x2=clip((cx+w/2-geometry.left)/geometry.gain,geometry.width),y2=clip((cy+h/2-geometry.top)/geometry.gain,geometry.height);
    if([x1,y1,x2,y2].every(Number.isFinite)&&x2>x1&&y2>y1)all.push({classId,className:CLASS_NAMES[classId],confidence:score,x1,y1,x2,y2});
  }
  all.sort((a,b)=>b.confidence-a.confidence);
  const kept=[];
  for(const box of all){if(!kept.some(other=>other.classId===box.classId&&iou(box,other)>nms))kept.push(box);if(kept.length===1000)break;}
  return kept;
}
