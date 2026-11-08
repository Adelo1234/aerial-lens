export function sampleTimes(duration,rate){
  if(!Number.isFinite(duration)||duration<=0||duration>60)throw new Error('Choose a video between 0 and 60 seconds long.');
  if(![2,5,10].includes(rate))throw new Error('Unsupported analysis rate.');
  return Array.from({length:Math.ceil(duration*rate)},(_,i)=>i/rate).filter(t=>t<duration);
}
export function nearestFrame(frames,time,rate){
  if(!frames.length)return null;
  let low=0,high=frames.length-1;
  while(low<high){const middle=Math.floor((low+high)/2);if(frames[middle].time<time)low=middle+1;else high=middle;}
  const best=low>0&&Math.abs(frames[low-1].time-time)<=Math.abs(frames[low].time-time)?frames[low-1]:frames[low];
  return Math.abs(best.time-time)<=1/rate+.001?best:null;
}
