import assert from 'node:assert/strict';
import {MotionTimeline} from '../uav_llm_demo/static/render_motion.js';

function source(t) {
  return {t,position:[t,0,2],velocity:[3,0,0],yaw:Math.PI/2,landed:false,path_index:Math.round(t*20),
    traffic:{bird:[t+3,0,2,1,0,0]}};
}
function replay(packets,frames=[1/60],make=source) {
  const timeline=new MotionTimeline();let now=0,next=0,n=0,frameIndex=0,previous=null;
  const velocities=[],samples=[];
  while(now<15) {
    const dt=frames[frameIndex++%frames.length];now+=dt;
    if(now+1e-9>=next) {
      const last=Math.floor((now+1e-9)*20);
      const data=Array.from({length:Math.min(32,last+1)},(_,i)=>make((last-Math.min(31,last)+i)/20));
      timeline.ingest(data,now);next=now+packets[n++%packets.length];
    }
    const current=timeline.sample(now);
    if(current){
      if(previous&&now>3)velocities.push((current.position[0]-previous.position[0])/dt);
      assert.ok(current.time<=timeline.samples.at(-1).t+1e-8,'must never extrapolate');
      assert.ok(Math.abs(current.traffic.bird[0]-current.position[0]-3)<1e-8,'UAV and bird clocks split');
      samples.push(current);previous=current;
    }
  }
  const mean=velocities.reduce((a,b)=>a+b,0)/velocities.length;
  const cv=Math.sqrt(velocities.reduce((sum,v)=>sum+(v-mean)**2,0)/velocities.length)/mean;
  assert.ok(Math.min(...velocities)>.9&&Math.max(...velocities)<1.1,JSON.stringify({packets,frames,min:Math.min(...velocities),max:Math.max(...velocities)}));
  assert.ok(cv<.025,`velocity CV ${cv}`);
  assert.ok(timeline.samples.length<=96);
  return {packetsMs:packets.map(t=>t*1000),minSpeed:+Math.min(...velocities).toFixed(4),maxSpeed:+Math.max(...velocities).toFixed(4),cv:+cv.toFixed(5)};
}
const timing=[replay([.125]),replay([.16]),replay([.125,.25,.15,.21]),replay([.125,.18],[1/30,1/60,1/120]),
  replay([.125],[.2]),replay([.125],[1/60,1/60,1/60,.2])];

const timeline=new MotionTimeline();
const corner=Array.from({length:31},(_,i)=>{const s=source(i*.05);s.position=[Math.min(s.t,1),Math.max(0,s.t-1),2];return s;});
timeline.ingest(corner,0);
for(let t=.8;t<1.21;t+=.003) {
  const p=timeline.at(t).position;
  assert.ok(p[1]===0||Math.abs(p[0]-1)<1e-10,'interpolation cut across the actual corner');
}
assert.ok(timeline.preview().every(p=>p[0]===1),'preview must stay on confirmed control segments');
const saved=timeline.samples.at(-1).position[0];corner.at(-1).position[0]=999;
assert.equal(timeline.samples.at(-1).position[0],saved,'input aliases stored telemetry');
const before=timeline.samples.length;timeline.ingest(null,1);timeline.ingest([{t:NaN}],1);timeline.ingest([source(.1)],1);
assert.equal(timeline.samples.length,before,'stale or invalid packet replaced current samples');
for(let i=0;i<120;i++)timeline.sample(i/60);
const frozen=timeline.sample(3);assert.equal(frozen.time,1.5);
assert.deepEqual(timeline.sample(4).position,frozen.position,'lost connection invented movement');
const newPacket=Array.from({length:32},(_,i)=>source(8+i*.05));
timeline.ingest(newPacket,4);assert.equal(timeline.sample(4).reset,true,'unobserved gap needs explicit reset');
assert.ok(timeline.sample(4.01).time>=8,'gap interpolated across an unknown trajectory');
timeline.reset();assert.equal(timeline.sample(5),null);

const stopped=new MotionTimeline();
stopped.ingest(Array.from({length:32},(_,i)=>({...source(i*.05),position:[0,0,0],velocity:[0,0,0],landed:true})),0);
const first=stopped.sample(0),second=stopped.sample(.1);
assert.deepEqual(first.position,second.position);assert.ok(second.traffic.bird[0]>first.traffic.bird[0]);
const yaw=new MotionTimeline();yaw.ingest([{...source(0),yaw:3.1},{...source(.05),yaw:-3.1}],0);
assert.ok(Math.abs(Math.abs(yaw.at(.025).yaw)-Math.PI)<1e-8,'yaw took the long rotation');
console.log(JSON.stringify({ok:true,checks:['constant speed','packet jitter','variable refresh rate','shared UAV/bird clock','control corner','packet ownership','stale/invalid packet','disconnect freeze','gap/reset','stationary UAV with moving birds','yaw wrap'],timing},null,2));
