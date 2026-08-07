// Solver for 080726/PROBLEM.md -- built on the statement, not on assumptions.
'use strict';
const A = 1/Math.SQRT2, VP = [];
for (let a=0;a<3;a++) for (const s of [1,-1]) { const p=[0,0,0]; p[a]=s*A; VP.push(p); }
for (const x of [1,-1]) for (const y of [1,-1]) for (const z of [1,-1]) VP.push([x*A,y*A,z*A]);
const NV=14, OCT=[0,1,2,3,4,5], APEX=[6,7,8,9,10,11,12,13];
const isApex=i=>i>=6;
const D=(i,j)=>Math.hypot(VP[i][0]-VP[j][0],VP[i][1]-VP[j][1],VP[i][2]-VP[j][2]);
const ADJ=VP.map((_,i)=>VP.map((_,j)=>j).filter(j=>j!==i&&Math.abs(D(i,j)-1)<1e-9));
const turn=(u,v,w)=>{const L=D(u,w);
  return Math.abs(L-1)<1e-9?60:Math.abs(L-Math.SQRT2)<1e-9?90:Math.abs(L-2)<1e-9?0:null;};
// the pole of an apex is its z-partner; that edge is the apex edge carrying its tet
const poleOf=a=>VP[a][2]>0?4:5;
const isApexEdge=(u,v)=>(isApex(u)&&v===poleOf(u))||(isApex(v)&&u===poleOf(v));
const tetOf=(u,v)=>isApex(u)?u:v;
const EK=(u,v)=>Math.min(u,v)*NV+Math.max(u,v);
const EDGES=[]; for(let i=0;i<NV;i++) for(const j of ADJ[i]) if(i<j) EDGES.push(EK(i,j));

function evaluate(W){
  const n=W.length, T=[];
  for(let i=0;i<n;i++){ if(!ADJ[W[i]].includes(W[(i+1)%n])) return null;
    const t=turn(W[(i-1+n)%n],W[i],W[(i+1)%n]); if(t===null) return null; T.push(t); }
  // requirement 1 -- tet active time, from the walk alone
  const act=new Map(APEX.map(a=>[a,0])); let live=null;
  for(let pass=0;pass<2;pass++) for(let i=0;i<n;i++){
    const u=W[i], v=W[(i+1)%n];
    if(isApexEdge(u,v)) live=tetOf(u,v);
    if(pass&&live!==null) act.set(live,act.get(live)+1); }
  const ticks=[...act.values()];
  // requirement 2 -- 2/3 at every scale, over the 90/60 ledger only
  const x=[]; for(const t of T){ if(t===90)x.push(1); else if(t===60)x.push(0); }
  const m=x.length, pre=new Int32Array(2*m+1);
  for(let i=0;i<2*m;i++) pre[i+1]=pre[i]+x[i%m];
  const per={}; let sum=0;
  for(const w of [3,6,12,25,50,100]){ if(w>m){per[w]=0;continue;}
    let se=0; for(let i=0;i<m;i++){const d=(pre[i+w]-pre[i])/w-2/3; se+=d*d;}
    per[w]=Math.sqrt(se/m); sum+=per[w]; }
  // requirement 3 -- coverage and its evenness
  const use=new Map(EDGES.map(e=>[e,0]));
  for(let i=0;i<n;i++) use.set(EK(W[i],W[(i+1)%n]), use.get(EK(W[i],W[(i+1)%n]))+1);
  const c=[...use.values()], mu=c.reduce((a,b)=>a+b,0)/c.length;
  const cv=Math.sqrt(c.reduce((a,b)=>a+(b-mu)**2,0)/c.length)/(mu||1);
  const n90=T.filter(t=>t===90).length, n60=T.filter(t=>t===60).length;
  return { T, n90, n60, n0:T.filter(t=>t===0).length, ratio:n90/(n90+n60),
    tetTicks:ticks, tetSpread:Math.max(...ticks)-Math.min(...ticks),
    tetNever:ticks.filter(v=>v===0).length,
    disc:{per, mean:sum/6, worst:Math.max(...Object.values(per))},
    edgesUsed:c.filter(v=>v>0).length, edgeCV:cv };
}
module.exports={ NV,OCT,APEX,ADJ,turn,isApex,poleOf,isApexEdge,tetOf,EK,EDGES,evaluate,VP };

if (require.main===module){
  console.log('graph      ', NV,'vertices,',EDGES.length,'edges');
  console.log('degrees    ', 'oct',ADJ[0].length,' apex',ADJ[6].length);
  const sc=EDGES.filter(e=>{const u=(e/NV)|0,v=e%NV; return isApexEdge(u,v);});
  console.log('apex edges ', sc.length,'(one per tet)');
  // the forced-60 fact, checked rather than asserted
  let bad=0; for(const a of APEX) for(const u of ADJ[a]) for(const w of ADJ[a])
    if(u!==w && turn(u,a,w)!==60) bad++;
  console.log('turns at an apex that are NOT 60:', bad);
  // ring vertices: two mutually orthogonal shortcut directions
  for(const v of OCT){ const ap=ADJ[v].filter(isApex).length, oc=ADJ[v].filter(x=>!isApex(x)).length;
    console.log('  oct vertex',v,' apex-nbrs',ap,' oct-nbrs',oc,' pole:',v===4||v===5); }
}
