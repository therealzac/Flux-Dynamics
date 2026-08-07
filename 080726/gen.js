// ===========================================================================
// GENERATOR for 080726/PROBLEM.md
// ===========================================================================
// Requirement 1 by CONSTRUCTION: a tet is active from its activating edge
// until the next, so eight activations spaced L apart give every tet exactly
// L ticks. A lap is therefore 8 segments of L steps, each opening with one
// tet's activating edge and touching no other activating edge until the next
// segment opens. Inside a segment the walk is otherwise free -- including
// straight through other apexes, which activate only on their POLE edge.
//
// EVERY TURN IS COUNTED EXACTLY ONCE. Segment i owns the L turns at nodes
// V[iL] .. V[iL+L-1]. The turn at its first node needs the node the walk
// ARRIVED from, so the arrival node is carried across the boundary as part of
// the state. Getting this wrong leaves 16 turns per lap unconstrained, which
// is what dragged the ratio to 0.64 instead of 0.667.
'use strict';
const A=1/Math.SQRT2, VP=[];
for(let a=0;a<3;a++) for(const s of [1,-1]){const p=[0,0,0];p[a]=s*A;VP.push(p);}
for(const x of [1,-1]) for(const y of [1,-1]) for(const z of [1,-1]) VP.push([x*A,y*A,z*A]);
const NV=14, APEX=[6,7,8,9,10,11,12,13], POLE=[4,5];
const isApex=i=>i>=6;
const D=(i,j)=>Math.hypot(VP[i][0]-VP[j][0],VP[i][1]-VP[j][1],VP[i][2]-VP[j][2]);
const ADJ=VP.map((_,i)=>VP.map((_,j)=>j).filter(j=>j!==i&&Math.abs(D(i,j)-1)<1e-9));
const TN=new Int8Array(NV*NV*NV).fill(-1);
for(let u=0;u<NV;u++)for(let v=0;v<NV;v++)for(let w=0;w<NV;w++){
  if(!ADJ[v].includes(u)||!ADJ[v].includes(w))continue;
  const L=D(u,w); TN[(u*NV+v)*NV+w]=Math.abs(L-1)<1e-9?60:Math.abs(L-Math.SQRT2)<1e-9?90
    :Math.abs(L-2)<1e-9?0:-1;}
const tn=(u,v,w)=>TN[(u*NV+v)*NV+w];
const DV=t=>t===90?1:t===60?-2:0;                 // balance: n90 - 2*n60
const poleOf=a=>VP[a][2]>0?4:5;
const ACT=APEX.map(a=>[a,poleOf(a)]);             // 8 activating edges, 4 per pole
const isAct=(u,v)=>(isApex(u)&&v===poleOf(u))||(isApex(v)&&u===poleOf(v));
const EK=(u,v)=>Math.min(u,v)*NV+Math.max(u,v);

// Layers for one segment. Starts having ARRIVED at `from` from `arrive`;
// the first move is forced along the activating edge.
// `gap` is the minimum spacing demanded between consecutive 60s inside a
// segment. The balanced word wants every gap to be exactly 3, and hoping an
// annealer stumbles onto that in a space of 10^68 does not work -- so it is
// carried in the state instead. `g` counts steps since the last 60, capped,
// and a 60 is refused while g < gap-1. The two 60s an activation forces sit
// outside the free steps and are not subject to it.
function segLayers(t,e,arrive,L,gap){
  const GC=Math.max(1,gap||1);
  const from=ACT[t][e], other=ACT[t][1-e];
  const VLO=-2*(L+2), VR=3*(L+2)+1;
  const key=(p,c,v,g)=>((((p*NV+c)*VR)+(v-VLO))*GC)+g;
  const t0=tn(arrive,from,other);
  if(t0<0) return null;
  // START READY. The two 60s an activation forces -- one at the apex, one at
  // the pole, since every non-activating edge off a pole is `apex+oct` and so
  // 60 or refused -- cannot obey any spacing rule. Beginning the counter at 0
  // refused them and emptied the DP outright. The constraint applies to the
  // FREE 60s, which are the ones actually available to place.
  let layer=new Map([[key(from,other,DV(t0),GC-1),1]]);
  const out=[layer];
  for(let s=2;s<L;s++){                            // free steps
    const nx=new Map();
    for(const [k,n] of layer){ const g=k%GC, k2=(k-g)/GC;
      const v=k2%VR+VLO, r=(k2-(k2%VR))/VR, c=r%NV, p=(r-c)/NV;
      for(const w of ADJ[c]){ if(isAct(c,w)) continue;
        const tt=tn(p,c,w); if(tt<0) continue;
        if(tt===60 && g<GC-1) continue;            // too soon for another 60
        const ng = tt===60 ? 0 : Math.min(GC-1,g+1);
        const nk=key(c,w,v+DV(tt),ng); nx.set(nk,(nx.get(nk)||0)+n); } }
    layer=nx; out.push(layer); }
  return {layers:out,VLO,VR,GC};
}
const dec=(k,VLO,VR,GC)=>{GC=GC||1; const g=k%GC, k2=(k-g)/GC;
  const v=k2%VR+VLO,r=(k2-(k2%VR))/VR,c=r%NV,p=(r-c)/NV;return{v,c,p,g};};

// Closing step: from the last free node, take the next tet's activating edge.
// That turn belongs to THIS segment, so it is added here.
function segTrans(t,e,arrive,L,targetV,gap){
  const S=segLayers(t,e,arrive,L,gap); if(!S) return new Map();
  const {layers,VLO,VR,GC}=S, last=layers[layers.length-1], out=new Map();
  for(const [k,n] of last){ const {v,c,p}=dec(k,VLO,VR,GC);
    for(let t2=0;t2<8;t2++){ if(t2===t) continue;
      for(let e2=0;e2<2;e2++){ const nf=ACT[t2][e2];
        if(!ADJ[c].includes(nf)) continue;
        // THE CONNECTING STEP MUST NOT ACTIVATE. If c is an apex and nf its
        // pole, this edge fires c's tet -- an extra, unscheduled activation.
        // Measured before this check: 9 activations per lap, spaced
        // 30,30,30,29,1,30,30,30,30, which is exactly the tet spread of 2 that
        // requirement 1 is supposed to make structurally impossible.
        if(isAct(c,nf)) continue;
        const tt=tn(p,c,nf); if(tt<0) continue;
        if(targetV!==undefined && v+DV(tt)!==targetV) continue;
        const kk=t2+'|'+e2+'|'+c;                  // c is the next arrival node
        out.set(kk,(out.get(kk)||0)+n); } } }
  return out;
}
// Uniform path within a segment realising a chosen closing transition.
function segPath(t,e,arrive,L,t2,e2,exit,rnd,targetV,gap){
  const S=segLayers(t,e,arrive,L,gap); if(!S) return null;
  const {layers,VLO,VR,GC}=S, last=layers[layers.length-1];
  const nf=ACT[t2][e2], fin=[]; let tot=0;
  for(const [k,n] of last){ const {v,c,p}=dec(k,VLO,VR,GC);
    if(c!==exit||isAct(c,nf)) continue;
    const tt=tn(p,c,nf); if(tt<0) continue;
    if(targetV!==undefined && v+DV(tt)!==targetV) continue;
    fin.push([k,n]); tot+=n; }
  if(!tot) return null;
  let r=rnd()*tot,pick=fin[fin.length-1];
  for(const f of fin){ r-=f[1]; if(r<=0){pick=f;break;} }
  const path=[]; let k=pick[0];
  for(let s=layers.length-1;s>=1;s--){ const st=dec(k,VLO,VR,GC); path.unshift(st.c);
    const cands=[]; let sum=0;
    for(const [pk,n] of layers[s-1]){ const q=dec(pk,VLO,VR,GC);
      if(q.c!==st.p||isAct(q.c,st.c)) continue;
      const tt=tn(q.p,q.c,st.c); if(tt<0||q.v+DV(tt)!==st.v) continue;
      if(tt===60&&q.g<GC-1) continue;
      const ng=tt===60?0:Math.min(GC-1,q.g+1); if(ng!==st.g) continue;
      cands.push([pk,n]); sum+=n; }
    if(!sum) return null;
    let rr=rnd()*sum,np=cands[cands.length-1][0];
    for(const c2 of cands){ rr-=c2[1]; if(rr<=0){np=c2[0];break;} }
    k=np; }
  path.unshift(dec(k,VLO,VR,GC).c);
  return [ACT[t][e], ...path];                     // L nodes: from, other, ...
}
module.exports={NV,APEX,POLE,ADJ,tn,DV,isApex,isAct,ACT,poleOf,EK,VP,D,
                segLayers,segTrans,segPath};
