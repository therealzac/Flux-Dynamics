// ===========================================================================
// COLOUR — requirements 4 and 5 of 080726/PROBLEM.md
// ===========================================================================
// The walk decides WHICH tet is active; the colouring decides WHAT it is.
// Colour is constant across an activation, so with 8 activations per lap the
// whole colour problem is a choice of one of 24 options per activation.
//
// A colouring is a bijection from the octahedron's three axes to the three
// colour/anti pairs. The pole axis is forced -- its pair is the one whose
// primary is the tet's colour, oriented so the touching pole is primary. The
// two equator axes take the remaining two pairs: 2 allocations x 2 x 2
// orientations = 8, times 3 tet colours = 24.
'use strict';
const G=require('./gen.js');
const {ACT,VP,POLE}=G;
// 0 R, 1 G, 2 B, 3 C(anti-R), 4 M(anti-G), 5 Y(anti-B)
const NAME=['R','G','B','C','M','Y'];
const anti=c=>(c+3)%6;
// The tet's own face carries TWO equator vertices -- the x-partner and the
// y-partner of its apex -- and they are ADJACENT on the equator square, not
// antipodal. Zac's rule: the equator edge touching the actualized tet must be
// a colour/anti pair. So pairing runs ALONG that face edge, not across the
// octahedron, and which vertices are paired depends on which tet is live.
const poleOfTet=t=>ACT[t][1];                        // 4 or 5
const faceEquator=t=>{const a=ACT[t][0];
  return [VP[a][0]>0?0:1, VP[a][1]>0?2:3];};         // x-partner, y-partner
const otherEquator=t=>{const f=faceEquator(t);
  return [0,1,2,3].filter(v=>!f.includes(v));};

// every colouring for tet t: 3 tet colours x (2 pair choices x 2 x 2) = 24
function colourings(t){
  const P=poleOfTet(t), other=P===4?5:4;
  const [e1,e2]=faceEquator(t), [f1,f2]=otherEquator(t), out=[];
  for(let c=0;c<3;c++){
    const rest=[0,1,2].filter(x=>x!==c);
    for(const [onFace,onRest] of [[rest[0],rest[1]],[rest[1],rest[0]]])
      for(const o1 of [0,1]) for(const o2 of [0,1]){
        const v=new Array(6);
        v[P]=c; v[other]=anti(c);
        // the face edge is a colour/anti pair -- the crucial constraint
        v[o1?e2:e1]=onFace; v[o1?e1:e2]=anti(onFace);
        v[o2?f2:f1]=onRest; v[o2?f1:f2]=anti(onRest);
        out.push({tet:c, vert:v}); }
  }
  return out;
}

// exact search: K laps x 8 activations, every tet R/G/B equally, every vertex
// all six equally. Backtracking -- the space is small and the answer is exact.
function solve(order,K){
  const N=order.length*K, opts=order.map(colourings);
  const perTet=K/3;                                   // times each tet takes each colour
  const perVert=N/6;                                  // times each vertex takes each colour
  const tetCount=new Map(), vertCount=[0,1,2,3,4,5].map(()=>new Array(6).fill(0));
  for(const t of order) tetCount.set(t,[0,0,0]);
  const pick=new Array(N).fill(null);
  let nodes=0;
  function rec(i){
    // A SUCCESSFUL SOLVE TAKES ~54 NODES. The old 4e6 budget meant an
    // unlucky random ordering burned seconds before admitting defeat, which
    // made searching many colourings absurdly slow. Fail fast and redraw.
    if(++nodes>2e4) return false;
    if(i===N){ return true; }
    const t=order[i%order.length], list=opts[i%order.length];
    // try in a shuffled order so repeated runs give different valid solutions
    const idx=list.map((_,j)=>j);
    for(let j=idx.length-1;j>0;j--){const k=Math.random()*(j+1)|0;[idx[j],idx[k]]=[idx[k],idx[j]];}
    for(const j of idx){ const C=list[j];
      if(tetCount.get(t)[C.tet]+1>perTet) continue;
      let ok=true;
      for(let v=0;v<6;v++) if(vertCount[v][C.vert[v]]+1>perVert){ok=false;break;}
      if(!ok) continue;
      tetCount.get(t)[C.tet]++; for(let v=0;v<6;v++) vertCount[v][C.vert[v]]++;
      pick[i]=C;
      if(rec(i+1)) return true;
      tetCount.get(t)[C.tet]--; for(let v=0;v<6;v++) vertCount[v][C.vert[v]]--;
      pick[i]=null; }
    return false; }
  return rec(0)?{pick,nodes}:null;
}
// how evenly, over sliding windows of activations, does each thing sit?
function balance(order,pick,K){
  const N=pick.length;
  const per={};
  for(const w of [3,6,12,24]){
    if(w>N){per[w]=null;continue;}
    let tetSE=0,tetN=0,vSE=0,vN=0;
    for(let i=0;i<N;i++){
      const tc=[0,0,0], vc=[0,1,2,3,4,5].map(()=>new Array(6).fill(0));
      for(let k=0;k<w;k++){ const C=pick[(i+k)%N];
        tc[C.tet]++; for(let v=0;v<6;v++) vc[v][C.vert[v]]++; }
      for(let c=0;c<3;c++){const d=tc[c]/w-1/3; tetSE+=d*d; tetN++;}
      for(let v=0;v<6;v++) for(let c=0;c<6;c++){const d=vc[v][c]/w-1/6; vSE+=d*d; vN++;} }
    per[w]={tet:Math.sqrt(tetSE/tetN), vert:Math.sqrt(vSE/vN)}; }
  return per;
}
// The exact solver returns the FIRST valid colouring. There are many, and they
// differ enormously in window balance, so search them: draw valid colourings
// (the backtracker is randomised) and keep the one whose sliding windows are
// flattest. Global balance is exact in every candidate -- only the timing moves.
const wScore=B=>{let s=0,n=0;
  for(const w of [3,6,12,24]){ const b=B[w]; if(!b) continue; s+=b.tet+b.vert; n++; }
  return n?s/n:1e9;};
function best(order,K,tries){
  let bp=null,bs=Infinity,bb=null,found=0;
  for(let i=0;i<tries;i++){ const S=solve(order,K); if(!S) continue; found++;
    const B=balance(order,S.pick,K), sc=wScore(B);
    if(sc<bs){bs=sc;bp=S.pick;bb=B;} }
  return bp?{pick:bp,score:bs,balance:bb,found}:null;
}
module.exports={colourings,solve,balance,best,wScore,NAME,anti,poleOfTet};

if(require.main===module){
  const walk=require('/tmp/p87lap.json').walk;
  // recover the activation order from the walk
  const order=[]; const n=walk.length;
  for(let i=0;i<n;i++){ const u=walk[i],v=walk[(i+1)%n];
    if(G.isAct(u,v)){ const a=G.isApex(u)?u:v; order.push(ACT.findIndex(e=>e[0]===a)); } }
  console.log('activation order      ', order.join(' '));
  console.log('pole touched by each  ', order.map(poleOfTet).join(' '),
              '  alternates:', order.every((t,i)=>poleOfTet(t)!==poleOfTet(order[(i+1)%order.length])));
  console.log('colourings per activation', colourings(order[0]).length);
  for(const K of [3,6]){
    const t0=Date.now(); const S=solve(order,K);
    if(!S){ console.log('K='+K,' no exact colouring found'); continue; }
    const B=balance(order,S.pick,K);
    console.log('\nK='+K+' laps  ('+(order.length*K)+' activations, '+(Date.now()-t0)+'ms, '+S.nodes+' nodes)');
    console.log('  tet colours by activation:', S.pick.map(c=>NAME[c.tet]).join(''));
    console.log('  vertex 0 over time       :', S.pick.map(c=>NAME[c.vert[0]]).join(''));
    console.log('  pole 4 over time         :', S.pick.map(c=>NAME[c.vert[4]]).join(''));
    for(const [w,b] of Object.entries(B)) if(b)
      console.log('   window '+String(w).padStart(2)+'  tet rms '+(100*b.tet).toFixed(2)+
                  '%   vertex rms '+(100*b.vert).toFixed(2)+'%');
  }
}
