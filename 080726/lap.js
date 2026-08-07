// Assemble, sample and anneal a full lap for 080726/PROBLEM.md
'use strict';
const G=require('./gen.js');
const {NV,ACT,ADJ,tn,isAct,isApex,EK,segTrans,segPath}=G;
const SCALES=[3,6,12,25,50,100];

function measure(W){
  const n=W.length,T=[];
  for(let i=0;i<n;i++){ if(!ADJ[W[i]].includes(W[(i+1)%n])) return null;
    const t=tn(W[(i-1+n)%n],W[i],W[(i+1)%n]); if(t<0) return null; T.push(t); }
  const tetOf=(u,v)=>{const a=isApex(u)?u:v; return ACT.findIndex(e=>e[0]===a);};
  const act=new Map(); for(let t=0;t<8;t++) act.set(t,0);
  let live=null;
  for(let pass=0;pass<2;pass++) for(let i=0;i<n;i++){
    const u=W[i],v=W[(i+1)%n];
    if(isAct(u,v)) live=tetOf(u,v);
    if(pass&&live!==null) act.set(live,act.get(live)+1); }
  const ticks=[...act.values()];
  const x=[]; for(const t of T){ if(t===90)x.push(1); else if(t===60)x.push(0); }
  const m=x.length, pre=new Int32Array(2*m+1);
  for(let i=0;i<2*m;i++) pre[i+1]=pre[i]+x[i%m];
  const per={}; let sum=0;
  for(const w of SCALES){ if(w>m){per[w]=0;continue;} let se=0;
    for(let i=0;i<m;i++){const d=(pre[i+w]-pre[i])/w-2/3; se+=d*d;}
    per[w]=Math.sqrt(se/m); sum+=per[w]; }
  const use=new Map();
  for(let i=0;i<n;i++){const k=EK(W[i],W[(i+1)%n]); use.set(k,(use.get(k)||0)+1);}
  const c=[...use.values()], mu=c.reduce((a,b)=>a+b,0)/c.length;
  const cv=Math.sqrt(c.reduce((a,b)=>a+(b-mu)**2,0)/c.length)/(mu||1);
  const n90=T.filter(t=>t===90).length,n60=T.filter(t=>t===60).length;
  return {T,n90,n60,n0:T.filter(t=>t===0).length,ratio:n90/(n90+n60),
    ticks,spread:Math.max(...ticks)-Math.min(...ticks),never:ticks.filter(v=>!v).length,
    disc:{per,mean:sum/SCALES.length,worst:Math.max(...Object.values(per))},
    edges:use.size,edgeCV:cv};
}
// 1 and 2 tied first; 3 breaks ties. Requirement 1 is structural, so its term
// is normally zero -- it is here only so the ranking stays right if it is not.
const score=m=>100*m.spread + m.disc.mean + 0.004*m.edgeCV + (m.edges<36?0.02:0);

function build(L,targetV,gap){
  const cache=new Map();
  const trans=(t,e,arr)=>{ const k=t+'|'+e+'|'+arr; let v=cache.get(k);
    if(v===undefined){ v=segTrans(t,e,arr,L,targetV,gap); cache.set(k,v); } return v; };
  const plans=[];
  for(let e0=0;e0<2;e0++) for(const arr0 of ADJ[ACT[0][e0]]){
    if(isAct(arr0,ACT[0][e0])) continue;
    const start=`0|${e0}|${arr0}`;
    const layers=[new Map([[`0#${start}`,1]])];
    for(let d=0;d<8;d++){ const nx=new Map();
      for(const [k,n] of layers[d]){
        const [ms,rest]=k.split('#'); const mask=Number(ms);
        const [t,e,arr]=rest.split('|').map(Number);
        for(const [kk,cnt] of trans(t,e,arr)){
          const [t2,e2,exit]=kk.split('|').map(Number);
          const nm=mask|(1<<t);
          if(nm!==255 && (t2===0 || (nm&(1<<t2)))) continue;
          if(nm===255 && `${t2}|${e2}|${exit}`!==start) continue;
          const nk=`${nm}#${t2}|${e2}|${exit}`;
          nx.set(nk,(nx.get(nk)||0)+n*cnt); } }
      layers.push(nx); }
    const fin=`255#${start}`;
    if(layers[8].has(fin)) plans.push({start,layers,total:layers[8].get(fin),fin});
  }
  return {L,targetV,gap,trans,plans};
}
function sample(M,rnd){
  let tot=0; for(const p of M.plans) tot+=p.total; if(!tot) return null;
  let r=rnd()*tot,P=M.plans[M.plans.length-1];
  for(const p of M.plans){ r-=p.total; if(r<=0){P=p;break;} }
  const chain=[]; let k=P.fin;
  for(let d=8;d>=1;d--){
    const [ms,rest]=k.split('#'); const mask=Number(ms);
    const [t2,e2,exit]=rest.split('|').map(Number);
    const cands=[]; let s=0;
    for(const [pk,n] of P.layers[d-1]){
      const [pms,prest]=pk.split('#'); const pm=Number(pms);
      const [t,e,arr]=prest.split('|').map(Number);
      if((pm|(1<<t))!==mask) continue;
      const cnt=M.trans(t,e,arr).get(`${t2}|${e2}|${exit}`); if(!cnt) continue;
      cands.push({pk,t,e,arr,w:n*cnt}); s+=n*cnt; }
    if(!s) return null;
    let rr=rnd()*s,pick=cands[cands.length-1];
    for(const c of cands){ rr-=c.w; if(rr<=0){pick=c;break;} }
    chain.unshift({t:pick.t,e:pick.e,arr:pick.arr,t2,e2,exit}); k=pick.pk; }
  const V=[];
  for(const c of chain){
    const p=segPath(c.t,c.e,c.arr,M.L,c.t2,c.e2,c.exit,rnd,M.targetV,M.gap);
    if(!p) return null; V.push(...p); }
  return {V,chain};
}
// local move: redraw ONE segment within the same transition -- every hard
// requirement is preserved by construction, so only the objective moves
function anneal(M,secs,rnd){
  rnd=rnd||Math.random;
  let cur=null,curS=Infinity,best=null,bestM=null,bestS=Infinity,tried=0,stall=0;
  const rebuild=st=>{const V=[];for(const p of st.paths)V.push(...p);return V;};
  const t0=Date.now();
  while(Date.now()-t0<secs*1000){
    tried++;
    if(!cur||stall>3000){
      const S=sample(M,rnd); if(!S) continue;
      const paths=[]; for(let i=0;i<8;i++) paths.push(S.V.slice(i*M.L,(i+1)*M.L));
      const m=measure(S.V); if(!m) continue;
      cur={chain:S.chain,paths}; curS=score(m); stall=0;
      if(curS<bestS){bestS=curS;best=S.V.slice();bestM=m;}
      continue; }
    const i=Math.floor(rnd()*8), c=cur.chain[i], keep=cur.paths[i];
    const p=segPath(c.t,c.e,c.arr,M.L,c.t2,c.e2,c.exit,rnd,M.targetV,M.gap);
    if(!p){ stall++; continue; }
    cur.paths[i]=p;
    const V=rebuild(cur), m=measure(V);
    if(!m){ cur.paths[i]=keep; stall++; continue; }
    const s=score(m), T=0.004*Math.exp(-stall/1500);
    if(s<=curS||rnd()<Math.exp((curS-s)/Math.max(1e-9,T))){
      curS=s; if(s<bestS){bestS=s;best=V.slice();bestM=m;stall=0;} else stall++;
    } else { cur.paths[i]=keep; stall++; } }
  return {best,bestM,bestS,tried};
}
module.exports={build,sample,measure,score,anneal};

if(require.main===module){
  const L=+(process.argv[2]||30), secs=+(process.argv[3]||30), gap=+(process.argv[4]||1);
  const tv=0; const t0=Date.now(); const M=build(L,tv,gap);
  const tot=M.plans.reduce((s,p)=>s+p.total,0);
  console.log('L='+L,' lap',8*L,' gap>=',gap,
              ' build',(Date.now()-t0)+'ms',' plans',M.plans.length,
              ' feasible laps',tot.toExponential(3));
  if(!tot){console.log('INFEASIBLE');process.exit(0);}
  const R=anneal(M,secs);
  if(!R.best){console.log('no lap assembled');process.exit(0);}
  const m=R.bestM;
  console.log('tried',R.tried);
  console.log('  1 tet ticks  ',m.ticks.join(' '),'  spread',m.spread,' never',m.never);
  console.log('  2 identity   ratio',m.ratio.toFixed(6),
              '  typical window off 2/3 by',(100*m.disc.mean).toFixed(2)+'%',
              ' worst',(100*m.disc.worst).toFixed(1)+'%');
  console.log('    per scale  ',Object.entries(m.disc.per).map(([k,v])=>'w'+k+':'+(100*v).toFixed(1)+'%').join('  '));
  console.log('  3 edges      ',m.edges,'/36   use cv',m.edgeCV.toFixed(3));
  console.log('  n90/n60/n0  ',m.n90,'/',m.n60,'/',m.n0);
  console.log('  walk        ',R.best.join(' '));
  if(process.env.DUMP){
    require('fs').writeFileSync(process.env.DUMP, JSON.stringify({
      L, gap, lap:R.best.length, walk:R.best,
      ticks:m.ticks, spread:m.spread, ratio:m.ratio,
      disc:{mean:m.disc.mean, per:m.disc.per}, n90:m.n90, n60:m.n60, n0:m.n0,
      edges:m.edges, edgeCV:m.edgeCV}, null, 1));
    console.log('  dumped to   ', process.env.DUMP); }
}
