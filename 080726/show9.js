'use strict';
const NBR=[]; for(const s of [[1,1,0],[1,-1,0],[1,0,1],[1,0,-1],[0,1,1],[0,1,-1]])
  {NBR.push(s);NBR.push([-s[0],-s[1],-s[2]]);}
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cls=(a,b)=>{const d=dot(a,b);return d===2?0:d===0?90:d===-1?60:null;};
const add=(a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]];
const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const len=a=>Math.hypot(...a); const key=p=>p.join(',');
const PATTERN=[90,90,60];
const NEXT=NBR.map((d,i)=>PATTERN.map(t=>NBR.map((e,j)=>j).filter(j=>cls(d,NBR[j])===t)));
function search(maxLen){const R=6,SZ=2*R+1;
  const enc=(x,y,z,d,ph)=>(((x+R)*SZ+(y+R))*SZ+(z+R))*36+d*3+ph;
  for(let d0=0;d0<12;d0++){const start={p:NBR[d0].slice(),d:d0,ph:1};
    const seen=new Set([enc(...start.p,d0,1)]),Q=[[start,[d0]]];let h=0;
    while(h<Q.length){const [st,path]=Q[h++]; if(path.length>maxLen)continue;
      for(const j of NEXT[st.d][st.ph]){const np=add(st.p,NBR[j]);
        if(Math.max(...np.map(Math.abs))>R)continue;
        const npath=path.concat(j);
        if(!np[0]&&!np[1]&&!np[2]&&npath.length%3===0){
          const f=npath[0],l=j,pv=npath[npath.length-2];
          if(cls(NBR[l],NBR[f])===PATTERN[0]&&cls(NBR[pv],NBR[l])===PATTERN[(npath.length-1)%3])return npath;}
        const k=enc(...np,j,(st.ph+1)%3); if(seen.has(k))continue; seen.add(k);
        Q.push([{p:np,d:j,ph:(st.ph+1)%3},npath]);}}}
  return null;}

const w=search(60);
const P=[[0,0,0]]; for(const j of w) P.push(add(P[P.length-1],NBR[j]));
const V=P.slice(0,-1);
const T=[]; for(let i=0;i<w.length;i++) T.push(cls(NBR[w[i]],NBR[w[(i+1)%w.length]]));
console.log('=== THE PERFECTLY BALANCED CLOSED WALK ===\n');
console.log('length        ', w.length, ' steps, closes on itself');
console.log('turn sequence ', T.join('  '));
console.log('n90 / n60     ', T.filter(x=>x===90).length, '/', T.filter(x=>x===60).length,
            ' = ratio', (T.filter(x=>x===90).length/T.length).toFixed(6));
console.log('every window of 3 holds exactly two 90s:',
  T.every((_,i)=>[0,1,2].map(k=>T[(i+k)%T.length]).filter(x=>x===90).length===2));
console.log('\nvertices (FCC integer coords):');
V.forEach((p,i)=>console.log('  '+String(i).padStart(2)+'  '+key(p).padEnd(12)+
  '  step dir '+key(NBR[w[i]]).padEnd(10)+'  turn here '+T[(i-1+T.length)%T.length]));
console.log('  closes back to', key(P[P.length-1]));
console.log('distinct vertices', new Set(V.map(key)).size, 'of', V.length, 'steps');

// ---- is it inside ONE stella octangula? ----------------------------------
const isFCC=p=>((p[0]+p[1]+p[2])%2+2)%2===0;
const adj=(a,b)=>Math.abs(len(sub(a,b))-Math.SQRT2)<1e-9;
let found=null;
for(let cx=-4;cx<=4&&!found;cx++)for(let cy=-4;cy<=4&&!found;cy++)for(let cz=-4;cz<=4&&!found;cz++){
  const C=[cx,cy,cz]; if(((cx+cy+cz)%2+2)%2!==1)continue;
  const O=[]; for(const d of [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]){
    const p=add(C,d); if(!isFCC(p)){O.length=0;break;} O.push(p);}
  if(O.length!==6)continue;
  const oset=new Set(O.map(key)), AP=[];
  for(let i=0;i<6;i++)for(let j=i+1;j<6;j++)for(let k=j+1;k<6;k++){
    if(!(adj(O[i],O[j])&&adj(O[j],O[k])&&adj(O[i],O[k])))continue;
    for(const n of NBR){const p=add(O[i],n); if(oset.has(key(p)))continue;
      if([O[i],O[j],O[k]].every(x=>adj(p,x))){AP.push(p);break;}}}
  const comp=new Set([...O.map(key),...AP.map(key)]);
  if(V.every(p=>comp.has(key(p)))) found={C,O,AP};}
console.log('\nfits inside a single stella octangula:', !!found, found?('centre '+key(found.C)):'');
