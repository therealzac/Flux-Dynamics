// Does the honeycomb admit a CLOSED walk whose turn sequence is exactly the
// balanced word (90,90,60) repeated? That word is the 2/3 target at every
// scale simultaneously -- discrepancy 0.55%, the floor. If it exists the
// identity requirement is solvable outright and only actualization is left.
//
// In the twelve neighbour vectors the turn rule is pure arithmetic:
//     90  <=>  d1 . d2 == 0        60  <=>  d1 . d2 == -1       0 <=> d1 == d2
// A walk closes when the direction vectors sum to zero and the wrap turns are
// legal too. So this is a shortest-closed-path search over (direction, phase).
'use strict';
const NBR = [];
for (const s of [[1,1,0],[1,-1,0],[1,0,1],[1,0,-1],[0,1,1],[0,1,-1]])
  { NBR.push(s); NBR.push([-s[0],-s[1],-s[2]]); }
const dot = (a,b) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cls = (a,b) => { const d = dot(a,b);
  return d === 2 ? 0 : d === 0 ? 90 : d === -1 ? 60 : null; };

const PATTERN = [90, 90, 60];
// legal successors for each direction under each required class
const NEXT = NBR.map((d,i) => PATTERN.map(t => NBR.map((e,j)=>j).filter(j => cls(d, NBR[j]) === t)));

function search(maxLen) {
  const R = 6, SZ = 2*R+1;
  const enc = (x,y,z,d,ph) => (((x+R)*SZ + (y+R))*SZ + (z+R))*36 + d*3 + ph;
  for (let d0 = 0; d0 < 12; d0++) {
    // BFS from just after taking direction d0, back to the same state at origin
    const start = { p:[NBR[d0][0],NBR[d0][1],NBR[d0][2]], d:d0, ph:1 };
    const seen = new Set(), Q = [[start, [d0]]];
    seen.add(enc(...start.p, start.d, start.ph));
    let head = 0;
    while (head < Q.length) {
      const [st, path] = Q[head++];
      if (path.length > maxLen) continue;
      for (const j of NEXT[st.d][st.ph]) {
        const np = [st.p[0]+NBR[j][0], st.p[1]+NBR[j][1], st.p[2]+NBR[j][2]];
        if (Math.abs(np[0])>R||Math.abs(np[1])>R||Math.abs(np[2])>R) continue;
        const nph = (st.ph+1) % 3, npath = path.concat(j);
        // closed? back at origin, and the two wrap turns must also match
        if (np[0]===0&&np[1]===0&&np[2]===0 && npath.length % 3 === 0) {
          const first = npath[0], last = j, prev = npath[npath.length-2];
          if (cls(NBR[last], NBR[first]) === PATTERN[0] &&
              cls(NBR[prev], NBR[last]) === PATTERN[(npath.length-1) % 3])
            return npath;
        }
        const k = enc(...np, j, nph);
        if (seen.has(k)) continue; seen.add(k);
        Q.push([{p:np, d:j, ph:nph}, npath]);
      }
    }
  }
  return null;
}

const w = search(60);
if (!w) { console.log('NO closed balanced walk up to length 60'); process.exit(0); }
console.log('FOUND a closed walk with turn sequence (90,90,60) repeated');
console.log('  length', w.length, ' turns', w.length, ' n90', w.length*2/3, ' n60', w.length/3);
const pos = [[0,0,0]]; for (const j of w) pos.push([pos[pos.length-1][0]+NBR[j][0],
  pos[pos.length-1][1]+NBR[j][1], pos[pos.length-1][2]+NBR[j][2]]);
console.log('  closes at', pos[pos.length-1].join(','), ' distinct vertices', new Set(pos.slice(0,-1).map(p=>p.join(','))).size);
const T = []; for (let i=0;i<w.length;i++) T.push(cls(NBR[w[i]], NBR[w[(i+1)%w.length]]));
console.log('  turn sequence', T.join(','));
console.log('  all match pattern:', T.every((t,i) => t === PATTERN[(i+1) % 3]));
