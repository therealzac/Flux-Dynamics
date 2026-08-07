// ===========================================================================
// FIRST PRINCIPLES: the actualized void lattice is the tet-oct honeycomb.
// ===========================================================================
// Every unit-length connection actualized, the lattice is FCC: integer points
// of even coordinate sum, twelve neighbours each at distance sqrt2. Two
// neighbours of a vertex sit at chord/edge in {1, sqrt2, sqrt3, 2} and nothing
// else -- that IS the turn classification, with no tolerance bands:
//     1     -> 60      sqrt2 -> 90      sqrt3 -> 120 REFUSED      2 -> 0
// The stella octangula is not a special object here. It is one octahedral cell
// (centre at an odd-sum point, six vertices at distance 1) plus the eight
// tetrahedral cells capping its faces. So the compound and the wider lattice
// are the same graph, and an apex's other neighbours are simply there.
'use strict';
const E = Math.SQRT2;                                  // FCC edge length
const key = p => p.join(',');
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const len = a => Math.hypot(a[0], a[1], a[2]);
const isFCC = p => ((p[0] + p[1] + p[2]) % 2 + 2) % 2 === 0;

const NBR = [];
for (const s of [[1,1,0],[1,-1,0],[1,0,1],[1,0,-1],[0,1,1],[0,1,-1]])
  { NBR.push(s); NBR.push([-s[0], -s[1], -s[2]]); }

// turn class from the chord between the two neighbours, in edge units
function turnOf(u, v, w) {
  const r = len(sub(u, w)) / E;
  for (const [t, c] of [[60,1],[90,Math.SQRT2],[0,2]])
    if (Math.abs(r - c) < 1e-9) return t;
  return null;                                          // sqrt3 -> 120, refused
}

// ---- an octahedral cell and its eight tet apexes --------------------------
const C = [1, 0, 0];                                    // odd sum -> an oct hole
const OCTV = [];
for (const d of [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]) {
  const p = add(C, d); if (!isFCC(p)) throw new Error('oct vertex not FCC');
  OCTV.push(p); }
// faces: triples of oct vertices that are mutually adjacent
const adjacent = (a, b) => Math.abs(len(sub(a, b)) - E) < 1e-9;
const FACES = [];
for (let i = 0; i < 6; i++) for (let j = i+1; j < 6; j++) for (let k = j+1; k < 6; k++)
  if (adjacent(OCTV[i],OCTV[j]) && adjacent(OCTV[j],OCTV[k]) && adjacent(OCTV[i],OCTV[k]))
    FACES.push([OCTV[i], OCTV[j], OCTV[k]]);
// apex of a face: the FCC point adjacent to all three that is not an oct vertex
const octSet = new Set(OCTV.map(key));
const APEXES = FACES.map(f => {
  for (const n of NBR) { const p = add(f[0], n);
    if (octSet.has(key(p))) continue;
    if (f.every(x => adjacent(p, x))) return p; }
  return null; });

console.log('oct vertices      ', OCTV.length, ' all FCC:', OCTV.every(isFCC));
console.log('triangular faces  ', FACES.length);
console.log('apexes found      ', APEXES.filter(Boolean).length, ' distinct:',
            new Set(APEXES.filter(Boolean).map(key)).size);
console.log('edge lengths      ', 'oct-oct', (len(sub(OCTV[0],OCTV[2]))/E).toFixed(6),
            ' apex-oct', (len(sub(APEXES[0], FACES[0][0]))/E).toFixed(6));

// ---- does the honeycomb reproduce the compound census? --------------------
const COMP = [...OCTV, ...APEXES];
const compSet = new Map(COMP.map((p,i) => [key(p), i]));
const isApex = i => i >= 6;
const cadj = COMP.map((p,i) => COMP.map((q,j)=>j).filter(j => j !== i && adjacent(p, COMP[j])));
const cls = {};
for (let v = 0; v < COMP.length; v++) for (const u of cadj[v]) for (const w of cadj[v]) {
  if (u === w) continue;
  const t = turnOf(COMP[u], COMP[v], COMP[w]);
  const kind = (isApex(v)?'apex':'oct') + ' : ' + [isApex(u)?'apex':'oct', isApex(w)?'apex':'oct'].sort().join('+');
  cls[kind] = cls[kind] || {}; const kk = t === null ? 'REFUSED(120)' : t;
  cls[kind][kk] = (cls[kind][kk] || 0) + 1; }
console.log('\nCOMPOUND census (should match the engine exactly):');
for (const k of Object.keys(cls).sort()) console.log('  ' + k.padEnd(22), JSON.stringify(cls[k]));

// ---- and what the apexes gain once the walk may leave the compound --------
const fullDeg = APEXES.map(a => NBR.map(n => add(a,n)).filter(isFCC).length);
const offComp = APEXES.map(a => NBR.map(n => add(a,n)).filter(p => !compSet.has(key(p))).length);
console.log('\napex degree in the honeycomb', fullDeg[0], ' of which OFF the compound', offComp[0]);
const a = APEXES[0], nb = NBR.map(n => add(a,n));
const tally = {};
for (const u of nb) for (const w of nb) { if (key(u)===key(w)) continue;
  const t = turnOf(u, a, w); const kk = t===null?'REFUSED(120)':t; tally[kk]=(tally[kk]||0)+1; }
console.log('turns available AT an apex, full honeycomb:', JSON.stringify(tally));
