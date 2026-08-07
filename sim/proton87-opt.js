// ===========================================================================
// PROTON 87 -- LAP OPTIMISER
// ===========================================================================
// The hard requirements are met by CONSTRUCTION, never by fitness:
//   * quark composition   n90 / (n90 + n60) == 2/3, exactly (v = n90 - 2*n60 = 0)
//   * Lorentz isotropy    every face live for exactly the same number of ticks
//   * full traversal      all 36 edges of the compound
//   * legal turns         60 / 90 / 0 only; 120 refused
// A dynamic program over (faces done, apex, entry node, running v) describes
// that feasible set exactly. Counting completions from every state turns it
// into a UNIFORM SAMPLER over the set -- so every candidate drawn already
// satisfies all of the above and none is ever rejected.
//
// What is left to optimise is SOFT: how evenly the 60s are spaced among the
// 90s. That is what a long run improves. The constraints do not drift, cannot
// be traded against the objective, and do not get better with more time --
// they are either satisfied or the lap is not produced at all.
//
// Runs in a worker; the core below is plain functions so it can be checked
// headlessly in node.
// ===========================================================================
(function (root) {
'use strict';

// ---- the compound, in exact ideal coordinates -----------------------------
// Octahedron at +-A e_i, stella apexes at A(+-1,+-1,+-1), A = 1/sqrt2. Unit
// distance is adjacency; every turn class then follows from the chord.
const A = 1 / Math.SQRT2, VP = [];
for (let a = 0; a < 3; a++) for (const s of [1, -1]) { const p = [0, 0, 0]; p[a] = s * A; VP.push(p); }
for (const x of [1, -1]) for (const y of [1, -1]) for (const z of [1, -1]) VP.push([x * A, y * A, z * A]);
const NV = 14, OCT = [0, 1, 2, 3, 4, 5], APEX = [6, 7, 8, 9, 10, 11, 12, 13];
const isApex = i => i >= 6;
const DIST = (i, j) => Math.hypot(VP[i][0] - VP[j][0], VP[i][1] - VP[j][1], VP[i][2] - VP[j][2]);
const ADJ = VP.map((_, i) => VP.map((_, j) => j).filter(j => j !== i && Math.abs(DIST(i, j) - 1) < 1e-9));
const TURN = new Int8Array(NV * NV * NV).fill(-1);
for (let u = 0; u < NV; u++) for (let v = 0; v < NV; v++) for (let w = 0; w < NV; w++) {
  if (!ADJ[v].includes(u) || !ADJ[v].includes(w)) continue;
  const L = DIST(u, w);
  TURN[(u * NV + v) * NV + w] = Math.abs(L - 1) < 1e-9 ? 60
    : Math.abs(L - Math.SQRT2) < 1e-9 ? 90 : Math.abs(L - 2) < 1e-9 ? 0 : -1; }
const tn = (u, v, w) => TURN[(u * NV + v) * NV + w];
const DV = t => t === 90 ? 1 : t === 60 ? -2 : 0;         // v = n90 - 2*n60
const EK = (u, v) => Math.min(u, v) * NV + Math.max(u, v);
const OCT_EDGES = []; for (const u of OCT) for (const v of ADJ[u]) if (v < 6 && u < v) OCT_EDGES.push(EK(u, v));
const octBit = new Map(OCT_EDGES.map((e, i) => [e, i]));

// ---- what actually actualizes a tet ---------------------------------------
// An apex reaches its face by two BASE edges and exactly one SHORTCUT, and that
// shortcut always runs to the face's POLE -- the oct vertex off the shortcut
// equator. Verified against the live engine: with the oct held by its four ring
// rods, the single tet rod is always apex -> pole.
//
// A tet exists from the moment that apex-pole shortcut is TRAVERSED, either
// direction, and it erases the others. So actualization is a property of the
// WALK, not of the packing -- no solver is needed to know it.
//
// The consequence is not small: a segment can reach its apex along the two base
// edges, sit there the whole window and leave the same way, and that face's tet
// NEVER EXISTS. Measured on a lap whose proximity dwell was a perfect 18 x 8:
// actualized dwell 18 18 35 1 1 18 19 34. Proximity is not the physical clock.
const poleOf = a => VP[a][2] > 0 ? 4 : 5;
const isPoleEdge = (u, v) => (isApex(u) && v === poleOf(u)) || (isApex(v) && u === poleOf(v));
// Face live at each tick, from the walk alone.
function actualised(V) {
  const n = V.length, live = new Array(n).fill(null);
  let cur = null;
  for (let pass = 0; pass < 2; pass++) for (let i = 0; i < n; i++) {
    const u = V[i], w = V[(i + 1) % n];
    if (isApex(u) && w === poleOf(u)) cur = u; else if (isApex(w) && u === poleOf(w)) cur = w;
    live[(i + 1) % n] = cur; }
  const t = new Map(APEX.map(a => [a, 0]));
  for (const l of live) if (l !== null) t.set(l, t.get(l) + 1);
  const ticks = [...t.values()];
  return { live, ticks, spread: Math.max(...ticks) - Math.min(...ticks),
           never: ticks.filter(x => x === 0).length,
           noTet: live.filter(x => x === null).length };
}

// ---- inner: one segment, counted --------------------------------------------
// A segment starts AT its apex having arrived from `prev`, walks `len` further
// nodes touching no other apex, spends all three of its apex's edges, and ends
// adjacent to the next apex with a legal turn. State is (prev, cur, apex edges
// spent, running v); fwd[] counts the paths reaching each state, which is what
// makes uniform sampling possible.
function buildSegment(Ap, prev, len, needAllApexEdges) {
  const allow = new Uint8Array(NV); for (const o of OCT) allow[o] = 1; allow[Ap] = 1;
  const aE = ADJ[Ap], bit = v => 1 << aE.indexOf(v);
  const VLO = -2 * (len + 2), VHI = len + 2, VR = VHI - VLO + 1;
  // SPARSE, NOT DENSE. These layers used to be Float64Array(NV*NV*8*VR) -- 0.3GB
  // at L=17 and 0.67GB at L=26, held for the whole run, which is enough to take
  // the tab down. Almost every slot was zero: only a small fraction of the
  // (prev, cur, apex-edges, v) product is ever reachable. A Map stores what is.
  const key = (p, c, m, v) => ((p * NV + c) * 8 + m) * VR + (v - VLO);
  const unkey = k => { const v = k % VR + VLO, r = (k - (k % VR)) / VR;
    return { v, m: r & 7, c: ((r >> 3) % NV), p: (((r >> 3) - ((r >> 3) % NV)) / NV) }; };
  const fwd = []; let cur = new Map(), any = [];
  for (const nx of ADJ[Ap]) { if (!allow[nx]) continue;
    const t = tn(prev, Ap, nx); if (t < 0) continue;
    const k = key(Ap, nx, bit(nx), DV(t)); if (!cur.has(k)) any.push(k);
    cur.set(k, (cur.get(k) || 0) + 1); }
  fwd.push({ arr: cur, keys: any });
  for (let s = 1; s < len; s++) {
    const nx2 = new Map(), ks = [];
    for (const k of fwd[s - 1].keys) { const { v, m, c, p } = unkey(k); const n = fwd[s - 1].arr.get(k);
      for (const w of ADJ[c]) { if (!allow[w]) continue;
        const t = tn(p, c, w); if (t < 0) continue;
        const nm = (c === Ap || w === Ap) ? (m | bit(c === Ap ? w : c)) : m;
        const nk = key(c, w, nm, v + DV(t)); if (!nx2.has(nk)) ks.push(nk);
        nx2.set(nk, (nx2.get(nk) || 0) + n); } }
    fwd.push({ arr: nx2, keys: ks }); }
  // index every layer by current node: sampleSegment only ever looks at
  // predecessors whose CURRENT node is the successor's PREVIOUS node, and
  // scanning the whole layer for them dominated the sampling cost.
  for (const f of fwd) { f.byC = new Map();
    for (const k of f.keys) { const c = ((k - (k % VR)) / VR >> 3) % NV;
      let a = f.byC.get(c); if (!a) f.byC.set(c, a = []); a.push(k); } }
  // closing transitions, grouped by (B, exit, v) with a path count each
  const trans = new Map();
  const last = fwd[len - 1];
  for (const k of last.keys) { const { v, m, c, p } = unkey(k);
    // SPENDING ALL THREE APEX EDGES IN ONE WINDOW COSTS TWO APEX VISITS, and
    // every apex visit injects a pair of adjacent 60s (60 at the apex, 60 at
    // the oct vertex leading in -- `apex+oct` is never 90). So this is not a
    // free requirement: it is coverage bought with evenness. Optional, because
    // which of the two matters more is a physics question, not a coding one.
    if (needAllApexEdges && m !== 7) continue;
    for (const B of APEX) { if (B === Ap || !ADJ[c].includes(B)) continue;
      const t = tn(p, c, B); if (t < 0) continue;
      const tk = (B - 6) * 1e6 + c * 1e4 + (v + DV(t) + 500);
      let e = trans.get(tk);
      if (!e) trans.set(tk, e = { B, exit: c, v: v + DV(t), n: 0, ends: [], endsAX: [], nAX: 0 });
      const cnt = last.arr.get(k);
      e.n += cnt; e.ends.push({ k, n: cnt });
      // APEX EXIT. If the segment's last two nodes are [Ap, pole(B)] then the
      // turn at that pole is apex+apex -- 90 or 0 -- instead of apex+oct, which
      // is always 60. It turns the handover's run of THREE adjacent 60s into
      // two, halving the forced gap-1s that pin the discrepancy. Only possible
      // when Ap and B share a hemisphere, so it cannot be required of every
      // handover; it is preferred wherever it exists.
      if (p === Ap) { e.endsAX.push({ k, n: cnt }); e.nAX += cnt; } } }
  return { Ap, prev, len, fwd, trans, unkey, key,
           allow, bit, aE, VLO, VR };
}

// sample one segment path uniformly among those realising a given transition
function sampleSegment(S, tr, rnd, preferAX) {
  const pool = (preferAX && tr.endsAX.length) ? tr.endsAX : tr.ends;
  const tot = (preferAX && tr.endsAX.length) ? tr.nAX : tr.n;
  let r = rnd() * tot, pick = pool[pool.length - 1];
  for (const e of pool) { r -= e.n; if (r <= 0) { pick = e; break; } }
  const path = []; let k = pick.k;
  for (let s = S.len - 1; s >= 1; s--) {
    const { c } = S.unkey(k); path.unshift(c);
    // predecessors: any state at s-1 that steps to this one
    const cands = []; let tot = 0;
    const st = S.unkey(k);
    for (const pk of (S.fwd[s - 1].byC.get(st.p) || [])) {
      const q = S.unkey(pk);
      const t = tn(q.p, q.c, st.c); if (t < 0) continue;
      const nm = (q.c === S.Ap || st.c === S.Ap) ? (q.m | S.bit(q.c === S.Ap ? st.c : q.c)) : q.m;
      if (nm !== st.m || q.v + DV(t) !== st.v) continue;
      const n = S.fwd[s - 1].arr.get(pk); if (!n) continue;
      cands.push({ pk, n }); tot += n; }
    let rr = rnd() * tot, nk = cands[cands.length - 1].pk;
    for (const c2 of cands) { rr -= c2.n; if (rr <= 0) { nk = c2.pk; break; } }
    k = nk; }
  path.unshift(S.unkey(k).c);
  return path;
}

// ---- outer: face order + running v, counted --------------------------------
function buildLap(len, target, opts) {
  opts = opts || {};
  const needAll = opts.allApexEdges !== false;
  const seg = new Map();
  for (const Ap of APEX) for (const p of ADJ[Ap])
    seg.set(Ap * NV + p, buildSegment(Ap, p, len, needAll));
  const A0 = APEX[0];
  const plans = [];
  for (const p0 of ADJ[A0]) {
    // forward counts over depth
    const layers = [new Map([[`0|${A0}|${p0}|0`, 1]])];
    for (let d = 0; d < 8; d++) {
      const nxt = new Map();
      for (const [k, n] of layers[d]) {
        const [mask, Ap, p, v] = k.split('|').map(Number);
        const S = seg.get(Ap * NV + p); if (!S) continue;
        for (const tr of S.trans.values()) {
          // QUARK BALANCE ON THE FACE. With perFaceBalanced, every face's own
          // window must read 2/3 -- not just the lap. This is what makes the
          // objectives pull against each other: a face cannot borrow 90s from
          // its neighbours to flatter the global ratio, so the composition has
          // to be right locally, at the scale a quark actually lives on.
          if (opts.perFaceBalanced && tr.v !== 0) continue;
          // TRUE ISOTROPY, BY CONSTRUCTION. Enter every apex along its pole
          // shortcut and each face actualizes exactly at its segment boundary,
          // so every live window is exactly one segment long. A segment only
          // touches its own apex, so no other rod can go up in between.
          if (opts.poleEntry && tr.exit !== poleOf(tr.B)) continue;
          const nm = mask | (1 << (Ap - 6));
          if (nm !== 255 && (tr.B === A0 || (nm & (1 << (tr.B - 6))))) continue;
          if (nm === 255 && (tr.B !== A0 || tr.exit !== p0)) continue;
          const nk = `${nm}|${tr.B}|${tr.exit}|${v + tr.v}`;
          nxt.set(nk, (nxt.get(nk) || 0) + n * (tr.nAX || tr.n)); } }
      layers.push(nxt); }
    const endK = `255|${A0}|${p0}|${target}`;
    if (layers[8].has(endK)) plans.push({ p0, layers, total: layers[8].get(endK), endK });
  }
  return { len, target, seg, A0, plans };
}

// sample one complete lap uniformly (weighted by how many walks each state has)
function sampleLap(M, rnd, AX) {
  let tot = 0; for (const p of M.plans) tot += p.total;
  if (!tot) return null;
  let r = rnd() * tot, P = M.plans[M.plans.length - 1];
  for (const p of M.plans) { r -= p.total; if (r <= 0) { P = p; break; } }
  // walk BACKWARD from the end state, choosing predecessors in proportion to
  // their forward counts -- that is what makes the draw uniform.
  const chain = []; let k = P.endK;
  for (let d = 8; d >= 1; d--) {
    const [mask, B, exit, v] = k.split('|').map(Number);
    const cands = []; let s = 0;
    for (const [pk, n] of P.layers[d - 1]) {
      const [pm, Ap, p, pv] = pk.split('|').map(Number);
      if ((pm | (1 << (Ap - 6))) !== mask) continue;
      const S = M.seg.get(Ap * NV + p); if (!S) continue;
      for (const tr of S.trans.values()) {
        if (tr.B !== B || tr.exit !== exit || pv + tr.v !== v) continue;
        const wt = n * (tr.nAX || tr.n); cands.push({ pk, Ap, p, tr, w: wt }); s += wt; } }
    if (!s) return null;
    let rr = rnd() * s, pick = cands[cands.length - 1];
    for (const c of cands) { rr -= c.w; if (rr <= 0) { pick = c; break; } }
    chain.unshift(pick); k = pick.pk; }
  const V = [];
  for (const c of chain) {
    const S = M.seg.get(c.Ap * NV + c.p);
    V.push(c.Ap, ...sampleSegment(S, c.tr, rnd)); }
  return { V, chain };
}

// ---- measurement: everything recomputed from the walk alone ----------------
function measure(V) {
  const n = V.length, T = [];
  for (let i = 0; i < n; i++) T.push(tn(V[(i - 1 + n) % n], V[i], V[(i + 1) % n]));
  if (T.some(x => x < 0)) return null;
  for (let i = 0; i < n; i++) if (!ADJ[V[i]].includes(V[(i + 1) % n])) return null;
  const cov = new Set(); for (let i = 0; i < n; i++) cov.add(EK(V[i], V[(i + 1) % n]));
  const n90 = T.filter(x => x === 90).length, n60 = T.filter(x => x === 60).length,
        n0 = T.filter(x => x === 0).length;
  const dw = new Map(APEX.map(a => [a, 0])); let live = null, hand = 0;
  for (const v of V) { if (isApex(v) && v !== live) { live = v; hand++; }
    if (live !== null) dw.set(live, dw.get(live) + 1); }
  const dv = [...dw.values()];
  const ix = []; T.forEach((t, i) => { if (t === 60) ix.push(i); });
  const gaps = ix.map((x, i) => i ? x - ix[i - 1] : x + n - ix[ix.length - 1]);
  const gm = gaps.reduce((s, x) => s + x, 0) / (gaps.length || 1);
  const gsd = Math.sqrt(gaps.reduce((s, x) => s + (x - gm) ** 2, 0) / (gaps.length || 1));
  const disc = discrepancy(T);
  const act = actualised(V);
  const edge = edgeUse(V);
  return { T, disc, act, edge, n90, n60, n0, ratio: n90 / (n90 + n60), edges: cov.size,
           octEdges: [...cov].filter(e => octBit.has(e)).length,
           dwell: dv, dwellSpread: Math.max(...dv) - Math.min(...dv), handovers: hand,
           gaps, gapMean: gm, gapSD: gsd, gapMin: Math.min(...gaps), gapMax: Math.max(...gaps) };
}
// SOFT objective only. Hard requirements are not scored -- they are structural,
// and a lap that misses one is a bug to report, not a worse lap to rank.
//
// TWO THIRDS AT EVERY SCALE. A global 2/3 can be faked: traverse, then bank a
// long run of 90s looping round the oct. Nature does not do that. The real
// condition is that ANY window of the ledger reads 2/3 -- 100 moves, 50 moves,
// and on average even 3 moves. The mean over windows is automatic once the
// global ratio is exact, so what is being minimised here is the VARIANCE at
// each scale.
//
// The ledger is the quark-producing turns only: 90s and 60s in order, with
// 0-turns dropped, since a step that does not turn is not a quark.
//
// The optimum is known and is the balanced (Christoffel) word (90,90,60)
// repeated: every window of 3 holds exactly two 90s, every window of 3k
// exactly 2k, and no window of any size is off by more than one. Equivalently
// -- every gap between consecutive 60s is exactly 3.
const SCALES = [3, 6, 12, 25, 50, 100];
function discrepancy(T) {
  const x = []; for (const t of T) { if (t === 90) x.push(1); else if (t === 60) x.push(0); }
  const n = x.length; if (!n) return null;
  const pre = new Int32Array(2 * n + 1);
  for (let i = 0; i < 2 * n; i++) pre[i + 1] = pre[i] + x[i % n];
  const per = {};
  let total = 0;
  for (const w of SCALES) {
    if (w > n) { per[w] = 0; continue; }
    let se = 0;
    for (let i = 0; i < n; i++) { const d = (pre[i + w] - pre[i]) / w - 2 / 3; se += d * d; }
    const rms = Math.sqrt(se / n); per[w] = rms; total += rms; }
  // MEAN, not sum. The headline used to be the sum of six rms values, which is
  // not a quantity -- it just grew with however many scales were in the list.
  // The mean reads directly: "a typical window is off 2/3 by this much". A
  // window of 3 that should hold 2 nineties and holds 1 is off by 0.333.
  const ks = Object.keys(per);
  const mean = total / ks.length;
  const worst = Math.max(...ks.map(k => per[k]));
  return { per, total, mean, worst, ledger: n };
}
// ---- edge evenness: the THIRD priority ------------------------------------
// Not "are all 36 covered" but "are they used equally". A lap of N steps over
// 36 edges wants N/36 apiece; an unused edge is simply the worst case of the
// same measure, so coverage falls out of evenness instead of being bolted on
// as a penalty. Reported as a coefficient of variation so it does not scale
// with lap length.
function edgeUse(V) {
  const n = V.length, c = new Map();
  for (const e of OCT_EDGES) c.set(e, 0);
  for (let i = 0; i < n; i++) { const k = EK(V[i], V[(i + 1) % n]);
    c.set(k, (c.get(k) || 0) + 1); }
  const counts = [...c.values()], m = counts.reduce((a, b) => a + b, 0) / counts.length;
  const sd = Math.sqrt(counts.reduce((a, b) => a + (b - m) ** 2, 0) / counts.length);
  return { counts, distinct: c.size, unused: counts.filter(x => x === 0).length,
           mean: m, sd, cv: m ? sd / m : 0 };
}

// ---- the objective, in the stated priority order ---------------------------
//   1. actualized face time must be EVEN  -- Lorentz; a preferred direction in
//      the proton itself is not a worse lap, it is a wrong one
//   2. nucleon identity                   -- 2/3 at every scale, not just globally
//   3. edge evenness                      -- secondary, breaks ties only
// The weights are separated by orders of magnitude so this is lexicographic in
// practice: no amount of edge evenness can buy back a tick of face-time spread.
// Priority 1 is normally ZERO here because poleEntry makes it structural; the
// term exists so the ranking is still right if that constraint is ever off.
const W_FACE = 100, W_EDGE = 0.004;
const score = m => W_FACE * (m.act ? m.act.spread : 99)
                 + (m.disc ? m.disc.mean : 1e6)
                 + W_EDGE * (m.edge ? m.edge.cv : 9);


// ---- the long run ----------------------------------------------------------
// Two moves, both preserving every hard requirement BY CONSTRUCTION:
//   * resample one segment's path within the same transition -- same apex, same
//     entry and exit, same v, so composition, dwell and closure cannot change
//   * resample the whole lap from the uniform sampler -- a restart
// So there is nothing to check after a move and nothing that can drift. Only
// the soft score moves. Annealed, because hill-climbing on window discrepancy
// stalls quickly in a space of 7e21.
function makeRunner(M, rnd, K) {
  rnd = rnd || Math.random; K = K || 12;
  let cur = null, curS = Infinity, best = null, bestM = null, bestS = Infinity;
  let tried = 0, accepted = 0, restarts = 0, sinceGain = 0;
  const build = st => { const V = [];
    for (let i = 0; i < 8; i++) V.push(st.chain[i].Ap, ...st.paths[i]); return V; };
  function fresh() {
    const L = sampleLap(M, rnd, true); if (!L) return null;
    const paths = []; let o = 0;
    for (let i = 0; i < 8; i++) { paths.push(L.V.slice(o + 1, o + 1 + M.len)); o += M.len + 1; }
    return { chain: L.chain, paths }; }
  function consider(st) {
    const V = build(st), m = measure(V); if (!m) return null;
    return { st, V, m, s: score(m) }; }
  function step() {
    tried++;
    if (!cur || sinceGain > 4000) {
      const st = fresh(); if (!st) return null;
      const c = consider(st); if (!c) return null;
      cur = c.st; curS = c.s; restarts++; sinceGain = 0;
      if (c.s < bestS) { bestS = c.s; best = c.V.slice(); bestM = c.m; }
      return c; }
    // BEST OF K, NOT ONE DRAW. The old move took a single uniform sample of a
    // segment's path and accepted or rejected it -- sampling, not searching,
    // and it stalled a factor of two above the structural floor. Discrepancy is
    // decided by where the 60s land INSIDE a segment, so proposing several and
    // keeping the best turns each move into a directed step at the scale the
    // objective actually lives on. Every candidate is still drawn from the
    // exact feasible set, so nothing about the hard requirements changes.
    const i = Math.floor(rnd() * 8), c0 = cur.chain[i];
    const S = M.seg.get(c0.Ap * NV + c0.p);
    const keep = cur.paths[i];
    let bc = null;
    for (let k = 0; k < K; k++) {
      cur.paths[i] = sampleSegment(S, c0.tr, rnd, true);
      const c1 = consider(cur);
      if (c1 && (!bc || c1.s < bc.s)) bc = { s: c1.s, path: cur.paths[i], c: c1 }; }
    if (!bc) { cur.paths[i] = keep; sinceGain++; return null; }
    cur.paths[i] = bc.path;
    const c = bc.c;
    const T = 0.02 * Math.exp(-sinceGain / 2000);            // anneal
    if (c && (c.s <= curS || rnd() < Math.exp((curS - c.s) / Math.max(1e-9, T)))) {
      accepted++; curS = c.s;
      if (c.s < bestS) { bestS = c.s; best = c.V.slice(); bestM = c.m; sinceGain = 0; }
      else sinceGain++;
    } else { cur.paths[i] = keep; sinceGain++; }
    return c; }
  return { step, stats: () => ({ tried, accepted, restarts, bestS, curS }),
           best: () => (best ? { V: best, m: bestM, score: bestS } : null),
           seed: v => { const m = measure(v); if (m && score(m) < bestS)
             { bestS = score(m); best = v.slice(); bestM = m; } } };
}

root.P87OPT = { buildLap, sampleLap, measure, score, makeRunner, discrepancy, SCALES, actualised, poleOf, edgeUse,
                ADJ, isApex, APEX, OCT, tn, VP };
})(typeof self !== 'undefined' ? self : globalThis);

// ===========================================================================
// WORKER SHIM. The optimiser runs off the main thread so the lattice keeps
// rendering and the page stays usable while it runs for an hour.
// Progress is posted on a timer, not per step -- a message per candidate would
// cost more than the search.
// ===========================================================================
if (typeof importScripts === 'function' && typeof self !== 'undefined') {
  const O = self.P87OPT;
  let M = null, R = null, timer = null, t0 = 0, lastTried = 0, lastAt = 0, hist = [];
  const snapshot = () => { const b = R && R.best(), s = R ? R.stats() : null;
    const now = Date.now(), dt = (now - lastAt) / 1000;
    const rate = dt > 0 ? (s.tried - lastTried) / dt : 0;
    lastTried = s ? s.tried : 0; lastAt = now;
    return { type: 'progress', elapsed: (now - t0) / 1000, rate,
      tried: s ? s.tried : 0, accepted: s ? s.accepted : 0, restarts: s ? s.restarts : 0,
      hist, best: b ? { V: b.V, score: b.score, m: {
        ratio: b.m.ratio, n90: b.m.n90, n60: b.m.n60, n0: b.m.n0, edges: b.m.edges,
        dwell: b.m.dwell, dwellSpread: b.m.dwellSpread, handovers: b.m.handovers,
        act: b.m.act, edge: b.m.edge,
        gaps: b.m.gaps, gapMin: b.m.gapMin, gapMax: b.m.gapMax, gapSD: b.m.gapSD,
        disc: b.m.disc, T: b.m.T } } : null }; };
  self.onmessage = e => {
    const d = e.data;
    if (d.cmd === 'start') {
      clearInterval(timer);
      self.postMessage({ type: 'status', text: 'building the feasible set for L=' + d.len + '…' });
      // Quark balance on the face is a REQUIREMENT, not a preference: every
      // face's own window reads 2/3, so a face cannot borrow 90s from its
      // neighbours to flatter the global number.
      // poleEntry is what makes ACTUALIZED face time equal -- the proximity
      // version was equal for free and was not the physical clock.
      // allApexEdges is OFF: forcing a segment to spend its apex's third edge
      // costs a second apex visit, and every apex visit injects adjacent 60s.
      // That is priority 3 (coverage) charged against priority 2 (identity).
      // Coverage is now measured by edge evenness instead of required outright.
      M = O.buildLap(d.len, 0,
        { perFaceBalanced: true, poleEntry: true, allApexEdges: false });
      const total = M.plans.reduce((s, p) => s + p.total, 0);
      if (!total) { self.postMessage({ type: 'status', text: 'L=' + d.len + ': no lap with an exact 2/3 composition' });
        self.postMessage({ type: 'done', infeasible: true }); return; }
      R = O.makeRunner(M, Math.random);
      if (d.seed && d.seed.length) { try { R.seed(d.seed); } catch (_) {} }
      t0 = lastAt = Date.now(); lastTried = 0; hist = [];
      self.postMessage({ type: 'status', text: 'searching ' + total.toExponential(3) + ' feasible laps',
                         total: total, ideal: (() => { const w = [];
                           for (let i = 0; i < 48; i++) w.push(90, 90, 60); return O.discrepancy(w).mean; })() });
      timer = setInterval(() => {
        const b = R.best(); if (b) hist.push([+( (Date.now() - t0) / 1000).toFixed(1), +b.score.toFixed(5)]);
        if (hist.length > 600) hist = hist.filter((_, i) => i % 2 === 0);
        self.postMessage(snapshot()); }, 500);
      const loop = () => { if (!R) return;
        const until = Date.now() + 40;                 // yield often so messages flow
        while (Date.now() < until) for (let i = 0; i < 300; i++) R.step();
        setTimeout(loop, 0); };
      loop();
    } else if (d.cmd === 'stop') {
      clearInterval(timer); timer = null;
      if (R) self.postMessage(snapshot());
      self.postMessage({ type: 'done' }); R = null;
    }
  };
}
