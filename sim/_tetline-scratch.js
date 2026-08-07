// ============================================================================
// TET ON A LINE  --  scratch, not part of the simulation.
//
// Draw an arbitrary straight line through the lattice centre, then tumble a
// tetrahedron along it: in at one boundary, out at the other, one flux event
// per step.
//
// The tumble rule is the one Zac fixed on 4 Aug: consecutive tets must SHARE A
// FACE.  That leaves exactly four moves from any position, one per face, and
// each of them swaps a single shortcut for one on the axis that was not in
// use.  Motion and flux-mode change are the same event.
//
// STEERING.  Three policies, chosen with {policy:...}:
//
//   'monotone'  (default)  Among the flips that ADVANCE along the line, take
//                          the one minimising  perp - mu*ds.  Because s
//                          strictly increases every step the walker cannot
//                          cycle and must reach the far boundary in finite
//                          time.  Termination is a theorem, not a hope.
//   'greedy'               Maximise ds alone.  Ignores the line; measures the
//                          bare drift speed available in a given direction.
//   'pursuit'              Chase a point a fixed lookahead down the line.
//
// Pursuit was tried first and FAILED: it fell into a period-4 limit cycle
// after three steps (60 legal moves, net travel 1.2 units).  The target was
// computed from the tet's own projection, so when the tet slid backwards the
// target slid backwards with it and the rule lost all memory of where it was
// meant to be going.  It is kept here so the failure stays reproducible.
//
// That a forward move ALWAYS exists is not an assumption.  The four flips are
// 1/2(sigma*e_C + eps*e_A) and 1/2(-sigma*e_C + eps*e_B), where A,B are the
// two live axes, C the third, eps free.  Best progress is
// 1/2 max(sigma*u_C + |u_A|, -sigma*u_C + |u_B|); for this to be <= 0 needs
// u_C = 0 and |u_A| = |u_B| = 0, i.e. u = 0.  So it is strictly positive for
// every direction, and 'monotone' can never be stuck.
//
// Supersedes the _isTet / _faceFlips in _tetwalk-scratch.js, which cached the
// adjacency at load time and went silently stale if the lattice was resized.
// Everything here rebuilds when NODE.length or LCENTER changes.
//
// All internal geometry is in INTEGER lattice coordinates.  Multiply by S for
// world coordinates.  One base edge = 1 world unit = sqrt(3) integer units.
//
//   _LINE()                 random line through the centre, drawn green
//   _LINE({dir:[1,1,0]})    a chosen line
//   _LINE({seed:12345})     a reproducible random line
//   _TETLINE()              tumble a tet down whatever line is set
//
// The loops registered from here are the SIMPLE ones: "simple" meaning the
// shortcuts are IMPOSED. We write the tet's two shortcuts into `active` and the
// solver holds them. Nothing here derives the shortcuts from anything moving.
// The next model has xons hop, and the shortcuts are whatever their motion
// leaves behind -- see the CA notes at the bottom of this file.
// ============================================================================

(function () {
  'use strict';

  // ---- adjacency, rebuilt whenever the lattice changes ---------------------
  const AXOF = d => (d[0] ? 'X' : (d[1] ? 'Y' : 'Z'));
  const K = (i, j) => i * 100000 + j;
  let _sig = null, _baseSet = null, _scAx = null, _baseNbr = null;

  function _sync() {
    const sig = NODE.length + '@' + LCENTER.join(',') + '@' + shells;
    if (sig === _sig) return;
    _sig = sig;
    _baseSet = new Set(); _baseNbr = NODE.map(() => []);
    for (const [i, j] of BASE) {
      _baseSet.add(K(i, j)); _baseSet.add(K(j, i));
      _baseNbr[i].push(j); _baseNbr[j].push(i);
    }
    _scAx = new Map();                       // undirected pair -> axis letter
    for (const [k, j] of SCOPT) {
      const i = +k.split(':')[0], a = +k.split(':')[1], L = AXOF(AX[a]);
      _scAx.set(K(i, j), L); _scAx.set(K(j, i), L);
    }
  }

  // ---- what is a tet ------------------------------------------------------
  // 4 base edges + 2 shortcuts on DISTINCT axes.  Distinct is forced, not
  // imposed: two shortcuts on one axis are parallel and cannot close a clique.
  // Returns the two shortcuts as [i,j,axis], or null.
  function isTet(q) {
    _sync();
    if (new Set(q).size !== 4) return null;
    let base = 0; const sc = [];
    for (let x = 0; x < 4; x++) for (let y = x + 1; y < 4; y++) {
      const a = q[x], b = q[y];
      if (_baseSet.has(K(a, b))) { base++; continue; }
      const ax = _scAx.get(K(a, b));
      if (ax === undefined) return null;
      sc.push([a, b, ax]);
    }
    if (base !== 4 || sc.length !== 2) return null;
    if (sc[0][2] === sc[1][2]) return null;
    return sc;
  }

  // ---- face-sharing neighbours -------------------------------------------
  // Drop one vertex, scan EVERY remaining node for one that completes a tet on
  // the surviving face.  Exhaustive by construction: no analytic shortcut, so
  // nothing can be missed if the geometry ever surprises us.
  function faceFlips(q) {
    const out = [];
    for (let d = 0; d < 4; d++) {
      const face = q.filter((_, i) => i !== d);
      for (let E = 0; E < NODE.length; E++) {
        if (q.includes(E)) continue;
        const sc = isTet([...face, E]);
        if (!sc) continue;
        out.push({ drop: q[d], add: E, v: [...face, E], sc,
                   mode: sc.map(s => s[2]).sort().join('') });
      }
    }
    return out;
  }

  const centroid = q => [0, 1, 2].map(t => q.reduce((s, n) => s + NODE[n][t], 0) / 4);
  const vkey = q => q.slice().sort((a, b) => a - b).join(',');

  // Commit a tet to the shortcut set.  Axis index comes from SCOPT, never from
  // reconstructing the coordinate difference.
  function writeTet(q) {
    const sc = isTet(q);
    if (!sc) throw new Error('writeTet: not a tet: ' + q);
    active.clear();
    for (const [i, j] of sc) {
      let done = false;
      for (let a = 0; a < AXN.length && !done; a++) {
        if (SCOPT.get(i + ':' + a) === j) { active.set(i + ':' + a, [i, j]); done = true; }
        else if (SCOPT.get(j + ':' + a) === i) { active.set(j + ':' + a, [j, i]); done = true; }
      }
      if (!done) throw new Error('writeTet: ' + i + '-' + j + ' is not a shortcut candidate');
    }
  }

  // ---- every tet in the lattice ------------------------------------------
  // Enumerated from the shortcut candidates: for each candidate pair (a,b),
  // the partner shortcut must join two common base-neighbours of a and b.
  function allTets() {
    _sync();
    const nbr = NODE.map(() => []);
    for (const [i, j] of BASE) { nbr[i].push(j); nbr[j].push(i); }
    const seenSC = new Set(), scs = [];
    for (const [k, j] of SCOPT) {
      const i = +k.split(':')[0], pid = Math.min(i, j) + '-' + Math.max(i, j);
      if (seenSC.has(pid)) continue; seenSC.add(pid);
      scs.push([i, j, _scAx.get(K(i, j))]);
    }
    const out = [], seen = new Set();
    for (const [a, b, ax] of scs) {
      const sh = nbr[a].filter(x => nbr[b].includes(x));
      for (let x = 0; x < sh.length; x++) for (let y = x + 1; y < sh.length; y++) {
        const q = [a, b, sh[x], sh[y]], sc = isTet(q);
        if (!sc) continue;
        const k = vkey(q); if (seen.has(k)) continue; seen.add(k);
        out.push({ v: q, sc, mode: sc.map(s => s[2]).sort().join(''), c: centroid(q) });
      }
    }
    return out;
  }

  // ---- the line -----------------------------------------------------------
  // p0 + s*u in integer coordinates.  s is arclength in integer units.
  let _L = null, _rod = null;

  const _mul = a => () => { a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };

  const sOf = c => (c[0] - _L.p0[0]) * _L.u[0] + (c[1] - _L.p0[1]) * _L.u[1]
                 + (c[2] - _L.p0[2]) * _L.u[2];
  const perpOf = c => { const s = sOf(c);
    return Math.hypot(c[0] - _L.p0[0] - s * _L.u[0],
                      c[1] - _L.p0[1] - s * _L.u[1],
                      c[2] - _L.p0[2] - s * _L.u[2]); };

  // How far the node set actually reaches along u, both ways.  Measured from
  // the nodes rather than assumed from RADIUS, so it is right for any centre,
  // any shell, any direction.
  function reach() {
    let lo = 0, hi = 0;
    for (const c of NODE) { const s = sOf(c); if (s < lo) lo = s; if (s > hi) hi = s; }
    return { lo, hi };
  }

  function drawLine() {
    const r = reach(), h = Math.max(-r.lo, r.hi) * 1.12, len = 2 * h * S;
    const dir = new THREE.Vector3(_L.u[0], _L.u[1], _L.u[2]);
    if (!_rod) {
      _rod = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 1, 10),
        new THREE.MeshBasicMaterial({ color: 0x00ff44, transparent: true, opacity: 0.85 }));
      _rod.frustumCulled = false; scene.add(_rod);
    }
    _rod.scale.set(0.022, len, 0.022);
    _rod.position.set(_L.p0[0] * S, _L.p0[1] * S, _L.p0[2] * S);
    _rod.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
    _rod.visible = true;
  }

  window._LINE = function (opt) {
    opt = opt || {};
    let u = opt.dir, seed = opt.seed;
    if (!u) {                                   // uniform on the sphere (Marsaglia)
      if (seed === undefined) seed = (Math.random() * 2147483647) | 0;
      const r = _mul(seed);
      let x, y, q; do { x = 2 * r() - 1; y = 2 * r() - 1; q = x * x + y * y; }
      while (q >= 1 || q === 0);
      const t = 2 * Math.sqrt(1 - q);
      u = [x * t, y * t, 1 - 2 * q];
    }
    const n = Math.hypot(u[0], u[1], u[2]);
    if (!(n > 0)) throw new Error('_LINE: direction is zero');
    // p0 defaults to the lattice centre; the electron overrides it, because a
    // mode-locked tet cannot steer onto a line -- the line is wherever it is.
    _L = { p0: (opt.p0 || LCENTER).slice(), u: u.map(v => v / n),
           seed: seed === undefined ? null : seed };
    _sync(); drawLine();
    const r = reach();
    console.log('LINE', JSON.stringify({ seed: _L.seed, dir: _L.u.map(v => +v.toFixed(5)),
      centre: _L.p0, reach: [+r.lo.toFixed(2), +r.hi.toFixed(2)] }));
    return _L;
  };
  window._LINEOFF = () => { if (_rod) _rod.visible = false; };
  window._LINEGET = () => _L;

  // ---- the tumble path, drawn ---------------------------------------------
  // Centroid trace in world coordinates, so a single still shows the crossing.
  let _trail = null;
  function drawTrail(cs) {
    if (!_trail) {
      _trail = new THREE.Line(new THREE.BufferGeometry(),
        new THREE.LineBasicMaterial({ color: 0xffb020, transparent: true, opacity: 0.95 }));
      _trail.frustumCulled = false; scene.add(_trail);
    }
    const a = new Float32Array(Math.max(3, cs.length * 3));
    cs.forEach((c, k) => { a[k * 3] = c[0] * S; a[k * 3 + 1] = c[1] * S; a[k * 3 + 2] = c[2] * S; });
    _trail.geometry.setAttribute('position', new THREE.BufferAttribute(a, 3));
    _trail.geometry.setDrawRange(0, cs.length);
    _trail.visible = cs.length > 1;
  }
  window._TRAILOFF = () => { if (_trail) _trail.visible = false; };

  // ---- the flip graph, cached ---------------------------------------------
  // Every tet and its face-sharing neighbours. Built once per lattice (~0.4 s
  // for 1392 tets) so the mode-locked walker can search several flips deep
  // without re-scanning the node list at every level.
  let _gsig = null, _G = null;
  function graph() {
    _sync();
    if (_G && _gsig === _sig) return _G;
    const T = allTets(), idx = new Map();
    T.forEach((t, i) => idx.set(vkey(t.v), i));
    const adj = T.map(() => []);
    for (let i = 0; i < T.length; i++) for (const f of faceFlips(T[i].v)) {
      const j = idx.get(vkey(f.v));
      if (j !== undefined) adj[i].push(j);
    }
    _G = { T, idx, adj }; _gsig = _sig;
    return _G;
  }
  window._tetGraph = graph;

  // ---- mode-locked moves --------------------------------------------------
  // A single face-flip ALWAYS changes the flux mode, so a tet that must keep
  // its mode cannot move in one event. It has to leave the mode and come back.
  // One STEP is therefore an excursion, cut at the FIRST return to the locked
  // mode -- anything longer is two steps glued together.
  //
  // The excursion length is the whole story, and getting it wrong is easy.
  // Measured exhaustively over all 1392 tets of the 339-node lattice:
  //
  //   flips | distinct steps | axis-aligned | off-axis | lengths   | per event
  //   ------|----------------|--------------|----------|-----------|----------
  //     2   |        6       |      6       |    0     | 1         |  0.500
  //     3   |       20       |      0       |   20     | v2, v3    |  0.577
  //     4   |       36       |     12       |   24     | 1, 2, v5  |  0.559
  //     5   |       20       |      0       |   20     | v2, v3    |  0.346
  //
  // The 2-flip move is the shortest and by far the most restrictive: its six
  // displacements are +-1 on a cube axis and nothing else, so a walker
  // restricted to it is stuck on a line parallel to the axis its mode does not
  // use. That is a fact about the 2-flip move, NOT about mode-locking -- an
  // easy and wrong conclusion to draw, and one drawn here before the odd
  // lengths were checked.
  //
  // The 3-flip excursion is omnidirectional AND the fastest thing available
  // (0.577 per flux event, the same as the neutrino's best). Allowing lengths
  // 2..4 gives a mode-locked tet steps along <100>, <110>, <111> and <210> --
  // ample to follow an arbitrary line.
  function modeChains(q, lock, maxLen) {
    maxLen = maxLen || 4;
    const g = graph(), i0 = g.idx.get(vkey(q));
    if (i0 === undefined) return [];
    const out = [], seen = new Set();
    const walk = (cur, d, chain) => {
      if (d > 0 && g.T[cur].mode === lock) {          // first return: step ends
        if (cur === i0) return;                       // came back to itself
        const key = chain.join('>');
        if (!seen.has(key)) {
          seen.add(key);
          out.push({ chain: chain.map(k => g.T[k].v), v: g.T[cur].v,
                     mode: lock, len: d, via: g.T[chain[0]].mode, key });
        }
        return;
      }
      if (d >= maxLen) return;
      for (const j of g.adj[cur]) walk(j, d + 1, chain.concat(j));
    };
    walk(i0, 0, []);
    return out;
  }
  window._modeChains = modeChains;

  // ---- THE HOP TABLE ------------------------------------------------------
  // FULLY ANALYTIC. Built once by integer vector arithmetic over the TWO
  // canonical XZ tet orientations. It touches no lattice, no graph and no
  // search -- the same status as a multiplication table, and the runtime does
  // nothing but look up and translate.
  //
  // WHY TWO CLASSES IS THE WHOLE STORY. Canonicalise an XZ tet by putting the
  // lexicographically smaller endpoint of its X rod at the origin. Measured
  // over all 464 XZ tets of the lattice, exactly TWO signatures result:
  //   U   (0,0,0) (2,0,0) (1, 1,1) (1, 1,-1)      Z rod above
  //   D   (0,0,0) (2,0,0) (1,-1,1) (1,-1,-1)      Z rod below
  // mirror images in Y, 232 members each. Each offers exactly SIX moves --
  // rail (0,+-1,0) and steer (+-1,0,+-1) -- and EVERY move flips the class,
  // U->D->U without exception. So orientation has period two and the whole
  // sublattice is described by 12 entries.
  //
  // WHY THE HOPS ARE FORCED, NOT FOUND. Measured over the same 464 tets:
  //   rail  -- both endpoints of the new rod are EXACTLY one base hop from the
  //            tet, 1648 of 1648. Never inside it, never two away.
  //   steer -- each new rod has one endpoint IN the tet and one exactly one hop
  //            away, 3040 and 3040. The shared base edge's two vertices ARE the
  //            two in-tet endpoints.
  // So the xon is never routed anywhere. Chirality then picks the sense and the
  // 60-degree rule picks the pivot, and what is left is a fixed hop list:
  // TWO hops for a rail, FOUR for a steer.
  // TRAVERSAL SENSE MAPS. A map fixes one sense per base axis -- 16 in
  // principle. The ones that COHERE are those expressible as sign(d.n) for some
  // n: a POLARIZATION. Four central planes cut the sphere into 14 regions, so
  // 14 of the 16 are polarizations and 2 are not, and the 2 exceptions are
  // exactly the ZERO-SUM sets: if sum(d) = 0 then sum(d.n) = 0, so no n can
  // make all four positive. Measured, and they are precisely A and B.
  //
  // That is the electron/nucleon split in one sentence. A and B are the only
  // two maps with NO preferred direction -- momentum-free, which is what a
  // handedness for a travelling particle has to be. Every other map has a
  // direction to wind about, which is what circulating a square requires.
  // Six of the fourteen polarize along a cardinal axis, one antipodal pair per
  // axis; those are the six named here.
  const HOP_CH = { A: ['1,1,1', '-1,-1,1', '-1,1,-1', '1,-1,-1'] };
  HOP_CH.B = HOP_CH.A.map(t => t.split(',').map(n => -(+n)).join(','));
  const _POL = { 'X+': [1,0,0], 'X-': [-1,0,0], 'Y+': [0,1,0],
                 'Y-': [0,-1,0], 'Z+': [0,0,1], 'Z-': [0,0,-1] };
  for (const k in _POL) { const n = _POL[k]; HOP_CH[k] = [];
    for (const x of [-1,1]) for (const y of [-1,1]) for (const z of [-1,1])
      if (x*n[0] + y*n[1] + z*n[2] > 0) HOP_CH[k].push([x,y,z].join(',')); }

  // ---- THE TWO CHIRALITIES, L and R -------------------------------------
  // Colour the four base axes and take "toward" to mean a positive component
  // along cardinal up:
  //     a red    (1,1,1)        b green  (1,1,-1)
  //     c blue   (1,-1,1)       d yellow (1,-1,-1)
  //
  //   L :  red toward, blue away, green toward, yellow away
  //   R :  red toward, blue away, green away,   yellow toward
  //
  // Flipping all four is not a third option -- it is the same object seen
  // upside-down, the antipodal vertex.
  //
  // WHY THESE TWO. The apex squares are AC (red/blue) and BD (green/yellow).
  // A square can only be circulated if one of its axes points toward and the
  // other away -- rise on one, fall on the other. Do that for BOTH squares and
  // the map's vector sum lands on a 4-VALENT <100> obtuse vertex of the
  // rhombic dodecahedron: L sums to (4,0,0), R to (0,0,4), both |4|. Let a
  // square's two axes agree instead and the sum lands on a 3-VALENT <111>
  // acute vertex, (2,+-2,2), |2*sqrt3| -- and those cannot circulate.
  // So chirality is the RELATIVE orientation of the two apex squares, same or
  // opposite, and nothing to do with which generation the particle is in.
  HOP_CH.L = ['1,1,1', '1,-1,1', '1,1,-1', '1,-1,-1'];    // == X+ polarization
  HOP_CH.R = ['1,1,1', '1,-1,1', '-1,-1,1', '-1,1,1'];    // == Z+ polarization
  const HOP_CLS = { U: [[0,0,0],[2,0,0],[1,1,1],[1,1,-1]],
                    D: [[0,0,0],[2,0,0],[1,-1,1],[1,-1,-1]] };
  // Where the NEXT tet's canonical origin sits, in this tet's frame.
  const HOP_TO = { 'U|0,1,0':[0,2,0], 'U|0,-1,0':[0,0,0], 'U|1,0,-1':[1,1,-1],
    'U|1,0,1':[1,1,1], 'U|-1,0,-1':[-1,1,-1], 'U|-1,0,1':[-1,1,1],
    'D|0,-1,0':[0,-2,0], 'D|0,1,0':[0,0,0], 'D|1,0,-1':[1,-1,-1],
    'D|1,0,1':[1,-1,1], 'D|-1,0,-1':[-1,-1,-1], 'D|-1,0,1':[-1,-1,1] };
  const HOP_MOV = {
    U: [{ k:'rail', disp:[0,1,0],  add:[[0,2,0],[2,2,0]],   sev:[[0,0,0],[2,0,0]] },
        { k:'rail', disp:[0,-1,0], add:[[1,-1,-1],[1,-1,1]],sev:[[1,1,-1],[1,1,1]] },
        { k:'steer',disp:[1,0,-1], b:[[[1,1,-1],[3,1,-1]],[[2,0,0],[2,0,-2]]] },
        { k:'steer',disp:[1,0,1],  b:[[[1,1,1],[3,1,1]],  [[2,0,0],[2,0,2]]] },
        { k:'steer',disp:[-1,0,-1],b:[[[-1,1,-1],[1,1,-1]],[[0,0,-2],[0,0,0]]] },
        { k:'steer',disp:[-1,0,1], b:[[[-1,1,1],[1,1,1]],  [[0,0,0],[0,0,2]]] }],
    D: [{ k:'rail', disp:[0,-1,0],add:[[0,-2,0],[2,-2,0]], sev:[[0,0,0],[2,0,0]] },
        { k:'rail', disp:[0,1,0], add:[[1,1,-1],[1,1,1]],  sev:[[1,-1,-1],[1,-1,1]] },
        { k:'steer',disp:[1,0,-1], b:[[[1,-1,-1],[3,-1,-1]],[[2,0,0],[2,0,-2]]] },
        { k:'steer',disp:[1,0,1],  b:[[[1,-1,1],[3,-1,1]], [[2,0,0],[2,0,2]]] },
        { k:'steer',disp:[-1,0,-1],b:[[[-1,-1,-1],[1,-1,-1]],[[0,0,-2],[0,0,0]]] },
        { k:'steer',disp:[-1,0,1], b:[[[-1,-1,1],[1,-1,1]],[[0,0,0],[0,0,2]]] }] };
  const hvK = v => v.join(',');
  const hvSub = (a, b) => [0,1,2].map(k => a[k] - b[k]);
  const hvBase = v => v.every(z => Math.abs(z) === 1);
  const hRods = c => c === 'U' ? [[[0,0,0],[2,0,0]], [[1,1,1],[1,1,-1]]]
                               : [[[0,0,0],[2,0,0]], [[1,-1,1],[1,-1,-1]]];
  const hSame = (x, y) => (hvK(x[0]) === hvK(y[0]) && hvK(x[1]) === hvK(y[1]))
                       || (hvK(x[0]) === hvK(y[1]) && hvK(x[1]) === hvK(y[0]));
  const hUnit = (rods, a, b) => hvBase(hvSub(b, a))
    || rods.some(r => (hvK(r[0]) === hvK(a) && hvK(r[1]) === hvK(b))
                   || (hvK(r[0]) === hvK(b) && hvK(r[1]) === hvK(a)));
  const HOP_BD = []; for (const x of [-1,1]) for (const y of [-1,1])
    for (const z of [-1,1]) HOP_BD.push([x,y,z]);
  // Bounded base-hop walk in RELATIVE coordinates from `at` (arrived from
  // `from`) to `target`, obeying chirality and the 60-degree rule at every
  // step. Returns the hop list, or null. Depth 4 is ample: the census says a
  // rod endpoint is 0 or 1 hops from the tet, and the extra depth only buys
  // the detours a polarization needs to arrive from a legal quarter.
  function walkTo(at, from, target, rods, ch) {
    if (hvK(at) === hvK(target)) return [];
    const seen = new Set([hvK(at) + '<' + hvK(from)]);
    let front = [{ at, from, path: [] }];
    for (let d = 0; d < 4 && front.length; d++) {
      const nxt = [];
      for (const st of front) for (const bd of HOP_CH[ch]) {
        const v = bd.split(',').map(Number);
        const to = [st.at[0] + v[0], st.at[1] + v[1], st.at[2] + v[2]];
        if (hvK(to) === hvK(st.from)) continue;              // no a-b-a
        if (!hUnit(rods, st.from, to)) continue;             // 60 degrees
        const path = st.path.concat([{ to: to.slice(), kind: 'base' }]);
        if (hvK(to) === hvK(target)) return path;
        const k = hvK(to) + '<' + hvK(st.at);
        if (seen.has(k)) continue; seen.add(k);
        nxt.push({ at: to, from: st.at, path });
      }
      front = nxt;
    }
    return null;
  }

  let _HOPT = null;
  function hopTable() {
    if (_HOPT) return _HOPT;
    const out = {};
    for (const c of ['U','D']) for (const m of HOP_MOV[c])
      for (const ch of Object.keys(HOP_CH)) {
      const R0 = hRods(c);
      // The agreed severance order: for a steer, ADD BEFORE SEVER so a tet is
      // closed on every tick. Rods pair by axis, b[0] with the old X rod and
      // b[1] with the old Z, which is what keeps the leftover rod from closing
      // a second tet with an incoming one.
      // BOTH ADD ORDERS. m.b is listed [X rod, Z rod], and hardcoding that as
      // the add order made the table non-equivariant under X<->Z: the swap
      // sends U to D and L to R, so L|U|d ought to mirror R|D|swap(d), and it
      // did not -- L lost all four of its +X steers while R lost none. Which
      // rod goes first is a free choice of route, not a property of the move,
      // so try it both ways and keep whichever yields an entry state.
      const plans = m.k === 'rail' ? [[{ add:m.add, sev:m.sev }]]
        : [[{ add:m.b[0], sev:null }, { add:m.b[1], sev:R0[0] },
            { add:null, sev:R0[1] }],
           [{ add:m.b[1], sev:null }, { add:m.b[0], sev:R0[1] },
            { add:null, sev:R0[0] }]];
      const newTet = m.k === 'rail'
        ? [...new Set(R0.filter(r => !hSame(r, m.sev)).concat([m.add])
            .reduce((a, r) => a.concat(r), []).map(hvK))].map(t => t.split(',').map(Number))
        : [...new Set(m.b.reduce((a, r) => a.concat(r), []).map(hvK))]
            .map(t => t.split(',').map(Number));
      const res = [];
      for (const start of HOP_CLS[c]) for (const prev of HOP_CLS[c]) {
        if (hvK(start) === hvK(prev) || !hUnit(R0, prev, start)) continue;
        let found = null;
        for (const plan of plans) {
        let at = start, from = prev, rods = R0.map(r => [r[0].slice(), r[1].slice()]);
        const hops = []; let ok = true;
        for (const step of plan) {
          if (step.add) {
            let did = false;
            for (const [p2, q2] of [[step.add[0], step.add[1]], [step.add[1], step.add[0]]]) {
              // APPROACH WALK, not a single hop. Reaching the endpoint of the
              // rod about to be built can take more than one base hop once a
              // chirality is in force: measured, A and B always manage it in
              // one, but a POLARIZATION often cannot, and capping the walk at
              // one hop is what emptied four of L's twelve moves. Same mistake
              // as gating tetWalk on tet membership, one level down.
              const pre = walkTo(at, from, p2, rods, ch);
              if (!pre) continue;
              const f2 = pre.length ? (pre.length > 1 ? pre[pre.length-2].to : at) : from;
              if (hvK(q2) === hvK(f2)) continue;
              if (!hUnit(rods.concat([[p2, q2]]), f2, q2)) continue;   // 60 degrees
              for (const h of pre) hops.push(h);
              hops.push({ to: q2.slice(), kind: 'sc', add: [p2.slice(), q2.slice()],
                          sev: step.sev ? step.sev.map(z => z.slice()) : null });
              rods = rods.concat([[p2.slice(), q2.slice()]]);
              if (step.sev) rods = rods.filter(r => !hSame(r, step.sev));
              from = pre.length ? p2 : at; at = q2; did = true; break;
            }
            if (!did) { ok = false; break; }
          } else {
            let fin = null;
            for (const pass of [0, 1]) { if (fin) break;
              for (const d of HOP_BD) {
                if (HOP_CH[ch].indexOf(hvK(d)) < 0) continue;
                const to = [at[0]+d[0], at[1]+d[1], at[2]+d[2]];
                if (pass === 0 && !newTet.some(v => hvK(v) === hvK(to))) continue;
                if (hvK(to) === hvK(from) || !hUnit(rods, from, to)) continue;
                fin = to; break;
              } }
            if (!fin) { ok = false; break; }
            hops.push({ to: fin.slice(), kind: 'base', sev: step.sev.map(z => z.slice()) });
            rods = rods.filter(r => !hSame(r, step.sev));
            from = at; at = fin;
          }
        }
        if (!ok) continue;
        const o2 = HOP_TO[c + '|' + hvK(m.disp)];
        found = { start: hvK(start), prev: hvK(prev), hops,
                  exitAt: hvK(hvSub(at, o2)), exitFrom: hvK(hvSub(from, o2)) };
        break;
        }
        if (found) res.push(found);
      }
      out[c + '|' + hvK(m.disp) + '|' + ch] = { kind: m.k, res };
    }
    _HOPT = out;
    return out;
  }
  window._hopTable = hopTable;
  // REPOSITIONING, as the CLOSURE OF A 16-STATE MACHINE.
  //
  // A xon inside its tet is fully described by (which vertex it is on, which
  // vertex it came from) -- four by four, sixteen states, and the legal
  // transitions between them are fixed by chirality, the 60-degree rule and
  // the no-a-b-a rule. Both base edges and ROD TRAVERSALS are transitions:
  // rail entry demands arriving along the kept rod itself, which no base hop
  // can supply, and under one chirality that costs three hops rather than two.
  // Capping the walk at two was why every rail move came back unavailable.
  //
  // This enumerates the whole closure once per (class, chirality) and caches
  // it. It is the transitive closure of a sixteen-entry table -- the lattice is
  // never consulted, and the result is a lookup from then on.
  const _HOPR = {};
  function hopReposition(cls, ch, atRel, fromRel) {
    const ck = cls + '|' + ch;
    if (!_HOPR[ck]) {
      const verts = HOP_CLS[cls], rods = hRods(cls);
      const outs = (at, from) => {
        const o = [];
        for (const v of verts) {
          if (hvK(v) === hvK(at) || hvK(v) === hvK(from)) continue;  // no a-b-a
          if (!hUnit(rods, from, v)) continue;                       // 60 degrees
          const d = hvSub(v, at);
          if (hvBase(d)) {
            if (HOP_CH[ch].indexOf(hvK(d)) < 0) continue;            // chirality
            o.push({ to: v.slice(), kind: 'base' });
          } else if (rods.some(r => hSame(r, [at, v]))) {
            o.push({ to: v.slice(), kind: 'sc' });                   // along a rod
          }
        }
        return o;
      };
      const tbl = {};
      for (const a0 of verts) for (const f0 of verts) {
        if (hvK(a0) === hvK(f0) || !hUnit(rods, f0, a0)) continue;
        const src = hvK(a0) + '<' + hvK(f0);
        const seen = new Set([src]), paths = {};
        let front = [{ at: a0, from: f0, hops: [] }];
        for (let d = 0; d < 6 && front.length; d++) {
          const nxt = [];
          for (const st of front) for (const o of outs(st.at, st.from)) {
            const k = hvK(o.to) + '<' + hvK(st.at);
            if (seen.has(k)) continue; seen.add(k);
            const hops = st.hops.concat([o]);
            paths[k] = { hops, at: hvK(o.to), from: hvK(st.at) };
            nxt.push({ at: o.to, from: st.at, hops });
          }
          front = nxt;
        }
        tbl[src] = paths;
      }
      _HOPR[ck] = tbl;
    }
    const src = atRel + '<' + fromRel;
    const paths = _HOPR[ck][src];
    if (!paths) return [];
    return Object.values(paths).sort((x, y) => x.hops.length - y.hops.length);
  }
  window._hopReposition = hopReposition;
  // Every legal in-tet state of a class: (vertex the xon is on, vertex it came
  // from), the pairs being unit apart. Four by four minus the impossible ones.
  function hopStates(cls) {
    const verts = HOP_CLS[cls], rods = hRods(cls), out = [];
    for (const a of verts) for (const f of verts) {
      if (hvK(a) === hvK(f) || !hUnit(rods, f, a)) continue;
      out.push({ at: hvK(a), prev: hvK(f) });
    }
    return out;
  }
  window._hopStates = hopStates;

  // ---- the tumble ---------------------------------------------------------
  // A STEP is a chain of one or more face-flips run back to back. The
  // neutrino's chain is one flip; the electron's is two, because two is the
  // shortest chain that returns to the same flux mode. Every flip in a chain
  // is its own flux event and gets its own solve -- a chain is a choice of
  // route, never a way to skip physics.
  //
  // Illegal flips are handed back to the chooser as banned and it picks again,
  // so the vacuum has the last word on every event, as it must. A refusal
  // mid-chain abandons the whole chain and rewinds to where it started.
  window._TETLINE = async function (opt) {
    opt = opt || {};
    if (!_L) window._LINE(opt);
    const policy = opt.policy || 'monotone';
    const look = opt.look === undefined ? 1.2 : opt.look;
    const mu = opt.mu === undefined ? 0.35 : opt.mu;
    const exitAt = opt.exit === undefined ? reach().hi : opt.exit;

    // entry: the tet whose centroid is nearest where the line enters the ball
    let T = opt.start;
    if (!T) {
      const s0 = opt.entry === undefined ? reach().lo : opt.entry;
      const E = [_L.p0[0] + s0 * _L.u[0], _L.p0[1] + s0 * _L.u[1], _L.p0[2] + s0 * _L.u[2]];
      let best = null, bd = Infinity;
      for (const t of allTets()) {
        const d = Math.hypot(t.c[0] - E[0], t.c[1] - E[1], t.c[2] - E[2]);
        if (d < bd) { bd = d; best = t; }
      }
      if (!best) throw new Error('_TETLINE: no tet anywhere in this lattice');
      T = best.v;
    }

    // the locked mode, if any, is whatever the entry tet happens to be
    const lock = opt.lock ? (opt.lock === true ? isTet(T).map(s => s[2]).sort().join('') : opt.lock)
                          : null;

    // Step budget from the lattice, not a magic number. A chain advances at
    // least `perStep` along the line -- exactly 1 when mode-locked, since the
    // electron's step length is provably one lattice unit; 0.35 otherwise,
    // comfortably under the slowest crossing measured (0.438). Each chain is
    // `chainLen` flux events.
    //
    // An oversized budget is NOT free. Every leftover item after the tet has
    // exited still calls restate(), and a cache hit there re-runs detect() and
    // measure(), both O(n^2) on 339 nodes. A flat 200 spent seconds of blocked
    // main thread doing nothing, per crossing.
    // budget is in FLUX EVENTS, so it is per-event drift that matters. The
    // mode-locked walker manages at best 0.577/event and pays for line
    // following out of that; 0.25 is a floor well under anything measured.
    const span = reach().hi - reach().lo;
    const maxSteps = opt.steps || Math.ceil(span / (lock ? 0.25 : 0.35)) + 10;

    let prevKey = null, banned = new Set(), log = [], done = false;
    let pending = null, lastT = null, revisits = 0, events = 0;
    let queued = [], chainStart = null;
    const visited = new Set([vkey(T)]);
    const path = [centroid(T)]; drawTrail(path);

    const choose = () => {
      const here = sOf(centroid(T));
      const raw = lock ? modeChains(T, lock)
                       : faceFlips(T).map(f => ({ chain: [f.v], v: f.v, mode: f.mode,
                                                  key: vkey(f.v), drop: f.drop, add: f.add }));
      let opts = raw.map(m => {
        const c = centroid(m.v), s = sOf(c);
        return { m, c, s, ds: s - here, perp: perpOf(c), key: m.key, dest: vkey(m.v),
                 len: m.len || 1 };
      }).filter(o => !banned.has(o.key));
      if (!opts.length) return null;

      if (policy === 'monotone') {
        // hard constraint first: s must strictly increase.  Proven satisfiable.
        const fwd = opts.filter(o => o.ds > 1e-9);
        if (fwd.length) opts = fwd;                       // if empty, the vacuum
        else return null;                                 // banned every forward move
        // chains differ in length, so reward progress PER FLUX EVENT -- else a
        // slow 4-flip excursion outscores a fast 3-flip one just by going
        // further in total
        opts.sort((a, b) => (a.perp - mu * a.ds / a.len) - (b.perp - mu * b.ds / b.len)
                            || b.ds / b.len - a.ds / a.len);
      } else if (policy === 'greedy') {
        opts.sort((a, b) => b.ds / b.len - a.ds / a.len || a.perp - b.perp);
      } else {                                            // pursuit (kept: it cycles)
        const t = k => _L.p0[k] + (here + look) * _L.u[k];
        for (const o of opts)
          o.cost = Math.hypot(o.c[0] - t(0), o.c[1] - t(1), o.c[2] - t(2));
        const fresh = opts.filter(o => o.dest !== prevKey);
        opts = fresh.length ? fresh : opts;
        opts.sort((a, b) => a.cost - b.cost || b.ds - a.ds);
      }
      return opts[0];
    };

    await runExperiment(
      `tet tumbling along a line  dir=${_L.u.map(v => v.toFixed(3)).join(',')}`,
      Array.from({ length: maxSteps }, (_, k) => ({ k, label: 'step ' + k })),
      () => {
        if (done) return;
        if (opt.alive && !opt.alive()) { done = true; pending = null; return; }
        // the lattice was resized under us: node indices mean something else
        // now, so this crossing is meaningless. Abandon it; the next one
        // re-enumerates against the new lattice.
        if (!isTet(T)) { done = true; pending = null; return; }
        if (!queued.length) {                          // start a new chain
          pending = choose();
          if (!pending) { done = true; return; }
          chainStart = T; queued = pending.m.chain.slice();
        }
        lastT = T; T = queued.shift(); writeTet(T);
      },
      (it) => {
        if (done && !pending) return { html: '<span style="color:#5d6e85">—</span>', skip: true };
        if (!pending) return { html: 'no move available', skip: true };
        detect();
        events++;
        if (!legal(resid)) {          // vacuum refused: abandon the CHAIN, rewind, ban
          banned.add(pending.key); queued = []; T = chainStart; writeTet(T); freezeOff();
          log.push({ step: it.k, rejected: true, key: pending.key });
          pending = null;
          return { html: '<span style="color:#ff5c5c">vacuum refused</span> chain abandoned' };
        }
        path.push(centroid(T)); drawTrail(path);
        if (queued.length) {                           // mid-chain: a real flux event
          const m = isTet(T).map(s => s[2]).sort().join('');
          return { html: '<span style="color:#8fa3ba">via ' + m + '</span>' };
        }
        banned = new Set(); prevKey = vkey(chainStart);
        if (visited.has(pending.dest)) revisits++; else visited.add(pending.dest);
        const c = pending.c, s = pending.s, pp = pending.perp;
        let mx = 0; for (let i = 0; i < NODE.length; i++)
          mx = Math.max(mx, P[i].distanceTo(REST[i]));
        log.push({ step: it.k, s: +s.toFixed(3), perp: +pp.toFixed(3),
                   c: c.map(x => +x.toFixed(2)), mode: pending.m.mode,
                   via: pending.m.via || null, flips: pending.m.chain.length,
                   tets: solids.tets.length, sc: active.size, maxDisp: +mx.toFixed(4) });
        pending = null;
        if (s >= exitAt) done = true;
        return { html: `<span style="color:#7fd4a8">ok</span> `
                     + `s=${s.toFixed(2)} perp=${pp.toFixed(2)} `
                     + `${log[log.length - 1].mode} tets=${solids.tets.length}`
                     + (done ? ' <b>exit</b>' : '') };
      },
      // paced only while a loop is driving, so measurement runs stay full speed
      { onFreeze: () => freezeOff(),
        pace: () => (window._loopNow && window._loopNow() !== 'none'
                     && window._loopPace) ? window._loopPace() : 0 });

    const mv = log.filter(r => !r.rejected);
    const perp = mv.map(r => r.perp);
    const mean = a => a.reduce((x, y) => x + y, 0) / Math.max(1, a.length);
    const out = {
      seed: _L.seed, dir: _L.u.map(v => +v.toFixed(5)), centre: _L.p0, policy, mu, look, lock,
      moves: mv.length, fluxEvents: events, refused: log.length - mv.length, revisits,
      // per EVENT, counted over the same intervals as driftPerMove -- i.e.
      // excluding the events that produced the first logged position
      driftPerEvent: mv.length > 1
        ? +((mv[mv.length - 1].s - mv[0].s)
            / mv.slice(1).reduce((n, r) => n + (r.flips || 1), 0)).toFixed(4) : null,
      sFrom: mv.length ? mv[0].s : null, sTo: mv.length ? mv[mv.length - 1].s : null,
      travelled: mv.length ? +(mv[mv.length - 1].s - mv[0].s).toFixed(3) : 0,
      driftPerMove: mv.length > 1
        ? +((mv[mv.length - 1].s - mv[0].s) / (mv.length - 1)).toFixed(4) : null,
      perpMean: +mean(perp).toFixed(3), perpMax: +Math.max(...perp, 0).toFixed(3),
      exited: mv.length ? mv[mv.length - 1].s >= exitAt : false,
      ranOut: !done && mv.length < maxSteps,
      modes: mv.map(r => r.mode).join(' '),
      maxDisp: [Math.min(...mv.map(r => r.maxDisp)), Math.max(...mv.map(r => r.maxDisp))],
      // geometric ceiling on one step's progress: half the sum of the two
      // largest |u| components, since every flip displaces by (1/2)<110>
      stepCeiling: +((a => (a[0] + a[1]) / 2)(_L.u.map(Math.abs).sort((x, y) => y - x))).toFixed(4),
      path, log
    };
    window._TL = out;
    console.log('TETLINE', JSON.stringify({ policy, moves: out.moves,
      refused: out.refused, revisits: out.revisits,
      travelled: out.travelled, driftPerMove: out.driftPerMove,
      perpMean: out.perpMean, perpMax: out.perpMax, exited: out.exited,
      ranOut: out.ranOut, maxDisp: out.maxDisp }, null, 1));
    return out;
  };

  // ---- robustness: many random lines --------------------------------------
  // The point of the exercise.  A rule that only works down one line is a
  // coincidence; this runs the same rule down n independent directions and
  // reports every crossing, so a failure cannot hide in an average.
  window._TETLINES = async function (n, opt) {
    n = n || 6; opt = opt || {};
    const rows = [];
    for (let k = 0; k < n; k++) {
      const seed = (opt.seed0 === undefined ? 1000 : opt.seed0) + k * 7717;
      window._LINE({ seed });
      const o = await window._TETLINE(Object.assign({ steps: 200 }, opt));
      rows.push({ seed, dir: o.dir, moves: o.moves, refused: o.refused,
        revisits: o.revisits, travelled: o.travelled, driftPerMove: o.driftPerMove,
        stepCeiling: o.stepCeiling, perpMean: o.perpMean, perpMax: o.perpMax,
        reachedEdge: !o.ranOut });
      console.log('line', k, JSON.stringify(rows[k]));
    }
    window._TLS = rows;
    console.log('TETLINES', JSON.stringify(rows, null, 1));
    return rows;
  };

  // ---- the neutrino loop --------------------------------------------------
  // Endless: pick a random direction through the centre, tumble a tet the
  // whole way across, pick another.  Named for Zac's reading of the mode
  // sequence -- every face-flip is forced to swap one flux mode for another,
  // so a travelling tet cannot help cycling XY -> YZ -> XZ as it goes.  Along
  // <111> that cycle is exactly period 3 and never breaks.
  //
  // One crossing is ~21-26 flux events, and each event is a full solve of the
  // 339-node lattice, so it moves at the speed of the physics, not the speed
  // of an animation.  There is no shortcut to make it faster that would leave
  // the geometry honest.
  // MUST be awaited once per crossing by every loop. settle() resolves
  // SYNCHRONOUSLY when the solve queue is already empty, which is exactly what
  // happens on a fully cached configuration -- so a crossing that is entirely
  // cache hits returns without ever reaching a macrotask, and `while (alive)
  // await crossing()` becomes a tight microtask loop that locks the renderer
  // out completely. This froze the tab the first time the electron loop ran,
  // because the electron has only six possible lines and cached all of them.
  const breathe = () => new Promise(r => setTimeout(r, 60));

  // 'no loop' must leave nothing behind: no tet, no line, no trail. Only runs
  // if nothing else has taken over in the meantime -- a loop's own cleanup
  // fires after its last crossing drains, which can be after the NEXT loop has
  // already put its tet down.
  const cleanup = () => {
    if (window._loopNow && window._loopNow() !== 'none') return;
    active.clear(); window._LINEOFF(); window._TRAILOFF(); window._XONOFF(); window._ATTROFF();
    lastChange = 'loop stopped'; restate(true);
  };
  window._loopCleanup = cleanup;

  // BETWEEN RUNS: take the particle away, then let the lattice actually relax
  // before the next one spawns.
  //
  // Clearing the shortcuts is itself a flux event -- the release propagates
  // outward at c exactly like the events that laid them -- so the next run must
  // not begin until those cones have drained. Otherwise the new particle's
  // geometry superposes on the old one's still-arriving relaxation, and what is
  // on screen belongs to two particles at once.
  const settleBetweenRuns = async (tok) => {
    // THE TRAVERSAL LINE STAYS UP through the settle. The particle is gone but
    // its news is not: the relaxation is still travelling out along that line,
    // so the line is the one thing on screen that still means something. It is
    // replaced when the next run draws its own, and cleared by cleanup() when
    // the loop is stopped.
    if (window._XONOFF) window._XONOFF();
    if (window._TRAILOFF) window._TRAILOFF();
    if (active.size) {
      active.clear(); lastChange = 'run ended';
      restate(true); await settle();
    }
    for (let i = 0; i < 4000; i++) {
      if (!window._loopAlive(tok)) return;
      if (!window._FLUX || !window._FLUX().running) break;
      await new Promise(r => requestAnimationFrame(r));
    }
    await breathe();
  };

  // THE SIMPLE NEUTRINO LOOP IS RETIRED. A tet tumbling the full width of the
  // lattice on a random line, one flux event per step, with no mode rule, no
  // chirality and no 60-degree constraint on the xon -- it predated all of
  // them and was superseded by 'neutrino (mode-cycling)', which is the same
  // crossing under the rules the rest of the CAs are held to. It survives as
  // _TETLINE({policy:'monotone'}) for control runs; it is no longer a loop.

  // ---- the electron loop --------------------------------------------------
  // Same tumble, same solver, same arbitrary random line. One rule added: the
  // flux mode may not change. Born with XY, it carries an X shortcut and a Y
  // shortcut the whole way across.
  if (window._registerLoop) {
;
  }

  // ==========================================================================
  // TOWARDS A CELLULAR AUTOMATON  --  measured, not yet built.
  //
  // The simple loops IMPOSE the shortcuts. The next model should have them be
  // whatever something moving leaves behind. These are the measurements that
  // constrain such a rule; none of it is implemented yet.
  //
  // (1) WHAT A FACE-FLIP IS, exactly. Over all 5184 flips in the 339-node
  //     lattice, with no exceptions whatsoever:
  //       - exactly ONE of the two shortcuts survives untouched
  //       - the destroyed and created shortcuts share exactly ONE vertex
  //       - the swing about that shared vertex is 90 degrees. 5184 of 5184.
  //     So a flip is one <100> ROD PIVOTING 90 DEGREES ABOUT ONE END. That is
  //     already a local rule, and it is not one anybody wrote down -- it fell
  //     out of "consecutive tets share a face".
  //
  // (2) A WALK DECOMPOSES INTO TWO RODS TAKING TURNS. On a 24-step monotone
  //     crossing: exactly one rod moves per tick (0 ticks where both moved),
  //     and consecutive swings of the SAME rod are 90 degrees apart every
  //     time (0 of 22 violations). The rods do not alternate strictly -- one
  //     took 13 turns and the other 11.
  //
  // (3) THE ROD WALKS END OVER END. The new pivot is the end just planted in
  //     19 of 22 consecutive pairs; in the other 3 the same pivot is reused
  //     and the same end swings twice.
  //
  // Which leaves ONE open question, and it is a physics choice, not a coding
  // one: what is the xon?
  //
  //   (a) the xon owns the rod, and the shortcut is its last <100> hop. Then
  //       19 of 22 moves are a single hop, and the 3 pivot-reuse moves need
  //       the xon to retrace along its own rod first -- two hops, one tick
  //       "wasted". Costs differ per move, which is measurable.
  //   (b) a xon sits at each rod END (four xons). A swing is then one xon
  //       travelling from -> pivot -> to, two <100> hops, every time.
  //
  // Note (b)'s constraint: the free end moves by 2*sqrt(2) integer units,
  // which is the THIRD shell (1.633 base edges) and NOT one of FD's ten
  // traversal directions. So the swing cannot be a single xon step under any
  // reading -- it must go through the pivot. That is a real result: the
  // 90-degree swing is two flux events, not one.
  // ==========================================================================

  // ==========================================================================
  // SHARED XON MACHINERY  --  used by the momentum xon below.
  //
  // ONE xon. A 0-dimensional spark on a 1-dimensional continuous path: every
  // tick it takes exactly one lattice edge, and it may not jump.
  //
  //   base hop  (<111>)  -- no geometric change. Transit.
  //   shortcut hop (<100>) -- the pair is pulled to unit length. A flux event.
  //
  // A tet's two shortcuts are OPPOSITE edges, disjoint, so one xon can never
  // make them back to back -- it has to walk from one rod to the other. All
  // four cross-pairs of a tet are base edges, so that walk is always a single
  // base hop. The 1:1 base/shortcut rhythm is not imposed; the tet's own
  // geometry produces it.
  //
  // WHICH SHORTCUT DIES. Not the oldest -- that is wrong 3 times in 24, at
  // exactly the flips where the same rod swings twice running. The rule that
  // is always right is local and needs no memory:
  //
  //   the new rod SHARES A VERTEX with the rod it replaces (5184 of 5184),
  //   and is DISJOINT from the rod that survives.
  //
  // So on a shortcut hop the xon severs whichever live shortcut touches the
  // one it just made. Nothing counts ticks; nothing looks at the tet.
  //
  // REMOVED: _XONLINE / 'electron loop (xon)', the tet-led predecessor, in
  // which a tet-level rule chose the tets and the xon walked BFS base paths to
  // realise them. Superseded by the momentum xon. Its measurements are kept in
  // observation-tet-traversal.md because they are what motivated the 60-degree
  // rule: mean turn ~120 degrees, 125 degrees on 91% of hops, straightness
  // 0.27, and no steering weight ever fixed it.
  // ==========================================================================
  // Shortest walk on base edges only; `cost` breaks ties between equally short
  // geodesics. No longer used by any loop -- the momentum xon never needs to
  // path anywhere, since the 60-degree rule keeps it inside its own tet. Kept
  // as an exported measurement tool (`window._basePath`).
  function basePath(from, to, cost) {
    _sync();
    if (from === to) return [];
    const prev = new Map([[from, -1]]); let frontier = [from];
    while (frontier.length) {
      const next = [];
      for (const v of frontier) {
        const nbr = cost ? _baseNbr[v].slice().sort((a, b) => cost(v, a) - cost(v, b))
                         : _baseNbr[v];
        for (const w of nbr) {
        if (prev.has(w)) continue;
        prev.set(w, v);
        if (w === to) {
          const p = []; let c = w;
          while (c !== from) { p.unshift(c); c = prev.get(c); }
          return p;
        }
        next.push(w);
        }
      }
      frontier = next;
    }
    return null;                             // base graph disconnected: impossible
  }

  function scKeyOf(i, j) {
    for (let a = 0; a < AXN.length; a++) {
      if (SCOPT.get(i + ':' + a) === j) return [i + ':' + a, [i, j]];
      if (SCOPT.get(j + ':' + a) === i) return [j + ':' + a, [j, i]];
    }
    return null;
  }

  // xon + its wake, drawn.
  //
  // CAUSALITY. paint() runs every frame while the solver is still working, so
  // the new shortcut and the deforming lattice are on screen immediately. If
  // the spark is only redrawn after the solve completes it sits at its old
  // node for the whole solve and the geometry appears to move FIRST -- which
  // is exactly backwards for a model whose whole claim is that the xon causes
  // the deformation. So: draw the xon the instant it hops, before restate(),
  // and track its node every frame via _onFrame while the lattice relaxes.
  let _spark = null, _wake = null, _xonNode = null;
  window._XONNODE = () => _xonNode;

  // Layer dimmer, driven by the 'xon' slider. Held outside the meshes because
  // they are created lazily on the first hop — a level set while no loop is
  // running has to survive until there is something to apply it to.
  const XON_A = 1, WAKE_A = 0.55;      // opacity at 100%
  let _xonDim = 1;
  function applyXonDim() {
    if (!_spark) return;
    _spark.material.opacity = XON_A * _xonDim;
    _wake.material.opacity = WAKE_A * _xonDim;
  }
  window._XONDIM = (v) => {
    _xonDim = Math.max(0, Math.min(1, v));
    applyXonDim();
    if (_spark && _xonDim === 0) { _spark.visible = false; _wake.visible = false; }
  };
  // The layers panel is built before this file loads, so adopt whatever level
  // came back from saved state instead of assuming 100%.
  { const s = document.getElementById('s-xon'); if (s) _xonDim = +s.value / 100; }

  window._onFrame = () => {
    // The spark rides the DRAWN geometry, not the solved geometry -- otherwise
    // it floats off the node it is standing on while that node's own news is
    // still arriving.
    if (_spark && _spark.visible && _xonNode !== null) {
      const D = window._PDRAW ? window._PDRAW() : P;
      if (D[_xonNode]) _spark.position.copy(D[_xonNode]);
    }
  };
  function drawXon(node, trail) {
    // THE SIMULATION TICK. Every CA moves its xon through here, so this is the
    // one place that knows a tick happened -- and the flux wave is clamped to
    // that count, so a stalled solver stalls the propagation with it. Arming
    // the clamp here also means it is armed exactly when a particle exists.
    if (node !== _xonNode && node !== null && node !== undefined) {
      if (window._SIMTICK) window._SIMTICK();
      if (window._FLUXCLAMP) window._FLUXCLAMP(true);
    }
    _xonNode = node;
    if (!_spark) {
      _spark = new THREE.Mesh(new THREE.SphereGeometry(0.13, 16, 12),
        new THREE.MeshBasicMaterial({ color: 0x40e0ff, transparent: true, opacity: XON_A }));
      _spark.frustumCulled = false; scene.add(_spark);
      _wake = new THREE.Line(new THREE.BufferGeometry(),
        new THREE.LineBasicMaterial({ color: 0x40e0ff, transparent: true, opacity: WAKE_A }));
      _wake.frustumCulled = false; scene.add(_wake);
      applyXonDim();
    }
    _spark.position.copy(P[node]); _spark.visible = _xonDim > 0;
    const a = new Float32Array(Math.max(3, trail.length * 3));
    trail.forEach((n, k) => { a[k * 3] = P[n].x; a[k * 3 + 1] = P[n].y; a[k * 3 + 2] = P[n].z; });
    _wake.geometry.setAttribute('position', new THREE.BufferAttribute(a, 3));
    _wake.geometry.setDrawRange(0, trail.length);
    _wake.visible = trail.length > 1 && _xonDim > 0;
  }
  // No xon means no tick to sync to, so the clamp comes off -- otherwise the
  // waves left over from the last run could never drain and settleBetweenRuns
  // would wait for ever.
  window._XONOFF = () => { if (_spark) { _spark.visible = false; _wake.visible = false; }
    _xonNode = null; if (window._FLUXCLAMP) window._FLUXCLAMP(false); };

  // ---- the attractor, drawn ------------------------------------------------
  // A small white ball at the deepest hole -- the point the xon is pulled
  // toward. It sits BETWEEN nodes, so it is the one thing on screen that is not
  // on the graph. Watching it move is the clearest read on whether the well the
  // xon is digging is the one it ends up in.
  let _attr = null, _attrDim = 1;
  function applyAttrDim() {
    if (!_attr) return;
    _attr.material.opacity = _attrDim;
    if (_attrDim === 0) _attr.visible = false;
  }
  window._ATTRDIM = (v) => { _attrDim = Math.max(0, Math.min(1, v)); applyAttrDim(); };
  { const s = document.getElementById('s-attractor'); if (s) _attrDim = +s.value / 100; }
  function drawAttractor(pt) {
    if (!_attr) {
      _attr = new THREE.Mesh(new THREE.SphereGeometry(0.085, 16, 12),
        new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 1 }));
      _attr.frustumCulled = false; scene.add(_attr);
      applyAttrDim();
    }
    if (!pt) { _attr.visible = false; return; }
    _attr.position.set(pt[0], pt[1], pt[2]);
    _attr.visible = _attrDim > 0;
  }
  window._ATTROFF = () => { if (_attr) _attr.visible = false; };

  // ==========================================================================
  // A YZ OCTAHEDRON AT THE CENTRE, and the cardinal-up arrow.
  //
  // The equator of an oct is four shortcuts closing a square, and the PAIR OF
  // AXES that square uses IS the flux mode. So a YZ oct is the ring
  //     i --Y+--> a --Z+--> b <--Y+-- d <--Z+-- i
  // with X left dormant. Its two apexes are then forced, not chosen: they are
  // the nodes base-joined to all four ring vertices, and they sit at +-X, so
  // the two apex squares are the ones chirality actually bites on.
  //
  // The ring's centre lands on a HALF-INTEGER point -- centroid (0,1,1) for a
  // ring anchored at the origin -- because ring vertices share a parity and
  // their mean cannot. So there is no YZ ring centred exactly on the lattice
  // centre; the search below takes the nearest one and reports how far off it
  // is rather than pretending otherwise.
  // ==========================================================================
  window._YZOCT = async function (opt) {
    opt = opt || {};
    _sync();
    const AY = 2, AZ = 4;                       // AX indices for Y+ and Z+
    let best = null;
    for (let i = 0; i < NODE.length; i++) {
      const a = SCOPT.get(i + ':' + AY); if (a === undefined) continue;
      const d = SCOPT.get(i + ':' + AZ); if (d === undefined) continue;
      const b = SCOPT.get(a + ':' + AZ); if (b === undefined) continue;
      if (SCOPT.get(d + ':' + AY) !== b) continue;      // must close the square
      const ctr = [0, 1, 2].map(k =>
        (NODE[i][k] + NODE[a][k] + NODE[b][k] + NODE[d][k]) / 4);
      const off = Math.hypot(...[0, 1, 2].map(k => ctr[k] - LCENTER[k]));
      if (!best || off < best.off) best = { i, a, b, d, ctr, off };
    }
    if (!best) return { ok: false, why: 'no YZ ring in this lattice' };
    if (!opt.keep) active.clear();
    const keys = [[best.i, AY, best.a], [best.i, AZ, best.d],
                  [best.a, AZ, best.b], [best.d, AY, best.b]];
    for (const [n, ax2, j] of keys) active.set(n + ':' + ax2, [n, j]);
    restate(true); await settle(); detect();
    const ring = [best.i, best.a, best.b, best.d];
    const apex = [];
    for (let v = 0; v < NODE.length; v++)
      if (!ring.includes(v) && ring.every(r => _baseSet.has(K(v, r)))) apex.push(v);
    return { ok: true, ring, ringXYZ: ring.map(v => NODE[v].join(',')),
      apex, apexXYZ: apex.map(v => NODE[v].join(',')),
      centre: best.ctr, offCentre: +best.off.toFixed(4),
      rods: keys.map(([n, , j]) => n + '-' + j),
      vacuum: legal(resid), octs: solids.octs.length, tets: solids.tets.length };
  };


  // THE FLUX PROPAGATION LOOP IS RETIRED. It placed one shortcut on a blank
  // lattice, held it, severed it, and let the wave run -- which was the right
  // demonstration while propagation was a special mode. It is not one any more:
  // restate() emits a wave from whatever rods changed, so every flux event in
  // the simulator does this, whether it comes from a click, a CA, or anything
  // else. A dedicated loop for it only invited the reading that propagation is
  // something you switch on.
  //
  // `_CAUSAL`, the harness behind it, went too. Its measurement modes ('front', the
  // shell-by-shell wave, and 'cone') produced the numbers that settled how
  // propagation should work -- that a clamped cone accumulates strain and
  // releases it all at once, that a front-only wave propagates but leaves a
  // permanent 5% wake, and that validate-then-reveal does neither. Those are
  // recorded in the session notes; the code is gone because its reveal path
  // wrote to P, and rendering has read PD since the engine took the job over,
  // so it could no longer draw anything.

  // CARDINAL UP. Not a lattice object -- a reference direction, so the two
  // chiralities can be read off the screen: "toward" means a positive
  // component along this arrow.
  let _upArrow = null, _upLen = null;
  window._UPARROW = function (on, len) {
    if (on === false) { if (_upArrow) _upArrow.visible = false; return; }
    // Rebuild on a length change: the geometry is baked, so resizing means
    // replacing it rather than scaling a cached group.
    if (_upArrow && len && len !== _upLen) {
      scene.remove(_upArrow);
      _upArrow.traverse(o => { if (o.geometry) o.geometry.dispose(); });
      _upArrow = null;
    }
    if (!_upArrow) {
      const L = _upLen = (len || 1.9), headL = L * 0.28,
            headR = L * 0.10, shaftR = L * 0.026;
      const mat = new THREE.MeshBasicMaterial({ color: 0x22ff66,
        transparent: true, opacity: 0.9, depthTest: false });
      _upArrow = new THREE.Group();
      const shaft = new THREE.Mesh(
        new THREE.CylinderGeometry(shaftR, shaftR, L - headL, 16), mat);
      shaft.position.y = (L - headL) / 2;
      const head = new THREE.Mesh(new THREE.ConeGeometry(headR, headL, 20), mat);
      head.position.y = L - headL / 2;
      _upArrow.add(shaft); _upArrow.add(head);
      _upArrow.renderOrder = 999;
      _upArrow.frustumCulled = false;
      scene.add(_upArrow);
    }
    _upArrow.position.set(LCENTER[0] * S, LCENTER[1] * S, LCENTER[2] * S);
    _upArrow.visible = true;
    return 'up arrow on, +Y';
  };

  // Realise a tet-flip sequence with a single walking xon. The tet-level rule
  // still chooses WHERE to go; what changes is that the shortcuts are now made
  // by the spark rather than written down.
  // ==========================================================================
  // MOMENTUM XON  --  the inversion. There is NO LINE. The xon carries a
  // conserved momentum and the tet goes wherever the xon takes it.
  //
  // BALANCE AND MOMENTUM, AND WHY THEY ARE NOT IN CONFLICT.
  //
  // Balancing ALL directions is not a law of motion. The eight signed base
  // <111> directions sum to exactly zero and so do the six shortcut <100>
  // directions, so a xon using every direction equally has exactly zero net
  // displacement, by arithmetic. Uniform balance is the condition for being at
  // REST. An earlier build here optimised that and then wondered why the
  // particle would not travel.
  //
  // The resolution is in FD's own counting: TEN traversal directions, but as
  // FOUR base plus SIX shortcut. Four, not eight -- the base directions are
  // AXES, agnostic to sense by construction; the shortcuts are counted signed.
  // So the balance principle applies asymmetrically:
  //
  //   6 shortcut SENSES balanced  -> they sum to zero -> flux events
  //                                  transport the particle not at all
  //   4 base AXES balanced, but +/- free within each axis
  //                               -> net displacement = sum (n+ - n-) d_axis,
  //                                  which can be ANY vector
  //
  // So the momentum lives entirely in the sign choices within the base axes,
  // and nothing about balance forbids motion. The geometry changes are
  // sense-symmetric; the travel is carried by the base lattice. (Which is the
  // shape of the Frank/Eshelby split -- a directed Burgers-like quantity in
  // the base lattice, a sense-symmetric shear event on top. Stated as a
  // reading, not a result.)
  //
  // THE TET IS A REQUIREMENT, NOT AN OBSERVATION. The xon's job as an electron
  // is to keep a tumbling tet manifest and move it. So: base hops are always
  // allowed (they leave the rods untouched, so the tet survives), and a
  // shortcut hop is allowed ONLY if the pair it leaves behind is still a tet.
  // The tet never lapses. What is emergent is where it GOES, and that is now
  // the xon's momentum rather than a line handed to it.
  // ==========================================================================
  window._XONMOM = async function (opt) {
    opt = opt || {};
    _sync();
    const mu = opt.mu === undefined ? 0.35 : opt.mu;
    // Balance is NOT imposed by default. Base-axis balance is exactly the
    // condition for zero momentum, and shortcut-sense balance is more
    // interesting as an OUTPUT than as an input -- if it emerges it is a
    // result, if it is weighted in it is an assumption. Set opt.bal to
    // reintroduce it for comparison.
    const bal = opt.bal === undefined ? 0 : opt.bal;
    const TRANSIT = opt.transit === undefined ? 0.05 : opt.transit;  // cost of a hop
                                                          // that moves no geometry
    // ticks without gaining ground before we call it cycling rather than
    // travelling. Declared HERE because the tick budget below uses it.
    const STALL = opt.stall === undefined ? 32 : opt.stall;
    // Declared up here because the START TET choice depends on it.
    const lockAxes = !!opt.lockAxes;
    const axisOf = d => (d[0] ? 'X' : (d[1] ? 'Y' : 'Z'));
    let axesBlocked = 0;

    // ---- BANNED SHORTCUT AXES --------------------------------------------
    // opt.banAxis: 'Y', or a list. Shortcuts on these axes DO NOT EXIST for
    // this particle -- not "are avoided", do not exist. The ban is applied
    // wherever a shortcut is proposed (candidates, planner, chain expansion,
    // start tet), so there is one rule and no path around it.
    //
    // This is a statement about the lattice rather than about the walker, and
    // it decides more than it looks like it does. A tet is two rods on DISTINCT
    // axes, so banning Y leaves only {X,Z} -- the generation is XZ by
    // construction, not by bookkeeping, and nothing has to promise to hold it.
    const BAN = opt.banAxis
      ? (Array.isArray(opt.banAxis) ? opt.banAxis.slice() : [opt.banAxis])
      : null;
    let banBlocked = 0;

    // CHIRALITY. Of the 16 sign patterns over the four base axes, exactly TWO
    // are zero-sum -- and they are exact negations of each other:
    //   A = { (1,1,1) (-1,-1,1) (-1,1,-1) (1,-1,-1) }      B = -A
    // Measured over the whole lattice: all 1392 tets have their base 4-cycle
    // lying ENTIRELY in one set or the other, never mixed (other: 0). So a xon
    // restricted to one set winds every tet one way and the other set winds
    // every tet the other way. Handedness is global and binary, not per-solid.
    //
    // opt.chirality: 'A' | 'B' | null (unrestricted) | undefined (drawn from seed)
    const CHIRAL_A = ['1,1,1', '-1,-1,1', '-1,1,-1', '1,-1,-1'];
    const CHIRAL_B = CHIRAL_A.map(s => s.split(',').map(n => -(+n)).join(','));
    // A polarization ('X+','Y-',...) is as valid a sense map as A or B; the
    // hop table already knows them, so the runtime gate must too.
    const CHIRAL_POL = HOP_CH;
    let chirality = opt.chirality;
    if (chirality === undefined) {
      const cs = opt.seed === undefined ? (Math.random() * 2147483647) | 0 : opt.seed;
      chirality = _mul(cs ^ 0x5bf03635)() < 0.5 ? 'A' : 'B';
    }
    const chiralSet = chirality === null ? null
      : new Set(CHIRAL_POL[chirality]
                || (chirality === 'A' ? CHIRAL_A : CHIRAL_B));
    const chiralOK = (a, b) => !chiralSet
      || chiralSet.has([0, 1, 2].map(k => NODE[b][k] - NODE[a][k]).join(','));
    let chiralBlocked = 0;      // base hops refused for wrong handedness

    // momentum: a direction, fixed for the life of the particle
    let v = opt.dir;
    if (!v) { const r = _mul(opt.seed === undefined ? (Math.random() * 2147483647) | 0 : opt.seed);
      let x, y, q; do { x = 2 * r() - 1; y = 2 * r() - 1; q = x * x + y * y; }
      while (q >= 1 || q === 0);
      const t = 2 * Math.sqrt(1 - q); v = [x * t, y * t, 1 - 2 * q]; }
    const vn = Math.hypot(...v); v = v.map(x => x / vn);

    // start on a tet near the upstream edge, xon on one of its nodes
    let lo = 0; for (const c of NODE) { const s = (c[0] - LCENTER[0]) * v[0]
      + (c[1] - LCENTER[1]) * v[1] + (c[2] - LCENTER[2]) * v[2]; if (s < lo) lo = s; }
    const E0 = [0, 1, 2].map(k => LCENTER[k] + lo * v[k]);
    // A MODE-LOCKED TET TRAVELS ALONG ITS DORMANT AXIS, AND ONLY THAT.
    // Measured: the centroid displacement set of a mode-preserving flip has
    // RANK 1 -- mode XZ moves +/-Y, XY moves +/-Z, YZ moves +/-X. So the birth
    // mode must be chosen to match the direction of travel, or the particle
    // cannot advance at all: picking the start tet by proximity alone gave 0
    // flux events whenever the nearest tet had the wrong mode.
    let wantMode = null;
    if (lockAxes) {
      const av = [Math.abs(v[0]), Math.abs(v[1]), Math.abs(v[2])];
      wantMode = ['YZ', 'XZ', 'XY'][av.indexOf(Math.max(...av))];
      // The ban outranks the direction. Asking for YZ while Y is banned names a
      // tet that cannot exist, and the search would come back empty -- so drop
      // the preference and let the ban pick the mode, which it does uniquely.
      if (BAN && BAN.some(a => wantMode.indexOf(a) >= 0)) wantMode = null;
    }
    let best = null, bd = Infinity, hiTet = -Infinity;
    // A tet built from a banned axis cannot be a start tet -- it is not a tet
    // this particle can occupy at all.
    const tetOK = t => !BAN || t.sc.every(s => BAN.indexOf(s[2]) < 0);
    for (const t of allTets()) { const d = Math.hypot(...[0, 1, 2].map(k => t.c[k] - E0[k]));
      if ((!wantMode || t.mode === wantMode) && tetOK(t) && d < bd) { bd = d; best = t; }
      // furthest along the ray any tet centroid can sit: the real finish line,
      // measured off the lattice rather than picked as a threshold
      const s = [0, 1, 2].reduce((a, k) => a + (t.c[k] - LCENTER[k]) * v[k], 0);
      if (s > hiTet) hiTet = s; }
    let live = best.sc.map(s => [s[0], s[1]]);
    writeTet(best.v);
    // STARTING VERTEX. It was live[0][0] -- an arbitrary endpoint of whichever
    // rod happened to be listed first. That is harmless when both base senses
    // are available, but under a chirality the xon can only circulate the tet
    // ONE way, so the start vertex fixes which pivot it can ever reach. Traced:
    // chirality A started on the wrong vertex and sat in a period-6 limit cycle
    // between two tets, its forward and backward flux events cancelling to a
    // net drift of exactly 0.000.
    //   opt.startVertex: 0-3, an index into the tet. undefined = auto.
    let xon = live[0][0];
    if (opt.startVertex !== undefined) xon = best.v[opt.startVertex % 4];

    // THE TET IS THE PARTICLE; THE XON IS THE MECHANISM.
    //
    // The momentum ray belongs to the TET CENTROID, not to the xon. Scoring
    // the xon's own position against it was a category error: the 60-degree
    // rule confines the xon to its own tet, so its position can never track a
    // line at all -- it is always within a unit of the tet whatever it does.
    // Under that metric base hops looked like progress while moving the
    // particle NOT AT ALL, so the walker dithered among them. Measured over
    // ten traversals: 47-65 base hops against 1-15 flux events, and five runs
    // of ten ended exactly sqrt(3) -- one base edge -- from where they began,
    // having visited 5 to 8 distinct nodes in ~75 hops.
    //
    // Scored on the centroid, a base hop has ds = 0 EXACTLY. It stops being
    // progress and becomes what it is: transit, worth taking only to reach a
    // pivot from which a good flux event is available.
    const centOf = ns => [0, 1, 2].map(t => ns.reduce((s, n) => s + NODE[n][t], 0) / 4);
    // THE PARTICLE IS THE CLOSED TET, WHICH IS NOT THE WHOLE ROD SET.
    //
    // An edge-flip carries THREE rods for two ticks: the incoming rod is added
    // BEFORE the outgoing one is severed, and that ordering is the entire
    // trick -- sever first and the set drops to a 3-node hinge with no tet at
    // all (measured, 4560 of 4560). Add first and a tet is closed on every
    // single tick, A handing off directly to B.
    //
    // The tet is never ambiguous in a 3-rod set. With {a1,a2,b1}: a1a2 closes A,
    // a1b1 are the same axis, a2b1 meet at a vertex -- exactly one pair closes.
    // With {a2,b1,b2}: only b1b2. So "which tet is the particle" has one answer
    // throughout, and it is the one this returns.
    const tetOf = rods => {
      for (let i = 0; i < rods.length; i++) for (let j = i + 1; j < rods.length; j++) {
        const n = [...new Set([...rods[i], ...rods[j]])];
        if (n.length === 4 && isTet(n)) return n;
      }
      return null;
    };
    const tetNodes = () => tetOf(live) || [...new Set([].concat(...live))];
    // The ray ALWAYS passes through the lattice centre; only its orientation is
    // random, and freshly so each traversal. The tet starts wherever the ray
    // enters the lattice, which is generally a little off it -- so how fast the
    // particle closes onto the line is itself a measurement, not a given.
    const origin = LCENTER.slice();
    window._LINE({ dir: v, p0: origin });      // the target, now actually drawn

    const alongC = c => [0, 1, 2].reduce((s, k) => s + (c[k] - origin[k]) * v[k], 0);
    const offC = c => { const s = alongC(c);
      return Math.hypot(...[0, 1, 2].map(k => c[k] - origin[k] - s * v[k])); };
    const along = n => alongC(NODE[n]);
    const off = n => offC(NODE[n]);

    // Budget the FULL crossing, lo -> hi. Sizing it from `hi` alone budgeted
    // half the journey: the ray now runs through the lattice centre, so the
    // particle starts at lo (about -3.5) and must reach hi, not 0. All ten
    // traversals used their entire budget and stopped mid-lattice because of
    // it. Measured drift is ~0.085 per tick; 0.05 leaves real headroom.
    // `lo` was already measured against LCENTER for the entry tet, and the ray
    // now runs through LCENTER, so the same reference serves.
    let hi = 0;
    for (const c of NODE) { const s = (c[0] - origin[0]) * v[0]
      + (c[1] - origin[1]) * v[1] + (c[2] - origin[2]) * v[2]; if (s > hi) hi = s; }
    // Sized from MEASURED drift, not a guess. Ten traversals crossed in 37-69
    // ticks over spans of 8.2-8.8, i.e. 0.13-0.23 per tick; 0.10 is a floor
    // well under that. The worst honest case is a crossing plus one stall
    // window, so that is the budget. Over-budgeting is not free: every unused
    // item is still iterated after the traversal ends.
    const ticks = opt.ticks || Math.ceil((hi - lo) / 0.10) + STALL + 10;

    // ---- EDGE DETECTION ---------------------------------------------------
    // A tet is BORN touching the lattice bound. Once it has carried itself off
    // the bound, the next time it touches again it has reached the far side and
    // the run is over. Measured off the lattice rather than against a chosen
    // threshold: a node is on the bound when it is missing base neighbours,
    // i.e. when they would have fallen outside the ball.
    //
    // The test is per VERTEX, not per face. A face needs all three of its
    // vertices on the bound at once, which a tet can avoid while plainly
    // sitting on the edge -- two of the twelve runs came back bornAtBound
    // false for exactly that reason. One vertex touching is touching.
    const _maxDeg = _baseNbr.reduce((m, a2) => Math.max(m, a2.length), 0);
    const onBound = n => _baseNbr[n].length < _maxDeg;
    const tetAtBound = ns => !!ns && ns.length === 4 && ns.some(onBound);
    let leftBound = false, bornAtBound = null;

    // ---- THE TRAVERSAL LINE -----------------------------------------------
    // The line of AVERAGE motion of the tet centre -- the first principal
    // component of the centroid track, not simply last-minus-first, so a
    // wobble in the middle counts against it the way it should. The angle
    // between this and the target line is the number to drive to zero.
    const fitAxis = pts => {
      const n = pts.length; if (n < 2) return null;
      const m = [0, 1, 2].map(k => pts.reduce((sm, q) => sm + q[k], 0) / n);
      const C = [[0,0,0],[0,0,0],[0,0,0]];
      for (const q of pts) { const d = [0,1,2].map(k => q[k] - m[k]);
        for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) C[i][j] += d[i] * d[j]; }
      let u = [1, 1, 1];
      for (let it = 0; it < 400; it++) {
        const w = [0,1,2].map(i => C[i][0]*u[0] + C[i][1]*u[1] + C[i][2]*u[2]);
        const nn = Math.hypot(...w); if (nn < 1e-12) return null;
        u = w.map(z => z / nn);
      }
      return { dir: u, mean: m };
    };
    const track = [];        // tet centroid, one per tick

    const wake = [xon]; const turns = {}, useB = {}, useS = {}, useAxis = {};
    let lastDir = null, events = 0, baseHops = 0, refused = 0, stuck = 0, done = false;
    let pending = null, pendingBucket = null, prev2 = null, lastWasTransit = false;
    let bestS = -Infinity, sinceGain = 0, stopped = 'budget';
    let noAdv = 0;      // consecutive ticks with NO advancing flip reachable

    // MODE LOCK -- the electron. A single flip ALWAYS rotates the flux mode:
    // measured, 0 of 5184 flips across all 1392 tets preserve it, and it is
    // forced (isTet needs two DISTINCT axes, and the replaced rod shares a
    // vertex with the new one, so the new rod has nowhere to go but the third
    // axis). So the mode cannot be held per tick.
    //
    // It is held per STEP. A chain of flips returns to the starting mode --
    // shortest is 2 (XY->XZ->XY), and every tet has one. The chain is a ROUTE
    // planned in advance and drained ONE HOP PER TICK; each flip is still its
    // own flux event with its own solve. One shortcut per tick, always.
    //
    // That is the whole electron/neutrino distinction: the neutrino's chain is
    // length 1 and never returns, so its mode tumbles XY->XZ->YZ forever.
    // opt.lockAxes: the STRICT reading. If the xon is born into mode YZ then
    // the only shortcuts it may ever touch are Y+ Y- Z+ Z-. X is forbidden for
    // life. This is a per-TICK invariant, not a per-step one.
    const modeOf = vs => { const sc = isTet(vs); return sc ? sc.map(s => s[2]).sort().join('') : null; };
    const LOCK = lockAxes ? modeOf(best.v) : null;
    const LOCKAX = LOCK ? LOCK.split('') : null;   // the two permitted axes
    // Use the TRUE 60-degree test (the unit triangle) rather than the stricter
    // tet-membership shorthand. Only the mode-locked walkers need it -- the free
    // neutrino's rod is always inside its own tet, it traverses 24/24 as it is,
    // and widening its move set is a change to a working control, not a fix.
    const FREE60 = lockAxes;
    let chainQ = [];            // remaining hops of the current step

    // ---- THE BIAS ---------------------------------------------------------
    // A gen-locked electron runs a RAIL: holding the mode every tick admits
    // only edge-flips, and those displace +-1 along the dormant axis and
    // nothing else. The bias is the second direction, taken deliberately and
    // paid for.
    //
    // The move exists and it is a mode-RETURNING 3-flip chain. Measured over
    // the whole lattice: every one of the 1392 tets has such a chain whose net
    // centroid displacement is EXACTLY a base <111> direction (|d| = sqrt3);
    // all eight appear, and each tet offers 4 to 8 of them (mean 6.34). Chains
    // of 2 flips give |d| = 1 on the dormant axis -- the rail again, no lateral
    // gain -- so 3 is the shortest thing that steers.
    //
    // THE PRICE, stated plainly: the two intermediate tets of the chain are NOT
    // in the birth generation. A bias move costs exactly 2 ticks off-mode and
    // returns. That is the whole cost of steering, and it cannot be avoided --
    // the tet IS its two rods, so moving it means swapping one, and a
    // mode-preserving swap is the dormant axis by construction.
    //
    // ---- THE TRAVERSAL LOOP, SOLVED AT BIRTH ------------------------------
    // opt.fit: fit the target line. The loop is COMPUTED, not searched for.
    //
    // A gen-locked electron has exactly two move families, and they are
    // orthogonal to each other:
    //   rail   (0,+-1,0)     the DORMANT axis      1 flux event
    //   steer  (+-1,0,+-1)   the MODE PLANE        the 3-tick edge-flip
    // (written for XZ; for XY the dormant axis is Z, for YZ it is X.)
    //
    // Writing the plane steps as p = e0+e1 and q = e0-e1, any integer
    // combination is a*p + b*q = (a+b, a-b) in the plane -- so the reachable
    // (x,z) are exactly those with x = z (mod 2), an index-2 sublattice, and
    // the counts INVERT directly:
    //
    //     nRail = Y        nP = (X+Z)/2        nQ = (X-Z)/2
    //
    // No search: pick the integer target (X,Y,Z) closest in ANGLE to the line,
    // read off three counts, and order them so the path hugs the line. On a
    // vertical line X = Z = 0, both plane counts vanish and the loop is pure
    // rail -- e6 reduces to e5 identically, by arithmetic rather than by a
    // special case.
    const solveLoop = (dir) => {
      if (!LOCK) return null;
      const dorm = LOCK === 'XZ' ? 1 : (LOCK === 'XY' ? 2 : 0);
      const pl = [0, 1, 2].filter(k => k !== dorm);
      const cands = [];
      // Scale scan: the loop should be the SHORTEST one that points where the
      // line points. Bounded because a longer loop cannot beat an exact fit and
      // the sublattice is periodic -- this enumerates candidate loops, it does
      // not search the traversal.
      for (let t = 1; t <= 16; t++) {
        const Y = Math.round(t * dir[dorm]);
        const X = Math.round(t * dir[pl[0]]);
        for (const dz of [0, 1, -1]) {
          const Z = Math.round(t * dir[pl[1]]) + dz;
          if ((((X - Z) % 2) + 2) % 2 !== 0) continue;     // parity is the rule
          const a = (X + Z) / 2, b = (X - Z) / 2;
          const n = Math.abs(Y) + Math.abs(a) + Math.abs(b);
          if (!n) continue;
          const D = [0, 0, 0]; D[dorm] = Y; D[pl[0]] = X; D[pl[1]] = Z;
          const nd = Math.hypot(...D); if (nd < 1e-9) continue;
          const cos = [0, 1, 2].reduce((sm, k) => sm + D[k] * dir[k], 0) / nd;
          cands.push({ cos, n, Y, a, b, D });
        }
      }
      // RANKED, not reduced to one. Which multiset the loop is made of decides
      // whether it can close at all, and the closest-in-angle candidate is not
      // always closable -- so the caller walks this list and takes the first
      // that does. Parity is the one obstruction that is exact and needs no
      // test: every move flips the tet's orientation class, so a cycle must
      // have an EVEN number of moves. Measured, 0 of 37 odd multisets close and
      // 32 of 38 even ones do. The remaining even failures have no simple
      // closed form -- |Z| <= |Y| holds when the two steer counts share a sign
      // and breaks in 92 of 340 cases when they do not -- so closability is
      // TESTED rather than predicted.
      cands.sort((x, y) => y.cos - x.cos || x.n - y.n);
      const best = cands[0];
      if (!best) return null;
      const order = (b2) => {
      // ORDER THE LOOP. Same idea as Bresenham: hold the running position and
      // take, at each place in the sequence, whichever remaining move leaves
      // the path closest to the ideal ray. Deterministic, done once, and the
      // loop's total is exactly D however it is ordered -- the ordering only
      // decides how far the path strays in between.
      const mk = (k0, k1, s0, s1) => { const w = [0, 0, 0];
        if (k0 !== null) w[k0] = s0; if (k1 !== null) w[k1] = s1; return w; };
      const rem = [
        { kind: 'rail',  v: mk(dorm, null, Math.sign(b2.Y) || 1, 0), n: Math.abs(b2.Y) },
        { kind: 'steer', v: mk(pl[0], pl[1], Math.sign(b2.a) || 1, Math.sign(b2.a) || 1),
          n: Math.abs(b2.a) },
        { kind: 'steer', v: mk(pl[0], pl[1], Math.sign(b2.b) || 1, -(Math.sign(b2.b) || 1)),
          n: Math.abs(b2.b) }];
      const nd = Math.hypot(...b2.D), u = b2.D.map(z => z / nd);
      const perp = c => { const sp = c[0] * u[0] + c[1] * u[1] + c[2] * u[2];
        return Math.hypot(c[0] - sp * u[0], c[1] - sp * u[1], c[2] - sp * u[2]); };
      const steps = []; let cur = [0, 0, 0];
      const total = rem.reduce((sm, r) => sm + r.n, 0);
      for (let i2 = 0; i2 < total; i2++) {
        let pick = null, pd = Infinity;
        for (const r of rem) { if (!r.n) continue;
          const c = [0, 1, 2].map(k => cur[k] + r.v[k]), d2 = perp(c);
          if (d2 < pd - 1e-12) { pd = d2; pick = r; } }
        if (!pick) break;
        pick.n--; cur = [0, 1, 2].map(k => cur[k] + pick.v[k]);
        steps.push({ kind: pick.kind, d: pick.v.slice() });
      }
        return { steps, disp: b2.D, cos: +b2.cos.toFixed(6),
                 nRail: b2.Y, nP: b2.a, nQ: b2.b };
      };
      // The ranked candidates, each already ordered, best angle first.
      // ALL candidates, best angle first, parity included. An odd multiset can
      // never close on its own -- every move flips the orientation class -- but
      // TWICE round it is even and points in exactly the same direction, so the
      // best-fitting loop is salvageable rather than something to trade away.
      return { best: order(best), all: cands.slice(0, 24).map(order) };
    };
    // ---- CLOSING THE LOOP -------------------------------------------------
    // solveLoop says WHICH moves the loop is made of. This says in what ORDER,
    // and it is the step that makes the pattern actually repeat.
    //
    // The xon inside its tet is a state: (class, vertex it is on, vertex it
    // came from). Each move carries that state into the next tet's frame --
    // hopTable records the exit -- and every move flips the class, so a closed
    // loop needs an EVEN number of moves. Repositioning walks the state inside
    // one tet without spending a displacement.
    //
    // So closing the loop is finding a CYCLE in a 32-state machine that spends
    // exactly the multiset of displacements solveLoop asked for and returns to
    // the state it began in. Then the tet has moved by one lattice translation
    // with the xon congruent to where it started, and the whole hop pattern
    // repeats verbatim -- forever, with nothing recomputed.
    //
    // Bounded and exact: 32 states times the product of the move counts. This
    // enumerates orderings of a known multiset, it does not search the lattice.
    const closeLoop = (steps, ch, dir) => {
      if (!steps || !steps.length) return null;
      // The cycle must not merely CLOSE, it must stay near the line the whole
      // way round. Any closing order spends the same displacements and ends in
      // the same place, but a run that meets the lattice edge part-way through
      // only ever executes a PREFIX -- and a prefix of a badly ordered cycle is
      // not parallel to anything. Measured: an ordering that front-loaded all
      // eleven rails left the traversal 21 degrees off, its four steers never
      // reached. So every cycle is scored by the worst perpendicular excursion
      // of its prefixes, and the best is taken.
      const un = Math.hypot(...dir) || 1, u = dir.map(z => z / un);
      const perpOf = q => { const sp = q[0]*u[0] + q[1]*u[1] + q[2]*u[2];
        return Math.hypot(q[0]-sp*u[0], q[1]-sp*u[1], q[2]-sp*u[2]); };
      const T = hopTable();
      const need0 = {};
      for (const st of steps) { const k = st.d.join(',');
        need0[k] = (need0[k] || 0) + 1; }
      const dk = Object.keys(need0);
      const cnt0 = dk.map(k => need0[k]);
      const flip = c => c === 'U' ? 'D' : 'U';
      const sk = (cls, at, prev) => cls + '|' + at + '<' + prev;
      // in-tet states reachable without spending a move, including staying put
      const reach = {};
      const reachOf = (cls, at, prev) => {
        const k = sk(cls, at, prev);
        if (reach[k]) return reach[k];
        const out = [{ at, prev, hops: 0 }];
        for (const p of hopReposition(cls, ch, at, prev))
          out.push({ at: p.at, prev: p.from, hops: p.hops.length });
        return (reach[k] = out);
      };
      const entryOf = (cls, d) => (T[cls + '|' + d + '|' + ch] || { res: [] }).res;
      // BFS over (state, remaining counts). Shortest in moves, then in hops.
      for (const s0 of hopStates('U').map(x => ({ cls: 'U', ...x }))
                     .concat(hopStates('D').map(x => ({ cls: 'D', ...x })))) {
        const start = sk(s0.cls, s0.at, s0.prev);
        const seen = new Map([[start + '#' + cnt0.join('.'), 0]]);
        let front = [{ cls: s0.cls, at: s0.at, prev: s0.prev,
                       cnt: cnt0.slice(), seq: [], pos: [0,0,0], dev: 0 }];
        let found = null;
        for (let depth = 0; depth < steps.length && front.length; depth++) {
          const nxt = [];
          for (const st of front) {
            for (let i = 0; i < dk.length; i++) {
              if (!st.cnt[i]) continue;
              const d = dk[i], ents = entryOf(st.cls, d);
              if (!ents.length) continue;
              for (const r of reachOf(st.cls, st.at, st.prev)) {
                const hit = ents.find(e => e.start === r.at && e.prev === r.prev);
                if (!hit) continue;
                const cnt = st.cnt.slice(); cnt[i]--;
                const ncls = flip(st.cls);
                const dv = d.split(',').map(Number);
                const pos = [0,1,2].map(k => st.pos[k] + dv[k]);
                const dev = Math.max(st.dev, perpOf(pos));
                const seq = st.seq.concat([{ d: dv,
                  entryAt: r.at, entryPrev: r.prev, repHops: r.hops }]);
                if (cnt.every(z => z === 0)) {
                  if (sk(ncls, hit.exitAt, hit.exitFrom) === start
                      && (!found || dev < found.dev - 1e-9))
                    found = { start: s0, seq, dev: +dev.toFixed(4) };
                  continue;
                }
                const key = sk(ncls, hit.exitAt, hit.exitFrom) + '#' + cnt.join('.');
                // Keep the LOWEST-excursion route to each (state, counts); a
                // plain visited-set would lock in whichever arrived first.
                const prev2 = seen.get(key);
                if (prev2 !== undefined && prev2 <= dev + 1e-9) continue;
                seen.set(key, dev);
                nxt.push({ cls: ncls, at: hit.exitAt, prev: hit.exitFrom, cnt, seq,
                           pos, dev });
              }
            }
          }
          front = nxt;
        }
        if (found) return found;
      }
      return null;
    };
    let LOOPFIT = null, loopIdx = 0, LOOPSEQ = null;
    const fitStats = { steps: 0, railDone: 0, steerDone: 0,
                       railMissing: 0, steerMissing: 0, routeFail: 0 };
    if (opt.fit) {
      LOOPFIT = solveLoop(v);
      // A closed loop repeats verbatim; an unclosed one is re-derived every
      // iteration and drifts. If the exact multiset will not close, try it
      // DOUBLED -- twice round the same displacements points the same way and
      // gives the ordering room to return to its own start state.
      // Walk the ranked candidates and take the FIRST that closes. Angle order
      // means the closest-fitting closable loop wins; testing beats predicting,
      // since the closure condition beyond parity has no simple closed form.
      if (LOOPFIT) {
        const tried = [];
        outer:
        for (const cand of LOOPFIT.all) {
          for (const mult of [1, 2]) {
            // 1x is hopeless for an odd multiset: parity forbids it outright.
            if (mult === 1 && cand.steps.length % 2) continue;
            const steps = mult === 1 ? cand.steps : cand.steps.concat(cand.steps);
            const seq = closeLoop(steps, chirality || 'A', v);
            tried.push({ disp: cand.disp.join(','), n: steps.length,
                         mult, cos: cand.cos, closed: !!seq });
            if (seq) {
              LOOPFIT = mult === 1 ? cand
                : { steps, disp: cand.disp.map(z => z * 2), cos: cand.cos,
                    nRail: cand.nRail * 2, nP: cand.nP * 2, nQ: cand.nQ * 2 };
              LOOPSEQ = seq; break outer;
            }
          }
        }
        if (!LOOPSEQ) LOOPFIT = LOOPFIT.best || null;
        fitStats.candidatesTried = tried.length;
        fitStats.candidates = tried;
      }
    }

    let biasActive = false;     // draining a table move right now
    const biasStats = { planned: 0, done: 0, noChain: 0, expandFail: 0,
                        offModeTicks: 0, brokeTet: 0 };
    // The xon must START as though it had just arrived along a tet edge. Left
    // as null the first hop is unconstrained, the xon can step straight off its
    // own tet, and then NOTHING satisfies the 60-degree rule -- a completed run
    // showed hops:1. live[0][0] and live[1][0] are ends of opposite rods, so
    // they are a cross-pair of the tet and therefore base-adjacent.
    let prevNode = live[1][0];
    const trace = [];
    const dvec = (a, b) => [0, 1, 2].map(k => NODE[b][k] - NODE[a][k]);
    const dkey = (a, b) => dvec(a, b).join(',');
    const rodSame = (x, y) => (x[0] === y[0] && x[1] === y[1]) || (x[0] === y[1] && x[1] === y[0]);
    // Which rod a new one replaces.
    //   neutrino (face-flip): the rod SHARING A VERTEX with the new one. The
    //     new rod is then forced onto the third axis, so the mode rotates.
    //   electron (edge-flip): the rod on the SAME AXIS. The two are parallel
    //     and disjoint, so the mode is preserved by construction. Measured:
    //     500 such swaps in 300 tets, and in every one the new rod lies wholly
    //     outside the tet with both ends base-adjacent to the kept rod.
    //   BIAS CHAIN: the chain's flips are FACE-flips by construction (that is
    //     what makes them able to leave the dormant axis at all), so while one
    //     is draining the killer is the share-a-vertex rod, not the same-axis
    //     one. Getting this wrong silently dissolves the tet mid-chain.
    const killerOf = rod => (lockAxes && !biasActive)
      ? live.find(r => axisOf(dvec(r[0], r[1])) === axisOf(dvec(rod[0], rod[1])))
      : live.find(r => r[0] === rod[0] || r[0] === rod[1]
                    || r[1] === rod[0] || r[1] === rod[1]);
    // the four nodes of the tet this rod would leave standing, or null
    const tetNodesAfter = rod => {
      const after = live.filter(r => r !== killerOf(rod)).concat([rod]);
      if (after.length !== 2) return null;
      const n = [...new Set([].concat(...after))];
      return (n.length === 4 && isTet(n)) ? n : null;
    };
    // UNIT APART, COMBINATORIALLY. The 60-degree rule is "the node before and
    // the node after close a unit triangle with the pivot", so it needs to know
    // which pairs are unit length. Measured off P this is WRONG on exactly the
    // ticks that matter: writeTet has already changed the shortcut set but P
    // still holds the PREVIOUS solve, so every candidate is rejected and the run
    // dies at hop 1 (the same trap the tet-membership test was written to dodge).
    // The lattice settles it exactly instead -- a pair is unit iff it is a base
    // edge or a LIVE rod; an unbuilt shortcut candidate sits at 2/sqrt3.
    // `rods` is explicit because a chain's later flips act on rod sets that do
    // not exist yet.
    const unitIn = (rods, a, b) => _baseSet.has(K(a, b))
      || rods.some(r => (r[0] === a && r[1] === b) || (r[0] === b && r[1] === a));
    const unitApart = (a, b) => unitIn(live, a, b);

    // AUTO START. The start was live[0][0] with arrival live[1][0] -- arbitrary
    // endpoints of whichever rods were listed first. Harmless while both base
    // senses are open; DECISIVE under a chirality, where the xon can circulate
    // the tet only one way and the start fixes which pivot it can ever reach.
    //
    // Measured on a locked line: chirality A crossed 99% from one vertex and
    // sat in a period-6 limit cycle at 7% from the other three, its forward and
    // backward flux events cancelling to drift exactly 0.000. Even the
    // unrestricted electron went 96%/stalled -> 100%/traversed on a better
    // start, at HALF the flux events (24 vs 43) and offMean 0.74 vs 1.26.
    //
    // Choose the (vertex, arrival) pair whose best available flux event sits
    // CLOSEST TO THE LINE -- offset first, advance only as a tie-break. That is
    // the scoring priority, and the geometry agrees: the start tet {25,30,55,54}
    // offers three flips, and ranking them by advance picks pivot 30 (ds 0.544,
    // off 0.68) while ranking by offset picks pivot 55 (ds 0.005, off 0.43).
    // Measured, 55 is the start that TRAVERSES -- 100%, offMean 0.74, drift
    // 0.265 on 24 flux events, against 96%/1.15/0.132 on 43 for the greedy
    // pick. The best opening move is not on the best trajectory.
    if (opt.startVertex !== undefined) {
      xon = best.v[opt.startVertex % 4];
      prevNode = best.v.find(n => n !== xon && _baseSet.has(K(n, xon)));
    } else if (opt.autoStart !== false) {
      let bx = xon, bp = prevNode, bOff = Infinity, bAdv = -Infinity;
      for (const cv of best.v) for (const pv of best.v) {
        if (pv === cv || !_baseSet.has(K(pv, cv))) continue;
        for (let a = 0; a < AXN.length; a++) {
          const j = SCOPT.get(cv + ':' + a); if (j === undefined) continue;
          const rod = [cv, j];
          if (live.some(r => rodSame(r, rod))) continue;
          const af = tetNodesAfter(rod);
          if (!af || !af.includes(pv)) continue;        // 60 deg once it forms
          const c0 = centOf(af), o = offC(c0), s = alongC(c0);
          if (o < bOff - 1e-9 || (Math.abs(o - bOff) < 1e-9 && s > bAdv)) {
            bOff = o; bAdv = s; bx = cv; bp = pv;
          }
        }
      }
      xon = bx; prevNode = bp;
    }
    wake.length = 0; wake.push(xon);

      // ENTER THE START TET IN THE STATE THE LOOP BEGINS IN. Otherwise the
      // very first step asks for a state the run never occupied and the whole
      // traversal dies at tick 0 with an empty trace -- which is exactly what
      // happened. The cycle alternates class every move, so rotating it by one
      // flips which class it starts in; that is all the alignment needed.
      if (LOOPSEQ) {
        // dvec/axisOf are declared further down, so the axis test is inlined.
        const axOf0 = r => { const d = [0,1,2].map(k => NODE[r[1]][k] - NODE[r[0]][k]);
          return d[0] ? 'X' : (d[1] ? 'Y' : 'Z'); };
        const X = live.find(r => axOf0(r) === 'X'), Z = live.find(r => axOf0(r) === 'Z');
        if (X && Z) {
          const lexLess0 = (a2, b2) => { for (let k = 0; k < 3; k++)
            if (NODE[a2][k] !== NODE[b2][k]) return NODE[a2][k] < NODE[b2][k];
            return false; };
          const o0 = lexLess0(X[0], X[1]) ? X[0] : X[1];
          const rel0 = n => [0, 1, 2].map(k => NODE[n][k] - NODE[o0][k]);
          const cls0 = rel0(Z[0])[1] > 0 ? 'U' : 'D';
          if (LOOPSEQ.start.cls !== cls0 && LOOPSEQ.seq.length > 1)
            LOOPSEQ = { start: { cls: cls0 }, dev: LOOPSEQ.dev,
                        seq: LOOPSEQ.seq.slice(1).concat([LOOPSEQ.seq[0]]) };
          const want = LOOPSEQ.seq[0];
          const at = KEY.get([0,1,2].map(k => NODE[o0][k]
                       + (+want.entryAt.split(',')[k])).join(','));
          const pv = KEY.get([0,1,2].map(k => NODE[o0][k]
                       + (+want.entryPrev.split(',')[k])).join(','));
          if (at !== undefined && pv !== undefined) { xon = at; prevNode = pv;
            wake.length = 0; wake.push(xon); }
        }
      }

    await runExperiment(
      `momentum xon — no line, v=${v.map(x => x.toFixed(3)).join(',')}`,
      Array.from({ length: ticks }, (_, k) => ({ k, label: 'tick ' + k })),
      () => {
        if (done) { pending = null; return; }
        if (opt.alive && !opt.alive()) { done = true; pending = null; return; }
        // THE 60-DEGREE RULE. Every turn the xon makes is 60 degrees measured
        // in the geometry that exists AFTER the move. Nothing else is needed:
        // no path search, no scoring of turn angles, no lookahead. The rule is
        // self-defining, and it is the whole choreography.
        //
        //   TRAVERSAL of geometry that already exists -- test it directly:
        //   two unit hops subtend 60 degrees exactly when the node before and
        //   the node after are themselves unit apart, i.e. the three of them
        //   close a unit triangle.
        //
        //   CREATION of a new rod -- the angle cannot be tested before the
        //   event, because the pair is still at 2/sqrt3 and the turn measures
        //   ~56 degrees. But a unit tet has six unit edges, so EVERY angle in
        //   it is 60 degrees. So the turn is 60 degrees after the flux event
        //   precisely when the node the xon came from is one of the four nodes
        //   of the tet that results. The flux event is what makes the turn
        //   legal. (Zac's point, and it is what closes the rule.)
        //
        // Consequence, verified over all 1272 flips in the lattice: the pivot
        // is ALWAYS exactly one base hop from an endpoint of the rod being
        // kept, 1272 of 1272. So the constraint never strands the xon, and the
        // electron's step is the t0/t1 pair it was specified to be -- base hop
        // (no geometric change), then shortcut hop (flux event).
        // Tested COMBINATORIALLY, not against P. A tet has six unit edges, so
        // every angle between two of its edges is 60 degrees -- membership of
        // the tet is the whole test, and it is exact. Testing solved
        // coordinates instead looks more direct and is wrong: on the first
        // tick writeTet() has changed the shortcut set but P still holds the
        // PREVIOUS configuration, so every candidate was rejected and the run
        // died at hop 1.
        // ONE TRANSIT PER FLUX EVENT. The pivot is always exactly one base hop
        // from an endpoint of the kept rod -- 1272 of 1272 flips -- so the xon
        // NEVER needs two transits running. Allowing them let it take a hop
        // whose lookahead promised a good flux event, then hop back, and never
        // commit: stalled runs showed 194 transits against 13 flux events and
        // sat at the same s after 204 ticks as after 54. Forbidding the second
        // transit is not a tuning knob, it is the t0/t1 rhythm the geometry
        // already dictates.
        const here = alongC(centOf(tetNodes()));   // where the PARTICLE is
        const cand = [];
        const now = tetNodes();                    // the four tet nodes
        // THE 60-DEGREE TEST, not tet membership. prev-xon-j is a unit triangle
        // exactly when prev and j are unit apart, and that is the whole rule.
        // Requiring j to be a vertex of the CURRENT tet is stricter than the
        // physics and it is what made the electron immobile: the rod it must
        // build lies wholly outside its tet, reachable only by a 60-degree hop
        // off an endpoint of the rod it keeps.
        for (const j of _baseNbr[xon]) {
          if (j === prevNode || prevNode === null) continue;
          // TET MEMBERSHIP IS NOT THE RULE, it is a sufficient special case of
          // it -- a tet has six unit edges, so any two of its vertices are unit
          // apart. The rule is the unit triangle, and under a mode lock the
          // difference is decisive: the rod the electron must build lies wholly
          // OUTSIDE its current tet, so a xon confined to its own four vertices
          // can never reach an endpoint of it. That confinement is what made
          // every mode-locked run end 'unroutable' with advancing chains still
          // on the table -- the geometry was there, the walk could not get to it.
          const ok = FREE60 ? unitApart(prevNode, j)
                            : (now.includes(j) && now.includes(prevNode));
          if (!ok) continue;
          if (!chiralOK(xon, j)) { chiralBlocked++; continue; }   // base senses only
          cand.push({ to: j, kind: 'base' });
        }
        for (let a = 0; a < AXN.length; a++) {
          const j = SCOPT.get(xon + ':' + a); if (j === undefined) continue;
          const rod = [xon, j];
          // THE AXIS BAN IS ABSOLUTE. No bias exemption, no chain exemption --
          // a banned shortcut does not exist, so it is refused before any other
          // rule gets a say.
          if (BAN && BAN.indexOf(axisOf(dvec(xon, j))) >= 0) { banBlocked++; continue; }
          // STRICT MODE LOCK: a shortcut off the birth mode's two axes is not
          // this particle's to use -- EXCEPT inside a bias chain, which is the
          // one sanctioned excursion and which returns to the birth mode by
          // construction. The exemption is deliberate and it is the price
          // measured in biasStats.offModeTicks, not a hole in the lock: outside
          // a chain the filter is absolute, and the chain's endpoint is always
          // back in LOCK.
          if (lockAxes && LOCKAX && !biasActive
              && LOCKAX.indexOf(axisOf(dvec(xon, j))) < 0) {
            axesBlocked++; continue;
          }
          if (live.some(r => rodSame(r, rod))) {          // traversing an existing rod
            const okT = FREE60 ? unitApart(prevNode, j) : now.includes(prevNode);
            if (okT && j !== prevNode) cand.push({ to: j, kind: 'sc', rod });
            continue;
          }
          const after = tetNodesAfter(rod);               // creating one
          if (!after) continue;                           // would dissolve the tet
          if (!after.includes(prevNode)) continue;        // 60 deg once the tet forms
          cand.push({ to: j, kind: 'sc', rod });
        }
        // A CREATION moves the particle, so it is scored directly: how far the
        // resulting tet centroid advances along the ray, less how far it sits
        // off it. A TRANSIT moves the particle not at all (ds = 0 exactly), so
        // it is scored by the best flux event it brings within reach -- one
        // step of lookahead, which is all the geometry ever needs, because the
        // pivot is always exactly one base hop from an endpoint of the kept rod
        // (1272 of 1272 flips).
        //
        // Balance rides on top, asymmetrically: shortcut senses (6, signed) and
        // base AXES (4, sense-agnostic), so the +/- carrying the momentum stays
        // completely free.
        // MOMENTUM FIRST, deviation second. A conserved velocity means the
        // particle does not go backwards along its own ray, so advancing moves
        // outrank retreating ones outright, and among advancing moves the one
        // that keeps closest to the line wins. Weighing the two against each
        // other instead (perp - mu*ds) hugged the line without pushing along
        // it: 10 traversals crossed 18-95% of the lattice, mean 60, with runs
        // stalling. Monotone crossed 79-93%, mean 87, none stalling.
        const creationScore = (rod, cameFrom) => {
          const af = tetNodesAfter(rod);
          if (!af || !af.includes(cameFrom)) return null;
          const c = centOf(af);
          return (alongC(c) - here) > 1e-9 ? offC(c) : 100 + offC(c);
        };
        const bestReach = (at, cameFrom) => {
          let best = null;
          for (let a = 0; a < AXN.length; a++) {
            const j = SCOPT.get(at + ':' + a); if (j === undefined) continue;
            const rod = [at, j];
            if (live.some(r => rodSame(r, rod))) continue;
            const s = creationScore(rod, cameFrom);
            if (s !== null && (best === null || s < best)) best = s;
          }
          return best;
        };
        // ---- PLAN TO THE BEST FLIP --------------------------------------
        // A transit moves the particle NOT AT ALL -- it only changes which
        // vertex the xon sits on. So the geometry is FIXED across any sequence
        // of transits, and the set of flips reachable from the current tet can
        // be enumerated exactly: four vertices, at most three hops. This is a
        // complete search of the local options, not a rollout.
        //
        // One step of lookahead was the bug. Under a chirality the base moves
        // form a DIRECTED cycle, so the pivot the xon needs is often two or
        // three hops away and invisible to a one-step probe -- the xon then
        // took whatever flip was adjacent, which alternately advanced and
        // retreated, and it sat in a period-6 limit cycle with drift 0.000.
        //
        // Ranked offset-first, advance second, fewest hops third.
        const planBest = () => {
          const nowT = tetNodes();
          const outs = (at, from) => {
            const o = [];
            for (const j of _baseNbr[at]) {
              if (j === from) continue;
              // Same 60-degree test the candidate list uses. Under a mode lock
              // the xon MUST be able to leave its tet -- the rod it needs to
              // build lies outside it -- so tet membership is the wrong gate.
              const ok = lockAxes ? unitApart(from, j)
                                  : (nowT.includes(j) && nowT.includes(from));
              if (!ok || !chiralOK(at, j)) continue;
              o.push({ to: j, kind: 'base' });
            }
            for (let a = 0; a < AXN.length; a++) {
              const j = SCOPT.get(at + ':' + a); if (j === undefined) continue;
              if (!live.some(r => rodSame(r, [at, j]))) continue;
              if (nowT.includes(from) && j !== from) o.push({ to: j, kind: 'sc', rod: [at, j] });
            }
            return o;
          };
          let bp = null;
          const consider = (st) => {
            for (let a = 0; a < AXN.length; a++) {
              const j = SCOPT.get(st.at + ':' + a); if (j === undefined) continue;
              const rod = [st.at, j];
              if (live.some(r => rodSame(r, rod))) continue;
              if (BAN && BAN.indexOf(axisOf(dvec(st.at, j))) >= 0) continue;
              if (lockAxes && LOCKAX
                  && LOCKAX.indexOf(axisOf(dvec(st.at, j))) < 0) continue;
              const af = tetNodesAfter(rod);
              if (!af || !af.includes(st.from)) continue;   // 60 deg once it forms
              const c = centOf(af), adv = alongC(c) - here;
              if (adv <= 1e-9) continue;                    // never plan a retreat
              const off = offC(c);
              if (!bp || off < bp.off - 1e-9
                  || (Math.abs(off - bp.off) < 1e-9 && adv > bp.adv + 1e-9)
                  || (Math.abs(off - bp.off) < 1e-9 && Math.abs(adv - bp.adv) < 1e-9
                      && st.hops < bp.hops))
                bp = { off, adv, hops: st.hops,
                       first: st.first || { to: j, kind: 'sc', rod } };
            }
          };
          const seen = new Set([xon + '|' + prevNode]);
          let front = [{ at: xon, from: prevNode, first: null, hops: 0 }];
          consider(front[0]);
          for (let d = 0; d < (lockAxes?4:3) && front.length; d++) {
            const nxt = [];
            for (const st of front) for (const o of outs(st.at, st.from)) {
              const k = o.to + '|' + st.at;
              if (seen.has(k)) continue; seen.add(k);
              const ns = { at: o.to, from: st.at, first: st.first || o, hops: st.hops + 1 };
              consider(ns); nxt.push(ns);
            }
            front = nxt;
          }
          return bp;
        };
        // ---- THE ANALYTIC MOVER -----------------------------------------
        // tetWalk is GONE. Every hop now comes from hopTable(), which is fixed
        // integer geometry over the two canonical orientations. Nothing here
        // searches: the tet is canonicalised, the move is looked up, the hop
        // vectors are translated back to node ids, and that is the move.
        const nodeAt = v => KEY.get(v[0] + ',' + v[1] + ',' + v[2]);
        const lexLess = (a2, b2) => { for (let k = 0; k < 3; k++)
          if (NODE[a2][k] !== NODE[b2][k]) return NODE[a2][k] < NODE[b2][k];
          return false; };
        // Canonical frame: origin = the lex-smaller endpoint of the X rod;
        // class = whether the Z rod sits above or below it.
        const frameOf = () => {
          if (live.length !== 2) return null;
          const ax = r => axisOf(dvec(r[0], r[1]));
          const X = live.find(r => ax(r) === 'X'), Z = live.find(r => ax(r) === 'Z');
          if (!X || !Z) return null;
          const o = lexLess(X[0], X[1]) ? X[0] : X[1];
          const rel = n => [0, 1, 2].map(k => NODE[n][k] - NODE[o][k]);
          return { o, rel, cls: rel(Z[0])[1] > 0 ? 'U' : 'D' };
        };
        // Look the move up and translate it. Returns the hop list, or null if
        // this tet's frame does not offer it (the lattice has run out that way).
        const analyticMove = (disp) => {
          const F = frameOf(); if (!F) return null;
          const key = F.cls + '|' + disp.join(',') + '|' + (chirality || 'A');
          const ent = hopTable()[key]; if (!ent) return null;
          const xr = F.rel(xon).join(','), pr = F.rel(prevNode).join(',');
          const hit = ent.res.find(r => r.start === xr && r.prev === pr);
          if (!hit) return { reposition: true, cls: F.cls, at: xr, prev: pr, F };
          const abs = v => nodeAt([0, 1, 2].map(k => NODE[F.o][k] + v[k]));
          const outHops = [];
          for (const h of hit.hops) {
            const to = abs(h.to); if (to === undefined) return null;   // off lattice
            const o2 = { to, kind: h.kind };
            if (h.add) { const p = abs(h.add[0]), q = abs(h.add[1]);
              if (p === undefined || q === undefined) return null;
              o2.add = [p, q]; o2.rod = [p, q]; }
            if (h.sev) { const p = abs(h.sev[0]), q = abs(h.sev[1]);
              if (p === undefined || q === undefined) return null;
              o2.sever = [p, q]; }
            outHops.push(o2);
          }
          return { hops: outHops, kind: ent.kind };
        };
        // Walk to a NAMED state. Used by the closed loop, which knows exactly
        // which state its next move enters from.
        const analyticRepositionTo = (wantAt, wantPrev) => {
          const F = frameOf(); if (!F) return null;
          const ch = chirality || 'A';
          const xr = F.rel(xon).join(','), pr = F.rel(prevNode).join(',');
          for (const path of hopReposition(F.cls, ch, xr, pr)) {
            if (path.at !== wantAt || path.from !== wantPrev) continue;
            const hops = []; let bad = false;
            for (const h of path.hops) {
              const to = nodeAt([0, 1, 2].map(k => NODE[F.o][k] + h.to[k]));
              if (to === undefined) { bad = true; break; }
              hops.push({ to, kind: h.kind });
            }
            if (!bad && hops.length) return hops;
          }
          return null;
        };
        // ---- THE COMPUTED LOOP (e6) -------------------------------------
        // The loop was solved at birth and is simply REPEATED, and every hop of
        // it comes out of hopTable(). NOTHING here searches: the tet is
        // canonicalised into one of two orientations, the move is looked up by
        // (class, displacement, chirality, entry state), and the stored hop
        // vectors are translated back into node ids. Two hops for a rail, four
        // for a steer, fixed.
        //
        // A move whose entry state does not match takes ONE tabulated
        // repositioning hop first -- also a lookup, since chirality and the
        // 60-degree rule leave at most one legal choice inside the tet.
        // THE CLOSED LOOP, REPLAYED. LOOPSEQ is a cycle in the state machine:
        // it spends exactly the displacements solveLoop asked for and returns
        // the (class, xon, prev) state to where it began, so one iteration
        // leaves the tet translated and the xon congruent. Replaying it is the
        // whole traversal -- nothing is recomputed, and each entry already
        // names the state to enter the move from.
        if (lockAxes && LOOPSEQ && LOOPSEQ.seq.length && !chainQ.length
            && live.length === 2) {
          const st = LOOPSEQ.seq[loopIdx % LOOPSEQ.seq.length];
          const F = frameOf();
          if (!F) { done = true; stopped = 'noframe'; pending = null; return; }
          const xr = F.rel(xon).join(','), pr = F.rel(prevNode).join(',');
          if (xr !== st.entryAt || pr !== st.entryPrev) {
            // Not yet in the state this step enters from: walk there. The path
            // is a lookup in the 16-state closure, and the loop guarantees such
            // a path exists -- it was checked when the cycle was built.
            const rep = analyticRepositionTo(st.entryAt, st.entryPrev);
            if (rep) { chainQ = rep; fitStats.repositions =
              (fitStats.repositions || 0) + 1; }
            else { done = true; stopped = 'lostState'; pending = null; return; }
          } else {
            const got = analyticMove(st.d);
            if (got && got.hops) {
              chainQ = got.hops;
              biasActive = Math.abs(st.d[0]) + Math.abs(st.d[2]) > 0;
              fitStats.steps++;
              if (st.d[0] === 0 && st.d[2] === 0) fitStats.railDone++;
              else fitStats.steerDone++;
              loopIdx++;
            } else {
              // The move is legal in the abstract but its tet is off the edge
              // of this lattice. That is the end of the traversal, not a fault.
              done = true; stopped = 'traversed'; pending = null; return;
            }
          }
        } else if (lockAxes && LOOPFIT && !LOOPSEQ && !chainQ.length) {
          // NO CLOSING CYCLE. There is no fallback and there should not be:
          // a scheduler that re-derives its way forward hides exactly the
          // question worth answering, which is why this multiset admits no
          // cycle. Fail loudly and leave the loop on the result for inspection.
          done = true; stopped = 'loopNotClosed'; pending = null; return;
        }
        // ---- DRAIN ------------------------------------------------------
        // One hop per tick. Adds land BEFORE severs so a tet is closed on every
        // tick of an edge-flip, and both are explicit on the hop rather than
        // inferred from a killerOf rule -- inferring it is what produced a
        // density violation once already.
        if (lockAxes && chainQ.length) {
          const want = chainQ.shift();
          const lastHop = !chainQ.length;
          pending = { to: want.to, kind: want.kind, rod: want.rod, bias: true,
                      from: xon, nCand: cand.length,
                      nBase: cand.filter(c => c.kind === 'base').length,
                      nSC: cand.filter(c => c.kind === 'sc').length };
          lastDir = dvec(xon, want.to);
          const dkb = dkey(xon, want.to);
          prev2 = prevNode; prevNode = xon;
          xon = want.to; wake.push(xon); if (wake.length > 40) wake.shift();
          drawXon(xon, wake);
          if (want.add && !live.some(r => rodSame(r, want.add))) {
            const nk = scKeyOf(want.add[0], want.add[1]);
            if (nk) active.set(nk[0], nk[1]);
            live = live.concat([want.add.slice()]);
            events++;
          }
          if (want.sever) {
            const k = scKeyOf(want.sever[0], want.sever[1]);
            if (k) active.delete(k[0]);
            live = live.filter(r => !rodSame(r, want.sever));
          }
          pending.killed = want.sever || null;
          if (want.kind === 'base') { baseHops++; useB[dkb] = (useB[dkb] || 0) + 1; }
          else useS[dkb] = (useS[dkb] || 0) + 1;
          // STRUCTURAL GUARDS. A tet closed after every tick, and never a rod
          // on a banned axis. Loud on purpose -- a hit is a coding fault.
          if (!tetOf(live)) { biasStats.brokeTet++;
            console.error('ANALYTIC MOVE LEFT NO CLOSED TET', { rods: live, step: want }); }
          if (BAN && live.some(r => BAN.indexOf(axisOf(dvec(r[0], r[1]))) >= 0)) {
            biasStats.bannedRod = (biasStats.bannedRod || 0) + 1;
            console.error('ANALYTIC MOVE INSTALLED A BANNED ROD', { rods: live, step: want }); }
          if (lastHop) biasActive = false;
          return;
        }
        if (opt.plan !== false) {
          const bp = planBest();
          // END OF THE RAY. planBest is an EXACT enumeration of every flip
          // reachable from this tet, so a null result means no advancing flux
          // event exists at all -- the particle has gone as far along the line
          // as the lattice allows. Left running it takes a retreat, comes back,
          // and sits in a period-6 limit cycle burning the budget until the
          // stall detector fires: that is what "collides with the edge until
          // the run dies" was. Traced at s=5.59 against hiTet 5.82, oscillating
          // 5.59 <-> 5.37 for the final 14 ticks.
          // End IMMEDIATELY, do not fall through. Falling through let the old
          // scoring take a retreat, and from the retreated tet an advancing
          // flip exists again -- so the streak counter reset every other tick
          // and never fired. The retreat was resetting its own detector.
          if (!bp || !bp.first) { noAdv++; done = true; stopped = 'traversed';
            pending = null; return; }
          noAdv = 0;
          if (bp && bp.first) {
            const pick = cand.find(c => c.to === bp.first.to && c.kind === bp.first.kind);
            if (pick) {
              pending = pick; pending.nCand = cand.length;
              pending.nBase = cand.filter(c => c.kind === 'base').length;
              pending.nSC = cand.filter(c => c.kind === 'sc').length;
              pending.from = xon;
              pendingBucket = null; lastWasTransit = pick.kind !== 'sc'
                || live.some(r => rodSame(r, pick.rod));
              lastDir = dvec(xon, pick.to);
              const dk0 = dkey(xon, pick.to);
              prev2 = prevNode; prevNode = xon;
              xon = pick.to; wake.push(xon); if (wake.length > 40) wake.shift();
              drawXon(xon, wake);
              if (pick.kind === 'base') { baseHops++; useB[dk0] = (useB[dk0] || 0) + 1; return; }
              useS[dk0] = (useS[dk0] || 0) + 1;
              if (live.some(r => rodSame(r, pick.rod))) return;
              const kill0 = killerOf(pick.rod);
              if (kill0) { const k = scKeyOf(kill0[0], kill0[1]); if (k) active.delete(k[0]); }
              const nk0 = scKeyOf(pick.rod[0], pick.rod[1]);
              if (nk0) active.set(nk0[0], nk0[1]);
              pending.killed = kill0;
              live = live.filter(r => r !== kill0).concat([pick.rod]);
              events++; return;
            }
          }
        }
        const scored = cand.map(o => {
          const d = dvec(xon, o.to);
          const isNew = o.kind === 'sc' && !live.some(r => rodSame(r, o.rod));
          const bucket = o.kind === 'base'
            ? 'ax' + d.map(x => x * (d.find(y => y !== 0) < 0 ? -1 : 1)).join(',')
            : d.join(',');
          const tbl = o.kind === 'base' ? useAxis : useS;
          const n = o.kind === 'base' ? 4 : 6;
          let tot = 0; for (const x of Object.values(tbl)) tot += x;
          const excess = (tbl[bucket] || 0) - tot / n;
          let s;
          if (isNew) s = creationScore(o.rod, prevNode);
          else { const r = bestReach(o.to, xon); s = (r === null ? 1e3 : r) + TRANSIT; }
          return { o, bucket, isNew, score: s + bal * excess };
        });
        if (!scored.length) { done = true; stopped = 'blocked'; pending = null; return; }
        const pool = lastWasTransit ? scored.filter(s => s.isNew) : scored;
        const use = pool.length ? pool : scored;
        use.sort((a, b) => a.score - b.score);
        const h = use[0].o;
        pendingBucket = use[0].bucket;
        lastWasTransit = !use[0].isNew;
        pending = h;
        pending.nCand = cand.length;          // how much choice existed this tick
        pending.nBase = cand.filter(c => c.kind === 'base').length;
        pending.nSC = cand.filter(c => c.kind === 'sc').length;
        pending.from = xon;
        lastDir = dvec(xon, h.to);
        const dk = dkey(xon, h.to);
        prev2 = prevNode; prevNode = xon;
        xon = h.to; wake.push(xon); if (wake.length > 40) wake.shift();
        drawXon(xon, wake);                       // xon moves BEFORE the geometry
        if (h.kind === 'base') { baseHops++; useB[dk] = (useB[dk] || 0) + 1;
          useAxis[pendingBucket] = (useAxis[pendingBucket] || 0) + 1; return; }
        useS[dk] = (useS[dk] || 0) + 1;
        if (live.some(r => rodSame(r, h.rod))) return;   // already live: a plain traversal
        const kill = killerOf(h.rod);
        if (kill) { const k = scKeyOf(kill[0], kill[1]); if (k) active.delete(k[0]); }
        const nk = scKeyOf(h.rod[0], h.rod[1]); if (nk) active.set(nk[0], nk[1]);
        pending.killed = kill; live = live.filter(r => r !== kill).concat([h.rod]);
        events++;
      },
      (it) => {
        if (!pending) return { html: '<span style="color:#5d6e85">—</span>', skip: true };
        detect(); drawXon(xon, wake);
        if (!legal(resid)) {
          refused++; freezeOff();
          const nk = scKeyOf(pending.rod[0], pending.rod[1]); if (nk) active.delete(nk[0]);
          if (pending.killed) { const k = scKeyOf(pending.killed[0], pending.killed[1]);
            if (k) active.set(k[0], k[1]);
            live = live.filter(r => r !== pending.rod).concat([pending.killed]); }
          else live = live.filter(r => r !== pending.rod);
          // A REFUSAL VOIDS THE REST OF THE CHAIN. The remaining hops were
          // planned against geometry that has just been rolled back, so
          // draining them would build a rod set nothing checked -- exactly the
          // route to a density violation. The vacuum has the last word on every
          // event, including events that were part of a plan.
          if (chainQ.length) { chainQ = []; biasStats.abandoned =
            (biasStats.abandoned || 0) + 1; }
          biasActive = false;
          return { html: '<span style="color:#ff5c5c">vacuum refused the rod</span>' };
        }
        // the turn the xon just made, measured in the geometry that now
        // exists. THIS is the number the 60-degree rule is a claim about.
        if (prev2 !== null && prevNode !== null) {
          const u = P[prev2].clone().sub(P[prevNode]).normalize();
          const v = P[xon].clone().sub(P[prevNode]).normalize();
          const a = Math.round(180 / Math.PI * Math.acos(Math.max(-1, Math.min(1, u.dot(v)))) * 10) / 10;
          turns[a] = (turns[a] || 0) + 1;
        }
        if (!solids.tets.length) stuck++;
        const c = centOf(tetNodes());            // the PARTICLE's position

        // FINISH DETECTION. Without it every traversal burned its whole tick
        // budget: the particle reached the far end and then milled about until
        // the budget expired, which is what made the loop look like it was
        // idling between crossings.
        //
        // `hiTet` is not a threshold I chose -- it is the furthest along the
        // ray that ANY tet centroid in this lattice can sit, so reaching it
        // means there is nowhere further to go. The stall test is a choice:
        // the 60-degree cage is 8 nodes and ~43 directed states, so a particle
        // that has not improved its best position in STALL ticks is cycling,
        // not travelling.
        const sNow = alongC(c);
        track.push(c.slice());
        // EDGE. Born touching the bound; the run ends the first time any
        // vertex touches the bound again after having left it.
        const atB = tetAtBound(tetNodes());
        if (bornAtBound === null) bornAtBound = atB;
        if (!atB) leftBound = true;
        else if (leftBound) { done = true; stopped = 'edge'; }
        if (sNow > bestS + 1e-9) { bestS = sNow; sinceGain = 0; } else sinceGain++;
        if (!done && sNow >= hiTet - 1e-6) { done = true; stopped = 'traversed'; }
        else if (!done && sinceGain >= STALL) { done = true; stopped = 'stalled'; }
        // The generation as it stands AFTER the move -- recorded per tick so a
        // bias excursion is visible in the trace rather than inferred from it.
        const mNow = modeOf(tetNodes());
        if (LOCK && mNow && mNow !== LOCK) biasStats.offModeTicks++;
        // LIVE READOUT. Everything the four rules are about, on screen while it
        // runs: what was drawn at birth and what is holding tick by tick.
        if (opt.status) opt.status({
          chirality, gen: modeOf(tetNodes()), lock: LOCK,
          dir: v.map(z => +z.toFixed(2)),
          loopLen: LOOPSEQ ? LOOPSEQ.seq.length : 0,
          loopCos: LOOPFIT ? LOOPFIT.cos : null,
          i: loopIdx, rail: fitStats.railDone, steer: fitStats.steerDone,
          rods: live.length, events, tick: it.k,
          angle: (() => { const f = fitAxis(track); if (!f) return null;
            const dt = Math.abs([0,1,2].reduce((sm,k)=>sm+f.dir[k]*v[k],0));
            return +(180/Math.PI*Math.acos(Math.min(1,dt))).toFixed(1); })() });
        trace.push({ tick: it.k, s: +alongC(c).toFixed(2), off: +offC(c).toFixed(2),
                     xs: +along(xon).toFixed(2), tets: solids.tets.length,
                     kind: pending.kind, made: !!(pending.kind === 'sc' && pending.killed),
                     mv: pending.from + '>' + xon, xon, gen: mNow,
                     bias: !!pending.bias,
                     tv: tetNodes().slice().sort((a, b) => a - b).join('.'),
                     nc: pending.nCand, nb: pending.nBase, ns: pending.nSC,
                     blk: chiralBlocked });
        return { html: (pending.kind === 'base'
            ? '<span style="color:#8fa3ba">base</span>'
            : '<span style="color:#7fd4a8">SHORTCUT — flux event</span>')
          + ` &nbsp; tet s=${alongC(centOf(tetNodes())).toFixed(1)}`
          + ` off=${offC(centOf(tetNodes())).toFixed(2)}`
          + ` &nbsp; ${solids.tets.length ? '<b style="color:#ffd166">TET</b>' : 'NO TET'}` };
      },
      // Pace the traversal, NOT the drain. runExperiment iterates every
      // remaining item after `done`, and paying the step delay on each of them
      // is a dead wait between crossings -- ~160 no-op ticks at 50ms was about
      // eight seconds of the loop looking hung.
      { onFreeze: () => freezeOff(),
        pace: () => (!done && window._loopNow && window._loopNow() !== 'none'
                     && window._loopPace) ? window._loopPace() : 0 });

    const hops = trace.length;
    const mt = (() => { let s = 0, n = 0;
      for (const [a, c] of Object.entries(turns)) { s += (+a) * c; n += c; } return n ? s / n : 0; })();
    const out = { chirality, chiralBlocked, mode: LOCK,
      biasStats, loopClosed: !!LOOPSEQ, loopDev: LOOPSEQ && LOOPSEQ.dev, bornAtBound,
      traversal: (() => { const f = fitAxis(track); if (!f) return null;
        const dot = Math.abs([0,1,2].reduce((sm, k) => sm + f.dir[k] * v[k], 0));
        return { dir: f.dir.map(z => +z.toFixed(4)),
                 angleDeg: +(180 / Math.PI * Math.acos(Math.min(1, dot))).toFixed(3),
                 nPts: track.length }; })(),
      loopSeq: LOOPSEQ && LOOPSEQ.seq.map(x => x.d.join(',') + '@' + x.entryAt),
      loop: LOOPFIT && { steps: LOOPFIT.steps.map(x => x.kind + ' ' + x.d.join(',')),
        disp: LOOPFIT.disp, cos: LOOPFIT.cos,
        nRail: LOOPFIT.nRail, nP: LOOPFIT.nP, nQ: LOOPFIT.nQ }, fitStats,
      banAxis: BAN, banBlocked,
      momentum: v.map(x => +x.toFixed(4)), stopped, hops,
      fluxEvents: events, baseHops, refused, ticksWithoutATet: stuck,
      crossedPct: +(100 * (bestS - lo) / (hiTet - lo)).toFixed(0),
      travelled: hops ? +(trace[hops - 1].s - trace[0].s).toFixed(2) : 0,
      driftPerTick: hops > 1 ? +((trace[hops - 1].s - trace[0].s) / (hops - 1)).toFixed(4) : null,
      offMax: hops ? +Math.max(...trace.map(t => t.off)).toFixed(2) : 0,
      meanTurn: +mt.toFixed(1), turnAngles: turns,
      baseAxisUse: Object.values(useAxis).sort((a, b) => a - b),   // want even
      baseSignedUse: Object.values(useB).sort((a, b) => a - b),    // free: the momentum
      scDirUse: Object.values(useS).sort((a, b) => a - b),         // want even
      trace };
    window._XM = out;
    console.log('XONMOM', JSON.stringify({ momentum: out.momentum, stopped: out.stopped,
      crossedPct: out.crossedPct, hops: out.hops, fluxEvents: out.fluxEvents,
      baseHops: out.baseHops, ticksWithoutATet: out.ticksWithoutATet,
      travelled: out.travelled, offMax: out.offMax, meanTurn: out.meanTurn }, null, 1));
    return out;
  };

  // ==========================================================================
  // THE PROTON  --  stationary. No line, no momentum, no destination.
  //
  // One xon starting at the CENTRE of the lattice, building an octahedron and
  // then living in it.
  //
  // WHAT AN OCTAHEDRON IS HERE, measured: four shortcuts forming a closed
  // 4-cycle. All 642 such cycles in the 339-node lattice alternate two axes,
  // i.e. each sits in ONE flux mode/plane (XY 214, XZ 214, YZ 214). Two
  // families by sublattice: oct+oct+pack+pack (336) and tet x4 (306).
  //
  // THE TWO APEXES ARE FREE. Actualising the ring pulls the square's side from
  // 2/sqrt3 to 1, and the apexes are already exactly one BASE edge from all
  // four corners -- so square side 1 and apex-to-corner 1 forces a regular
  // octahedron, apex-to-apex sqrt2, at no extra flux cost. Confirmed in the
  // engine: 4 shortcuts -> 1 octahedron detected, six vertices, and exactly
  // TWELVE 90-degree angles, which is 6 vertices x 2 right-angle pairs and not
  // one more.
  //
  // THE ANGLES. At any vertex of a regular octahedron the four neighbours sit
  // in a square around it: 2 pairs at 90 degrees, 4 pairs at 60. So
  //
  //   90 deg  =  shortcut -> shortcut, FOLLOWING THE EQUATORIAL RING
  //   60 deg  =  shortcut <-> base edge, STEPPING OUT TO AN APEX
  //
  // Uniform choice would give 90 one third of the time. Zac's rule -- 90 two
  // thirds, 60 one third -- is a two-fold bias toward circulating the ring,
  // which is the thing that builds and holds the solid together. Note 4 ring
  // hops plus one apex excursion (out and back) is 6 moves at exactly 4:2.
  //
  // TESTING THE TURN. Both hops are unit, so the far pair fixes the angle:
  //   |prev - cand| = 1      -> 60 degrees (equilateral)
  //   |prev - cand| = sqrt2  -> 90 degrees (right isoceles)
  // For a CREATION the new rod is not unit yet, so 90 is tested instead as
  // integer orthogonality of the two <100> directions -- which is exact in the
  // rest lattice and survives contraction, because a square corner stays a
  // square corner. That is why the proton has no chicken-and-egg where the
  // electron did.
  // ==========================================================================
  // PROBE: the live rods, as integer coordinate pairs. Read-only.
  window._RODS = () => [...active.values()].map(([i, j]) =>
    ({ a: NODE[i].slice(), b: NODE[j].slice(),
       d: [0, 1, 2].map(k => NODE[j][k] - NODE[i][k]) }));

  // ---- PROBE: deep holes ---------------------------------------------------
  // MEASUREMENT ONLY. No CA reads this; it changes no physics. It answers one
  // design question: is "the point inside the lattice furthest from any node" a
  // well-defined attractor, or is it degenerate? In an UNDEFORMED lattice every
  // Voronoi vertex sits at exactly the covering radius, so there are many
  // equally-deep holes -- and "the nearest one" would then track the xon rather
  // than hold it. Deformation should break that tie; this measures whether it
  // does, and by how much.
  //
  // Samples a grid inside a ball about LCENTER and reports the deepest points
  // of the CURRENT solved positions P, clustered so that one hole counts once.
  window._DEEPHOLE = function (opt) {
    opt = opt || {};
    _sync();
    const R = opt.radius === undefined ? 2.4 : opt.radius;
    const h = opt.step || 0.1;
    // opt.centre: search ball centre in WORLD units. Defaults to the lattice
    // centre. The CA passes the xon's own position -- depth is created locally
    // by the xon's rods, and centring on the xon also realises the "if several
    // are equally deep, take the nearest" clause for free.
    const c = opt.centre ? opt.centre.slice()
                         : [LCENTER[0] * S, LCENTER[1] * S, LCENTER[2] * S];
    // Whole lattice: with the bucket index below, indexing every node costs
    // nothing and removes a radius parameter that could silently clip contacts.
    const near = [];
    for (let i = 0; i < P.length; i++) near.push(P[i]);
    // Bucketed nearest-node lookup. Scanning every node per sample is O(N) and
    // makes an interior-wide search unaffordable per tick; this makes it O(1).
    // Purely a speed change -- the value returned is the same distance.
    // Flat integer-indexed cells, NOT a string-keyed Map: this is the hot path
    // (hundreds of thousands of calls per tick) and string key construction
    // alone dominated the whole attractor.
    const B = 0.9;
    let lox = Infinity, loy = Infinity, loz = Infinity;
    let hix = -Infinity, hiy = -Infinity, hiz = -Infinity;
    for (const q of near) {
      if (q.x < lox) lox = q.x; if (q.x > hix) hix = q.x;
      if (q.y < loy) loy = q.y; if (q.y > hiy) hiy = q.y;
      if (q.z < loz) loz = q.z; if (q.z > hiz) hiz = q.z;
    }
    const nx = Math.max(1, Math.ceil((hix - lox) / B) + 1);
    const ny = Math.max(1, Math.ceil((hiy - loy) / B) + 1);
    const nz = Math.max(1, Math.ceil((hiz - loz) / B) + 1);
    const cells = new Array(nx * ny * nz);
    const cidx = (a, b, g) => (a * ny + b) * nz + g;
    for (const q of near) {
      const a = Math.floor((q.x - lox) / B), b = Math.floor((q.y - loy) / B),
            g = Math.floor((q.z - loz) / B);
      const t = cidx(a, b, g);
      if (!cells[t]) cells[t] = [];
      cells[t].push(q);
    }
    const depth = (px, py, pz) => {
      const ia = Math.floor((px - lox) / B), ib = Math.floor((py - loy) / B),
            ig = Math.floor((pz - loz) / B);
      let m = Infinity;
      for (let r = 0; r <= 12; r++) {
        for (let a = ia - r; a <= ia + r; a++) {
          if (a < 0 || a >= nx) continue;
          const da = Math.abs(a - ia);
          for (let b = ib - r; b <= ib + r; b++) {
            if (b < 0 || b >= ny) continue;
            const db = Math.abs(b - ib);
            for (let g = ig - r; g <= ig + r; g++) {
              if (g < 0 || g >= nz) continue;
              // shell only: interior cells were covered at smaller r
              if (Math.max(da, db, Math.abs(g - ig)) !== r) continue;
              const arr = cells[cidx(a, b, g)]; if (!arr) continue;
              for (let t = 0; t < arr.length; t++) {
                const q = arr[t];
                const dx = px - q.x, dy = py - q.y, dz = pz - q.z;
                const d2 = dx * dx + dy * dy + dz * dz;
                if (d2 < m) m = d2;
              }
            }
          }
        }
        // Anything still unscanned lies at least r*B away, so once the best
        // found beats that, no further shell can improve on it.
        if (m < Infinity && m <= r * B * (r * B)) break;
      }
      return Math.sqrt(m);
    };
    // A hole is a LOCAL MAXIMUM of depth, i.e. a Voronoi vertex -- not "any
    // grid sample that came out deep". The grid only SEEDS the search; reading
    // depths straight off it is what produced the bogus four-way tie, because a
    // 0.1 grid on a cone-shaped peak undershoots the true summit by ~0.04 and a
    // tolerance band then promotes the shoulders to equal rank.
    // Local maxima of depth on a grid over one ball. Seeds only -- the true
    // summit comes from refine().
    const regionSeeds = (cc, RR, hh) => {
      const n1 = Math.round(2 * RR / hh) + 1;
      const at = (i, j, k) => [cc[0] - RR + i * hh, cc[1] - RR + j * hh, cc[2] - RR + k * hh];
      const G = new Float64Array(n1 * n1 * n1);
      const gi = (i, j, k) => (i * n1 + j) * n1 + k;
      for (let i = 0; i < n1; i++) for (let j = 0; j < n1; j++) for (let k = 0; k < n1; k++) {
        const p = at(i, j, k);
        G[gi(i, j, k)] = Math.hypot(p[0] - cc[0], p[1] - cc[1], p[2] - cc[2]) > RR
          ? -1 : depth(p[0], p[1], p[2]);
      }
      const out = [];
      for (let i = 1; i < n1 - 1; i++) for (let j = 1; j < n1 - 1; j++) for (let k = 1; k < n1 - 1; k++) {
        const v = G[gi(i, j, k)]; if (v < 0) continue;
        let top = true;
        for (let a = -1; a <= 1 && top; a++) for (let b = -1; b <= 1 && top; b++)
          for (let g = -1; g <= 1 && top; g++) {
            if (!a && !b && !g) continue;
            if (G[gi(i + a, j + b, k + g)] > v) top = false;
          }
        if (top) out.push({ p: at(i, j, k), v });
      }
      return out;
    };
    // ONLY VOIDS ADJACENT TO A SHORTCUT ARE CONTENDERS. The undeformed lattice
    // sits at the covering radius everywhere (measured: 0.6455 at every one of
    // its 504 holes), so depth can only be created by the deformation a rod
    // causes. Searching the whole interior wastes nearly all of its work on
    // ground that is provably flat. With no rods there is no contender at all,
    // which is correct -- nothing has been built yet to be attracted to.
    const rods = [...active.values()].map(([i, j]) => [P[i], P[j], i, j]);
    // The rod path does no grid search at all -- it solves for the voids
    // exactly, below. Only the whole-interior survey still needs seeds.
    const seeds = opt.aroundRods ? [] : regionSeeds(c, R, h);
    // opt.maxSeeds: refine only the deepest N grid maxima. The summit can only
    // be at the deepest seeds, so this is a cost cut, not an accuracy one --
    // but it does truncate the FULL hole census, so leave it unset for surveys.
    // opt.tieNear: "if several are equally deep, take the one nearest the xon".
    // Grid depths are quantised to 0.01 so the degenerate holes of an
    // undeformed lattice group together and proximity decides among them; a
    // genuinely deeper hole still wins outright.
    const dToT = p => opt.tieNear
      ? Math.hypot(p[0] - opt.tieNear[0], p[1] - opt.tieNear[1], p[2] - opt.tieNear[2]) : 0;
    seeds.sort((a, b) => {
      const va = Math.round(a.v * 100), vb = Math.round(b.v * 100);
      return vb !== va ? vb - va : dToT(a.p) - dToT(b.p);
    });
    const seedList = (opt.maxSeeds ? seeds.slice(0, opt.maxSeeds) : seeds).map(s => s.p);
    // Shrinking hill-climb to the true summit. max-min of distances is not
    // smooth, so this walks rather than differentiates.
    const DIRS = [];
    for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (let g = -1; g <= 1; g++) {
      if (!a && !b && !g) continue;
      const n = Math.hypot(a, b, g); DIRS.push([a / n, b / n, g / n]);
    }
    const refine = (p0) => {
      let bp = p0.slice(), bd = depth(bp[0], bp[1], bp[2]), step = h;
      while (step > 1e-6) {
        let moved = false;
        for (const d of DIRS) {
          const q = [bp[0] + d[0] * step, bp[1] + d[1] * step, bp[2] + d[2] * step];
          if (Math.hypot(q[0] - c[0], q[1] - c[1], q[2] - c[2]) > R) continue;  // stay inside
          const dq = depth(q[0], q[1], q[2]);
          if (dq > bd + 1e-13) { bd = dq; bp = q; moved = true; }
        }
        if (!moved) step *= 0.5;
      }
      return { p: bp, d: bd };
    };
    // contacts: how many nodes sit at the hole's own depth. 4 = tetrahedral
    // void, 6 = octahedral void. Anything else means the tolerance is wrong or
    // the hole is not a real Voronoi vertex.
    const cTol = opt.contactTol === undefined ? 0.01 : opt.contactTol;
    const contactIdx = (p, d) => {
      const out = [];
      for (let i = 0; i < P.length; i++) {
        const q = P[i];
        if (Math.hypot(p[0] - q.x, p[1] - q.y, p[2] - q.z) <= d * (1 + cTol)) out.push(i);
      }
      return out;
    };
    const holes = [];
    for (const s of seedList) {
      const r = refine(s);
      // A refinement that terminated ON the sampling boundary never reached a
      // Voronoi vertex -- it was clamped. Those are the spurious 1/2/3-contact
      // entries, and an attractor must never be allowed to chase one.
      if (opt.dropBoundary &&
          Math.hypot(r.p[0] - c[0], r.p[1] - c[1], r.p[2] - c[2]) > R - 1e-3) continue;
      if (holes.some(k => Math.hypot(k.at[0] - r.p[0], k.at[1] - r.p[1],
                                     k.at[2] - r.p[2]) < 0.08)) continue;
      const ci = contactIdx(r.p, r.d);
      holes.push({ d: r.d, at: r.p, n: ci.length, idx: ci });
    }
    // EXACT VOIDS ADJACENT TO A ROD -- no grid, no refinement, no tolerance on
    // position. A deep hole IS the circumcentre of four nodes whose circumsphere
    // is empty (a Voronoi vertex). A hole adjacent to rod (a,b) has both a and b
    // among those four, so it is enough to try pairs drawn from the handful of
    // nodes near the rod. Grid-sampling a box per rod cost ~60k distance queries
    // a tick to locate three voids; this costs a few hundred and is exact.
    if (opt.aroundRods) {
      const circum = (p0, p1, p2, p3) => {
        const a = [p1.x - p0.x, p1.y - p0.y, p1.z - p0.z];
        const b = [p2.x - p0.x, p2.y - p0.y, p2.z - p0.z];
        const g = [p3.x - p0.x, p3.y - p0.y, p3.z - p0.z];
        const la = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) / 2;
        const lb = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]) / 2;
        const lg = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]) / 2;
        const det = a[0] * (b[1] * g[2] - b[2] * g[1])
                  - a[1] * (b[0] * g[2] - b[2] * g[0])
                  + a[2] * (b[0] * g[1] - b[1] * g[0]);
        if (Math.abs(det) < 1e-9) return null;           // the four are coplanar
        const x = (la * (b[1] * g[2] - b[2] * g[1]) - a[1] * (lb * g[2] - b[2] * lg)
                                                    + a[2] * (lb * g[1] - b[1] * lg)) / det;
        const y = (a[0] * (lb * g[2] - b[2] * lg) - la * (b[0] * g[2] - b[2] * g[0])
                                                  + a[2] * (b[0] * lg - lb * g[0])) / det;
        const z = (a[0] * (b[1] * lg - lb * g[1]) - a[1] * (b[0] * lg - lb * g[0])
                                                  + la * (b[0] * g[1] - b[1] * g[0])) / det;
        return [p0.x + x, p0.y + y, p0.z + z];
      };
      holes.length = 0;
      for (const [qa, qb, ia, ib] of rods) {
        const nb = [];
        for (let i = 0; i < P.length; i++) {
          if (i === ia || i === ib) continue;
          const q = P[i];
          if (Math.hypot(q.x - qa.x, q.y - qa.y, q.z - qa.z) < 1.9 &&
              Math.hypot(q.x - qb.x, q.y - qb.y, q.z - qb.z) < 1.9) nb.push(i);
        }
        for (let u = 0; u < nb.length; u++) for (let v = u + 1; v < nb.length; v++) {
          const p = circum(qa, qb, P[nb[u]], P[nb[v]]);
          if (!p) continue;
          const d = Math.hypot(p[0] - qa.x, p[1] - qa.y, p[2] - qa.z);
          // A near-coplanar quadruple has a circumcentre that runs off to
          // infinity, and such a sphere passes the emptiness test vacuously --
          // measured, one came back at radius 12831 with all 339 nodes as
          // "contacts". No real void can exceed the covering radius, so cap it.
          if (d > (opt.maxVoidRadius || 1.2)) continue;
          // empty circumsphere <=> nothing nearer than the four that define it
          if (depth(p[0], p[1], p[2]) < d * (1 - 1e-6)) continue;
          if (holes.some(k => Math.hypot(k.at[0] - p[0], k.at[1] - p[1],
                                         k.at[2] - p[2]) < 0.05)) continue;
          const ci = contactIdx(p, d);
          holes.push({ d, at: p, n: ci.length, idx: ci });
        }
      }
    }
    holes.sort((a, b) => b.d - a.d);
    if (opt.tieNear && holes.length) {                 // same rule, refined values
      const lim = holes[0].d * (1 - cTol);
      const tie = holes.filter(k => k.d >= lim);
      tie.sort((a, b) => dToT(a.at) - dToT(b.at));
      holes.splice(0, tie.length, ...tie);
    }
    const kind = n => n === 6 ? 'oct' : (n === 4 ? 'tet' : 'other:' + n);
    const deepest = holes[0] || null;
    // how many holes tie the deepest, to within the contact tolerance
    const tied = deepest ? holes.filter(k => k.d >= deepest.d * (1 - cTol)).length : 0;
    const tally = {};
    for (const k of holes) tally[kind(k.n)] = (tally[kind(k.n)] || 0) + 1;
    const out = { coveringRadius: deepest ? +deepest.d.toFixed(4) : null,
      deepestKind: deepest ? kind(deepest.n) : null,
      deepestContacts: deepest ? deepest.n : null,
      holesTiedAtDeepest: tied,
      totalHoles: holes.length, byKind: tally,
      sampledBallRadius: R, grid: h, activeRods: active.size,
      nearestHoleFromCentre: deepest
        ? +Math.hypot(deepest.at[0] - c[0], deepest.at[1] - c[1], deepest.at[2] - c[2]).toFixed(3)
        : null,
      top: holes.slice(0, 6).map(k => ({ d: +k.d.toFixed(4), contacts: k.n,
        kind: kind(k.n), at: k.at.map(v => +v.toFixed(3)) })) };
    if (!opt.quiet) console.log('DEEPHOLE', JSON.stringify(out, null, 1));
    return out;
  };

  // ==========================================================================
  // THE NUCLEON  --  the rule set worked out by hand on 6 Aug, coded straight.
  //
  // Rules, in priority order. Each is a HARD filter except 5, which ranks.
  //   1  no shuttling -- a-b-a is forbidden outright
  //   2  directional balance -- move only in a least-traversed BASE direction.
  //      Shortcuts do NOT count: they were tried in the balance set three ways
  //      (signed, pooled, pooled-at-2x) and every one of them strangles the
  //      nucleon. Pooled is worst: one X+ makes the whole X axis read as spent,
  //      so the X- that closes the equator is barred and no oct can ever form.
  //      Out of the set entirely, a shortcut axis is re-usable and the equator
  //      closes -- measured by hand, 2 octs by tick 7.
  //   3  nucleon identity -- rolling blocks of three turns.
  //      proton  = two 90 and one 60      neutron = two 60 and one 90
  //      Order inside a block is free. A PROTON may never break it; a neutron
  //      may, and only to keep rules 1 and 2.
  //   4  attractor -- rank by distance to the deepest hole
  //   5  identity preference -- proton prefers 90, neutron prefers 60
  //
  // THE CASCADE WAS REORDERED ON 6 AUG and PREFER-SHORTCUTS WAS RETIRED.
  //
  // It used to run: block, then the identity preference, then prefer-shortcuts,
  // then the attractor. Both of the middle two were HARD FILTERS ahead of the
  // attractor, so the attractor never chose anything -- it only ranked what
  // they left. Measured on a YZ proton: at t2 the two nearest moves at dA 0.645
  // were 60s, the preference binned them, and the nearest surviving 90 was
  // 0.947. Prefer-shortcuts was retired outright rather than demoted; the
  // identity preference survives as the LAST tie-break, among moves already
  // tied for nearest, where it cannot cost distance.
  //
  // Note what R3 still discards regardless: `need` only ever holds 60 and/or
  // 90, so every 75 and off-class move dies at R3 and can only be taken by a
  // neutron breaking its block.
  //
  // Anything written before 6 Aug calls the attractor "r5".
  //
  // Plus GENERATION (once two shortcut axes carry rods the third is barred,
  // fixing XY / XZ / YZ) and CHIRALITY -- which is NOT fixed at birth: it is
  // discovered, needing the mode first and then base traversals to single out
  // one of the mode's two polarizations. See the determination note below.
  //
  // SEVERANCE, as in the scaffolded build: two tets may not share an edge
  // while attached to the oct, since the pair raises a second octahedron. When
  // a new tet shares an edge with an older one THE OLD TET DIES -- its enabling
  // rod, the one the new tet does not carry, is severed. `octs` above 1 is the
  // symptom this exists to prevent, and it is the number to watch.
  // THE NUCLEON STEPPER. One source of truth for the rules: the loop below is
  // a thin driver over this, so anything played by hand and anything played at
  // speed run identical code. `look()` reports the board, `step()` applies one
  // decision. Nothing here scores in a way the rules do not name.
  window._NSTEP = (() => {
    // CHIRALITY IS DISCOVERED, NOT DRAWN AT BIRTH.
    //
    // There are exactly THREE maps in the whole scheme -- the three cardinal
    // polarizations X, Y, Z -- and each flux mode uses the two lying along its
    // own ACTIVE axes. Computed over all three modes, every one of the six
    // resulting maps lands on a 4-valent <100> zonotope vertex of magnitude 4,
    // which is §6's circulation criterion; the dormant axis's polarization is
    // never a chirality of that mode.
    //
    //     XY -> {X+, Y+}      XZ -> {X+, Z+}      YZ -> {Y+, Z+}
    //
    // The old code hardcoded L = X+ and R = Z+, i.e. the XZ pair, and applied
    // it to every mode. Correct for the electron, which is XZ by construction;
    // simply the wrong map for a YZ or XY nucleon, and that is why a
    // geometrically valid 90 kept coming back barred.
    //
    // ORDER OF DETERMINATION. Nothing about chirality exists until the mode
    // does, because "toward" is measured along the DORMANT axis and there is
    // no dormant axis until two shortcut axes carry rods. After that, base
    // traversals decide -- but ONE IS NEVER ENOUGH. Measured over all three
    // modes: each of the 8 base directions lies in exactly 2 of the mode's 4
    // sign-sets, one per family, so a single hop fixes the orientation and
    // never the family. Two hops settle it in 32 of 64 ordered pairs.
    //
    // So the candidates are carried and filtered, and chirality LATCHES when
    // one survives. While more than one survives a base move is legal if any
    // surviving candidate admits it, which is also what stops the xon walking
    // into a direction consistent with none of them.
    const POLK = { X: 0, Y: 1, Z: 2 };
    const MODE_FAM = { XY: ['X', 'Y'], XZ: ['X', 'Z'], YZ: ['Y', 'Z'] };
    const POL_SET = (() => {
      const dirs = [];
      for (const x of [1, -1]) for (const y of [1, -1]) for (const z of [1, -1])
        dirs.push([x, y, z]);
      const out = {};
      for (const a of ['X', 'Y', 'Z']) for (const s of [1, -1])
        out[a + (s > 0 ? '+' : '-')] =
          new Set(dirs.filter(d => d[POLK[a]] * s > 0).map(d => d.join(',')));
      return out;
    })();

    let identity = 'proton', chSet = null, chName = null, chCand = null;
    let xon = null, prevNode = null, live = [], wake = [];
    let tick = 0, events = 0, baseHops = 0, refused = 0, severed = 0, stalled = null;
    const baseCount = new Map(), scCount = new Map(), classLog = [], log = [];
    // UNDO. Every scrap of state the automaton reads, captured before a move
    // and restored whole -- including `active`, since the engine's own rod set
    // is what the solver rebuilds from. Stepping back and forward again may
    // land on a DIFFERENT move now that r5 is sampled, which is the point.
    const history = [], chJournal = [];

    // CHIRALITY DETERMINATION, run after every applied move.
    //   1. no mode yet            -> nothing to determine
    //   2. mode just clicked in   -> open the four sign-sets of that mode
    //   3. a base traversal       -> discard candidates that do not admit it
    //   4. one candidate left     -> LATCH, for life
    // Step 3 is why the candidates are carried rather than resolved on the
    // spot: measured over all three modes, every base direction lies in two of
    // the four sets, so no single traversal can ever pick one out.
    function chUpdate(dirKey, wasBase) {
      if (chSet) return;
      const mode = modeName();
      if (!mode) return;
      if (!chCand) {
        chCand = [];
        for (const a of MODE_FAM[mode]) chCand.push(a + '+', a + '-');
        chJournal.push('t' + tick + ' mode ' + mode + ' -> candidates '
          + chCand.join(' '));
      }
      if (!wasBase) return;
      const kept = chCand.filter(n => POL_SET[n].has(dirKey));
      if (!kept.length) {                 // admitted by none: should be barred
        chJournal.push('t' + tick + ' ' + dirKey + ' admitted by NO candidate');
        return;
      }
      if (kept.length !== chCand.length)
        chJournal.push('t' + tick + ' base ' + dirKey + ' -> ' + kept.join(' '));
      chCand = kept;
      if (chCand.length === 1) {
        chName = chCand[0]; chSet = POL_SET[chName];
        chJournal.push('t' + tick + ' LATCHED ' + chName);
      }
    }
    // ALL EIGHT base directions and ALL SIX shortcut senses, so the readout can
    // show what chirality and generation are excluding, not only what was used.
    const ALL_BASE = [1, -1].flatMap(x => [1, -1].flatMap(y => [1, -1]
      .map(z => [x, y, z].join(','))));
    const ALL_SC = ['2,0,0', '-2,0,0', '0,2,0', '0,-2,0', '0,0,2', '0,0,-2'];
    const dv = (a2, b2) => [0, 1, 2].map(k => NODE[b2][k] - NODE[a2][k]);
    const ax = d => (d[0] ? 'X' : (d[1] ? 'Y' : 'Z'));
    const same = (x, y) => (x[0] === y[0] && x[1] === y[1])
                        || (x[0] === y[1] && x[1] === y[0]);
    const K2 = (i, j) => Math.min(i, j) * 100000 + Math.max(i, j);

    // TETS AND OCTS COME FROM THE ENGINE'S DETECTOR, not a parallel
    // combinatorial one. Writing my own was the bug: `solids` reported 7 octs
    // and 11 tets while my rod-pair scan reported ZERO of each, so both the
    // no-tets-before-the-oct gate and the severance rule were silent no-ops for
    // every tick of every run. Never re-derive what the engine already detects.
    //
    // AN OCT IS `[apexA, apexB, [ring x4]]` -- a NESTED triple, not a flat
    // vertex list. Treating it as flat made `rodsIn` match nothing, so
    // `ringOf()` returned [] on a board with two octs on it and severance's
    // "never cut a ring rod" guard was vacuous: at t8 it cut 200-131, a rod
    // holding the very oct it was meant to protect.
    const engineTets = () => solids.tets.map(t => t.slice());
    const engineOcts = () => solids.octs.map(o => o.slice());
    const octVerts = o => [o[0], o[1], ...o[2]];
    // Which live rods lie inside a given solid's vertex set.
    const rodsIn = vs => live.filter(r => vs.includes(r[0]) && vs.includes(r[1]));
    // THE HOME OCT is latched the first time one appears and never re-picked;
    // `solids.octs[0]` is not stable once a second oct exists, and the whole
    // point of severance is to protect the FIRST one.
    let homeOct = null;
    // WHAT SEVERANCE MUST NOT CUT: every rod of the home oct, found over ALL
    // SIX of its vertices.
    //
    // Asking for `homeOct[2]` was wrong and cost the oct at t12. An octahedron
    // has THREE great squares and `detect()` names one of them arbitrarily --
    // it scans `diag` for a sqrt(2) pair and takes the first, so `[a,b]` is
    // whichever diagonal came up, not the equator's apexes. Measured at t11:
    // the oct was {168,238,169,200,206,237} with equator 168-169-238-237, and
    // detect reported apexes (168,238) and sh=[169,200,206,237] -- a square
    // holding NOT ONE of the four equator rods. `rodsIn(sh)` returned [], so
    // nothing was protected and severance cut 238-237, an equator rod, and the
    // octahedron died for no geometric reason.
    //
    // Over all six vertices there is no ambiguity: a rod is a unit edge, an
    // oct's diagonals are sqrt(2), so every rod between two oct vertices is an
    // edge of that oct and cutting any one of them destroys it.
    const ringOf = () => homeOct ? rodsIn(octVerts(homeOct)) : [];

    // NOT EVERY EDGE OF A DETECTED SOLID IS A ROD WE OWN. `detect()` is
    // geometric on the SOLVED positions, so the solver squeezing a shortcut
    // slot to 1.0 raises a solid nobody built -- an implied shortcut. Measured
    // at t8 of the first proton: tet {131,168,169,200} has edges 131-200 and
    // 168-169 at d=1.000000 with neither in `active`. That tet, and the second
    // oct it completes, cannot be severed: there is no rod to cut. Only the
    // gate can prevent it, which is why the gate has to solve.
    const impliedEdgesOf = vs => {
      const out = [];
      for (let i = 0; i < vs.length; i++) for (let j = i + 1; j < vs.length; j++) {
        const a2 = vs[i], b2 = vs[j];
        if (Math.abs(P[a2].distanceTo(P[b2]) - 1) > 1e-6) continue;
        if (_baseSet.has(K(a2, b2))) continue;
        if (live.some(r => same(r, [a2, b2]))) continue;
        out.push(a2 + '-' + b2);
      }
      return out;
    };
    const sharingPairs = () => {
      const t = engineTets(), out = [];
      for (let i = 0; i < t.length; i++) for (let j = i + 1; j < t.length; j++) {
        const sh = t[i].filter(v => t[j].includes(v));
        if (sh.length >= 2) out.push([i, j, sh]);
      }
      return out;
    };

    const modeAxes = () => {
      const m = [...new Set(live.map(r => ax(dv(r[0], r[1]))))];
      return m.length >= 2 ? m.slice(0, 2) : ['X', 'Y', 'Z'];
    };
    // The flux mode as a NAME, or null while fewer than two shortcut axes
    // carry rods. Null is the honest answer: with no dormant axis there is no
    // "toward", so chirality is not merely unknown, it does not yet exist.
    const modeName = () => {
      const m = [...new Set(live.map(r => ax(dv(r[0], r[1]))))].sort();
      if (m.length !== 2) return null;
      const n = m.join('');
      return MODE_FAM[n] ? n : null;
    };
    // Which base directions chirality currently permits: the latched map once
    // one exists, otherwise the union of the surviving candidates, otherwise
    // -- no mode yet -- all eight.
    const allowedDirs = () => {
      if (chSet) return chSet;
      if (chCand && chCand.length) {
        const u = new Set();
        for (const n of chCand) POL_SET[n].forEach(d => u.add(d));
        return u;
      }
      return new Set(ALL_BASE);
    };
    // Balance is measured over whatever chirality currently permits, so it
    // widens and narrows with the candidate set rather than assuming four.
    const balanceRows = () => {
      const rows = [...allowedDirs()].map(id => ({ id, n: baseCount.get(id) || 0 }));
      const min = Math.min(...rows.map(r => r.n));
      rows.forEach(r => r.ok = (r.n === min));
      return rows;
    };
    // THE READOUT'S OWN VIEW. Counts every direction ever offered, with the
    // reason it is or is not spendable -- balance binds base directions only,
    // generation binds shortcut axes only, and chirality bars four of the
    // eight base senses outright. Purely descriptive: nothing here is read by
    // a rule.
    const dirStats = () => {
      const rows = balanceRows(), axes = modeAxes(), ok = allowedDirs();
      return {
        base: ALL_BASE.map(id => ({ id, n: baseCount.get(id) || 0,
          allowed: ok.has(id),
          atMin: !!(rows.find(r => r.id === id) || {}).ok })),
        sc: ALL_SC.map(id => ({ id, n: scCount.get(id) || 0,
          allowed: axes.indexOf(ax(id.split(',').map(Number))) >= 0 })),
        axes,
        turns: { n60: classLog.filter(c => c === 60).length,
                 n90: classLog.filter(c => c === 90).length,
                 total: classLog.length },
      };
    };
    // NUCLEON IDENTITY: A BALANCED BLOCK OF THREE, PLUS A PREFERENCE.
    //
    // 90 is an up quark, 60 is a down. A proton is uud, a neutron is udd, and
    // EVERY GROUP OF THREE MUST BALANCE -- but the order inside a group is
    // free. A proton may run uud, udu or duu; a neutron dud, ddu or udd. So the
    // first turn of a block is a free choice of either class, the second is
    // free only if the first left both still owing, and the third is whatever
    // the block still owes.
    //
    // THE PREFERENCE ONLY BREAKS TIES INSIDE THAT FREEDOM: the proton takes a
    // 90 when the block permits either, the neutron a 60. It never overrides
    // the composition.
    //
    // Both halves are load-bearing and neither works alone:
    //   - quota with no preference -- the greedy shortcut rule spent both 90s
    //     on the first two ticks, so when the move that CLOSED the octahedron
    //     appeared at t4 (legal, vacuum-approved, tet-free, nearest the
    //     attractor, +1 OCT) the block owed a 60 and forbade it.
    //   - preference with no quota -- the proton simply took three consecutive
    //     90s and closed the equatorial square outright. The oct formed, but
    //     for no reason: a square of four right angles is not uud.
    //   - a strict 90-90-60 order changed nothing, since that greedy prefix is
    //     exactly what the strict pattern demands.
    const PREF = { proton: 90, neutron: 60 };
    const blockState = () => {
      const spent = classLog.slice(Math.floor(classLog.length / 3) * 3);
      const want90 = identity === 'proton' ? 2 : 1;
      const want60 = identity === 'proton' ? 1 : 2;
      const left90 = want90 - spent.filter(c => c === 90).length;
      const left60 = want60 - spent.filter(c => c === 60).length;
      const n90 = classLog.filter(c => c === 90).length;
      return { identity, spent, left90, left60, pref: PREF[identity],
               // what the block still PERMITS this tick -- both while it owes
               // both, one class once the other is spent
               need: [].concat(left90 > 0 ? [90] : [], left60 > 0 ? [60] : []),
               n90, n60: classLog.filter(c => c === 60).length,
               total: classLog.length,
               ratio: classLog.length ? n90 / classLog.length : null };
    };
    const angleAt = (a2, b2, c2) => (a2 && b2 && c2)
      ? +(Math.acos(Math.max(-1, Math.min(1, (a2 * a2 + b2 * b2 - c2 * c2) / (2 * a2 * b2))))
          * 180 / Math.PI).toFixed(1) : null;
    const cls = t => t === null ? 'first'
      : (t >= 45 && t <= 60 ? 60 : (t > 60 && t < 75 ? 75
      : (t >= 75 && t <= 105 ? 90 : 'off')));

    // THE ATTRACTOR IS TRACKED, NOT RE-ELECTED EVERY TICK.
    //
    // `tieNear` used to be the XON'S OWN POSITION, and _DEEPHOLE quantises
    // depth to 0.01 before breaking ties by proximity. Measured on a live
    // board: four holes tied at q=66 and the winner was the SHALLOWER one, at
    // depth 0.6591 and 0.659 from the xon, beating a genuinely deeper hole at
    // 0.6624 that was 1.565 away. So the target was re-elected each tick by
    // where the walker happened to be -- move the xon and a different hole
    // wins. That is the "it moved away and created a new attractor" failure,
    // and it makes r5 partly circular: you cannot steer toward a goal that
    // follows you.
    //
    // Fixing the TIE-BREAK alone was not enough, and the measurement says why.
    // After one move the tracked hole was still there, shifted 0.039 -- but the
    // xon's own new rod had dug a DEEPER hole beside it, q=67 against q=64, and
    // depth beats a tie. So the target still jumped 0.651 to a void the xon had
    // just created by moving. Re-election by depth is the same circularity as
    // re-election by proximity: the walker's own rods deepen the ground under
    // its feet, so "the deepest hole" tracks the walker no matter how the tie
    // is broken.
    //
    // THE ATTRACTOR IS THEREFORE LATCHED BY IDENTITY, NOT BY DEPTH. It is
    // acquired once -- deepest hole, xon-nearest tie-break -- and thereafter
    // the target is the SAME VOID, found each tick as the hole nearest to
    // where it was last seen. It moves only as far as that void deforms.
    // Re-acquisition happens only if the census comes back empty, i.e. the
    // void is genuinely gone.
    //
    // The deliberate cost: a genuinely deeper hole elsewhere is now ignored.
    // That is the point. A goal that can be superseded by the consequences of
    // moving is not a goal.
    let homeAttr = null, attrJump = null;
    const attract = () => {
      const d = window._DEEPHOLE({ aroundRods: true, quiet: true,
        tieNear: homeAttr
          || (xon !== null ? [P[xon].x, P[xon].y, P[xon].z] : null) });
      const tops = (d.top && d.top.length) ? d.top : null;
      if (!tops) { homeAttr = null; attrJump = null; drawAttractor(null); return null; }
      let top = tops[0];
      if (homeAttr) {                      // follow the void we already chose
        const near = (h) => Math.hypot(h.at[0] - homeAttr[0],
          h.at[1] - homeAttr[1], h.at[2] - homeAttr[2]);
        top = tops.reduce((best, h) => near(h) < near(best) ? h : best, tops[0]);
      }
      attrJump = homeAttr ? +Math.hypot(top.at[0] - homeAttr[0],
        top.at[1] - homeAttr[1], top.at[2] - homeAttr[2]).toFixed(3) : 0;
      homeAttr = top.at.slice();
      drawAttractor(top.at);
      return top;
    };

    function look() {
      const A = attract(), rows = balanceRows(), axes = modeAxes();
      const okDirs = allowedDirs();
      const haveOct = engineOcts().length > 0;
      const cand = [];
      const add = (j, kind, rod) => {
        const d = dv(xon, j), key = d.join(',');
        const c = { to: j, xyz: NODE[j].join(','), kind, rod, dir: key,
          shuttle: j === prevNode, okChirality: true, okBalance: true,
          okGen: true, okTetGate: null };      // null == NOT YET GATED
        if (kind === 'base') {
          // Unconstrained before the mode exists; the union of the survivors
          // while chirality is still being determined; the latched map after.
          c.okChirality = okDirs.has(key);
          const r = rows.find(x => x.id === key);
          c.okBalance = !!(r && r.ok);
        } else {
          c.okGen = axes.indexOf(ax(d)) >= 0;
        }
        const dist = prevNode === null ? null : P[prevNode].distanceTo(P[j]);
        c.angle = prevNode === null ? null
          : angleAt(P[prevNode].distanceTo(P[xon]), P[xon].distanceTo(P[j]), dist);
        c.turn = prevNode === null ? 'first' : cls(c.angle);
        c.dA = A ? +Math.hypot(P[j].x - A.at[0], P[j].y - A.at[1], P[j].z - A.at[2]).toFixed(3) : null;
        c.legal = !c.shuttle && c.okChirality && c.okBalance && c.okGen
                  && c.okTetGate !== false;
        cand.push(c);
      };
      for (const j of _baseNbr[xon]) add(j, 'base', null);
      for (let a2 = 0; a2 < AXN.length; a2++) {
        const j = SCOPT.get(xon + ':' + a2); if (j === undefined) continue;
        add(j, live.some(r => same(r, [xon, j])) ? 'sc-live' : 'sc-new', [xon, j]);
      }
      const sharing = sharingPairs();
      return { tick, xon, xyz: xon !== null ? NODE[xon].join(',') : null, prevNode,
        haveOct,
        identity, rods: live.length, rodList: live.map(r => r.join('-')),
        // CHIRALITY, as it actually stands: the mode (null until two shortcut
        // axes carry rods), the surviving candidate maps, and the latched one.
        mode: modeName(), chirality: chName,
        chCandidates: chCand ? chCand.slice() : null,
        chLatched: !!chSet,
        dormant: modeName() ? ['X', 'Y', 'Z'].find(a =>
          MODE_FAM[modeName()].indexOf(a) < 0) : null,
        octs: solids.octs.length, tets: solids.tets.length,
        tetsSharingAnEdge: sharing.length, sharingDetail: sharing,
        ring: ringOf().map(r => r.join('-')),
        balance: rows.map(r => r.id + '=' + r.n + (r.ok ? '*' : '')),
        block: blockState(), classLog: classLog.slice(),
        attractor: A ? { at: A.at.map(v => +v.toFixed(3)), kind: A.kind } : null,
        attrJump,                     // 0 = same hole held; large = target moved
        distA: A ? +Math.hypot(P[xon].x - A.at[0], P[xon].y - A.at[1], P[xon].z - A.at[2]).toFixed(3) : null,
        stats: dirStats(), lastMove: log.length ? log[log.length - 1] : null,
        implied: engineTets().concat(engineOcts().map(octVerts))
                   .flatMap(impliedEdgesOf).filter((v, i, a) => a.indexOf(v) === i),
        stalled, events, severed, refused, candidates: cand };
    }

    // ---- THE TET GATE ------------------------------------------------------
    // NO TET MAY ACTUALISE BEFORE THE OCT. The previous version materialised
    // the candidate rod and called detect() WITHOUT SOLVING; detect() measures
    // P[], so the probe rod's endpoints were still sqrt(2) apart and it found
    // nothing -- the gate passed everything, every tick of every run.
    //
    // The probe therefore has to go all the way round: add -> restate ->
    // settle -> detect. That is the only way to see the IMPLIED tets, which is
    // exactly the case that broke t7, and it is not optional: an implied tet
    // has no rod and so cannot be severed after the fact.
    //
    // Cost is one solve per untried candidate. `enqueue` is LRU-cached on the
    // pair set, so the undo is always a hit and repeated probes are free.
    async function gate(s) {
      // EVERY tick, oct or no oct. An earlier version skipped the probe once
      // the oct existed, on the grounds that "no tet before the oct" no longer
      // decides anything -- but the same probe carries the VACUUM answer, and
      // skipping it puts rods in without asking the solver first. The tet rule
      // has a scope; the vacuum does not.
      // PROBES MUST NOT PROPAGATE. Each one materialises a rod, solves, and
      // undoes it, so with the flux wave armed a single tick would emit a
      // dozen waves from rods that were never really laid. Silence them for
      // the probe run; the move that is actually taken emits its wave as
      // normal. `startWave` with no changed rods snaps rather than animating,
      // so the restore below leaves the drawing exactly where it was.
      // `quiet`, not `on:false` -- the master switch SNAPS, which would kill
      // any wave still in flight from the previous move.
      const fluxWas = window._FLUX ? window._FLUX().quiet : false;
      if (window._FLUX) window._FLUX({ quiet: true });
      const probes = s.candidates.filter(c => c.kind === 'sc-new' && c.legal);
      for (const c of probes) {
        const nk = scKeyOf(c.rod[0], c.rod[1]);
        if (!nk) { c.okTetGate = false; c.legal = false; continue; }
        active.set(nk[0], nk[1]);
        restate(true); await settle(); detect();
        c.wouldTets = solids.tets.length;
        c.wouldOcts = solids.octs.length;
        c.wouldShare = sharingPairs().length;
        c.wouldVacuum = legal(resid);
        active.delete(nk[0]);
        c.okTetGate = !(!s.haveOct && c.wouldTets > 0);
        // THE VACUUM ANSWER IS BINDING. The probe already asked it; choosing a
        // move it refuses adds a rod only to roll it back, spending the tick
        // and displacing the xon with nothing built. Measured at t7: 63 came
        // back vac=false, was chosen over 133 (vac=true) on distance alone,
        // and the xon walked out to (-2,0,8) with 5 rods and no oct.
        c.legal = c.legal && c.okTetGate && c.wouldVacuum;
      }
      // put the board back exactly as it was, then re-read it
      restate(true); await settle(); detect();
      if (window._FLUX) window._FLUX({ quiet: fluxWas });
      s.gated = true;
      // Tag every candidate with the odds r5 would draw it at, so the panel can
      // show them before the click. `narrow` is defined below -- hoisted, and
      // only ever called after the gate has settled legality.
      const n = narrow(s);
      s.poolSize = n.use ? n.use.length : 0;
      s.narrowWhy = n.why || n.stop;
      return s;
    }

    async function lookGated() { return gate(look()); }

    // The rule cascade, stated ONCE. `narrow` runs rules 3-5 down to the pool
    // r5 samples from and tags each survivor with its probability; `choose`
    // draws from it. The panel calls `narrow` too, so the odds it shows you
    // before you click are the odds actually used -- not a UI re-derivation
    // that could drift from the rule.
    function narrow(s) {
      const legal = s.candidates.filter(c => c.legal);
      if (!legal.length) return { stop: 'no legal move (rules 1/2/gen/tet-gate)' };
      // r3, in two stages: the BLOCK says which classes may still be spent in
      // this group of three, then the PREFERENCE picks among them. The
      // preference is a tie-break inside the block's freedom -- it can never
      // widen it.
      // R3, NUCLEON IDENTITY BLOCK -- composition only. Rolling group of three
      // turns: proton 2x90 + 1x60, neutron 2x60 + 1x90, order free. The
      // PREFERENCE that used to live here has moved to R5.
      const b = s.block;
      let pool = legal.filter(c => b.need.includes(c.turn) || c.turn === 'first');
      let why = 'r3 block permits ' + b.need.join('/');
      if (!pool.length) {
        if (identity !== 'neutron') return { stop: 'identity block unsatisfiable',
                                             need: b.need, legal };
        pool = legal; why = 'r3 BROKEN (neutron may, to keep r1+r2)';
      }
      // R4, THE ATTRACTOR: NEAREST WINS. Runs BEFORE prefer-shortcuts as of
      // 6 Aug. As a filter behind prefer-shortcuts it was toothless -- measured
      // at t12 of a YZ proton, the block forced a 90, prefer-shortcuts had
      // already discarded every base move, and the attractor was left ranking
      // two shortcuts that both led 0.567 further out.
      //
      // Chance decides ONLY exact ties. An earlier build sampled the whole pool
      // with weights 1/(rank+1); that was far too broad -- the nearest option
      // held only 48% and one draw took dA 1.36 over 0.947, walking the xon
      // from 0.645 to 1.249. Weighting the pool does not break ties in R4, it
      // overrides R4. Unseeded on purpose: the ask is variation across runs.
      pool.sort((x, y) => (x.dA ?? 9e9) - (y.dA ?? 9e9));
      const best = pool[0].dA;
      const near = pool.filter(c => c.dA === best
        || (c.dA !== null && best !== null && Math.abs(c.dA - best) < 1e-9));
      why += '; r4 nearest ' + (best === null ? 'n/a' : best)
           + (near.length > 1 ? ' (' + near.length + ' tied)' : '');

      // R5, IDENTITY PREFERENCE -- proton prefers 90, neutron prefers 60. The
      // LAST word, among moves already tied for nearest, so it can never cost
      // distance. Naturally vacuous when R3 has already forced a single class,
      // since every survivor is then that class anyway.
      const want = near.filter(c => c.turn === b.pref);
      const use = want.length ? want : near;
      if (want.length && want.length !== near.length)
        why += '; r5 prefers ' + b.pref;
      use.forEach(c => { c.pPick = 1 / use.length; });
      return { use, w: use.map(() => 1), tot: use.length, why };
    }

    function choose(s) {
      const n = narrow(s);
      if (n.stop) return n;
      let r = Math.random() * n.tot, k = 0;
      while (k < n.use.length - 1 && (r -= n.w[k]) > 0) k++;
      const pick = n.use[k];
      return { pick, why: n.why + (n.use.length > 1
        ? '; drew 1 of ' + n.use.length + ' tied' : '') };
    }

    async function start(opt) {
      opt = opt || {};
      identity = opt.identity === 'neutron' ? 'neutron' : 'proton';
      // Nothing is drawn at birth. A nucleon spawns with no mode, therefore no
      // dormant axis, therefore no chirality -- all eight base senses are open
      // until the lattice says otherwise.
      chSet = null; chName = null; chCand = null;
      _sync(); active.clear();
      live = []; prevNode = null; tick = 0; events = 0; baseHops = 0;
      refused = 0; severed = 0; stalled = null; homeOct = null;
      homeAttr = null; attrJump = null; chJournal.length = 0;
      baseCount.clear(); scCount.clear(); classLog.length = 0; log.length = 0;
      history.length = 0;
      xon = KEY.get(LCENTER.join(','));
      if (xon === undefined) { let bd = Infinity;
        for (let i = 0; i < NODE.length; i++) {
          const d = Math.hypot(...[0, 1, 2].map(k => NODE[i][k] - LCENTER[k]));
          if (d < bd) { bd = d; xon = i; } } }
      wake = [xon];
      restate(true); await settle(); detect(); drawXon(xon, wake);
      return look();
    }

    const snap = () => ({ xon, prevNode, tick, events, baseHops, refused, severed,
      stalled, live: live.map(r => r.slice()), wake: wake.slice(),
      homeOct: homeOct ? homeOct.slice() : null,
      homeAttr: homeAttr ? homeAttr.slice() : null, attrJump,
      chSet, chName, chCand: chCand ? chCand.slice() : null,
      chJournalLen: chJournal.length,
      baseCount: new Map(baseCount), scCount: new Map(scCount),
      classLog: classLog.slice(), logLen: log.length, active: new Map(active) });

    async function back() {
      const h = history.pop();
      if (!h) return { none: true, state: look() };
      xon = h.xon; prevNode = h.prevNode; tick = h.tick; events = h.events;
      baseHops = h.baseHops; refused = h.refused; severed = h.severed;
      stalled = h.stalled; live = h.live; wake = h.wake; homeOct = h.homeOct;
      homeAttr = h.homeAttr; attrJump = h.attrJump;
      chSet = h.chSet; chName = h.chName; chCand = h.chCand;
      chJournal.length = h.chJournalLen;
      baseCount.clear(); h.baseCount.forEach((v, k2) => baseCount.set(k2, v));
      scCount.clear(); h.scCount.forEach((v, k2) => scCount.set(k2, v));
      classLog.length = 0; h.classLog.forEach(c => classLog.push(c));
      log.length = h.logLen;
      active.clear(); h.active.forEach((v, k2) => active.set(k2, v));
      freezeOff(); restate(true); await settle(); detect(); drawXon(xon, wake);
      return { state: look() };
    }

    // `pre` is a gated state already computed for THIS tick -- the panel shows
    // the candidate table before you click, and passing it back guarantees the
    // automaton acts on the very table you were shown, not a re-derived one
    // (and saves a second round of probe solves).
    async function step(forceTo, pre) {
      if (stalled) return { stalled, state: look() };
      const s = (pre && pre.gated && pre.tick === tick) ? pre : await lookGated();
      const ch = forceTo !== undefined
        ? { pick: s.candidates.find(c => c.to === forceTo), why: 'manual' }
        : choose(s);
      if (!ch.pick) { stalled = ch.stop || 'no such move'; return { stalled, detail: ch, state: look() }; }
      history.push(snap());          // only once a move is certain to be applied
      const pick = ch.pick, from = xon;
      if (pick.turn === 60 || pick.turn === 90) classLog.push(pick.turn);
      const d = dv(xon, pick.to);
      if (pick.kind === 'base') { baseHops++;
        baseCount.set(d.join(','), (baseCount.get(d.join(',')) || 0) + 1); }
      else scCount.set(d.join(','), (scCount.get(d.join(',')) || 0) + 1);
      let made = false;
      if (pick.kind === 'sc-new') { const nk = scKeyOf(xon, pick.to);
        if (nk) { active.set(nk[0], nk[1]); live = live.concat([pick.rod]);
                  made = true; events++; } }
      prevNode = xon; xon = pick.to; tick++;
      wake.push(xon); if (wake.length > 40) wake.shift();
      restate(true); await settle(); detect(); drawXon(xon, wake);
      const ok = legal(resid);
      // Chirality is settled AFTER the rod is in and the vacuum has spoken --
      // a refused rod never fixed a mode, so it must not fix a chirality.
      if (ok) chUpdate(d.join(','), pick.kind === 'base');
      if (!ok && made) { const nk = scKeyOf(from, pick.to);
        if (nk) active.delete(nk[0]);
        live = live.filter(r => !same(r, pick.rod));
        refused++; freezeOff(); restate(true); await settle(); detect(); }

      // LATCH THE HOME OCT the moment one appears. `solids.octs[0]` stops
      // meaning "ours" as soon as a second one exists, and severance protects
      // the first.
      if (!homeOct && solids.octs.length) homeOct = solids.octs[0].slice();

      // SEVERANCE, against the ENGINE's tets. Two tets may not share an edge:
      // the pair raises a second oct. The old tet dies -- and the rod that goes
      // is the one NOT on the home equator, since cutting a ring rod would
      // break the oct itself.
      //
      // THIS CANNOT ALWAYS SUCCEED, and the failure is not a bug in the rule.
      // A tet whose edges are IMPLIED owns no rod, so there is nothing to cut;
      // `unsevered` records exactly that case rather than reporting a clean
      // pass. Prevention is the gate's job, not severance's.
      const cut = [], unsevered = [];
      if (made && ok) {
        const tets = engineTets();
        const ring = ringOf();
        const onRing = r => ring.some(q => same(q, r));
        const isNew = vs => vs.includes(from) && vs.includes(pick.to);
        for (const nt of tets.filter(isNew)) for (const ot of tets) {
          if (ot === nt) continue;
          if (ot.filter(v => nt.includes(v)).length < 2) continue;
          const cands = rodsIn(ot).filter(r => !onRing(r) && !same(r, [from, pick.to]));
          const kill = cands[0];
          if (!kill) { unsevered.push({ tet: ot.join(','),
            rodsOwned: rodsIn(ot).map(r => r.join('-')),
            implied: impliedEdgesOf(ot) }); continue; }
          const k = scKeyOf(kill[0], kill[1]);
          if (k) active.delete(k[0]);
          live = live.filter(r => !same(r, kill));
          cut.push(kill.join('-')); severed++;
        }
        if (cut.length) { restate(true); await settle(); detect(); }
      }
      const st = look();
      log.push({ tick, mv: from + '>' + pick.to, from, to: pick.to, dir: pick.dir,
        kind: pick.kind, turn: pick.turn,
        made, vacuumOK: ok, cut, unsevered, why: ch.why, octs: st.octs, tets: st.tets,
        sharing: st.tetsSharingAnEdge, rods: st.rods, distA: st.distA });
      st.lastMove = log[log.length - 1];    // look() ran before the push
      return { applied: st.lastMove, state: st };
    }
    return { start, step, back, look, lookGated, sheet: () => log.slice(),
             chirality: () => chJournal.slice(), depth: () => history.length };
  })();

  // ==========================================================================
  // THE STEPPER PANEL -- one tick per click, with the whole decision on screen.
  //
  // Built here rather than in the engine because it knows what a quark turn is.
  // It drives `_NSTEP` and nothing else, so what it shows is what the automaton
  // used: the candidate table is the very object `choose()` ranked, not a
  // re-derivation of it.
  // ==========================================================================
  (function stepperPanel() {
    if (!document.body || document.getElementById('nstep')) return;
    const css = document.createElement('style');
    css.textContent = `
      /* RIGHT side, below the size stepper. It used to sit at left:12px/top:12px,
         directly over #chip and the #stats dropdown that opens under it. */
      #nstep{position:fixed;right:12px;top:calc(66px + env(safe-area-inset-top));
             width:330px;z-index:6;
             background:#0d1219;border:1px solid #1f2937;border-radius:13px;
             font:11px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;
             color:#8fa3ba;overflow:hidden;
             max-height:calc(100vh - 150px - env(safe-area-inset-top));
             display:flex;flex-direction:column}
      #nshead{height:40px;display:flex;align-items:center;gap:8px;padding:0 12px;
              font-size:12px;color:#8fa3ba;cursor:pointer;flex:none}
      #nshead b{color:#c8d0dc;font-weight:600}
      #nscaret{margin-left:auto;font-size:9px;transition:transform .18s}
      #nstep.open #nscaret{transform:rotate(180deg)}
      #nsbody{display:none;padding:0 12px 12px;overflow:auto}
      #nstep.open #nsbody{display:block}
      .nsrow{display:flex;gap:6px;margin-bottom:7px}
      .nsbtn{flex:1;height:34px;display:flex;align-items:center;justify-content:center;
             border-radius:9px;border:1.5px solid #2a3543;background:#111721;
             color:#93a6bb;cursor:pointer;font-size:11.5px;font-weight:600;
             letter-spacing:.03em;user-select:none}
      .nsbtn:active{transform:scale(.985)}
      .nsbtn.on{background:#12352a;border-color:#3f9e77;color:#7fd4a8}
      .nsbtn.go{background:#12352a;border-color:#3f9e77;color:#7fd4a8}
      .nsbtn.busy{opacity:.45;pointer-events:none}
      .nsbtn.dead{background:#301a1a;border-color:#8a3f3f;color:#ff8f8f;
                  pointer-events:none}
      .nssec{margin:9px 0 3px;color:#4d5b6d;font-size:9.5px;letter-spacing:.09em;
             text-transform:uppercase}
      .nsk{color:#5d6e85}  .nsv{color:#c8d0dc}
      .nsg{color:#7fd4a8}  .nsr{color:#ff6b6b}  .nsy{color:#e0b860}
      .nsdim{color:#3d4855}
      table.nst{width:100%;border-collapse:collapse;font-size:10px}
      table.nst td{padding:1px 3px 1px 0;white-space:nowrap}
      table.nst td.n{text-align:right;font-variant-numeric:tabular-nums}
      .nsbar{display:inline-block;height:7px;background:#2f7d5f;border-radius:2px;
             vertical-align:middle}`;
    document.head.appendChild(css);

    const box = document.createElement('div');
    box.id = 'nstep';          // collapsed by default; click the header to open
    box.innerHTML = '<div id="nshead"><b>nucleon stepper</b>'
      + '<span id="nstick" class="nsdim">idle</span>'
      + '<span id="nscaret">&#9650;</span></div><div id="nsbody"></div>';
    document.body.appendChild(box);
    document.getElementById('nshead').onclick = () => box.classList.toggle('open');

    let ident = 'proton', busy = false, gated = null, depth = 0;
    const $n = id => document.getElementById(id);
    const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;');

    // Why a candidate is out, in the order the cascade would reject it. One
    // reason, the first that applies -- listing all four would hide which rule
    // is actually doing the work.
    const veto = c => c.shuttle ? 'shuttle'
      : !c.okChirality ? 'chirality'
      : !c.okGen ? 'generation'
      : !c.okBalance ? 'balance'
      : c.okTetGate === false ? 'TET GATE (' + c.wouldTets + 't/' + c.wouldOcts + 'o)'
      : c.wouldVacuum === false ? 'vacuum'
      : null;

    const bar = (n, max) => '<span class="nsbar" style="width:'
      + Math.round(46 * (max ? n / max : 0)) + 'px"></span>';

    function render() {
      const s = gated, body = $n('nsbody');
      if (!s) { body.innerHTML = '<div class="nsdim" style="padding:6px 0 2px">'
        + 'press <b>start</b> to spawn a nucleon at the lattice centre.</div>'
        + controls(); wire(); return; }
      const st = s.stats, m = s.lastMove;
      const mx = Math.max(1, ...st.base.map(r => r.n), ...st.sc.map(r => r.n));
      const legal = s.candidates.filter(c => c.legal);

      $n('nstick').textContent = 't' + s.tick + (s.stalled ? ' · STALLED' : '');
      $n('nstick').className = s.stalled ? 'nsr' : 'nsdim';

      body.innerHTML = controls()
        + '<div class="nssec">board</div>'
        + '<span class="nsk">at</span> <span class="nsv">' + s.xon + '</span>'
        + ' <span class="nsdim">(' + s.xyz + ')</span>'
        + ' &nbsp;<span class="nsk">dA</span> <span class="nsv">' + s.distA + '</span>'
        // attractor drift: 0 means the same hole is still the target. Anything
        // large means the goal moved, which is the failure this tracks.
        + ' &nbsp;<span class="nsk">target moved</span> <span class="'
          + (s.attrJump > 0.35 ? 'nsr' : s.attrJump ? 'nsy' : 'nsg') + '">'
          + (s.attrJump === null ? '—' : s.attrJump) + '</span><br>'
        + '<span class="nsk">rods</span> <span class="nsv">' + s.rods + '</span>'
        + ' &nbsp;<span class="nsk">octs</span> <span class="'
          + (s.octs > 1 ? 'nsr' : s.octs === 1 ? 'nsg' : 'nsv') + '">' + s.octs + '</span>'
        + ' &nbsp;<span class="nsk">tets</span> <span class="nsv">' + s.tets + '</span>'
        + ' &nbsp;<span class="nsk">sharing</span> <span class="'
          + (s.tetsSharingAnEdge ? 'nsr' : 'nsv') + '">' + s.tetsSharingAnEdge + '</span>'
        + ' &nbsp;<span class="nsk">implied</span> <span class="'
          + (s.implied.length ? 'nsy' : 'nsv') + '">' + s.implied.length + '</span>'
        + (s.implied.length ? ' <span class="nsdim">' + esc(s.implied.join(' ')) + '</span>' : '')
        // MODE FIRST, THEN CHIRALITY. Until two shortcut axes carry rods there
        // is no dormant axis, so there is no "toward" and no chirality to name.
        + '<br><span class="nsk">mode</span> <span class="'
          + (s.mode ? 'nsg' : 'nsdim') + '">' + (s.mode || 'not yet fixed') + '</span>'
        + (s.dormant ? ' <span class="nsdim">(dormant ' + s.dormant
            + ', arrow along it)</span>' : '')
        + '<br><span class="nsk">chirality</span> '
        + (s.chLatched
            ? '<span class="nsg">' + s.chirality + ' latched</span>'
            : s.chCandidates
              ? '<span class="nsy">undecided</span> <span class="nsv">'
                + s.chCandidates.join(' ') + '</span>'
              : '<span class="nsdim">not yet declared (needs the mode)</span>')

        + '<div class="nssec">turns &nbsp;·&nbsp; ' + s.identity + ' = '
          + (s.identity === 'proton' ? '2x90 + 1x60 (uud)' : '2x60 + 1x90 (udd)')
          + ' per 3, order free</div>'
        + '<span class="nsk">60</span> <span class="nsv">' + st.turns.n60 + '</span>'
        + ' &nbsp;<span class="nsk">90</span> <span class="nsv">' + st.turns.n90 + '</span>'
        + ' &nbsp;<span class="nsk">ratio 90</span> <span class="nsv">'
          + (s.block.ratio === null ? '—' : s.block.ratio.toFixed(3)) + '</span>'
        + ' <span class="nsdim">(target ' + (s.identity === 'proton' ? '.667' : '.333')
          + ')</span><br>'
        + '<span class="nsk">block</span> '
          + [0, 1, 2].map(i => { const c = s.block.spent[i];
              return '<span class="' + (c ? 'nsv' : 'nsdim') + '">'
                + (c ? (c === 90 ? 'u' : 'd') + c : '&middot;') + '</span>'; }).join(' ')
        + ' &nbsp;<span class="nsk">permits</span> <span class="nsy">'
          + (s.block.need.join('/') || '—') + '</span>'
        + (s.block.need.length > 1
            ? ' <span class="nsdim">&rarr; prefers ' + s.block.pref + '</span>'
            : ' <span class="nsdim">forced</span>')

        + '<div class="nssec">base directions &nbsp;·&nbsp; balance binds these</div>'
        + '<table class="nst">' + st.base.map(r => '<tr>'
            + '<td class="' + (r.allowed ? 'nsv' : 'nsdim') + '">' + r.id + '</td>'
            + '<td class="n ' + (r.allowed ? 'nsv' : 'nsdim') + '">' + r.n + '</td>'
            + '<td>' + (r.allowed ? bar(r.n, mx) : '') + '</td>'
            + '<td class="' + (r.allowed ? (r.atMin ? 'nsg' : 'nsdim') : 'nsdim') + '">'
              + (r.allowed ? (r.atMin ? 'spendable' : 'ahead') : 'chirality') + '</td>'
          + '</tr>').join('') + '</table>'

        + '<div class="nssec">shortcut senses &nbsp;·&nbsp; balance does NOT bind these</div>'
        + '<table class="nst">' + st.sc.map(r => '<tr>'
            + '<td class="' + (r.allowed ? 'nsv' : 'nsdim') + '">' + r.id + '</td>'
            + '<td class="n ' + (r.allowed ? 'nsv' : 'nsdim') + '">' + r.n + '</td>'
            + '<td>' + (r.allowed ? bar(r.n, mx) : '') + '</td>'
            + '<td class="nsdim">' + (r.allowed ? '' : 'dormant axis') + '</td>'
          + '</tr>').join('') + '</table>'

        + (m ? '<div class="nssec">last move</div>'
            + '<span class="nsv">' + m.mv + '</span> <span class="nsdim">' + m.dir + '</span>'
            + ' <span class="nsk">' + m.kind + '</span> '
            + '<span class="nsv">' + m.turn + (m.turn === 60 || m.turn === 90 ? '&deg;' : '') + '</span>'
            + (m.made ? ' <span class="nsg">rod laid</span>' : '')
            + (m.vacuumOK ? '' : ' <span class="nsr">VACUUM REFUSED</span>')
            + (m.cut.length ? ' <span class="nsy">cut ' + m.cut.join(' ') + '</span>' : '')
            + (m.unsevered && m.unsevered.length
                ? ' <span class="nsr">UNSEVERABLE (implied tet)</span>' : '')
            + '<br><span class="nsdim">' + esc(m.why || '') + '</span>' : '')

        + '<div class="nssec">candidates &nbsp;·&nbsp; ' + legal.length + ' legal of '
          + s.candidates.length + ' &nbsp;·&nbsp; ' + (s.poolSize || 0)
          + ' in the draw</div>'
        + '<div class="nsdim" style="margin-bottom:2px">' + esc(s.narrowWhy || '') + '</div>'
        + '<table class="nst">' + s.candidates.slice()
            .sort((a, b) => (a.legal === b.legal) ? (a.dA ?? 9e9) - (b.dA ?? 9e9)
                                                  : (a.legal ? -1 : 1))
            .map(c => { const v = veto(c); return '<tr>'
              + '<td class="' + (c.legal ? 'nsg' : 'nsdim') + '">' + (c.legal ? '&#10003;' : '&#215;') + '</td>'
              + '<td class="' + (c.legal ? 'nsv' : 'nsdim') + '">' + c.to + '</td>'
              + '<td class="' + (c.legal ? 'nsv' : 'nsdim') + '">' + c.dir + '</td>'
              + '<td class="nsk">' + c.kind.replace('sc-', '') + '</td>'
              + '<td class="' + (c.legal ? 'nsv' : 'nsdim') + '">' + c.turn + '</td>'
              + '<td class="n nsdim">' + (c.dA ?? '—') + '</td>'
              + '<td class="n ' + (c.pPick ? 'nsy' : 'nsdim') + '">'
                + (c.pPick ? (100 * c.pPick).toFixed(0) + '%' : '') + '</td>'
              + '<td class="' + (v && v.indexOf('TET') === 0 ? 'nsr' : 'nsdim') + '">'
                + (v || (c.wouldOcts ? '<span class="nsg">+' + c.wouldOcts + ' OCT</span>' : ''))
              + '</td></tr>'; }).join('') + '</table>'
        + (s.stalled ? '<div class="nssec nsr">stalled</div><span class="nsr">'
            + esc(s.stalled) + '</span>' : '');
      wire();
    }

    // No L/R buttons: chirality is DISCOVERED from the mode plus the base
    // traversals, so there is nothing here to choose. The readout below shows
    // the determination as it happens.
    const controls = () => '<div class="nsrow">'
        + '<div class="nsbtn' + (ident === 'proton' ? ' on' : '') + '" id="ns-p">proton</div>'
        + '<div class="nsbtn' + (ident === 'neutron' ? ' on' : '') + '" id="ns-n">neutron</div>'
      + '</div><div class="nsrow">'
        + '<div class="nsbtn" id="ns-start">start</div>'
        + '<div class="nsbtn' + (busy || !depth ? ' busy' : '') + '" id="ns-back">'
          + '&#9664; back</div>'
        + '<div class="nsbtn ' + (busy ? 'busy' : (gated && gated.stalled ? 'dead' : 'go'))
          + '" id="ns-next" style="flex:1.4">'
          + (busy ? 'solving...' : (gated && gated.stalled ? 'stalled' : 'next &#9654;')) + '</div>'
      + '</div>';

    function wire() {
      const on = (id, fn) => { const e = $n(id); if (e) e.onclick = fn; };
      on('ns-p', () => { ident = 'proton'; render(); });
      on('ns-n', () => { ident = 'neutron'; render(); });
      on('ns-start', async () => {
        if (busy) return; busy = true; render();
        // A registered loop drives _NSTEP too, and two drivers on one piece of
        // shared state is what made hand-stepping look like it took twenty
        // turns at once. Stand the loop down first.
        if (window._loopNow && window._loopNow() !== 'none' && window._setLoopNone)
          window._setLoopNone();
        try { await window._NSTEP.start({ identity: ident });
              gated = await window._NSTEP.lookGated(); }
        finally { depth = window._NSTEP.depth(); busy = false; render(); }
      });
      on('ns-next', async () => {
        if (busy || !gated || gated.stalled) return; busy = true; render();
        try { const r = await window._NSTEP.step(undefined, gated);
              gated = await window._NSTEP.lookGated();
              gated.lastMove = r.applied || gated.lastMove;
              if (r.stalled) gated.stalled = r.stalled; }
        finally { depth = window._NSTEP.depth(); busy = false; render(); }
      });
      // BACK re-gates the restored board rather than replaying a stored table,
      // so the candidate list you get is measured against the board as it now
      // stands. Stepping forward again re-samples r5 and may take a different
      // move -- that is the point of having it.
      on('ns-back', async () => {
        if (busy || !depth) return; busy = true; render();
        try { await window._NSTEP.back(); gated = await window._NSTEP.lookGated(); }
        finally { depth = window._NSTEP.depth(); busy = false; render(); }
      });
    }
    render();
    window._NPANEL = { render, state: () => gated };
  })();

  window._XONPROTON = async function (opt) {
    opt = opt || {};
    _sync();
    const ticks = opt.ticks || 240;
    const pRing = opt.pRing === undefined ? 2 / 3 : opt.pRing;
    // freeBuild: run the 90/60 ratio from tick 0 with NO ring scaffolding --
    // no forced d1/d2/-d1/-d2 sequence, no build-before-ratio priority. The
    // question it answers is whether the octahedron condenses out of the turn
    // statistics alone or only because the CA was told to build one.
    const freeBuild = !!opt.freeBuild;
    // legacy: the 1.0 REFERENCE BUILDS. freeBuild and attractor were always
    // opt-in, but no-shuttling and direction balance were added unconditionally
    // and leaked into them, so the "untouched controls" were quietly running
    // two rules the originals never had. This restores them.
    const legacy = !!opt.legacy;
    // ATTRACTOR. The xon is drawn toward the deepest hole -- the point inside
    // the lattice furthest from any node. This does NOT touch the 90/60 draw:
    // the class is chosen first by the ratio, and the attractor only picks the
    // DIRECTION within that class. So it cannot shift the quark content; all it
    // does is stop the xon walking off to the edge of the sim.
    const attractor = !!opt.attractor;
    // The search must span the lattice INTERIOR, not a ball around the xon: a
    // ball that travels with the xon loses sight of the well the xon dug as
    // soon as it strays, and then nothing pulls it home. Measured -- a
    // xon-centred ball let it wander 3.7 from home and never close a ring.
    // 4.0 reaches the lattice boundary, where depth grows without bound and
    // every hole is discarded as a clamp artifact -- measured, it returns none
    // at all. 3.2 stays inside.
    const attrR = opt.attractorRadius || 3.2;
    const attrStep = opt.attractorStep || 0.15;
    const attrSeeds = opt.attractorSeeds || 24;
    const rnd = (() => { let s = (opt.seed === undefined
      ? (Math.random() * 2147483647) | 0 : opt.seed) | 0;
      return () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; }; })();

    const dvec = (a, b) => [0, 1, 2].map(k => NODE[b][k] - NODE[a][k]);
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const rodSame = (x, y) => (x[0] === y[0] && x[1] === y[1]) || (x[0] === y[1] && x[1] === y[0]);
    const axOf = d => (d[0] ? 'X' : (d[1] ? 'Y' : 'Z'));

    // DIRECTION BALANCE. The xon wants to traverse every direction of its
    // GENERATION equally, sign-agnostically -- forward and back are the same
    // direction. Gen 1 is 4 base <111> axes plus the 2 shortcut axes of one
    // flux mode, so the balance target is exactly 6 directions. The third
    // shortcut axis exerts no pull at all: it is not part of this generation.
    //
    // This is what replaces the plane rule. A closed square needs exactly two
    // rods on each of its two axes, so balancing the selected pair IS
    // squareness -- arrived at rather than dictated.
    const dirKey = (d) => {                  // canonical, sign-agnostic axis id
      let s = 0;
      for (const v of d) { if (v) { s = v; break; } }
      return (s < 0 ? d.map(v => -v) : d).join(',');
    };
    const dirCount = new Map();
    const scAxes = [];      // the flux mode, fixed by the first two axes used
    // BALANCE IS MEASURED ABOUT THE ATTRACTOR, not just by direction. A move
    // has a SIDE: which half of the void its midpoint falls on, judged against
    // the first offset ever seen for that direction. Wanting each (direction,
    // side) bucket equally is what makes a square -- the two X edges of a ring
    // lie on opposite sides of its centre, and so do the two Y edges. Plain
    // direction counting cannot tell those apart, which is how the xon found
    // perfectly balanced limit cycles that built nothing.
    const dirRef = new Map();        // dirKey -> first offset from the attractor
    const sideCount = new Map();     // "dirKey|+" / "dirKey|-" -> count
    const sideOf = (k, off) => {
      const r = dirRef.get(k);
      if (!r) return '+';
      return (off[0] * r[0] + off[1] * r[1] + off[2] * r[2]) >= 0 ? '+' : '-';
    };

    // start at the lattice centre
    let xon = KEY.get(LCENTER.join(','));
    if (xon === undefined) { let bd = Infinity;
      for (let i = 0; i < NODE.length; i++) {
        const d = Math.hypot(...[0, 1, 2].map(k => NODE[i][k] - LCENTER[k]));
        if (d < bd) { bd = d; xon = i; } } }
    const home = xon;
    active.clear();
    let live = [], prevNode = null, lastDir = null;
    const ringDirs = [];            // the square's edge directions, in order

    const wake = [xon]; const turns = {};
    let events = 0, baseHops = 0, refused = 0, severed = 0, ringClosedAt = null, pending = null;
    const attrDepths = []; let wander = 0;   // how deep the well got, how far the xon strayed
    const fullPath = [], dirLog = [], kindLog = [], poolLog = [];
    // The plane rule is gone. It closed the ring 0 times in 6 at a 4-position
    // window and 1 in 4 at 5, against 6 in 14 with no rule at all -- it could
    // only ever cost closures, because a square's four corners are coplanar.
    // Direction balance replaces it.
    let forced = 0, debt90 = 0, debt60 = 0; const avail90 = [], avail60 = [];
    const octTicks = [], tetTicks = [];

    // ---- QUARKS: three tets on the faces of the octahedron -----------------
    // A face is two adjacent ring vertices A,B (already joined by a ring rod)
    // plus an apex T (already base-joined to both). A tet's two shortcuts are
    // OPPOSITE edges, so the second rod is forced to be T-E: each quark costs
    // exactly ONE flux event and REUSES a ring rod.
    //
    // Measured on a closed ring: exactly ONE placement per face (6 of 8 faces
    // here; the other two run off the lattice edge). Once actualised A-T-E is
    // equilateral, so creating a quark is a 60-degree turn at an apex --
    // precisely the third of moves the rule already spends visiting apexes.
    let ringNodes = null, apexes = null;
    const quarkRods = [];              // at most ONE lives at a time
    const faceVisits = new Set();      // which faces have carried a quark
    const learnOct = () => {
      ringNodes = [...new Set([].concat(...live))];
      apexes = [];
      for (let v = 0; v < NODE.length; v++) {
        if (ringNodes.includes(v)) continue;
        if (ringNodes.every(r => _baseSet.has(K(v, r)))) apexes.push(v);
      }
    };
    // adjacent ring pairs through a given vertex. Must look at RING rods only:
    // `live` also holds the quark rods once they exist, and those hang off the
    // solid rather than belonging to the square.
    const ringPairsAt = a => live
      .filter(r => !quarkRods.some(q => rodSame(q, r)))
      .filter(r => r[0] === a || r[1] === a)
      .map(r => (r[0] === a ? r[1] : r[0]));
    const quarkOptions = () => {                 // xon must be AT an apex
      if (!apexes || !apexes.includes(xon) || prevNode === null) return [];
      if (!ringNodes.includes(prevNode)) return [];
      const T = xon, A = prevNode, out = [];
      for (const B of ringPairsAt(A)) {
        for (let a = 0; a < AXN.length; a++) {
          const E = SCOPT.get(T + ':' + a); if (E === undefined) continue;
          if (ringNodes.includes(E) || apexes.includes(E)) continue;
          const rod = [T, E];
          if (live.some(r => rodSame(r, rod))) continue;
          if (!isTet([A, B, T, E])) continue;
          out.push({ to: E, kind: 'sc', rod, turn: 60, makes: true, quark: true });
        }
      }
      return out;
    };

    // rods must stay a connected path (or cycle) of at most 4, all in one mode
    const ringOK = rods => {
      if (rods.length > 4) return false;
      const ax = [...new Set(rods.map(r => axOf(dvec(r[0], r[1]))))];
      if (ax.length > 2) return false;                 // one flux mode only
      const deg = new Map();
      for (const r of rods) for (const v of r) deg.set(v, (deg.get(v) || 0) + 1);
      for (const d of deg.values()) if (d > 2) return false;   // no branching
      return true;
    };
    const isClosedRing = rods => {
      if (rods.length !== 4) return false;
      const deg = new Map();
      for (const r of rods) for (const v of r) deg.set(v, (deg.get(v) || 0) + 1);
      return deg.size === 4 && [...deg.values()].every(d => d === 2);
    };

    await runExperiment('proton — stationary, building an octahedron',
      Array.from({ length: ticks }, (_, k) => ({ k, label: 'tick ' + k })),
      () => {
        if (opt.alive && !opt.alive()) { pending = null; return; }
        let cand = [];                       // reassigned by the plane-rule filter
        const U = (a, b) => Math.abs(P[a].distanceTo(P[b]) - 1) < 1e-4;
        const R2 = (a, b) => Math.abs(P[a].distanceTo(P[b]) - Math.SQRT2) < 1e-4;
        // base hops: turn classified from the far pair, needs solved geometry
        for (const j of _baseNbr[xon]) {
          if (j === prevNode) continue;
          if (prevNode === null) { cand.push({ to: j, kind: 'base', turn: 90 }); continue; }
          if (U(prevNode, j)) cand.push({ to: j, kind: 'base', turn: 60 });
          else if (R2(prevNode, j)) cand.push({ to: j, kind: 'base', turn: 90 });
        }
        for (let a = 0; a < AXN.length; a++) {
          const j = SCOPT.get(xon + ':' + a); if (j === undefined || j === prevNode) continue;
          const rod = [xon, j];
          const already = live.some(r => rodSame(r, rod));
          if (already) {                                  // traversing a live rod
            if (prevNode === null) { cand.push({ to: j, kind: 'sc', rod, turn: 90 }); continue; }
            if (U(prevNode, j)) cand.push({ to: j, kind: 'sc', rod, turn: 60 });
            else if (R2(prevNode, j)) cand.push({ to: j, kind: 'sc', rod, turn: 90 });
            continue;
          }
          // CREATION: only as a 90-degree ring corner, AND only if it keeps the
          // path on ONE square. Consecutive orthogonality alone is not enough --
          // +X,+Y,+X,+Y satisfies it and walks off diagonally forever, which is
          // what the first build did: four rods, a path, no octahedron. The
          // first two edges DETERMINE the square, so the third must be -d1 and
          // the fourth -d2. Closure is then forced, not hoped for.
          if (!ringOK(live.concat([rod]))) continue;
          const nd = dvec(xon, j);
          const k = ringDirs.length;
          if (!freeBuild) {
            if (k === 1 && dot(ringDirs[0], nd) !== 0) continue;
            if (k === 2 && !(nd[0] === -ringDirs[0][0] && nd[1] === -ringDirs[0][1]
                                                       && nd[2] === -ringDirs[0][2])) continue;
            if (k === 3 && !(nd[0] === -ringDirs[1][0] && nd[1] === -ringDirs[1][1]
                                                       && nd[2] === -ringDirs[1][2])) continue;
            if (k >= 4) continue;
          }
          // Under freeBuild the k>=4 cap is redundant: ringOK already refuses a
          // fifth rod, so nothing here can exceed a four-rod ring either way.
          //
          // THE ANGLE IS THE ONE THAT EXISTS AFTER THE ROD MANIFESTS, not
          // before -- the same lesson the electron CA taught. Materialising
          // (xon,j) pulls them to unit distance, which can place prevNode and j
          // in one tet, making the turn 60 rather than 90. Judged in the
          // CURRENT geometry every creation looks like 90, so rods could only
          // ever be laid on a 90 draw: two-thirds of proton ticks but only one
          // third of neutron ticks, which is why the neutron built worse. Test
          // it combinatorially, never against P, because P still holds the
          // previous configuration.
          const post60 = prevNode !== null &&
            (_baseSet.has(K(prevNode, j)) || live.some(r => rodSame(r, [prevNode, j])));
          cand.push({ to: j, kind: 'sc', rod, turn: post60 ? 60 : 90, makes: true });
        }
        // NO SHUTTLING. a-b-a is forbidden: the xon may not step straight back
        // where it came from. This is what killed the old ping-pong, where a
        // dead-ended xon spent 1332 of 2000 ticks bouncing on base edges.
        if (!legacy && prevNode !== null) {
          const fwd = cand.filter(cc => cc.to !== prevNode);
          if (fwd.length) cand = fwd;
        }
        if (!cand.length) { pending = null; return; }
        // WHILE THE RING IS OPEN, WORK ON THE RING. The 2/3-1/3 ratio is the
        // steady state, not the build: 60-degree turns become available the
        // moment two rods are live (a shortcut plus two base edges is a unit
        // triangle -- a tet face), so without this the xon wanders off into
        // tet-shaped geometry and never closes the square. The apexes are not
        // part of a solid until the ring exists, so there is nothing there to
        // visit yet.
        const closed = isClosedRing(live) || ringClosedAt !== null;
        if (closed && !ringNodes) learnOct();
        if (closed) for (const q of quarkOptions()) cand.push(q);
        let pool;
        if (!closed && !freeBuild) {
          const build = cand.filter(c => c.makes);          // extend the ring
          const onRing = cand.filter(c => c.kind === 'sc'); // or ride what exists
          pool = build.length ? build : (onRing.length ? onRing : cand);
        } else {
          // a quark placement is taken whenever one is on offer -- there is
          // exactly one per face, so these are rare and must not be passed over
          // THE RATIO IS A PROBABILITY, and it governs everything.
          //
          // Draw the class first -- 90 with probability pRing, else 60 -- and
          // only then look at what is on offer within it. Quark placements are
          // 60-degree moves and are drawn from the 60 class like any other;
          // they do NOT bypass the draw. Letting them jump the queue was what
          // dragged the achieved ratio to 0.258.
          //
          // If the drawn class is empty we must still move, so the other class
          // is used and the substitution is COUNTED. That count is the honest
          // measure of whether the geometry can support the requested ratio at
          // all -- if forced substitutions are common, 2/3 is unreachable here
          // and no amount of drawing will produce it.
          const nine = cand.filter(c => c.turn === 90);
          const six = cand.filter(c => c.turn === 60);
          avail90.push(nine.length); avail60.push(six.length);
          // A forced substitution silently loses that draw forever, so the
          // achieved ratio can never reach the requested one. Carry the
          // shortfall as a DEBT and repay it the moment the class is available
          // again; the long-run ratio then converges on pRing exactly instead
          // of sagging below it.
          let want90;
          if (debt90 > 0 && nine.length) want90 = true;
          else if (debt60 > 0 && six.length) want90 = false;
          else want90 = rnd() < pRing;
          const first = want90 ? nine : six, other = want90 ? six : nine;
          if (first.length) { pool = first;
            if (want90 && debt90 > 0) debt90--; if (!want90 && debt60 > 0) debt60--; }
          else { pool = other.length ? other : cand; forced++;
            if (want90) debt90++; else debt60++; }
          // within the 60 class, a quark placement is preferred over a plain
          // apex hop -- the tet is the point of going there
          const q = pool.filter(c => c.quark);
          if (q.length) pool = q;
        }
        // DIRECTION BIAS. The class is already fixed by the draw above, so
        // choosing within `pool` cannot move the ratio by a single count. Among
        // the moves the ratio permits, take the one landing nearest the deepest
        // hole; ties go to chance as before.
        // The class is already fixed by the draw above, so nothing chosen here
        // can move the 90/60 statistics by a single count. Within that class:
        //   1. take the least-used direction of this generation  (balance)
        //   2. break ties by landing nearest the largest void     (attractor)
        //   3. break remaining ties by chance
        let AP = null;
        if (attractor) {
          const q0 = P[xon];
          const dh = window._DEEPHOLE({ radius: attrR, aroundRods: true,
            quiet: true, tieNear: [q0.x, q0.y, q0.z] });
          if (dh.top && dh.top.length) { AP = dh.top[0].at; attrDepths.push(dh.coveringRadius); }
          drawAttractor(AP);
        }
        let tied = null, bBal = Infinity, bDist = Infinity;
        for (const cc of pool) {
          const dv = dvec(xon, cc.to);
          // A shortcut outside the selected duple has NO pull: not this generation.
          const outOfMode = cc.kind === 'sc' && scAxes.length >= 2
            && scAxes.indexOf(axOf(dv)) < 0;
          const k = dirKey(dv);
          const q = P[cc.to], p0 = P[xon];
          let bal;
          // legacy: balance is inert, so every candidate ties here and the pick
          // falls through to a uniform draw -- the original 1.0 behaviour.
          if (legacy) bal = 0;
          else if (outOfMode) bal = Infinity;
          else if (AP) {
            const off = [(p0.x + q.x) / 2 - AP[0], (p0.y + q.y) / 2 - AP[1],
                         (p0.z + q.z) / 2 - AP[2]];
            bal = sideCount.get(k + '|' + sideOf(k, off)) || 0;
          } else bal = dirCount.get(k) || 0;      // no void yet: plain balance
          const dist = AP ? Math.hypot(q.x - AP[0], q.y - AP[1], q.z - AP[2]) : 0;
          if (bal < bBal - 1e-9 || (bal === bBal && dist < bDist - 1e-9)) {
            bBal = bal; bDist = dist; tied = [cc];
          } else if (bal === bBal && Math.abs(dist - bDist) < 1e-9) {
            tied.push(cc);
          }
        }
        // How much freedom did balance actually have this tick?
        poolLog.push({ c: cand.length, p: pool.length, t: tied ? tied.length : 0,
          spread: pool.length > 1 ? 1 : 0 });
        const h = (tied && tied.length) ? tied[Math.floor(rnd() * tied.length)]
                                        : pool[Math.floor(rnd() * pool.length)];
        pending = h;
        turns[h.turn] = (turns[h.turn] || 0) + 1;
        lastDir = dvec(xon, h.to); prevNode = xon;
        // Record the direction, and let the first two shortcut axes used fix
        // the flux mode. Nothing selects the generation up front -- it is
        // whatever the xon happens to traverse first, and then it is locked.
        const dk = dirKey(lastDir);
        if (!fullPath.length) fullPath.push(prevNode);
        fullPath.push(xon); dirLog.push(dk); kindLog.push(h.kind);
        dirCount.set(dk, (dirCount.get(dk) || 0) + 1);
        if (AP) {
          const a = P[prevNode], b = P[xon];
          const off = [(a.x + b.x) / 2 - AP[0], (a.y + b.y) / 2 - AP[1],
                       (a.z + b.z) / 2 - AP[2]];
          const sd = sideOf(dk, off);
          if (!dirRef.has(dk)) dirRef.set(dk, off);   // first offset defines the sides
          sideCount.set(dk + '|' + sd, (sideCount.get(dk + '|' + sd) || 0) + 1);
        }
        if (h.kind === 'sc') {
          const ax = axOf(lastDir);
          if (scAxes.indexOf(ax) < 0 && scAxes.length < 2) scAxes.push(ax);
        }
        xon = h.to; wake.push(xon); if (wake.length > 60) wake.shift();
        wander = Math.max(wander, Math.hypot(P[xon].x - P[home].x,
          P[xon].y - P[home].y, P[xon].z - P[home].z));
        drawXon(xon, wake);                           // xon moves BEFORE geometry
        if (h.kind === 'base') { baseHops++; return; }
        if (!h.makes) return;                          // traversal of a live rod
        if (h.quark) {
          // ONE TET AT A TIME. Making a new quark severs the old one, always.
          //
          // This replaces a conditional rule that severed only when the two
          // faces were adjacent in the same hemisphere -- which does prevent
          // the second octahedron (two quark rods from one apex on different
          // axes put their far ends at sqrt2, exactly detect()'s antipodal
          // distance, so a second oct closes) but needed 9 severances to hold
          // 3 quarks and dragged the 90-degree ratio down to 0.14.
          //
          // Unconditional is simpler and stronger: the proton is one oct and
          // ONE tet, and the tet moves around the faces. The three quarks are
          // realised in sequence rather than simultaneously.
          for (const q of quarkRods.slice()) {
            const ok = scKeyOf(q[0], q[1]); if (ok) active.delete(ok[0]);
            live = live.filter(r => r !== q);
            severed++;
          }
          quarkRods.length = 0;
          faceVisits.add(h.rod.slice().sort((a, b) => a - b).join(','));
        }
        const nk = scKeyOf(h.rod[0], h.rod[1]); if (nk) active.set(nk[0], nk[1]);
        live = live.concat([h.rod]); events++;
        if (h.quark) quarkRods.push(h.rod);       // hangs off the ring, not part of it
        else ringDirs.push(lastDir.slice());
      },
      (it) => {
        if (!pending) return { html: '<span style="color:#5d6e85">—</span>', skip: true };
        detect(); drawXon(xon, wake);
        if (!legal(resid)) {                           // vacuum refuses the rod
          refused++; freezeOff();
          if (pending.makes) { const nk = scKeyOf(pending.rod[0], pending.rod[1]);
            if (nk) active.delete(nk[0]); live = live.filter(r => r !== pending.rod); }
          return { html: '<span style="color:#ff5c5c">vacuum refused</span>' };
        }
        if (isClosedRing(live) && ringClosedAt === null) ringClosedAt = it.k;
        octTicks.push(solids.octs.length); tetTicks.push(solids.tets.length);
        return { html: `<span style="${pending.turn === 90 ? 'color:#7fd4a8' : 'color:#8fa3ba'}">`
          + `${pending.turn}°</span> `
          + (pending.quark ? '<b style="color:#ff9e6b">QUARK</b>'
                           : (pending.makes ? '<b>FLUX</b>' : pending.kind))
          + ` &nbsp; rods=${live.length} sc=${active.size}`
          + ` &nbsp; ${solids.octs.length ? '<b style="color:#ffd166">OCT</b>' : 'no oct'}`
          + ` &nbsp; tets=${solids.tets.length}` };
      },
      { onFreeze: () => freezeOff(),
        pace: () => (window._loopNow && window._loopNow() !== 'none'
                     && window._loopPace) ? window._loopPace() : 0 });

    const n90 = turns[90] || 0, n60 = turns[60] || 0;
    const out = { home: NODE[home], ticks, fluxEvents: events, baseHops, refused,
      ringClosedAtTick: ringClosedAt, rodsAtEnd: live.length,
      attractorOn: attractor, freeBuild, legacy,
      fluxMode: scAxes.join('') || null,
      dirBalance: [...dirCount.entries()].map(([k, v]) => k + ':' + v).join(' '),
      sideBalance: [...sideCount.entries()].map(([k, v]) => k + ':' + v).join(' '),
      path: fullPath.slice(),          // every node visited, in order
      pathDirs: dirLog.slice(),        // the direction key of every step
      pathKinds: kindLog.slice(),      // 'base' | 'sc' per step
      meanCand: +(poolLog.reduce((s, x) => s + x.c, 0) / Math.max(1, poolLog.length)).toFixed(2),
      meanPool: +(poolLog.reduce((s, x) => s + x.p, 0) / Math.max(1, poolLog.length)).toFixed(2),
      ticksWithNoChoice: poolLog.filter(x => x.p <= 1).length,
      ticksBalanceDecided: poolLog.filter(x => x.p > 1 && x.t < x.p).length,
      maxWanderFromHome: +wander.toFixed(3),
      meanWellDepth: attrDepths.length
        ? +(attrDepths.reduce((s, x) => s + x, 0) / attrDepths.length).toFixed(4) : null,
      maxWellDepth: attrDepths.length ? +Math.max(...attrDepths).toFixed(4) : null,
      quarksPlaced: severed + quarkRods.length,   // total placements over the run
      distinctFacesUsed: faceVisits.size,
      pRingRequested: pRing, forcedSubstitutions: forced,
      mean90Available: +(avail90.reduce((s, x) => s + x, 0) / Math.max(1, avail90.length)).toFixed(2),
      mean60Available: +(avail60.reduce((s, x) => s + x, 0) / Math.max(1, avail60.length)).toFixed(2),
      ticksWithNo90: avail90.filter(x => x === 0).length,
      ticksWithNo60: avail60.filter(x => x === 0).length,
      maxTets: Math.max(0, ...tetTicks),
      tetTickFraction: +(tetTicks.filter(x => x > 0).length
                         / Math.max(1, tetTicks.length)).toFixed(3),
      turn90: n90, turn60: n60,
      ratio90: +(n90 / Math.max(1, n90 + n60)).toFixed(3),
      octTickFraction: +(octTicks.filter(x => x > 0).length
                         / Math.max(1, octTicks.length)).toFixed(3),
      maxOcts: Math.max(0, ...octTicks) };
    window._XP = out;
    console.log('XONPROTON', JSON.stringify(out, null, 1));
    return out;
  };

  // ---- MANUAL CA HARNESS ---------------------------------------------------
  // The CA's rules, played BY HAND. Nothing in here scores, ranks or chooses:
  // it reports the state a player needs and applies the move they name. The
  // point is to test the RULES without a scorer bug being able to validate or
  // invalidate them -- every automated run so far has confounded the two.
  //
  //   await _MANUAL.start()        -> board state
  //   _MANUAL.look()               -> the same, any time
  //   await _MANUAL.move(node,why) -> commit a step, solve, render, record
  //   _MANUAL.sheet()              -> the traversal log
  window._MANUAL = (() => {
    let xon = null, prevNode = null, live = [], tick = 0, wake = [], lastWasSC = false;
    const dirCount = new Map(), sideCount = new Map(), dirRef = new Map();
    const log = [];
    const dv = (a, b) => [0, 1, 2].map(k => NODE[b][k] - NODE[a][k]);
    // RULE 2 counts TRAVEL SENSE, for base edges as well as shortcuts: (1,-1,1)
    // and (-1,1,-1) are different directions, not one axis traversed twice.
    // Collapsing them barred a legal move -- 168->200 was refused because
    // 238->206 had used the opposite sense of the same axis.
    const dkey = d => d.join(',');
    const ax = d => (d[0] ? 'X' : (d[1] ? 'Y' : 'Z'));
    const same = (x, y) => (x[0] === y[0] && x[1] === y[1]) || (x[0] === y[1] && x[1] === y[0]);

    // ---- RULE 2 bookkeeping -------------------------------------------------
    // Base directions are sense-AGNOSTIC (4 of them). Shortcut directions are
    // sense-SIGNED, so X+ and X- are different directions (6, or 4 once the
    // flux mode locks). 4 + 6 = the ten traversal directions of the lattice.
    const senseCount = new Map();
    const classLog = [];                        // 60/90 for each classified turn
    let identity = 'proton';                    // 'proton' | 'neutron'
    let chirality = 'A';
    let pooled = true;                          // shortcut senses pooled by axis
    let pooledFlat = false;                     // ...at floor(n/2), not n
    let scBalance = true;                       // do shortcuts count for rule 2?
    // CHIRALITY, the same two zero-sum sign sets the electron uses. Of the 16
    // sign patterns over the four base axes exactly two sum to zero, and they
    // are exact negations: A = {+a,-b,-c,+d}, B = -A. A xon has one for life,
    // so the opposite sense of any base axis is simply not a move it has.
    // The sense maps. A and B are the apolar zero-sum pair; L and R are the two
    // polarizations that can circulate an oct's apex squares -- red toward,
    // blue away, then green/yellow either way round.
    const CH_SETS = {
      A: ['1,1,1', '-1,-1,1', '-1,1,-1', '1,-1,-1'],
      B: ['-1,-1,-1', '1,1,-1', '1,-1,1', '-1,1,1'],
      L: ['1,1,1', '1,-1,1', '1,1,-1', '1,-1,-1'],
      R: ['1,1,1', '1,-1,1', '-1,-1,1', '-1,1,1'] };
    const chSet = () => new Set(CH_SETS[chirality] || CH_SETS.A);
    // RULE 2 identities. Base is a SIGNED direction, but chirality admits only
    // one sense per axis, so there are exactly four.
    //
    // Shortcuts are SIGNED TOO: X+ and X- are different directions, not one
    // axis traversed twice. That makes the balance set 4 base + 4 in-mode
    // shortcut senses = EIGHT, so a balanced run spends shortcuts and base
    // edges in equal measure -- e.g. Z a b X c d X Z. Pooling them by axis was
    // tried and it strangles the ring: after one X+ the whole X axis reads as
    // spent, so the X- that CLOSES the equator is barred and the nucleon can
    // never finish its oct. Set opt.pooled to compare.
    const dirId = (d, isSC) => !isSC ? d.join(',')
      : (pooled ? ax(d) : ax(d) + ((d[0] || d[1] || d[2]) > 0 ? '+' : '-'));
    // The balance set is THIS GENERATION's directions: the four chirality-
    // allowed base senses plus the two in-mode shortcut AXES. The dormant axis
    // is excluded outright -- it is not a direction this particle has, so
    // leaving it at zero would deadlock the minimum forever.
    // The locked generation's two shortcut axes -- once two distinct axes carry
    // rods the third is barred outright, which IS the generation lock.
    const modeAxes = () => {
      const m = [...new Set(live.map(r => ax(dv(r[0], r[1]))))];
      return m.length >= 2 ? m.slice(0, 2) : ['X', 'Y', 'Z'];
    };
    // opt.scBalance:false takes shortcuts out of rule 2 entirely -- only the
    // four base directions are balanced, so a shortcut axis may be re-used
    // freely. The generation lock is then enforced by modeAxes() directly,
    // because balance had been doing it as a side effect: the dormant axis
    // simply had no counter, so the lookup failed.
    const balanceSet = () => {
      const axes = modeAxes();
      if (!scBalance) return [...chSet()];
      return [...chSet()].concat(pooled ? axes
        : [].concat(...axes.map(a => [a + '+', a + '-'])));
    };
    // Shortcut axes are POOLED but carry TWICE the allowance: X+ and X- charge
    // one X counter, and that counter may reach 2 for every 1 on a base
    // direction. So a balanced run spends four base directions against two
    // shortcut axes used twice each -- e.g. Z a b X c d X Z, equal measure of
    // shortcut and base. The comparison level is therefore floor(n/2) for a
    // shortcut and n for a base direction.
    //
    // Plain pooling (level = n) strangles the nucleon: one X+ makes the whole
    // X axis read as spent, so the X- that closes the equator is barred. Fully
    // signed senses free that but let one axis run away. Pooled-with-2x is the
    // middle, and it is what puts a 60-degree SHORTCUT within reach of an apex.
    const balanceTable = () => {
      const rows = balanceSet().map(id => {
        const kind = id.indexOf(',') >= 0 ? 'base' : 'sc';
        const n = senseCount.get(id) || 0;
        return { id, n, kind, lvl: kind === 'sc' && !pooledFlat
                                   ? Math.floor(n / 2) : n };
      });
      const min = Math.min(...rows.map(x => x.lvl));
      rows.forEach(x => x.ok = (x.lvl === min));   // RULE 2 is hard: only the minimum
      return { rows, min };
    };
    // RULE 3, the rolling block of three turns.
    //   proton  = two 90 and one 60      neutron = two 60 and one 90
    // The order inside a block is free; only the composition is fixed. The rule
    // may be broken ONLY to keep rules 1 and 2, and ONLY for a neutron.
    const blockState = () => {
      const spent = classLog.slice(Math.floor(classLog.length / 3) * 3);
      const want90 = identity === 'proton' ? 2 : 1;
      const want60 = identity === 'proton' ? 1 : 2;
      return { identity, spent,
               left90: want90 - spent.filter(c => c === 90).length,
               left60: want60 - spent.filter(c => c === 60).length,
               mayBreak: identity === 'neutron' };
    };
    const sideOf = (k, off) => { const r = dirRef.get(k); if (!r) return '+';
      return (off[0] * r[0] + off[1] * r[1] + off[2] * r[2]) >= 0 ? '+' : '-'; };
    const ringOKl = rods => {
      if (rods.length > 4) return false;
      if ([...new Set(rods.map(r => ax(dv(r[0], r[1]))))].length > 2) return false;
      const deg = new Map();
      for (const r of rods) for (const v of r) deg.set(v, (deg.get(v) || 0) + 1);
      for (const d of deg.values()) if (d > 2) return false;
      return true;
    };
    const closedRing = rods => {
      if (rods.length !== 4) return false;
      const deg = new Map();
      for (const r of rods) for (const v of r) deg.set(v, (deg.get(v) || 0) + 1);
      return [...deg.values()].length === 4 && [...deg.values()].every(d => d === 2);
    };
    const attract = () => {
      const d = window._DEEPHOLE({ aroundRods: true, quiet: true,
        tieNear: xon !== null ? [P[xon].x, P[xon].y, P[xon].z] : null });
      const top = (d.top && d.top.length) ? d.top[0] : null;
      drawAttractor(top ? top.at : null);
      return top;
    };

    // The turn at the xon, in DEGREES, from the triangle prev-xon-next.
    // cos t = (a^2 + b^2 - c^2) / 2ab. Reporting the far-pair distance instead
    // invited comparing 90 to 1.489, which are not the same kind of thing.
    const angleAt = (a, b, c) => (a && b && c)
      ? +(Math.acos(Math.max(-1, Math.min(1, (a * a + b * b - c * c) / (2 * a * b))))
          * 180 / Math.PI).toFixed(1) : null;
    // RAW MEASURED ANGLE with a permissive band.
    //
    // The combinatorial formalization is PARKED, not solved. Two attempts were
    // both too loose: counting every possible shortcut as a unit pair made 122
    // and 133 degree turns classify as right angles. The relaxed lattice does
    // not render a half-built ring at exactly 60/90 either -- a ring corner
    // measures 92.9, an apex diagonal 92.9, a tet face 60.0, and the hop back
    // to the origin 66.9. So the bands are wide on purpose, chosen to match the
    // measured board rather than to be principled.
    //
    // This is a STOPGAP so play can continue. It will misjudge genuinely
    // ambiguous turns, and the real rule is structural, not metric.
    // Bands, revised: a THIRD class appears between 60 and 75. The relaxed
    // lattice renders a right angle anywhere from 75 up (measured 75.5 at a
    // proto-apex, 92.9 at a true ring corner), so 75 is the lower edge of 90 --
    // not the top of 60. 105 stays the upper edge so ring corners still read 90.
    //   [45,60]  -> 60
    //   (60,75)  -> 75   <- neither 60 nor 90; satisfies no proton slot
    //   [75,105] -> 90
    const cls = t => t === null ? 'first'
      : (t >= 45 && t <= 60 ? 60
      : (t > 60 && t < 75 ? 75
      : (t >= 75 && t <= 105 ? 90 : 'off-model')));


    function row(j, kind, A) {
      const d = dv(xon, j), k = dkey(d);
      const dist = prevNode === null ? null : +P[prevNode].distanceTo(P[j]).toFixed(4);
      const angNow = prevNode === null ? null
        : angleAt(P[prevNode].distanceTo(P[xon]), P[xon].distanceTo(P[j]), dist);
      const turnNow = angNow === null ? 'first' : cls(angNow);
      // For a creation the angle that counts is the one AFTER the rod pulls
      // xon and j to unit distance -- combinatorial, never measured off P.
      const post60 = prevNode !== null &&
        (_baseSet.has(K(prevNode, j)) || live.some(r => same(r, [prevNode, j])));
      const q = P[j];
      let side = null, sideUsed = null;
      if (A) {
        const p0 = P[xon];
        const off = [(p0.x + q.x) / 2 - A.at[0], (p0.y + q.y) / 2 - A.at[1],
                     (p0.z + q.z) / 2 - A.at[2]];
        side = sideOf(k, off); sideUsed = sideCount.get(k + '|' + side) || 0;
      }
      const okCh = kind !== 'base' || chSet().has(k);
      const isSC = kind !== 'base';
      const bId = dirId(d, isSC);
      const bTab = balanceTable();
      // Generation lock, stated explicitly. Balance used to enforce it as a
      // side effect -- the dormant axis had no counter, so the lookup failed --
      // and that stops being true once shortcuts leave the balance set.
      const okGen = !isSC || modeAxes().indexOf(ax(d)) >= 0;
      const okBal = (isSC && !scBalance) ? true
                  : bTab.rows.some(r => r.id === bId && r.ok);
      return { to: j, kind, axis: kind === 'base' ? '-' : ax(d), dir: k,
        balId: bId, balUsed: senseCount.get(bId) || 0,
        okChirality: okCh, okBalance: okBal && okGen, okGen,
        dirUsed: dirCount.get(k) || 0, side, sideUsed,
        angleNow: angNow, turnNow,
        turn: turnNow,                                    // the RULE (metric band)
        turnIfCreated: kind === 'sc-new' ? (post60 ? 60 : 90) : null,
        ringOK: kind === 'sc-new' ? ringOKl(live.concat([[xon, j]])) : null,
        shuttle: j === prevNode,
        toAttractor: A ? +Math.hypot(q.x - A.at[0], q.y - A.at[1], q.z - A.at[2]).toFixed(3) : null };
    }

    function look() {
      const A = attract(), rows = [];
      for (const j of _baseNbr[xon]) rows.push(row(j, 'base', A));
      for (let a = 0; a < AXN.length; a++) {
        const j = SCOPT.get(xon + ':' + a); if (j === undefined) continue;
        rows.push(row(j, live.some(r => same(r, [xon, j])) ? 'sc-live' : 'sc-new', A));
      }
      return { tick, xon, prevNode, rods: live.length, closedRing: closedRing(live),
        identity, chirality,
        balance: balanceTable(),
        block: blockState(), classLog: classLog.slice(),
        fluxMode: [...new Set(live.map(r => ax(dv(r[0], r[1]))))].join('') || null,
        solids: { octs: solids.octs.length, tets: solids.tets.length },
        attractor: A ? { at: A.at.map(v => +v.toFixed(3)), depth: +A.d.toFixed(4),
                         contacts: A.contacts, kind: A.kind } : null,
        dirCount: [...dirCount.entries()].map(([k, v]) => k + ':' + v).join('  ') || '(none)',
        sideCount: [...sideCount.entries()].map(([k, v]) => k + ':' + v).join('  ') || '(none)',
        candidates: rows };
    }

    async function start(opt) {
      opt = opt || {};
      identity = opt.identity === 'neutron' ? 'neutron' : 'proton';
      chirality = CH_SETS[opt.chirality] ? opt.chirality : 'A';
      pooled = opt.signed ? false : true;
      pooledFlat = !!opt.pooledFlat;
      scBalance = opt.scBalance !== false;
      _sync();
      active.clear(); live = []; prevNode = null; tick = 0; log.length = 0;
      dirCount.clear(); sideCount.clear(); dirRef.clear();
      senseCount.clear();
      classLog.length = 0;      // must reset, or the identity block carries over
      lastWasSC = false;
      xon = KEY.get(LCENTER.join(','));
      if (xon === undefined) { let bd = Infinity;
        for (let i = 0; i < NODE.length; i++) {
          const d = Math.hypot(...[0, 1, 2].map(k => NODE[i][k] - LCENTER[k]));
          if (d < bd) { bd = d; xon = i; } } }
      wake = [xon];
      restate(true); await settle(); detect(); drawXon(xon, wake);
      return look();
    }

    async function move(to, why) {
      if (xon === null) throw new Error('call _MANUAL.start() first');
      // Class of THIS move, read off the board BEFORE it is applied. Only 60s
      // and 90s advance the identity block; a 75 is playable but fills no slot.
      const row0 = look().candidates.find(c => c.to === to);
      const cls0 = row0 ? row0.turn : null;
      if (cls0 === 60 || cls0 === 90) classLog.push(cls0);
      const d = dv(xon, to), k = dkey(d);
      const A = attract();
      const isBase = _baseSet.has(K(xon, to));
      let created = false, rod = null;
      if (!isBase) {
        rod = [xon, to];
        if (!live.some(r => same(r, rod))) {
          const nk = scKeyOf(xon, to);
          if (!nk) throw new Error('no base edge and no shortcut option ' + xon + '->' + to);
          active.set(nk[0], nk[1]); live = live.concat([rod]); created = true;
        }
      }
      const from = xon;
      prevNode = xon; xon = to; wake.push(xon); tick++;
      restate(true); await settle(); detect(); drawXon(xon, wake);
      const ok = legal(resid);
      if (!ok && created) {                    // vacuum refuses it: undo the rod
        const nk = scKeyOf(from, to); if (nk) active.delete(nk[0]);
        live = live.filter(r => !same(r, rod));
        restate(true); await settle(); detect();
      }
      lastWasSC = !isBase;
      dirCount.set(k, (dirCount.get(k) || 0) + 1);
      // RULE 2's ledger. Base counts the signed sense (chirality admits one per
      // axis); shortcuts count the AXIS, pooled over both senses.
      const bId = dirId(d, !isBase);
      senseCount.set(bId, (senseCount.get(bId) || 0) + 1);
      if (A) {
        const a = P[from], b = P[xon];
        const off = [(a.x + b.x) / 2 - A.at[0], (a.y + b.y) / 2 - A.at[1],
                     (a.z + b.z) / 2 - A.at[2]];
        const sd = sideOf(k, off);
        if (!dirRef.has(k)) dirRef.set(k, off);
        sideCount.set(k + '|' + sd, (sideCount.get(k + '|' + sd) || 0) + 1);
      }
      log.push({ tick, from, to, dir: k, created, vacuumOK: ok, why: why || '' });
      return { applied: { tick, from, to, dir: k, created, vacuumRefused: !ok }, state: look() };
    }

    // GROUND TRUTH for creation candidates. `turnIfCreated` above is only a
    // combinatorial PREDICTION; here the rod is actually materialised, solved,
    // measured and reverted. Reporting both is the whole point of the harness:
    // if prediction and measurement ever disagree, the CA is mis-classifying
    // its own moves and no amount of scoring will save it.
    async function probe() {
      const s = look();
      for (const c of s.candidates)
        if (c.kind !== 'sc-new') { c.turnMeasured = c.turnNow; c.vacuumOK = true; }
      const news = s.candidates.filter(c => c.kind === 'sc-new');
      // ONE XON STEP IS ONE FLUX EVENT. A probe must never leave a rod behind:
      // a leaked rod is a deformation no move paid for, and every angle
      // measured afterwards would be against geometry the xon never created.
      // Snapshot/restore in a finally makes that hold even if the run throws --
      // which is exactly how a stray Y rod appeared on move 1.
      const snapshot = new Map(active);
      try {
      if (news.length) {
        // Through runExperiment, not a bare loop: the anti-batch guard is right
        // that bulk solving must be visible on screen, and watching the harness
        // try each rod in turn is exactly what we want to see anyway.
        await runExperiment('manual probe — angle after each rod manifests',
          news.map(c => ({ c, label: 'rod ' + xon + '->' + c.to })),
          (it) => { it.nk = scKeyOf(xon, it.c.to); if (it.nk) active.set(it.nk[0], it.nk[1]); },
          (it) => {
            const c = it.c;
            // Measured in the SOLVED geometry with the rod in place, so both
            // arms are at unit length and the angle is the real one.
            const far = prevNode === null ? null : P[prevNode].distanceTo(P[c.to]);
            c.distAfter = far === null ? null : +far.toFixed(4);
            c.angleAfter = prevNode === null ? null
              : angleAt(P[prevNode].distanceTo(P[xon]), P[xon].distanceTo(P[c.to]), far);
            c.armAfter = +P[xon].distanceTo(P[c.to]).toFixed(4);   // did the rod contract to 1?
            c.turnMeasured = c.angleAfter === null ? 'first' : cls(c.angleAfter);
            c.vacuumOK = legal(resid);
            c.predictionOK = (String(c.turnIfCreated) === String(c.turnMeasured));
            if (it.nk) active.delete(it.nk[0]);        // revert before the next item
            return { html: xon + '&rarr;' + c.to + ' &nbsp; <b>' + c.turnMeasured + '</b>'
              + (c.vacuumOK ? '' : ' <b style="color:#ff5c5c">vacuum refused</b>') };
          }, { onFreeze: () => freezeOff() });
      }
      } finally {
        active.clear(); for (const [k, v] of snapshot) active.set(k, v);
        restate(true); await settle(); detect(); drawXon(xon, wake);
      }
      s.probed = true;
      return s;
    }

    return { start, look, probe, move, sheet: () => log.slice() };
  })();

  // PROTON AND NEUTRON ARE THE SAME CA, ONE PARAMETER APART.
  //   proton  = 90° two-thirds of the time, 60° one-third   (uud)
  //   neutron = 60° two-thirds of the time, 90° one-third   (udd)
  // The turn ratio IS the quark content: 90° is up, 60° is down. It is a
  // probability, not a sequence.
  if (window._registerLoop) {
    // ticks is read live, so the lifespan slider takes effect on the next
    // particle rather than mid-life -- a hadron does not grow longer while it
    // exists. `extra` is what separates the emergent build from the 1.0 build.
    const hadron = (id, name, p, sub, extra) => window._registerLoop(id, name, sub,
      async (tok) => {
        while (window._loopAlive(tok)) {
          await window._XONPROTON(Object.assign({ pRing: p,
            ticks: window._loopLife ? window._loopLife() : 240,
            alive: () => window._loopAlive(tok) }, extra));
          await breathe();
        }
        cleanup();
      }, { life: true });
    hadron('proton', 'proton (uud)', 2 / 3,
      'One xon at the lattice centre builds an octahedron — four shortcuts in a '
      + 'closed ring — then turns 90° two-thirds of the time and 60° one-third. '
      + '90 is up, 60 is down: the ratio IS uud. One tet at a time. NO ring '
      + 'scaffolding — the ratio picks the class of turn, the attractor picks '
      + 'the direction. The oct is emergent, and sometimes fails to form.',
      { freeBuild: true, attractor: true });
    // THE NEUTRON IS OCT-CENTRED TOO. A 90-degree turn needs the far pair at
    // sqrt2, and no pair inside a regular tet is at sqrt2 -- measured, a lone
    // actualized tet supplies TWENTY 60-degree angles and ZERO right angles,
    // and none in its second shell either. Right angles exist only across an
    // octahedron's equatorial ring, where shortcut meets shortcut.
    //
    // So if 90 is up and 60 is down, an up quark REQUIRES an octahedron. udd
    // contains an up, so the neutron needs a ring exactly as the proton does.
    // The two differ only in the ratio. The tet-only version was built and
    // run: 240 turns, every one 60 degrees, ratio90 exactly 0 against a
    // requested 0.333 -- unsatisfiable, not merely unmet. It was retired.
    hadron('neutron', 'neutron (udd)', 1 / 3,
      'The same automaton as the proton with the ratio inverted: 60° two-thirds '
      + 'of the time, 90° one-third. udd. Achieves 0.308 against 0.333 — the '
      + 'lattice is 60°-rich, so udd sits far more comfortably than uud. Same '
      + 'emergent build as the proton, one parameter apart.',
      { freeBuild: true, attractor: true });

    // 1.0 = the original scaffolded build, kept for reference. The ring is
    // FORCED: rod 2 orthogonal to rod 1, rod 3 = -d1, rod 4 = -d2, and ring
    // moves take priority over the ratio until it closes. The oct is therefore
    // guaranteed rather than emergent, which is exactly why these are useful as
    // a control. No attractor and no plane rule, so they are the untouched
    // originals.
    hadron('proton-1', 'proton 1.0 (scaffolded)', 2 / 3,
      'REFERENCE BUILD. The ring is forced — d1, d2, −d1, −d2 — and ring moves '
      + 'outrank the ratio until it closes, so the octahedron always forms. '
      + 'No attractor, no direction balance, no no-shuttle rule. The control.',
      { legacy: true });
    hadron('neutron-1', 'neutron 1.0 (scaffolded)', 1 / 3,
      'REFERENCE BUILD. Forced ring, ratio inverted: 60° two-thirds, 90° '
      + 'one-third. No attractor, no direction balance, no no-shuttle rule. '
      + 'The control for udd.',
      { legacy: true });
  }

  if (window._registerLoop) {
    // The unrestricted 'electron (60° xon)' loop is RETIRED. It was the same
    // automaton with chirality off, and once the planner replaced the one-step
    // lookahead the chiral version matched it on every measure -- 24/24
    // traversals across 24 lines, offMean 0.55 vs 0.55. It survives as
    // _XONMOM({chirality:null}) for control runs; it is no longer a loop.
    //
    // Chirality: base senses restricted to one of the two zero-sum sign
    // patterns, drawn at spawn and fixed for the life of the particle;
    // shortcuts are unaffected. Measured, it costs the electron nothing --
    // which is what a chirality ought to look like.
    const randDir = () => { let x, y, q;
      do { x = 2 * Math.random() - 1; y = 2 * Math.random() - 1; q = x * x + y * y; }
      while (q >= 1 || q === 0);
      const t = 2 * Math.sqrt(1 - q); return [x * t, y * t, 1 - 2 * q]; };

    // e5 -- THE RAIL-ONLY ELECTRON -- IS RETIRED, because the electron below
    // SUBSUMES it exactly rather than merely replacing it. On a vertical line
    // the loop solver returns nRail = 1, nP = nQ = 0 by arithmetic, and the two
    // then run hop for hop identically: 25 hops chirality A, 24 chirality B, 11
    // flux events, 100%, offMax 1.00, net 0,11,0 in both. Keeping a second loop
    // that is a special case of the first only invites them to drift apart.
    //
    // What e5 established, and what still holds:
    //   Holding the generation every tick admits ONLY edge-flips -- the new rod
    //   goes on the same axis as the one it replaces, parallel and disjoint.
    //   Measured exhaustively: 2472 such flips, 2472 preserve the mode, 0 lose
    //   it. Their centroid displacement is +-1 along the DORMANT axis and
    //   nothing else -- XZ moves Y, XY moves Z, YZ moves X, 824 each. Per-tet
    //   rank is 1 in all 1392 tets and the same-mode tets fall into 156
    //   mutually disconnected rails.
    //   So a rail-only electron cannot be steered off its axis at any quality
    //   of choreography. Steering needed the OTHER edge-flip family -- sharing
    //   a base edge, both rods swinging -- which is what the electron below
    //   uses. It survives as _XONMOM({lockAxes:true, dir:[0,1,0]}) for control
    //   runs; it is no longer a loop.
    //
    // Also retired with it: the direction<->generation table. It is not gone,
    // it is FORCED, and worth stating once -- the birth mode is fixed by the
    // direction of travel, one-to-one and without exception:
    //   dir +-X -> YZ      dir +-Y -> XZ      dir +-Z -> XY
    // each generation running on the one axis it does not use.

    // THE BIASED ELECTRON -- the rail plus a second direction.
    //
    // One base <111> direction drawn at birth from the chirality-allowed four,
    // and one extra move along it per rail step. The move is a mode-RETURNING
    // 3-flip chain: measured, every one of the 1392 tets has chains whose net
    // centroid displacement is exactly a base direction (|d| = sqrt3), 4 to 8
    // of the 8 per tet. Chains of 2 flips land back on the dormant axis, so 3
    // is the shortest thing that steers at all.
    //
    // MEASURED, vertical line, all four chirality-A directions:
    //   no bias    net 0,11,0     -- the pure rail
    //   1,1,1      net 4,9,4      -- 4 bias moves, +4 +4
    //   -1,-1,1    net -3,9,3     -- 3 bias moves, -3 +3
    //   -1,1,-1    net -3,9,-3    -- 3 bias moves, -3 -3
    //   1,-1,-1    net 4,9,-4     -- 4 bias moves, +4 -4
    // Each completed bias contributes exactly its own +-1 in X and Z; the count
    // and the displacement agree exactly. Turns 60 only, 0 vacuum refusals, 0
    // broken tets.
    //
    // Live status into the loop panel's subtitle, so the rules are visible
    // while it runs rather than only in a returned object.
    const e6Status = (st) => {
      const el = document.getElementById('lpsub'); if (!el) return;
      const genOK = st.gen === st.lock;
      el.innerHTML =
        '<b style="color:#7fd4a8">chirality ' + st.chirality + '</b>'
        + ' &nbsp;·&nbsp; <b style="color:' + (genOK ? '#7fd4a8' : '#ff5c5c') + '">gen '
        + (st.gen || '—') + '</b>'
        + ' &nbsp;·&nbsp; line ' + st.dir.join(',')
        + '<br>loop ' + st.loopLen + ' moves (cos '
        + (st.loopCos !== null ? st.loopCos.toFixed(4) : '—') + ')'
        + ' &nbsp;·&nbsp; step ' + (st.loopLen ? (st.i % st.loopLen) + 1 : 0) + '/' + st.loopLen
        + ' &nbsp;·&nbsp; rail ' + st.rail + ' steer ' + st.steer
        + '<br>rods ' + st.rods + ' &nbsp;·&nbsp; flux ' + st.events
        + ' &nbsp;·&nbsp; tick ' + st.tick
        + ' &nbsp;·&nbsp; <b>angle to line ' + (st.angle === null ? '—' : st.angle + '°') + '</b>';
    };

    // THE NUCLEONS, on the 6 Aug rule set: no shuttling, base-only directional
    // balance, a TURN PREFERENCE (not a quota), shortcuts preferred, attractor
    // last -- with chirality (L or R) and generation both fixed at birth, the
    // no-tet-before-the-oct gate, and severance so a second oct cannot form.
    for (const [id, label, ident, blurb] of [
      ['nucleon-p', 'proton (uud) — 6 Aug rules', 'proton',
       'Prefers 90° whenever a 90° is legal, and takes whatever else is legal '
       + 'when none is — the uud ratio is an outcome, not a quota. Directional '
       + 'balance over the four base directions only; shortcuts are re-usable, '
       + 'which is what lets the equator close. Chirality L/R and generation '
       + 'fixed at spawn.'],
      ['nucleon-n', 'neutron (udd) — 6 Aug rules', 'neutron',
       'The same automaton with the preference inverted: 60° whenever one is '
       + 'legal. udd. Nothing else differs.']]) {
      window._registerLoop(id, label, blurb, async (tok) => {
        while (window._loopAlive(tok)) {
          await window._NSTEP.start({ identity: ident });
          for (let n = 0; n < 200 && window._loopAlive(tok); n++) {
            const r = await window._NSTEP.step();
            if (r.stalled) break;
            const p2 = (window._loopPace && window._loopPace()) || 0;
            if (p2) await new Promise(res => setTimeout(res, p2));
          }
          await breathe();
        }
        cleanup();
      });
    }

    // THE STEERING ELECTRON. Y shortcuts are banned, the generation is held at
    // EVERY tick, a tet is closed at EVERY tick, and it still steers.
    //
    // Y SHORTCUTS DO NOT EXIST for this particle -- refused at the point of
    // proposal in candidates, planner, walk and start tet alike. That leaves
    // the 464 XZ tets of 1392, so the generation is XZ BY CONSTRUCTION and
    // nothing has to hold it.
    //
    // The bias is an EDGE-FLIP, not a face-flip. Face-flips are what needed Y:
    // 1728 lead out of the XZ tets and 0 land legally, because a face-flip must
    // lay its new rod on the third axis. An edge-flip shares a BASE EDGE with
    // the next tet, swings BOTH rods, and stays in the mode -- 2280 such pairs
    // in the lattice, displacing in the mode's own plane, XZ: (1,0,+-1). With
    // the rail on the dormant axis that is RANK 3.
    //
    // BOTH RODS CHANGE, SO THE ORDER OF SEVERANCE IS EVERYTHING:
    //   tick 1:  add b1            -> {a1, a2, b1}   tet A still closed
    //   tick 2:  add b2, sever a1  -> {a2, b1, b2}   tet B now closed
    //   tick 3:  sever a2          -> {b1, b2}       tet B
    // Sever first and the set drops to a 3-node hinge with no tet at all
    // (4560 of 4560). Add first and a tet is closed throughout, unambiguously:
    // in each 3-rod set exactly one pair closes, the others being same-axis or
    // meeting at a vertex. The vacuum charges nothing for the third rod --
    // 16 of 16 probed states legal, minSep exactly 1.000.
    //
    // AUDITED, 4 bias directions x both chiralities, vertical line:
    //   net  1,1,1 -> 4,9,4    -1,-1,1 -> -3,9,3   -1,1,-1 -> -3,9,-3
    //        1,-1,-1 -> 4,9,-4  and B chirality mirrors each
    //   every bias move planned was completed (expandFail 0)
    //   generations seen: XZ only          off-generation ticks: 0
    //   ticks with no closed tet: 0        banned rods installed: 0
    //   shuttling 0, teleports 0, vacuum refusals 0, turns 60 only
    window._registerLoop('electron', 'electron',
      'Line, generation and handedness each drawn at spawn and fixed for life. '
      + 'No a-b-a, 60° motions only, a closed tet every tick, no Y shortcut '
      + 'ever. The traversal is a PRE-COMPUTED loop replayed verbatim — solved '
      + 'at birth, never searched for — and it steers by edge-flips that share '
      + 'a base edge, adding the incoming rod before severing the outgoing one '
      + 'so the tet never opens.',
      async (tok) => {
        while (window._loopAlive(tok)) {
          await window._XONMOM({ lockAxes: true, dir: randDir(), fit: true,
                                 banAxis: 'Y', status: e6Status,
                                 chirality: Math.random() < 0.5 ? 'L' : 'R',
                                 alive: () => window._loopAlive(tok) });
          await settleBetweenRuns(tok);
        }
        cleanup();
      });

    // THE CHAIN WALKER IS RETIRED. Mode-locked per STEP rather than per tick:
    // face-flip chains that return to the birth mode. It followed any random
    // line (38 of 40 runs at 100%, offMax mean 1.3) and paid for it in the one
    // currency that is not negotiable -- audited over 907 ticks it sat OFF its
    // birth generation for 561 of them, 62%, breaking in 10 of 10 runs. It
    // passed no-shuttling, chirality and the 60-degree rule cleanly and failed
    // "generation fixed at birth", so it was never an electron. It survives as
    // Its chain machinery has now been removed outright along with tetWalk --
    // e6 needs no router, so nothing was left to keep it alive.

    // THE NEUTRINO. Same automaton, mode UNLOCKED: it takes face-flips, whose
    // new rod is forced onto the third axis, so the mode cycles XY->XZ->YZ for
    // life. That costs it a fixed generation and buys it rank-3 motion -- it
    // can follow any line at all.
    window._registerLoop('xon-chiral', 'neutrino (mode-cycling)',
      'Mode ROTATES every flux event — XY→XZ→YZ, forced, because a face-flip '
      + 'can only lay its new rod on the axis the mode is not using. No fixed '
      + 'generation, but rank-3 motion: it follows an arbitrary line. '
      + 'Handedness still applies to base senses.',
      async (tok) => {
        while (window._loopAlive(tok)) {
          await window._XONMOM({ alive: () => window._loopAlive(tok) });
          await breathe();
        }
        cleanup();
      });
  }

  window._basePath = basePath;
  window._allTets = allTets;
  window._tetFlips = faceFlips;
  window._tetIs = isTet;
  window._tetWrite = writeTet;
  console.log('tetline-scratch loaded — _LINE, _TETLINE, _allTets, _tetFlips');
})();
