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
  const HOP_CH = { A: ['1,1,1', '-1,-1,1', '-1,1,-1', '1,-1,-1'] };
  HOP_CH.B = HOP_CH.A.map(t => t.split(',').map(n => -(+n)).join(','));
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
  let _HOPT = null;
  function hopTable() {
    if (_HOPT) return _HOPT;
    const out = {};
    for (const c of ['U','D']) for (const m of HOP_MOV[c]) for (const ch of ['A','B']) {
      const R0 = hRods(c);
      // The agreed severance order: for a steer, ADD BEFORE SEVER so a tet is
      // closed on every tick. Rods pair by axis, b[0] with the old X rod and
      // b[1] with the old Z, which is what keeps the leftover rod from closing
      // a second tet with an incoming one.
      const plan = m.k === 'rail' ? [{ add:m.add, sev:m.sev }]
        : [{ add:m.b[0], sev:null }, { add:m.b[1], sev:R0[0] },
           { add:null, sev:R0[1] }];
      const newTet = m.k === 'rail'
        ? [...new Set(R0.filter(r => !hSame(r, m.sev)).concat([m.add])
            .reduce((a, r) => a.concat(r), []).map(hvK))].map(t => t.split(',').map(Number))
        : [...new Set(m.b.reduce((a, r) => a.concat(r), []).map(hvK))]
            .map(t => t.split(',').map(Number));
      const res = [];
      for (const start of HOP_CLS[c]) for (const prev of HOP_CLS[c]) {
        if (hvK(start) === hvK(prev) || !hUnit(R0, prev, start)) continue;
        let at = start, from = prev, rods = R0.map(r => [r[0].slice(), r[1].slice()]);
        const hops = []; let ok = true;
        for (const step of plan) {
          if (step.add) {
            let did = false;
            for (const [p2, q2] of [[step.add[0], step.add[1]], [step.add[1], step.add[0]]]) {
              const pre = [];
              if (hvK(at) !== hvK(p2)) {
                const d = hvSub(p2, at);
                if (!hvBase(d) || HOP_CH[ch].indexOf(hvK(d)) < 0) continue;
                if (hvK(p2) === hvK(from) || !hUnit(rods, from, p2)) continue;
                pre.push({ to: p2.slice(), kind: 'base' });
              }
              const f2 = pre.length ? at : from;
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
        res.push({ start: hvK(start), prev: hvK(prev), hops,
                   exitAt: hvK(hvSub(at, o2)), exitFrom: hvK(hvSub(from, o2)) });
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
    if (_spark && _spark.visible && _xonNode !== null && P[_xonNode])
      _spark.position.copy(P[_xonNode]);
  };
  function drawXon(node, trail) {
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
  window._XONOFF = () => { if (_spark) { _spark.visible = false; _wake.visible = false; } };

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
    let chirality = opt.chirality;
    if (chirality === undefined) {
      const cs = opt.seed === undefined ? (Math.random() * 2147483647) | 0 : opt.seed;
      chirality = _mul(cs ^ 0x5bf03635)() < 0.5 ? 'A' : 'B';
    }
    const chiralSet = chirality === null ? null
      : new Set(chirality === 'A' ? CHIRAL_A : CHIRAL_B);
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
      let best = null;
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
          if (!best || cos > best.cos + 1e-9
              || (Math.abs(cos - best.cos) < 1e-9 && n < best.n))
            best = { cos, n, Y, a, b, D };
        }
      }
      if (!best) return null;
      // ORDER THE LOOP. Same idea as Bresenham: hold the running position and
      // take, at each place in the sequence, whichever remaining move leaves
      // the path closest to the ideal ray. Deterministic, done once, and the
      // loop's total is exactly D however it is ordered -- the ordering only
      // decides how far the path strays in between.
      const mk = (k0, k1, s0, s1) => { const w = [0, 0, 0];
        if (k0 !== null) w[k0] = s0; if (k1 !== null) w[k1] = s1; return w; };
      const rem = [
        { kind: 'rail',  v: mk(dorm, null, Math.sign(best.Y) || 1, 0), n: Math.abs(best.Y) },
        { kind: 'steer', v: mk(pl[0], pl[1], Math.sign(best.a) || 1, Math.sign(best.a) || 1),
          n: Math.abs(best.a) },
        { kind: 'steer', v: mk(pl[0], pl[1], Math.sign(best.b) || 1, -(Math.sign(best.b) || 1)),
          n: Math.abs(best.b) }];
      const nd = Math.hypot(...best.D), u = best.D.map(z => z / nd);
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
      return { steps, disp: best.D, cos: +best.cos.toFixed(6),
               nRail: best.Y, nP: best.a, nQ: best.b };
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
    const closeLoop = (steps, ch) => {
      if (!steps || !steps.length) return null;
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
        const seen = new Set([start + '#' + cnt0.join('.')]);
        let front = [{ cls: s0.cls, at: s0.at, prev: s0.prev,
                       cnt: cnt0.slice(), seq: [] }];
        let found = null;
        for (let depth = 0; depth < steps.length && front.length && !found; depth++) {
          const nxt = [];
          for (const st of front) {
            for (let i = 0; i < dk.length && !found; i++) {
              if (!st.cnt[i]) continue;
              const d = dk[i], ents = entryOf(st.cls, d);
              if (!ents.length) continue;
              for (const r of reachOf(st.cls, st.at, st.prev)) {
                const hit = ents.find(e => e.start === r.at && e.prev === r.prev);
                if (!hit) continue;
                const cnt = st.cnt.slice(); cnt[i]--;
                const ncls = flip(st.cls);
                const seq = st.seq.concat([{ d: d.split(',').map(Number),
                  entryAt: r.at, entryPrev: r.prev, repHops: r.hops }]);
                if (cnt.every(z => z === 0)) {
                  if (sk(ncls, hit.exitAt, hit.exitFrom) === start)
                    { found = { start: s0, seq }; }
                  continue;
                }
                const key = sk(ncls, hit.exitAt, hit.exitFrom) + '#' + cnt.join('.');
                if (seen.has(key)) continue; seen.add(key);
                nxt.push({ cls: ncls, at: hit.exitAt, prev: hit.exitFrom, cnt, seq });
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
    if (opt.fit) {
      LOOPFIT = solveLoop(v);
      // A closed loop repeats verbatim; an unclosed one is re-derived every
      // iteration and drifts. If the exact multiset will not close, try it
      // DOUBLED -- twice round the same displacements points the same way and
      // gives the ordering room to return to its own start state.
      if (LOOPFIT) {
        LOOPSEQ = closeLoop(LOOPFIT.steps, chirality || 'A');
        if (!LOOPSEQ && LOOPFIT.steps.length <= 12)
          LOOPSEQ = closeLoop(LOOPFIT.steps.concat(LOOPFIT.steps), chirality || 'A');
      }
    }
    const fitStats = { steps: 0, railDone: 0, steerDone: 0,
                       railMissing: 0, steerMissing: 0, routeFail: 0 };

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
            LOOPSEQ = { start: { cls: cls0 }, seq: LOOPSEQ.seq.slice(1)
                          .concat([LOOPSEQ.seq[0]]) };
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
        // The one repositioning hop, also tabulated.
        const analyticReposition = (disp) => {
          const F = frameOf(); if (!F) return null;
          const ch = chirality || 'A';
          const key = F.cls + '|' + disp.join(',') + '|' + ch;
          const ent = hopTable()[key]; if (!ent) return null;
          const xr = F.rel(xon).join(','), pr = F.rel(prevNode).join(',');
          for (const path of hopReposition(F.cls, ch, xr, pr)) {
            // only worth taking if it LANDS in a state this move accepts
            if (!ent.res.some(r => r.start === path.at && r.prev === path.from)) continue;
            const hops = [];
            let bad = false;
            for (const h of path.hops) {
              const to = nodeAt([0, 1, 2].map(k => NODE[F.o][k] + h.to[k]));
              if (to === undefined) { bad = true; break; }
              hops.push({ to, kind: h.kind,
                          rod: h.kind === 'sc' ? [xon, to] : undefined });
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
        } else if (lockAxes && LOOPFIT && !LOOPSEQ && LOOPFIT.steps.length
                   && !chainQ.length && live.length === 2) {
          // NO CLOSED CYCLE for this multiset. Fall back to the open schedule
          // and SAY SO in `stopped` rather than pretending the loop repeats.
          let placed = false;
          for (let tries = 0; tries < LOOPFIT.steps.length * 2 && !placed; tries++) {
            const st2 = LOOPFIT.steps[loopIdx % LOOPFIT.steps.length];
            const got = analyticMove(st2.d);
            if (got && got.hops) { chainQ = got.hops;
              biasActive = st2.kind === 'steer'; fitStats.steps++;
              if (st2.kind === 'rail') fitStats.railDone++; else fitStats.steerDone++;
              loopIdx++; placed = true; break; }
            if (got && got.reposition) { const rep = analyticReposition(st2.d);
              if (rep) { chainQ = rep; fitStats.repositions =
                (fitStats.repositions || 0) + 1; placed = true; break; } }
            if (st2.kind === 'rail') fitStats.railMissing++; else fitStats.steerMissing++;
            loopIdx++;
          }
          if (!placed) { done = true; stopped = 'openloopblocked'; pending = null; return; }
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
        if (sNow > bestS + 1e-9) { bestS = sNow; sinceGain = 0; } else sinceGain++;
        if (sNow >= hiTet - 1e-6) { done = true; stopped = 'traversed'; }
        else if (sinceGain >= STALL) { done = true; stopped = 'stalled'; }
        // The generation as it stands AFTER the move -- recorded per tick so a
        // bias excursion is visible in the trace rather than inferred from it.
        const mNow = modeOf(tetNodes());
        if (LOCK && mNow && mNow !== LOCK) biasStats.offModeTicks++;
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
      biasStats, loopClosed: !!LOOPSEQ,
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
    const dirId = (d, isSC) => {
      if (isSC) { const m = d[0] || d[1] || d[2]; return ax(d) + (m > 0 ? '+' : '-'); }
      let s = 0; for (const v of d) { if (v) { s = v; break; } }
      return 'b' + (s < 0 ? d.map(v => -v) : d).join(',');
    };
    const BASE_IDS = ['b1,1,1', 'b1,1,-1', 'b1,-1,1', 'b1,-1,-1'];
    // The balance set is THIS GENERATION's directions: the 4 base axes plus the
    // signed senses of the in-mode shortcut axes. An out-of-mode axis is not a
    // direction the particle has, so it must not sit at zero forever and
    // deadlock the minimum.
    const balanceSet = () => {
      const m = [...new Set(live.map(r => ax(dv(r[0], r[1]))))];
      const axes = m.length >= 2 ? m : ['X', 'Y', 'Z'];
      return BASE_IDS.concat(...axes.map(a => [a + '+', a + '-']));
    };
    const balanceTable = () => {
      const rows = balanceSet().map(id => ({ id, n: senseCount.get(id) || 0 }));
      const min = Math.min(...rows.map(x => x.n));
      rows.forEach(x => x.ok = (x.n === min));    // RULE 2 is hard: only the minimum
      return { rows, min };
    };
    // RULE 3: 3-turn blocks of 2x90 + 1x60 for the proton. What is left to spend.
    const blockState = () => {
      const spent = classLog.slice(Math.floor(classLog.length / 3) * 3);
      return { spent, left90: 2 - spent.filter(c => c === 90).length,
                      left60: 1 - spent.filter(c => c === 60).length };
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
      return { to: j, kind, axis: kind === 'base' ? '-' : ax(d), dir: k,
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
        block: blockState(), classLog: classLog.slice(),
        fluxMode: [...new Set(live.map(r => ax(dv(r[0], r[1]))))].join('') || null,
        solids: { octs: solids.octs.length, tets: solids.tets.length },
        attractor: A ? { at: A.at.map(v => +v.toFixed(3)), depth: +A.d.toFixed(4),
                         contacts: A.contacts, kind: A.kind } : null,
        dirCount: [...dirCount.entries()].map(([k, v]) => k + ':' + v).join('  ') || '(none)',
        sideCount: [...sideCount.entries()].map(([k, v]) => k + ':' + v).join('  ') || '(none)',
        candidates: rows };
    }

    async function start() {
      _sync();
      active.clear(); live = []; prevNode = null; tick = 0; log.length = 0;
      dirCount.clear(); sideCount.clear(); dirRef.clear();
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
    const AXDIRS = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]];
    // PINNED TO VERTICAL while steering is worked on -- one line, repeatable,
    // easy to read off the screen. Restore `AXDIRS[(Math.random()*6)|0]` to go
    // back to all six. Note this pins the GENERATION too, and not by choice:
    // measured over 11 consecutive spawns the birth mode is forced by the
    // direction of travel, one-to-one and without exception --
    //   dir +-X -> YZ      dir +-Y -> XZ      dir +-Z -> XY
    // each mode running on the one axis it does not use. So a vertical line is
    // always an XZ electron. Handedness stays random per spawn.
    const VERTICAL = AXDIRS[2];                       // [0,1,0]
    // e6 gets a RANDOM orientation through the centre -- fitting an arbitrary
    // line is the whole point of it, and a vertical one reduces to e5 anyway.
    const randDirE6 = () => { let x, y, q;
      do { x = 2 * Math.random() - 1; y = 2 * Math.random() - 1; q = x * x + y * y; }
      while (q >= 1 || q === 0);
      const t = 2 * Math.sqrt(1 - q); return [x * t, y * t, 1 - 2 * q]; };

    // THE ELECTRON -- lockAxes, the STRICT per-tick generation lock.
    //
    // Audited over 18 runs / 436 ticks, all six axes, both handednesses, all
    // three generations:
    //   no shuttling (a-b-a)      0 violations
    //   chirality fixed at birth  0 violations
    //   generation fixed at birth 0 mode breaks, 0 of 18 runs
    //   60-degree motions only    436 of 436 turns
    //   crossings                 100% x 18, offMax 1.00 every run
    //   teleports / vacuum refusals   0 / 0
    //
    // THE LINE IS DRAWN FROM THE SIX LATTICE AXES, NOT THE SPHERE, and that is
    // forced rather than chosen. Holding the mode every tick admits only
    // edge-flips -- the new rod goes on the SAME axis as the one it replaces,
    // parallel and disjoint. Measured exhaustively: 2472 such flips in the
    // lattice, 2472 preserve the mode, 0 lose it, every one sharing exactly two
    // vertices. Their centroid displacement is +-1 along the DORMANT axis and
    // nothing else -- XZ moves Y, XY moves Z, YZ moves X, 824 each, no
    // exceptions. Per-tet displacement rank is 1 in all 1392 tets, and the
    // same-mode tets fall into 156 components under edge-flips, every one rank
    // 1 and mutually disconnected. So there is no second axis to trade against
    // and no neighbouring rail to hop to: a diagonal is not something a
    // generation-locked electron can be steered onto, at any quality of
    // choreography. An off-axis target is a NEUTRINO test.
    window._registerLoop('electron', 'e5 — electron (gen-locked, rail only)',
      'Generation FIXED at birth and held EVERY tick — only edge-flips preserve '
      + 'it, and they move the tet along the one axis its mode does not use. '
      + 'Line PINNED VERTICAL for now, which pins the generation to XZ; '
      + 'handedness is still drawn at spawn. Every turn is 60°, no a-b-a, '
      + '100% crossings at offMax 1.00.',
      async (tok) => {
        while (window._loopAlive(tok)) {
          await window._XONMOM({ lockAxes: true, dir: VERTICAL,
                                 alive: () => window._loopAlive(tok) });
          await breathe();
        }
        cleanup();
      });

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
    window._registerLoop('electron-bias', 'e6 — electron (gen-locked, line-fitting)',
      'Steers while holding everything: generation XZ at every tick, a closed '
      + 'tet at every tick, no Y shortcut ever. The bias is an EDGE-flip — the '
      + 'next tet shares a base edge, both rods swing, and the incoming rod is '
      + 'added BEFORE the outgoing one is severed so the tet never opens. '
      + 'Net 4,9,4 for bias 1,1,1 against 0,11,0 for the pure rail.',
      async (tok) => {
        while (window._loopAlive(tok)) {
          await window._XONMOM({ lockAxes: true, dir: randDirE6(), fit: true,
                                 banAxis: 'Y',
                                 alive: () => window._loopAlive(tok) });
          await breathe();
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
