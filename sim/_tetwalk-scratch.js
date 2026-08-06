// TEMPORARY — tet traversal by face-flip. Scratch, not part of the simulation.
//
// A tet = 4 base edges + 2 shortcuts on DISTINCT axes, one shortcut from each
// parity sublattice (forced: base edges join opposite parities, so two
// same-parity shortcuts could never close a 4-clique).
//
// ZAC'S CONSTRAINT (4 Aug): consecutive tets must SHARE A FACE — three of four
// vertices. That makes traversal a tumble, not a teleport, and it has exactly
// four options per position, one per face.
//
// What each flip does, measured:
//   * changes exactly ONE shortcut -> one flux event per step (the coupled
//     add/sever pair, not two independent moves)
//   * always recruits the axis that was NOT active and drops one of the pair,
//     so MOTION AND FLUX-MODE CHANGE ARE THE SAME EVENT
//   * displaces the centroid by exactly sqrt(1/2) = 0.7071 lattice units
//   * but no flip has an axial component above 0.5, so straight-line DRIFT is
//     capped at 0.5 per flux event -- a speed limit from geometry alone
//
// Measured walk (8 flips, seed configuration below, drift along +x): every step
// legal, always exactly 1 tet, x advancing +0.5 monotonically, y and z
// oscillating about a mean, and max lattice displacement pinned at 0.107-0.111
// throughout -- i.e. the deformation field travels with it unchanged. A soliton.
//
// Ratio instantaneous/drift = 0.7071/0.5 = sqrt(2). Zitterbewegung-shaped, in
// that the particle is faster than its own trajectory; NOT claimed to be the
// Dirac result, since the instantaneous speed here is not c.

(function () {
  const baseSet = new Set();
  for (const [i, j] of BASE) { baseSet.add(i * 1e5 + j); baseSet.add(j * 1e5 + i); }
  const scPair = new Map();
  for (const [k, j] of SCOPT) scPair.set((+k.split(':')[0]) * 1e5 + j, +k.split(':')[1]);
  const AXOF = d => (d[0] ? 'X' : (d[1] ? 'Y' : 'Z'));

  // Is this 4-node set a tet? Returns its two shortcuts, or null.
  window._isTet = function (q) {
    if (new Set(q).size !== 4) return null;
    let base = 0; const sc = [];
    for (let x = 0; x < 4; x++) for (let y = x + 1; y < 4; y++) {
      const a = q[x], b = q[y];
      if (baseSet.has(a * 1e5 + b)) { base++; continue; }
      const ax = scPair.get(a * 1e5 + b), ax2 = scPair.get(b * 1e5 + a);
      if (ax !== undefined) sc.push([a, b, AXOF(AX[ax])]);
      else if (ax2 !== undefined) sc.push([b, a, AXOF(AX[ax2])]);
      else return null;
    }
    if (base !== 4 || sc.length !== 2) return null;
    if (sc[0][2] === sc[1][2]) return null;          // distinct axes required
    return sc;
  };

  // Face-sharing neighbours: drop one vertex, find every E completing a tet.
  window._faceFlips = function (q) {
    const out = [];
    for (let drop = 0; drop < 4; drop++) {
      const face = q.filter((_, i) => i !== drop);
      for (let E = 0; E < NODE.length; E++) {
        if (q.includes(E)) continue;
        const cand = [...face, E];
        const sc = window._isTet(cand);
        if (sc) out.push({ drop: q[drop], add: E, tet: cand,
                           axes: sc.map(s => s[2]).sort().join('') });
      }
    }
    return out;
  };

  window._tetCentroid = q => {
    const c = [0, 0, 0];
    for (const v of q) { c[0] += NODE[v][0] / 4; c[1] += NODE[v][1] / 4; c[2] += NODE[v][2] / 4; }
    return c;
  };

  window._writeTet = function (q) {
    active.clear();
    for (const [a, b] of window._isTet(q)) {
      const d = [NODE[b][0] - NODE[a][0], NODE[b][1] - NODE[a][1], NODE[b][2] - NODE[a][2]];
      const ax = AX.findIndex(v => v[0] === d[0] && v[1] === d[1] && v[2] === d[2]);
      if (ax >= 0) active.set(a + ':' + ax, [a, b]);
    }
  };

  // The seed tet used in the 4 Aug measurements.
  window._SEED_TET = () => [[-2,0,0],[0,0,0],[-1,-1,1],[-1,1,1]].map(c => KEY.get(c.join(',')));

  // Greedy walk along `dir`, one flux event per step, refusing to immediately undo.
  window._TETFLIP = async function (steps, dir, seed) {
    const D = dir || [1, 0, 0];
    let T = seed || window._SEED_TET();
    let prev = null; const log = []; const c0 = window._tetCentroid(T);
    await runExperiment('tet face-flip walk (one flux event per step)',
      Array.from({ length: steps }, (_, k) => ({ k, label: 'flip ' + k })),
      () => {
        const opts = window._faceFlips(T)
          .filter(f => prev === null || f.add !== prev)
          .map(f => { const c = window._tetCentroid(f.tet);
                      return { f, gain: c[0]*D[0] + c[1]*D[1] + c[2]*D[2] }; })
          .sort((a, b) => b.gain - a.gain);
        if (!opts.length) return;
        prev = opts[0].f.drop; T = opts[0].f.tet; window._writeTet(T);
      },
      (it) => {
        detect(); const ok = legal(resid); const c = window._tetCentroid(T);
        let mx = 0; for (let i = 0; i < NODE.length; i++) mx = Math.max(mx, P[i].distanceTo(REST[i]));
        log.push({ flip: it.k, legal: ok, tets: solids.tets.length, sc: active.size,
                   centroid: c.map(x => +x.toFixed(2)),
                   netFromStart: +Math.hypot(c[0]-c0[0], c[1]-c0[1], c[2]-c0[2]).toFixed(3),
                   axes: window._isTet(T).map(s => s[2]).join(''), maxDisp: +mx.toFixed(4) });
        return { html: (ok ? '<span style="color:#7fd4a8">ok</span>'
                           : '<span style="color:#ff5c5c">illegal</span>')
                 + ' c=' + c.map(x => x.toFixed(1)).join(',') };
      },
      { onFreeze: () => freezeOff() });
    window._TF = log; return log;
  };
  console.log('tetwalk-scratch loaded — _isTet, _faceFlips, _writeTet, _TETFLIP, _SEED_TET');
})();
