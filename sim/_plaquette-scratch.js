// TEMPORARY — Branch B test harness. Scratch, not part of the simulation.
//
// WHY A NEW SAMPLER. The add/sever sampler in fluxDynamics changes one bond at
// a time, so it cannot stay inside a fixed-degree ensemble: every single add or
// sever breaks |S| = 4. It samples the NON-OVERLAP constraint (minSep >= 1),
// which is an inequality, and that is why it returned a broad degree histogram.
// Close-packing is an EQUALITY -- kissing number saturated, tau + |S| = 12 --
// so sampling it needs a move that preserves every vertex's degree.
//
// THE MOVE. On the shortcut graph (simple cubic, coordination 6, bipartite),
// take a plaquette: v, v+a, v+a+b, v+b for two distinct axes a,b. Its four
// edges form a 4-cycle. If occupancy ALTERNATES around the cycle, flipping all
// four leaves every vertex's degree unchanged -- each of the four vertices
// touches exactly two consecutive edges, one active and one inactive, and the
// flip swaps them. This is the standard loop move for constrained bond models.
//
// The physics is untouched: this proposes configurations and asks the EXISTING
// solver whether they are legal. No solver setting, threshold, or acceptance
// rule is modified. An illegal proposal is reverted, never accepted.

// Is the shortcut between nodes i and j (direction index da from i) active?
function _scActive(i, j, da) {
  return active.has(i + ':' + da) || active.has(j + ':' + (da ^ 1));
}
function _scSet(i, j, da, on) {
  const ka = i + ':' + da, kb = j + ':' + (da ^ 1);
  if (on) { if (!active.has(ka) && !active.has(kb)) active.set(ka, [i, j]); }
  else { active.delete(ka); active.delete(kb); }
}

// The four (node, node, dirIndex) edges of the plaquette at v spanned by da, db.
// Returns null if any corner falls outside the ball.
function _plaquette(v, da, db) {
  const c = NODE[v];
  const step = (c0, d) => KEY.get([c0[0] + d[0], c0[1] + d[1], c0[2] + d[2]].join(','));
  const A = AX[da], B = AX[db];
  const p1 = step(c, A); if (p1 === undefined) return null;
  const p2 = step(NODE[p1], B); if (p2 === undefined) return null;
  const p3 = step(c, B); if (p3 === undefined) return null;
  // cycle v -> p1 -> p2 -> p3 -> v
  return [[v, p1, da], [p1, p2, db], [p2, p3, da ^ 1], [p3, v, db ^ 1]];
}

window._PLAQ = async function (nProposals, seed) {
  _psSeed = seed || 999;
  const t0 = performance.now();
  let alternating = 0, accepted = 0, rejected = 0, notAlt = 0, offLattice = 0;
  const startKey = [...active.keys()].sort().join('|');
  let pending = null, pendState = null;

  // Routed through runExperiment so every proposal is solved on screen -- the
  // engine refuses to step the solver any other way, and that guard stays.
  await runExperiment(`plaquette flips (degree-preserving), seed=${seed || 999}`,
    Array.from({ length: nProposals }, (_, k) => ({ k, label: 'flip ' + k })),
    () => {
      pending = null;
      const v = Math.floor(_psRand() * NODE.length);
      const da = Math.floor(_psRand() * 6);
      let db = Math.floor(_psRand() * 6);
      if ((da >> 1) === (db >> 1)) db = (db + 2) % 6;          // must be a different axis
      const P = _plaquette(v, da, db);
      if (!P) { offLattice++; return; }
      const st = P.map(([i, j, d]) => _scActive(i, j, d));
      if (!((st[0] !== st[1]) && (st[1] !== st[2]) && (st[2] !== st[3]) && (st[3] !== st[0]))) {
        notAlt++; return;
      }
      alternating++;
      P.forEach(([i, j, d], k) => _scSet(i, j, d, !st[k]));
      pending = P; pendState = st;
    },
    (it) => {
      if (!pending) return { html: '<span style="color:#5a6b7d">skip</span>', step: it.k, kind: 'skip' };
      const ok = legal(resid);
      if (ok) { accepted++; return { html: '<span style="color:#7fd4a8">accepted</span>', step: it.k, kind: 'acc' }; }
      pending.forEach(([i, j, d], k) => _scSet(i, j, d, pendState[k]));
      rejected++; freezeOff();
      return { html: '<span style="color:#ff5c5c">rejected (Kepler)</span>', step: it.k, kind: 'rej' };
    },
    { onFreeze: () => { freezeOff(); } });

  // degree audit: the sum rule must still hold on interior vertices
  const d = new Array(NODE.length).fill(0), seen = new Set();
  for (const [k, v2] of active) {
    const i = v2[0], j = v2[1], pid = Math.min(i, j) + '-' + Math.max(i, j);
    if (seen.has(pid)) continue; seen.add(pid); d[i]++; d[j]++;
  }
  const bd = new Array(NODE.length).fill(0); for (const [i, j] of BASE) { bd[i]++; bd[j]++; }
  const histInt = {};
  for (let i = 0; i < NODE.length; i++) if (bd[i] === 8) histInt[d[i]] = (histInt[d[i]] || 0) + 1;

  const endKey = [...active.keys()].sort().join('|');
  window._PQ = {
    proposals: nProposals, offLattice, notAlternating: notAlt,
    alternating, accepted, rejected,
    acceptRate: alternating ? +(accepted / alternating).toFixed(3) : null,
    activeUnique: seen.size, degreeHistInterior: histInt,
    legal: legal(resid), minSep: resid.minSep,
    movedAwayFromStart: startKey !== endKey,
    secs: +((performance.now() - t0) / 1000).toFixed(1)
  };
  console.log('PLAQ', JSON.stringify(window._PQ));
  return window._PQ;
};
console.log('plaquette-scratch loaded');
