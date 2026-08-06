// TEMPORARY — flux-event point-pattern recorder. Scratch, not part of the simulation.
//
// Question: are the SOLVED positions of flux events lattice-like or Poisson-like?
// Causal-set Lorentz invariance rests on the sprinkling being Poisson -- the
// unique boost-invariant point process, since boosts preserve 4-volume and
// Poisson depends on volume alone. A lattice is what BHS contrasts against.
//
// A flux event = an accepted change to the shortcut field. Its position is the
// midpoint of the affected edge IN SOLVED COORDINATES, so it moves as the
// lattice deforms. Node identity is combinatorially regular; position is not
// (RMS displacement 0.361, max 0.547 under a global mode).
//
// ONE SEED = ONE SPRINKLING SAMPLE. A causal set is a whole spacetime history,
// so a seed's full run is one realisation. Seeds are kept SEPARATE: number
// variance and g(r) measure within-realisation correlations, and pooling across
// seeds manufactures pairs between points that never coexisted.
//
// Accept/reject logic mirrors fluxDynamics exactly; the ensemble is unchanged.

function _edgeList() {
  const edges = [], seenE = new Set();
  for (const [k, j] of SCOPT) {
    const i = +k.split(':')[0];
    const pid = Math.min(i, j) + '-' + Math.max(i, j);
    if (seenE.has(pid)) continue; seenE.add(pid); edges.push([k, i, j, pid]);
  }
  return edges;
}

async function _oneSeed(steps, pAdd, seed) {
  _psSeed = seed;
  freezeOff(); active.clear(); restate(); await settle();
  const edges = _edgeList();
  const events = [];
  let lastKey = null, lastIJ = null, lastPid = null, mode = '';

  await runExperiment(`flux-event positions, seed=${seed}`,
    Array.from({ length: steps }, (_, k) => ({ k, label: 'step ' + k })),
    () => {
      lastKey = null; lastIJ = null; lastPid = null;
      const doAdd = active.size === 0 || _psRand() < pAdd;
      if (doAdd) {
        const free = edges.filter(([k, i, j]) => !active.has(k) && !active.has(j + ':' + (+k.split(':')[1] ^ 1)));
        if (!free.length) { mode = 'full'; return; }
        const [k, i, j, pid] = free[Math.floor(_psRand() * free.length)];
        active.set(k, [i, j]); lastKey = k; lastIJ = [i, j]; lastPid = pid; mode = 'add';
      } else {
        const ks = [...active.keys()];
        const k = ks[Math.floor(_psRand() * ks.length)];
        const v = active.get(k); lastIJ = v.slice();
        lastPid = Math.min(v[0], v[1]) + '-' + Math.max(v[0], v[1]);
        active.delete(k); mode = 'sever';
      }
    },
    (it) => {
      const ok = legal(resid);
      if (mode === 'add' && !ok) {
        active.delete(lastKey); freezeOff();
        return { html: '<span style="color:#ff5c5c">rejected</span>', step: it.k, kind: 'rej' };
      }
      if (lastIJ && P.length === NODE.length) {
        const a = P[lastIJ[0]], b = P[lastIJ[1]];
        events.push({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2,
                      e: lastPid, step: it.k, kind: mode });
      }
      return { html: `<span style="color:#7fd4a8">${mode}</span> n=${active.size} ev=${events.length}`,
               step: it.k, kind: mode };
    },
    { onFreeze: () => { freezeOff(); } });
  return events;
}

window._EVENTPOS_MULTI = async function (seeds, steps, pAdd) {
  steps = steps || 400; pAdd = pAdd || 0.92;
  const R = RADIUS[shells];
  const runs = [];
  for (const sd of seeds) {
    const ev = await _oneSeed(steps, pAdd, sd);
    runs.push({ seed: sd, events: ev });
    console.log('seed', sd, ev.length, 'events');
    window._EVM = { R, nodes: NODE.length, runs };     // exposed incrementally
  }
  // Controls in the same ball. The RIGHT lattice null is the 228 shortcut-edge
  // midpoints in REST coordinates -- that is the site set events are drawn from.
  const mids = [];
  for (const [k, i, j] of _edgeList())
    mids.push({ x: (REST[i].x + REST[j].x) / 2, y: (REST[i].y + REST[j].y) / 2,
                z: (REST[i].z + REST[j].z) / 2 });
  const nAvg = Math.round(runs.reduce((s, r) => s + r.events.length, 0) / runs.length);
  const pois = [];
  for (let n = 0; n < nAvg; n++) {
    for (;;) {
      const x = (2 * _psRand() - 1) * R, y = (2 * _psRand() - 1) * R, z = (2 * _psRand() - 1) * R;
      if (x * x + y * y + z * z <= R * R) { pois.push({ x, y, z }); break; }
    }
  }
  window._EVM = { R, nodes: NODE.length, runs, edgeMidpointsRest: mids, poisson: pois,
                  nAvgEvents: nAvg };
  window._EVMDONE = true;
  console.log('EVENTPOS_MULTI done', runs.map(r => r.events.length));
  return window._EVM;
};
console.log('events-scratch v2 loaded');
