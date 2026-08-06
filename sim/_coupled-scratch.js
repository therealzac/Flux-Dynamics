// TEMPORARY — coupled add/sever sampler. Scratch, not part of the simulation.
//
// THE MECHANISM (Zac, this session): an xon activates a shortcut; where that
// activation is mutually exclusive with a latent shortcut, the latent one is
// severed IN THE SAME TICK. Add and sever are one event pair, not two draws.
//
// Consequences this is built to test:
//   * below saturation adds are free and |S| climbs; at saturation every add
//     must evict, so |S| is conserved. Close-packing becomes an ATTRACTOR of
//     the dynamics rather than an axiom.
//   * a tick contains 1..3 simultaneous events, so runs produce genuine
//     ANTICHAINS -- the {a,b,c} inside [A,B,C]. The previous sampler drew adds
//     and severs independently and flattened every run into a pure chain.
//
// EXCLUSIVITY AS IMPLEMENTED: the kissing bound. tau + |S| <= 12, so a vertex
// of base degree tau admits at most 12 - tau shortcuts (4 in the interior).
// An add that would exceed the cap at an endpoint evicts one shortcut there.
// Eviction only LOWERS degree at its far endpoint, which is always legal, so
// there is no cascade -- at most two evictions per add.
//
// Annihilation severs are a different mechanism (independent, own tick) and are
// NOT modelled here. Every sever this sampler makes is exclusivity-induced and
// therefore simultaneous with its add.
//
// The physics is untouched: proposals go to the EXISTING solver, and an illegal
// tick is reverted whole. No solver setting or threshold is modified.

function _degAt() {
  const d = new Array(NODE.length).fill(0), at = NODE.map(() => []);
  const seen = new Set();
  for (const [k, v] of active) {
    const i = v[0], j = v[1], pid = Math.min(i, j) + '-' + Math.max(i, j);
    if (seen.has(pid)) continue; seen.add(pid);
    d[i]++; d[j]++; at[i].push(k); at[j].push(k);
  }
  return { d, at };
}

// cap = 12 - tau, the kissing bound, with tau the BULK base coordination of 8.
//
// Do NOT use each node's measured base degree. A boundary node's degree is low
// only because its neighbours lie outside the simulated ball -- physically it
// would have 8. Deriving the cap from the truncated degree hands boundary nodes
// a cap of 8, which is unreachable (there are only 6 shortcut directions), so
// they never evict and simply fill. With this ball that was 86 of 113 nodes,
// and it drove |S| to 152 with the exclusivity mechanism effectively disabled
// across most of the lattice.
const _BULK_TAU = 8;
function _capOf() {
  return NODE.map(() => 12 - _BULK_TAU);       // = 4, everywhere
}

window._COUPLED = async function (ticks, seed) {
  ticks = ticks || 300; _psSeed = seed || 5150;

  // The page persists `active` to localStorage (saveState, and saveNow on
  // beforeunload), and replays it at startup -- so a fresh load inherits the
  // PREVIOUS run's final configuration. An experiment starting from that is
  // measuring the tail of the last one. Clear the store, stop it being
  // rewritten, and refuse to start unless the vacuum is verified.
  try { localStorage.removeItem(SKEY); } catch (_) {}
  saveState = () => {};

  freezeOff(); active.clear(); restate(); await settle();
  if (active.size !== 0 || !legal(resid)) {
    console.error('REFUSED: not a clean vacuum', { active: active.size, resid });
    return { error: 'unclean start', active: active.size, resid };
  }
  const vacuumMinSep = resid.minSep;          // must be 2/sqrt3 = 1.154700...

  const CAP = _capOf();
  const edges = [], seenE = new Set();
  for (const [k, j] of SCOPT) {
    const i = +k.split(':')[0];
    const pid = Math.min(i, j) + '-' + Math.max(i, j);
    if (seenE.has(pid)) continue; seenE.add(pid); edges.push([k, i, j]);
  }
  const interior = [];
  { const bd = new Array(NODE.length).fill(0);
    for (const [i, j] of BASE) { bd[i]++; bd[j]++; }
    for (let i = 0; i < NODE.length; i++) if (bd[i] === 8) interior.push(i); }

  const history = [];          // one entry per tick: the simultaneous event set
  const trace = [];            // mean interior degree, to watch the attractor
  let pend = null;             // {addKey, addIJ, evicted:[[key,val],...]}
  let accepted = 0, rejected = 0, skipped = 0;
  // Rejections must be split by class: only EVICTION ticks failing would show
  // that the coupled move itself is blocked, which is the whole question.
  let propEv = 0, accEv = 0, rejEv = 0, propPlain = 0, accPlain = 0, rejPlain = 0;

  await runExperiment(`coupled add/sever, seed=${seed || 5150}`,
    Array.from({ length: ticks }, (_, k) => ({ k, label: 'tick ' + k })),
    () => {
      pend = null;
      const free = edges.filter(([k, i, j]) =>
        !active.has(k) && !active.has(j + ':' + (+k.split(':')[1] ^ 1)));
      if (!free.length) { skipped++; return; }
      const [k, i, j] = free[Math.floor(_psRand() * free.length)];
      const { d, at } = _degAt();
      const evicted = [];
      // exclusivity: adding (i,j) would breach the kissing cap at an endpoint
      if (d[i] >= CAP[i] && at[i].length) {
        const ek = at[i][Math.floor(_psRand() * at[i].length)];
        evicted.push([ek, active.get(ek)]);
      }
      if (d[j] >= CAP[j] && at[j].length) {
        const pool = at[j].filter(x => !evicted.some(e => e[0] === x));
        if (pool.length) {
          const ek = pool[Math.floor(_psRand() * pool.length)];
          evicted.push([ek, active.get(ek)]);
        }
      }
      active.set(k, [i, j]);
      for (const [ek] of evicted) active.delete(ek);
      if (evicted.length) propEv++; else propPlain++;
      pend = { addKey: k, addIJ: [i, j], evicted };
    },
    (it) => {
      if (!pend) return { html: '<span style="color:#5a6b7d">no free edge</span>', step: it.k };
      if (!legal(resid)) {                       // revert the WHOLE tick
        active.delete(pend.addKey);
        for (const [ek, val] of pend.evicted) active.set(ek, val);
        rejected++; pend.evicted.length ? rejEv++ : rejPlain++; freezeOff();
        return { html: '<span style="color:#ff5c5c">tick rejected'
                 + (pend.evicted.length ? ' (had eviction)' : '') + '</span>', step: it.k, kind: 'rej' };
      }
      accepted++; pend.evicted.length ? accEv++ : accPlain++;
      const ev = [{ kind: 'add', ij: pend.addIJ }];
      for (const [, val] of pend.evicted) ev.push({ kind: 'sever', ij: val });
      // positions are recorded in SOLVED coordinates, at this tick
      for (const e of ev) {
        const a = P[e.ij[0]], b = P[e.ij[1]];
        e.x = (a.x + b.x) / 2; e.y = (a.y + b.y) / 2; e.z = (a.z + b.z) / 2;
      }
      history.push({ tick: history.length, events: ev });   // ev = one ANTICHAIN
      const { d } = _degAt();
      // |S| is recorded from the FIRST accepted tick, so a saturated start
      // cannot hide: tick 0 must show nSC = 1.
      trace.push({ t: history.length - 1, nSC: active.size,
                   degInt: +(interior.reduce((s, i) => s + d[i], 0) / interior.length).toFixed(3) });
      return { html: `<span style="color:#7fd4a8">tick</span> |S|=${active.size} `
                     + `ev=${ev.length} degInt=${trace[trace.length - 1].degInt}`,
               step: it.k, kind: 'acc' };
    },
    { onFreeze: () => { freezeOff(); } });

  const { d } = _degAt();
  const histInt = {};
  for (const i of interior) histInt[d[i]] = (histInt[d[i]] || 0) + 1;
  const anti = {};
  for (const h of history) anti[h.events.length] = (anti[h.events.length] || 0) + 1;

  window._CP = {
    ticks, accepted, rejected, skipped,
    evictionTicks: { proposed: propEv, accepted: accEv, rejected: rejEv,
                     acceptRate: propEv ? +(accEv / propEv).toFixed(3) : null },
    plainTicks: { proposed: propPlain, accepted: accPlain, rejected: rejPlain,
                  acceptRate: propPlain ? +(accPlain / propPlain).toFixed(3) : null },
    activeSC: active.size, interiorNodes: interior.length,
    finalInteriorDegreeHist: histInt,
    antichainSizeHist: anti,
    trace,                                     // [{t, nSC, degInt}] from tick 0
    startedFromVacuum: { active0: 0, minSep: vacuumMinSep },
    legal: legal(resid),
    history
  };
  console.log('COUPLED', JSON.stringify({ accepted, rejected, anti, histInt,
    firstTrace: trace.slice(0, 5), lastTrace: trace.slice(-5) }));
  return window._CP;
};
console.log('coupled-scratch loaded');
