// TEMPORARY measurement harness — scratch, not part of the simulation.
// Read-only with respect to physics: it drives the existing fluxDynamics /
// modeCensus and records what they return. It changes no solver setting, adds
// no cutoff, and alters no acceptance rule.
//
// Purpose: raise the effective independent sample count for the mode-population
// triple. The earlier study took three snapshots of ONE trajectory, which share
// nearly all their solids (N_eff ~ 3). Here each seed is an independent
// configuration sampled exactly once (burn = steps-1).

window._INIT = async function (targetShells) {
  saveState = () => {};                 // worker tabs must not fight over localStorage
  freezeOff(); active.clear();          // drop any restored config (a stale one can be illegal)
  if (targetShells !== undefined && targetShells !== shells) setSize(targetShells);
  restate(); await settle();
  return { FROZEN, shells, nodes: NODE.length, active: active.size,
           legal: legal(resid), resid };
};

const _AXOF2 = d => (d[0] ? 'X' : (d[1] ? 'Y' : 'Z'));

// Axis balance of the shortcuts still available. Static pool is exactly
// balanced (76/76/76 at shells=1), so any skew here is dynamic: it is the
// lock-in mechanism, not a biased sampler.
function _freeByAxis() {
  const seen = new Set(), f = { X: 0, Y: 0, Z: 0 };
  for (const [k, j] of SCOPT) {
    const i = +k.split(':')[0], a = +k.split(':')[1];
    const pid = Math.min(i, j) + '-' + Math.max(i, j);
    if (seen.has(pid)) continue; seen.add(pid);
    if (active.has(k) || active.has(j + ':' + (a ^ 1))) continue;
    f[_AXOF2(AX[a])]++;
  }
  return f;
}

window._SEEDSTUDY = async function (seeds, steps, pAdd) {
  const out = []; window._SS = out; window._DONE = false;
  for (const sd of seeds) {
    const t0 = performance.now();
    await fluxDynamics(steps, pAdd, sd, steps - 1, 1);   // exactly one sample, at the end
    const c = modeCensus();
    const pop = { XY: 0, XZ: 0, YZ: 0 };
    for (const m of c) pop[m.mode]++;
    const tot = c.length;
    const sorted = Object.entries(pop).sort((a, b) => b[1] - a[1]);
    out.push({
      seed: sd, shells, n: active.size, solids: tot,
      tets: c.filter(m => m.type === 'tet').length,
      octs: c.filter(m => m.type === 'oct').length,
      pop, leader: tot ? sorted[0][0] : null,
      triple: tot ? sorted.map(e => +(e[1] / tot).toFixed(3)) : null,
      freeAx: _freeByAxis(),
      secs: +((performance.now() - t0) / 1000).toFixed(1)
    });
    console.log('SEED', sd, JSON.stringify(out[out.length - 1]));
  }
  window._DONE = true;
  return out;
};

// Two concurrent runs would mutate the same `active` map and silently produce
// garbage (EXP.i climbing past EXP.n is the tell). Refuse to start a second.
window._BUSY = false;
window._RUN = async function (targetShells, seeds, steps, pAdd) {
  if (window._BUSY) { console.error('REFUSED: a study is already running'); return; }
  window._BUSY = true;
  try {
    window._INITR = await window._INIT(targetShells);
    console.log('INIT', JSON.stringify(window._INITR));
    if (!window._INITR.legal) { console.error('INIT NOT LEGAL', window._INITR); return; }
    return await window._SEEDSTUDY(seeds, steps, pAdd);
  } finally { window._BUSY = false; }
};
console.log('study-scratch loaded');
