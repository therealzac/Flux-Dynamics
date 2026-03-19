// flux-demo-ui.js — Demo panels, profiling, choreo logging, pause/resume/stop

// Lightweight tick counter update — safe to call during backtrack retry loops.
// Updates the timeline scrubber and left panel title.
function _updateTickCounter() {
    const dpT = document.getElementById('dp-title');
    if (dpT) dpT.innerHTML = `${_planckSeconds} Flux Events<br><span style="font-size:0.7em; color:#8a9aaa; letter-spacing:0.05em;">${_demoTick} Planck Seconds</span><br><span style="font-size:0.55em; color:#556677; margin-top:-2px; display:block; line-height:1;">(Best ${_maxTickReached})</span>`;
    _updateTimelineScrubber();
}

// Builds the persistent meta rows below the tick counter.
// Returns HTML string with <br>-separated lines for: highest, current layer, options.
function _tickerMetaLines() {
    const s = 'font-size:0.65em; letter-spacing:0.03em;';

    // ── Highest ──
    const highest = `<span style="${s} color:#8a9aaa;">highest: ${_maxTickReached}</span>`;

    // ── Current Layer ──
    // Which tick we're deciding on, and how many times we've revisited it.
    let layerTick = _demoTick;
    let layerVisits = 0;
    if (_btActive || _bfsFailTick >= 0) {
        // During BFS, the anchor tick is _bfsFailTick - _bfsLayer
        layerTick = _bfsFailTick >= 0 ? _bfsFailTick - _bfsLayer : _demoTick;
        const fpSet = _btTriedFingerprints.get(layerTick);
        layerVisits = fpSet ? fpSet.size : 0;
    }
    const layerColor = _btActive ? '#ff9944' : '#556677';
    const layer = `<span style="${s} color:${layerColor};">layer: ps ${layerTick} (visit ${layerVisits + 1})</span>`;

    // ── Options this layer ──
    // How many valid options exist at the current tick, and which one we're on.
    let optCurrent = 0, optTotal = 0;
    if (_btActive || _bfsFailTick >= 0) {
        const fpSet = _btTriedFingerprints.get(layerTick);
        optCurrent = fpSet ? fpSet.size : 0;
        // Total = enumerated options if available, otherwise just show tried count
        if (_relayPhase === 'replaying' && _relayScoredQueue) {
            optTotal = optCurrent + (_relayScoredQueue.length - _relayScoredIndex);
        } else if (_relayPhase === 'enumerating' && _relayEnumFingerprints) {
            optTotal = optCurrent + _relayEnumFingerprints.size;
        } else {
            optTotal = optCurrent; // still discovering
        }
    }
    const optColor = _btActive ? '#66bbff' : '#556677';
    const opts = `<span style="${s} color:${optColor};">options: ${optCurrent}${optTotal > optCurrent ? '/' + optTotal : '+'}</span>`;

    return `<br>${highest}<br>${layer}<br>${opts}`;
}

// Force-update bottom-stats (shortcuts + density) regardless of simHalted.
// Called during playback/rewind so the popover reflects the restored state.
function _updateBottomStats() {
    const planckEl = document.getElementById('st-planck');
    if (planckEl) planckEl.textContent = _planckSeconds;
    const ticksEl = document.getElementById('st-ticks');
    if (ticksEl) ticksEl.textContent = _demoTick;
    const _allSCs = new Set([...activeSet, ...impliedSet, ...(typeof xonImpliedSet !== 'undefined' ? xonImpliedSet : [])]);
    const totalOpen = _allSCs.size;
    const scEl = document.getElementById('st-sc');
    if (scEl) scEl.textContent = totalOpen + ' / ' + ALL_SC.length;
    const densEl = document.getElementById('st-dens');
    if (densEl && typeof computeActualDensity === 'function') {
        const actual = (computeActualDensity() * 100).toFixed(4);
        const ideal = typeof computeIdealDensity === 'function' ? computeIdealDensity() * 100 : 74.048;
        const dev = Math.abs(parseFloat(actual) - ideal);
        densEl.textContent = actual + '%';
        densEl.style.color = dev < 0.001 ? '#6a8aaa' : dev < 0.01 ? '#ffaa44' : '#ff4444';
    }
    // Highest ps
    const hpEl = document.getElementById('st-highest-ps');
    if (hpEl) hpEl.textContent = _maxTickReached || 0;
    // Quark balance scores
    const visits = _actualizationVisits || _demoVisits;
    if (visits) {
        const types6 = ['pu1', 'pu2', 'pd', 'nd1', 'nd2', 'nu'];
        const calcEvenness = (arr) => {
            const m = arr.reduce((a, b) => a + b, 0) / arr.length;
            if (m === 0) return 0;
            const sd = Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length);
            return Math.max(0, 1 - sd / m);
        };
        const totals = [], protonPerFace = [], neutronPerFace = [];
        for (let f = 1; f <= 8; f++) {
            const v = visits[f];
            if (!v) continue;
            let ft = 0;
            for (const t of types6) ft += v[t] || 0;
            totals.push(ft);
            protonPerFace.push((v.pu1 || 0) + (v.pu2 || 0) + (v.pd || 0));
            neutronPerFace.push((v.nd1 || 0) + (v.nd2 || 0) + (v.nu || 0));
        }
        const evenness = totals.length > 0 ? calcEvenness(totals) : 0;
        const pEven = protonPerFace.length > 0 ? calcEvenness(protonPerFace) : 0;
        const nEven = neutronPerFace.length > 0 ? calcEvenness(neutronPerFace) : 0;
        const oEl = document.getElementById('st-balance-overall');
        if (oEl) oEl.textContent = (evenness * 100).toFixed(1) + '%';
        const pEl = document.getElementById('st-balance-proton');
        if (pEl) pEl.textContent = (pEven * 100).toFixed(1) + '%';
        const nEl = document.getElementById('st-balance-neutron');
        if (nEl) nEl.textContent = (nEven * 100).toFixed(1) + '%';
    }
}

function dumpProfile() {
    if (_tickCount === 0) { console.log('[PROFILE] No ticks recorded'); return; }
    const n = _tickCount;
    const total = _tickTotalMs;
    const ph = _profPhases;
    const phaseTot = ph.wb + ph.p0 + ph.p05 + ph.p1 + ph.p2 + ph.p3 + ph.solver + ph.render + ph.guards + (ph._temporalK||0) + (ph._uiUpdate||0);
    const pct = (v) => ((v / total) * 100).toFixed(1) + '%';
    const avg = (v) => (v / n).toFixed(1);
    console.log(`\n[PROFILE] ${n} ticks, total ${(total/1000).toFixed(1)}s, avg ${avg(total)}ms/tick, max ${_tickMaxMs.toFixed(0)}ms`);
    console.log(`  WB(setup):  ${avg(ph.wb)}ms/tick  ${pct(ph.wb)}`);
    console.log(`  PHASE 0:    ${avg(ph.p0)}ms/tick  ${pct(ph.p0)}`);
    console.log(`  PHASE 0.5:  ${avg(ph.p05)}ms/tick  ${pct(ph.p05)}`);
    console.log(`  PHASE 1:    ${avg(ph.p1)}ms/tick  ${pct(ph.p1)}`);
    console.log(`  PHASE 2:    ${avg(ph.p2)}ms/tick  ${pct(ph.p2)}  (gpuBatch: ${avg(ph.gpuBatch||0)}ms)`);
    console.log(`  PHASE 3:    ${avg(ph.p3)}ms/tick  ${pct(ph.p3)}`);
    // PHASE 3b/4/5: Removed — backtracker is the universal safety net
    console.log(`  Solver+glu: ${avg(ph.solver)}ms/tick  ${pct(ph.solver)}`);
    console.log(`  Render:     ${avg(ph.render)}ms/tick  ${pct(ph.render)}`);
    console.log(`  Guards:     ${avg(ph.guards)}ms/tick  ${pct(ph.guards)}`);
    console.log(`  TemporalK:  ${avg(ph._temporalK||0)}ms/tick  ${pct(ph._temporalK||0)}`);
    console.log(`  UI update:  ${avg(ph._uiUpdate||0)}ms/tick  ${pct(ph._uiUpdate||0)}`);
    console.log(`  Accounted:  ${avg(phaseTot)}ms/tick  ${pct(phaseTot)}`);
    if (typeof _solveCallCount !== 'undefined') {
        console.log(`  _solve() calls: ${_solveCallCount} (${(_solveCallCount/n).toFixed(1)}/tick), avg ${_solveCallCount?(_solveTotalMs/_solveCallCount).toFixed(1):'0'}ms`);
    }
    if (typeof _cmqCallCount !== 'undefined') {
        console.log(`  CMQ: ${_cmqCallCount} calls, ${_cmqCpuCount} CPU, ${_cmqCacheHits} cached, avg ${_cmqCpuCount?(_cmqTotalMs/_cmqCpuCount).toFixed(1):'0'}ms/CPU`);
    }
}

function resetProfile() {
    _tickTotalMs = 0; _tickCount = 0; _tickMaxMs = 0;
    for (const k in _profPhases) _profPhases[k] = 0;
    if (typeof _solveCallCount !== 'undefined') { _solveCallCount = 0; _solveTotalMs = 0; _solveMaxMs = 0; _solveIterTotal = 0; }
    if (typeof _cmqCallCount !== 'undefined') { _cmqCallCount = 0; _cmqCpuCount = 0; _cmqCacheHits = 0; _cmqTotalMs = 0; }
    console.log('[PROFILE] Counters reset');
}

function updateDemoPanel() {
    const epoch = Math.floor(_demoTick / 64);

    // ── Update left panel coverage bars (skip during non-PPO test execution) ──
    if (_testRunning && !_ppoTraining) return;
    const el = document.getElementById('dp-coverage-bars');
    if (!el) return;

    // Use actualization-based counts for display (falls back to _demoVisits)
    const visits = _actualizationVisits || _demoVisits;

    // Compute evenness: CV across all faces' total visits
    const types6 = ['pu1', 'pu2', 'pd', 'nd1', 'nd2', 'nu'];
    const totals = [];
    for (let f = 1; f <= 8; f++) {
        const v = visits[f];
        let faceTotal = 0;
        for (const t of types6) faceTotal += v[t] || 0;
        totals.push(faceTotal);
    }
    const mean = totals.reduce((a, b) => a + b, 0) / totals.length;
    const stddev = Math.sqrt(totals.reduce((s, v) => s + (v - mean) ** 2, 0) / totals.length);
    const cv = mean > 0 ? (stddev / mean) : 1; // no visits = worst evenness, not best
    const evenness = Math.max(0, 1 - cv);

    // Find max for bar normalization
    const _toHex6 = c => '#' + c.toString(16).padStart(6, '0');
    const typeColors = { pu1: _toHex6(QUARK_COLORS.pu1), pu2: _toHex6(QUARK_COLORS.pu2), pd: _toHex6(QUARK_COLORS.pd), nd1: _toHex6(QUARK_COLORS.nd1), nd2: _toHex6(QUARK_COLORS.nd2), nu: _toHex6(QUARK_COLORS.nu) };
    const typeLabels = { pu1: 'pu\u2081', pu2: 'pu\u2082', pd: 'pd', nd1: 'nd\u2081', nd2: 'nd\u2082', nu: 'nu' };
    let maxCount = 1;
    for (let f = 1; f <= 8; f++) {
        for (const t of types6) {
            maxCount = Math.max(maxCount, visits[f][t] || 0);
        }
    }

    // Build bars (6 per face)
    let html = '';
    for (let f = 1; f <= 8; f++) {
        const v = visits[f];
        let faceTotal = 0;
        for (const t of types6) faceTotal += v[t] || 0;
        const isA = [1, 3, 6, 8].includes(f);
        html += `<div style="display:flex; align-items:center; gap:1px;">`
            + `<span style="width:18px; color:${isA ? '#cc8866' : '#6688aa'}; font-size:8px; font-weight:bold;">F${f}</span>`;
        for (const t of types6) {
            html += `<div class="dp-bar-bg" style="flex:1;" title="${typeLabels[t]} ${v[t] || 0}"><div class="dp-bar-fill" style="width:${((v[t] || 0) / maxCount * 100).toFixed(1)}%; background:${typeColors[t]};"></div></div>`;
        }
        html += `<span style="width:22px; text-align:right; font-size:7px; color:var(--text-3);">${faceTotal}</span>`
            + `</div>`;
    }

    // ── Per-hadron evenness ──
    const protonPerFace = [], neutronPerFace = [];
    for (let f = 1; f <= 8; f++) {
        const v = visits[f];
        protonPerFace.push((v.pu1 || 0) + (v.pu2 || 0) + (v.pd || 0));
        neutronPerFace.push((v.nd1 || 0) + (v.nd2 || 0) + (v.nu || 0));
    }
    // Per-type global totals for 3-way evenness
    const typeTotals = {};
    for (const t of types6) typeTotals[t] = 0;
    for (let f = 1; f <= 8; f++) {
        for (const t of types6) typeTotals[t] += visits[f][t] || 0;
    }
    const calcEvenness = (arr) => {
        const m = arr.reduce((a, b) => a + b, 0) / arr.length;
        if (m === 0) return 0;
        const sd = Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length);
        return Math.max(0, 1 - sd / m);
    };
    const protonEvenness = calcEvenness(protonPerFace);
    const neutronEvenness = calcEvenness(neutronPerFace);
    // 3-way type evenness within each hadron
    const pTotal = typeTotals.pu1 + typeTotals.pu2 + typeTotals.pd;
    const nTotal = typeTotals.nd1 + typeTotals.nd2 + typeTotals.nu;
    const pEven = pTotal > 0 ? 1 - (Math.abs(typeTotals.pu1/pTotal - 1/3) + Math.abs(typeTotals.pu2/pTotal - 1/3) + Math.abs(typeTotals.pd/pTotal - 1/3)) : 0;
    const nEven = nTotal > 0 ? 1 - (Math.abs(typeTotals.nd1/nTotal - 1/3) + Math.abs(typeTotals.nd2/nTotal - 1/3) + Math.abs(typeTotals.nu/nTotal - 1/3)) : 0;
    const ratioAccuracy = (pEven + nEven) / 2;
    const evColor = (e) => e > 0.99 ? '#66dd66' : e > 0.95 ? '#ccaa66' : '#ff6644';

    // Evenness + rule compliance
    html += `<div style="margin-top:6px; border-top:1px solid rgba(80,100,120,0.25); padding-top:4px;">`;
    html += `<div style="display:flex; justify-content:space-between; font-size:9px;">`
        + `<span style="color:var(--text-3);">overall</span>`
        + `<span style="color:${evColor(evenness)}; font-weight:bold;">${(evenness * 100).toFixed(1)}%</span>`
        + `</div>`;
    html += `<div style="display:flex; justify-content:space-between; font-size:9px;">`
        + `<span style="color:#cc8866;">proton</span>`
        + `<span style="color:${evColor(protonEvenness)}; font-weight:bold;">${(protonEvenness * 100).toFixed(1)}%</span>`
        + `</div>`;
    html += `<div style="display:flex; justify-content:space-between; font-size:9px;">`
        + `<span style="color:#6688aa;">neutron</span>`
        + `<span style="color:${evColor(neutronEvenness)}; font-weight:bold;">${(neutronEvenness * 100).toFixed(1)}%</span>`
        + `</div>`;
    html += `<div style="display:flex; justify-content:space-between; font-size:9px;">`
        + `<span style="color:var(--text-3);">p 1:1:1</span>`
        + `<span style="color:${pEven > 0.7 ? '#66dd66' : '#ccaa66'}; font-weight:bold;">${(pEven * 100).toFixed(1)}%</span>`
        + `</div>`;
    html += `<div style="display:flex; justify-content:space-between; font-size:9px;">`
        + `<span style="color:var(--text-3);">n 1:1:1</span>`
        + `<span style="color:${nEven > 0.7 ? '#66dd66' : '#ccaa66'}; font-weight:bold;">${(nEven * 100).toFixed(1)}%</span>`
        + `</div>`;

    // ── Ratio accuracy history sparkline (fixed single row, sliding window) ──
    _demoTypeBalanceHistory.push(ratioAccuracy * 100);
    const SPARK_SLOTS = 28;  // fits one row at 22px in ~345px panel
    const hist = _demoTypeBalanceHistory;
    const sparkData = hist.slice(-SPARK_SLOTS);
    const sparkMin = Math.min(...sparkData, 90);
    const sparkMax = 100;
    const SPARK = ['\u2581', '\u2582', '\u2583', '\u2584', '\u2585', '\u2586', '\u2587', '\u2588'];
    let sparkline = '';
    for (const v of sparkData) {
        const norm = Math.max(0, Math.min(1, (v - sparkMin) / (sparkMax - sparkMin)));
        const idx = Math.min(7, Math.floor(norm * 7.99));
        const c = v >= 99.5 ? '#66dd66' : v >= 96 ? '#ccaa66' : '#cc8855';
        sparkline += `<span style="color:${c};">${SPARK[idx]}</span>`;
    }
    html += `<div style="margin-top:4px; overflow:hidden; width:100%;">`
        + `<div style="font-size:8px; color:var(--text-3); margin-bottom:2px;">ratio accuracy (last ${Math.min(hist.length, SPARK_SLOTS)} windows)</div>`
        + `<div style="font-size:22px; letter-spacing:-1px; line-height:1; font-family:monospace; white-space:nowrap; overflow:hidden;">${sparkline}</div>`
        + `<div style="display:flex; justify-content:space-between; font-size:7px; color:#445566; margin-top:2px;">`
        + `<span>${sparkMin.toFixed(0)}%</span><span>100%</span></div>`
        + `</div>`;
    // Rule compliance indicators
    const rules = [
        { name: 'anti-phase', ok: true },  // guaranteed by schedule construction
        { name: 'pauli', ok: _demoPauliViolations === 0 },
        { name: 'spread', ok: _demoSpreadViolations === 0 },
        { name: 'coverage', ok: evenness > 0.9 },
    ];
    html += `<div style="margin-top:3px; font-size:8px; color:var(--text-3);">`;
    for (const r of rules) {
        html += `<span style="color:${r.ok ? '#44aa66' : '#cc4444'}; margin-right:6px;">${r.ok ? '\u2713' : '\u2717'} ${r.name}</span>`;
    }
    html += `</div></div>`;
    el.innerHTML = html;
}

// ── Edge Balance Panel ────────────────────────────────────────────────────
function _updateEdgeBalancePanel() {
    const el = document.getElementById('dp-edge-balance');
    if (!el || !_edgeBalance || _edgeBalance.size === 0) return;

    const types = ['pu1', 'pu2', 'pd', 'nd1', 'nd2', 'nu'];
    const _toH = c => '#' + c.toString(16).padStart(6, '0');
    const typeColors = { pu1: _toH(QUARK_COLORS.pu1), pu2: _toH(QUARK_COLORS.pu2), pd: _toH(QUARK_COLORS.pd), nd1: _toH(QUARK_COLORS.nd1), nd2: _toH(QUARK_COLORS.nd2), nu: _toH(QUARK_COLORS.nu) };

    // Sort edges: oct↔oct first, then oct↔ext, by pairId
    // pairId() returns a number (a*20000+b), so decode it back to node pair
    const octOct = [], octExt = [];
    for (const [pid, counts] of _edgeBalance) {
        const a = Math.floor(pid / 20000), b = pid % 20000;
        if (_octNodeSet && _octNodeSet.has(a) && _octNodeSet.has(b)) {
            octOct.push([pid, counts, a, b]);
        } else {
            octExt.push([pid, counts, a, b]);
        }
    }
    octOct.sort((a, b) => a[0] - b[0]);
    octExt.sort((a, b) => a[0] - b[0]);

    // Overall evenness score
    const evenness = _computeEdgeEvenness();
    const wColor = evenness > 0.95 ? '#66dd66' : evenness > 0.8 ? '#ccaa66' : '#ff6644';

    let html = `<div style="display:flex; justify-content:space-between; margin-bottom:4px;">`;
    html += `<span style="color:var(--text-3); font-size:9px;">evenness</span>`;
    html += `<span style="color:${wColor}; font-weight:bold; font-size:9px;">${(evenness * 100).toFixed(1)}%</span>`;
    html += `</div>`;

    // Render edge rows
    const renderEdge = ([pid, counts, a, b], label) => {
        if (counts.total === 0) {
            html += `<div style="display:flex; align-items:center; gap:1px;">`;
            html += `<span style="width:36px; color:var(--text-3); font-size:7px; font-family:var(--font-mono);">${label}</span>`;
            html += `<span style="color:var(--text-3); font-size:7px;">—</span>`;
            html += `</div>`;
            return;
        }
        const maxT = Math.max(1, ...types.map(t => counts[t]));
        // Compute per-edge type evenness
        const ideal = counts.total / 6;
        let dev = 0;
        for (const t of types) dev += Math.abs(counts[t] - ideal);
        const maxDev = counts.total * (10 / 6);
        const edgeW = 1 - (dev / maxDev);
        const eColor = edgeW > 0.95 ? '#66dd66' : edgeW > 0.8 ? '#ccaa66' : '#ff6644';

        // Directional balance indicator
        const fwd = counts.fwd || 0, rev = counts.rev || 0;
        const dirTotal = fwd + rev;
        const dirPct = dirTotal > 0 ? (fwd / dirTotal * 100) : 50;
        const dirSkew = Math.abs(dirPct - 50);
        const dirColor = dirSkew < 10 ? '#66dd66' : dirSkew < 25 ? '#ccaa66' : '#ff6644';
        const dirArrow = dirPct > 55 ? '→' : dirPct < 45 ? '←' : '⇄';

        html += `<div style="display:flex; align-items:center; gap:1px;">`;
        html += `<span style="width:36px; color:${eColor}; font-size:7px; font-family:var(--font-mono);">${label}</span>`;
        for (const t of types) {
            const pct = (counts[t] / maxT * 100).toFixed(1);
            html += `<div class="dp-bar-bg" style="flex:1;" title="${t}: ${counts[t]}"><div class="dp-bar-fill" style="width:${pct}%; background:${typeColors[t]};"></div></div>`;
        }
        html += `<span style="width:14px; text-align:center; font-size:7px; color:${dirColor};" title="fwd ${fwd} / rev ${rev} (${dirPct.toFixed(0)}%→)">${dirArrow}</span>`;
        html += `<span style="width:20px; text-align:right; font-size:6px; color:var(--text-3);">${counts.total}</span>`;
        html += `</div>`;
    };

    if (octOct.length > 0) {
        html += `<div style="color:var(--text-3); font-size:7px; margin:3px 0 1px;">oct↔oct</div>`;
        for (const e of octOct) renderEdge(e, `${e[2]}–${e[3]}`);
    }
    if (octExt.length > 0) {
        html += `<div style="color:var(--text-3); font-size:7px; margin:3px 0 1px;">oct↔ext</div>`;
        for (const e of octExt) renderEdge(e, `${e[2]}–${e[3]}`);
    }

    el.innerHTML = html;
}

// ── Ejection Balance Panel ─────────────────────────────────────────────────
function _updateEjectionBalancePanel() {
    const el = document.getElementById('dp-ejection-balance');
    if (!el || !_ejectionBalance || _ejectionBalance.size === 0) return;

    // Overall ejection evenness
    const evenness = _computeEjectionEvenness();
    const eColor = evenness > 0.95 ? '#66dd66' : evenness > 0.8 ? '#ccaa66' : '#ff6644';

    // Collect edges with ejection data, decode pairId
    const edges = [];
    let maxCount = 1;
    for (const [pid, count] of _ejectionBalance) {
        const a = Math.floor(pid / 20000), b = pid % 20000;
        edges.push([pid, count, a, b]);
        if (count > maxCount) maxCount = count;
    }
    // Sort: oct↔oct first, then oct↔ext
    edges.sort((a, b) => a[0] - b[0]);

    let html = `<div style="display:flex; justify-content:space-between; margin-bottom:4px;">`;
    html += `<span style="color:var(--text-3); font-size:9px;">evenness</span>`;
    html += `<span style="color:${eColor}; font-weight:bold; font-size:9px;">${(evenness * 100).toFixed(1)}%</span>`;
    html += `</div>`;

    // All edges in _ejectionBalance are chirality-eligible — show all of them
    for (const [pid, count, a, b] of edges) {
        const pct = maxCount > 0 ? (count / maxCount * 100).toFixed(1) : '0';
        html += `<div style="display:flex; align-items:center; gap:1px;">`;
        html += `<span style="width:36px; color:var(--text-3); font-size:7px; font-family:var(--font-mono);">${a}–${b}</span>`;
        html += `<div class="dp-bar-bg" style="flex:1;" title="ejections: ${count}"><div class="dp-bar-fill" style="width:${pct}%; background:rgba(255,255,255,0.45);"></div></div>`;
        html += `<span style="width:20px; text-align:right; font-size:6px; color:var(--text-3);">${count}</span>`;
        html += `</div>`;
    }

    html += `<div style="font-size:7px; color:var(--text-3); margin-top:3px; font-style:italic;">chirality-forbidden edges (a↔b, c↔d) omitted</div>`;

    el.innerHTML = html;
}

// ── Balance History Chart ──

function _drawBalanceChart() {
    const canvas = document.getElementById('dp-balance-chart');
    if (!canvas || _balanceHistory.length < 2) return;

    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const PAD = { top: 42, right: 14, bottom: 28, left: 46 };

    ctx.clearRect(0, 0, W, H);

    // Filter data by timeframe
    let data = _balanceHistory;
    if (_balanceTimeframe === '1000') {
        const cutoff = _planckSeconds - 1000;
        data = data.filter(d => d.ps >= cutoff);
    } else if (_balanceTimeframe === '250') {
        const cutoff = _planckSeconds - 250;
        data = data.filter(d => d.ps >= cutoff);
    }
    if (data.length < 2) return;

    const plotW = W - PAD.left - PAD.right;
    const plotH = H - PAD.top - PAD.bottom;
    const minPS = data[0].ps;
    const maxPS = data[data.length - 1].ps;
    const psRange = maxPS - minPS || 1;

    const toX = (ps) => PAD.left + ((ps - minPS) / psRange) * plotW;
    const toY = (v) => PAD.top + plotH - (Math.min(100, Math.max(0, v)) / 100) * plotH;

    // Background
    ctx.fillStyle = 'rgba(8, 12, 20, 0.6)';
    ctx.fillRect(0, 0, W, H);

    // Gridlines
    ctx.setLineDash([2, 2]);
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 1;
    for (const pct of [25, 50, 75]) {
        const y = toY(pct);
        ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(W - PAD.right, y); ctx.stroke();
    }
    ctx.setLineDash([]);

    // Axes
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.beginPath();
    ctx.moveTo(PAD.left, PAD.top); ctx.lineTo(PAD.left, H - PAD.bottom);
    ctx.lineTo(W - PAD.right, H - PAD.bottom);
    ctx.stroke();

    // Y-axis labels
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.font = '13px monospace';
    ctx.textAlign = 'right';
    for (const pct of [0, 50, 100]) {
        ctx.fillText(pct + '%', PAD.left - 5, toY(pct) + 5);
    }

    // Series definitions (avg computed inline from the 3 metrics)
    const series = [
        { key: 'quark',    color: '#66dd66', label: 'Quark' },
        { key: 'edge',     color: '#6699ff', label: 'Edge' },
        { key: 'ejection', color: '#ff8844', label: 'Ejection' },
        { key: 'avg',      color: '#ffffff', label: 'Avg' },
    ];

    // Downsample if too many points
    let drawData = data;
    if (data.length > plotW) {
        const step = data.length / plotW;
        drawData = [];
        for (let i = 0; i < plotW; i++) drawData.push(data[Math.floor(i * step)]);
    }

    // Draw lines
    for (const s of series) {
        ctx.strokeStyle = s.color;
        ctx.lineWidth = s.key === 'avg' ? 2.8 : 2.2;
        ctx.beginPath();
        for (let i = 0; i < drawData.length; i++) {
            const d = drawData[i];
            const x = toX(d.ps);
            const v = s.key === 'avg' ? (d.quark + d.edge + d.ejection) / 3 : d[s.key];
            const y = toY(v);
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }

    // Legend with latest values — 2 rows, 2 items each
    ctx.font = '13px monospace';
    ctx.textAlign = 'left';
    const last = data[data.length - 1];
    const rowH = 16;
    for (let i = 0; i < series.length; i++) {
        const s = series[i];
        const row = Math.floor(i / 2);
        const col = i % 2;
        const lx = PAD.left + 4 + col * ((W - PAD.left - PAD.right) / 2);
        const ly = PAD.top - 26 + row * rowH;
        const latest = s.key === 'avg' ? (last.quark + last.edge + last.ejection) / 3 : last[s.key];
        // Colored swatch
        ctx.fillStyle = s.color;
        ctx.fillRect(lx, ly - 7, 12, 7);
        // Label + value
        ctx.fillStyle = s.color;
        ctx.fillText(s.label, lx + 15, ly);
        ctx.fillStyle = 'rgba(255,255,255,0.7)';
        ctx.fillText(latest.toFixed(1) + '%', lx + 15 + ctx.measureText(s.label).width + 5, ly);
    }
}

function _initBalanceChartControls() {
    const wrap = document.getElementById('dp-balance-timeframe');
    if (!wrap) return;
    wrap.addEventListener('click', (e) => {
        const btn = e.target.closest('.dp-tf-btn');
        if (!btn) return;
        _balanceTimeframe = btn.dataset.tf;
        wrap.querySelectorAll('.dp-tf-btn').forEach(b => {
            b.classList.toggle('dp-tf-active', b === btn);
            b.style.background = b === btn ? 'rgba(255,255,255,0.08)' : 'transparent';
            b.style.color = b === btn ? 'var(--text-2)' : 'var(--text-3)';
        });
        _drawBalanceChart();
    });
}

// Init controls on load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', _initBalanceChartControls);
} else {
    _initBalanceChartControls();
}

// ── Choreographer logging helper ──
function _logChoreo(msg) {
    _choreoLog.push({ tick: _demoTick, msg });
    if (_choreoLog.length > _CHOREO_LOG_MAX) _choreoLog.shift();
}

// ── Log PHASE 2 summary: called after bipartite matching + fallbacks ──
function _logPhase2Summary(octPlans) {
    const _QL = { pu1: 'pu\u2081', pu2: 'pu\u2082', pd: 'pd', nd1: 'nd\u2081', nd2: 'nd\u2082', nu: 'nu' };
    const lines = [];
    for (const plan of octPlans) {
        const x = plan.xon;
        const idx = _demoXons.indexOf(x);
        const label = x._mode === 'oct' ? 'idle' : x._mode === 'weak' ? 'weak' :
                      x._quarkType ? _QL[x._quarkType] || x._quarkType : x._mode;
        const cands = plan.candidates.length;
        if (plan.assigned) {
            lines.push(`X${idx}(${label}) n${plan.fromNode}: ${cands}c->n${plan.assigned.node}`);
        } else if (plan.idleTet) {
            lines.push(`X${idx}(${label}) n${plan.fromNode}: ${cands}c->tet f${x._assignedFace}`);
        } else {
            const reasons = [];
            if (cands === 0) reasons.push('0 cands');
            else reasons.push(`${cands}c taken`);
            if (x._evictedThisTick) reasons.push('evicted');
            lines.push(`X${idx}(${label}) n${plan.fromNode}: STUCK(${reasons.join(',')})`);
        }
    }
    _logChoreo('PH2: ' + lines.join(' | '));
}

// ── Xon panel update (sidebar) ──
// ── Xon Panel: build-once / update-in-place ──
// Structure is built once on first call; subsequent calls only patch dynamic values
// (node numbers, colors, highlight state) so sliders remain interactive.
let _xonPanelBuilt = false;

const _XP_ROLE_DISPLAY = {
    pu1: 'Prot. ▲1', pu2: 'Prot. ▲2', pd: 'Prot. ▼',
    nd1: 'Neut. ▼1', nd2: 'Neut. ▼2', nu: 'Neut. ▲',
    oct: 'Oct', gluon: 'Gluon', weak: 'Weak',
};
const _XP_ROLE_INITIALS = {
    pu1: 'PU1', pu2: 'PU2', pd: 'PD',
    nd1: 'ND1', nd2: 'ND2', nu: 'NU',
    oct: 'Oct', gluon: 'Glu', weak: 'Wk',
};
// _XP_ROLE_PATTERNS removed — traversal names no longer shown on cards

function _xpRoleColor(key) {
    if (key === 'oct') return 0xffffff;
    if (key === 'gluon') return typeof GLUON_COLOR !== 'undefined' ? GLUON_COLOR : 0x80ff00;
    if (key === 'weak') return typeof WEAK_FORCE_COLOR !== 'undefined' ? WEAK_FORCE_COLOR : 0x7f00ff;
    return (typeof QUARK_COLORS !== 'undefined' && QUARK_COLORS[key]) || 0x888888;
}
function _xpHex(c) { return '#' + (c || 0).toString(16).padStart(6, '0'); }
function _xpPicTextColor(c) {
    const r = ((c >> 16) & 0xff) / 255, g = ((c >> 8) & 0xff) / 255, b = (c & 0xff) / 255;
    return (0.299*r + 0.587*g + 0.114*b) > 0.45 ? 'rgba(0,0,0,0.55)' : 'rgba(255,255,255,0.65)';
}

function _buildXonPanel() {
    const _STAT_LABELS = {
        pu1:'Prot. ▲1', pu2:'Prot. ▲2', pd:'Prot. ▼',
        nd1:'Neut. ▼1', nd2:'Neut. ▼2', nu:'Neut. ▲',
        oct:'Oct', gluon:'Gluon', weak:'Weak',
    };
    const _ALL_ROLES = ['pu1','pu2','pd','nd1','nd2','nu','oct','gluon','weak'];

    // ── Build stats grid HTML (shared by both sections) ──
    function _buildStatsGrid(idSuffix) {
        let shtml = '';
        for (const key of _ALL_ROLES) {
            shtml += `<div style="display:flex;justify-content:space-between;align-items:baseline;gap:4px;">`
                + `<span style="font-size:10px;color:var(--text-3);font-family:var(--font-sans);">${_STAT_LABELS[key]}</span>`
                + `<span id="role-stat-${key}${idSuffix}" style="font-size:10px;color:var(--text-2);font-family:var(--font-sans);font-weight:600;">0</span>`
                + `</div>`;
        }
        return shtml;
    }

    // ── Role cards (Quark Types section) ──
    const roleEl = document.getElementById('role-card-list');
    if (roleEl) {
        let rhtml = '';
        for (const key of _ALL_ROLES) {
            const color = _xpRoleColor(key);
            const hex = _xpHex(color);
            const opVal = Math.round((_roleOpacity[key] != null ? _roleOpacity[key] : 1) * 100);
            rhtml += `<div class="role-card" data-role="${key}">`
                + `<div class="card-pic" style="background:${hex};color:${_xpPicTextColor(color)};">${_XP_ROLE_INITIALS[key]}</div>`
                + `<span class="role-card-name">${_XP_ROLE_DISPLAY[key]}</span>`
                + `<button class="card-sm-btn" data-action="solo" title="Solo">S</button>`
                + `<button class="card-sm-btn" data-action="mute" title="Mute">M</button>`
                + `<input type="range" class="op-slider" id="role-opacity-${key}" min="0" max="100" value="${opVal}" step="1">`
                + `</div>`;
        }
        roleEl.innerHTML = rhtml;

        // Delegate role slider + solo/mute events
        roleEl.addEventListener('input', (e) => {
            if (!e.target.classList.contains('op-slider')) return;
            const card = e.target.closest('.role-card');
            if (!card) return;
            const key = card.dataset.role;
            if (key) _roleOpacity[key] = (+e.target.value) / 100;
        });
        roleEl.addEventListener('click', (e) => {
            const btn = e.target.closest('.card-sm-btn');
            if (!btn) return;
            const card = btn.closest('.role-card');
            if (!card) return;
            const key = card.dataset.role;
            if (!key) return;
            _handleSoloMute('role', key, btn.dataset.action, _ALL_ROLES);
        });
    }

    // ── Role stats grid (Quark Types totals) ──
    const statsBar2 = document.getElementById('role-stats-bar-2');
    if (statsBar2) statsBar2.innerHTML = _buildStatsGrid('-b');

    // ── Xon cards ──
    const listEl = document.getElementById('xon-card-list');
    if (!listEl) return;

    let html = '';
    for (let i = 0; i < 6; i++) {
        html += `<div class="xon-card" data-xon-idx="${i}">`
            + `<div class="card-pic card-pic-xon" id="xon-pic-${i}" style="background:#888;">X${i}</div>`
            + `<span class="xon-card-id">X${i}</span>`
            + `<div class="card-pic-sm" id="xon-node-pic-${i}">?</div>`
            + `<span class="xon-card-node-label" id="xon-node-lbl-${i}">Node ?</span>`
            + `<button class="card-sm-btn" data-action="solo" title="Solo">S</button>`
            + `<button class="card-sm-btn" data-action="mute" title="Mute">M</button>`
            + `<input type="range" class="op-slider" id="xon-opacity-${i}" min="0" max="100" value="100" step="1">`
            + `</div>`;
    }
    listEl.innerHTML = html;

    // Click + opacity slider + solo/mute delegation
    listEl.addEventListener('mousedown', (e) => {
        if (e.target.classList.contains('op-slider') || e.target.classList.contains('card-sm-btn')) return;
        const card = e.target.closest('.xon-card');
        if (!card) return;
        const idx = parseInt(card.dataset.xonIdx, 10);
        if (!isNaN(idx)) _highlightXon(idx);
    });
    listEl.addEventListener('input', (e) => {
        if (!e.target.classList.contains('op-slider')) return;
        const card = e.target.closest('.xon-card');
        if (!card) return;
        const idx = parseInt(card.dataset.xonIdx, 10);
        if (!isNaN(idx)) _xonOpacity[idx] = (+e.target.value) / 100;
    });
    listEl.addEventListener('click', (e) => {
        const btn = e.target.closest('.card-sm-btn');
        if (!btn) return;
        const card = btn.closest('.xon-card');
        if (!card) return;
        const idx = parseInt(card.dataset.xonIdx, 10);
        if (!isNaN(idx)) _handleSoloMute('xon', idx, btn.dataset.action, [0,1,2,3,4,5]);
    });

    // ── Role stats grid (Xons totals) ──
    const statsBar = document.getElementById('role-stats-bar');
    if (statsBar) statsBar.innerHTML = _buildStatsGrid('');

    _xonPanelBuilt = true;
}

// ── Solo / Mute logic ──
function _handleSoloMute(type, key, action, allKeys) {
    if (type === 'role') {
        if (action === 'solo') {
            // If already solo'd on this key, un-solo (restore all to 100%)
            const isSolo = allKeys.every(k => k === key ? _roleOpacity[k] === 1 : _roleOpacity[k] === 0);
            for (const k of allKeys) {
                _roleOpacity[k] = isSolo ? 1 : (k === key ? 1 : 0);
                const sl = document.getElementById(`role-opacity-${k}`);
                if (sl) sl.value = Math.round(_roleOpacity[k] * 100);
            }
        } else if (action === 'mute') {
            _roleOpacity[key] = _roleOpacity[key] === 0 ? 1 : 0;
            const sl = document.getElementById(`role-opacity-${key}`);
            if (sl) sl.value = Math.round(_roleOpacity[key] * 100);
        }
        _refreshSoloMuteBtns('role', allKeys);
    } else {
        if (action === 'solo') {
            const isSolo = allKeys.every(k => k === key ? _xonOpacity[k] === 1 : _xonOpacity[k] === 0);
            for (const k of allKeys) {
                _xonOpacity[k] = isSolo ? 1 : (k === key ? 1 : 0);
                const sl = document.getElementById(`xon-opacity-${k}`);
                if (sl) sl.value = Math.round(_xonOpacity[k] * 100);
            }
        } else if (action === 'mute') {
            _xonOpacity[key] = _xonOpacity[key] === 0 ? 1 : 0;
            const sl = document.getElementById(`xon-opacity-${key}`);
            if (sl) sl.value = Math.round(_xonOpacity[key] * 100);
        }
        _refreshSoloMuteBtns('xon', allKeys);
    }
}

function _refreshSoloMuteBtns(type, allKeys) {
    if (type === 'role') {
        const listEl = document.getElementById('role-card-list');
        if (!listEl) return;
        for (const card of listEl.children) {
            const key = card.dataset.role;
            if (!key) continue;
            const btns = card.querySelectorAll('.card-sm-btn');
            const muted = _roleOpacity[key] === 0;
            const soloed = allKeys.filter(k => _roleOpacity[k] > 0).length === 1 && _roleOpacity[key] > 0;
            for (const b of btns) {
                b.classList.remove('active-solo', 'active-mute');
                if (b.dataset.action === 'solo' && soloed) b.classList.add('active-solo');
                if (b.dataset.action === 'mute' && muted) b.classList.add('active-mute');
            }
        }
    } else {
        const listEl = document.getElementById('xon-card-list');
        if (!listEl) return;
        for (const card of listEl.children) {
            const idx = parseInt(card.dataset.xonIdx, 10);
            if (isNaN(idx)) continue;
            const btns = card.querySelectorAll('.card-sm-btn');
            const muted = _xonOpacity[idx] === 0;
            const soloed = allKeys.filter(k => _xonOpacity[k] > 0).length === 1 && _xonOpacity[idx] > 0;
            for (const b of btns) {
                b.classList.remove('active-solo', 'active-mute');
                if (b.dataset.action === 'solo' && soloed) b.classList.add('active-solo');
                if (b.dataset.action === 'mute' && muted) b.classList.add('active-mute');
            }
        }
    }
}

function updateXonPanel() {
    if (_testRunning) return;
    const panel = document.getElementById('xon-panel');
    if (!panel) return;
    panel.style.display = (_demoActive || _playbackMode) ? 'block' : 'none';
    if (!_demoActive && !_playbackMode) return;

    // Build structure once; after that only patch dynamic values
    if (!_xonPanelBuilt) _buildXonPanel();

    // ── Update role stats bars in-place (both Xons and Quark Types totals) ──
    const _roleKeys = ['pu1','pu2','pd','nd1','nd2','nu','oct','gluon','weak'];
    for (const key of _roleKeys) {
        const v = _globalRoleStats[key] || 0;
        const t = v >= 1000 ? (v / 1000).toFixed(1) + 'k' : '' + v;
        const el1 = document.getElementById(`role-stat-${key}`);
        if (el1 && el1.textContent !== t) el1.textContent = t;
        const el2 = document.getElementById(`role-stat-${key}-b`);
        if (el2 && el2.textContent !== t) el2.textContent = t;
    }

    // ── Update xon cards in-place (node, color, highlight) ──
    for (let i = 0; i < _demoXons.length; i++) {
        const x = _demoXons[i];
        if (!x || !x.alive) continue;

        const role = typeof _xonRole === 'function' ? _xonRole(x) : 'oct';
        const color = _xpRoleColor(role);
        const hex = _xpHex(color);
        const tc = _xpPicTextColor(color);

        // Profile pic color
        const pic = document.getElementById(`xon-pic-${i}`);
        if (pic) { pic.style.background = hex; pic.style.color = tc; }

        // Node number
        const nodePic = document.getElementById(`xon-node-pic-${i}`);
        if (nodePic && nodePic.textContent !== '' + x.node) nodePic.textContent = x.node;
        const nodeLbl = document.getElementById(`xon-node-lbl-${i}`);
        if (nodeLbl) { const t = 'Node ' + x.node; if (nodeLbl.textContent !== t) nodeLbl.textContent = t; }

        // Highlight state
        const card = pic?.closest('.xon-card');
        if (card) {
            const hl = _xonHighlightTimers.has(i);
            if (hl && !card.classList.contains('highlighted')) card.classList.add('highlighted');
            else if (!hl && card.classList.contains('highlighted')) card.classList.remove('highlighted');
        }
    }
}

function _highlightXon(idx) {
    if (idx < 0 || idx >= _demoXons.length) return;
    const xon = _demoXons[idx];
    if (!xon) return;

    // Set highlight timer — decayed per-frame in _tickDemoXons
    xon._highlightT = 2.0; // seconds

    // Track which xons are highlighted for button border styling
    if (_xonHighlightTimers.has(idx)) clearTimeout(_xonHighlightTimers.get(idx));
    _xonHighlightTimers.set(idx, setTimeout(() => _xonHighlightTimers.delete(idx), 2000));
}

function _showReplayCorruption(tick, msg) {
    // Use sim-start-buttons (always visible) as anchor, fallback to pause button
    const anchor = document.getElementById('sim-start-buttons') || document.getElementById('btn-nucleus-pause');
    if (!anchor) return;
    let container = document.getElementById('replay-corruption-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'replay-corruption-container';
        container.style.cssText = 'margin-top:4px;';
        anchor.parentNode.insertBefore(container, anchor.nextSibling);
    }
    container.innerHTML = `<span style="color:#ff4444;font-weight:bold;font-size:11px;">CORRUPTED @ t=${tick}</span>` +
        `<button id="btn-clear-replay-test" style="margin-left:8px;font-size:10px;padding:2px 8px;cursor:pointer;background:#333;color:#ccc;border:1px solid #555;border-radius:3px;">Clear test</button>` +
        `<div style="color:#ff6666;font-size:9px;margin-top:2px;max-width:300px;word-wrap:break-word;">${msg}</div>`;
    document.getElementById('btn-clear-replay-test')?.addEventListener('click', function() {
        localStorage.removeItem('flux_replay_test');
        _clearReplayCorruption();
        _guardHardStop = false;
    });
    console.error(`[REPLAY GUARD] Corruption detected at tick ${tick}: ${msg}`);
}

function _clearReplayCorruption() {
    const el = document.getElementById('replay-corruption-container');
    if (el) el.remove();
}

function pauseDemo() {
    _demoPaused = true;
    if (_demoInterval) { clearInterval(_demoInterval); _demoInterval = null; }
    if (_demoUncappedId) { clearTimeout(_demoUncappedId); _demoUncappedId = null; }
    if (typeof _updateLatticeSliderLock === 'function') _updateLatticeSliderLock();
}
function resumeDemo() {
    _demoPaused = false;
    if (typeof _updateLatticeSliderLock === 'function') _updateLatticeSliderLock();
    simHalted = false;
    const _pb = document.getElementById('btn-nucleus-pause');
    if (_pb) { _pb.textContent = '\u23F8'; _pb.title = 'Pause simulation'; }
    if (_demoActive && !_demoInterval && !_demoUncappedId) {
        // Unified: demoTick handles both replay cursor and live play internally.
        const intervalMs = _getDemoIntervalMs();
        if (intervalMs === 0) {
            _demoUncappedId = setTimeout(_demoUncappedLoop, 0);
        } else {
            _demoInterval = setInterval(demoTick, intervalMs);
        }
    }
}
function isDemoPaused() {
    return _demoPaused;
}

function stopDemo() {
    _demoActive = false;
    _demoPaused = false;
    _xonPanelBuilt = false; // rebuild panel on next demo start
    // Only reset visual defaults flag when sweep is fully over —
    // mid-sweep seed transitions must NOT reapply slider defaults.
    if (!_sweepActive) _demoOpDefaultsApplied = false;
    // Clear council replay state if active
    _sweepReplayActive = false;
    _sweepReplayMember = null;
    if (typeof _updateLatticeSliderLock === 'function') _updateLatticeSliderLock();
    _setSimUIActive(false);
    _demoReversing = false;
    if (_reverseInterval) { clearTimeout(_reverseInterval); _reverseInterval = null; }
    _tickLog.length = 0;
    _movieFrames.length = 0;
    _lastMoviePos = null;
    _stopMoviePlayback();
    _replayCursor = -1;
    _openingPhase = false;
    const pbEl = document.getElementById('playback-controls');
    if (pbEl) pbEl.style.display = 'none';
    if (typeof _liveGuardsActive !== 'undefined') _liveGuardsActive = false;
    if (_demoInterval) { clearInterval(_demoInterval); _demoInterval = null; }
    if (_demoUncappedId) { clearTimeout(_demoUncappedId); _demoUncappedId = null; }
    const ds = document.getElementById('demo-status');
    if (ds) ds.style.display = 'none';
    // Dispose wavefunction surface
    if (typeof disposeWavefunction === 'function') disposeWavefunction();
    // Clean up Demo 3.0 xons and gluons
    _cleanupDemo3();
    // Clean up tet SCs from xonImpliedSet + oct SCs from activeSet
    for (const [, fd] of Object.entries(_nucleusTetFaceData)) {
        for (const scId of fd.scIds) {
            xonImpliedSet.delete(scId);
            _scAttribution.delete(scId);
        }
    }
    _scAttribution.clear(); // full cleanup on demo stop
    for (const scId of _octSCIds) {
        activeSet.delete(scId);
    }
    // Clear tet annotations
    _ruleAnnotations.tetColors.clear();
    _ruleAnnotations.tetOpacity.clear();
    _ruleAnnotations.dirty = true;
    bumpState();
    const pClean = detectImplied();
    applyPositions(pClean);
    updateSpheres();
    // Show xon sparks again
    const quarks = NucleusSimulator?.quarkExcitations || [];
    for (const q of quarks) {
        if (q.spark) q.spark.visible = true;
        if (q.trailLine) q.trailLine.visible = true;
    }
    // Restore panel title
    const dpTitle = document.querySelector('#deuteron-panel > div:first-child');
    if (dpTitle) dpTitle.textContent = 'DEUTERON';
}

