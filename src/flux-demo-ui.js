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
    const totalOpen = activeSet.size + impliedSet.size + (typeof xonImpliedSet !== 'undefined' ? xonImpliedSet.size : 0);
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
        _replayGuardMode = false;
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
        if (_redoStack.length > 0) {
            // Drain redo stack at playback speed (responsive to slider changes).
            // Uses setTimeout chaining (like rewind) so speed adjustments take
            // effect immediately — setInterval would lock to the initial rate.
            let _lastForwardVisual = 0;
            const FWD_VISUAL_INTERVAL = 33; // ~30fps
            function _forwardReplayStep() {
                if (_demoPaused || !_demoActive) {
                    _demoInterval = null;
                    return;
                }
                if (_redoStack.length > 0) {
                    _btSaveSnapshot(); // save current state so rewind can reach it
                    let snap = _redoStack.pop();
                    // Safety: skip any undefined/null entries (stale IDB data)
                    while (!snap && _redoStack.length > 0) snap = _redoStack.pop();
                    if (!snap) { _demoInterval = null; return; }
                    // Guard snapshot BEFORE restore — captures previous tick's state
                    // as "pre-move" analog. Mirrors live play: snapshot → moves → check.
                    if (typeof _liveGuardSnapshot === 'function') _liveGuardSnapshot();
                    _btRestoreSnapshot(snap);
                    // Archive replayed snapshots so _councilSnapArchive has full t=0→t=N
                    // when auto-retry-best extends a replay run and saves back to IDB.
                    if (_sweepReplayActive && typeof _serializeSnapshot === 'function') {
                        _councilSnapArchive.push(_serializeSnapshot(snap));
                    }
                    simHalted = false;
                    // Update tet coloring BEFORE guards — T58 reads _ruleAnnotations
                    // which must reflect the restored SC sets, not the previous frame.
                    if (typeof _applyTetColoring === 'function') _applyTetColoring(false);
                    // Guards always fire during replay — no exceptions.
                    if (typeof _liveGuardCheck === 'function') _liveGuardCheck();
                    if (simHalted) {
                        _demoInterval = null;
                        // Update display so user sees the failure tick state
                        _playbackUpdateDisplay();
                        return;
                    }
                    // Throttle visual updates to ~30fps for buttery speed
                    const now = performance.now();
                    if (now - _lastForwardVisual >= FWD_VISUAL_INTERVAL) {
                        _playbackUpdateDisplay();
                        _lastForwardVisual = now;
                    }
                    // Re-read slider each step so speed changes take effect immediately
                    const nextMs = Math.max(4, _getDemoIntervalMs());
                    _demoInterval = setTimeout(_forwardReplayStep, nextMs);
                } else {
                    // Redo exhausted — seamless transition to live execution.
                    // NO state clearing, NO guard resets. The restored state IS
                    // the live state. Guards keep firing continuously.
                    _demoInterval = null;
                    if (_sweepActive && _sweepReplayActive) {
                        _sweepReplayActive = false;
                        _sweepReplayMember = null;
                        console.log('%c[REPLAY] Redo drain complete — continuing live from tick ' + _demoTick, 'color:#66ccff;font-weight:bold');
                    }
                    _maxTickReached = _demoTick;
                    const liveMs = _getDemoIntervalMs();
                    if (liveMs === 0) {
                        _demoUncappedId = setTimeout(_demoUncappedLoop, 0);
                    } else {
                        _demoInterval = setInterval(demoTick, liveMs);
                    }
                }
            }
            const intervalMs = Math.max(4, _getDemoIntervalMs());
            _demoInterval = setTimeout(_forwardReplayStep, intervalMs);
        } else {
            const intervalMs = _getDemoIntervalMs();
            if (intervalMs === 0) {
                _demoUncappedId = setTimeout(_demoUncappedLoop, 0);
            } else {
                _demoInterval = setInterval(demoTick, intervalMs);
            }
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
    _redoStack.length = 0;
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

// ══════════════════════════════════════════════════════════════════════════
// PLAYBACK CONTROLS — rewind, step, reverse, export
// ══════════════════════════════════════════════════════════════════════════

// Step back one tick by restoring the previous snapshot.
// Snapshot {tick: N} = state before tick N ran = state after tick N-1 completed.
// Pop it and restore → _demoTick becomes N, undoing tick N's effects.
// Forward replay uses redo stack for deterministic replay of the same path.
function _playbackStepBack() {
    if (_tickInProgress || _btSnapshots.length < 1) return false;
    if (_demoTick <= 0) return false;
    // Save current state to redo stack (for deterministic forward replay)
    _btSaveSnapshot();
    const redoSnap = _btSnapshots.pop(); // the one we just saved = current state
    _redoStack.push(redoSnap);
    // Now pop the actual previous state
    const snap = _btSnapshots.pop();
    if (!snap) {
        // No snapshot available — put everything back
        _redoStack.pop();
        _btSnapshots.push(redoSnap);
        return false;
    }
    _btRestoreSnapshot(snap, true); // reverse=true for fighterjet reverse animation
    // Clear halt flag — rewind should always allow forward replay
    simHalted = false;
    // Clear backtracker BFS state so forward replay starts clean
    _bfsReset();
    _btReset();
    // Reset live guards to grace period — replayed choreography may diverge
    if (typeof _liveGuardResetForRewind === 'function') _liveGuardResetForRewind();
    // Also trim the tick log to match
    while (_tickLog.length > 0 && _tickLog[_tickLog.length - 1].tick >= _demoTick) {
        _tickLog.pop();
    }
    _playbackUpdateDisplay();
    return true;
}

// Step forward one tick — instant from redo stack if available, else re-execute.
async function _playbackStepForward() {
    if (_tickInProgress || !_demoActive) return;
    simHalted = false;
    if (_redoStack.length > 0) {
        // Deterministic replay from redo stack (no solver, no choreography)
        _btSaveSnapshot(); // save current state so rewind can reach it
        const redoSnap = _redoStack.pop();
        _btRestoreSnapshot(redoSnap);
        simHalted = false;
        _playbackUpdateDisplay();
        return;
    }
    // No redo available — re-execute tick (live path)
    const wasPaused = _demoPaused;
    _demoPaused = false;
    await demoTick();
    if (wasPaused) _demoPaused = true;
    _playbackUpdateDisplay();
}

// Begin reverse playback at current speed (responsive to slider changes).
function startReverse() {
    if (_demoReversing) return;
    pauseDemo();
    _demoReversing = true;
    // Throttle visual updates during reverse: restore state every tick but only
    // rebuild 3D scene at ~30fps max to avoid expensive rebuildShortcutLines calls
    let _lastVisualUpdate = 0;
    const VISUAL_INTERVAL = 33; // ~30fps
    function _reverseStep() {
        if (!_demoReversing) return;
        if (_btSnapshots.length < 1 || _demoTick <= 0) {
            stopReverse();
            return;
        }
        // Save current state to redo stack before restoring
        _btSaveSnapshot();
        _redoStack.push(_btSnapshots.pop());
        // Restore previous state
        const snap = _btSnapshots.pop();
        if (!snap) { stopReverse(); return; }
        _btRestoreSnapshot(snap, true); // reverse=true for fighterjet reverse animation
        simHalted = false;
        _bfsReset();
        _btReset();
        if (typeof _liveGuardResetForRewind === 'function') _liveGuardResetForRewind();
        while (_tickLog.length > 0 && _tickLog[_tickLog.length - 1].tick >= _demoTick) {
            _tickLog.pop();
        }
        // Only update visuals at throttled rate
        const now = performance.now();
        if (now - _lastVisualUpdate >= VISUAL_INTERVAL) {
            _playbackUpdateDisplay();
            _lastVisualUpdate = now;
        }
        // Re-read slider each step so speed changes take effect immediately
        const nextMs = Math.max(4, _getDemoIntervalMs());
        _reverseInterval = setTimeout(_reverseStep, nextMs);
    }
    const intervalMs = Math.max(4, _getDemoIntervalMs());
    _reverseInterval = setTimeout(_reverseStep, intervalMs);
    _updatePlaybackButtons();
}

// Stop reverse playback.
function stopReverse() {
    _demoReversing = false;
    if (_reverseInterval) { clearTimeout(_reverseInterval); _reverseInterval = null; }
    _demoPaused = true;
    // Final visual sync so display matches current state
    _playbackUpdateDisplay();
    _updatePlaybackButtons();
}

// Refresh display after a manual step (back or forward).
// Snapshot already contains pos[], activeSet, impliedSet, xonImpliedSet —
// NO solver or detectImplied() needed. Just update 3D scene from restored state.
function _playbackUpdateDisplay() {
    // Tick counter
    const dpT = document.getElementById('dp-title');
    if (dpT) dpT.innerHTML = `${_planckSeconds} Flux Events<br><span style="font-size:0.7em; color:#8a9aaa; letter-spacing:0.05em;">${_demoTick} Planck Seconds</span><br><span style="font-size:0.55em; color:#556677; margin-top:-2px; display:block; line-height:1;">(Best ${_maxTickReached})</span>`;
    _updateTimelineScrubber();
    // Apply restored solver positions to the 3D scene (no re-solve needed)
    if (typeof applyPositions === 'function' && typeof pos !== 'undefined') applyPositions(pos);
    // Rebuild state + SC lines + void spheres from restored SC sets
    if (typeof bumpState === 'function') bumpState();
    if (typeof rebuildShortcutLines === 'function') rebuildShortcutLines();
    // Re-apply tet coloring from restored edge balance (no counting during replay)
    if (typeof _applyTetColoring === 'function') _applyTetColoring(false);
    if (typeof updateVoidSpheres === 'function') updateVoidSpheres();
    // Lazily build wavefunction + shells on first replay frame (cold-storage
    // replay may not have had valid REST positions during init).
    if (typeof _wfMesh !== 'undefined' && !_wfMesh && typeof buildWavefunction === 'function') buildWavefunction();
    if (typeof _braneShells !== 'undefined' && _braneShells.length === 0 && typeof buildBranes === 'function') buildBranes();
    if (typeof updateWavefunction === 'function') updateWavefunction();
    if (typeof updateSpheres === 'function') updateSpheres();
    if (typeof updateStatus === 'function') updateStatus();
    // Force-update bottom-stats even when simHalted (updateStatus bails early)
    _updateBottomStats();
    // Sync spark colors from role (snapshot col may be stale white from oct mode)
    for (let i = 0; i < _demoXons.length; i++) {
        const x = _demoXons[i];
        if (!x || !x.alive) continue;
        const role = typeof _xonRole === 'function' ? _xonRole(x) : 'oct';
        const roleCol = _xpRoleColor(role);
        if (x.col !== roleCol) { x.col = roleCol; if (x.sparkMat) x.sparkMat.color.setHex(roleCol); }
    }
    updateDemoPanel();
    _updateEdgeBalancePanel();
    updateXonPanel();
}

// ── Timeline Scrubber ──
// Updates the range slider to reflect current position in the snapshot timeline.
// Total range = _btSnapshots.length (past) + _redoStack.length (future).
// Current position = _btSnapshots.length (we're at the end of past snapshots).
function _updateTimelineScrubber() {
    const slider = document.getElementById('timeline-scrubber');
    const valEl = document.getElementById('timeline-val');
    if (!slider) return;
    const total = (typeof _btSnapshots !== 'undefined' ? _btSnapshots.length : 0)
                + (typeof _redoStack !== 'undefined' ? _redoStack.length : 0);
    const pos = typeof _btSnapshots !== 'undefined' ? _btSnapshots.length : 0;
    slider.max = total;
    slider.value = pos;
    if (valEl) valEl.textContent = _demoTick;
}

// Seek to a specific position in the snapshot timeline.
function _timelineScrubTo(targetPos) {
    // Movie playback scrubbing
    if (_playbackMode && _importedMovie) {
        const idx = Math.max(0, Math.min(targetPos, _importedMovie.totalFrames - 1));
        _playbackFrame = idx;
        _pbPosCache = null; // invalidate cache for random seek
        _applyMovieFrame(idx);
        const val = document.getElementById('timeline-val');
        if (val) val.textContent = idx;
        return;
    }
    if (!_demoActive) return;
    if (_demoReversing && typeof stopReverse === 'function') stopReverse();
    if (!_demoPaused && typeof pauseDemo === 'function') pauseDemo();
    const currentPos = _btSnapshots.length;
    if (targetPos === currentPos) return;

    if (targetPos < currentPos) {
        // Move backward: push snapshots to redo stack
        const steps = currentPos - targetPos;
        for (let i = 0; i < steps; i++) {
            if (_btSnapshots.length === 0) break;
            _btSaveSnapshot();
            _redoStack.push(_btSnapshots.pop());
            const snap = _btSnapshots.pop();
            if (!snap) break;
            _btRestoreSnapshot(snap);
        }
    } else {
        // Move forward: pop from redo stack
        const steps = targetPos - currentPos;
        for (let i = 0; i < steps; i++) {
            if (_redoStack.length === 0) break;
            _btSaveSnapshot();
            const snap = _redoStack.pop();
            _btRestoreSnapshot(snap);
        }
    }
    if (typeof simHalted !== 'undefined') simHalted = false;
    _bfsReset(); _btReset();
    if (typeof _liveGuardResetForRewind === 'function') _liveGuardResetForRewind();
    _playbackUpdateDisplay();
    const pb = document.getElementById('btn-nucleus-pause');
    if (pb) pb.textContent = '\u25B6';
}

// Init scrubber event listener
(function _initTimelineScrubber() {
    function attach() {
        const slider = document.getElementById('timeline-scrubber');
        if (!slider) return;
        slider.addEventListener('input', function() {
            _timelineScrubTo(parseInt(this.value, 10));
        });
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', attach);
    } else {
        attach();
    }
})();

// Sync playback button visual states.
function _updatePlaybackButtons() {
    const revBtn = document.getElementById('btn-reverse');
    const pauseBtn = document.getElementById('btn-nucleus-pause');
    if (revBtn) {
        revBtn.style.borderColor = _demoReversing ? '#ff8844' : '#6688aa';
        revBtn.style.color = _demoReversing ? '#ffaa66' : '#88aacc';
    }
    if (pauseBtn) {
        pauseBtn.textContent = (_demoPaused && !_demoReversing) ? '\u25B6' : '\u23F8';
    }
}

// Export tick log or council replay snapshots as downloadable JSON.
function exportTickLog() {
    // During council replay or paused demo, export full snapshots
    if (_btSnapshots && _btSnapshots.length > 0 && typeof _serializeSnapshot === 'function') {
        // Serialize then reconstruct trails for export (modern snapshots use trailLen only)
        const snapData = _btSnapshots.map(s => _serializeSnapshot(s));
        // Reconstruct unified trail arrays so exported file is self-contained
        const numXons = snapData[0] && snapData[0].xons ? snapData[0].xons.length : 0;
        for (let si = 0; si < snapData.length; si++) {
            const snap = snapData[si];
            if (!snap.xons) continue;
            for (let xi = 0; xi < numXons && xi < snap.xons.length; xi++) {
                const xon = snap.xons[xi];
                const role = xon._role || 'oct';
                xon.col = (typeof _xpRoleColor === 'function') ? _xpRoleColor(role) : xon.col || 0xffffff;
                const p = snap.pos && snap.pos[xon.node];
                const entry = { node: xon.node, role, pos: p ? [p[0], p[1], p[2]] : [0, 0, 0] };
                if (si === 0) {
                    xon.trail = [entry];
                } else {
                    const prev = snapData[si - 1].xons[xi];
                    xon.trail = prev.trail.concat([entry]);
                }
            }
        }
        const seed = _forceSeed || _runSeed || 0;
        const peak = _btSnapshots[_btSnapshots.length - 1].tick || 0;
        const data = {
            version: 2, type: 'council-replay',
            exported: new Date().toISOString(),
            seed: seed, peak: peak,
            totalSnapshots: snapData.length,
            snapshots: snapData
        };
        const blob = new Blob([JSON.stringify(data)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        const seedHex = seed.toString(16).padStart(8, '0');
        a.download = `flux-replay-0x${seedHex}-t${peak}-${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
        console.log(`[export] Downloaded ${snapData.length} snapshots (seed 0x${seedHex}, peak t${peak})`);
        return;
    }
    if (_tickLog.length === 0) {
        console.warn('[export] No data to export');
        return;
    }
    const data = {
        version: 2,
        exported: new Date().toISOString(),
        totalTicks: _tickLog.length,
        format: { guards: 'delta — tick 0 has full state, subsequent ticks have only changed keys (null = no change)' },
        params: JSON.parse(JSON.stringify(_choreoParams)),
        log: _tickLog
    };
    const blob = new Blob([JSON.stringify(data)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `flux-log-${_tickLog.length}ticks-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    console.log(`[export] Downloaded ${_tickLog.length} tick log entries`);
}

// Import a previously exported council replay and play it back.
function importReplay(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const data = JSON.parse(e.target.result);
            if (data.type !== 'council-replay' || !data.snapshots || data.snapshots.length === 0) {
                console.warn('[import] Not a valid council replay export');
                alert('Not a valid council replay file.');
                return;
            }
            // Deserialize snapshots
            const snapshots = data.snapshots.map(s => _deserializeSnapshot(s));
            // Inject as a temporary council member
            const member = {
                seed: data.seed || 1,
                peak: data.peak || snapshots[snapshots.length - 1].tick || 0,
                snapshots: snapshots,
                _cold: false,
                _imported: true,
            };
            // Add to front of council and trigger replay
            _sweepGoldenCouncil.unshift(member);
            console.log(`%c[import] Loaded replay: seed 0x${member.seed.toString(16).padStart(8,'0')}, peak t${member.peak}, ${snapshots.length} snapshots`, 'color:#66ccff;font-weight:bold');
            // Start replay on the newly inserted member (index 0)
            if (typeof startCouncilReplay === 'function') {
                startCouncilReplay(0);
            }
        } catch (err) {
            console.error('[import] Failed to parse replay file:', err);
            alert('Failed to parse replay file: ' + err.message);
        }
    };
    reader.readAsText(file);
}

// ── Movie export ──
function exportMovie() {
    if (_movieFrames.length === 0) {
        console.warn('[movie] No movie frames to export');
        return;
    }
    const data = {
        version: 3,
        type: 'movie',
        exported: new Date().toISOString(),
        latticeLevel: latticeLevel,
        nodeCount: N,
        totalFrames: _movieFrames.length,
        frames: _movieFrames
    };
    const blob = new Blob([JSON.stringify(data)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `flux-movie-${_movieFrames.length}f-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    console.log(`[movie] Downloaded ${_movieFrames.length} frames`);
}

// ── Movie import ──
function importMovie() {
    const input = document.getElementById('movie-file-input');
    if (input) input.click();
}

function _handleMovieFileSelect(e) {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = function(ev) {
        try {
            const data = JSON.parse(ev.target.result);
            if (data.version < 3 || data.type !== 'movie') {
                console.error('[movie] Invalid movie file (need version ≥ 3, type=movie)');
                return;
            }
            if (!data.frames || data.frames.length === 0) {
                console.error('[movie] No frames in movie file');
                return;
            }
            _importedMovie = data;
            _startMoviePlayback();
        } catch (err) {
            console.error('[movie] Failed to parse movie file:', err);
        }
    };
    reader.readAsText(file);
    e.target.value = ''; // reset so same file can be re-imported
}

// ── Movie playback engine ──
// Create a bare xon with Three.js visuals for movie playback (no nucleus data needed).
function _spawnPlaybackXon(startNode) {
    const col = 0xffffff;
    const sparkMat = new THREE.SpriteMaterial({
        color: col, map: _sparkTex, transparent: true, opacity: 1.0,
        blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false,
    });
    const spark = new THREE.Sprite(sparkMat);
    spark.scale.set(0.28, 0.28, 1);
    spark.renderOrder = 22;
    const group = new THREE.Group();
    group.add(spark);
    if (pos[startNode]) group.position.set(pos[startNode][0], pos[startNode][1], pos[startNode][2]);
    scene.add(group);

    const trailGeo = new THREE.BufferGeometry();
    const trailPos = new Float32Array((XON_TRAIL_VERTS + 1) * 3);
    const trailCol = new Float32Array((XON_TRAIL_VERTS + 1) * 3);
    trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPos, 3));
    trailGeo.setAttribute('color', new THREE.BufferAttribute(trailCol, 3));
    const trailMat = new THREE.LineBasicMaterial({
        vertexColors: true, transparent: true, opacity: 1.0,
        depthTest: false, blending: THREE.AdditiveBlending,
    });
    const trailLine = new THREE.Line(trailGeo, trailMat);
    trailLine.renderOrder = 20;
    scene.add(trailLine);

    const xon = {
        node: startNode, prevNode: startNode, sign: 1,
        _loopType: null, _loopSeq: null, _loopStep: 0,
        _assignedFace: null, _quarkType: null,
        _mode: 'oct', _lastDir: null, _dirHistory: [],
        _dirBalance: new Array(10).fill(0),
        _modeStats: { oct: 0, tet: 0, idle_tet: 0, weak: 0, gluon: 0 },
        col, group, spark, sparkMat,
        trailLine, trailGeo, trailPos, trailCol,
        trail: [{ node: startNode, role: 'oct', pos: pos[startNode] ? [pos[startNode][0], pos[startNode][1], pos[startNode][2]] : [0,0,0] }],
        tweenT: 1, flashT: 1.0, _highlightT: 0, alive: true,
    };
    _demoXons.push(xon);
    return xon;
}

function _startMoviePlayback() {
    if (!_importedMovie) return;
    // Stop any running demo
    if (_demoActive) {
        if (_demoInterval) { clearInterval(_demoInterval); _demoInterval = null; }
        _demoActive = false;
    }
    // Build lattice if not already built
    if (!pos || pos.length === 0) {
        buildLattice(_importedMovie.latticeLevel || 2);
    }
    // Apply frame 0 positions so lattice is correct before spawning
    const f0 = _importedMovie.frames[0];
    if (f0 && f0.pos) {
        for (let i = 0; i < f0.pos.length && i < pos.length; i++) {
            pos[i][0] = f0.pos[i][0]; pos[i][1] = f0.pos[i][1]; pos[i][2] = f0.pos[i][2];
        }
    }
    // Spawn playback xons (clean slate)
    for (const x of _demoXons) {
        if (x.group) scene.remove(x.group);
        if (x.trailLine) scene.remove(x.trailLine);
    }
    _demoXons.length = 0;
    const xonCount = f0 ? f0.xons.length : 6;
    for (let i = 0; i < xonCount; i++) {
        const startNode = f0 ? f0.xons[i].n : 0;
        _spawnPlaybackXon(startNode);
    }
    _playbackMode = true;
    _playbackFrame = 0;
    _demoPaused = false;
    _demoReversing = false;
    // Show playback controls + pause button
    const pc = document.getElementById('playback-controls');
    if (pc) pc.style.display = '';
    const pauseBtn = document.getElementById('btn-nucleus-pause');
    if (pauseBtn) { pauseBtn.style.display = ''; pauseBtn.textContent = '\u23F8'; }
    const scrubber = document.getElementById('timeline-scrubber');
    if (scrubber) {
        scrubber.max = _importedMovie.totalFrames - 1;
        scrubber.value = 0;
    }
    const val = document.getElementById('timeline-val');
    if (val) val.textContent = '0';
    // Show deuteron panel + populate legend (normally done by enterNucleusMode)
    // Respect user's sidebar toggle
    const dp = document.getElementById('deuteron-panel');
    const leftVis = typeof _isLeftSidebarVisible === 'function' ? _isLeftSidebarVisible() : true;
    if (dp && leftVis) dp.style.display = 'block';
    if (typeof _populateDeuteronQuarkLegend === 'function') _populateDeuteronQuarkLegend();
    // Reset visit tracking for sidebar
    _demoVisits = {};
    for (let f = 1; f <= 8; f++) _demoVisits[f] = {};
    _demoTypeBalanceHistory = [];
    _demoTick = 0;
    _pbVisitsUpTo = -1;
    // Apply first frame
    _applyMovieFrame(0);
    // Start playback interval (same speed as demo: 60fps tick)
    _demoInterval = setInterval(_moviePlaybackTick, 17);
    console.log(`[movie] Playing back ${_importedMovie.totalFrames} frames`);
}

function _moviePlaybackTick() {
    if (!_playbackMode || !_importedMovie) return;
    if (_demoPaused) return;
    if (_demoReversing) {
        if (_playbackFrame > 0) {
            _playbackFrame--;
            _applyMovieFrame(_playbackFrame);
        }
    } else {
        if (_playbackFrame < _importedMovie.totalFrames - 1) {
            _playbackFrame++;
            _applyMovieFrame(_playbackFrame);
        }
    }
    // Update scrubber
    const scrubber = document.getElementById('timeline-scrubber');
    if (scrubber) scrubber.value = _playbackFrame;
    const val = document.getElementById('timeline-val');
    if (val) val.textContent = _playbackFrame;
}

// Cached position reconstruction — avoids re-walking from frame 0 each time.
// _pbPosCache stores { idx, pos[][] } for the last reconstructed frame.
let _pbPosCache = null;

function _reconstructPos(frameIdx) {
    const frames = _importedMovie.frames;
    // If cache is at or before target, continue from cache; else restart from 0
    let startIdx = 0;
    let p;
    if (_pbPosCache && _pbPosCache.idx <= frameIdx) {
        startIdx = _pbPosCache.idx + 1;
        p = _pbPosCache.pos.map(v => [v[0], v[1], v[2]]);
    } else {
        p = frames[0].pos.map(v => [v[0], v[1], v[2]]);
        startIdx = 1;
    }
    for (let i = startIdx; i <= frameIdx; i++) {
        if (frames[i].dp) {
            for (const [nodeIdx, x, y, z] of frames[i].dp) {
                p[nodeIdx][0] = x; p[nodeIdx][1] = y; p[nodeIdx][2] = z;
            }
        }
    }
    _pbPosCache = { idx: frameIdx, pos: p.map(v => [v[0], v[1], v[2]]) };
    return p;
}

// Color for a given mode+quark
function _modeColor(mode, quark) {
    if (quark && QUARK_COLORS[quark]) return QUARK_COLORS[quark];
    if (mode === 'weak') return WEAK_FORCE_COLOR;
    if (mode === 'gluon') return GLUON_COLOR;
    return 0xffffff; // oct or unassigned
}

// Playback trail length: read from trails slider (matches live trail behavior)
function _playbackTrailLen() {
    const el = document.getElementById('tracer-lifespan-slider');
    return el ? +el.value : 55;
}

// Incremental visit tracker for playback
let _pbVisitsUpTo = -1; // last frame index we've counted visits for

// Reconstruct _demoVisits from movie frames [0..idx] for sidebar display
function _reconstructPlaybackVisits(idx) {
    const frames = _importedMovie.frames;
    // If scrubbing backwards, reset and rebuild
    if (idx < _pbVisitsUpTo) {
        _demoVisits = {};
        for (let f = 1; f <= 8; f++) _demoVisits[f] = {};
        _pbVisitsUpTo = -1;
    }
    // Ensure _demoVisits initialized
    if (!_demoVisits || _pbVisitsUpTo < 0) {
        _demoVisits = {};
        for (let f = 1; f <= 8; f++) _demoVisits[f] = {};
        _pbVisitsUpTo = -1;
    }
    // Incrementally add frames from _pbVisitsUpTo+1 to idx
    for (let t = _pbVisitsUpTo + 1; t <= idx && t < frames.length; t++) {
        const fr = frames[t];
        if (!fr || !fr.xons) continue;
        for (const x of fr.xons) {
            if (x.f && x.f > 0 && x.q && (x.m === 'tet' || x.m === 'idle_tet')) {
                _demoVisits[x.f][x.q] = (_demoVisits[x.f][x.q] || 0) + 1;
            }
        }
    }
    _pbVisitsUpTo = idx;
    _actualizationVisits = _demoVisits;
}

function _applyMovieFrame(idx) {
    if (!_importedMovie) return;
    const frames = _importedMovie.frames;
    const frame = frames[idx];
    if (!frame) return;

    // Set tick counter for sidebar
    _demoTick = idx;

    // 1. Reconstruct positions
    const rPos = _reconstructPos(idx);
    for (let i = 0; i < rPos.length && i < pos.length; i++) {
        pos[i][0] = rPos[i][0];
        pos[i][1] = rPos[i][1];
        pos[i][2] = rPos[i][2];
    }

    // 2. Restore SC sets
    activeSet.clear(); for (const id of frame.a) activeSet.add(id);
    xonImpliedSet.clear(); for (const id of frame.xi) xonImpliedSet.add(id);
    impliedSet.clear(); for (const id of frame.im) impliedSet.add(id);
    // Fold xonImplied into implied for rendering (mirrors what _detectImplied does)
    for (const id of frame.xi) impliedSet.add(id);

    // 3. Position xons with trails
    for (let i = 0; i < _demoXons.length && i < frame.xons.length; i++) {
        const x = _demoXons[i];
        const fx = frame.xons[i];
        // Derive prevNode from previous frame
        const prevFrame = idx > 0 ? frames[idx - 1] : null;
        const prevNode = (prevFrame && prevFrame.xons[i]) ? prevFrame.xons[i].n : fx.n;

        x._restoring = true;
        x.prevNode = prevNode;
        x.node = fx.n;
        x._restoring = false;
        x._mode = fx.m;
        x._quarkType = fx.q;
        x.col = _modeColor(fx.m, fx.q);
        x.alive = true;

        // Update spark color
        if (x.sparkMat) x.sparkMat.color.setHex(x.col);

        // Build trail from history (last N frames)
        // Use current reconstructed positions for all trail entries (close enough
        // since node positions barely shift between adjacent frames)
        const trailStart = Math.max(0, idx - _playbackTrailLen() + 1);
        x.trail = [];
        for (let t = trailStart; t <= idx; t++) {
            const tf = frames[t];
            if (!tf || !tf.xons[i]) continue;
            const tn = tf.xons[i].n;
            const m = tf.xons[i].m, q = tf.xons[i].q;
            const role = (m === 'tet' || m === 'idle_tet') ? (q || 'oct') :
                         m === 'gluon' ? 'gluon' : m === 'weak' ? 'weak' : 'oct';
            x.trail.push({ node: tn, role, pos: [rPos[tn][0], rPos[tn][1], rPos[tn][2]] });
        }

        // Snap sprite to position
        if (x.group && pos[x.node]) {
            x.group.position.set(pos[x.node][0], pos[x.node][1], pos[x.node][2]);
        }
        x.tweenT = 1;
    }

    // 4. Rebuild visuals
    rebuildBaseLines();
    rebuildShortcutLines();
    updateVoidSpheres();
    updateSpheres();
    // 5. Reconstruct visit data and update sidebar panels
    try {
        _reconstructPlaybackVisits(idx);
        // Reset balance history so sparkline builds up during playback
        if (idx === 0) _demoTypeBalanceHistory = [];
        if (typeof updateDemoPanel === 'function') updateDemoPanel();
    } catch(e) {}
    try { if (typeof updateXonPanel === 'function') updateXonPanel(); } catch(e) {}
}

function _stopMoviePlayback() {
    _playbackMode = false;
    _importedMovie = null;
    _playbackFrame = 0;
    _pbPosCache = null;
    if (_demoInterval) { clearInterval(_demoInterval); _demoInterval = null; }
}

// ── Trail curve sliders ──
{
    const _fadeSlider = document.getElementById('trail-fade-slider');
    const _fadeVal = document.getElementById('trail-fade-val');
    if (_fadeSlider) {
        _fadeSlider.addEventListener('input', e => {
            _trailFadeFloor = +e.target.value / 100;
            if (_fadeVal) _fadeVal.textContent = e.target.value + '%';
        });
    }
    const _curveSlider = document.getElementById('trail-curve-slider');
    const _curveVal = document.getElementById('trail-curve-val');
    if (_curveSlider) {
        _curveSlider.addEventListener('input', e => {
            _fjCurvature = +e.target.value * 0.89 / 100;
            if (_curveVal) _curveVal.textContent = e.target.value + '%';
        });
    }
    const _tensionSlider = document.getElementById('trail-tension-slider');
    const _tensionVal = document.getElementById('trail-tension-val');
    if (_tensionSlider) {
        _tensionSlider.addEventListener('input', e => {
            _fjTension = +e.target.value / 100;
            if (_tensionVal) _tensionVal.textContent = _fjTension.toFixed(2);
        });
    }
    const _alphaSlider = document.getElementById('trail-alpha-slider');
    const _alphaVal = document.getElementById('trail-alpha-val');
    if (_alphaSlider) {
        _alphaSlider.addEventListener('input', e => {
            _fjAlpha = +e.target.value / 100;
            if (_alphaVal) _alphaVal.textContent = _fjAlpha.toFixed(2);
        });
    }
}

// ── Rules switchboard toggles ──
{
    const _t20El = document.getElementById('rule-t20-strict-toggle');
    if (_t20El) {
        _t20El.checked = _ruleT20StrictMode; // sync to JS default
        _t20El.addEventListener('change', e => { _ruleT20StrictMode = e.target.checked; _populateCouncilDropdown(); });
    }
}
{
    const _octFullEl = document.getElementById('rule-oct-full-slider');
    if (_octFullEl) {
        _octFullEl.value = T79_MAX_FULL_TICKS; // reset to JS default
        _octFullEl.addEventListener('input', e => {
            T79_MAX_FULL_TICKS = parseInt(e.target.value, 10);
            const lbl = document.getElementById('rule-oct-full-value');
            if (lbl) lbl.textContent = T79_MAX_FULL_TICKS;
            _populateCouncilDropdown();
        });
    }
}
// OCT_CAPACITY_MAX and _populateCouncilDropdown live in flux-tests.js which loads AFTER this file.
// All switchboard listeners that reference late-loaded globals must be deferred to DOMContentLoaded
// (by which time all <script> tags have executed).
window.addEventListener('DOMContentLoaded', () => {
    const _octCapEl = document.getElementById('rule-oct-capacity-slider');
    if (_octCapEl) {
        _octCapEl.value = OCT_CAPACITY_MAX;
        _octCapEl.addEventListener('input', e => {
            OCT_CAPACITY_MAX = parseInt(e.target.value, 10);
            const lbl = document.getElementById('rule-oct-capacity-value');
            if (lbl) lbl.textContent = OCT_CAPACITY_MAX;
            _populateCouncilDropdown();
        });
    }
    const _gluonEl = document.getElementById('rule-gluon-mediated-toggle');
    if (_gluonEl) {
        _gluonEl.checked = _ruleGluonMediatedSC;
        _gluonEl.addEventListener('change', e => { _ruleGluonMediatedSC = e.target.checked; _populateCouncilDropdown(); });
    }
    const _bareEl = document.getElementById('rule-bare-tet-toggle');
    if (_bareEl) {
        _bareEl.checked = _ruleBareTetrahedra;
        _bareEl.addEventListener('change', e => { _ruleBareTetrahedra = e.target.checked; _populateCouncilDropdown(); });
    }
    const _projGuardEl = document.getElementById('rule-projected-guards-toggle');
    if (_projGuardEl) {
        _projGuardEl.checked = _ruleProjectedGuards;
        _projGuardEl.addEventListener('change', e => { _ruleProjectedGuards = e.target.checked; _populateCouncilDropdown(); });
    }
    const _t90TolEl = document.getElementById('rule-t90-tolerance-slider');
    if (_t90TolEl) {
        _t90TolEl.value = T90_TOLERANCE;
        _t90TolEl.addEventListener('input', e => {
            T90_TOLERANCE = parseInt(e.target.value, 10);
            const lbl = document.getElementById('rule-t90-tolerance-value');
            if (lbl) lbl.textContent = T90_TOLERANCE;
            _populateCouncilDropdown();
        });
    }
    const _t91TolEl = document.getElementById('rule-t91-tolerance-slider');
    if (_t91TolEl) {
        _t91TolEl.value = T91_TOLERANCE;
        _t91TolEl.addEventListener('input', e => {
            T91_TOLERANCE = parseInt(e.target.value, 10);
            const lbl = document.getElementById('rule-t91-tolerance-value');
            if (lbl) lbl.textContent = T91_TOLERANCE;
            _populateCouncilDropdown();
        });
    }
    const _t92TolEl = document.getElementById('rule-t92-tolerance-slider');
    if (_t92TolEl) {
        _t92TolEl.value = T92_TOLERANCE;
        _t92TolEl.addEventListener('input', e => {
            T92_TOLERANCE = parseInt(e.target.value, 10);
            const lbl = document.getElementById('rule-t92-tolerance-value');
            if (lbl) lbl.textContent = T92_TOLERANCE;
            _populateCouncilDropdown();
        });
    }
    // Rule 9 & 10: adaptive ejection — mutually exclusive with rule 8 sliders and each other
    const _adaptEl = document.getElementById('rule-adaptive-ejection-toggle');
    const _cubeEl = document.getElementById('rule-cuberoot-ejection-toggle');
    const _rule8Sliders = ['rule-t90-tolerance-slider', 'rule-t91-tolerance-slider', 'rule-t92-tolerance-slider'];
    function _syncRule8vs9vs10() {
        const on = _ruleAdaptiveEjection || _ruleCubeRootEjection;
        for (const id of _rule8Sliders) {
            const sl = document.getElementById(id);
            if (sl) { sl.disabled = on; sl.style.opacity = on ? '0.3' : '1'; }
        }
        // Also disable ticker buttons for rule 8 sliders
        document.querySelectorAll('.tol-tick').forEach(btn => {
            if (_rule8Sliders.includes(btn.dataset.slider)) {
                btn.disabled = on; btn.style.opacity = on ? '0.3' : '1';
            }
        });
        // Dim the other adaptive checkbox when one is active
        if (_adaptEl) { _adaptEl.disabled = _ruleCubeRootEjection; _adaptEl.parentElement.style.opacity = _ruleCubeRootEjection ? '0.3' : '1'; }
        if (_cubeEl) { _cubeEl.disabled = _ruleAdaptiveEjection; _cubeEl.parentElement.style.opacity = _ruleAdaptiveEjection ? '0.3' : '1'; }
    }
    if (_adaptEl) {
        _adaptEl.checked = _ruleAdaptiveEjection;
        _adaptEl.addEventListener('change', e => {
            _ruleAdaptiveEjection = e.target.checked;
            if (e.target.checked) { _ruleCubeRootEjection = false; if (_cubeEl) _cubeEl.checked = false; }
            _syncRule8vs9vs10();
            _populateCouncilDropdown();
        });
    }
    if (_cubeEl) {
        _cubeEl.checked = _ruleCubeRootEjection;
        _cubeEl.addEventListener('change', e => {
            _ruleCubeRootEjection = e.target.checked;
            if (e.target.checked) { _ruleAdaptiveEjection = false; if (_adaptEl) _adaptEl.checked = false; }
            _syncRule8vs9vs10();
            _populateCouncilDropdown();
        });
    }
    _syncRule8vs9vs10(); // apply initial state
    // Wire left/right ticker buttons for tolerance sliders
    document.querySelectorAll('.tol-tick').forEach(btn => {
        btn.addEventListener('click', () => {
            const sl = document.getElementById(btn.dataset.slider);
            if (!sl) return;
            const dir = parseInt(btn.dataset.dir, 10);
            const nv = Math.max(+sl.min, Math.min(+sl.max, +sl.value + dir));
            if (nv !== +sl.value) { sl.value = nv; sl.dispatchEvent(new Event('input')); }
        });
    });
    // Sync JS from DOM after browser form restoration.
    // Read DOM → JS so JS always matches what the user sees.
    const _syncJSFromDOM = () => {
        const t20El = document.getElementById('rule-t20-strict-toggle');
        const gluEl = document.getElementById('rule-gluon-mediated-toggle');
        const bareEl = document.getElementById('rule-bare-tet-toggle');
        const projEl = document.getElementById('rule-projected-guards-toggle');
        const octFullEl = document.getElementById('rule-oct-full-slider');
        const octCapEl = document.getElementById('rule-oct-capacity-slider');
        if (t20El) _ruleT20StrictMode = t20El.checked;
        if (gluEl) _ruleGluonMediatedSC = gluEl.checked;
        if (bareEl) _ruleBareTetrahedra = bareEl.checked;
        if (projEl) _ruleProjectedGuards = projEl.checked;
        if (octFullEl) T79_MAX_FULL_TICKS = parseInt(octFullEl.value, 10);
        if (octCapEl) OCT_CAPACITY_MAX = parseInt(octCapEl.value, 10);
        const t90TolEl = document.getElementById('rule-t90-tolerance-slider');
        const t91TolEl = document.getElementById('rule-t91-tolerance-slider');
        const t92TolEl = document.getElementById('rule-t92-tolerance-slider');
        if (t90TolEl) T90_TOLERANCE = parseInt(t90TolEl.value, 10);
        if (t91TolEl) T91_TOLERANCE = parseInt(t91TolEl.value, 10);
        if (t92TolEl) T92_TOLERANCE = parseInt(t92TolEl.value, 10);
        const adaptEl = document.getElementById('rule-adaptive-ejection-toggle');
        if (adaptEl) _ruleAdaptiveEjection = adaptEl.checked;
        const cubeEl = document.getElementById('rule-cuberoot-ejection-toggle');
        if (cubeEl) _ruleCubeRootEjection = cubeEl.checked;
        // Per-role opacity sliders
        for (const key of Object.keys(_roleOpacity)) {
            const sl = document.getElementById(`role-opacity-${key}`);
            if (sl) _roleOpacity[key] = +sl.value / 100;
        }
        // Per-xon opacity sliders
        for (let i = 0; i < 6; i++) {
            const sl = document.getElementById(`xon-opacity-${i}`);
            if (sl) _xonOpacity[i] = +sl.value / 100;
        }
        _populateCouncilDropdown();
    };
    // Fire after load, pageshow, AND with escalating delays to catch late browser restoration
    window.addEventListener('load', () => { setTimeout(_syncJSFromDOM, 0); setTimeout(_syncJSFromDOM, 100); });
    window.addEventListener('pageshow', () => { setTimeout(_syncJSFromDOM, 0); setTimeout(_syncJSFromDOM, 100); });

}); // end DOMContentLoaded

// ── Simulation UI state: button swap + rule locking ──
function _setSimUIActive(active) {
    const startRow = document.getElementById('sim-start-buttons');
    const activeRow = document.getElementById('sim-active-buttons');
    if (startRow) startRow.style.display = active ? 'none' : 'flex';
    if (activeRow) activeRow.style.display = active ? 'flex' : 'none';
    // Lock/unlock rule toggles
    const toggleIds = ['rule-t20-strict-toggle', 'rule-gluon-mediated-toggle', 'rule-bare-tet-toggle', 'rule-oct-full-slider', 'rule-oct-capacity-slider', 'rule-projected-guards-toggle', 'rule-t90-tolerance-slider', 'rule-t91-tolerance-slider', 'rule-t92-tolerance-slider', 'rule-adaptive-ejection-toggle'];
    for (const id of toggleIds) {
        const el = document.getElementById(id);
        if (el) { el.disabled = active; el.style.opacity = active ? '0.4' : '1'; }
    }
    const sbPanel = document.getElementById('switchboard-panel');
    if (sbPanel) sbPanel.style.opacity = active ? '0.6' : '1';
}

// Clear Simulation button with confirmation + cancel
function _clearSimReset() {
    const btn = document.getElementById('btn-clear-simulation');
    const cancel = document.getElementById('btn-clear-cancel');
    if (btn) { btn._confirmed = false; btn.textContent = 'Clear Simulation'; btn.style.background = 'rgba(255,60,60,0.15)'; }
    if (cancel) cancel.style.display = 'none';
}
document.getElementById('btn-clear-simulation')?.addEventListener('click', function() {
    if (!this._confirmed) {
        this._confirmed = true;
        this.textContent = 'Are you sure?';
        this.style.background = 'rgba(255,60,60,0.3)';
        const cancel = document.getElementById('btn-clear-cancel');
        if (cancel) cancel.style.display = '';
        return;
    }
    window.location.reload();
});
document.getElementById('btn-clear-cancel')?.addEventListener('click', _clearSimReset);
