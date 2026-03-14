// flux-demo-ui.js — Demo panels, profiling, choreo logging, pause/resume/stop

// Lightweight tick counter update — safe to call during backtrack retry loops.
// Updates the timeline scrubber and left panel title.
function _updateTickCounter() {
    const dpT = document.getElementById('dp-title');
    if (dpT) dpT.innerHTML = `${_planckSeconds} Flux events<br><span style="font-size:0.7em; color:#8a9aaa; letter-spacing:0.05em;">${_demoTick} Planck seconds</span>${_tickerMetaLines()}`;
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

    // Hide density/sync rows during demo (not relevant)
    const densityRow = document.querySelector('#deuteron-panel > div:nth-child(2)');
    const syncRow = document.querySelector('#deuteron-panel > div:nth-child(3)');
    if (densityRow) densityRow.style.display = 'none';
    if (syncRow) syncRow.style.display = 'none';
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
function updateXonPanel() {
    if (_testRunning) return;
    const panel = document.getElementById('xon-panel');
    if (!panel) return;
    panel.style.display = (_demoActive || _playbackMode) ? 'block' : 'none';
    if (!_demoActive && !_playbackMode) return;

    const listEl = document.getElementById('xon-panel-list');
    if (!listEl) return;

    let html = '';
    for (let i = 0; i < _demoXons.length; i++) {
        const x = _demoXons[i];
        if (!x.alive) continue;
        const modeCol = x._mode === 'oct' ? '#ffffff' :
                        x._mode === 'weak' ? '#' + (WEAK_FORCE_COLOR || 0x7f00ff).toString(16).padStart(6, '0') :
                        x._mode === 'gluon' ? '#' + (GLUON_COLOR || 0x000000).toString(16).padStart(6, '0') :
                        x._mode === 'oct_formation' ? '#ffffff' :
                        x._mode === 'tet' ? '#' + (x.col || 0xffffff).toString(16).padStart(6, '0') :
                        x._mode === 'idle_tet' ? '#' + (x.col || 0x888888).toString(16).padStart(6, '0') : '#888888';
        // Display labels: oct=idle, tet/idle_tet=hadron type (p_u, p_d, n_u, n_d)
        const QUARK_LABELS = { pu1: 'pu\u2081', pu2: 'pu\u2082', pd: 'pd', nd1: 'nd\u2081', nd2: 'nd\u2082', nu: 'nu' };
        let modeLabel, faceStr;
        if (x._mode === 'oct') {
            modeLabel = 'idle';
            faceStr = '';
        } else if (x._mode === 'weak') {
            modeLabel = 'weak';
            faceStr = '';
        } else if (x._mode === 'gluon') {
            modeLabel = 'gluon';
            faceStr = '';
        } else if (x._mode === 'oct_formation') {
            modeLabel = 'form';
            faceStr = '';
        } else {
            // tet or idle_tet — show hadron type
            modeLabel = x._quarkType ? QUARK_LABELS[x._quarkType] || x._quarkType : x._mode;
            faceStr = x._assignedFace ? ` f${x._assignedFace}` : '';
        }
        // Balance bar: 1 - coeffOfVariation(_dirBalance), clamped [0,1]
        let balPct = 0;
        let balColor = '#ff8800';
        if (x._dirBalance) {
            const counts = x._dirBalance;
            let sum = 0, n = 0;
            for (let d = 0; d < 10; d++) { sum += counts[d]; n++; }
            const mean = sum / n;
            if (mean > 0) {
                let variance = 0;
                for (let d = 0; d < 10; d++) variance += (counts[d] - mean) ** 2;
                const cv = Math.sqrt(variance / n) / mean;
                balPct = Math.max(0, Math.min(1, 1 - cv));
            }
            balColor = balPct > 0.6 ? '#44cc44' : balPct > 0.4 ? '#cccc44' : '#ff8800';
        }
        const barFull = Math.round(balPct * 8);
        const barEmpty = 8 - barFull;
        const barStr = '\u2588'.repeat(barFull) + '\u2591'.repeat(barEmpty);
        const balStr = `${Math.round(balPct * 100)}%`;

        // Mode stats line
        const ms = x._modeStats || { oct: 0, tet: 0, idle_tet: 0, weak: 0, gluon: 0 };
        const msStr = `o:${ms.oct} t:${ms.tet} i:${ms.idle_tet}` + (ms.gluon > 0 ? ` g:${ms.gluon}` : '') + (ms.weak > 0 ? ` w:${ms.weak}` : '');

        // Tooltip: full 10-direction breakdown
        const db = x._dirBalance || new Array(10).fill(0);
        const tipDirs = `base[0-3]: ${db.slice(0, 4).join(',')} sc[4-9]: ${db.slice(4).join(',')}`;

        const highlighted = _xonHighlightTimers.has(i);
        const border = highlighted ? `2px solid ${modeCol}` : '1px solid #334455';
        const bg = highlighted ? 'rgba(255,255,255,0.15)' : '#0d1520';
        html += `<button class="xon-btn" data-xon-idx="${i}" style="display:flex; flex-direction:column; align-items:center; justify-content:center; width:48px; height:52px; padding:2px; cursor:pointer; border-radius:4px; background:${bg}; border:${border}; font-family:monospace; outline:none;" title="X${i}: n${x.node} ${modeLabel}${faceStr}\n${tipDirs}">`
            + `<span style="color:${modeCol}; font-weight:bold; font-size:11px;">X${i}</span>`
            + `<span style="color:var(--text-2); font-size:8px;">n${x.node}</span>`
            + `<span style="color:var(--text-3); font-size:7px;">${modeLabel}${faceStr}</span>`
            + `<span style="color:${balColor}; font-size:6px; letter-spacing:-0.5px;">${barStr} ${balStr}</span>`
            + `<span style="color:var(--text-3); font-size:6px;">${msStr}</span>`
            + `</button>`;
    }
    // Running totals: xon-ticks spent in each mode (global across all xons)
    const g = typeof _globalModeStats !== 'undefined' ? _globalModeStats : { oct: 0, tet: 0, idle_tet: 0, weak: 0, gluon: 0 };
    const totalTicks = g.oct + g.tet + g.idle_tet + g.weak + g.gluon;
    const pct = (v) => totalTicks > 0 ? Math.round(v / totalTicks * 100) : 0;
    // Current mode counts (live)
    const now = { oct: 0, tet: 0, idle_tet: 0, weak: 0, gluon: 0 };
    for (const x of _demoXons) { if (x.alive && now[x._mode] !== undefined) now[x._mode]++; }
    const nowParts = [];
    if (now.oct > 0) nowParts.push(`<span style="color:#fff">oct:${now.oct}</span>`);
    if (now.tet > 0) nowParts.push(`<span style="color:#5bf">tet:${now.tet}</span>`);
    if (now.idle_tet > 0) nowParts.push(`<span style="color:#888">idle:${now.idle_tet}</span>`);
    if (now.gluon > 0) nowParts.push(`<span style="color:#80ff00">gluon:${now.gluon}</span>`);
    if (now.weak > 0) nowParts.push(`<span style="color:#7f00ff">weak:${now.weak}</span>`);
    html += `<div style="width:100%; text-align:center; font-size:9px; margin-top:4px; letter-spacing:0.03em;">${nowParts.join(' &middot; ')}</div>`;
    // Historical running totals
    const histParts = [];
    histParts.push(`<span style="color:#aaa">oct:${g.oct}</span>`);
    histParts.push(`<span style="color:#5bf">tet:${g.tet}</span>`);
    histParts.push(`<span style="color:#888">idle:${g.idle_tet}</span>`);
    histParts.push(`<span style="color:#80ff00">gluon:${g.gluon}</span>`);
    histParts.push(`<span style="color:#7f00ff">weak:${g.weak}</span>`);
    html += `<div style="width:100%; text-align:center; font-size:8px; margin-top:2px; color:var(--text-3);">totals: ${histParts.join(' &middot; ')}</div>`;

    listEl.innerHTML = html;

    // Click delegation: attach ONE handler to parent (survives innerHTML rebuilds).
    // Guard against duplicate attachment with a flag.
    if (!listEl._xonDelegated) {
        listEl._xonDelegated = true;
        listEl.addEventListener('mousedown', (e) => {
            // Find closest .xon-btn ancestor of the click target
            const btn = e.target.closest('.xon-btn');
            if (!btn) return;
            const idx = parseInt(btn.dataset.xonIdx, 10);
            if (!isNaN(idx)) _highlightXon(idx);
        });
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
                    const snap = _redoStack.pop();
                    _btRestoreSnapshot(snap);
                    simHalted = false;
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
                    // Redo exhausted — switch to live execution
                    _demoInterval = null;
                    _bfsReset(); _btReset();
                    if (typeof _liveGuardResetForRewind === 'function') _liveGuardResetForRewind();
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
    // Restore density/sync rows
    const densityRow = document.querySelector('#deuteron-panel > div:nth-child(2)');
    const syncRow = document.querySelector('#deuteron-panel > div:nth-child(3)');
    if (densityRow) densityRow.style.display = '';
    if (syncRow) syncRow.style.display = '';
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
    if (dpT) dpT.innerHTML = `${_planckSeconds} Flux events<br><span style="font-size:0.7em; color:#8a9aaa; letter-spacing:0.05em;">${_demoTick} Planck seconds</span>${_tickerMetaLines()}`;
    _updateTimelineScrubber();
    // Apply restored solver positions to the 3D scene (no re-solve needed)
    if (typeof applyPositions === 'function' && typeof pos !== 'undefined') applyPositions(pos);
    // Rebuild state + SC lines + void spheres from restored SC sets
    if (typeof bumpState === 'function') bumpState();
    if (typeof rebuildShortcutLines === 'function') rebuildShortcutLines();
    // Re-apply tet coloring from restored edge balance (no counting during replay)
    if (typeof _applyTetColoring === 'function') _applyTetColoring(false);
    if (typeof updateVoidSpheres === 'function') updateVoidSpheres();
    if (typeof updateSpheres === 'function') updateSpheres();
    if (typeof updateStatus === 'function') updateStatus();
    // Force-update bottom-stats even when simHalted (updateStatus bails early)
    _updateBottomStats();
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

// Export tick log as downloadable JSON.
function exportTickLog() {
    if (_tickLog.length === 0) {
        console.warn('[export] No tick log data to export');
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
    const trailPos = new Float32Array((XON_TRAIL_LENGTH + 1) * 3);
    const trailCol = new Float32Array((XON_TRAIL_LENGTH + 1) * 3);
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
        trail: [startNode], trailColHistory: [col], _trailRoleHistory: ['oct'], _trailFrozenPos: [],
        tweenT: 1, flashT: 1.0, _highlightT: 0, alive: true,
    };
    _trailInitFrozen(xon);
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
    const dp = document.getElementById('deuteron-panel');
    if (dp) dp.style.display = 'block';
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

const _PLAYBACK_TRAIL_LEN = 12;

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
        const trailStart = Math.max(0, idx - _PLAYBACK_TRAIL_LEN + 1);
        x.trail = [];
        x.trailColHistory = [];
        x._trailRoleHistory = [];
        x._trailFrozenPos = [];
        for (let t = trailStart; t <= idx; t++) {
            const tf = frames[t];
            if (!tf || !tf.xons[i]) continue;
            const tn = tf.xons[i].n;
            const m = tf.xons[i].m, q = tf.xons[i].q;
            x.trail.push(tn);
            x.trailColHistory.push(_modeColor(m, q));
            // Derive role from mode + quark type
            const role = (m === 'tet' || m === 'idle_tet') ? (q || 'oct') :
                         m === 'gluon' ? 'gluon' : m === 'weak' ? 'weak' : 'oct';
            x._trailRoleHistory.push(role);
            x._trailFrozenPos.push([rPos[tn][0], rPos[tn][1], rPos[tn][2]]);
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

// ── Fighterjet mode toggle ──
document.getElementById('fighterjet-toggle')?.addEventListener('change', e => {
    _fighterjetMode = e.target.checked;
});

// ── Rules switchboard toggles ──
document.getElementById('rule-relinquish-toggle')?.addEventListener('change', e => {
    _ruleRelinquishSCs = e.target.checked;
});
document.getElementById('rule-gluon-mediated-toggle')?.addEventListener('change', e => {
    _ruleGluonMediatedSC = e.target.checked;
});
document.getElementById('rule-bare-tet-toggle')?.addEventListener('change', e => {
    _ruleBareTetrahedra = e.target.checked;
});

// ── Simulation UI state: button swap + rule locking ──
function _setSimUIActive(active) {
    const startRow = document.getElementById('sim-start-buttons');
    const activeRow = document.getElementById('sim-active-buttons');
    if (startRow) startRow.style.display = active ? 'none' : 'flex';
    if (activeRow) activeRow.style.display = active ? 'flex' : 'none';
    // Lock/unlock rule toggles
    const toggleIds = ['rule-relinquish-toggle', 'rule-gluon-mediated-toggle', 'rule-bare-tet-toggle'];
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
    _clearSimReset();
    if (typeof stopDemo === 'function') stopDemo();
    _setSimUIActive(false);
});
document.getElementById('btn-clear-cancel')?.addEventListener('click', _clearSimReset);
