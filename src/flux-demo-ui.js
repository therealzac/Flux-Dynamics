// flux-demo-ui.js — Demo panels, profiling, choreo logging, pause/resume/stop

// Lightweight tick counter update — safe to call during backtrack retry loops.
// Updates the timeline scrubber and left panel title.
function _updateTickCounter() {
    const dpT = document.getElementById('dp-title');
    if (dpT) dpT.innerHTML = `${_planckSeconds} Planck seconds<br><span style="font-size:0.7em; color:#8a9aaa; letter-spacing:0.05em;">${_demoTick} ticks</span>${_tickerMetaLines()}`;
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
    const layer = `<span style="${s} color:${layerColor};">layer: tick ${layerTick} (visit ${layerVisits + 1})</span>`;

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
    const typeColors = { pu1: '#0040ff', pu2: '#00ff40', pd: '#00ffff', nd1: '#ffbf00', nd2: '#ff00bf', nu: '#ff0000' };
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
    const typeColors = { pu1: '#0040ff', pu2: '#00ff40', pd: '#00ffff', nd1: '#ffbf00', nd2: '#ff00bf', nu: '#ff0000' };

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
    panel.style.display = _demoActive ? 'block' : 'none';
    if (!_demoActive) return;

    const listEl = document.getElementById('xon-panel-list');
    if (!listEl) return;

    let html = '';
    for (let i = 0; i < _demoXons.length; i++) {
        const x = _demoXons[i];
        if (!x.alive) continue;
        const modeCol = x._mode === 'oct' ? '#ffffff' :
                        x._mode === 'weak' ? '#080808' :
                        x._mode === 'gluon' ? '#ffffff' :
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
        const ms = x._modeStats || { oct: 0, tet: 0, idle_tet: 0, weak: 0 };
        const msStr = `o:${ms.oct} t:${ms.tet} i:${ms.idle_tet}` + (ms.weak > 0 ? ` w:${ms.weak}` : '');

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
    _demoReversing = false;
    if (_reverseInterval) { clearTimeout(_reverseInterval); _reverseInterval = null; }
    _tickLog.length = 0;
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
    _btRestoreSnapshot(snap);
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
        _btRestoreSnapshot(snap);
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
    if (dpT) dpT.innerHTML = `${_planckSeconds} Planck seconds<br><span style="font-size:0.7em; color:#8a9aaa; letter-spacing:0.05em;">${_demoTick} ticks</span>${_tickerMetaLines()}`;
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
