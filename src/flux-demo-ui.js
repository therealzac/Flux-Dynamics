// flux-demo-ui.js — Demo panels, profiling, choreo logging, pause/resume/stop

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
    // Demand-driven: use fixed 64-tick epoch for display purposes
    const epoch = Math.floor(_demoTick / 64);

    // ── Update demo-status (right panel, below button) ──
    const ds = document.getElementById('demo-status');
    if (ds) {
        ds.innerHTML = `<span style="color:#88bbdd;">epoch ${epoch}</span>`;
    }

    // ── Update left panel coverage bars (skip during test execution) ──
    if (_testRunning) return;
    const el = document.getElementById('dp-coverage-bars');
    if (!el) return;

    // Compute evenness: CV across all faces' total visits
    const totals = [];
    for (let f = 1; f <= 8; f++) totals.push(_demoVisits[f].total);
    const mean = totals.reduce((a, b) => a + b, 0) / totals.length;
    const stddev = Math.sqrt(totals.reduce((s, v) => s + (v - mean) ** 2, 0) / totals.length);
    const cv = mean > 0 ? (stddev / mean) : 0;
    const evenness = Math.max(0, 1 - cv);

    // Find max for bar normalization
    const types6 = ['pu1', 'pu2', 'pd', 'nd1', 'nd2', 'nu'];
    const typeColors = { pu1: '#0040ff', pu2: '#00ff40', pd: '#00ffff', nd1: '#ffbf00', nd2: '#ff00bf', nu: '#ff0000' };
    const typeLabels = { pu1: 'pu\u2081', pu2: 'pu\u2082', pd: 'pd', nd1: 'nd\u2081', nd2: 'nd\u2082', nu: 'nu' };
    let maxCount = 1;
    for (let f = 1; f <= 8; f++) {
        for (const t of types6) {
            maxCount = Math.max(maxCount, _demoVisits[f][t] || 0);
        }
    }

    // Build bars (6 per face)
    let html = '';
    for (let f = 1; f <= 8; f++) {
        const v = _demoVisits[f];
        const isA = [1, 3, 6, 8].includes(f);
        html += `<div style="display:flex; align-items:center; gap:1px;">`
            + `<span style="width:18px; color:${isA ? '#cc8866' : '#6688aa'}; font-size:8px; font-weight:bold;">F${f}</span>`;
        for (const t of types6) {
            html += `<div class="dp-bar-bg" style="flex:1;" title="${typeLabels[t]} ${v[t] || 0}"><div class="dp-bar-fill" style="width:${((v[t] || 0) / maxCount * 100).toFixed(1)}%; background:${typeColors[t]};"></div></div>`;
        }
        html += `<span style="width:22px; text-align:right; font-size:7px; color:#667788;">${v.total}</span>`
            + `</div>`;
    }

    // ── Per-hadron evenness ──
    const protonPerFace = [], neutronPerFace = [];
    for (let f = 1; f <= 8; f++) {
        const v = _demoVisits[f];
        protonPerFace.push((v.pu1 || 0) + (v.pu2 || 0) + (v.pd || 0));
        neutronPerFace.push((v.nd1 || 0) + (v.nd2 || 0) + (v.nu || 0));
    }
    // Per-type global totals for 3-way evenness
    const typeTotals = {};
    for (const t of types6) typeTotals[t] = 0;
    for (let f = 1; f <= 8; f++) {
        for (const t of types6) typeTotals[t] += _demoVisits[f][t] || 0;
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
        + `<span style="color:#6a8a9a;">overall</span>`
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
        + `<span style="color:#6a8a9a;">p 1:1:1</span>`
        + `<span style="color:${pEven > 0.7 ? '#66dd66' : '#ccaa66'}; font-weight:bold;">${(pEven * 100).toFixed(1)}%</span>`
        + `</div>`;
    html += `<div style="display:flex; justify-content:space-between; font-size:9px;">`
        + `<span style="color:#6a8a9a;">n 1:1:1</span>`
        + `<span style="color:${nEven > 0.7 ? '#66dd66' : '#ccaa66'}; font-weight:bold;">${(nEven * 100).toFixed(1)}%</span>`
        + `</div>`;
    html += `<div style="display:flex; justify-content:space-between; font-size:9px;">`
        + `<span style="color:#6a8a9a;">epoch</span>`
        + `<span style="color:#88aacc;">${epoch}</span>`
        + `</div>`;

    // ── Ratio accuracy history sparkline ──
    _demoTypeBalanceHistory.push(ratioAccuracy * 100);
    const hist = _demoTypeBalanceHistory;
    const sparkLen = Math.min(hist.length, 24);  // show last 24 cycles
    const sparkData = hist.slice(-sparkLen);
    // Scale: map [min..100] to 8-level block chars
    const sparkMin = Math.min(...sparkData, 90);
    const sparkMax = 100;
    const SPARK = ['\u2581', '\u2582', '\u2583', '\u2584', '\u2585', '\u2586', '\u2587', '\u2588'];
    let sparkline = '';
    for (const v of sparkData) {
        const norm = Math.max(0, Math.min(1, (v - sparkMin) / (sparkMax - sparkMin)));
        const idx = Math.min(7, Math.floor(norm * 7.99));
        // Color: green at 100, yellow at 95, orange below
        const c = v >= 99.5 ? '#66dd66' : v >= 96 ? '#ccaa66' : '#cc8855';
        sparkline += `<span style="color:${c};">${SPARK[idx]}</span>`;
    }
    html += `<div style="margin-top:4px; overflow:hidden;">`
        + `<div style="font-size:7px; color:#556677; margin-bottom:1px;">ratio accuracy (last ${sparkLen} windows)</div>`
        + `<div style="font-size:10px; letter-spacing:-1px; line-height:1; font-family:monospace; overflow:hidden;">${sparkline}</div>`
        + `<div style="display:flex; justify-content:space-between; font-size:6px; color:#445566; margin-top:1px;">`
        + `<span>${sparkMin.toFixed(0)}%</span><span>100%</span></div>`
        + `</div>`;
    // Rule compliance indicators
    const rules = [
        { name: 'anti-phase', ok: true },  // guaranteed by schedule construction
        { name: 'pauli', ok: _demoPauliViolations === 0 },
        { name: 'spread', ok: _demoSpreadViolations === 0 },
        { name: 'coverage', ok: evenness > 0.9 },
    ];
    html += `<div style="margin-top:3px; font-size:8px; color:#556677;">`;
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
                        x._mode === 'weak' ? '#cc44ff' :
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
            + `<span style="color:#88aacc; font-size:8px;">n${x.node}</span>`
            + `<span style="color:#667788; font-size:7px;">${modeLabel}${faceStr}</span>`
            + `<span style="color:${balColor}; font-size:6px; letter-spacing:-0.5px;">${barStr} ${balStr}</span>`
            + `<span style="color:#556677; font-size:6px;">${msStr}</span>`
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
}
function resumeDemo() {
    _demoPaused = false;
    simHalted = false;
    if (_demoActive && !_demoInterval && !_demoUncappedId) {
        // Always re-execute fresh ticks (redo stack is no longer used)
        _redoStack.length = 0;
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
// When user steps forward, demoTick() re-executes the tick fresh (no redo replay).
function _playbackStepBack() {
    if (_tickInProgress || _btSnapshots.length < 1) return false;
    if (_demoTick <= 0) return false;
    // Discard redo stack — forward replay should re-execute ticks fresh,
    // not replay the same (possibly failed) sequence from the original run.
    _redoStack.length = 0;
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

// Step forward one tick — always re-execute fresh (backtracker can find better paths).
async function _playbackStepForward() {
    if (_tickInProgress || !_demoActive) return;
    simHalted = false;
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
        // Restore previous state (no redo stack — forward replay re-executes fresh)
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
    const el = document.getElementById('nucleus-status');
    if (el) el.innerHTML = `${_planckSeconds} Planck seconds<br><span style="font-size:0.8em; color:#556677;">${_demoTick} ticks</span>`;
    const dpT = document.getElementById('dp-title');
    if (dpT) dpT.innerHTML = `${_planckSeconds} Planck seconds<br><span style="font-size:0.7em; color:#8a9aaa; letter-spacing:0.05em;">${_demoTick} ticks</span>`;
    // Apply restored solver positions to the 3D scene (no re-solve needed)
    if (typeof applyPositions === 'function' && typeof pos !== 'undefined') applyPositions(pos);
    // Rebuild state + SC lines + void spheres from restored SC sets
    if (typeof bumpState === 'function') bumpState();
    if (typeof rebuildShortcutLines === 'function') rebuildShortcutLines();
    if (typeof updateVoidSpheres === 'function') updateVoidSpheres();
    if (typeof updateSpheres === 'function') updateSpheres();
    if (typeof updateStatus === 'function') updateStatus();
    updateDemoPanel();
    updateXonPanel();
}

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
