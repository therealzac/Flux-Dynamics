// flux-tests-replay.js — Replay pipeline, UI wiring, council dropdown, DOMContentLoaded IIFE
// Split from flux-tests.js (lines 4134-5255). Loaded LAST in script order.


function _updateReplayPanel(member, startTime, message) {
    const el = document.getElementById('bfs-test-results');
    if (!el) return;
    const elapsed = startTime ? ((performance.now() - startTime) / 1000).toFixed(1) : '?';
    const seedHex = '0x' + member.seed.toString(16).padStart(8, '0');
    const pastPeak = _demoTick > member.peak;
    let html = '';

    if (message) {
        html += `<div style="color:#9abccc; font-size:11px; margin-bottom:6px;">${message}</div>`;
    } else if (pastPeak) {
        html += `<div style="color:#ff9966; font-size:11px; margin-bottom:6px;">` +
            `Live exploration — seed ${seedHex} (past peak t${member.peak}) — ` +
            `tick ${_demoTick}, ${elapsed}s</div>`;
    } else {
        html += `<div style="color:#66ccff; font-size:11px; margin-bottom:6px;">` +
            `Replaying seed ${seedHex} — tick ${_demoTick} / peak t${member.peak} — ${elapsed}s</div>`;
    }

    // Blacklist stats
    html += `<div style="font-size:10px; color:#aaa;">` +
        `Blacklist: ${_sweepTotalBlacklisted.toLocaleString()} states, ` +
        `hits: ${_sweepBlacklistHits.toLocaleString()}</div>`;

    // Stop button
    html += `<div style="margin-top:6px;"><button onclick="_stopSweep()" ` +
        `style="font-size:10px; padding:2px 8px; cursor:pointer;">Stop Replay</button></div>`;

    el.innerHTML = html;
}

// Populate council dropdown — works with cold stubs (no snapshots in RAM)
async function _populateCouncilDropdown() {
    const sel = document.getElementById('council-replay-select');
    if (!sel) return;

    const slider = document.getElementById('lattice-slider');
    const lvl = slider ? +slider.value : 2;

    // Always load from IDB when no sweep is active (rules may have changed)
    let council = [];
    if (_sweepActive) {
        council = _sweepGoldenCouncil;
    } else {
        try {
            const cached = await _blIDBLoad(lvl);
            if (cached && cached.goldenCouncil) {
                council = cached.goldenCouncil;
            }
        } catch (e) {
            console.warn('[Council dropdown] Failed to load:', e);
        }
        // Persist loaded stubs so replay can find them by index
        _sweepGoldenCouncil = council;
        // Trim to current max size
        const _maxC2 = _goldenCouncilSize();
        if (_sweepGoldenCouncil.length > _maxC2) _sweepGoldenCouncil.length = _maxC2;
    }

    // Sort by peak descending
    council.sort((a, b) => (b.peak || 0) - (a.peak || 0));

    sel.innerHTML = '';
    const blankOpt = document.createElement('option');
    blankOpt.value = '';
    blankOpt.textContent = 'New run';
    sel.appendChild(blankOpt);

    for (let i = 0; i < council.length; i++) {
        const m = council[i];
        const seedHex = '0x' + m.seed.toString(16).padStart(8, '0');
        const opt = document.createElement('option');
        opt.value = i;
        opt.textContent = `${seedHex} (t${m.peak})`;
        sel.appendChild(opt);
    }
}

function _updateSweepPanel(message, sweepStartTime) {
    const el = document.getElementById('bfs-test-results');
    if (!el) return;

    let html = '';

    // Header
    html += `<div style="color:#9abccc; font-size:11px; margin-bottom:6px;">`;
    if (message) {
        html += message;
    } else {
        html += `Take ${_sweepSeedIdx + 1} &mdash; tick ${_demoTick}, retries ${_totalBacktrackRetries}, ` +
            `layer ${typeof _bfsLayer !== 'undefined' ? _bfsLayer : '?'}`;
    }
    html += `</div>`;

    // Learnings stats
    html += `<div style="padding:4px; background:rgba(100,180,255,0.06); border:1px solid rgba(100,180,255,0.15); border-radius:3px; margin-bottom:6px;">`;
    html += `<div style="font-size:10px; color:#9abccc;">` +
        `Learnings: <b>${_sweepTotalBlacklisted.toLocaleString()}</b>` +
        ` &middot; Applied: <b>${_sweepBlacklistHits.toLocaleString()}</b> (${_sweepBlacklistHitsSeed} this take)</div>`;
    if (_sweepGoldenCouncil.length > 0 || (_demoActive && _lastAutosavePeak > 0)) {
        // Build sorted entries: council members, marking the current round's seed with green *
        const hasRecentAutosave = _demoActive && _maxTickReached > 0
            && _lastAutosavePeak > 0 && (_maxTickReached - _lastAutosavePeak) < 100
            && typeof _isCouncilEligible === 'function' && _isCouncilEligible();
        const liveSeed = hasRecentAutosave ? (_forceSeed || _runSeed || 0) : null;
        const entries = _sweepGoldenCouncil.map(m => ({ peak: m.peak, live: liveSeed !== null && m.seed === liveSeed }));
        // If live run isn't in council yet but is autosave-eligible, add it as a separate green entry
        if (hasRecentAutosave && !entries.some(e => e.live)) {
            entries.push({ peak: _lastAutosavePeak, live: true });
        }
        entries.sort((a, b) => b.peak - a.peak);
        const peakStrs = entries.map(e => e.live
            ? `<span style="color:#80ff80;">t${e.peak}*</span>`
            : 't' + e.peak);
        html += `<div style="font-size:10px; color:#ffcc66; margin-top:2px;">` +
            `Best Takes: ${peakStrs.join(', ')} &middot; ` +
            `Votes: <b>${_sweepGoldenHits.toLocaleString()}</b> (${_sweepGoldenHitsSeed} this take)</div>`;
        if (hasRecentAutosave) {
            html += `<div style="font-size:9px; color:#80ff80; margin-top:1px;">* autosaved</div>`;
        }
    }
    html += `</div>`;

    // ── Seed peak-tick sparkline bar chart (same style as ratio accuracy) ──
    if (_sweepResults.length > 0) {
        const SPARK_SLOTS = 32;
        const SPARK = ['\u2581', '\u2582', '\u2583', '\u2584', '\u2585', '\u2586', '\u2587', '\u2588'];
        const peaks = _sweepResults.map(r => r.maxTick || 0);
        const chartData = peaks.slice(-SPARK_SLOTS);
        const chartMax = Math.max(...chartData, 1);
        const chartMin = 0;
        let sparkline = '';
        for (const v of chartData) {
            const norm = Math.max(0, Math.min(1, (v - chartMin) / (chartMax - chartMin)));
            const idx = Math.min(7, Math.floor(norm * 7.99));
            // Color: green if high, orange if mid, red if low
            const pct = chartMax > 0 ? v / chartMax : 0;
            const c = pct >= 0.7 ? '#66dd66' : pct >= 0.3 ? '#ccaa66' : '#cc5544';
            sparkline += `<span style="color:${c};">${SPARK[idx]}</span>`;
        }
        html += `<div style="margin-top:4px; overflow:hidden; width:100%;">`
            + `<div style="font-size:12px; color:#667788; margin-bottom:2px;">Progress (last ${chartData.length})</div>`
            + `<div style="font-size:33px; letter-spacing:-1px; line-height:1; font-family:monospace; white-space:nowrap; overflow:hidden;">${sparkline}</div>`
            + `<div style="display:flex; justify-content:space-between; font-size:11px; color:#445566; margin-top:2px;">`
            + `<span>t0</span><span>t${chartMax}</span></div>`
            + `</div>`;
    }

    // ── Blacklist contribution sparkline bar chart ──
    if (_sweepResults.length > 0) {
        const SPARK_SLOTS = 32;
        const SPARK = ['\u2581', '\u2582', '\u2583', '\u2584', '\u2585', '\u2586', '\u2587', '\u2588'];
        const bls = _sweepResults.map(r => r.newBlacklisted || 0);
        const chartData = bls.slice(-SPARK_SLOTS);
        const chartMax = Math.max(...chartData, 1);
        let sparkline = '';
        for (const v of chartData) {
            const norm = Math.max(0, Math.min(1, v / chartMax));
            const idx = Math.min(7, Math.floor(norm * 7.99));
            const c = v === 0 ? '#445566' : norm >= 0.7 ? '#66aaff' : norm >= 0.3 ? '#7799bb' : '#556688';
            sparkline += `<span style="color:${c};">${SPARK[idx]}</span>`;
        }
        html += `<div style="margin-top:4px; overflow:hidden; width:100%;">`
            + `<div style="font-size:12px; color:#667788; margin-bottom:2px;">Learning Rate (last ${chartData.length})</div>`
            + `<div style="font-size:33px; letter-spacing:-1px; line-height:1; font-family:monospace; white-space:nowrap; overflow:hidden;">${sparkline}</div>`
            + `<div style="display:flex; justify-content:space-between; font-size:11px; color:#445566; margin-top:2px;">`
            + `<span>0</span><span>${chartMax.toLocaleString()}</span></div>`
            + `</div>`;
    }

    // Download button placeholder (listener attached after innerHTML)
    if (_sweepResults.length > 0 && !_sweepActive) {
        html += `<button id="btn-sweep-download" style="margin-top:6px;padding:4px 10px;font-size:10px;cursor:pointer;` +
            `background:#1a3a4a;color:#9abccc;border:1px solid #3a6a7a;border-radius:3px;display:block;width:100%;"` +
            `>Download Sweep Log</button>`;
    }

    el.innerHTML = html;

    // Attach download listener after innerHTML is set
    const dlBtn = document.getElementById('btn-sweep-download');
    if (dlBtn) dlBtn.addEventListener('click', _downloadSweepLog);

    // Clear cache button removed from choreographer panel — use the standalone button instead

    // Hide old traversal log button during sweep
    const oldDlBtn = document.getElementById('btn-traversal-log');
    if (oldDlBtn) oldDlBtn.style.display = 'none';
}

// ── Compact event encoding for download ──
// Strips candidates from rewind events (option 1), keeps them only on success/escalation (option 2),
// and uses compact string encoding for candidates (option 3): "node:score" or "node:score:x" if excluded.
function _compactEvent(e) {
    const compact = {
        id: e.eventId, nid: e.nodeId, pid: e.parentId,
        t: e.tick, r: e.retry, L: e.bfsLayer,
        fp: e.fingerprint, o: e.outcome,
    };
    // Wall: only include type + guard ID (strip verbose details)
    if (e.wall) {
        const guardMatch = e.wall.details?.[0]?.match(/^(T\d+\w*)/);
        compact.w = guardMatch ? guardMatch[1] : e.wall.type;
    }
    // Moves: compact "xonIdx:from>to:mode[0]" e.g. "0:9>13:o"
    if (e.moves && e.moves.length > 0) {
        compact.m = e.moves.map(m => m.xonIdx + ':' + m.from + '>' + m.to + ':' + m.mode[0]);
    }
    // Candidates: only on success/escalation/canary events (not rewinds — option 1+2)
    if (e.outcome !== 'rewind' && e.candidates) {
        const cc = {};
        for (const [xi, arr] of Object.entries(e.candidates)) {
            if (arr.length > 0) {
                // Compact: "node:score" or "node:score:x" if excluded (option 3)
                cc[xi] = arr.map(c => c.node + ':' + (c.score || 0) + (c.excluded ? ':x' : ''));
            }
        }
        if (Object.keys(cc).length > 0) compact.c = cc;
    }
    // Exclusion count (not full list)
    if (e.exclusionTotal > 0) compact.ex = e.exclusionTotal;
    return compact;
}

function _downloadSweepLog() {
    if (!_sweepResults || _sweepResults.length === 0) return;

    const seeds = _sweepResults.map(r => ({
        seedIdx: r.seedIdx,
        seed: '0x' + (r.seed >>> 0).toString(16).padStart(8, '0'),
        summary: {
            maxTick: r.maxTick,
            totalRetries: r.totalRetries,
            totalFingerprints: r.totalFingerprints,
            haltReason: r.haltReason,
            haltViolation: r.haltViolation || '',
            elapsedMs: Math.round(r.elapsedMs),
            newBlacklisted: r.newBlacklisted || 0,
        },
        events: (r.traversalLog || []).map(_compactEvent),
    }));

    const payload = {
        version: 4,
        mode: 'sweep',
        timestamp: new Date().toISOString(),
        latticeLevel: typeof latticeLevel !== 'undefined' ? latticeLevel : '?',
        rules: {
            t20Strict: _ruleT20StrictMode,
            maxFullOctTicks: T79_MAX_FULL_TICKS,
            octCapacityMax: OCT_CAPACITY_MAX,
            gluonMediated: _ruleGluonMediatedSC,
            bareTetrahedra: _ruleBareTetrahedra,
        },
        totalSeeds: _sweepResults.length,
        totalBlacklisted: _sweepTotalBlacklisted,
        seeds,
    };

    // No pretty-print — compact JSON saves ~40% more
    const blob = new Blob([JSON.stringify(payload)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `sweep-log-L${payload.latticeLevel}-${payload.totalSeeds}seeds-${new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function _downloadTraversalLog() {
    if (!_bfsTestResults || _bfsTestResults.length < 2) return;

    const runs = _bfsTestResults.map(r => ({
        mode: r.mode || 'unknown',
        seed: '0x' + (r.seed >>> 0).toString(16).padStart(8, '0'),
        summary: {
            maxTick: r.maxTick,
            totalRetries: r.totalRetries,
            totalFingerprints: r.totalFingerprints,
            haltReason: r.haltReason,
            haltViolation: r.haltViolation || '',
            elapsedMs: Math.round(r.elapsedMs),
        },
        events: (r.traversalLog || []).map(_compactEvent),
    }));

    const payload = {
        version: 4,
        timestamp: new Date().toISOString(),
        latticeLevel: typeof latticeLevel !== 'undefined' ? latticeLevel : '?',
        rules: {
            t20Strict: _ruleT20StrictMode,
            maxFullOctTicks: T79_MAX_FULL_TICKS,
            octCapacityMax: OCT_CAPACITY_MAX,
            gluonMediated: _ruleGluonMediatedSC,
            bareTetrahedra: _ruleBareTetrahedra,
        },
        comparison: _bfsTestComparison ? {
            identical: _bfsTestComparison.identical,
            earlyAbort: _bfsTestComparison.earlyAbort || false,
            novelCount: _bfsTestComparison.novelCount || 0,
            summary: _bfsTestComparison.summary,
        } : null,
        runs,
    };

    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `traversal-log-L${payload.latticeLevel}-${new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Show/hide BFS panel and move to top of left panels during tests
function _setBfsTestPanelVisible(visible) {
    const bfsSection = document.getElementById('bfs-section');
    const qbSection = document.getElementById('quark-balance-section');
    if (!bfsSection) return;
    if (visible) {
        bfsSection.style.display = '';
        // Move BFS section before quark balance (top of panel)
        if (qbSection && bfsSection.parentNode === qbSection.parentNode) {
            qbSection.parentNode.insertBefore(bfsSection, qbSection);
        }
    } else {
        bfsSection.style.display = 'none';
    }
}

// ── Export BFS test results to JSON file ──
function _exportBfsTestResults() {
    if (!_bfsTestResults[0] || !_bfsTestResults[1] || !_bfsTestComparison) {
        alert('No BFS test results to export. Run the test first.');
        return;
    }

    const now = new Date();
    const timestamp = now.toISOString().replace(/[:.]/g, '-');

    // Build per-run detail objects
    function buildRunDetail(result, label) {
        // Per-tick fingerprint breakdown: which solutions were tried at each tick, in order
        const perTickDetail = [];
        const sortedTicks = [...result.perTickFingerprints.keys()].sort((a, b) => a - b);
        for (const tick of sortedTicks) {
            const fps = [...result.perTickFingerprints.get(tick)];
            perTickDetail.push({
                tick,
                fingerprintCount: fps.length,
                fingerprints: fps,
            });
        }

        // Per-tick bad-move ledger (exclusions that accumulated)
        const perTickExclusions = [];
        const ledgerTicks = [...result.perTickLedger.keys()].sort((a, b) => a - b);
        for (const tick of ledgerTicks) {
            const exclusions = [...result.perTickLedger.get(tick)];
            perTickExclusions.push({
                tick,
                exclusionCount: exclusions.length,
                exclusions,
            });
        }

        // Identify starting fingerprint (tick 0, first entry) and longest/stopping ticks
        const startingFP = sortedTicks.length > 0
            ? [...result.perTickFingerprints.get(sortedTicks[0])][0] || null
            : null;

        // The "longest found" is the highest tick that has any fingerprint
        const longestTick = sortedTicks.length > 0 ? sortedTicks[sortedTicks.length - 1] : 0;
        const longestFPs = longestTick >= 0 && result.perTickFingerprints.has(longestTick)
            ? [...result.perTickFingerprints.get(longestTick)]
            : [];

        // Stopping solution: the last fingerprint tried at the highest tick
        const stoppingFP = longestFPs.length > 0 ? longestFPs[longestFPs.length - 1] : null;

        return {
            label,
            seed: '0x' + result.seed.toString(16).padStart(8, '0'),
            seedDecimal: result.seed,
            maxTickReached: result.maxTick,
            haltReason: result.haltReason,
            haltViolation: result.haltViolation || null,
            totalBacktrackRetries: result.totalRetries,
            totalUniqueFingerprints: result.totalFingerprints,
            searchTimeMs: Math.round(result.elapsedMs),
            searchTimeSec: +(result.elapsedMs / 1000).toFixed(2),
            ticksExplored: sortedTicks.length,
            startingSolution: startingFP,
            longestSolutionTick: longestTick,
            longestSolutions: longestFPs,
            stoppingSolution: stoppingFP,
            perTickFingerprints: perTickDetail,
            perTickExclusions,
        };
    }

    const runA = buildRunDetail(_bfsTestResults[0], 'Test 1: CHOREOGRAPHER (normal)');
    runA.choreographerMode = _bfsTestResults[0].mode || 'choreographer';
    const runB = buildRunDetail(_bfsTestResults[1], 'Test 2: RANDOM');
    runB.choreographerMode = _bfsTestResults[1].mode || 'random';

    // Comparison detail
    const cmp = _bfsTestComparison;
    const comparison = {
        verdict: cmp.identical ? 'EXHAUSTIVE (PASS)' : 'DIVERGENT (FAIL)',
        identical: cmp.identical,
        sameLongestSolution: cmp.sameLongestSolution,
        sameTotalFingerprints: cmp.sameTotalFingerprints,
        differentPaths: cmp.differentPaths,
        maxTickMatch: cmp.maxTickMatch,
        haltReasonMatch: cmp.haltReasonMatch,
        violationMatch: cmp.violationMatch,
        allFingerprintsMatch: cmp.allFPMatch,
        summary: cmp.summary,
        divergentTicks: cmp.fingerprintDiff.ticksMismatch.map(m => ({
            tick: m.tick,
            sharedFingerprints: m.shared,
            onlyInChoreographer: m.onlyA,
            onlyInRandom: m.onlyB,
        })),
        ticksOnlyInChoreographer: cmp.fingerprintDiff.ticksOnlyA,
        ticksOnlyInRandom: cmp.fingerprintDiff.ticksOnlyB,
    };

    // Fingerprint overlap analysis: for each tick, what % of fingerprints are shared?
    const overlapAnalysis = [];
    const allTicks = new Set([
        ..._bfsTestResults[0].perTickFingerprints.keys(),
        ..._bfsTestResults[1].perTickFingerprints.keys(),
    ]);
    for (const tick of [...allTicks].sort((a, b) => a - b)) {
        const setA = _bfsTestResults[0].perTickFingerprints.get(tick);
        const setB = _bfsTestResults[1].perTickFingerprints.get(tick);
        const countA = setA ? setA.size : 0;
        const countB = setB ? setB.size : 0;
        let shared = 0;
        if (setA && setB) {
            for (const fp of setA) { if (setB.has(fp)) shared++; }
        }
        const union = countA + countB - shared;
        overlapAnalysis.push({
            tick,
            countA,
            countB,
            shared,
            onlyA: countA - shared,
            onlyB: countB - shared,
            jaccardSimilarity: union > 0 ? +(shared / union).toFixed(4) : 1,
        });
    }

    // Aggregate stats
    const totalShared = overlapAnalysis.reduce((s, o) => s + o.shared, 0);
    const totalUnion = overlapAnalysis.reduce((s, o) => s + o.countA + o.countB - o.shared, 0);

    const report = {
        exportedAt: now.toISOString(),
        latticeLevel: typeof latticeLevel !== 'undefined' ? latticeLevel : 'unknown',
        nodeCount: typeof pos !== 'undefined' ? pos.length : 'unknown',
        xonCount: typeof _demoXons !== 'undefined' ? _demoXons.length : 6,
        comparison,
        aggregateStats: {
            totalFingerprintsA: runA.totalUniqueFingerprints,
            totalFingerprintsB: runB.totalUniqueFingerprints,
            totalSharedFingerprints: totalShared,
            overallJaccardSimilarity: totalUnion > 0 ? +(totalShared / totalUnion).toFixed(4) : 1,
            totalSearchTimeMs: runA.searchTimeMs + runB.searchTimeMs,
            totalBacktrackRetries: runA.totalBacktrackRetries + runB.totalBacktrackRetries,
        },
        runA,
        runB,
        perTickOverlap: overlapAnalysis,
    };

    // Download as JSON
    const json = JSON.stringify(report, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `dfs-exhaustiveness-${timestamp}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    // Console summary
    const sz = (json.length / 1024).toFixed(1);
    console.log(`%c[DFS TEST] ══════════════════════════════════════════`, 'color:cyan');
    console.log(`%c[DFS TEST] Exported ${sz} KB report`, 'color:lime');
    console.log(`%c[DFS TEST] Verdict: ${comparison.verdict}`, comparison.identical ? 'color:lime;font-weight:bold' : 'color:red;font-weight:bold');
    console.log(`  ✦ Same longest solution: ${comparison.sameLongestSolution ? '✓ YES' : '✗ NO'}`);
    console.log(`  ✦ Same total fingerprints: ${comparison.sameTotalFingerprints ? '✓ YES' : '✗ NO'} (${runA.totalUniqueFingerprints} vs ${runB.totalUniqueFingerprints})`);
    console.log(`  ✦ Took different paths: ${comparison.differentPaths ? '✓ YES' : '✗ NO'} (retries: ${runA.totalBacktrackRetries} vs ${runB.totalBacktrackRetries})`);
    console.log(`  Test 1 (CHOREOGRAPHER): highest=${runA.maxTickReached}, fps=${runA.totalUniqueFingerprints}, ${runA.searchTimeSec}s`);
    console.log(`  Test 2 (RANDOM): highest=${runB.maxTickReached}, fps=${runB.totalUniqueFingerprints}, ${runB.searchTimeSec}s`);
    console.log(`  Starting solution (choreographer): ${runA.startingSolution}`);
    console.log(`  Starting solution (random): ${runB.startingSolution}`);
    console.log(`  Longest solution tick: ${runA.longestSolutionTick} (choreographer) vs ${runB.longestSolutionTick} (random)`);
    console.log(`%c[DFS TEST] ══════════════════════════════════════════`, 'color:cyan');
}

// ── Wire up nucleus UI ──
(function(){
    NucleusSimulator.populateModelSelect();

    // Static clear cache button (always visible in H-2 panel)
    document.getElementById('btn-clear-cache-static')?.addEventListener('click', async function() {
        if (this.dataset.confirming) return;
        this.dataset.confirming = '1';
        const origText = this.textContent;
        this.textContent = 'Confirm? Click again to clear';
        this.style.background = 'rgba(255,60,60,0.2)';
        const timeout = setTimeout(() => {
            delete this.dataset.confirming;
            this.textContent = origText;
            this.style.background = '';
        }, 3000);
        this.addEventListener('click', async function onConfirm() {
            this.removeEventListener('click', onConfirm);
            clearTimeout(timeout);
            delete this.dataset.confirming;
            await _clearCacheExecute();
            this.textContent = 'Cache cleared!';
            this.style.background = '';
            setTimeout(() => { this.textContent = origText; }, 2000);
        }, { once: true });
    });

    // Play button — new run or council replay depending on dropdown selection
    document.getElementById('btn-simulate-nucleus')?.addEventListener('click', function(){
        if (_sweepActive || _bfsTestActive || _demoActive) return;
        const slider = document.getElementById('lattice-slider');
        const lvl = slider ? +slider.value : 2;
        const sel = document.getElementById('council-replay-select');
        const selectedVal = sel ? sel.value : '';
        if (selectedVal && selectedVal !== '') {
            const idx = parseInt(selectedVal, 10);
            if (!isNaN(idx)) {
                startCouncilReplay(idx);
                return;
            }
        }
        startSweepTest(lvl);
    });

    // Tournament button
    document.getElementById('btn-tournament')?.addEventListener('click', function(){
        if(tournamentActive) stopTournament();
        else startTournament();
    });

    // Populate council dropdown on page load
    _populateCouncilDropdown();

    // Re-populate when lattice slider changes
    document.getElementById('lattice-slider')?.addEventListener('change', function(){
        _populateCouncilDropdown();
    });

    // Re-populate when rule checkboxes/sliders change (affects the blacklist rule key)
    for (const id of ['rule-t20-strict-toggle', 'rule-gluon-mediated-toggle', 'rule-bare-tet-toggle']) {
        document.getElementById(id)?.addEventListener('change', function(){
            _populateCouncilDropdown();
        });
    }
    for (const id of ['rule-oct-full-slider', 'rule-oct-capacity-slider']) {
        document.getElementById(id)?.addEventListener('input', function(){
            _populateCouncilDropdown();
        });
    }

    // BFS export button
    document.getElementById('btn-bfs-export')?.addEventListener('click', function(){
        _exportBfsTestResults();
    });

    // Play/pause button — pauses/resumes the demo tick interval
    document.getElementById('btn-nucleus-pause')?.addEventListener('click', function(){
        // Movie playback mode: toggle pause
        if (_playbackMode) {
            _demoPaused = !_demoPaused;
            this.textContent = _demoPaused ? '\u25B6' : '\u23F8';
            this.title = _demoPaused ? 'Resume playback' : 'Pause playback';
            return;
        }
        if (typeof isDemoPaused === 'function' && _demoActive) {
            // If reversing, just stop reverse and stay paused — don't toggle
            if (_demoReversing) {
                stopReverse();
                this.textContent = '\u25B6';
                this.title = 'Resume simulation';
                return;
            }
            if (!isDemoPaused()) {
                pauseDemo();
                this.textContent = '\u25B6';
                this.title = 'Resume simulation';
            } else {
                resumeDemo();
                this.textContent = '\u23F8';
                this.title = 'Pause simulation';
            }
        } else if (excitationClockTimer) {
            stopExcitationClock();
            this.textContent = '▶';
            this.title = 'Resume simulation';
            document.getElementById('nucleus-status').textContent = 'paused';
        } else {
            startExcitationClock();
            this.textContent = '⏸';
            this.title = 'Pause simulation';
            document.getElementById('nucleus-status').textContent = 'running';
        }
    });

    // ── Playback controls ──
    document.getElementById('btn-step-back')?.addEventListener('click', function() {
        if (_playbackMode && _importedMovie) {
            _demoPaused = true;
            if (_playbackFrame > 0) { _playbackFrame--; _pbPosCache = null; _applyMovieFrame(_playbackFrame); }
            const s = document.getElementById('timeline-scrubber'); if (s) s.value = _playbackFrame;
            const v = document.getElementById('timeline-val'); if (v) v.textContent = _playbackFrame;
            return;
        }
        if (!_demoActive) return;
        if (_demoReversing) stopReverse();
        if (!_demoPaused) pauseDemo();
        _playbackStepBack();
        const pb = document.getElementById('btn-nucleus-pause');
        if (pb) pb.textContent = '\u25B6';
    });

    document.getElementById('btn-step-forward')?.addEventListener('click', function() {
        if (_playbackMode && _importedMovie) {
            _demoPaused = true;
            if (_playbackFrame < _importedMovie.totalFrames - 1) { _playbackFrame++; _applyMovieFrame(_playbackFrame); }
            const s = document.getElementById('timeline-scrubber'); if (s) s.value = _playbackFrame;
            const v = document.getElementById('timeline-val'); if (v) v.textContent = _playbackFrame;
            return;
        }
        if (!_demoActive) return;
        if (_demoReversing) stopReverse();
        if (!_demoPaused) pauseDemo();
        _playbackStepForward();
        const pb = document.getElementById('btn-nucleus-pause');
        if (pb) pb.textContent = '\u25B6';
    });

    document.getElementById('btn-reverse')?.addEventListener('click', function() {
        if (_playbackMode) {
            _demoReversing = !_demoReversing;
            _demoPaused = false;
            _updatePlaybackButtons();
            return;
        }
        if (!_demoActive) return;
        if (_demoReversing) {
            stopReverse();
        } else {
            startReverse();
        }
    });

    // Rewind all the way to t=0
    document.getElementById('btn-rewind-start')?.addEventListener('click', function() {
        if (_playbackMode && _importedMovie) {
            _demoPaused = true; _playbackFrame = 0; _pbPosCache = null;
            _applyMovieFrame(0);
            const s = document.getElementById('timeline-scrubber'); if (s) s.value = 0;
            const v = document.getElementById('timeline-val'); if (v) v.textContent = 0;
            return;
        }
        if (!_demoActive) return;
        if (_demoReversing) stopReverse();
        if (!_demoPaused) pauseDemo();
        // Jump to t=0: restore first snapshot, set cursor
        if (_btSnapshots.length > 0) {
            _replayCursor = 0;
            _btRestoreSnapshot(_btSnapshots[0]);
        }
        simHalted = false;
        _bfsReset(); _btReset();
        if (typeof _liveGuardResetForRewind === 'function') _liveGuardResetForRewind();
        _tickLog.length = 0;
        _playbackUpdateDisplay();
        const pb = document.getElementById('btn-nucleus-pause');
        if (pb) pb.textContent = '\u25B6';
    });

    // Fast-forward: drain redo stack instantly, then pause
    document.getElementById('btn-forward-end')?.addEventListener('click', function() {
        if (_playbackMode && _importedMovie) {
            _demoPaused = true; _playbackFrame = _importedMovie.totalFrames - 1;
            _pbPosCache = null; _applyMovieFrame(_playbackFrame);
            const s = document.getElementById('timeline-scrubber'); if (s) s.value = _playbackFrame;
            const v = document.getElementById('timeline-val'); if (v) v.textContent = _playbackFrame;
            return;
        }
        if (!_demoActive) return;
        if (_demoReversing) stopReverse();
        if (!_demoPaused) pauseDemo();
        // Jump to end: restore last snapshot, clear cursor
        if (_btSnapshots.length > 0) {
            _replayCursor = -1;
            _btRestoreSnapshot(_btSnapshots[_btSnapshots.length - 1]);
        }
        simHalted = false;
        _bfsReset(); _btReset();
        if (typeof _liveGuardResetForRewind === 'function') _liveGuardResetForRewind();
        _playbackUpdateDisplay();
        const pb = document.getElementById('btn-nucleus-pause');
        if (pb) pb.textContent = '\u25B6';
    });

    document.getElementById('btn-export-log')?.addEventListener('click', function() {
        exportTickLog();
    });

    // Import replay button
    document.getElementById('btn-import-replay')?.addEventListener('click', function() {
        document.getElementById('replay-file-input')?.click();
    });
    document.getElementById('replay-file-input')?.addEventListener('change', function(e) {
        if (e.target.files && e.target.files[0]) {
            importReplay(e.target.files[0]);
            e.target.value = ''; // reset so same file can be re-imported
        }
    });

    // ── Test Replay Button ──
    document.getElementById('btn-test-replay')?.addEventListener('click', function(){
        // Clear any previous test state
        localStorage.removeItem(_REPLAY_TEST_KEY);
        if (typeof _clearReplayCorruption === 'function') _clearReplayCorruption();
        if (_sweepActive || _bfsTestActive || _demoActive) return;
        _startReplayTest();
    });

    // ── On-load: check for pending replay test phase ──
    const _rTestQ = localStorage.getItem('flux_replay_test');
    if (_rTestQ) {
        try {
            const q = JSON.parse(_rTestQ);
            if (q && q.phase === 'done' && q.result && q.result.startsWith('FAIL')) {
                // Test fixture: re-run the replay phase on every reload.
                // The IDB data is our fixture. Replay it, hit the failure, stop.
                // localStorage is NOT cleared — this is the persistent fixture.
                setTimeout(() => {
                    _replayTestLog(`Re-running test fixture (previous: ${q.result})`);
                    // Reset phase to replay so _runReplayTestPhase picks it up
                    const replayQ = { phase: 'replay', targetPS: q.targetPS || 100,
                        seed: q.seed, startedAt: q.startedAt };
                    _runReplayTestPhase(replayQ);
                }, 1500);
            } else if (q && q.phase === 'done' && q.result) {
                // PASS result — show static message
                setTimeout(() => {
                    _replayTestLog(`Previous test result: ${q.result}`);
                }, 1000);
            } else if (q && q.phase && q.phase !== 'done') {
                // Interrupted mid-pipeline — resume the phase
                setTimeout(() => _runReplayTestPhase(q), 1500);
            }
        } catch (e) { localStorage.removeItem('flux_replay_test'); }
    }

    // Stop/clear button
    document.getElementById('btn-stop-nucleus')?.addEventListener('click', function(){
        NucleusSimulator.deactivate();
        activeSet.clear();
        impliedSet.clear(); xonImpliedSet.clear(); blockedImplied.clear(); impliedBy.clear();
        while(excitations.length > 0){
            const e = excitations.pop();
            if(e.group) scene.remove(e.group);
            if(e.trailLine) scene.remove(e.trailLine);
        }
        if(typeof stopExcitationClock === 'function') stopExcitationClock();
        bumpState();
        const pFinal = detectImplied();
        applyPositions(pFinal);
        updateCandidates(); updateSpheres(); updateStatus();
        rebuildShortcutLines();
        updateExcitationSidebar();
        // Reset pause button state
        const pauseBtn = document.getElementById('btn-nucleus-pause');
        if(pauseBtn){ pauseBtn.textContent = '⏸'; pauseBtn.title = 'Pause simulation'; }
    });

})();

// ╔══════════════════════════════════════════════════════════════════════╗
// ║  REPLAY INTEGRITY TEST PIPELINE                                     ║
// ║  Record → Save → Reload → Replay w/ Guards → Scrub → Extend        ║
// ╚══════════════════════════════════════════════════════════════════════╝

const _REPLAY_TEST_KEY = 'flux_replay_test';
const _REPLAY_TEST_TARGETS = [100, 200, 300]; // planck-second milestones

function _replayTestLog(msg) {
    console.log(`%c[REPLAY TEST] ${msg}`, 'color:#ff66ff;font-weight:bold');
}

function _replayTestFail(phase, tick, reason) {
    _guardHardStop = false;  // stop guards from re-triggering
    const q = { phase: 'done', result: `FAIL @ ${phase} t=${tick}: ${reason}` };
    localStorage.setItem(_REPLAY_TEST_KEY, JSON.stringify(q));
    _replayTestLog(`FAILED: ${q.result}`);
    if (typeof _showReplayCorruption === 'function') _showReplayCorruption(tick, reason);
    // Kill the sweep so it doesn't death-loop through more seeds.
    // Keep _demoActive = true so the timeline scrubber remains functional.
    simHalted = true;
    _demoPaused = true;
    _sweepActive = false;
    _sweepReplayActive = false;
    _sweepReplayMember = null;
    if (_demoInterval) { clearInterval(_demoInterval); _demoInterval = null; }
    if (_demoUncappedId) { clearTimeout(_demoUncappedId); _demoUncappedId = null; }
    // Update scrubber so user can scrub through the failed replay
    if (typeof _updateTimelineScrubber === 'function') _updateTimelineScrubber();
    // Update pause button to show play icon
    const pb = document.getElementById('btn-nucleus-pause');
    if (pb) { pb.textContent = '\u25B6'; pb.title = 'Resume simulation'; }
}

function _replayTestPass() {
    const q = { phase: 'done', result: 'PASS — 300ps replay perfection achieved' };
    localStorage.setItem(_REPLAY_TEST_KEY, JSON.stringify(q));
    _replayTestLog('PASSED — all milestones verified');
    if (typeof _clearReplayCorruption === 'function') _clearReplayCorruption();
    const pb = document.getElementById('btn-nucleus-pause');
    if (pb) {
        let el = document.getElementById('replay-corruption-msg');
        if (!el) {
            el = document.createElement('span');
            el.id = 'replay-corruption-msg';
            el.style.cssText = 'color:#44ff44;font-weight:bold;display:block;font-size:11px;margin-top:4px;';
            pb.parentNode.appendChild(el);
        }
        el.style.color = '#44ff44';
        el.textContent = 'REPLAY TEST PASSED — 300ps verified';
    }
}

// Phase: record — clear IDB, run sweep until target planck-seconds, save, reload
async function _startReplayTest() {
    _replayTestLog('Starting replay integrity test — clearing IDB for default ruleset...');
    if (typeof _clearReplayCorruption === 'function') _clearReplayCorruption();

    // Clear IDB for default ruleset
    const slider = document.getElementById('lattice-slider');
    const lvl = slider ? +slider.value : 2;
    if (_blIDB) {
        try {
            const baseKey = _blacklistRuleKey(lvl);
            const tx = _blIDB.transaction([_BL_IDB_STORE, _CS_IDB_STORE], 'readwrite');
            const blStore = tx.objectStore(_BL_IDB_STORE);
            const csStore = tx.objectStore(_CS_IDB_STORE);
            // Delete all keys with this base prefix
            const blKeys = await new Promise(r => { const req = blStore.getAllKeys(); req.onsuccess = () => r(req.result); req.onerror = () => r([]); });
            for (const k of blKeys) { if (typeof k === 'string' && k.startsWith(baseKey)) blStore.delete(k); }
            const csKeys = await new Promise(r => { const req = csStore.getAllKeys(); req.onsuccess = () => r(req.result); req.onerror = () => r([]); });
            for (const k of csKeys) { if (typeof k === 'string' && k.startsWith(baseKey)) csStore.delete(k); }
        } catch (e) { console.warn('[REPLAY TEST] IDB clear error:', e); }
    }

    // Reset council
    _sweepGoldenCouncil.length = 0;

    const targetPS = _REPLAY_TEST_TARGETS[0];
    const q = { phase: 'record', targetPS, startedAt: Date.now() };
    localStorage.setItem(_REPLAY_TEST_KEY, JSON.stringify(q));

    _replayTestLog(`Recording run to ${targetPS} planck-seconds...`);

    // Start sweep
    _sweepActive = false; // ensure clean
    await new Promise(r => setTimeout(r, 100));

    // Start a normal sweep
    startSweepTest(lvl);

    // Poll until we reach target planck-seconds or sim halts
    const pollId = setInterval(() => {
        if (!_sweepActive && !_demoActive) {
            clearInterval(pollId);
            _replayTestLog('Sweep ended before reaching target — checking council...');
            _finishRecordPhase(targetPS);
            return;
        }
        if (_demoTick >= targetPS) {
            clearInterval(pollId);
            _replayTestLog(`Reached tick ${_demoTick} (target ${targetPS}) — saving...`);
            // Stop sweep and save
            _sweepActive = false;
            setTimeout(() => _finishRecordPhase(targetPS), 500);
        }
    }, 200);
}

function _finishRecordPhase(targetPS) {
    // Force save current run to council
    if (typeof _saveCurrentRunToCouncil === 'function') _saveCurrentRunToCouncil();

    // Wait for IDB write to settle, then check we have a member
    setTimeout(() => {
        if (_sweepGoldenCouncil.length === 0) {
            _replayTestFail('record', _demoTick, 'No council member saved — run did not produce valid data');
            return;
        }
        const member = _sweepGoldenCouncil[0];
        const q = { phase: 'replay', targetPS, seed: member.seed, startedAt: Date.now() };
        localStorage.setItem(_REPLAY_TEST_KEY, JSON.stringify(q));
        _replayTestLog(`Saved seed 0x${member.seed.toString(16).padStart(8,'0')} (peak t${member.peak}) — reloading...`);

        if (typeof stopDemo === 'function') stopDemo();
        setTimeout(() => location.reload(), 500);
    }, 1000);
}

// Dispatch to the correct phase handler
async function _runReplayTestPhase(q) {
    _replayTestLog(`Resuming test — phase: ${q.phase}, targetPS: ${q.targetPS}, seed: 0x${(q.seed || 0).toString(16).padStart(8,'0')}`);

    if (q.phase === 'replay') {
        await _replayTestReplayPhase(q);
    } else if (q.phase === 'extend') {
        await _replayTestExtendPhase(q);
    }
}

// Phase: replay — hydrate from IDB, play with guards
async function _replayTestReplayPhase(q) {
    _replayTestLog('Replay phase — loading from IDB with guard mode...');

    // Wait for IDB
    if (!_blIDBReady) await _blIDBOpen();
    await new Promise(r => setTimeout(r, 500));

    // Populate council from IDB
    const slider = document.getElementById('lattice-slider');
    const lvl = slider ? +slider.value : 2;
    await _populateCouncilDropdown();
    await new Promise(r => setTimeout(r, 300));

    // Find the member by seed
    let memberIdx = -1;
    for (let i = 0; i < _sweepGoldenCouncil.length; i++) {
        if (_sweepGoldenCouncil[i].seed === q.seed) { memberIdx = i; break; }
    }
    if (memberIdx < 0) {
        // Try first member if seed not found
        if (_sweepGoldenCouncil.length > 0) memberIdx = 0;
        else {
            _replayTestFail('replay', 0, 'No council member found in IDB');
            return;
        }
    }

    // Set replay guard mode BEFORE starting replay
    _guardHardStop = true;
    _liveGuardsActive = true;

    // Load blacklist so learnings apply during replay + extend
    const cached = await _blIDBLoad(lvl);
    if (cached) {
        _sweepBlacklist = cached.map;
        _sweepTotalBlacklisted = cached.total;
        _sweepSeedIdx = cached.seedIdx;
        _sweepUsedSeeds = new Set(cached.usedSeeds || []);
        if (cached.goldenCouncil && cached.goldenCouncil.length > 0) {
            _sweepGoldenCouncil = cached.goldenCouncil;
            _trimCouncil(lvl);
        }
        if (_blBucketVersion >= 1) await _blPrefetchBucket(lvl, 0);
    }

    // Start council replay — this calls startSweepSeed directly
    // (sweep is the only execution mode; replay = sweep with pre-seeded redo stack)
    const member = _sweepGoldenCouncil[memberIdx];
    _sweepActive = true;
    await startSweepSeed(member.seed, member, lvl);

    // Poll for redo drain completion or corruption (observer only — no pause/resume)
    await new Promise(resolve => {
        const pollId = setInterval(() => {
            if (simHalted) {
                clearInterval(pollId);
                const failed = Object.entries(_liveGuards).filter(([, g]) => g.failed);
                if (failed.length > 0) {
                    const msg = failed.map(([k, g]) => `${k}: ${g.msg}`).join('; ');
                    _replayTestFail('replay', _demoTick, msg);
                } else {
                    _replayTestFail('replay', _demoTick, 'simHalted without guard failure');
                }
                resolve();
                return;
            }
            // Replay cursor exhausted — replay validated up to member's peak
            if (_demoActive && !_sweepReplayActive && _replayCursor === -1) {
                clearInterval(pollId);
                _replayTestLog(`Replay phase passed — reached tick ${_demoTick}`);
                _guardHardStop = false;

                // Did we reach the current target? If not, extend to it first.
                if (_demoTick < q.targetPS) {
                    _replayTestLog(`Replay ended at tick ${_demoTick}, need ${q.targetPS} — extending...`);
                    const extQ = { ...q, phase: 'extend' };
                    localStorage.setItem(_REPLAY_TEST_KEY, JSON.stringify(extQ));
                    setTimeout(() => _replayTestExtendPhase(extQ), 500);
                    resolve();
                    return;
                }

                // Current target reached — determine next target
                const currentTarget = q.targetPS;
                const nextIdx = _REPLAY_TEST_TARGETS.indexOf(currentTarget) + 1;
                if (nextIdx >= _REPLAY_TEST_TARGETS.length) {
                    // All targets done!
                    _replayTestPass();
                    resolve();
                    return;
                }

                // Move to extend phase for next milestone
                const nextTarget = _REPLAY_TEST_TARGETS[nextIdx];
                const extQ = { ...q, phase: 'extend', targetPS: nextTarget };
                localStorage.setItem(_REPLAY_TEST_KEY, JSON.stringify(extQ));
                _replayTestLog(`Extending to ${nextTarget} planck-seconds...`);
                setTimeout(() => _replayTestExtendPhase(extQ), 500);
                resolve();
            }
        }, 200);
    });
}

// Phase: extend — continue live until target planck-seconds, save, reload
async function _replayTestExtendPhase(q) {
    _replayTestLog(`Extend phase — running live until ${q.targetPS} planck-seconds...`);
    _guardHardStop = false; // live play uses normal guards

    // Ensure demo is running
    if (_demoPaused && typeof resumeDemo === 'function') resumeDemo();

    // Poll for target planck-seconds
    await new Promise(resolve => {
        const pollId = setInterval(() => {
            if (simHalted || !_demoActive) {
                clearInterval(pollId);
                // Sweep regression or guard halt — restart from council replay
                // (same as auto-retry-best behavior in normal sweep).
                _replayTestLog(`Extend halted at tick ${_demoTick} — restarting from council replay...`);
                if (typeof stopDemo === 'function') stopDemo();
                setTimeout(async () => {
                    const replayQ = { ...q, phase: 'replay' };
                    localStorage.setItem(_REPLAY_TEST_KEY, JSON.stringify(replayQ));
                    setTimeout(() => location.reload(), 500);
                    resolve();
                }, 500);
                return;
            }
            if (_demoTick >= q.targetPS) {
                clearInterval(pollId);
                _replayTestLog(`Reached tick ${_demoTick} (target ${q.targetPS}) — saving extended replay...`);

                // Save to council
                if (typeof _saveCurrentRunToCouncil === 'function') _saveCurrentRunToCouncil();

                setTimeout(() => {
                    const replayQ = { ...q, phase: 'replay' };
                    localStorage.setItem(_REPLAY_TEST_KEY, JSON.stringify(replayQ));
                    _replayTestLog(`Saved — reloading for replay verification at ${q.targetPS}ps...`);
                    if (typeof stopDemo === 'function') stopDemo();
                    setTimeout(() => location.reload(), 500);
                    resolve();
                }, 1000);
            }
        }, 200);
    });
}

// Export for console debugging
function _exportReplayFixture() {
    if (_sweepGoldenCouncil.length === 0) {
        console.warn('No council members to export');
        return;
    }
    const member = _sweepGoldenCouncil[0];
    if (member._cold) {
        console.warn('Member is cold — hydrate first by selecting from dropdown');
        return;
    }
    const data = {
        seed: member.seed,
        peak: member.peak,
        snapshots: member.snapshots ? member.snapshots.map(s => _serializeSnapshot(s)) : null,
    };
    localStorage.setItem('flux_test_fixture', JSON.stringify(data));
    console.log(`Exported replay fixture: seed 0x${member.seed.toString(16).padStart(8,'0')}, peak t${member.peak}, ${data.snapshots ? data.snapshots.length : 0} snapshots`);
}

// ╔══════════════════════════════════════════════════════════════════════╗
// ║  TOURNAMENT / RL TRAINING — REMOVED                                ║
// ║  Quark balance with zero jitter proven optimal. Code stubbed out   ║
// ║  in flux-tournament.js / flux-rl.js / flux-rules-v2.js.           ║
// ╚══════════════════════════════════════════════════════════════════════╝

// Compatibility stub — other code may check this flag
let _tournamentRunning = false;
// (Tournament variables, functions, and UI code removed — see stubbed files)

