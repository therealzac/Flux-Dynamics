// flux-tests-idb.js — IDB persistence, blacklist buckets, serialization, council cold storage
// Split from flux-tests.js (lines 2570-3659). Loaded after flux-tests-unit.js.

// ═══ BFS Exhaustiveness Test — Runner ════════════════════════════════════════

function _captureBfsRunResult() {
    const totalFP = [..._btTriedFingerprints.values()].reduce((s, set) => s + set.size, 0);
    return {
        seed: _runSeed,
        maxTick: _maxTickReached,
        haltReason: simHalted ? 'canary' : 'stopped',
        haltViolation: _rewindViolation || '',
        totalRetries: _totalBacktrackRetries,
        totalFingerprints: totalFP,
        perTickFingerprints: new Map(
            [..._btTriedFingerprints].map(([t, s]) => [t, new Set(s)])
        ),
        perTickLedger: new Map(
            [..._btBadMoveLedger].map(([t, s]) => [t, new Set(s)])
        ),
        elapsedMs: performance.now() - _searchStartTime,
        traversalLog: _searchTraversalLog.slice(),
    };
    _searchTraversalLog = []; // reset for next run
    _searchEventCounter = 0;
    _searchPathStack = [];
    _searchParentNodeId = null;
    _searchLastCandidates = null;
}

function _executeBfsTestRun(runIdx) {
    return new Promise((resolve) => {
        _forceSeed = _bfsTestSeeds[runIdx];

        // Start demo loop — nucleus is already set up by the orchestrator
        startDemoLoop();

        // Poll for completion: only canary (simHalted) or demo stopped.
        // No artificial tick limit — exhaustive search must run until BFS
        // either proves the rules impossible (canary at t=0) or succeeds.
        const pollId = setInterval(() => {
            // Update progress panel
            const testLabel = runIdx === 0 ? 'Test 1 (choreographer)' : 'Test 2 (random)';
            const novelWarning = (runIdx === 1 && _bfsTestNovelCount > 0)
                ? `<br><span style="color:#ff4444;font-weight:bold;font-size:1.1em;">⚠ NOT EXHAUSTIVE — ${_bfsTestNovelCount} novel solution${_bfsTestNovelCount !== 1 ? 's' : ''} found</span>`
                : '';
            _updateBfsTestPanel(
                `${testLabel}: tick ${_demoTick}, highest ${_maxTickReached}, ` +
                `retries ${_totalBacktrackRetries}, ` +
                `layer ${typeof _bfsLayer !== 'undefined' ? _bfsLayer : '?'}` +
                novelWarning
            );
            // Check termination: canary fired or demo stopped (no early abort — let random run to completion)
            if (simHalted || !_demoActive) {
                clearInterval(pollId);
                resolve();
            }
        }, 100);
    });
}

function _setBfsTestTitle(runIdx, test1Best) {
    const titleEl = document.getElementById('topbar-title');
    if (!titleEl) return;
    if (runIdx === 0) {
        titleEl.innerHTML = 'Test 1 <span style="font-size:0.6em; color:#66bbff;">CHOREOGRAPHER</span>';
    } else if (runIdx === 1) {
        titleEl.innerHTML = `Test 2 <span style="font-size:0.6em; color:#ff9944;">RANDOM</span><br><span style="font-size:0.65em; color:#556677; font-weight:400;">Test 1 solutions: ${test1Best}</span>`;
    } else {
        // Restore default
        if (typeof RULE_REGISTRY !== 'undefined' && typeof activeRuleIndex !== 'undefined') {
            titleEl.textContent = RULE_REGISTRY[activeRuleIndex]?.name || '';
        } else {
            titleEl.textContent = '';
        }
    }
}

async function startBfsExhaustivenessTest(latticeLevel) {
    if (_bfsTestActive || _demoActive) return;
    const lvl = latticeLevel || 1; // default L1

    _bfsTestActive = true;
    _bfsTestRunIdx = 0;
    _bfsTestResults = [null, null];
    _bfsTestComparison = null;
    _bfsTestReferenceFingerprints = null;
    _bfsTestEarlyAbort = false;
    _searchTraversalLog = [];
    _searchEventCounter = 0;
    _searchPathStack = [];
    _searchParentNodeId = null;
    _searchLastCandidates = null;
    _searchStartTime = performance.now();
    _setBfsTestPanelVisible(true);
    _bfsTestNovelCount = 0;
    _bfsTestNovelDetail = null;
    _bfsTestDecisionTrace = [];

    // Set lattice level
    const slider = document.getElementById('lattice-slider');
    if (slider && +slider.value !== lvl) {
        slider.value = lvl;
        slider.dispatchEvent(new Event('input'));
        await new Promise(r => setTimeout(r, 100));
    }

    // Both runs use the same seed — the difference is CHOREOGRAPHER STRATEGY
    const seed = (Math.random() * 0xFFFFFFFF) >>> 0;
    _bfsTestSeeds[0] = seed;
    _bfsTestSeeds[1] = seed;

    console.log(`%c[DFS TEST] Starting model exhaustiveness test (L${lvl})\n` +
        `  Seed: 0x${seed.toString(16).padStart(8,'0')}\n` +
        `  Test 1: CHOREOGRAPHER (normal heuristic scoring)\n` +
        `  Test 2: RANDOM (no heuristics) — early abort if novel solution found`,
        'color:cyan;font-weight:bold');

    // ── Test 1: Normal Choreographer ──
    _bfsTestRunIdx = 0;
    _bfsTestRandomChoreographer = false; // normal heuristic scoring
    simHalted = false;
    if (typeof stopDemo === 'function' && _demoActive) stopDemo();
    NucleusSimulator.simulateNucleus();
    await new Promise(r => setTimeout(r, 100));
    _setBfsTestTitle(0);
    _updateBfsTestPanel('Test 1: CHOREOGRAPHER (normal scoring)...');
    await new Promise(r => setTimeout(r, 200));
    await _executeBfsTestRun(0);
    _bfsTestResults[0] = _captureBfsRunResult();
    _bfsTestResults[0].mode = 'choreographer';
    console.log(`%c[DFS TEST] Test 1 (choreographer) done: highest ${_bfsTestResults[0].maxTick}, ` +
        `${_bfsTestResults[0].totalFingerprints} fps, ` +
        `retries ${_bfsTestResults[0].totalRetries}, ${(_bfsTestResults[0].elapsedMs / 1000).toFixed(1)}s`,
        'color:cyan');
    stopDemo();

    // Store Test 1's fingerprints as the reference set for live comparison
    _bfsTestReferenceFingerprints = new Map(
        [..._bfsTestResults[0].perTickFingerprints].map(([t, s]) => [t, new Set(s)])
    );

    // Pause between tests
    _updateBfsTestPanel(`Test 1 (choreographer) done — highest: ${_bfsTestResults[0].maxTick}, ` +
        `fps: ${_bfsTestResults[0].totalFingerprints}. Starting Test 2 (random)...`);
    await new Promise(r => setTimeout(r, 1000));
    simHalted = false;
    _btBadMoveLedger.clear();
    _btTriedFingerprints.clear();

    // ── Test 2: Random Choreographer (with live comparison) ──
    _bfsTestRunIdx = 1;
    _bfsTestRandomChoreographer = true; // all scoring → uniform random
    _bfsTestEarlyAbort = false;
    _bfsTestNovelCount = 0;
    NucleusSimulator.simulateNucleus();
    await new Promise(r => setTimeout(r, 100));
    _setBfsTestTitle(1, _bfsTestResults[0].totalFingerprints);
    _updateBfsTestPanel('Test 2: RANDOM — checking against choreographer solutions...');
    await new Promise(r => setTimeout(r, 200));
    await _executeBfsTestRun(1);
    _bfsTestResults[1] = _captureBfsRunResult();
    _bfsTestResults[1].mode = 'random';
    console.log(`%c[DFS TEST] Test 2 (random) done: highest ${_bfsTestResults[1].maxTick}, ` +
        `${_bfsTestResults[1].totalFingerprints} fps, ${(_bfsTestResults[1].elapsedMs / 1000).toFixed(1)}s`,
        'color:cyan');
    stopDemo();

    // ── Result ──
    _bfsTestActive = false;
    _bfsTestRandomChoreographer = false;
    _forceSeed = null;

    // Save reference count before clearing (needed for fail console output)
    const refFPs = _bfsTestReferenceFingerprints;
    _bfsTestReferenceFingerprints = null;

    if (_bfsTestEarlyAbort) {
        // ── FAIL: Random found novel solutions ──
        const nd = _bfsTestNovelDetail || { tick: '?', fingerprint: '?' };
        const refCount = refFPs ? (refFPs.get(nd.tick)?.size || 0) : '?';
        const r1 = _bfsTestResults[0], r2 = _bfsTestResults[1];
        const randomHalted = r2 && r2.maxTick < 200;
        const failMsg = `FAIL: Choreographer not exhaustive — ` +
            `random found ${_bfsTestNovelCount} novel solution${_bfsTestNovelCount !== 1 ? 's' : ''} (first at tick ${nd.tick}). ` +
            `Random reached tick ${r2 ? r2.maxTick : '?'}${randomHalted ? ' (halted — finite paths)' : ''}.`;
        _bfsTestComparison = {
            identical: false,
            earlyAbort: true,
            novelTick: nd.tick,
            novelFingerprint: nd.fingerprint,
            novelCount: _bfsTestNovelCount,
            summary: failMsg,
            sameLongestSolution: false,
            sameTotalFingerprints: false,
            differentPaths: true,
            randomMaxTick: r2 ? r2.maxTick : 0,
            randomHalted,
            fingerprintDiff: { ticksOnlyA: [], ticksOnlyB: [], ticksMismatch: [] },
        };
        _updateBfsTestPanel(failMsg);
        _setBfsTestTitle(-1);
        console.log(`%c[DFS TEST] ${failMsg}`, 'color:red;font-weight:bold');
        console.log(`  Total novel fingerprints: ${_bfsTestNovelCount}`);
        console.log(`  First novel at tick: ${nd.tick}`);
        console.log(`  Choreographer: highest=${r1 ? r1.maxTick : '?'}, fps=${r1 ? r1.totalFingerprints : '?'}`);
        console.log(`  Random: highest=${r2 ? r2.maxTick : '?'}, fps=${r2 ? r2.totalFingerprints : '?'}${randomHalted ? ' (HALTED — rules may be impossible)' : ''}`);
        console.log(`  Choreographer had ${refCount} solutions at first novel tick`);

        // ── Dump decision trace around divergent tick ──
        if (_bfsTestDecisionTrace && _bfsTestDecisionTrace.length > 0 && nd.tick) {
            const divergeTick = nd.tick;
            // Get choreographer decisions (runIdx=0) near the divergent tick
            const choreoTraces = _bfsTestDecisionTrace
                .filter(e => e.runIdx === 0 && Math.abs(e.tick - divergeTick) <= 3)
                .sort((a, b) => a.tick - b.tick);
            // Get random decisions (runIdx=1) near the divergent tick
            const randomTraces = _bfsTestDecisionTrace
                .filter(e => e.runIdx === 1 && Math.abs(e.tick - divergeTick) <= 3)
                .sort((a, b) => a.tick - b.tick);

            console.log(`%c[DFS TEST] ═══ DECISION TRACE (ticks ${divergeTick-3}..${divergeTick+3}) ═══`,
                'color:orange;font-weight:bold');

            console.log(`%c  CHOREOGRAPHER decisions:`, 'color:#66bbff;font-weight:bold');
            for (const t of choreoTraces) {
                const faces = t.faceAssignments.map(a => `X${a.xonIdx}→F${a.face}(${a.quarkType},s=${a.score})`).join(', ');
                const octs = t.octMatching.map(m => `X${m.xonIdx}:${m.from}→${m.to ?? 'null'}(${m.candidateCount}c)`).join(', ');
                console.log(`    t=${t.tick} [${t.matchingMethod || '?'}${t.btActive ? ' BT' : ''}]` +
                    (faces ? ` faces:{${faces}}` : '') +
                    (octs ? ` oct:{${octs}}` : '') +
                    (t.totalMatchings ? ` (${t.matchingIdx}/${t.totalMatchings} matchings)` : ''));
            }

            console.log(`%c  RANDOM decisions:`, 'color:#ff9944;font-weight:bold');
            for (const t of randomTraces) {
                const faces = t.faceAssignments.map(a => `X${a.xonIdx}→F${a.face}(${a.quarkType},s=${a.score})`).join(', ');
                const octs = t.octMatching.map(m => `X${m.xonIdx}:${m.from}→${m.to ?? 'null'}(${m.candidateCount}c)`).join(', ');
                console.log(`    t=${t.tick} [${t.matchingMethod || '?'}${t.btActive ? ' BT' : ''}]` +
                    (faces ? ` faces:{${faces}}` : '') +
                    (octs ? ` oct:{${octs}}` : '') +
                    (t.totalMatchings ? ` (${t.matchingIdx}/${t.totalMatchings} matchings)` : ''));
            }

            // Key insight: show face assignments at the divergent tick for both
            const choreoAtTick = choreoTraces.find(t => t.tick === divergeTick);
            const randomAtTick = randomTraces.find(t => t.tick === divergeTick);
            if (choreoAtTick || randomAtTick) {
                console.log(`%c  ═══ CRITICAL DIFF AT TICK ${divergeTick} ═══`, 'color:red;font-weight:bold');
                if (choreoAtTick?.faceAssignments?.length) {
                    console.log(`    Choreographer assigned faces: ${choreoAtTick.faceAssignments.map(a => `X${a.xonIdx}→F${a.face}(${a.quarkType})`).join(', ')}`);
                } else {
                    console.log(`    Choreographer: no face assignments at this tick`);
                }
                if (randomAtTick?.faceAssignments?.length) {
                    console.log(`    Random assigned faces: ${randomAtTick.faceAssignments.map(a => `X${a.xonIdx}→F${a.face}(${a.quarkType})`).join(', ')}`);
                } else {
                    console.log(`    Random: no face assignments at this tick`);
                }
            }

            console.log(`%c[DFS TEST] ═══════════════════════════════════════`, 'color:orange;font-weight:bold');
        }
    } else {
        // ── PASS: Random finished without finding anything new ──
        _bfsTestComparison = _compareBfsRuns(_bfsTestResults[0], _bfsTestResults[1]);
        const passMsg = `PASS: Model is exhaustive — random found no novel solutions. ` +
            `Both explored ${_bfsTestResults[0].totalFingerprints} fingerprints.`;
        _bfsTestComparison.summary = passMsg;
        _bfsTestComparison.identical = true;
        _updateBfsTestPanel(passMsg);
        _setBfsTestTitle(-1);

        console.log(`%c[DFS TEST] ${passMsg}`, 'color:lime;font-weight:bold');
        console.log(`  Choreographer: highest=${_bfsTestResults[0].maxTick}, fps=${_bfsTestResults[0].totalFingerprints}, ${(_bfsTestResults[0].elapsedMs / 1000).toFixed(1)}s`);
        console.log(`  Random: highest=${_bfsTestResults[1].maxTick}, fps=${_bfsTestResults[1].totalFingerprints}, ${(_bfsTestResults[1].elapsedMs / 1000).toFixed(1)}s`);
        console.log(`  Same longest solution: ${_bfsTestComparison.sameLongestSolution ? '✓ YES' : '✗ NO'}`);
        console.log(`  Different paths: ${_bfsTestComparison.differentPaths ? '✓ YES' : '✗ NO'} ` +
            `(retries: ${_bfsTestResults[0].totalRetries} vs ${_bfsTestResults[1].totalRetries})`);
    }
}

function _updateBfsTestPanel(message) {
    const el = document.getElementById('bfs-test-results');
    if (!el) return;

    let html = `<div style="color:#9abccc; font-size:11px; margin-bottom:8px;">${message}</div>`;

    // Show choreographer stats from Test 1 if available (during Test 2 or after)
    if (_bfsTestResults[0] && !_bfsTestComparison) {
        const r1 = _bfsTestResults[0];
        html += `<div style="margin-bottom:6px; padding:4px; background:rgba(100,180,255,0.06); border:1px solid rgba(100,180,255,0.15); border-radius:3px;">`;
        html += `<div style="color:#6a8aaa; font-size:9px;">CHOREOGRAPHER (completed)</div>`;
        html += `<div style="font-size:10px; color:#9abccc;">` +
            `highest: <b>${r1.maxTick}</b> &middot; fps: ${r1.totalFingerprints} &middot; ` +
            `retries: ${r1.totalRetries} &middot; ${(r1.elapsedMs / 1000).toFixed(1)}s</div>`;
        html += `</div>`;
    }

    if (_bfsTestComparison) {
        const c = _bfsTestComparison;
        const _bigRedBox = (label) =>
            `<div style="color:#ff4444; font-weight:bold; font-size:13px; margin-bottom:6px; ` +
            `padding:8px; background:rgba(255,50,50,0.12); border:2px solid #ff4444; border-radius:4px; ` +
            `text-align:center; animation:dp-alarm 0.8s ease-in-out infinite alternate;">` +
            `✗ ${label}</div>`;

        if (c.earlyAbort) {
            // ── Box 1: CHOREOGRAPHER NOT EXHAUSTIVE ──
            html += _bigRedBox('CHOREOGRAPHER NOT EXHAUSTIVE');
            html += `<div style="font-size:10px; color:#ff8866; margin-bottom:6px; line-height:1.5;">` +
                `Random found <b>${c.novelCount || 1}</b> novel solution${(c.novelCount || 1) !== 1 ? 's' : ''} (first at tick <b>${c.novelTick}</b>)<br>` +
                `<span style="font-size:9px; color:#cc7755; word-break:break-all;">${c.novelFingerprint}</span>` +
                `</div>`;

            // ── Box 2: RANDOM HALTED (if random halted before reaching choreographer's peak) ──
            if (c.randomHalted) {
                html += _bigRedBox('RANDOM HALTED');
                html += `<div style="font-size:10px; color:#ffaa44; margin-bottom:6px;">` +
                    `Random halted at tick ${c.randomMaxTick} — finite valid paths (rules may be impossible)</div>`;
            }

            // ── Box 3: RANDOM INCOMPLETE (if random found fewer total fps than choreographer) ──
            const r1 = _bfsTestResults[0], r2 = _bfsTestResults[1];
            if (r1 && r2 && r2.totalFingerprints < r1.totalFingerprints) {
                html += `<div style="color:#ffaa00; font-weight:bold; font-size:13px; margin-bottom:6px; ` +
                    `padding:8px; background:rgba(255,170,0,0.12); border:2px solid #ffaa00; border-radius:4px; ` +
                    `text-align:center;">` +
                    `⚠ RANDOM INCOMPLETE</div>`;
                html += `<div style="font-size:10px; color:#cc9944; margin-bottom:6px;">` +
                    `Random explored only ${r2.totalFingerprints} fps vs choreographer's ${r1.totalFingerprints}</div>`;
            }

            // Sanity check: if choreographer found MORE outcomes than random, random is broken
            if (r1 && r2 && r1.totalFingerprints > r2.totalFingerprints && !c.randomHalted) {
                html += _bigRedBox('RANDOM BROKEN — FEWER OUTCOMES THAN CHOREOGRAPHER');
                html += `<div style="font-size:10px; color:#ff8866; margin-bottom:6px;">` +
                    `Choreographer: ${r1.totalFingerprints} fps, Random: ${r2.totalFingerprints} fps. ` +
                    `Random should find at least as many — something is wrong.</div>`;
            }
        } else {
            // ── PASS: exhaustive ──
            html += `<div style="color:#44ff44; font-weight:bold; font-size:13px; margin-bottom:6px; ` +
                    `padding:8px; background:rgba(50,255,50,0.1); border:2px solid #44ff44; border-radius:4px; ` +
                    `text-align:center;">` +
                    `✓ MODEL IS EXHAUSTIVE</div>`;
        }

        for (let i = 0; i < 2; i++) {
            const r = _bfsTestResults[i];
            if (!r) continue;
            const modeLabel = r.mode === 'random' ? 'RANDOM' : 'CHOREOGRAPHER';
            html += `<div style="margin-bottom:4px; padding:4px; background:rgba(255,255,255,0.03); border-radius:3px;">`;
            html += `<div style="color:#6a8aaa; font-size:9px;">${modeLabel} &mdash; seed 0x${r.seed.toString(16).padStart(8,'0')}</div>`;
            html += `<div style="font-size:10px; color:#9abccc;">` +
                `highest tick: <b>${r.maxTick}</b> &middot; halt: ${r.haltReason} &middot; ` +
                `fps: ${r.totalFingerprints} &middot; retries: ${r.totalRetries} &middot; ` +
                `${(r.elapsedMs / 1000).toFixed(1)}s</div>`;
            if (r.haltViolation) {
                html += `<div style="font-size:9px; color:#cc8866; margin-top:2px;">${r.haltViolation}</div>`;
            }
            html += `</div>`;
        }

        if (!c.earlyAbort && c.fingerprintDiff && c.fingerprintDiff.ticksMismatch.length > 0) {
            html += `<div style="color:#ff8844; font-size:9px; margin-top:4px;">Divergent ticks: `;
            html += c.fingerprintDiff.ticksMismatch.slice(0, 10)
                .map(m => `t${m.tick}(+${m.onlyA}/-${m.onlyB})`)
                .join(', ');
            html += `</div>`;
        }
    }

    el.innerHTML = html;

    // Show export button only on success (not on early abort fail)
    const exportBtn = document.getElementById('btn-bfs-export');
    if (exportBtn) {
        const showExport = _bfsTestComparison && !_bfsTestComparison.earlyAbort;
        exportBtn.style.display = showExport ? 'inline-block' : 'none';
    }

    // Show traversal log download button whenever comparison exists
    if (_bfsTestComparison) {
        let dlBtn = document.getElementById('btn-traversal-log');
        if (!dlBtn) {
            dlBtn = document.createElement('button');
            dlBtn.id = 'btn-traversal-log';
            dlBtn.textContent = 'Download Traversal Log';
            dlBtn.style.cssText = 'margin-top:6px;padding:4px 10px;font-size:10px;cursor:pointer;' +
                'background:#1a3a4a;color:#9abccc;border:1px solid #3a6a7a;border-radius:3px;display:block;width:100%;';
            dlBtn.addEventListener('click', _downloadTraversalLog);
            el.parentElement.appendChild(dlBtn);
        }
        dlBtn.style.display = 'block';
    }
}

// ═══ Sweep Mode: Sequential Seeds with Cross-Seed Fingerprint Blacklist ════════

// ── IndexedDB persistence for cross-session blacklist ──
const _BL_IDB_NAME = 'FluxBlacklist';
const _BL_IDB_VERSION = 3;
const _BL_IDB_STORE = 'blacklists';
const _AS_IDB_STORE = 'autosave';
const _CS_IDB_STORE = 'council';
let _blIDB = null;
let _blIDBReady = false;

function _blIDBOpen() {
    return new Promise((resolve) => {
        try {
            const req = indexedDB.open(_BL_IDB_NAME, _BL_IDB_VERSION);
            req.onupgradeneeded = (e) => {
                const db = e.target.result;
                if (!db.objectStoreNames.contains(_BL_IDB_STORE)) db.createObjectStore(_BL_IDB_STORE);
                if (!db.objectStoreNames.contains(_AS_IDB_STORE)) db.createObjectStore(_AS_IDB_STORE);
                if (!db.objectStoreNames.contains(_CS_IDB_STORE)) db.createObjectStore(_CS_IDB_STORE);
            };
            req.onsuccess = (e) => { _blIDB = e.target.result; _blIDBReady = true; _migrateOldKeys().then(resolve); };
            req.onerror = () => { console.warn('[Blacklist] IndexedDB unavailable'); resolve(); };
        } catch (e) { resolve(); }
    });
}

// One-time migration: old key format was `|name=value` for every rule.
// New format: concat tag only when rule is active. Absent = OFF.
// Example old: v2|L2|t20=1|oct=1|cap=6|glu=1|bare=1|proj=1|idleOct=0
// Example new: v2|L2|t20|oct1|glu|bare|proj
function _migrateOldKeyFormat(oldKey) {
    // Only migrate keys matching old format (contain '=')
    if (!oldKey.includes('=')) return null;
    // Split into base (v..|L..) and rule segments
    const parts = oldKey.split('|');
    // Find the base prefix (v...|L...) — everything before first '='
    let base = '';
    let ruleParts = [];
    let seedSuffix = '';
    for (const p of parts) {
        if (p.includes('=')) {
            ruleParts.push(p);
        } else if (!base) {
            base = p;
        } else if (base && !base.includes('|L')) {
            base += '|' + p;
        } else {
            // Seed suffix (numeric after rule parts)
            seedSuffix = '|' + p;
        }
    }
    // Rebuild: base + only-when-active tags
    let k = base;
    for (const rp of ruleParts) {
        const [name, val] = rp.split('=');
        if (name === 't20' && val === '1')       k += '|t20';
        else if (name === 'oct' && +val > 0)     k += `|oct${val}`;
        else if (name === 'cap' && +val < 6)     k += `|cap${val}`;
        else if (name === 'glu' && val === '1')   k += '|glu';
        else if (name === 'bare' && val === '1')  k += '|bare';
        else if (name === 'proj' && val === '1')  k += '|proj';
        else if (name === 'idleOct' && val === '1') k += '|idle';
        // OFF rules: omitted entirely
    }
    return k + seedSuffix;
}

async function _migrateOldKeys() {
    if (!_blIDB) return;
    if (localStorage.getItem('_idbKeyMigrated_v1')) return; // already done
    let migrated = 0;
    for (const storeName of [_BL_IDB_STORE, _AS_IDB_STORE, _CS_IDB_STORE]) {
        try {
            const tx = _blIDB.transaction(storeName, 'readwrite');
            const store = tx.objectStore(storeName);
            const allKeys = await new Promise((res, rej) => {
                const r = store.getAllKeys(); r.onsuccess = () => res(r.result); r.onerror = rej;
            });
            for (const key of allKeys) {
                if (typeof key !== 'string') continue;
                const newKey = _migrateOldKeyFormat(key);
                if (!newKey || newKey === key) continue;
                const val = await new Promise((res, rej) => {
                    const r = store.get(key); r.onsuccess = () => res(r.result); r.onerror = rej;
                });
                // If new key already has data, keep the newer one
                const existing = await new Promise((res, rej) => {
                    const r = store.get(newKey); r.onsuccess = () => res(r.result); r.onerror = rej;
                });
                if (!existing) store.put(val, newKey);
                store.delete(key);
                migrated++;
            }
            await new Promise((res) => { tx.oncomplete = res; });
        } catch (e) { console.warn('[IDB migrate]', storeName, e); }
    }
    localStorage.setItem('_idbKeyMigrated_v1', '1');
    if (migrated > 0) console.log(`[IDB] Migrated ${migrated} keys to new format`);
}

// ── Council trim — enforce maxSize, delete evicted from IDB ──
function _trimCouncil(lvl) {
    const maxSize = _goldenCouncilSize();
    if (_sweepGoldenCouncil.length <= maxSize) return;
    _sweepGoldenCouncil.sort((a, b) => b.peak - a.peak);
    const evicted = _sweepGoldenCouncil.splice(maxSize);
    for (const m of evicted) {
        _blIDBDeleteCouncilMember(lvl, m.seed);
    }
    if (evicted.length > 0) {
        console.log(`%c[COUNCIL] Trimmed ${evicted.length} members (max ${maxSize}): evicted [${evicted.map(m => 't' + m.peak).join(', ')}]`, 'color:#ff8866');
        _blIDBSave(lvl);
        _populateCouncilDropdown();
    }
}

// ── Autosave helpers (council-eligible crash recovery) ──

function _isCouncilEligible() {
    if (!_sweepActive) return true;  // manual demo = always save
    const maxSize = _goldenCouncilSize();
    const currentSeed = _forceSeed || _runSeed || 0;
    // Current seed already in council → still eligible (keep autosaving updates)
    if (_sweepGoldenCouncil.some(m => m.seed === currentSeed)) return true;
    // Council not full → always eligible
    if (_sweepGoldenCouncil.length < maxSize) return true;
    // Must beat the lowest council member to be admitted
    const lowestPeak = _sweepGoldenCouncil[_sweepGoldenCouncil.length - 1].peak;
    return _maxTickReached > lowestPeak;
}

async function _autosaveToIDB() {
    if (_btSnapshots.length === 0) return;
    if (!_blIDBReady) await _blIDBOpen();
    if (!_blIDB) return;
    // Incremental serialization: cold snapshots are already serialized (from IDB),
    // only serialize the hot (live) portion beyond _btColdBoundary.
    const coldPart = _btColdSnapshots.slice(0, Math.min(_btColdBoundary, _btColdSnapshots.length));
    const hotPart = _btSnapshots.slice(_btColdBoundary).map(_serializeSnapshot);
    const allSnaps = coldPart.concat(hotPart);
    const lvl = typeof latticeLevel !== 'undefined' ? latticeLevel : 2;
    const key = _blacklistRuleKey(lvl);
    try {
        const tx = _blIDB.transaction(_AS_IDB_STORE, 'readwrite');
        tx.objectStore(_AS_IDB_STORE).put({
            snapshots: allSnaps,
            tick: _demoTick,
            seed: _runSeed,
            maxTickReached: _maxTickReached,
            sweepSeedIdx: _sweepSeedIdx,
            timestamp: new Date().toISOString(),
        }, key);
        console.log(`%c[Autosave] tick ${_demoTick} saved (peak ${_maxTickReached}, ${allSnaps.length} snapshots)`, 'color:#80ff80');
    } catch (e) { console.warn('[Autosave] Save failed:', e); }
}

async function _autosaveIDBLoad(lvl) {
    if (!_blIDBReady) await _blIDBOpen();
    if (!_blIDB) return null;
    const key = _blacklistRuleKey(lvl || 2);
    return new Promise((resolve) => {
        try {
            const tx = _blIDB.transaction(_AS_IDB_STORE, 'readonly');
            const req = tx.objectStore(_AS_IDB_STORE).get(key);
            req.onsuccess = () => resolve(req.result || null);
            req.onerror = () => resolve(null);
        } catch (e) { resolve(null); }
    });
}

async function _autosaveIDBClear(lvl) {
    if (!_blIDBReady) await _blIDBOpen();
    if (!_blIDB) return;
    const key = _blacklistRuleKey(lvl || 2);
    try {
        const tx = _blIDB.transaction(_AS_IDB_STORE, 'readwrite');
        tx.objectStore(_AS_IDB_STORE).delete(key);
    } catch (e) { /* ignore */ }
}

// Canonical rule key: deterministic fingerprint of the ENTIRE rule config + snapshot version.
// If ANY config value differs, data goes into a separate IDB bucket.
function _blacklistRuleKey(lvl) {
    // Concat-only-when-active: each rule appends a tag only when ON.
    // OFF rules are absent → key is identical to before that rule existed.
    let k = `v${_SNAPSHOT_VERSION}|L${lvl}`;
    if (_ruleT20StrictMode)     k += '|t20';
    if (T79_MAX_FULL_TICKS > 0) k += `|oct${T79_MAX_FULL_TICKS}`;
    if (OCT_CAPACITY_MAX < 6)   k += `|cap${OCT_CAPACITY_MAX}`;
    if (_ruleGluonMediatedSC)   k += '|glu';
    if (_ruleBareTetrahedra)    k += '|bare';
    if (_ruleProjectedGuards)   k += '|proj';
    if (_ruleAdaptiveEjection) {
        k += '|adpt';
    } else if (_ruleCubeRootEjection) {
        k += '|cbrt';
    } else {
        if (T90_TOLERANCE > 1) k += `|eq${T90_TOLERANCE}`;
        if (T91_TOLERANCE > 1) k += `|ef${T91_TOLERANCE}`;
        if (T92_TOLERANCE > 1) k += `|eh${T92_TOLERANCE}`;
    }
    return k;
}

// Load blacklist from IndexedDB for current rules
async function _blIDBLoad(lvl) {
    if (!_blIDBReady) await _blIDBOpen();
    if (!_blIDB) return null;
    const key = _blacklistRuleKey(lvl);
    const t0 = performance.now();
    return new Promise((resolve) => {
        try {
            const tx = _blIDB.transaction(_BL_IDB_STORE, 'readonly');
            const req = tx.objectStore(_BL_IDB_STORE).get(key);
            req.onsuccess = async () => {
                const data = req.result;
                if (!data) { resolve(null); return; }

                // ── Parse council index (shared by both formats) ──
                let goldenCouncil = [];
                if (data.councilIndex && Array.isArray(data.councilIndex)) {
                    for (const stub of data.councilIndex) {
                        goldenCouncil.push({ peak: stub.peak, seed: stub.seed, _cold: true });
                    }
                } else if (data.goldenCouncil && Array.isArray(data.goldenCouncil)) {
                    // Old format (v2): migrate to cold storage
                    console.log(`[BL] Migrating ${data.goldenCouncil.length} council members to cold storage...`);
                    for (const member of data.goldenCouncil) {
                        try {
                            const cKey = key + '|' + member.seed;
                            const migTx = _blIDB.transaction(_CS_IDB_STORE, 'readwrite');
                            const migStore = migTx.objectStore(_CS_IDB_STORE);
                            migStore.put({
                                seed: member.seed, snapshots: member.snapshots, moves: member.moves,
                                snapshotVersion: (member.snapshots && member.snapshots[0] && member.snapshots[0]._v) || 0,
                                timestamp: new Date().toISOString(),
                            }, cKey);
                            // Also store moves separately for fast hydration
                            if (member.moves) {
                                const movesArr = [];
                                for (const [tick, moveMap] of (member.moves instanceof Map ? member.moves : new Map(member.moves))) {
                                    movesArr.push([tick, moveMap instanceof Map ? [...moveMap.entries()] : moveMap]);
                                }
                                migStore.put({ seed: member.seed, moves: movesArr }, cKey + '|mv');
                            }
                        } catch (e) { console.warn('[BL] Migration: failed to write member:', e); }
                        goldenCouncil.push({ peak: member.peak, seed: member.seed, _cold: true });
                    }
                }
                // Dedup: keep only the highest-peak entry per seed
                const seedBest = new Map();
                for (const m of goldenCouncil) {
                    const existing = seedBest.get(m.seed);
                    if (!existing || m.peak > existing.peak) seedBest.set(m.seed, m);
                }
                goldenCouncil = [...seedBest.values()].sort((a, b) => b.peak - a.peak);
                const peaks = goldenCouncil.map(m => 't' + m.peak).join(', ');

                // ── Bucketed format: metadata only, fingerprints loaded on demand ──
                if (data.bucketVersion >= 1) {
                    _blBucketVersion = data.bucketVersion;
                    _blBucketCount = data.bucketCount || 0;
                    _blBucketSize = data.bucketSize || 64;
                    _blLoadedBuckets = new Set();
                    const ms = (performance.now() - t0).toFixed(1);
                    console.log(`[BL] Loaded metadata: ${data.total} fps across ${_blBucketCount} buckets + council [${peaks}] in ${ms}ms`);
                    resolve({ map: new Map(), total: data.total || 0, seedIdx: data.seedIdx || 0, usedSeeds: data.usedSeeds || [], goldenCouncil });
                    return;
                }

                // ── Legacy single-blob format: deserialize all inline ──
                if (data.entries) {
                    _blBucketVersion = 0;
                    const map = new Map();
                    let total = 0;
                    for (const [tick, fps] of data.entries) {
                        map.set(tick, new Set(fps));
                        total += fps.length;
                    }
                    // Mark all covered buckets as loaded
                    _blLoadedBuckets = new Set();
                    for (const tick of map.keys()) {
                        _blLoadedBuckets.add(Math.floor(tick / _blBucketSize));
                    }
                    const ms = (performance.now() - t0).toFixed(1);
                    console.log(`[BL] Loaded legacy blob: ${total} fps + council [${peaks}] in ${ms}ms (will migrate on next save)`);
                    resolve({ map, total, seedIdx: data.seedIdx || 0, usedSeeds: data.usedSeeds || [], goldenCouncil });
                } else {
                    resolve(null);
                }
            };
            req.onerror = () => resolve(null);
        } catch (e) { resolve(null); }
    });
}

// Save blacklist + council index to IndexedDB (debounced). Council snapshots saved separately.
let _blIDBSaveTimer = null;
function _blIDBSave(lvl) {
    if (_blIDBSaveTimer) clearTimeout(_blIDBSaveTimer);
    _blIDBSaveTimer = setTimeout(() => {
        _blIDBSaveTimer = null;
        _blIDBSaveBlacklist(lvl);
    }, 2000);
}

// ── Bucketed blacklist: on-demand loading ──

// Load a single bucket from IDB into _sweepBlacklist. Returns a Promise.
const _BL_MAX_LOADED_BUCKETS = 3; // keep at most current ± 1 in RAM
async function _blPrefetchBucket(lvl, bucketIdx) {
    if (_blBucketVersion < 1) return;            // legacy format, already fully loaded
    if (_blLoadedBuckets.has(bucketIdx)) return;  // already in memory
    if (bucketIdx >= _blBucketCount) return;      // beyond stored range

    // Evict distant buckets to cap RAM usage
    if (_blLoadedBuckets.size >= _BL_MAX_LOADED_BUCKETS) {
        for (const bi of _blLoadedBuckets) {
            if (Math.abs(bi - bucketIdx) > 1) {
                // Remove all fingerprints in this bucket's tick range
                const lo = bi * _blBucketSize;
                const hi = lo + _blBucketSize;
                for (let t = lo; t < hi; t++) _sweepBlacklist.delete(t);
                _blLoadedBuckets.delete(bi);
                console.log(`[BL] Evicted bucket ${bi} (ticks ${lo}-${hi - 1}) — too far from bucket ${bucketIdx}`);
            }
        }
    }

    const t0 = performance.now();
    const baseKey = _blacklistRuleKey(lvl);
    const bucketKey = baseKey + '|bl|' + bucketIdx;
    const tickLo = bucketIdx * _blBucketSize;
    const tickHi = tickLo + _blBucketSize - 1;
    console.log(`[BL] Prefetching bucket ${bucketIdx} (ticks ${tickLo}-${tickHi})...`);

    if (!_blIDBReady) await _blIDBOpen();
    if (!_blIDB) { _blLoadedBuckets.add(bucketIdx); return; }

    return new Promise((resolve) => {
        try {
            const tx = _blIDB.transaction(_BL_IDB_STORE, 'readonly');
            const req = tx.objectStore(_BL_IDB_STORE).get(bucketKey);
            req.onsuccess = () => {
                const data = req.result;
                let count = 0;
                if (data && data.entries) {
                    for (const [tick, fps] of data.entries) {
                        if (!_sweepBlacklist.has(tick)) _sweepBlacklist.set(tick, new Set());
                        const set = _sweepBlacklist.get(tick);
                        for (const fp of fps) { set.add(fp); count++; }
                    }
                }
                _blLoadedBuckets.add(bucketIdx);
                const ms = (performance.now() - t0).toFixed(1);
                console.log(`[BL] Bucket ${bucketIdx} loaded: ${count} fps in ${ms}ms`);
                resolve();
            };
            req.onerror = () => { _blLoadedBuckets.add(bucketIdx); resolve(); };
        } catch (e) { _blLoadedBuckets.add(bucketIdx); resolve(); }
    });
}

// Prefetch all buckets covering a tick range. Parallel IDB reads.
async function _blPrefetchRange(lvl, tickLow, tickHigh) {
    if (_blBucketVersion < 1) return;
    const lo = Math.floor(Math.max(0, tickLow) / _blBucketSize);
    const hi = Math.floor(tickHigh / _blBucketSize);
    const needed = [];
    for (let bi = lo; bi <= hi; bi++) {
        if (!_blLoadedBuckets.has(bi) && bi < _blBucketCount) needed.push(bi);
    }
    if (needed.length === 0) return;
    console.log(`[BL] Range prefetch: buckets ${needed.join(',')} (ticks ${tickLow}-${tickHigh})`);
    await Promise.all(needed.map(bi => _blPrefetchBucket(lvl, bi)));
}

// Ensure the bucket for a given tick is loaded. Blocks if needed.
async function _blEnsureTick(lvl, tick) {
    if (_blBucketVersion < 1) return;
    const bi = Math.floor(tick / _blBucketSize);
    if (_blLoadedBuckets.has(bi)) return;
    const t0 = performance.now();
    console.warn(`[BL] ⚠️ Tick ${tick} blocked — awaiting bucket ${bi}`);
    await _blPrefetchBucket(lvl, bi);
    const ms = (performance.now() - t0).toFixed(1);
    console.warn(`[BL] ⚠️ Bucket ${bi} loaded after ${ms}ms block`);
}

// Serialize a single backtracker snapshot for IndexedDB storage.
// Converts Sets → arrays and Maps → [key, value] pairs.
function _serializeSnapshot(snap) {
    return {
        _v: snap._v || 0,
        tick: snap.tick,
        openingPhase: snap.openingPhase,
        xons: snap.xons.map(x => {
            const role = x._role || (x._mode === 'gluon' ? 'gluon' : x._mode === 'weak' ? 'weak' : x._quarkType || 'oct');
            // Strip live trail array — only store trailLen + per-tick delta (_tDelta/_tRecolor).
            const { trail, _trailLenAtTickStart, ...rest } = x;
            return { ...rest, _role: role, trailLen: x.trailLen != null ? x.trailLen : (trail ? trail.length : 0) };
        }),
        activeSet: [...snap.activeSet],
        xonImpliedSet: [...snap.xonImpliedSet],
        impliedSet: [...snap.impliedSet],
        scAttribution: [...snap.scAttribution.entries()],
        pos: snap.pos, // array of [x,y,z] already
        octFullConsecutive: snap.octFullConsecutive,
        demoVisits: snap.demoVisits,
        actualizationVisits: snap.actualizationVisits,
        faceEdgeEpoch: snap.faceEdgeEpoch,
        faceWasActualized: snap.faceWasActualized,
        edgeBalance: snap.edgeBalance ? [...snap.edgeBalance].map(([k, v]) => [k, { ...v }]) : null,
        ejectionBalance: snap.ejectionBalance ? [...snap.ejectionBalance] : null,
        octWindingDirection: snap.octWindingDirection,
        planckSeconds: snap.planckSeconds,
        globalModeStats: snap.globalModeStats,
        globalRoleStats: snap.globalRoleStats || null,
        // Nucleus topology (v2+)
        octNodeSet: snap.octNodeSet ? [...snap.octNodeSet] : null,
        octSCIds: snap.octSCIds ? snap.octSCIds : null,
        octEdgeSet: snap.octEdgeSet ? [...snap.octEdgeSet] : null,
        nucleusTetFaceData: snap.nucleusTetFaceData || null,
        octEquatorCycle: snap.octEquatorCycle || null,
        octCageSCCycle: snap.octCageSCCycle || null,
        octSeedCenter: snap.octSeedCenter != null ? snap.octSeedCenter : null,
        octVoidIdx: snap.octVoidIdx != null ? snap.octVoidIdx : -1,
        octAntipodal: snap.octAntipodal ? [...snap.octAntipodal] : null,
    };
}

// Deserialize a snapshot from IndexedDB back into live format (with Sets/Maps).
function _deserializeSnapshot(s) {
    return {
        _v: s._v || 0,
        tick: s.tick,
        openingPhase: s.openingPhase,
        xons: s.xons.map(x => {
            const role = x._role || (x._mode === 'gluon' ? 'gluon' : x._mode === 'weak' ? 'weak' : x._quarkType || 'oct');
            // Trail: unified entry format or null (modern snapshots use trailLen + _tDelta)
            let trail = null;
            if (x.trail && Array.isArray(x.trail) && x.trail.length > 0 && typeof x.trail[0] === 'object') {
                trail = x.trail.map(e => ({ node: e.node, role: e.role, pos: e.pos ? [e.pos[0], e.pos[1], e.pos[2]] : [0,0,0] }));
            }
            return {
                ...x,
                _role: role,
                _loopSeq: x._loopSeq ? x._loopSeq.slice() : null,
                trail: trail,
                trailLen: x.trailLen != null ? x.trailLen : (trail ? trail.length : 0),
                _dirBalance: x._dirBalance ? x._dirBalance.slice() : new Array(10).fill(0),
                _modeStats: x._modeStats ? { ...x._modeStats } : { oct: 0, tet: 0, idle_tet: 0, weak: 0, gluon: 0 },
                _gluonBoundSCs: x._gluonBoundSCs ? x._gluonBoundSCs.slice() : null,
            };
        }),
        activeSet: new Set(s.activeSet),
        xonImpliedSet: new Set(s.xonImpliedSet),
        impliedSet: new Set(s.impliedSet),
        scAttribution: new Map(s.scAttribution),
        pos: s.pos.map(p => [p[0], p[1], p[2]]),
        octFullConsecutive: s.octFullConsecutive,
        demoVisits: s.demoVisits ? JSON.parse(JSON.stringify(s.demoVisits)) : null,
        actualizationVisits: s.actualizationVisits ? JSON.parse(JSON.stringify(s.actualizationVisits)) : null,
        faceEdgeEpoch: s.faceEdgeEpoch ? JSON.parse(JSON.stringify(s.faceEdgeEpoch)) : null,
        faceWasActualized: s.faceWasActualized ? { ...s.faceWasActualized } : null,
        edgeBalance: s.edgeBalance ? new Map(s.edgeBalance.map(([k, v]) => [k, { ...v }])) : null,
        ejectionBalance: s.ejectionBalance ? new Map(s.ejectionBalance) : null,
        octWindingDirection: s.octWindingDirection,
        planckSeconds: s.planckSeconds,
        globalModeStats: s.globalModeStats ? { ...s.globalModeStats } : null,
        globalRoleStats: s.globalRoleStats ? { ...s.globalRoleStats } : null,
        // Nucleus topology (v2+)
        octNodeSet: s.octNodeSet ? new Set(s.octNodeSet) : null,
        octSCIds: s.octSCIds ? s.octSCIds.slice() : null,
        octEdgeSet: s.octEdgeSet ? new Set(s.octEdgeSet) : null,
        nucleusTetFaceData: s.nucleusTetFaceData ? JSON.parse(JSON.stringify(s.nucleusTetFaceData)) : null,
        octEquatorCycle: s.octEquatorCycle ? s.octEquatorCycle.slice() : null,
        octCageSCCycle: s.octCageSCCycle ? s.octCageSCCycle.slice() : null,
        octSeedCenter: s.octSeedCenter != null ? s.octSeedCenter : null,
        octVoidIdx: s.octVoidIdx != null ? s.octVoidIdx : -1,
        octAntipodal: s.octAntipodal ? new Map(s.octAntipodal) : null,
    };
}

// ── Hot/Cold storage: blacklist + council index saved together, council snapshots saved separately ──

function _blIDBSaveBlacklist(lvl) {
    if (!_blIDB) return;
    const t0 = performance.now();
    const baseKey = _blacklistRuleKey(lvl);

    // Group fingerprints by bucket index
    const buckets = new Map(); // bucketIdx → [[tick, [fps...]], ...]
    let maxBucket = -1;
    for (const [tick, fpSet] of _sweepBlacklist) {
        const bi = Math.floor(tick / _blBucketSize);
        if (bi > maxBucket) maxBucket = bi;
        if (!buckets.has(bi)) buckets.set(bi, []);
        buckets.get(bi).push([tick, [...fpSet]]);
    }
    const bucketCount = maxBucket + 1;

    // Council index: lightweight stubs only (no snapshots, no moves)
    const councilIndex = _sweepGoldenCouncil.map(m => ({
        peak: m.peak, seed: m.seed, snapshotVersion: _SNAPSHOT_VERSION,
    }));

    try {
        const tx = _blIDB.transaction(_BL_IDB_STORE, 'readwrite');
        const store = tx.objectStore(_BL_IDB_STORE);

        // Write each bucket as a separate key
        for (const [bi, entries] of buckets) {
            const count = entries.reduce((s, [, fps]) => s + fps.length, 0);
            store.put({ entries, count }, baseKey + '|bl|' + bi);
        }

        // Write metadata (no fingerprints — they're in buckets now)
        store.put({
            key: baseKey,
            bucketVersion: 1,
            bucketSize: _blBucketSize,
            bucketCount,
            total: _sweepTotalBlacklisted,
            seedIdx: _sweepSeedIdx,
            usedSeeds: [..._sweepUsedSeeds],
            councilIndex,
            timestamp: new Date().toISOString(),
        }, baseKey);

        _blBucketVersion = 1;
        _blBucketCount = bucketCount;
        const peaks = _sweepGoldenCouncil.map(m => 't' + m.peak).join(', ');
        const ms = (performance.now() - t0).toFixed(1);
        console.log(`[BL] Saved ${buckets.size} buckets (${_sweepTotalBlacklisted} total fps) + council [${peaks}] in ${ms}ms`);
    } catch (e) { console.warn('[BL] Save failed:', e); }
}

async function _blIDBSaveCouncilMember(lvl, seed, snapshots, moves) {
    if (!_blIDBReady) await _blIDBOpen();
    if (!_blIDB) return;
    const key = _blacklistRuleKey(lvl) + '|' + seed;

    // ── Take only the last contiguous run (auto-retry may concatenate multiple t=0→t=x runs) ──
    {
        let lastStart = 0;
        for (let i = 1; i < snapshots.length; i++) {
            if (snapshots[i].tick <= snapshots[i - 1].tick) lastStart = i;
        }
        if (lastStart > 0) {
            console.log(`%c[Council IDB] Trimmed ${lastStart} stale snapshots (kept last run: t${snapshots[lastStart].tick}→t${snapshots[snapshots.length-1].tick})`, 'color:#ffaa44');
            snapshots = snapshots.slice(lastStart);
        }
    }

    // Full overwrite — _btSnapshots IS the complete traversal.
    // Incremental serialization: cold snapshots are already serialized,
    // only serialize the hot (live) portion beyond _btColdBoundary.
    const coldCount = Math.min(_btColdBoundary, snapshots.length, _btColdSnapshots.length);
    const coldPart = _btColdSnapshots.slice(0, coldCount);
    const hotPart = snapshots.slice(coldCount).map(_serializeSnapshot);
    const snapsArr = coldPart.concat(hotPart);
    const movesArr = [];
    for (const [tick, moveMap] of moves) {
        movesArr.push([tick, [...moveMap.entries()]]);
    }
    try {
        const tx = _blIDB.transaction(_CS_IDB_STORE, 'readwrite');
        const store = tx.objectStore(_CS_IDB_STORE);
        // Store snapshots+moves together (full blob for replay hydration)
        store.put({
            seed, snapshots: snapsArr, moves: movesArr,
            snapshotVersion: _SNAPSHOT_VERSION,
            timestamp: new Date().toISOString(),
        }, key);
        // Store moves separately under key|mv (lightweight read for golden boost)
        store.put({ seed, moves: movesArr }, key + '|mv');
        // Wait for transaction to complete before returning — auto-retry-best
        // needs the data committed before it can hydrate the new best.
        await new Promise((resolve, reject) => {
            tx.oncomplete = resolve;
            tx.onerror = () => reject(tx.error);
        });
        console.log(`%c[Council IDB] Saved member seed 0x${seed.toString(16).padStart(8,'0')} (${snapsArr.length} snapshots + ${movesArr.length} moves) to cold storage`, 'color:#66ccff');
    } catch (e) { console.warn('[Council IDB] Save failed:', e); }
}

async function _blIDBDeleteCouncilMember(lvl, seed) {
    if (!_blIDBReady) await _blIDBOpen();
    if (!_blIDB) return;
    const key = _blacklistRuleKey(lvl) + '|' + seed;
    try {
        const tx = _blIDB.transaction(_CS_IDB_STORE, 'readwrite');
        const store = tx.objectStore(_CS_IDB_STORE);
        store.delete(key);
        store.delete(key + '|mv');
        console.log(`[Council IDB] Deleted evicted member seed 0x${seed.toString(16).padStart(8,'0')} from cold storage`);
    } catch (e) { /* ignore */ }
}

async function _hydrateCouncilMember(lvl, member) {
    if (!member._cold) return; // already hydrated
    if (!_blIDBReady) await _blIDBOpen();
    if (!_blIDB) return;
    const t0 = performance.now();
    const key = _blacklistRuleKey(lvl) + '|' + member.seed;
    return new Promise((resolve) => {
        try {
            const tx = _blIDB.transaction(_CS_IDB_STORE, 'readonly');
            const req = tx.objectStore(_CS_IDB_STORE).get(key);
            req.onsuccess = () => {
                const data = req.result;
                if (data) {
                    const t1 = performance.now();
                    if (data.snapshots) {
                        // Lazy deserialization: store raw IDB snapshots as-is.
                        // _btRestoreSnapshot handles both raw arrays and Sets/Maps.
                        // Trail reconstruction happens incrementally during playback.
                        member.snapshots = data.snapshots;
                    }
                    const t2 = performance.now();
                    if (data.moves) {
                        member.moves = new Map();
                        for (const [tick, pairs] of data.moves) {
                            member.moves.set(tick, new Map(pairs));
                        }
                    }
                    member._cold = false;
                    const t3 = performance.now();
                    console.log(`%c[Council IDB] Hydrated member seed 0x${member.seed.toString(16).padStart(8,'0')} — IDB read: ${(t1-t0).toFixed(1)}ms, ${member.snapshots ? member.snapshots.length : 0} snapshots (lazy), moves deser: ${(t3-t2).toFixed(1)}ms, total: ${(t3-t0).toFixed(1)}ms`, 'color:#66ccff');
                }
                resolve();
            };
            req.onerror = () => resolve();
        } catch (e) { resolve(); }
    });
}

async function _hydrateCouncilMoves(lvl, member) {
    if (member.moves) return; // already has moves
    if (!_blIDBReady) await _blIDBOpen();
    if (!_blIDB) return;
    const t0 = performance.now();
    const key = _blacklistRuleKey(lvl) + '|' + member.seed;
    // Try lightweight moves-only key first, fall back to full blob
    const mvKey = key + '|mv';
    return new Promise((resolve) => {
        try {
            const tx = _blIDB.transaction(_CS_IDB_STORE, 'readonly');
            const store = tx.objectStore(_CS_IDB_STORE);
            const req = store.get(mvKey);
            req.onsuccess = () => {
                const data = req.result;
                if (data && data.moves) {
                    // Fast path: moves-only record
                    member.moves = new Map();
                    for (const [tick, pairs] of data.moves) {
                        member.moves.set(tick, new Map(pairs));
                    }
                    console.log(`[STARTUP] Hydrated moves (fast) for seed 0x${member.seed.toString(16).padStart(8,'0')}: ${member.moves.size} ticks in ${(performance.now()-t0).toFixed(1)}ms`);
                    resolve();
                } else {
                    // Fallback: read full blob (legacy, no |mv key yet)
                    const req2 = store.get(key);
                    req2.onsuccess = () => {
                        const full = req2.result;
                        if (full && full.moves) {
                            member.moves = new Map();
                            for (const [tick, pairs] of full.moves) {
                                member.moves.set(tick, new Map(pairs));
                            }
                            console.log(`[STARTUP] Hydrated moves (legacy fallback) for seed 0x${member.seed.toString(16).padStart(8,'0')}: ${member.moves.size} ticks in ${(performance.now()-t0).toFixed(1)}ms`);
                            // Migrate: write |mv key so next load is fast
                            try {
                                const wTx = _blIDB.transaction(_CS_IDB_STORE, 'readwrite');
                                wTx.objectStore(_CS_IDB_STORE).put({ seed: member.seed, moves: full.moves }, mvKey);
                                console.log(`[STARTUP] Migrated |mv key for seed 0x${member.seed.toString(16).padStart(8,'0')}`);
                            } catch (e) { /* best-effort */ }
                        }
                        resolve();
                    };
                    req2.onerror = () => resolve();
                }
            };
            req.onerror = () => resolve();
        } catch (e) { resolve(); }
    });
}

function _dehydrateCouncilMember(member) {
    member.snapshots = null;
    if (!_sweepActive) member.moves = null;
    member._cold = true;
}
