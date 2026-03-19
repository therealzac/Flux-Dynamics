// flux-tests-unit.js — Run-once unit tests (runDemo3Tests), BFS comparison logic
// Split from flux-tests.js (lines 2074-2569). Loaded after flux-tests-guards.js.

// ═══ BFS Exhaustiveness Test — Comparison Logic ═════════════════════════════
// Pure function: compares two BFS run results for identical state-space exploration.
function _compareBfsRuns(resultA, resultB) {
    const maxTickMatch = resultA.maxTick === resultB.maxTick;
    const haltReasonMatch = resultA.haltReason === resultB.haltReason;
    const violationMatch = resultA.haltViolation === resultB.haltViolation;

    // Per-tick fingerprint comparison
    const allTicks = new Set([
        ...resultA.perTickFingerprints.keys(),
        ...resultB.perTickFingerprints.keys()
    ]);
    const ticksOnlyA = [], ticksOnlyB = [], ticksMismatch = [];
    let allFPMatch = true;

    for (const tick of [...allTicks].sort((a, b) => a - b)) {
        const setA = resultA.perTickFingerprints.get(tick);
        const setB = resultB.perTickFingerprints.get(tick);
        if (setA && !setB) { ticksOnlyA.push(tick); allFPMatch = false; }
        else if (!setA && setB) { ticksOnlyB.push(tick); allFPMatch = false; }
        else {
            let onlyA = 0, onlyB = 0, shared = 0;
            for (const fp of setA) { if (setB.has(fp)) shared++; else onlyA++; }
            for (const fp of setB) { if (!setA.has(fp)) onlyB++; }
            if (onlyA > 0 || onlyB > 0) {
                ticksMismatch.push({ tick, onlyA, onlyB, shared });
                allFPMatch = false;
            }
        }
    }

    // ── Key booleans ──
    // Same longest tick solution: both runs found the same highest-tick fingerprints
    const longestTickA = [...resultA.perTickFingerprints.keys()].sort((a, b) => b - a)[0] ?? -1;
    const longestTickB = [...resultB.perTickFingerprints.keys()].sort((a, b) => b - a)[0] ?? -1;
    const sameLongestTick = longestTickA === longestTickB;
    let sameLongestSolution = false;
    if (sameLongestTick && longestTickA >= 0) {
        const fpsA = resultA.perTickFingerprints.get(longestTickA);
        const fpsB = resultB.perTickFingerprints.get(longestTickB);
        if (fpsA && fpsB && fpsA.size === fpsB.size) {
            sameLongestSolution = true;
            for (const fp of fpsA) { if (!fpsB.has(fp)) { sameLongestSolution = false; break; } }
        }
    }

    // Same total solutions explored
    const sameTotalFingerprints = resultA.totalFingerprints === resultB.totalFingerprints;

    // Took different paths: first fingerprints at each tick were tried in different order
    // (i.e. not identical sequence — proves different exploration paths)
    let differentPaths = false;
    if (resultA.totalRetries !== resultB.totalRetries) {
        differentPaths = true; // different retry counts = clearly different paths
    } else {
        // Check if the per-tick fingerprint LISTS are in different order
        for (const tick of allTicks) {
            const listA = resultA.perTickFingerprints.get(tick);
            const listB = resultB.perTickFingerprints.get(tick);
            if (listA && listB) {
                const arrA = [...listA];
                const arrB = [...listB];
                if (arrA.length === arrB.length) {
                    for (let i = 0; i < arrA.length; i++) {
                        if (arrA[i] !== arrB[i]) { differentPaths = true; break; }
                    }
                }
                if (differentPaths) break;
            }
        }
    }

    const identical = maxTickMatch && haltReasonMatch && violationMatch && allFPMatch;
    let summary;
    if (identical) {
        summary = `PASS: Both runs explored identical state space. ` +
                  `Longest tick: ${longestTickA}. Halt: ${resultA.haltReason}. ` +
                  `Fingerprints: ${resultA.totalFingerprints}. ` +
                  `Paths: ${differentPaths ? 'DIFFERENT ✓' : 'same ✗'}. DFS is exhaustive.`;
    } else {
        const diffs = [];
        if (!maxTickMatch) diffs.push(`maxTick: ${resultA.maxTick} vs ${resultB.maxTick}`);
        if (!haltReasonMatch) diffs.push(`reason: ${resultA.haltReason} vs ${resultB.haltReason}`);
        if (!violationMatch) diffs.push(`violation: "${resultA.haltViolation}" vs "${resultB.haltViolation}"`);
        if (!allFPMatch) diffs.push(`${ticksMismatch.length} tick(s) with different fingerprints, ` +
            `${ticksOnlyA.length} ticks only in A, ${ticksOnlyB.length} ticks only in B`);
        summary = `FAIL: Runs diverged. ${diffs.join('; ')}`;
    }

    return {
        identical, maxTickMatch, haltReasonMatch, violationMatch, allFPMatch,
        sameLongestTick, sameLongestSolution, sameTotalFingerprints, differentPaths,
        fingerprintDiff: { ticksOnlyA, ticksOnlyB, ticksMismatch },
        summary,
    };
}

function runDemo3Tests() {
    _testRunning = true;
    const results = [];
    const pass = (name) => { results.push({ name, ok: true }); };
    const fail = (name, msg) => { results.push({ name, ok: false, msg }); };
    const skip = (name, msg) => { results.push({ name, ok: null, msg: msg || 'unproven' }); };
    const assert = (name, cond, msg) => cond ? pass(name) : fail(name, msg || 'assertion failed');

    // ── Ensure nucleus is simulated so we have valid state ──
    if (!NucleusSimulator.active) {
        NucleusSimulator.simulateNucleus();
    }

    // T01-T07: Now handled as live convergence guards in LIVE_GUARD_REGISTRY.
    // They start null and only pass when runtime conditions are met (e.g. tet faces discovered).

    // T08 REMOVED: Schedule structure test eliminated (window system removed)

    // T09 REMOVED: Tet face data test eliminated (dynamic discovery, face count varies)

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // T10 DISABLED: requires face data which is deferred during discovery
    // skip('T10 Xon spawning', 'disabled');

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // TEST 11: Xon advancement — _advanceXon updates state correctly
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // T11 DISABLED: requires face data which is deferred during discovery
    // skip('T11 Xon advancement (4 hops + wrap)', 'disabled');

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    //  PERSISTENT 6-XON MODEL (T12–T27)
    //  ALL deferred to LIVE MONITORING — continuous per-tick validation
    //  with grace period, permanent fail + halt on violation
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Auto-register all live guards from LIVE_GUARD_REGISTRY
    for (const entry of LIVE_GUARD_REGISTRY) {
        skip(`${entry.id} ${entry.name}`, 'grace period (live)');
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // T-OctSymmetry: No oct node is closer to a boundary than any other.
    // A boundary node has fewer than 8 base neighbors (interior nodes have 8).
    // For each oct node, find minimum hop distance to nearest boundary node.
    // All 6 oct nodes must have the same minimum distance.
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if (_octNodeSet && _octNodeSet.size === 6 && baseNeighbors && baseNeighbors.length > 0) {
        // Find boundary nodes: nodes with < 8 base neighbors
        const boundaryNodes = new Set();
        for (let i = 0; i < N; i++) {
            if (baseNeighbors[i].length < 8) boundaryNodes.add(i);
        }
        // BFS from each oct node to find min hops to nearest boundary
        function hopsToNearestBoundary(startNode) {
            if (boundaryNodes.has(startNode)) return 0;
            const visited = new Set([startNode]);
            let frontier = [startNode], dist = 0;
            while (frontier.length > 0) {
                dist++;
                const next = [];
                for (const n of frontier) {
                    for (const nb of baseNeighbors[n]) {
                        if (!visited.has(nb.node)) {
                            if (boundaryNodes.has(nb.node)) return dist;
                            visited.add(nb.node);
                            next.push(nb.node);
                        }
                    }
                }
                frontier = next;
            }
            return Infinity;
        }
        const octNodes = [..._octNodeSet];
        const distances = octNodes.map(n => hopsToNearestBoundary(n));
        const allEqual = distances.every(d => d === distances[0]);
        assert('T-OctSymmetry Lattice boundary equidistant from all oct nodes',
            allEqual,
            `oct node boundary distances: [${distances.join(', ')}] (nodes: [${octNodes.join(', ')}])`);
    } else {
        skip('T-OctSymmetry Lattice boundary equidistant from all oct nodes', 'no oct data');
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // T-RLStrategicInference: strategic model returns finite scores
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if (typeof _rlAvailable !== 'undefined' && _rlAvailable && typeof createStrategicModel === 'function') {
        try {
            const testModel = createStrategicModel();
            const testFeatures = new Float32Array(RL_STRATEGIC_FEATURES);
            for (let i = 0; i < RL_STRATEGIC_FEATURES; i++) testFeatures[i] = Math.random();
            const score = scoreStrategicRL(testFeatures, testModel);
            assert('T-RLStrat Strategic inference', isFinite(score), `score=${score} not finite`);
            testModel.dispose();
        } catch (e) {
            fail('T-RLStrat Strategic inference', e.message);
        }
    } else {
        skip('T-RLStrat Strategic inference', 'TF.js unavailable');
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // T-HadronicFitness: hadronic ratio fitness on known inputs
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if (typeof _evaluateHadronicRatioFitness === 'function') {
        // Uniform visits → fitness near 1.0
        const uniformVisits = {};
        for (let f = 1; f <= 8; f++) uniformVisits[f] = { pu1: 10, pu2: 10, pd: 10, nd1: 10, nd2: 10, nu: 10, total: 60 };
        const uniformF = _evaluateHadronicRatioFitness(uniformVisits);
        assert('T-HadFit Uniform → ~1.0', uniformF.fitness > 0.85, `uniform fitness=${uniformF.fitness.toFixed(3)} should be > 0.85`);

        // Single-type visits → fitness near 0.0
        const singleVisits = {};
        for (let f = 1; f <= 8; f++) singleVisits[f] = { pu1: 60, pu2: 0, pd: 0, nd1: 0, nd2: 0, nu: 0, total: 60 };
        const singleF = _evaluateHadronicRatioFitness(singleVisits);
        assert('T-HadFit Single-type → low', singleF.fitness < 0.3, `single-type fitness=${singleF.fitness.toFixed(3)} should be < 0.3`);

        // Empty visits → penalty
        const emptyVisits = {};
        for (let f = 1; f <= 8; f++) emptyVisits[f] = { pu1: 0, pu2: 0, pd: 0, nd1: 0, nd2: 0, nu: 0, total: 0 };
        const emptyF = _evaluateHadronicRatioFitness(emptyVisits);
        assert('T-HadFit Empty → -20', emptyF.fitness === -20, `empty fitness=${emptyF.fitness} should be -20`);
    } else {
        skip('T-HadFit Hadronic fitness', 'function not defined yet');
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // T-GenomeSplit: 2858 genome round-trips through model→genome→model
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if (typeof _rlAvailable !== 'undefined' && _rlAvailable &&
        typeof createStrategicModel === 'function' && typeof createPolicyModel === 'function') {
        try {
            const s1 = createStrategicModel();
            const t1 = createPolicyModel();
            const genome1 = modelToGenome(s1, t1);
            assert('T-GenSplit Genome size', genome1.length === getGenomeSize(),
                `genome.length=${genome1.length} != ${getGenomeSize()}`);

            // Load into fresh models and extract again — should match
            const s2 = createStrategicModel();
            const t2 = createPolicyModel();
            genomeToModel(genome1, s2, t2);
            const genome2 = modelToGenome(s2, t2);
            let maxDiff = 0;
            for (let i = 0; i < genome1.length; i++) {
                maxDiff = Math.max(maxDiff, Math.abs(genome1[i] - genome2[i]));
            }
            assert('T-GenSplit Round-trip fidelity', maxDiff < 1e-6,
                `max diff=${maxDiff} (should be < 1e-6)`);
            s1.dispose(); t1.dispose(); s2.dispose(); t2.dispose();
        } catch (e) {
            fail('T-GenSplit Genome round-trip', e.message);
        }
    } else {
        skip('T-GenSplit Genome round-trip', 'TF.js unavailable');
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // T-Temporal: strategic features include 6 temporal features (f[16]-f[21])
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if (typeof extractStrategicFeatures === 'function' && typeof RL_STRATEGIC_FEATURES !== 'undefined') {
        assert('T-Temporal Feature count', RL_STRATEGIC_FEATURES === 22,
            `RL_STRATEGIC_FEATURES=${RL_STRATEGIC_FEATURES} should be 22`);
        // Create a mock xon with temporal state
        const mockXon = {
            node: 0, prevNode: 0, alive: true, _mode: 'oct',
            _dirBalance: new Array(10).fill(5),
            _modeStats: { oct: 50, tet: 30, idle_tet: 10, weak: 10 },
            _octModeSince: 10,
        };
        const mockOccupied = new Map();
        // Ensure temporal state exists
        if (typeof _rlTemporalState !== 'undefined') {
            _rlTemporalState.faceLastVisitTick[1] = 100;
            _rlTemporalState.prevFaceCV[1] = 0.5;
        }
        const feats = extractStrategicFeatures(mockXon, 1, 'pu1', mockOccupied);
        assert('T-Temporal Feature length', feats.length === 22,
            `feature vector length=${feats.length} should be 22`);
        // f[16]-f[21] should be finite numbers
        let allFinite = true;
        for (let i = 16; i < 22; i++) {
            if (!isFinite(feats[i])) { allFinite = false; break; }
        }
        assert('T-Temporal All finite', allFinite, 'temporal features contain non-finite values');
    } else {
        skip('T-Temporal Temporal features', 'function not defined yet');
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // T-Sphericity: Lattice boundary must be spherical, not cuboctahedral.
    //
    // For all boundary nodes (baseNeighbors.length < 8), compute Euclidean
    // distance from the oct centroid [0, -r3, 0]. The "sphericity ratio"
    // is minDist / maxDist.
    //
    // A continuous sphere gives ratio = 1.0, but a DISCRETE FCC lattice
    // has a Kepler-like limit: the boundary spans ~1 lattice plane spacing
    // (~1.0) in depth, so the theoretical limit is 1 - thickness/R_max.
    // Empirically: L2≈0.77, L4≈0.86, L6≈0.90, L8≈0.92, approaching 1.0.
    //
    // An UNTRIMMED cuboctahedral lattice gives much worse ratios:
    // L2≈0.45, L4≈0.57, L6≈0.62, L8≈0.64 (bounded by 1/sqrt(2)≈0.707).
    //
    // SIZE-DEPENDENT THRESHOLD: max(0.7, 1 - 1.5/R_max).
    // This is below the discrete-lattice theoretical limit (1 - 1.0/R_max)
    // but well above the untrimmed cuboctahedral ratio at every level.
    // MATHEMATICALLY GUARANTEED separation between spherical and
    // cuboctahedral shapes at all lattice sizes L2+.
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if (baseNeighbors && baseNeighbors.length > 0 && N > 15) {
        const octCy = -r3;  // oct centroid y-coordinate
        // Identify boundary nodes: fewer than 8 base neighbors
        const boundaryDists = [];
        for (let i = 0; i < N; i++) {
            if (baseNeighbors[i].length < 8) {
                const [x, y, z] = REST[i];
                const dy = y - octCy;
                const dist = Math.sqrt(x * x + dy * dy + z * z);
                boundaryDists.push(dist);
            }
        }
        if (boundaryDists.length > 0) {
            const minDist = Math.min(...boundaryDists);
            const maxDist = Math.max(...boundaryDists);
            const sphericity = minDist / maxDist;
            // Size-dependent threshold: accounts for discrete lattice boundary thickness
            const threshold = Math.max(0.7, 1 - 1.5 / maxDist);
            assert(`T-Sphericity Lattice boundary is spherical (ratio > ${threshold.toFixed(3)})`,
                sphericity > threshold,
                `sphericity ratio = ${sphericity.toFixed(4)} (threshold=${threshold.toFixed(4)}, min=${minDist.toFixed(4)}, max=${maxDist.toFixed(4)}, boundary=${boundaryDists.length})`);
        } else {
            skip('T-Sphericity', 'no boundary nodes found');
        }
    } else {
        skip('T-Sphericity', 'lattice not built yet');
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // RESULTS
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    const passed = results.filter(r => r.ok === true).length;
    const nulled = results.filter(r => r.ok === null).length;
    const failed = results.filter(r => r.ok === false);
    console.log(`%c═══ Demo 3.0 Tests: ${passed}/${results.length} passed${nulled ? `, ${nulled} null` : ''} ═══`, 'font-weight:bold; font-size:14px');
    for (const r of results) {
        if (r.ok === true) console.log(`  %c✓ ${r.name}`, 'color:#44cc66');
        else if (r.ok === null) console.log(`  %c– ${r.name}: ${r.msg}`, 'color:#ccaa44');
        else console.log(`  %c✗ ${r.name}: ${r.msg}`, 'color:#ff4444; font-weight:bold');
    }
    if (failed.length === 0 && nulled === 0) {
        console.log('%c  ALL TESTS PASSED', 'color:#44cc66; font-weight:bold; font-size:12px');
    }

    // ── Update left panel ──
    const testSection = document.getElementById('dp-test-section');
    const testResultsEl = document.getElementById('dp-test-results');
    const testSummary = document.getElementById('dp-test-summary');
    if (testSection && testResultsEl) {
        testSection.style.display = '';
        const allPassed = failed.length === 0 && nulled === 0;
        testSummary.textContent = `${passed}/${results.length}${nulled ? ` (${nulled}?)` : ''}`;
        testSummary.style.color = allPassed ? '#66dd66' : (failed.length > 0 ? '#ff6644' : '#ccaa44');
        let html = '';
        for (const r of results) {
            const icon = r.ok === true ? '✓' : (r.ok === null ? '–' : '✗');
            const color = r.ok === true ? '#44cc66' : (r.ok === null ? '#ccaa44' : '#ff4444');
            const label = r.name.replace(/^T\d+\w?\s*/, '');
            const num = r.name.match(/^T(\d+\w?)/)?.[1] || '';
            html += `<div style="display:flex; gap:3px; align-items:baseline;">`
                + `<span style="color:${color}; font-weight:bold; min-width:10px;">${icon}</span>`
                + `<span style="color:#556677; min-width:18px;">T${num}</span>`
                + `<span style="color:${r.ok === true ? '#7a9aaa' : color};">${label}</span>`
                + (r.ok === true ? '' : `<span style="color:${r.ok === null ? '#aa8833' : '#aa4444'}; font-size:7px; margin-left:2px;">${r.msg || ''}</span>`)
                + `</div>`;
        }
        testResultsEl.innerHTML = html;
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // T-ENUM: Matching enumerator produces all valid matchings
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if (typeof _enumerateAllMatchings === 'function') {
        // 3 xons, triangle of candidates: A→{1,2}, B→{2,3}, C→{1,3}
        // Max cardinality = 3 (each gets a distinct node)
        // Valid matchings: A1-B2-C3, A1-B3-C?(no: 1 taken), A2-B3-C1, A2-B?(no 2 taken)...
        // Should be exactly 2: [1,2,3] and [2,3,1]
        const testPlans = [
            { xon: {}, candidates: [{node:1},{node:2}] },
            { xon: {}, candidates: [{node:2},{node:3}] },
            { xon: {}, candidates: [{node:1},{node:3}] },
        ];
        const matchings = _enumerateAllMatchings(testPlans, new Set());
        assert('T-ENUM Matching enumerator completeness',
            matchings.length === 2,
            `expected 2 matchings, got ${matchings.length}`);
        // Verify all matchings are maximum cardinality (3)
        const allMaxCard = matchings.every(m => m.filter(c => c !== null).length === 3);
        assert('T-ENUM All matchings are max cardinality',
            allMaxCard,
            `some matchings are not max cardinality`);
        // Verify no two matchings are identical
        const fps = new Set(matchings.map(m => m.map(c => c ? c.node : 'null').join(',')));
        assert('T-ENUM All matchings are distinct',
            fps.size === matchings.length,
            `duplicate matchings found`);
    } else {
        skip('T-ENUM Matching enumerator', '_enumerateAllMatchings not defined');
    }

    // ── T-BfsCompare: _compareBfsRuns correctness ──
    {
        const mockBase = () => ({
            seed: 0x11111111, maxTick: 5, haltReason: 'halted',
            haltViolation: 'T19: tick 5: Pauli', totalRetries: 10, totalFingerprints: 3,
            perTickFingerprints: new Map([
                [0, new Set(['X0:0->1|X1:stay@2'])],
                [1, new Set(['X0:1->3|X1:2->4', 'X0:1->5|X1:2->4'])]
            ]),
            perTickLedger: new Map(), elapsedMs: 100,
        });
        // Identical runs → PASS
        const a1 = mockBase(), b1 = mockBase();
        b1.seed = 0x22222222; b1.totalRetries = 8; b1.elapsedMs = 120;
        const cmp1 = _compareBfsRuns(a1, b1);
        assert('T-BfsCompare identical runs',
            cmp1.identical === true,
            `expected identical=true, got ${cmp1.identical}: ${cmp1.summary}`);
        // Divergent fingerprints → FAIL
        const a2 = mockBase(), b2 = mockBase();
        b2.perTickFingerprints.get(1).add('X0:1->7|X1:2->4'); // extra fingerprint in B
        b2.totalFingerprints = 4;
        const cmp2 = _compareBfsRuns(a2, b2);
        assert('T-BfsCompare divergent fingerprints',
            cmp2.identical === false,
            `expected identical=false, got ${cmp2.identical}`);
        // Different max tick → FAIL
        const a3 = mockBase(), b3 = mockBase();
        b3.maxTick = 7;
        const cmp3 = _compareBfsRuns(a3, b3);
        assert('T-BfsCompare different maxTick',
            cmp3.identical === false && cmp3.maxTickMatch === false,
            `expected maxTickMatch=false, got ${cmp3.maxTickMatch}`);
    }

    // ── T-DfsSecondary: secondary choices deterministic when _btActive ──
    {
        // _selectBestPermutation should NOT add PRNG noise when _btActive = true
        // Verify by calling it multiple times and checking for identical results
        if (typeof _selectBestPermutation === 'function' && _demoXons.length > 0 &&
            _nucleusTetFaceData && Object.keys(_nucleusTetFaceData).length > 0) {
            const testXon = _demoXons[0];
            const faceId = Object.keys(_nucleusTetFaceData)[0];
            const fd = _nucleusTetFaceData[faceId];
            const cycle = fd.cycle;
            const qType = 'pu1';

            // Test with _btActive = true: results should be identical across calls
            const prevBtActive = _btActive;
            _btActive = true;
            const results = [];
            for (let i = 0; i < 5; i++) {
                const seq = _selectBestPermutation(testXon, cycle, qType);
                results.push(seq ? seq.join(',') : 'null');
            }
            const allSame = results.every(r => r === results[0]);
            assert('T-DfsSecondary _btActive=true → deterministic permutation',
                allSame,
                `expected identical results, got ${results.length} unique: ${[...new Set(results)].join(' / ')}`);

            // Restore
            _btActive = prevBtActive;
        } else {
            skip('T-DfsSecondary', 'prerequisites not available');
        }
    }

    // ── Reset demo state after tests so visual demo starts clean ──
    _demoTick = 0;
    _planckSeconds = 0;
    _demoVisitedFaces = new Set();
    _demoTypeBalanceHistory = [];
    _demoPrevFaces = new Set();
    if (_demoVisits) for (let f = 1; f <= 8; f++) {
        _demoVisits[f] = { pu1: 0, pu2: 0, pd: 0, nd1: 0, nd2: 0, nu: 0, total: 0 };
    }
    // Return xons to oct mode at their current positions
    for (const xon of _demoXons) {
        if (xon.alive && (xon._mode === 'tet' || xon._mode === 'idle_tet')) _returnXonToOct(xon);
    }
    // Clear any tet SCs accumulated during tests
    for (const [fIdStr, fd] of Object.entries(_nucleusTetFaceData)) {
        for (const scId of fd.scIds) xonImpliedSet.delete(scId);
    }
    _testRunning = false;

    return { passed, total: results.length, failed: failed.map(f => f.name) };
}

