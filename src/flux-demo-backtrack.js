// flux-demo-backtrack.js — Backtracker: state snapshots, BFS layers, exclusion ledger

// Save a full snapshot of choreography state before a tick executes.
function _btSaveSnapshot() {
    const snap = {
        tick: _demoTick,
        openingPhase: _openingPhase,
        // Per-xon state (deep copy of mutable fields)
        xons: _demoXons.map(x => ({
            node: x.node, prevNode: x.prevNode, _mode: x._mode,
            _assignedFace: x._assignedFace, _quarkType: x._quarkType,
            _loopSeq: x._loopSeq ? x._loopSeq.slice() : null,
            _loopStep: x._loopStep, col: x.col,
            _movedThisTick: x._movedThisTick, _evictedThisTick: x._evictedThisTick,
            _lastDir: x._lastDir, alive: x.alive, _highlightT: x._highlightT,
            _t60Ejected: !!x._t60Ejected, _weakLeftOct: !!x._weakLeftOct, _pendingWeakEjection: !!x._pendingWeakEjection,
            _dirBalance: x._dirBalance ? x._dirBalance.slice() : new Array(10).fill(0),
            _modeStats: x._modeStats ? { ...x._modeStats } : { oct: 0, tet: 0, idle_tet: 0, weak: 0 },
            trail: x.trail.slice(),
            trailColHistory: x.trailColHistory.slice(),
            _trailFrozenPos: x._trailFrozenPos ? x._trailFrozenPos.map(p => [p[0], p[1], p[2]]) : [],
        })),
        // Global SC sets (shallow copy — Set of primitive IDs)
        activeSet: new Set(activeSet),
        xonImpliedSet: new Set(xonImpliedSet),
        impliedSet: new Set(impliedSet),
        scAttribution: new Map(_scAttribution),
        // Solver vertex positions (deep copy)
        pos: pos.map(p => [p[0], p[1], p[2]]),
        // T79 state
        octFullConsecutive: _octFullConsecutive,
        // Hadron balance counts (must rewind with backtracker for correct T22)
        demoVisits: _demoVisits ? JSON.parse(JSON.stringify(_demoVisits)) : null,
        // Actualization-based quark counts (must rewind to stay in sync)
        actualizationVisits: _actualizationVisits ? JSON.parse(JSON.stringify(_actualizationVisits)) : null,
        // Per-face edge epoch (edges since last manifestation)
        faceEdgeEpoch: _faceEdgeEpoch ? JSON.parse(JSON.stringify(_faceEdgeEpoch)) : null,
        faceWasActualized: _faceWasActualized ? { ..._faceWasActualized } : null,
        // Edge balance counters (must rewind with backtracker)
        edgeBalance: _edgeBalance ? new Map([..._edgeBalance].map(([k, v]) => [k, { ...v }])) : null,
        // Ejection balance counters
        ejectionBalance: _ejectionBalance ? new Map(_ejectionBalance) : null,
        // Matter/antimatter winding direction
        octWindingDirection: _octWindingDirection,
        // Planck second counter (deformation events)
        planckSeconds: _planckSeconds,
    };
    _btSnapshots.push(snap);
}

// Restore choreography state from a snapshot.
function _btRestoreSnapshot(snap) {
    _demoTick = snap.tick;
    // Restore per-xon state
    for (let i = 0; i < _demoXons.length && i < snap.xons.length; i++) {
        const x = _demoXons[i], s = snap.xons[i];
        // Bypass node setter validation (adjacency/SC checks) during restore
        x._restoring = true;
        x.prevNode = s.prevNode;
        x.node = s.node;
        x._restoring = false;
        x._mode = s._mode;
        x._assignedFace = s._assignedFace;
        x._quarkType = s._quarkType;
        x._loopSeq = s._loopSeq ? s._loopSeq.slice() : null;
        x._loopStep = s._loopStep;
        x.col = s.col;
        x._movedThisTick = s._movedThisTick;
        x._evictedThisTick = s._evictedThisTick;
        x._lastDir = s._lastDir;
        x.alive = s.alive;
        x._highlightT = s._highlightT;
        x._t60Ejected = !!s._t60Ejected;
        x._weakLeftOct = !!s._weakLeftOct;
        x._pendingWeakEjection = !!s._pendingWeakEjection;
        x._dirBalance = s._dirBalance ? s._dirBalance.slice() : new Array(10).fill(0);
        x._modeStats = s._modeStats ? { ...s._modeStats } : { oct: 0, tet: 0, idle_tet: 0, weak: 0 };
        x.trail = s.trail.slice();
        x.trailColHistory = s.trailColHistory.slice();
        x._trailFrozenPos = s._trailFrozenPos ? s._trailFrozenPos.map(p => [p[0], p[1], p[2]]) : [];
        // Update visuals
        if (x.sparkMat) x.sparkMat.color.setHex(x.col);
        if (x.group && pos[x.node]) {
            x.group.position.set(pos[x.node][0], pos[x.node][1], pos[x.node][2]);
        }
        x.tweenT = 1; // snap to position (no interpolation)
    }
    // Restore SC sets
    activeSet.clear(); for (const id of snap.activeSet) activeSet.add(id);
    xonImpliedSet.clear(); for (const id of snap.xonImpliedSet) xonImpliedSet.add(id);
    impliedSet.clear(); for (const id of snap.impliedSet) impliedSet.add(id);
    _scAttribution.clear(); for (const [k, v] of snap.scAttribution) _scAttribution.set(k, v);
    // Restore solver positions
    for (let i = 0; i < pos.length && i < snap.pos.length; i++) {
        pos[i][0] = snap.pos[i][0];
        pos[i][1] = snap.pos[i][1];
        pos[i][2] = snap.pos[i][2];
    }
    // Frozen trail positions are already restored from snapshot (line 71) —
    // do NOT re-derive from pos[] as that defeats the purpose of static trails.
    // Restore opening phase flag
    if ('openingPhase' in snap) _openingPhase = snap.openingPhase;
    // Restore T79 state
    if ('octFullConsecutive' in snap) _octFullConsecutive = snap.octFullConsecutive;
    // Restore hadron balance counts
    if (snap.demoVisits) _demoVisits = JSON.parse(JSON.stringify(snap.demoVisits));
    // Restore actualization-based quark counts
    if (snap.actualizationVisits) _actualizationVisits = JSON.parse(JSON.stringify(snap.actualizationVisits));
    // Restore per-face edge epoch
    if (snap.faceEdgeEpoch) _faceEdgeEpoch = JSON.parse(JSON.stringify(snap.faceEdgeEpoch));
    if (snap.faceWasActualized) _faceWasActualized = { ...snap.faceWasActualized };
    // Restore edge balance counters
    if (snap.edgeBalance) _edgeBalance = new Map([...snap.edgeBalance].map(([k, v]) => [k, { ...v }]));
    // Restore ejection balance counters
    if (snap.ejectionBalance) _ejectionBalance = new Map(snap.ejectionBalance);
    // Restore matter/antimatter winding direction
    if ('octWindingDirection' in snap) _octWindingDirection = snap.octWindingDirection;
    // Restore Planck second counter
    if ('planckSeconds' in snap) _planckSeconds = snap.planckSeconds;
    // Clear tick-level state
    _moveRecord.clear();
    _moveTrace.length = 0;
}

// Extract which moves to exclude from a violation.
// Returns array of "xonIdx:destNode" strings.
function _btExtractExclusions() {
    // Use _moveTrace to find the moves that led to the violation
    const exclusions = [];
    if (!_rewindViolation) return exclusions;
    // T19: "node X has 2+ xons" — find all xons that moved TO that node
    const nodeMatch = _rewindViolation.match(/node (\d+)/);
    if (nodeMatch) {
        const collisionNode = parseInt(nodeMatch[1], 10);
        for (const trace of _moveTrace) {
            if (trace.to === collisionNode) {
                exclusions.push(`${trace.xonIdx}:${collisionNode}`);
            }
        }
        // If no trace found (xon didn't move = was already there), exclude
        // the OTHER xon that moved to it
        if (exclusions.length === 0) {
            for (let i = 0; i < _demoXons.length; i++) {
                if (_demoXons[i].node === collisionNode) {
                    exclusions.push(`${i}:${collisionNode}`);
                }
            }
        }
    }
    // T20: "stuck at node X" — the xon couldn't move because all exits were
    // blocked. Exclude the BLOCKER xons' moves that occupied those exits,
    // forcing them to choose different destinations on retry.
    const stuckMatch = _rewindViolation.match(/stuck at node (\d+)/);
    if (stuckMatch) {
        const stuckNode = parseInt(stuckMatch[1], 10);
        // Find oct-cage neighbors of stuckNode (the exits that were blocked)
        const exitNodes = new Set();
        for (const nb of (baseNeighbors[stuckNode] || [])) {
            if (!_octNodeSet || _octNodeSet.has(nb.node)) exitNodes.add(nb.node);
        }
        for (const sc of _localScNeighbors(stuckNode)) {
            const other = sc.a === stuckNode ? sc.b : sc.a;
            if (!_octNodeSet || _octNodeSet.has(other)) exitNodes.add(other);
        }
        // Exclude antipodal (it's already filtered from candidates)
        const stuckAntipodal = _octAntipodal.get(stuckNode);
        if (stuckAntipodal !== undefined) exitNodes.delete(stuckAntipodal);

        // For each blocked exit, exclude the move that put a xon there
        for (const exitNode of exitNodes) {
            for (const trace of _moveTrace) {
                if (trace.to === exitNode) {
                    exclusions.push(`${trace.xonIdx}:${exitNode}`);
                }
            }
            // If no xon moved there this tick (blocker was already there),
            // exclude the blocker staying at that position
            if (!_moveTrace.some(t => t.to === exitNode)) {
                for (let i = 0; i < _demoXons.length; i++) {
                    if (_demoXons[i].alive && _demoXons[i].node === exitNode) {
                        exclusions.push(`${i}:${exitNode}`);
                    }
                }
            }
        }
        // Also exclude the stuck xon staying at its own node
        for (let i = 0; i < _demoXons.length; i++) {
            if (_demoXons[i].node === stuckNode) {
                exclusions.push(`${i}:${stuckNode}`);
            }
        }
    }

    // T55: "N oct xons > capacity C" — too many xons in oct mode.
    // Exclude excess oct xons' moves to oct-cage nodes, forcing them
    // into idle_tet or tet paths instead.
    const capMatch = _rewindViolation.match(/(\d+) oct xons > capacity (\d+)/);
    if (capMatch && _octNodeSet) {
        const octCount = parseInt(capMatch[1], 10);
        const capacity = parseInt(capMatch[2], 10);
        const excess = octCount - capacity;
        // Find all oct-mode xons, sorted by most recently moved (from moveTrace)
        const octXons = [];
        for (let i = 0; i < _demoXons.length; i++) {
            if (_demoXons[i].alive && _demoXons[i]._mode === 'oct') {
                octXons.push(i);
            }
        }
        // Pick the last `excess` oct xons to exclude from oct nodes
        // (prefer to eject the ones that moved most recently)
        const toEject = octXons.slice(-excess);
        for (const xi of toEject) {
            // Exclude this xon from going to ANY oct cage node
            for (const octNode of _octNodeSet) {
                exclusions.push(`${xi}:${octNode}`);
            }
        }
    }

    // ── Universal fallback: if no specific handler matched, exclude every
    // move that happened this tick. This ensures the backtracker always
    // gets new information from any guard failure, not just T19/T20/T55.
    if (exclusions.length === 0 && _moveTrace.length > 0) {
        for (const trace of _moveTrace) {
            exclusions.push(`${trace.xonIdx}:${trace.to}`);
        }
    }

    return exclusions;
}

// Check if a candidate move is excluded by the persistent bad-move ledger.
// Consulted during ALL movement decisions, not just PHASE 2.
function _btIsMoveExcluded(xonIdx, destNode) {
    if (!_btActive) return false;
    const tickLedger = _btBadMoveLedger.get(_demoTick);
    if (!tickLedger) return false;
    return tickLedger.has(`${xonIdx}:${destNode}`);
}

// ══════════════════════════════════════════════════════════════════════════
// DETERMINISTIC MATCHING ENUMERATION
// ══════════════════════════════════════════════════════════════════════════

// Enumerate ALL valid maximum-cardinality bipartite matchings.
// plans: array of { xon, candidates: [{node, ...}] }
// blocked: Set of nodes reserved by higher-priority moves (tet)
// Returns: array of assignment arrays, where each assignment[i] = candidate for plan i (or null)
function _enumerateAllMatchings(plans, blocked) {
    const n = plans.length;
    if (n === 0) return [[]];

    // Build adjacency: for each plan, list of valid candidate nodes
    const adj = plans.map(p =>
        p.candidates.filter(c => !blocked.has(c.node)).map(c => c.node)
    );

    // First, find maximum cardinality via Kuhn's so we know the target size
    const maxCard = _kuhnMaxCardinality(adj, n);

    // Enumerate all matchings of size maxCard using recursive backtracking
    const results = [];
    const used = new Set(); // nodes claimed so far
    const current = new Array(n).fill(null);

    function enumerate(idx, matched) {
        // Pruning: can we still reach maxCard even if all remaining plans match?
        const remaining = n - idx;
        if (matched + remaining < maxCard) return;

        if (idx === n) {
            if (matched === maxCard) {
                results.push(current.slice());
            }
            return;
        }

        // Option A: assign plan[idx] to each available candidate
        for (const node of adj[idx]) {
            if (used.has(node)) continue;
            used.add(node);
            current[idx] = node;
            enumerate(idx + 1, matched + 1);
            current[idx] = null;
            used.delete(node);
        }

        // Option B: skip plan[idx] (only if we can still reach maxCard without it)
        if (matched + (n - idx - 1) >= maxCard) {
            current[idx] = null;
            enumerate(idx + 1, matched);
        }
    }

    enumerate(0, 0);

    // Convert node IDs back to candidate objects for compatibility
    return results.map(assignment =>
        assignment.map((node, i) => {
            if (node === null) return null;
            return plans[i].candidates.find(c => c.node === node) || null;
        })
    );
}

// Kuhn's algorithm to find maximum cardinality (just the count, not all matchings)
function _kuhnMaxCardinality(adj, n) {
    const match = new Map(); // node → plan index
    function augment(idx, visited) {
        for (const node of adj[idx]) {
            if (visited.has(node)) continue;
            visited.add(node);
            const existing = match.get(node);
            if (existing === undefined || augment(existing, visited)) {
                match.set(node, idx);
                return true;
            }
        }
        return false;
    }
    let card = 0;
    for (let i = 0; i < n; i++) {
        if (augment(i, new Set())) card++;
    }
    return card;
}

// Compute a canonical fingerprint of ALL moves this tick (not just oct).
// Format: "X0:5->9|X1:stay@0|X2:13->4|..." sorted by xon index.
function _computeTickFingerprint() {
    // Build a map from xon index to their move
    const moves = new Map();
    for (const trace of _moveTrace) {
        moves.set(trace.xonIdx, trace);
    }
    const parts = [];
    for (let i = 0; i < _demoXons.length; i++) {
        const x = _demoXons[i];
        if (!x.alive) continue;
        const trace = moves.get(i);
        if (trace) {
            parts.push(`X${i}:${trace.from}->${trace.to}`);
        } else {
            parts.push(`X${i}:stay@${x.node}`);
        }
    }
    return parts.join('|');
}

// Record the fingerprint for the current tick. Returns true if this is
// a genuinely new fingerprint (never tried before), false if duplicate.
function _btRecordFingerprint() {
    const tick = _demoTick - 1; // tick was already incremented
    const fp = _computeTickFingerprint();
    if (!_btTriedFingerprints.has(tick)) _btTriedFingerprints.set(tick, new Set());
    const fpSet = _btTriedFingerprints.get(tick);
    if (fpSet.has(fp)) return false;
    fpSet.add(fp);

    // ── Live comparison during Test 2 (random) ──
    // If we're in Test 2 and have a reference set from Test 1, check for novel fingerprints
    if (_bfsTestActive && _bfsTestRunIdx === 1 && _bfsTestReferenceFingerprints) {
        const refSet = _bfsTestReferenceFingerprints.get(tick);
        if (!refSet || !refSet.has(fp)) {
            // Novel fingerprint! Test 1 (choreographer) missed this solution
            _bfsTestEarlyAbort = true;
            _bfsTestNovelDetail = { tick, fingerprint: fp };
            console.log(`%c[DFS TEST] EARLY ABORT: Random found novel fingerprint at tick ${tick}`,
                'color:red;font-weight:bold');
            console.log(`  Fingerprint: ${fp}`);
            console.log(`  Reference set at tick ${tick}: ${refSet ? refSet.size : 0} entries`);
            // Stop the demo — the poll loop in _executeBfsTestRun will detect !_demoActive
            if (typeof stopDemo === 'function') stopDemo();
        }
    }

    return true;
}

// Check if the current tick is provably exhausted.
// Returns true if ALL matchings have been tried (no more options).
function _btIsTickExhausted() {
    // If we have a matching cache and we've tried all of them, exhausted
    if (_btMatchingCache !== null && _btMatchingIndex >= _btMatchingCache.length) {
        return true;
    }
    return false;
}

// Get the next matching to try from the cache.
// Returns the assignment array, or null if exhausted.
function _btNextMatching() {
    if (_btMatchingCache === null || _btMatchingIndex >= _btMatchingCache.length) return null;
    return _btMatchingCache[_btMatchingIndex++];
}

// Reset matching cache (called when moving to a new tick or BFS layer)
function _btResetMatchingCache() {
    _btMatchingCache = null;
    _btMatchingIndex = 0;
    _btMatchingCacheLedgerSize = 0;
}

// Reset per-tick backtracking state (called after a clean tick).
// BFS state (_bfsFailTick, _bfsLayer, _bfsLayerRetries) is NOT reset here —
// it persists across demoTick() calls until the failure tick passes.
function _btReset() {
    _btRetryCount = 0;
    _btActive = false;
    _rewindRequested = false;
    _rewindViolation = null;
    _btResetMatchingCache();
    _btStaleRetries = 0;
}

// Clear all BFS state (called when the failure tick finally passes or on demo restart).
function _bfsReset() {
    _bfsFailTick = -1;
    _bfsLayer = 0;
    _bfsLayerRetries = 0;
    // During BFS exhaustiveness test, preserve fingerprints and ledger for capture
    if (!_bfsTestActive) {
        _btBadMoveLedger.clear();
        _btTriedFingerprints.clear();
    }
    _btResetMatchingCache();
    _btStaleRetries = 0;
}
