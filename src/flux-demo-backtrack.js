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
        // Matter/antimatter winding direction
        octWindingDirection: _octWindingDirection,
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
    // Restore opening phase flag
    if ('openingPhase' in snap) _openingPhase = snap.openingPhase;
    // Restore T79 state
    if ('octFullConsecutive' in snap) _octFullConsecutive = snap.octFullConsecutive;
    // Restore hadron balance counts
    if (snap.demoVisits) _demoVisits = JSON.parse(JSON.stringify(snap.demoVisits));
    // Restore matter/antimatter winding direction
    if ('octWindingDirection' in snap) _octWindingDirection = snap.octWindingDirection;
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

// Reset per-tick backtracking state (called after a clean tick).
// BFS state (_bfsFailTick, _bfsLayer, _bfsLayerRetries) is NOT reset here —
// it persists across demoTick() calls until the failure tick passes.
function _btReset() {
    _btRetryCount = 0;
    _btActive = false;
    _rewindRequested = false;
    _rewindViolation = null;
}

// Clear all BFS state (called when the failure tick finally passes or on demo restart).
function _bfsReset() {
    _bfsFailTick = -1;
    _bfsLayer = 0;
    _bfsLayerRetries = 0;
    _btBadMoveLedger.clear();
}
