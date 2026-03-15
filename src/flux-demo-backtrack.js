// flux-demo-backtrack.js — Backtracker: state snapshots, BFS layers, exclusion ledger

// Gut a snapshot's heavy payload to free memory while keeping it as a
// tombstone in the stack. The tick field is preserved so _btSnapshots.find()
// still returns truthy, but restore will skip it. Call this when a tick is
// fully exhausted (all three relay phases completed, all options tried).
function _btPruneSnapshot(snap) {
    if (!snap || snap._pruned) return;
    snap._pruned = true;
    // Free heavy arrays/objects — keep only tick for lookup
    snap.xons = null;
    snap.activeSet = null;
    snap.xonImpliedSet = null;
    snap.impliedSet = null;
    snap.scAttribution = null;
    snap.pos = null;
    snap.demoVisits = null;
    snap.actualizationVisits = null;
    snap.faceEdgeEpoch = null;
    snap.faceWasActualized = null;
    snap.edgeBalance = null;
    snap.ejectionBalance = null;
}

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
            _gluonForFace: x._gluonForFace, _gluonBoundSCs: x._gluonBoundSCs ? x._gluonBoundSCs.slice() : null,
            _dirBalance: x._dirBalance ? x._dirBalance.slice() : new Array(10).fill(0),
            _modeStats: x._modeStats ? { ...x._modeStats } : { oct: 0, tet: 0, idle_tet: 0, weak: 0, gluon: 0 },
            trail: x.trail.slice(),
            trailColHistory: x.trailColHistory.slice(),
            _trailRoleHistory: x._trailRoleHistory ? x._trailRoleHistory.slice() : [],
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
        // Global mode stats (running totals)
        globalModeStats: { ..._globalModeStats },
    };
    _btSnapshots.push(snap);
}

// Restore choreography state from a snapshot.
function _btRestoreSnapshot(snap, reverse) {
    _demoTick = snap.tick;
    // Fighterjet reverse: capture current sprite positions BEFORE any state changes
    const fjReverse = _fighterjetMode && reverse;
    const savedPositions = fjReverse ? _demoXons.map(x =>
        x.group ? [x.group.position.x, x.group.position.y, x.group.position.z] : null
    ) : null;
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
        x._gluonForFace = s._gluonForFace != null ? s._gluonForFace : null;
        x._gluonBoundSCs = s._gluonBoundSCs ? s._gluonBoundSCs.slice() : null;
        // Re-derive _gluonClientXon from face binding (can't serialize xon references)
        x._gluonClientXon = null;
        if (x._gluonForFace != null) {
            const client = _demoXons.find((cx, ci) => cx.alive && snap.xons[ci] &&
                (snap.xons[ci]._mode === 'tet' || snap.xons[ci]._mode === 'idle_tet') &&
                snap.xons[ci]._assignedFace === x._gluonForFace && cx !== x);
            if (client) x._gluonClientXon = client;
        }
        x._dirBalance = s._dirBalance ? s._dirBalance.slice() : new Array(10).fill(0);
        x._modeStats = x._modeStats ? { ...s._modeStats } : { oct: 0, tet: 0, idle_tet: 0, weak: 0, gluon: 0 };
        x.trail = s.trail.slice();
        x.trailColHistory = s.trailColHistory.slice();
        x._trailRoleHistory = s._trailRoleHistory ? s._trailRoleHistory.slice() : [];
        x._trailFrozenPos = s._trailFrozenPos ? s._trailFrozenPos.map(p => [p[0], p[1], p[2]]) : [];
        // Pop most recent trail entry on reverse — visually removes the last hop
        if (fjReverse && x.trail.length > 0) {
            x.trail.pop();
            if (x.trailColHistory.length > 0) x.trailColHistory.pop();
            if (x._trailRoleHistory && x._trailRoleHistory.length > 0) x._trailRoleHistory.pop();
            if (x._trailFrozenPos && x._trailFrozenPos.length > 0) x._trailFrozenPos.pop();
        }
        // Update visuals
        if (x.sparkMat) x.sparkMat.color.setHex(x.col);
        if (fjReverse && savedPositions[i] && x.group) {
            // Reverse animation: sprite starts at old position, animates to restored pos
            // Trails are destroyed instantly (restored from snapshot)
            x._fjReverseFrom = savedPositions[i];
            x._fjRevT = 1; // independent reverse timer (1→0)
            x.group.position.set(savedPositions[i][0], savedPositions[i][1], savedPositions[i][2]);
            x.tweenT = 1;
        } else if (_fighterjetMode && x.group && pos[x.prevNode]) {
            // Forward fighterjet: start at prevNode, animate to node
            x._fjReverseFrom = null;
            x._fjRevT = 0;
            x.group.position.set(pos[x.prevNode][0], pos[x.prevNode][1], pos[x.prevNode][2]);
            x.tweenT = 0;
        } else {
            x._fjReverseFrom = null;
            x._fjRevT = 0;
            if (x.group && pos[x.node]) {
                x.group.position.set(pos[x.node][0], pos[x.node][1], pos[x.node][2]);
            }
            x.tweenT = 1; // snap to position
        }
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
    // Frozen trail positions are already restored from snapshot —
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
    if (snap.globalModeStats) _globalModeStats = { ...snap.globalModeStats };
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

// ══════════════════════════════════════════════════════════════════════════
// FACE ASSIGNMENT ENUMERATION (greedy + ranked alternatives)
// ══════════════════════════════════════════════════════════════════════════

// Enumerate all valid face assignment combos for a set of idle oct xons.
// Each combo is a one-to-one mapping: at most one xon per face, one face per xon.
// Returns: array of combos, each combo = array of {xon, face, quarkType, score}.
//   Sorted: greedy (highest aggregate score) first, then descending.
//
// proposals: array of {xon, face, quarkType, score, onFace} — ALL valid proposals
//   (multiple proposals per xon, multiple quark types per face, etc.)
function _enumerateAllFaceAssignments(proposals) {
    if (proposals.length === 0) return [[]];

    // Group proposals by xon (identity) to build the search tree
    const xonSet = [];
    const xonMap = new Map(); // xon → index in xonSet
    for (const p of proposals) {
        if (!xonMap.has(p.xon)) {
            xonMap.set(p.xon, xonSet.length);
            xonSet.push(p.xon);
        }
    }

    // Per-xon: list of {face, quarkType, score}
    const perXon = xonSet.map(() => []);
    for (const p of proposals) {
        const idx = xonMap.get(p.xon);
        perXon[idx].push({ face: p.face, quarkType: p.quarkType, score: p.score, xon: p.xon, onFace: p.onFace });
    }
    // Sort each xon's options by score descending (greedy preference)
    for (const opts of perXon) opts.sort((a, b) => b.score - a.score);

    const n = xonSet.length;
    const results = [];
    const usedFaces = new Set();
    const current = new Array(n).fill(null);

    // Cap enumeration to prevent combinatorial explosion
    const MAX_COMBOS = 200;

    function enumerate(idx) {
        if (results.length >= MAX_COMBOS) return;
        if (idx === n) {
            // Record this combo (only assigned entries)
            const combo = current.filter(Boolean).slice();
            results.push(combo);
            return;
        }

        // Option A: assign xon[idx] to each of its valid faces
        for (const opt of perXon[idx]) {
            if (usedFaces.has(opt.face)) continue;
            usedFaces.add(opt.face);
            current[idx] = opt;
            enumerate(idx + 1);
            current[idx] = null;
            usedFaces.delete(opt.face);
            if (results.length >= MAX_COMBOS) return;
        }

        // Option B: skip this xon (it doesn't get a face this tick)
        current[idx] = null;
        enumerate(idx + 1);
    }

    enumerate(0);

    // Sort combos: highest total score first (greedy = first combo tried)
    results.sort((a, b) => {
        const sa = a.reduce((s, p) => s + p.score, 0);
        const sb = b.reduce((s, p) => s + p.score, 0);
        return sb - sa;
    });

    return results;
}

// Parse a fingerprint string into structured moves array.
// Returns [{xonIdx, move, from, to, stayed}]
function _parseFingerprintMoves(fp) {
    return fp.split('|').map(part => {
        const match = part.match(/^X(\d+):(.+)$/);
        if (!match) return null;
        const xonIdx = +match[1];
        const move = match[2];
        const stayMatch = move.match(/^stay@(\d+)$/);
        const moveMatch = move.match(/^(\d+)->(\d+)$/);
        return {
            xonIdx, move,
            from: stayMatch ? +stayMatch[1] : (moveMatch ? +moveMatch[1] : null),
            to: stayMatch ? +stayMatch[1] : (moveMatch ? +moveMatch[2] : null),
            stayed: !!stayMatch,
        };
    }).filter(Boolean);
}

// Find the closest fingerprint from a reference set to a novel fingerprint.
// Returns {fp, sharedMoves, diffs: [{xonIdx, novelMove, choreoMove}]}
function _findClosestFingerprint(novelFP, novelMoves, refFPs) {
    if (refFPs.length === 0) return null;
    let bestMatch = null;
    let bestShared = -1;

    for (const refFP of refFPs) {
        const refMoves = _parseFingerprintMoves(refFP);
        let shared = 0;
        const diffs = [];
        for (const nm of novelMoves) {
            const rm = refMoves.find(r => r.xonIdx === nm.xonIdx);
            if (rm && rm.move === nm.move) {
                shared++;
            } else {
                diffs.push({
                    xonIdx: nm.xonIdx,
                    novelMove: nm.move,
                    choreoMove: rm ? rm.move : '(missing)',
                });
            }
        }
        if (shared > bestShared) {
            bestShared = shared;
            bestMatch = { fp: refFP, sharedMoves: shared, diffs };
        }
    }
    return bestMatch;
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
    // Cross-seed blacklist check: skip states proven dead in previous seeds
    if (_sweepActive && _sweepBlacklist.has(tick)) {
        if (_sweepBlacklist.get(tick).has(fp)) {
            _sweepBlacklistHits++;
            _sweepBlacklistHitsSeed++;
            return false;
        }
    }

    if (!_btTriedFingerprints.has(tick)) _btTriedFingerprints.set(tick, new Set());
    const fpSet = _btTriedFingerprints.get(tick);
    if (fpSet.has(fp)) return false;
    fpSet.add(fp);

    // ── Live comparison during Test 2 (random) ──
    // If we're in Test 2 and have a reference set from Test 1, log novel fingerprints
    // but DO NOT abort — let random run to completion to test rule satisfiability.
    if (_bfsTestActive && _bfsTestRunIdx === 1 && _bfsTestReferenceFingerprints) {
        const refSet = _bfsTestReferenceFingerprints.get(tick);
        if (!refSet || !refSet.has(fp)) {
            // Novel fingerprint — choreographer missed this solution
            // Track it but keep going
            if (!_bfsTestNovelCount) _bfsTestNovelCount = 0;
            _bfsTestNovelCount++;
            _bfsTestEarlyAbort = true; // flag that novel solutions exist

            // Only log first 5 novel fingerprints in detail to avoid spam
            if (_bfsTestNovelCount <= 5) {
                const novelMoves = _parseFingerprintMoves(fp);
                const refFPs = refSet ? [...refSet] : [];
                const closest = _findClosestFingerprint(fp, novelMoves, refFPs);

                console.log(`%c[DFS TEST] Novel fingerprint #${_bfsTestNovelCount} at tick ${tick}`,
                    'color:orange;font-weight:bold');
                console.log(`  Novel:   ${fp}`);
                console.log(`  Ref set: ${refSet ? refSet.size : 0} choreographer fingerprints at this tick`);
                if (closest) {
                    console.log(`  Closest match (${closest.sharedMoves}/${novelMoves.length} same):`);
                    for (const diff of closest.diffs) {
                        console.log(`    X${diff.xonIdx}: ${diff.novelMove} (random) vs ${diff.choreoMove} (choreographer)`);
                    }
                }

                // Capture first novel detail for the results panel
                if (_bfsTestNovelCount === 1) {
                    const xonStates = _demoXons.map((x, i) => ({
                        idx: i, node: x.node, prevNode: x.prevNode,
                        mode: x._mode, face: x._assignedFace, quark: x._quarkType,
                        step: x._loopStep, seq: x._loopSeq ? x._loopSeq.join(',') : null,
                    }));
                    _bfsTestNovelDetail = {
                        tick, fingerprint: fp, novelMoves, closest, xonStates,
                        refCount: refSet ? refSet.size : 0,
                        choreographerTicks: [..._bfsTestReferenceFingerprints.keys()].sort((a,b) => a-b),
                    };
                }
            }
            // DO NOT stop — let random continue to test satisfiability
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
    _btFaceAssignCache = null;
    _btFaceAssignIndex = 0;
    _btFaceAssignLedgerSize = 0;
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
    // Reset relay state on clean tick
    _relayPhase = 'normal';
    _bfsTestRandomChoreographer = false;
    _relayEnumFingerprints = null;
    _relayEnumAttempts = 0;
    _relayEnumStale = 0;
    _relayScoredQueue = null;
    _relayScoredIndex = 0;
}

// Clear all BFS state (called when the failure tick finally passes or on demo restart).
function _bfsReset() {
    _bfsFailTick = -1;
    _bfsLayer = 0;
    _bfsLayerRetries = 0;
    // During BFS exhaustiveness test or sweep, preserve fingerprints and ledger for capture
    if (!_bfsTestActive && !_sweepActive) {
        _btBadMoveLedger.clear();
        _btTriedFingerprints.clear();
    }
    _btResetMatchingCache();
    _btStaleRetries = 0;
    // Reset relay state
    _relayPhase = 'normal';
    _bfsTestRandomChoreographer = false;
    _relayEnumFingerprints = null;
    _relayEnumAttempts = 0;
    _relayEnumStale = 0;
    _relayScoredQueue = null;
    _relayScoredIndex = 0;
}

// Score a fingerprint for choreographer preference ordering.
// Higher score = more desirable path. Uses quark balance deficit
// and xon direction balance as scoring axes.
function _relayScoreFingerprint(fp, snapshot) {
    // Parse fingerprint to see where xons end up
    const moves = _parseFingerprintMoves(fp);
    let score = 0;

    // 1. Prefer moves where more xons actually moved (not stayed)
    const movedCount = moves.filter(m => !m.stayed).length;
    score += movedCount * 10;

    // 2. Prefer xon states with better quark balance
    // (We can't fully evaluate without replaying, but we can check
    //  if the move pattern distributes xons across more distinct nodes)
    const uniqueDestinations = new Set(moves.map(m => m.to));
    score += uniqueDestinations.size * 5;

    // 3. Penalize fingerprints where xons cluster on the same nodes
    // (early indicator of congestion that leads to dead ends)
    const destCounts = {};
    for (const m of moves) {
        destCounts[m.to] = (destCounts[m.to] || 0) + 1;
    }
    for (const count of Object.values(destCounts)) {
        if (count > 1) score -= count * 20; // heavy penalty for stacking
    }

    return score;
}
