// flux-demo-state.js — Shared state, constants, and PRNG for the demo choreography system

// ── Visual defaults (single source of truth) ────────────────────────────
// Applied once when user clicks play, and for council replay init.
const DEMO_VISUAL_DEFAULTS = [
    ['sphere-opacity-slider', 3], ['void-opacity-slider', 13],
    ['graph-opacity-slider', 13], ['trail-opacity-slider', 100],
    ['spark-opacity-slider', 100],
    ['brane-opacity-slider', 0], ['wf-opacity-slider', 0],
    ['bg-gray-slider', 0],
    ['orbit-speed-slider', 8], ['tracer-lifespan-slider', 1000],
];

// ── Core demo state ──────────────────────────────────────────────────────
let _demoActive = false;
let _demoOpDefaultsApplied = false; // true after first startDemoLoop() applies visual defaults
let _demoInterval = null;
let _demoPaused = false;  // true when user has paused via pause button
let _demoReversing = false;    // true during reverse playback
let _reverseInterval = null;   // setInterval ID for reverse playback
let _tickLog = [];             // lightweight per-tick log for export
let _tickLogLastGuards = {};   // last full guard state for delta encoding
// ── Movie export/import ──
let _movieFrames = [];         // lean per-tick frames for movie export
let _lastMoviePos = null;      // previous tick's pos[][] for delta compression
let _playbackMode = false;     // true during imported movie playback
let _playbackFrame = 0;        // current frame index in playback
let _importedMovie = null;     // parsed movie JSON during playback
let _replayCursor = -1;        // -1 = live play, >= 0 = replaying _btSnapshots[_replayCursor]
let _rlActiveModel = null;     // active RL model for oct scoring (null = use heuristic)
// PPO training state
let _ppoTraining = false;       // true during PPO training (enables trajectory collection)
let _ppoStrategicBuffer = null; // PPOTrajectoryBuffer for strategic decisions
let _ppoTacticalBuffer = null;  // PPOTrajectoryBuffer for tactical decisions
let _ppoStrategicAC = null;     // strategic actor-critic during training
let _ppoTacticalAC = null;      // tactical actor-critic during training
// T45 bounce guard — prevents A→B→A oscillation for oct AND weak xons.
// Only tet/idle_tet xons are exempt (actualized hadronic patterns like fork: a→b→a→c→a).
// Bounces are only allowed in actualized hadronic patterns that require them.
const _T45_BOUNCE_GUARD = false;
let _demoTick = 0;
let _planckSeconds = 0;  // ticks where lattice deformation occurred (SC adds/removes)
let _demoVisits = null;       // {face: {pu1:0, pu2:0, pd:0, nd1:0, nd2:0, nu:0}}
let _demoTetAssignments = 0;  // total tet assignments (for hit rate = completions / assignments)
let _actualizationVisits = null; // {face: {pu1:0,...}} — counts per-Planck-second tet actualization
// Choreographer always uses quark balance (jitter and emphasis dropdown removed)

// ── Per-Face Edge Epoch — tracks edge traversals since last manifestation ──
// _faceEdgeEpoch[faceId] = { pu1:0, pu2:0, pd:0, nd1:0, nd2:0, nu:0 }
// Resets when the face loses actualization (not all SCs active).
// _faceWasActualized[faceId] = bool — tracks previous-tick actualization state.
let _faceEdgeEpoch = null;
let _faceWasActualized = null;

// ── Edge Balance System — per-edge quark traversal counters ──
// Tracks how evenly all 6 quark types traverse each edge connected to the oct.
// "White" = perfectly balanced (all 6 types equal). Tinted = imbalanced.
// _octEdgeSet: Set of pairId strings for all edges with ≥1 oct endpoint
// _edgeBalance: Map<pairId, {pu1,pu2,pd,nd1,nd2,nu,total,fwd,rev}>
// fwd = traversals from min-node → max-node, rev = max → min
let _octEdgeSet = null;
let _edgeBalance = null;

// ── Ejection Balance System — per-edge weak force exit counters ──
// Tracks how evenly weak particles exit through the "dead edges" of each tet face
// (edges that handedness prevents quarks from traversing).
// _ejectionBalance: Map<pairId, count> — incremented each time a weak xon traverses the edge
// _ejectionEdgeSet: Set of pairId — all tet-face edges (same set as _octEdgeSet, weak uses the complement)
let _ejectionBalance = null;

// Initialize edge balance system — call after _nucleusTetFaceData and _octSCIds are populated
function _initEdgeBalance() {
    _octEdgeSet = new Set();
    _edgeBalance = new Map();
    if (!_nucleusTetFaceData) return;

    const addEdge = (a, b) => {
        const pid = pairId(a, b);
        if (!_octEdgeSet.has(pid)) {
            _octEdgeSet.add(pid);
            _edgeBalance.set(pid, { pu1: 0, pu2: 0, pd: 0, nd1: 0, nd2: 0, nu: 0, total: 0, fwd: 0, rev: 0 });
        }
    };

    // Collect edges from tet face cycles — these are the only edges quarks traverse.
    // Each face cycle = [a, b, c, d] (4 nodes of the tet). Quark loops traverse
    // all 6 edges of the K4 tetrahedron (a↔b, a↔c, a↔d, b↔c, b↔d, c↔d).
    for (let f = 1; f <= 8; f++) {
        const fd = _nucleusTetFaceData[f];
        if (!fd || !fd.cycle) continue;
        const [a, b, c, d] = fd.cycle;
        addEdge(a, b); addEdge(a, c); addEdge(a, d);
        addEdge(b, c); addEdge(b, d); addEdge(c, d);
    }

    // Also include oct cage edges (oct↔oct SC edges)
    if (_octSCIds) {
        for (const scId of _octSCIds) {
            const sc = SC_BY_ID[scId];
            if (sc) addEdge(sc.a, sc.b);
        }
    }

    console.log(`[edgeBalance] Tracking ${_octEdgeSet.size} edges from ${Object.keys(_nucleusTetFaceData).length} tet faces + oct cage`);
}

// Record a quark traversal on an edge (called from _advanceXon)
function _recordEdgeTraversal(fromNode, toNode, quarkType, faceId) {
    if (!_edgeBalance || !quarkType) return;
    const pid = pairId(fromNode, toNode);
    const entry = _edgeBalance.get(pid);
    if (!entry) return; // not an oct-adjacent edge
    if (entry[quarkType] !== undefined) {
        entry[quarkType]++;
        entry.total++;
    }
    // Track directionality: fwd = min→max, rev = max→min
    if (fromNode < toNode) entry.fwd++;
    else entry.rev++;
    // Accumulate into per-face epoch (edges since last manifestation)
    if (faceId != null && _faceEdgeEpoch) {
        if (!_faceEdgeEpoch[faceId]) {
            _faceEdgeEpoch[faceId] = { pu1: 0, pu2: 0, pd: 0, nd1: 0, nd2: 0, nu: 0 };
        }
        _faceEdgeEpoch[faceId][quarkType]++;
    }
}

// Determine dominant quark type for a face based on edge traversals since last manifestation.
// Uses _faceEdgeEpoch (resets when face loses actualization).
// Tie → most needed quark (lowest in _actualizationVisits). No contestants → null (bare tet).
function _dominantQuarkForFace(faceId) {
    const types = ['pu1', 'pu2', 'pd', 'nd1', 'nd2', 'nu'];
    const epoch = _faceEdgeEpoch ? _faceEdgeEpoch[faceId] : null;

    // Check if any edges were traversed since last manifestation
    let total = 0;
    if (epoch) {
        for (const t of types) total += epoch[t] || 0;
    }
    if (!epoch || total === 0) return null; // bare tetrahedra — no contestants

    // Find max epoch value
    let bestVal = 0;
    for (const t of types) {
        if ((epoch[t] || 0) > bestVal) bestVal = epoch[t] || 0;
    }

    // Collect all types tied at max
    const tied = [];
    for (const t of types) {
        if ((epoch[t] || 0) === bestVal) tied.push(t);
    }

    if (tied.length === 1) return tied[0];

    // Tiebreak: most needed quark (lowest in _actualizationVisits across ALL faces)
    const globalCounts = {};
    for (const t of types) globalCounts[t] = 0;
    if (_actualizationVisits) {
        for (const fv of Object.values(_actualizationVisits)) {
            for (const t of types) globalCounts[t] += fv[t] || 0;
        }
    }
    let mostNeeded = tied[0], lowestCount = Infinity;
    for (const t of tied) {
        if (globalCounts[t] < lowestCount) {
            lowestCount = globalCounts[t];
            mostNeeded = t;
        }
    }
    return mostNeeded;
}

// Apply tet face coloring based on geometric actualization.
// Called from demoTick (live) and _playbackUpdateDisplay (replay).
// countVisits: if true, increment _actualizationVisits (only during live ticks, not replay).
function _applyTetColoring(countVisits) {
    if (!_nucleusTetFaceData) return;
    if (!_faceEdgeEpoch) _faceEdgeEpoch = {};
    if (!_faceWasActualized) _faceWasActualized = {};
    for (const [fIdStr, fd] of Object.entries(_nucleusTetFaceData)) {
        const fId = parseInt(fIdStr);
        const allSCsActive = fd.scIds.every(scId =>
            activeSet.has(scId) || impliedSet.has(scId) || xonImpliedSet.has(scId));
        const wasActualized = !!_faceWasActualized[fId];

        if (allSCsActive) {
            const dominant = _dominantQuarkForFace(fId);
            if (dominant) {
                _ruleAnnotations.tetColors.set(fd.voidIdx, QUARK_COLORS[dominant]);
            } else {
                // Bare tetrahedra — no edge contributors since last manifestation
                _ruleAnnotations.tetColors.set(fd.voidIdx, 0x444466);
            }
            _ruleAnnotations.tetOpacity.set(fd.voidIdx, 1.0);
            if (countVisits && _actualizationVisits) {
                if (!_actualizationVisits[fId]) {
                    _actualizationVisits[fId] = { pu1: 0, pu2: 0, pd: 0, nd1: 0, nd2: 0, nu: 0 };
                }
                if (dominant) _actualizationVisits[fId][dominant]++;
            }
            _faceWasActualized[fId] = true;
        } else {
            _ruleAnnotations.tetColors.set(fd.voidIdx, 0x1a1a2a);
            _ruleAnnotations.tetOpacity.set(fd.voidIdx, 0.0);
            // Face lost actualization → reset epoch for next manifestation
            if (wasActualized && countVisits) {
                _faceEdgeEpoch[fId] = null;
            }
            _faceWasActualized[fId] = false;
        }
    }
    _ruleAnnotations.dirty = true;
}

// Compute scalar edge-evenness score [0,1] — 1.0 = all 6 quark types visit each edge equally
// in both directions. Blends quark-type evenness (70%) with directional evenness (30%).
function _computeEdgeEvenness() {
    if (!_edgeBalance || _edgeBalance.size === 0) return 0;
    const types = ['pu1', 'pu2', 'pd', 'nd1', 'nd2', 'nu'];
    let totalTypeScore = 0, totalDirScore = 0;
    let edgesWithData = 0;
    for (const [pid, counts] of _edgeBalance) {
        if (counts.total === 0) continue;
        edgesWithData++;
        // Quark-type evenness: perfect = each type is 1/6 of total
        const ideal = counts.total / 6;
        let deviation = 0;
        for (const t of types) deviation += Math.abs(counts[t] - ideal);
        const maxDev = counts.total * (10 / 6); // worst case: all one type
        totalTypeScore += 1 - (deviation / maxDev);
        // Directional evenness: perfect = fwd ≈ rev (50/50)
        const dirIdeal = counts.total / 2;
        const dirDev = Math.abs(counts.fwd - dirIdeal);
        const maxDirDev = counts.total / 2; // worst case: all one direction
        totalDirScore += maxDirDev > 0 ? 1 - (dirDev / maxDirDev) : 1;
    }
    if (edgesWithData === 0) return 0;
    const typeEvenness = totalTypeScore / edgesWithData;
    const dirEvenness = totalDirScore / edgesWithData;
    return typeEvenness * 0.7 + dirEvenness * 0.3;
}

// ── Ejection Balance — track weak xon traversals through tet-face edges ──

function _initEjectionBalance() {
    _ejectionBalance = new Map();
    if (!_nucleusTetFaceData) return;
    // Chirality determines which edges weak xons can use.
    // Per face cycle [a, b, c, d]: a↔b (pole↔ext) and c↔d (oct↔oct) are chirality-forbidden.
    // Eligible ejection edges: a↔c, a↔d, b↔c, b↔d (4 per face).
    const forbidden = new Set();
    const eligible = new Set();
    for (let f = 1; f <= 8; f++) {
        const fd = _nucleusTetFaceData[f];
        if (!fd || !fd.cycle) continue;
        const [a, b, c, d] = fd.cycle;
        forbidden.add(pairId(a, b)); // pole↔ext
        forbidden.add(pairId(c, d)); // oct↔oct non-cage
        eligible.add(pairId(a, c));
        eligible.add(pairId(a, d));
        eligible.add(pairId(b, c));
        eligible.add(pairId(b, d));
    }
    // Remove any that ended up in both (shouldn't happen, but safety)
    for (const pid of forbidden) eligible.delete(pid);
    for (const pid of eligible) {
        _ejectionBalance.set(pid, 0);
    }
    console.log(`[ejectionBalance] Tracking ${_ejectionBalance.size} eligible edges (${forbidden.size} chirality-forbidden)`);
}

// Record a weak xon traversal on a tet-face edge
function _recordEjectionTraversal(fromNode, toNode) {
    if (!_ejectionBalance) return;
    const pid = pairId(fromNode, toNode);
    if (_ejectionBalance.has(pid)) {
        _ejectionBalance.set(pid, _ejectionBalance.get(pid) + 1);
    }
}

// Compute scalar ejection evenness [0,1] — 1.0 = all eligible edges have equal ejection counts
// All edges in _ejectionBalance are chirality-eligible, so consider all of them.
function _computeEjectionEvenness() {
    if (!_ejectionBalance || _ejectionBalance.size === 0) return 0;
    const counts = [];
    for (const [, c] of _ejectionBalance) counts.push(c);
    const total = counts.reduce((a, b) => a + b, 0);
    if (total === 0) return 0;
    const n = counts.length;
    const ideal = total / n;
    let deviation = 0;
    for (const c of counts) deviation += Math.abs(c - ideal);
    const maxDev = 2 * total * (1 - 1 / n);
    return maxDev > 0 ? 1 - (deviation / maxDev) : 0;
}

// ── Balance History — ring buffer for time-series chart ──
// Records one sample per planck second: {ps, quark, edge, ejection} as percentages [0-100]
let _balanceHistory = [];
let _balanceTimeframe = 'all';  // 'all' | '1000' | '250'

function _recordBalanceSample() {
    if (!_demoVisits) return;
    // Quark evenness (same CV logic as updateDemoPanel)
    const totals = [];
    for (let f = 1; f <= 8; f++) totals.push(_demoVisits[f] ? _demoVisits[f].total : 0);
    const mean = totals.reduce((a, b) => a + b, 0) / totals.length;
    const stddev = Math.sqrt(totals.reduce((s, v) => s + (v - mean) ** 2, 0) / totals.length);
    const cv = mean > 0 ? (stddev / mean) : 1;
    const quark = Math.max(0, 1 - cv) * 100;

    const edge = _computeEdgeEvenness() * 100;
    const ejection = _computeEjectionEvenness() * 100;

    _balanceHistory.push({ ps: _planckSeconds, quark, edge, ejection });
}

// ── Oct Center Bias — xon movement prefers proximity to geometric oct center ──
// The oct geometric center {0, -0.5774, 0} is the centroid of the 6 oct nodes.
// This bias replaces the emergent spawn-point bias from _dirBalance with an
// explicit positional preference towards the nuclear center.
const _OCT_CENTER = [0.0, -0.5774, 0.0];

// Compute oct-center proximity score for a candidate node.
// Returns a small positive bonus for nodes closer to oct center.
// Uses solver positions (pos[]) for live deformed coordinates.
function _octCenterBias(nodeIdx) {
    if (!pos || !pos[nodeIdx]) return 0;
    const dx = pos[nodeIdx][0] - _OCT_CENTER[0];
    const dy = pos[nodeIdx][1] - _OCT_CENTER[1];
    const dz = pos[nodeIdx][2] - _OCT_CENTER[2];
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
    // Invert: closer = higher score. Scale so 1 unit away ≈ 1.0 score.
    // Clamp at 0 to avoid rewarding being at the center itself too heavily.
    return Math.max(0, 2.0 - dist);
}

// ── Rolling Ratio Tracker — demand-driven quark type selection ──
// Syncs from _demoVisits each tick. Computes deficit for any quark type.
// Target fractions: each type = 1/3 within its hadron (3-way split).
const _ratioTracker = {
    pu1: 0, pu2: 0, pd: 0, nd1: 0, nd2: 0, nu: 0,
    sync() {
        this.pu1 = 0; this.pu2 = 0; this.pd = 0;
        this.nd1 = 0; this.nd2 = 0; this.nu = 0;
        for (let f = 1; f <= 8; f++) {
            if (!_demoVisits || !_demoVisits[f]) continue;
            this.pu1 += _demoVisits[f].pu1 || 0;
            this.pu2 += _demoVisits[f].pu2 || 0;
            this.pd  += _demoVisits[f].pd  || 0;
            this.nd1 += _demoVisits[f].nd1 || 0;
            this.nd2 += _demoVisits[f].nd2 || 0;
            this.nu  += _demoVisits[f].nu  || 0;
        }
    },
    // Returns positive value when type is underrepresented vs 1/3 target
    deficit(type) {
        const protonTotal = this.pu1 + this.pu2 + this.pd;
        const neutronTotal = this.nd1 + this.nd2 + this.nu;
        if (type === 'pu1') return protonTotal === 0 ? 1.0 : (1/3) - this.pu1 / protonTotal;
        if (type === 'pu2') return protonTotal === 0 ? 1.0 : (1/3) - this.pu2 / protonTotal;
        if (type === 'pd')  return protonTotal === 0 ? 1.0 : (1/3) - this.pd / protonTotal;
        if (type === 'nd1') return neutronTotal === 0 ? 1.0 : (1/3) - this.nd1 / neutronTotal;
        if (type === 'nd2') return neutronTotal === 0 ? 1.0 : (1/3) - this.nd2 / neutronTotal;
        if (type === 'nu')  return neutronTotal === 0 ? 1.0 : (1/3) - this.nu / neutronTotal;
        return 0;
    }
};
// ── Temporal state for RL strategic features ──
// Updated each tick in demoTick. Read by extractStrategicFeatures.
const _rlTemporalState = {
    faceLastVisitTick: {},   // face → tick when last tet loop completed on this face
    prevFaceCV: {},          // face → CV of quark types from 64 ticks ago (for velocity)
    globalPressure: 0,       // fraction of faces below-target on balance
    reset() {
        this.faceLastVisitTick = {};
        this.prevFaceCV = {};
        this.globalPressure = 0;
    }
};

let _demoPauliViolations = 0;
let _demoSpreadViolations = 0;
let _demoTypeBalanceHistory = [];  // type balance % at each cycle boundary
let _demoVisitedFaces = new Set(); // faces activated so far (for oct reveal)
let _demoOctRevealed = false;      // oct renders once all 8 faces visited

// ── Demo 3.0: Xon-choreographed particle manifestation ──────────────
// Xons physically trace loop topologies to cut shortcuts.
// Gluons maintain the octahedral cage between fermionic loops.
let _demoXons = [];               // active xon objects (dynamic count)
let _demoGluons = [];             // active gluon objects (lightweight)
let _globalModeStats = { oct: 0, tet: 0, idle_tet: 0, weak: 0, gluon: 0 }; // running total of xon-ticks per mode

// ── Rule switches (switchboard) ────────────────────────────────────────────
let _ruleRelinquishSCs = false; // When true, SCs actively removed after loops. When false, persist until vacuum severs.
let _ruleGluonMediatedSC = true; // When true, gluon xons physically maintain tet face SCs instead of code-based relinquishment.
let _ruleBareTetrahedra = false; // When true, actualized tets with no edge contributors are violations (T86). When false, bare tets are allowed — they simply don't count as quarks.
let _demoPrevFaces = new Set();   // faces active in previous window (for relinquishing)
let _idleTetManifested = false;   // set by _startIdleTetLoop when new SCs are materialised
let T79_MAX_FULL_TICKS = 1;       // T79: max consecutive ticks allowed with all 6 xons on oct nodes
let _ruleT20StrictMode = true;   // When true, T20 has no mode-transition exemption
let _octFullConsecutive = 0;      // T79: running count of consecutive full-oct ticks
// T41: tick-level move record — tracks destNode → fromNode for all xon moves this tick.
// Used to prevent adjacent xon swaps (A→B while B→A in the same tick).
const _moveRecord = new Map();
let _noSwapRule = true; // T41: swap prevention always active — xons may not swap positions
// Annihilation toggle — set false to disable pair annihilation/genesis.
// When off, PHASE 4 uses scatter-only; unresolvable Pauli collisions
// fall through to weak force escape instead of gluon storage.
let _annihilationEnabled = false;
// Choreographer debug log — ring buffer of last N entries
let _choreoLog = [];
const _CHOREO_LOG_MAX = 20;
// Xon panel highlight state
let _xonHighlightTimers = new Map(); // xon index → timeout id
// Flash toggle — set false to disable mode-transition flash effects.
// Re-enable by setting to true. Flash = sparkle scale/brightness pulse on mode change.
let _flashEnabled = false;
// Kuhn's bipartite matching toggle. When false, Phase 2 uses greedy
// first-fit assignment instead of augmenting-path matching. Greedy is
// simpler and avoids the swap-removal Pauli gap where unmatched xons
// collide with matched ones. Re-enable for optimal throughput once the
// collision bug is fixed.
let _kuhnEnabled = false;
let _ruleProjectedGuards = true; // When true, projected guards gate candidate moves pre-execution
let T90_TOLERANCE = 12; // T90: ticks a first-place quark may stay actualized before ejection
let T91_TOLERANCE = 12; // T91: ticks a first-place face may stay actualized before ejection
let T92_TOLERANCE = 12; // T92: ticks a leading hadron face may stay actualized before ejection
let _ruleAdaptiveEjection = true; // Rule 9: adaptive √n tolerance (mutually exclusive with rule 8 family)
let _ruleCubeRootEjection = false; // Rule 10: adaptive ∛n tolerance (mutually exclusive with rules 8 & 9)
const _SNAPSHOT_VERSION = 2; // Bump when snapshot format changes; separates IDB keyspaces

// ── Per-xon and per-role opacity ──
let _xonOpacity = new Float32Array(6).fill(1.0);   // Per-xon opacity multiplier (0–1)
let _roleOpacity = { pu1:1, pu2:1, pd:1, nd1:1, nd2:1, nu:1, oct:1, gluon:1, weak:0.13 };

// ── All-time role stats (ticks spent in each role across all xons) ──
let _globalRoleStats = { pu1:0, pu2:0, pd:0, nd1:0, nd2:0, nu:0, oct:0, gluon:0, weak:0 };

// ── Trail visual parameters ──
let _trailFadeFloor = 0.15; // 0 = full fade (tail invisible), 1 = no fade (uniform brightness)
// ── Trail curve parameters ──
let _fjCurvature = 0.89; // 0 = straight, 1 = standard CR, 2 = exaggerated
let _fjTension   = 1.0;  // CR tension τ: 0 = flat tangents, 0.5 = standard, 1 = aggressive
let _fjAlpha     = 0.0;  // CR parameterization: 0 = uniform, 0.5 = centripetal, 1 = chordal

// ── Seeded PRNG for deterministic backtracker replay ─────────────────
// Mulberry32: fast 32-bit seeded PRNG. Returns float in [0, 1).
// The backtracker requires deterministic forward replay from the same
// snapshot + exclusions. Math.random() breaks this because it's unseeded.
// All choreography randomness MUST use _sRng() instead of Math.random().
let _sRngState = 0;
let _runSeed = 0; // randomized per demo run so DFS explores different branches
let _forceSeed = null; // set via console or URL param to replay a specific seed
let _maxTickReached = 0; // high-water mark for current run
let _searchStartTime = 0; // performance.now() when demo started
let _totalBacktrackRetries = 0; // total retries across all ticks/layers
let _bestPathFingerprint = ''; // fingerprint of the tick that achieved _maxTickReached

// ── BFS Exhaustiveness Test Mode ──
let _bfsTestActive = false;       // true while BFS exhaustiveness test is running
let _bfsTestRunIdx = 0;           // 0 = Run A, 1 = Run B
let _bfsTestResults = [null, null]; // captured results for Run A and Run B
let _bfsTestSeeds = [0, 0];      // seeds for the two runs
let _bfsTestMaxTick = 200;       // max ticks before forced termination
let _bfsTestComparison = null;   // comparison result after both runs complete
let _bfsTestRandomChoreographer = false; // true = totally random decisions (no heuristic scoring)
let _bfsTestReferenceFingerprints = null; // Map<tick, Set<string>> from Test 1 (choreographer)
let _bfsTestEarlyAbort = false;           // true if Test 2 found a novel fingerprint
let _bfsTestNovelCount = 0;              // total novel fingerprints found by random
let _bfsTestNovelDetail = null;           // {tick, fingerprint} of the first novel solution
let _bfsTestDecisionTrace = [];          // [{tick, faceAssignments, octMatching, btActive}] — per-tick decisions during test

// ── Hybrid Relay State (greedy + random enumeration at each layer) ──
// When normal choreographer exhausts a tick, we enumerate ALL valid fingerprints
// via random seeds, then let the choreographer score and try them in order.
let _relayPhase = 'normal';           // 'normal' | 'enumerating' | 'replaying'
let _relayEnumFingerprints = null;    // Map<string, {fp, snapshot, moves}> — all valid FPs found during enumeration
let _relayEnumAttempts = 0;           // how many random seeds tried during enumeration
let _relayEnumStale = 0;             // consecutive attempts with no new FP
let _relayScoredQueue = null;         // Array of {fp, snapshot, score} — sorted best-first for replay
let _relayScoredIndex = 0;           // which scored option we're currently trying
let _relayEscapes = 0;               // total successful relay-assisted advances this run
let _relayEnumTotal = 0;             // total fingerprints enumerated across all stuck ticks

// Save run result to localStorage for DFS audit. Appends to a history array
// so multiple runs can be compared. Key: 'flux_run_history'.
function _saveRunResult(reason, violation) {
    const elapsed = ((performance.now() - _searchStartTime) / 1000).toFixed(1);
    const totalFP = [..._btTriedFingerprints.values()].reduce((s, set) => s + set.size, 0);
    const entry = {
        seed: '0x' + _runSeed.toString(16).padStart(8, '0'),
        maxTick: _maxTickReached,
        reason,
        violation: violation || '',
        searchSeconds: parseFloat(elapsed),
        totalRetries: _totalBacktrackRetries,
        totalFingerprints: totalFP,
        bestPathId: _bestPathFingerprint,
        ts: new Date().toISOString(),
    };
    const seedStr = entry.seed;
    console.log(
        `%c╔══════════════════════════════════════════════════════════╗\n` +
        `║  SEARCH COMPLETE                                        ║\n` +
        `╠══════════════════════════════════════════════════════════╣\n` +
        `║  Seed:          ${seedStr.padEnd(39)}║\n` +
        `║  Peak tick:     ${String(_maxTickReached).padEnd(39)}║\n` +
        `║  Reason:        ${reason.padEnd(39)}║\n` +
        `║  Search time:   ${(elapsed + 's').padEnd(39)}║\n` +
        `║  Total retries: ${String(_totalBacktrackRetries).padEnd(39)}║\n` +
        `║  Fingerprints:  ${String(totalFP).padEnd(39)}║\n` +
        `║  Best path ID:  ${(_bestPathFingerprint || 'n/a').substring(0, 39).padEnd(39)}║\n` +
        `╚══════════════════════════════════════════════════════════╝`,
        'color:cyan;font-weight:bold;font-size:12px'
    );
    try {
        const hist = JSON.parse(localStorage.getItem('flux_run_history') || '[]');
        hist.push(entry);
        if (hist.length > 50) hist.splice(0, hist.length - 50);
        localStorage.setItem('flux_run_history', JSON.stringify(hist));
    } catch (e) { console.error('[RUN RESULT] save failed:', e); }
}
function _sRng() {
    _sRngState |= 0;
    _sRngState = (_sRngState + 0x6D2B79F5) | 0;
    let t = Math.imul(_sRngState ^ (_sRngState >>> 15), 1 | _sRngState);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
// Seed from tick number — call at the start of each tick so replays
// from the same snapshot produce the same random sequence.
function _sRngSeed(tick) { _sRngState = (tick * 2654435761 + _runSeed) | 0; }
// Fisher-Yates shuffle using seeded PRNG. Consumes exactly arr.length-1
// PRNG values — deterministic unlike .sort(() => _sRng() - 0.5) which
// consumes a variable number depending on the sort algorithm internals.
function _sRngShuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(_sRng() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
}

// ── Diagnostic trace — permanent, extensible ─────────────────────────
// Records every physical xon move with source code path label.
// Used by T41/T26/T27 diagnostics and future debugging.
const _moveTrace = []; // [{xonIdx, from, to, path, mode, tick}] — current tick only
const _moveTraceHistory = []; // rolling 5-tick history for dump audits
// Set of all legitimate nucleus nodes (oct cage + tet face vertices).
// Built lazily on first _traceMove call to ensure all nucleus data is ready.
let _nucleusNodeSet = null;
let _ejectionTargetNodes = null; // nodes 1 hop from oct, not in any tet/oct void
// ── Node classification sets (see CHOREOGRAPHER DESIGN INTENT §3) ──
let _actualizedTetNodes = null;  // nodes on currently actualized (all SCs active) tets — DYNAMIC
let _nucleusFaceNodes = null;    // union of _nucleusTetFaceData[1..8].allNodes — STATIC
let _ejectionForbidden = null;   // _octNodeSet ∪ _actualizedTetNodes — DYNAMIC
let _purelyTetNodes = null;      // _actualizedTetNodes \ _octNodeSet — DYNAMIC
// ── Gluon mode ──
let GLUON_COLOR = 0x000000;   // overwritten by _recomputeColors()

// ── SC attribution ──
const _scAttribution = new Map();

// ══════════════════════════════════════════════════════════════════════════
// BACKTRACKING CHOREOGRAPHER — rewind on violation, try different choices
// ══════════════════════════════════════════════════════════════════════════
let _rewindRequested = false;        // set by guard check when T19/T20 fails
let _rewindViolation = null;         // description of the violation that triggered rewind
let _searchTraversalLog = [];        // per-event log for search space analysis
let _searchEventCounter = 0;         // sequential event ID within a run
let _searchPathStack = [];           // chain of success fingerprints [fp_at_t0, fp_at_t1, ...]
let _searchParentNodeId = null;      // nodeId of current parent success event
let _searchLastCandidates = null;    // snapshot of PHASE 2 candidates before matching

// ── Sweep mode: sequential seeds with cross-seed fingerprint blacklist ──
let _sweepActive = false;            // true during multi-seed sweep
let _sweepSeedIdx = 0;               // current take number (0, 1, 2, ...)
let _sweepUsedSeeds = new Set();     // Set of seeds already tried (prevents reuse)
let _sweepBlacklist = new Map();     // Map<tick, Set<fingerprint>> — cross-seed dead states
let _sweepResults = [];              // per-seed summary results
let _sweepTotalBlacklisted = 0;      // running count of blacklisted fingerprints
let _sweepBlacklistHits = 0;         // how many times a blacklisted fingerprint was actually matched
let _sweepBlacklistHitsSeed = 0;     // hits for current seed only

// ── Bucketed blacklist IDB storage ──
let _blBucketSize = 64;              // ticks per bucket (aligned with cycle length)
let _blLoadedBuckets = new Set();    // Set<bucketIndex> — which buckets are in memory
let _blBucketCount = 0;              // total bucket count from IDB metadata
let _blBucketVersion = 0;            // 0 = legacy single-blob, 1 = bucketed

// ── Golden council: move traces from the top-performing seeds ──
// Council size scales with experience: min(10, max(1, floor(sqrt(totalSeeds))))
let _sweepGoldenCouncil = [];        // Array of {peak, seed, moves: Map<tick, Map<xonIdx, toNode>>}, sorted by peak desc
let _sweepGoldenHits = 0;            // times a candidate matched any council member
let _sweepGoldenHitsSeed = 0;        // golden hits for current seed
let _sweepSeedMoves = null;          // Map<tick, Map<xonIdx, toNode>> — current seed's move buffer

// ── Council replay: deterministic frame-by-frame replay of a council member ──
let _sweepReplayActive = false;      // true during council member replay
let _sweepReplayMember = null;       // council member being replayed {peak, seed, moves}
let _replayAncestorPeak = -1;        // ancestor's peak tick when extending a replay (-1 = fresh run)
let _councilReplayMode = false;      // true when replaying via snapshot playback (pause at end)
let _guardHardStop = false;        // when true, guards halt replay on failure instead of silent reset

function _goldenCouncilSize() {
    const topScore = _sweepGoldenCouncil.length > 0 ? _sweepGoldenCouncil[0].peak : 0;
    return Math.max(1, Math.floor(Math.sqrt(topScore / 100)));
}

function _fnv1aHash(str) {
    let h = 0x811c9dc5;
    for (let i = 0; i < str.length; i++) {
        h ^= str.charCodeAt(i);
        h = Math.imul(h, 0x01000193);
    }
    return (h >>> 0).toString(16).padStart(8, '0');
}
let _lastAutosavePeak = 0;          // last _maxTickReached at which autosave fired
const _BT_MAX_SNAPSHOTS = Infinity; // no cap — must be able to rewind all the way to t=0
const _BT_MAX_RETRIES = Infinity;   // no artificial cap — L2 lattice is inherently finite
let _btSnapshots = [];               // stack of state snapshots (one per tick) — source of truth for IDB saves
let _btColdBoundary = 0;             // index where cold (raw IDB) snapshots end and live snapshots begin
let _btColdSnapshots = [];           // raw IDB snapshots (already serialized) — immutable, used at save time
let _btRetryCount = 0;               // retries at current depth within a single demoTick() call
let _btActive = false;               // true while inside a backtrack retry loop

// ── BFS backtracker state (persists across demoTick() calls) ──
// When a tick fails, we exhaust all options at that tick (layer 0),
// then go one tick back (layer 1), try all rotations there, replay forward,
// then two ticks back (layer 2), etc. This is BFS over tick layers.
let _bfsFailTick = -1;               // the tick that originally failed (-1 = no active BFS)
let _bfsLayer = 0;                   // how many ticks back from _bfsFailTick we're exploring
let _bfsLayerRetries = 0;            // retries at the current BFS layer's anchor tick
const _BFS_MAX_LAYERS = Infinity;   // no artificial cap — can go all the way back to t=0

// ── Persistent bad-move ledger ──
// Key: tick number → Set of "xonIdx:destNode" strings.
// Accumulates across retries so the search space shrinks monotonically.
let _btBadMoveLedger = new Map();

// ── Deterministic matching enumeration ──
// Instead of PRNG-based retries, we enumerate ALL valid oct matchings
// for a tick and try them sequentially. Escalation only when provably exhausted.
let _btTriedFingerprints = new Map();   // tick → Set of canonical fingerprint strings
let _btMatchingCache = null;            // Array of all valid matchings for current tick (or null)
let _btMatchingIndex = 0;              // next matching to try from _btMatchingCache
let _btMatchingCacheLedgerSize = 0;    // ledger size when cache was built (invalidate on change)
let _btStaleRetries = 0;               // consecutive retries with no new fingerprint/exclusion

// ── Face assignment enumeration (greedy + ranked alternatives) ──
// During backtracking, enumerate ALL valid (xon, face, quarkType) assignments.
// Greedy (best score) is tried first; on retry, alternatives ranked by score.
let _btFaceAssignCache = null;         // Array of valid assignment combos [{xon, face, quarkType, score}[]]
let _btFaceAssignIndex = 0;            // next face assignment combo to try
let _btFaceAssignLedgerSize = 0;       // ledger size when cache was built

// ── Tunable choreography parameters (genome for GA tournament) ──
// All hardcoded magic numbers extracted here for parameterized optimization.
const _choreoParams = {
    // Movement genes
    lookahead: 12,              // PHASE 0 eviction foresight depth
    congestionMax: 4,           // oct cage xon count triggering idle_tet demotion
    octDeadEndPenalty: 10,      // PHASE 2: penalize 1-move dead ends
    // Face scoring genes (per spec §6: quark type → xonic balance → vacancy)
    faceOccupiedPenalty: 30,    // another xon already looping this face
    ratioDeficitWeight: 10,     // quark type ratio deficit bonus (applied at 10× hadronic weight)
    ratioThreshold: 0.05,       // min deficit gap to prefer secondary quark type
    assignmentThreshold: 5,     // minimum face score to attempt tet assignment
};
// Ranges for GA mutation (used by tournament engine in flux-tests.js)
// Float genes use [min, max, 'float'] to signal continuous mutation.
const _choreoParamRanges = {
    lookahead:              [2, 30],
    congestionMax:          [1, 8],
    octDeadEndPenalty:      [0, 50],
    faceOccupiedPenalty:    [0, 100],
    ratioDeficitWeight:     [0, 50],
    ratioThreshold:         [0.0, 0.3, 'float'],
    assignmentThreshold:    [0, 30],
};

// Loop topology → concrete node sequence, given tet cycle [a, b, c, d]
// a=octNode0, b=extNode, c=octNode1, d=octNode2
const LOOP_SEQUENCES = {
    pu1: ([a, b, c, d]) => [a, b, c, d, a],     // Hamiltonian CW (proton up 1)
    pu2: ([a, b, c, d]) => [a, d, c, b, a],     // Hamiltonian CCW (proton up 2)
    pd:  ([a, b, c, d]) => [a, b, c, b, a],     // Hook (proton down)
    nd1: ([a, b, c, d]) => [a, b, c, d, a],     // Hamiltonian CW (neutron down 1)
    nd2: ([a, b, c, d]) => [a, d, c, b, a],     // Hamiltonian CCW (neutron down 2)
    nu:  ([a, b, c, d]) => [a, b, a, c, a],     // Fork (neutron up)
};

const LOOP_TYPE_NAMES = { pu1: 'ham_cw', pu2: 'ham_ccw', pd: 'hook', nd1: 'ham_cw', nd2: 'ham_ccw', nu: 'fork' };

// All valid loop permutations per topology.
// Each topology maps to an array of generator functions.
// Input: cycle [a,b,c,d] where a=oct entry, b=ext, c=oct, d=oct.
const LOOP_PERMUTATIONS = {
    fork: [
        ([a,b,c,d]) => [a,b,a,c,a],
        ([a,b,c,d]) => [a,c,a,b,a],
        ([a,b,c,d]) => [a,b,a,d,a],
        ([a,b,c,d]) => [a,d,a,b,a],
        ([a,b,c,d]) => [a,c,a,d,a],
        ([a,b,c,d]) => [a,d,a,c,a],
    ],
    hook: [
        ([a,b,c,d]) => [a,b,c,b,a],
        ([a,b,c,d]) => [a,c,b,c,a],
        ([a,b,c,d]) => [a,b,d,b,a],
        ([a,b,c,d]) => [a,d,b,d,a],
        ([a,b,c,d]) => [a,c,d,c,a],
        ([a,b,c,d]) => [a,d,c,d,a],
    ],
    hamCW: [
        ([a,b,c,d]) => [a,b,c,d,a],
        ([a,b,c,d]) => [a,c,d,b,a],
        ([a,b,c,d]) => [a,d,b,c,a],
    ],
    hamCCW: [
        ([a,b,c,d]) => [a,d,c,b,a],
        ([a,b,c,d]) => [a,c,b,d,a],
        ([a,b,c,d]) => [a,b,d,c,a],
    ],
};

// Map quark type → topology name
const QUARK_TOPOLOGY = {
    pu1: 'hamCW', pu2: 'hamCCW', pd: 'hook',
    nd1: 'hamCW', nd2: 'hamCCW', nu: 'fork',
};

// Weak force escape color — overwritten by _recomputeColors()
let WEAK_FORCE_COLOR = 0x7f00ff;

const TRAIL_MAX = 1000; // single constant: max trail length (slider max + vertex buffer size)
const XON_TRAIL_LENGTH = TRAIL_MAX; // alias for vertex buffer allocation
const FJ_SUBS = 12; // fighterjet subdivisions per trail segment
const XON_TRAIL_VERTS = XON_TRAIL_LENGTH * FJ_SUBS + FJ_SUBS; // max vertices in FJ mode

// ── Weak Force Lifecycle Recorder ──
// Records up to 10 full lifecycles of weak force excitations for debugging.
// Each record: { xonIdx, entryTick, entryNode, exitTick, exitNode, exitReason, path }
const _weakLifecycleLog = [];
const _weakActiveTracking = new Map(); // xonIdx → { entryTick, entryNode, path }
const WEAK_LIFECYCLE_MAX = 10;

// ╔══════════════════════════════════════════════════════════════════════╗
// ║  Annihilation: 2 xons at same node → pair stored, both removed       ║
// ║  Creation: stored pair → 2 new xons on free adjacent oct nodes       ║
// ╚══════════════════════════════════════════════════════════════════════╝
let _gluonStoredPairs = 0;

const QUARK_COLORS = { pu1: 0x0040ff, pu2: 0x00ff40, pd: 0x00ffff, nd1: 0xffbf00, nd2: 0xff00bf, nu: 0xff0000 };

// ── Color Wheel System ──
// 8 roles at fixed 45° intervals; phase slider rotates all uniformly.
// Opposite roles (proton↔neutron counterparts) are always 180° apart.
let _colorPhaseShift = 0;
const COLOR_ROLE_OFFSETS = {
    gluon: 0, pu2: 45, pd: 90, pu1: 135,
    weak: 180, nd2: 225, nu: 270, nd1: 315,
};

// Port of getExactColor() from colors.html — maps angle → {r, g, b, hex}
function _hslToRgb(h, s, l) {
    l /= 100;
    const a = s * Math.min(l, 1 - l) / 100;
    const f = n => {
        const k = (n + h / 30) % 12;
        return l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
    };
    return [f(0) * 255, f(8) * 255, f(4) * 255];
}

function _getWheelColor(angle) {
    let h = ((angle % 360) + 360) % 360;
    let r, g, b;
    if (h >= 180) {
        const rgb = _hslToRgb(h - 180, 100, 50);
        r = 255 - Math.round(rgb[0]);
        g = 255 - Math.round(rgb[1]);
        b = 255 - Math.round(rgb[2]);
    } else {
        const rgb = _hslToRgb(h, 100, 50);
        r = Math.round(rgb[0]);
        g = Math.round(rgb[1]);
        b = Math.round(rgb[2]);
    }
    return (r << 16) | (g << 8) | b;
}

// Resolve a role key to the CURRENT phase color
function _roleToColor(role) {
    if (role === 'gluon') return GLUON_COLOR;
    if (role === 'weak') return WEAK_FORCE_COLOR;
    if (role === 'oct') return 0xffffff;
    return QUARK_COLORS[role] !== undefined ? QUARK_COLORS[role] : 0xffffff;
}

// Get the role key for a xon's current mode
function _xonRole(xon) {
    if (xon._mode === 'tet' || xon._mode === 'idle_tet') return xon._quarkType || 'oct';
    if (xon._mode === 'gluon') return 'gluon';
    if (xon._mode === 'weak') return 'weak';
    return 'oct';
}

function _recomputeColors(phase) {
    _colorPhaseShift = phase;

    // Update all color constants from wheel
    QUARK_COLORS.nu  = _getWheelColor(COLOR_ROLE_OFFSETS.nu + phase);
    QUARK_COLORS.nd1 = _getWheelColor(COLOR_ROLE_OFFSETS.nd1 + phase);
    QUARK_COLORS.nd2 = _getWheelColor(COLOR_ROLE_OFFSETS.nd2 + phase);
    QUARK_COLORS.pd  = _getWheelColor(COLOR_ROLE_OFFSETS.pd + phase);
    QUARK_COLORS.pu1 = _getWheelColor(COLOR_ROLE_OFFSETS.pu1 + phase);
    QUARK_COLORS.pu2 = _getWheelColor(COLOR_ROLE_OFFSETS.pu2 + phase);
    GLUON_COLOR      = _getWheelColor(COLOR_ROLE_OFFSETS.gluon + phase);
    WEAK_FORCE_COLOR = _getWheelColor(COLOR_ROLE_OFFSETS.weak + phase);

    // Update live xon spark colors — trail colors are derived at render time from entry.role
    if (typeof _demoXons !== 'undefined' && _demoXons) {
        for (const xon of _demoXons) {
            if (!xon || !xon.alive) continue;
            xon.col = _roleToColor(_xonRole(xon));
            if (xon.sparkMat) xon.sparkMat.color.setHex(xon.col);
        }
    }
    // Refresh shapes (tet void colors) to match new phase
    if (typeof _applyTetColoring === 'function' && typeof _nucleusTetFaceData !== 'undefined' && _nucleusTetFaceData) _applyTetColoring(false);
    // Refresh legend + stats + xon panels
    if (typeof _updateLegend === 'function') _updateLegend();
    if (typeof updateDemoPanel === 'function') updateDemoPanel();
    if (typeof updateXonPanel === 'function') updateXonPanel();
}

// Initialize colors from wheel at default phase
_recomputeColors(0);

const A_SET = new Set([1, 3, 6, 8]);

const L1_VALID_TRIPLES = [
    [3, 5, 6], [1, 6, 7], [3, 5, 8], [1, 7, 8],  // 2A+1B
    [4, 5, 6], [4, 6, 7], [2, 5, 8], [2, 7, 8],  // 1A+2B
];

let _tickInProgress = false; // guard against overlapping async ticks
// ─── Profiling ───
let _tickTotalMs = 0, _tickCount = 0, _tickMaxMs = 0;
let _profPhases = { wb: 0, p0: 0, p05: 0, p1: 0, p2: 0, p3: 0, p3b: 0, p4: 0, p5: 0, solver: 0, cleanup: 0, render: 0, guards: 0 };

let _demoUncappedId = null;  // setTimeout chain for uncapped mode

// ── Precomputed pattern schedule for algos ──
let _activePatternSchedule = null;

// ════════════════════════════════════════════════════════════════════
// QUARK_ALGO_REGISTRY — pluggable movement strategies
// ════════════════════════════════════════════════════════════════════
const QUARK_ALGO_REGISTRY = [];

// ── Network naming for tournament/training ──
const _NETWORK_NAMES = [
    'Helios','Andromeda','Quasar','Pulsar','Cassini','Kepler','Orion','Vega',
    'Sirius','Nova','Nebula','Zenith','Eclipse','Photon','Boson','Fermion',
    'Tachyon','Planck','Hawking','Dirac','Faraday','Tesla','Maxwell','Euler',
    'Gauss','Riemann','Lagrange','Lorentz','Noether','Feynman','Curie','Hubble',
    'Sagan','Cosmos','Horizon','Meridian','Polaris','Aether','Prism','Singularity',
    'Vertex','Lattice','Flux','Radiance','Lumina','Solstice','Equinox','Aurora',
    'Nimbus','Helix'
];
let _networkName = null;   // current network identity (assigned at training start)
let _networkGen = 0;       // generation counter within current training run
let _networkLeaderboard = []; // [{name, gen, fitness, cv, guardFailures, actualizationRate, avgReward}]

// ── Opening phase flag ──
let _openingPhase = false;
// ── Matter/antimatter winding direction ──
// Set during opening tick 1 merry-go-round. 'CW' = matter (follows _octEquatorCycle order).
// Once set, must remain constant for the entire simulation (guarded by T81).
let _octWindingDirection = null;
