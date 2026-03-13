// flux-demo-state.js — Shared state, constants, and PRNG for the demo choreography system

// ── Core demo state ──────────────────────────────────────────────────────
let _demoActive = false;
let _demoInterval = null;
let _demoPaused = false;  // true when user has paused via pause button
let _demoReversing = false;    // true during reverse playback
let _reverseInterval = null;   // setInterval ID for reverse playback
let _tickLog = [];             // lightweight per-tick log for export
let _tickLogLastGuards = {};   // last full guard state for delta encoding
let _redoStack = [];           // snapshots saved during rewind for instant step-forward
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

// ── Edge Balance System — per-edge quark traversal counters ──
// Tracks how evenly all 6 quark types traverse each edge connected to the oct.
// "White" = perfectly balanced (all 6 types equal). Tinted = imbalanced.
// _octEdgeSet: Set of pairId strings for all edges with ≥1 oct endpoint
// _edgeBalance: Map<pairId, {pu1,pu2,pd,nd1,nd2,nu,total}>
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
            _edgeBalance.set(pid, { pu1: 0, pu2: 0, pd: 0, nd1: 0, nd2: 0, nu: 0, total: 0 });
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
function _recordEdgeTraversal(fromNode, toNode, quarkType) {
    if (!_edgeBalance || !quarkType) return;
    const pid = pairId(fromNode, toNode);
    const entry = _edgeBalance.get(pid);
    if (!entry) return; // not an oct-adjacent edge
    if (entry[quarkType] !== undefined) {
        entry[quarkType]++;
        entry.total++;
    }
}

// Compute scalar edge-evenness score [0,1] — 1.0 = all 6 quark types visit each edge equally
function _computeEdgeEvenness() {
    if (!_edgeBalance || _edgeBalance.size === 0) return 0;
    const types = ['pu1', 'pu2', 'pd', 'nd1', 'nd2', 'nu'];
    let totalScore = 0;
    let edgesWithData = 0;
    for (const [pid, counts] of _edgeBalance) {
        if (counts.total === 0) continue;
        edgesWithData++;
        // Perfect balance = each type is 1/6 of total
        const ideal = counts.total / 6;
        let deviation = 0;
        for (const t of types) deviation += Math.abs(counts[t] - ideal);
        // Normalize: max deviation is when all traversals are one type = 5/6 * total * 2
        // But simpler: deviation / (2 * total * (1 - 1/6)) = deviation / (5/3 * total)
        // Score = 1 - normalized deviation
        const maxDev = counts.total * (10 / 6); // worst case: all one type
        totalScore += 1 - (deviation / maxDev);
    }
    return edgesWithData > 0 ? totalScore / edgesWithData : 0;
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
let _demoPrevFaces = new Set();   // faces active in previous window (for relinquishing)
let _idleTetManifested = false;   // set by _startIdleTetLoop when new SCs are materialised
const T79_MAX_FULL_TICKS = 1;     // T79: max consecutive ticks allowed with all 6 xons on oct nodes
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

// ── Seeded PRNG for deterministic backtracker replay ─────────────────
// Mulberry32: fast 32-bit seeded PRNG. Returns float in [0, 1).
// The backtracker requires deterministic forward replay from the same
// snapshot + exclusions. Math.random() breaks this because it's unseeded.
// All choreography randomness MUST use _sRng() instead of Math.random().
let _sRngState = 0;
function _sRng() {
    _sRngState |= 0;
    _sRngState = (_sRngState + 0x6D2B79F5) | 0;
    let t = Math.imul(_sRngState ^ (_sRngState >>> 15), 1 | _sRngState);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
// Seed from tick number — call at the start of each tick so replays
// from the same snapshot produce the same random sequence.
function _sRngSeed(tick) { _sRngState = tick * 2654435761; }
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
const GLUON_COLOR = 0x7f00ff;   // violet — cage maintenance mode (formerly weak color)

// ── SC attribution ──
const _scAttribution = new Map();

// ══════════════════════════════════════════════════════════════════════════
// BACKTRACKING CHOREOGRAPHER — rewind on violation, try different choices
// ══════════════════════════════════════════════════════════════════════════
let _rewindRequested = false;        // set by guard check when T19/T20 fails
let _rewindViolation = null;         // description of the violation that triggered rewind
const _BT_MAX_SNAPSHOTS = Infinity; // no cap — must be able to rewind all the way to t=0
const _BT_MAX_RETRIES = Infinity;   // no artificial cap — L2 lattice is inherently finite
let _btSnapshots = [];               // stack of state snapshots (one per tick)
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
    pu1: ([a, b, c, d]) => [a, b, a, c, a],     // Fork (proton up 1)
    pu2: ([a, b, c, d]) => [a, b, c, b, a],     // Hook (proton up 2)
    pd:  ([a, b, c, d]) => [a, b, c, d, a],     // Hamiltonian CW (proton down)
    nd1: ([a, b, c, d]) => [a, b, a, c, a],     // Fork (neutron down 1)
    nd2: ([a, b, c, d]) => [a, b, c, b, a],     // Hook (neutron down 2)
    nu:  ([a, b, c, d]) => [a, d, c, b, a],     // Hamiltonian CCW (neutron up)
};

const LOOP_TYPE_NAMES = { pu1: 'fork', pu2: 'hook', pd: 'ham_cw', nd1: 'fork', nd2: 'hook', nu: 'ham_ccw' };

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
    pu1: 'fork', pu2: 'hook', pd: 'hamCW',
    nd1: 'fork', nd2: 'hook', nu: 'hamCCW',
};

// Weak force escape color — purple/magenta, distinct from all quark + oct colors.
// Used when a xon breaks confinement and enters the 'weak' mode.
const WEAK_FORCE_COLOR = 0x080808; // near-black — renders as black against grey (#333) scene background

const XON_TRAIL_LENGTH = 250;

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
