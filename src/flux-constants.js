// flux-constants.js — Global constants, state declarations, geometry
'use strict';

// ─── Single-cell constants ────────────────────────────────────────────────────
const S3 = Math.sqrt(3), r4 = 2 / S3, r3 = 1 / S3, SHORT_D = 2 / S3;
const UNIT_REST = [
    [0,0,0],
    [r4,0,0],[-r4,0,0],[0,r4,0],[0,-r4,0],[0,0,r4],[0,0,-r4],
    [r3,r3,r3],[r3,r3,-r3],[r3,-r3,r3],[r3,-r3,-r3],
    [-r3,r3,r3],[-r3,r3,-r3],[-r3,-r3,r3],[-r3,-r3,-r3],
];
function vd(a,b){ const dx=a[0]-b[0],dy=a[1]-b[1],dz=a[2]-b[2]; return Math.sqrt(dx*dx+dy*dy+dz*dz); }
const LATTICE_OFFSETS = [
    [2,2,0],[2,-2,0],[-2,2,0],[-2,-2,0],
    [2,0,2],[2,0,-2],[-2,0,2],[-2,0,-2],
    [0,2,2],[0,2,-2],[0,-2,2],[0,-2,-2]
].map(v=>[v[0]*r3,v[1]*r3,v[2]*r3]);
// Shortcut colors: RMS blend of their two parent base-direction colors
//   s1=v1⊕v4  s2=v1⊕v3  s3=v1⊕v2  s4=v2⊕v3  s5=v2⊕v4  s6=v3⊕v4
const SC_BASE_DIRS = {1:[0,3], 2:[0,2], 3:[0,1], 4:[1,2], 5:[1,3], 6:[2,3]};
const S_COLOR   = {1:0xC69BC2,2:0xC3AC74,3:0xFF8D5D,4:0xC3C467,5:0xC6B5BA,6:0x6FCEC7};
const S_COLOR_CSS = {1:'#c69bc2',2:'#c3ac74',3:'#ff8d5d',4:'#c3c467',5:'#c6b5ba',6:'#6fcec7'};

// ─── All mutable state — declared early to prevent TDZ errors ─────────────────
// Any let/const used before its natural source position must live here.
let latticeLevel = 2;
let simHalted = false;
let selectMode = false;
let stateVersion = 0;
let sidePanelVersion = -1;
let candidateCacheKey = '';
const blockedImplied = new Set();


// Void sphere mesh handles — hoisted to avoid TDZ before first rebuildVoidSpheres() call
let _voidMeshPool = []; // {fillMesh, wireMesh, type} per void
// (_forceActualizedVoids removed — all octs render unconditionally as gray)

// ╔══════════════════════════════════════════════════════════════════════╗
// ║  DO NOT DELETE — FLUX DYNAMICS CANONICAL MECHANICS REFERENCE       ║
// ╠══════════════════════════════════════════════════════════════════════╣
// ║                                                                    ║
// ║  ┌─────────────────────────────────────────────────────────────┐   ║
// ║  │  1. LATTICE GEOMETRY                                       │   ║
// ║  └─────────────────────────────────────────────────────────────┘   ║
// ║  Octahedron at center of FCC lattice. 8 tet faces form K₄,₄       ║
// ║  bipartite structure (cube dual / face-at-a-point coloring):      ║
// ║    Group A (proton): faces 1(0,7,9) 3(0,13,11) 6(5,9,13)         ║
// ║                            8(5,11,7)                              ║
// ║    Group B (neutron): faces 2(0,9,13) 4(0,11,7) 5(5,7,9)         ║
// ║                             7(5,13,11)                            ║
// ║  Oct SC slots (4 edges forming the cage square):                  ║
// ║    SC 39 → tets {1,5}   SC 48 → tets {2,6}                       ║
// ║    SC 58 → tets {3,7}   SC 37 → tets {4,8}                       ║
// ║  Each tet has 2 SCs: 1 shared with oct (free) + 1 unique.         ║
// ║  Dense sphere packing (74.048%) is the lattice's heart.           ║
// ║                                                                    ║
// ║  ┌─────────────────────────────────────────────────────────────┐   ║
// ║  │  2. L1 SOLVER CONSTRAINTS (verified)                       │   ║
// ║  └─────────────────────────────────────────────────────────────┘   ║
// ║  Max simultaneous tets in L1:                                     ║
// ║    1 tet:  8 valid configs (any single tet)                        ║
// ║    2 tets: 20 valid configs (20 of 28 pairs)                      ║
// ║    3 tets: 8 valid configs (all have 2:1 group split)             ║
// ║    4 tets: 1 valid config ({2,3,5,8} only)                        ║
// ║    5+ tets: 0 (impossible)                                        ║
// ║  Forbidden pairs (never coexist):                                 ║
// ║    Same oct slot: (1,5) (2,6) (3,7) (4,8)                        ║
// ║    Same group adj: (1,3) (2,4) (5,7) (6,8)                       ║
// ║  Valid triples (the 8 configs):                                   ║
// ║    2A+1B: {3,5,6} {1,6,7} {3,5,8} {1,7,8}                       ║
// ║    1A+2B: {4,5,6} {4,6,7} {2,5,8} {2,7,8}                       ║
// ║  Note: 2:1 group split maps to hadron structure (uud / udd).     ║
// ║                                                                    ║
// ║  ┌─────────────────────────────────────────────────────────────┐   ║
// ║  │  3. XON TRAVERSAL TOPOLOGIES — PARTICLE IDENTITY           │   ║
// ║  └─────────────────────────────────────────────────────────────┘   ║
// ║  Each tet is K₄ (4 nodes, 6 edges). A 4-move closed loop on K₄   ║
// ║  has 21 instances but only 4 topologically distinct shapes        ║
// ║  (pendulum excluded as degenerate 1D):                            ║
// ║                                                                    ║
// ║  Shape         Example         Nodes  Quark Type                  ║
// ║  ─────────────────────────────────────────────────                 ║
// ║  Fork          a→b→a→c→a       3      p-up (proton majority)      ║
// ║  Lollipop      a→b→c→b→a       3      n-down (neutron majority)   ║
// ║  Hamilton CW   a→b→c→d→a       4      p-down (proton anchor)      ║
// ║  Hamilton CCW  a→d→c→b→a       4      n-up (neutron anchor)       ║
// ║                                                                    ║
// ║  The oct face is the BASIS that orients the tet traversal.         ║
// ║  3 base nodes (on oct face) + 1 apex node = oriented tet.         ║
// ║  Without a basis (no oct face binding), traversal topology is      ║
// ║  indistinguishable → this is the ELECTRON (flavorless).           ║
// ║  One complete 4-step loop = one hadron actualization.              ║
// ║                                                                    ║
// ║  Fork/hook assignment is a gauge choice (isospin conjugates).  ║
// ║  Pick whichever makes physical sense for xon movement, as long    ║
// ║  as the distribution is balanced.                                 ║
// ║                                                                    ║
// ║  ┌─────────────────────────────────────────────────────────────┐   ║
// ║  │  4. PARTICLE ZOO                                           │   ║
// ║  └─────────────────────────────────────────────────────────────┘   ║
// ║  Particle     Traversal          Basis      Identity              ║
// ║  ────────────────────────────────────────────────────              ║
// ║  p-up (×2)    Fork               oct face   oriented              ║
// ║  p-down (×1)  Hamiltonian CW     oct face   oriented + chiral     ║
// ║  n-up (×1)    Hamiltonian CCW    oct face   oriented + chiral     ║
// ║  n-down (×2)  Lollipop           oct face   oriented              ║
// ║  gluon        oct traversal      oct cage   anonymous, any length ║
// ║  electron     any loop           none       unoriented=flavorless ║
// ║                                                                    ║
// ║  ┌─────────────────────────────────────────────────────────────┐   ║
// ║  │  5. HADRON CHOREOGRAPHY RULES                              │   ║
// ║  └─────────────────────────────────────────────────────────────┘   ║
// ║  Proton (uud): 2 up + 1 down on Group A faces                    ║
// ║  Neutron (udd): 1 up + 2 down on Group B faces                   ║
// ║  Anti-phase: proton on A when neutron on B (alternating)          ║
// ║  Pauli: no two same-type quarks on same face per tick             ║
// ║  Max spread: all 3 quarks of each hadron on different faces       ║
// ║  Full coverage: every face visited equally over complete cycles   ║
// ║  Bosonic cage: 4 oct SCs must be materialized (maintained by      ║
// ║    gluon xons — live arbitrarily long, traverse oct freely,      ║
// ║    but cannot leave the octahedron)                               ║
// ║                                                                    ║
// ║  ┌─────────────────────────────────────────────────────────────┐   ║
// ║  │  6. SYSTEM ARCHITECTURE                                    │   ║
// ║  └─────────────────────────────────────────────────────────────┘   ║
// ║  Pattern machine: computes ideal activation schedules (what       ║
// ║    SHOULD happen for perfect coverage). Feeds tournament algos.   ║
// ║  Tournament algos: try to approximate pattern machine output      ║
// ║    using legal lattice moves in 3D (what CAN happen).            ║
// ║  Each tet needs 4 ticks to actualize (one full xon loop).        ║
// ║  Gluons can be created at will to maintain the cage.             ║
// ║  Stochastic type assignment ensures flavor democracy over time.   ║
// ║                                                                    ║
// ╚══════════════════════════════════════════════════════════════════════╝

// All 8 tet definitions: {face, octNodes, group}
// Actual node arrays (with external node) populated at runtime in simulateNucleus()
const DEUTERON_TET_FACES = {
    // Group A (proton)
    1: { octNodes: [0,7,9],   group:'A' },
    3: { octNodes: [0,13,11], group:'A' },
    6: { octNodes: [5,9,13],  group:'A' },
    8: { octNodes: [5,11,7],  group:'A' },
    // Group B (neutron)
    2: { octNodes: [0,9,13],  group:'B' },
    4: { octNodes: [0,11,7],  group:'B' },
    5: { octNodes: [5,7,9],   group:'B' },
    7: { octNodes: [5,13,11], group:'B' },
};

// XON DEFINITIONS — anonymous excitation workers (like gluons).
// The quarks ARE the tets; xons just actualize SC patterns.
// Xons are WHITE — they carry no quark identity.
const XON_COLOR = 0xffffff;
const DEUTERON_XONS = [
    { id: 'x0', group: 'a', startFace: 1, color: XON_COLOR },
    { id: 'x1', group: 'a', startFace: 3, color: XON_COLOR },
    { id: 'x2', group: 'a', startFace: 8, color: XON_COLOR },
    { id: 'x3', group: 'b', startFace: 2, color: XON_COLOR },
    { id: 'x4', group: 'b', startFace: 5, color: XON_COLOR },
    { id: 'x5', group: 'b', startFace: 7, color: XON_COLOR },
];

// TET QUARK COLORS — the quarks ARE the tets, colored by type.
// Proton (uud) on Group A {1,3,6,8}: up=yellow, down=green
// Neutron (udd) on Group B {2,4,5,7}: up=blue, down=red
const TET_QUARK_COLORS = {
    1: 0xffdd44,  // proton up (yellow)
    3: 0xffdd44,  // proton up (yellow)
    6: 0x44cc66,  // proton down (green)
    8: 0x44cc66,  // proton down (green)
    2: 0x4488ff,  // neutron up (blue)
    4: 0x4488ff,  // neutron up (blue)
    5: 0xff4444,  // neutron down (red)
    7: 0xff4444,  // neutron down (red)
};

// Both groups can hop to ANY of the 8 tet faces.
// The A/B split is just initial seeding; coverage spans the whole octa.
const DEUTERON_HOP_GROUPS = {
    a: [1, 2, 3, 4, 5, 6, 7, 8],
    b: [1, 2, 3, 4, 5, 6, 7, 8],
};

const QUARK_TRAIL_LENGTH = 4;      // exactly 1 full tet cycle → closed loop "string"
let _activeQuarkAlgo = 0;          // index into QUARK_ALGO_REGISTRY

// ════════════════════════════════════════════════════════════════════
// ACTIVATION PATTERN ANALYSIS
// ════════════════════════════════════════════════════════════════════
//
// KEY INSIGHT: The quarks ARE the tets, not the excitations.
// Excitations are anonymous workers (like gluons) whose job is to
// materialize SCs to actualize the target activation pattern.
//
// The octahedron's 8 face-tets form a BIPARTITE graph (K₄,₄):
//   Group A = {1,3,6,8}  ←→  Group B = {2,4,5,7}
//   Every A face is adjacent ONLY to B faces and vice versa.
//   This bipartite division demonstrates proton-neutron complementarity.
//
// Activation patterns specify which tets are "active" (fully actualized)
// at each tick. Active = all SCs of the tet are open.
//
// Anti-phase complementarity:
//   Even ticks: proton activations on A, neutron activations on B
//   Odd  ticks: proton activations on B, neutron activations on A
//   → Maximum spatial distribution across the octahedron.
//
// Each hadron activates 3 of 4 group-faces per half-tick:
//   3 active out of 4 → 1 tet empty at each tick
//   Over an 8-tick cycle, each tet is active 6 times, empty 2 times
//   The empty-tet rotation has D(4) = 9 valid derangement phasings.
//
// Excitations are dispatched anonymously to fulfill the target pattern.
// If the lattice drifts from target, a nearby valid pattern guides recovery.
//
// Full pattern = (A-phase, B-phase) × anchor compatibility
// ════════════════════════════════════════════════════════════════════

// All derangements of [0,1,2,3] — permutations with no fixed points
const _DERANGEMENTS_4 = (function() {
    const result = [];
    function gen(chosen) {
        if (chosen.length === 4) {
            if (chosen.every((v, i) => v !== i)) result.push([...chosen]);
            return;
        }
        for (let v = 0; v < 4; v++) {
            if (chosen.includes(v)) continue;
            chosen.push(v);
            gen(chosen);
            chosen.pop();
        }
    }
    gen([]);
    return result;
})(); // D(4) = 9

// All permutations of [0,1,2,3]
const _PERMS_4 = (function() {
    const result = [];
    function gen(chosen) {
        if (chosen.length === 4) { result.push([...chosen]); return; }
        for (let v = 0; v < 4; v++) {
            if (chosen.includes(v)) continue;
            chosen.push(v);
            gen(chosen);
            chosen.pop();
        }
    }
    gen([]);
    return result;
})(); // 4! = 24

/**
 * Compute all valid "activation patterns" for the deuteron.
 *
 * A pattern specifies, for each hadron, the face-assignment schedule
 * of its 3 quarks over an 8-tick cycle (4 ticks on A, 4 on B).
 *
 * Returns: { patterns, summary }
 *   patterns[]: { aDerang, bDerang, anchorsA, anchorsB, spread }
 *   summary: human-readable analysis string
 */
