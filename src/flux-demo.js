// flux-demo.js — Demo mode: pattern computation, xon management, demo loop
//
// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║  CHOREOGRAPHER DESIGN INTENT — Authoritative Reference                      ║
// ║  This comment IS the spec. Code that contradicts it is wrong.               ║
// ╚══════════════════════════════════════════════════════════════════════════════╝
//
// ┌─────────────────────────────────────────────────────────────────────────────┐
// │ 1. OVERVIEW                                                                 │
// └─────────────────────────────────────────────────────────────────────────────┘
//
// The choreographer advances a deuteron simulation one Planck second per tick.
// Six anonymous fermionic workers ("xons") traverse an FCC lattice, executing
// hadronic excitation loops on tetrahedral faces surrounding an octahedral cage.
// The PBD constraint solver ("the vacuum") has absolute authority over all
// shortcut activations. The backtracker provides exhaustive Monte Carlo DFS
// search over choreography solutions — the only true failure is backing all
// the way to t=0, which proves the rules are mathematically impossible.
//
// ┌─────────────────────────────────────────────────────────────────────────────┐
// │ 2. XON MODES                                                                │
// └─────────────────────────────────────────────────────────────────────────────┘
//
//   oct       (white)   Cruising the octahedral cage. Participates in bipartite
//                        matching for destination assignment. May be assigned to
//                        a tet face or designated as a gluon.
//
//   tet       (quark)   Executing a scheduled hadronic excitation loop on an
//                        assigned face. Follows a fixed 5-node sequence (fork,
//                        hook, ham CW, or ham CCW). First-class priority:
//                        the planner defers to tet xons' paths.
//
//   idle_tet  (quark)   Executing an unscheduled loop on an actualized face.
//                        Expendable: evicted to make room for tet or oct xons.
//
//   gluon     (orange)  Cage maintenance. Assigned when the PBD solver indicates
//                        the cage would break without this xon's action this tick.
//                        Participates in bipartite matching (must still move).
//                        Reverts to oct at the start of the next tick. Gluon
//                        assignment can chain across consecutive ticks if needed.
//
//   weak      (purple)  Ejected from confinement. Cannot enter oct nodes or any
//                        true geometric tet node. Must first move to "ejection
//                        space" (1+ moves), which flips _mayReturn = true. Then
//                        its top priority is returning to the oct cage (still
//                        avoiding non-oct tet nodes). Transitions to oct on
//                        arrival at an oct node when _mayReturn && oct has vacancy.
//
// ┌─────────────────────────────────────────────────────────────────────────────┐
// │ 3. NODE CLASSIFICATION                                                      │
// └─────────────────────────────────────────────────────────────────────────────┘
//
//   _octNodeSet          6 nodes of the octahedral cage.
//
//   _actualizedTetNodes  Every node on any CURRENTLY ACTUALIZED tetrahedron —
//                        i.e. tets whose ALL bounding SCs are active. Recomputed
//                        each tick. Only geometrically precise tets count.
//
//   _nucleusFaceNodes    Union of _nucleusTetFaceData[1..8].allNodes. Includes
//                        oct nodes + 8 ext (apex) nodes.
//
//   _ejectionForbidden   _octNodeSet ∪ _actualizedTetNodes (dynamic).
//                        Weak xons may NOT occupy these nodes. Non-actualized
//                        tet nodes are valid transit space for weak xons.
//
//   _purelyTetNodes      _actualizedTetNodes \ _octNodeSet. Actualized tet nodes
//                        that are not oct nodes.
//
// ┌─────────────────────────────────────────────────────────────────────────────┐
// │ 4. TICK PIPELINE                                                            │
// └─────────────────────────────────────────────────────────────────────────────┘
//
// Each tick is wrapped in the backtracker retry loop. On guard failure, the
// tick is rewound and retried with the offending move excluded.
//
//   PER-TICK SETUP
//     - Clear _movedThisTick, _evictedThisTick on all xons.
//     - Clear _moveRecord, _moveTrace.
//     - Snapshot live guard state (_liveGuardSnapshot).
//     - Revert any gluon xons to oct mode (fresh evaluation each tick).
//     - Build occupied map.
//
//   OPENING PHASE (ticks 0–2)
//     Tick 0→1: 4 xons move to equatorial (square) positions via base edges.
//               2 free xons move stochastically with xonic movement balance.
//               Free xons may NOT use shortcut directions (protects vacuum
//               state for cage formation). Base edges only.
//     Tick 1→2: Continue spreading. No cage yet.
//     Tick 2:   Discover octahedron, set up all data structures.
//               Normal phases begin at tick 3.
//
//   PHASE 0 — PRE-CHECK TET/IDLE_TET XONS
//     For each tet/idle_tet xon:
//     a) T60 check: are ALL face SCs still active? If not → weak mode,
//        _t60Ejected = true, relinquish face SCs. Movement deferred to Phase 0.5.
//     b) Destination blocked by idle_tet → evict the BLOCKER (expendable).
//        Blocked by tet → the blocked xon defers (tet has first-class priority).
//        Must find alternative or trigger backtrack.
//     c) Loop-shape-aware lookahead: simulate remaining path while projecting
//        all other tet xons forward. Dead end → evict to weak + _t60Ejected.
//     Eviction is NEVER _returnXonToOct. It is ALWAYS weak + _t60Ejected.
//
//   PHASE 0.5 — WEAK FORCE RETURN
//     Handles ALL weak xon movement (including _t60Ejected).
//     Pre-_mayReturn: move to ejection-space nodes only (_isValidEjectionTarget).
//       After 1+ valid ejection moves, flip _mayReturn = true.
//     Post-_mayReturn: top priority is returning to oct cage. BFS toward nearest
//       oct node. May enter oct nodes and ejection-space nodes. May NOT enter
//       _purelyTetNodes. On arrival at oct node → transition to oct mode.
//     No-prevNode rule applies (absolute for weak mode).
//
//   PHASE 1 — PLAN TET/IDLE_TET MOVES
//     For each tet/idle_tet xon, plan the next hop in its loop sequence.
//     Destination free + lookahead passes → approve, reserve node.
//     Destination has oct xon → tentatively approve, mark for forced vacate.
//     Destination has idle_tet → evict blocker (weak + _t60Ejected).
//     Destination has tet → defer (tet priority). Find alternative or backtrack.
//     Vacuum negotiation for SC-only edges:
//       canMaterialiseQuick → BFS single severances → BFS 2-severance combos → reject.
//       Each severance incurs a penalty tracked by _severanceCount.
//
//   PHASE 1.5 — NATURAL LOOP COMPLETION
//     For each tet/idle_tet with _loopStep >= 4:
//     Actualization check uses accumulated _tetActualized flag (true if all
//     face SCs were active during at least one tick of the loop — NOT re-checked
//     at completion). If not actualized → weak + _t60Ejected.
//     Otherwise → revert to oct, relinquish face SCs (respecting traversal lock).
//
//   PHASE 2a — DEMAND-DRIVEN FACE SELECTION
//     Before scoring, check each oct xon: does the PBD solver indicate the cage
//     would break without this xon's action? If yes → assign as gluon (orange),
//     skip face scoring. Gluons participate in Phase 2 matching.
//     For remaining oct xons, score all 8 faces. Priority order:
//       1. Quark type selection (proton/neutron ratio deficit from _ratioTracker)
//       2. Xonic movement balance (which directions would this loop traverse?
//          Prefer loops that use the xon's least-used directions)
//       3. Vacancy (is another xon already looping this face?)
//     Reachability is NOT a separate factor. Anti-phase balance is subsumed
//     by xonic movement balance.
//     Vacuum feasibility: all face SCs must be materializable.
//     Lookahead viability: loop sequence must pass _lookaheadTetPath.
//     If all pass → _assignXonToTet: promote face SCs, build loop, set tet mode.
//
//   PHASE 2 — COORDINATED OCT MOVEMENT (Bipartite Matching)
//     Gather oct/weak/gluon xons not yet moved.
//     Generate candidates via _getOctCandidates:
//       - Weak: filtered by _mayReturn rules (see Phase 0.5).
//       - Oct/gluon on cage: base + SC neighbors on oct cage, excluding
//         antipodal. Scored by xonic movement balance (direction deficit).
//     Vacuum pre-filter: canMaterialiseQuick for SC candidates.
//     Backtrack exclusion filter: remove candidates in bad-move ledger.
//     Kuhn's algorithm (maximum bipartite matching): most-constrained first.
//     Collision avoidance for unassigned xons:
//       Strategy 1: idle_tet diversion (productive — manifests a hadron).
//       Strategy 2: eject as weak particle (_t60Ejected = true).
//     No T55 enforcement here — T55 is a live guard; backtracker handles it.
//     No 2-step verification — backtracker handles downstream violations.
//
//   PHASE 3 — EXECUTE ALL MOVES (single pass)
//     All moves execute in one pass (no oct-first/tet-second ordering).
//     For oct/gluon moves: _executeOctMove with anti-bounce, SC re-check,
//       vacuum negotiation. If vacuum rejects → _rewindRequested = true.
//     For tet/idle_tet moves: final Pauli check, _advanceXon.
//     If any move fails → _rewindRequested = true (backtracker handles it).
//     Post-execution: Pauli check on all nodes. If 2+ xons on any node →
//       _rewindRequested = true.
//     No escape hatch. No Phase 3b. No Phase 3.5. No Phase 4. No safety net.
//     The backtracker is the universal safety net.
//
//   POST-PHASE PROCESSING
//     - Advance gluons (visual sprites along oct edges).
//     - Run PBD solver if _solverNeeded (rebuild impliedSet, apply positions).
//     - Kepler density check (deviation > 0.01% → halt with red screen).
//     - SC cleanup: remove ONLY non-unit-length SCs from xonImpliedSet.
//       All unit-length SCs remain as traversal paths and severance options.
//       SCs not traversed on the previous turn are severance candidates
//       (unless load-bearing per PBD solver).
//     - Decay dying xon trails.
//     - Color tet voids (only on actualized loop completion).
//     - Increment _demoTick.
//     - Live guard checks → if failed, _rewindRequested triggers backtracker.
//
// ┌─────────────────────────────────────────────────────────────────────────────┐
// │ 5. BACKTRACKER                                                              │
// └─────────────────────────────────────────────────────────────────────────────┘
//
// The backtracker wraps every tick in a retry loop. It is an exhaustive Monte
// Carlo DFS for choreography solutions.
//
//   - Full state snapshot before each tick (xon positions, modes, trails, SC
//     sets, attributions, solver positions).
//   - On guard failure: extract exclusions (which xon→node caused it), record
//     in persistent bad-move ledger, restore snapshot, retry with exclusion.
//   - BFS over tick layers: exhaust all candidate rotations at failing tick
//     (layer 0), then rewind 1 tick (layer 1), then 2 ticks, etc.
//   - NO CAPS: _BT_TOTAL_CAP = Infinity, _BFS_MAX_LAYERS = Infinity,
//     _BT_MAX_RETRIES = Infinity. L2 lattice is inherently finite.
//   - Event-loop yield every 32 retries (prevents browser freeze, does not
//     limit retries).
//   - The ONLY failure condition: BFS reaches t=0 and exhausts all options.
//     This halts with: "CANARY: Rules are mathematically impossible —
//     backtracker exhausted all possibilities to t=0"
//     This is the canary in the coal mine: it proves the physics rules need
//     to be revised.
//
// ┌─────────────────────────────────────────────────────────────────────────────┐
// │ 6. MOVEMENT RULES                                                           │
// └─────────────────────────────────────────────────────────────────────────────┘
//
// XONIC MOVEMENT BALANCE
//   Each xon tracks 10 movement vector counters:
//     - 4 base directions (v1, v2, v3, v4 — positive direction only)
//     - 6 shortcut directions (s1+, s1-, s2+, s2-, s3+, s3- — 3 types × 2 signs)
//   Goal: equalize frequency across all 10 vectors. Least-used directions
//   score highest. This naturally creates center bias, coverage evenness,
//   and directional diversity without special-case heuristics.
//
// HADRONIC BALANCE > XONIC BALANCE
//   The system prioritizes producing the correct hadron ratios (from
//   _ratioTracker) over individual xon direction balance. Hadronic deficit
//   is weighted 10× relative to xonic deficit.
//
// NO-PREVNODE RULE
//   A xon may NEVER travel to prevNode UNLESS it is in tet mode following
//   its _loopSeq (fork/hook loops require node revisits). This is
//   absolute for oct, weak, gluon, and idle_tet modes.
//
// EJECTION RULES
//   _t60Ejected = true: xon has been ejected from confinement.
//   Pre-_mayReturn: may ONLY move to _isValidEjectionTarget nodes (not oct,
//     not any geometric tet, not any nucleus face node).
//   _mayReturn flips true after 1+ moves to a valid ejection target.
//   Post-_mayReturn: top priority return to oct. May enter oct nodes and
//     ejection-space nodes. May NOT enter _purelyTetNodes.
//
// OCT CAPACITY OVERFLOW (2-tier + T79 pressure)
//   1. Send xon into unscheduled hadronic loop (idle_tet) to vacate the cage.
//   2. If impossible, eject with _t60Ejected = true.
//   T79 pressure: after T79_MAX_FULL_TICKS-1 consecutive full-oct ticks,
//   overflow adds +1 to excess so that exactly-at-capacity (6) triggers
//   a shed. T79_MAX_FULL_TICKS is a tunable constant (default 6).
//
// ┌─────────────────────────────────────────────────────────────────────────────┐
// │ 7. VACUUM NEGOTIATION                                                       │
// └─────────────────────────────────────────────────────────────────────────────┘
//
// The PBD solver is the ONLY source of physical truth. Every SC activation
// goes through:
//   1. canMaterialiseQuick(scId) — dry-run solver, measures strain. No side effects.
//   2. If rejected: BFS all single-severance options. For each severable SC:
//      sever → canMaterialiseQuick → if yes, done; if no, undo sever.
//   3. If no single works: BFS all 2-severance combos. For each pair:
//      sever both → canMaterialiseQuick → if yes, done; if no, undo both.
//   4. If no 2-combo works: REJECTED. Vacuum wins. Xon must find another path.
//
// Severance penalty: each severance increments _severanceCount. Face scoring
// deducts _severanceCount × severancePenalty, discouraging severance-heavy plans.
//
// Severance eligibility: any SC not traversed on the previous turn, as long as
// it is not load-bearing (removing it would cause Kepler violation per solver).
// Traversal-locked SCs (actively being traversed by a xon) are NEVER severed.
//
// The solver also determines gluon assignment: if the cage would break without
// a particular xon's action (per PBD solver check), that xon becomes a gluon.
//
// ┌─────────────────────────────────────────────────────────────────────────────┐
// │ 8. ACTUALIZATION                                                            │
// └─────────────────────────────────────────────────────────────────────────────┘
//
// A tet loop is "actualized" if all face SCs were simultaneously active during
// at least one tick of the excitation loop. This is tracked per-step by the
// _tetActualized flag in _advanceXon — NOT re-checked at loop completion.
//
// Mid-loop T60 check (Phase 0) IS real-time: if a face SC disappears mid-loop,
// the xon is ejected immediately. The solver should prioritize not breaking
// loops mid-traversal (excitationSeverForRoom protects traversal-locked SCs).
//
// ┌─────────────────────────────────────────────────────────────────────────────┐
// │ 9. SC LIFECYCLE                                                             │
// └─────────────────────────────────────────────────────────────────────────────┘
//
// SC cleanup removes ONLY non-unit-length SCs from xonImpliedSet. All
// unit-length SCs remain as traversal paths and severance options, regardless
// of attribution. SCs with no active attribution are flagged "none" for
// diagnostics but NOT removed.
//
// Attribution (_scAttribution) is kept for debugging only — it does not drive
// cleanup decisions. The distance check (unit-length or not) is the sole
// removal criterion.
//
// ┌─────────────────────────────────────────────────────────────────────────────┐
// │ 10. KEY INVARIANTS (Live Guards)                                            │
// └─────────────────────────────────────────────────────────────────────────────┘
//
//   T19  Pauli exclusion     Max 1 xon per node at any tick.
//   T20  Never stand still   Every xon moves every tick. NO exemptions.
//   T21  Oct cage permanence Oct SCs stay in at least one SC set.
//   T26  No unactivated SC   SC-only traversals must use activated SCs.
//   T27  No teleportation    Xons only move via connected edges.
//   T45  No bouncing         No A→B→A for non-tet modes.
//   T55  Oct capacity        Max 4 oct-mode xons at any time.
//   T59  Trail consistency   Xon position matches trail head.
//   T60  Non-actualized tet  Ejection on face SC loss.
//   T61  No weak on oct      Weak xons cannot occupy oct nodes.
//   T62  Weak re-entry       Weak xons may only re-enter at oct nodes.
//   T79  Oct full limit       All 6 xons on oct nodes may persist for at most
//                              T79_MAX_FULL_TICKS consecutive ticks (tunable,
//                              default 6). Choreographer must shed at least 1.
//
//   All guards fire into the backtracker. The backtracker is the universal
//   resolution mechanism. There are no escape hatches, rescue phases, or
//   safety nets in the pipeline — only the guards and the backtracker.
//
// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║  END CHOREOGRAPHER DESIGN INTENT                                            ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

// ── Xonic Movement Balance: direction identification ──
// Maps SC stype (1-6) to _dirBalance index (4-9). Base dirs use indices 0-3 directly.
let _demoPanelDirty = false; // rAF coalescing flag for left-panel UI updates
let _demoPanelTimer = null;  // setTimeout handle for throttled heavy panel rebuilds
let _demoLightDirty = false; // rAF coalescing flag for lightweight updates (xon panel)
const _STYPE_TO_DIR = {1:4, 2:5, 3:6, 4:7, 5:8, 6:9};

// Identify the direction index (0-9) for a move from→to.
// Returns 0-3 for base edges, 4-9 for SC edges, or -1 if unknown.
function _identifyMoveDir(from, to) {
    // Check base edge first (preferred — base dirs 0-3)
    const bnb = baseNeighbors[from];
    if (bnb) {
        for (const nb of bnb) {
            if (nb.node === to) return nb.dirIdx; // 0-3
        }
    }
    // Check SC edge (stype → index 4-9)
    const pid = pairId(from, to);
    const scId = scPairToId.get(pid);
    if (scId !== undefined) {
        const sc = SC_BY_ID[scId];
        if (sc && _STYPE_TO_DIR[sc.stype] !== undefined) return _STYPE_TO_DIR[sc.stype];
    }
    return -1; // shouldn't happen if T27 holds
}

// Compute xonic balance score for a candidate direction.
// Higher = this direction is more underused = better to take it.
function _dirBalanceScore(xon, dirIdx) {
    if (!xon._dirBalance || dirIdx < 0 || dirIdx > 9) return 0;
    const counts = xon._dirBalance;
    let maxCount = 0;
    for (let i = 0; i < 10; i++) if (counts[i] > maxCount) maxCount = counts[i];
    return maxCount - counts[dirIdx]; // deficit: how far behind this dir is
}

// Compute aggregate xonic balance score for a set of direction indices.
// Used by face scoring to evaluate how much a tet loop would help this xon.
function _dirBalanceScoreMulti(xon, dirIndices) {
    if (!xon._dirBalance || !dirIndices || dirIndices.length === 0) return 0;
    const counts = xon._dirBalance;
    let maxCount = 0;
    for (let i = 0; i < 10; i++) if (counts[i] > maxCount) maxCount = counts[i];
    let totalDeficit = 0;
    for (const d of dirIndices) {
        if (d >= 0 && d <= 9) totalDeficit += maxCount - counts[d];
    }
    return totalDeficit;
}

// Increment _dirBalance for a move from→to.
function _updateDirBalance(xon, from, to) {
    const dirIdx = _identifyMoveDir(from, to);
    if (dirIdx >= 0 && dirIdx <= 9 && xon._dirBalance) {
        xon._dirBalance[dirIdx]++;
    }
}

// ── Locality filter: only return SCs whose endpoints are approximately unit-length apart ──
// This prevents non-local SC candidates from entering ANY decision path.
// Threshold 0.50: rejects teleportation-range (d > 1.5) but allows pre-solver SC edges (~1.15).
// Cached per (node, stateVersion) to avoid redundant filtering + sqrt calls.
let _localScCache = null;
let _localScCacheVersion = -1;
function _localScNeighbors(node) {
    // Invalidate cache when SC sets change
    if (_localScCacheVersion !== stateVersion) {
        _localScCache = new Array(N || 0);
        _localScCacheVersion = stateVersion;
    }
    if (_localScCache[node]) return _localScCache[node];
    const scs = scByVert[node] || [];
    const pa = pos[node];
    if (!pa) return scs;
    // Squared distance check: |d - 1| <= 0.50 ⟺ d² ∈ [0.25, 2.25]
    const result = scs.filter(sc => {
        const other = sc.a === node ? sc.b : sc.a;
        const pb = pos[other];
        if (!pb) return false;
        const dx = pb[0]-pa[0], dy = pb[1]-pa[1], dz = pb[2]-pa[2];
        const d2 = dx*dx + dy*dy + dz*dz;
        return d2 >= 0.25 && d2 <= 2.25;
    });
    _localScCache[node] = result;
    return result;
}

// ── Oct-distance helper: sort neighbors by proximity to nearest oct node ──
// Used by weak force paths to prevent wandering away from the nucleus.
function _distToNearestOct(node) {
    if (!_octNodeSet || !pos[node]) return Infinity;
    if (_octNodeSet.has(node)) return 0;
    let best = Infinity;
    for (const octN of _octNodeSet) {
        const p = pos[octN];
        if (!p) continue;
        const dx = pos[node][0]-p[0], dy = pos[node][1]-p[1], dz = pos[node][2]-p[2];
        const d = dx*dx + dy*dy + dz*dz; // squared distance (no sqrt needed for comparison)
        if (d < best) best = d;
    }
    return best;
}

// ── Universal nucleus-local neighbor filter ──
// Hard-filters a baseNeighbors array to ONLY nucleus nodes, then sorts by
// oct proximity (closest first). Every movement path should use this instead
// of raw baseNeighbors to guarantee no non-local moves escape.
function _localBaseNeighbors(node) {
    _ensureNucleusNodeSet();
    const nbs = baseNeighbors[node] || [];
    if (!_nucleusNodeSet) return nbs; // fallback pre-init
    return nbs.filter(nb => _nucleusNodeSet.has(nb.node))
              .sort((a, b) => _distToNearestOct(a.node) - _distToNearestOct(b.node));
}

function computeActivationPatterns() {
    const A = [1, 3, 6, 8];
    const B = [2, 4, 5, 7];
    const D4 = _DERANGEMENTS_4;
    const P4 = _PERMS_4;

    const lines = [];
    lines.push('═══════════════════════════════════════════');
    lines.push('  ACTIVATION PATTERN ANALYSIS');
    lines.push('  Octahedron K₄,₄ · 8 faces · 6 quarks');
    lines.push('═══════════════════════════════════════════');
    lines.push(`D(4) = ${D4.length} derangements of {0,1,2,3}:`);
    D4.forEach((d, i) => {
        lines.push(`  d${i}: [${d}] → A-faces: [${d.map(j => A[j])}]  B-faces: [${d.map(j => B[j])}]`);
    });

    // For each derangement d (F2 relative to F1 = identity):
    // Find anchors that avoid BOTH identity and d at every position
    // → anchor(i) ≠ i AND anchor(i) ≠ d(i) ∀i
    // These give "max spread" patterns (all 3 quarks on different faces)

    const patternData = [];

    lines.push('\n─── FOLLOWER-PAIR PHASING × ANCHOR COMPATIBILITY ───');
    lines.push('For each follower derangement d, which anchor schedules');
    lines.push('place all 3 quarks on DIFFERENT faces every tick?\n');

    for (let di = 0; di < D4.length; di++) {
        const d = D4[di];
        // Find all permutations that are derangements of BOTH identity and d
        const validAnchors = P4.filter(p =>
            p.every((v, i) => v !== i && v !== d[i])
        );
        lines.push(`  d${di} [${d}]: ${validAnchors.length} anchors → ${validAnchors.map(a => `[${a}]`).join('  ')}`);

        patternData.push({
            derangIdx: di,
            derang: d,
            anchors: validAnchors,
            anchorCount: validAnchors.length
        });
    }

    // Full hadron pattern = (A-derang, B-derang, A-anchor, B-anchor)
    // Max-spread patterns: both A and B sub-cycles have valid anchors
    let totalMaxSpread = 0;
    const fullPatterns = [];

    for (let ai = 0; ai < D4.length; ai++) {
        const aData = patternData[ai];
        for (let bi = 0; bi < D4.length; bi++) {
            const bData = patternData[bi];
            const count = aData.anchorCount * bData.anchorCount;
            if (count > 0) {
                totalMaxSpread += count;
                // Store the first anchor combo as representative
                fullPatterns.push({
                    aDerang: ai,
                    bDerang: bi,
                    anchorsA: aData.anchors,
                    anchorsB: bData.anchors,
                    combos: count
                });
            }
        }
    }

    lines.push(`\n─── SUMMARY ───`);
    lines.push(`Total follower phasings: ${D4.length}² = ${D4.length ** 2} per hadron`);
    lines.push(`Max-spread patterns (all 3 on different faces): ${totalMaxSpread} per hadron`);
    lines.push(`Full deuteron (proton × neutron): ${totalMaxSpread ** 2} max-spread combos`);

    // Show a few concrete patterns
    lines.push(`\n─── EXAMPLE MAX-SPREAD PATTERNS ───`);
    const examples = fullPatterns.slice(0, 3);
    for (const pat of examples) {
        const d_a = D4[pat.aDerang];
        const d_b = D4[pat.bDerang];
        const anchor_a = pat.anchorsA[0]; // first valid anchor for A
        const anchor_b = pat.anchorsB[0]; // first valid anchor for B

        // Build 8-tick schedule
        // Even ticks (0,2,4,6) → A faces; Odd ticks (1,3,5,7) → B faces
        const ticks = [0,1,2,3,4,5,6,7];
        const anchorSched = [], f1Sched = [], f2Sched = [];
        for (let i = 0; i < 4; i++) {
            // Even tick 2i: A faces
            anchorSched.push(A[anchor_a[i]]);
            f1Sched.push(A[i]);           // F1 = identity on A
            f2Sched.push(A[d_a[i]]);      // F2 = derangement of F1
            // Odd tick 2i+1: B faces
            anchorSched.push(B[anchor_b[i]]);
            f1Sched.push(B[i]);
            f2Sched.push(B[d_b[i]]);
        }

        const colW = 4;
        lines.push(`\nPattern A:d${pat.aDerang} × B:d${pat.bDerang} (${pat.combos} anchor combos)`);
        lines.push(`  tick:    ${ticks.map(t => String(t).padStart(colW)).join('')}`);
        lines.push(`  group:   ${ticks.map(t => (t%2===0?'A':'B').padStart(colW)).join('')}`);
        lines.push(`  anchor:  ${anchorSched.map(f => String(f).padStart(colW)).join('')}  (different type)`);
        lines.push(`  foll-1:  ${f1Sched.map(f => String(f).padStart(colW)).join('')}  (same type)`);
        lines.push(`  foll-2:  ${f2Sched.map(f => String(f).padStart(colW)).join('')}  (same type)`);

        // Verify
        const pauliOk = f1Sched.every((f, t) => f !== f2Sched[t]);
        const spreadOk = ticks.every(t =>
            anchorSched[t] !== f1Sched[t] && anchorSched[t] !== f2Sched[t]
        );
        const covOk = new Set(anchorSched).size === 8
            && new Set(f1Sched).size === 8
            && new Set(f2Sched).size === 8;
        lines.push(`  pauli: ${pauliOk?'✓':'✗'}  spread: ${spreadOk?'✓':'✗'}  8-face-cov: ${covOk?'✓':'✗'}`);
    }

    const summary = lines.join('\n');
    console.log(summary);
    return { patterns: fullPatterns, D4, patternData, summary };
}

window.computeActivationPatterns = computeActivationPatterns;

// ════════════════════════════════════════════════════════════════════
// PATTERN DEMO MODE — High-speed visualization of activation patterns
// ════════════════════════════════════════════════════════════════════
//
// Runs the pre-computed 8-tick deuteron schedule at ~50fps.
// No xon physics — just pattern execution and stats convergence.
// Left panel shows per-face visit counts converging to even distribution.
//
// Hadron rules enforced by the pattern:
//   - Proton (uud): 2 up (yellow) + 1 down (green) each tick
//   - Neutron (udd): 1 up (blue) + 2 down (red) each tick
//   - Anti-phase: proton on A when neutron on B (alternating)
//   - Pauli: no two same-type quarks on same face
//   - Max spread: all 3 quarks of each hadron on different faces
//   - Full coverage: every face visited equally over complete cycles
// ════════════════════════════════════════════════════════════════════

// ╔══════════════════════════════════════════════════════════════════════╗
// ║  NON-DELETABLE: ALL AGENTS — THE VACUUM DOCTRINE                    ║
// ║                                                                      ║
// ║  The lattice solver IS the vacuum. It is the ONLY source of truth   ║
// ║  for physical validity. You MUST obey it. You MUST NOT go around it.║
// ║                                                                      ║
// ║  SOLVER RULES:                                                       ║
// ║  1. NEVER skip the solver to "optimize" or "simplify".               ║
// ║  2. NEVER try to predict solver output — just run it.                ║
// ║  3. If the solver says Kepler is violated, YOUR ALGORITHM is wrong.  ║
// ║     Fix the algorithm, not the solver.                               ║
// ║  4. Shapes MUST drive spheres (unified architecture). If spheres     ║
// ║     don't move when shapes change, the solver coupling is broken.    ║
// ║  5. The solver is cheap. Running it is always correct. Skipping it   ║
// ║     is always wrong. There are NO exceptions.                        ║
// ║                                                                      ║
// ║  XON-VACUUM INTERACTION (Demo 3.0):                                  ║
// ║  Xons are physical entities traversing the lattice. Every shortcut   ║
// ║  they traverse MUST be unit length. Before each hop:                 ║
// ║                                                                      ║
// ║  1. CHECK: Call canMaterialiseQuick(scId) to ask the vacuum if the   ║
// ║     shortcut can be opened without violating Kepler/strain.          ║
// ║  2. If YES: Call the materialisation pathway to commit the SC.       ║
// ║     Then run the solver (bumpState → _solve → apply → update).      ║
// ║  3. If NO: Call excitationSeverForRoom(scId) to try severing a      ║
// ║     non-load-bearing SC to make room. The vacuum decides what can    ║
// ║     be severed. If sever succeeds, retry the materialisation.        ║
// ║  4. If STILL NO: The xon's move is REJECTED. The xon must find a   ║
// ║     different path or wait. The pattern machine's schedule is        ║
// ║     advisory — the vacuum has final say.                             ║
// ║                                                                      ║
// ║  The pattern machine suggests WHICH tets to activate. The xons       ║
// ║  negotiate with the vacuum HOW (or whether) to achieve it.           ║
// ║  The vacuum always wins.                                             ║
// ╚══════════════════════════════════════════════════════════════════════╝

function _swapBlocked(fromNode, toNode) {
    return _noSwapRule && _moveRecord.get(fromNode) === toNode;
}
function _ensureNucleusNodeSet() {
    if (_nucleusNodeSet) return;
    if (!_octNodeSet || _octNodeSet.size === 0) return; // not ready yet
    if (!_nucleusTetFaceData) return;
    _nucleusNodeSet = new Set(_octNodeSet);
    for (let f = 1; f <= 8; f++) {
        const fd = _nucleusTetFaceData[f];
        if (fd) for (const n of fd.allNodes) _nucleusNodeSet.add(n);
    }
    // Build ejection targets: base neighbors of oct nodes NOT in nucleus
    _ejectionTargetNodes = new Set();
    for (const octN of _octNodeSet) {
        const nbs = baseNeighbors[octN] || [];
        for (const nb of nbs) {
            if (_nucleusNodeSet.has(nb.node)) continue;
            _ejectionTargetNodes.add(nb.node);
        }
    }
    // Build _nucleusFaceNodes: the 8 nucleus tet faces' nodes (STATIC)
    _nucleusFaceNodes = new Set();
    for (let f = 1; f <= 8; f++) {
        const fd = _nucleusTetFaceData[f];
        if (fd) for (const n of fd.allNodes) _nucleusFaceNodes.add(n);
    }
    // Initial computation of dynamic sets
    _recomputeActualizedTetNodes();
    console.log(`[FLASHLIGHT] Nucleus node set: ${_nucleusNodeSet.size} nodes: [${Array.from(_nucleusNodeSet).sort((a,b)=>a-b).join(',')}]`);
    console.log(`[FLASHLIGHT] Ejection targets: ${_ejectionTargetNodes.size} nodes: [${Array.from(_ejectionTargetNodes).sort((a,b)=>a-b).join(',')}]`);
}

// Recompute dynamic ejection sets based on currently actualized tets.
// A tet is "actualized" (geometrically precise) iff ALL its bounding SCs are active.
// Called at tick start and after SC set changes.
function _recomputeActualizedTetNodes() {
    _actualizedTetNodes = new Set();
    // No actualized tets until oct is discovered — geometry isn't established yet
    if (!_octNodeSet || _octNodeSet.size === 0) {
        _ejectionForbidden = new Set();
        _purelyTetNodes = new Set();
        return;
    }
    if (typeof voidNeighborData !== 'undefined') {
        for (const v of voidNeighborData) {
            if (v.type !== 'tet') continue;
            const allActive = v.scIds.every(scId =>
                activeSet.has(scId) || impliedSet.has(scId) || xonImpliedSet.has(scId));
            if (allActive) {
                for (const n of v.nbrs) _actualizedTetNodes.add(n);
            }
        }
    }
    // _ejectionForbidden: oct ∪ actualizedTets (dynamic only — weak xons may visit non-actualized tet nodes)
    _ejectionForbidden = new Set(_octNodeSet || []);
    if (_actualizedTetNodes) for (const n of _actualizedTetNodes) _ejectionForbidden.add(n);
    // _purelyTetNodes: actualizedTets \ oct (forbidden even post-_mayReturn)
    _purelyTetNodes = new Set();
    for (const n of _actualizedTetNodes) {
        if (!_octNodeSet || !_octNodeSet.has(n)) _purelyTetNodes.add(n);
    }
}
// ── Ejection target validity check (see CHOREOGRAPHER DESIGN INTENT §3) ──
function _isValidEjectionTarget(node) {
    _ensureNucleusNodeSet();
    return _ejectionForbidden ? !_ejectionForbidden.has(node) : false;
}
// Returns true if a hadron (tet or idle_tet xon) currently occupies this node.
// Weak xons may traverse actualized tet geometry but must avoid active hadrons.
function _isHadronOccupied(node) {
    for (const x of _demoXons) {
        if (x.alive && x.node === node && (x._mode === 'tet' || x._mode === 'idle_tet')) return true;
    }
    return false;
}
// ── Cage integrity check: would the cage break without this xon? ──
// Uses SC set membership to determine if cage SCs remain stable
// without this xon's participation this tick.
function _cageWouldBreak(xon) {
    if (!_octSCIds || !_octNodeSet || !xon.alive) return false;
    if (!_octNodeSet.has(xon.node)) return false; // not on cage

    // Check if all cage SCs are physically maintained. Only count activeSet
    // and impliedSet (physics-derived) — xonImpliedSet protection doesn't
    // count as genuine stability (it's a safety net, not xon-induced).
    for (const scId of _octSCIds) {
        if (!activeSet.has(scId) && !impliedSet.has(scId)) {
            return true;
        }
    }

    // All cage SCs are active — cage is stable even without this xon
    return false;
}
function _traceMove(xon, from, to, path) {
    const entry = {xonIdx: _demoXons.indexOf(xon), from, to, path, mode: xon._mode, tick: _demoTick};
    _moveTrace.push(entry);
    _moveTraceHistory.push(entry);
    if (_moveTraceHistory.length > 60) _moveTraceHistory.splice(0, _moveTraceHistory.length - 60);
    // FLASHLIGHT: log (but don't freeze) if xon moves to a non-nucleus node
    _ensureNucleusNodeSet();
    if (_nucleusNodeSet && !_nucleusNodeSet.has(to) &&
        !(xon._mode === 'weak' && _isValidEjectionTarget(to))) {
        console.warn(`[FLASHLIGHT] tick=${_demoTick} X${entry.xonIdx} moved ${from}→${to} (outside nucleus) mode=${xon._mode}`);
    }
}


// ── Fighterjet mode: Catmull-Rom spline + curvature easing ──────────
// Uniform Catmull-Rom evaluation on segment p1→p2, parameter t∈[0,1].
// Catmull-Rom spline with variable tension τ (_fjTension).
// τ=0: flat tangents (smooth ease), τ=0.5: standard CR, τ=1: aggressive.
// p0,p1,p2,p3 are [x,y,z] arrays. Returns [x,y,z].
const _crOut = [0, 0, 0]; // reusable return to avoid allocation
const _fjP3  = [0, 0, 0]; // reusable extrapolated future point
function _catmullRom(p0, p1, p2, p3, t) {
    const tau = _fjTension;
    const t2 = t * t, t3 = t2 * t;
    const h00 = 2*t3 - 3*t2 + 1;
    const h10 = t3 - 2*t2 + t;
    const h01 = -2*t3 + 3*t2;
    const h11 = t3 - t2;
    _crOut[0] = h00*p1[0] + h10*tau*(p2[0]-p0[0]) + h01*p2[0] + h11*tau*(p3[0]-p1[0]);
    _crOut[1] = h00*p1[1] + h10*tau*(p2[1]-p0[1]) + h01*p2[1] + h11*tau*(p3[1]-p1[1]);
    _crOut[2] = h00*p1[2] + h10*tau*(p2[2]-p0[2]) + h01*p2[2] + h11*tau*(p3[2]-p1[2]);
    return _crOut;
}

// Centripetal Catmull-Rom (Barry & Goldman).
// _fjAlpha: 0 = uniform (standard), 0.5 = centripetal, 1 = chordal.
// More expensive than uniform CR but prevents cusps and self-intersections.
const _ccrOut = [0, 0, 0];
function _centripetalCR(p0, p1, p2, p3, t) {
    function _dist2(a, b) {
        const dx = a[0]-b[0], dy = a[1]-b[1], dz = a[2]-b[2];
        return dx*dx + dy*dy + dz*dz;
    }
    const a = _fjAlpha * 0.5; // half-alpha for pow(dist², α/2) = pow(dist, α)
    const d01 = Math.pow(_dist2(p0, p1), a) || 1e-6;
    const d12 = Math.pow(_dist2(p1, p2), a) || 1e-6;
    const d23 = Math.pow(_dist2(p2, p3), a) || 1e-6;
    // Knot values
    const t1 = d01;
    const t2 = t1 + d12;
    const t3 = t2 + d23;
    // Remap t from [0,1] to [t1, t2]
    const tt = t1 + t * d12;
    // Barry-Goldman pyramid
    for (let c = 0; c < 3; c++) {
        const A1 = (t1 - tt) / (t1) * p0[c] + tt / t1 * p1[c];
        const A2 = (t2 - tt) / d12 * p1[c] + (tt - t1) / d12 * p2[c];
        const A3 = (t3 - tt) / d23 * p2[c] + (tt - t2) / d23 * p3[c];
        const B1 = (t2 - tt) / t2 * A1 + tt / t2 * A2;
        const B2 = (t3 - tt) / (t3 - t1) * A2 + (tt - t1) / (t3 - t1) * A3;
        _ccrOut[c] = (t2 - tt) / d12 * B1 + (tt - t1) / d12 * B2;
    }
    return _ccrOut;
}

// Curvature-blended interpolation: blend between linear and CR.
// _fjCurvature: 0 = straight line, 1 = standard CR, 2 = exaggerated.
// Uses centripetal CR when _fjAlpha > 0, uniform CR otherwise.
const _fjBlendOut = [0, 0, 0]; // reusable return
function _fjBlend(p0, p1, p2, p3, t) {
    if (_fjCurvature <= 0) {
        _fjBlendOut[0] = p1[0] + (p2[0] - p1[0]) * t;
        _fjBlendOut[1] = p1[1] + (p2[1] - p1[1]) * t;
        _fjBlendOut[2] = p1[2] + (p2[2] - p1[2]) * t;
        return _fjBlendOut;
    }
    const cr = _fjAlpha > 0 ? _centripetalCR(p0, p1, p2, p3, t)
                             : _catmullRom(p0, p1, p2, p3, t);
    if (_fjCurvature === 1.0) return cr;
    const lx = p1[0] + (p2[0] - p1[0]) * t;
    const ly = p1[1] + (p2[1] - p1[1]) * t;
    const lz = p1[2] + (p2[2] - p1[2]) * t;
    _fjBlendOut[0] = lx + _fjCurvature * (cr[0] - lx);
    _fjBlendOut[1] = ly + _fjCurvature * (cr[1] - ly);
    _fjBlendOut[2] = lz + _fjCurvature * (cr[2] - lz);
    return _fjBlendOut;
}

// Curvature-dependent easing: blends linear (straight) with cubic
// ease-in-out (sharp turns). angle = turn angle in radians [0,PI].
function _fjEase(t, angle) {
    const sharpness = Math.min(1, angle / (Math.PI * 0.7)); // normalize; 126° = full ease
    const linear = t;
    const eio = t < 0.5 ? 4*t*t*t : 1 - (-2*t+2)**3 / 2; // cubic ease-in-out
    return linear * (1 - sharpness) + eio * sharpness;
}

// Compute turn angle between two consecutive segments (v1→v2 and v2→v3).
// Returns angle in radians. 0 = straight, PI = reversal.
function _fjTurnAngle(p0, p1, p2) {
    const ax = p1[0]-p0[0], ay = p1[1]-p0[1], az = p1[2]-p0[2];
    const bx = p2[0]-p1[0], by = p2[1]-p1[1], bz = p2[2]-p1[2];
    const dot = ax*bx + ay*by + az*bz;
    const la = Math.sqrt(ax*ax+ay*ay+az*az);
    const lb = Math.sqrt(bx*bx+by*by+bz*bz);
    if (la < 1e-8 || lb < 1e-8) return 0;
    return Math.acos(Math.max(-1, Math.min(1, dot / (la * lb))));
}

// Animate all demo xons — called every frame from the render loop.
// Handles tween interpolation, spark flash, trail rendering, and trail decay.
function _tickDemoXons(dt) {
    const sparkOp = (+document.getElementById('spark-opacity-slider').value) / 100;
    const weakOp = (typeof _roleOpacity !== 'undefined' && _roleOpacity.weak != null) ? _roleOpacity.weak : 0.13;
    const demoStepSec = _getDemoIntervalMs() * 0.001;

    for (let xi = _demoXons.length - 1; xi >= 0; xi--) {
        const xon = _demoXons[xi];

        // ── Dying xons: render frozen trail (decay happens in demoTick) ──
        if (xon._dying) {
            if (!xon._frozenPos || xon._frozenPos.length === 0 || !xon.trailGeo) {
                // Trail fade complete — if group intact (annihilated), keep in array
                // for _manifestXonPair reactivation. Only splice fully destroyed xons.
                if (xon.group) {
                    // Annihilated xon: keep slot, just finish dying
                    xon._dying = false;
                    xon._dyingStartTick = null; // reset for T14
                    if (xon.trailLine) xon.trailLine.visible = false;
                } else {
                    _finalCleanupXon(xon);
                    _demoXons.splice(xi, 1);
                }
                continue;
            }
            // Render from frozen (historical) positions — per-segment colors
            const n = xon._frozenPos.length;
            for (let i = 0; i < n; i++) {
                const fp = xon._frozenPos[i];
                xon.trailPos[i * 3] = fp[0];
                xon.trailPos[i * 3 + 1] = fp[1];
                xon.trailPos[i * 3 + 2] = fp[2];
                const segCol = (xon._frozenRoles && xon._frozenRoles[i]) ? _roleToColor(xon._frozenRoles[i]) : xon.col;
                const cr = ((segCol >> 16) & 0xff) / 255;
                const cg = ((segCol >> 8) & 0xff) / 255;
                const cb = (segCol & 0xff) / 255;
                const alpha = sparkOp * (0.5 + 0.5 * (i / Math.max(n - 1, 1)) ** 0.8);
                xon.trailCol[i * 3] = cr * alpha;
                xon.trailCol[i * 3 + 1] = cg * alpha;
                xon.trailCol[i * 3 + 2] = cb * alpha;
            }
            xon.trailGeo.setDrawRange(0, n);
            xon.trailGeo.attributes.position.needsUpdate = true;
            xon.trailGeo.attributes.color.needsUpdate = true;
            continue;
        }

        if (!xon.alive || !xon.group) continue;

        // Ensure main trail stays AdditiveBlending always
        if (xon.trailLine) {
            const mat = xon.trailLine.material;
            if (mat.blending !== THREE.AdditiveBlending) {
                mat.blending = THREE.AdditiveBlending;
                mat.needsUpdate = true;
            }
        }

        // ── Live xons: tween + spark + trail ──
        // Curved trails: enforce minimum tween duration so curves are visible
        const stepSec = (_fjCurvature > 0 && demoStepSec < 0.06) ? 0.06 : demoStepSec;
        // Reverse animation: decrement _fjRevT from 1→0 (independent timer)
        if (xon._fjReverseFrom) {
            const revDt = Math.min(dt, 0.033); // cap to prevent instant completion
            xon._fjRevT = Math.max(0, (xon._fjRevT || 1) - revDt / stepSec);
            if (xon._fjRevT <= 0) {
                xon._fjReverseFrom = null; // done reversing
                xon._fjRevT = 0;
            }
            // Freeze tweenT at 1 during reverse so forward path never runs
            xon.tweenT = 1;
        } else {
            xon.tweenT = Math.min(1, xon.tweenT + dt / stepSec);
        }
        const pf = pos[xon.prevNode], pt = pos[xon.node];
        if (pf && pt) {
            const _tdx = pt[0] - pf[0], _tdy = pt[1] - pf[1], _tdz = pt[2] - pf[2];
            const _hopDist = Math.sqrt(_tdx*_tdx + _tdy*_tdy + _tdz*_tdz);
            // Reverse animation takes priority — check BEFORE teleport guard
            if (xon._fjReverseFrom && _fjCurvature > 0) {
                // ── REVERSE: lerp from saved position → restored node ──
                const from = xon._fjReverseFrom;
                const to = pos[xon.node]; // restored tick's node position
                const t = xon._fjRevT; // 1=old pos, 0=restored pos
                xon.group.position.set(
                    to[0] + (from[0] - to[0]) * t,
                    to[1] + (from[1] - to[1]) * t,
                    to[2] + (from[2] - to[2]) * t
                );
            } else if (_hopDist > 1.2) {
                xon.group.position.set(pf[0], pf[1], pf[2]);
            } else if (_fjCurvature > 0) {
                // ── FORWARD: curvature-blended spline ──
                const _tl = xon.trail ? xon.trail.length : 0;
                const p1 = pf, p2 = pt;
                const p0 = (_tl >= 3) ? xon.trail[_tl - 3].pos : p1;
                _fjP3[0] = 2*p2[0] - p1[0];
                _fjP3[1] = 2*p2[1] - p1[1];
                _fjP3[2] = 2*p2[2] - p1[2];
                const bl = _fjBlend(p0, p1, p2, _fjP3, xon.tweenT);
                xon.group.position.set(bl[0], bl[1], bl[2]);
            } else {
                // ── Curvature 0: cubic ease-out lerp ──
                const s = 1 - (1 - xon.tweenT) ** 3;
                const px = pf[0] + (pt[0] - pf[0]) * s;
                const py = pf[1] + (pt[1] - pf[1]) * s;
                const pz = pf[2] + (pt[2] - pf[2]) * s;
                xon.group.position.set(px, py, pz);
            }
        }

        // Sparkle flash + flicker
        xon.flashT = Math.max(0, xon.flashT - dt * 6.0);
        const flicker = 0.85 + Math.random() * 0.3;
        const hlBoost = xon._highlightT > 0 ? 2.5 : 1.0;
        const isGluon = xon._mode === 'gluon';
        const isWeak = xon._mode === 'weak';
        const gluonBoost = isGluon ? 1.5 : 1.0;
        // Per-xon × per-role opacity hierarchy
        const _xoi = (typeof _xonOpacity !== 'undefined' && _xonOpacity[xi] != null) ? _xonOpacity[xi] : 1;
        const _curRole = typeof _xonRole === 'function' ? _xonRole(xon) : 'oct';
        const _roleOp = (typeof _roleOpacity !== 'undefined' && _roleOpacity[_curRole] != null) ? _roleOpacity[_curRole] : 1;
        const xonOp = _xoi * _roleOp;
        // Swap spark blending for gluon/weak xons (additive can't show dark colors)
        const needsNormal = isWeak || isGluon;
        if (needsNormal && xon.sparkMat.blending !== THREE.NormalBlending) {
            xon.sparkMat.blending = THREE.NormalBlending;
            xon.sparkMat.needsUpdate = true;
        } else if (!needsNormal && xon.sparkMat.blending !== THREE.AdditiveBlending) {
            xon.sparkMat.blending = THREE.AdditiveBlending;
            xon.sparkMat.needsUpdate = true;
        }
        const pulse = (0.22 + xon.flashT * 0.26) * flicker * hlBoost * gluonBoost;
        xon.spark.scale.set(pulse, pulse, 1);
        // Spark opacity: sparkOp × xonOp (per-xon × per-role)
        xon.sparkMat.opacity = isWeak ? sparkOp * xonOp :
            isGluon ? Math.min(0.5, (0.3 + xon.flashT * 0.2) * flicker * sparkOp * xonOp) :
            Math.min(1.0, (0.6 + xon.flashT * 0.4) * flicker * sparkOp * hlBoost * xonOp);
        // Decay highlight timer
        if (xon._highlightT > 0) xon._highlightT = Math.max(0, xon._highlightT - dt);
        xon.sparkMat.rotation = Math.random() * Math.PI * 2;

        // Trail: fading vertex-colored path
        // Trails knob controls how many trail points are visible (0-1000).
        // Always store full history; render only the last `visLen` points.
        const lifespan = +document.getElementById('tracer-lifespan-slider').value;
        const fullLen = xon.trail.length;
        const visLen = Math.min(fullLen, lifespan);
        const startIdx = fullLen - visLen; // skip older points beyond lifespan

        // During tween (tweenT < 1), the latest trail entry is the DESTINATION
        // which the sprite hasn't reached yet. Rendering it in the body creates
        // a backward line from destination back to sprite. Fix: exclude the
        // latest entry during tween and let the trail head animate the hop.
        // During reverse: trails show restored snapshot data instantly (no exclusion needed)
        const bodyLen = (xon.tweenT < 1 && visLen > 1 && !xon._fjReverseFrom) ? visLen - 1 : visLen;

        // ── Fighterjet curved trails: subdivide each segment with CR spline ──
        if (_fjCurvature > 0 && bodyLen >= 2) {
            // FJ_SUBS defined in flux-demo-state.js
            xon._lastTrailFlashBoost = 0;
            let out = 0; // output vertex index
            for (let vi = 0; vi < bodyLen - 1 && out < XON_TRAIL_VERTS - FJ_SUBS - 2; vi++) {
                const i = startIdx + vi;
                const e1 = xon.trail[i], e2 = xon.trail[i + 1];
                // 4 control points for this segment
                const cp1 = e1.pos || pos[e1.node];
                const cp2 = e2.pos || pos[e2.node];
                if (!cp1 || !cp2) continue;
                let cp0 = (i > 0 ? (xon.trail[i-1].pos || pos[xon.trail[i-1].node]) : null) || cp1;
                let cp3 = (i + 2 < fullLen ? (xon.trail[i+2].pos || pos[xon.trail[i+2].node]) : null)
                    || [2*cp2[0]-cp1[0], 2*cp2[1]-cp1[1], 2*cp2[2]-cp1[2]];
                { const d0x=cp0[0]-cp1[0],d0y=cp0[1]-cp1[1],d0z=cp0[2]-cp1[2];
                  if(d0x*d0x+d0y*d0y+d0z*d0z>1.44) cp0=cp1; }
                { const d3x=cp3[0]-cp2[0],d3y=cp3[1]-cp2[1],d3z=cp3[2]-cp2[2];
                  if(d3x*d3x+d3y*d3y+d3z*d3z>1.44) cp3=cp2; }
                const _tdx = cp2[0]-cp1[0], _tdy = cp2[1]-cp1[1], _tdz = cp2[2]-cp1[2];
                const teleport = (_tdx*_tdx + _tdy*_tdy + _tdz*_tdz > 1.44);
                // Segment color: quarks/gluons use e2 (destination) flat color.
                // Oct↔non-oct transitions interpolate for smooth blending.
                const col1 = _roleToColor(e1.role);
                const c1r = ((col1 >> 16) & 0xff) / 255;
                const c1g = ((col1 >> 8) & 0xff) / 255;
                const c1b = (col1 & 0xff) / 255;
                const col2 = _roleToColor(e2.role);
                const c2r = ((col2 >> 16) & 0xff) / 255;
                const c2g = ((col2 >> 8) & 0xff) / 255;
                const c2b = (col2 & 0xff) / 255;
                const op1 = (typeof _roleOpacity !== 'undefined' && _roleOpacity[e1.role] != null) ? _roleOpacity[e1.role] : 1;
                const op2 = (typeof _roleOpacity !== 'undefined' && _roleOpacity[e2.role] != null) ? _roleOpacity[e2.role] : 1;
                // Blend any transition involving oct (oct↔weak, oct↔tet, tet↔oct)
                const _blend = e1.role !== e2.role && (e1.role === 'oct' || e2.role === 'oct');
                // Emit FJ_SUBS vertices along curve (skip last to avoid duplicates)
                for (let s = 0; s < FJ_SUBS && out < XON_TRAIL_VERTS; s++) {
                    const u = s / FJ_SUBS;
                    let px, py, pz;
                    if (teleport) {
                        px = cp1[0]; py = cp1[1]; pz = cp1[2];
                    } else {
                        const bl = _fjBlend(cp0, cp1, cp2, cp3, u);
                        px = bl[0]; py = bl[1]; pz = bl[2];
                    }
                    xon.trailPos[out * 3]     = px;
                    xon.trailPos[out * 3 + 1] = py;
                    xon.trailPos[out * 3 + 2] = pz;
                    // Progress through entire trail for alpha fade (0=tail, 1=head)
                    const progress = (vi + u) / Math.max(bodyLen - 1, 1);
                    // Gentle fade: linear with floor so long trails stay visible
                    const fadeCurve = _trailFadeFloor + (1 - _trailFadeFloor) * progress;
                    if (teleport) {
                        xon.trailCol[out*3] = 0; xon.trailCol[out*3+1] = 0; xon.trailCol[out*3+2] = 0;
                    } else {
                        // Blend oct↔non-oct transitions; flat color otherwise
                        const scr = _blend ? c1r + (c2r - c1r) * u : c2r;
                        const scg = _blend ? c1g + (c2g - c1g) * u : c2g;
                        const scb = _blend ? c1b + (c2b - c1b) * u : c2b;
                        const segRoleOp = _blend ? op1 + (op2 - op1) * u : op2;
                        const flashBoost = xon.flashT * 0.4 * progress;
                        xon._lastTrailFlashBoost = Math.max(xon._lastTrailFlashBoost || 0, flashBoost);
                        const alpha = sparkOp * _xoi * segRoleOp * Math.min(1, fadeCurve + flashBoost);
                        xon.trailCol[out*3] = scr * alpha; xon.trailCol[out*3+1] = scg * alpha; xon.trailCol[out*3+2] = scb * alpha;
                    }
                    out++;
                }
            }
            // Trail head: CR-subdivide current hop up to tweenT (matches sprite path)
            // During reverse animation: skip head (trails show restored state instantly)
            if (!xon._fjReverseFrom && out < XON_TRAIL_VERTS - FJ_SUBS - 2 && xon.tweenT < 1 && bodyLen >= 1) {
                const hi = startIdx + bodyLen - 1; // last body index
                const _he = xon.trail[hi];
                const hp1 = _he ? (_he.pos || pos[_he.node]) : null; // prevNode pos
                const hp2 = pos[xon.node]; // destination
                if (hp1 && hp2) {
                    const _he0 = hi > 0 ? xon.trail[hi - 1] : null;
                    const hp0 = (_he0 ? (_he0.pos || pos[_he0.node]) : null) || hp1;
                    _fjP3[0] = 2*hp2[0]-hp1[0]; _fjP3[1] = 2*hp2[1]-hp1[1]; _fjP3[2] = 2*hp2[2]-hp1[2];
                    const _tdx = hp2[0]-hp1[0], _tdy = hp2[1]-hp1[1], _tdz = hp2[2]-hp1[2];
                    const teleport = (_tdx*_tdx + _tdy*_tdy + _tdz*_tdz > 1.44);
                    const headCol = xon.col;
                    const headRole = _xonRole(xon);
                    const headRoleOp = (typeof _roleOpacity !== 'undefined' && _roleOpacity[headRole] != null) ? _roleOpacity[headRole] : 1;
                    const hcr = ((headCol >> 16) & 0xff) / 255;
                    const hcg = ((headCol >> 8) & 0xff) / 255;
                    const hcb = (headCol & 0xff) / 255;
                    const headSubs = Math.max(1, Math.round(FJ_SUBS * xon.tweenT));
                    for (let s = 1; s <= headSubs && out < XON_TRAIL_VERTS; s++) {
                        const u = (s / FJ_SUBS) * (xon.tweenT > 0 ? 1 : 0);
                        const t = u * xon.tweenT / (headSubs / FJ_SUBS);
                        let px, py, pz;
                        if (teleport) { px = hp1[0]; py = hp1[1]; pz = hp1[2]; }
                        else { const bl = _fjBlend(hp0, hp1, hp2, _fjP3, s / headSubs * xon.tweenT); px = bl[0]; py = bl[1]; pz = bl[2]; }
                        xon.trailPos[out*3] = px; xon.trailPos[out*3+1] = py; xon.trailPos[out*3+2] = pz;
                        const headAlpha = sparkOp * _xoi * headRoleOp;
                        xon.trailCol[out*3] = hcr*headAlpha; xon.trailCol[out*3+1] = hcg*headAlpha; xon.trailCol[out*3+2] = hcb*headAlpha;
                        out++;
                    }
                }
            }
            xon.trailGeo.setDrawRange(0, Math.min(out, XON_TRAIL_VERTS));
            xon.trailGeo.attributes.position.needsUpdate = true;
            xon.trailGeo.attributes.color.needsUpdate = true;
        } else {
        // ── Normal straight-line trails ──

        // Per-segment color from unified trail entry — segments retain their original role
        xon._lastTrailFlashBoost = 0; // reset per frame for T37 measurement
        for (let vi = 0; vi < bodyLen; vi++) {
            const i = startIdx + vi;
            const _te = xon.trail[i];
            if (!_te) continue;
            const np = _te.pos || pos[_te.node];
            if (!np) continue;
            // Teleport suppression
            if (vi > 0) {
                const _spx = xon.trailPos[(vi-1) * 3], _spy = xon.trailPos[(vi-1) * 3 + 1], _spz = xon.trailPos[(vi-1) * 3 + 2];
                const _sdx = np[0] - _spx, _sdy = np[1] - _spy, _sdz = np[2] - _spz;
                if (_sdx*_sdx + _sdy*_sdy + _sdz*_sdz > 1.44) {
                    xon.trailPos[vi * 3] = _spx;
                    xon.trailPos[vi * 3 + 1] = _spy;
                    xon.trailPos[vi * 3 + 2] = _spz;
                    xon.trailCol[vi * 3] = 0;
                    xon.trailCol[vi * 3 + 1] = 0;
                    xon.trailCol[vi * 3 + 2] = 0;
                    continue;
                }
            }
            xon.trailPos[vi * 3] = np[0];
            xon.trailPos[vi * 3 + 1] = np[1];
            xon.trailPos[vi * 3 + 2] = np[2];
            // Color from next entry (destination) so segments show the move's role,
            // not the origin's. Matches Catmull-Rom path which uses e2.role.
            const _teNext = (vi < bodyLen - 1) ? xon.trail[startIdx + vi + 1] : _te;
            const segCol = _roleToColor(_teNext.role);
            const cr = ((segCol >> 16) & 0xff) / 255;
            const cg = ((segCol >> 8) & 0xff) / 255;
            const cb = (segCol & 0xff) / 255;
            const segRoleOp = (typeof _roleOpacity !== 'undefined' && _roleOpacity[_te.role] != null) ? _roleOpacity[_te.role] : 1;
            // Gentle fade: linear with floor so long trails stay visible
            const progress = vi / Math.max(bodyLen - 1, 1); // 0=tail, 1=head
            const fadeCurve = _trailFadeFloor + (1 - _trailFadeFloor) * progress;
            const flashBoost = xon.flashT * 0.4 * progress;
            xon._lastTrailFlashBoost = Math.max(xon._lastTrailFlashBoost || 0, flashBoost);
            const alpha = sparkOp * _xoi * segRoleOp * Math.min(1, fadeCurve + flashBoost);
            xon.trailCol[vi * 3] = cr * alpha;
            xon.trailCol[vi * 3 + 1] = cg * alpha;
            xon.trailCol[vi * 3 + 2] = cb * alpha;
        }
        // Current interpolated position as trail head — extends from last BODY
        // entry toward sprite. During tween this smoothly animates the hop.
        // Skip trail head during reverse animation — no new trail should be traced.
        const last = bodyLen;
        let _drawHead = false;
        if (last < XON_TRAIL_LENGTH && bodyLen > 0) {
            // Distance from last body point to current group position
            const _lfi = startIdx + bodyLen - 1;
            const _lfe = xon.trail[_lfi];
            const _lfp = _lfe ? (_lfe.pos || pos[_lfe.node]) : null;
            if (_lfp) {
                const _hdx = xon.group.position.x - _lfp[0];
                const _hdy = xon.group.position.y - _lfp[1];
                const _hdz = xon.group.position.z - _lfp[2];
                if (_hdx*_hdx + _hdy*_hdy + _hdz*_hdz <= 1.44) { // 1.2^2
                    _drawHead = true;
                    xon.trailPos[last * 3] = xon.group.position.x;
                    xon.trailPos[last * 3 + 1] = xon.group.position.y;
                    xon.trailPos[last * 3 + 2] = xon.group.position.z;
                    const headCol = xon.col;
                    const headRole = _xonRole(xon);
                    const headRoleOp = (typeof _roleOpacity !== 'undefined' && _roleOpacity[headRole] != null) ? _roleOpacity[headRole] : 1;
                    const hcr = ((headCol >> 16) & 0xff) / 255;
                    const hcg = ((headCol >> 8) & 0xff) / 255;
                    const hcb = (headCol & 0xff) / 255;
                    const headAlpha = sparkOp * xonOp * headRoleOp;
                    xon.trailCol[last * 3] = hcr * headAlpha;
                    xon.trailCol[last * 3 + 1] = hcg * headAlpha;
                    xon.trailCol[last * 3 + 2] = hcb * headAlpha;
                }
            }
        }
        const n = _drawHead ? bodyLen + 1 : bodyLen;
        xon.trailGeo.setDrawRange(0, Math.min(n, XON_TRAIL_LENGTH));
        xon.trailGeo.attributes.position.needsUpdate = true;
        xon.trailGeo.attributes.color.needsUpdate = true;
        } // end normal trail else-block

        // T37 fallback: if trail body was too short for the boost loop to execute
        // (e.g. tick 1 with only 1 trail entry), synthesize boost from flashT so
        // the guard sees a nonzero value whenever flashT is nonzero.
        if (xon._lastTrailFlashBoost <= 0 && xon.flashT > 0) {
            xon._lastTrailFlashBoost = xon.flashT * 0.4;
        }
    }
}

// Emit a gluon between two tet faces along oct edges
// Clear historical trail segments along gluon edge so the laser beam pops.
// Sets frozen positions to NaN for any trail vertex at the gluon edge nodes,
// which breaks the line at those points (Three.js skips NaN vertices).
function _clearTrailsAlongEdges(edgeNodes) {
    const nodeSet = new Set(edgeNodes);
    const _nan3 = [NaN, NaN, NaN];
    for (const xon of _demoXons) {
        if (!xon.alive || !xon.trail || xon.trail.length < 2) continue;
        // Skip the gluon xon itself — don't clear its own trail
        if (xon._mode === 'gluon') continue;
        for (let i = 0; i < xon.trail.length; i++) {
            if (nodeSet.has(xon.trail[i].node)) {
                xon.trail[i].pos = _nan3;
            }
        }
    }
}

function _emitGluon(fromFace, toFace) {
    const fdFrom = _nucleusTetFaceData[fromFace];
    const fdTo = _nucleusTetFaceData[toFace];
    if (!fdFrom || !fdTo || !_octNodeSet) return;

    // Find shared oct nodes between the two faces
    const fromOctNodes = fdFrom.allNodes.filter(n => _octNodeSet.has(n));
    const toOctNodes = fdTo.allNodes.filter(n => _octNodeSet.has(n));
    const shared = fromOctNodes.filter(n => toOctNodes.includes(n));

    if (shared.length === 0) {
        // No shared nodes — need 2-hop path through oct
        // Find a bridging oct node connected to both
        for (const fn of fromOctNodes) {
            for (const tn of toOctNodes) {
                const pid = pairId(fn, tn);
                const scId = scPairToId.get(pid);
                if (scId !== undefined) {
                    // Direct oct edge exists
                    const sprite = _createGluonSprite();
                    if (pos[fn]) sprite.position.set(pos[fn][0], pos[fn][1], pos[fn][2]);
                    _demoGluons.push({
                        fromFace, toFace,
                        path: [fn, tn],
                        step: 0,
                        scIds: [scId],
                        sprite: sprite,
                    });
                    // Clear historical trails along this edge for laser pop effect
                    _clearTrailsAlongEdges([fn, tn]);
                    return;
                }
            }
        }
    } else {
        // Shared node — gluon is a zero-hop bridge (instant)
        // Oct SCs will be added when the oct is revealed (all 8 faces visited).
        // Don't add individual oct SCs here — let the oct reveal handle it atomically.
    }
}

// Advance all active gluons one step. Returns true if any SCs were changed.
// Gluons also negotiate with the vacuum — oct SCs are validated before adding.
function _advanceGluons() {
    let changed = false;
    for (let i = _demoGluons.length - 1; i >= 0; i--) {
        const g = _demoGluons[i];
        if (g.step < g.path.length - 1) {
            g.step++;
            const toNode = g.path[g.step];
            // Negotiate with vacuum before materializing oct SC
            const scId = g.scIds[g.step - 1];
            if (scId !== undefined && !activeSet.has(scId) && !xonImpliedSet.has(scId)) {
                if (canMaterialiseQuick(scId)) {
                    xonImpliedSet.add(scId);
                    _scAttribution.set(scId, { reason: 'gluonAdvance', tick: _demoTick });
                    stateVersion++; // invalidate cache
                    changed = true;
                }
                // If vacuum rejects, gluon still moves visually
            }
            // Move sprite
            if (g.sprite && pos[toNode]) {
                g.sprite.position.set(pos[toNode][0], pos[toNode][1], pos[toNode][2]);
            }
        } else {
            // Gluon arrived — remove
            if (g.sprite) {
                scene.remove(g.sprite);
                g.sprite.material.dispose();
            }
            _demoGluons.splice(i, 1);
        }
    }
    return changed;
}

// Clean up all demo 3.0 xons and gluons (immediate, for stop/reset)
function _cleanupDemo3() {
    for (const xon of _demoXons) {
        if (xon.alive) _destroyXon(xon);
        _finalCleanupXon(xon);
    }
    _demoXons = [];
    _gluonStoredPairs = 0;
    for (const g of _demoGluons) {
        if (g.sprite) { scene.remove(g.sprite); g.sprite.material.dispose(); }
    }
    _demoGluons = [];
    _demoPrevFaces = new Set();
}

// Map speed slider (1-100) to demo interval: 1→2000ms (2s cycle), 50→~60ms, 100→uncapped
function _getDemoIntervalMs() {
    const slider = document.getElementById('excitation-speed-slider');
    if (!slider) return 2000; // default = slowest
    const t = +slider.value / 100;
    if (t >= 1.0) return 0; // 100% = uncapped, as fast as possible
    return Math.max(4, Math.round(Math.exp(Math.log(2000) * (1 - t) + Math.log(4) * t)));
}
function _demoUncappedLoop() {
    if (!_demoActive || _demoInterval || _demoPaused) { _demoUncappedId = null; return; }
    demoTick().then(() => {
        if (_demoActive && !_demoInterval && !_demoPaused) {
            _demoUncappedId = setTimeout(_demoUncappedLoop, 0);
        } else {
            _demoUncappedId = null;
        }
    }).catch(err => {
        console.error('[uncapped loop] demoTick error:', err);
        // Don't kill the loop — schedule next tick anyway
        if (_demoActive && !_demoInterval && !_demoPaused) {
            _demoUncappedId = setTimeout(_demoUncappedLoop, 0);
        }
    });
}

/**
 * Build a full deuteron 8-tick schedule from two patterns.
 * Each entry has protonFaces[3] and neutronFaces[3], plus quark-type assignments:
 *   protonFaces[0] = anchor (proton-down), [1],[2] = followers (proton-up)
 *   neutronFaces[0] = anchor (neutron-up), [1],[2] = followers (neutron-down)
 */
function buildDeuteronSchedule(patP, patN, D4) {
    const A = [1, 3, 6, 8];
    const B = [2, 4, 5, 7];
    const dA_p = D4[patP.aDerang], dB_p = D4[patP.bDerang];
    const ancA_p = patP.anchorsA[0], ancB_p = patP.anchorsB[0];
    const dA_n = D4[patN.aDerang], dB_n = D4[patN.bDerang];
    const ancA_n = patN.anchorsA[0], ancB_n = patN.anchorsB[0];

    const schedule = [];
    for (let i = 0; i < 4; i++) {
        // Even tick 2i: proton on A, neutron on B (anti-phase)
        schedule.push({
            protonFaces: [A[ancA_p[i]], A[i], A[dA_p[i]]],
            neutronFaces: [B[ancB_n[i]], B[i], B[dB_n[i]]],
            // Quark-type map: face → quarkType
            faceQuarks: {
                [A[ancA_p[i]]]: 'pd',   // proton anchor (hook)
                [A[i]]: 'pu1',           // proton follower-1 (ham CW)
                [A[dA_p[i]]]: 'pu2',     // proton follower-2 (ham CCW)
                [B[ancB_n[i]]]: 'nu',    // neutron anchor (fork)
                [B[i]]: 'nd1',           // neutron follower-1 (ham CW)
                [B[dB_n[i]]]: 'nd2',     // neutron follower-2 (ham CCW)
            },
        });
        // Odd tick 2i+1: proton on B, neutron on A
        schedule.push({
            protonFaces: [B[ancB_p[i]], B[i], B[dB_p[i]]],
            neutronFaces: [A[ancA_n[i]], A[i], A[dA_n[i]]],
            faceQuarks: {
                [B[ancB_p[i]]]: 'pd',
                [B[i]]: 'pu1',
                [B[dB_p[i]]]: 'pu2',
                [A[ancA_n[i]]]: 'nu',
                [A[i]]: 'nd1',
                [A[dA_n[i]]]: 'nd2',
            },
        });
    }
    return schedule;
}


/**
 * Start the demand-driven demo: sets up lattice, runs high-speed loop.
 * Called AFTER simulateNucleus() has built the octahedron.
 * No schedule or windows — xons self-assign via _scoreFaceOpportunity.
 */
function startDemoLoop() {
    // Init visit counters (demand-driven — no schedule needed)
    _demoVisits = {};
    _actualizationVisits = {};
    _faceEdgeEpoch = {};
    _faceWasActualized = {};
    for (let f = 1; f <= 8; f++) {
        _demoVisits[f] = { pu1: 0, pu2: 0, pd: 0, nd1: 0, nd2: 0, nu: 0, total: 0 };
        _actualizationVisits[f] = { pu1: 0, pu2: 0, pd: 0, nd1: 0, nd2: 0, nu: 0 };
    }
    _demoTick = 0;
    _planckSeconds = 0;
    _globalModeStats = { oct: 0, tet: 0, idle_tet: 0, weak: 0, gluon: 0 };
    _globalRoleStats = { pu1:0, pu2:0, pd:0, nd1:0, nd2:0, nu:0, oct:0, gluon:0, weak:0 };
    // Allow fixed seed via URL param (?seed=0xABCD1234) or global override for replay.
    const _urlSeed = new URLSearchParams(window.location.search).get('seed');
    if (typeof _forceSeed !== 'undefined' && _forceSeed !== null) {
        _runSeed = _forceSeed >>> 0;
    } else if (_urlSeed) {
        _runSeed = parseInt(_urlSeed, 16) >>> 0;
    } else {
        _runSeed = (Math.random() * 0xFFFFFFFF) >>> 0;
    }
    _maxTickReached = 0;
    _searchStartTime = performance.now();
    _totalBacktrackRetries = 0;
    _bestPathFingerprint = '';
    _relayEscapes = 0;
    _relayEnumTotal = 0;
    console.log(`%c[DEMO] seed=0x${_runSeed.toString(16).padStart(8,'0')}`, 'color:cyan;font-weight:bold');
    _balanceHistory = [];
    _bfsReset(); // fresh demo = clean BFS + ledger
    _lastAutosavePeak = 0; // autosave not cleared — new run overwrites naturally at tick 100
    _btSnapshots.length = 0;
    _btColdBoundary = 0;
    _btColdSnapshots.length = 0;
    _tickLog.length = 0;
    _tickLogLastGuards = {};
    _movieFrames.length = 0;
    _lastMoviePos = null;
    _replayCursor = -1;
    _demoReversing = false;
    if (_reverseInterval) { clearInterval(_reverseInterval); _reverseInterval = null; }
    _demoTetAssignments = 0;
    _demoPauliViolations = 0;
    _demoSpreadViolations = 0;
    // Edge balance + ejection balance reset (will be re-initialized after oct discovery)
    _octEdgeSet = null;
    _edgeBalance = null;
    _ejectionBalance = null;
    if (typeof _rlTemporalState !== 'undefined') _rlTemporalState.reset();
    _demoTypeBalanceHistory = [];
    _demoVisitedFaces = new Set();  // track which faces have been activated
    _demoOctRevealed = false;       // oct only renders once all 8 faces visited
    // Clean up any existing xon visuals before reinit
    for (const xon of _demoXons) {
        if (xon.group) { scene.remove(xon.group); }
        if (xon.sparkMat) xon.sparkMat.dispose();
        if (xon.trailLine) scene.remove(xon.trailLine);
        if (xon.trailGeo) xon.trailGeo.dispose();
        // Clean up weak overlay trail (NormalBlending mesh) — orphaned ones paint black
        if (xon._weakTrailLine) { scene.remove(xon._weakTrailLine); }
        if (xon._weakTrailLine && xon._weakTrailLine.geometry) xon._weakTrailLine.geometry.dispose();
        if (xon._weakTrailLine && xon._weakTrailLine.material) xon._weakTrailLine.material.dispose();
        xon._weakTrailLine = null; xon._weakTrailPos = null; xon._weakTrailCol = null;
    }
    _demoXons = [];
    _demoGluons = [];               // Demo 3.1: clear gluon pool
    _demoPrevFaces = new Set();     // Demo 3.1: no previous window faces
    _demoActive = true;
    _demoPaused = false;
    if (typeof _updateLatticeSliderLock === 'function') _updateLatticeSliderLock();
    if (typeof _setSimUIActive === 'function') _setSimUIActive(true);

    // Apply demo-mode visual defaults only on the FIRST startDemoLoop() call
    // (user clicked play). Subsequent seed restarts within a sweep preserve
    // whatever the user has set. Source: DEMO_VISUAL_DEFAULTS in flux-demo-state.js
    if (!_demoOpDefaultsApplied) {
        _demoOpDefaultsApplied = true;
        for (const [id, val] of DEMO_VISUAL_DEFAULTS) {
            const el = document.getElementById(id);
            if (el && +el.value !== val) { el.value = val; el.dispatchEvent(new Event('input')); }
        }
    }

    // Stop excitation clock (we drive our own loop)
    if (typeof stopExcitationClock === 'function') stopExcitationClock();

    // Do NOT pre-open all 8 tet SCs — only 1-3 tets can coexist at a time.
    // Tets activate/deactivate per window via xonImpliedSet, and the
    // solver re-runs each time so spheres physically respond to geometry.
    // Oct emerges visually once all 8 faces have been visited.
    bumpState();
    const pSolved = detectImplied();
    applyPositions(pSolved);
    // Clear implied SCs before void check — FCC rest geometry has some SC pairs
    // at unit distance which detectImplied promotes. These are phantom voids at t=0.
    impliedSet.clear(); impliedBy.clear();
    bumpState(); // invalidate void cache so updateVoidSpheres re-evaluates
    updateVoidSpheres();

    // Hide xon sparks/trails
    const quarks = NucleusSimulator?.quarkExcitations || [];
    for (const q of quarks) {
        if (q.spark) q.spark.visible = false;
        if (q.trailLine) q.trailLine.visible = false;
    }

    // Demo status (hidden but kept for compatibility)
    const ds = document.getElementById('demo-status');
    if (ds) ds.style.display = 'none';

    // Update left panel header
    const dpTitle = document.getElementById('dp-title');
    if (dpTitle) dpTitle.textContent = '0 Flux events';

    // Camera defaults are set in flux-ui.js — don't override user preferences here.

    // Speed and trail lifespan defaults are set in HTML — don't override here.
    // Spawn 6 persistent xons at center node
    _initPersistentXons();
    _nucleusNodeSet = null; // reset so lazy builder re-runs on next demo
    _openingPhase = true; // 2-tick opening choreography (ticks 0-1)
    _octWindingDirection = null; // reset — will be set by merry-go-round at tick 1

    // Clear any orphaned timers that the speed slider dispatch (above) may have
    // started — the slider handler sees _demoActive=true and starts a loop, but
    // startDemo() needs exactly ONE loop. Without this, pause can never clear the
    // orphaned timer and the demo appears to ignore the pause button.
    if (_demoInterval) { clearInterval(_demoInterval); _demoInterval = null; }
    if (_demoUncappedId) { clearTimeout(_demoUncappedId); _demoUncappedId = null; }

    const intervalMs = _getDemoIntervalMs();
    if (intervalMs === 0) {
        // Uncapped: self-scheduling async loop (as fast as GPU/CPU allows)
        _demoUncappedId = setTimeout(_demoUncappedLoop, 0);
        console.log(`[demo] Pattern demo started UNCAPPED (max speed)`);
    } else {
        _demoInterval = setInterval(demoTick, intervalMs);
        console.log(`[demo] Pattern demo started at ${intervalMs}ms interval`);
    }

    // Auto-run unit tests — HALT DEMO if any test fails (BFS test mode: warn but don't halt)
    try {
        const testResult = runDemo3Tests();
        if (testResult.failed.length > 0) {
            if (_bfsTestActive) {
                console.warn(`[BFS TEST] ${testResult.failed.length} unit test(s) failed — continuing anyway: ${testResult.failed.join(', ')}`);
            } else {
                console.error(`[demo] HALTED: ${testResult.failed.length} test(s) failed: ${testResult.failed.join(', ')}`);
                stopDemo();
                return;
            }
        }
    } catch (e) { console.warn('[demo] Test suite error:', e); }

    // Activate live guards (T19, T21, T26, T27) — start with null during grace
    if (typeof _liveGuards !== 'undefined') {
        for (const entry of LIVE_GUARD_REGISTRY) {
            const g = _liveGuards[entry.id];
            if (!g) continue;
            g.ok = true;
            g.msg = '';
            g.failed = false;
            // Re-apply init fields so state is clean across demo restarts
            if (entry.init) Object.assign(g, entry.init);
        }
        if (typeof _liveGuardActivated !== 'undefined') _liveGuardActivated = false;
        _liveGuardsActive = true;
        _liveGuardRender();
    }
}

// L2/L3 toggle for demo mode — switches lattice and restarts demo
function _updateDemoLatticeButtons() {
    const lv = +document.getElementById('lattice-slider').value;
    for (const [id, level] of [['demo-l2-btn', 2], ['demo-l3-btn', 3], ['demo-l4-btn', 4]]) {
        const btn = document.getElementById(id);
        if (!btn) continue;
        const active = lv === level;
        btn.style.background = active ? '#1a2a3a' : '#0a1a2a';
        btn.style.color = active ? '#88bbdd' : '#556677';
        btn.style.borderColor = active ? '#3a5a7a' : '#2a3a4a';
    }
}
function _setDemoLattice(level) {
    const slider = document.getElementById('lattice-slider');
    if (!slider || +slider.value === level) return;
    // Stop demo, change lattice, re-simulate, restart demo
    stopDemo();
    slider.value = level;
    slider.dispatchEvent(new Event('input'));
    // Call simulateNucleus directly (not via button click) and restart demo
    setTimeout(() => {
        NucleusSimulator.simulateNucleus();
        setTimeout(() => startDemoLoop(), 150);
    }, 50);
}


// ─── 2-Tick Opening Choreography ─────────────────────────────────────────
// The oct is DISCOVERED through choreography, not imposed.
// Tick 0: 4 xons move center → 4 base neighbors below center (z-axis).
//         2 free xons move to other base neighbors above center.
//         The 4 below-z nodes form a shortcut-connected equatorial square.
// Tick 1: Discover the oct from the 6 xon positions.
//         4 equatorial xons merry-go-round via cage SCs.
//         2 free xons stay within 1-step of center.
function _executeOpeningTick(occupied) {
    const center = _octSeedCenter;

    if (_demoTick === 0) {
        // ── Tick 0: 4 xons → below center (y-axis), 2 → above center ──
        // Deterministic: sort by y, lowest 4 go to equatorial formation.
        const cy = pos[center][1];
        const allNbs = baseNeighbors[center].slice();
        const belowY = allNbs.filter(nb => pos[nb.node][1] < cy);
        const aboveY = allNbs.filter(nb => pos[nb.node][1] >= cy);
        // Sort each group by y for determinism (lowest first)
        belowY.sort((a, b) => pos[a.node][1] - pos[b.node][1]);
        aboveY.sort((a, b) => pos[a.node][1] - pos[b.node][1]);

        // First 4 xons → below center (equatorial square formation)
        for (let i = 0; i < 4 && i < belowY.length; i++) {
            const xon = _demoXons[i];
            const pick = belowY[i];
            _executeOctMove(xon, { node: pick.node, dirIdx: pick.dirIdx, _needsMaterialise: false, _scId: undefined });
        }
        // Remaining 2 xons: move to above-center neighbors
        for (let i = 4; i < 6; i++) {
            const xon = _demoXons[i];
            if (xon._mode === 'oct_formation') {
                const aboveIdx = i - 4; // 0 and 1
                if (aboveIdx < aboveY.length) {
                    _executeOctMove(xon, { node: aboveY[aboveIdx].node, dirIdx: aboveY[aboveIdx].dirIdx, _needsMaterialise: false, _scId: undefined });
                }
            }
        }
        // All 6 xons start as weak — they must enter oct via the merry-go-round
        for (const xon of _demoXons) {
            if (!xon.alive) continue;
            xon._mode = 'weak';
            xon.col = WEAK_FORCE_COLOR;
            if (xon.sparkMat) xon.sparkMat.color.setHex(WEAK_FORCE_COLOR);
        }

    } else if (_demoTick === 1) {
        // ── Tick 1: Discover oct from the 6 xon positions, then merry-go-round ──
        const xonNodes = _demoXons.map(x => x.node);
        const xonNodeSet = new Set(xonNodes);
        const centerBaseNbSet = new Set(baseNeighbors[center].map(nb => nb.node));

        // Find oct candidates whose equator is a subset of our 6 xon positions
        const validOcts = [];
        for (const sc of _localScNeighbors(center)) {
            const pole = sc.a === center ? sc.b : sc.a;
            const equator = baseNeighbors[pole].map(nb => nb.node).filter(n => centerBaseNbSet.has(n));
            if (equator.length !== 4) continue;
            if (!equator.every(n => xonNodeSet.has(n))) continue;
            const cageSCIds = [];
            for (let i = 0; i < equator.length; i++)
                for (let j = i + 1; j < equator.length; j++) {
                    const scId = scPairToId.get(pairId(equator[i], equator[j]));
                    if (scId !== undefined && !(baseNeighbors[equator[i]] || []).some(nb => nb.node === equator[j]))
                        cageSCIds.push(scId);
                }
            if (cageSCIds.length !== 4) continue;
            validOcts.push({ pole, equator, cageSCIds, octNodes: new Set([center, pole, ...equator]) });
        }

        if (validOcts.length === 0) {
            console.error('[opening] No valid oct among 6 xon positions!');
            return;
        }

        // Deterministic: pick the oct whose equator has lowest average y
        // (the one the 4 below-center xons naturally form)
        const chosen = validOcts.reduce((best, oct) => {
            const yAvg = oct.equator.reduce((s, n) => s + pos[n][1], 0) / 4;
            const bestYAvg = best.equator.reduce((s, n) => s + pos[n][1], 0) / 4;
            return yAvg < bestYAvg ? oct : best;
        });
        const equatorSet = new Set(chosen.equator);

        // ── Set up all oct data structures ──
        _octNodeSet = chosen.octNodes;
        _octSCIds = chosen.cageSCIds;

        // Camera position set at demo start — don't override here.

        // Chain-walk equator into cycle order
        const eq = chosen.equator.slice();
        const ordered = [eq[0]], used = new Set([0]), scCycle = [];
        for (let step = 0; step < 3; step++) {
            const cur = ordered[ordered.length - 1];
            for (let j = 0; j < eq.length; j++) {
                if (used.has(j)) continue;
                const scId = scPairToId.get(pairId(cur, eq[j]));
                if (scId !== undefined && chosen.cageSCIds.includes(scId)) {
                    ordered.push(eq[j]); scCycle.push(scId); used.add(j); break;
                }
            }
        }
        const closeScId = scPairToId.get(pairId(ordered[3], ordered[0]));
        if (closeScId !== undefined) scCycle.push(closeScId);
        _octEquatorCycle = ordered;
        _octCageSCCycle = scCycle;

        // Build antipodal map: eq[0]↔eq[2], eq[1]↔eq[3], pole↔pole
        _octAntipodal = new Map();
        _octAntipodal.set(ordered[0], ordered[2]);
        _octAntipodal.set(ordered[2], ordered[0]);
        _octAntipodal.set(ordered[1], ordered[3]);
        _octAntipodal.set(ordered[3], ordered[1]);
        const poles = [];
        for (const n of _octNodeSet) {
            if (!ordered.includes(n)) poles.push(n);
        }
        if (poles.length === 2) {
            _octAntipodal.set(poles[0], poles[1]);
            _octAntipodal.set(poles[1], poles[0]);
        }

        // Find oct void
        _octVoidIdx = -1;
        for (let vi = 0; vi < voidNeighborData.length; vi++) {
            const v = voidNeighborData[vi];
            if (v.type === 'oct' && v.nbrs.every(n => _octNodeSet.has(n))) { _octVoidIdx = vi; break; }
        }

        // Discover adjacent tet voids → face IDs
        _nucleusTetFaceData = {};
        const adjTets = [];
        for (let vi = 0; vi < voidNeighborData.length; vi++) {
            const v = voidNeighborData[vi];
            if (v.type !== 'tet') continue;
            const inOct = v.nbrs.filter(n => _octNodeSet.has(n));
            if (inOct.length !== 3) continue;
            adjTets.push({ voidIdx: vi, octNodes: inOct, extNode: v.nbrs.find(n => !_octNodeSet.has(n)),
                allNodes: [...v.nbrs], scIds: [...v.scIds] });
        }
        const tetGroup = new Map();
        if (adjTets.length > 0) {
            tetGroup.set(adjTets[0].voidIdx, 'A');
            const queue = [adjTets[0]];
            while (queue.length > 0) {
                const cur = queue.shift();
                const otherG = tetGroup.get(cur.voidIdx) === 'A' ? 'B' : 'A';
                for (const other of adjTets) {
                    if (tetGroup.has(other.voidIdx)) continue;
                    if (cur.octNodes.filter(n => other.octNodes.includes(n)).length === 2) {
                        tetGroup.set(other.voidIdx, otherG); queue.push(other);
                    }
                }
            }
            for (const t of adjTets) if (!tetGroup.has(t.voidIdx)) tetGroup.set(t.voidIdx, 'A');
        }
        const gA = [1,3,6,8], gB = [2,4,5,7];
        const tA = adjTets.filter(t => tetGroup.get(t.voidIdx) === 'A');
        const tB = adjTets.filter(t => tetGroup.get(t.voidIdx) === 'B');
        for (let i = 0; i < tA.length && i < gA.length; i++) {
            const t = tA[i];
            _nucleusTetFaceData[gA[i]] = { voidIdx: t.voidIdx, allNodes: t.allNodes, extNode: t.extNode,
                scIds: t.scIds, cycle: [t.octNodes[0], t.extNode, t.octNodes[1], t.octNodes[2]], group: 'A' };
        }
        for (let i = 0; i < tB.length && i < gB.length; i++) {
            const t = tB[i];
            _nucleusTetFaceData[gB[i]] = { voidIdx: t.voidIdx, allNodes: t.allNodes, extNode: t.extNode,
                scIds: t.scIds, cycle: [t.octNodes[0], t.extNode, t.octNodes[1], t.octNodes[2]], group: 'B' };
        }

        console.log(`[opening] Oct discovered: equator=[${ordered}], pole=${chosen.pole}, ${adjTets.length} tets`);

        // Build edge balance + ejection balance tracking
        _initEdgeBalance();
        _initEjectionBalance();

        // Build wavefunction surface and concentric shells now that oct is known
        if (typeof buildWavefunction === 'function') buildWavefunction();
        if (typeof buildBranes === 'function') buildBranes();

        // ── Merry-go-round: equatorial xons rotate one position via cage SCs ──
        // This establishes the matter/antimatter winding direction (CW = matter).
        const eqXonMap = new Map();
        for (let i = 0; i < 6; i++) if (equatorSet.has(xonNodes[i])) eqXonMap.set(xonNodes[i], i);

        const choreoMoves = [];
        for (let i = 0; i < 4; i++) {
            const src = _octEquatorCycle[i], dst = _octEquatorCycle[(i+1)%4];
            const scId = _octCageSCCycle[i];
            const xon = _demoXons[eqXonMap.get(src)];
            const isActive = activeSet.has(scId) || impliedSet.has(scId) || xonImpliedSet.has(scId);
            let dirIdx = 0;
            const nb = baseNeighbors[xon.node]?.find(nb => nb.node === dst);
            if (nb) dirIdx = nb.dirIdx;
            choreoMoves.push({ xon, target: { node: dst, dirIdx, _needsMaterialise: !isActive, _scId: scId } });
        }
        for (const { xon, target } of choreoMoves) _executeOctMove(xon, target);

        // Lock winding direction — merry-go-round follows cycle order = CW
        _octWindingDirection = 'CW';

        // Free xons: handled by normal phases (free to choose any action)

        // ── Transition all xons out of oct_formation mode ──
        for (const xon of _demoXons) {
            if (xon._mode !== 'oct_formation') continue;
            if (equatorSet.has(xon.node) || _octNodeSet.has(xon.node)) {
                // On an oct node (equatorial or arrived via 1-hop) → oct mode
                _clearModeProps(xon);
                xon._mode = 'oct';
                xon.col = 0xffffff;
                if (xon.sparkMat) xon.sparkMat.color.setHex(0xffffff);
            } else {
                // Not on oct node → weak, navigate back via PHASE 0.5
                xon._mode = 'weak';
                xon._t60Ejected = true;
                xon.col = WEAK_FORCE_COLOR;
                if (xon.sparkMat) xon.sparkMat.color.setHex(WEAK_FORCE_COLOR);
            }
        }
    }
}

// ── Periodic garbage collection (every 10 advancement ticks) ──
// Trims auxiliary in-memory structures that are NOT needed for replay or backtracking.
// NEVER touches:
//   _btSnapshots  — needed for backtracker rewind + council replay capture
//   _sweepSeedMoves — needed for golden council move recording
function _gc10() {
    let trimmed = 0;
    // 1. Cap balance history (display-only, not needed for replay)
    if (_balanceHistory.length > 256) {
        trimmed += _balanceHistory.length - 256;
        _balanceHistory.splice(0, _balanceHistory.length - 256);
    }
    // 2. Cap tick log (export-only)
    if (_tickLog.length > 500) {
        trimmed += _tickLog.length - 500;
        _tickLog.splice(0, _tickLog.length - 500);
    }
    // 3. Cap movie frames (export-only)
    if (_movieFrames.length > 500) {
        trimmed += _movieFrames.length - 500;
        _movieFrames.splice(0, _movieFrames.length - 500);
    }
    // 4. Cap search traversal log (debug/analysis-only)
    if (_searchTraversalLog && _searchTraversalLog.length > 1000) {
        trimmed += _searchTraversalLog.length - 1000;
        _searchTraversalLog.splice(0, _searchTraversalLog.length - 1000);
    }
    // 5. Cap type balance history (display-only)
    if (_demoTypeBalanceHistory.length > 64) {
        trimmed += _demoTypeBalanceHistory.length - 64;
        _demoTypeBalanceHistory.splice(0, _demoTypeBalanceHistory.length - 64);
    }
    if (trimmed > 0) console.log(`[GC] tick ${_demoTick}: trimmed ${trimmed} stale entries`);
}

async function demoTick() {
    if (!_demoActive || _demoPaused) return;
    if (simHalted) {
        // Tournament: fire callback on halt so GA can score the failed trial
        if (typeof _tournamentTickCheck === 'function') _tournamentTickCheck();
        return;
    }
    if (_tickInProgress) return; // previous async tick still running
    _tickInProgress = true;
    const _tickT0 = performance.now();
    try {

    // ── REPLAY CURSOR: restore from _btSnapshots instead of live choreography ──
    if (_replayCursor >= 0 && _replayCursor < _btSnapshots.length) {
        const snap = _btSnapshots[_replayCursor];
        // On LAST snapshot: switch off hard-stop before restoring,
        // so guard failures on this tick trigger backtracking, not corruption halt.
        if (_replayCursor === _btSnapshots.length - 1 && _sweepReplayActive) {
            _sweepReplayActive = false;
            _sweepReplayMember = null;
            _guardHardStop = false;
            console.log(`%c[REPLAY] Final snapshot — switching to live mode`, 'color:#66ccff;font-weight:bold');
        }
        // Guard snapshot BEFORE restore (mirrors live: snapshot → moves → check)
        if (typeof _liveGuardSnapshot === 'function') _liveGuardSnapshot();
        _btRestoreSnapshot(snap);
        // No re-snapshot needed — cold snapshots stay raw. At save time,
        // _btColdSnapshots provides pre-serialized data for indices < _btColdBoundary.
        _replayCursor++;
        simHalted = false;
        // Apply tet coloring BEFORE guards — T58 reads _ruleAnnotations
        if (typeof _applyTetColoring === 'function') _applyTetColoring(false);
        // Guards fire during replay — detect corruption
        if (typeof _liveGuardCheck === 'function') _liveGuardCheck();
        if (simHalted) {
            console.error(`[REPLAY] Guard failure at tick ${_demoTick} during snapshot replay`);
            _tickInProgress = false;
            return;
        }
        // Update display (throttled by setInterval rate — no extra throttle needed)
        if (!_testRunning) _playbackUpdateDisplay();
        // Check if cursor exhausted — transition to live
        if (_replayCursor >= _btSnapshots.length) {
            _replayCursor = -1;
            _maxTickReached = _demoTick;
            console.log(`%c[REPLAY] Cursor reached end — transitioning to live play at tick ${_demoTick}`, 'color:#66ccff;font-weight:bold');
        }
        _tickInProgress = false;
        return;
    }

    // ── Bucketed blacklist: ensure current tick's bucket is loaded ──
    if (_sweepActive && _blBucketVersion >= 1 && typeof _blEnsureTick === 'function') {
        const _lvl = typeof latticeLevel !== 'undefined' ? latticeLevel : 2;
        await _blEnsureTick(_lvl, _demoTick);
        // Eager prefetch: when within 8 ticks of next bucket boundary, prefetch it
        const nextBucket = Math.floor((_demoTick + 8) / _blBucketSize);
        const curBucket = Math.floor(_demoTick / _blBucketSize);
        if (nextBucket !== curBucket && !_blLoadedBuckets.has(nextBucket) && nextBucket < _blBucketCount) {
            console.log(`[BL] Tick ${_demoTick} approaching boundary → prefetching bucket ${nextBucket}`);
            _blPrefetchBucket(_lvl, nextBucket); // fire-and-forget
        }
    }

    // ── BACKTRACKING RETRY LOOP ──
    // Save state before tick, run choreography, check guards.
    // If T19/T20 violation → rewind, exclude offending moves, retry.
    // Reset per-tick RL reward accumulators so they don't leak across ticks
    if (typeof _ppoTetCompletionsThisTick !== 'undefined') {
        _ppoTetCompletionsThisTick = 0;
        _ppoGuardFailedThisTick = false;
        _ppoDeformationThisTick = false;
        _ppoBacktracksThisTick = 0;
    }
    const _inReplayPhase = _sweepReplayActive && _sweepReplayMember
        && _demoTick <= _sweepReplayMember.peak;
    // Set trail baseline BEFORE snapshot so tick-start snapshots have _tDelta=null
    // (no moves have happened yet). Reset again inside retry loop after backtrack restores.
    for (const xon of _demoXons) { xon._trailLenAtTickStart = xon.trail ? xon.trail.length : 0; }
    _btSaveSnapshot();
    _rewindRequested = false;
    _rewindViolation = null;

    // If we're in an active BFS and this tick is at or near the failure tick,
    // activate backtracking so exclusions and rotations apply during forward replay.
    if (_bfsFailTick >= 0) {
        _btActive = true;
    }

    // No artificial cap — L2 lattice is inherently finite.
    // Only true failure: BFS reaches t=0 (canary for impossible rules).
    for (let _btAttempt = 0; ; _btAttempt++) {
    // Yield to event loop every 8 retries so the browser can paint & stay responsive
    if (_btAttempt > 0 && _btAttempt % 2 === 0) {
        // Yield to event loop every 2 retries so browser stays responsive.
        // Heavy panel updates handled by the 1/sec timer.
        _demoPanelDirty = true;
        updateXonPanel();
        await new Promise(r => setTimeout(r, 0));
    }

    // Reset per-tick RL reward accumulators at each retry attempt so failed
    // retries don't leak completions/penalties into the successful attempt.
    if (typeof _ppoTetCompletionsThisTick !== 'undefined') {
        _ppoTetCompletionsThisTick = 0;
        _ppoGuardFailedThisTick = false;
        _ppoDeformationThisTick = false;
        _ppoBacktracksThisTick = 0;
    }
    // Seed PRNG from tick number + retry context. The retry count and BFS layer
    // ensure each backtracker attempt gets a different PRNG sequence, so type
    // selection, shuffles, and other randomized decisions explore new paths.
    // During PPO training, use tick-only seed so the RL model sees consistent
    // random outcomes for the same state (reproducibility for policy learning).
    if (typeof _ppoTraining !== 'undefined' && _ppoTraining) {
        _sRngSeed(_demoTick * 65537 + _bfsLayer * 31);
    } else {
        _sRngSeed(_demoTick * 65537 + _btRetryCount * 997 + _bfsLayer * 31 + _bfsLayerRetries * 7919);
    }
    // Clear stale movement flags from previous tick so WB processing isn't blocked
    for (const xon of _demoXons) { xon._movedThisTick = false; xon._evictedThisTick = false; xon._trailLenAtTickStart = xon.trail ? xon.trail.length : 0; }
    // Gluon mode is now sticky — cleared in PHASE 2a only when cage is stable.
    // (Previously reverted here every tick, allowing gluons to immediately enter tets.)
    _moveRecord.clear(); // T41: clear tick-level move record
    _moveTrace.length = 0; // diagnostic: clear trace for this tick
    // _scAttribution persists across ticks — only cleared on SC deletion

    // Recompute dynamic ejection sets (actualized tets change as SCs activate/deactivate)
    _recomputeActualizedTetNodes();

    // Snapshot xon positions BEFORE advancement for live guard T26/T27
    if (typeof _liveGuardSnapshot === 'function') _liveGuardSnapshot();

    let _solverNeeded = false;

    // ╔══════════════════════════════════════════════════════════════════╗
    // ║  UNIFIED DEMAND-DRIVEN CHOREOGRAPHY (no windows)                ║
    // ║                                                                  ║
    // ║  Window boundary block REMOVED. Face assignment is now           ║
    // ║  demand-driven via PHASE 1.5 (natural completion) + PHASE 2a    ║
    // ║  (decentralized face scoring). Loops complete organically.       ║
    // ║                                                                  ║
    // ║  The demo MUST manage tet SCs in xonImpliedSet and              ║
    // ║  re-solve the lattice so spheres physically respond.             ║
    // ║  Shapes drive spheres (unified architecture).                    ║
    // ╚══════════════════════════════════════════════════════════════════╝

    // ╔══════════════════════════════════════════════════════════════════╗
    // ║  XON-VACUUM NEGOTIATION (every tick)                            ║
    // ║                                                                  ║
    // ║  Each xon attempts one hop per tick. Before hopping:             ║
    // ║  1. Check if the traversed edge is a tet SC                     ║
    // ║  2. If so, ask the vacuum: canMaterialiseQuick(scId)            ║
    // ║  3. If blocked, try excitationSeverForRoom(scId)                ║
    // ║  4. If still blocked, xon's move is rejected (vacuum wins)      ║
    // ║  5. If allowed, commit the SC and run the solver                ║
    // ║                                                                  ║
    // ║  The pattern schedule is advisory. The vacuum has final say.     ║
    // ╚══════════════════════════════════════════════════════════════════╝

    _idleTetManifested = false; // reset per-tick; _startIdleTetLoop sets if new SCs added

    // ── GLUON CREATION: Manifest stored xon pairs when there's room ──
    // Conservation: alive + 2*stored = 6. Pairs spawn on free adjacent oct nodes.
    // Only runs when annihilation is enabled (genesis is the reverse of annihilation).
    if (_annihilationEnabled) {
        const aliveCount = _demoXons.filter(x => x.alive).length;
        if (aliveCount < 6 && _gluonStoredPairs > 0) {
            _manifestXonPair();
        }
    }

    // NOTE: _movedThisTick is NOT reset here. WB movements (scatter, _returnXonToOct, _walkToFace)
    // are real moves that count toward the one-hop-per-tick limit. The flag was already cleared
    // at tick start (line above snapshot). Xons moved during WB won't be moved again by the planner.

    let occupied = _occupiedNodes();

    // ── Opening phase: scripted 2-tick formation choreography ──
    let _skipNormalPhases = false;
    let _pT5 = performance.now(); // profiling anchor (updated by PHASE 5 if normal phases run)
    if (_openingPhase) {
        if (_demoTick < 2) {
            _executeOpeningTick(occupied);
            // Don't skip normal phases — free xons need coordinated movement
            // Propagate per-xon _solverNeeded flags (set by _executeOctMove when SCs
            // are actually materialized). Only set global _solverNeeded when real
            // vacuum deformation occurred — tick 0 uses base edges only (no SCs).
            for (const xon of _demoXons) {
                if (xon._solverNeeded) { _solverNeeded = true; xon._solverNeeded = false; }
            }
            occupied = _occupiedNodes(); // refresh after opening moves
        } else {
            _openingPhase = false;
            _demoOctRevealed = true; // Oct formed — enable oct-dependent guards (T96)
            _ticksSinceLastQuark = 0; // Start the quark clock when oct forms, not at tick 0
        }
    }

    if (!_skipNormalPhases) {
    // ══════════════════════════════════════════════════════════════════
    //  COORDINATED MOVE PLANNER
    //  All moves are planned before execution to prevent Pauli violations.
    //  Priority: tet/idle_tet (fixed path) > oct (flexible).
    // ══════════════════════════════════════════════════════════════════

    const planned = new Set();  // globally reserved destination nodes
    let anyMoved = false;
    const _pT = performance.now(); _profPhases.wb += _pT - _tickT0; // phase timer anchor (wb = window boundary + setup)

    // T60 consistency: ensure ejected xons stay in weak mode.
    // If any code path set mode='oct' while _t60Ejected > 0, correct it here.
    for (const xon of _demoXons) {
        if (!xon.alive) continue;
        if (xon._t60Ejected && xon._mode !== 'weak') {
            xon._mode = 'weak';
            xon._assignedFace = null;
            xon._quarkType = null;
            xon._loopSeq = null;
            xon._loopStep = 0;
            xon.col = WEAK_FORCE_COLOR;
            if (xon.sparkMat) xon.sparkMat.color.setHex(WEAK_FORCE_COLOR);
        }
    }

    // ── PHASE 0: Pre-check tet/idle_tet xons with blocked next steps ──
    // If a tet/idle_tet xon's next step is blocked by another tet/idle_tet xon
    // (which the oct planner can't move), OR if N-depth lookahead shows the loop
    // leads to a dead end, return the xon to oct mode NOW so PHASE 2's bipartite
    // matching with full lookahead can find it an optimal move.
    {
        let phase0Changed = false;
        for (const xon of _demoXons) {
            if (!xon.alive) continue;
            if (xon._mode !== 'tet' && xon._mode !== 'idle_tet') continue;

            // T60 check: face must be actualized every step.
            // If the vacuum withdrew support (severed a face SC), switch to weak mode.
            // Recolor existing trail segments to purple — no colored trail without an actualized tet.
            // Don't physically move — PHASE 0.5 handles weak xon movement.
            if (xon._assignedFace != null && _nucleusTetFaceData) {
                const fd60 = _nucleusTetFaceData[xon._assignedFace];
                const faceActualized = fd60 && fd60.scIds.every(scId =>
                    activeSet.has(scId) || impliedSet.has(scId) || xonImpliedSet.has(scId));
                if (!faceActualized) {
                    // Trail segments keep their original recorded color.
                    // Only new trail pushes after mode change use WEAK_FORCE_COLOR.
                    // Relinquish face SCs (tight space = delete, sticky space = no-op)
                    _relinquishFaceSCs(xon);
                    // If already on an oct node, go directly to oct mode (no weak ejection needed)
                    if (_octNodeSet && _octNodeSet.has(xon.node)) {
                        _logChoreo(`X${_demoXons.indexOf(xon)} non-actualized face ${xon._assignedFace} → oct (already on cage)`);
                        _clearModeProps(xon);
                        xon._mode = 'oct';
                        xon._assignedFace = null;
                        xon._quarkType = null;
                        xon._loopSeq = null;
                        xon._loopStep = 0;
                        xon.col = 0xffffff;
                        if (xon.sparkMat) xon.sparkMat.color.setHex(0xffffff);
                        if (_flashEnabled) xon.flashT = 1.0;
                    } else {
                        _logChoreo(`X${_demoXons.indexOf(xon)} non-actualized face ${xon._assignedFace} → weak`);
                        xon._mode = 'weak';
                        xon._assignedFace = null;
                        xon._quarkType = null;
                        xon._loopSeq = null;
                        xon._loopStep = 0;
                        xon._tetActualized = false;
                        xon._t60Ejected = true;
                        xon.col = WEAK_FORCE_COLOR;
                        if (xon.sparkMat) xon.sparkMat.color.setHex(WEAK_FORCE_COLOR);
                        _weakLifecycleEnter(xon, 'non_actualized_tet');
                    }
                    phase0Changed = true;
                    continue;
                }
            }

            const effectiveStep = xon._loopStep >= 4 ? 0 : xon._loopStep;
            const nextNode = xon._loopSeq[effectiveStep + 1];

            let shouldEvictSelf = false;

            // Check 1: destination blocked by another xon
            // idle_tet blocker → evict the BLOCKER (expendable), not the blocked xon.
            // tet blocker → the blocked xon defers (tet has first-class priority).
            //   Backtracker handles any resulting T20 violation.
            // oct blocker → oct planner may vacate; don't evict.
            if ((occupied.get(nextNode) || 0) > 0) {
                const blocker = _demoXons.find(x => x.alive && x.node === nextNode &&
                    x !== xon && (x._mode === 'tet' || x._mode === 'idle_tet'));
                if (blocker && blocker._mode === 'idle_tet') {
                    // Evict the BLOCKER (idle_tet is expendable, but protect cage SCs)
                    _relinquishFaceSCs(blocker);
                    _logChoreo(`X${_demoXons.indexOf(blocker)} idle_tet blocker at n${nextNode} → weak (evicted by X${_demoXons.indexOf(xon)})`);
                    blocker._mode = 'weak';
                    blocker._t60Ejected = true;
                    blocker._assignedFace = null;
                    blocker._quarkType = null;
                    blocker._loopSeq = null;
                    blocker._loopStep = 0;
                    blocker._tetActualized = false;
                    blocker.col = WEAK_FORCE_COLOR;
                    if (blocker.sparkMat) blocker.sparkMat.color.setHex(WEAK_FORCE_COLOR);
                    _weakLifecycleEnter(blocker, 'evicted_by_tet');
                    blocker._evictedThisTick = true;
                    phase0Changed = true;
                } else if (blocker && blocker._mode === 'tet') {
                    // Blocked by tet → defer. Backtracker handles T20 if needed.
                    // Don't evict either xon.
                } else if (xon._mode === 'idle_tet') {
                    // idle_tet blocked by oct → evict SELF (expendable)
                    shouldEvictSelf = true;
                }
                // tet blocked by oct → don't evict, oct planner may vacate
            }

            // Dead-end lookahead ejection removed — let tet xons attempt their loops.
            // If they get stuck, PHASE 3 stuck-ejection handles it.

            if (shouldEvictSelf) {
                // Eviction is ALWAYS weak + _t60Ejected, NEVER _returnXonToOct
                _relinquishFaceSCs(xon);
                _logChoreo(`X${_demoXons.indexOf(xon)} evicted (dead end/blocked) → weak`);
                xon._mode = 'weak';
                xon._t60Ejected = true;
                xon._assignedFace = null;
                xon._quarkType = null;
                xon._loopSeq = null;
                xon._loopStep = 0;
                xon._tetActualized = false;
                xon.col = WEAK_FORCE_COLOR;
                if (xon.sparkMat) xon.sparkMat.color.setHex(WEAK_FORCE_COLOR);
                _weakLifecycleEnter(xon, 'phase0_eviction');
                xon._evictedThisTick = true;
                phase0Changed = true;
            }
        }
        if (phase0Changed) occupied = _occupiedNodes();
    }
    const _pT0 = performance.now(); _profPhases.p0 += _pT0 - _pT;

    // ── PHASE 0.5: Weak force xon movement ──
    // Handles ALL weak xon movement. Weak xons BFS toward nearest oct node.
    // May traverse any node that isn't hadron-occupied.
    // On arrival at oct node → transition to oct mode immediately.
    for (const xon of _demoXons) {
        if (!xon.alive || xon._mode !== 'weak') continue;

        // Already at oct node → becomes oct immediately, UNLESS just ejected
        // (must leave oct cage first to prevent eject→return oscillation).
        if (_octNodeSet && _octNodeSet.has(xon.node)) {
            if (!xon._t60Ejected || xon._weakLeftOct) {
                _weakLifecycleExit(xon, 'arrived_oct_immediate');
                _clearModeProps(xon);
                xon._mode = 'oct';
                if (_flashEnabled) xon.flashT = 1.0;
                xon.col = 0xffffff;
                if (xon.sparkMat) xon.sparkMat.color.setHex(0xffffff);
                continue;
            }
            // Ejected and haven't left oct yet — must move off first.
            // Fall through to BFS which will step toward ejection space.
        }

        // Log decision ledger for this weak xon (first attempt only)
        _logWeakDecisionLedger(xon, occupied);

        // BFS toward nearest oct node — collect ALL first-steps at optimal depth
        // Post-_mayReturn: weak xons may enter non-actualized tet nodes (no SCs active
        // = no geometry to disrupt) but NOT actualized tet nodes (active hadron space).
        // This gives weak xons breathing room to navigate back to the oct cage.
        // Avoid recent trail nodes to prevent cycling outside oct cage.
        const recentTrail = new Set(xon.trail ? xon.trail.slice(-6) : []);
        const visited = new Set([xon.node]);
        const queue = [[xon.node, null, 0]]; // [node, firstStep, depth]
        const bestSteps = []; // all first-steps reaching oct at same depth
        let bestDepth = Infinity;
        while (queue.length > 0) {
            const [cur, step, depth] = queue.shift();
            if (depth > bestDepth) break; // past optimal depth
            const nbs = baseNeighbors[cur] || [];
            for (const nb of nbs) {
                if (visited.has(nb.node)) continue;
                // Post-_mayReturn: may enter actualized tet nodes unless a hadron is there
                if (_isHadronOccupied(nb.node)) continue;
                visited.add(nb.node);
                const nextStep = step || nb.node;
                if (_octNodeSet && _octNodeSet.has(nb.node)) {
                    if (depth + 1 <= bestDepth) {
                        bestDepth = depth + 1;
                        if (!bestSteps.includes(nextStep)) bestSteps.push(nextStep);
                    }
                } else if (depth + 1 < bestDepth) {
                    queue.push([nb.node, nextStep, depth + 1]);
                }
            }
        }
        // Shuffle first for random tiebreaking, then stable-sort by priority.
        // This avoids a non-transitive comparator (_sRng() - 0.5 violates contract).
        // PRNG seed changes per retry, so shuffle produces different orderings.
        _sRngShuffle(bestSteps);
        bestSteps.sort((a, b) => {
            const aInTrail = recentTrail.has(a) ? 1 : 0;
            const bInTrail = recentTrail.has(b) ? 1 : 0;
            if (aInTrail !== bInTrail) return aInTrail - bInTrail;
            const aIsPrev = a === xon.prevNode ? 1 : 0;
            const bIsPrev = b === xon.prevNode ? 1 : 0;
            if (aIsPrev !== bIsPrev) return aIsPrev - bIsPrev;
            return 0; // tied — shuffle already randomized order
        });
        // Try each first-step: first that passes guards + occupancy + swap wins
        let bestStep = null;
        for (const step of bestSteps) {
            if (occupied.get(step) || 0) continue;
            if (_swapBlocked(xon.node, step)) continue;
            if (_moveViolatesGuards(xon, xon.node, step)) continue;
            bestStep = step;
            break;
        }
        if (bestStep !== null) {
            const fromWk = xon.node;
            // Track when ejected xon leaves oct cage (enables return)
            if (xon._t60Ejected && _octNodeSet && _octNodeSet.has(fromWk) && !_octNodeSet.has(bestStep)) {
                xon._weakLeftOct = true;
            }
            _occDel(occupied, xon.node);
            xon.prevNode = xon.node;
            xon.node = bestStep;
            // Proxy may have blocked (already moved this tick) — verify
            if (xon.node !== bestStep) { _occAdd(occupied, xon.node); continue; }
            _occAdd(occupied, bestStep);
            xon._movedThisTick = true;
            _moveRecord.set(bestStep, fromWk);
            _traceMove(xon, fromWk, bestStep, 'weakBFS');
            _recordEjectionTraversal(fromWk, bestStep);
            xon.tweenT = 0;
            anyMoved = true;
            _weakLifecycleStep(xon);
            // Check if arrived at oct node — transition immediately (no _mayReturn gate).
            // Weak xons that reach the oct cage have successfully returned.
            if (_octNodeSet && _octNodeSet.has(bestStep)) {
                const octCountNow = _demoXons.filter(x => x.alive && x._mode === 'oct').length;
                if (octCountNow < OCT_CAPACITY_MAX) {
                    _weakLifecycleExit(xon, 'arrived_oct_bfs');
                    _clearModeProps(xon);
                    xon._mode = 'oct';
                    if (_flashEnabled) xon.flashT = 1.0;
                    xon.col = 0xffffff;
                    if (xon.sparkMat) xon.sparkMat.color.setHex(0xffffff);
                }
            }
        } else {
            // All BFS steps blocked — try free neighbor (avoiding hadron-occupied nodes)
            // Use full baseNeighbors (not _localBaseNeighbors which restricts to nucleus)
            // — weak xons have freedom to roam the full lattice.
            // Shuffle to prevent lattice-order bias in fallback selection.
            const allNbs = _sRngShuffle((baseNeighbors[xon.node] || []).slice());
            const hadronFilter = nb => !_isHadronOccupied(nb.node);
            // Tier 1: guard-safe, no hadron, not in recent trail, not prevNode
            let freeNb = allNbs.find(nb => !(occupied.get(nb.node) || 0) &&
                hadronFilter(nb) &&
                !recentTrail.has(nb.node) && nb.node !== xon.prevNode &&
                !_swapBlocked(xon.node, nb.node) &&
                !_moveViolatesGuards(xon, xon.node, nb.node));
            // Tier 2: guard-safe, no hadron, not prevNode
            if (!freeNb) {
                freeNb = allNbs.find(nb => !(occupied.get(nb.node) || 0) &&
                    hadronFilter(nb) &&
                    nb.node !== xon.prevNode &&
                    !_swapBlocked(xon.node, nb.node) &&
                    !_moveViolatesGuards(xon, xon.node, nb.node));
            }
            // Tier 3: guard-safe, no hadron, allow prevNode
            if (!freeNb) {
                freeNb = allNbs.find(nb => !(occupied.get(nb.node) || 0) &&
                    hadronFilter(nb) &&
                    !_swapBlocked(xon.node, nb.node) &&
                    !_moveViolatesGuards(xon, xon.node, nb.node));
            }
            // No guard bypass — if no guard-safe move exists, xon stays put
            if (freeNb) {
                const fromWk2 = xon.node;
                if (xon._t60Ejected && _octNodeSet && _octNodeSet.has(fromWk2) && !_octNodeSet.has(freeNb.node)) {
                    xon._weakLeftOct = true;
                }
                _occDel(occupied, xon.node);
                xon.prevNode = xon.node;
                xon.node = freeNb.node;
                // Proxy may have blocked — verify
                if (xon.node !== freeNb.node) { _occAdd(occupied, xon.node); continue; }
                _occAdd(occupied, freeNb.node);
                xon._movedThisTick = true;
                _moveRecord.set(freeNb.node, fromWk2);
                _traceMove(xon, fromWk2, freeNb.node, 'weakDetour');
                xon.tweenT = 0;
                anyMoved = true;
                _weakLifecycleStep(xon);
                if (_octNodeSet && _octNodeSet.has(freeNb.node)) {
                    const octCountNow = _demoXons.filter(x => x.alive && x._mode === 'oct').length;
                    if (octCountNow < OCT_CAPACITY_MAX) {
                        _weakLifecycleExit(xon, 'arrived_oct_detour');
                        _clearModeProps(xon);
                        xon._mode = 'oct';
                        if (_flashEnabled) xon.flashT = 1.0;
                        xon.col = 0xffffff;
                        if (xon.sparkMat) xon.sparkMat.color.setHex(0xffffff);
                    }
                }
            }
        }
    }
    const _pT05 = performance.now(); _profPhases.p05 += _pT05 - _pT0;

    // ── PHASE 1: Plan tet/idle_tet moves (fixed sequences) ──
    const tetPlans = [];
    const tetBlockedBy = new Map(); // toNode → xon (tet xons blocked by oct occupants)
    for (const xon of _demoXons) {
        if (!xon.alive) continue;
        if (xon._mode !== 'tet' && xon._mode !== 'idle_tet') continue;
        // Wrap completed loops — xons cycle continuously in their tet
        const effectiveStep = xon._loopStep >= 4 ? 0 : xon._loopStep;
        const fromNode = xon._loopSeq[effectiveStep];
        const toNode = xon._loopSeq[effectiveStep + 1];
        tetPlans.push({ xon, fromNode, toNode, approved: false });
    }

    // Approve tet moves to free destinations; track oct-blocked ones
    // Uses loop-shape-aware lookahead: checks the xon's actual loop path, not generic graph
    for (const plan of tetPlans) {
        if (planned.has(plan.toNode)) continue; // another tet already claimed this
        const occCount = occupied.get(plan.toNode) || 0;
        if (occCount === 0) {
            // Loop-shape-aware lookahead: verify the xon's specific loop path is viable
            const tmpOcc = new Map(occupied);
            _occDel(tmpOcc, plan.fromNode);
            _occAdd(tmpOcc, plan.toNode);
            const effectiveStep = plan.xon._loopStep >= 4 ? 0 : plan.xon._loopStep;
            if (_lookaheadTetPath(plan.xon._loopSeq, effectiveStep + 1, tmpOcc, _choreoParams.lookahead, plan.xon)) {
                plan.approved = true;
                planned.add(plan.toNode);
            }
            // If lookahead fails, xon's escape hatch will return it to oct
        } else {
            // Blocked — check if blocker is an oct xon we can ask to move
            const blocker = _demoXons.find(x => x.alive && x._mode === 'oct' && x.node === plan.toNode);
            if (blocker) {
                tetBlockedBy.set(plan.toNode, plan);
                // Tentatively claim — oct planner will be forced to vacate this node
                planned.add(plan.toNode);
                plan.approved = true;
                plan._needsOctVacate = blocker;
            }
            // If blocker is tet/idle_tet (or no oct blocker found), approve as annihilation.
            // The cooperative lookahead treats collisions as annihilation opportunities.
            // Skip when annihilation disabled — xon will escape via PHASE 3 hatch instead.
            if (!plan.approved && _annihilationEnabled) {
                plan.approved = true;
                plan._annihilateMove = true;
                planned.add(plan.toNode);
            }
        }
    }

    // Vacuum negotiation for approved tet moves — hard requirement.
    // If ANY SC exists on this edge and isn't active, it must be materialised.
    // If materialisation fails, the tet xon cannot traverse.
    for (const plan of tetPlans) {
        if (!plan.approved) continue;
        const pid = pairId(plan.fromNode, plan.toNode);
        const scId = scPairToId.get(pid);
        if (scId === undefined) continue; // no SC on this edge, base edge only

        // Check if edge also has a base connection — if so, xon uses base edge, no SC needed
        const hasBaseEdge = (baseNeighbors[plan.fromNode] || []).some(nb => nb.node === plan.toNode);
        if (hasBaseEdge) continue;

        // Edge is SC-only — must be activated
        if (!xonImpliedSet.has(scId) && !activeSet.has(scId) && !impliedSet.has(scId)) {
            let activated = false;
            const xi = _demoXons.indexOf(plan.xon);
            if (canMaterialiseQuick(scId)) {
                xonImpliedSet.add(scId);
                _scAttribution.set(scId, { reason: 'tetTraversal', xonIdx: xi, face: plan.xon._assignedFace, tick: _demoTick });
                stateVersion++; // invalidate _getBasePairs cache for subsequent checks
                _solverNeeded = true;
                activated = true;
            } else if (excitationSeverForRoom(scId)) {
                if (canMaterialiseQuick(scId)) {
                    xonImpliedSet.add(scId);
                    _scAttribution.set(scId, { reason: 'tetTraversal', xonIdx: xi, face: plan.xon._assignedFace, tick: _demoTick });
                    stateVersion++; // invalidate _getBasePairs cache
                    _solverNeeded = true;
                    activated = true;
                }
            }
            if (!activated) {
                // Vacuum rejected — revoke tet move
                plan.approved = false;
                planned.delete(plan.toNode);
            }
        }
    }

    const _pT1 = performance.now(); _profPhases.p1 += _pT1 - _pT05;

    // ── PHASE 1.5: Natural loop completion — return xons that finished their loops ──
    // Replaces forced window-boundary returns. Loops complete organically at step >= 4.
    {
        const locked15 = _traversalLockedSCs();
        for (const xon of _demoXons) {
            if (!xon.alive) continue;
            if (xon._mode !== 'tet' && xon._mode !== 'idle_tet') continue;
            if (xon._loopStep < 4) continue; // still mid-loop — let it finish

            // T60: non-actualized loop → eject as weak particle.
            // Fresh check at completion: are all face SCs still present RIGHT NOW?
            // If the vacuum severed one during the loop, the tet lost support → weak eject.
            let _t60actualized = false;
            if (xon._assignedFace != null && _nucleusTetFaceData) {
                const fd60 = _nucleusTetFaceData[xon._assignedFace];
                if (fd60 && fd60.scIds.every(scId =>
                    activeSet.has(scId) || impliedSet.has(scId) || xonImpliedSet.has(scId))) {
                    _t60actualized = true;
                }
            }
            if (!_t60actualized) {
                // Trail segments keep their original recorded color.
                // Relinquish face SCs (tight space = delete, sticky space = no-op)
                _relinquishFaceSCs(xon);
                _logChoreo(`X${_demoXons.indexOf(xon)} non-actualized face ${xon._assignedFace} (completion) → weak`);
                xon._mode = 'weak';
                xon._assignedFace = null;
                xon._quarkType = null;
                xon._loopSeq = null;
                xon._loopStep = 0;
                xon._tetActualized = false;
                xon._t60Ejected = true;
                xon.col = WEAK_FORCE_COLOR;
                if (xon.sparkMat) xon.sparkMat.color.setHex(WEAK_FORCE_COLOR);
                _weakLifecycleEnter(xon, 'non_actualized_tet');
                continue; // PHASE 0.5 will move it; skip normal return-to-oct
            }

            // Loop complete + actualized — return to oct
            _returnXonToOct(xon, occupied);
            // Relinquish face SCs (tight space = delete, sticky space = no-op)
            _relinquishFaceSCs(xon);
        }
        occupied = _occupiedNodes(); // refresh after returns
    }

    // ── PHASE 2a: Demand-driven face selection (decentralized, no order precedence) ──
    // Each oct xon scores ALL reachable faces independently. Conflicts resolved by
    // random shuffling — no xon gets order-precedence over another.
    let _cageCriticalXons = new Set(); // hoisted for PHASE 3 gluon activation
    {
        _ratioTracker.sync();
        let octIdle = _demoXons.filter(x => x.alive && x._mode === 'oct' && !x._movedThisTick && !x._evictedThisTick);

        // ── GLUON / CAGE CHECK: cage integrity takes priority over face assignment ──
        // Gluon mode = xon that moved OFF the oct cage to preserve cage integrity.
        // Oct→oct moves for cage maintenance stay in oct mode.
        // Gluon mode activates ONLY when the xon actually moves to a non-oct node.
        //
        // Here we: (1) demote gluons back to oct if they've returned to the cage,
        //          (2) exclude cage-critical oct xons from face assignment.
        // ── Gluon-mediated orphan cleanup: release gluons whose client is no longer active ──
        if (_ruleGluonMediatedSC) {
            for (const xon of _demoXons) {
                if (!xon.alive || xon._gluonForFace == null) continue;
                const client = xon._gluonClientXon;
                if (!client || !client.alive ||
                    (client._mode !== 'tet' && client._mode !== 'idle_tet') ||
                    client._assignedFace !== xon._gluonForFace) {
                    _releaseGluon(xon._gluonForFace);
                }
            }
        }
        const gluonXons = _demoXons.filter(x => x.alive && x._mode === 'gluon' && !x._movedThisTick && !x._evictedThisTick);
        for (const xon of gluonXons) {
            // Face-bound gluons stay gluon until released — skip auto-demotion
            if (xon._gluonForFace != null) continue;
            // Legacy gluon (cage-critical) that returned to oct cage → demote to oct
            if (_octNodeSet && _octNodeSet.has(xon.node)) {
                xon._mode = 'oct';
                xon.col = 0xffffff;
                if (xon.sparkMat) xon.sparkMat.color.setHex(0xffffff);
                octIdle.push(xon);
                _logChoreo(`GLUON: X${_demoXons.indexOf(xon)} returned to oct cage → oct mode`);
            }
            // Gluon still off-cage → stays gluon, excluded from face assignment
        }
        // Mark cage-critical oct xons — exclude from face assignment but keep oct mode
        // (they'll become gluon only if their move takes them off the oct cage)
        _cageCriticalXons = new Set();
        for (const xon of octIdle) {
            if (_cageWouldBreak(xon)) {
                _cageCriticalXons.add(xon);
                _logChoreo(`CAGE: X${_demoXons.indexOf(xon)} cage-critical (excluded from face assignment)`);
            }
        }
        // Remove cage-critical xons from face assignment candidates (but NOT from oct movement)
        octIdle = octIdle.filter(x => !_cageCriticalXons.has(x));
        // Remove gluon-bound xons from face assignment (they're serving a tet face)
        if (_ruleGluonMediatedSC) {
            octIdle = octIdle.filter(x => x._gluonForFace == null);
        }

        if (octIdle.length > 0 && _nucleusTetFaceData) {
            // ── FACE ASSIGNMENT: greedy + ranked enumeration ──
            // During backtracking, enumerate ALL valid (xon, face, quarkType) combos
            // and cycle through them ranked by aggregate score. Greedy = first combo.
            let assignedCombo = null; // the combo that actually gets assigned

            if (_btActive && typeof _enumerateAllFaceAssignments === 'function') {
                // Build/refresh face assignment cache when ledger changes
                const ledgerSize = (_btBadMoveLedger.get(_demoTick) || {size:0}).size || 0;
                if (_btFaceAssignCache === null || _btFaceAssignLedgerSize !== ledgerSize) {
                    // Collect ALL valid proposals across all xons, faces, quark types
                    const allProposals = [];
                    for (const xon of octIdle) {
                        const opps = _allFaceOpportunities(xon, occupied);
                        for (const opp of opps) allProposals.push(opp);
                    }
                    _btFaceAssignCache = _enumerateAllFaceAssignments(allProposals);
                    _btFaceAssignIndex = 0;
                    _btFaceAssignLedgerSize = ledgerSize;
                    _logChoreo(`FACE ENUM: ${_btFaceAssignCache.length} valid combos for tick ${_demoTick}`);
                }

                // Try the next combo from the cache
                if (_btFaceAssignCache.length > 0) {
                    const idx = _btFaceAssignIndex % _btFaceAssignCache.length;
                    _btFaceAssignIndex++;
                    const combo = _btFaceAssignCache[idx];
                    // Validate each proposal in the combo (vacuum + lookahead)
                    const validatedCombo = [];
                    const usedXons = new Set();
                    const usedFaces = new Set();
                    for (const prop of combo) {
                        if (usedXons.has(prop.xon) || usedFaces.has(prop.face)) continue;
                        // Vacuum feasibility
                        const fd = _nucleusTetFaceData[prop.face];
                        let canMaterialize = true;
                        for (const scId of fd.scIds) {
                            if (activeSet.has(scId) || impliedSet.has(scId) || xonImpliedSet.has(scId)) continue;
                            if (!canMaterialiseQuick(scId)) {
                                if (!excitationSeverForRoom(scId) || !canMaterialiseQuick(scId)) {
                                    canMaterialize = false;
                                    break;
                                }
                            }
                        }
                        if (!canMaterialize) continue;
                        // Lookahead viability
                        const seq = _selectBestPermutation(prop.xon, fd.cycle, prop.quarkType);
                        const tmpOcc = new Map(occupied);
                        if (!_lookaheadTetPath(seq, 0, tmpOcc, _choreoParams.lookahead, prop.xon)) continue;
                        validatedCombo.push(prop);
                        usedXons.add(prop.xon);
                        usedFaces.add(prop.face);
                    }
                    assignedCombo = validatedCombo;
                }
            } else {
                // ── GREEDY PATH (normal, non-backtrack) ──
                // Each xon picks its single best face
                const proposals = [];
                for (const xon of octIdle) {
                    let bestOpp = null;
                    for (let f = 1; f <= 8; f++) {
                        const opp = _scoreFaceOpportunity(xon, f, occupied);
                        if (opp && opp.score >= _choreoParams.assignmentThreshold) {
                            if (!bestOpp || opp.score > bestOpp.score) bestOpp = { xon, ...opp };
                        }
                    }
                    if (bestOpp) proposals.push(bestOpp);
                }

                // Shuffle proposals — no xon gets priority by index order
                for (let i = proposals.length - 1; i > 0; i--) {
                    const j = Math.floor(_sRng() * (i + 1));
                    [proposals[i], proposals[j]] = [proposals[j], proposals[i]];
                }

                // Resolve conflicts: one xon per face, first-after-shuffle wins
                const validatedCombo = [];
                const usedXons = new Set();
                const usedFaces = new Set();
                for (const prop of proposals) {
                    if (usedXons.has(prop.xon) || usedFaces.has(prop.face)) continue;
                    if (prop.score < _choreoParams.assignmentThreshold) continue;
                    // Vacuum feasibility
                    const fd = _nucleusTetFaceData[prop.face];
                    let canMaterialize = true;
                    for (const scId of fd.scIds) {
                        if (activeSet.has(scId) || impliedSet.has(scId) || xonImpliedSet.has(scId)) continue;
                        if (!canMaterialiseQuick(scId)) {
                            if (!excitationSeverForRoom(scId) || !canMaterialiseQuick(scId)) {
                                canMaterialize = false;
                                break;
                            }
                        }
                    }
                    if (!canMaterialize) continue;
                    // Lookahead viability
                    const seq = _selectBestPermutation(prop.xon, fd.cycle, prop.quarkType);
                    const tmpOcc = new Map(occupied);
                    if (!_lookaheadTetPath(seq, 0, tmpOcc, _choreoParams.lookahead, prop.xon)) continue;
                    validatedCombo.push(prop);
                    usedXons.add(prop.xon);
                    usedFaces.add(prop.face);
                }
                assignedCombo = validatedCombo;
            }

            // ── Execute the chosen face assignment combo ──
            const assignedXons = new Set();
            const assignedFaces = new Set();
            if (assignedCombo) {
                for (const prop of assignedCombo) {
                    _assignXonToTet(prop.xon, prop.face, prop.quarkType);
                    _demoTetAssignments++;
                    assignedXons.add(prop.xon);
                    assignedFaces.add(prop.face);
                    _demoVisitedFaces.add(prop.face);
                    _solverNeeded = true;
                }
            }

            // ── Immediate first step for newly-assigned tet xons ──
            // Xons assigned in PHASE 2a missed PHASE 1's tet planning.
            // If they haven't moved (already on face node), plan their first step
            // so PHASE 3 can execute it (prevents standing still for 1 tick).
            for (const xon of assignedXons) {
                if (xon._movedThisTick) continue; // walked to face during assignment
                if (!xon._loopSeq || xon._mode !== 'tet') continue;
                const effectiveStep = xon._loopStep >= 4 ? 0 : xon._loopStep;
                const fromNode = xon._loopSeq[effectiveStep];
                const toNode = xon._loopSeq[effectiveStep + 1];
                if (toNode === undefined) continue;
                if (planned.has(toNode)) continue;
                if ((occupied.get(toNode) || 0) > 0) continue;
                // Vacuum negotiation for first hop
                const pid = pairId(fromNode, toNode);
                const scId = scPairToId.get(pid);
                const hasBase = (baseNeighbors[fromNode] || []).some(nb => nb.node === toNode);
                let scOk = hasBase;
                if (!scOk && scId !== undefined) {
                    scOk = activeSet.has(scId) || impliedSet.has(scId) || xonImpliedSet.has(scId);
                    if (!scOk && canMaterialiseQuick(scId)) {
                        const xi = _demoXons.indexOf(xon);
                        xonImpliedSet.add(scId);
                        _scAttribution.set(scId, { reason: 'tetTraversal', xonIdx: xi, face: xon._assignedFace, tick: _demoTick });
                        stateVersion++;
                        _solverNeeded = true;
                        scOk = true;
                    }
                }
                if (scOk || scId === undefined) {
                    tetPlans.push({ xon, fromNode, toNode, approved: true });
                    planned.add(toNode);
                }
            }

            // ── BFS Test decision trace: capture face assignments ──
            if (_bfsTestActive && assignedXons.size > 0) {
                const faceAssignments = [];
                if (assignedCombo) {
                    for (const prop of assignedCombo) {
                        faceAssignments.push({
                            xonIdx: _demoXons.indexOf(prop.xon),
                            face: prop.face,
                            quarkType: prop.quarkType,
                            score: +prop.score.toFixed(2),
                        });
                    }
                }
                if (!_bfsTestDecisionTrace) _bfsTestDecisionTrace = [];
                let entry = _bfsTestDecisionTrace.find(e => e.tick === _demoTick && e.runIdx === _bfsTestRunIdx);
                if (!entry) {
                    entry = { tick: _demoTick, runIdx: _bfsTestRunIdx, btActive: _btActive, faceAssignments: [], octMatching: [] };
                    _bfsTestDecisionTrace.push(entry);
                }
                entry.faceAssignments = faceAssignments;
            }
        }

        occupied = _occupiedNodes(); // refresh after assignments
    }

    // ── OCT CAPACITY OVERFLOW — proactive shedding ──
    // Count ALL xons on oct nodes (any mode — oct, idle_tet, weak all count).
    // T79 pressure: if approaching the consecutive-full limit, shed 1 extra.
    // Priority: oct-mode first (least disruptive), then idle_tet, then weak.
    // Tier 1: _startIdleTetLoop (productive — manifests a hadron, moves off cage).
    // Tier 2: _pendingWeakEjection (becomes weak when it steps off oct).
    {
        const allOnOct = (_octNodeSet && _octNodeSet.size > 0)
            ? _demoXons.filter(x => x.alive && _octNodeSet.has(x.node)).length : 0;
        const t79Pressure = (_octFullConsecutive >= T79_MAX_FULL_TICKS - 1 && allOnOct >= OCT_CAPACITY_MAX) ? 1 : 0;
        let excess = allOnOct - OCT_CAPACITY_MAX + t79Pressure;
        if (excess > 0) {
            // Priority 1: oct-mode xons (easiest to redirect)
            const octCandidates = _sRngShuffle(_demoXons.filter(x =>
                x.alive && x._mode === 'oct' && !x._movedThisTick && !x._evictedThisTick &&
                _octNodeSet.has(x.node)
            ));
            for (const xon of octCandidates) {
                if (excess <= 0) break;
                if (_startIdleTetLoop(xon, occupied)) {
                    _logChoreo(`X${_demoXons.indexOf(xon)} oct overflow -> idle_tet f${xon._assignedFace}`);
                    _solverNeeded = true;
                    excess--;
                    continue;
                }
                // No direct weak ejection from oct — xons can only leave
                // by starting a tet loop. If no loop available, stay on oct.
                _logChoreo(`X${_demoXons.indexOf(xon)} oct overflow — no tet available, stays on oct`);
            }
            // Priority 2: idle_tet xons on oct nodes — revert to oct (no weak ejection)
            if (excess > 0) {
                const idleCandidates = _sRngShuffle(_demoXons.filter(x =>
                    x.alive && x._mode === 'idle_tet' && !x._movedThisTick &&
                    _octNodeSet.has(x.node)
                ));
                for (const xon of idleCandidates) {
                    if (excess <= 0) break;
                    const xi = _demoXons.indexOf(xon);
                    _logChoreo(`X${xi} idle_tet on oct -> oct (shed, no weak ejection)`);
                    xon._assignedFace = null;
                    xon._loopSeq = null;
                    xon._loopStep = 0;
                    xon._mode = 'oct';
                    excess--;
                }
            }
            // Priority 3 removed: no direct weak ejection from oct.
            // Xons can only leave oct by starting a tet loop.
            if (allOnOct - OCT_CAPACITY_MAX + t79Pressure > 0) occupied = _occupiedNodes();
        }
    }

    // ── PHASE 2: Coordinated oct movement planning ──
    let octXons = [];
    let octPlans = [];
    {
    octXons = _demoXons.filter(x => x.alive && (x._mode === 'oct' || x._mode === 'gluon' || x._mode === 'weak') && !x._movedThisTick);
    for (const xon of octXons) _occDel(occupied, xon.node);

    octPlans = octXons.map(xon => ({
        xon,
        candidates: _getOctCandidates(xon, occupied, planned),
        assigned: null,
        fromNode: xon.node,
    }));

    // Restore occupied for later use
    for (const xon of octXons) _occAdd(occupied, xon.node);

    // Pre-filter candidates: remove those where vacuum would definitely reject.
    // EXCEPTION: oct cage SCs bypass this filter — they use full vacuum negotiation
    // (including excitationSeverForRoom) in _executeOctMove, which the quick check
    // doesn't account for. Without this exception, cumulative strain from the first
    // 3 cage SCs can permanently block the 4th from ever being attempted.
    //
    // GPU/Worker acceleration: batch all canMaterialiseQuick calls into one Worker
    // round-trip when available. Falls back to synchronous main-thread solver.
    let _batchResults = null; // Map<scId, {pass, worst, avg}>
    if (typeof SolverProxy !== 'undefined' && SolverProxy.isReady()) {
        // Collect unique SC IDs needing materialisation check
        // Pre-filter: skip grossly non-local SC edges (saves solver calls)
        const candidateScIds = new Set();
        for (const plan of octPlans) {
            for (const c of plan.candidates) {
                if (!c._needsMaterialise) continue;
                if (c._scId === undefined) continue;
                if (_octSCIds && _octSCIds.includes(c._scId)) continue;
                // Distance pre-filter: reject obviously non-local SC candidates (d > 1.5)
                const sc = SC_BY_ID[c._scId];
                if (sc) {
                    const pa = pos[sc.a], pb = pos[sc.b];
                    const dx = pb[0]-pa[0], dy = pb[1]-pa[1], dz = pb[2]-pa[2];
                    const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
                    if (Math.abs(dist - 1) > 0.50) continue; // teleportation-range, skip solver
                }
                candidateScIds.add(c._scId);
            }
        }
        if (candidateScIds.size > 0) {
            const candidateScIdArray = [...candidateScIds];
            // Only use Worker batch when enough candidates to amortize round-trip.
            // Worker overhead is ~50ms for postMessage round-trip; CPU CMQ is ~12ms each.
            // Break-even: ~4 candidates. Below that, CPU is faster.
            const MIN_BATCH_SIZE = 5;
            if (candidateScIdArray.length >= MIN_BATCH_SIZE) {
                const basePairs = _getBasePairs();
                const candidateScPairs = candidateScIdArray.map(id => { const sc = SC_BY_ID[id]; return [sc.a, sc.b]; });
                const _batchT0 = performance.now();
                const results = await SolverProxy.solveBatch(basePairs, candidateScPairs);
                _profPhases.gpuBatch = (_profPhases.gpuBatch || 0) + (performance.now() - _batchT0);
                if (results) {
                    _batchResults = new Map();
                    for (let i = 0; i < candidateScIdArray.length; i++) {
                        _batchResults.set(candidateScIdArray[i], results[i]);
                    }
                    SolverProxy.cacheBatchResults(candidateScIdArray, results, stateVersion);
                }
            } else {
                // Small batch: run CMQ on CPU inline (faster than Worker round-trip)
                _batchResults = new Map();
                for (const scId of candidateScIdArray) {
                    _batchResults.set(scId, { pass: canMaterialiseQuick(scId) });
                }
            }
        }
    }
    for (const plan of octPlans) {
        plan.candidates = plan.candidates.filter(c => {
            if (!c._needsMaterialise) return true; // base edge or already active SC
            if (c._scId === undefined) return true;
            // Oct cage SCs get full vacuum negotiation in _executeOctMove
            if (_octSCIds && _octSCIds.includes(c._scId)) return true;
            // Distance pre-filter: reject grossly non-local before hitting solver
            const sc = SC_BY_ID[c._scId];
            if (sc) {
                const pa = pos[sc.a], pb = pos[sc.b];
                const dx = pb[0]-pa[0], dy = pb[1]-pa[1], dz = pb[2]-pa[2];
                if (Math.abs(Math.sqrt(dx*dx + dy*dy + dz*dz) - 1) > 0.50) return false;
            }
            // Use batch results if available, otherwise fall back to sync
            if (_batchResults && _batchResults.has(c._scId)) {
                return _batchResults.get(c._scId).pass;
            }
            return canMaterialiseQuick(c._scId); // fallback: sync main-thread solver
        });
    }

    // ── CANDIDATE SNAPSHOT (pre-exclusion) for traversal log ──
    let _preExclusionCandidates = null;
    if (_btActive && _searchTraversalLog) {
        _preExclusionCandidates = {};
        for (const plan of octPlans) {
            const xi = _demoXons.indexOf(plan.xon);
            _preExclusionCandidates[xi] = plan.candidates.map(c => ({
                node: c.node, score: +(c.score?.toFixed(3) || 0),
            }));
        }
    }

    // ── BACKTRACK EXCLUSION FILTER: remove moves that caused violations on previous attempts ──
    if (_btActive) {
        for (const plan of octPlans) {
            const xonIdx = _demoXons.indexOf(plan.xon);
            plan.candidates = plan.candidates.filter(c => !_btIsMoveExcluded(xonIdx, c.node));
        }
    }

    // ── CANDIDATE SNAPSHOT (post-exclusion) — mark excluded candidates ──
    if (_btActive && _searchTraversalLog && _preExclusionCandidates) {
        _searchLastCandidates = {};
        for (const plan of octPlans) {
            const xi = _demoXons.indexOf(plan.xon);
            const remaining = new Set(plan.candidates.map(c => c.node));
            _searchLastCandidates[xi] = (_preExclusionCandidates[xi] || []).map(c => ({
                ...c, excluded: !remaining.has(c.node),
            }));
        }
    }

    // T55 enforcement: Removed — T55 live guard fires into backtracker

    // ── DETERMINISTIC MATCHING ENUMERATION ──
    // During backtrack retries, instead of rotating candidates and hoping Kuhn's
    // produces a different matching, we enumerate ALL valid maximum-cardinality
    // matchings and try them sequentially. This guarantees comprehensiveness.
    if (_btActive && typeof _enumerateAllMatchings === 'function') {
        // Enumerate all valid matchings for current candidate lists.
        // Re-enumerate when exclusions change (ledger size differs from cached state).
        const ledgerSize = (_btBadMoveLedger.get(_demoTick) || {size:0}).size || 0;
        if (_btMatchingCache === null || _btMatchingCacheLedgerSize !== ledgerSize) {
            _btMatchingCache = _enumerateAllMatchings(octPlans, planned);
            _btMatchingIndex = 0;
            _btMatchingCacheLedgerSize = ledgerSize;
            _logChoreo(`ENUM: ${_btMatchingCache.length} valid matchings for tick ${_demoTick} (ledger: ${ledgerSize})`);
        }
        // Apply the next matching from the cache, cycling with wrap-around.
        // Each cycle pairs matchings with a different PRNG seed (via _btRetryCount
        // / _bfsLayerRetries), so secondary choices (idle_tet face, weak BFS,
        // return-to-oct neighbor) explore different paths each cycle.
        if (_btMatchingCache.length > 0) {
            const idx = _btMatchingIndex % _btMatchingCache.length;
            _btMatchingIndex++;
            const matching = _btMatchingCache[idx];
            for (let i = 0; i < octPlans.length; i++) {
                octPlans[i].assigned = matching[i];
            }
        } else {
            // Zero valid matchings (all candidates excluded) — null assignments
            for (const plan of octPlans) plan.assigned = null;
        }
    } else if (_kuhnEnabled) {
        // Normal (non-backtrack) path: Kuhn's augmenting-path matching.
        _maxBipartiteAssignment(octPlans, planned);
    } else {
        // Greedy first-fit: assign each xon its best available candidate.
        const greedyClaimed = new Set();
        for (const plan of octPlans) {
            plan.assigned = null;
            for (const c of plan.candidates) {
                if (planned.has(c.node) || greedyClaimed.has(c.node)) continue;
                plan.assigned = c;
                greedyClaimed.add(c.node);
                break;
            }
        }
    }
    const octClaimed = new Set();
    for (const plan of octPlans) {
        if (plan.assigned) octClaimed.add(plan.assigned.node);
    }

    // ── BFS Test decision trace: capture oct matching ──
    if (_bfsTestActive) {
        let entry = _bfsTestDecisionTrace?.find(e => e.tick === _demoTick && e.runIdx === _bfsTestRunIdx);
        if (!entry) {
            if (!_bfsTestDecisionTrace) _bfsTestDecisionTrace = [];
            entry = { tick: _demoTick, runIdx: _bfsTestRunIdx, btActive: _btActive, faceAssignments: [], octMatching: [] };
            _bfsTestDecisionTrace.push(entry);
        }
        entry.octMatching = octPlans.map(p => ({
            xonIdx: _demoXons.indexOf(p.xon),
            from: p.fromNode,
            to: p.assigned ? p.assigned.node : null,
            candidateCount: p.candidates.length,
        }));
        entry.matchingMethod = _btActive ? 'enumerated' : (_kuhnEnabled ? 'kuhn' : 'greedy');
        if (_btActive && _btMatchingCache) {
            entry.totalMatchings = _btMatchingCache.length;
            entry.matchingIdx = _btMatchingIndex;
        }
    }

    // Verify needsOctVacate: if an oct xon was supposed to move but couldn't,
    // convert to annihilation move (lookahead treats collisions as valid terminal states).
    for (const plan of tetPlans) {
        if (!plan._needsOctVacate) continue;
        const blocker = plan._needsOctVacate;
        const octPlan = octPlans.find(p => p.xon === blocker);
        if (!octPlan || !octPlan.assigned) {
            if (_annihilationEnabled) {
                // Oct xon couldn't move — approve as annihilation instead of revoking.
                // PHASE 4 will resolve the on-node collision via gluon storage.
                plan._annihilateMove = true;
            } else {
                // Annihilation disabled — revoke this plan so escape hatch handles it.
                plan.approved = false;
                planned.delete(plan.toNode);
            }
        }
    }

    // Build a combined blocked set for idle_tet planning
    const allBlocked = new Map(occupied);
    for (const n of planned) _occAdd(allBlocked, n);
    for (const n of octClaimed) _occAdd(allBlocked, n);
    for (const plan of octPlans) {
        if (plan.assigned) _occDel(allBlocked, plan.fromNode);
    }

    // ── COLLISION AVOIDANCE: hierarchical strategy for unassigned xons ──
    // 1. Divert into unscheduled tet (productive work — manifest a hadron)
    // 2. If no tet available, eject as weak particle (safety valve)
    // This replaces the old congestion-relief / bounce-escape / idle_tet fallbacks
    // with a single unified pass.
    for (const plan of octPlans) {
        if (plan.assigned || plan.idleTet) continue;
        if (plan.xon._evictedThisTick) continue;
        // Gluon xons must stay on the oct cage — never divert into tet loops
        if (plan.xon._mode === 'gluon') continue;

        // ── Strategy 1: Divert into tet ──
        // Try _startIdleTetLoop first (uses Pauli-aware face selection)
        const _savedMode = plan.xon._mode;
        const _savedCol = plan.xon.col;
        let diverted = false;
        if (_startIdleTetLoop(plan.xon, allBlocked)) {
            const dest = plan.xon._loopSeq[plan.xon._loopStep + 1];
            const tmpCheck = new Map(allBlocked); _occAdd(tmpCheck, dest);
            if (dest !== undefined && !allBlocked.has(dest) && _lookaheadTetPath(plan.xon._loopSeq, 1, tmpCheck, _choreoParams.lookahead, plan.xon)) {
                plan.idleTet = true;
                _occAdd(allBlocked, dest);
                diverted = true;
                _logChoreo(`X${_demoXons.indexOf(plan.xon)} collision->tet f${plan.xon._assignedFace}`);
            } else {
                // Rollback tet attempt
                _relinquishFaceSCs(plan.xon);
                plan.xon._mode = _savedMode;
                if (_flashEnabled) plan.xon.flashT = 1.0;
                plan.xon._loopSeq = null;
                plan.xon._loopStep = 0;
                plan.xon._assignedFace = null;
                plan.xon.col = _savedCol;
                if (plan.xon.sparkMat) plan.xon.sparkMat.color.setHex(_savedCol);
            }
        }

        // ── Strategy 2: Eject as weak particle (_t60Ejected = true) ──
        if (!diverted) {
            plan.xon._mode = 'weak';
            plan.xon._t60Ejected = true;
            plan.xon.col = WEAK_FORCE_COLOR;
            if (plan.xon.sparkMat) plan.xon.sparkMat.color.setHex(WEAK_FORCE_COLOR);
            _weakLifecycleEnter(plan.xon, 'phase2_collision_eject');
            // Find a neighbor node to escape to
            const nbs = _localBaseNeighbors(plan.xon.node);
            let escaped = false;
            for (const nb of nbs) {
                if (allBlocked.has(nb.node)) continue;
                if (planned.has(nb.node)) continue;
                if (octClaimed.has(nb.node)) continue;
                plan.assigned = { node: nb.node, dirIdx: nb.dirIdx, score: -999, _needsMaterialise: false, _scId: undefined };
                octClaimed.add(nb.node);
                _occAdd(allBlocked, nb.node);
                escaped = true;
                _logChoreo(`X${_demoXons.indexOf(plan.xon)} collision->weak n${nb.node}`);
                break;
            }
            if (!escaped) {
                _logChoreo(`X${_demoXons.indexOf(plan.xon)} STUCK: no tet, no weak escape`);
            }
        }
    }
    occupied = _occupiedNodes(); // refresh after diversions

    // If idle_tet manifestation added new SCs, flag solver
    if (_idleTetManifested) _solverNeeded = true;

    // Log PHASE 2 decisions for debugging
    _logPhase2Summary(octPlans);

    // 2-step verification: Removed — backtracker handles downstream violations

    } // end PHASE 2 block
    const _pT2 = performance.now(); _profPhases.p2 += _pT2 - _pT1;

    // ── PHASE 3: Execute all planned moves ──
    // All moves execute in a single pass — simultaneous, not ordered.
    // If an oct move fails (vacuum rejection), revoke dependent tet approvals.

    // Build reverse map: oct xon → tet plan that depends on it vacating
    const octToTetDep = new Map(); // oct xon → tet plan
    for (const plan of tetPlans) {
        if (plan._needsOctVacate) octToTetDep.set(plan._needsOctVacate, plan);
    }

    // Execute oct moves first (includes idle_tet advances)
    for (const plan of octPlans) {
        if (plan.assigned) {
            if (plan.xon._movedThisTick) continue; // already moved (WB scatter/return) — one hop per tick
            const target = plan.assigned;
            const fromNode = plan.xon.node;
            // T41 swap check: reject if another xon just moved FROM target.node TO fromNode
            if (_swapBlocked(fromNode, target.node)) {
                const depTet = octToTetDep.get(plan.xon);
                if (depTet) { depTet.approved = false; planned.delete(depTet.toNode); }
                continue;
            }
            _occDel(occupied, plan.xon.node);
            const ok = _executeOctMove(plan.xon, target);
            if (!ok) {
                // Vacuum rejected at execution time — xon stays put
                // Revoke any tet move that depended on this xon vacating
                const depTet = octToTetDep.get(plan.xon);
                if (depTet) {
                    depTet.approved = false;
                    planned.delete(depTet.toNode);
                }
            } else {
                anyMoved = true;
                plan.xon._movedThisTick = true; // prevent double-move in PHASE 3.5/4
                _moveRecord.set(plan.xon.node, fromNode); // T41: record dest→origin
                _traceMove(plan.xon, fromNode, plan.xon.node, 'p3oct');
                // Immediate mode transition for xons that moved off-oct.
                // Mode must change in the SAME tick as the move (T95: oct mode on oct nodes only).
                if (_octNodeSet && !_octNodeSet.has(plan.xon.node)) {
                    if (_cageCriticalXons.has(plan.xon)) {
                        // Gluon activation: cage-critical xon moved off-cage
                        plan.xon._mode = 'gluon';
                        plan.xon.col = GLUON_COLOR;
                        if (plan.xon.sparkMat) plan.xon.sparkMat.color.setHex(GLUON_COLOR);
                        _logChoreo(`GLUON: X${_demoXons.indexOf(plan.xon)} moved off-cage ${fromNode}→${plan.xon.node} → gluon mode`);
                    } else if (plan.xon._pendingWeakEjection) {
                        // Pending-weak xon stepped off oct — transition immediately
                        plan.xon._pendingWeakEjection = false;
                        plan.xon._mode = 'weak';
                        plan.xon._t60Ejected = true;
                        plan.xon.col = WEAK_FORCE_COLOR;
                        if (plan.xon.sparkMat) plan.xon.sparkMat.color.setHex(WEAK_FORCE_COLOR);
                        _weakLifecycleEnter(plan.xon, 'pending_ejection_offcage');
                        _logChoreo(`X${_demoXons.indexOf(plan.xon)} pending-weak → weak at node ${plan.xon.node} (off oct, immediate)`);
                    }
                }
                if (plan.xon._solverNeeded) {
                    _solverNeeded = true;
                    plan.xon._solverNeeded = false;
                }
            }
            _occAdd(occupied, plan.xon.node);
        } else if (plan.idleTet) {
            // Verify SC is still active (may have been severed by oct move negotiation)
            if (!_canAdvanceSafely(plan.xon)) {
                _returnXonToOct(plan.xon, occupied); // abort idle_tet — SC was deactivated
                plan.xon._evictedThisTick = true;    // safety eviction (T52 exempt)
                continue;
            }
            // Pauli check: destination may have become occupied since planning
            const effectiveStep = plan.xon._loopStep >= 4 ? 0 : plan.xon._loopStep;
            const idleDest = plan.xon._loopSeq[effectiveStep + 1];
            if (idleDest !== undefined && (occupied.get(idleDest) || 0) > 0) {
                _returnXonToOct(plan.xon, occupied); // destination occupied — return to oct
                plan.xon._evictedThisTick = true;    // safety eviction (T52 exempt)
                continue;
            }
            // T41 swap check: reject if another xon just moved FROM idleDest TO xon.node
            if (idleDest !== undefined && _swapBlocked(plan.xon.node, idleDest)) {
                _returnXonToOct(plan.xon, occupied); // would swap — return to oct
                plan.xon._evictedThisTick = true;    // safety eviction (T52 exempt)
                continue;
            }
            const fromNode = plan.xon.node;
            _occDel(occupied, plan.xon.node);
            _advanceXon(plan.xon);
            _occAdd(occupied, plan.xon.node);
            _moveRecord.set(plan.xon.node, fromNode); // T41: record idle_tet move
            _traceMove(plan.xon, fromNode, plan.xon.node, 'p3idle');

            anyMoved = true;
        }
    }

    // Execute approved tet moves
    for (const plan of tetPlans) {
        if (!plan.approved) continue;
        // Final Pauli safety check before executing
        if ((occupied.get(plan.toNode) || 0) > 0) {
            if (!plan._annihilateMove) continue; // destination still occupied — skip to prevent collision
            // Annihilation move: allow advance into occupied node ONLY if occupant is non-weak.
            // Weak xons are protected from non-local annihilation (T38).
            const occupant = _demoXons.find(x => x.alive && x.node === plan.toNode && x !== plan.xon);
            if (occupant && occupant._mode === 'weak') continue; // don't collide with returning weak xon
        }
        // T41 swap check: reject if any xon just moved FROM plan.toNode TO plan.xon.node
        if (_swapBlocked(plan.xon.node, plan.toNode)) continue;
        // Verify SC is still active (may have been severed by oct move negotiation)
        if (!_canAdvanceSafely(plan.xon)) continue;
        const tetFrom = plan.xon.node;
        _advanceXon(plan.xon);
        _occDel(occupied, plan.xon.prevNode);
        _occAdd(occupied, plan.xon.node);
        _moveRecord.set(plan.xon.node, tetFrom); // T41: record tet move
        _traceMove(plan.xon, tetFrom, plan.xon.node, 'p3tet');

        anyMoved = true;
    }

    // ── Stuck tet ejection: eject as weak particle instead of staying stuck ──
    // If a tet/idle_tet xon couldn't move this tick (plan unapproved, blocked,
    // or vacuum-rejected), eject it as a weak particle with _mayReturn so it
    // can navigate back to the oct cage and re-enter the nucleus later.
    for (const plan of tetPlans) {
        if (plan.xon._movedThisTick) continue; // already moved — no problem
        if (plan.xon.node !== plan.fromNode) continue; // moved successfully
        // This tet xon is stuck — eject as weak particle
        const xi = _demoXons.indexOf(plan.xon);
        _logChoreo(`X${xi} tet stuck at n${plan.fromNode} (${plan.approved ? 'approved but blocked' : 'unapproved'}) → ejecting as weak`);
        _relinquishFaceSCs(plan.xon);
        _clearModeProps(plan.xon);
        plan.xon._mode = 'weak';
        plan.xon._t60Ejected = true;
        plan.xon._assignedFace = null;
        plan.xon._loopSeq = null;
        plan.xon._loopStep = 0;
        plan.xon._quarkType = null;
        plan.xon.col = WEAK_FORCE_COLOR;
        if (plan.xon.sparkMat) plan.xon.sparkMat.color.setHex(WEAK_FORCE_COLOR);
        _weakLifecycleEnter(plan.xon, 'tet_stuck_ejection');
        // Try to move to a free neighbor so T20 doesn't fire
        const stuckNbs = baseNeighbors[plan.xon.node] || [];
        for (const nb of stuckNbs) {
            if ((occupied.get(nb.node) || 0) > 0) continue;
            if (_swapBlocked(plan.xon.node, nb.node)) continue;
            const fromStuck = plan.xon.node;
            _occDel(occupied, plan.xon.node);
            plan.xon.prevNode = plan.xon.node;
            plan.xon.node = nb.node;
            // Proxy may have blocked — verify
            if (plan.xon.node !== nb.node) { _occAdd(occupied, plan.xon.node); continue; }
            _occAdd(occupied, nb.node);
            plan.xon._movedThisTick = true;
            _moveRecord.set(nb.node, fromStuck);
            _traceMove(plan.xon, fromStuck, nb.node, 'tetStuckEject');
            plan.xon.tweenT = 0;
            anyMoved = true;
            break;
        }
    }

    const _pT3 = performance.now(); _profPhases.p3 += _pT3 - _pT2;
    _pT5 = _pT3; // PHASE 3b/4/5 removed — bridge profiling timer to solver

    // PHASE 3b: Removed — backtracker handles stuck oct xons via rewind

    // PHASE 3.5: Removed — PHASE 0.5 handles all weak xon movement

    // ── PENDING WEAK EJECTION: transition xons that stepped off oct ──
    // A xon with _pendingWeakEjection stays in oct mode until it physically
    // lands on a non-oct node. Only THEN does it become weak.
    for (const xon of _demoXons) {
        if (!xon.alive || !xon._pendingWeakEjection) continue;
        const onOct = _octNodeSet && _octNodeSet.has(xon.node);
        if (!onOct) {
            // Stepped off oct — now transition to weak
            const xi = _demoXons.indexOf(xon);
            _logChoreo(`X${xi} pending-weak → weak at node ${xon.node} (off oct)`);
            xon._pendingWeakEjection = false;
            xon._mode = 'weak';
            xon._t60Ejected = true;
            xon.col = WEAK_FORCE_COLOR;
            if (xon.sparkMat) xon.sparkMat.color.setHex(WEAK_FORCE_COLOR);
            _weakLifecycleEnter(xon, 'pending_ejection_offcage');
        }
        // If still on oct: stays in oct mode with _pendingWeakEjection — will try again next tick
    }

    // ── POST-MOVE WEAK→OCT TRANSITION SWEEP ──
    // Any weak xon that landed on an oct node becomes oct immediately.
    for (const xon of _demoXons) {
        if (!xon.alive) continue;
        if (xon._mode !== 'weak') continue;
        if (!(_octNodeSet && _octNodeSet.has(xon.node))) continue;
        const xi = _demoXons.indexOf(xon);
        _weakLifecycleExit(xon, 'post_move_oct_arrival');
        _clearModeProps(xon);
        xon._mode = 'oct';
        if (_flashEnabled) xon.flashT = 1.0;
        xon.col = 0xffffff;
        if (xon.sparkMat) xon.sparkMat.color.setHex(0xffffff);
    }

    // ── POST-EXECUTION PAULI CHECK (replaces PHASE 4) ──
    // If planning was correct, no collisions exist. If one slipped through, trigger backtrack.
    // Post-exec Pauli check: enforced in both demo and tournament mode.
    {
        const _p4occ = new Map();
        for (const xon of _demoXons) {
            if (!xon.alive) continue;
            _p4occ.set(xon.node, (_p4occ.get(xon.node) || 0) + 1);
        }
        for (const [n, c] of _p4occ) {
            if (c > 1) {
                _rewindRequested = true;
                _rewindViolation = `POST-EXEC Pauli: node ${n} has ${c} xons`;
                break;
            }
        }
    }

    // Final safety net: Removed — backtracker handles stuck xons via rewind

    // PHASE 5: Removed — backtracker is the universal deadlock handler (see t=0 canary)
    } // end !_skipNormalPhases

    // ── Advance gluons along oct edges (also negotiates with vacuum) ──
    if (_advanceGluons()) _solverNeeded = true;

    // ── T90: First-place quark ejection ──
    // No direct enforcement here. T90 live guard detects first-place overstay
    // and fires into the backtracker, which rewinds and lets the choreographer
    // find a competing tet excitation that naturally destroys the dominant tet
    // through geometric mutual exclusion (the "octa immune system").

    // ── Run solver if any SCs changed (unified architecture) ──
    if (_solverNeeded) {
        _planckSeconds++;
        _recordBalanceSample();
        if (typeof _ppoDeformationThisTick !== 'undefined') _ppoDeformationThisTick = true;
    }
    if (_solverNeeded) {
        bumpState(true); // skip voids — solver will reposition everything
        const scPairs = [];
        activeSet.forEach(id => { const s = SC_BY_ID[id]; scPairs.push([s.a, s.b]); });
        xonImpliedSet.forEach(id => { const s = SC_BY_ID[id]; scPairs.push([s.a, s.b]); });
        const { p: pSolved } = _solve(scPairs, 5000, true); // noBailout: full convergence for Kepler
        impliedSet.clear(); impliedBy.clear();
        xonImpliedSet.forEach(id => {
            if (!activeSet.has(id)) { impliedSet.add(id); impliedBy.set(id, new Set()); }
        });
        stateVersion++;
        applyPositions(pSolved);
        // Re-freeze any trail entries captured before solver ran (stale [0,0,0])
        for (const xon of _demoXons) {
            if (!xon || !xon.alive || !xon.trail) continue;
            for (let ti = 0; ti < xon.trail.length; ti++) {
                const e = xon.trail[ti];
                if (!e.pos || (e.pos[0] === 0 && e.pos[1] === 0 && e.pos[2] === 0)) {
                    const p = pos[e.node];
                    if (p && (p[0] !== 0 || p[1] !== 0 || p[2] !== 0)) {
                        e.pos = [p[0], p[1], p[2]];
                    }
                }
            }
        }
        updateVoidSpheres(); // evaluate with final solved positions
        updateSpheres();
    }

    // ── KEPLER + INVARIANT CHECKS (every tick, non-negotiable) ──
    // Fast path: density check + edge/SC/repulsion validation.
    // These iterate flat arrays — total <1ms per tick.
    {
        // 1. Kepler density
        const _actualDens = computeActualDensity() * 100;
        const _idealDens = computeIdealDensity() * 100;
        const _densDev = Math.abs(_actualDens - _idealDens);
        if (_densDev > 0.01) {
            _keplerViolation(_actualDens, _idealDens);
        }
        const _densEl = document.getElementById('st-dens');
        if (_densEl) {
            _densEl.textContent = _actualDens.toFixed(4) + '%';
            _densEl.style.color = _densDev < 0.001 ? '#6a8aaa' : _densDev < 0.01 ? '#ffaa44' : '#ff4444';
        }

        // 2. Edge/SC/repulsion invariants (same checks as updateStatus but no side panel)
        const TOL = 1e-3;
        let violation = null;
        for (const [i,j] of BASE_EDGES) {
            const err = Math.abs(vd(pos[i],pos[j]) - 1.0);
            if (err > TOL) { violation = `R1 base edge v${i}-v${j} err=${err.toFixed(5)}`; break; }
        }
        if (!violation) {
            for (const id of activeSet) {
                const s = SC_BY_ID[id];
                const err = Math.abs(vd(pos[s.a],pos[s.b]) - 1.0);
                if (err > TOL) { violation = `R2 shortcut sc${id} v${s.a}-v${s.b} err=${err.toFixed(5)}`; break; }
            }
        }
        if (!violation) {
            for (const [i,j] of REPULSION_PAIRS) {
                const d = vd(pos[i],pos[j]);
                if (d < 1.0 - TOL) { violation = `R3 overlap v${i}-v${j} dist=${d.toFixed(5)}`; break; }
            }
        }
        if (violation) {
            // Soft recovery: try clearing xon-implied SCs (protect oct cage)
            // When relinquishment is off (and no gluon mediation), violations should halt
            if (_ruleGluonMediatedSC && xonImpliedSet.size && !simHalted) {
                // Release all gluon bindings first
                for (const x of _demoXons) {
                    if (x.alive && x._gluonForFace != null) {
                        x._gluonForFace = null; x._gluonBoundSCs = null; x._gluonClientXon = null;
                        x._mode = 'oct'; x.col = 0xffffff;
                        if (x.sparkMat) x.sparkMat.color.setHex(0xffffff);
                    }
                }
                const _cageSCSet = _octSCIds ? new Set(_octSCIds) : new Set();
                for (const id of [...xonImpliedSet]) {
                    if (_cageSCSet.has(id)) continue;
                    xonImpliedSet.delete(id); impliedSet.delete(id); impliedBy.delete(id);
                }
                bumpState();
                const pFinal = detectImplied();
                applyPositions(pFinal);
                let stillBad = false;
                for (const [i,j] of BASE_EDGES) {
                    if (Math.abs(vd(pos[i],pos[j]) - 1.0) > TOL) { stillBad = true; break; }
                }
                if (!stillBad) { /* recovered */ }
                else {
                    simHalted = true;
                    document.getElementById('violation-msg').textContent = 'HALTED: ' + violation;
                    document.getElementById('violation-banner').style.display = 'block';
                }
            } else if (_ruleRelinquishSCs && xonImpliedSet.size && !simHalted) {
                const _cageSCSet = _octSCIds ? new Set(_octSCIds) : new Set();
                for (const id of [...xonImpliedSet]) {
                    if (_cageSCSet.has(id)) continue; // NEVER clear oct cage SCs
                    xonImpliedSet.delete(id); impliedSet.delete(id); impliedBy.delete(id);
                }
                bumpState();
                const pFinal = detectImplied();
                applyPositions(pFinal);
                // Re-check after recovery
                let stillBad = false;
                for (const [i,j] of BASE_EDGES) {
                    if (Math.abs(vd(pos[i],pos[j]) - 1.0) > TOL) { stillBad = true; break; }
                }
                if (!stillBad) { /* recovered */ }
                else {
                    simHalted = true;
                    document.getElementById('violation-msg').textContent = 'HALTED: ' + violation;
                    document.getElementById('violation-banner').style.display = 'block';
                }
            } else if (!simHalted) {
                simHalted = true;
                document.getElementById('violation-msg').textContent = 'HALTED: ' + violation;
                document.getElementById('violation-banner').style.display = 'block';
            }
        }
    }

    // SC cleanup: remove ONLY non-unit-length SCs from xonImpliedSet (per spec §9).
    // All unit-length SCs remain as traversal paths and severance options.
    // Attribution is kept for diagnostics only — not a removal criterion.
    {
        const locked = typeof _traversalLockedSCs === 'function' ? _traversalLockedSCs() : new Set();
        const toRemove = [];
        for (const scId of xonImpliedSet) {
            if (activeSet.has(scId)) continue;  // not xonImpliedSet's responsibility
            if (locked.has(scId)) continue;     // xon currently traversing
            // Distance check: is this SC still approximately unit-length?
            const sc = SC_BY_ID[scId];
            if (!sc) { toRemove.push(scId); continue; }
            const pa = pos[sc.a], pb = pos[sc.b];
            if (!pa || !pb) continue; // pos not ready
            const dx = pb[0]-pa[0], dy = pb[1]-pa[1], dz = pb[2]-pa[2];
            const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
            if (Math.abs(dist - 1) > 0.15) {
                toRemove.push(scId); // non-unit-length → remove
            }
            // Flag SCs with no attribution for diagnostics (but don't remove)
            if (!_scAttribution.has(scId)) {
                _scAttribution.set(scId, { reason: 'none', xonIdx: -1, tick: _demoTick });
            }
        }
        const _cageSCCleanup = _octSCIds ? new Set(_octSCIds) : new Set();
        for (const id of toRemove) {
            if (_cageSCCleanup.has(id)) continue; // NEVER remove oct cage SCs
            xonImpliedSet.delete(id);
            _scAttribution.delete(id);
            stateVersion++;
        }
    }

    const _pTsolver = performance.now(); _profPhases.solver += _pTsolver - _pT5;

    // ── Decay dying xon trails (every simulation tick, not per-frame) ──
    _decayDyingXons();

    // ── Color tets based on geometric actualization (all SCs active) ──
    _applyTetColoring(true); // true = count actualization visits during live ticks
    // ── Per-xon/role SC opacity for shapes layer ──
    if (typeof _roleOpacity !== 'undefined' && typeof _ruleAnnotations !== 'undefined') {
        _ruleAnnotations.scOpacity.clear();
        const graphOp = +(document.getElementById('graph-opacity-slider')?.value || 1) / 100;
        for (let xi = 0; xi < _demoXons.length; xi++) {
            const xon = _demoXons[xi];
            if (!xon || !xon.alive) continue;
            const xoi = _xonOpacity[xi] != null ? _xonOpacity[xi] : 1;
            const role = typeof _xonRole === 'function' ? _xonRole(xon) : 'oct';
            const roleOp = _roleOpacity[role] != null ? _roleOpacity[role] : 1;
            const op = xoi * roleOp * graphOp;
            // Tet/idle_tet: annotate face SCs
            if ((xon._mode === 'tet' || xon._mode === 'idle_tet') && xon._assignedFace != null && _nucleusTetFaceData) {
                const fd = _nucleusTetFaceData[xon._assignedFace];
                if (fd && fd.scIds) {
                    for (const scId of fd.scIds) _ruleAnnotations.scOpacity.set(scId, op);
                }
            }
            // Gluon: annotate bound SCs
            if (xon._mode === 'gluon' && xon._gluonBoundSCs) {
                for (const scId of xon._gluonBoundSCs) _ruleAnnotations.scOpacity.set(scId, op);
            }
        }
        // Oct cage SCs: use oct role opacity × min of oct xon opacities
        if (_octSCIds && _roleOpacity.oct != null) {
            let minOctXoi = 1;
            for (let xi = 0; xi < _demoXons.length; xi++) {
                const xon = _demoXons[xi];
                if (xon && xon.alive && xon._mode === 'oct') {
                    minOctXoi = Math.min(minOctXoi, _xonOpacity[xi] != null ? _xonOpacity[xi] : 1);
                }
            }
            const octOp = minOctXoi * _roleOpacity.oct * graphOp;
            for (const scId of _octSCIds) {
                const prev = _ruleAnnotations.scOpacity.get(scId);
                _ruleAnnotations.scOpacity.set(scId, prev != null ? Math.min(prev, octOp) : octOp);
            }
        }
        _ruleAnnotations.dirty = true;
    }
    if (typeof updateVoidSpheres === 'function') updateVoidSpheres();

    const _pTrender = performance.now(); _profPhases.render += _pTrender - _pTsolver;

    // ── TRAIL: one entry per xon per tick (unified push) ──
    // All movement is resolved. Record FINAL position and role.
    // Always push — even if xon didn't move — so the trail captures role changes.
    // _trailRecolor is gone; this is the only place trail entries are created.
    for (const xon of _demoXons) {
        if (!xon.alive || xon._dying) continue;
        _trailPush(xon, xon.node);
    }

    // T79: track consecutive full-oct ticks (for next tick's overflow pressure)
    if (_octNodeSet && _demoXons.filter(x => x.alive && _octNodeSet.has(x.node)).length >= 6) {
        _octFullConsecutive++;
    } else {
        _octFullConsecutive = 0;
    }

    // Track consecutive oct ticks per xon (Rule 11 / T97)
    for (const xon of _demoXons) {
        if (!xon.alive) continue;
        if (xon._mode === 'oct') {
            xon._octConsecutive = (xon._octConsecutive || 0) + 1;
        } else {
            xon._octConsecutive = 0;
        }
    }
    if (_demoOctRevealed) _ticksSinceLastQuark++; // Only count after oct forms — reset in _advanceXon if quark actualized
    _demoTick++;
    if (_demoTick > _maxTickReached) {
        _maxTickReached = _demoTick;
    }
    // Size-based GC: cap traversal log regardless of forward progress
    if (_searchTraversalLog && _searchTraversalLog.length > 2000) {
        _searchTraversalLog.splice(0, _searchTraversalLog.length - 1000);
    }
    // Capture fingerprint of the tick that achieved the new high-water mark
    if (_demoTick > _maxTickReached && typeof _computeTickFingerprint === 'function') {
        _bestPathFingerprint = _computeTickFingerprint();
    }

    // Update tick + Planck-second ticker (both right-panel status and left-panel title)
    const _tickerEl = document.getElementById('nucleus-status');
    if (_tickerEl) _tickerEl.innerHTML = `${_planckSeconds} Flux Events<br><span style="font-size:0.8em; color:#556677;">${_demoTick} Planck Seconds</span>`;
    if (typeof _updateTickCounter === 'function') _updateTickCounter();
    // Top-center title is set once per trial by _runTournament — no per-tick update needed

    // Live guard checks (T19, T21, T26, T27) — after tick advances xons
    const _gT0 = performance.now();
    if (typeof _liveGuardCheck === 'function') _liveGuardCheck();
    const _gT1 = performance.now();

    // ── Record fingerprint for SUCCESSFUL ticks (guards passed) ──
    // This is essential for exhaustiveness testing: both successful and failed
    // fingerprints must be recorded so Test 2 (random) can detect novel solutions.
    if (!_rewindRequested && typeof _btRecordFingerprint === 'function') {
        _btRecordFingerprint();
        // Capture moves for golden path candidate (sweep mode)
        if (_sweepActive && _moveTrace && _moveTrace.length > 0) {
            const gt = _demoTick - 1;
            if (!_sweepSeedMoves) _sweepSeedMoves = new Map();
            const tickMoves = new Map();
            for (const t of _moveTrace) tickMoves.set(t.xonIdx, t.to);
            _sweepSeedMoves.set(gt, tickMoves);
        }
        // Log successful tick advancement with tree structure
        if (_btActive && _searchTraversalLog) {
            const successTick = _demoTick - 1;
            const fp = typeof _computeTickFingerprint === 'function' ? _computeTickFingerprint() : '';
            const pathKey = _searchPathStack.slice(0, successTick).join('\u2192') + ':' + fp;
            const nodeId = _fnv1aHash(pathKey);
            _searchTraversalLog.push({
                eventId: _searchEventCounter++, nodeId, parentId: _searchParentNodeId,
                tick: successTick, retry: _btRetryCount, bfsLayer: _bfsLayer,
                relayPhase: typeof _relayPhase !== 'undefined' ? _relayPhase : 'normal',
                fingerprint: fp, outcome: 'success', wall: null,
                moves: _moveTrace.map(t => ({ xonIdx: t.xonIdx, from: t.from, to: t.to, mode: t.mode })),
                candidates: _searchLastCandidates || {},
                cumulativeExclusions: _btBadMoveLedger.has(successTick) ? [..._btBadMoveLedger.get(successTick)] : [],
                exclusionTotal: _btBadMoveLedger.has(successTick) ? _btBadMoveLedger.get(successTick).size : 0,
                xonPositions: _demoXons.filter(x => x && x.alive).map(x => x.node),
                xonModes: _demoXons.filter(x => x && x.alive).map(x => x._mode),
                activeSCs: activeSet.size + xonImpliedSet.size,
                elapsed: _searchStartTime ? (performance.now() - _searchStartTime) / 1000 : 0,
            });
            // Update tree state: this success becomes the parent for the next tick
            _searchPathStack[successTick] = fp;
            _searchPathStack.length = successTick + 1;
            _searchParentNodeId = nodeId;
        }
    }

    // ── GC + Autosave on advancement milestones (every 100 NEW high-water ticks) ──
    // Fires when _maxTickReached crosses a new 100-tick boundary since last milestone.
    // GC always runs; autosave saves to council if eligible.
    {
        const _asEl = document.getElementById('autosave-interval');
        const _asInterval = _asEl ? +_asEl.value : 100;
        const nextMilestone = _lastAutosavePeak + _asInterval;
        if (_maxTickReached >= nextMilestone) {
            _lastAutosavePeak = Math.floor(_maxTickReached / _asInterval) * _asInterval;
            _gc10();
            // Only autosave when we're AT the peak (not replaying old ground)
            if (_demoTick >= _maxTickReached
                && typeof _isCouncilEligible === 'function' && _isCouncilEligible()
                && typeof _saveCurrentRunToCouncil === 'function') {
                _saveCurrentRunToCouncil();
            }
            if (!_testRunning && typeof _updateSweepPanel === 'function' && _sweepActive) {
                _updateSweepPanel(null, _searchStartTime);
            }
        }
    }

    // ── Sweep panel update ──
    if (!_testRunning && typeof _updateSweepPanel === 'function' && _sweepActive) {
        _updateSweepPanel(null, _searchStartTime);
    }

    // ── Council replay: suppress all rewinds during the happy path ──
    if (_inReplayPhase && _rewindRequested) {
        _rewindRequested = false;
        _rewindViolation = null;
    }

    // ── BACKTRACK CHECK (BFS): did guards request a rewind? ──
    if (_rewindRequested) {
        _rewindRequested = false;
        _btActive = true;
        _totalBacktrackRetries++;
        // Signal RL reward: penalize moves that trigger backtracks
        if (typeof _ppoBacktracksThisTick !== 'undefined') _ppoBacktracksThisTick++;

        // Record fingerprint of the failed attempt for deduplication
        const _isNewFingerprint = (typeof _btRecordFingerprint === 'function') ? _btRecordFingerprint() : true;

        // Extract exclusions and accumulate in persistent ledger.
        const newExclusions = _btExtractExclusions();
        const currentTick = _demoTick - 1; // tick was already incremented
        if (!_btBadMoveLedger.has(currentTick)) _btBadMoveLedger.set(currentTick, new Set());
        const ledger = _btBadMoveLedger.get(currentTick);
        let _addedNewExclusions = false;
        for (const ex of newExclusions) {
            if (!ledger.has(ex)) _addedNewExclusions = true;
            ledger.add(ex);
        }

        // Log rewind event with wall classification and tree structure
        if (_searchTraversalLog) {
            const fp = typeof _computeTickFingerprint === 'function' ? _computeTickFingerprint() : '';
            let wall;
            if (_rewindViolation && _rewindViolation.includes('Kepler')) {
                wall = { type: 'kepler', details: [_rewindViolation],
                    keplerDensity: typeof _actualDens !== 'undefined' ? _actualDens : null,
                    keplerIdeal: typeof _idealDens !== 'undefined' ? _idealDens : null };
            } else if (_rewindViolation && _rewindViolation.includes('POST-EXEC Pauli')) {
                wall = { type: 'pauli-postexec', details: [_rewindViolation] };
            } else {
                // Parse guard ID from violation string (e.g. "T20: tick 4: stuck...")
                const guardMatch = _rewindViolation?.match(/^(T\d+\w*)/);
                if (guardMatch) {
                    wall = { type: 'liveguard', details: [_rewindViolation] };
                } else {
                    wall = { type: 'other', details: [_rewindViolation || 'unknown'] };
                }
            }
            const pathKey = _searchPathStack.slice(0, currentTick).join('\u2192') + ':' + fp;
            const nodeId = _fnv1aHash(pathKey);
            _searchTraversalLog.push({
                eventId: _searchEventCounter++, nodeId, parentId: _searchParentNodeId,
                tick: currentTick, retry: _btRetryCount, bfsLayer: _bfsLayer,
                relayPhase: typeof _relayPhase !== 'undefined' ? _relayPhase : 'normal',
                fingerprint: fp, outcome: 'rewind', wall,
                moves: _moveTrace.map(t => ({ xonIdx: t.xonIdx, from: t.from, to: t.to, mode: t.mode })),
                candidates: _searchLastCandidates || {},
                cumulativeExclusions: [...ledger],
                exclusionTotal: ledger.size,
                xonPositions: _demoXons.filter(x => x && x.alive).map(x => x.node),
                xonModes: _demoXons.filter(x => x && x.alive).map(x => x._mode),
                activeSCs: activeSet.size + xonImpliedSet.size,
                elapsed: _searchStartTime ? (performance.now() - _searchStartTime) / 1000 : 0,
            });
        }

        // ── ESCALATION HELPER: attempt to go one layer deeper ──
        // Called when current layer is exhausted (no new exclusions).
        // Returns false if we've reached t=0 (canary: rules are impossible).
        const _escalateLayer = () => {
            if (_bfsFailTick >= 0 && currentTick === _bfsFailTick) {
                // At the failure tick after deeper-layer replay — go deeper
                _bfsLayer++;
                _bfsLayerRetries = 0;
            } else if (_bfsFailTick < 0) {
                // First time exhausting — this tick is now the BFS failure tick
                _bfsFailTick = currentTick;
                _bfsLayer = 1;
                _bfsLayerRetries = 0;
                _logChoreo(`BFS: tick ${currentTick} exhausted at layer 0, starting BFS layer 1`);
            } else {
                // Non-failure tick failed during forward replay — escalate
                _bfsLayer++;
                _bfsLayerRetries = 0;
                _logChoreo(`BFS: intermediate tick ${currentTick} also failed, escalating to layer ${_bfsLayer}`);
            }

            const targetTick = _bfsFailTick - _bfsLayer;

            // Eagerly prefetch blacklist buckets for the rewind range
            if (_blBucketVersion >= 1 && typeof _blPrefetchRange === 'function') {
                const _lvl = typeof latticeLevel !== 'undefined' ? latticeLevel : 2;
                console.log(`[BL] Layer escalation → prefetching ticks ${Math.max(0, targetTick)}-${_bfsFailTick}`);
                _blPrefetchRange(_lvl, Math.max(0, targetTick), _bfsFailTick); // fire-and-forget
            }

            // Log escalation event with tree structure
            if (_searchTraversalLog) {
                const escPathKey = _searchPathStack.slice(0, currentTick).join('\u2192') + ':escalate-L' + _bfsLayer;
                const escNodeId = _fnv1aHash(escPathKey);
                _searchTraversalLog.push({
                    eventId: _searchEventCounter++, nodeId: escNodeId, parentId: _searchParentNodeId,
                    tick: currentTick, retry: _btRetryCount, bfsLayer: _bfsLayer,
                    relayPhase: typeof _relayPhase !== 'undefined' ? _relayPhase : 'normal',
                    fingerprint: '', outcome: 'escalate',
                    wall: { type: 'stale', details: [`escalated to layer ${_bfsLayer}, target tick ${targetTick}`] },
                    moves: [], candidates: {},
                    cumulativeExclusions: ledger ? [...ledger] : [],
                    exclusionTotal: ledger ? ledger.size : 0,
                    xonPositions: _demoXons.filter(x => x && x.alive).map(x => x.node),
                    xonModes: _demoXons.filter(x => x && x.alive).map(x => x._mode),
                    activeSCs: activeSet.size + xonImpliedSet.size,
                    elapsed: _searchStartTime ? (performance.now() - _searchStartTime) / 1000 : 0,
                });
                // Truncate path stack to target tick and recompute parent
                _searchPathStack.length = Math.max(0, targetTick);
                if (_searchPathStack.length > 0) {
                    const lastFp = _searchPathStack[_searchPathStack.length - 1];
                    const parentKey = _searchPathStack.slice(0, -1).join('\u2192') + ':' + lastFp;
                    _searchParentNodeId = _fnv1aHash(parentKey);
                } else {
                    _searchParentNodeId = null;
                }
            }

            // ── SWEEP EARLY-OUT: Skip seed if backtracker has regressed too far.
            // Threshold: min(100, 50% of peak). Disabled if peak < 10 ticks. ──
            if (_sweepActive && _maxTickReached >= 10) {
                const threshold = Math.min(100, Math.floor(_maxTickReached * 0.5));
                if (_maxTickReached - targetTick >= threshold) {
                    console.error(`%c[SWEEP SKIP] Seed regressed ${_maxTickReached - targetTick} ticks below peak (peak=${_maxTickReached}, target=${targetTick}, threshold=${threshold}). Skipping to next seed.`, 'color:orange;font-weight:bold');
                    _saveRunResult('sweep-skip', `backtracked to ${targetTick}, peak was ${_maxTickReached}`);
                    simHalted = true;
                    _btReset();
                    _bfsReset();
                    return false;
                }
            }

            // ── t=0 CANARY: If we've backed all the way past tick 0, the rules
            // are mathematically impossible. This is the ONLY true failure. ──
            if (targetTick < 0) {
                // Log canary event with tree structure
                if (_searchTraversalLog) {
                    const canaryPathKey = _searchPathStack.join('\u2192') + ':canary';
                    const canaryNodeId = _fnv1aHash(canaryPathKey);
                    _searchTraversalLog.push({
                        eventId: _searchEventCounter++, nodeId: canaryNodeId, parentId: _searchParentNodeId,
                        tick: currentTick, retry: _btRetryCount, bfsLayer: _bfsLayer,
                        relayPhase: typeof _relayPhase !== 'undefined' ? _relayPhase : 'normal',
                        fingerprint: '', outcome: 'canary',
                        wall: { type: 'canary', details: [_rewindViolation || 'rules impossible'] },
                        moves: [], candidates: {},
                        cumulativeExclusions: [], exclusionTotal: 0,
                        xonPositions: _demoXons.filter(x => x && x.alive).map(x => x.node),
                        xonModes: _demoXons.filter(x => x && x.alive).map(x => x._mode),
                        activeSCs: activeSet.size + xonImpliedSet.size,
                        elapsed: _searchStartTime ? (performance.now() - _searchStartTime) / 1000 : 0,
                    });
                }
                console.error(`%c[CANARY] Rules are mathematically impossible — backtracker exhausted all possibilities to t=0. Last violation: ${_rewindViolation}`, 'color:red;font-weight:bold;font-size:14px');
                _saveRunResult('canary-t0', _rewindViolation);
                simHalted = true;
                _btReset();
                _bfsReset();
                return false;
            }

            const anchorSnap = _btSnapshots.find(s => s.tick === targetTick);
            if (!anchorSnap || anchorSnap._pruned) {
                // No snapshot (or pruned) for this tick — we've exhausted snapshot history.
                // This is equivalent to reaching t=0.
                console.error(`%c[CANARY] Rules are mathematically impossible — no snapshot for tick ${targetTick} (${anchorSnap ? 'pruned' : 'missing'}, backed to t=0). Last violation: ${_rewindViolation}`, 'color:red;font-weight:bold;font-size:14px');
                _saveRunResult('canary-no-snap', _rewindViolation);
                simHalted = true;
                _btReset();
                _bfsReset();
                return false;
            }

            // Clear ledger entries and fingerprints for ticks after anchor (state will diverge)
            // Collect keys first to avoid deleting during iteration (undefined behavior).
            const _staleKeys = [..._btBadMoveLedger.keys()].filter(t => t > targetTick);
            for (const t of _staleKeys) _btBadMoveLedger.delete(t);
            // During sweep, preserve fingerprints for cross-seed blacklist harvesting.
            // They're still valid dead-end states even though local state will diverge.
            if (!_sweepActive) {
                const _staleFPKeys = [..._btTriedFingerprints.keys()].filter(t => t > targetTick);
                for (const t of _staleFPKeys) _btTriedFingerprints.delete(t);
            }
            // Trim stale snapshots from the failed forward path so rewind
            // playback only shows the happy path, not BFS mistakes.
            while (_btSnapshots.length > 0 && _btSnapshots[_btSnapshots.length - 1].tick > targetTick) {
                _btSnapshots.pop();
            }
            // Adjust cold boundary if we rewound into the cold region
            if (_btColdBoundary > _btSnapshots.length) {
                _btColdBoundary = _btSnapshots.length;
            }
            // Snapshots <= targetTick are kept intact for rewind/playback.
            // Also trim tick log — entries past the anchor tick will be
            // re-generated when the BFS replays forward with different moves.
            while (_tickLog.length > 0 && _tickLog[_tickLog.length - 1].tick >= targetTick) {
                _tickLog.pop();
            }
            // Reset guard delta baseline so replayed ticks get full guard state
            _tickLogLastGuards = {};
            _btRetryCount = 0;
            // Reset relay state — deeper layer starts fresh with normal choreographer
            _relayPhase = 'normal';
            _bfsTestRandomChoreographer = false;
            _relayEnumFingerprints = null;
            _relayEnumAttempts = 0;
            _relayEnumStale = 0;
            _relayScoredQueue = null;
            _relayScoredIndex = 0;
            _btRestoreSnapshot(anchorSnap);
            _updateTickCounter(); // show tick decrement during backtrack
            _logChoreo(`BFS: rewound to layer ${_bfsLayer} anchor tick ${targetTick}`);
            return true;
        };

        // ══════════════════════════════════════════════════════════════
        // THREE-PHASE EXHAUSTION: normal → enumerate → scored replay
        //
        // Phase 1 (normal): Greedy choreographer tries all matchings
        //   with PRNG-varied secondary choices. Escalates to Phase 2
        //   when stale (no new fingerprints or exclusions).
        //
        // Phase 2 (enumerating): Random scoring explores the full
        //   decision space (face assignments, quark types, etc.).
        //   Collects ALL valid fingerprints into _relayEnumFingerprints.
        //   Transitions to Phase 3 when stale.
        //
        // Phase 3 (replaying): Choreographer scores all enumerated
        //   fingerprints by preference, tries them best-first.
        //   Escalates to deeper BFS layer when all exhausted.
        // ══════════════════════════════════════════════════════════════
        const _relayBlocked = _bfsTestActive && _bfsTestRunIdx === 1;

        if (_relayPhase === 'replaying') {
            // ── PHASE 3: Scored replay — trying enumerated options in order ──
            // A failed replay attempt means this fingerprint's path also fails.
            // Move to the next scored option.
            _relayScoredIndex++;
            if (_relayScoredQueue && _relayScoredIndex < _relayScoredQueue.length) {
                // Try the next scored option
                const nextOpt = _relayScoredQueue[_relayScoredIndex];
                _logChoreo(`RELAY: tick ${currentTick} — replay option ${_relayScoredIndex + 1}/${_relayScoredQueue.length} (score: ${nextOpt.score.toFixed(1)})`);
                const snap = _btSnapshots[_btSnapshots.length - 1];
                _btRestoreSnapshot(snap);
                _btRetryCount++;
                // Seed PRNG to reproduce this fingerprint's decisions
                // (use a hash of the fingerprint string for reproducibility)
                let fpHash = 0;
                for (let i = 0; i < nextOpt.fp.length; i++) fpHash = ((fpHash << 5) - fpHash + nextOpt.fp.charCodeAt(i)) | 0;
                _sRngSeed(fpHash >>> 0);
                continue;
            }
            // All scored options exhausted — escalate to deeper layer
            _logChoreo(`RELAY: tick ${currentTick} — all ${_relayScoredQueue ? _relayScoredQueue.length : 0} scored options exhausted, escalating`);
            _relayPhase = 'normal';
            _bfsTestRandomChoreographer = false;
            _relayEnumFingerprints = null;
            _relayScoredQueue = null;
            _btStaleRetries = 0;
            _btResetMatchingCache();
            if (!_escalateLayer()) break;
            continue;
        }

        if (!_addedNewExclusions && !_isNewFingerprint) {
            if (typeof _btStaleRetries === 'undefined') _btStaleRetries = 0;
            _btStaleRetries++;
            const cacheSize = (_btMatchingCache ? _btMatchingCache.length : 1);
            const minRetries = Math.max(3, cacheSize * 3);

            if (_btStaleRetries >= minRetries) {
                if (_relayPhase === 'normal' && !_relayBlocked) {
                    // ── Transition: normal → enumerating ──
                    _relayPhase = 'enumerating';
                    _bfsTestRandomChoreographer = true;
                    _relayEnumFingerprints = new Map();
                    _relayEnumAttempts = 0;
                    _relayEnumStale = 0;
                    _btStaleRetries = 0;
                    _btResetMatchingCache();
                    _logChoreo(`RELAY: tick ${currentTick} — normal exhausted, starting enumeration phase`);
                    const snap = _btSnapshots[_btSnapshots.length - 1];
                    _btRestoreSnapshot(snap);
                    _btRetryCount++;
                    continue;
                }

                if (_relayPhase === 'enumerating') {
                    // Enumeration stale — check if we've found any options
                    _relayEnumStale++;
                    const enumMinStale = Math.max(5, cacheSize * 3);
                    if (_relayEnumStale >= enumMinStale) {
                        _bfsTestRandomChoreographer = false;
                        if (_relayEnumFingerprints && _relayEnumFingerprints.size > 0) {
                            // ── Transition: enumerating → replaying ──
                            // Score all enumerated fingerprints and sort best-first
                            const scored = [];
                            for (const [fp, data] of _relayEnumFingerprints) {
                                const score = _relayScoreFingerprint(fp, null);
                                scored.push({ fp, score });
                            }
                            scored.sort((a, b) => b.score - a.score);
                            _relayScoredQueue = scored;
                            _relayScoredIndex = 0;
                            _relayPhase = 'replaying';
                            _relayEnumTotal += scored.length;
                            _logChoreo(`RELAY: tick ${currentTick} — enumerated ${scored.length} valid options, starting scored replay (best: ${scored[0].score.toFixed(1)})`);
                            const snap = _btSnapshots[_btSnapshots.length - 1];
                            _btRestoreSnapshot(snap);
                            _btRetryCount++;
                            // Seed PRNG for first scored option
                            let fpHash = 0;
                            for (let i = 0; i < scored[0].fp.length; i++) fpHash = ((fpHash << 5) - fpHash + scored[0].fp.charCodeAt(i)) | 0;
                            _sRngSeed(fpHash >>> 0);
                            continue;
                        }
                        // No valid options found — escalate directly
                        _logChoreo(`RELAY: tick ${currentTick} — enumeration found 0 options, escalating`);
                        _relayPhase = 'normal';
                        _relayEnumFingerprints = null;
                        _btStaleRetries = 0;
                        _btResetMatchingCache();
                        if (!_escalateLayer()) break;
                        continue;
                    }
                }

                if (_relayBlocked) {
                    // Relay blocked (Test 2 explicit random) — escalate directly
                    _btStaleRetries = 0;
                    _logChoreo(`BFS: tick ${currentTick} exhausted, escalating`);
                    _btResetMatchingCache();
                    if (!_escalateLayer()) break;
                    continue;
                }
            }
        } else {
            // Got new info — reset stale counter
            if (typeof _btStaleRetries !== 'undefined') _btStaleRetries = 0;

            // During enumeration phase, collect valid fingerprints
            if (_relayPhase === 'enumerating' && _isNewFingerprint) {
                _relayEnumAttempts++;
                _relayEnumStale = 0; // got new info, reset stale
                const fp = _computeTickFingerprint();
                if (!_relayEnumFingerprints.has(fp)) {
                    _relayEnumFingerprints.set(fp, { fp });
                }
            }
            // During normal phase, new info is just progress — keep going
            if (_relayPhase === 'normal' && _isNewFingerprint) {
                _relayEscapes++; // normal choreographer found a path
            }
        }

        // ── SAME-TICK RETRY (new exclusions were added) ──
        if (_bfsFailTick >= 0 && currentTick === _bfsFailTick) {
            _bfsLayerRetries++;
        } else {
            _btRetryCount++;
        }

        const snap = _btSnapshots[_btSnapshots.length - 1];
        _btRestoreSnapshot(snap);
        _updateTickCounter(); // show tick decrement during backtrack
        _logChoreo(`BACKTRACK retry at tick ${currentTick} (ledger: ${ledger.size} exclusions)`);
        continue;
    }

    // ── Clean tick — commit and reset per-tick backtrack state ──
    const cleanTick = _demoTick - 1; // the tick that just succeeded
    // If we just passed the BFS failure tick, the BFS succeeded!
    if (_bfsFailTick >= 0 && cleanTick >= _bfsFailTick) {
        _logChoreo(`BFS: failure tick ${_bfsFailTick} PASSED at layer ${_bfsLayer}! Clearing BFS state.`);
        _bfsReset();
    }
    _btReset();
    _profPhases.guards += _gT1 - _gT0;
    break; // exit retry loop

    } // end backtracking retry loop

    // Increment per-xon mode stats + global running totals + track oct idle duration
    for (const x of _demoXons) {
        if (!x.alive || !x._modeStats) continue;
        const m = x._mode;
        if (m === 'oct' || m === 'oct_formation') {
            x._modeStats.oct++;
            _globalModeStats.oct++;
            _globalRoleStats.oct++;
            if (!x._octModeSince) x._octModeSince = _demoTick;
        } else {
            if (m === 'tet') { x._modeStats.tet++; _globalModeStats.tet++; }
            else if (m === 'idle_tet') { x._modeStats.idle_tet++; _globalModeStats.idle_tet++; }
            else if (m === 'weak') { x._modeStats.weak++; _globalModeStats.weak++; _globalRoleStats.weak++; }
            else if (m === 'gluon') { x._modeStats.gluon++; _globalModeStats.gluon++; _globalRoleStats.gluon++; }
            x._octModeSince = 0;
        }
        // Per-role stats for tet/idle_tet (maps quarkType to specific role)
        if (m === 'tet' || m === 'idle_tet') {
            const role = x._quarkType || 'oct';
            if (_globalRoleStats[role] != null) _globalRoleStats[role]++;
        }
    }

    // Update RL temporal state (for strategic feature extraction)
    if (typeof _rlTemporalState !== 'undefined') {
        // Record face visit ticks (when tet loops complete)
        for (const x of _demoXons) {
            if (x.alive && x._loopStep === 4 && x._tetActualized && x._assignedFace != null) {
                _rlTemporalState.faceLastVisitTick[x._assignedFace] = _demoTick;
            }
        }
        // Every 64 ticks: snapshot per-face CV for ratio velocity calculation
        if (_demoTick % 64 === 0 && _demoVisits) {
            for (let face = 1; face <= 8; face++) {
                const v = _demoVisits[face] || {};
                const types = ['pu1', 'pu2', 'pd', 'nd1', 'nd2', 'nu'];
                const counts = types.map(t => v[t] || 0);
                const s = counts.reduce((a, b) => a + b, 0);
                if (s > 0) {
                    const m = s / 6;
                    let var_ = 0;
                    for (let i = 0; i < 6; i++) var_ += (counts[i] - m) ** 2;
                    _rlTemporalState.prevFaceCV[face] = Math.sqrt(var_ / 6) / m;
                }
            }
        }
        // Global pressure: fraction of faces with CV > 0.5 (below target)
        if (_demoVisits) {
            let belowTarget = 0;
            for (let face = 1; face <= 8; face++) {
                const v = _demoVisits[face] || {};
                const types = ['pu1', 'pu2', 'pd', 'nd1', 'nd2', 'nu'];
                const counts = types.map(t => v[t] || 0);
                const s = counts.reduce((a, b) => a + b, 0);
                if (s === 0) { belowTarget++; continue; }
                const m = s / 6;
                let var_ = 0;
                for (let i = 0; i < 6; i++) var_ += (counts[i] - m) ** 2;
                if (Math.sqrt(var_ / 6) / m > 0.5) belowTarget++;
            }
            _rlTemporalState.globalPressure = belowTarget / 8;
        }
    }

    // Update UI — throttled to ~1/sec for heavy panels, rAF for lightweight updates.
    // This prevents 19ms+ panel rebuilds from eating every frame budget.
    _demoPanelDirty = true;
    // Update all panels every tick
    if (!_testRunning) {
        updateXonPanel();
        _updateEdgeBalancePanel();
        updateDemoPanel();
        _updateEjectionBalancePanel();
        _drawBalanceChart();
        updateStatus();
        if (typeof _updateBottomStats === 'function') _updateBottomStats();
    }

    // Tick log entry (lightweight, for export)
    // Guards use delta encoding: full snapshot on first tick, then only changes
    if (!_testRunning) {
        const currentGuards = Object.fromEntries(
            Object.entries(_liveGuards || {}).map(([k, g]) => [k, g.ok])
        );
        let guardEntry;
        if (_tickLog.length === 0) {
            // First tick: store full guard state
            guardEntry = currentGuards;
            _tickLogLastGuards = currentGuards;
        } else {
            // Delta: only keys that changed
            const delta = {};
            let changed = false;
            for (const [k, v] of Object.entries(currentGuards)) {
                if (_tickLogLastGuards[k] !== v) { delta[k] = v; changed = true; }
            }
            guardEntry = changed ? delta : null; // null = no changes
            _tickLogLastGuards = currentGuards;
        }
        _tickLog.push({
            tick: _demoTick - 1,
            planck: _planckSeconds,
            xons: _demoXons.filter(x => x.alive).map(x => ({
                node: x.node, mode: x._mode,
                quark: x._quarkType || null,
                face: x._assignedFace || null,
                step: x._loopStep
            })),
            activeSCs: [...activeSet],
            xonImplied: [...xonImpliedSet],
            moves: _moveTrace.map(t => ({ xi: t.xonIdx, from: t.from, to: t.to, path: t.path })),
            guards: guardEntry
        });
    }

    // ─── Movie frame recording (lean, for export/playback) ───
    if (!_testRunning) {
        const frame = {
            xons: _demoXons.filter(x => x.alive).map(x => ({
                n: x.node, m: x._mode, q: x._quarkType || null,
                f: x._assignedFace || 0
            })),
            a: [...activeSet],
            xi: [...xonImpliedSet],
            im: [...impliedSet]
        };
        // Delta-compress positions: full on first frame, deltas after
        if (_movieFrames.length === 0 || !_lastMoviePos) {
            frame.pos = Array.from(pos, p => [+p[0].toFixed(4), +p[1].toFixed(4), +p[2].toFixed(4)]);
            _lastMoviePos = pos.map(p => [p[0], p[1], p[2]]);
        } else {
            const deltas = [];
            for (let i = 0; i < pos.length; i++) {
                const dx = pos[i][0] - _lastMoviePos[i][0];
                const dy = pos[i][1] - _lastMoviePos[i][1];
                const dz = pos[i][2] - _lastMoviePos[i][2];
                if (Math.abs(dx) > 1e-4 || Math.abs(dy) > 1e-4 || Math.abs(dz) > 1e-4) {
                    deltas.push([i, +pos[i][0].toFixed(4), +pos[i][1].toFixed(4), +pos[i][2].toFixed(4)]);
                    _lastMoviePos[i][0] = pos[i][0];
                    _lastMoviePos[i][1] = pos[i][1];
                    _lastMoviePos[i][2] = pos[i][2];
                }
            }
            frame.dp = deltas; // dp = delta positions (absent means no change)
        }
        _movieFrames.push(frame);
    }

    // Tournament hook: check if trial has reached its target tick
    if (typeof _tournamentTickCheck === 'function') _tournamentTickCheck();

    // ─── Profiling: record tick time ───
    const _tickDt = performance.now() - _tickT0;
    _tickTotalMs += _tickDt;
    _tickCount++;
    if (_tickDt > _tickMaxMs) _tickMaxMs = _tickDt;

    // Auto-dump every 50 ticks
    if (_tickCount > 0 && _tickCount % 50 === 0) dumpProfile();

    } finally {
        _tickInProgress = false;
    }
}

// ── Precomputed pattern schedule for algos ──

function getOrComputePatternSchedule() {
    if (_activePatternSchedule) return _activePatternSchedule;
    const result = computeActivationPatterns();
    if (!result.patterns.length) return null;
    const patP = result.patterns[0];
    const patN = result.patterns.length > 1 ? result.patterns[1] : result.patterns[0];
    _activePatternSchedule = buildDeuteronSchedule(patP, patN, result.D4);
    return _activePatternSchedule;
}

// ════════════════════════════════════════════════════════════════════
// XON ALGORITHM REGISTRY — PHYSICS-BASED CHOREOGRAPHY STRATEGIES
// ════════════════════════════════════════════════════════════════════
//
// Xons are anonymous excitation workers (like gluons). They don't
// carry quark identity — the quarks ARE the tets. Xons just
// materialize SCs to actualize the target activation pattern.
//
// Each algorithm controls TWO decision points:
//   stepQuark(e, freeOpts, costlyOpts, tetSCsOpen, faceData, ctx)
//     → {dest, scId} or null (choose where xon moves within its tet)
//   shouldHop(e, groupFaces, occupiedFaces, ctx)
//     → {targetFace} or null (decide if/where xon hops between faces)
//
// ctx = { allOpen, quarkList, faceCoverage, nucleusTick, tetFaceData,
//         canMaterialise, materialise, severForRoom, hopGroups }
//
// Tournament swaps algorithms and measures coverage evenness.
// ════════════════════════════════════════════════════════════════════
// ── Shared xon stepQuark: SC materialisation, identity-agnostic ──
function _xonStep(e, freeOpts, costlyOpts, tetSCsOpen, faceData, ctx) {
    let chosen = null;
    if (tetSCsOpen < 2 && costlyOpts.length > 0) {
        for (const opt of costlyOpts) {
            if (ctx.canMaterialise(opt.scId)) {
                if (ctx.materialise(e, opt.scId)) { chosen = opt; break; }
            } else if (ctx.severForRoom(opt.scId)) {
                if (ctx.materialise(e, opt.scId)) { chosen = opt; break; }
            }
        }
    }
    if (!chosen && freeOpts.length > 0) {
        chosen = freeOpts[Math.floor(_sRng() * freeOpts.length)];
    }
    if (!chosen && costlyOpts.length > 0) {
        for (const opt of costlyOpts) {
            if (ctx.canMaterialise(opt.scId)) {
                if (ctx.materialise(e, opt.scId)) { chosen = opt; break; }
            } else if (ctx.severForRoom(opt.scId)) {
                if (ctx.materialise(e, opt.scId)) { chosen = opt; break; }
            }
        }
    }
    return chosen;
}

// Helper: compute per-tet activation state (how many SCs open / total)
function _tetActivationMap(ctx) {
    const map = {};
    for (const [faceId, fd] of Object.entries(ctx.tetFaceData)) {
        const open = fd.scIds.filter(id => ctx.allOpen.has(id)).length;
        const total = fd.scIds.length;
        const covKey = Object.keys(ctx.faceCoverage)
            .filter(k => k.endsWith('_' + faceId));
        let totalCov = 0;
        for (const k of covKey) totalCov += ctx.faceCoverage[k] || 0;
        map[faceId] = { open, total, full: open === total, totalCov };
    }
    return map;
}

// ── Algorithm 4: "xon-least-action" (Lagrangian mechanics) ──
// Nature takes the path of minimum energy. Xons minimize SC
// materializations needed — hop to tets requiring fewest new SCs.
QUARK_ALGO_REGISTRY.push({
    name: 'xon-least-action',
    description: 'Lagrangian: minimize SC materializations per hop (path of least resistance)',
    minDwell: 3,
    timeout: 10,

    stepQuark: _xonStep,

    shouldHop(e, groupFaces, occupiedFaces, ctx) {
        const sif = e._stepsInFace || 0;
        if (sif < this.minDwell) return null;
        const fd = ctx.tetFaceData[e._currentFace];
        const tetFull = fd && fd.scIds.every(id => ctx.allOpen.has(id));
        if (!tetFull && sif < this.timeout) return null;

        const unoccupied = groupFaces.filter(f => !occupiedFaces.has(f));
        if (unoccupied.length === 0) return null;

        // Score each candidate by "action" = closed SCs + coverage (lower = better)
        const actMap = _tetActivationMap(ctx);
        let bestFace = null, bestAction = Infinity;
        for (const f of unoccupied) {
            const a = actMap[f];
            if (!a) continue;
            // Action = closed SCs (materialisation cost) - coverage deficit bonus
            const closedSCs = a.total - a.open;
            const avgCov = Object.values(actMap).reduce((s, x) => s + x.totalCov, 0)
                / Object.keys(actMap).length;
            const deficit = Math.max(0, avgCov - a.totalCov);
            const action = closedSCs - deficit * 0.1; // deficit lowers action cost
            if (action < bestAction) { bestAction = action; bestFace = f; }
        }
        if (bestFace === null) return null;

        // Always hop when ready (deterministic — least action is decisive)
        return { targetFace: bestFace };
    }
});

// ── Algorithm 5: "xon-diffusion" (Fick's law / heat equation) ──
// Coverage diffuses from high-density to low-density regions.
// Hop rate ∝ coverage gradient. Like thermal equilibration.
QUARK_ALGO_REGISTRY.push({
    name: 'xon-diffusion',
    description: "Fick's law: coverage flows from over-served to under-served tets",
    minDwell: 2,
    timeout: 8,

    stepQuark: _xonStep,

    shouldHop(e, groupFaces, occupiedFaces, ctx) {
        const sif = e._stepsInFace || 0;
        if (sif < this.minDwell) return null;
        const fd = ctx.tetFaceData[e._currentFace];
        const tetFull = fd && fd.scIds.every(id => ctx.allOpen.has(id));
        if (!tetFull && sif < this.timeout) return null;

        // Compute coverage "temperature" — current face vs average
        const actMap = _tetActivationMap(ctx);
        const allCovs = Object.values(actMap).map(a => a.totalCov);
        const avgCov = allCovs.reduce((a, b) => a + b, 0) / allCovs.length;
        const curCov = actMap[e._currentFace]?.totalCov || 0;

        // Gradient = how much hotter we are than average
        const gradient = (curCov - avgCov) / Math.max(1, avgCov);

        // Hop probability ∝ gradient (leave hot spots, stay in cold spots)
        const prob = Math.min(0.8, Math.max(0.05, 0.1 + gradient * 0.6));
        if (_sRng() >= prob) return null;

        const unoccupied = groupFaces.filter(f => !occupiedFaces.has(f));
        if (unoccupied.length === 0) return null;

        // Boltzmann-weighted target selection: prefer coldest tet
        const candidates = unoccupied.map(f => ({
            face: f,
            cov: actMap[f]?.totalCov || 0
        }));
        candidates.sort((a, b) => a.cov - b.cov);

        // During backtracking, deterministic: always pick coldest
        if (_btActive) return { targetFace: candidates[0].face };

        // Boltzmann: P(f) ∝ exp(-cov/T), T = temperature parameter
        const T = Math.max(1, avgCov * 0.3);
        const weights = candidates.map(c => Math.exp(-c.cov / T));
        const wTotal = weights.reduce((a, b) => a + b, 0);
        let r = _sRng() * wTotal;
        for (let i = 0; i < candidates.length; i++) {
            r -= weights[i];
            if (r <= 0) return { targetFace: candidates[i].face };
        }
        return { targetFace: candidates[0].face };
    }
});

// ── Algorithm 6: "xon-resonance" (Standing waves / normal modes) ──
// Phase-locked to a deterministic cycle from the pre-computed
// activation patterns. Flexible: if current state drifts too far
// from target, switches to the closest achievable pattern.
QUARK_ALGO_REGISTRY.push({
    name: 'xon-resonance',
    description: 'Standing wave: phase-locked to pre-computed activation pattern cycle',
    minDwell: 2,
    timeout: 6,
    _patternPhase: 0, // which of the 9 derangements we're using

    stepQuark: _xonStep,

    shouldHop(e, groupFaces, occupiedFaces, ctx) {
        const sif = e._stepsInFace || 0;
        if (sif < this.minDwell) return null;
        const fd = ctx.tetFaceData[e._currentFace];
        const tetFull = fd && fd.scIds.every(id => ctx.allOpen.has(id));
        if (!tetFull && sif < this.timeout) return null;

        // Determine target face from the pattern cycle
        // Pattern period = 4 (one full rotation through 4 group faces)
        const cycleIdx = Math.floor(ctx.nucleusTick / 3) % 4; // hop every ~3 ticks
        const targetFace = groupFaces[cycleIdx];

        if (targetFace === e._currentFace) return null;
        if (occupiedFaces.has(targetFace)) {
            // Target occupied — find next available in cycle
            for (let offset = 1; offset < groupFaces.length; offset++) {
                const alt = groupFaces[(cycleIdx + offset) % groupFaces.length];
                if (!occupiedFaces.has(alt) && alt !== e._currentFace) {
                    return { targetFace: alt };
                }
            }
            return null;
        }
        return { targetFace };
    }
});

// ── Algorithm 7: "xon-cooperative" (Many-body / entanglement) ──
// Xons coordinate: each checks what others are doing and avoids
// duplication. Collectively maximizes coverage spread.
QUARK_ALGO_REGISTRY.push({
    name: 'xon-cooperative',
    description: 'Many-body coordination: xons communicate to avoid duplication and maximize spread',
    minDwell: 3,
    timeout: 10,

    stepQuark: _xonStep,

    shouldHop(e, groupFaces, occupiedFaces, ctx) {
        const sif = e._stepsInFace || 0;
        if (sif < this.minDwell) return null;
        const fd = ctx.tetFaceData[e._currentFace];
        const tetFull = fd && fd.scIds.every(id => ctx.allOpen.has(id));
        if (!tetFull && sif < this.timeout) return null;

        // Count xons per face in this group
        const facePop = {};
        for (const f of groupFaces) facePop[f] = 0;
        for (const q of ctx.quarkList) {
            if (q._hopGroup === e._hopGroup) facePop[q._currentFace]++;
        }

        // Only hop if current face is "over-populated" or has excess coverage
        const actMap = _tetActivationMap(ctx);
        const myPop = facePop[e._currentFace] || 0;
        if (myPop <= 1) {
            // I'm the only one here — only leave if coverage is excessive
            const curCov = actMap[e._currentFace]?.totalCov || 0;
            const avgCov = Object.values(actMap).reduce((s, a) => s + a.totalCov, 0)
                / Object.keys(actMap).length;
            if (curCov <= avgCov * 1.2) return null;
        }

        const unoccupied = groupFaces.filter(f => !occupiedFaces.has(f));
        if (unoccupied.length === 0) return null;

        // Target: face with fewest xons AND lowest coverage
        let bestFace = unoccupied[0], bestScore = Infinity;
        for (const f of unoccupied) {
            const pop = facePop[f] || 0;
            const cov = actMap[f]?.totalCov || 0;
            const score = pop * 100 + cov; // population dominates
            if (score < bestScore) { bestScore = score; bestFace = f; }
        }
        return { targetFace: bestFace };
    }
});

// ── Algorithm 8: "xon-flux-tube" (QCD string/confinement) ──
// Xons prefer to maintain connected "flux tubes" of open SCs.
// They hop along the bosonic cage's edges, extending activation
// along the octahedral adjacency graph.
QUARK_ALGO_REGISTRY.push({
    name: 'xon-flux-tube',
    description: 'QCD confinement: xons extend flux tubes along connected SC chains',
    minDwell: 3,
    timeout: 8,

    stepQuark(e, freeOpts, costlyOpts, tetSCsOpen, faceData, ctx) {
        // Flux tube preference: free edges that connect to other active tets
        if (freeOpts.length > 0) {
            // Prefer edges toward oct nodes shared with other actualized tets
            const actMap = _tetActivationMap(ctx);
            const scored = freeOpts.map(opt => {
                let connectivity = 0;
                // Check if dest node is shared with another active tet
                for (const [fId, fd] of Object.entries(ctx.tetFaceData)) {
                    if (String(fId) === String(e._currentFace)) continue;
                    if (actMap[fId]?.full && fd.allNodes.includes(opt.dest)) {
                        connectivity++;
                    }
                }
                return { ...opt, connectivity };
            });
            scored.sort((a, b) => b.connectivity - a.connectivity);
            if (scored[0].connectivity > 0) return scored[0];
        }
        // Fall back to standard xon step
        return _xonStep(e, freeOpts, costlyOpts, tetSCsOpen, faceData, ctx);
    },

    shouldHop(e, groupFaces, occupiedFaces, ctx) {
        const sif = e._stepsInFace || 0;
        if (sif < this.minDwell) return null;
        const fd = ctx.tetFaceData[e._currentFace];
        const tetFull = fd && fd.scIds.every(id => ctx.allOpen.has(id));
        if (!tetFull && sif < this.timeout) return null;

        const unoccupied = groupFaces.filter(f => !occupiedFaces.has(f));
        if (unoccupied.length === 0) return null;

        // Prefer faces adjacent on the octahedron (sharing an oct node)
        const curDef = DEUTERON_TET_FACES[e._currentFace];
        if (!curDef) return { targetFace: unoccupied[0] };

        const adjacent = unoccupied.filter(f => {
            const tgtDef = DEUTERON_TET_FACES[f];
            if (!tgtDef) return false;
            return curDef.octNodes.some(n => tgtDef.octNodes.includes(n));
        });

        const actMap = _tetActivationMap(ctx);
        const candidates = (adjacent.length > 0 ? adjacent : unoccupied);

        // Among candidates, prefer least-covered
        let bestFace = candidates[0], minCov = Infinity;
        for (const f of candidates) {
            const cov = actMap[f]?.totalCov || 0;
            if (cov < minCov) { minCov = cov; bestFace = f; }
        }

        if (_sRng() >= 0.35) return null;
        return { targetFace: bestFace };
    }
});
