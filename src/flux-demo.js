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
//   T21  Oct cage permanence Oct SCs never leave activeSet.
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
function _localScNeighbors(node) {
    const scs = scByVert[node] || [];
    const pa = pos[node];
    if (!pa) return scs; // pos not ready yet (pre-lattice)
    return scs.filter(sc => {
        const other = sc.a === node ? sc.b : sc.a;
        const pb = pos[other];
        if (!pb) return false;
        const dx = pb[0]-pa[0], dy = pb[1]-pa[1], dz = pb[2]-pa[2];
        return Math.abs(Math.sqrt(dx*dx + dy*dy + dz*dz) - 1) <= 0.50;
    });
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

    // Count how many oct-mode xons are currently on the cage
    let octOnCage = 0;
    for (const x of _demoXons) {
        if (x.alive && x._mode === 'oct' && _octNodeSet.has(x.node)) octOnCage++;
    }

    // If plenty of oct xons remain (3+), removing one won't break the cage
    if (octOnCage >= 3) return false;

    // If only 1-2 oct xons on cage, check if all cage SCs are stable.
    // If any cage SC is not in any active set, the cage needs this xon.
    for (const scId of _octSCIds) {
        if (!activeSet.has(scId) && !xonImpliedSet.has(scId) && !impliedSet.has(scId)) {
            // A cage SC is not in any active set — cage is already broken!
            // This xon is needed to help repair it.
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


// Animate all demo xons — called every frame from the render loop.
// Handles tween interpolation, spark flash, trail rendering, and trail decay.
function _tickDemoXons(dt) {
    const sparkOp = (+document.getElementById('spark-opacity-slider').value) / 100;
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
                const segCol = (xon._frozenColors && xon._frozenColors[i]) || xon.col;
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

        // ── Live xons: tween + spark + trail ──
        // Tween interpolation (cubic ease-out)
        xon.tweenT = Math.min(1, xon.tweenT + dt / demoStepSec);
        const s = 1 - (1 - xon.tweenT) ** 3;
        const pf = pos[xon.prevNode], pt = pos[xon.node];
        if (pf && pt) {
            // Distance check: if prevNode→node exceeds valid hop distance,
            // hold at source (don't flash sprite at non-adjacent target)
            const _tdx = pt[0] - pf[0], _tdy = pt[1] - pf[1], _tdz = pt[2] - pf[2];
            const _hopDist = Math.sqrt(_tdx*_tdx + _tdy*_tdy + _tdz*_tdz);
            if (_hopDist > 1.2) {
                xon.group.position.set(pf[0], pf[1], pf[2]);
            } else {
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
        const pulse = (0.22 + xon.flashT * 0.26) * flicker * hlBoost;
        xon.spark.scale.set(pulse, pulse, 1);
        xon.sparkMat.opacity = Math.min(1.0, (0.6 + xon.flashT * 0.4) * flicker * sparkOp * hlBoost);
        // Decay highlight timer
        if (xon._highlightT > 0) xon._highlightT = Math.max(0, xon._highlightT - dt);
        xon.sparkMat.rotation = Math.random() * Math.PI * 2;

        // Trail: fading vertex-colored path
        // Lifespan knob controls how many trail points are visible (0-50).
        // Always store full 50-tick history; render only the last `visLen` points.
        const lifespan = +document.getElementById('tracer-lifespan-slider').value;
        const fullLen = xon.trail.length;
        const visLen = Math.min(fullLen, lifespan);
        const startIdx = fullLen - visLen; // skip older points beyond lifespan

        // During tween (tweenT < 1), the latest trail entry is the DESTINATION
        // which the sprite hasn't reached yet. Rendering it in the body creates
        // a backward line from destination back to sprite. Fix: exclude the
        // latest entry during tween and let the trail head animate the hop.
        const bodyLen = (xon.tweenT < 1 && visLen > 1) ? visLen - 1 : visLen;

        // Per-segment color from trailColHistory — segments retain their original color
        // flashT boosts trail brightness near the head (mode transition / birth flash)
        xon._lastTrailFlashBoost = 0; // reset per frame for T37 measurement
        for (let vi = 0; vi < bodyLen; vi++) {
            const i = startIdx + vi;
            // Use frozen positions (recorded at trail push time) so trails don't deform with solver
            const np = (xon._trailFrozenPos && xon._trailFrozenPos[i]) || pos[xon.trail[i]];
            if (!np) continue;
            // Teleport suppression: if this segment jumps > 1.5 from previous point,
            // collapse to previous point (zero-length segment hides the teleport line)
            if (vi > 0) {
                const _spx = xon.trailPos[(vi-1) * 3], _spy = xon.trailPos[(vi-1) * 3 + 1], _spz = xon.trailPos[(vi-1) * 3 + 2];
                const _sdx = np[0] - _spx, _sdy = np[1] - _spy, _sdz = np[2] - _spz;
                if (_sdx*_sdx + _sdy*_sdy + _sdz*_sdz > 1.44) { // 1.2^2
                    xon.trailPos[vi * 3] = _spx;
                    xon.trailPos[vi * 3 + 1] = _spy;
                    xon.trailPos[vi * 3 + 2] = _spz;
                    // Zero alpha to fully hide collapsed point
                    xon.trailCol[vi * 3] = 0;
                    xon.trailCol[vi * 3 + 1] = 0;
                    xon.trailCol[vi * 3 + 2] = 0;
                    continue;
                }
            }
            xon.trailPos[vi * 3] = np[0];
            xon.trailPos[vi * 3 + 1] = np[1];
            xon.trailPos[vi * 3 + 2] = np[2];
            const segCol = (xon.trailColHistory && xon.trailColHistory[i]) || xon.col;
            const cr = ((segCol >> 16) & 0xff) / 255;
            const cg = ((segCol >> 8) & 0xff) / 255;
            const cb = (segCol & 0xff) / 255;
            const baseAlpha = 0.5 + 0.5 * (vi / Math.max(bodyLen, 1)) ** 0.8;
            // Flash boost: head segments get up to 40% brighter during flash
            const headProximity = vi / Math.max(bodyLen - 1, 1); // 0=tail, 1=head
            const flashBoost = xon.flashT * 0.4 * headProximity;
            xon._lastTrailFlashBoost = Math.max(xon._lastTrailFlashBoost || 0, flashBoost);
            const alpha = sparkOp * Math.min(1, baseAlpha + flashBoost);
            xon.trailCol[vi * 3] = cr * alpha;
            xon.trailCol[vi * 3 + 1] = cg * alpha;
            xon.trailCol[vi * 3 + 2] = cb * alpha;
        }
        // Current interpolated position as trail head — extends from last BODY
        // entry toward sprite. During tween this smoothly animates the hop.
        const last = bodyLen;
        let _drawHead = false;
        if (last < XON_TRAIL_LENGTH && bodyLen > 0) {
            // Distance from last body point to current group position
            const _lfi = startIdx + bodyLen - 1;
            const _lfp = (xon._trailFrozenPos && xon._trailFrozenPos[_lfi]) || pos[xon.trail[startIdx + bodyLen - 1]];
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
                    const hcr = ((headCol >> 16) & 0xff) / 255;
                    const hcg = ((headCol >> 8) & 0xff) / 255;
                    const hcb = (headCol & 0xff) / 255;
                    xon.trailCol[last * 3] = hcr * sparkOp;
                    xon.trailCol[last * 3 + 1] = hcg * sparkOp;
                    xon.trailCol[last * 3 + 2] = hcb * sparkOp;
                }
            }
        }
        const n = _drawHead ? bodyLen + 1 : bodyLen;
        xon.trailGeo.setDrawRange(0, Math.min(n, XON_TRAIL_LENGTH));
        xon.trailGeo.attributes.position.needsUpdate = true;
        xon.trailGeo.attributes.color.needsUpdate = true;
    }
}

// Emit a gluon between two tet faces along oct edges
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
            if (scId !== undefined && !activeSet.has(scId)) {
                if (canMaterialiseQuick(scId)) {
                    activeSet.add(scId);
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
                [A[ancA_p[i]]]: 'pd',   // proton anchor (ham CW)
                [A[i]]: 'pu1',           // proton follower-1 (fork)
                [A[dA_p[i]]]: 'pu2',     // proton follower-2 (hook)
                [B[ancB_n[i]]]: 'nu',    // neutron anchor (ham CCW)
                [B[i]]: 'nd1',           // neutron follower-1 (fork)
                [B[dB_n[i]]]: 'nd2',     // neutron follower-2 (hook)
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
    for (let f = 1; f <= 8; f++) {
        _demoVisits[f] = { pu1: 0, pu2: 0, pd: 0, nd1: 0, nd2: 0, nu: 0, total: 0 };
    }
    _demoTick = 0;
    _bfsReset(); // fresh demo = clean BFS + ledger
    _btSnapshots.length = 0;
    _demoTetAssignments = 0;
    _demoPauliViolations = 0;
    _demoSpreadViolations = 0;
    _demoTypeBalanceHistory = [];
    _demoVisitedFaces = new Set();  // track which faces have been activated
    _demoOctRevealed = false;       // oct only renders once all 8 faces visited
    // Clean up any existing xon visuals before reinit
    for (const xon of _demoXons) {
        if (xon.group) { scene.remove(xon.group); }
        if (xon.sparkMat) xon.sparkMat.dispose();
        if (xon.trailLine) scene.remove(xon.trailLine);
        if (xon.trailGeo) xon.trailGeo.dispose();
    }
    _demoXons = [];
    _demoGluons = [];               // Demo 3.1: clear gluon pool
    _demoPrevFaces = new Set();     // Demo 3.1: no previous window faces
    _demoActive = true;
    _demoPaused = false;

    // Stop excitation clock (we drive our own loop)
    if (typeof stopExcitationClock === 'function') stopExcitationClock();

    // Do NOT pre-open all 8 tet SCs — only 1-3 tets can coexist at a time.
    // Tets activate/deactivate per window via xonImpliedSet, and the
    // solver re-runs each time so spheres physically respond to geometry.
    // Oct emerges visually once all 8 faces have been visited.
    bumpState();
    const pSolved = detectImplied();
    applyPositions(pSolved);
    updateVoidSpheres();

    // Hide xon sparks/trails
    const quarks = NucleusSimulator?.quarkExcitations || [];
    for (const q of quarks) {
        if (q.spark) q.spark.visible = false;
        if (q.trailLine) q.trailLine.visible = false;
    }

    // Show demo status + L2/L3 toggle
    const ds = document.getElementById('demo-status');
    if (ds) {
        ds.style.display = 'block';
        // Add L2/L3 toggle if not already present
        if (!document.getElementById('demo-lattice-toggle')) {
            const toggleDiv = document.createElement('div');
            toggleDiv.id = 'demo-lattice-toggle';
            toggleDiv.style.cssText = 'margin-top:4px; text-align:center;';
            toggleDiv.innerHTML = `<span style="font-size:8px; color:#667788; margin-right:4px;">lattice:</span>`
                + `<button id="demo-l2-btn" style="font-size:8px; padding:1px 6px; margin:0 2px; background:#1a2a3a; color:#88bbdd; border:1px solid #3a5a7a; border-radius:3px; cursor:pointer;">L2</button>`
                + `<button id="demo-l3-btn" style="font-size:8px; padding:1px 6px; margin:0 2px; background:#0a1a2a; color:#556677; border:1px solid #2a3a4a; border-radius:3px; cursor:pointer;">L3</button>`
                + `<button id="demo-l4-btn" style="font-size:8px; padding:1px 6px; margin:0 2px; background:#0a1a2a; color:#556677; border:1px solid #2a3a4a; border-radius:3px; cursor:pointer;">L4</button>`;
            ds.parentNode.insertBefore(toggleDiv, ds.nextSibling);
            document.getElementById('demo-l2-btn').addEventListener('click', () => _setDemoLattice(2));
            document.getElementById('demo-l3-btn').addEventListener('click', () => _setDemoLattice(3));
            document.getElementById('demo-l4-btn').addEventListener('click', () => _setDemoLattice(4));
        }
        _updateDemoLatticeButtons();
    }

    // Update left panel header
    const dpTitle = document.getElementById('dp-title');
    if (dpTitle) dpTitle.textContent = '0 Planck seconds';

    // Demo 3.0 visual setup: opacity defaults
    const spheresSlider = document.getElementById('sphere-opacity-slider');
    if (spheresSlider) { spheresSlider.value = 3; spheresSlider.dispatchEvent(new Event('input')); }
    const shapesSlider = document.getElementById('void-opacity-slider');
    if (shapesSlider) { shapesSlider.value = 5; shapesSlider.dispatchEvent(new Event('input')); }
    const graphSlider = document.getElementById('graph-opacity-slider');
    if (graphSlider) { graphSlider.value = 21; graphSlider.dispatchEvent(new Event('input')); }
    const trailSlider = document.getElementById('trail-opacity-slider');
    if (trailSlider) { trailSlider.value = 55; trailSlider.dispatchEvent(new Event('input')); }

    // Center camera on bosonic cage (oct node centroid) at eye level
    if (_octNodeSet && _octNodeSet.size > 0 && pos) {
        let cx = 0, cy = 0, cz = 0, count = 0;
        for (const n of _octNodeSet) {
            if (pos[n]) { cx += pos[n][0]; cy += pos[n][1]; cz += pos[n][2]; count++; }
        }
        if (count > 0) {
            panTarget.x = cx / count;
            panTarget.y = cy / count;
            panTarget.z = cz / count;
        }
    }
    applyCamera();

    // Default to maximum speed (uncapped)
    const speedSlider = document.getElementById('excitation-speed-slider');
    if (speedSlider) { speedSlider.value = 100; speedSlider.dispatchEvent(new Event('input')); }
    // Default lifespan: visible trail length (how many of 50 stored ticks to show)
    const lifespanSlider = document.getElementById('tracer-lifespan-slider');
    if (lifespanSlider) { lifespanSlider.value = 50; lifespanSlider.dispatchEvent(new Event('input')); }
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

    // Auto-run unit tests — HALT DEMO if any test fails (tournament: run but don't halt)
    try {
        const testResult = runDemo3Tests();
        if (!_tournamentRunning && testResult.failed.length > 0) {
            console.error(`[demo] HALTED: ${testResult.failed.length} test(s) failed: ${testResult.failed.join(', ')}`);
            stopDemo();
            return;
        }
    } catch (e) { console.warn('[demo] Test suite error:', e); }

    // Activate live guards (T19, T21, T26, T27) — start with null during grace
    if (typeof _liveGuards !== 'undefined') {
        for (const entry of LIVE_GUARD_REGISTRY) {
            const g = _liveGuards[entry.id];
            if (!g) continue;
            g.ok = null;
            g.msg = 'grace period';
            g.failed = false;
            // Re-apply init fields so state is clean across demo restarts
            if (entry.init) Object.assign(g, entry.init);
        }
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
        // Remaining 2 xons: free to move, become weak when they step off oct
        for (let i = 4; i < 6; i++) {
            const xon = _demoXons[i];
            if (xon._mode === 'oct_formation') {
                xon._mode = 'oct';
                xon._pendingWeakEjection = true;
            }
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

    // ── BACKTRACKING RETRY LOOP ──
    // Save state before tick, run choreography, check guards.
    // If T19/T20 violation → rewind, exclude offending moves, retry.
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
    // Yield to event loop every 32 retries to prevent browser freeze
    if (_btAttempt > 0 && _btAttempt % 32 === 0) await new Promise(r => setTimeout(r, 0));

    // Seed PRNG from tick number + retry context. The retry count and BFS layer
    // ensure each backtracker attempt gets a different PRNG sequence, so type
    // selection, shuffles, and other randomized decisions explore new paths.
    _sRngSeed(_demoTick * 65537 + _btRetryCount * 997 + _bfsLayer * 31);
    // Clear stale movement flags from previous tick so WB processing isn't blocked
    for (const xon of _demoXons) { xon._movedThisTick = false; xon._evictedThisTick = false; }
    // Revert gluon xons to oct (fresh evaluation each tick per spec §2)
    for (const xon of _demoXons) {
        if (xon.alive && xon._mode === 'gluon') {
            xon._mode = 'oct';
            xon.col = 0xffffff;
            if (xon.sparkMat) xon.sparkMat.color.setHex(0xffffff);
        }
    }
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
            _solverNeeded = true;
            occupied = _occupiedNodes(); // refresh after opening moves
        } else {
            _openingPhase = false;
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
                    // Recolor trail to purple
                    for (let ti = 0; ti < xon.trailColHistory.length; ti++) {
                        xon.trailColHistory[ti] = WEAK_FORCE_COLOR;
                    }
                    // Relinquish face SCs
                    const locked60 = _traversalLockedSCs();
                    if (fd60) {
                        for (const scId of fd60.scIds) {
                            if (locked60.has(scId)) continue;
                            if (xonImpliedSet.delete(scId)) {
                                _scAttribution.delete(scId);
                                _solverNeeded = true;
                                stateVersion++;
                            }
                        }
                    }
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
                    phase0Changed = true;
                    continue; // skip normal eviction checks; PHASE 0.5 moves it
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
                    // Evict the BLOCKER (idle_tet is expendable)
                    const blockerFd = blocker._assignedFace != null ? _nucleusTetFaceData[blocker._assignedFace] : null;
                    if (blockerFd) {
                        const locked0 = _traversalLockedSCs();
                        for (const scId of blockerFd.scIds) {
                            if (locked0.has(scId)) continue;
                            if (xonImpliedSet.delete(scId)) {
                                _scAttribution.delete(scId);
                                _solverNeeded = true;
                                stateVersion++;
                            }
                        }
                    }
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
                const evictFd = xon._assignedFace != null ? _nucleusTetFaceData[xon._assignedFace] : null;
                if (evictFd) {
                    const locked0 = _traversalLockedSCs();
                    for (const scId of evictFd.scIds) {
                        if (locked0.has(scId)) continue;
                        if (xonImpliedSet.delete(scId)) {
                            _scAttribution.delete(scId);
                            _solverNeeded = true;
                            stateVersion++;
                        }
                    }
                }
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
        // Sort: prefer (1) non-trail nodes, (2) non-prevNode, (3) anything
        bestSteps.sort((a, b) => {
            const aInTrail = recentTrail.has(a) ? 1 : 0;
            const bInTrail = recentTrail.has(b) ? 1 : 0;
            if (aInTrail !== bInTrail) return aInTrail - bInTrail;
            const aIsPrev = a === xon.prevNode ? 1 : 0;
            const bIsPrev = b === xon.prevNode ? 1 : 0;
            return aIsPrev - bIsPrev;
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
            _trailPush(xon, bestStep, WEAK_FORCE_COLOR);
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
            const allNbs = baseNeighbors[xon.node] || [];
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
                _trailPush(xon, freeNb.node, WEAK_FORCE_COLOR);
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
                // Recolor trail to purple — no colored trail without an actualized tet
                for (let ti = 0; ti < xon.trailColHistory.length; ti++) {
                    xon.trailColHistory[ti] = WEAK_FORCE_COLOR;
                }
                // Relinquish face SCs (before mode change)
                if (xon._assignedFace != null) {
                    const fd = _nucleusTetFaceData[xon._assignedFace];
                    if (fd) {
                        const locked60 = _traversalLockedSCs();
                        for (const scId of fd.scIds) {
                            if (locked60.has(scId)) continue;
                            if (xonImpliedSet.delete(scId)) {
                                _scAttribution.delete(scId);
                                _solverNeeded = true;
                                stateVersion++;
                            }
                        }
                    }
                }
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
            // Relinquish face SCs that are no longer needed (respects traversal lock)
            if (xon._assignedFace != null) {
                const fd = _nucleusTetFaceData[xon._assignedFace];
                if (fd) {
                    for (const scId of fd.scIds) {
                        if (locked15.has(scId)) continue;
                        if (xonImpliedSet.delete(scId)) {
                            _scAttribution.delete(scId);
                            _solverNeeded = true;
                            stateVersion++;
                        }
                    }
                }
            }
        }
        occupied = _occupiedNodes(); // refresh after returns
    }

    // ── PHASE 2a: Demand-driven face selection (decentralized, no order precedence) ──
    // Each oct xon scores ALL reachable faces independently. Conflicts resolved by
    // random shuffling — no xon gets order-precedence over another.
    {
        _ratioTracker.sync();
        let octIdle = _demoXons.filter(x => x.alive && x._mode === 'oct' && !x._movedThisTick && !x._evictedThisTick);

        // ── GLUON CHECK: cage integrity takes priority over face assignment ──
        // (see spec §2: gluon mode, §4: PHASE 2a)
        for (const xon of octIdle) {
            if (_cageWouldBreak(xon)) {
                _clearModeProps(xon);
                xon._mode = 'gluon';
                xon.col = GLUON_COLOR;
                if (xon.sparkMat) xon.sparkMat.color.setHex(GLUON_COLOR);
                _logChoreo(`GLUON: X${_demoXons.indexOf(xon)} designated as gluon (cage integrity)`);
            }
        }
        // Remove gluon xons from face assignment candidates
        octIdle = octIdle.filter(x => x._mode !== 'gluon');

        if (octIdle.length > 0 && _nucleusTetFaceData) {
            // Each xon independently scores all faces
            const proposals = []; // {xon, face, quarkType, score, onFace}
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
            const assignedXons = new Set();
            const assignedFaces = new Set();
            for (const prop of proposals) {
                if (assignedXons.has(prop.xon)) continue;
                if (assignedFaces.has(prop.face)) continue;

                // Skip low-scoring proposals (vacancy penalty already handles crowding)
                if (prop.score < _choreoParams.assignmentThreshold) continue;

                // Vacuum feasibility: can we materialize the face SCs?
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

                // Lookahead viability: can the loop complete?
                const seq = LOOP_SEQUENCES[prop.quarkType](fd.cycle);
                const tmpOcc = new Map(occupied);
                if (!_lookaheadTetPath(seq, 0, tmpOcc, _choreoParams.lookahead, prop.xon)) continue;

                // ASSIGN — decentralized decision accepted by the system
                _assignXonToTet(prop.xon, prop.face, prop.quarkType);
                _demoTetAssignments++;
                assignedXons.add(prop.xon);
                assignedFaces.add(prop.face);
                _demoVisitedFaces.add(prop.face);
                _solverNeeded = true;
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
                _logChoreo(`X${_demoXons.indexOf(xon)} oct overflow -> pendingWeak`);
                xon._pendingWeakEjection = true;
                excess--;
            }
            // Priority 2: idle_tet xons on oct nodes (interrupt loop, mark for ejection)
            if (excess > 0) {
                const idleCandidates = _sRngShuffle(_demoXons.filter(x =>
                    x.alive && x._mode === 'idle_tet' && !x._movedThisTick &&
                    _octNodeSet.has(x.node)
                ));
                for (const xon of idleCandidates) {
                    if (excess <= 0) break;
                    const xi = _demoXons.indexOf(xon);
                    _logChoreo(`X${xi} idle_tet on oct -> pendingWeak (T79 shed)`);
                    xon._assignedFace = null;
                    xon._loopSeq = null;
                    xon._loopStep = 0;
                    xon._mode = 'oct';
                    xon._pendingWeakEjection = true;
                    excess--;
                }
            }
            // Priority 3: weak xons on oct nodes (force off-oct movement)
            if (excess > 0) {
                const weakOnOct = _sRngShuffle(_demoXons.filter(x =>
                    x.alive && x._mode === 'weak' && !x._movedThisTick &&
                    _octNodeSet.has(x.node)
                ));
                for (const xon of weakOnOct) {
                    if (excess <= 0) break;
                    const xi = _demoXons.indexOf(xon);
                    _logChoreo(`X${xi} weak on oct -> pendingWeak (T79 shed)`);
                    xon._mode = 'oct';
                    xon._pendingWeakEjection = true;
                    excess--;
                }
            }
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

    // ── BACKTRACK EXCLUSION FILTER: remove moves that caused violations on previous attempts ──
    if (_btActive) {
        for (const plan of octPlans) {
            const xonIdx = _demoXons.indexOf(plan.xon);
            plan.candidates = plan.candidates.filter(c => !_btIsMoveExcluded(xonIdx, c.node));
        }
    }

    // T55 enforcement: Removed — T55 live guard fires into backtracker

    // ── BFS BACKTRACK: systematic candidate rotation ──
    // During retries, rotate each xon's candidate list so Kuhn's algorithm
    // produces genuinely different matchings on each attempt.
    // The effective seed combines retryCount + BFS layer * MAX_RETRIES so that
    // layer escalation produces different rotations even when retryCount resets to 0.
    if (_btActive) {
        const effectiveSeed = _btRetryCount + _bfsLayer * _BT_MAX_RETRIES + _bfsLayerRetries;
        if (effectiveSeed > 0) {
            for (let i = 0; i < octPlans.length; i++) {
                const cands = octPlans[i].candidates;
                if (cands.length <= 1) continue;
                // Each xon gets a different rotation based on seed.
                // Stagger per xon so we explore the Cartesian product.
                const shift = Math.floor(effectiveSeed / Math.max(1, i + 1)) % cands.length;
                if (shift > 0) {
                    octPlans[i].candidates = [...cands.slice(shift), ...cands.slice(0, shift)];
                }
            }
        }
    }

    // Move assignment: either Kuhn's augmenting-path matching or greedy first-fit.
    if (_kuhnEnabled) {
        // Maximum bipartite matching with arbitrary-depth backtracking (Kuhn's algorithm).
        // Finds augmenting paths so the maximum number of oct xons get a valid destination.
        _maxBipartiteAssignment(octPlans, planned);
    } else {
        // Greedy first-fit: assign each xon its best available candidate.
        // Simpler than Kuhn's — no swap-removal, so no Pauli gap.
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
                // (removed: _mayReturn flip — no longer used)
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
            _trailPush(plan.xon, nb.node, WEAK_FORCE_COLOR);
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

    // ── Run solver if any SCs changed (unified architecture) ──
    if (_solverNeeded) {
        bumpState();
        const scPairs = [];
        activeSet.forEach(id => { const s = SC_BY_ID[id]; scPairs.push([s.a, s.b]); });
        xonImpliedSet.forEach(id => { const s = SC_BY_ID[id]; scPairs.push([s.a, s.b]); });
        const { p: pSolved } = _solve(scPairs, 5000, true); // noBailout: full convergence for Kepler
        impliedSet.clear(); impliedBy.clear();
        xonImpliedSet.forEach(id => {
            if (!activeSet.has(id)) { impliedSet.add(id); impliedBy.set(id, new Set()); }
        });
        // Bump state again AFTER solving so applyPositions → updateVoidSpheres
        // re-evaluates geometric checks with the deformed (solved) positions.
        // bumpState() above calls updateVoidSpheres() with pre-solver positions,
        // caching stale actualization; this second bump invalidates that cache.
        stateVersion++;
        applyPositions(pSolved);
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
            // Soft recovery: try clearing electron-implied SCs
            if (xonImpliedSet.size && !simHalted) {
                for (const id of [...xonImpliedSet]) {
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
        for (const id of toRemove) {
            xonImpliedSet.delete(id);
            _scAttribution.delete(id);
            stateVersion++;
        }
    }

    const _pTsolver = performance.now(); _profPhases.solver += _pTsolver - _pT5;

    // ── Decay dying xon trails (every simulation tick, not per-frame) ──
    _decayDyingXons();

    // ── Color tets with progressive opacity (ramps as xon loop completes) ──
    // Demand-driven: derive active faces from xon state, not schedule.
    if (_nucleusTetFaceData) {
        // Build active face map from xon assignments
        const activeFaces = new Map(); // face → {quarkType, loopStep, actualized}
        for (const xon of _demoXons) {
            if (!xon.alive || xon._assignedFace == null) continue;
            if (xon._mode === 'tet' || xon._mode === 'idle_tet') {
                activeFaces.set(xon._assignedFace, {
                    quarkType: xon._quarkType, loopStep: xon._loopStep,
                    actualized: !!xon._tetActualized
                });
            }
        }
        for (const [fIdStr, fd] of Object.entries(_nucleusTetFaceData)) {
            const fId = parseInt(fIdStr);
            const active = activeFaces.get(fId);
            // T58: only color tet faces that have COMPLETED a loop and counted
            // in the hadronic balance. loopStep === 4 is the completion tick
            // (same gate as _demoVisits increment in _advanceXon).
            // SCs must also be active right now to confirm genuine actualization.
            const completedNow = active && active.actualized && active.loopStep >= 4
                && fd.scIds.every(scId =>
                    activeSet.has(scId) || impliedSet.has(scId) || xonImpliedSet.has(scId));
            if (completedNow) {
                _ruleAnnotations.tetColors.set(fd.voidIdx, QUARK_COLORS[active.quarkType]);
                _ruleAnnotations.tetOpacity.set(fd.voidIdx, 0.85);
            } else {
                _ruleAnnotations.tetColors.set(fd.voidIdx, 0x1a1a2a);
                _ruleAnnotations.tetOpacity.set(fd.voidIdx, 0.0);
            }
        }
        _ruleAnnotations.dirty = true;
        if (typeof updateVoidSpheres === 'function') updateVoidSpheres();
    }

    const _pTrender = performance.now(); _profPhases.render += _pTrender - _pTsolver;

    // T79: track consecutive full-oct ticks (for next tick's overflow pressure)
    if (_octNodeSet && _demoXons.filter(x => x.alive && _octNodeSet.has(x.node)).length >= 6) {
        _octFullConsecutive++;
    } else {
        _octFullConsecutive = 0;
    }

    _demoTick++;

    // Update Planck-second ticker (both right-panel status and left-panel title)
    const _tickerEl = document.getElementById('nucleus-status');
    if (_tickerEl) _tickerEl.textContent = `${_demoTick} Planck seconds`;
    const _dpTitle = document.getElementById('dp-title');
    if (_dpTitle) _dpTitle.textContent = `${_demoTick} Planck seconds`;
    // Top-center title is set once per trial by _runTournament — no per-tick update needed

    // Live guard checks (T19, T21, T26, T27) — after tick advances xons
    const _gT0 = performance.now();
    if (typeof _liveGuardCheck === 'function') _liveGuardCheck();
    const _gT1 = performance.now();

    // ── BACKTRACK CHECK (BFS): did guards request a rewind? ──
    if (_rewindRequested) {
        _rewindRequested = false;
        _btActive = true;

        // Extract exclusions and accumulate in persistent ledger.
        // Track whether any GENUINELY NEW exclusions were added — if not, this
        // layer is exhausted and we must escalate to a deeper BFS layer.
        const newExclusions = _btExtractExclusions();
        const currentTick = _demoTick - 1; // tick was already incremented
        if (!_btBadMoveLedger.has(currentTick)) _btBadMoveLedger.set(currentTick, new Set());
        const ledger = _btBadMoveLedger.get(currentTick);
        let _addedNewExclusions = false;
        for (const ex of newExclusions) {
            if (!ledger.has(ex)) _addedNewExclusions = true;
            ledger.add(ex);
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

            // ── t=0 CANARY: If we've backed all the way past tick 0, the rules
            // are mathematically impossible. This is the ONLY true failure. ──
            if (targetTick < 0) {
                console.error(`%c[CANARY] Rules are mathematically impossible — backtracker exhausted all possibilities to t=0. Last violation: ${_rewindViolation}`, 'color:red;font-weight:bold;font-size:14px');
                simHalted = true;
                _btReset();
                _bfsReset();
                return false;
            }

            const anchorSnap = _btSnapshots.find(s => s.tick === targetTick);
            if (!anchorSnap) {
                // No snapshot for this tick — we've exhausted snapshot history.
                // This is equivalent to reaching t=0.
                console.error(`%c[CANARY] Rules are mathematically impossible — no snapshot for tick ${targetTick} (backed to t=0). Last violation: ${_rewindViolation}`, 'color:red;font-weight:bold;font-size:14px');
                simHalted = true;
                _btReset();
                _bfsReset();
                return false;
            }

            // Clear ledger entries for ticks after anchor (state will diverge)
            for (const [t] of _btBadMoveLedger) {
                if (t > targetTick) _btBadMoveLedger.delete(t);
            }
            _btRetryCount = 0;
            _btRestoreSnapshot(anchorSnap);
            _logChoreo(`BFS: rewound to layer ${_bfsLayer} anchor tick ${targetTick}`);
            return true;
        };

        // ── BFS LAYER TRACKING ──
        if (!_addedNewExclusions) {
            // No new information from this retry — layer is genuinely exhausted.
            // Escalate to a deeper BFS layer.
            _logChoreo(`BFS: no new exclusions at tick ${currentTick} (ledger: ${ledger.size}), escalating`);
            if (!_escalateLayer()) break; // t=0 canary fired
            continue;
        }

        // ── SAME-TICK RETRY (new exclusions were added) ──
        if (_bfsFailTick >= 0 && currentTick === _bfsFailTick) {
            _bfsLayerRetries++;
        } else {
            _btRetryCount++;
        }

        const snap = _btSnapshots[_btSnapshots.length - 1];
        _btRestoreSnapshot(snap);
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

    // Update UI — every tick (un-throttled)
    updateDemoPanel();
    updateStatus();
    updateXonPanel();

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
