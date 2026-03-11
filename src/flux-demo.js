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
//                        lollipop, ham CW, or ham CCW). First-class priority:
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
//   its _loopSeq (fork/lollipop loops require node revisits). This is
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
// OCT CAPACITY OVERFLOW (2-tier)
//   1. Send xon into unscheduled hadronic loop (idle_tet) to vacate the cage.
//   2. If impossible, eject with _t60Ejected = true.
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

let _demoActive = false;
let _demoInterval = null;
let _demoPaused = false;  // true when user has paused via pause button
// T45 bounce guard — prevents A→B→A oscillation for oct AND weak xons.
// Only tet/idle_tet xons are exempt (actualized hadronic patterns like fork: a→b→a→c→a).
// Bounces are only allowed in actualized hadronic patterns that require them.
const _T45_BOUNCE_GUARD = true;
let _demoTick = 0;
let _demoVisits = null;       // {face: {pu:0, pd:0, nu:0, nd:0}}
let _demoTetAssignments = 0;  // total tet assignments (for hit rate = completions / assignments)

// ── Rolling Ratio Tracker — demand-driven quark type selection ──
// Syncs from _demoVisits each tick. Computes deficit for any quark type.
// Target fractions: pu=2/3 of proton total, pd=1/3; nd=2/3 of neutron total, nu=1/3.
const _ratioTracker = {
    pu: 0, pd: 0, nu: 0, nd: 0,
    sync() {
        this.pu = 0; this.pd = 0; this.nu = 0; this.nd = 0;
        for (let f = 1; f <= 8; f++) {
            if (!_demoVisits || !_demoVisits[f]) continue;
            this.pu += _demoVisits[f].pu || 0;
            this.pd += _demoVisits[f].pd || 0;
            this.nu += _demoVisits[f].nu || 0;
            this.nd += _demoVisits[f].nd || 0;
        }
    },
    // Returns positive value when type is underrepresented vs target ratio
    deficit(type) {
        const protonTotal = this.pu + this.pd;
        const neutronTotal = this.nu + this.nd;
        if (type === 'pu') return protonTotal === 0 ? 1.0 : (2/3) - this.pu / protonTotal;
        if (type === 'pd') return protonTotal === 0 ? 1.0 : (1/3) - this.pd / protonTotal;
        if (type === 'nu') return neutronTotal === 0 ? 1.0 : (1/3) - this.nu / neutronTotal;
        if (type === 'nd') return neutronTotal === 0 ? 1.0 : (2/3) - this.nd / neutronTotal;
        return 0;
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
// T41: tick-level move record — tracks destNode → fromNode for all xon moves this tick.
// Used to prevent adjacent xon swaps (A→B while B→A in the same tick).
const _moveRecord = new Map();
let _noSwapRule = true; // T41: swap prevention always active — xons may not swap positions
function _swapBlocked(fromNode, toNode) {
    return _noSwapRule && _moveRecord.get(fromNode) === toNode;
}
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
const GLUON_COLOR = 0xff8800;   // orange — cage maintenance mode
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
    // FLASHLIGHT TRAP: freeze if xon moves to a non-nucleus node
    // Exception: weak-mode xons may visit ejection-space nodes (T60 ejection)
    _ensureNucleusNodeSet();
    if (_nucleusNodeSet && !_nucleusNodeSet.has(to) &&
        !(xon._mode === 'weak' && _isValidEjectionTarget(to))) {
        console.error(`[FLASHLIGHT] tick=${_demoTick} X${entry.xonIdx} moved ${from}→${to} via "${path}" mode=${xon._mode} face=${xon._assignedFace} quark=${xon._quarkType} loopStep=${xon._loopStep} loopSeq=${JSON.stringify(xon._loopSeq)}`);
        console.error(`[FLASHLIGHT] nucleus nodes: [${Array.from(_nucleusNodeSet).sort((a,b)=>a-b).join(',')}]`);
        console.error(`[FLASHLIGHT] FREEZING — node ${to} is outside the nucleus`);
        simHalted = true;
    }
}
// SC Attribution Registry — tracks why each SC entered xonImpliedSet.
// Maps scId → { reason, xonIdx, tick, face? }
// Reasons: 'faceAssign' (face SC promotion), 'manifest' (idle tet void creation),
//          'tetTraversal' (PHASE 1 SC-only edge activation)
// T42 guard: every eSC must have a valid attribution entry.
// Lookahead can query _scAttribution to make informed SC decisions.
const _scAttribution = new Map();

// ══════════════════════════════════════════════════════════════════════════
// BACKTRACKING CHOREOGRAPHER — rewind on violation, try different choices
// ══════════════════════════════════════════════════════════════════════════
let _rewindRequested = false;        // set by guard check when T19/T20 fails
let _rewindViolation = null;         // description of the violation that triggered rewind
const _BT_MAX_SNAPSHOTS = 50;       // cap snapshot stack to prevent unbounded memory
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
            _t60Ejected: !!x._t60Ejected, _mayReturn: !!x._mayReturn,
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
    };
    _btSnapshots.push(snap);
    // Keep stack bounded (cap at _BT_MAX_SNAPSHOTS)
    if (_btSnapshots.length > _BT_MAX_SNAPSHOTS) _btSnapshots.shift();
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
        x._mayReturn = !!s._mayReturn;
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
    pu: ([a, b, c, d]) => [a, b, a, c, a],      // Fork (p-up)
    nd: ([a, b, c, d]) => [a, b, c, b, a],      // Lollipop (n-down)
    pd: ([a, b, c, d]) => [a, b, c, d, a],      // Hamiltonian CW (p-down)
    nu: ([a, b, c, d]) => [a, d, c, b, a],      // Hamiltonian CCW (n-up)
};

const LOOP_TYPE_NAMES = { pu: 'fork', nd: 'lollipop', pd: 'ham_cw', nu: 'ham_ccw' };

// Weak force escape color — purple/magenta, distinct from all quark + oct colors.
// Used when a xon breaks confinement and enters the 'weak' mode.
const WEAK_FORCE_COLOR = 0xcc44ff;

const XON_TRAIL_LENGTH = 50;

// ── Weak Force Lifecycle Recorder ──
// Records up to 10 full lifecycles of weak force excitations for debugging.
// Each record: { xonIdx, entryTick, entryNode, exitTick, exitNode, exitReason, path }
const _weakLifecycleLog = [];
const _weakActiveTracking = new Map(); // xonIdx → { entryTick, entryNode, path }
const WEAK_LIFECYCLE_MAX = 10;
function _weakLifecycleEnter(xon, source) {
    if (_weakLifecycleLog.length >= WEAK_LIFECYCLE_MAX && _weakActiveTracking.size === 0) return;
    const idx = _demoXons.indexOf(xon);
    _weakActiveTracking.set(idx, {
        entryTick: _demoTick, entryNode: xon.node, source, path: [xon.node]
    });
    console.log(`[WEAK LIFECYCLE] ENTER #${_weakLifecycleLog.length + 1}: xon${idx} at node ${xon.node} (${source}) tick=${_demoTick}`);
}
function _weakLifecycleStep(xon) {
    const idx = _demoXons.indexOf(xon);
    const track = _weakActiveTracking.get(idx);
    if (track) track.path.push(xon.node);
}
function _weakLifecycleExit(xon, reason) {
    const idx = _demoXons.indexOf(xon);
    const track = _weakActiveTracking.get(idx);
    if (!track) return;
    const record = {
        xonIdx: idx, entryTick: track.entryTick, entryNode: track.entryNode,
        exitTick: _demoTick, exitNode: xon.node, exitReason: reason,
        source: track.source, path: track.path, duration: _demoTick - track.entryTick
    };
    _weakLifecycleLog.push(record);
    _weakActiveTracking.delete(idx);
    console.log(`[WEAK LIFECYCLE] EXIT #${_weakLifecycleLog.length}: xon${idx} at node ${xon.node} reason="${reason}" duration=${record.duration} path=[${record.path.join('→')}]`);
    if (_weakLifecycleLog.length === WEAK_LIFECYCLE_MAX) {
        console.log('[WEAK LIFECYCLE] ═══ 10 LIFECYCLES RECORDED ═══');
        console.table(_weakLifecycleLog.map(r => ({
            xon: r.xonIdx, src: r.source, entry: `${r.entryNode}@${r.entryTick}`,
            exit: `${r.exitNode}@${r.exitTick}`, reason: r.exitReason,
            ticks: r.duration, hops: r.path.length - 1
        })));
    }
}

// ── Clear class-specific properties on mode transitions ──
// Called BEFORE setting new mode to prevent stale properties leaking across classes.
// weak-class: _t60Ejected, _mayReturn
// tet-class: _assignedFace, _quarkType, _loopType, _loopSeq, _loopStep, _tetActualized
function _clearModeProps(xon) {
    xon._t60Ejected = false;
    xon._mayReturn = false;
    xon._tetActualized = false;
}

// ── Trail helper: freeze 3D positions at record time so trails don't deform with solver ──
function _trailPush(xon, node, color) {
    xon.trail.push(node);
    xon.trailColHistory.push(color);
    const p = pos[node];
    xon._trailFrozenPos.push(p ? [p[0], p[1], p[2]] : [0, 0, 0]);
    if (xon.trail.length > XON_TRAIL_LENGTH) {
        xon.trail.shift();
        xon.trailColHistory.shift();
        xon._trailFrozenPos.shift();
    }
}
// Initialize frozen pos array from current trail (for init/reset)
function _trailInitFrozen(xon) {
    xon._trailFrozenPos = xon.trail.map(n => {
        const p = pos[n];
        return p ? [p[0], p[1], p[2]] : [0, 0, 0];
    });
}

// Spawn a xon at a node with spark, trail, and tween — mirrors excitation visuals.
// Color by quark function: pu=yellow, pd=green, nu=blue, nd=red.
function _spawnXon(face, quarkType, sign) {
    const fd = _nucleusTetFaceData[face];
    if (!fd) return null;
    const seq = LOOP_SEQUENCES[quarkType](fd.cycle);
    const col = QUARK_COLORS[quarkType];

    // Spark sprite — uses shared _sparkTex for sparkle effect
    const sparkMat = new THREE.SpriteMaterial({
        color: col, map: _sparkTex, transparent: true, opacity: 1.0,
        blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false,
    });
    const spark = new THREE.Sprite(sparkMat);
    spark.scale.set(0.28, 0.28, 1);
    spark.renderOrder = 22;
    const group = new THREE.Group();
    group.add(spark);
    if (pos[seq[0]]) group.position.set(pos[seq[0]][0], pos[seq[0]][1], pos[seq[0]][2]);
    scene.add(group);

    // Trail line — fading vertex-colored path
    const trailGeo = new THREE.BufferGeometry();
    const trailPos = new Float32Array((XON_TRAIL_LENGTH + 1) * 3);
    const trailCol = new Float32Array((XON_TRAIL_LENGTH + 1) * 3);
    trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPos, 3));
    trailGeo.setAttribute('color', new THREE.BufferAttribute(trailCol, 3));
    const trailMat = new THREE.LineBasicMaterial({
        vertexColors: true, transparent: true, opacity: 1.0,
        depthTest: false, blending: THREE.AdditiveBlending,
    });
    const trailLine = new THREE.Line(trailGeo, trailMat);
    trailLine.renderOrder = 20;
    scene.add(trailLine);

    const xon = {
        node: seq[0], prevNode: seq[0], sign,
        _loopType: LOOP_TYPE_NAMES[quarkType],
        _loopSeq: seq, _loopStep: 0,
        _assignedFace: face, _quarkType: quarkType,
        _mode: 'tet',           // 'tet', 'oct', 'idle_tet', 'weak', or 'gluon'
        _lastDir: null,         // last direction index (0-3) for momentum
        _dirHistory: [],        // direction vector history for T16 test
        _mayReturn: false,      // weak mode: true after 1+ valid ejection moves
        _dirBalance: new Array(10).fill(0), // xonic movement balance: 4 base + 6 SC dirs
        col, group, spark, sparkMat,
        trailLine, trailGeo, trailPos, trailCol,
        trail: [seq[0]], trailColHistory: [col], _trailFrozenPos: [], tweenT: 1, flashT: 1.0,
        _highlightT: 0,
        alive: true,
    };
    _trailInitFrozen(xon);
    _demoXons.push(xon);
    return xon;
}

// Create a lightweight gluon sprite (white spark on oct edges)
function _createGluonSprite() {
    const col = 0xffffff; // white for gluon
    const sparkMat = new THREE.SpriteMaterial({
        color: col, map: _sparkTex, transparent: true, opacity: 0.8,
        blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false,
    });
    const sprite = new THREE.Sprite(sparkMat);
    sprite.scale.set(0.18, 0.18, 1);
    sprite.renderOrder = 21;
    scene.add(sprite);
    return sprite;
}

// Mark a xon as dying — spark vanishes, trail decays naturally.
// The tail "chases" into the annihilation point, shrinking each tick.
// Full cleanup happens in _tickDemoXons when trail is empty.
function _destroyXon(xon) {
    xon.alive = false;
    // Hide spark immediately (annihilated), but keep trail for decay
    if (xon.group) { scene.remove(xon.group); xon.group = null; }
    if (xon.sparkMat) { xon.sparkMat.dispose(); xon.sparkMat = null; }
    xon.spark = null;
    // Snapshot trail positions + colors — dying tracers keep historical state,
    // they do NOT follow the live solver. This creates a cool ghosting effect.
    xon._frozenPos = xon.trail.map(nodeIdx => {
        const p = pos[nodeIdx];
        return p ? [p[0], p[1], p[2]] : [0, 0, 0];
    });
    xon._frozenColors = xon.trailColHistory ? [...xon.trailColHistory] : null;
    xon._dying = true; // signal to _tickDemoXons: decay trail
}

// Final cleanup after trail has fully decayed
function _finalCleanupXon(xon) {
    if (xon.trailLine) { scene.remove(xon.trailLine); }
    if (xon.trailGeo) xon.trailGeo.dispose();
    xon.trailLine = null; xon.trailGeo = null;
    xon._dying = false;
}

// Decay dying xon trails — called ONCE per demoTick (simulation tick).
// Every dying tracer experiences every simulation tick (no frame-rate dependency).
// Removes one frozen trail point per tick (constant decay).
function _decayDyingXons() {
    for (const xon of _demoXons) {
        if (!xon._dying || !xon._frozenPos) continue;
        // Remove one trail point per tick
        if (xon._frozenPos.length > 0) {
            xon._frozenPos.shift();
            if (xon._frozenColors) xon._frozenColors.shift();
        }
        // Cleanup in simulation domain — don't wait for render frame.
        // Without this, a fully-decayed xon can sit in _dying=true for
        // extra ticks if render frames lag behind simulation ticks (T14).
        if (xon._frozenPos.length === 0) {
            if (xon.group) {
                xon._dying = false;
                xon._dyingStartTick = null;
                if (xon.trailLine) xon.trailLine.visible = false;
            }
        }
    }
}

// Check if the next hop in a xon's loop crosses an SC-only edge that is still activated.
// Returns true if traversal is safe (base edge or SC is active), false if SC was deactivated.
function _canAdvanceSafely(xon) {
    if (!xon.alive || !xon._loopSeq) return false;
    const effectiveStep = xon._loopStep >= 4 ? 0 : xon._loopStep;
    const fromNode = xon._loopSeq[effectiveStep];
    const toNode = xon._loopSeq[effectiveStep + 1];
    if (toNode === undefined) return false;
    const hasBase = (baseNeighbors[fromNode] || []).some(nb => nb.node === toNode);
    if (hasBase) return true; // base edge, no SC needed
    const pid = pairId(fromNode, toNode);
    const scId = scPairToId.get(pid);
    if (scId === undefined) return true; // no SC on this edge
    return activeSet.has(scId) || impliedSet.has(scId) || xonImpliedSet.has(scId);
}

// Advance a xon one hop: update position state, push trail, start tween.
// SC negotiation with the vacuum happens BEFORE this call in demoTick.
function _advanceXon(xon) {
    if (!xon.alive) return false;
    if (xon._loopStep >= 4) {
        xon._loopStep = 0; // wrap for continuous cycling
        xon._tetActualized = false; // reset actualization flag for new loop
    }
    const fromNode = xon._loopSeq[xon._loopStep];
    const toNode = xon._loopSeq[xon._loopStep + 1];
    if (_swapBlocked(fromNode, toNode)) return false; // T41: no swap
    xon.prevNode = fromNode;
    xon.node = toNode;
    // Verify node setter accepted the move (it blocks non-adjacent hops)
    if (xon.node !== toNode) return false;
    xon._loopStep++;

    // Update xonic movement balance counters
    _updateDirBalance(xon, fromNode, toNode);

    // Check if tet face is actualized this step (all face SCs active)
    if (xon._assignedFace != null && _nucleusTetFaceData) {
        const fd = _nucleusTetFaceData[xon._assignedFace];
        if (fd && fd.scIds.every(scId =>
            activeSet.has(scId) || impliedSet.has(scId) || xonImpliedSet.has(scId))) {
            xon._tetActualized = true;
        }
    }

    // Count when loop completes AND tet was actualized at some point during the loop
    if (xon._loopStep === 4 && xon._tetActualized &&
        xon._assignedFace != null && xon._quarkType) {
        if (_demoVisits && _demoVisits[xon._assignedFace]) {
            _demoVisits[xon._assignedFace][xon._quarkType]++;
            _demoVisits[xon._assignedFace].total++;
        }
    }

    // Push trail history + per-segment color, start tween
    _trailPush(xon, toNode, xon.col);
    xon.tweenT = 0;
    if (_flashEnabled) xon.flashT = 1.0;
    return true;
}

// ╔══════════════════════════════════════════════════════════════════════╗
// ║  PERSISTENT 6-XON MODEL — Demo 3.1                                  ║
// ╚══════════════════════════════════════════════════════════════════════╝

// (_completeOctDiscovery removed — oct is now formed deterministically in simulateNucleus)

// Spawn exactly 6 persistent xons on oct nodes. Called once from startDemoLoop.
// 3 sign=+1, 3 sign=-1. All start in oct mode (white, cruising cage).
function _initPersistentXons() {
    _demoXons = [];
    if (_octSeedCenter < 0) {
        console.error('[demo] Cannot init persistent xons: no center node');
        return;
    }
    // All 6 xons start at center (opening choreography spreads them out in 2 ticks)
    const startNode = _octSeedCenter;
    for (let i = 0; i < 6; i++) {
        const sign = i < 3 ? +1 : -1;

        // Create spark + trail visuals (white for oct mode)
        const col = 0xffffff;
        const sparkMat = new THREE.SpriteMaterial({
            color: col, map: _sparkTex, transparent: true, opacity: 1.0,
            blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false,
        });
        const spark = new THREE.Sprite(sparkMat);
        spark.scale.set(0.28, 0.28, 1);
        spark.renderOrder = 22;
        const group = new THREE.Group();
        group.add(spark);
        if (pos[startNode]) group.position.set(pos[startNode][0], pos[startNode][1], pos[startNode][2]);
        scene.add(group);

        const trailGeo = new THREE.BufferGeometry();
        const trailPos = new Float32Array((XON_TRAIL_LENGTH + 1) * 3);
        const trailCol = new Float32Array((XON_TRAIL_LENGTH + 1) * 3);
        trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPos, 3));
        trailGeo.setAttribute('color', new THREE.BufferAttribute(trailCol, 3));
        const trailMat = new THREE.LineBasicMaterial({
            vertexColors: true, transparent: true, opacity: 1.0,
            depthTest: false, blending: THREE.AdditiveBlending,
        });
        const trailLine = new THREE.Line(trailGeo, trailMat);
        trailLine.renderOrder = 20;
        scene.add(trailLine);

        const initDir = 0; // direction doesn't matter at center (opening choreography assigns)

        const xon = {
            prevNode: startNode, sign,
            _loopType: null,
            _loopSeq: null, _loopStep: 0,
            _assignedFace: null, _quarkType: null,
            _mode: 'oct_formation',
            _lastDir: initDir,
            _dirHistory: [],
            _mayReturn: false,
            _dirBalance: new Array(10).fill(0),
            col, group, spark, sparkMat,
            trailLine, trailGeo, trailPos, trailCol,
            trail: [startNode], trailColHistory: [col], _trailFrozenPos: [], tweenT: 1, flashT: 1.0,
            _highlightT: 0,
            alive: true,
        };
        _trailInitFrozen(xon);
        // Interceptor: enforce single-hop-per-tick + validate each individual movement
        let _nodeVal = startNode;
        xon._movedThisTick = false;
        Object.defineProperty(xon, 'node', {
            get() { return _nodeVal; },
            set(v) {
                const from = _nodeVal;
                if (from === v) { _nodeVal = v; return; } // no-op assignment
                // Bypass validation during backtracker snapshot restore
                if (xon._restoring) { _nodeVal = v; return; }
                // Validate: nodes must be adjacent (base edge or SC edge)
                if (typeof scPairToId !== 'undefined' && scPairToId && scPairToId.size > 0) {
                    const hasBase = (baseNeighbors[from] || []).some(nb => nb.node === v);
                    if (!hasBase) {
                        // Check if there's an SC between them
                        const scs = _localScNeighbors(from);
                        const hasSC = scs.some(sc => (sc.a === from ? sc.b : sc.a) === v);
                        if (!hasSC) {
                            console.warn(`[MOVEMENT BLOCKED] tick=${_demoTick} xon: ${from}→${v} NO EDGE (not adjacent)`);
                            return; // BLOCK: not adjacent at all
                        }
                        // SC exists — verify it's active
                        const pid = pairId(from, v);
                        const scId = scPairToId.get(pid);
                        if (scId !== undefined && !activeSet.has(scId) && !impliedSet.has(scId) && !xonImpliedSet.has(scId)) {
                            console.warn(`[MOVEMENT BLOCKED] tick=${_demoTick} xon: ${from}→${v} SC ${scId} INACTIVE`);
                            return; // BLOCK: SC not active
                        }
                    }
                    // Enforce single-hop-per-tick
                    if (xon._movedThisTick) {
                        console.warn(`[MOVEMENT BLOCKED] tick=${_demoTick} xon: ${from}→${v} ALREADY MOVED (no FTL)`);
                        return; // BLOCK: already hopped this tick
                    }
                    xon._movedThisTick = true;
                }
                _nodeVal = v;
            },
            enumerable: true, configurable: true
        });
        _demoXons.push(xon);
    }
    console.log(`[demo] Initialized 6 persistent xons at center node ${startNode}`);
}

// ╔══════════════════════════════════════════════════════════════════════╗
// ║  GLUON STORAGE — Xon Pair Annihilation / Creation                    ║
// ║                                                                      ║
// ║  Conservation: alive_count + 2 * stored_pairs = 6 (always)           ║
// ║  Annihilation: 2 xons at same node → pair stored, both removed       ║
// ║  Creation: stored pair → 2 new xons on free adjacent oct nodes       ║
// ╚══════════════════════════════════════════════════════════════════════╝
let _gluonStoredPairs = 0;

// Annihilate two xons into a stored gluon pair.
// Both xons are deactivated and visually removed.
function _annihilateXonPair(xonA, xonB) {
    // Record weak lifecycle exit if either was in weak mode
    if (xonA._mode === 'weak') _weakLifecycleExit(xonA, 'ANNIHILATED');
    if (xonB._mode === 'weak') _weakLifecycleExit(xonB, 'ANNIHILATED');
    // T42: clean up face SCs before death (must run while alive)
    _relinquishFaceSCs(xonA);
    _relinquishFaceSCs(xonB);
    // Graceful trail fade (T40): hide spark, freeze trail for decay.
    // Keep group/sparkMat intact so _manifestXonPair can reactivate later.
    for (const xon of [xonA, xonB]) {
        xon.alive = false;
        if (xon.group) xon.group.visible = false; // spark vanishes
        // Freeze trail positions for dying decay
        xon._frozenPos = xon.trail.map(nodeIdx => {
            const p = pos[nodeIdx];
            return p ? [p[0], p[1], p[2]] : [0, 0, 0];
        });
        xon._frozenColors = xon.trailColHistory ? [...xon.trailColHistory] : null;
        xon._dying = true;
    }
    _gluonStoredPairs++;
    console.log(`[gluon] Annihilation at node ${xonA.node}: stored=${_gluonStoredPairs}, alive=${_demoXons.filter(x=>x.alive).length} modes=[${xonA._mode},${xonB._mode}]`);
}

// Manifest a xon pair from gluon storage onto free oct nodes.
// Returns true if a pair was created, false if no room or no stored pairs.
function _manifestXonPair() {
    if (_gluonStoredPairs <= 0) return false;
    const aliveCount = _demoXons.filter(x => x.alive).length;
    if (aliveCount >= 6) return false;

    // Find dead xons to reactivate (recycle slots) — skip dying (trail still fading)
    const dead = _demoXons.filter(x => !x.alive && !x._dying);
    if (dead.length < 2) return false;

    // Find two free adjacent oct nodes
    if (!_octNodeSet) return false;
    const occupied = _occupiedNodes();
    let nodeA = null, nodeB = null;
    for (const n of _octNodeSet) {
        if (occupied.get(n) || 0) continue;
        // Check for a free adjacent oct node
        const nbs = baseNeighbors[n] || [];
        for (const nb of nbs) {
            if (!_octNodeSet.has(nb.node)) continue;
            if (occupied.get(nb.node) || 0) continue;
            if (n === nb.node) continue;
            nodeA = n;
            nodeB = nb.node;
            break;
        }
        if (nodeA !== null) break;
    }
    if (nodeA === null) return false; // no room

    // Reactivate two dead xons at nodeA and nodeB
    const xonA = dead[0];
    const xonB = dead[1];
    // Clear any residual dying state from trail fade
    xonA._dying = false; xonA._frozenPos = null; xonA._frozenColors = null; xonA._dyingStartTick = null;
    xonB._dying = false; xonB._frozenPos = null; xonB._frozenColors = null; xonB._dyingStartTick = null;
    xonA.alive = true;
    xonA.node = nodeA; // bypass interceptor for respawn
    xonA.prevNode = nodeA;
    xonA._mode = 'oct';
    xonA._mayReturn = false;
    xonA._assignedFace = null;
    xonA._quarkType = null;
    xonA._loopType = null;
    xonA._loopSeq = null;
    xonA._loopStep = 0;
    xonA.col = 0xffffff;
    xonA._movedThisTick = false;
    xonA.trail = [nodeA];
    xonA.trailColHistory = [0xffffff];
    _trailInitFrozen(xonA);
    xonA.tweenT = 1;
    if (_flashEnabled) xonA.flashT = 1.0;
    if (xonA.sparkMat) xonA.sparkMat.color.setHex(0xffffff);
    if (xonA.group) xonA.group.visible = true;
    if (xonA.trailLine) xonA.trailLine.visible = true;

    xonB.alive = true;
    xonB.node = nodeB;
    xonB.prevNode = nodeB;
    xonB._mode = 'oct';
    xonB._mayReturn = false;
    xonB._assignedFace = null;
    xonB._quarkType = null;
    xonB._loopType = null;
    xonB._loopSeq = null;
    xonB._loopStep = 0;
    xonB.col = 0xffffff;
    xonB._movedThisTick = false;
    xonB.trail = [nodeB];
    xonB.trailColHistory = [0xffffff];
    _trailInitFrozen(xonB);
    xonB.tweenT = 1;
    if (_flashEnabled) xonB.flashT = 1.0;
    if (xonB.sparkMat) xonB.sparkMat.color.setHex(0xffffff);
    if (xonB.group) xonB.group.visible = true;
    if (xonB.trailLine) xonB.trailLine.visible = true;

    // Genesis tracking — T31 verifies: spawns on oct nodes, in pairs, in oct mode
    xonA._genesisNode = nodeA;
    xonA._genesisTick = _demoTick;
    xonB._genesisNode = nodeB;
    xonB._genesisTick = _demoTick;

    _gluonStoredPairs--;
    console.log(`[gluon] Manifested pair at nodes ${nodeA},${nodeB}: stored=${_gluonStoredPairs}, alive=${_demoXons.filter(x=>x.alive).length}`);
    return true;
}

// Build a count map of currently occupied nodes (for Pauli exclusion)
// Uses counts because multiple xons can share a node temporarily (after tet return)
function _occupiedNodes() {
    const occ = new Map(); // node → count
    for (const xon of _demoXons) {
        if (xon.alive) occ.set(xon.node, (occ.get(xon.node) || 0) + 1);
    }
    return occ;
}
function _occAdd(occ, node) { occ.set(node, (occ.get(node) || 0) + 1); }
function _occDel(occ, node) {
    const c = (occ.get(node) || 0) - 1;
    if (c <= 0) occ.delete(node);
    else occ.set(node, c);
}

// Maximum bipartite matching for oct xon move assignment (Kuhn's algorithm).
// Finds an augmenting path of arbitrary depth so that the maximum number of
// xons get a valid destination. This prevents deadlocks that greedy assignment misses.
//   plans: array of { xon, candidates: [{node, ...}], assigned: null }
//   blocked: Set of nodes reserved by higher-priority moves (tet)
function _maxBipartiteAssignment(plans, blocked) {
    const n = plans.length;
    const assignment = new Array(n).fill(null); // plan index → candidate
    const claimed = new Map(); // dest node → plan index

    // Augmenting path search: try to assign plans[idx] to a free candidate.
    // If candidate is already taken by plans[other], recursively try to
    // reassign plans[other] to a different candidate (arbitrary depth).
    function augment(idx, visited) {
        for (const c of plans[idx].candidates) {
            if (blocked.has(c.node)) continue;
            if (visited.has(c.node)) continue;
            visited.add(c.node);

            const existing = claimed.get(c.node);
            if (existing === undefined || augment(existing, visited)) {
                assignment[idx] = c;
                claimed.set(c.node, idx);
                return true;
            }
        }
        return false;
    }

    // Most constrained first: try xons with fewest candidates first
    const order = plans.map((_, i) => i);
    order.sort((a, b) => plans[a].candidates.length - plans[b].candidates.length);

    for (const i of order) {
        augment(i, new Set());
    }

    // Apply results
    for (let i = 0; i < n; i++) {
        plans[i].assigned = assignment[i];
    }
}

// ── 6-Step Awareness System (bookended fermionic loop) ──
// Every xon must know its next 6 valid steps before committing a move.
// This covers: entry step + 4-hop tet loop + exit step.
// The lookahead uses PROJECTED occupation (where neighbors will be after
// their 1st moves) to account for cooperative multi-agent dynamics.
//
// Two lookahead modes:
// 1. Generic graph traversal (_lookahead) — for oct xons with flexible movement
// 2. Loop-shape-aware (_lookaheadTetPath) — for tet/idle_tet xons following
//    their specific fermionic loop (fork, lollipop, ham CW/CCW).
//    This simulates the xon stepping through its ACTUAL sequence, tracking
//    self-occupation to handle revisited nodes (fork: a→b→a→c→a).
//
// Lookahead depth reads from _choreoParams.lookahead (GA-tunable)

// Generic graph lookahead for oct xons (flexible movement).
// Validates against: T19 (Pauli), T26 (SC activation), T27 (connectivity),
// T29 (white trails only on oct nodes).
function _lookahead(node, occupied, depth, _visited, _selfXon) {
    if (depth <= 0) return true;
    if (!_visited) _visited = new Set();
    _visited.add(node);

    // Base-edge neighbors
    const nbs = baseNeighbors[node] || [];
    for (const nb of nbs) {
        if (_visited.has(nb.node)) continue;
        // Prefer oct nodes for normal movement
        if (_octNodeSet && !_octNodeSet.has(nb.node)) continue;
        if (occupied.get(nb.node) || 0) {
            // Occupied node = ANNIHILATION OPPORTUNITY (valid terminal move).
            return true;
        }
        if (_lookahead(nb.node, occupied, depth - 1, new Set(_visited), _selfXon)) return true;
    }
    // Active SC neighbors — T26: only traverse activated SCs
    const scs = _localScNeighbors(node);
    for (const sc of scs) {
        const other = sc.a === node ? sc.b : sc.a;
        if (_visited.has(other)) continue;
        // Prefer oct nodes for normal movement
        if (_octNodeSet && !_octNodeSet.has(other)) continue;
        if (_annihilationEnabled && (occupied.get(other) || 0)) return true; // annihilation opportunity
        // T26: SC must be activated
        if (!(activeSet.has(sc.id) || impliedSet.has(sc.id) || xonImpliedSet.has(sc.id))) continue;
        if (_lookahead(other, occupied, depth - 1, new Set(_visited), _selfXon)) return true;
    }
    // WEAK FORCE FALLBACK: if all oct-restricted paths fail, a free base neighbor
    // CLOSE TO the oct cage is a valid escape via the weak force.
    // Only consider neighbors within 2 hops of an oct node (prevents flashlight).
    for (const nb of nbs) {
        if (_visited.has(nb.node)) continue;
        if (!(occupied.get(nb.node) || 0)) {
            // Structural guard check: reject if move would violate ANY active test
            if (_selfXon && _moveViolatesGuards(_selfXon, node, nb.node)) continue;
            // Hard filter: only nucleus nodes allowed
            _ensureNucleusNodeSet();
            if (_nucleusNodeSet && !_nucleusNodeSet.has(nb.node)) continue;
            return true;
        }
    }
    return false;
}

// Loop-shape-aware COOPERATIVE lookahead for tet/idle_tet xons.
// Simulates ALL tet/idle_tet xons advancing simultaneously through their loops.
// At each timestep, checks if our xon's destination collides with any other
// tet xon's projected position (Pauli exclusion lookahead).
// Oct xons are ignored — the planner will move them.
//
// `selfXon` is the xon being checked (excluded from "others" simulation).
// If null, falls back to static occupation check.
function _lookaheadTetPath(loopSeq, fromStep, occupied, depth, selfXon) {
    // Build list of other tet/idle_tet xons with their loop state
    const others = [];
    if (selfXon) {
        for (const x of _demoXons) {
            if (!x.alive || x === selfXon) continue;
            if ((x._mode === 'tet' || x._mode === 'idle_tet') && x._loopSeq) {
                others.push({
                    step: x._loopStep >= 4 ? 0 : x._loopStep,
                    seq: x._loopSeq,
                    node: x.node,
                    face: x._assignedFace,
                    col: x.col,
                });
            }
        }
    }

    let myStep = fromStep >= 4 ? 0 : fromStep;
    let myNode = loopSeq[myStep];
    const myColor = selfXon ? selfXon.col : 0;
    const myFace = selfXon ? selfXon._assignedFace : null;

    for (let i = 0; i < depth; i++) {
        // Advance our xon
        myStep++;
        if (myStep > 4) myStep = 1;
        const myNextNode = loopSeq[myStep];
        if (myStep >= 4) myStep = 0;

        // ── T26: SC activation check ──
        // Every edge in the loop must have either a base edge or an active SC.
        const pid = pairId(myNode, myNextNode);
        const scId = scPairToId.get(pid);
        if (scId !== undefined) {
            const hasBaseEdge = (baseNeighbors[myNode] || []).some(nb => nb.node === myNextNode);
            if (!hasBaseEdge) {
                // SC-only edge: must be activated
                if (!xonImpliedSet.has(scId) && !activeSet.has(scId) && !impliedSet.has(scId)) {
                    return false; // T26 violation — path uses unactivated SC
                }
            }
        }

        // ── T27: Connectivity check ──
        // Verify nodes are actually connected (base edge or SC)
        const hasBase = (baseNeighbors[myNode] || []).some(nb => nb.node === myNextNode);
        if (!hasBase && scId === undefined) {
            return false; // T27 violation — no edge exists between these nodes
        }

        // Advance all other tet xons simultaneously
        for (const o of others) {
            o.step++;
            if (o.step > 4) o.step = 1;
            o.node = o.seq[o.step];
            if (o.step >= 4) o.step = 0;
        }

        // ── T19: Pauli check — collision with another tet xon ──
        const tetCollision = others.some(o => o.node === myNextNode);
        if (tetCollision) {
            // Collision = ANNIHILATION OPPORTUNITY.
            // Same-node collisions are resolved via gluon storage (pair annihilation).
            // Annihilation is a legitimate tool — it always happens in pairs and
            // genesis restores xons on oct edges. This is a valid terminal state.
            return true;
        }

        myNode = myNextNode;
    }
    return true; // path clears all guard checks for projected timesteps
}

// Unified lookahead dispatcher: uses loop-shape-aware check for tet/idle_tet,
// generic graph traversal for oct.
function _lookaheadForXon(xon, node, occupied, depth) {
    if ((xon._mode === 'tet' || xon._mode === 'idle_tet') && xon._loopSeq) {
        // Find which step in the loop corresponds to `node`
        let currentStep = -1;
        for (let i = 0; i <= 4; i++) {
            if (xon._loopSeq[i] === node) { currentStep = i; break; }
        }
        if (currentStep === -1) return _lookahead(node, occupied, depth); // fallback
        if (currentStep >= 4) currentStep = 0;
        return _lookaheadTetPath(xon._loopSeq, currentStep, occupied, depth, xon);
    }
    return _lookahead(node, occupied, depth);
}

// Compute the projected occupation map after all planned moves execute.
// Returns a Map<node, count> of where xons will be.
function _projectOccupation(tetPlans, octPlans) {
    const result = new Map();
    for (const xon of _demoXons) {
        if (!xon.alive) continue;
        let futureNode = xon.node;
        // Check tet plans
        const tp = tetPlans.find(p => p.xon === xon && p.approved);
        if (tp) { futureNode = tp.toNode; }
        // Check oct plans (assigned or idleTet)
        const op = octPlans ? octPlans.find(p => p.xon === xon) : null;
        if (op) {
            if (op.assigned) futureNode = op.assigned.node;
            else if (op.idleTet && xon._loopSeq) {
                const nextStep = xon._loopStep >= 4 ? 1 : xon._loopStep + 1;
                futureNode = xon._loopSeq[nextStep] || xon.node;
            }
        }
        _occAdd(result, futureNode);
    }
    return result;
}

// ── Cooperative 2-Step Awareness ──
// After all planning, verify every xon has a valid 2nd move by projecting
// where ALL xons will be after their 1st moves (neighbors' choices).
// For tet/idle_tet xons: 2nd move is deterministic (next loop step) — check THAT node.
// For oct xons: 2nd move is flexible — check that ANY neighbor is reachable.
// Returns array of stuck xon info. Iteratively fixes conflicts.

function _getXonFutureNode(xon, tetPlans, octPlans) {
    let futureNode = xon.node;
    const tp = tetPlans.find(p => p.xon === xon && p.approved);
    if (tp) return tp.toNode;
    const op = octPlans ? octPlans.find(p => p.xon === xon) : null;
    if (op && op.assigned) return op.assigned.node;
    if (op && op.idleTet && xon._loopSeq) {
        const nextStep = xon._loopStep >= 4 ? 1 : xon._loopStep + 1;
        return xon._loopSeq[nextStep] || xon.node;
    }
    return futureNode;
}

function _xonHas2ndMove(xon, futureNode, projected, tetPlans, octPlans) {
    // Remove self from projected so we don't block ourselves
    _occDel(projected, futureNode);

    let has2nd = false;
    const futureMode = xon._mode; // mode after 1st move

    if (futureMode === 'tet' || futureMode === 'idle_tet') {
        // Loop-shape-aware: check the full remaining loop path, not just 1 step.
        // Uses the xon's actual loop sequence (fork, lollipop, ham CW/CCW).
        if (xon._loopSeq) {
            const tp = tetPlans.find(p => p.xon === xon && p.approved);
            let stepAfter1st;
            if (tp) {
                const effective = xon._loopStep >= 4 ? 0 : xon._loopStep;
                stepAfter1st = effective + 1;
            } else {
                stepAfter1st = (xon._loopStep >= 4 ? 0 : xon._loopStep) + 1;
            }
            if (stepAfter1st >= 4) stepAfter1st = 0;
            // Check remaining loop path for _choreoParams.lookahead - 1 steps (we already used 1)
            has2nd = _lookaheadTetPath(xon._loopSeq, stepAfter1st, projected, _choreoParams.lookahead - 1, xon);
        }
    } else {
        // Oct mode: any reachable neighbor is a valid 2nd move
        has2nd = _lookahead(futureNode, projected, 1);
    }

    _occAdd(projected, futureNode);
    return has2nd;
}

// ── Single-Move Guard Check ──
// Validates a proposed move for one xon against ALL projected guards.
// STRUCTURAL GUARANTEE: any guard with projected() in LIVE_GUARD_REGISTRY
// is automatically checked here. Add projected() to your test = covered everywhere.
// Used by: _lookahead weak-force fallback, PHASE 0.5 weak BFS, PHASE 3/5 escape hatches.
function _moveViolatesGuards(xon, fromNode, toNode) {
    // Check persistent bad-move ledger (during backtrack retries)
    if (_btActive) {
        const xonIdx = _demoXons.indexOf(xon);
        if (_btIsMoveExcluded(xonIdx, toNode)) return true;
    }
    if (typeof PROJECTED_GUARD_CHECKS === 'undefined' || !PROJECTED_GUARD_CHECKS.length) return false;
    // Build futures: this xon at toNode, all others at current positions
    const futures = [];
    for (const x of _demoXons) {
        if (!x.alive) continue;
        if (x === xon) {
            futures.push({ xon: x, futureNode: toNode, fromNode, futureMode: x._mode, futureColor: x.col });
        } else {
            futures.push({ xon: x, futureNode: x.node, fromNode: x.node, futureMode: x._mode, futureColor: x.col });
        }
    }
    return _validateProjectedGuards(futures).length > 0;
}

// ── Detailed guard violation checker (for decision ledger logging) ──
// Returns an array of {reason} objects explaining why a move is blocked,
// or empty array if the move is allowed.
function _moveViolatesGuardsDetailed(xon, fromNode, toNode) {
    const reasons = [];
    if (_btActive) {
        const xonIdx = _demoXons.indexOf(xon);
        if (_btIsMoveExcluded(xonIdx, toNode)) {
            reasons.push({ reason: 'backtracker-excluded' });
            return reasons; // backtracker exclusion is terminal
        }
    }
    if (typeof PROJECTED_GUARD_CHECKS === 'undefined' || !PROJECTED_GUARD_CHECKS.length) return reasons;
    const futures = [];
    for (const x of _demoXons) {
        if (!x.alive) continue;
        if (x === xon) {
            futures.push({ xon: x, futureNode: toNode, fromNode, futureMode: x._mode, futureColor: x.col });
        } else {
            futures.push({ xon: x, futureNode: x.node, fromNode: x.node, futureMode: x._mode, futureColor: x.col });
        }
    }
    for (const check of PROJECTED_GUARD_CHECKS) {
        const result = check(futures);
        if (result) {
            const items = Array.isArray(result) ? result : [result];
            for (const v of items) {
                if (v) reasons.push({ reason: `${v.guard || '?'}: ${v.msg || JSON.stringify(v)}` });
            }
        }
    }
    return reasons;
}

// ── Decision Ledger Logger ──
// Logs a complete decision ledger for a weak xon showing every base neighbor
// and why it was accepted or rejected. Includes backtracker context.
function _logWeakDecisionLedger(xon, occupied) {
    const xi = _demoXons.indexOf(xon);
    const allNbs = baseNeighbors[xon.node] || [];
    const btLabel = _btActive ? ` [BT retry #${typeof _btRetryCount !== 'undefined' ? _btRetryCount : '?'}]` : ' [FIRST attempt]';
    const lines = [`[DECISION LEDGER] tick=${_demoTick} X${xi} at node ${xon.node} (${xon._mode})${btLabel} — ${allNbs.length} neighbors:`];
    let anyOpen = false;
    for (const nb of allNbs) {
        const checks = [];
        let blocked = false;
        // Occupancy
        const occ = occupied.get(nb.node) || 0;
        if (occ > 0) { checks.push(`OCCUPIED(${occ})`); blocked = true; }
        else checks.push('free');
        // Swap blocked
        if (_swapBlocked(xon.node, nb.node)) { checks.push('SWAP-BLOCKED'); blocked = true; }
        // Guard violations (detailed — shows backtracker-excluded and specific guards)
        const guardViolations = _moveViolatesGuardsDetailed(xon, xon.node, nb.node);
        if (guardViolations.length > 0) {
            for (const v of guardViolations) { checks.push(`GUARD:${v.reason}`); }
            blocked = true;
        }
        if (!blocked) anyOpen = true;
        // Node classification
        const tags = [];
        if (_octNodeSet && _octNodeSet.has(nb.node)) tags.push('oct');
        if (_purelyTetNodes && _purelyTetNodes.has(nb.node)) tags.push('pureTet');
        if (_nucleusNodeSet && _nucleusNodeSet.has(nb.node)) tags.push('nucleus');
        if (_ejectionTargetNodes && _ejectionTargetNodes.has(nb.node)) tags.push('ejTarget');
        const tagStr = tags.length ? ` [${tags.join(',')}]` : '';
        const status = blocked ? '✗' : '✓';
        lines.push(`  ${status} node ${nb.node}${tagStr}: ${checks.join(', ')}`);
    }
    lines.push(anyOpen ? `  → HAS viable moves` : `  → ALL BLOCKED — xon will be stuck!`);
    console.error(lines.join('\n'));
}

// ── Projected Guard Validator ──
// Iterates the PROJECTED_GUARD_CHECKS array (defined in flux-tests.js).
// Each check function receives the projected xon states and returns violations.
// Adding a new test to that array = automatically covered by lookahead.
//
// `xonFutures` is an array of { xon, futureNode, futureMode, futureColor, fromNode }
function _validateProjectedGuards(xonFutures) {
    if (typeof PROJECTED_GUARD_CHECKS === 'undefined' || !PROJECTED_GUARD_CHECKS.length) {
        return []; // guard checks not loaded yet (flux-tests.js loads after flux-demo.js)
    }
    const violations = [];
    for (const check of PROJECTED_GUARD_CHECKS) {
        const result = check(xonFutures);
        if (result) {
            const items = Array.isArray(result) ? result : [result];
            for (const v of items) if (v) violations.push(v);
        }
    }
    return violations;
}

// _verifyPlan: Removed — backtracker handles downstream violations
// function _verifyPlan(tetPlans, octPlans) { ... }

// ═══════════════════════════════════════════════════════════════════════
// Demand-driven face scoring — nucleus-as-one-system approach
// Scores a (xon, face) pair. Returns {face, quarkType, score} or null.
// Pure function, no side effects. Used as edge weight in global bipartite matching.
//
// Priority order (per spec §6):
//   1. Quark type selection (hadronic ratio deficit — weighted 10×)
//   2. Xonic movement balance (which directions the loop would exercise)
//   3. Vacancy (is another xon already on this face?)
// Reachability is pass/fail only (return null if unreachable).
// Anti-phase and coverage deficit are subsumed by xonic balance.
// ═══════════════════════════════════════════════════════════════════════
function _scoreFaceOpportunity(xon, face, occupied) {
    if (!_nucleusTetFaceData || !_nucleusTetFaceData[face]) return null;
    const fd = _nucleusTetFaceData[face];

    // REACHABILITY (pass/fail): xon must be on a face oct node or 1 hop away
    const faceOctNodes = [];
    for (const n of fd.cycle) {
        if (_octNodeSet && _octNodeSet.has(n)) faceOctNodes.push(n);
    }
    const onFace = faceOctNodes.includes(xon.node);
    if (!onFace) {
        let nearFace = false;
        for (const nb of (baseNeighbors[xon.node] || [])) {
            if (faceOctNodes.includes(nb.node)) { nearFace = true; break; }
        }
        if (!nearFace) return null; // unreachable this tick
    }

    let score = 0;

    // 1. QUARK TYPE SELECTION (hadronic ratio deficit — weighted 10×)
    const isProtonFace = A_SET.has(face);
    const primaryType = isProtonFace ? 'pu' : 'nd';
    const secondaryType = isProtonFace ? 'pd' : 'nu';
    const primaryDeficit = _ratioTracker.deficit(primaryType);
    const secondaryDeficit = _ratioTracker.deficit(secondaryType);
    let quarkType;
    if (secondaryDeficit > primaryDeficit + _choreoParams.ratioThreshold) {
        quarkType = secondaryType;
        score += secondaryDeficit * _choreoParams.ratioDeficitWeight * 10; // 10× hadronic weight
    } else {
        quarkType = primaryType;
        score += Math.max(0, primaryDeficit) * _choreoParams.ratioDeficitWeight * 10; // 10× hadronic weight
    }

    // 2. XONIC MOVEMENT BALANCE: score by how much the loop's directions
    //    help balance this xon's 10-direction counters.
    //    A tet loop traverses 4 edges (5-node sequence). Compute direction
    //    indices for those edges and sum the balance deficits.
    if (fd.cycle && fd.cycle.length === 4) {
        // Get the loop sequence for the chosen quark type
        const loopSeq = LOOP_SEQUENCES[quarkType] ? LOOP_SEQUENCES[quarkType](fd.cycle) : null;
        if (loopSeq && loopSeq.length === 5) {
            const loopDirs = [];
            for (let s = 0; s < 4; s++) {
                const d = _identifyMoveDir(loopSeq[s], loopSeq[s + 1]);
                if (d >= 0) loopDirs.push(d);
            }
            score += _dirBalanceScoreMulti(xon, loopDirs);
        }
    }

    // 3. VACANCY: penalize if another xon is already executing a loop on this face
    for (const x of _demoXons) {
        if (!x.alive || x === xon) continue;
        if ((x._mode === 'tet' || x._mode === 'idle_tet') && x._assignedFace === face) {
            score -= _choreoParams.faceOccupiedPenalty;
            break;
        }
    }

    return { face, quarkType, score, onFace };
}

// Get scored oct-mode candidates for a xon. Returns array sorted by momentum score (desc).
// `blocked` is an optional Set of additional nodes to treat as occupied (for coordinated planning).
function _getOctCandidates(xon, occupied, blocked) {
    if (!xon.alive) return [];
    if (xon._mode !== 'oct' && xon._mode !== 'weak') return [];

    // Weak xons: T60-ejected must move AWAY from oct cage; others navigate freely
    if (xon._mode === 'weak') {
        const candidates = [];
        const isEjected = !!xon._t60Ejected;
        for (const nb of _localBaseNeighbors(xon.node)) {
            if ((occupied.get(nb.node) || 0) > 0) continue;
            if (blocked && blocked.has(nb.node)) continue;
            if (nb.node === xon.prevNode && xon.prevNode !== xon.node) continue;
            // T61: ejected weak xons must NOT target oct nodes (must eject away)
            if (isEjected && _octNodeSet && _octNodeSet.has(nb.node)) continue;
            candidates.push({ node: nb.node, dirIdx: nb.dirIdx, score: 1, _scId: undefined, _needsMaterialise: false });
        }
        return candidates;
    }

    // Constrain oct movement to cage nodes only.
    if (!_octNodeSet) return [];

    // Constrain oct movement to cage nodes only.
    // Off-cage xons get no candidates here; the fallback _startIdleTetLoop handles them.
    const onCage = _octNodeSet.has(xon.node);
    if (!onCage) return [];

    // Get neighbors: base edges + SC edges (filtered to oct cage, excluding antipodal)
    const antipodal = _octAntipodal.get(xon.node);
    const allOctNeighbors = [];
    for (const nb of baseNeighbors[xon.node]) {
        if (_octNodeSet.has(nb.node) && nb.node !== antipodal) {
            allOctNeighbors.push({ node: nb.node, dirIdx: nb.dirIdx });
        }
    }
    const scs = _localScNeighbors(xon.node);
    for (const sc of scs) {
        const other = sc.a === xon.node ? sc.b : sc.a;
        if (_octNodeSet.has(other) && other !== antipodal && !allOctNeighbors.find(n => n.node === other)) {
            const scId = sc.id;
            const alreadyActive = activeSet.has(scId) || impliedSet.has(scId) || xonImpliedSet.has(scId);
            // Use stype-based direction index (4-9) for xonic movement balance
            const scDirIdx = _STYPE_TO_DIR[sc.stype] !== undefined ? _STYPE_TO_DIR[sc.stype] : 4;
            allOctNeighbors.push({
                node: other, dirIdx: scDirIdx,
                _scId: scId, _needsMaterialise: !alreadyActive
            });
        }
    }

    if (allOctNeighbors.length === 0) return [];

    // Score candidates by xonic movement balance (least-used direction = highest score)
    const candidates = [];
    for (const nb of allOctNeighbors) {
        if (occupied.has(nb.node)) continue; // Pauli: already occupied
        if (blocked && blocked.has(nb.node)) continue; // Pauli: reserved by another planned move
        // No bouncing: don't go back to the node we just came from
        if (nb.node === xon.prevNode && xon.prevNode !== xon.node) continue;
        const balScore = _dirBalanceScore(xon, nb.dirIdx);
        candidates.push({ node: nb.node, dirIdx: nb.dirIdx, score: balScore, _scId: nb._scId, _needsMaterialise: nb._needsMaterialise });
    }

    // 2-step awareness SCORING — penalize candidates that appear to lack a
    // 2nd move. This is a heuristic using partial occupation (oct xons removed).
    // The AUTHORITATIVE hard check happens in the cooperative post-plan
    // verification, which uses full projected state (neighbors' 1st moves).
    const tmpOcc = new Map(occupied);
    if (blocked) for (const n of blocked) _occAdd(tmpOcc, n);
    for (const c of candidates) {
        _occAdd(tmpOcc, c.node);
        if (!_lookahead(c.node, tmpOcc, 1)) {
            c.score -= _choreoParams.octDeadEndPenalty; // strong penalty — but NOT eliminated, since other
                           // oct xons may vacate and open up 2nd-move paths
        }
        _occDel(tmpOcc, c.node);
    }

    // Sort by score descending (prefer xonic balance + 2-step awareness)
    candidates.sort((a, b) => b.score - a.score);
    return candidates;
}

// Execute an oct move to a specific target. Handles vacuum negotiation.
// Returns true if the move succeeded, false if vacuum rejected.
function _executeOctMove(xon, target) {
    // Reject self-moves (target is current node) — these are no-ops that corrupt prevNode
    if (target.node === xon.node) return false;
    // T45: anti-bounce guard — reject move back to prevNode for oct/weak xons
    if (_T45_BOUNCE_GUARD && (xon._mode === 'oct' || xon._mode === 'weak') && target.node === xon.prevNode && xon.prevNode !== xon.node) {
        return false;
    }
    // Re-check SC activation at execution time (may have changed since planning)
    if (target._scId !== undefined) {
        const stillActive = activeSet.has(target._scId) || impliedSet.has(target._scId) || xonImpliedSet.has(target._scId);
        const hasBase = (baseNeighbors[xon.node] || []).some(nb => nb.node === target.node);
        if (!stillActive && !hasBase) {
            // SC was deactivated since planning — need materialization now
            target._needsMaterialise = true;
        }
    }
    // Vacuum negotiation: if target SC is inactive, try to materialise
    if (target._needsMaterialise && target._scId !== undefined) {
        let materialised = false;
        if (canMaterialiseQuick(target._scId)) {
            activeSet.add(target._scId);
            stateVersion++; // invalidate cache
            materialised = true;
        } else if (excitationSeverForRoom(target._scId)) {
            if (canMaterialiseQuick(target._scId)) {
                activeSet.add(target._scId);
                stateVersion++; // invalidate cache
                materialised = true;
            }
        }
        if (!materialised) return false; // vacuum rejected
        xon._solverNeeded = true;
    }

    // Record direction history for T16 momentum test
    if (pos[xon.node] && pos[target.node]) {
        const dx = pos[target.node][0] - pos[xon.node][0];
        const dy = pos[target.node][1] - pos[xon.node][1];
        const dz = pos[target.node][2] - pos[xon.node][2];
        const len = Math.sqrt(dx*dx + dy*dy + dz*dz) || 1;
        xon._dirHistory.push([dx/len, dy/len, dz/len]);
        if (xon._dirHistory.length > 200) xon._dirHistory.splice(0, 100);
    }

    // Move
    const fromNode = xon.node;
    xon.prevNode = xon.node;
    xon.node = target.node;
    xon._lastDir = target.dirIdx;

    // Update xonic movement balance counters
    _updateDirBalance(xon, fromNode, target.node);

    // Push trail history + per-segment color, start tween
    _trailPush(xon, target.node, xon.col);
    xon.tweenT = 0;
    if (_flashEnabled) xon.flashT = 1.0;
    return true;
}

// Legacy wrapper — used by collision scatter in PASS 1.5
function _advanceOctXon(xon, occupied) {
    const candidates = _getOctCandidates(xon, occupied);
    if (candidates.length === 0) return false;
    // Try candidates in order; skip those needing materialisation that fails
    for (const c of candidates) {
        if (_executeOctMove(xon, c)) return true;
    }
    return false;
}

// ── Traversal Lock ──────────────────────────────────────────────
// Returns a Set of SC IDs that xons are currently sitting on (prevNode→node).
// These SCs MUST NOT be removed from any set until the next tick.
// Call this before any SC deletion to check if the SC is locked.
function _traversalLockedSCs(excludeXon) {
    // EDGE-ONLY lock: only the SC on the edge a xon just traversed (prevNode↔node).
    // Physics: "if I used a shortcut on my last turn, it must exist on this turn."
    // No face-level lock — xons negotiate with the vacuum before each hop.
    const locked = new Set();
    for (const xon of _demoXons) {
        if (!xon.alive || xon.prevNode == null) continue;
        if (xon === excludeXon) continue;
        const pid = pairId(xon.prevNode, xon.node);
        const scId = scPairToId.get(pid);
        if (scId !== undefined) locked.add(scId);
    }
    return locked;
}

// Promote impliedSet-only face SCs into xonImpliedSet so they persist.
// impliedSet is ephemeral (rebuilt each solver tick). When a xon is assigned
// to a face, the SCs it will traverse must be in a persistent set.
function _promoteFaceSCs(face, xon) {
    const fd = _nucleusTetFaceData[face];
    if (!fd) return;
    const xi = xon ? _demoXons.indexOf(xon) : -1;
    for (const scId of fd.scIds) {
        if (impliedSet.has(scId) && !xonImpliedSet.has(scId) && !activeSet.has(scId)) {
            xonImpliedSet.add(scId);
            _scAttribution.set(scId, { reason: 'faceAssign', xonIdx: xi, face, tick: _demoTick });
            stateVersion++;
        }
    }
}

// Transition xon from oct mode to tet mode (assigned to actualize a face)
function _assignXonToTet(xon, face, quarkType) {
    const fd = _nucleusTetFaceData[face];
    if (!fd) return;
    _demoTetAssignments++;  // track for hit rate
    _promoteFaceSCs(face, xon);

    let seq = LOOP_SEQUENCES[quarkType](fd.cycle);
    const col = QUARK_COLORS[quarkType];
    const cycle = fd.cycle; // [a, b, c, d]

    // If xon is already at seq[0], use the sequence as-is.
    // If xon is at a different oct node on this face, rotate the cycle
    // so the xon starts from where it already is (no teleportation / Pauli safe).
    if (xon.node !== seq[0]) {
        const octNodesOnFace = cycle.filter(n => _octNodeSet.has(n));
        const currentIdx = octNodesOnFace.indexOf(xon.node);
        if (currentIdx >= 0) {
            // Rotate cycle so xon's current node is in position 0
            const a = cycle[0], b = cycle[1], c = cycle[2], d = cycle[3];
            let rotated;
            if (xon.node === a) rotated = [a, b, c, d];
            else if (xon.node === c) rotated = [c, b, a, d]; // swap a↔c
            else if (xon.node === d) rotated = [d, b, c, a]; // swap a↔d
            else rotated = cycle; // fallback
            seq = LOOP_SEQUENCES[quarkType](rotated);
        } else {
            // Xon is NOT on this face — walk ONE HOP toward nearest face oct node.
            const faceOctNodes = new Set(octNodesOnFace);
            const target = _walkToFace(xon, faceOctNodes);
            if (target !== null) {
                // Reached a face node in one hop — rotate cycle
                const a = cycle[0], b = cycle[1], c = cycle[2], d = cycle[3];
                let rotated;
                if (target === a) rotated = [a, b, c, d];
                else if (target === c) rotated = [c, b, a, d];
                else if (target === d) rotated = [d, b, c, a];
                else rotated = cycle;
                seq = LOOP_SEQUENCES[quarkType](rotated);
            } else {
                // Didn't reach face in one hop — abort assignment (no teleportation).
                // Xon stays in oct mode; assignment will retry next window.
                return;
            }
        }
    }

    _clearModeProps(xon);
    xon._mode = 'tet';
    xon._assignedFace = face;
    xon._quarkType = quarkType;
    xon._loopType = LOOP_TYPE_NAMES[quarkType];
    xon._loopSeq = seq;
    xon._loopStep = 0;
    xon.col = col;

    // Update spark color
    if (xon.sparkMat) xon.sparkMat.color.setHex(col);

    // Safety: if xon isn't at seq[0], abort instead of teleporting (T27)
    if (xon.node !== seq[0]) {
        _clearModeProps(xon);
        xon._mode = 'oct';
        xon._assignedFace = null;
        xon._quarkType = null;
        xon._loopType = null;
        xon._loopSeq = null;
        xon._loopStep = 0;
        xon.col = 0xffffff;
        if (xon.sparkMat) xon.sparkMat.color.setHex(0xffffff);
        return;
    }
}

// Walk xon ONE HOP toward nearest node in targetNodes via connected edges (BFS).
// Returns the target node if xon is already there, or the first step if it moved.
// Returns null if no path exists. ONE HOP PER TICK — no teleportation (T27).
function _walkToFace(xon, targetNodes) {
    if (targetNodes.has(xon.node)) return xon.node;
    if (xon._movedThisTick) return null; // one hop per tick — no double-move (T27)

    // Build occupied set (exclude self)
    const occupiedNodes = new Set();
    for (const x of _demoXons) {
        if (x !== xon && x.alive) occupiedNodes.add(x.node);
    }

    // BFS from xon.node to nearest target, only via base edges + active SCs
    // Exclude antipodal oct node hops (diagonal traversal)
    const visited = new Set([xon.node]);
    const parent = new Map();
    const queue = [xon.node];
    let found = null;

    while (queue.length > 0 && !found) {
        const curr = queue.shift();
        const currAntipodal = _octAntipodal.get(curr);
        const nbs = baseNeighbors[curr] || [];
        for (const nb of nbs) {
            if (visited.has(nb.node)) continue;
            if (!_octNodeSet.has(nb.node)) continue;
            if (nb.node === currAntipodal) continue; // no diagonal hops
            visited.add(nb.node);
            parent.set(nb.node, curr);
            // Pauli: only accept unoccupied target nodes (T19)
            if (targetNodes.has(nb.node) && !occupiedNodes.has(nb.node)) { found = nb.node; break; }
            if (occupiedNodes.has(nb.node)) continue;
            queue.push(nb.node);
        }
        if (found) break;
        const scs = _localScNeighbors(curr);
        for (const sc of scs) {
            if (!activeSet.has(sc.id) && !impliedSet.has(sc.id) && !xonImpliedSet.has(sc.id)) continue;
            const neighbor = sc.a === curr ? sc.b : sc.a;
            if (visited.has(neighbor)) continue;
            if (!_octNodeSet.has(neighbor)) continue;
            if (neighbor === currAntipodal) continue; // no diagonal hops
            visited.add(neighbor);
            parent.set(neighbor, curr);
            // Pauli: only accept unoccupied target nodes (T19)
            if (targetNodes.has(neighbor) && !occupiedNodes.has(neighbor)) { found = neighbor; break; }
            if (occupiedNodes.has(neighbor)) continue;
            queue.push(neighbor);
        }
    }

    if (!found) return null;

    // Reconstruct path
    const path = [];
    let n = found;
    while (n !== xon.node) { path.push(n); n = parent.get(n); }
    path.reverse();

    // ONE HOP ONLY — no teleportation (T27)
    const step = path[0];
    if (_swapBlocked(xon.node, step)) return null; // T41: abort if swap
    const fromWF = xon.node;
    xon.prevNode = xon.node;
    xon.node = step;
    xon._movedThisTick = true; // one hop per tick — prevent double-move
    _moveRecord.set(step, fromWF);
    _traceMove(xon, fromWF, step, 'walkToFace');

    _trailPush(xon, step, 0xffffff);
    xon.tweenT = 0;

    // Return the target if we reached it in one hop, otherwise null (still walking)
    return targetNodes.has(step) ? step : null;
}

// T42: Clean up face SCs from xonImpliedSet when a xon abandons its tet face.
// Respects traversal lock — won't remove SCs being traversed by other xons.
function _relinquishFaceSCs(xon) {
    if (xon._assignedFace == null) return;
    const fd = _nucleusTetFaceData ? _nucleusTetFaceData[xon._assignedFace] : null;
    if (!fd) return;
    const locked = _traversalLockedSCs(xon); // exclude self — don't self-lock
    for (const scId of fd.scIds) {
        if (locked.has(scId)) continue;
        if (xonImpliedSet.has(scId) && !activeSet.has(scId)) {
            xonImpliedSet.delete(scId);
            _scAttribution.delete(scId);
            stateVersion++;
        }
    }
}

// Transition xon from tet mode back to oct mode after loop completion.
// Optional `occupied` map prevents Pauli violations when multiple xons return simultaneously.
function _returnXonToOct(xon, occupied) {
    // If at a non-oct node, check if we can actually reach an oct node first.
    // Only clear assignment and switch to oct mode if we can get there.
    if (_octNodeSet && !_octNodeSet.has(xon.node)) {
        const nbs = baseNeighbors[xon.node] || [];
        let target = null;
        for (const nb of nbs) {
            if (!_octNodeSet.has(nb.node)) continue;
            if (_swapBlocked(xon.node, nb.node)) continue;
            if (occupied && (occupied.get(nb.node) || 0) > 0) continue;
            target = nb;
            break;
        }
        if (!target) {
            // Can't reach an oct node — DON'T switch to oct mode (would violate T16).
            // Keep current mode; will retry next tick.
            return;
        }
        // Can reach an oct node — proceed with mode transition + move
        _relinquishFaceSCs(xon); // T42: clean up face SCs before clearing assignment
        _clearModeProps(xon);
        xon._mode = 'oct';
        xon._assignedFace = null;
        xon._quarkType = null;
        xon._loopType = null;
        xon._loopSeq = null;
        xon._loopStep = 0;
        xon.col = 0xffffff;
        if (_flashEnabled) xon.flashT = 1.0;
        if (xon.sparkMat) xon.sparkMat.color.setHex(0xffffff);

        const fromRTO = xon.node;
        xon.prevNode = xon.node;
        xon.node = target.node;
        xon._movedThisTick = true;
        _moveRecord.set(target.node, fromRTO);
        _traceMove(xon, fromRTO, target.node, 'returnToOct');
        if (occupied) { _occDel(occupied, fromRTO); _occAdd(occupied, target.node); }
        _trailPush(xon, target.node, xon.col);
    } else {
        // Already at an oct node — just switch mode
        _relinquishFaceSCs(xon);
        _clearModeProps(xon);
        xon._mode = 'oct';
        xon._assignedFace = null;
        xon._quarkType = null;
        xon._loopType = null;
        xon._loopSeq = null;
        xon._loopStep = 0;
        xon.col = 0xffffff;
        if (_flashEnabled) xon.flashT = 1.0;
        if (xon.sparkMat) xon.sparkMat.color.setHex(0xffffff);
    }
}

// Start an idle tet loop for a xon boxed in on the oct surface.
// CONSTRAINT: xons can ONLY idle in already-actualized tets — faces whose
// SCs are already in xonImpliedSet or activeSet. No new geometry created.
// Returns true if a loop was started, false if no actualized face found.
function _startIdleTetLoop(xon, occupied) {
    if (!_nucleusTetFaceData) return false;

    const types = ['pu', 'nd', 'pd', 'nu'];

    // ── Pass 1: Try already-actualized faces ──
    const actualizedFaces = [];
    const manifestCandidates = []; // faces we could try to manifest
    for (const [fStr, fd] of Object.entries(_nucleusTetFaceData)) {
        if (!fd.cycle.includes(xon.node)) continue;
        const actualized = fd.scIds.every(scId =>
            xonImpliedSet.has(scId) || activeSet.has(scId) || impliedSet.has(scId));
        if (actualized) {
            actualizedFaces.push(parseInt(fStr));
        } else {
            manifestCandidates.push(parseInt(fStr));
        }
    }

    // Helper: try to assign xon to a face with free destination
    function tryFaces(faces) {
        const shuffled = faces.sort(() => Math.random() - 0.5);
        const shuffledTypes = types.slice().sort(() => Math.random() - 0.5);
        let bestSeq = null, bestFace = null, bestType = null;
        for (const face of shuffled) {
            const existingXon = _demoXons.find(x =>
                x.alive && x !== xon && x._assignedFace === face &&
                (x._mode === 'tet' || x._mode === 'idle_tet'));
            const fd = _nucleusTetFaceData[face];
            const cycle = fd.cycle;
            const [a, b, c, d] = cycle;
            let rotated;
            if (xon.node === a) rotated = [a, b, c, d];
            else if (xon.node === c) rotated = [c, b, a, d];
            else if (xon.node === d) rotated = [d, b, c, a];
            else if (xon.node === b) rotated = [b, a, d, c];
            else continue;

            for (const qType of shuffledTypes) {
                const seq = LOOP_SEQUENCES[qType](rotated);
                const dest = seq[1];
                if (occupied && occupied.has(dest)) continue;
                _promoteFaceSCs(face, xon);
                _clearModeProps(xon);
                xon._mode = 'idle_tet';
                xon._loopSeq = seq;
                xon._loopStep = 0;
                xon._assignedFace = face;
                xon._quarkType = qType;
                xon._loopType = LOOP_TYPE_NAMES[qType];
                xon.col = QUARK_COLORS[qType];
                if (_flashEnabled) xon.flashT = 1.0;
                if (xon.sparkMat) xon.sparkMat.color.setHex(xon.col);
                return true;
            }
            if (!bestSeq) {
                const fallbackType = existingXon
                    ? shuffledTypes.find(t => QUARK_COLORS[t] === existingXon.col) || shuffledTypes[0]
                    : shuffledTypes[0];
                bestSeq = LOOP_SEQUENCES[fallbackType](rotated);
                bestFace = face;
                bestType = fallbackType;
            }
        }
        if (bestSeq) {
            _promoteFaceSCs(bestFace, xon);
            _clearModeProps(xon);
            xon._mode = 'idle_tet';
            xon._loopSeq = bestSeq;
            xon._loopStep = 0;
            xon._assignedFace = bestFace;
            xon._quarkType = bestType;
            xon._loopType = bestType ? LOOP_TYPE_NAMES[bestType] : null;
            xon.col = bestType ? QUARK_COLORS[bestType] : 0x888888;
            if (_flashEnabled) xon.flashT = 1.0;
            if (xon.sparkMat) xon.sparkMat.color.setHex(xon.col);
            return true;
        }
        return false;
    }

    if (tryFaces(actualizedFaces)) return true;

    // ── Pass 2: Manifest new tet voids ──
    // Try to materialise the missing SCs for non-actualized faces.
    // This creates new loiter space when the oct cage is congested.
    const newlyActualized = [];
    for (const face of manifestCandidates.sort(() => Math.random() - 0.5)) {
        const fd = _nucleusTetFaceData[face];
        const missingSCs = fd.scIds.filter(scId =>
            !xonImpliedSet.has(scId) && !activeSet.has(scId) && !impliedSet.has(scId));
        // Try to materialise all missing SCs
        let allOk = true;
        const justAdded = [];
        const xi = _demoXons.indexOf(xon);
        for (const scId of missingSCs) {
            if (canMaterialiseQuick(scId)) {
                xonImpliedSet.add(scId);
                _scAttribution.set(scId, { reason: 'manifest', xonIdx: xi, face, tick: _demoTick });
                stateVersion++; // invalidate cache for next check
                justAdded.push(scId);
            } else if (excitationSeverForRoom(scId)) {
                if (canMaterialiseQuick(scId)) {
                    xonImpliedSet.add(scId);
                    _scAttribution.set(scId, { reason: 'manifest', xonIdx: xi, face, tick: _demoTick });
                    stateVersion++; // invalidate cache
                    justAdded.push(scId);
                } else {
                    allOk = false; break;
                }
            } else {
                allOk = false; break;
            }
        }
        if (allOk) {
            newlyActualized.push(face);
            if (justAdded.length > 0) {
                _idleTetManifested = true;
                console.log(`[MANIFEST] Actualized tet face ${face} (${justAdded.length} new SCs) for idle loitering`);
            }
        } else {
            // Roll back partial materialisation
            for (const scId of justAdded) {
                xonImpliedSet.delete(scId);
                _scAttribution.delete(scId);
                stateVersion++; // invalidate cache
            }
        }
    }

    if (newlyActualized.length > 0) {
        const _idleLocked = _traversalLockedSCs();
        if (tryFaces(newlyActualized)) {
            // Rollback SCs for faces we manifested but didn't use
            const assignedFace = xon._assignedFace;
            for (const face of newlyActualized) {
                if (face === assignedFace) continue;
                const fd = _nucleusTetFaceData[face];
                for (const scId of fd.scIds) {
                    if (_idleLocked.has(scId)) continue; // xon traversing this SC
                    if (xonImpliedSet.has(scId) && !activeSet.has(scId) && !impliedSet.has(scId)) {
                        xonImpliedSet.delete(scId);
                        _scAttribution.delete(scId);
                        stateVersion++;
                    }
                }
            }
            return true;
        }
        // tryFaces failed — rollback ALL newly manifested SCs
        for (const face of newlyActualized) {
            const fd = _nucleusTetFaceData[face];
            for (const scId of fd.scIds) {
                if (_idleLocked.has(scId)) continue; // xon traversing this SC
                if (xonImpliedSet.has(scId) && !activeSet.has(scId) && !impliedSet.has(scId)) {
                    xonImpliedSet.delete(scId);
                    _scAttribution.delete(scId);
                    stateVersion++;
                }
            }
        }
    }

    // ── Fallback: use any blocked actualized face ──
    // (caller handles Pauli if this destination is occupied)
    if (actualizedFaces.length > 0) return tryFaces(actualizedFaces);
    return false;
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
let _demoUncappedId = null;  // setTimeout chain for uncapped mode
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
                [A[ancA_p[i]]]: 'pd',   // proton anchor = down
                [A[i]]: 'pu',            // proton follower-1 = up
                [A[dA_p[i]]]: 'pu',      // proton follower-2 = up
                [B[ancB_n[i]]]: 'nu',    // neutron anchor = up
                [B[i]]: 'nd',            // neutron follower-1 = down
                [B[dB_n[i]]]: 'nd',      // neutron follower-2 = down
            },
        });
        // Odd tick 2i+1: proton on B, neutron on A
        schedule.push({
            protonFaces: [B[ancB_p[i]], B[i], B[dB_p[i]]],
            neutronFaces: [A[ancA_n[i]], A[i], A[dA_n[i]]],
            faceQuarks: {
                [B[ancB_p[i]]]: 'pd',
                [B[i]]: 'pu',
                [B[dB_p[i]]]: 'pu',
                [A[ancA_n[i]]]: 'nu',
                [A[i]]: 'nd',
                [A[dA_n[i]]]: 'nd',
            },
        });
    }
    return schedule;
}

// ── L1-valid tet configurations (verified by solver) ──
const L1_VALID_TRIPLES = [
    [3, 5, 6], [1, 6, 7], [3, 5, 8], [1, 7, 8],  // 2A+1B
    [4, 5, 6], [4, 6, 7], [2, 5, 8], [2, 7, 8],  // 1A+2B
];
// ╔══════════════════════════════════════════════════════════════════════╗
// ║  NON-DELETABLE: MINIMAL ACTION PRINCIPLE                           ║
// ║                                                                    ║
// ║  "Relinquish if necessary for the desired transformation,          ║
// ║   otherwise do not change."                                        ║
// ║                                                                    ║
// ║  When switching tet configurations:                                ║
// ║  - Remove ONLY the SCs that need to go (old tets not in new set)  ║
// ║  - Add ONLY the SCs that are new (new tets not in old set)        ║
// ║  - Keep everything else UNCHANGED                                  ║
// ║  - Do NOT clear-and-rebuild from scratch                           ║
// ║  - Do NOT cascade-detect implied shortcuts during demo             ║
// ║    (cascade deforms FCC geometry → Kepler violation)               ║
// ║                                                                    ║
// ║  This is a physics principle, not an optimization.                 ║
// ╚══════════════════════════════════════════════════════════════════════╝

/**
 * Start the demand-driven demo: sets up lattice, runs high-speed loop.
 * Called AFTER simulateNucleus() has built the octahedron.
 * No schedule or windows — xons self-assign via _scoreFaceOpportunity.
 */
function startDemoLoop() {
    // Init visit counters (demand-driven — no schedule needed)
    _demoVisits = {};
    for (let f = 1; f <= 8; f++) {
        _demoVisits[f] = { pu: 0, pd: 0, nu: 0, nd: 0, total: 0 };
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

const QUARK_COLORS = { pu: 0xffdd44, pd: 0x44cc66, nu: 0x4488ff, nd: 0xff4444 };
const A_SET = new Set([1, 3, 6, 8]);

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
        // Remaining 2 xons → above center
        for (let i = 0; i < 2 && i < aboveY.length; i++) {
            const xon = _demoXons[4 + i];
            const pick = aboveY[i];
            _executeOctMove(xon, { node: pick.node, dirIdx: pick.dirIdx, _needsMaterialise: false, _scId: undefined });
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

        // ── Free xons: move AWAY from oct cage (T55: max 4 on oct nodes) ──
        const center1hop = new Set([center]);
        for (const nb of baseNeighbors[center]) center1hop.add(nb.node);
        for (const sc of _localScNeighbors(center)) center1hop.add(sc.a === center ? sc.b : sc.a);
        const freeXonData = [];
        const takenNodes = new Set(_octEquatorCycle);
        for (let i = 0; i < 6; i++) {
            if (equatorSet.has(xonNodes[i])) continue;
            const xon = _demoXons[i];
            // Exclude oct nodes — non-equatorial xons must NOT land on the oct cage
            const candidates = baseNeighbors[xon.node].filter(nb =>
                center1hop.has(nb.node) && !takenNodes.has(nb.node) &&
                !_octNodeSet.has(nb.node) && nb.node !== xon.prevNode
            );
            freeXonData.push({ xon, candidates });
        }
        if (freeXonData.length === 2 && freeXonData[0].candidates.length > 0 && freeXonData[1].candidates.length > 0) {
            let bestPair = null, bestDist = -Infinity;
            for (const c0 of freeXonData[0].candidates) {
                for (const c1 of freeXonData[1].candidates) {
                    if (c0.node === c1.node) continue;
                    const d = Math.hypot(
                        pos[c0.node][0] - pos[c1.node][0],
                        pos[c0.node][1] - pos[c1.node][1],
                        pos[c0.node][2] - pos[c1.node][2]
                    );
                    if (d > bestDist) { bestDist = d; bestPair = [c0, c1]; }
                }
            }
            if (bestPair) {
                _executeOctMove(freeXonData[0].xon, { node: bestPair[0].node, dirIdx: bestPair[0].dirIdx, _needsMaterialise: false, _scId: undefined });
                takenNodes.add(bestPair[0].node);
                _executeOctMove(freeXonData[1].xon, { node: bestPair[1].node, dirIdx: bestPair[1].dirIdx, _needsMaterialise: false, _scId: undefined });
            }
        } else {
            for (const { xon, candidates } of freeXonData) {
                if (candidates.length > 0) {
                    const pick = candidates[0];
                    _executeOctMove(xon, { node: pick.node, dirIdx: pick.dirIdx, _needsMaterialise: false, _scId: undefined });
                    takenNodes.add(pick.node);
                }
            }
        }

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
                xon._mayReturn = true;
                xon.col = WEAK_FORCE_COLOR;
                if (xon.sparkMat) xon.sparkMat.color.setHex(WEAK_FORCE_COLOR);
            }
        }
    }
}

let _tickInProgress = false; // guard against overlapping async ticks
// ─── Profiling ───
let _tickTotalMs = 0, _tickCount = 0, _tickMaxMs = 0;
let _profPhases = { wb: 0, p0: 0, p05: 0, p1: 0, p2: 0, p3: 0, p3b: 0, p4: 0, p5: 0, solver: 0, cleanup: 0, render: 0, guards: 0 };
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

    // Clear stale movement flags from previous tick so WB processing isn't blocked
    for (const xon of _demoXons) { xon._movedThisTick = false; xon._evictedThisTick = false; }
    // Revert gluon xons to oct (fresh evaluation each tick per spec §2)
    for (const xon of _demoXons) {
        if (xon.alive && xon._mode === 'gluon') {
            xon._mode = 'oct';
            xon._mayReturn = false;
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
            _skipNormalPhases = true;
            _solverNeeded = true;
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
            xon._mayReturn = false;
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
                    xon._t60Ejected = true; // must reach ejection target before returning
                    xon._mayReturn = false;
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
                    blocker._mayReturn = false;
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

            // Check 2: Loop-shape-aware lookahead — will this specific loop lead to a dead end?
            // Uses the xon's actual loop sequence (fork, lollipop, ham CW/CCW)
            // instead of generic graph traversal.
            if (!shouldEvictSelf && !(occupied.get(nextNode) || 0)) {
                const tmpOcc = new Map(occupied);
                _occDel(tmpOcc, xon.node);
                _occAdd(tmpOcc, nextNode);
                if (!_lookaheadTetPath(xon._loopSeq, effectiveStep + 1, tmpOcc, _choreoParams.lookahead, xon)) shouldEvictSelf = true;
            }

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
                xon._mayReturn = false;
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
    // Handles ALL weak xon movement (including _t60Ejected).
    // Pre-_mayReturn: move to ejection-space nodes only (_isValidEjectionTarget).
    //   After 1+ valid ejection moves, flip _mayReturn = true.
    // Post-_mayReturn: BFS toward nearest oct node. May enter oct nodes and
    //   ejection-space nodes. May NOT enter _purelyTetNodes.
    //   On arrival at oct node → transition to oct mode.
    for (const xon of _demoXons) {
        if (!xon.alive || xon._mode !== 'weak') continue;

        // ── Pre-_mayReturn: ejected xons must reach ejection space first ──
        if (xon._t60Ejected && !xon._mayReturn) {
            // Already on ejection space node? Flip _mayReturn and proceed to BFS return below.
            if (_isValidEjectionTarget(xon.node)) {
                xon._mayReturn = true;
                // Fall through to post-_mayReturn BFS below
            } else {
                // Must move to an ejection-space node
                // Use full baseNeighbors — weak xons have freedom to roam the full lattice
                const allNbs = baseNeighbors[xon.node] || [];
                const recentTrail = new Set(xon.trail ? xon.trail.slice(-6) : []);
                // Tier 1: ejection target, guard-safe, not in trail, not prevNode
                let target = allNbs.find(nb => !(occupied.get(nb.node) || 0) &&
                    _isValidEjectionTarget(nb.node) &&
                    !recentTrail.has(nb.node) && nb.node !== xon.prevNode &&
                    !_swapBlocked(xon.node, nb.node) &&
                    !_moveViolatesGuards(xon, xon.node, nb.node));
                // Tier 2: ejection target, guard-safe, not prevNode
                if (!target) {
                    target = allNbs.find(nb => !(occupied.get(nb.node) || 0) &&
                        _isValidEjectionTarget(nb.node) &&
                        nb.node !== xon.prevNode &&
                        !_swapBlocked(xon.node, nb.node) &&
                        !_moveViolatesGuards(xon, xon.node, nb.node));
                }
                // Tier 3: ejection target, guard-safe, allow prevNode
                if (!target) {
                    target = allNbs.find(nb => !(occupied.get(nb.node) || 0) &&
                        _isValidEjectionTarget(nb.node) &&
                        !_swapBlocked(xon.node, nb.node) &&
                        !_moveViolatesGuards(xon, xon.node, nb.node));
                }
                if (target) {
                    const fromWk = xon.node;
                    _occDel(occupied, xon.node);
                    xon.prevNode = xon.node;
                    xon.node = target.node;
                    _occAdd(occupied, target.node);
                    xon._movedThisTick = true;
                    _moveRecord.set(target.node, fromWk);
                    _traceMove(xon, fromWk, target.node, 'weakEjection');
                    _trailPush(xon, target.node, WEAK_FORCE_COLOR);
                    xon.tweenT = 0;
                    anyMoved = true;
                    _weakLifecycleStep(xon);
                    // Arrived at ejection space — flip _mayReturn
                    xon._mayReturn = true;
                }
                continue; // done for this xon this tick (whether moved or not)
            }
        }

        // ── Post-_mayReturn (or non-ejected weak): return to oct cage ──
        // Weak xon with _mayReturn already at oct node → becomes oct immediately (no capacity gate).
        // The choreographer is responsible for not over-populating the oct cage.
        if (xon._mayReturn && _octNodeSet.has(xon.node)) {
            _weakLifecycleExit(xon, 'arrived_oct_immediate');
            _clearModeProps(xon);
            xon._mode = 'oct';
            if (_flashEnabled) xon.flashT = 1.0;
            xon.col = 0xffffff;
            if (xon.sparkMat) xon.sparkMat.color.setHex(0xffffff);
            continue;
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
                // Post-_mayReturn: may NOT enter actualized tet nodes (active geometry)
                if (_purelyTetNodes && _purelyTetNodes.has(nb.node)) continue;
                visited.add(nb.node);
                const nextStep = step || nb.node;
                if (_octNodeSet.has(nb.node)) {
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
            _occDel(occupied, xon.node);
            xon.prevNode = xon.node;
            xon.node = bestStep;
            _occAdd(occupied, bestStep);
            xon._movedThisTick = true;
            _moveRecord.set(bestStep, fromWk);
            _traceMove(xon, fromWk, bestStep, 'weakBFS');
            _trailPush(xon, bestStep, WEAK_FORCE_COLOR);
            xon.tweenT = 0;
            anyMoved = true;
            _weakLifecycleStep(xon);
            // Check if arrived at oct node — transition if _mayReturn and capacity
            if (_octNodeSet.has(bestStep) && xon._mayReturn) {
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
            // All BFS steps blocked — try free neighbor (avoiding actualized tet nodes)
            // Use full baseNeighbors (not _localBaseNeighbors which restricts to nucleus)
            // — weak xons have freedom to roam the full lattice.
            const allNbs = baseNeighbors[xon.node] || [];
            const purelyTetFilter = nb => !(_purelyTetNodes && _purelyTetNodes.has(nb.node));
            // Tier 1: guard-safe, not purelyTet, not in recent trail, not prevNode
            let freeNb = allNbs.find(nb => !(occupied.get(nb.node) || 0) &&
                purelyTetFilter(nb) &&
                !recentTrail.has(nb.node) && nb.node !== xon.prevNode &&
                !_swapBlocked(xon.node, nb.node) &&
                !_moveViolatesGuards(xon, xon.node, nb.node));
            // Tier 2: guard-safe, not purelyTet, not prevNode
            if (!freeNb) {
                freeNb = allNbs.find(nb => !(occupied.get(nb.node) || 0) &&
                    purelyTetFilter(nb) &&
                    nb.node !== xon.prevNode &&
                    !_swapBlocked(xon.node, nb.node) &&
                    !_moveViolatesGuards(xon, xon.node, nb.node));
            }
            // Tier 3: guard-safe, not purelyTet, allow prevNode
            if (!freeNb) {
                freeNb = allNbs.find(nb => !(occupied.get(nb.node) || 0) &&
                    purelyTetFilter(nb) &&
                    !_swapBlocked(xon.node, nb.node) &&
                    !_moveViolatesGuards(xon, xon.node, nb.node));
            }
            // No guard bypass — if no guard-safe move exists, xon stays put
            if (freeNb) {
                const fromWk2 = xon.node;
                _occDel(occupied, xon.node);
                xon.prevNode = xon.node;
                xon.node = freeNb.node;
                _occAdd(occupied, freeNb.node);
                xon._movedThisTick = true;
                _moveRecord.set(freeNb.node, fromWk2);
                _traceMove(xon, fromWk2, freeNb.node, 'weakDetour');
                _trailPush(xon, freeNb.node, WEAK_FORCE_COLOR);
                xon.tweenT = 0;
                anyMoved = true;
                _weakLifecycleStep(xon);
                if (_octNodeSet.has(freeNb.node) && xon._mayReturn) {
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
                xon._t60Ejected = true; // must reach ejection target before returning
                xon._mayReturn = false;
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
                const j = Math.floor(Math.random() * (i + 1));
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

    // ── OCT CAPACITY OVERFLOW — 2-tier relief ──
    // If more than OCT_CAPACITY_MAX xons are in oct mode, shed the excess.
    // Tier 1: _startIdleTetLoop (productive — manifests a hadron).
    // Tier 2: Eject as weak particle with _t60Ejected = true.
    {
        const octModeXons = _demoXons.filter(x => x.alive && x._mode === 'oct' && !x._movedThisTick && !x._evictedThisTick);
        let excess = octModeXons.length - OCT_CAPACITY_MAX;
        if (excess > 0) {
            // Shuffle to avoid order bias
            const candidates = octModeXons.slice().sort(() => Math.random() - 0.5);
            for (const xon of candidates) {
                if (excess <= 0) break;
                // Tier 1: try idle_tet diversion
                if (_startIdleTetLoop(xon, occupied)) {
                    _logChoreo(`X${_demoXons.indexOf(xon)} oct overflow -> idle_tet f${xon._assignedFace}`);
                    _solverNeeded = true;
                    excess--;
                    continue;
                }
                // Tier 2: eject as weak particle
                _logChoreo(`X${_demoXons.indexOf(xon)} oct overflow -> weak (no idle_tet available)`);
                xon._mode = 'weak';
                xon._t60Ejected = true;
                xon._mayReturn = false;
                xon.col = WEAK_FORCE_COLOR;
                if (xon.sparkMat) xon.sparkMat.color.setHex(WEAK_FORCE_COLOR);
                _weakLifecycleEnter(xon, 'oct_capacity_overflow');
                excess--;
            }
            if (excess <= 0) occupied = _occupiedNodes(); // refresh after overflow relief
        }
    }

    // ── PHASE 2: Coordinated oct movement planning ──
    let octXons = [];
    let octPlans = [];
    {
    octXons = _demoXons.filter(x => x.alive && (x._mode === 'oct' || x._mode === 'gluon' || (x._mode === 'weak' && !x._mayReturn)) && !x._movedThisTick);
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

    // Maximum bipartite matching with arbitrary-depth backtracking (Kuhn's algorithm).
    // Finds augmenting paths so the maximum number of oct xons get a valid destination.
    _maxBipartiteAssignment(octPlans, planned);
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
            plan.xon._mayReturn = false;
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
                // T60: check if ejected weak xon reached ejection target → flip _mayReturn
                if (plan.xon._t60Ejected && !plan.xon._mayReturn &&
                    _isValidEjectionTarget(plan.xon.node)) {
                    plan.xon._mayReturn = true;
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

    // Escape hatch: Removed — backtracker handles stuck tet xons via rewind.
    // If a tet plan was approved and vacuum-negotiated, it should succeed.
    // If it doesn't (vacuum changed), trigger backtrack.
    for (const plan of tetPlans) {
        if (!plan.approved) continue;
        if (plan.xon.node !== plan.fromNode) continue; // already moved — success
        if (plan.xon._movedThisTick) continue; // moved by another path
        // Tet move was approved but couldn't execute — trigger backtrack
        _rewindRequested = true;
        _rewindViolation = `PHASE 3 tet stuck: xon at ${plan.fromNode} couldn't reach ${plan.toNode}`;
    }

    const _pT3 = performance.now(); _profPhases.p3 += _pT3 - _pT2;
    _pT5 = _pT3; // PHASE 3b/4/5 removed — bridge profiling timer to solver

    // PHASE 3b: Removed — backtracker handles stuck oct xons via rewind

    // PHASE 3.5: Removed — PHASE 0.5 handles all weak xon movement

    // ── POST-MOVE WEAK→OCT TRANSITION SWEEP ──
    // Any weak xon with _mayReturn=true that landed on an oct node (via PHASE 2
    // bipartite matching or any other phase) becomes oct immediately.
    // This ensures the user's directive: "weak xons become type oct on the turn
    // that they land on the oct, since it has successfully returned."
    for (const xon of _demoXons) {
        if (!xon.alive) continue;
        if (xon._mode !== 'weak') continue;
        if (!xon._mayReturn) continue;
        if (!_octNodeSet || !_octNodeSet.has(xon.node)) continue;
        _weakLifecycleExit(xon, 'post_move_oct_arrival');
        _clearModeProps(xon);
        xon._mode = 'oct';
        if (_flashEnabled) xon.flashT = 1.0;
        xon.col = 0xffffff;
        if (xon.sparkMat) xon.sparkMat.color.setHex(0xffffff);
    }

    // ── POST-EXECUTION PAULI CHECK (replaces PHASE 4) ──
    // If planning was correct, no collisions exist. If one slipped through, trigger backtrack.
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

function dumpProfile() {
    if (_tickCount === 0) { console.log('[PROFILE] No ticks recorded'); return; }
    const n = _tickCount;
    const total = _tickTotalMs;
    const ph = _profPhases;
    const phaseTot = ph.wb + ph.p0 + ph.p05 + ph.p1 + ph.p2 + ph.p3 + ph.solver + ph.render + ph.guards + (ph._temporalK||0) + (ph._uiUpdate||0);
    const pct = (v) => ((v / total) * 100).toFixed(1) + '%';
    const avg = (v) => (v / n).toFixed(1);
    console.log(`\n[PROFILE] ${n} ticks, total ${(total/1000).toFixed(1)}s, avg ${avg(total)}ms/tick, max ${_tickMaxMs.toFixed(0)}ms`);
    console.log(`  WB(setup):  ${avg(ph.wb)}ms/tick  ${pct(ph.wb)}`);
    console.log(`  PHASE 0:    ${avg(ph.p0)}ms/tick  ${pct(ph.p0)}`);
    console.log(`  PHASE 0.5:  ${avg(ph.p05)}ms/tick  ${pct(ph.p05)}`);
    console.log(`  PHASE 1:    ${avg(ph.p1)}ms/tick  ${pct(ph.p1)}`);
    console.log(`  PHASE 2:    ${avg(ph.p2)}ms/tick  ${pct(ph.p2)}  (gpuBatch: ${avg(ph.gpuBatch||0)}ms)`);
    console.log(`  PHASE 3:    ${avg(ph.p3)}ms/tick  ${pct(ph.p3)}`);
    // PHASE 3b/4/5: Removed — backtracker is the universal safety net
    console.log(`  Solver+glu: ${avg(ph.solver)}ms/tick  ${pct(ph.solver)}`);
    console.log(`  Render:     ${avg(ph.render)}ms/tick  ${pct(ph.render)}`);
    console.log(`  Guards:     ${avg(ph.guards)}ms/tick  ${pct(ph.guards)}`);
    console.log(`  TemporalK:  ${avg(ph._temporalK||0)}ms/tick  ${pct(ph._temporalK||0)}`);
    console.log(`  UI update:  ${avg(ph._uiUpdate||0)}ms/tick  ${pct(ph._uiUpdate||0)}`);
    console.log(`  Accounted:  ${avg(phaseTot)}ms/tick  ${pct(phaseTot)}`);
    if (typeof _solveCallCount !== 'undefined') {
        console.log(`  _solve() calls: ${_solveCallCount} (${(_solveCallCount/n).toFixed(1)}/tick), avg ${_solveCallCount?(_solveTotalMs/_solveCallCount).toFixed(1):'0'}ms`);
    }
    if (typeof _cmqCallCount !== 'undefined') {
        console.log(`  CMQ: ${_cmqCallCount} calls, ${_cmqCpuCount} CPU, ${_cmqCacheHits} cached, avg ${_cmqCpuCount?(_cmqTotalMs/_cmqCpuCount).toFixed(1):'0'}ms/CPU`);
    }
}

function resetProfile() {
    _tickTotalMs = 0; _tickCount = 0; _tickMaxMs = 0;
    for (const k in _profPhases) _profPhases[k] = 0;
    if (typeof _solveCallCount !== 'undefined') { _solveCallCount = 0; _solveTotalMs = 0; _solveMaxMs = 0; _solveIterTotal = 0; }
    if (typeof _cmqCallCount !== 'undefined') { _cmqCallCount = 0; _cmqCpuCount = 0; _cmqCacheHits = 0; _cmqTotalMs = 0; }
    console.log('[PROFILE] Counters reset');
}

function updateDemoPanel() {
    // Demand-driven: use fixed 64-tick epoch for display purposes
    const epoch = Math.floor(_demoTick / 64);

    // ── Update demo-status (right panel, below button) ──
    const ds = document.getElementById('demo-status');
    if (ds) {
        ds.innerHTML = `<span style="color:#88bbdd;">epoch ${epoch}</span>`;
    }

    // ── Update left panel coverage bars (skip during test execution) ──
    if (_testRunning) return;
    const el = document.getElementById('dp-coverage-bars');
    if (!el) return;

    // Compute evenness: CV across all faces' total visits
    const totals = [];
    for (let f = 1; f <= 8; f++) totals.push(_demoVisits[f].total);
    const mean = totals.reduce((a, b) => a + b, 0) / totals.length;
    const stddev = Math.sqrt(totals.reduce((s, v) => s + (v - mean) ** 2, 0) / totals.length);
    const cv = mean > 0 ? (stddev / mean) : 0;
    const evenness = Math.max(0, 1 - cv);

    // Find max for bar normalization
    let maxCount = 1;
    for (let f = 1; f <= 8; f++) {
        for (const t of ['pu', 'pd', 'nu', 'nd']) {
            maxCount = Math.max(maxCount, _demoVisits[f][t]);
        }
    }

    // Build bars
    let html = '';
    for (let f = 1; f <= 8; f++) {
        const v = _demoVisits[f];
        const isA = [1, 3, 6, 8].includes(f);
        html += `<div style="display:flex; align-items:center; gap:2px;">`
            + `<span style="width:18px; color:${isA ? '#cc8866' : '#6688aa'}; font-size:8px; font-weight:bold;">F${f}</span>`
            + `<div class="dp-bar-bg" style="flex:1;" title="p\u2191 ${v.pu}"><div class="dp-bar-fill" style="width:${(v.pu / maxCount * 100).toFixed(1)}%; background:#ddcc44;"></div></div>`
            + `<div class="dp-bar-bg" style="flex:1;" title="p\u2193 ${v.pd}"><div class="dp-bar-fill" style="width:${(v.pd / maxCount * 100).toFixed(1)}%; background:#44cc66;"></div></div>`
            + `<div class="dp-bar-bg" style="flex:1;" title="n\u2191 ${v.nu}"><div class="dp-bar-fill" style="width:${(v.nu / maxCount * 100).toFixed(1)}%; background:#4488ff;"></div></div>`
            + `<div class="dp-bar-bg" style="flex:1;" title="n\u2193 ${v.nd}"><div class="dp-bar-fill" style="width:${(v.nd / maxCount * 100).toFixed(1)}%; background:#ff4444;"></div></div>`
            + `<span style="width:22px; text-align:right; font-size:7px; color:#667788;">${v.total}</span>`
            + `</div>`;
    }

    // ── Per-hadron evenness ──
    // Proton visits per face = pu + pd, Neutron = nu + nd
    const protonPerFace = [], neutronPerFace = [], typePerFace = [];
    for (let f = 1; f <= 8; f++) {
        const v = _demoVisits[f];
        protonPerFace.push(v.pu + v.pd);
        neutronPerFace.push(v.nu + v.nd);
    }
    // Per-type global totals
    // Physical ratio: pu ≈ 2×pd (proton uud), nd ≈ 2×nu (neutron udd)
    const typeTotals = { pu: 0, pd: 0, nu: 0, nd: 0 };
    for (let f = 1; f <= 8; f++) {
        for (const t of ['pu', 'pd', 'nu', 'nd']) typeTotals[t] += _demoVisits[f][t];
    }
    const calcEvenness = (arr) => {
        const m = arr.reduce((a, b) => a + b, 0) / arr.length;
        if (m === 0) return 0; // no visits = 0% balance
        const sd = Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length);
        return Math.max(0, 1 - sd / m);
    };
    const protonEvenness = calcEvenness(protonPerFace);
    const neutronEvenness = calcEvenness(neutronPerFace);
    // Type ratio balance: check pu:pd ≈ 2:1 and nd:nu ≈ 2:1
    const puPdRatio = typeTotals.pd > 0 ? typeTotals.pu / typeTotals.pd : 0;
    const ndNuRatio = typeTotals.nu > 0 ? typeTotals.nd / typeTotals.nu : 0;
    // How close each ratio is to the target 2.0
    const ratioAccuracy = (puPdRatio > 0 && ndNuRatio > 0)
        ? 1 - (Math.abs(puPdRatio - 2) + Math.abs(ndNuRatio - 2)) / 4
        : 0;
    const evColor = (e) => e > 0.99 ? '#66dd66' : e > 0.95 ? '#ccaa66' : '#ff6644';

    // Evenness + rule compliance
    html += `<div style="margin-top:6px; border-top:1px solid rgba(80,100,120,0.25); padding-top:4px;">`;
    html += `<div style="display:flex; justify-content:space-between; font-size:9px;">`
        + `<span style="color:#6a8a9a;">overall</span>`
        + `<span style="color:${evColor(evenness)}; font-weight:bold;">${(evenness * 100).toFixed(1)}%</span>`
        + `</div>`;
    html += `<div style="display:flex; justify-content:space-between; font-size:9px;">`
        + `<span style="color:#cc8866;">proton</span>`
        + `<span style="color:${evColor(protonEvenness)}; font-weight:bold;">${(protonEvenness * 100).toFixed(1)}%</span>`
        + `</div>`;
    html += `<div style="display:flex; justify-content:space-between; font-size:9px;">`
        + `<span style="color:#6688aa;">neutron</span>`
        + `<span style="color:${evColor(neutronEvenness)}; font-weight:bold;">${(neutronEvenness * 100).toFixed(1)}%</span>`
        + `</div>`;
    html += `<div style="display:flex; justify-content:space-between; font-size:9px;">`
        + `<span style="color:#6a8a9a;">pu:pd ratio</span>`
        + `<span style="color:${Math.abs(puPdRatio - 2) < 0.3 ? '#66dd66' : '#ccaa66'}; font-weight:bold;">${puPdRatio.toFixed(2)} (\u21922.0)</span>`
        + `</div>`;
    html += `<div style="display:flex; justify-content:space-between; font-size:9px;">`
        + `<span style="color:#6a8a9a;">nd:nu ratio</span>`
        + `<span style="color:${Math.abs(ndNuRatio - 2) < 0.3 ? '#66dd66' : '#ccaa66'}; font-weight:bold;">${ndNuRatio.toFixed(2)} (\u21922.0)</span>`
        + `</div>`;
    html += `<div style="display:flex; justify-content:space-between; font-size:9px;">`
        + `<span style="color:#6a8a9a;">epoch</span>`
        + `<span style="color:#88aacc;">${epoch}</span>`
        + `</div>`;

    // ── Ratio accuracy history sparkline ──
    _demoTypeBalanceHistory.push(ratioAccuracy * 100);
    const hist = _demoTypeBalanceHistory;
    const sparkLen = Math.min(hist.length, 24);  // show last 24 cycles
    const sparkData = hist.slice(-sparkLen);
    // Scale: map [min..100] to 8-level block chars
    const sparkMin = Math.min(...sparkData, 90);
    const sparkMax = 100;
    const SPARK = ['\u2581', '\u2582', '\u2583', '\u2584', '\u2585', '\u2586', '\u2587', '\u2588'];
    let sparkline = '';
    for (const v of sparkData) {
        const norm = Math.max(0, Math.min(1, (v - sparkMin) / (sparkMax - sparkMin)));
        const idx = Math.min(7, Math.floor(norm * 7.99));
        // Color: green at 100, yellow at 95, orange below
        const c = v >= 99.5 ? '#66dd66' : v >= 96 ? '#ccaa66' : '#cc8855';
        sparkline += `<span style="color:${c};">${SPARK[idx]}</span>`;
    }
    html += `<div style="margin-top:4px; overflow:hidden;">`
        + `<div style="font-size:7px; color:#556677; margin-bottom:1px;">ratio accuracy (last ${sparkLen} windows)</div>`
        + `<div style="font-size:10px; letter-spacing:-1px; line-height:1; font-family:monospace; overflow:hidden;">${sparkline}</div>`
        + `<div style="display:flex; justify-content:space-between; font-size:6px; color:#445566; margin-top:1px;">`
        + `<span>${sparkMin.toFixed(0)}%</span><span>100%</span></div>`
        + `</div>`;
    // Rule compliance indicators
    const rules = [
        { name: 'anti-phase', ok: true },  // guaranteed by schedule construction
        { name: 'pauli', ok: _demoPauliViolations === 0 },
        { name: 'spread', ok: _demoSpreadViolations === 0 },
        { name: 'coverage', ok: evenness > 0.9 },
    ];
    html += `<div style="margin-top:3px; font-size:8px; color:#556677;">`;
    for (const r of rules) {
        html += `<span style="color:${r.ok ? '#44aa66' : '#cc4444'}; margin-right:6px;">${r.ok ? '\u2713' : '\u2717'} ${r.name}</span>`;
    }
    html += `</div></div>`;
    el.innerHTML = html;

    // Hide density/sync rows during demo (not relevant)
    const densityRow = document.querySelector('#deuteron-panel > div:nth-child(2)');
    const syncRow = document.querySelector('#deuteron-panel > div:nth-child(3)');
    if (densityRow) densityRow.style.display = 'none';
    if (syncRow) syncRow.style.display = 'none';
}

// ── Choreographer logging helper ──
function _logChoreo(msg) {
    _choreoLog.push({ tick: _demoTick, msg });
    if (_choreoLog.length > _CHOREO_LOG_MAX) _choreoLog.shift();
}

// ── Log PHASE 2 summary: called after bipartite matching + fallbacks ──
function _logPhase2Summary(octPlans) {
    const _QL = { pu: 'p_u', pd: 'p_d', nu: 'n_u', nd: 'n_d' };
    const lines = [];
    for (const plan of octPlans) {
        const x = plan.xon;
        const idx = _demoXons.indexOf(x);
        const label = x._mode === 'oct' ? 'idle' : x._mode === 'weak' ? 'weak' :
                      x._quarkType ? _QL[x._quarkType] || x._quarkType : x._mode;
        const cands = plan.candidates.length;
        if (plan.assigned) {
            lines.push(`X${idx}(${label}) n${plan.fromNode}: ${cands}c->n${plan.assigned.node}`);
        } else if (plan.idleTet) {
            lines.push(`X${idx}(${label}) n${plan.fromNode}: ${cands}c->tet f${x._assignedFace}`);
        } else {
            const reasons = [];
            if (cands === 0) reasons.push('0 cands');
            else reasons.push(`${cands}c taken`);
            if (x._evictedThisTick) reasons.push('evicted');
            lines.push(`X${idx}(${label}) n${plan.fromNode}: STUCK(${reasons.join(',')})`);
        }
    }
    _logChoreo('PH2: ' + lines.join(' | '));
}

// ── Xon panel update (sidebar) ──
function updateXonPanel() {
    if (_testRunning) return;
    const panel = document.getElementById('xon-panel');
    if (!panel) return;
    panel.style.display = _demoActive ? 'block' : 'none';
    if (!_demoActive) return;

    const listEl = document.getElementById('xon-panel-list');
    if (!listEl) return;

    let html = '';
    for (let i = 0; i < _demoXons.length; i++) {
        const x = _demoXons[i];
        if (!x.alive) continue;
        const modeCol = x._mode === 'oct' ? '#ffffff' :
                        x._mode === 'weak' ? '#cc44ff' :
                        x._mode === 'tet' ? '#' + (x.col || 0xffffff).toString(16).padStart(6, '0') :
                        x._mode === 'idle_tet' ? '#' + (x.col || 0x888888).toString(16).padStart(6, '0') : '#888888';
        // Display labels: oct=idle, tet/idle_tet=hadron type (p_u, p_d, n_u, n_d)
        const QUARK_LABELS = { pu: 'p_u', pd: 'p_d', nu: 'n_u', nd: 'n_d' };
        let modeLabel, faceStr;
        if (x._mode === 'oct') {
            modeLabel = 'idle';
            faceStr = '';
        } else if (x._mode === 'weak') {
            modeLabel = 'weak';
            faceStr = '';
        } else if (x._mode === 'oct_formation') {
            modeLabel = 'form';
            faceStr = '';
        } else {
            // tet or idle_tet — show hadron type
            modeLabel = x._quarkType ? QUARK_LABELS[x._quarkType] || x._quarkType : x._mode;
            faceStr = x._assignedFace ? ` f${x._assignedFace}` : '';
        }
        const highlighted = _xonHighlightTimers.has(i);
        const border = highlighted ? `2px solid ${modeCol}` : '1px solid #334455';
        const bg = highlighted ? 'rgba(255,255,255,0.15)' : '#0d1520';
        html += `<button class="xon-btn" data-xon-idx="${i}" style="display:flex; flex-direction:column; align-items:center; justify-content:center; width:42px; height:36px; padding:2px; cursor:pointer; border-radius:4px; background:${bg}; border:${border}; font-family:monospace; outline:none;" title="X${i}: n${x.node} ${modeLabel}${faceStr}">`
            + `<span style="color:${modeCol}; font-weight:bold; font-size:11px;">X${i}</span>`
            + `<span style="color:#88aacc; font-size:8px;">n${x.node}</span>`
            + `<span style="color:#667788; font-size:7px;">${modeLabel}${faceStr}</span>`
            + `</button>`;
    }
    listEl.innerHTML = html;

    // Click delegation: attach ONE handler to parent (survives innerHTML rebuilds).
    // Guard against duplicate attachment with a flag.
    if (!listEl._xonDelegated) {
        listEl._xonDelegated = true;
        listEl.addEventListener('mousedown', (e) => {
            // Find closest .xon-btn ancestor of the click target
            const btn = e.target.closest('.xon-btn');
            if (!btn) return;
            const idx = parseInt(btn.dataset.xonIdx, 10);
            if (!isNaN(idx)) _highlightXon(idx);
        });
    }
}

function _highlightXon(idx) {
    if (idx < 0 || idx >= _demoXons.length) return;
    const xon = _demoXons[idx];
    if (!xon) return;

    // Set highlight timer — decayed per-frame in _tickDemoXons
    xon._highlightT = 2.0; // seconds

    // Track which xons are highlighted for button border styling
    if (_xonHighlightTimers.has(idx)) clearTimeout(_xonHighlightTimers.get(idx));
    _xonHighlightTimers.set(idx, setTimeout(() => _xonHighlightTimers.delete(idx), 2000));
}

function pauseDemo() {
    _demoPaused = true;
    if (_demoInterval) { clearInterval(_demoInterval); _demoInterval = null; }
    if (_demoUncappedId) { clearTimeout(_demoUncappedId); _demoUncappedId = null; }
}
function resumeDemo() {
    _demoPaused = false;
    if (_demoActive && !_demoInterval && !_demoUncappedId) {
        const intervalMs = _getDemoIntervalMs();
        if (intervalMs === 0) {
            _demoUncappedId = setTimeout(_demoUncappedLoop, 0);
        } else {
            _demoInterval = setInterval(demoTick, intervalMs);
        }
    }
}
function isDemoPaused() {
    return _demoPaused;
}

function stopDemo() {
    _demoActive = false;
    _demoPaused = false;
    _openingPhase = false;
    if (typeof _liveGuardsActive !== 'undefined') _liveGuardsActive = false;
    if (_demoInterval) { clearInterval(_demoInterval); _demoInterval = null; }
    if (_demoUncappedId) { clearTimeout(_demoUncappedId); _demoUncappedId = null; }
    const ds = document.getElementById('demo-status');
    if (ds) ds.style.display = 'none';
    // Clean up Demo 3.0 xons and gluons
    _cleanupDemo3();
    // Clean up tet SCs from xonImpliedSet + oct SCs from activeSet
    for (const [, fd] of Object.entries(_nucleusTetFaceData)) {
        for (const scId of fd.scIds) {
            xonImpliedSet.delete(scId);
            _scAttribution.delete(scId);
        }
    }
    _scAttribution.clear(); // full cleanup on demo stop
    for (const scId of _octSCIds) {
        activeSet.delete(scId);
    }
    // Clear tet annotations
    _ruleAnnotations.tetColors.clear();
    _ruleAnnotations.tetOpacity.clear();
    _ruleAnnotations.dirty = true;
    bumpState();
    const pClean = detectImplied();
    applyPositions(pClean);
    updateSpheres();
    // Show xon sparks again
    const quarks = NucleusSimulator?.quarkExcitations || [];
    for (const q of quarks) {
        if (q.spark) q.spark.visible = true;
        if (q.trailLine) q.trailLine.visible = true;
    }
    // Restore density/sync rows
    const densityRow = document.querySelector('#deuteron-panel > div:nth-child(2)');
    const syncRow = document.querySelector('#deuteron-panel > div:nth-child(3)');
    if (densityRow) densityRow.style.display = '';
    if (syncRow) syncRow.style.display = '';
    // Restore panel title
    const dpTitle = document.querySelector('#deuteron-panel > div:first-child');
    if (dpTitle) dpTitle.textContent = 'DEUTERON';
}

// ── Precomputed pattern schedule for algos ──
let _activePatternSchedule = null;

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
const QUARK_ALGO_REGISTRY = [];
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
        chosen = freeOpts[Math.floor(Math.random() * freeOpts.length)];
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
        if (Math.random() >= prob) return null;

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
        let r = Math.random() * wTotal;
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

        if (Math.random() >= 0.35) return null;
        return { targetFace: bestFace };
    }
});
