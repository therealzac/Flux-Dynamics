// flux-demo.js — Orchestrator: design intent, rendering (_tickDemoXons), startDemoLoop, lattice
// Helper functions, phases, opening, and algorithms split into:
//   flux-demo-opening.js  — 2-tick opening choreography
//   flux-demo-ca.js       — dir balance, locality, patterns, algorithm registry
//   flux-demo-phases.js   — phase helpers, gluon system, demoTick() main loop
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
            g.ok = null;
            g.msg = 'grace period';
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


