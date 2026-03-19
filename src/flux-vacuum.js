// flux-vacuum.js — Vacuum negotiation, excitation materialisation, strain checks, xon pathfinding, Big Bang
// ─── Excitation system ──────────────────────────────────────────────────────────
const XON_COLORS=[0xffee44,0x44ffcc,0xff44aa,0x44aaff,0xffaa44,0xaa44ff];
const XON_COLORS_CSS=['#ffee44','#44ffcc','#ff44aa','#44aaff','#ffaa44','#aa44ff'];
let excitations=[], xonNextId=0, placingExcitation=false;
let _deferUIUpdates = false, _uiDirty = false; // batch UI updates during tick
const XON_ALPHA=3.0, TRAIL_LENGTH=24;
let XON_STEP_MS=0;  // 0 = uncapped (MAX speed), matches slider default of 100
let _severanceCount = 0;

// scPairToId hoisted to early block (line ~315) to avoid TDZ in computeVoidNeighbors
function rebuildScPairLookup(){ scPairToId = new Map(); ALL_SC.forEach(s=>{ scPairToId.set(pairId(s.a, s.b), s.id); }); }

// ─── excitationInduceShortcut — with post-induction rollback ────────────────
// WHY ROLLBACK EXISTS:
//   shortcutCompatible() is a greedy one-at-a-time check. With multiple
//   excitations firing in the same interval, each new SC passes because it's
//   checked against the set at the moment it was queried. But the combined set
//   of all 7–9 simultaneous implied SCs can be overconstrained — the PBD solver
//   drifts, base edges stretch, updateStatus() detects R1 and HALTS THE SIM.
//
//   The fix: after adding the SC and running the solver, measure the actual
//   worst base-edge error. If it exceeds tolerance, roll back immediately:
//   remove the SC from all sets, revert positions, and return. This is far
//   better than letting the violation accumulate and halt the sim.
//
// DO NOT remove the rollback block. It is the only thing that prevents the
// multi-excitation overconstrained-set halt that has recurred multiple times.
// ─────────────────────────────────────────────────────────────────────────────
// ─── Excitation SC materialisation ────────────────────────────────────────────
// Takes an explicit scId. Adds to xonImpliedSet, runs solver, rolls back
// if overconstrained. Returns true on success, false on rollback.
// Preserves ownShortcut for tet completion detection.
function excitationMaterialiseSC(e, scId, isBridge){
    if(xonImpliedSet.has(scId)){ e.ownShortcut=scId; return true; }
    if(activeSet.has(scId)){ e.ownShortcut=scId; return true; }
    const prevShortcut = e.ownShortcut;
    const posBefore = pos.map(p=>[p[0],p[1],p[2]]);
    e.ownShortcut = scId;
    xonImpliedSet.add(scId);
    if(typeof _scAttribution !== 'undefined') _scAttribution.set(scId, { reason: 'excitationMaterialise', tick: typeof _demoTick !== 'undefined' ? _demoTick : 0 });
    impliedSet.add(scId);
    impliedBy.set(scId, new Set());
    bumpState();
    const pFinal = detectImplied();
    applyPositions(pFinal);

    // Strain check
    const ROLLBACK_TOL = 5e-4, AVG_TOL = 1e-5;
    let worstErr=0, sumErr=0;
    for(const [i,j] of BASE_EDGES){
        const err=Math.abs(vd(pos[i],pos[j])-1.0);
        if(err>worstErr) worstErr=err;
        sumErr+=err;
    }
    if(worstErr>ROLLBACK_TOL || sumErr/BASE_EDGES.length>AVG_TOL){
        if(isBridge){
            // Bridge shortcut: solver may have bailed out early (slow convergence
            // with many constraints). Re-solve with no bail-out to give it a fair
            // chance — bridge shortcuts are known tet partners and should converge.
            const pairs=[];
            activeSet.forEach(id=>{ const s=SC_BY_ID[id]; pairs.push([s.a,s.b]); });
            impliedSet.forEach(id=>{ const s=SC_BY_ID[id]; pairs.push([s.a,s.b]); });
            const {p:p2, converged}=_solve(pairs, 10000, /*noBailout=*/true);
            if(converged){
                applyPositions(p2);
                worstErr=0; sumErr=0;
                for(const [i,j] of BASE_EDGES){
                    const err=Math.abs(vd(pos[i],pos[j])-1.0);
                    if(err>worstErr) worstErr=err;
                    sumErr+=err;
                }
            }
        }
        if(worstErr>ROLLBACK_TOL || sumErr/BASE_EDGES.length>AVG_TOL){
            xonImpliedSet.delete(scId);
            impliedSet.delete(scId);
            impliedBy.delete(scId);
            e.ownShortcut = prevShortcut;
            applyPositions(posBefore);
            bumpState();
            return false;
        }
    }

    // Tet completion: if prevShortcut + scId are tet partners, record zero-point
    // Quark excitations skip void-binding (confinement via rankCandidates instead)
    if(prevShortcut !== null && !e._isQuark){
        const partners = tetPartnerMap.get(prevShortcut);
        if(partners && partners.includes(scId)){
            for(const v of voidNeighborData){
                if(v.type==='tet' && v.scIds.length===2 &&
                   v.scIds.includes(prevShortcut) && v.scIds.includes(scId)){
                    e.zeroPoint = v.nbrs.reduce((acc,i)=>
                        [acc[0]+pos[i][0]/4, acc[1]+pos[i][1]/4, acc[2]+pos[i][2]/4], [0,0,0]);
                    e.voidType = 'tet';
                    e.voidScIds = [prevShortcut, scId];
                    e.voidNodes = new Set(v.nbrs);
                    break;
                }
            }
        }
    }
    if(_deferUIUpdates){
        _uiDirty = true;
    } else {
        updateVoidSpheres(); updateCandidates(); updateSpheres(); updateStatus();
    }
    return true;
}

// ─── canMaterialiseQuick: lightweight dry-run strain check ─────────────────
// Returns true iff adding scId to the constraint system passes the strain
// thresholds. Uses a SINGLE solver call (no detectImplied) and never
// leaves side effects — safe to call in lookahead.
let _cmqCallCount = 0, _cmqCpuCount = 0, _cmqCacheHits = 0, _cmqTotalMs = 0;
// CMQ result cache: keyed by stateVersion + scId, cleared on SC set changes
let _cmqResultCache = new Map();
let _cmqResultCacheVersion = -1;
function canMaterialiseQuick(scId){
    _cmqCallCount++;
    if(activeSet.has(scId)||impliedSet.has(scId)||xonImpliedSet.has(scId)) return true;
    // Fast rejection: SC endpoints must be near the ideal FCC shortcut
    // length 2/sqrt(3) ≈ 1.1547 (unactivated) or already at unit length.
    const _sc=SC_BY_ID[scId];
    if(_sc && pos[_sc.a] && pos[_sc.b]){
        const _dx=pos[_sc.b][0]-pos[_sc.a][0],_dy=pos[_sc.b][1]-pos[_sc.a][1],_dz=pos[_sc.b][2]-pos[_sc.a][2];
        const _dist=Math.sqrt(_dx*_dx+_dy*_dy+_dz*_dz);
        const _SC_IDEAL=2/Math.sqrt(3); // 1.1547
        if(Math.abs(_dist-1)>0.05 && Math.abs(_dist-_SC_IDEAL)>0.10) return false;
    }
    // CMQ result cache: same SC config + same candidate → same pass/fail
    if (_cmqResultCacheVersion !== stateVersion) {
        _cmqResultCache.clear();
        _cmqResultCacheVersion = stateVersion;
    }
    if (_cmqResultCache.has(scId)) {
        _cmqCacheHits++;
        return _cmqResultCache.get(scId);
    }
    // Check GPU batch cache (avoids redundant CPU solve)
    if (typeof SolverProxy !== 'undefined' && SolverProxy.isReady()) {
        const cached = SolverProxy.getBatchResult(scId);
        if (cached) { _cmqCacheHits++; _cmqResultCache.set(scId, cached.pass); return cached.pass; }
    }
    // Build constraint pairs: push/pop avoids spread-copy (safe: _solve copies input)
    _cmqCpuCount++;
    const _cmqT0 = performance.now();
    const basePairs = _getBasePairs();
    const sc=SC_BY_ID[scId];
    basePairs.push([sc.a, sc.b]);
    const cmqIters = Math.max(500, Math.ceil(N * 5));
    const {p}=_solve(basePairs, cmqIters);
    basePairs.pop();
    _cmqTotalMs += performance.now() - _cmqT0;
    // Don't bail on !converged — solver may not reach 1e-9 on L3+
    // but positions can still be within strain tolerance. Let strain check decide.
    const ROLLBACK_TOL=5e-4, AVG_TOL=1e-5;
    let worst=0,sum=0,edgeLenSum=0;
    for(const [i,j] of BASE_EDGES){
        const dx=p[i][0]-p[j][0],dy=p[i][1]-p[j][1],dz=p[i][2]-p[j][2];
        const d=Math.sqrt(dx*dx+dy*dy+dz*dz);
        const err=Math.abs(d-1.0);
        if(err>worst) worst=err; sum+=err;
        edgeLenSum+=d;
    }
    if(worst>ROLLBACK_TOL || sum/BASE_EDGES.length>AVG_TOL) { _cmqResultCache.set(scId, false); return false; }
    // Kepler density check: reject if adding this SC would push density beyond 0.01%
    const lAvg=edgeLenSum/BASE_EDGES.length;
    const idealDens=computeIdealDensity();
    const actualDens=idealDens/(lAvg*lAvg*lAvg);
    const densDev=Math.abs(actualDens*100 - idealDens*100);
    if(densDev > 0.01) { _cmqResultCache.set(scId, false); return false; }
    _cmqResultCache.set(scId, true);
    return true;
}

// ─── excitationSeverForRoom: sever a non-load-bearing implied SC ────────
// When an excitation needs to materialise a shortcut but strain is too high,
// it may try severing ONE non-load-bearing xonImplied shortcut to make
// room. Candidates are ranked by fewest excitation references (orphans first).
// The function tries each candidate in rank order: sever it, check if the
// target SC can now be materialized, and if so keep the sever. If not, undo
// and try the next. Only ONE SC is ever severed.
// "Load-bearing" means: part of a completed tet pair, part of an actualized
// oct cycle, or part of any excitation's claimed void.
// Returns true if a sever enabled the target materialization.
function excitationSeverForRoom(targetScId){
    if(!xonImpliedSet.size) return false;

    // Build protected set (same logic as strainMonitorCheck)
    const protectedSCs = new Set();
    // Protect tet pairs
    for(const scId of xonImpliedSet){
        const partners = tetPartnerMap.get(scId);
        if(partners){
            for(const pid of partners){
                if(xonImpliedSet.has(pid) || activeSet.has(pid)){
                    protectedSCs.add(scId);
                    protectedSCs.add(pid);
                }
            }
        }
    }
    // Protect complete oct cycles (real-time check, not cached flag)
    for(const v of voidNeighborData){
        if(v.type !== 'oct' || !v.cycles) continue;
        for(const cycle of v.cycles){
            const allPresent = cycle.scIds.every(id =>
                xonImpliedSet.has(id) || activeSet.has(id) || impliedSet.has(id));
            if(!allPresent) continue;
            for(const id of cycle.scIds) protectedSCs.add(id);
        }
    }
    // Protect voidScIds of all void-bound excitations
    for(const ex of excitations){
        if(ex.voidScIds){
            for(const id of ex.voidScIds) protectedSCs.add(id);
        }
    }
    // Protect oct SCs (bosonic cage must never be severed)
    for(const id of _octSCIds) protectedSCs.add(id);

    // Protect SCs currently being traversed by xons (traversal lock).
    // If a xon is on an SC or needs it for its face loop, it cannot be severed.
    if (typeof _traversalLockedSCs === 'function') {
        for (const id of _traversalLockedSCs()) protectedSCs.add(id);
    }

    // Collect severable candidates, scored by fewest excitation references
    // (prefer severing orphan shortcuts that no excitation currently owns)
    const ranked = [];
    for(const scId of xonImpliedSet){
        if(protectedSCs.has(scId)) continue;
        let refs = 0;
        for(const ex of excitations){
            if(ex.ownShortcut === scId) refs++;
        }
        ranked.push({ scId, score: refs + Math.random() * 0.5 });
    }
    if(!ranked.length) return false;
    ranked.sort((a,b) => a.score - b.score);

    // Recursive N-depth severance: try removing 1, 2, 3... SCs until either
    // the target can materialize or all combinations are exhausted.
    // No artificial depth cap — the search is bounded by ranked.length.
    const severed = []; // stack of severed SC IDs

    function _trySeverDepth(startIdx) {
        // Check if target can materialize with current severances
        if (canMaterialiseQuick(targetScId)) {
            // Success — finalize all severed SCs
            for (const vid of severed) {
                impliedBy.delete(vid);
                for (const ex of excitations) {
                    if (ex.ownShortcut === vid) ex.ownShortcut = null;
                    if (ex.voidScIds && ex.voidScIds.includes(vid)) {
                        ex.zeroPoint = null; ex.voidType = null;
                        ex.voidScIds = null; ex.voidNodes = null;
                    }
                }
            }
            _severanceCount += severed.length;
            bumpState();
            return true;
        }

        // Try severing each remaining candidate
        for (let i = startIdx; i < ranked.length; i++) {
            const vid = ranked[i].scId;
            xonImpliedSet.delete(vid);
            impliedSet.delete(vid);
            stateVersion++;
            severed.push(vid);

            if (_trySeverDepth(i + 1)) return true;

            // Undo
            severed.pop();
            xonImpliedSet.add(vid);
            if(typeof _scAttribution !== 'undefined') _scAttribution.set(vid, { reason: 'severUndo', tick: typeof _demoTick !== 'undefined' ? _demoTick : 0 });
            impliedSet.add(vid);
            stateVersion++;
        }
        return false;
    }

    if (_trySeverDepth(0)) return true;
    return false;
}

// ══════════════════════════════════════════════════════════════════════════
// DO NOT DELETE — EXCITATION MOVEMENT RULES (comprehensive reference)
// ══════════════════════════════════════════════════════════════════════════
//
// An excitation is a sparkle that hops between lattice nodes. Its movement
// obeys a strict hierarchy that determines how it travels and what actions
// it may take at each step.
//
// ── PRIMARY MOVEMENT: BASE-DIRECTION 2-HOP TRAVERSAL ────────────────────
//
// The preferred, highest-priority way for an excitation to move is along
// TWO consecutive base-direction edges: node → mid → far. This 2-hop
// traversal REQUIRES the shortcut (node ↔ far) to exist. The excitation
// may CREATE this shortcut if it doesn't already exist — this is the
// preferred, canonical way excitations travel and build structure.
//
// If the shortcut cannot be materialized (solver strain too high), the
// excitation may SEVER a non-load-bearing implied shortcut elsewhere in
// the lattice to relieve strain, then retry. "Load-bearing" means:
//   • Part of a completed tet void (both tet-pair SCs present)
//   • Part of an actualized oct cycle (all cycle SCs present)
//   • Part of any excitation's claimed void (voidScIds)
//
// All base-direction candidates are ranked by ZERO-SUM BALANCE: the
// excitation tracks how many times it has used each of the 4 base
// directions (dirCounts), and prefers the least-used pair (d1, d2).
// This balance constraint causes the walk to converge to a closed cycle,
// naturally producing the 4-step tetrahedral fermion loop.
//
// OVERLAPPING TRAVERSAL: A continuous path w → x → y → z implies TWO
// shortcuts: w–y AND x–z. When the excitation moves node → mid → far,
// the overlapping shortcut lastMid–mid is also materialized if possible.
//
// ── VOID STICKING ───────────────────────────────────────────────────────
//
// Once an excitation claims a void (tet or oct cycle), it restricts its
// candidates to only those whose mid AND far nodes are within voidNodes.
// The excitation traces the boundary of its void shape and never strays.
//
// ── VOID DETECTION ──────────────────────────────────────────────────────
//
// After each successful base traversal, the excitation proactively checks
// whether its current node, mid, or far is part of a complete void:
//   Priority 1: Tetrahedral void (fermion, max 1 excitation per tet)
//   Priority 2: Octahedral cycle (boson, up to 8 excitations per cycle)
// An oct excitation needs only a single square cycle (4 nodes) of the
// full octahedron — not all 3 cycles.
//
// ── FALLBACK: SHORTCUT-DIRECTION TRAVEL ─────────────────────────────────
//
// If NO base-direction 2-hop traversal is possible (all candidates fail
// even after attempting sever-for-room), the excitation falls back to
// traveling directly along a shortcut edge (single hop, shortcut
// direction rather than base direction):
//   1. Prefer existing open shortcuts from the current node
//   2. Last resort: materialize a new shortcut and travel along it
//      (with sever-for-room if strain blocks materialization)
//
// ── STUCK / DISSOLUTION ─────────────────────────────────────────────────
//
// If neither base-direction nor shortcut-direction movement is possible,
// the excitation increments stuckTicks. Seeking excitations (no void
// claimed) are subject to stochastic decay: each step has probability
// 1/avgLifespan of dissolving (radioactive-decay model). Stuck
// excitations decay 4x faster. Void-bound excitations are immortal —
// they only dissolve via dedup (tet=1, oct cycle=8 max).
//
// ── DEGENERATE LOOP DETECTION ─────────────────────────────────────────────
//
// A seeking excitation whose trail (last 24 nodes) visits fewer than
// 4 unique nodes after 12+ steps is stuck in a degenerate cycle (e.g.
// a 3-node equilateral triangle). Both tet and oct voids require at
// least 4 distinct nodes, so such a loop can never converge to a valid
// void. These excitations are dissolved immediately.
//
// ── SEVERANCE HEURISTICS (excitationSeverForRoom) ───────────────────────
//
// When materialization fails due to lattice strain, the excitation may
// sever exactly ONE non-load-bearing implied shortcut to make room.
// The algorithm:
//   1. Build a protected set (tet pairs, actualized oct cycles, voidScIds)
//   2. Rank severable candidates by fewest excitation references (orphan
//      shortcuts first — those not owned by any excitation)
//   3. Try each in rank order: temporarily sever it, check if the target
//      SC can now be materialized (canMaterialiseQuick). If yes, finalize
//      the sever and return success. If no, undo and try the next.
//   4. At most ONE shortcut is ever severed per attempt. If no single
//      sever enables the target, the excitation is simply stuck.
//
// ── RULE ARENA MODE ─────────────────────────────────────────────────────
//
// In arena mode, candidate ranking is driven by pluggable rules from
// RULE_REGISTRY[activeRuleIndex]. Each rule's rankCandidates() sets a
// .score property on candidates; highest score wins (ties shuffled).
//
// The arena framework tests rules headlessly, measuring TEMPORAL
// K-complexity — LZ76 on the concatenated sequence of state strings
// over time. This captures how much new information each state change
// adds to the "3D movie" of lattice events.
//
// Built-in rules: 'classic' (zero-sum direction balance, index 0),
// 'adam' (genome-weighted features). New rules are added to
// RULE_REGISTRY by Claude across conversations.
//
// ── MAINTAINER NOTE ─────────────────────────────────────────────────────
//
// When excitation movement rules change, UPDATE THIS COMMENT BLOCK to
// keep it as the single source of truth. Do not let the code and this
// comment diverge.
//
// ══════════════════════════════════════════════════════════════════════════
//
// canMaterialise: when false, excitation only walks on already-open
// shortcuts (no solver calls). Used to keep ticks fast at L2+ with many
// excitations — only a few per tick get to materialise.
function excitationStep(e, canMaterialise){
    // ── Phase 2: complete pending traversal (second base step) ────────
    if(e.travelDest !== null){
        e.trail.push(e.node); if(e.trail.length>TRAIL_LENGTH) e.trail.shift();
        e.prevNode = e.node;
        e.node = e.travelDest;
        e.travelDest = null;
        e.tweenT = 0; e.flashT = 1.0;
        return;
    }

    // ── Phase 1: build candidate direction pairs, ranked by balance ──
    // Base-direction 2-hop traversal WITH materialization is the preferred
    // primary movement. If shortcut isn't open, the excitation creates it.
    const allOpen = getAllOpen();
    const candidates = [];
    for(let d1=0; d1<4; d1++){
        const mid = basePosNeighbor[e.node]?.[d1];
        if(mid === undefined) continue;
        for(let d2=0; d2<4; d2++){
            if(d2 === d1) continue;
            const far = basePosNeighbor[mid]?.[d2];
            if(far === undefined) continue;
            const scId = scPairToId.get(pairId(e.node, far));
            if(scId === undefined) continue;
            candidates.push({d1, d2, mid, far, scId,
                priority: e.dirCounts[d1] + e.dirCounts[d2]});
        }
    }

    // Void-sticking: once excitation has claimed a void (tet or oct),
    // strictly stay on void nodes. Both mid AND far must be in voidNodes
    // so the excitation never strays off its shape.
    if(e.voidNodes && candidates.length){
        const voidLocal = candidates.filter(c =>
            e.voidNodes.has(c.mid) && e.voidNodes.has(c.far));
        if(voidLocal.length){
            candidates.length = 0;
            candidates.push(...voidLocal);
        }
    }

    // ── Rank candidates using active rule (if it provides rankCandidates) ──
    const rule = getActiveRule();
    if(rule && rule.rankCandidates){
        const needsK = activeRuleIndex > 0 && canMaterialise;
        const { stateStr: kStr, baseline: kBase } = needsK
            ? getKStateAndBaseline() : { stateStr: '', baseline: 0 };
        const ruleCtx = {
            allOpen, kStr, kBase, pos, ALL_SC,
            frameCount: _temporalFrames.length, temporalK: _temporalKValue,
            avgHamming: _avgHamming, stuckTicks: _stuckTickCount,
            isFallback: false,
            quarks: (typeof NucleusSimulator !== 'undefined') ? NucleusSimulator.quarkExcitations : [],
            basePosNeighbor: basePosNeighbor,
        };
        rule.rankCandidates(candidates, e, ruleCtx);
    } else {
        // No rankCandidates: random movement (excitation dynamics driven by tick())
        for(const c of candidates) c.score = Math.random();
    }
    candidates.sort((a, b) => b.score - a.score);
    // Shuffle within top-score tier (epsilon tolerance for floats)
    if(candidates.length){
        const topScore = candidates[0].score;
        let shuffleEnd = candidates.findIndex(c => topScore - c.score > 0.01);
        if(shuffleEnd < 0) shuffleEnd = candidates.length;
        for(let i = shuffleEnd - 1; i > 0; i--){
            const j = Math.floor(Math.random() * (i + 1));
            [candidates[i], candidates[j]] = [candidates[j], candidates[i]];
        }
    }

    // ── Try each candidate: materialise shortcut if not already open ──
    for(const {d1, d2, mid, far, scId} of candidates){
        // If shortcut isn't open, try to materialise it (preferred primary movement).
        // If strain is too high, sever a non-load-bearing implied SC to make room.
        if(!allOpen.has(scId)){
            if(!canMaterialise) continue;
            if(!canMaterialiseQuick(scId)){
                // Try severing a non-load-bearing SC to make room for this one
                if(!excitationSeverForRoom(scId)) continue;
            }
            if(!excitationMaterialiseSC(e, scId)) continue;
        } else {
            e.ownShortcut = scId;
        }

        // Overlapping traversal shortcut: the continuous base-edge path
        // lastMid → node → mid implies a shortcut lastMid–mid.
        // (path w→x→y→z implies both w-y and x-z)
        if(e.lastMid !== null && canMaterialise){
            const overlapScId = scPairToId.get(pairId(e.lastMid, mid));
            if(overlapScId !== undefined && !allOpen.has(overlapScId)){
                if(canMaterialiseQuick(overlapScId)){
                    const savedOwn = e.ownShortcut;
                    excitationMaterialiseSC(e, overlapScId, /*isBridge=*/true);
                    e.ownShortcut = savedOwn;
                }
            }
        }

        // Proactive tet detection: check if the excitation's current node,
        // mid node, or far node is part of a COMPLETE tet void (both scIds open).
        // This handles tets whose two shortcuts don't share endpoints —
        // the excitation just needs to VISIT any node of the tet.
        // Quark excitations skip void-binding — their confinement is handled
        // by nucleus-sustain rankCandidates, not by zeroPoint sticking.
        if(e.zeroPoint === null && !e._isQuark){
            for(const checkNode of [e.node, mid, far]){
                const tetVoids = _nodeTetVoids.get(checkNode);
                if(!tetVoids) continue;
                for(const tv of tetVoids){
                    if(tv.scIds.every(id => allOpen.has(id))){
                        e.zeroPoint = tv.nbrs.reduce((acc,i)=>
                            [acc[0]+pos[i][0]/4, acc[1]+pos[i][1]/4, acc[2]+pos[i][2]/4], [0,0,0]);
                        e.voidType = 'tet';
                        e.voidScIds = [...tv.scIds];
                        e.voidNodes = new Set(tv.nbrs);
                        break;
                    }
                }
                if(e.zeroPoint !== null) break;
            }
        }

        // Priority 2: oct void CYCLES (boson — up to 8 per void)
        // An excitation can survive on a single square cycle (4-node loop)
        // of an oct void — it doesn't need the full octahedron to be complete.
        // The cycle must be a geometric square (per-cycle actualized flag).
        if(e.zeroPoint === null && !e._isQuark){
            for(const checkNode of [e.node, mid, far]){
                const octVoids = _nodeOctVoids.get(checkNode);
                if(!octVoids) continue;
                for(const ov of octVoids){
                    if(!ov.cycles) continue;
                    for(const cycle of ov.cycles){
                        if(!cycle.actualized) continue;
                        // Check that this node is actually part of this cycle
                        if(!cycle.verts.includes(checkNode)) continue;
                        const nv = cycle.verts.length;
                        let cx=0,cy=0,cz=0;
                        for(const n of cycle.verts){ cx+=pos[n][0]; cy+=pos[n][1]; cz+=pos[n][2]; }
                        e.zeroPoint = [cx/nv, cy/nv, cz/nv];
                        e.voidType = 'oct';
                        e.voidScIds = [...cycle.scIds];
                        e.voidNodes = new Set(cycle.verts);
                        break;
                    }
                    if(e.zeroPoint !== null) break;
                }
                if(e.zeroPoint !== null) break;
            }
        }

        // Bridge + tet partner materialization only when allowed
        if(canMaterialise){
            if(e.lastMid !== null && e.lastMid !== mid){
                const bridgeScId = scPairToId.get(pairId(e.lastMid, mid));
                if(bridgeScId !== undefined && !allOpen.has(bridgeScId)){
                    excitationMaterialiseSC(e, bridgeScId, /*isBridge=*/true);
                }
            }
            if(e.ownShortcut !== null){
                const tetPartners = tetPartnerMap.get(e.ownShortcut);
                if(tetPartners){
                    const savedOwn = e.ownShortcut;
                    for(const pid of tetPartners){
                        if(!xonImpliedSet.has(pid) && !activeSet.has(pid)){
                            if(canMaterialiseQuick(pid)){
                                excitationMaterialiseSC(e, pid, /*isBridge=*/true);
                                e.ownShortcut = savedOwn;
                            }
                        }
                    }
                }
            }
        }
        e.lastMid = mid;

        // Update direction counts
        e.dirCounts[d1]++;
        e.dirCounts[d2]++;

        // Take first base step
        e.stuckTicks = 0;
        e.trail.push(e.node); if(e.trail.length>TRAIL_LENGTH) e.trail.shift();
        e.prevNode = e.node;
        e.node = mid;
        e.travelDest = far;
        e.tweenT = 0; e.flashT = 1.0;
        return;
    }

    // ── All base-direction candidates failed → shortcut-direction fallback ──
    // Prefer existing open shortcuts; materialize new ones as a last resort.
    // 1. Try existing open shortcuts
    const scFallback = [];
    for(const sc of scByVert[e.node]){
        if(!allOpen.has(sc.id)) continue;
        const dest = sc.a === e.node ? sc.b : sc.a;
        if(dest === e.prevNode) continue;
        scFallback.push({scId: sc.id, dest});
    }
    if(scFallback.length){
        // Rank fallback using active rule (if available)
        if(rule && rule.rankCandidates){
            const fbCtx = { ...ruleCtx, isFallback: true };
            rule.rankCandidates(scFallback, e, fbCtx);
        } else {
            for(const c of scFallback) c.score = Math.random();
        }
        scFallback.sort((a, b) => b.score - a.score);
        const pick = scFallback[0];
        e.ownShortcut = pick.scId;
        e.stuckTicks = 0;
        e.trail.push(e.node); if(e.trail.length>TRAIL_LENGTH) e.trail.shift();
        e.prevNode = e.node;
        e.node = pick.dest;
        e.tweenT = 0; e.flashT = 1.0;
        return;
    }
    // 2. Last resort: materialize a new shortcut and travel on it
    if(canMaterialise){
        const scNew = [];
        for(const sc of scByVert[e.node]){
            if(allOpen.has(sc.id)) continue;
            const dest = sc.a === e.node ? sc.b : sc.a;
            if(dest === e.prevNode) continue;
            scNew.push({sc, dest});
        }
        // Rank last-resort candidates using active rule (add scId for feature extraction)
        for(const item of scNew) item.scId = item.sc.id;
        if(rule && rule.rankCandidates){
            const lrCtx = { ...ruleCtx, isFallback: true };
            rule.rankCandidates(scNew, e, lrCtx);
        } else {
            for(const c of scNew) c.score = Math.random();
        }
        scNew.sort((a, b) => b.score - a.score);
        for(const {sc, dest} of scNew){
            if(!canMaterialiseQuick(sc.id)){
                if(!excitationSeverForRoom(sc.id)) continue;
            }
            if(!excitationMaterialiseSC(e, sc.id)) continue;
            e.ownShortcut = sc.id;
            e.stuckTicks = 0;
            e.trail.push(e.node); if(e.trail.length>TRAIL_LENGTH) e.trail.shift();
            e.prevNode = e.node;
            e.node = dest;
            e.tweenT = 0; e.flashT = 1.0;
            return;
        }
    }
    e.stuckTicks++;
}

function createExcitation(nodeIdx){
    // Verify at least one shortcut traversal is possible from this node
    let hasPath = false;
    for(let d1=0; d1<4 && !hasPath; d1++){
        const mid = basePosNeighbor[nodeIdx]?.[d1];
        if(mid === undefined) continue;
        for(let d2=0; d2<4; d2++){
            if(d2===d1) continue;
            if(basePosNeighbor[mid]?.[d2] !== undefined){ hasPath=true; break; }
        }
    }
    if(!hasPath){
        setStatus('\u26a0 no valid excitation path from this node');
        return null;
    }

    const id=xonNextId++; const colIdx=id%XON_COLORS.length; const col=XON_COLORS[colIdx];
    // Spark sprite — sparkle point that travels along base edges
    const sparkMat=new THREE.SpriteMaterial({color:col,map:_sparkTex,transparent:true,opacity:1.0,
        blending:THREE.AdditiveBlending,depthWrite:false,depthTest:false});
    const spark=new THREE.Sprite(sparkMat); spark.scale.set(0.28,0.28,1); spark.renderOrder=22;
    const group=new THREE.Group(); group.add(spark);
    group.position.set(...pos[nodeIdx]); scene.add(group);
    const trailGeo=new THREE.BufferGeometry();
    const trailPos=new Float32Array(TRAIL_LENGTH*3); const trailCol=new Float32Array(TRAIL_LENGTH*3);
    trailGeo.setAttribute('position',new THREE.BufferAttribute(trailPos,3));
    trailGeo.setAttribute('color',new THREE.BufferAttribute(trailCol,3));
    const trailMat=new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:1.0,depthTest:false,blending:THREE.AdditiveBlending});
    const trailLine=new THREE.Line(trailGeo,trailMat); trailLine.renderOrder=20; scene.add(trailLine);
    const e={id,node:nodeIdx,prevNode:nodeIdx,travelDest:null,stuckTicks:0,
        dirCounts:[0,0,0,0],lastMid:null,sameSCSteps:0,totalSteps:0,
        trail:[],tweenT:1,flashT:0,ownShortcut:null,zeroPoint:null,
        voidType:null,voidScIds:null,voidNodes:null,
        group,spark,sparkMat,trailLine,trailGeo,trailPos,trailCol,colorIdx:colIdx,col};
    excitations.push(e);
    startExcitationClock();
    selectedVert=-1; hoveredVert=-1;
    updateCandidates(); updateSpheres();
    updateExcitationSidebar();
    return e;
}

// V2: Color-parameterized excitation creator (used by NucleusSimulator)
function _createExcitation(nodeIdx, customColor){
    let hasPath = false;
    for(let d1=0; d1<4 && !hasPath; d1++){
        const mid = basePosNeighbor[nodeIdx]?.[d1];
        if(mid === undefined) continue;
        for(let d2=0; d2<4; d2++){
            if(d2===d1) continue;
            if(basePosNeighbor[mid]?.[d2] !== undefined){ hasPath=true; break; }
        }
    }
    if(!hasPath) return null;
    const id=xonNextId++;
    const col = customColor || XON_COLORS[id % XON_COLORS.length];
    const sparkMat=new THREE.SpriteMaterial({color:col,map:_sparkTex,transparent:true,opacity:1.0,
        blending:THREE.AdditiveBlending,depthWrite:false,depthTest:false});
    const spark=new THREE.Sprite(sparkMat); spark.scale.set(0.28,0.28,1); spark.renderOrder=22;
    const group=new THREE.Group(); group.add(spark);
    group.position.set(...pos[nodeIdx]); scene.add(group);
    const trailGeo=new THREE.BufferGeometry();
    const trailPos=new Float32Array(TRAIL_LENGTH*3); const trailCol=new Float32Array(TRAIL_LENGTH*3);
    trailGeo.setAttribute('position',new THREE.BufferAttribute(trailPos,3));
    trailGeo.setAttribute('color',new THREE.BufferAttribute(trailCol,3));
    const trailMat=new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:1.0,depthTest:false,blending:THREE.AdditiveBlending});
    const trailLine=new THREE.Line(trailGeo,trailMat); trailLine.renderOrder=20; scene.add(trailLine);
    const e={id,node:nodeIdx,prevNode:nodeIdx,travelDest:null,stuckTicks:0,
        dirCounts:[0,0,0,0],lastMid:null,sameSCSteps:0,totalSteps:0,
        trail:[],tweenT:1,flashT:0,ownShortcut:null,zeroPoint:null,
        voidType:null,voidScIds:null,voidNodes:null,
        group,spark,sparkMat,trailLine,trailGeo,trailPos,trailCol,colorIdx:0,col};
    excitations.push(e);
    return e;
}

let _batchRemoveMode = false; // when true, skip per-removal sidebar rebuilds
function removeExcitation(id){
    const idx=excitations.findIndex(e=>e.id===id); if(idx<0) return;
    const e=excitations[idx];
    if(!e._headless){
        scene.remove(e.group); e.sparkMat.dispose();
        scene.remove(e.trailLine); e.trailGeo.dispose();
    }
    excitations.splice(idx,1);
    if(!excitations.length && !bigBangActive) stopExcitationClock();
    if(!_batchRemoveMode && !_deferUIUpdates) updateExcitationSidebar();
}
function removeAllExcitations(){
    _batchRemoveMode=true;
    [...excitations].forEach(e=>removeExcitation(e.id));
    _batchRemoveMode=false;
    // Xon-implied SCs are real structure — keep them unless they cause
    // strain violations. The old behavior of wiping all xon-implied SCs
    // on clear was too aggressive (shortcuts visibly vanished).
    if(xonImpliedSet.size){
        // Check for actual strain before clearing anything
        const TOL = 1e-3;
        let hasStrain = false;
        for(const [i,j] of BASE_EDGES){
            if(Math.abs(vd(pos[i],pos[j])-1.0) > TOL){ hasStrain = true; break; }
        }
        if(hasStrain){
            // Only clear if strain exists — soft recovery
            for(const id of [...xonImpliedSet]){
                xonImpliedSet.delete(id);
                impliedSet.delete(id);
                impliedBy.delete(id);
            }
            bumpState();
            const pFinal = detectImplied();
            applyPositions(pFinal);
            toast('strain reset: cleared xon-implied SCs');
        }
    }
    updateExcitationSidebar(); excitationPaused=false; syncExcitationPlayBtn();
}

// ─── Big Bang: batch-create excitations at every node ─────────────────────
// Performance: skips per-excitation UI updates (rebuildScPairLookup,
// updateCandidates, updateSpheres, updateExcitationSidebar) and does
// them once at the end. Critical for L2+ lattices (100+ nodes).
function _doBigBang(){
    removeAllExcitations();
    // Survivors (excitations with claimed voids) get 2× energy on re-bang
    rebuildScPairLookup();
    for(let i = 0; i < N; i++){
        let hasPath = false;
        for(let d1=0; d1<4 && !hasPath; d1++){
            const mid = basePosNeighbor[i]?.[d1];
            if(mid === undefined) continue;
            for(let d2=0; d2<4; d2++){
                if(d2===d1) continue;
                if(basePosNeighbor[mid]?.[d2] !== undefined){ hasPath=true; break; }
            }
        }
        if(!hasPath) continue;
        const id=xonNextId++; const colIdx=id%XON_COLORS.length; const col=XON_COLORS[colIdx];
        const sparkMat=new THREE.SpriteMaterial({color:col,map:_sparkTex,transparent:true,opacity:1.0,
            blending:THREE.AdditiveBlending,depthWrite:false,depthTest:false});
        const spark=new THREE.Sprite(sparkMat); spark.scale.set(0.28,0.28,1); spark.renderOrder=22;
        const group=new THREE.Group(); group.add(spark);
        group.position.set(...pos[i]); scene.add(group);
        const trailGeo=new THREE.BufferGeometry();
        const trailPos=new Float32Array(TRAIL_LENGTH*3); const trailCol=new Float32Array(TRAIL_LENGTH*3);
        trailGeo.setAttribute('position',new THREE.BufferAttribute(trailPos,3));
        trailGeo.setAttribute('color',new THREE.BufferAttribute(trailCol,3));
        const trailMat=new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:1.0,depthTest:false,blending:THREE.AdditiveBlending});
        const trailLine=new THREE.Line(trailGeo,trailMat); trailLine.renderOrder=20; scene.add(trailLine);
        excitations.push({id,node:i,prevNode:i,travelDest:null,stuckTicks:0,
            dirCounts:[0,0,0,0],lastMid:null,sameSCSteps:0,totalSteps:0,
            trail:[],tweenT:1,flashT:0,ownShortcut:null,zeroPoint:null,
            voidType:null,voidScIds:null,voidNodes:null,
            group,spark,sparkMat,trailLine,trailGeo,trailPos,trailCol,colorIdx:colIdx,col});
    }
    selectedVert=-1; hoveredVert=-1;
    updateCandidates(); updateSpheres();
    updateExcitationSidebar();
    startExcitationClock();
}

// ─── Big Bang toggle mode ────────────────────────────────────────────────
// When active, monitors for stale states and re-bangs automatically:
//   - Empty lattice (all excitations dissolved) → instant re-bang
//   - All survivors settled (everyone has a void) with no activity
//     (no dissolutions) for staleTicks → re-bang with 2× energy for survivors
let bigBangActive = false;
let _bbStaleCounter = 0;
let _bbLastExcCount = 0;
const BB_STALE_TICKS = 40; // ticks of no change before re-bang

function toggleBigBang(){
    bigBangActive = !bigBangActive;
    _syncBigBangBtn();
    if(bigBangActive){
        _bbStaleCounter = 0;
        _bbLastExcCount = 0;
        _doBigBang();
    }
}

// Deactivate big bang without toggling — called when user modifies graph
function deactivateBigBang(){
    if(!bigBangActive) return;
    bigBangActive = false;
    _syncBigBangBtn();
}

function _syncBigBangBtn(){
    const btn = document.getElementById('btn-big-bang');
    btn.classList.toggle('active', bigBangActive);
    btn.textContent = bigBangActive ? 'stop rebang ▾' : 'big bang ▾';
}

function bigBangStaleCheck(){
    if(!bigBangActive) return;
    // Empty lattice → instant re-bang
    if(!excitations.length){
        _bbStaleCounter = 0;
        _bbLastExcCount = 0;
        _doBigBang();
        return;
    }
    // Check if population changed (dissolution activity)
    if(excitations.length !== _bbLastExcCount){
        _bbLastExcCount = excitations.length;
        _bbStaleCounter = 0;
        return;
    }
    // All survivors have claimed voids — check if stable
    const allSettled = excitations.every(e => e.zeroPoint !== null);
    if(!allSettled){
        _bbStaleCounter = 0;
        return;
    }
    // Settled and no population change — count stale ticks
    _bbStaleCounter++;
    if(_bbStaleCounter >= BB_STALE_TICKS){
        _bbStaleCounter = 0;
        _bbLastExcCount = 0;
        _doBigBang();
    }
}

let excitationClockTimer=null, excitationClockCursor=0, excitationPaused=false;
let _strainCheckCounter = 0;

// ─── Background strain monitor ────────────────────────────────────────────
// WHY THIS EXISTS:
//   The per-induction rollback (in excitationInduceShortcut) checks whether
//   adding a SINGLE new SC pushes avgEdge error over a threshold. But each
//   SC contributes only ~0.15–0.5 ppm individually, so all pass. With 20
//   simultaneous xon-implied SCs the cumulative drift reaches 17+ ppm
//   (density 74.0442%) because no individual induction ever trips the rollback.
//
//   Fix: every STRAIN_CHECK_INTERVAL ticks, measure the actual running avgErr
//   across all base edges. If it exceeds STRAIN_EVICT_TOL (3 ppm), find the
//   single xon-implied SC whose removal most reduces avgErr and evict it.
//   The owning excitation loses ownShortcut and keeps walking — no freeze.
//
//   1 ppm threshold keeps actual strain negligible. Density is always
//   displayed as Kepler max (74.0480%) — solver noise is not physics.
//
// DO NOT remove this monitor. Without it cumulative drift from many concurrent
// SCs is undetectable by per-induction checks and causes underdensity readings.
// ─────────────────────────────────────────────────────────────────────────────
const STRAIN_CHECK_INTERVAL = 8;  // check every N ticks (not every tick — perf)
const STRAIN_EVICT_TOL = 1e-6;    // 1 ppm avgErr threshold

function strainMonitorCheck(){
    if(!xonImpliedSet.size) return;

    // Check strain level
    let sumErr=0;
    for(const [i,j] of BASE_EDGES) sumErr+=Math.abs(vd(pos[i],pos[j])-1.0);
    const avgErr = sumErr / BASE_EDGES.length;
    if(avgErr <= STRAIN_EVICT_TOL) return;

    // Build set of protected SCs — shortcuts whose tet partner is ALSO in
    // xonImpliedSet form a completed tet void and must not be evicted.
    const protectedSCs = new Set();
    for(const scId of xonImpliedSet){
        const partners = tetPartnerMap.get(scId);
        if(partners){
            for(const pid of partners){
                if(xonImpliedSet.has(pid)){
                    protectedSCs.add(scId);
                    protectedSCs.add(pid);
                }
            }
        }
    }

    // Protect oct void members: protect SCs of any complete cycle.
    // Computed in real-time (not from cached cycle.actualized) to avoid
    // stale flags when updateVoidSpheres is deferred during excitation ticks.
    for(const v of voidNeighborData){
        if(v.type !== 'oct' || !v.cycles) continue;
        for(const cycle of v.cycles){
            const allPresent = cycle.scIds.every(id =>
                xonImpliedSet.has(id) || activeSet.has(id) || impliedSet.has(id));
            if(!allPresent) continue;
            for(const id of cycle.scIds) protectedSCs.add(id);
        }
    }

    // Find the SC whose removal most reduces avgErr (try removing each one).
    // Evict at most 1 per tick — the monitor runs every STRAIN_CHECK_INTERVAL
    // ticks, so high strain converges over a few intervals without frame freezes.
    let bestId=null, bestAvg=avgErr;
    for(const scId of xonImpliedSet){
        if(protectedSCs.has(scId)) continue; // don't evict tet pair members
        const sc=SC_BY_ID[scId];
        // Test: remove this SC and re-solve
        const testPairs=[...[...activeSet,...xonImpliedSet]
            .filter(id=>id!==scId)
            .map(id=>{ const s=SC_BY_ID[id]; return [s.a,s.b]; })];
        const {p:tp, converged} = _solve(testPairs);
        if(!converged) continue;
        let ts=0;
        for(const [i,j] of BASE_EDGES){
            const dx=tp[i][0]-tp[j][0],dy=tp[i][1]-tp[j][1],dz=tp[i][2]-tp[j][2];
            ts+=Math.abs(Math.sqrt(dx*dx+dy*dy+dz*dz)-1.0);
        }
        const ta=ts/BASE_EDGES.length;
        if(ta<bestAvg){ bestAvg=ta; bestId=scId; }
    }
    if(bestId===null) return; // all remaining SCs are protected

    // Evict the worst SC
    xonImpliedSet.delete(bestId);
    impliedSet.delete(bestId);
    impliedBy.delete(bestId);
    // Clear ownShortcut on any excitation that owned it
    for(const e of excitations){
        if(e.ownShortcut===bestId){ e.ownShortcut=null; e.zeroPoint=null; e.voidType=null; e.voidScIds=null; e.voidNodes=null; }
    }
    const evicted = true;

    if(!evicted) return;
    bumpState();
    const pFinal=detectImplied();
    applyPositions(pFinal);
    updateVoidSpheres(); updateCandidates(); updateSpheres(); updateStatus();
    return true;
}
