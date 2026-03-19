// flux-demo-ca.js — Helpers (dir balance, locality, patterns) + Algorithm Registry
// Split from flux-demo.js — no logic changes.

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
