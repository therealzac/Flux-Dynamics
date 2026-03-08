// flux-demo.js — Demo mode: pattern computation, xon management, demo loop
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
let _demoTick = 0;
let _demoSchedule = null;     // 8-window physical schedule (32 ticks/cycle)
let _demoVisits = null;       // {face: {pu:0, pd:0, nu:0, nd:0}}
let _demoFaceDecks = null;    // {face: shuffled array} — stochastic type assignment
let _demoWindowTypes = null;  // current window's face→type map (persists 4 ticks)
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

// Loop topology → concrete node sequence, given tet cycle [a, b, c, d]
// a=octNode0, b=extNode, c=octNode1, d=octNode2
const LOOP_SEQUENCES = {
    pu: ([a, b, c, d]) => [a, b, a, c, a],      // Fork (p-up)
    nd: ([a, b, c, d]) => [a, b, c, b, a],      // Lollipop (n-down)
    pd: ([a, b, c, d]) => [a, b, c, d, a],      // Hamiltonian CW (p-down)
    nu: ([a, b, c, d]) => [a, d, c, b, a],      // Hamiltonian CCW (n-up)
};

const LOOP_TYPE_NAMES = { pu: 'fork', nd: 'lollipop', pd: 'ham_cw', nu: 'ham_ccw' };

const XON_TRAIL_LENGTH = 12;

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
    const trailPos = new Float32Array(XON_TRAIL_LENGTH * 3);
    const trailCol = new Float32Array(XON_TRAIL_LENGTH * 3);
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
        _mode: 'tet',           // 'tet' or 'oct'
        _lastDir: null,         // last direction index (0-3) for momentum
        _dirHistory: [],        // direction vector history for T16 test
        col, group, spark, sparkMat,
        trailLine, trailGeo, trailPos, trailCol,
        trail: [seq[0]], trailColHistory: [col], tweenT: 1, flashT: 1.0,
        alive: true,
    };
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
// Lifespan slider (0-100): 0 = instant kill, N = N ticks between each trail point removal.
function _decayDyingXons() {
    const lifespan = +document.getElementById('tracer-lifespan-slider').value;
    for (const xon of _demoXons) {
        if (!xon._dying || !xon._frozenPos) continue;
        if (lifespan === 0) {
            // Instant: clear entire frozen trail
            xon._frozenPos.length = 0;
            continue;
        }
        // Increment tick counter; remove one trail point every `lifespan` ticks
        xon._decayTicks = (xon._decayTicks || 0) + 1;
        if (xon._decayTicks >= lifespan) {
            xon._decayTicks = 0;
            xon._frozenPos.shift();
        }
    }
}

// Check if the next hop in a xon's loop crosses an SC-only edge that is still activated.
// Returns true if traversal is safe (base edge or SC is active), false if SC was deactivated.
function _canAdvanceSafely(xon) {
    if (!xon.alive || xon._loopStep >= 4) return false;
    const fromNode = xon._loopSeq[xon._loopStep];
    const toNode = xon._loopSeq[xon._loopStep + 1];
    if (toNode === undefined) return false;
    const hasBase = (baseNeighbors[fromNode] || []).some(nb => nb.node === toNode);
    if (hasBase) return true; // base edge, no SC needed
    const pid = pairId(fromNode, toNode);
    const scId = scPairToId.get(pid);
    if (scId === undefined) return true; // no SC on this edge
    return activeSet.has(scId) || impliedSet.has(scId) || electronImpliedSet.has(scId);
}

// Advance a xon one hop: update position state, push trail, start tween.
// SC negotiation with the vacuum happens BEFORE this call in demoTick.
function _advanceXon(xon) {
    if (!xon.alive || xon._loopStep >= 4) return;
    const fromNode = xon._loopSeq[xon._loopStep];
    const toNode = xon._loopSeq[xon._loopStep + 1];
    xon.prevNode = fromNode;
    xon.node = toNode;
    xon._loopStep++;

    // Push trail history + per-segment color, start tween
    xon.trail.push(toNode);
    xon.trailColHistory.push(xon.col);
    if (xon.trail.length > XON_TRAIL_LENGTH) { xon.trail.shift(); xon.trailColHistory.shift(); }
    xon.tweenT = 0;
    xon.flashT = 1.0;
}

// ╔══════════════════════════════════════════════════════════════════════╗
// ║  PERSISTENT 6-XON MODEL — Demo 3.1                                  ║
// ╚══════════════════════════════════════════════════════════════════════╝

// Spawn exactly 6 persistent xons on oct nodes. Called once from startDemoLoop.
// 3 sign=+1, 3 sign=-1. All start in oct mode (white, cruising cage).
function _initPersistentXons() {
    _demoXons = [];
    if (!_octNodeSet || _octNodeSet.size < 6) {
        console.error('[demo] Cannot init persistent xons: need 6 oct nodes, have', _octNodeSet?.size);
        return;
    }
    const octNodes = [..._octNodeSet];
    for (let i = 0; i < 6; i++) {
        const startNode = octNodes[i % octNodes.length];
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
        const trailPos = new Float32Array(XON_TRAIL_LENGTH * 3);
        const trailCol = new Float32Array(XON_TRAIL_LENGTH * 3);
        trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPos, 3));
        trailGeo.setAttribute('color', new THREE.BufferAttribute(trailCol, 3));
        const trailMat = new THREE.LineBasicMaterial({
            vertexColors: true, transparent: true, opacity: 1.0,
            depthTest: false, blending: THREE.AdditiveBlending,
        });
        const trailLine = new THREE.Line(trailGeo, trailMat);
        trailLine.renderOrder = 20;
        scene.add(trailLine);

        // Pick initial direction: random valid oct neighbor
        const octNeighbors = baseNeighbors[startNode].filter(nb => _octNodeSet.has(nb.node));
        const initNb = octNeighbors.length > 0 ? octNeighbors[Math.floor(Math.random() * octNeighbors.length)] : null;
        const initDir = initNb ? initNb.dirIdx : 0;

        const xon = {
            prevNode: startNode, sign,
            _loopType: null,
            _loopSeq: null, _loopStep: 0,
            _assignedFace: null, _quarkType: null,
            _mode: 'oct',
            _lastDir: initDir,
            _dirHistory: [],
            col, group, spark, sparkMat,
            trailLine, trailGeo, trailPos, trailCol,
            trail: [startNode], trailColHistory: [col], tweenT: 1, flashT: 1.0,
            alive: true,
        };
        // Interceptor: enforce single-hop-per-tick + validate each individual movement
        let _nodeVal = startNode;
        xon._movedThisTick = false;
        Object.defineProperty(xon, 'node', {
            get() { return _nodeVal; },
            set(v) {
                const from = _nodeVal;
                if (from === v) { _nodeVal = v; return; } // no-op assignment
                // Validate: nodes must be adjacent (base edge or SC edge)
                if (typeof scPairToId !== 'undefined' && scPairToId && scPairToId.size > 0) {
                    const hasBase = (baseNeighbors[from] || []).some(nb => nb.node === v);
                    if (!hasBase) {
                        // Check if there's an SC between them
                        const scs = scByVert[from] || [];
                        const hasSC = scs.some(sc => (sc.a === from ? sc.b : sc.a) === v);
                        if (!hasSC) {
                            console.warn(`[MOVEMENT BLOCKED] tick=${_demoTick} xon: ${from}→${v} NO EDGE (not adjacent)`);
                            return; // BLOCK: not adjacent at all
                        }
                        // SC exists — verify it's active
                        const pid = pairId(from, v);
                        const scId = scPairToId.get(pid);
                        if (scId !== undefined && !activeSet.has(scId) && !impliedSet.has(scId) && !electronImpliedSet.has(scId)) {
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
    console.log(`[demo] Initialized 6 persistent xons on oct nodes: [${_demoXons.map(x => x.node).join(',')}]`);
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

// Get scored oct-mode candidates for a xon. Returns array sorted by momentum score (desc).
// `blocked` is an optional Set of additional nodes to treat as occupied (for coordinated planning).
function _getOctCandidates(xon, occupied, blocked) {
    if (!xon.alive || xon._mode !== 'oct') return [];

    // Get ALL oct neighbors: base edges + SC edges
    const allOctNeighbors = [];
    for (const nb of baseNeighbors[xon.node]) {
        if (_octNodeSet.has(nb.node)) {
            allOctNeighbors.push({ node: nb.node, dirIdx: nb.dirIdx });
        }
    }
    const scs = scByVert[xon.node] || [];
    for (const sc of scs) {
        const other = sc.a === xon.node ? sc.b : sc.a;
        if (_octNodeSet.has(other) && !allOctNeighbors.find(n => n.node === other)) {
            const scId = sc.id;
            const alreadyActive = activeSet.has(scId) || impliedSet.has(scId) || electronImpliedSet.has(scId);
            const dx = pos[other][0] - pos[xon.node][0];
            const dy = pos[other][1] - pos[xon.node][1];
            const dz = pos[other][2] - pos[xon.node][2];
            const d = Math.sqrt(dx*dx + dy*dy + dz*dz) || 1;
            let bestDir = 0, bestDot = -Infinity;
            for (let k = 0; k < 4; k++) {
                const v = DIR_VEC[k];
                const dot = Math.abs((dx/d)*v[0] + (dy/d)*v[1] + (dz/d)*v[2]);
                if (dot > bestDot) { bestDot = dot; bestDir = k; }
            }
            allOctNeighbors.push({
                node: other, dirIdx: bestDir,
                _scId: scId, _needsMaterialise: !alreadyActive
            });
        }
    }

    if (allOctNeighbors.length === 0) return [];

    // Score candidates by momentum conservation
    const candidates = [];
    for (const nb of allOctNeighbors) {
        if (occupied.has(nb.node)) continue; // Pauli: already occupied
        if (blocked && blocked.has(nb.node)) continue; // Pauli: reserved by another planned move
        if (xon._lastDir === null || xon.prevNode === xon.node) {
            candidates.push({ node: nb.node, dirIdx: nb.dirIdx, score: 1, _scId: nb._scId, _needsMaterialise: nb._needsMaterialise });
        } else {
            const dx = pos[nb.node][0] - pos[xon.node][0];
            const dy = pos[nb.node][1] - pos[xon.node][1];
            const dz = pos[nb.node][2] - pos[xon.node][2];
            const len = Math.sqrt(dx*dx + dy*dy + dz*dz) || 1;
            const actualDir = [dx/len, dy/len, dz/len];
            const pdx = pos[xon.node][0] - pos[xon.prevNode][0];
            const pdy = pos[xon.node][1] - pos[xon.prevNode][1];
            const pdz = pos[xon.node][2] - pos[xon.prevNode][2];
            const plen = Math.sqrt(pdx*pdx + pdy*pdy + pdz*pdz) || 1;
            const prevActual = [pdx/plen, pdy/plen, pdz/plen];
            const dot = prevActual[0]*actualDir[0] + prevActual[1]*actualDir[1] + prevActual[2]*actualDir[2];
            // Accept all directions (including backtrack) but score them
            candidates.push({ node: nb.node, dirIdx: nb.dirIdx, score: dot, _scId: nb._scId, _needsMaterialise: nb._needsMaterialise });
        }
    }

    // Sort by score descending (prefer forward momentum)
    candidates.sort((a, b) => b.score - a.score);
    return candidates;
}

// Execute an oct move to a specific target. Handles vacuum negotiation.
// Returns true if the move succeeded, false if vacuum rejected.
function _executeOctMove(xon, target) {
    // Re-check SC activation at execution time (may have changed since planning)
    if (target._scId !== undefined) {
        const stillActive = activeSet.has(target._scId) || impliedSet.has(target._scId) || electronImpliedSet.has(target._scId);
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
    xon.prevNode = xon.node;
    xon.node = target.node;
    xon._lastDir = target.dirIdx;

    // Push trail history + per-segment color, start tween
    xon.trail.push(target.node);
    xon.trailColHistory.push(xon.col);
    if (xon.trail.length > XON_TRAIL_LENGTH) { xon.trail.shift(); xon.trailColHistory.shift(); }
    xon.tweenT = 0;
    xon.flashT = 1.0;
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

// Transition xon from oct mode to tet mode (assigned to actualize a face)
function _assignXonToTet(xon, face, quarkType) {
    const fd = _nucleusTetFaceData[face];
    if (!fd) return;

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
            // Xon is NOT on this face — walk it to the nearest face oct node
            // via connected edges (BFS) to avoid teleportation.
            const faceOctNodes = new Set(octNodesOnFace);
            const target = _walkToFace(xon, faceOctNodes);
            if (target !== null) {
                // Rotate cycle so the arrived-at node is position 0
                const a = cycle[0], b = cycle[1], c = cycle[2], d = cycle[3];
                let rotated;
                if (target === a) rotated = [a, b, c, d];
                else if (target === c) rotated = [c, b, a, d];
                else if (target === d) rotated = [d, b, c, a];
                else rotated = cycle;
                seq = LOOP_SEQUENCES[quarkType](rotated);
            }
            // If walk failed (shouldn't happen), seq stays as-is and xon
            // will already be at a face node from the walk.
        }
    }

    xon._mode = 'tet';
    xon._assignedFace = face;
    xon._quarkType = quarkType;
    xon._loopType = LOOP_TYPE_NAMES[quarkType];
    xon._loopSeq = seq;
    xon._loopStep = 0;
    xon.col = col;

    // Update spark color
    if (xon.sparkMat) xon.sparkMat.color.setHex(col);

    // Start from xon's current position (should already be at seq[0] after walk)
    xon.prevNode = xon.node;
    xon.node = seq[0];
}

// Walk xon to nearest node in targetNodes via connected edges (BFS).
// Moves the xon step-by-step, updating trail. Returns the target node reached.
// Pauli-aware: avoids nodes occupied by other xons (except the target itself).
function _walkToFace(xon, targetNodes) {
    if (targetNodes.has(xon.node)) return xon.node;

    // Build occupied set (exclude self)
    const occupiedNodes = new Set();
    for (const x of _demoXons) {
        if (x !== xon && x.alive) occupiedNodes.add(x.node);
    }

    // BFS from xon.node to nearest target, only via base edges + active SCs
    const visited = new Set([xon.node]);
    const parent = new Map();
    const queue = [xon.node];
    let found = null;

    while (queue.length > 0 && !found) {
        const curr = queue.shift();
        // Base neighbors
        const nbs = baseNeighbors[curr] || [];
        for (const nb of nbs) {
            if (visited.has(nb.node)) continue;
            if (!_octNodeSet.has(nb.node)) continue;
            visited.add(nb.node);
            parent.set(nb.node, curr);
            if (targetNodes.has(nb.node)) { found = nb.node; break; }
            // Skip occupied intermediate nodes (target is OK to land on)
            if (occupiedNodes.has(nb.node)) continue;
            queue.push(nb.node);
        }
        if (found) break;
        // SC neighbors (only activated SCs)
        const scs = scByVert[curr] || [];
        for (const sc of scs) {
            if (!activeSet.has(sc.id) && !impliedSet.has(sc.id) && !electronImpliedSet.has(sc.id)) continue;
            const neighbor = sc.a === curr ? sc.b : sc.a;
            if (visited.has(neighbor)) continue;
            if (!_octNodeSet.has(neighbor)) continue;
            visited.add(neighbor);
            parent.set(neighbor, curr);
            if (targetNodes.has(neighbor)) { found = neighbor; break; }
            if (occupiedNodes.has(neighbor)) continue;
            queue.push(neighbor);
        }
    }

    if (!found) return null; // no path (shouldn't happen on connected oct surface)

    // Reconstruct path and walk xon along it
    const path = [];
    let n = found;
    while (n !== xon.node) {
        path.push(n);
        n = parent.get(n);
    }
    path.reverse(); // path[0] is first step from xon.node, path[last] is target

    for (const step of path) {
        xon.prevNode = xon.node;
        xon.node = step;
        xon.trail.push(step);
        xon.trailColHistory.push(0xffffff); // white while walking to face
        if (xon.trail.length > XON_TRAIL_LENGTH) {
            xon.trail.shift();
            xon.trailColHistory.shift();
        }
    }
    xon.tweenT = 0;
    return found;
}

// Transition xon from tet mode back to oct mode after loop completion
function _returnXonToOct(xon) {
    xon._mode = 'oct';
    xon._assignedFace = null;
    xon._quarkType = null;
    xon._loopType = null;
    xon._loopSeq = null;
    xon._loopStep = 0;
    xon.col = 0xffffff; // white for oct mode

    // Update spark color to white
    if (xon.sparkMat) xon.sparkMat.color.setHex(0xffffff);

    // If xon is at a non-oct node (e.g. returning from idle_tet mid-loop),
    // move it to the nearest oct neighbor
    if (_octNodeSet && !_octNodeSet.has(xon.node)) {
        const nbs = baseNeighbors[xon.node] || [];
        for (const nb of nbs) {
            if (_octNodeSet.has(nb.node)) {
                xon.prevNode = xon.node;
                xon.node = nb.node;
                xon.trail.push(nb.node);
                xon.trailColHistory.push(xon.col);
                if (xon.trail.length > XON_TRAIL_LENGTH) { xon.trail.shift(); xon.trailColHistory.shift(); }
                break;
            }
        }
    }
}

// Start an idle tet loop for a xon boxed in on the oct surface.
// CONSTRAINT: xons can ONLY idle in already-actualized tets — faces whose
// SCs are already in electronImpliedSet or activeSet. No new geometry created.
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
            electronImpliedSet.has(scId) || activeSet.has(scId) || impliedSet.has(scId));
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
        let bestSeq = null, bestFace = null;
        for (const face of shuffled) {
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
                xon._mode = 'idle_tet';
                xon._loopSeq = seq;
                xon._loopStep = 0;
                xon._assignedFace = face;
                xon._quarkType = null;
                xon._loopType = null;
                return true;
            }
            if (!bestSeq) {
                bestSeq = LOOP_SEQUENCES[shuffledTypes[0]](rotated);
                bestFace = face;
            }
        }
        if (bestSeq) {
            xon._mode = 'idle_tet';
            xon._loopSeq = bestSeq;
            xon._loopStep = 0;
            xon._assignedFace = bestFace;
            xon._quarkType = null;
            xon._loopType = null;
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
            !electronImpliedSet.has(scId) && !activeSet.has(scId) && !impliedSet.has(scId));
        // Try to materialise all missing SCs
        let allOk = true;
        const justAdded = [];
        for (const scId of missingSCs) {
            if (canMaterialiseQuick(scId)) {
                electronImpliedSet.add(scId);
                stateVersion++; // invalidate cache for next check
                justAdded.push(scId);
            } else if (excitationSeverForRoom(scId)) {
                if (canMaterialiseQuick(scId)) {
                    electronImpliedSet.add(scId);
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
                electronImpliedSet.delete(scId);
                stateVersion++; // invalidate cache
            }
        }
    }

    if (newlyActualized.length > 0 && tryFaces(newlyActualized)) return true;

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
                _finalCleanupXon(xon);
                _demoXons.splice(xi, 1);
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
                const alpha = sparkOp * (0.15 + 0.85 * (i / Math.max(n - 1, 1)) ** 1.6);
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
            const px = pf[0] + (pt[0] - pf[0]) * s;
            const py = pf[1] + (pt[1] - pf[1]) * s;
            const pz = pf[2] + (pt[2] - pf[2]) * s;
            xon.group.position.set(px, py, pz);
        }

        // Sparkle flash + flicker
        xon.flashT = Math.max(0, xon.flashT - dt * 6.0);
        const flicker = 0.85 + Math.random() * 0.3;
        const pulse = (0.22 + xon.flashT * 0.26) * flicker;
        xon.spark.scale.set(pulse, pulse, 1);
        xon.sparkMat.opacity = (0.6 + xon.flashT * 0.4) * flicker * sparkOp;
        xon.sparkMat.rotation = Math.random() * Math.PI * 2;

        // Trail: fading vertex-colored path (12 steps)
        // Per-segment color from trailColHistory — segments retain their original color
        const n = xon.trail.length + 1;
        for (let i = 0; i < xon.trail.length; i++) {
            const np = pos[xon.trail[i]];
            if (!np) continue;
            xon.trailPos[i * 3] = np[0];
            xon.trailPos[i * 3 + 1] = np[1];
            xon.trailPos[i * 3 + 2] = np[2];
            const segCol = (xon.trailColHistory && xon.trailColHistory[i]) || xon.col;
            const cr = ((segCol >> 16) & 0xff) / 255;
            const cg = ((segCol >> 8) & 0xff) / 255;
            const cb = (segCol & 0xff) / 255;
            const alpha = sparkOp * (0.15 + 0.85 * (i / (n - 1)) ** 1.6);
            xon.trailCol[i * 3] = cr * alpha;
            xon.trailCol[i * 3 + 1] = cg * alpha;
            xon.trailCol[i * 3 + 2] = cb * alpha;
        }
        // Current interpolated position as trail head
        const last = xon.trail.length;
        if (last < XON_TRAIL_LENGTH) {
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
    for (const g of _demoGluons) {
        if (g.sprite) { scene.remove(g.sprite); g.sprite.material.dispose(); }
    }
    _demoGluons = [];
    _demoPrevFaces = new Set();
}

// Map speed slider (1-100) to demo interval: 1→200ms (slow), 50→20ms, 100→2ms (turbo)
function _getDemoIntervalMs() {
    const slider = document.getElementById('excitation-speed-slider');
    if (!slider) return 20;
    const t = +slider.value / 100;
    return Math.max(2, Math.round(Math.exp(Math.log(200) * (1 - t) + Math.log(2) * t)));
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
const L1_INNER_FACES = [1, 2, 3, 4];

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
 * Build a physically valid 16-window schedule (64 ticks per cycle).
 * 8 triple windows (all L1-valid triples) + 8 single-tet windows
 * (2 per inner face for coverage equalization).
 * Each face gets exactly 4 activations per cycle.
 * Fisher-Yates shuffled for stochastic ordering.
 * Returns array of 16 entries: {faces: [f1, f2?, f3?]}
 */
function buildPhysicalSchedule() {
    const windows = [];
    // 8 triple windows — hadron activations
    for (const triple of L1_VALID_TRIPLES) {
        windows.push({ faces: [...triple] });
    }
    // 8 single-tet windows — inner face coverage equalization
    // Each inner face (1-4) gets 2 singles to match outer faces' 4 total
    for (const f of L1_INNER_FACES) {
        windows.push({ faces: [f] });
        windows.push({ faces: [f] });
    }
    // Fisher-Yates shuffle
    for (let i = windows.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [windows[i], windows[j]] = [windows[j], windows[i]];
    }
    return windows;
}

/**
 * Start the pattern demo: sets up lattice, computes schedule, runs high-speed loop.
 * Called AFTER simulateNucleus() has built the octahedron.
 */
function startDemoLoop() {
    // Build L1-valid physical schedule (8 windows = 32 ticks, reshuffled each cycle)
    _demoSchedule = buildPhysicalSchedule();

    // Init visit counters + per-face shuffled decks for stochastic type assignment
    _demoVisits = {};
    _demoFaceDecks = {};
    for (let f = 1; f <= 8; f++) {
        _demoVisits[f] = { pu: 0, pd: 0, nu: 0, nd: 0, total: 0 };
        _demoFaceDecks[f] = [];  // empty → will reshuffle on first draw
    }
    _demoTick = 0;
    _demoPauliViolations = 0;
    _demoSpreadViolations = 0;
    _demoTypeBalanceHistory = [];
    _demoWindowTypes = null;  // current window's face→type assignments
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

    // Stop excitation clock (we drive our own loop)
    if (typeof stopExcitationClock === 'function') stopExcitationClock();

    // Do NOT pre-open all 8 tet SCs — only 1-3 tets can coexist at a time.
    // Tets activate/deactivate per window via electronImpliedSet, and the
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
                + `<button id="demo-l3-btn" style="font-size:8px; padding:1px 6px; margin:0 2px; background:#0a1a2a; color:#556677; border:1px solid #2a3a4a; border-radius:3px; cursor:pointer;">L3</button>`;
            ds.parentNode.insertBefore(toggleDiv, ds.nextSibling);
            document.getElementById('demo-l2-btn').addEventListener('click', () => _setDemoLattice(2));
            document.getElementById('demo-l3-btn').addEventListener('click', () => _setDemoLattice(3));
        }
        _updateDemoLatticeButtons();
    }

    // Update left panel header
    const dpTitle = document.querySelector('#deuteron-panel > div:first-child');
    if (dpTitle) dpTitle.textContent = '0 Planck seconds';

    // Demo 3.0 visual setup: shapes subtle, graph visible, spheres ghostly
    const shapesSlider = document.getElementById('void-opacity-slider');
    if (shapesSlider) { shapesSlider.value = 20; shapesSlider.dispatchEvent(new Event('input')); }
    const spheresSlider = document.getElementById('sphere-opacity-slider');
    if (spheresSlider) { spheresSlider.value = 1; spheresSlider.dispatchEvent(new Event('input')); }
    const graphSlider = document.getElementById('graph-opacity-slider');
    if (graphSlider) { graphSlider.value = 10; graphSlider.dispatchEvent(new Event('input')); }

    // Zoom camera out for better demo overview
    sph.r = Math.max(12, latticeLevel * 4.5);
    applyCamera();

    // Default to maximum speed for fast iteration
    const speedSlider = document.getElementById('excitation-speed-slider');
    if (speedSlider) { speedSlider.value = 100; speedSlider.dispatchEvent(new Event('input')); }
    // Default lifespan: 4 ticks between each trail point removal
    const lifespanSlider = document.getElementById('tracer-lifespan-slider');
    if (lifespanSlider) { lifespanSlider.value = 4; lifespanSlider.dispatchEvent(new Event('input')); }
    // Demo 3.1: Spawn 6 persistent xons on oct cage nodes
    _initPersistentXons();

    const intervalMs = _getDemoIntervalMs();
    _demoInterval = setInterval(demoTick, intervalMs);
    console.log(`[demo] Pattern demo started at ${intervalMs}ms interval`);

    // Auto-run unit tests — HALT DEMO if any test fails
    try {
        const testResult = runDemo3Tests();
        if (testResult.failed.length > 0) {
            console.error(`[demo] HALTED: ${testResult.failed.length} test(s) failed: ${testResult.failed.join(', ')}`);
            stopDemo();
            return;
        }
    } catch (e) { console.warn('[demo] Test suite error:', e); }

    // Activate live guards (T19, T21, T26, T27) — start with null during grace
    if (typeof _liveGuards !== 'undefined') {
        for (const key of Object.keys(_liveGuards)) {
            _liveGuards[key].ok = null;
            _liveGuards[key].msg = 'grace period';
            _liveGuards[key].failed = false;
            if (key === 'T21') _liveGuards[key]._octSnapshot = null;
        }
        _liveGuardsActive = true;
        _liveGuardRender();
    }
}

// L2/L3 toggle for demo mode — switches lattice and restarts demo
function _updateDemoLatticeButtons() {
    const l2 = document.getElementById('demo-l2-btn');
    const l3 = document.getElementById('demo-l3-btn');
    if (!l2 || !l3) return;
    const lv = +document.getElementById('lattice-slider').value;
    l2.style.background = lv === 2 ? '#1a2a3a' : '#0a1a2a';
    l2.style.color = lv === 2 ? '#88bbdd' : '#556677';
    l2.style.borderColor = lv === 2 ? '#3a5a7a' : '#2a3a4a';
    l3.style.background = lv === 3 ? '#1a2a3a' : '#0a1a2a';
    l3.style.color = lv === 3 ? '#88bbdd' : '#556677';
    l3.style.borderColor = lv === 3 ? '#3a5a7a' : '#2a3a4a';
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

function demoTick() {
    if (!_demoActive || !_demoSchedule) return;
    if (simHalted) return;

    // Clear stale movement flags from previous tick so WB processing isn't blocked
    for (const xon of _demoXons) xon._movedThisTick = false;

    // Snapshot xon positions BEFORE advancement for live guard T26/T27
    if (typeof _liveGuardSnapshot === 'function') _liveGuardSnapshot();

    let _solverNeeded = false;

    const CYCLE_LEN = 64;       // 16 windows × 4 ticks
    const WINDOW_LEN = 4;       // ticks per actualization window
    const WINDOWS_PER_CYCLE = 16;

    const tickInCycle = _demoTick % CYCLE_LEN;
    const windowIdx = Math.floor(tickInCycle / WINDOW_LEN);
    const tickInWindow = tickInCycle % WINDOW_LEN;

    // ── On window boundary: assign new config + spawn xons ──
    if (tickInWindow === 0) {
        // Reshuffle schedule at start of each cycle
        if (windowIdx === 0) {
            _demoSchedule = buildPhysicalSchedule();
        }

        const window = _demoSchedule[windowIdx];
        const faces = window.faces;

        // Determine hadron type from group composition
        // 2A+1B = proton; 1A+2B = neutron
        const aCount = faces.filter(f => A_SET.has(f)).length;
        const isProton = aCount >= faces.length / 2;

        // Stochastic type assignment
        _demoWindowTypes = {};
        if (faces.length === 3) {
            // Triple: minority quark on random face
            // Proton (uud): 2 pu + 1 pd; Neutron (udd): 1 nu + 2 nd
            const minorityIdx = Math.floor(Math.random() * 3);
            if (isProton) {
                for (let i = 0; i < 3; i++) {
                    _demoWindowTypes[faces[i]] = (i === minorityIdx) ? 'pd' : 'pu';
                }
            } else {
                for (let i = 0; i < 3; i++) {
                    _demoWindowTypes[faces[i]] = (i === minorityIdx) ? 'nu' : 'nd';
                }
            }
        } else {
            // Single-tet (inner face catch-up): OPPOSITE hadron deck for uniform coverage.
            // Inner A-faces only see proton types from triples → singles compensate
            // with neutron types. Inner B-faces only see neutron → compensate with proton.
            // Result: every face converges to identical distribution:
            //   33.3% pu, 16.7% pd, 16.7% nu, 33.3% nd
            // Global 2:1 ratio maintained: pu:pd = nd:nu = 2:1
            const f = faces[0];
            if (!_demoFaceDecks[f] || _demoFaceDecks[f].length === 0) {
                _demoFaceDecks[f] = A_SET.has(f)
                    ? ['nd', 'nd', 'nu']    // A-face (proton triples) → neutron singles
                    : ['pu', 'pu', 'pd'];   // B-face (neutron triples) → proton singles
                // Fisher-Yates shuffle
                for (let i = _demoFaceDecks[f].length - 1; i > 0; i--) {
                    const j = Math.floor(Math.random() * (i + 1));
                    [_demoFaceDecks[f][i], _demoFaceDecks[f][j]] = [_demoFaceDecks[f][j], _demoFaceDecks[f][i]];
                }
            }
            _demoWindowTypes[f] = _demoFaceDecks[f].pop();
        }

        // ── Accumulate visit counts (once per window, not per tick) ──
        for (const [fStr, qType] of Object.entries(_demoWindowTypes)) {
            const f = parseInt(fStr);
            if (_demoVisits[f]) {
                _demoVisits[f][qType]++;
                _demoVisits[f].total++;
            }
        }

        // ── Track visited faces ──
        for (const f of faces) _demoVisitedFaces.add(f);
        // Oct cage is EMERGENT — xons in oct mode materialize oct SCs
        // through traversal. No free activeSet.add here.

        // ╔══════════════════════════════════════════════════════════════════╗
        // ║  NON-DELETABLE: UNIFIED ARCHITECTURE — SHAPES DRIVE SPHERES     ║
        // ║                                                                  ║
        // ║  The demo MUST manage tet SCs in electronImpliedSet and          ║
        // ║  re-solve the lattice so spheres physically respond.             ║
        // ║  Without this, the demo is visually detached from physics —      ║
        // ║  a CARDINAL SIN. NEVER remove the solver coupling.              ║
        // ║                                                                  ║
        // ║  MINIMAL ACTION PRINCIPLE: relinquish only what is necessary     ║
        // ║  for the desired transformation. Do not clear-and-rebuild.       ║
        // ║  Diff-based: remove old tet SCs, add new ones, keep rest.       ║
        // ║  No cascade detection (detectImplied) during demo — tet SCs     ║
        // ║  are nearest-neighbor (distance 1 in REST) so they don't        ║
        // ║  deform FCC geometry. Cascade implies extra shortcuts that       ║
        // ║  CAN deform geometry and violate Kepler. Minimal action.        ║
        // ║                                                                  ║
        // ║  NEVER remove, bypass, or skip the solver coupling below.        ║
        // ╚══════════════════════════════════════════════════════════════════╝

        // ── Demo 3.1: Persistent xons — return tet-mode xons to oct, assign new ones ──
        const newFaceSet = new Set(faces);

        // 1. Return all tet & idle_tet xons to oct (their loops are done)
        //    Then scatter returned xons so no two share a node (Pauli).
        const returningXons = [];
        for (const xon of _demoXons) {
            if (xon.alive && (xon._mode === 'tet' || xon._mode === 'idle_tet')) {
                _returnXonToOct(xon);
                returningXons.push(xon);
            }
        }
        // Scatter: if multiple xons landed on the same node, move extras
        {
            const taken = new Set();
            for (const xon of _demoXons) {
                if (!xon.alive) continue;
                if (!taken.has(xon.node)) {
                    taken.add(xon.node);
                } else {
                    // Collision — move this xon to a free oct neighbor (only via active edges)
                    const allNb = baseNeighbors[xon.node].filter(nb => _octNodeSet.has(nb.node));
                    const scNb = (scByVert[xon.node] || []).filter(sc => activeSet.has(sc.id)).map(sc => sc.a === xon.node ? sc.b : sc.a).filter(n => _octNodeSet.has(n));
                    const combined = [...allNb.map(nb => nb.node), ...scNb];
                    let moved = false;
                    for (const n of combined) {
                        if (!taken.has(n)) {
                            xon.prevNode = xon.node;
                            xon.node = n;
                            taken.add(n);
                            moved = true;
                            break;
                        }
                    }
                    if (!moved) taken.add(xon.node); // can't move, accept collision
                }
            }
        }

        // 2. Relinquish SCs for faces LEAVING active set
        for (const [fIdStr, fd] of Object.entries(_nucleusTetFaceData)) {
            const fId = parseInt(fIdStr);
            if (!newFaceSet.has(fId)) {
                for (const scId of fd.scIds) {
                    if (electronImpliedSet.delete(scId)) {
                        _solverNeeded = true;
                        stateVersion++; // invalidate _getBasePairs cache
                    }
                }
            }
        }

        // 3. Assign oct-mode xons to new tet faces (Pauli-aware)
        for (const f of faces) {
            const qType = _demoWindowTypes[f];
            if (!qType) continue;

            const fd = _nucleusTetFaceData[f];
            if (!fd) continue;

            // Find best idle xon: prefer one already on an oct node of this face
            const faceOctNodes = new Set(fd.cycle.filter(n => _octNodeSet.has(n)));
            let bestXon = null;
            let bestScore = -Infinity;
            for (const xon of _demoXons) {
                if (!xon.alive || xon._mode !== 'oct') continue;
                let score = 0;
                if (faceOctNodes.has(xon.node)) score = 10;
                else {
                    const nbs = baseNeighbors[xon.node];
                    for (const nb of nbs) {
                        if (faceOctNodes.has(nb.node)) { score = 5; break; }
                    }
                }
                if (score > bestScore) { bestScore = score; bestXon = xon; }
            }

            if (bestXon) {
                _assignXonToTet(bestXon, f, qType);
            }
        }

        _demoPrevFaces = newFaceSet;
    }

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

    // Reset single-hop-per-tick flag AFTER window boundary (WB movements are exempt from T26/T27)
    for (const xon of _demoXons) xon._movedThisTick = false;

    let occupied = _occupiedNodes();

    // ══════════════════════════════════════════════════════════════════
    //  COORDINATED MOVE PLANNER
    //  All moves are planned before execution to prevent Pauli violations.
    //  Priority: tet/idle_tet (fixed path) > oct (flexible).
    // ══════════════════════════════════════════════════════════════════

    const planned = new Set();  // globally reserved destination nodes
    let anyMoved = false;

    // ── PHASE 1: Plan tet/idle_tet moves (fixed sequences) ──
    const tetPlans = [];
    const tetBlockedBy = new Map(); // toNode → xon (tet xons blocked by oct occupants)
    for (const xon of _demoXons) {
        if (!xon.alive) continue;
        if (xon._mode !== 'tet' && xon._mode !== 'idle_tet') continue;
        if (xon._loopStep >= 4) continue;
        const fromNode = xon._loopSeq[xon._loopStep];
        const toNode = xon._loopSeq[xon._loopStep + 1];
        tetPlans.push({ xon, fromNode, toNode, approved: false });
    }

    // Approve tet moves to free destinations; track oct-blocked ones
    for (const plan of tetPlans) {
        if (planned.has(plan.toNode)) continue; // another tet already claimed this
        const occCount = occupied.get(plan.toNode) || 0;
        if (occCount === 0) {
            plan.approved = true;
            planned.add(plan.toNode);
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
        if (!electronImpliedSet.has(scId) && !activeSet.has(scId) && !impliedSet.has(scId)) {
            let activated = false;
            if (canMaterialiseQuick(scId)) {
                electronImpliedSet.add(scId);
                stateVersion++; // invalidate _getBasePairs cache for subsequent checks
                _solverNeeded = true;
                activated = true;
            } else if (excitationSeverForRoom(scId)) {
                if (canMaterialiseQuick(scId)) {
                    electronImpliedSet.add(scId);
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

    // ── PHASE 2: Coordinated oct movement planning ──
    // Remove all oct xons from occupied so they can see each other's positions as available
    // (enables position swaps and chain moves)
    const octXons = _demoXons.filter(x => x.alive && x._mode === 'oct');
    for (const xon of octXons) _occDel(occupied, xon.node);

    const octPlans = octXons.map(xon => ({
        xon,
        candidates: _getOctCandidates(xon, occupied, planned),
        assigned: null,
        fromNode: xon.node,
    }));

    // Restore occupied for later use
    for (const xon of octXons) _occAdd(occupied, xon.node);

    // Pre-filter candidates: remove those where vacuum would definitely reject
    for (const plan of octPlans) {
        plan.candidates = plan.candidates.filter(c => {
            if (!c._needsMaterialise) return true; // base edge or already active SC
            if (c._scId === undefined) return true;
            return canMaterialiseQuick(c._scId); // keep only if vacuum would allow
        });
    }

    // Maximum bipartite matching with arbitrary-depth backtracking (Kuhn's algorithm).
    // Finds augmenting paths so the maximum number of oct xons get a valid destination.
    _maxBipartiteAssignment(octPlans, planned);
    const octClaimed = new Set();
    for (const plan of octPlans) {
        if (plan.assigned) octClaimed.add(plan.assigned.node);
    }

    // Verify needsOctVacate: if an oct xon was supposed to move but couldn't, revoke tet approval
    for (const plan of tetPlans) {
        if (!plan._needsOctVacate) continue;
        const blocker = plan._needsOctVacate;
        const octPlan = octPlans.find(p => p.xon === blocker);
        if (!octPlan || !octPlan.assigned) {
            // Oct xon couldn't move — revoke tet approval
            plan.approved = false;
            planned.delete(plan.toNode);
        }
    }

    // Build a combined blocked set for idle_tet planning
    const allBlocked = new Map(occupied);
    for (const n of planned) _occAdd(allBlocked, n);
    for (const n of octClaimed) _occAdd(allBlocked, n);
    for (const plan of octPlans) {
        if (plan.assigned) _occDel(allBlocked, plan.fromNode);
    }

    // Proactive congestion relief: if the oct cage is crowded,
    // send some assigned oct xons into idle_tet to reduce density.
    // With 12 oct nodes and 6 xons, >4 xons on oct is congested.
    const octOnCage = octPlans.filter(p => p.assigned || (!p.assigned && !p.idleTet)).length;
    if (octOnCage > 4) {
        // Demote the lowest-scored assigned oct xons to idle_tet
        const demotable = octPlans
            .filter(p => p.assigned && p.assigned.score !== undefined)
            .sort((a, b) => (a.assigned.score || 0) - (b.assigned.score || 0));
        for (const plan of demotable) {
            if (octOnCage - (demotable.indexOf(plan) < demotable.length ? 1 : 0) <= 4) break;
            // Try idle_tet for this xon instead of its oct move
            if (_startIdleTetLoop(plan.xon, allBlocked)) {
                const dest = plan.xon._loopSeq[plan.xon._loopStep + 1];
                if (dest !== undefined && !allBlocked.has(dest)) {
                    octClaimed.delete(plan.assigned.node);
                    plan.assigned = null;
                    plan.idleTet = true;
                    _occAdd(allBlocked, dest);
                } else {
                    plan.xon._mode = 'oct';
                    plan.xon._loopSeq = null;
                    plan.xon._loopStep = 0;
                    plan.xon._assignedFace = null;
                }
            }
        }
    }

    // Unassigned oct xons: try idle_tet with Pauli-aware face selection
    for (const plan of octPlans) {
        if (plan.assigned || plan.idleTet) continue;
        if (_startIdleTetLoop(plan.xon, allBlocked)) {
            const dest = plan.xon._loopSeq[plan.xon._loopStep + 1];
            if (dest !== undefined && !allBlocked.has(dest)) {
                plan.idleTet = true;
                _occAdd(allBlocked, dest);
            } else {
                plan.xon._mode = 'oct';
                plan.xon._loopSeq = null;
                plan.xon._loopStep = 0;
                plan.xon._assignedFace = null;
            }
        }
    }

    // If idle_tet manifestation added new SCs, flag solver
    if (_idleTetManifested) _solverNeeded = true;

    // ── PHASE 3: Execute all planned moves ──
    // Oct moves execute FIRST (to vacate nodes for tet xons).
    // If an oct move fails (vacuum rejection), revoke dependent tet approvals.

    // Build reverse map: oct xon → tet plan that depends on it vacating
    const octToTetDep = new Map(); // oct xon → tet plan
    for (const plan of tetPlans) {
        if (plan._needsOctVacate) octToTetDep.set(plan._needsOctVacate, plan);
    }

    // Execute oct moves first
    for (const plan of octPlans) {
        if (plan.assigned) {
            const target = plan.assigned;
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
                if (plan.xon._solverNeeded) {
                    _solverNeeded = true;
                    plan.xon._solverNeeded = false;
                }
            }
            _occAdd(occupied, plan.xon.node);
        } else if (plan.idleTet) {
            // Verify SC is still active (may have been severed by oct move negotiation)
            if (!_canAdvanceSafely(plan.xon)) {
                _returnXonToOct(plan.xon); // abort idle_tet — SC was deactivated
                continue;
            }
            _occDel(occupied, plan.xon.node);
            _advanceXon(plan.xon);
            _occAdd(occupied, plan.xon.node);
            anyMoved = true;
        }
    }

    // Then execute approved tet moves (nodes should now be vacated)
    for (const plan of tetPlans) {
        if (!plan.approved) continue;
        // Final Pauli safety check before executing
        if ((occupied.get(plan.toNode) || 0) > 0) {
            continue; // destination still occupied — skip to prevent collision
        }
        // Verify SC is still active (may have been severed by oct move negotiation)
        if (!_canAdvanceSafely(plan.xon)) continue;
        _advanceXon(plan.xon);
        _occDel(occupied, plan.xon.prevNode);
        _occAdd(occupied, plan.xon.node);
        anyMoved = true;
    }

    // ── PHASE 4: Auto-return completed loops, scatter collisions ──
    for (const xon of _demoXons) {
        if (!xon.alive) continue;
        if ((xon._mode === 'tet' || xon._mode === 'idle_tet') && xon._loopStep >= 4) {
            _returnXonToOct(xon);
        }
    }
    occupied = _occupiedNodes();
    // Collect collision groups: all oct xons at nodes with >1 occupancy
    const collisionNodes = new Set();
    for (const xon of _demoXons) {
        if (xon.alive && xon._mode === 'oct' && (occupied.get(xon.node) || 0) > 1) {
            collisionNodes.add(xon.node);
        }
    }
    for (const cNode of collisionNodes) {
        // Find all oct xons at this node; prefer moving ones that haven't moved yet
        const atNode = _demoXons.filter(x => x.alive && x._mode === 'oct' && x.node === cNode);
        // Sort: moved xons FIRST (keep them — can't move again), unmoved last (scatter them)
        atNode.sort((a, b) => (a._movedThisTick ? 0 : 1) - (b._movedThisTick ? 0 : 1));
        // Move extras (keep the first one)
        for (let k = 1; k < atNode.length; k++) {
            const xon = atNode[k];
            if (xon._movedThisTick) continue; // already moved this tick — can't move again
            const allNb = (baseNeighbors[xon.node] || []).filter(nb => _octNodeSet.has(nb.node));
            const scNb = (scByVert[xon.node] || []).filter(sc => activeSet.has(sc.id)).map(sc => sc.a === xon.node ? sc.b : sc.a).filter(n => _octNodeSet.has(n));
            const candidates = [...allNb.map(nb => nb.node), ...scNb];
            let moved = false;
            for (const n of candidates) {
                if (!occupied.has(n)) {
                    _occDel(occupied, xon.node);
                    xon.prevNode = xon.node;
                    xon.node = n;
                    _occAdd(occupied, n);
                    xon.trail.push(n);
                    xon.trailColHistory.push(xon.col);
                    if (xon.trail.length > XON_TRAIL_LENGTH) { xon.trail.shift(); xon.trailColHistory.shift(); }
                    xon.tweenT = 0;
                    moved = true;
                    break;
                }
            }
            if (!moved && !xon._movedThisTick) {
                // Try idle_tet as escape
                if (_startIdleTetLoop(xon, occupied)) {
                    const dest = xon._loopSeq[xon._loopStep + 1];
                    if (dest !== undefined && !occupied.has(dest) && _canAdvanceSafely(xon)) {
                        _advanceXon(xon);
                        _occDel(occupied, xon.prevNode);
                        _occAdd(occupied, xon.node);
                    }
                }
            }
        }
        // If collision persists and the KEPT xon hasn't moved, try moving IT instead
        const remaining = _demoXons.filter(x => x.alive && x._mode === 'oct' && x.node === cNode);
        if (remaining.length > 1) {
            for (const xon of remaining) {
                if (xon._movedThisTick) continue;
                const allNb = (baseNeighbors[xon.node] || []).filter(nb => _octNodeSet.has(nb.node));
                const scNb = (scByVert[xon.node] || []).filter(sc => activeSet.has(sc.id)).map(sc => sc.a === xon.node ? sc.b : sc.a).filter(n => _octNodeSet.has(n));
                const candidates = [...allNb.map(nb => nb.node), ...scNb];
                for (const n of candidates) {
                    if (!occupied.has(n)) {
                        _occDel(occupied, xon.node);
                        xon.prevNode = xon.node;
                        xon.node = n;
                        _occAdd(occupied, n);
                        xon.trail.push(n);
                        xon.trailColHistory.push(xon.col);
                        if (xon.trail.length > XON_TRAIL_LENGTH) { xon.trail.shift(); xon.trailColHistory.shift(); }
                        xon.tweenT = 0;
                        break;
                    }
                }
                if ((occupied.get(cNode) || 0) <= 1) break; // resolved
            }
        }
    }

    // ── PHASE 5: Global deadlock detection (non-fatal, warn only) ──
    if (typeof _globalStuckTicks === 'undefined') _globalStuckTicks = 0;
    if (!anyMoved && _demoXons.some(x => x.alive)) {
        _globalStuckTicks++;
        if (_globalStuckTicks === 8) {
            console.warn('[STALL] No xon could move for 8 ticks — waiting for vacuum/excitation to free space');
        }
    } else {
        _globalStuckTicks = 0;
    }

    // ── Advance gluons along oct edges (also negotiates with vacuum) ──
    if (_advanceGluons()) _solverNeeded = true;

    // ── Run solver if any SCs changed (unified architecture) ──
    if (_solverNeeded) {
        bumpState();
        const scPairs = [];
        activeSet.forEach(id => { const s = SC_BY_ID[id]; scPairs.push([s.a, s.b]); });
        electronImpliedSet.forEach(id => { const s = SC_BY_ID[id]; scPairs.push([s.a, s.b]); });
        const { p: pSolved } = _solve(scPairs, 5000, true); // noBailout: full convergence for Kepler
        impliedSet.clear(); impliedBy.clear();
        electronImpliedSet.forEach(id => {
            if (!activeSet.has(id)) { impliedSet.add(id); impliedBy.set(id, new Set()); }
        });
        applyPositions(pSolved);
        updateSpheres();
    }

    // ── Decay dying xon trails (every simulation tick, not per-frame) ──
    _decayDyingXons();

    // ── Color tets with progressive opacity (ramps as xon loop completes) ──
    // Void wireframes are the SOLE edge rendering system during demo.
    if (_demoWindowTypes) {
        for (const [fIdStr, fd] of Object.entries(_nucleusTetFaceData)) {
            const fId = parseInt(fIdStr);
            const qType = _demoWindowTypes[fId];
            if (qType) {
                _ruleAnnotations.tetColors.set(fd.voidIdx, QUARK_COLORS[qType]);
                // Progressive opacity: ramps from 0.3 to 0.85 over 4 ticks
                // Find the xon for this face to get its loop progress
                const xon = _demoXons.find(x => x.alive && x._assignedFace === fId);
                const step = xon ? xon._loopStep : 0;
                const opacity = 0.3 + (step / 4) * 0.55; // 0.3 → 0.85
                _ruleAnnotations.tetOpacity.set(fd.voidIdx, opacity);
            } else {
                _ruleAnnotations.tetColors.set(fd.voidIdx, 0x1a1a2a);
                _ruleAnnotations.tetOpacity.set(fd.voidIdx, 0.0);
            }
        }
        _ruleAnnotations.dirty = true;
        if (typeof updateVoidSpheres === 'function') updateVoidSpheres();
    }

    _demoTick++;

    // Update Planck-second ticker (both right-panel status and left-panel title)
    const _tickerEl = document.getElementById('nucleus-status');
    if (_tickerEl) _tickerEl.textContent = `${_demoTick} Planck seconds`;
    const _dpTitle = document.querySelector('#deuteron-panel > div:first-child');
    if (_dpTitle) _dpTitle.textContent = `${_demoTick} Planck seconds`;

    // Live guard checks (T19, T21, T26, T27) — after tick advances xons
    if (typeof _liveGuardCheck === 'function') _liveGuardCheck();

    // Capture temporal K frame every tick (tracks lattice state movie)
    if (typeof captureTemporalFrame === 'function') captureTemporalFrame();

    // Update UI at window boundaries (every 4 ticks)
    if (_demoTick % WINDOW_LEN === 0) {
        updateDemoPanel();
        updateStatus();
    }
}

function updateDemoPanel() {
    const CYCLE_LEN = 64;
    const cycles = Math.floor(_demoTick / CYCLE_LEN);
    const windows = Math.floor(_demoTick / 4);

    // ── Update demo-status (right panel, below button) ──
    const ds = document.getElementById('demo-status');
    if (ds) {
        ds.innerHTML = `<span style="color:#88bbdd;">cycle ${cycles}</span> · ` +
            `<span style="color:#667788;">window ${windows}</span>`;
    }

    // ── Update left panel coverage bars (skip during test execution) ──
    if (_testRunning) { _demoTick++; return; }
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
        if (m === 0) return 1;
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
        + `<span style="color:#6a8a9a;">cycles</span>`
        + `<span style="color:#88aacc;">${cycles}</span>`
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

function stopDemo() {
    _demoActive = false;
    if (typeof _liveGuardsActive !== 'undefined') _liveGuardsActive = false;
    if (_demoInterval) { clearInterval(_demoInterval); _demoInterval = null; }
    const ds = document.getElementById('demo-status');
    if (ds) ds.style.display = 'none';
    // Clean up Demo 3.0 xons and gluons
    _cleanupDemo3();
    // Clean up tet SCs from electronImpliedSet + oct SCs from activeSet
    for (const [, fd] of Object.entries(_nucleusTetFaceData)) {
        for (const scId of fd.scIds) {
            electronImpliedSet.delete(scId);
        }
    }
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
