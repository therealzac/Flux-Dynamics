// flux-demo-xon.js — Xon lifecycle: spawn, destroy, advance, trails, occupancy

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
// weak-class: _t60Ejected, _weakLeftOct
// tet-class: _assignedFace, _quarkType, _loopType, _loopSeq, _loopStep, _tetActualized
function _clearModeProps(xon) {
    xon._t60Ejected = false;
    xon._weakLeftOct = false;
    xon._pendingWeakEjection = false;
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
        _dirBalance: new Array(10).fill(0), // xonic movement balance: 4 base + 6 SC dirs
        _modeStats: { oct: 0, tet: 0, idle_tet: 0, weak: 0 }, // ticks spent in each mode
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
        // PPO reward signal: tet loop completion
        if (typeof _ppoTetCompletionsThisTick !== 'undefined') _ppoTetCompletionsThisTick++;
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
            _dirBalance: new Array(10).fill(0),
            _modeStats: { oct: 0, tet: 0, idle_tet: 0, weak: 0 },
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
