// flux-xons.js — Excitation clock tick, batched stepping, stochastic pruning, sidebar, utilities
// ─── excitationClockTick: batched stepping + stochastic pruning ────────
// Steps up to BATCH_SIZE excitations per tick (round-robin). This avoids
// hanging the browser at L2+ (107 nodes × solver calls = too slow for 1 tick).
// Pruning rules checked on ALL excitations every tick:
//   1. Degenerate loop: trail visits <4 unique nodes after 12+ steps → dissolve
//   2. Seeking (no void): stochastic decay with avg lifespan from energy slider
//   3. Stuck: stochastic decay with shorter avg lifespan (1/4 of seeking)
//   4. Dedup: multiple excitations on same tet → keep only first
// Excitations that have claimed a void (tet/oct) are IMMORTAL — no decay.
// Per-tick budget: up to MAT_BUDGET excitations may call the solver,
// the rest walk on existing open shortcuts (O(1) per step).
// WALK_BATCH controls how many walkers step per tick for throughput.
const MAT_BUDGET = 2;
const WALK_BATCH = 20;
function excitationClockTick(){
    if(!excitations.length||simHalted) return;

    // Defer UI updates during the entire tick — flush once at the end
    _deferUIUpdates = true;
    _uiDirty = false;

    // ── RULE tick() HOOK ──────────────────────────────────────────
    // If the active rule implements tick(), it gets FULL CONTROL over
    // the lattice state BEFORE excitation movement.  This is the
    // expanded rule interface — rules can directly open/close SCs,
    // spawn/kill excitations, and manipulate anything except the
    // base geometry and rendering pipeline.
    //
    // tick() receives a context with:
    //   - Direct references: activeSet, impliedSet, ALL_SC, pos, etc.
    //   - Helper functions: openSC(), closeSC(), toggleSC()
    //   - Metrics: temporalK, avgHamming, stuckTicks
    //   - Control flag: skipExcitations (set true to skip standard movement)
    //
    // Rules with tick() get first crack at state, then excitations run
    // (unless skipExcitations is set). GAUGE always runs after.
    const activeRule = getActiveRule();
    let skipExcitations = false;
    {
        // ── DENSITY SAFEGUARDS (fundamental layer) ──
        const DENSITY_MAX = 0.65;
        const DENSITY_MIN = 0.02;
        const maxOpen = Math.floor(ALL_SC.length * DENSITY_MAX);
        const minOpen = Math.ceil(ALL_SC.length * DENSITY_MIN);

        let _tickChanges = 0;

        // Combined allOpen set for models that check if an SC is active/implied
        const _allOpenTick = new Set([...activeSet, ...impliedSet, ...xonImpliedSet]);

        const tickCtx = {
            // ── Direct state references (read-only recommended) ──
            activeSet,
            impliedSet,
            xonImpliedSet,
            excitations,
            ALL_SC,
            pos,
            REST,
            N,                        // number of nodes
            voidTypes,                // 'tetrahedral'|'octahedral' per node
            stateVersion,
            allOpen: _allOpenTick,    // combined active+implied set

            // ── Nucleus model data ──
            quarks: (typeof NucleusSimulator !== 'undefined') ? NucleusSimulator.quarkExcitations : [],
            createVirtualPair: (typeof NucleusSimulator !== 'undefined') ? NucleusSimulator.buildSetupCtx().createVirtualPair : function(){ return [null,null]; },
            nodeTetVoids: _nodeTetVoids,
            nodeOctVoids: _nodeOctVoids,
            voidData: voidNeighborData,
            basePosNeighbor: basePosNeighbor,

            // ── Metrics ──
            temporalK: _temporalKValue,
            avgHamming: _avgHamming,
            hammingDistance: _hammingDistance,
            stuckTicks: _stuckTickCount,
            frameCount: _temporalFrames.length,
            density: activeSet.size / ALL_SC.length, // current active density

            // ── Helpers: safe SC manipulation with density guards ──
            // These enforce density limits to prevent lattice crash.
            openSC(scId){
                if(activeSet.has(scId)) return false;
                if(activeSet.size >= maxOpen) return false; // density cap
                if(_tickChanges >= MAX_TICK_CHANGES) return false; // throttle
                activeSet.add(scId);
                _tickChanges++;
                bumpState();
                return true;
            },
            closeSC(scId){
                if(!activeSet.has(scId)) return false;
                if(activeSet.size <= minOpen) return false; // density floor
                if(_tickChanges >= MAX_TICK_CHANGES) return false; // throttle
                activeSet.delete(scId);
                _tickChanges++;
                bumpState();
                return true;
            },
            toggleSC(scId){
                if(_tickChanges >= MAX_TICK_CHANGES) return false; // throttle
                if(activeSet.has(scId)){
                    if(activeSet.size <= minOpen) return false;
                    activeSet.delete(scId);
                } else {
                    if(activeSet.size >= maxOpen) return false;
                    activeSet.add(scId);
                }
                _tickChanges++;
                bumpState();
                return true;
            },
            isOpen(scId){ return activeSet.has(scId) || impliedSet.has(scId); },
            isActive(scId){ return activeSet.has(scId); },
            get changesRemaining(){ return MAX_TICK_CHANGES - _tickChanges; },
            maxChanges: MAX_TICK_CHANGES,

            // ── Physics update (call after bulk SC changes) ──
            applyPhysics(){
                const pFinal = detectImplied();
                applyPositions(pFinal);
                _tickChanges = 0; // prevent double-solve in auto-apply
            },

            // ── Control flags ──
            skipExcitations: false,

            // ── Animation annotations ──
            // Rules use these to visually show what they're doing.
            // Colors are 0xRRGGBB hex values.
            // Animation quality is a tournament criterion.
            annotate: {
                /** Set custom color for a shortcut line. */
                scColor(scId, hexColor){
                    _ruleAnnotations.scColors.set(scId, hexColor);
                    _ruleAnnotations.dirty = true;
                },
                /** Set custom color for a node sphere. */
                nodeColor(nodeIdx, hexColor){
                    _ruleAnnotations.nodeColors.set(nodeIdx, hexColor);
                    _ruleAnnotations.dirty = true;
                },
                /** Set custom opacity for a shortcut (0-1). */
                scOpacity(scId, opacity){
                    _ruleAnnotations.scOpacity.set(scId, opacity);
                    _ruleAnnotations.dirty = true;
                },
                // nodeScale API removed permanently — sphere sizes must NEVER vary.
                /** Clear all SC color annotations. */
                clearSC(){
                    _ruleAnnotations.scColors.clear();
                    _ruleAnnotations.scOpacity.clear();
                    _ruleAnnotations.dirty = true;
                },
                /** Clear all node annotations. */
                clearNodes(){
                    _ruleAnnotations.nodeColors.clear();
            
                    _ruleAnnotations.dirty = true;
                },
                // ── Void annotations ──
                /** Set custom color for a tetrahedral void mesh. */
                tetColor(voidIndex, hexColor){
                    _ruleAnnotations.tetColors.set(voidIndex, hexColor);
                    _ruleAnnotations.dirty = true;
                },
                /** Set custom color for an octahedral void mesh. */
                octColor(voidIndex, hexColor){
                    _ruleAnnotations.octColors.set(voidIndex, hexColor);
                    _ruleAnnotations.dirty = true;
                },
                /** Set custom opacity for a tet void (0-1). */
                tetOpacity(voidIndex, opacity){
                    _ruleAnnotations.tetOpacity.set(voidIndex, opacity);
                    _ruleAnnotations.dirty = true;
                },
                /** Set per-face colors for an oct void (array of hex colors). */
                octFaces(voidIndex, faceColorArray){
                    _ruleAnnotations.octFaceColors.set(voidIndex, faceColorArray);
                    _ruleAnnotations.dirty = true;
                },
                // ── Excitation annotations ──
                /** Override an excitation's spark color. */
                excitationColor(excIdx, hexColor){
                    _ruleAnnotations.excitationColors.set(excIdx, hexColor);
                    _ruleAnnotations.dirty = true;
                },
                /** Scale an excitation's spark size. */
                excitationScale(excIdx, scale){
                    _ruleAnnotations.excitationScale.set(excIdx, scale);
                    _ruleAnnotations.dirty = true;
                },
                /** Clear all annotations. */
                clear(){
                    _ruleAnnotations.scColors.clear();
                    _ruleAnnotations.nodeColors.clear();
                    _ruleAnnotations.scOpacity.clear();
            
                    _ruleAnnotations.tetColors.clear();
                    _ruleAnnotations.octColors.clear();
                    _ruleAnnotations.tetOpacity.clear();
                    _ruleAnnotations.octFaceColors.clear();
                    _ruleAnnotations.excitationColors.clear();
                    _ruleAnnotations.excitationScale.clear();
                    _ruleAnnotations.dirty = true;
                },
                // Pre-defined gauge group color palettes for convenience
                colors: {
                    // SU(3) color charges
                    RED:     0xff3333,
                    GREEN:   0x33ff33,
                    BLUE:    0x3333ff,
                    ANTI_RED:   0x00cccc,  // cyan (anti-red)
                    ANTI_GREEN: 0xcc00cc,  // magenta (anti-green)
                    ANTI_BLUE:  0xcccc00,  // yellow (anti-blue)
                    WHITE:   0xffffff,      // color-neutral
                    // SU(2) weak isospin
                    LEFT:    0xff8800,      // left-handed (orange)
                    RIGHT:   0x0088ff,      // right-handed (blue)
                    W_PLUS:  0xffcc00,      // W+ boson
                    W_MINUS: 0xff0066,      // W- boson
                    Z_BOSON: 0x88ff88,      // Z boson
                    // U(1) hypercharge
                    PHOTON:  0xffffaa,      // electromagnetic (golden)
                    HYPER_POS: 0xffaaff,    // positive hypercharge
                    HYPER_NEG: 0xaaffff,    // negative hypercharge
                    // Particles
                    FERMION: 0xff6644,      // tet void (fermion)
                    BOSON:   0x4466ff,       // oct void (boson)
                    GLUON:   0x44ffaa,       // gluon field
                    DOMAIN_WALL: 0xff44ff,   // domain boundary
                    CREATION: 0xffff00,      // pair creation flash
                    ANNIHILATION: 0xff0000,  // pair annihilation flash
                }
            },
        };
        // Clear all annotations before rule + GAUGE build new ones
        tickCtx.annotate.clear();
        if(activeRule.tick) {
            activeRule.tick(tickCtx);
            skipExcitations = tickCtx.skipExcitations;
        }

        // ── ANIMATION QUALITY MEASUREMENT ──
        // Measure how well the rule is using annotations to show its logic.
        // Coverage: fraction of SCs+nodes with custom colors
        // Dynamism: how much annotations changed since last tick
        {
            const totalElements = ALL_SC.length + N;
            const annotatedCount = _ruleAnnotations.scColors.size + _ruleAnnotations.nodeColors.size;
            _animCoverage = totalElements > 0 ? annotatedCount / totalElements : 0;

            // Build hash of current annotations for dynamism
            let hashParts = [];
            for(const [k, v] of _ruleAnnotations.scColors) hashParts.push(`s${k}:${v}`);
            for(const [k, v] of _ruleAnnotations.nodeColors) hashParts.push(`n${k}:${v}`);
            const currentHash = hashParts.join(',');

            // Dynamism = did annotations change? (binary per tick, averaged over time)
            _animDynamism = (currentHash !== _prevAnnotationHash && annotatedCount > 0) ? 1.0 : 0.0;
            _prevAnnotationHash = currentHash;

            // Combined animation quality: coverage matters, dynamism matters more
            const animQ = _animCoverage * 0.4 + _animDynamism * 0.6;
            _animHistory.push(animQ);
            if(_animHistory.length > 50) _animHistory.shift();
            _avgAnimQuality = _animHistory.reduce((a, b) => a + b, 0) / _animHistory.length;
        }

        // Auto-apply physics if rule made changes (and didn't already call applyPhysics)
        if(_tickChanges > 0){
            const pFinal = detectImplied();
            applyPositions(pFinal);
        }
        // ── POST-TICK STRAIN RECOVERY ──
        // Tick-based rules can create solver violations via bulk changes.
        // Attempt recovery here (before updateStatus halts the sim).
        // Strategy: check base edges; if any exceed tolerance, reset positions
        // from current activeSet by re-solving from REST positions.
        const TICK_TOL = 1e-3;
        let tickViolation = false;
        for(const [i,j] of BASE_EDGES){
            if(Math.abs(vd(pos[i],pos[j]) - 1.0) > TICK_TOL){ tickViolation = true; break; }
        }
        if(tickViolation){
            // Recovery: re-solve from REST positions with current constraints
            pos = REST.map(v => [...v]);
            const pRecov = detectImplied();
            applyPositions(pRecov);
        }

        // ── SHARED GAUGE GROUP POST-TICK ──
        // Always runs. Updates gauge state (SU(3), SU(2), U(1)) and
        // annotates excitations with gauge force colors,
        // tet voids with fermion type+gen, oct voids with gluon octet,
        // domain wall node scaling.
        // Skips SC colors if the rule already set them.
        if(typeof GAUGE !== 'undefined' && GAUGE.postTick){
            // Clear GAUGE-managed annotations (preserve rule SC colors)
            _ruleAnnotations.excitationColors.clear();
            _ruleAnnotations.excitationScale.clear();
            _ruleAnnotations.tetColors.clear();
            _ruleAnnotations.octColors.clear();
            _ruleAnnotations.tetOpacity.clear();
            _ruleAnnotations.octFaceColors.clear();
    

            tickCtx._ruleSetSCColors = _ruleAnnotations.scColors.size > 0;
            GAUGE.postTick(tickCtx);
        }
    }

    // ── Standard excitation movement (skippable by tick() rules) ──
    if(!skipExcitations){

    // ══════════════════════════════════════════════════════════════════
    // QUARK SINGLE-HOP STEPPING (LEGAL LATTICE MOVES)
    // ══════════════════════════════════════════════════════════════════
    // Quarks traverse within tet K₄ voids via single-hop movement.
    // Each tet has 6 edges: 4 base + 2 SCs. From any node, 3 directions.
    // Cost minimization: prefer free edges (base or open SC), only
    // materialise closed SCs when all free paths are Pauli-blocked.
    // Pauli exclusion: no two excitations may occupy the same node.
    //
    // NOTE on directional constraint (future): positive/negative vector
    // filtering may apply to base directions only (xons=negative,
    // positrons=positive). Not yet enforced for quarks.

    // Phase 1: Build occupancy map
    _quarkNodeOccupancy.clear();
    const _quarkList = [];
    for (const e of excitations) {
        if (!e._isQuark) continue;
        _quarkNodeOccupancy.set(e.node, e);
        _quarkList.push(e);
    }

    // Phase 2: Shuffle for random priority (fair over time)
    for (let i = _quarkList.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [_quarkList[i], _quarkList[j]] = [_quarkList[j], _quarkList[i]];
    }

    // Increment nucleus tick counter (for oct edge tracing)
    _nucleusTick++;

    // Phase 3: Pluggable quark movement via QUARK_ALGO_REGISTRY
    const _allOpenQ = getAllOpen();
    const _qDeparting = new Set();
    const _qArriving = new Set();
    const _qNodeFree = (node, self) => {
        if (_qArriving.has(node)) return false;
        if (_qDeparting.has(node)) return true;
        const occ = _quarkNodeOccupancy.get(node);
        return !occ || occ === self;
    };

    const _algo = QUARK_ALGO_REGISTRY[_activeQuarkAlgo] || QUARK_ALGO_REGISTRY[0];
    const _algoCtx = {
        allOpen: _allOpenQ,
        quarkList: _quarkList,
        faceCoverage: _faceCoverageTotal,
        nucleusTick: _nucleusTick,
        tetFaceData: _nucleusTetFaceData,
        hopGroups: DEUTERON_HOP_GROUPS,
        octSCIds: _octSCIds,
        canMaterialise: canMaterialiseQuick,
        materialise: excitationMaterialiseSC,
        severForRoom: excitationSeverForRoom,
    };

    for (const e of _quarkList) {
        const faceData = _nucleusTetFaceData[e._currentFace];
        if (!faceData || !e.voidNodes) continue;
        const tetScIdSet = new Set(faceData.scIds);

        // Classify edges: free vs costly
        // Include ALL SCs (tet SCs + oct SCs) — not just tet SCs
        const octSCSet = new Set(_octSCIds);
        const freeOptions = [];
        const costlyOptions = [];
        for (const dest of e.voidNodes) {
            if (dest === e.node) continue;
            if (!_qNodeFree(dest, e)) continue;
            const pid = pairId(e.node, dest);
            const scId = scPairToId.get(pid);
            if (scId !== undefined && (tetScIdSet.has(scId) || octSCSet.has(scId))) {
                if (_allOpenQ.has(scId)) freeOptions.push({ dest, scId });
                else costlyOptions.push({ dest, scId });
            } else {
                freeOptions.push({ dest, scId: null });
            }
        }

        const tetSCsOpen = faceData.scIds.filter(id => _allOpenQ.has(id)).length;

        // Priority: materialise oct SCs first (bosonic cage must stay intact)
        let chosen = null;
        const octCostly = costlyOptions.filter(o => octSCSet.has(o.scId));
        if (octCostly.length > 0) {
            for (const opt of octCostly) {
                if (_algoCtx.canMaterialise(opt.scId)) {
                    if (_algoCtx.materialise(e, opt.scId)) { chosen = opt; break; }
                } else if (_algoCtx.severForRoom(opt.scId)) {
                    if (_algoCtx.materialise(e, opt.scId)) { chosen = opt; break; }
                }
            }
        }

        // Cage rush: during first 12 ticks, if no oct SC from here,
        // prefer moving toward oct nodes that have closed oct SCs
        if (!chosen && _nucleusTick < 12 && _octNodeSet) {
            const cageComplete = _octSCIds.every(id => _allOpenQ.has(id));
            if (!cageComplete) {
                // Prefer free edges leading to oct nodes with closed oct SC neighbors
                const octFree = freeOptions.filter(o => _octNodeSet.has(o.dest));
                if (octFree.length > 0) {
                    chosen = octFree[Math.floor(Math.random() * octFree.length)];
                }
            }
        }

        // Delegate to active algorithm if oct SCs already handled
        if (!chosen) chosen = _algo.stepQuark(e, freeOptions, costlyOptions, tetSCsOpen, faceData, _algoCtx);

        // ── Illegal traversal check: if chosen edge is an SC, it must be open NOW ──
        // (can't use _allOpenQ — it's a stale snapshot from before materialisation)
        if (chosen && chosen.scId !== null && chosen.scId !== undefined && (tetScIdSet.has(chosen.scId) || octSCSet.has(chosen.scId))) {
            const liveOpen = getAllOpen();
            if (!liveOpen.has(chosen.scId)) {
                console.warn(`[illegal] ${e._xonId} traversed closed SC ${chosen.scId} (face ${e._currentFace}, ${e.node}→${chosen.dest})`);
                _illegalTraversalCount++;
                chosen = null; // block the move
            }
        }

        // Apply move
        e.trail.push(e.node);
        if (e.trail.length > QUARK_TRAIL_LENGTH) e.trail.shift();
        e.prevNode = e.node;

        if (chosen) {
            _qDeparting.add(e.node);
            _qArriving.add(chosen.dest);
            if (_octNodeSet && _octNodeSet.has(e.node) && _octNodeSet.has(chosen.dest)) {
                const edgeKey = e.node < chosen.dest ? e.node+','+chosen.dest : chosen.dest+','+e.node;
                _octEdgeLastTraced.set(edgeKey, _nucleusTick);
            }
            // Teleportation check: destination must be in current tet
            if (!e.voidNodes.has(chosen.dest)) {
                console.warn(`[teleport] ${e._xonId} jumped to node ${chosen.dest} outside tet face ${e._currentFace}`);
                _teleportationCount++;
            }
            e.node = chosen.dest;
            e.tweenT = 0; e.flashT = 1.0;
            e.totalSteps++; e.stuckTicks = 0;
            e._stepsInFace = (e._stepsInFace || 0) + 1;
        } else {
            e.tweenT = 0; e.flashT = 0.2; e.stuckTicks++;
            _xonStallCount++;
        }

        if (e._tetVoidIdx !== undefined) {
            _ruleAnnotations.tetColors.set(e._tetVoidIdx, TET_QUARK_COLORS[e._currentFace] || 0xffffff);
            _ruleAnnotations.tetOpacity.set(e._tetVoidIdx, 0.7);
        }
        const _eIdx = excitations.indexOf(e);
        if (_eIdx >= 0) _ruleAnnotations.excitationColors.set(_eIdx, e.col);
    }

    // Rebuild occupancy map after moves
    _quarkNodeOccupancy.clear();
    for (const e of _quarkList) _quarkNodeOccupancy.set(e.node, e);

    // ── Coverage-driven quark hopping (delegated to algorithm) ──
    for (const [groupId, groupFaces] of Object.entries(DEUTERON_HOP_GROUPS)) {
        const occupiedFaces = new Set();
        for (const e of _quarkList) {
            if (e._hopGroup === groupId) occupiedFaces.add(e._currentFace);
        }

        // Each quark asks the algorithm if it should hop
        for (const hopper of _quarkList) {
            if (hopper._hopGroup !== groupId) continue;
            const hopResult = _algo.shouldHop(hopper, groupFaces, occupiedFaces, _algoCtx);
            if (!hopResult) continue;

            const targetFace = hopResult.targetFace;
            const newFaceData = _nucleusTetFaceData[targetFace];
            if (!newFaceData) continue;

            // Find shared oct node for continuous transition
            const curFaceDef = DEUTERON_TET_FACES[hopper._currentFace];
            const tgtFaceDef = DEUTERON_TET_FACES[targetFace];
            let sharedNode = null;
            if (curFaceDef && tgtFaceDef) {
                for (const n of curFaceDef.octNodes) {
                    if (tgtFaceDef.octNodes.includes(n)) { sharedNode = n; break; }
                }
            }

            // Release old tet SCs if no other quark remains
            const oldFace = hopper._currentFace;
            const oldVoidIdx = hopper._tetVoidIdx;
            const oldFaceData = _nucleusTetFaceData[oldFace];
            const otherOnOldFace = _quarkList.some(
                e => e !== hopper && e._currentFace === oldFace
            );
            if (!otherOnOldFace) {
                if (oldVoidIdx !== undefined) {
                    _ruleAnnotations.tetColors.delete(oldVoidIdx);
                    _ruleAnnotations.tetOpacity.set(oldVoidIdx, 0.0);
                }
                if (oldFaceData) {
                    const octSCSet = new Set(_octSCIds);
                    for (const scId of oldFaceData.scIds) {
                        // Never sever oct SCs — the bosonic cage must stay intact
                        if (octSCSet.has(scId)) continue;
                        if (xonImpliedSet.has(scId)) {
                            xonImpliedSet.delete(scId);
                            impliedSet.delete(scId);
                            impliedBy.delete(scId);
                            for (const ex of excitations) {
                                if (ex.ownShortcut === scId) ex.ownShortcut = null;
                            }
                        }
                    }
                    bumpState();
                }
            }

            // Switch to new tet
            occupiedFaces.delete(oldFace);
            occupiedFaces.add(targetFace);
            hopper._currentFace = targetFace;
            hopper._tetVoidIdx = newFaceData.voidIdx;
            hopper.voidNodes = new Set(newFaceData.allNodes);
            hopper._stepsInFace = 0;

            // Land on shared oct node if available and unoccupied
            const sharedOccupant = _quarkNodeOccupancy.get(sharedNode);
            if (sharedNode !== null && (!sharedOccupant || sharedOccupant === hopper)) {
                hopper.node = sharedNode;
            }

            // If still outside new tet (no shared node, or it was occupied),
            // pick any unoccupied node in the new tet
            if (!hopper.voidNodes.has(hopper.node)) {
                const newNodes = [...hopper.voidNodes];
                const free = newNodes.filter(n => !_quarkNodeOccupancy.has(n) || _quarkNodeOccupancy.get(n) === hopper);
                hopper.node = free.length > 0 ? free[0] : newNodes[0];
            }

            // Teleportation check: after hop, node must be in new tet
            if (!hopper.voidNodes.has(hopper.node)) {
                console.warn(`[teleport] ${hopper._xonId} hop to face ${targetFace} but node ${hopper.node} not in new tet (nodes: ${[...hopper.voidNodes]})`);
                _teleportationCount++;
            }

            _ruleAnnotations.tetColors.set(newFaceData.voidIdx, TET_QUARK_COLORS[targetFace] || 0xffffff);
            _ruleAnnotations.tetOpacity.set(newFaceData.voidIdx, 0.7);

            break; // one hop per group per tick
        }
    }

    if (_quarkList.length > 0) _ruleAnnotations.dirty = true;

    // Average lifespan (in steps): energy=0% → 8 steps, energy=100% → 80 steps
    // Seeking excitations have per-step dissolution probability = 1/avgLifespan.
    // Stuck excitations decay 4× faster (avgLifespan/4).
    const avgLifespan = 8 + Math.round(excitationEnergy * 72);

    // Step a batch: first MAT_BUDGET get solver access, rest are walk-only
    const n = excitations.length;
    const batchSize = Math.min(n, MAT_BUDGET + WALK_BATCH);
    let matUsed = 0;
    for(let b = 0; b < batchSize; b++){
        excitationClockCursor = excitationClockCursor % excitations.length;
        const e = excitations[excitationClockCursor];
        if(!e) break;
        if(e._isQuark) { excitationClockCursor++; continue; } // already stepped above
        // Excitations with voids don't need materialization (just walking)
        const needsMat = !e.zeroPoint && matUsed < MAT_BUDGET;
        // Only count choice-making steps (Phase 1), not travelDest completions
        // (Phase 2). Each full shortcut traversal = 2 ticks but only 1 "step".
        // This ensures lifespan counts actual traversals, not half-steps.
        const wasCompleting = e.travelDest !== null;
        excitationStep(e, needsMat);
        if(needsMat) matUsed++;
        if(!wasCompleting) e.totalSteps++;
        excitationClockCursor++;
    }

    // Check ALL excitations for eviction (not just the stepped ones)
    const toRemove = new Set();
    const tetOwners = new Map();
    const octOwners = new Map();
    for(const e of excitations){
        // Excitations bound to a void are immortal — skip decay
        if(e.zeroPoint !== null){
            // Dedup: fermion (tet) = 1 per void, boson (oct) = up to 8 per void
            if(e.voidScIds){
                const key = [...e.voidScIds].sort((a,b)=>a-b).join(',');
                if(e.voidType === 'tet'){
                    if(tetOwners.has(key)) toRemove.add(e.id);
                    else tetOwners.set(key, e.id);
                } else if(e.voidType === 'oct'){
                    if(!octOwners.has(key)) octOwners.set(key, []);
                    const owners = octOwners.get(key);
                    if(owners.length >= 8) toRemove.add(e.id);
                    else owners.push(e.id);
                }
            }
            continue;
        }
        // Quark excitations are immortal (managed by virtual pair lifecycle)
        if(e._isQuark) continue;
        // In arena mode excitations are immortal (no lifespan decay).
        // They can still be evicted by void dedup above, but not by
        // loop detection or stochastic decay.
        if(activeRuleIndex > 0) continue;

        // Degenerate loop detection: if a seeking excitation's trail
        // visits fewer than 4 unique nodes after enough steps, it's stuck
        // in a cycle that can never form a valid void (both tet and oct
        // require 4 nodes). Dissolve immediately.
        if(e.totalSteps >= 12 && e.trail.length >= 8){
            const uniqueNodes = new Set(e.trail);
            if(uniqueNodes.size < 4){
                toRemove.add(e.id);
                continue;
            }
        }
        // Seeking excitations: stochastic decay (radioactive-decay model)
        // Must survive at least 4 steps (grace period to find a void)
        if(e.totalSteps >= 4){
            const life = e.stuckTicks > 0 ? Math.max(1, avgLifespan / 4) : avgLifespan;
            if(Math.random() < 1 / life){ toRemove.add(e.id); continue; }
        }
    }

    // Batch remove
    if(toRemove.size){
        if(toRemove.size <= 3){
            for(const id of toRemove) toast('excitation e'+id+' dissolved');
        } else {
            toast(toRemove.size+' excitations dissolved');
        }
        _batchRemoveMode = true;
        for(const id of toRemove) removeExcitation(id);
        _batchRemoveMode = false;
        updateExcitationSidebar();

        // If all excitations dissolved, clean up orphaned xon-implied SCs.
        // The excitation clock will stop (no excitations → no more ticks), so
        // strainMonitorCheck would never run again. Without cleanup, orphaned
        // SCs accumulate strain and trigger invariant violations.
        if(!excitations.length && xonImpliedSet.size){
            let cleaned = 0;
            while(xonImpliedSet.size){
                const before = xonImpliedSet.size;
                strainMonitorCheck();
                if(xonImpliedSet.size >= before) break; // couldn't evict (all protected)
                cleaned++;
                if(cleaned > 50) break; // safety cap
            }
            // If strain is still above halt threshold after cleanup, clear ALL
            // orphaned xon-implied SCs as a last resort.
            if(xonImpliedSet.size){
                let sumErr = 0;
                for(const [i,j] of BASE_EDGES) sumErr += Math.abs(vd(pos[i],pos[j]) - 1.0);
                if(sumErr / BASE_EDGES.length > 1e-3){
                    for(const id of [...xonImpliedSet]){
                        xonImpliedSet.delete(id);
                        impliedSet.delete(id);
                        impliedBy.delete(id);
                    }
                    bumpState();
                    const pFinal = detectImplied();
                    applyPositions(pFinal);
                }
            }
        }
    }

    } // end if(!skipExcitations)

    // ── VIRTUAL PAIR LIFECYCLE ──
    // Decay virtual excitations and handle quark-antiquark annihilation
    if(typeof NucleusSimulator !== 'undefined' && NucleusSimulator.active){
        const qExc = NucleusSimulator.quarkExcitations;
        // 1. Decay virtual excitations
        for(let i = excitations.length - 1; i >= 0; i--){
            const e = excitations[i];
            if(e && e._isVirtual && e._lifetime !== undefined){
                e._lifetime--;
                if(e._lifetime <= 0){
                    // Remove from quark excitations list
                    const qi = qExc.indexOf(e);
                    if(qi >= 0) qExc.splice(qi, 1);
                    // Visual removal
                    if(e.group) scene.remove(e.group);
                    if(e.trailLine) scene.remove(e.trailLine);
                    excitations.splice(i, 1);
                }
            }
        }
        // 2. Annihilation: quark + antiquark at same node with opposite direction
        const nodeMap = new Map(); // nodeIdx → [excitation, ...]
        for(const e of excitations){
            if(!e._isQuark || !e._isVirtual) continue;
            if(e.node === undefined) continue;
            if(!nodeMap.has(e.node)) nodeMap.set(e.node, []);
            nodeMap.get(e.node).push(e);
        }
        for(const [node, group] of nodeMap){
            const particles = group.filter(e => e._direction === 1);
            const antiparticles = group.filter(e => e._direction === -1);
            const pairs = Math.min(particles.length, antiparticles.length);
            for(let p = 0; p < pairs; p++){
                const a = particles[p], b = antiparticles[p];
                // Remove both
                for(const x of [a, b]){
                    const qi = qExc.indexOf(x);
                    if(qi >= 0) qExc.splice(qi, 1);
                    const ei = excitations.indexOf(x);
                    if(ei >= 0){
                        if(x.group) scene.remove(x.group);
                        if(x.trailLine) scene.remove(x.trailLine);
                        excitations.splice(ei, 1);
                    }
                }
                // Annihilation flash: activate nearby SCs as "binding energy"
                for(const sc of ALL_SC){
                    if(sc.a === node || sc.b === node){
                        if(!xonImpliedSet.has(sc.id) && Math.random() < 0.3){
                            xonImpliedSet.add(sc.id);
                            if (typeof _scAttribution !== 'undefined') _scAttribution.set(sc.id, { reason: 'annihilation', tick: typeof _demoTick !== 'undefined' ? _demoTick : 0 });
                        }
                        break; // just 1 SC per annihilation
                    }
                }
            }
        }
        // 3. Update nucleus metrics display + deuteron panel
        NucleusSimulator.updateMetrics();
        NucleusSimulator.updateDeuteronPanel();
    }

    // Run strain monitor every N ticks
    if(++_strainCheckCounter >= STRAIN_CHECK_INTERVAL){
        _strainCheckCounter=0;
        strainMonitorCheck();
    }

    // Sync jiggle: when both jiggle + excitations are active, run jiggle
    // inside the excitation tick instead of on its own independent timer.
    // This prevents interleaved solver calls from two competing intervals.
    if(jiggleActive && excitations.length) jiggleStep();

    // Capture temporal K frame (after all state mutations for this tick)
    captureTemporalFrame();

    // Tournament watchdog — check if current rule trial should advance
    tournamentCheckTick();

    // Flush deferred UI updates — single batch for the entire tick.
    // IMPORTANT: This MUST rebuild lines + spheres together to prevent
    // sphere-graph desynchronization. If only spheres update (via
    // _spheresDirty in render loop) without graph lines rebuilding,
    // the user sees spheres move while edges/lines stay frozen.
    // FIX (recurring bug): Always rebuild if EITHER flag is set, and
    // always sync both spheres AND graph together.
    _deferUIUpdates = false;
    if(_uiDirty || _spheresDirty || _ruleAnnotations.dirty){
        _uiDirty = false;
        rebuildBaseLines(); rebuildShortcutLines();
        updateVoidSpheres(); updateCandidates(); updateSpheres(); updateStatus();
    }

    // ── Sync health check: verify sphere ↔ pos[] alignment ──
    // Spot-checks a sample of sphere InstancedMesh positions against pos[].
    // Updates _syncMaxDeviation and _syncStatus for the deuteron panel.
    if(typeof NucleusSimulator !== 'undefined' && NucleusSimulator.active && bgMesh){
        let maxDev = 0;
        if(!excitationClockTick._chkMat) excitationClockTick._chkMat = new THREE.Matrix4();
        if(!excitationClockTick._chkPos) excitationClockTick._chkPos = new THREE.Vector3();
        const _chkMat = excitationClockTick._chkMat;
        const _chkPos = excitationClockTick._chkPos;
        // Sample up to 8 random nodes + all quark nodes
        const sampleSet = new Set();
        for(const q of (NucleusSimulator.quarkExcitations || [])){
            sampleSet.add(q.node);
        }
        for(let s = 0; s < 8 && sampleSet.size < 14; s++){
            sampleSet.add(Math.floor(Math.random() * N));
        }
        for(const idx of sampleSet){
            if(idx >= N) continue;
            bgMesh.getMatrixAt(idx, _chkMat);
            _chkPos.setFromMatrixPosition(_chkMat);
            const dx = _chkPos.x - pos[idx][0];
            const dy = _chkPos.y - pos[idx][1];
            const dz = _chkPos.z - pos[idx][2];
            // If scale is 0 (hidden in bg, shown in fg), check fg instead
            const scaleX = _chkMat.elements[0]; // approximate scale
            if(Math.abs(scaleX) < 0.01 && fgMesh){
                fgMesh.getMatrixAt(idx, _chkMat);
                _chkPos.setFromMatrixPosition(_chkMat);
                const dx2 = _chkPos.x - pos[idx][0];
                const dy2 = _chkPos.y - pos[idx][1];
                const dz2 = _chkPos.z - pos[idx][2];
                const dev2 = Math.sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2);
                maxDev = Math.max(maxDev, dev2);
            } else {
                const dev = Math.sqrt(dx*dx + dy*dy + dz*dz);
                maxDev = Math.max(maxDev, dev);
            }
        }
        _syncMaxDeviation = maxDev;
        _syncStatus = maxDev < 0.001 ? 'ok' : maxDev < 0.01 ? 'warn' : 'error';
    }

    // Big bang toggle: check for stale state and re-bang
    bigBangStaleCheck();
}
function startExcitationClock(){ if(excitationClockTimer||excitationPaused) return; excitationClockTimer=setInterval(excitationClockTick,XON_STEP_MS); }
function stopExcitationClock(){ clearInterval(excitationClockTimer); excitationClockTimer=null; excitationClockCursor=0; }
function syncExcitationPlayBtn(){
    const btn=document.getElementById('btn-excitation-play');
    btn.style.display=excitations.length?'':'none';
    btn.textContent=excitationPaused?'\u25b6':'\u23f8';
    btn.classList.toggle('active',excitationPaused);
}
function toggleExcitationPause(){
    excitationPaused=!excitationPaused;
    if(excitationPaused){ clearInterval(excitationClockTimer); excitationClockTimer=null; }
    else if(excitations.length){ startExcitationClock(); }
    syncExcitationPlayBtn();
}

function updateExcitationSidebar(){
    const el=document.getElementById('side-excitations'); el.innerHTML='';
    // (no longer hiding shortcuts when excitations active — both show in unified panel)
    syncExcitationPlayBtn();
    if(!excitations.length) return;
    const hdr=document.createElement('div'); hdr.className='el-header';
    hdr.innerHTML='excitations <span id="el-clear" title="remove all">✕</span>';
    el.appendChild(hdr);
    hdr.querySelector('#el-clear').addEventListener('click', removeAllExcitations);
    excitations.forEach(e=>{ const item=document.createElement('div'); item.className='el-item'; item.innerHTML=`<div class="el-dot" style="background:${XON_COLORS_CSS[e.colorIdx]}"></div><span class="el-label">e${e.id} · v${e.node}</span><span class="el-remove">remove</span>`; item.querySelector('.el-remove').addEventListener('click',()=>removeExcitation(e.id)); el.appendChild(item); });
}

function tickExcitations(dt){
    excitations.forEach(e=>{
        e.tweenT=Math.min(1,e.tweenT+dt/(XON_STEP_MS*0.001));
        const s=1-(1-e.tweenT)**3;
        const pfx=pos[e.prevNode][0],pfy=pos[e.prevNode][1],pfz=pos[e.prevNode][2];
        const ptx=pos[e.node][0],pty=pos[e.node][1],ptz=pos[e.node][2];
        const px=pfx+(ptx-pfx)*s, py=pfy+(pty-pfy)*s, pz=pfz+(ptz-pfz)*s;
        e.group.position.set(px,py,pz);
        // Sparkle flash: pulse + random flicker + linear travel sparkle
        e.flashT=Math.max(0,e.flashT-dt*6.0);
        const flicker=0.85+Math.random()*0.3; // random 0.85–1.15
        // Linear travel detection: consecutive moves in similar direction → extra sparkle
        let linearBoost = 1.0;
        if(e.trail.length >= 3){
            const t = e.trail;
            const len = t.length;
            // Check if last 3 trail positions form a roughly straight line
            const ax=pos[t[len-1]][0]-pos[t[len-2]][0], ay=pos[t[len-1]][1]-pos[t[len-2]][1], az=pos[t[len-1]][2]-pos[t[len-2]][2];
            const bx=pos[t[len-2]][0]-pos[t[len-3]][0], by=pos[t[len-2]][1]-pos[t[len-3]][1], bz=pos[t[len-2]][2]-pos[t[len-3]][2];
            const al=Math.sqrt(ax*ax+ay*ay+az*az)||1, bl=Math.sqrt(bx*bx+by*by+bz*bz)||1;
            const dot=(ax*bx+ay*by+az*bz)/(al*bl);
            if(dot > 0.8) linearBoost = 1.0 + (dot - 0.8) * 3.0; // up to 1.6x for perfectly straight
        }
        const pulse=(0.22+e.flashT*0.26)*flicker*linearBoost;
        e.spark.scale.set(pulse,pulse,1);
        const sparkSliderOp = (+document.getElementById('spark-opacity-slider').value) / 100;
        e.sparkMat.opacity=(0.6+e.flashT*0.4)*flicker*sparkSliderOp*Math.min(linearBoost, 1.3);
        // Enhanced sparkle for linear travel: faster rotation + scale jitter
        if(linearBoost > 1.1){
            e.sparkMat.rotation += Math.random()*Math.PI; // extra spin
            const jitter = 1.0 + Math.random()*0.15*(linearBoost-1.0);
            e.spark.scale.x *= jitter; e.spark.scale.y *= jitter;
        } else {
            e.sparkMat.rotation=Math.random()*Math.PI*2;
        }
        // ── Excitation annotation overrides ──
        const eIdx = excitations.indexOf(e);
        const annotExcCol = _ruleAnnotations.excitationColors.get(eIdx);
        const annotExcScale = _ruleAnnotations.excitationScale.get(eIdx);
        if(annotExcCol !== undefined){
            e.sparkMat.color.setHex(annotExcCol);
        }
        if(annotExcScale !== undefined){
            const aPulse = pulse * annotExcScale;
            e.spark.scale.set(aPulse, aPulse, 1);
        }
        // Trail: electrical path along base edges
        const useCol = annotExcCol !== undefined ? annotExcCol : e.col;
        const cr=((useCol>>16)&0xff)/255, cg=((useCol>>8)&0xff)/255, cb=(useCol&0xff)/255;
        const graphOp=+document.getElementById('trail-opacity-slider').value/100;
        const isQuarkTrail = !!e._isQuark;
        const n=e.trail.length+1;
        for(let i=0;i<e.trail.length;i++){
            const np=pos[e.trail[i]];
            e.trailPos[i*3]=np[0]; e.trailPos[i*3+1]=np[1]; e.trailPos[i*3+2]=np[2];
            // Quark trails: uniform brightness (string-like closed loops)
            // Normal trails: fade from dim to bright
            const alpha = isQuarkTrail ? graphOp * 0.9 : graphOp*(0.15+0.85*(i/(n-1))**1.6);
            e.trailCol[i*3]=cr*alpha; e.trailCol[i*3+1]=cg*alpha; e.trailCol[i*3+2]=cb*alpha;
        }
        const last=e.trail.length;
        e.trailPos[last*3]=px; e.trailPos[last*3+1]=py; e.trailPos[last*3+2]=pz;
        const lastAlpha = isQuarkTrail ? graphOp * 0.9 : graphOp;
        e.trailCol[last*3]=cr*lastAlpha; e.trailCol[last*3+1]=cg*lastAlpha; e.trailCol[last*3+2]=cb*lastAlpha;
        e.trailGeo.setDrawRange(0,n);
        e.trailGeo.attributes.position.needsUpdate=true; e.trailGeo.attributes.color.needsUpdate=true;
        if(e.tweenT>=1){ const lbl=document.querySelector(`#side-excitations .el-item:nth-child(${excitations.indexOf(e)+2}) .el-label`); if(lbl) lbl.textContent=`e${e.id} · v${e.node}`; }
    });
}

function toggleExcitationPlacement(){
    placingExcitation=!placingExcitation;
    document.getElementById('btn-add-excitation').classList.toggle('placing',placingExcitation);
    document.getElementById('hint').textContent=placingExcitation?'click a node to place excitation · Escape to cancel':'click sphere to select · click candidate (blue) to add shortcut · click edge to sever';
}

