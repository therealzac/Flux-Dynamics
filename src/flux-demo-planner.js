// flux-demo-planner.js — Move planner: bipartite matching, lookahead, face scoring, tet assignment

// Select the best loop permutation for a given quark type on a face.
// Scores each valid permutation by how well it balances the xon's direction counters.
// Returns a 5-node sequence (or falls back to LOOP_SEQUENCES if no permutations).
function _selectBestPermutation(xon, cycle, quarkType) {
    const topo = QUARK_TOPOLOGY[quarkType];
    const perms = topo ? LOOP_PERMUTATIONS[topo] : null;
    if (!perms || perms.length === 0) {
        return LOOP_SEQUENCES[quarkType](cycle);
    }
    // Random choreographer mode: pick a random permutation, skip scoring entirely
    if (_bfsTestRandomChoreographer && !_btActive) {
        const idx = Math.floor(_sRng() * perms.length);
        const seq = perms[idx](cycle);
        return (seq && seq.length === 5) ? seq : LOOP_SEQUENCES[quarkType](cycle);
    }

    let bestSeq = null, bestScore = -Infinity;
    for (const gen of perms) {
        const seq = gen(cycle);
        if (!seq || seq.length !== 5) continue;
        // Compute direction indices for the 4 edges
        const dirs = [];
        for (let s = 0; s < 4; s++) {
            const d = _identifyMoveDir(seq[s], seq[s + 1]);
            if (d >= 0) dirs.push(d);
        }
        const score = _dirBalanceScoreMulti(xon, dirs) + (_sRng() - 0.5) * 0.1;
        if (score > bestScore) {
            bestScore = score;
            bestSeq = seq;
        }
    }
    return bestSeq || LOOP_SEQUENCES[quarkType](cycle);
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

    const visited = new Set();
    for (const i of order) {
        visited.clear();
        augment(i, visited);
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
//    their specific fermionic loop (fork, hook, ham CW/CCW).
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
        // Uses the xon's actual loop sequence (fork, hook, ham CW/CCW).
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
// Checks backtracker exclusion + optional projected guard pre-validation.
function _moveViolatesGuards(xon, fromNode, toNode) {
    if (_btActive) {
        const xonIdx = _demoXons.indexOf(xon);
        if (_btIsMoveExcluded(xonIdx, toNode)) return true;
    }
    if (_ruleProjectedGuards) {
        const futures = [];
        for (const x of _demoXons) {
            if (!x.alive) continue;
            futures.push({
                xon: x,
                futureNode: x === xon ? toNode : x.node,
                fromNode: x === xon ? fromNode : x.node,
                futureMode: x._mode,
                futureColor: x.col,
            });
        }
        if (_validateProjectedGuards(futures).length > 0) return true;
    }
    return false;
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
        }
    }
    if (_ruleProjectedGuards) {
        const futures = [];
        for (const x of _demoXons) {
            if (!x.alive) continue;
            futures.push({
                xon: x,
                futureNode: x === xon ? toNode : x.node,
                fromNode: x === xon ? fromNode : x.node,
                futureMode: x._mode,
                futureColor: x.col,
            });
        }
        const violations = _validateProjectedGuards(futures);
        for (const v of violations) {
            reasons.push({ reason: `PROJECTED:${v.guard || '?'}: ${v.msg || JSON.stringify(v)}` });
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
        const status = blocked ? '\u2717' : '\u2713';
        lines.push(`  ${status} node ${nb.node}${tagStr}: ${checks.join(', ')}`);
    }
    lines.push(anyOpen ? `  \u2192 HAS viable moves` : `  \u2192 ALL BLOCKED \u2014 xon will be stuck!`);
    console.error(lines.join('\n'));
}

// ── Projected Guard Validator ──
// Iterates the PROJECTED_GUARD_CHECKS array (defined in flux-tests.js).
// Each check function receives the projected xon states and returns violations.
// Gated by _ruleProjectedGuards toggle (switchboard).
function _validateProjectedGuards(xonFutures) {
    if (typeof PROJECTED_GUARD_CHECKS === 'undefined' || !PROJECTED_GUARD_CHECKS.length) return [];
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

    // REACHABILITY: xon must be on a face oct node.
    // If face is NOT actualized, xon must be at the pole (cycle[0])
    // so the first step traverses the unique SC to activate the tet.
    const faceOctNodes = [];
    for (const n of fd.cycle) {
        if (_octNodeSet && _octNodeSet.has(n)) faceOctNodes.push(n);
    }
    const faceActualized = fd.scIds.every(scId =>
        activeSet.has(scId) || impliedSet.has(scId) || xonImpliedSet.has(scId));
    const onFace = faceOctNodes.includes(xon.node);
    if (!onFace) return null; // must be directly on a face oct node
    if (!faceActualized && xon.node !== fd.cycle[0]) return null; // must be at pole for non-actualized faces

    // Quark type selection
    const isProtonFace = A_SET.has(face);
    const candidates = isProtonFace ? ['pu1', 'pu2', 'pd'] : ['nd1', 'nd2', 'nu'];

    // Random choreographer mode: pick random compatible quark type, random score
    if (_bfsTestRandomChoreographer && !_btActive) {
        const compat = candidates.filter(qt => _bipartiteCompatible(face, qt));
        if (compat.length === 0) return null;
        const quarkType = compat[Math.floor(_sRng() * compat.length)];
        return { face, quarkType, score: _sRng() * 10, onFace };
    }

    // Pick most in-demand quark type that passes bipartite compatibility
    // Sort by deficit descending, pick first compatible
    const ranked = candidates.map(t => ({ t, d: _ratioTracker.deficit(t) }));
    ranked.sort((a, b) => b.d - a.d);
    let quarkType = null, bestDeficit = -Infinity;
    for (const { t, d } of ranked) {
        if (_bipartiteCompatible(face, t)) {
            quarkType = t;
            bestDeficit = d;
            break;
        }
    }
    if (!quarkType) return null; // no compatible quark type for this face

    // RL STRATEGIC SCORING: if strategic model is active, use it instead of heuristic
    if (typeof _rlStrategicModel !== 'undefined' && _rlStrategicModel &&
        typeof scoreStrategicRL === 'function' && typeof extractStrategicFeatures === 'function') {
        const features = extractStrategicFeatures(xon, face, quarkType, occupied);
        // PPO training mode: sample from policy + collect trajectory
        if (typeof _ppoTraining !== 'undefined' && _ppoTraining &&
            typeof _ppoStrategicBuffer !== 'undefined' && _rlStrategicModel.sampleAction) {
            const result = _rlStrategicModel.sampleAction(features, 1);
            _ppoStrategicBuffer.push(features, result.actionIdx, result.logProb, result.value, 1);
            return { face, quarkType, score: result.value, onFace };
        }
        // Inference mode: use forward pass score
        const rlScore = scoreStrategicRL(features, _rlStrategicModel);
        return { face, quarkType, score: rlScore, onFace };
    }

    // HEURISTIC SCORING — quark balance (deterministic, proven optimal)
    let score = Math.max(0, bestDeficit) * _choreoParams.ratioDeficitWeight * 10;

    // VACANCY: penalize if another xon is already executing a loop on this face
    for (const x of _demoXons) {
        if (!x.alive || x === xon) continue;
        if ((x._mode === 'tet' || x._mode === 'idle_tet') && x._assignedFace === face) {
            score -= _choreoParams.faceOccupiedPenalty;
            break;
        }
    }

    return { face, quarkType, score, onFace };
}

// ── Bipartite face compatibility check (T87/T88/T89) ──
// Pre-computed face pair topology: how many nodes each pair shares.
// Lazily built on first call, since _nucleusTetFaceData may not be populated at load time.
let _facePairTopology = null; // Map<'f1:f2' → sharedCount>

function _ensureFacePairTopology() {
    if (_facePairTopology) return;
    _facePairTopology = new Map();
    if (!_nucleusTetFaceData) return;
    const faces = Object.keys(_nucleusTetFaceData).map(Number);
    for (let i = 0; i < faces.length; i++) {
        for (let j = i + 1; j < faces.length; j++) {
            const c1 = new Set(_nucleusTetFaceData[faces[i]].cycle);
            const shared = _nucleusTetFaceData[faces[j]].cycle.filter(n => c1.has(n)).length;
            _facePairTopology.set(`${faces[i]}:${faces[j]}`, shared);
            _facePairTopology.set(`${faces[j]}:${faces[i]}`, shared);
        }
    }
}

const _PROTON_TYPES_SET = new Set(['pu1', 'pu2', 'pd']);

// Check if assigning quarkType to face would violate T87/T88/T89
// given currently active tet/idle_tet xons.
function _bipartiteCompatible(face, quarkType) {
    _ensureFacePairTopology();
    if (!_facePairTopology || _facePairTopology.size === 0) return true;
    const isProton = _PROTON_TYPES_SET.has(quarkType);
    for (const x of _demoXons) {
        if (!x.alive) continue;
        if (x._mode !== 'tet' && x._mode !== 'idle_tet') continue;
        if (x._assignedFace == null || !x._quarkType) continue;
        const otherFace = x._assignedFace;
        if (otherFace === face) continue; // same face — not a bipartite issue
        const shared = _facePairTopology.get(`${face}:${otherFace}`);
        if (shared === undefined) continue;
        const otherIsProton = _PROTON_TYPES_SET.has(x._quarkType);
        const sameHadron = (isProton === otherIsProton);
        // T87: shared ≥ 2 → must be opposite hadrons (sameHadron → FAIL)
        if (shared >= 2 && sameHadron) return false;
        // T88: shared = 1 → must be same hadron (opposite → FAIL)
        if (shared === 1 && !sameHadron) return false;
        // T89: shared = 0 → must be opposite hadrons (sameHadron → FAIL)
        if (shared === 0 && sameHadron) return false;
    }
    return true;
}

// Return ALL valid (face, quarkType, score) proposals for a xon across all faces and quark types.
// Used during backtracking to enumerate the full decision space.
function _allFaceOpportunities(xon, occupied) {
    const results = [];
    if (!_nucleusTetFaceData) return results;
    for (let f = 1; f <= 8; f++) {
        const fd = _nucleusTetFaceData[f];
        if (!fd) continue;
        // REACHABILITY: same check as _scoreFaceOpportunity
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
            if (!nearFace) continue;
        }
        // All valid quark types for this face
        const isProtonFace = A_SET.has(f);
        const qTypes = isProtonFace ? ['pu1', 'pu2', 'pd'] : ['nd1', 'nd2', 'nu'];
        for (const qt of qTypes) {
            // Bipartite compatibility: T87/T88/T89 check
            if (!_bipartiteCompatible(f, qt)) continue;
            // Score: deficit-weighted
            const deficit = _ratioTracker.deficit(qt);
            let score = Math.max(0, deficit) * _choreoParams.ratioDeficitWeight * 10;
            // Vacancy penalty
            for (const x of _demoXons) {
                if (!x.alive || x === xon) continue;
                if ((x._mode === 'tet' || x._mode === 'idle_tet') && x._assignedFace === f) {
                    score -= _choreoParams.faceOccupiedPenalty;
                    break;
                }
            }
            if (score >= _choreoParams.assignmentThreshold) {
                results.push({ xon, face: f, quarkType: qt, score, onFace });
            }
        }
    }
    return results;
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
        for (const nb of _sRngShuffle(_localBaseNeighbors(xon.node).slice())) {
            if ((occupied.get(nb.node) || 0) > 0) continue;
            if (blocked && blocked.has(nb.node)) continue;
            if (nb.node === xon.prevNode && xon.prevNode !== xon.node) continue;
            // T61: ejected weak xons must NOT target oct nodes (must eject away)
            if (isEjected && _octNodeSet && _octNodeSet.has(nb.node)) continue;
            // Random score with tiebreaker (PRNG seed changes per retry)
            const score = 1000 + (_sRng() - 0.5) * 0.5;
            candidates.push({ node: nb.node, dirIdx: nb.dirIdx, score, _scId: undefined, _needsMaterialise: false });
        }
        // Sort by score descending (fewest ejection traversals first)
        candidates.sort((a, b) => b.score - a.score);
        return candidates;
    }

    // Pending-weak xons: give ALL base neighbors (they need to step OFF the oct)
    if (xon._pendingWeakEjection) {
        const candidates = [];
        for (const nb of _sRngShuffle((baseNeighbors[xon.node] || []).slice())) {
            if (occupied.has(nb.node)) continue;
            if (blocked && blocked.has(nb.node)) continue;
            if (nb.node === xon.prevNode && xon.prevNode !== xon.node) continue;
            // Prefer non-oct destinations (score bonus for stepping off cage)
            const offCageBonus = (_octNodeSet && _octNodeSet.has(nb.node)) ? 0 : 10;
            candidates.push({ node: nb.node, dirIdx: nb.dirIdx, score: offCageBonus, _needsMaterialise: false, _scId: undefined });
        }
        return candidates;
    }

    // Constrain oct movement to cage nodes only.
    if (!_octNodeSet) return [];

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

    // Score candidates — RL model if available, else xonic movement balance heuristic
    const useRL = _rlActiveModel && typeof extractRLFeatures === 'function' && typeof scoreCandidateRL === 'function';
    const usePPO = useRL && typeof _ppoTraining !== 'undefined' && _ppoTraining &&
        typeof _ppoTacticalBuffer !== 'undefined' && _rlActiveModel.sampleAction;
    const validNeighbors = [];
    for (const nb of allOctNeighbors) {
        if (occupied.has(nb.node)) continue;
        if (blocked && blocked.has(nb.node)) continue;
        if (nb.node === xon.prevNode && xon.prevNode !== xon.node) continue;
        validNeighbors.push(nb);
    }
    const candidates = [];

    // Random choreographer mode: uniform random scores, skip all heuristics
    if (_bfsTestRandomChoreographer && !_btActive) {
        for (const nb of validNeighbors) {
            candidates.push({ node: nb.node, dirIdx: nb.dirIdx, score: _sRng(), _scId: nb._scId, _needsMaterialise: nb._needsMaterialise });
        }
        candidates.sort((a, b) => b.score - a.score);
        return candidates;
    }

    if (usePPO && validNeighbors.length > 0) {
        // PPO training: extract features for all valid candidates, sample from policy
        const featuresList = [];
        for (const nb of validNeighbors) {
            featuresList.push(extractRLFeatures(xon, nb, occupied));
        }
        // Use first candidate's features as the "state" (simplified — tactical state)
        // and sample which candidate to prefer
        const result = _rlActiveModel.sampleAction(featuresList[0], validNeighbors.length);
        _ppoTacticalBuffer.push(featuresList[0], result.actionIdx, result.logProb, result.value, validNeighbors.length);
        // Assign scores: sampled action gets highest, rest get decreasing
        for (let i = 0; i < validNeighbors.length; i++) {
            const nb = validNeighbors[i];
            const score = i === result.actionIdx ? 100 : (validNeighbors.length - i);
            candidates.push({ node: nb.node, dirIdx: nb.dirIdx, score, _scId: nb._scId, _needsMaterialise: nb._needsMaterialise });
        }
    } else {
        for (const nb of validNeighbors) {
            let score;
            if (useRL) {
                const features = extractRLFeatures(xon, nb, occupied);
                score = scoreCandidateRL(features, _rlActiveModel);
            } else {
                // Spatial bias + edge balance: prefer candidates closer to oct center
                // AND edges with fewer traversals. Random tiebreaker prevents cycling.
                score = _octCenterBias(nb.node);
                // Edge balance bonus: prefer less-traversed edges
                if (_edgeBalance) {
                    const pid = pairId(xon.node, nb.node);
                    const entry = _edgeBalance.get(pid);
                    if (entry && entry.total > 0) {
                        // Normalize: max traversals across all edges
                        score -= entry.total * 0.01; // slight penalty for overused edges
                    }
                }
                score += (_sRng() - 0.5) * 0.2;
            }
            candidates.push({ node: nb.node, dirIdx: nb.dirIdx, score, _scId: nb._scId, _needsMaterialise: nb._needsMaterialise });
        }
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

    // Council replay: force exact recorded moves during replay phase (up to peak tick)
    if (_sweepReplayActive && _sweepReplayMember && _demoTick <= _sweepReplayMember.peak) {
        const replayTick = _demoTick;
        const replayXi = _demoXons.indexOf(xon);
        const replayTickMoves = _sweepReplayMember.moves.get(replayTick);
        if (replayTickMoves) {
            const replayTarget = replayTickMoves.get(replayXi);
            if (replayTarget !== undefined) {
                const forced = candidates.filter(c => c.node === replayTarget);
                if (forced.length > 0) {
                    forced.sort((a, b) => b.score - a.score);
                    return forced;
                }
            }
        }
    }

    // Golden council boost: check all council members, boost proportional to agreement
    if (_sweepActive && _sweepGoldenCouncil.length > 0) {
        const tick = _demoTick;  // current tick (pre-move)
        const xi = _demoXons.indexOf(xon);
        // Count votes per candidate node from council members
        const votes = new Map();  // node → vote count
        for (const member of _sweepGoldenCouncil) {
            if (!member.moves) continue; // cold member, moves not loaded
            const tickMoves = member.moves.get(tick);
            if (tickMoves) {
                const target = tickMoves.get(xi);
                if (target !== undefined) votes.set(target, (votes.get(target) || 0) + 1);
            }
        }
        if (votes.size > 0) {
            for (const c of candidates) {
                const v = votes.get(c.node);
                if (v) {
                    c.score += 0.15 * v;  // +0.15 per agreeing council member
                    _sweepGoldenHits += v;
                    _sweepGoldenHitsSeed += v;
                }
            }
        }
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
        const xi = _demoXons.indexOf(xon);
        if (canMaterialiseQuick(target._scId)) {
            xonImpliedSet.add(target._scId);
            _scAttribution.set(target._scId, { reason: 'octMove', xonIdx: xi, tick: _demoTick });
            stateVersion++; // invalidate cache
            materialised = true;
        } else if (excitationSeverForRoom(target._scId)) {
            if (canMaterialiseQuick(target._scId)) {
                xonImpliedSet.add(target._scId);
                _scAttribution.set(target._scId, { reason: 'octMove', xonIdx: xi, tick: _demoTick });
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
    // Proxy may have blocked (e.g. already moved this tick) — verify
    if (xon.node !== target.node) return false;
    xon._lastDir = target.dirIdx;

    // Update xonic movement balance counters
    _updateDirBalance(xon, fromNode, target.node);

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
// _promoteFaceSCs removed: no preemptive SC promotion. SCs are only
// activated through vacuum negotiation when a xon traverses the edge.

// ── Gluon-mediated SC lifecycle ─────────────────────────────────────────────
// When _ruleGluonMediatedSC is ON, a companion gluon xon holds tet face SCs
// open in xonImpliedSet on behalf of the tet xon executing the loop.

// Find the best available oct-mode xon to serve as gluon for a tet face.
// Returns null if no candidate available (graceful degradation).
function _findGluonCandidate(tetXon, face) {
    if (!_ruleGluonMediatedSC) return null;
    const fd = _nucleusTetFaceData[face];
    if (!fd) return null;
    // Candidates: alive oct-mode xons that aren't already gluon-bound or the tet xon
    const candidates = _demoXons.filter(x =>
        x.alive && x !== tetXon &&
        x._mode === 'oct' &&
        x._gluonForFace == null
    );
    if (candidates.length === 0) return null;
    // Score by proximity to face SC endpoints (prefer xons near the face)
    const faceNodes = new Set(fd.allNodes);
    let best = null, bestScore = Infinity;
    for (const c of candidates) {
        // Simple: is the candidate ON a face node? (distance 0)
        if (faceNodes.has(c.node)) return c; // immediate match
        // Otherwise count shared base neighbors with face nodes
        const nbrs = baseNeighbors[c.node] || [];
        let dist = 2; // default: not adjacent
        for (const n of nbrs) {
            if (faceNodes.has(n.node)) { dist = 1; break; }
        }
        if (dist < bestScore) { bestScore = dist; best = c; }
    }
    return best;
}

// Assign a xon as gluon to physically maintain a tet face's 2 SCs.
function _assignGluon(gluonXon, face, clientXon) {
    const fd = _nucleusTetFaceData[face];
    if (!fd) return;
    const gi = _demoXons.indexOf(gluonXon);
    // Bind gluon to face
    gluonXon._mode = 'gluon';
    gluonXon._gluonForFace = face;
    gluonXon._gluonBoundSCs = fd.scIds.slice(); // copy the 2 SC IDs
    gluonXon._gluonClientXon = clientXon;
    gluonXon.col = GLUON_COLOR;
    if (gluonXon.sparkMat) gluonXon.sparkMat.color.setHex(GLUON_COLOR);
    // Promote face SCs into xonImpliedSet (ensure they persist)
    for (const scId of fd.scIds) {
        if (!xonImpliedSet.has(scId) && !activeSet.has(scId)) {
            if (canMaterialiseQuick(scId)) {
                xonImpliedSet.add(scId);
                _scAttribution.set(scId, { reason: 'gluonMaintain', xonIdx: gi, face, tick: _demoTick });
                stateVersion++;
            }
        }
    }
    _logChoreo(`X${gi} → gluon for face ${face} (client X${_demoXons.indexOf(clientXon)}), SCs [${fd.scIds.join(',')}]`);
}

// Release a gluon from its face SC maintenance duty.
// Removes face SCs from xonImpliedSet (same guards as _relinquishFaceSCs),
// clears gluon binding, returns xon to oct mode.
function _releaseGluon(face) {
    if (face == null) return;
    const gluonXon = _demoXons.find(x => x.alive && x._gluonForFace === face);
    if (!gluonXon) return; // no gluon bound — nothing to release
    const gi = _demoXons.indexOf(gluonXon);
    // Clear gluon binding, return to oct. SCs are NOT deleted here —
    // they persist until vacuum severs them (physics-only severance).
    _logChoreo(`X${gi} gluon released from face ${face} → oct`);
    gluonXon._gluonForFace = null;
    gluonXon._gluonBoundSCs = null;
    gluonXon._gluonClientXon = null;
    gluonXon._mode = 'oct';
    gluonXon.col = 0xffffff;
    if (gluonXon.sparkMat) gluonXon.sparkMat.color.setHex(0xffffff);
}

// Transition xon from oct mode to tet mode (assigned to actualize a face)
function _assignXonToTet(xon, face, quarkType) {
    const fd = _nucleusTetFaceData[face];
    if (!fd) return;
    // Tet loops can only be initiated from oct nodes — xon must be on the oct cage
    if (!_octNodeSet || !_octNodeSet.has(xon.node)) return;
    _demoTetAssignments++;  // track for hit rate
    // No preemptive face SC promotion — xons negotiate with the vacuum
    // before each hop. SCs are only locked if traversed (prevNode→node).

    const col = QUARK_COLORS[quarkType];
    const cycle = fd.cycle; // [a, b, c, d]

    // Determine the rotated cycle so position 0 = xon's current node.
    let rotated = cycle;
    const octNodesOnFace = cycle.filter(n => _octNodeSet.has(n));

    // Xon must be on one of the face's oct nodes to start a loop.
    // If the face isn't yet actualized, xon MUST be at the pole (cycle[0])
    // so the first step traverses the unique SC to activate the tet.
    if (!octNodesOnFace.includes(xon.node)) {
        return; // not on this face — can't assign
    }
    const [a, b, c, d] = cycle;
    const faceActualized = fd.scIds.every(scId =>
        activeSet.has(scId) || impliedSet.has(scId) || xonImpliedSet.has(scId));
    if (!faceActualized && xon.node !== a) {
        // Face needs activation — xon must be at the pole (cycle[0]) to traverse unique SC
        return;
    }
    // Rotate cycle so xon's current node is in position 0
    if (xon.node === a) rotated = [a, b, c, d];
    else if (xon.node === c) rotated = [c, b, a, d];
    else if (xon.node === d) rotated = [d, b, c, a];

    let seq = _selectBestPermutation(xon, rotated, quarkType);

    _clearModeProps(xon);
    xon._mode = 'tet';
    xon._assignedFace = face;
    xon._quarkType = quarkType;
    xon._loopType = LOOP_TYPE_NAMES[quarkType];
    xon._loopSeq = seq;
    xon._loopStep = 0;
    xon.col = col;
    // trail role captured at tick-end by _trailPush (T94: no retroactive edits)

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

    // Gluon-mediated: assign a companion gluon to hold face SCs open
    if (_ruleGluonMediatedSC) {
        const gluon = _findGluonCandidate(xon, face);
        if (gluon) _assignGluon(gluon, face, xon);
    }

    // Immediate advance: don't stall for one tick at seq[0].
    // Check Pauli at seq[1] before moving.
    // Also activate the unique SC via vacuum negotiation (Phase 3 won't run for this move).
    if (xon._loopStep === 0 && seq[1] !== undefined) {
        const dest = seq[1];
        const destOccupied = _demoXons.some(x => x.alive && x !== xon && x.node === dest);
        if (!destOccupied) {
            // Activate unique SC before moving (this edge is SC-only for pole→ext)
            const pid = pairId(xon.node, dest);
            const scId = scPairToId.get(pid);
            if (scId !== undefined && !xonImpliedSet.has(scId) && !activeSet.has(scId) && !impliedSet.has(scId)) {
                if (canMaterialiseQuick(scId)) {
                    xonImpliedSet.add(scId);
                    _scAttribution.set(scId, { reason: 'tetTraversal', xonIdx: _demoXons.indexOf(xon), face: face, tick: _demoTick });
                    _solverNeeded = true;
                } else if (excitationSeverForRoom(scId) && canMaterialiseQuick(scId)) {
                    xonImpliedSet.add(scId);
                    _scAttribution.set(scId, { reason: 'tetTraversal', xonIdx: _demoXons.indexOf(xon), face: face, tick: _demoTick });
                    _solverNeeded = true;
                } else {
                    // Can't activate SC — abort assignment
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
            _advanceXon(xon);
            xon._movedThisTick = true;
        }
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

    xon.tweenT = 0;

    // Return the target if we reached it in one hop, otherwise null (still walking)
    return targetNodes.has(step) ? step : null;
}

// T42: Clean up face SCs from xonImpliedSet when a xon abandons its tet face.
// Respects traversal lock — won't remove SCs being traversed by other xons.
function _relinquishFaceSCs(xon) {
    // Gluon binding cleanup (no SC deletion — gluons don't touch SCs anymore)
    if (_ruleGluonMediatedSC) _releaseGluon(xon._assignedFace);
    if (!_ruleRelinquishSCs) return; // sticky space: SCs persist until vacuum severs
    if (xon._assignedFace == null) return;
    const fd = _nucleusTetFaceData ? _nucleusTetFaceData[xon._assignedFace] : null;
    if (!fd) return;
    const locked = _traversalLockedSCs(xon); // exclude self — don't self-lock
    const cageSCs = _octSCIds ? new Set(_octSCIds) : new Set();
    for (const scId of fd.scIds) {
        if (locked.has(scId)) continue;
        if (cageSCs.has(scId)) continue; // NEVER relinquish oct cage SCs
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
            // Backtracker exclusion: skip moves the DFS has already tried
            const xi = _demoXons.indexOf(xon);
            if (xi >= 0 && typeof _btIsMoveExcluded === 'function' && _btIsMoveExcluded(xi, nb.node)) continue;
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
    // Tet loops can only be initiated from oct nodes
    if (!_octNodeSet || !_octNodeSet.has(xon.node)) return false;

    const types = ['pu1', 'pu2', 'nd1', 'nd2', 'pd', 'nu'];

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
        const shuffled = _sRngShuffle(faces.slice());
        const shuffledTypes = _sRngShuffle(types.slice());
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
                const seq = _selectBestPermutation(xon, rotated, qType);
                const dest = seq[1];
                if (occupied && occupied.has(dest)) continue;
                // Backtracker exclusion: skip face/dest combos the DFS has already tried
                const xi = _demoXons.indexOf(xon);
                if (xi >= 0 && typeof _btIsMoveExcluded === 'function' && _btIsMoveExcluded(xi, dest)) continue;
                // No preemptive face promotion — xons negotiate each SC on arrival.
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
                // Gluon-mediated: assign companion gluon for idle_tet face
                if (_ruleGluonMediatedSC) {
                    const gluon = _findGluonCandidate(xon, face);
                    if (gluon) _assignGluon(gluon, face, xon);
                }
                return true;
            }
            if (!bestSeq) {
                const fallbackType = existingXon && existingXon._quarkType
                    ? existingXon._quarkType
                    : shuffledTypes[0];
                bestSeq = _selectBestPermutation(xon, rotated, fallbackType);
                bestFace = face;
                bestType = fallbackType;
            }
        }
        if (bestSeq) {
            // No preemptive face promotion — xons negotiate each SC on arrival.
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
            // Gluon-mediated: assign companion gluon for idle_tet face
            if (_ruleGluonMediatedSC) {
                const gluon = _findGluonCandidate(xon, bestFace);
                if (gluon) _assignGluon(gluon, bestFace, xon);
            }
            return true;
        }
        return false;
    }

    if (tryFaces(actualizedFaces)) return true;

    // Pass 2 removed: no preemptive SC materialization. Xons negotiate
    // with the vacuum at traversal time (Phase 3). Only already-actualized
    // faces are available for idle tet loops.

    // ── Fallback: use any blocked actualized face ──
    // (caller handles Pauli if this destination is occupied)
    if (actualizedFaces.length > 0) return tryFaces(actualizedFaces);
    return false;
}
