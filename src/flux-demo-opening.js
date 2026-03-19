// flux-demo-opening.js — 2-Tick Opening Choreography
// Split from flux-demo.js — no logic changes.
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
        // Remaining 2 xons: move to above-center neighbors, become oct
        for (let i = 4; i < 6; i++) {
            const xon = _demoXons[i];
            if (xon._mode === 'oct_formation') {
                const aboveIdx = i - 4; // 0 and 1
                if (aboveIdx < aboveY.length) {
                    _executeOctMove(xon, { node: aboveY[aboveIdx].node, dirIdx: aboveY[aboveIdx].dirIdx, _needsMaterialise: false, _scId: undefined });
                }
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
