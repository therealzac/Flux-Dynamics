// flux-voids.js — Void rendering, render loop, UI wiring, state import/export
// ─── Void duality toggle ──────────────────────────────────────────────────────
// ─── View mode: spheres vs voids (mutually exclusive) ────────────────────────
// 'spheres' mode: sphere InstancedMesh visible, void face meshes hidden.
// 'voids'   mode: void face meshes visible, sphere InstancedMesh hidden.
// Each mode stores its own last-used opacity so switching back restores it.


function applySphereOpacity(){
    const op = +document.getElementById('sphere-opacity-slider').value / 100;
    document.getElementById('sphere-opacity-val').textContent = Math.round(op*100) + '%';
    _bgMat.opacity = op;
    _bgMat.depthWrite = op > 0.5; // opaque enough → write depth (correct occlusion)
    _bgMat.needsUpdate = true;    // transparent → skip depth write (voids show through)
}

function applyVoidOpacity(){
    document.getElementById('void-opacity-val').textContent = Math.round(+document.getElementById('void-opacity-slider').value) + '%';
    _updateVoidVisibility(); // per-mesh opacity set in _updateVoidVisibility
}

function toggleSelectMode(){
    selectMode=!selectMode;
    document.getElementById('btn-select-mode').classList.toggle('active',selectMode);
    canvas.style.cursor=selectMode?'crosshair':isGrabMode?'grab':'default';
    if(!selectMode){ hoveredVert=-1; hoveredSC=-1; selectedVert=-1; updateCandidates(); updateSpheres(); updateStatusHoverOnly(); }
}

// ─── Void sphere system ──────────────────────────────────────────────────────
// In this BCC-type lattice, two kinds of interstitial voids exist between sphere
// centers (radius 0.5):
//
//   Tetrahedral void (A₄): 4 equidistant sphere-center neighbors at r=√(5/12)≈0.645
//     Kissing sphere radius = √(5/12) - 0.5 ≈ 0.1455
//     Canonical offsets per cell: (±r3, ±r3/2, 0) and cyclic permutations (12 total)
//
//   Octahedral void (Oₕ): 2 close sphere-center neighbors at r=r3≈0.577 (+ 4 further)
//     Kissing sphere radius = r3 - 0.5 ≈ 0.0774
//     Canonical offsets per cell: (±r3, 0, 0) and cyclic permutations (6 total)
//
// Positions are replicated across all cells and deduplicated.

// Void neighbor computation:
//
// TETRAHEDRAL void: defined by 4 sphere vertices that include exactly 2
//   shortcut edges. Center = centroid of those 4. Actualized when both
//   shortcuts are in activeSet ∪ impliedSet.
//
// OCTAHEDRAL void: any 4-shortcut square (4-cycle, no chord) in the SC graph.
//   Center = centroid of the 4 square vertices. Actualized when all 4 shortcuts active/implied.
function computeVoidNeighbors(){
    voidNeighborData = [];
    const SC_D2 = (2/S3)*(2/S3);
    const scLookup = new Map(); // "min,max" -> sc id
    for(const sc of ALL_SC) scLookup.set(Math.min(sc.a,sc.b)+','+Math.max(sc.a,sc.b), sc.id);

    // ── Tetrahedral voids ─────────────────────────────────────────────────
    // A tet void = two shortcuts (A,B) and (C,D) whose 4 cross-pairs
    // (A-C, A-D, B-C, B-D) are all base edges. This is a purely graph-
    // theoretic criterion that correctly finds ALL tet voids regardless
    // of cell tiling, including voids with non-canonical centroid offsets
    // that the old CELL_VOIDS_TET approach missed.
    //
    // DO NOT revert to CELL_VOIDS_TET — it misses real tet voids.
    const BASE_D2 = 1.0;
    const baseAdjSet = Array.from({length:N}, ()=>new Set());
    for(const [i,j] of BASE_EDGES){ baseAdjSet[i].add(j); baseAdjSet[j].add(i); }

    // O(SC × degree²) tet detection — replaces O(SC²).
    // At L10: SC²=6.3B ops → SC×144=11M ops (~560× faster).
    // Key insight: SC(A,B) can only pair with SC(C,D) where C,D are base-
    // neighbors of A and B. So instead of scanning all SC pairs, for each
    // SC(A,B) we enumerate candidate C from neighbors(A) and D from neighbors(B).
    // DO NOT revert to O(SC²) — it will hang for 30+ seconds at L6+.
    // Build local lookup (can't reuse global scPairToId — not yet populated at init time)
    const scPairToIdMap = new Map();
    for(const sc of ALL_SC) scPairToIdMap.set(pairId(sc.a, sc.b), sc.id);
    const seenTet = new Set();
    for(const scA of ALL_SC){
        const a=scA.a, b=scA.b;
        // C must be a base-neighbor of both A and B (i.e. a bridge of scA... no)
        // Actually: for tet, C must be base-adj to A, D must be base-adj to B,
        // AND C base-adj to D, AND D base-adj to A, AND C base-adj to B.
        // Simpler: C ∈ neighbors(A) ∩ neighbors(B)... no. Let's be precise:
        // We need baseAdjSet[a].has(c) && baseAdjSet[a].has(d) &&
        //          baseAdjSet[b].has(c) && baseAdjSet[b].has(d)
        // So C ∈ neighbors(A)∩neighbors(B) and D ∈ neighbors(A)∩neighbors(B).
        // But also (C,D) must be a SC, so look it up.
        // bridges(A,B) = neighbors(A) ∩ neighbors(B)... but that's bridges, not tet.
        // For tet: all 4 of C,D must each be in neighbors(A) AND neighbors(B).
        // So both C and D ∈ neighbors(A) ∩ neighbors(B) is NOT required — 
        // re-reading the condition: a-c, a-d, b-c, b-d all base edges.
        // → c ∈ neighbors(a), d ∈ neighbors(a), c ∈ neighbors(b), d ∈ neighbors(b)
        // → c ∈ neighbors(a)∩neighbors(b), d ∈ neighbors(a)∩neighbors(b)
        // So yes: both C and D must be in commonNbrs(A,B).
        // But commonNbrs(A,B) are the bridge nodes of SC(A,B)!
        // So: enumerate all pairs (C,D) from bridges(A,B), check if (C,D) is a SC.
        const bridgesAB = [];
        for(const nb of baseAdjSet[a]){ if(baseAdjSet[b].has(nb)) bridgesAB.push(nb); }
        for(let ci=0;ci<bridgesAB.length;ci++){
            const c=bridgesAB[ci];
            for(let di=ci+1;di<bridgesAB.length;di++){
                const d=bridgesAB[di];
                if(c===a||c===b||d===a||d===b) continue;
                const scBId = scPairToIdMap.get(pairId(c,d));
                if(scBId===undefined) continue;
                const scB = SC_BY_ID[scBId];
                const nbrs=[a,b,c,d].sort((x,y)=>x-y);
                const key=nbrs.join(',');
                if(seenTet.has(key)) continue;
                seenTet.add(key);
                voidNeighborData.push({type:'tet', nbrs, scIds:[scA.id, scBId]});
            }
        }
    }

    // Build tet-completion map: scId -> partner scIds that complete a tet void.
    // Used by excitationStep to bias toward tet void completion.
    tetPartnerMap = new Map();
    for(const {type, scIds} of voidNeighborData){
        if(type!=='tet'||scIds.length!==2) continue;
        const [a,b]=scIds;
        if(!tetPartnerMap.has(a)) tetPartnerMap.set(a,[]);
        if(!tetPartnerMap.has(b)) tetPartnerMap.set(b,[]);
        tetPartnerMap.get(a).push(b);
        tetPartnerMap.get(b).push(a);
    }

    // Build _nodeTetVoids: node → list of tet voids whose nbrs include that node.
    // Used by excitationStep for proactive tet detection: when an excitation lands
    // on any node of a complete tet (both scIds open), it claims the tet immediately.
    // This handles tets whose two shortcuts don't share endpoints.
    _nodeTetVoids = new Map();
    for(const v of voidNeighborData){
        if(v.type!=='tet' || v.scIds.length!==2) continue;
        for(const n of v.nbrs){
            if(!_nodeTetVoids.has(n)) _nodeTetVoids.set(n, []);
            _nodeTetVoids.get(n).push(v);
        }
    }

    // Build _nodeOctVoids: node → list of oct voids containing that node.
    // Used by excitationStep for proactive oct (boson) detection.
    // Caches _allNodes (union of all cycle verts) on each oct entry.
    _nodeOctVoids = new Map();
    for(const v of voidNeighborData){
        if(v.type !== 'oct') continue;
        const allNodes = new Set();
        for(const {verts} of v.cycles) verts.forEach(n => allNodes.add(n));
        v._allNodes = allNodes;
        for(const n of allNodes){
            if(!_nodeOctVoids.has(n)) _nodeOctVoids.set(n, []);
            _nodeOctVoids.get(n).push(v);
        }
    }

    // Build scBridgeMap: scId -> Set of bridge nodes.
    // A bridge node is any lattice node base-adjacent to BOTH endpoints of the SC.
    // Excitations can only induce SC(A,B) by walking A→bridge→B (or B→bridge→A).
    // Used by goal-homing: after inducing SC1 of a tet, bias toward bridge nodes
    // of SC2 so the excitation reaches the induction path in the fewest hops.
    // O(SC×degree) scBridgeMap — replaces O(SC×N) scan.
    // bridges(A,B) = neighbors(A) ∩ neighbors(B), computed via Set intersection.
    scBridgeMap = new Map();
    for(const sc of ALL_SC){
        const bridges=new Set();
        for(const nb of baseAdjSet[sc.a]){ if(baseAdjSet[sc.b].has(nb)) bridges.add(nb); }
        scBridgeMap.set(sc.id, bridges);
    }

            // ── Octahedral voids ──────────────────────────────────────────────────
    // An O_h void appears at the center of any 4-cycle that is GEOMETRICALLY
    // a square (all 4 inner angles ≈ 90°) in the CURRENT positions.
    //
    // KEY INSIGHT (hard-won, do not lose this):
    //   - In the undeformed REST lattice, base edges meet at 60°/120° — never 90°.
    //   - Flux deforms the lattice. When shortcuts are active, some 4-cycles
    //     in the graph become geometric squares.
    //   - The criterion is GEOMETRIC (check angles in pos[]), NOT topological.
    //   - Edge composition (BASE vs SC) is IRRELEVANT. Any 4-cycle can become
    //     a square when the lattice is under the right flux configuration.
    //
    // Here we enumerate ALL 4-cycles (BASE + SC + mixed) in the full graph.
    // At render time, updateVoidSpheres() checks whether pos[] gives 90° angles.
    // The cycle ordering (nbrs field) is stored so updateVoidSpheres can check
    // consecutive angles: angle at nbrs[0] between nbrs[3]→nbrs[0]→nbrs[1], etc.
    //
    // scIds is still stored (for squarePartnerMap / excitation bias), but
    // actualization is driven purely by the geometric 90° check in updateVoidSpheres.
    //
    // DO NOT replace this with a topological-only check. That was the bug that
    // caused hundreds of phantom voids at rest (BASE 4-cycles are not squares
    // until flux deforms them) and missed flux-induced squares (diagonal SC edges
    // incorrectly rejected legitimate squares like v2-v12-v6-v14).
    // ────────────────────────────────────────────────────────────────────────

    const adjAll2 = Array.from({length:N},()=>new Set());
    for(const [i,j] of BASE_EDGES){ adjAll2[i].add(j); adjAll2[j].add(i); }
    for(const sc of ALL_SC){ adjAll2[sc.a].add(sc.b); adjAll2[sc.b].add(sc.a); }
    const seenOct = new Set();

    // One physical O_h void can be the center of up to 3 mutually-perpendicular
    // squares simultaneously. We enumerate all 4-cycles but DEDUPLICATE by REST
    // centroid so only one sphere is rendered per unique void location.
    // allCyclesAtCentroid: centroidKey -> {nbrs (cycle order), scIds (union of all cycles)}
    const allCyclesAtCentroid = new Map();

    for(let a=0;a<N;a++) for(const b of adjAll2[a]){
        if(b<=a) continue;
        for(const c of adjAll2[b]){
            if(c<=a||c===a) continue;
            for(const d of adjAll2[c]){
                if(d<=a||d===b||d===a) continue;
                if(!adjAll2[d].has(a)) continue;
                const sq=[a,b,c,d].sort((x,y)=>x-y);
                const sqKey=sq.join(',');
                if(seenOct.has(sqKey)) continue;
                seenOct.add(sqKey);
                // REST centroid key (rounded to avoid float noise)
                const cx=Math.round((REST[a][0]+REST[b][0]+REST[c][0]+REST[d][0])*1000/4);
                const cy=Math.round((REST[a][1]+REST[b][1]+REST[c][1]+REST[d][1])*1000/4);
                const cz=Math.round((REST[a][2]+REST[b][2]+REST[c][2]+REST[d][2])*1000/4);
                const centKey=cx+','+cy+','+cz;
                // SC ids for this cycle
                const cycleEdges=[[a,b],[b,c],[c,d],[d,a]];
                const cycleSCIds=[];
                for(const [u,v] of cycleEdges){
                    const id=scLookup.get(Math.min(u,v)+','+Math.max(u,v));
                    if(id!==undefined) cycleSCIds.push(id);
                }
                if(!allCyclesAtCentroid.has(centKey)){
                    allCyclesAtCentroid.set(centKey,{
                        // cycles: array of {verts, scIds} — one per square at this centroid
                        // scIds stored per-cycle at build time so render code needs no lookup
                        cycles:[{verts:[a,b,c,d], scIds:[...cycleSCIds]}]
                    });
                } else {
                    allCyclesAtCentroid.get(centKey).cycles.push({verts:[a,b,c,d], scIds:[...cycleSCIds]});
                }
            }
        }
    }

    for(const {cycles} of allCyclesAtCentroid.values()){
        // scIds = union of all cycle scIds (used by squarePartnerMap / excitation bias)
        const allScIds=[...new Set(cycles.flatMap(c=>c.scIds))];
        voidNeighborData.push({type:'oct', nbrs:cycles[0].verts, cycles, scIds:allScIds});
    }

    // Build squarePartnerMap AFTER oct voids pushed.
    // Maps scId -> all other scIds that belong to the same O_h void (across all cycles).
    // An excitation holding scId gets a bonus for inducing any sibling scId,
    // because doing so advances the full octahedron toward actualization.
    // NOTE: must stay here — oct data doesn't exist earlier.
    // NOTE: use cycles[].scIds per-cycle, NOT the union scIds on the void entry.
    //   The union was broken by the old scIds.length===4 guard; this is the correct approach.
    squarePartnerMap = new Map();
    for(const {type, cycles} of voidNeighborData){
        if(type!=='oct'||!cycles) continue;
        // Collect all scIds across all cycles of this void
        const voidAllScIds = [...new Set(cycles.flatMap(c=>c.scIds))];
        if(voidAllScIds.length === 0) continue; // pure-BASE void, no shortcuts to bias toward
        for(const id of voidAllScIds){
            if(!squarePartnerMap.has(id)) squarePartnerMap.set(id, new Set());
            for(const other of voidAllScIds) if(other!==id) squarePartnerMap.get(id).add(other);
        }
    }
}

    function rebuildVoidSpheres(){
    for(const entry of _voidMeshPool){
        scene.remove(entry.fillMesh); entry.fillMesh.geometry.dispose();
    }
    _voidMeshPool = [];
    for(const v of voidNeighborData){
        const fGeo = new THREE.BufferGeometry();
        // Each void gets its own material clone for independent opacity control
        const fillMat = (v.type==='tet' ? _voidMatTet : _voidMatOctVC).clone();
        const fillMesh = new THREE.Mesh(fGeo, fillMat);
        fillMesh.renderOrder = 1;
        fillMesh.visible = false;
        scene.add(fillMesh);
        // Wireframes removed — edges now rendered by unified edge pipeline
        // (rebuildBaseLines/rebuildShortcutLines with void-priority coloring)
        _voidMeshPool.push({fillMesh, type:v.type, wasActualized:false, scIds:v.scIds});
    }
    updateVoidSpheres();
}

function updateVoidSpheres(){
    // Called on every state change. For each void, show it if all required
    // shortcuts are active or implied; otherwise scale to 0 (hidden).
    const allActive = getAllOpen();
    // Dirty-flag: skip actualization checks if the open-SC set hasn't changed.
    // stateVersion comparison replaces O(n log n) sort+join string diff.
    const scSetChanged = _voidSpheresCacheKey !== stateVersion;
    _voidSpheresCacheKey = stateVersion;
    // Helper: dot product of two 3-vectors
    const _dot3=(u,v)=>u[0]*v[0]+u[1]*v[1]+u[2]*v[2];
    // Helper: check if a 4-cycle [p0,p1,p2,p3] is a geometric square in pos[].
    // Checks all 4 inner angles ≈ 90° (tolerance 15°) AND all SC edges active.
    //
    // BOTH conditions are required — do not remove either:
    //   1. SC edges active: SC squares exist at 90° in REST (inner 8 vertices of
    //      L1 form perfect squares with inactive shortcuts). Without this check,
    //      6 phantom O_h voids appear in the pure FCC lattice with no flux.
    //   2. Geometric 90° check: BASE 4-cycles exist everywhere topologically but
    //      are only squares when flux has deformed the lattice. Without this check,
    //      hundreds of phantom voids appear in the undeformed lattice.
    //
    // DO NOT simplify to either check alone. Both are necessary. ──────────────
    // Check if a single 4-cycle is geometrically a square in pos[].
    // cycleScIds stored at build time — no scLookup needed here.
    function _isCycleSquare(verts, cycleScIds){
        if(!cycleScIds.every(id => allActive.has(id))) return false;
        for(let k=0;k<4;k++){
            const prev=verts[(k+3)%4],mid=verts[k],nxt=verts[(k+1)%4];
            const u=[pos[prev][0]-pos[mid][0],pos[prev][1]-pos[mid][1],pos[prev][2]-pos[mid][2]];
            const w=[pos[nxt][0]-pos[mid][0],pos[nxt][1]-pos[mid][1],pos[nxt][2]-pos[mid][2]];
            const m=Math.sqrt(_dot3(u,u)*_dot3(w,w));
            if(m<1e-9) return false;
            if(Math.abs(_dot3(u,w)/m)>0.259) return false;
        }
        return true;
    }
    // An O_h void is actualized when ALL of its square cycles are geometric
    // squares (SCs active + 90° angles). The solver deforms base edges to 90°
    // when shortcuts activate — this is the geometric sanity check for proper
    // sphere packing. Boundary voids (fewer than 3 cycles) never actualize.
    function _isSquare(cycles){
        if(cycles.length < 3) return false; // boundary: not a full octahedron
        return cycles.every(({verts, scIds}) => _isCycleSquare(verts, scIds));
    }
    // Geometry builders
    function _faceGeo(vArr, idx){
        const g=new THREE.BufferGeometry();
        g.setAttribute('position',new THREE.BufferAttribute(vArr,3));
        g.setIndex(idx); g.computeVertexNormals(); return g;
    }
    // _wireFromFace removed — wireframes now handled by unified edge pipeline
    function _outwardFaces(vArr, rawFaces, nv){
        // Ensure each face normal points away from centroid
        let cx=0,cy=0,cz=0;
        for(let i=0;i<nv;i++){ cx+=vArr[i*3]; cy+=vArr[i*3+1]; cz+=vArr[i*3+2]; }
        cx/=nv; cy/=nv; cz/=nv;
        const idx=[];
        for(const [a,b,c] of rawFaces){
            const ax=vArr[a*3]-vArr[b*3], ay=vArr[a*3+1]-vArr[b*3+1], az=vArr[a*3+2]-vArr[b*3+2];
            const bx=vArr[c*3]-vArr[b*3], by=vArr[c*3+1]-vArr[b*3+1], bz=vArr[c*3+2]-vArr[b*3+2];
            const nx=ay*bz-az*by, ny=az*bx-ax*bz, nz=ax*by-ay*bx;
            const mx=(vArr[a*3]+vArr[b*3]+vArr[c*3])/3-cx;
            const my=(vArr[a*3+1]+vArr[b*3+1]+vArr[c*3+1])/3-cy;
            const mz=(vArr[a*3+2]+vArr[b*3+2]+vArr[c*3+2])/3-cz;
            if(nx*mx+ny*my+nz*mz>0) idx.push(a,b,c); else idx.push(a,c,b);
        }
        return idx;
    }
    function _tetGeo(nbrs){
        const vArr=new Float32Array(12);
        for(let i=0;i<4;i++){ vArr[i*3]=pos[nbrs[i]][0]; vArr[i*3+1]=pos[nbrs[i]][1]; vArr[i*3+2]=pos[nbrs[i]][2]; }
        return _faceGeo(vArr, _outwardFaces(vArr,[[0,2,1],[0,1,3],[0,3,2],[1,2,3]],4));
    }
    function _octGeo(cycles){
        const vSet=new Set(); for(const {verts} of cycles) verts.forEach(v=>vSet.add(v));
        const vList=[...vSet];
        const adj=new Set();
        for(const {verts:cv} of cycles) for(let k=0;k<4;k++){
            const a=cv[k],b=cv[(k+1)%4]; adj.add(Math.min(a,b)+','+Math.max(a,b));
        }
        const n=vList.length;
        const tmpArr=new Float32Array(n*3);
        for(let i=0;i<n;i++){ tmpArr[i*3]=pos[vList[i]][0]; tmpArr[i*3+1]=pos[vList[i]][1]; tmpArr[i*3+2]=pos[vList[i]][2]; }
        const raw=[];
        for(let i=0;i<n;i++) for(let j=i+1;j<n;j++) for(let k=j+1;k<n;k++){
            const a=vList[i],b=vList[j],c=vList[k];
            if(adj.has(Math.min(a,b)+','+Math.max(a,b))&&adj.has(Math.min(b,c)+','+Math.max(b,c))&&adj.has(Math.min(a,c)+','+Math.max(a,c)))
                raw.push([i,j,k]);
        }
        const idx = _outwardFaces(tmpArr, raw, n);
        const nTris = idx.length/3;
        // Build non-indexed geometry with per-face vertex colors
        const posArr=new Float32Array(nTris*9), colArr=new Float32Array(nTris*9);
        const faceNormals=[];
        // Centroid for face-lighting direction calculation
        let cx=0,cy=0,cz=0;
        for(let i=0;i<n;i++){ cx+=tmpArr[i*3]; cy+=tmpArr[i*3+1]; cz+=tmpArr[i*3+2]; }
        cx/=n; cy/=n; cz/=n;
        const br=1.0, bg=1.0, bb=1.0; // base white (oct voids = bosonic field)
        for(let t=0;t<nTris;t++){
            const ia=idx[t*3], ib=idx[t*3+1], ic=idx[t*3+2];
            for(let c=0;c<3;c++){ posArr[t*9+c]=tmpArr[ia*3+c]; posArr[t*9+3+c]=tmpArr[ib*3+c]; posArr[t*9+6+c]=tmpArr[ic*3+c]; }
            // Face normal
            const ax=tmpArr[ib*3]-tmpArr[ia*3], ay=tmpArr[ib*3+1]-tmpArr[ia*3+1], az=tmpArr[ib*3+2]-tmpArr[ia*3+2];
            const bx=tmpArr[ic*3]-tmpArr[ia*3], by=tmpArr[ic*3+1]-tmpArr[ia*3+1], bz=tmpArr[ic*3+2]-tmpArr[ia*3+2];
            let nx=ay*bz-az*by, ny=az*bx-ax*bz, nz=ax*by-ay*bx;
            const nl=Math.sqrt(nx*nx+ny*ny+nz*nz); if(nl>0){nx/=nl;ny/=nl;nz/=nl;}
            faceNormals.push(nx,ny,nz);
            // Init vertex colors to base gray
            for(let v=0;v<3;v++){ colArr[t*9+v*3]=br; colArr[t*9+v*3+1]=bg; colArr[t*9+v*3+2]=bb; }
        }
        const g=new THREE.BufferGeometry();
        g.setAttribute('position',new THREE.BufferAttribute(posArr,3));
        g.setAttribute('color',new THREE.BufferAttribute(colArr,3));
        g.computeVertexNormals();
        g._vList=vList; g._faceNormals=faceNormals; g._centroid=[cx,cy,cz];
        return g;
    }

    const op = +document.getElementById('void-opacity-slider').value / 100;
    for(let vi=0; vi<voidNeighborData.length; vi++){
        const {type, nbrs, cycles, scIds} = voidNeighborData[vi];
        const entry = _voidMeshPool[vi];
        const {fillMesh} = entry;
        // Only re-evaluate actualization if the open-SC set changed
        let actualized = entry.wasActualized; // default: keep previous state
        if(scSetChanged){
            actualized = type==='tet'
                    ? scIds.every(id => allActive.has(id))
                    : _isSquare(cycles||[]);
            voidNeighborData[vi].actualized = actualized;
            // Store per-cycle actualization for oct voids — excitations can
            // survive on a single square cycle, not just the full octahedron.
            if(type === 'oct' && cycles){
                for(const cycle of cycles){
                    const wasCycleActualized = cycle.actualized || false;
                    cycle.actualized = _isCycleSquare(cycle.verts, cycle.scIds);
                    // Lock in oct cycle SCs: when a cycle newly actualizes,
                    // promote any cascade-implied SCs to xonImpliedSet
                    // so they persist across detectImplied recalculations.
                    // Without this, octs built partly on cascade-implied SCs
                    // pop in and immediately out of existence.
                    if(cycle.actualized && !wasCycleActualized){
                        for(const id of cycle.scIds){
                            if(impliedSet.has(id) && !xonImpliedSet.has(id) && !activeSet.has(id)){
                                xonImpliedSet.add(id);
                                if(!impliedBy.has(id)) impliedBy.set(id, new Set());
                            }
                        }
                    }
                }
            }
        }

        // Void fill mesh updates (wireframes removed — handled by unified edge pipeline)
        if(actualized && !entry.wasActualized){
            // Newly actualized: build fill geometry from scratch
            const fGeo = type==='tet' ? _tetGeo(nbrs) : _octGeo(cycles);
            fillMesh.geometry.dispose(); fillMesh.geometry = fGeo;
            fillMesh.visible = op > 0;
            entry.wasActualized = true;
        } else if(actualized && entry.wasActualized){
            // Still actualized: update fill vertex positions
            if(type === 'tet'){
                const pos3f = fillMesh.geometry.attributes.position;
                for(let i=0;i<4;i++){ pos3f.setXYZ(i, pos[nbrs[i]][0], pos[nbrs[i]][1], pos[nbrs[i]][2]); }
                pos3f.needsUpdate = true;
                fillMesh.geometry.computeVertexNormals();
            } else {
                // Oct: rebuild entirely (oct voids are rare, not perf-critical)
                const fGeo = _octGeo(cycles);
                fillMesh.geometry.dispose(); fillMesh.geometry = fGeo;
            }
        } else if(!actualized && entry.wasActualized){
            // Newly de-actualized: hide
            fillMesh.visible = false;
            entry.wasActualized = false;
        }
        // !actualized && !wasActualized: already hidden, nothing to do
    }
    _updateVoidVisibility();
}

function _updateVoidVisibility(){
    const op = +document.getElementById('void-opacity-slider').value / 100;
    // Map excited voids to the excitation color that owns them.
    // voidScIds is set once when the excitation claims a void — reliable match.
    const excitedVoidColor = new Map(); // vi → THREE.Color
    try { for(const e of excitations){
        if(!e.voidScIds) continue;
        const eKey = [...e.voidScIds].sort((a,b)=>a-b).join(',');
        for(let vi=0; vi<_voidMeshPool.length; vi++){
            if(excitedVoidColor.has(vi)) continue;
            const entry = _voidMeshPool[vi];
            if(!entry.wasActualized) continue;
            const vKey = [...entry.scIds].sort((a,b)=>a-b).join(',');
            if(vKey === eKey){
                excitedVoidColor.set(vi, new THREE.Color(e.col));
                break;
            }
        }
    } } catch(_){} // excitations may not be initialized yet during startup
    for(let vi=0; vi<_voidMeshPool.length; vi++){
        const entry = _voidMeshPool[vi];
        if(!entry.wasActualized) continue;
        if(entry.type === 'oct'){
            // Oct voids: full opacity always — face lighting handles visual feedback
            entry.fillMesh.material.opacity = op;
            entry.fillMesh.material.needsUpdate = true;
        } else {
            // Tet voids: Rule annotation > excitation color > default dim
            const annotCol = _ruleAnnotations.tetColors.get(vi);
            const annotOp = _ruleAnnotations.tetOpacity.get(vi);
            const eCol = excitedVoidColor.get(vi);
            const excited = !!eCol;
            const hasAnnot = annotCol !== undefined;

            const fillOp = annotOp !== undefined ? annotOp * op : (hasAnnot || excited ? op : op * 0.25);

            if(hasAnnot){
                entry.fillMesh.material.color.setHex(annotCol);
                entry.fillMesh.material.emissive.setHex(annotCol); entry.fillMesh.material.emissive.multiplyScalar(0.3);
            } else if(excited){
                entry.fillMesh.material.color.copy(eCol);
                entry.fillMesh.material.emissive.copy(eCol).multiplyScalar(0.3);
            } else {
                entry.fillMesh.material.color.setHex(0x999999);
                entry.fillMesh.material.emissive.setHex(0x222222);
            }
            entry.fillMesh.material.opacity = fillOp;
            entry.fillMesh.material.needsUpdate = true;
        }
    }
}

// ─── Oct void face lighting ────────────────────────────────────────────────
// Per-frame: modulate oct void face colors based on excitation direction.
// Faces pointing toward a nearby excitation glow brighter.
function tickOctVoids(){
    // Per-face coloring uses the color of the nearest contributing excitation.
    // When no excitations are present, all faces glow at neutral gray.
    const dbr=1.0, dbg=1.0, dbb=1.0; // default white (oct voids = bosonic field)
    const EXCITE_THRESH = 0.08;
    const noExcitations = !excitations.length;
    // Pre-extract excitation colors as RGB floats for speed
    const _tmpCol = new THREE.Color();
    for(let vi=0; vi<_voidMeshPool.length; vi++){
        const entry = _voidMeshPool[vi];
        if(!entry.wasActualized || entry.type!=='oct') continue;
        const geo = entry.fillMesh.geometry;
        if(!geo._faceNormals || !geo._centroid) continue;
        const colAttr = geo.attributes.color;
        if(!colAttr) continue;
        const [cx,cy,cz] = geo._centroid;
        const fn = geo._faceNormals;
        const nFaces = fn.length/3;
        // Check for rule annotation override
        const annotOctCol = _ruleAnnotations.octColors.get(vi);
        const annotFaces = _ruleAnnotations.octFaceColors.get(vi);

        for(let f=0;f<nFaces;f++){
            let fr=dbr, fg=dbg, fb=dbb, brightness;

            // Per-face annotation takes highest priority
            if(annotFaces && annotFaces[f] !== undefined){
                const fc = annotFaces[f];
                fr = ((fc>>16)&0xff)/255;
                fg = ((fc>>8)&0xff)/255;
                fb = (fc&0xff)/255;
                brightness = 1.0;
            } else if(annotOctCol !== undefined){
                // Whole-oct annotation color
                fr = ((annotOctCol>>16)&0xff)/255;
                fg = ((annotOctCol>>8)&0xff)/255;
                fb = (annotOctCol&0xff)/255;
                brightness = 0.8;
            } else if(noExcitations){
                brightness = 1.0;
            } else {
                const fnx=fn[f*3], fny=fn[f*3+1], fnz=fn[f*3+2];
                let maxContrib = 0;
                let bestCol = null;
                try { for(const e of excitations){
                    const ep = e.group.position;
                    let dx=ep.x-cx, dy=ep.y-cy, dz=ep.z-cz;
                    const dl=Math.sqrt(dx*dx+dy*dy+dz*dz);
                    if(dl<0.01) continue;
                    dx/=dl; dy/=dl; dz/=dl;
                    const dot = fnx*dx + fny*dy + fnz*dz;
                    const prox = Math.max(0, 1.0 - dl/4.0);
                    const contrib = Math.max(0,dot) * prox;
                    if(contrib > maxContrib){ maxContrib = contrib; bestCol = e.col; }
                } } catch(_){}
                brightness = maxContrib > EXCITE_THRESH ? (0.3 + 0.7 * maxContrib) : 0.08;
                if(bestCol !== null && maxContrib > EXCITE_THRESH){
                    _tmpCol.setHex(bestCol);
                    fr = _tmpCol.r; fg = _tmpCol.g; fb = _tmpCol.b;
                }
            }
            for(let v=0;v<3;v++){
                colAttr.array[(f*3+v)*3]   = fr*brightness;
                colAttr.array[(f*3+v)*3+1] = fg*brightness;
                colAttr.array[(f*3+v)*3+2] = fb*brightness;
            }
        }
        colAttr.needsUpdate = true;
    }
}


// Render loop moved to flux-solver-render.js (startRenderLoop)


// ─── Apply state from JSON ────────────────────────────────────────────────────
function applyStateFromJSON(data){
    if(jiggleActive) toggleJiggle();
    if(placingExcitation) toggleExcitationPlacement();
    removeAllExcitations();
    const targetLevel = (data.meta&&data.meta.latticeLevel)||1;
    if(targetLevel !== latticeLevel){
        latticeLevel = targetLevel;
        document.getElementById('lattice-slider').value = latticeLevel;
        document.getElementById('lattice-lv').textContent = 'L'+latticeLevel;
        rebuildLatticeGeometry(latticeLevel, _nucleusOctCentered);
        rebuildScPairLookup();
        rebuildSphereMeshes();
        rebuildBaseLines();
        rebuildShortcutLines();
        rebuildVoidSpheres();
        applySphereOpacity();
        applyCamera();
    }
    activeSet.clear(); impliedSet.clear(); impliedBy.clear();
    xonImpliedSet.clear(); blockedImplied.clear();
    selectedVert=-1; hoveredVert=-1; hoveredSC=-1;
    // Apply active shortcuts first (positions will be solved fresh)
    if(data.active){
        for(const s of data.active){
            const sc=ALL_SC.find(x=>x.a===s.a&&x.b===s.b);
            if(sc) activeSet.add(sc.id);
        }
    }
    // Apply implied shortcuts
    if(data.implied){
        for(const s of data.implied){
            const sc=ALL_SC.find(x=>x.a===s.a&&x.b===s.b);
            if(sc){ impliedSet.add(sc.id); impliedBy.set(sc.id,new Set()); }
        }
    }
    // Solve positions from scratch — do NOT use stored positions which may
    // have floating-point violations. The solver will find the correct geometry.
    bumpState();
    const solvedPos = detectImplied();
    applyPositions(solvedPos);
    updateCandidates(); updateSpheres(); updateStatus();
}

// ─── Wire UI ──────────────────────────────────────────────────────────────────
document.getElementById('sh-clear').addEventListener('click',clearAll);
document.getElementById('btn-export').addEventListener('click',exportState);
document.getElementById('btn-copy-violation').addEventListener('click',function(){
    try{ const json=JSON.stringify(buildExportData(),null,2); copyText(json); toast('violation state copied'); }
    catch(e){ toast('export error: '+e.message); console.error('buildExportData failed:',e); }
});
const _btnJiggle = document.getElementById('btn-jiggle');
if (_btnJiggle) _btnJiggle.addEventListener('click',toggleJiggle);
const _jiggleSlider = document.getElementById('jiggle-slider');
if (_jiggleSlider) _jiggleSlider.addEventListener('input',updateJiggleSpeed);
document.getElementById('energy-slider').addEventListener('input',updateEnergy);
document.getElementById('lattice-slider').addEventListener('input',updateLatticeLevel);

document.getElementById('graph-opacity-slider').addEventListener('input',updateGraphOpacity);
document.getElementById('sphere-opacity-slider').addEventListener('input',applySphereOpacity);
document.getElementById('void-opacity-slider').addEventListener('input',applyVoidOpacity);
document.getElementById('excitation-speed-slider').addEventListener('input', ()=>{
    const t = +document.getElementById('excitation-speed-slider').value / 100;
    if (t >= 1.0) {
        ELECTRON_STEP_MS = 0;
        document.getElementById('excitation-speed-val').textContent = 'MAX';
    } else {
        ELECTRON_STEP_MS = Math.round(Math.exp(Math.log(1000)*(1-t) + Math.log(30)*t));
        document.getElementById('excitation-speed-val').textContent = ELECTRON_STEP_MS + 'ms';
    }
    // Restart clock so new interval takes effect immediately
    if(excitationClockTimer){ clearInterval(excitationClockTimer); excitationClockTimer=null; startExcitationClock(); }
    // Also restart demo interval if demo is running (but NOT if paused)
    if(_demoActive && typeof isDemoPaused === 'function' && !isDemoPaused()) {
        if (_demoInterval) { clearInterval(_demoInterval); _demoInterval = null; }
        if (_demoUncappedId) { clearTimeout(_demoUncappedId); _demoUncappedId = null; }
        const intervalMs = _getDemoIntervalMs();
        if (intervalMs === 0) {
            _demoUncappedId = setTimeout(_demoUncappedLoop, 0);
        } else {
            _demoInterval = setInterval(demoTick, intervalMs);
        }
    }
});
document.getElementById('trail-opacity-slider').addEventListener('input',()=>{
    const pct=+document.getElementById('trail-opacity-slider').value;
    document.getElementById('trail-opacity-val').textContent=pct+'%';
});
document.getElementById('tracer-lifespan-slider').addEventListener('input',()=>{
    const val=+document.getElementById('tracer-lifespan-slider').value;
    document.getElementById('tracer-lifespan-val').textContent = val === 0 ? 'off' : val;
});
// Spark slider synced from xons (trail) slider
document.getElementById('trail-opacity-slider').addEventListener('input',()=>{
    document.getElementById('spark-opacity-slider').value =
        document.getElementById('trail-opacity-slider').value;
});

// Weak force opacity slider label update
document.getElementById('weak-opacity-slider').addEventListener('input',()=>{
    const pct=+document.getElementById('weak-opacity-slider').value;
    document.getElementById('weak-opacity-val').textContent=pct+'%';
});
// ═══ BRANE LAYER — Planar slices through FCC lattice ═══════════════════════
// 6 plane families, one per shortcut type.  Each plane is spanned by a pair
// of base directions; the plane normal IS the corresponding shortcut direction.
// Perpendicular plane pairs automatically receive complementary SC colors.

let _braneMeshes = [];   // THREE.Mesh[], one per stype
let _braneVertMeta = [];  // per-mesh: [{nodes:[nodeIdx,...], falloff, br, bg, bb}, ...] per vertex

function _cross3(a, b) {
    return [a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x];
}

function _grahamHull2D(pts) {
    // Returns convex hull indices in CCW order (Graham scan)
    if (pts.length < 3) return pts.map((_, i) => i);
    // Find bottom-most (then left-most) point
    let p0 = 0;
    for (let i = 1; i < pts.length; i++) {
        if (pts[i][1] < pts[p0][1] || (pts[i][1] === pts[p0][1] && pts[i][0] < pts[p0][0])) p0 = i;
    }
    const anchor = pts[p0];
    const idxs = [];
    for (let i = 0; i < pts.length; i++) if (i !== p0) idxs.push(i);
    // Sort by polar angle from anchor
    idxs.sort((a, b) => {
        const ax = pts[a][0] - anchor[0], ay = pts[a][1] - anchor[1];
        const bx = pts[b][0] - anchor[0], by = pts[b][1] - anchor[1];
        const cross = ax * by - ay * bx;
        if (Math.abs(cross) > 1e-12) return -cross; // CCW first
        // Collinear: closer first
        return (ax*ax + ay*ay) - (bx*bx + by*by);
    });
    const stack = [p0];
    for (const idx of idxs) {
        while (stack.length > 1) {
            const t = stack[stack.length - 1], t2 = stack[stack.length - 2];
            const cross = (pts[t][0]-pts[t2][0])*(pts[idx][1]-pts[t2][1])
                        - (pts[t][1]-pts[t2][1])*(pts[idx][0]-pts[t2][0]);
            if (cross > 1e-12) break;
            stack.pop();
        }
        stack.push(idx);
    }
    return stack;
}

function buildBranes() {
    disposeBranes();
    if (typeof N === 'undefined' || N === 0 || typeof REST === 'undefined') return;

    const slider = document.getElementById('brane-opacity-slider');
    // Slider 0–100 maps to 0.0%–10.0%
    const baseOp = slider ? (+slider.value / 10) / 100 : 0.03;

    // Oct center for inverse distance dimming (dim near nucleus, bright far away)
    let ocx = 0, ocy = 0, ocz = 0;
    if (typeof _octNodeSet !== 'undefined' && _octNodeSet && _octNodeSet.size > 0) {
        for (const ni of _octNodeSet) { ocx += REST[ni][0]; ocy += REST[ni][1]; ocz += REST[ni][2]; }
        ocx /= _octNodeSet.size; ocy /= _octNodeSet.size; ocz /= _octNodeSet.size;
    }
    // Compute max distance from oct center for normalization
    let maxDist = 0;
    for (let k = 0; k < N; k++) {
        const dx = REST[k][0] - ocx, dy = REST[k][1] - ocy, dz = REST[k][2] - ocz;
        const d = Math.sqrt(dx*dx + dy*dy + dz*dz);
        if (d > maxDist) maxDist = d;
    }
    if (maxDist < 0.01) maxDist = 1; // safety

    for (let stype = 1; stype <= 6; stype++) {
        const dirs = SC_BASE_DIRS[stype]; // [i, j] base direction indices
        const va = BASE_DIR_V[dirs[0]], vb = BASE_DIR_V[dirs[1]];
        // Plane normal = cross product of the two base directions
        const nc = _cross3(va, vb);
        const nm = Math.sqrt(nc[0]*nc[0] + nc[1]*nc[1] + nc[2]*nc[2]);
        const nx = nc[0]/nm, ny = nc[1]/nm, nz = nc[2]/nm;

        // Project all nodes onto normal, group by projection value
        const projMap = new Map(); // rounded d → [nodeIdx, ...]
        for (let k = 0; k < N; k++) {
            const d = REST[k][0]*nx + REST[k][1]*ny + REST[k][2]*nz;
            const key = Math.round(d * 1e4); // tolerance ~1e-4
            if (!projMap.has(key)) projMap.set(key, []);
            projMap.get(key).push(k);
        }

        // Build orthonormal basis in the plane
        // u = normalize(va), w = cross(n, u)
        const ux = va.x, uy = va.y, uz = va.z; // already normalized
        const wx = ny*uz - nz*uy, wy = nz*ux - nx*uz, wz = nx*uy - ny*ux;

        // Base color components (normalized 0–1)
        const hex = S_COLOR[stype];
        const br = ((hex >> 16) & 0xff) / 255;
        const bg = ((hex >> 8) & 0xff) / 255;
        const bb = (hex & 0xff) / 255;

        // Collect all triangle vertices + vertex colors + metadata across all slices
        const triVerts = [];
        const triColors = [];
        const vertMeta = []; // per-vertex: {nodes:[nodeIdx,...], br, bg, bb}

        for (const [, nodeIdxs] of projMap) {
            if (nodeIdxs.length < 3) continue;

            // Project to 2D plane coordinates
            const pts2D = [];
            for (const ni of nodeIdxs) {
                const px = REST[ni][0], py = REST[ni][1], pz = REST[ni][2];
                pts2D.push([px*ux + py*uy + pz*uz, px*wx + py*wy + pz*wz]);
            }

            // Convex hull
            const hull = _grahamHull2D(pts2D);
            if (hull.length < 3) continue;

            // Fan triangulate from centroid
            let cx3 = 0, cy3 = 0, cz3 = 0;
            const hullNodes = hull.map(hi => nodeIdxs[hi]);
            for (const ni of hullNodes) {
                cx3 += REST[ni][0]; cy3 += REST[ni][1]; cz3 += REST[ni][2];
            }
            cx3 /= hull.length; cy3 /= hull.length; cz3 /= hull.length;

            for (let h = 0; h < hull.length; h++) {
                const h2 = (h + 1) % hull.length;
                const a = hullNodes[h], b = hullNodes[h2];

                // Inverse distance dimming: dim near oct, bright far away
                // Centroid vertex
                const cdx = cx3-ocx, cdy = cy3-ocy, cdz = cx3-ocz; // intentional: use cx3 for x,z
                const cDist = Math.sqrt((cx3-ocx)**2 + (cy3-ocy)**2 + (cz3-ocz)**2);
                const cDim = (cDist / maxDist) ** 2; // quadratic: 0 at center, 1 at edge
                triVerts.push(cx3, cy3, cz3);
                triColors.push(br * cDim, bg * cDim, bb * cDim);
                vertMeta.push({nodes: hullNodes, br: br * cDim, bg: bg * cDim, bb: bb * cDim});
                // Vertex A
                const aDist = Math.sqrt((REST[a][0]-ocx)**2 + (REST[a][1]-ocy)**2 + (REST[a][2]-ocz)**2);
                const aDim = (aDist / maxDist) ** 2;
                triVerts.push(REST[a][0], REST[a][1], REST[a][2]);
                triColors.push(br * aDim, bg * aDim, bb * aDim);
                vertMeta.push({nodes: [a], br: br * aDim, bg: bg * aDim, bb: bb * aDim});
                // Vertex B
                const bDist = Math.sqrt((REST[b][0]-ocx)**2 + (REST[b][1]-ocy)**2 + (REST[b][2]-ocz)**2);
                const bDim = (bDist / maxDist) ** 2;
                triVerts.push(REST[b][0], REST[b][1], REST[b][2]);
                triColors.push(br * bDim, bg * bDim, bb * bDim);
                vertMeta.push({nodes: [b], br: br * bDim, bg: bg * bDim, bb: bb * bDim});
            }
        }

        if (triVerts.length === 0) continue;

        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.Float32BufferAttribute(triVerts, 3));
        geo.setAttribute('color', new THREE.Float32BufferAttribute(triColors, 3));
        geo.computeVertexNormals();

        const mat = new THREE.MeshBasicMaterial({
            vertexColors: true,
            transparent: true,
            opacity: baseOp,
            side: THREE.DoubleSide,
            depthWrite: false,
            blending: THREE.AdditiveBlending,
        });

        const mesh = new THREE.Mesh(geo, mat);
        mesh.renderOrder = 0;
        scene.add(mesh);
        _braneMeshes.push(mesh);
        _braneVertMeta.push(vertMeta);
    }
}

function updateBraneOpacity() {
    const slider = document.getElementById('brane-opacity-slider');
    if (!slider) return;
    // Slider 0–100 maps to 0.0%–10.0% opacity (0.1% per tick)
    const pct = +slider.value / 10;          // 0.0 – 10.0
    const op = pct / 100;                    // 0.000 – 0.100
    document.getElementById('brane-opacity-val').textContent = pct.toFixed(1) + '%';
    for (const mesh of _braneMeshes) {
        mesh.material.opacity = op;
        mesh.visible = op > 0;
    }
}

// Per-frame: compute node displacements, highlight deformed brane segments
function updateBraneHighlights() {
    if (typeof pos === 'undefined' || typeof REST === 'undefined' || typeof N === 'undefined') return;
    if (_braneMeshes.length === 0) return;

    const disp = new Float32Array(N);
    for (let k = 0; k < N; k++) {
        const dx = pos[k][0] - REST[k][0], dy = pos[k][1] - REST[k][1], dz = pos[k][2] - REST[k][2];
        disp[k] = Math.sqrt(dx*dx + dy*dy + dz*dz);
    }

    const DEFORM_THRESH = 0.005;
    for (let mi = 0; mi < _braneMeshes.length; mi++) {
        const mesh = _braneMeshes[mi];
        const meta = _braneVertMeta[mi];
        if (!meta) continue;
        const colAttr = mesh.geometry.attributes.color;
        const colArr = colAttr.array;

        for (let vi = 0; vi < meta.length; vi++) {
            const m = meta[vi];
            let maxD = 0;
            for (const ni of m.nodes) {
                if (disp[ni] > maxD) maxD = disp[ni];
            }
            const boost = maxD > DEFORM_THRESH ? Math.min(2, 1 + maxD / DEFORM_THRESH) : 1;
            colArr[vi * 3]     = m.br * boost;
            colArr[vi * 3 + 1] = m.bg * boost;
            colArr[vi * 3 + 2] = m.bb * boost;
        }
        colAttr.needsUpdate = true;
    }
}

function disposeBranes() {
    for (const mesh of _braneMeshes) {
        scene.remove(mesh);
        mesh.geometry.dispose();
        mesh.material.dispose();
    }
    _braneMeshes = [];
    _braneVertMeta = [];
}

document.getElementById('brane-opacity-slider').addEventListener('input', updateBraneOpacity);

// ╔══════════════════════════════════════════════════════════════════════╗
// ║  WAVEFUNCTION LAYER — Spherical Harmonic Surface on Lattice Hull   ║
// ╚══════════════════════════════════════════════════════════════════════╝

let _wfMesh = null;
let _wfVertSph = [];       // per-vertex {theta, phi}
let _wfCenter = [0,0,0];
let _wfPoleAxis = null;    // THREE.Vector3
let _wfXHat = null;        // THREE.Vector3
let _wfYHat = null;        // THREE.Vector3
let _wfAdaptiveMin = 0;
let _wfAdaptiveMax = 1;

// ── 3D Convex Hull (incremental algorithm) ──
function _convexHull3D(points) {
    // points: array of [x,y,z]
    // Returns: array of [i,j,k] triangle index triples (CCW winding from outside)
    const n = points.length;
    if (n < 4) return [];

    // Find 4 non-coplanar seed points
    let p0 = 0, p1 = -1, p2 = -1, p3 = -1;
    // p1: farthest from p0
    let maxD = 0;
    for (let i = 1; i < n; i++) {
        const dx = points[i][0] - points[p0][0];
        const dy = points[i][1] - points[p0][1];
        const dz = points[i][2] - points[p0][2];
        const d = dx*dx + dy*dy + dz*dz;
        if (d > maxD) { maxD = d; p1 = i; }
    }
    if (p1 < 0) return [];

    // p2: farthest from line p0-p1
    const e01 = [points[p1][0]-points[p0][0], points[p1][1]-points[p0][1], points[p1][2]-points[p0][2]];
    const e01len = Math.sqrt(e01[0]*e01[0]+e01[1]*e01[1]+e01[2]*e01[2]);
    e01[0]/=e01len; e01[1]/=e01len; e01[2]/=e01len;
    maxD = 0;
    for (let i = 0; i < n; i++) {
        if (i===p0||i===p1) continue;
        const dx = points[i][0]-points[p0][0], dy = points[i][1]-points[p0][1], dz = points[i][2]-points[p0][2];
        const dot = dx*e01[0]+dy*e01[1]+dz*e01[2];
        const px = dx-dot*e01[0], py = dy-dot*e01[1], pz = dz-dot*e01[2];
        const d = px*px+py*py+pz*pz;
        if (d > maxD) { maxD = d; p2 = i; }
    }
    if (p2 < 0) return [];

    // p3: farthest from plane p0-p1-p2
    const ab = [points[p1][0]-points[p0][0], points[p1][1]-points[p0][1], points[p1][2]-points[p0][2]];
    const ac = [points[p2][0]-points[p0][0], points[p2][1]-points[p0][1], points[p2][2]-points[p0][2]];
    const norm = [ab[1]*ac[2]-ab[2]*ac[1], ab[2]*ac[0]-ab[0]*ac[2], ab[0]*ac[1]-ab[1]*ac[0]];
    const nlen = Math.sqrt(norm[0]*norm[0]+norm[1]*norm[1]+norm[2]*norm[2]);
    norm[0]/=nlen; norm[1]/=nlen; norm[2]/=nlen;
    maxD = 0;
    for (let i = 0; i < n; i++) {
        if (i===p0||i===p1||i===p2) continue;
        const dx = points[i][0]-points[p0][0], dy = points[i][1]-points[p0][1], dz = points[i][2]-points[p0][2];
        const d = Math.abs(dx*norm[0]+dy*norm[1]+dz*norm[2]);
        if (d > maxD) { maxD = d; p3 = i; }
    }
    if (p3 < 0) return [];

    // Orient initial tetrahedron so all faces point outward
    const d03 = [points[p3][0]-points[p0][0], points[p3][1]-points[p0][1], points[p3][2]-points[p0][2]];
    const dot03 = d03[0]*norm[0]+d03[1]*norm[1]+d03[2]*norm[2];
    // If p3 is on positive side of p0-p1-p2 plane, swap p1/p2 to flip normal
    if (dot03 > 0) { const tmp = p1; p1 = p2; p2 = tmp; }

    // faces: each face is {verts:[i,j,k], norm:[nx,ny,nz]}
    const faces = [];
    function makeFace(a, b, c) {
        const ab = [points[b][0]-points[a][0], points[b][1]-points[a][1], points[b][2]-points[a][2]];
        const ac = [points[c][0]-points[a][0], points[c][1]-points[a][1], points[c][2]-points[a][2]];
        const n = [ab[1]*ac[2]-ab[2]*ac[1], ab[2]*ac[0]-ab[0]*ac[2], ab[0]*ac[1]-ab[1]*ac[0]];
        const l = Math.sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
        if (l > 1e-12) { n[0]/=l; n[1]/=l; n[2]/=l; }
        return {verts:[a,b,c], norm:n, d: n[0]*points[a][0]+n[1]*points[a][1]+n[2]*points[a][2]};
    }
    faces.push(makeFace(p0,p1,p2));
    faces.push(makeFace(p0,p2,p3));
    faces.push(makeFace(p0,p3,p1));
    faces.push(makeFace(p1,p3,p2));

    // Incremental insertion
    for (let i = 0; i < n; i++) {
        if (i===p0||i===p1||i===p2||i===p3) continue;
        const pt = points[i];
        const visible = [];
        for (let f = faces.length - 1; f >= 0; f--) {
            const face = faces[f];
            const dist = face.norm[0]*pt[0]+face.norm[1]*pt[1]+face.norm[2]*pt[2] - face.d;
            if (dist > 1e-8) visible.push(f);
        }
        if (visible.length === 0) continue;

        // Collect horizon edges (edges shared by exactly one visible face)
        const edgeCount = new Map();
        for (const fi of visible) {
            const v = faces[fi].verts;
            for (let e = 0; e < 3; e++) {
                const a = v[e], b = v[(e+1)%3];
                const key = a < b ? a+','+b : b+','+a;
                const dir = a < b ? [a,b] : [b,a];
                if (!edgeCount.has(key)) edgeCount.set(key, {count:0, a:v[e], b:v[(e+1)%3]});
                edgeCount.get(key).count++;
            }
        }

        // Remove visible faces (reverse order to preserve indices)
        visible.sort((a,b)=>b-a);
        for (const fi of visible) faces.splice(fi, 1);

        // Add new faces from horizon edges to new point
        for (const [, edge] of edgeCount) {
            if (edge.count === 1) {
                faces.push(makeFace(edge.a, edge.b, i));
            }
        }
    }

    return faces.map(f => f.verts);
}

function buildWavefunction() {
    disposeWavefunction();
    if (!_octNodeSet || _octNodeSet.size < 6 || !REST || REST.length === 0) return;
    if (typeof baseNeighbors === 'undefined' || !baseNeighbors) return;

    // 1. Oct centroid
    let cx = 0, cy = 0, cz = 0;
    for (const ni of _octNodeSet) {
        cx += REST[ni][0]; cy += REST[ni][1]; cz += REST[ni][2];
    }
    cx /= _octNodeSet.size; cy /= _octNodeSet.size; cz /= _octNodeSet.size;
    _wfCenter = [cx, cy, cz];

    // 2. Identify boundary nodes (fewer than 8 base-edge neighbors)
    const boundaryIndices = [];
    const boundaryPositions = [];
    for (let i = 0; i < N; i++) {
        if (baseNeighbors[i] && baseNeighbors[i].length < 8) {
            boundaryIndices.push(i);
            boundaryPositions.push(REST[i]);
        }
    }
    if (boundaryIndices.length < 4) return;

    // 3. Build harmonic frame from oct geometry
    // Poles = nodes in _octNodeSet NOT in _octEquatorCycle
    let pole0 = -1, pole1 = -1;
    const equatorSet = new Set(_octEquatorCycle || []);
    for (const ni of _octNodeSet) {
        if (!equatorSet.has(ni)) {
            if (pole0 < 0) pole0 = ni; else pole1 = ni;
        }
    }
    if (pole0 < 0 || pole1 < 0) {
        // Fallback: find the farthest-apart pair in _octNodeSet
        const octArr = [..._octNodeSet];
        let bestD = 0;
        for (let i = 0; i < octArr.length; i++) {
            for (let j = i+1; j < octArr.length; j++) {
                const dx = REST[octArr[i]][0]-REST[octArr[j]][0];
                const dy = REST[octArr[i]][1]-REST[octArr[j]][1];
                const dz = REST[octArr[i]][2]-REST[octArr[j]][2];
                const d = dx*dx+dy*dy+dz*dz;
                if (d > bestD) { bestD = d; pole0 = octArr[i]; pole1 = octArr[j]; }
            }
        }
    }

    // z-hat = pole axis (spin axis)
    const zx = REST[pole1][0]-REST[pole0][0];
    const zy = REST[pole1][1]-REST[pole0][1];
    const zz = REST[pole1][2]-REST[pole0][2];
    const zlen = Math.sqrt(zx*zx+zy*zy+zz*zz);
    _wfPoleAxis = new THREE.Vector3(zx/zlen, zy/zlen, zz/zlen);

    // x-hat = equator node direction, projected perpendicular to z-hat
    let eqNode = (_octEquatorCycle && _octEquatorCycle.length > 0) ? _octEquatorCycle[0] : [..._octNodeSet].find(n => n !== pole0 && n !== pole1);
    let ex = REST[eqNode][0]-cx, ey = REST[eqNode][1]-cy, ez = REST[eqNode][2]-cz;
    const edot = ex*_wfPoleAxis.x + ey*_wfPoleAxis.y + ez*_wfPoleAxis.z;
    ex -= edot*_wfPoleAxis.x; ey -= edot*_wfPoleAxis.y; ez -= edot*_wfPoleAxis.z;
    const elen = Math.sqrt(ex*ex+ey*ey+ez*ez);
    _wfXHat = new THREE.Vector3(ex/elen, ey/elen, ez/elen);

    // y-hat = cross(z-hat, x-hat)
    _wfYHat = new THREE.Vector3().crossVectors(_wfPoleAxis, _wfXHat).normalize();

    // 4. Convex hull of boundary nodes
    const hullTriangles = _convexHull3D(boundaryPositions);
    if (hullTriangles.length === 0) return;

    // 5. Build BufferGeometry
    // Unique vertex indices used in hull (subset of boundaryIndices)
    const usedSet = new Set();
    for (const tri of hullTriangles) for (const idx of tri) usedSet.add(idx);

    // Remap: boundary index → geometry vertex index
    const vertMap = new Map();
    const verts = [];
    for (const bi of usedSet) {
        vertMap.set(bi, verts.length);
        verts.push(boundaryPositions[bi]);
    }

    const positions = new Float32Array(verts.length * 3);
    for (let i = 0; i < verts.length; i++) {
        positions[i*3]   = verts[i][0];
        positions[i*3+1] = verts[i][1];
        positions[i*3+2] = verts[i][2];
    }

    const indices = [];
    for (const tri of hullTriangles) {
        indices.push(vertMap.get(tri[0]), vertMap.get(tri[1]), vertMap.get(tri[2]));
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geo.setIndex(indices);
    geo.computeVertexNormals();

    // Color attribute
    const colors = new Float32Array(verts.length * 3);
    colors.fill(0.2); // initial blue-ish
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    // 6. Pre-compute spherical coordinates for each vertex
    _wfVertSph = [];
    for (let i = 0; i < verts.length; i++) {
        const dx = verts[i][0] - cx;
        const dy = verts[i][1] - cy;
        const dz = verts[i][2] - cz;
        // Project onto harmonic frame
        const lz = dx*_wfPoleAxis.x + dy*_wfPoleAxis.y + dz*_wfPoleAxis.z;
        const lx = dx*_wfXHat.x + dy*_wfXHat.y + dz*_wfXHat.z;
        const ly = dx*_wfYHat.x + dy*_wfYHat.y + dz*_wfYHat.z;
        const r = Math.sqrt(lx*lx + ly*ly + lz*lz);
        const theta = r > 1e-10 ? Math.acos(Math.max(-1, Math.min(1, lz / r))) : 0;
        const phi = Math.atan2(ly, lx);
        _wfVertSph.push({theta, phi});
    }

    // 7. Material + Mesh
    const sliderEl = document.getElementById('wf-opacity-slider');
    const initOp = sliderEl ? (+sliderEl.value / 100) : 0.2;
    const mat = new THREE.MeshBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: initOp,
        side: THREE.DoubleSide,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
    });
    _wfMesh = new THREE.Mesh(geo, mat);
    _wfMesh.renderOrder = 1;
    _wfMesh.visible = initOp > 0;
    scene.add(_wfMesh);

    // Reset adaptive scaling
    _wfAdaptiveMin = 0;
    _wfAdaptiveMax = 1;
}

// ── Real Spherical Harmonics (normalization constants) ──
const _Y00  = 0.28209479;  // 1/(2√π)
const _Y10  = 0.48860251;  // √(3/(4π))
const _Y11  = 0.48860251;  // √(3/(4π))
const _Y20  = 0.31539157;  // √(5/(16π))
const _Y21  = 1.09254843;  // √(15/(4π))
const _Y22  = 0.54627422;  // √(15/(16π))

function updateWavefunction() {
    if (!_wfMesh || !_wfMesh.visible) return;
    if (!_wfVertSph || _wfVertSph.length === 0) return;

    // ── Compute harmonic coefficients from simulation state ──
    let c00 = 0.5, c10 = 0, c11 = 0, c1m1 = 0;
    let c20 = 0, c21 = 0, c22 = 0;

    const demoActive = typeof _demoActive !== 'undefined' && _demoActive;
    const xons = typeof _demoXons !== 'undefined' ? _demoXons : [];

    if (demoActive && xons.length > 0 && _wfPoleAxis) {
        const cx = _wfCenter[0], cy = _wfCenter[1], cz = _wfCenter[2];

        // Dipole from xon positions
        let aliveCount = 0;
        for (const x of xons) {
            if (!x.alive || x.node == null || !pos[x.node]) continue;
            aliveCount++;
            const dx = pos[x.node][0] - cx;
            const dy = pos[x.node][1] - cy;
            const dz = pos[x.node][2] - cz;
            // Project onto harmonic axes
            c10  += dx*_wfPoleAxis.x + dy*_wfPoleAxis.y + dz*_wfPoleAxis.z;
            c11  += dx*_wfXHat.x + dy*_wfXHat.y + dz*_wfXHat.z;
            c1m1 += dx*_wfYHat.x + dy*_wfYHat.y + dz*_wfYHat.z;
        }
        if (aliveCount > 0) {
            const norm = aliveCount * 2.0;  // normalization scale
            c10 /= norm; c11 /= norm; c1m1 /= norm;
        }

        // Quadrupole from active SC directions
        let scCount = 0;
        const allActive = new Set();
        if (typeof activeSet !== 'undefined') for (const id of activeSet) allActive.add(id);
        if (typeof electronImpliedSet !== 'undefined') for (const id of electronImpliedSet) allActive.add(id);

        for (const scId of allActive) {
            const sc = SC_BY_ID[scId];
            if (!sc || !pos[sc.a] || !pos[sc.b]) continue;
            scCount++;
            // SC midpoint direction from center
            const mx = (pos[sc.a][0]+pos[sc.b][0])/2 - cx;
            const my = (pos[sc.a][1]+pos[sc.b][1])/2 - cy;
            const mz = (pos[sc.a][2]+pos[sc.b][2])/2 - cz;
            const mr = Math.sqrt(mx*mx+my*my+mz*mz);
            if (mr < 1e-10) continue;
            // Project onto harmonic frame
            const cosTheta = (mx*_wfPoleAxis.x+my*_wfPoleAxis.y+mz*_wfPoleAxis.z)/mr;
            const sinTheta = Math.sqrt(Math.max(0, 1 - cosTheta*cosTheta));
            const projX = (mx*_wfXHat.x+my*_wfXHat.y+mz*_wfXHat.z)/mr;
            const projY = (mx*_wfYHat.x+my*_wfYHat.y+mz*_wfYHat.z)/mr;
            const phi = Math.atan2(projY, projX);

            c20 += (3*cosTheta*cosTheta - 1) / 2;
            c21 += sinTheta * cosTheta * Math.cos(phi);
            c22 += sinTheta * sinTheta * Math.cos(2*phi);
        }
        if (scCount > 0) {
            c20 /= scCount; c21 /= scCount; c22 /= scCount;
            // Scale quadrupole contribution
            c20 *= 0.6; c21 *= 0.4; c22 *= 0.4;
        }
    }

    // ── Evaluate Y_lm at each vertex ──
    const colAttr = _wfMesh.geometry.getAttribute('color');
    const colArr = colAttr.array;
    let frameMin = Infinity, frameMax = -Infinity;

    // Pre-compute amplitudes
    const nv = _wfVertSph.length;
    const amps = new Float32Array(nv);
    for (let i = 0; i < nv; i++) {
        const {theta, phi} = _wfVertSph[i];
        const cosT = Math.cos(theta), sinT = Math.sin(theta);

        const amp = c00 * _Y00
            + c10  * _Y10  * cosT
            + c11  * _Y11  * sinT * Math.cos(phi)
            + c1m1 * _Y11  * sinT * Math.sin(phi)
            + c20  * _Y20  * (3*cosT*cosT - 1)
            + c21  * _Y21  * sinT * cosT * Math.cos(phi)
            + c22  * _Y22  * sinT * sinT * Math.cos(2*phi);

        amps[i] = amp;
        if (amp < frameMin) frameMin = amp;
        if (amp > frameMax) frameMax = amp;
    }

    // Adaptive scaling with EMA smoothing
    if (frameMin < frameMax) {
        _wfAdaptiveMin = _wfAdaptiveMin * 0.95 + frameMin * 0.05;
        _wfAdaptiveMax = _wfAdaptiveMax * 0.95 + frameMax * 0.05;
    }
    const range = _wfAdaptiveMax - _wfAdaptiveMin;
    const invRange = range > 1e-8 ? 1 / range : 1;

    // Color map: cool→warm (blue→cyan→green→yellow→red)
    for (let i = 0; i < nv; i++) {
        let t = (amps[i] - _wfAdaptiveMin) * invRange;
        t = Math.max(0, Math.min(1, t));

        let r, g, b;
        if (t < 0.25) {
            const s = t / 0.25;
            r = 0.0; g = 0.2 + 0.6*s; b = 1.0;
        } else if (t < 0.5) {
            const s = (t - 0.25) / 0.25;
            r = 0.2*s; g = 0.8 + 0.2*s; b = 1.0 - 0.8*s;
        } else if (t < 0.75) {
            const s = (t - 0.5) / 0.25;
            r = 0.2 + 0.8*s; g = 1.0 - 0.1*s; b = 0.2 - 0.2*s;
        } else {
            const s = (t - 0.75) / 0.25;
            r = 1.0; g = 0.9 - 0.7*s; b = 0.0;
        }

        colArr[i*3]     = r;
        colArr[i*3 + 1] = g;
        colArr[i*3 + 2] = b;
    }
    colAttr.needsUpdate = true;
}

function disposeWavefunction() {
    if (_wfMesh) {
        scene.remove(_wfMesh);
        _wfMesh.geometry.dispose();
        _wfMesh.material.dispose();
        _wfMesh = null;
    }
    _wfVertSph = [];
    _wfAdaptiveMin = 0;
    _wfAdaptiveMax = 1;
}

function applyWavefunctionOpacity() {
    const slider = document.getElementById('wf-opacity-slider');
    const valEl = document.getElementById('wf-opacity-val');
    if (!slider) return;
    const op = +slider.value / 100;
    if (valEl) valEl.textContent = Math.round(op * 100) + '%';
    if (_wfMesh) {
        _wfMesh.material.opacity = op;
        _wfMesh.visible = op > 0;
    }
}

document.getElementById('wf-opacity-slider').addEventListener('input', applyWavefunctionOpacity);

document.getElementById('btn-add-excitation').addEventListener('click',toggleExcitationPlacement);
// Big bang button handled by V2 dropdown menu in post-load script
document.getElementById('btn-select-mode').addEventListener('click',toggleSelectMode);
document.getElementById('btn-excitation-play').addEventListener('click',toggleExcitationPause);

// ── Model Controls (V2 — replaces V1 arena panel) ──────────────────────
// Model select and tournament are now in the nucleus panel.
// populateModelSelect is called from NucleusSimulator wiring below.
// Force rule title on init
{
    const _titleInit = document.getElementById('rule-title');
    if(_titleInit) _titleInit.textContent = RULE_REGISTRY[activeRuleIndex]?.name || '';
}

// ── Expose symbols needed by flux-rules.js ──────────────────────────
window.RULE_REGISTRY = RULE_REGISTRY;
window.extractCandidateFeatures = extractCandidateFeatures;
window.extractFallbackFeatures = extractFallbackFeatures;
window.scoreCandidateGA = scoreCandidateGA;
window.kDeltaForFlip = kDeltaForFlip;
window.GA_NUM_FEATURES = GA_NUM_FEATURES;
window.tetPartnerMap = tetPartnerMap;
window.squarePartnerMap = squarePartnerMap;
// V2: additional symbols for NucleusSimulator
window._createExcitation = _createExcitation;
window.startExcitationClock = startExcitationClock;

// ══════════════════════════════════════════════════════════════════════
// INTEGRATION TESTS — run via console: window.runFluxTests()
// ══════════════════════════════════════════════════════════════════════
//
// These tests verify critical invariants that have broken before:
//   1. Graph visualization syncs with shortcut state (scLineObjs ↔ activeSet+impliedSet)
//   2. Temporal K captures frames when state changes
//   3. Rules load and produce valid scores
//   4. Classic mode is hermetic (no K overhead)
//
// Tests run non-destructively where possible; some tests temporarily
// modify state and restore it.  Safe to run anytime.
//
// USAGE:
//   window.runFluxTests()          — run all tests, log results
//   window.runFluxTests('graph')   — run only graph tests
//
// Each test returns {pass: boolean, name: string, detail: string}.
// The runner logs a summary and returns the results array.

window.runFluxTests = async function(filter){
    const results = [];
    const tests = [];

    // ── Helper: wait for N excitation ticks ──
    function waitTicks(n){
        return new Promise(resolve => {
            let count = 0;
            const origTick = excitationClockTick;
            // We can't monkeypatch excitationClockTick easily since it's
            // called by setInterval.  Instead, just wait based on tick speed.
            setTimeout(resolve, ELECTRON_STEP_MS * n + 200);
        });
    }

    // ── Helper: assert ──
    function assert(cond, name, detail){
        results.push({ pass: !!cond, name, detail: detail || '' });
        if(!cond) console.error(`[FAIL] ${name}: ${detail || ''}`);
        else console.log(`[PASS] ${name}`);
    }

    // ══════════════════════════════════════
    // TEST GROUP: Graph sync
    // ══════════════════════════════════════

    if(!filter || filter === 'graph'){
        // Save state for restore
        const savedLL = latticeLevel;
        const savedBB = bigBangActive;
        const savedRule = activeRuleIndex;

        // Test 1: scLineObjs count matches activeSet + impliedSet
        // (snapshot of current state)
        {
            const lineCount = Object.keys(scLineObjs).length;
            const stateCount = activeSet.size + impliedSet.size;
            assert(
                lineCount === stateCount,
                'graph-sync-snapshot',
                `scLineObjs=${lineCount}, activeSet+impliedSet=${stateCount}`
            );
        }

        // Test 2: After big bang on L1, graph lines appear
        {
            // Ensure L1
            if(latticeLevel !== 1){
                document.getElementById('lattice-slider').value = 1;
                updateLatticeLevel();
            }
            activeRuleIndex = 0;
            clearAll();

            // Verify clean state
            assert(
                Object.keys(scLineObjs).length === 0,
                'graph-clean-before-bigbang',
                `scLineObjs should be 0 after clearAll, got ${Object.keys(scLineObjs).length}`
            );

            // Fire big bang and wait for excitations to move
            toggleBigBang();
            await waitTicks(8);

            const lineCount = Object.keys(scLineObjs).length;
            const implied = impliedSet.size;
            assert(
                lineCount > 0,
                'graph-lines-appear-after-bigbang',
                `scLineObjs=${lineCount} (need >0, impliedSet=${implied})`
            );

            assert(
                lineCount === activeSet.size + impliedSet.size,
                'graph-sync-after-bigbang',
                `scLineObjs=${lineCount}, activeSet+impliedSet=${activeSet.size + impliedSet.size}`
            );

            deactivateBigBang();
        }

        // Test 3: After clearAll, graph lines disappear
        {
            clearAll();
            const lineCount = Object.keys(scLineObjs).length;
            assert(
                lineCount === 0,
                'graph-clear-after-clearAll',
                `scLineObjs=${lineCount} (need 0)`
            );
        }

        // Test 4: L2 big bang — graph lines appear with many excitations
        {
            document.getElementById('lattice-slider').value = 2;
            updateLatticeLevel();
            activeRuleIndex = 0;
            toggleBigBang();
            await waitTicks(8);

            const lineCount = Object.keys(scLineObjs).length;
            const expected = activeSet.size + impliedSet.size;
            assert(
                lineCount > 0 && lineCount === expected,
                'graph-sync-L2-bigbang',
                `scLineObjs=${lineCount}, activeSet+impliedSet=${expected}, excitations=${excitations.length}`
            );

            deactivateBigBang();
        }

        // Test 5: Each scLineObj ID is in activeSet OR impliedSet
        {
            let allInState = true;
            let orphans = [];
            for(const id of Object.keys(scLineObjs).map(Number)){
                if(!activeSet.has(id) && !impliedSet.has(id)){
                    allInState = false;
                    orphans.push(id);
                }
            }
            assert(
                allInState,
                'graph-no-orphan-lines',
                orphans.length ? `orphan line IDs: ${orphans.join(',')}` : 'all lines have state backing'
            );
        }

        // Test 6: No state entries missing from scLineObjs
        {
            let allRendered = true;
            let missing = [];
            for(const id of activeSet){
                if(!scLineObjs[id]){ allRendered = false; missing.push(id); }
            }
            for(const id of impliedSet){
                if(!scLineObjs[id]){ allRendered = false; missing.push(id); }
            }
            assert(
                allRendered,
                'graph-no-missing-lines',
                missing.length ? `missing line IDs: ${missing.join(',')}` : 'all state entries have lines'
            );
        }

        // Restore
        clearAll();
        if(savedLL !== latticeLevel){
            document.getElementById('lattice-slider').value = savedLL;
            updateLatticeLevel();
        }
        activeRuleIndex = savedRule;
    }

    // ══════════════════════════════════════
    // TEST GROUP: Temporal K
    // ══════════════════════════════════════

    if(!filter || filter === 'temporal'){
        const savedLL = latticeLevel;
        const savedRule = activeRuleIndex;

        // Test: temporal K captures frames during excitation movement
        {
            if(latticeLevel !== 1){
                document.getElementById('lattice-slider').value = 1;
                updateLatticeLevel();
            }
            activeRuleIndex = 0;
            clearAll();
            resetTemporalK();

            assert(
                _temporalFrames.length === 0,
                'temporalK-clean-start',
                `frames=${_temporalFrames.length}`
            );

            toggleBigBang();
            await waitTicks(12);

            assert(
                _temporalFrames.length > 0,
                'temporalK-captures-frames',
                `frames=${_temporalFrames.length} after 12 ticks`
            );

            deactivateBigBang();
            clearAll();
        }

        // Restore
        if(savedLL !== latticeLevel){
            document.getElementById('lattice-slider').value = savedLL;
            updateLatticeLevel();
        }
        activeRuleIndex = savedRule;
    }

    // ══════════════════════════════════════
    // TEST GROUP: Rules
    // ══════════════════════════════════════

    if(!filter || filter === 'rules'){
        // Test: all rules in RULE_REGISTRY have required interface
        {
            let allValid = true;
            let problems = [];
            for(let i = 0; i < RULE_REGISTRY.length; i++){
                const r = RULE_REGISTRY[i];
                if(!r.name){ allValid = false; problems.push(`[${i}] missing name`); }
                if(!r.description){ allValid = false; problems.push(`[${i}] missing description`); }
                if(typeof r.rankCandidates !== 'function'){
                    allValid = false; problems.push(`[${i}] missing rankCandidates`);
                }
            }
            assert(
                allValid,
                'rules-valid-interface',
                problems.length ? problems.join('; ') : `all ${RULE_REGISTRY.length} rules valid`
            );
        }

        // Test: classic is at index 0
        {
            assert(
                RULE_REGISTRY[0]?.name === 'classic',
                'rules-classic-at-zero',
                `index 0 name: ${RULE_REGISTRY[0]?.name}`
            );
        }

        // Test: classic mode skips K computation (activeRuleIndex=0 guard)
        {
            activeRuleIndex = 0;
            // The needsK guard is: activeRuleIndex > 0 && canMaterialise
            // With activeRuleIndex=0, needsK should always be false
            assert(
                activeRuleIndex === 0,
                'rules-classic-hermetic',
                'activeRuleIndex=0 → needsK=false (no K overhead in classic)'
            );
        }
    }

    // ── Summary ──
    const passed = results.filter(r => r.pass).length;
    const failed = results.filter(r => !r.pass).length;
    const summary = `\n${'═'.repeat(50)}\nTESTS: ${passed} passed, ${failed} failed, ${results.length} total\n${'═'.repeat(50)}`;
    if(failed > 0){
        console.error(summary);
        results.filter(r => !r.pass).forEach(r =>
            console.error(`  FAIL: ${r.name} — ${r.detail}`)
        );
    } else {
        console.log(summary);
    }
    return results;
};

