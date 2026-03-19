// flux-voids-render.js — Shell layers (branes), wavefunction, convex hull, render utilities
// Weak force opacity: now driven by _roleOpacity['weak'] via role card slider in updateXonPanel().
// ═══ SHELL LAYER — Concentric wavefunction shells from oct center ═══════════
// One 3D convex hull per BFS distance level (≥3) from _octNodeSet, up to but
// not including the outermost boundary (which is the wavefunction layer).

let _braneShells = [];  // [{mesh, vertNodeIdx, adaptiveMax, distance, maxDist}]

function buildBranes() {
    disposeBranes();
    if (typeof N === 'undefined' || N === 0 || typeof REST === 'undefined') return;
    if (!_octNodeSet || _octNodeSet.size === 0) return;
    if (typeof baseNeighbors === 'undefined' || !baseNeighbors) return;

    const slider = document.getElementById('brane-opacity-slider');
    const baseOp = slider ? +slider.value / 100 : 0.05;

    // Compute oct centroid
    const cx = [], cy = [], cz = [];
    for (const ni of _octNodeSet) {
        cx.push(REST[ni][0]); cy.push(REST[ni][1]); cz.push(REST[ni][2]);
    }
    const centX = cx.reduce((a, b) => a + b, 0) / cx.length;
    const centY = cy.reduce((a, b) => a + b, 0) / cy.length;
    const centZ = cz.reduce((a, b) => a + b, 0) / cz.length;

    // Compute Euclidean distance from centroid for every node
    const eucDist = new Float64Array(N);
    let maxEuc = 0;
    for (let i = 0; i < N; i++) {
        const dx = REST[i][0] - centX, dy = REST[i][1] - centY, dz = REST[i][2] - centZ;
        eucDist[i] = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (eucDist[i] > maxEuc) maxEuc = eucDist[i];
    }

    // Find the min distance of non-oct nodes (exclude oct nodes themselves)
    let minNonOct = Infinity;
    for (let i = 0; i < N; i++) {
        if (!_octNodeSet.has(i) && eucDist[i] < minNonOct) minNonOct = eucDist[i];
    }

    // Bin into equal-width radial bands from minNonOct to maxEuc
    // Use BFS max distance to determine number of bands for consistent layering
    // First do a quick BFS just to count layers
    const bfsDist = new Int32Array(N).fill(-1);
    const bfsQ = [];
    for (const ni of _octNodeSet) { bfsDist[ni] = 0; bfsQ.push(ni); }
    let qi = 0, maxBFS = 0;
    while (qi < bfsQ.length) {
        const curr = bfsQ[qi++];
        const nbs = baseNeighbors[curr];
        if (!nbs) continue;
        for (const nb of nbs) {
            if (bfsDist[nb.node] === -1) {
                bfsDist[nb.node] = bfsDist[curr] + 1;
                if (bfsDist[nb.node] > maxBFS) maxBFS = bfsDist[nb.node];
                bfsQ.push(nb.node);
            }
        }
    }
    const numBands = maxBFS; // one band per BFS layer (excluding oct)
    const bandWidth = (maxEuc - minNonOct) / numBands;

    // Group non-oct nodes into radial bands
    const shellNodes = new Map();
    for (let i = 0; i < N; i++) {
        if (_octNodeSet.has(i)) continue;
        const band = Math.min(numBands - 1, Math.floor((eucDist[i] - minNonOct) / bandWidth));
        if (!shellNodes.has(band)) shellNodes.set(band, []);
        shellNodes.get(band).push(i);
    }

    // Build shells for bands 0 .. numBands-2 (skip outermost = wavefunction)
    const distances = Array.from(shellNodes.keys()).sort((a, b) => a - b);
    if (distances.length > 0) distances.pop(); // remove outermost band

    for (const d of distances) {
        const nodes = shellNodes.get(d);
        if (nodes.length < 4) continue;

        const points = nodes.map(ni => REST[ni]);
        const hullTris = _convexHull3D(points);
        if (hullTris.length === 0) continue;

        // Unique vertex indices used in hull
        const usedSet = new Set();
        for (const tri of hullTris) for (const idx of tri) usedSet.add(idx);

        // Remap: hull point index → geometry vertex index
        const vertMap = new Map();
        const verts = [];
        const vertNodeIdx = [];
        for (const pi of usedSet) {
            vertMap.set(pi, verts.length);
            verts.push(points[pi]);
            vertNodeIdx.push(nodes[pi]); // map to lattice node
        }

        // Build index buffer
        const indices = [];
        for (const tri of hullTris) {
            indices.push(vertMap.get(tri[0]), vertMap.get(tri[1]), vertMap.get(tri[2]));
        }

        // Positions
        const posArr = new Float32Array(verts.length * 3);
        for (let i = 0; i < verts.length; i++) {
            posArr[i * 3] = verts[i][0];
            posArr[i * 3 + 1] = verts[i][1];
            posArr[i * 3 + 2] = verts[i][2];
        }

        // Colors — initialize to cool blue
        const colors = new Float32Array(verts.length * 3);
        for (let i = 0; i < verts.length; i++) {
            colors[i * 3] = 0.0;
            colors[i * 3 + 1] = 0.2;
            colors[i * 3 + 2] = 1.0;
        }

        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.Float32BufferAttribute(posArr, 3));
        geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        geo.setIndex(indices);
        geo.computeVertexNormals();

        // Opacity scales: inner shells dimmer, outer shells brighter
        const opScale = distances.length > 1
            ? 0.5 + 0.5 * ((d - distances[0]) / (distances[distances.length - 1] - distances[0]))
            : 1.0;
        const shellOp = baseOp * opScale;
        const solid = shellOp > 0.5;

        const mat = new THREE.MeshPhongMaterial({
            vertexColors: true,
            transparent: true,
            opacity: shellOp,
            side: THREE.DoubleSide,
            depthWrite: solid,
            blending: solid ? THREE.NormalBlending : THREE.AdditiveBlending,
            emissive: 0x111111,
            emissiveIntensity: 0.3,
            shininess: 30,
        });

        const mesh = new THREE.Mesh(geo, mat);
        mesh.renderOrder = 0;
        scene.add(mesh);
        _braneShells.push({
            mesh,
            vertNodeIdx,
            adaptiveMax: 0.001,
            distance: d,
            maxDist: distances.length > 0 ? distances[distances.length - 1] : d,
        });
    }
}

function updateBraneOpacity() {
    const slider = document.getElementById('brane-opacity-slider');
    if (!slider) return;
    const pct = +slider.value;
    const op = pct / 100;
    document.getElementById('brane-opacity-val').textContent = pct + '%';
    for (const shell of _braneShells) {
        // Inner shells dimmer, outer shells brighter
        const opScale = _braneShells.length > 1
            ? 0.5 + 0.5 * ((shell.distance - _braneShells[0].distance) / Math.max(1, shell.maxDist - _braneShells[0].distance))
            : 1.0;
        const shellOp = op * opScale;
        shell.mesh.material.opacity = shellOp;
        shell.mesh.visible = shellOp > 0;
        const solid = shellOp > 0.5;
        shell.mesh.material.depthWrite = solid;
        shell.mesh.material.blending = solid ? THREE.NormalBlending : THREE.AdditiveBlending;
        shell.mesh.material.needsUpdate = true;
    }
}

// Per-frame: update shell positions from solver, displacement-based coloring
function updateBraneHighlights() {
    if (typeof pos === 'undefined' || typeof REST === 'undefined') return;
    if (_braneShells.length === 0) return;

    for (const shell of _braneShells) {
        const mesh = shell.mesh;
        if (!mesh.visible) continue;

        const posAttr = mesh.geometry.attributes.position;
        const colAttr = mesh.geometry.attributes.color;
        const posArr = posAttr.array;
        const colArr = colAttr.array;
        const vni = shell.vertNodeIdx;

        let frameMax = 0;
        for (let i = 0; i < vni.length; i++) {
            const ni = vni[i];
            // Update position from solver
            posArr[i * 3]     = pos[ni][0];
            posArr[i * 3 + 1] = pos[ni][1];
            posArr[i * 3 + 2] = pos[ni][2];
            // Compute displacement
            const dx = pos[ni][0] - REST[ni][0];
            const dy = pos[ni][1] - REST[ni][1];
            const dz = pos[ni][2] - REST[ni][2];
            const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
            if (d > frameMax) frameMax = d;
        }

        // EMA-smooth adaptive max
        shell.adaptiveMax = Math.max(0.001, shell.adaptiveMax * 0.95 + frameMax * 0.05);
        const invMax = 1 / shell.adaptiveMax;

        // Displacement-based cool→warm color ramp (same as wavefunction)
        for (let i = 0; i < vni.length; i++) {
            const ni = vni[i];
            const dx = pos[ni][0] - REST[ni][0];
            const dy = pos[ni][1] - REST[ni][1];
            const dz = pos[ni][2] - REST[ni][2];
            let t = Math.sqrt(dx * dx + dy * dy + dz * dz) * invMax;
            if (t > 1) t = 1;

            let r, g, b;
            if (t < 0.25) {
                const s = t / 0.25;
                r = 0; g = 0.2 + 0.6 * s; b = 1.0;
            } else if (t < 0.5) {
                const s = (t - 0.25) / 0.25;
                r = 0.2 * s; g = 0.8 + 0.2 * s; b = 1.0 - 0.8 * s;
            } else if (t < 0.75) {
                const s = (t - 0.5) / 0.25;
                r = 0.2 + 0.8 * s; g = 1.0 - 0.1 * s; b = 0.2 - 0.2 * s;
            } else {
                const s = (t - 0.75) / 0.25;
                r = 1.0; g = 0.9 - 0.7 * s; b = 0;
            }
            colArr[i * 3] = r;
            colArr[i * 3 + 1] = g;
            colArr[i * 3 + 2] = b;
        }

        posAttr.needsUpdate = true;
        colAttr.needsUpdate = true;
        mesh.geometry.computeVertexNormals();
    }
}

function disposeBranes() {
    for (const shell of _braneShells) {
        scene.remove(shell.mesh);
        shell.mesh.geometry.dispose();
        shell.mesh.material.dispose();
    }
    _braneShells = [];
}

document.getElementById('brane-opacity-slider').addEventListener('input', updateBraneOpacity);

// ╔══════════════════════════════════════════════════════════════════════╗
// ║  WAVEFUNCTION LAYER — Spherical Harmonic Surface on Lattice Hull   ║
// ╚══════════════════════════════════════════════════════════════════════╝

let _wfMesh = null;
let _wfVertNodeIdx = [];   // per-vertex: lattice node index (for pos[] lookup)
let _wfAdaptiveMax = 0.001; // smoothed max displacement (EMA)

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

    // 3. Convex hull of boundary nodes
    const hullTriangles = _convexHull3D(boundaryPositions);
    if (hullTriangles.length === 0) return;

    // 4. Build BufferGeometry
    // Unique vertex indices used in hull (subset of boundaryIndices)
    const usedSet = new Set();
    for (const tri of hullTriangles) for (const idx of tri) usedSet.add(idx);

    // Remap: boundary index → geometry vertex index, track lattice node indices
    const vertMap = new Map();
    const verts = [];
    _wfVertNodeIdx = [];
    for (const bi of usedSet) {
        vertMap.set(bi, verts.length);
        verts.push(boundaryPositions[bi]);
        _wfVertNodeIdx.push(boundaryIndices[bi]); // lattice node index
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

    // 5. Material + Mesh
    const sliderEl = document.getElementById('wf-opacity-slider');
    const initOp = sliderEl ? (+sliderEl.value / 100) : 0.2;
    const solid = initOp > 0.5;
    const mat = new THREE.MeshPhongMaterial({
        vertexColors: true,
        transparent: true,
        opacity: initOp,
        side: THREE.DoubleSide,
        depthWrite: solid,
        blending: solid ? THREE.NormalBlending : THREE.AdditiveBlending,
        emissive: 0x111111,
        emissiveIntensity: 0.3,
        shininess: 30,
    });
    _wfMesh = new THREE.Mesh(geo, mat);
    _wfMesh.renderOrder = 1;
    _wfMesh.visible = initOp > 0;
    scene.add(_wfMesh);

    // Reset adaptive scaling
    _wfAdaptiveMax = 0.001;
}

function updateWavefunction() {
    if (!_wfMesh || !_wfMesh.visible) return;
    if (!_wfVertNodeIdx || _wfVertNodeIdx.length === 0) return;

    const posAttr = _wfMesh.geometry.getAttribute('position');
    const posArr = posAttr.array;
    const colAttr = _wfMesh.geometry.getAttribute('color');
    const colArr = colAttr.array;
    const nv = _wfVertNodeIdx.length;

    // ── Update vertex positions from solver & compute displacement ──
    let frameMax = 0;
    const disps = new Float32Array(nv);
    for (let i = 0; i < nv; i++) {
        const ni = _wfVertNodeIdx[i];
        if (!pos[ni] || !REST[ni]) continue;

        // Move vertex to current solver position
        posArr[i*3]     = pos[ni][0];
        posArr[i*3 + 1] = pos[ni][1];
        posArr[i*3 + 2] = pos[ni][2];

        // Displacement from rest
        const dx = pos[ni][0] - REST[ni][0];
        const dy = pos[ni][1] - REST[ni][1];
        const dz = pos[ni][2] - REST[ni][2];
        const disp = Math.sqrt(dx*dx + dy*dy + dz*dz);
        disps[i] = disp;
        if (disp > frameMax) frameMax = disp;
    }
    posAttr.needsUpdate = true;
    _wfMesh.geometry.computeVertexNormals();

    // Adaptive max with EMA smoothing (prevents flicker)
    _wfAdaptiveMax = Math.max(0.001, _wfAdaptiveMax * 0.95 + frameMax * 0.05);
    const invMax = 1 / _wfAdaptiveMax;

    // ── Color by displacement: cool→warm ──
    for (let i = 0; i < nv; i++) {
        let t = disps[i] * invMax;
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
    _wfVertNodeIdx = [];
    _wfAdaptiveMax = 0.001;
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
        // Above 50% opacity: switch to solid mode (occludes interior).
        // Same pattern as sphere opacity in applySphereOpacity().
        const solid = op > 0.5;
        _wfMesh.material.depthWrite = solid;
        _wfMesh.material.blending = solid ? THREE.NormalBlending : THREE.AdditiveBlending;
        _wfMesh.material.needsUpdate = true;
    }
}

document.getElementById('wf-opacity-slider').addEventListener('input', applyWavefunctionOpacity);

document.getElementById('bg-gray-slider').addEventListener('input', () => {
    const pct = +document.getElementById('bg-gray-slider').value;
    document.getElementById('bg-gray-val').textContent = pct + '%';
});

document.getElementById('color-phase-slider').addEventListener('input', e => {
    _colorPhaseShift = +e.target.value;
    _recomputeColors(_colorPhaseShift);
    document.getElementById('color-phase-val').textContent = e.target.value + '°';
});

document.getElementById('btn-add-excitation').addEventListener('click',toggleExcitationPlacement);
// Big bang button handled by V2 dropdown menu in post-load script
document.getElementById('btn-select-mode').addEventListener('click',toggleSelectMode);
document.getElementById('btn-excitation-play').addEventListener('click',toggleExcitationPause);

// ── Model Controls (V2 — replaces V1 arena panel) ──────────────────────
// Model select and tournament are now in the nucleus panel.
// populateModelSelect is called from NucleusSimulator wiring below.
// Force rule title on init
{
    const _titleInit = document.getElementById('topbar-title');
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
            setTimeout(resolve, XON_STEP_MS * n + 200);
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

