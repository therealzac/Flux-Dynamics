// flux-rl.js — Two-layer RL for deuteron choreography
// Strategic layer: face/quark assignment scoring (22→24→24→1)
//   16 spatial features + 6 temporal features
// Tactical layer: oct move candidate scoring (22→32→32→1)
// Both layers' weights form a single genome evolved via GA.

// ── Architecture constants ──
const RL_STRATEGIC_FEATURES = 22;
const RL_STRATEGIC_HIDDEN = 24;
const RL_NUM_FEATURES = 22;   // tactical (unchanged)
const RL_HIDDEN_1 = 32;
const RL_HIDDEN_2 = 32;

// ── Genome layout ──
// Strategic: 22×24 + 24 + 24×24 + 24 + 24×1 + 1 = 1177
// Tactical:  22×32 + 32 + 32×32 + 32 + 32×1 + 1 = 1825
// Total: 3002
const RL_STRATEGIC_SIZE = RL_STRATEGIC_FEATURES * RL_STRATEGIC_HIDDEN + RL_STRATEGIC_HIDDEN
    + RL_STRATEGIC_HIDDEN * RL_STRATEGIC_HIDDEN + RL_STRATEGIC_HIDDEN
    + RL_STRATEGIC_HIDDEN * 1 + 1;
const RL_TACTICAL_SIZE = RL_NUM_FEATURES * RL_HIDDEN_1 + RL_HIDDEN_1
    + RL_HIDDEN_1 * RL_HIDDEN_2 + RL_HIDDEN_2
    + RL_HIDDEN_2 * 1 + 1;

// ── TF.js availability ──
let _rlAvailable = false;
let _rlModel = null;           // tactical model (oct move scoring)
let _rlStrategicModel = null;  // strategic model (face/quark scoring)

function _rlCheckAvailable() {
    if (typeof tf !== 'undefined' && tf.sequential) {
        _rlAvailable = true;
        return true;
    }
    return false;
}

// ── Create tactical MLP (22→32→32→1) ──
function createPolicyModel() {
    if (!_rlCheckAvailable()) return null;
    const model = tf.sequential();
    model.add(tf.layers.dense({
        inputShape: [RL_NUM_FEATURES], units: RL_HIDDEN_1,
        activation: 'relu', kernelInitializer: 'glorotNormal',
    }));
    model.add(tf.layers.dense({
        units: RL_HIDDEN_2, activation: 'relu', kernelInitializer: 'glorotNormal',
    }));
    model.add(tf.layers.dense({
        units: 1, activation: 'linear', kernelInitializer: 'glorotNormal',
    }));
    model.predict(tf.zeros([1, RL_NUM_FEATURES])).dispose();
    return model;
}

// ── Create strategic MLP (16→24→24→1) ──
function createStrategicModel() {
    if (!_rlCheckAvailable()) return null;
    const model = tf.sequential();
    model.add(tf.layers.dense({
        inputShape: [RL_STRATEGIC_FEATURES], units: RL_STRATEGIC_HIDDEN,
        activation: 'relu', kernelInitializer: 'glorotNormal',
    }));
    model.add(tf.layers.dense({
        units: RL_STRATEGIC_HIDDEN, activation: 'relu', kernelInitializer: 'glorotNormal',
    }));
    model.add(tf.layers.dense({
        units: 1, activation: 'linear', kernelInitializer: 'glorotNormal',
    }));
    model.predict(tf.zeros([1, RL_STRATEGIC_FEATURES])).dispose();
    return model;
}

// ── Inference ──
function scoreCandidateRL(features, model) {
    if (!model || !_rlAvailable) return 0;
    const input = tf.tensor2d([features], [1, RL_NUM_FEATURES]);
    const output = model.predict(input);
    const score = output.dataSync()[0];
    input.dispose(); output.dispose();
    return isFinite(score) ? score : 0;
}

function scoreStrategicRL(features, model) {
    if (!model || !_rlAvailable) return 0;
    const input = tf.tensor2d([features], [1, RL_STRATEGIC_FEATURES]);
    const output = model.predict(input);
    const score = output.dataSync()[0];
    input.dispose(); output.dispose();
    return isFinite(score) ? score : 0;
}

function scoreCandidatesBatchRL(featureArray, model) {
    if (!model || !_rlAvailable || featureArray.length === 0) {
        return new Float32Array(featureArray.length);
    }
    const flat = new Float32Array(featureArray.length * RL_NUM_FEATURES);
    for (let i = 0; i < featureArray.length; i++) flat.set(featureArray[i], i * RL_NUM_FEATURES);
    const input = tf.tensor2d(flat, [featureArray.length, RL_NUM_FEATURES]);
    const output = model.predict(input);
    const scores = output.dataSync();
    input.dispose(); output.dispose();
    return scores;
}

// ── Genome ↔ Model conversion (combined genome) ──
function getGenomeSize() { return RL_STRATEGIC_SIZE + RL_TACTICAL_SIZE; }

function modelToGenome(strategicModel, tacticalModel) {
    const size = getGenomeSize();
    const genome = new Float32Array(size);
    let offset = 0;
    // Strategic weights first
    if (strategicModel) {
        for (const w of strategicModel.getWeights()) {
            genome.set(w.dataSync(), offset);
            offset += w.size;
        }
    } else {
        offset = RL_STRATEGIC_SIZE;
    }
    // Tactical weights second
    if (tacticalModel) {
        for (const w of tacticalModel.getWeights()) {
            genome.set(w.dataSync(), offset);
            offset += w.size;
        }
    }
    return genome;
}

function genomeToModel(genome, strategicModel, tacticalModel) {
    if (!genome || genome.length === 0) return;
    // Load strategic weights [0..RL_STRATEGIC_SIZE)
    if (strategicModel) {
        const weights = strategicModel.getWeights();
        let offset = 0;
        const newWeights = [];
        for (const w of weights) {
            newWeights.push(tf.tensor(genome.slice(offset, offset + w.size), w.shape));
            offset += w.size;
        }
        strategicModel.setWeights(newWeights);
        for (const w of newWeights) w.dispose();
    }
    // Load tactical weights [RL_STRATEGIC_SIZE..end)
    if (tacticalModel) {
        const weights = tacticalModel.getWeights();
        let offset = RL_STRATEGIC_SIZE;
        const newWeights = [];
        for (const w of weights) {
            newWeights.push(tf.tensor(genome.slice(offset, offset + w.size), w.shape));
            offset += w.size;
        }
        tacticalModel.setWeights(newWeights);
        for (const w of newWeights) w.dispose();
    }
}

// ── GA operations on combined genome ──
function rlMutateGenome(genome) {
    const m = new Float32Array(genome);
    const nMutations = 1 + Math.floor(Math.random() * Math.max(1, genome.length / 50));
    for (let i = 0; i < nMutations; i++) {
        const idx = Math.floor(Math.random() * m.length);
        m[idx] += (Math.random() - 0.5) * 0.4;
    }
    return m;
}

function rlCrossoverGenome(a, b) {
    const child = new Float32Array(a.length);
    // Block-level crossover: swap strategic or tactical block with 50% each
    const blocks = [
        [0, RL_STRATEGIC_SIZE],                          // strategic block
        [RL_STRATEGIC_SIZE, RL_STRATEGIC_SIZE + RL_TACTICAL_SIZE], // tactical block
    ];
    for (const [start, end] of blocks) {
        const src = Math.random() < 0.5 ? a : b;
        child.set(src.slice(start, end), start);
    }
    return child;
}

function rlRandomGenome() {
    const size = getGenomeSize();
    const g = new Float32Array(size);
    let idx = 0;
    // Strategic layer (22→24→24→1)
    const ss1 = Math.sqrt(2 / RL_STRATEGIC_FEATURES);
    const ss2 = Math.sqrt(2 / RL_STRATEGIC_HIDDEN);
    for (let i = 0; i < RL_STRATEGIC_FEATURES * RL_STRATEGIC_HIDDEN; i++) g[idx++] = (Math.random() - 0.5) * 2 * ss1;
    for (let i = 0; i < RL_STRATEGIC_HIDDEN; i++) g[idx++] = 0;
    for (let i = 0; i < RL_STRATEGIC_HIDDEN * RL_STRATEGIC_HIDDEN; i++) g[idx++] = (Math.random() - 0.5) * 2 * ss2;
    for (let i = 0; i < RL_STRATEGIC_HIDDEN; i++) g[idx++] = 0;
    for (let i = 0; i < RL_STRATEGIC_HIDDEN; i++) g[idx++] = (Math.random() - 0.5) * 2 * ss2;
    g[idx++] = 0;
    // Tactical layer (22→32→32→1)
    const st1 = Math.sqrt(2 / RL_NUM_FEATURES);
    const st2 = Math.sqrt(2 / RL_HIDDEN_1);
    const st3 = Math.sqrt(2 / RL_HIDDEN_2);
    for (let i = 0; i < RL_NUM_FEATURES * RL_HIDDEN_1; i++) g[idx++] = (Math.random() - 0.5) * 2 * st1;
    for (let i = 0; i < RL_HIDDEN_1; i++) g[idx++] = 0;
    for (let i = 0; i < RL_HIDDEN_1 * RL_HIDDEN_2; i++) g[idx++] = (Math.random() - 0.5) * 2 * st2;
    for (let i = 0; i < RL_HIDDEN_2; i++) g[idx++] = 0;
    for (let i = 0; i < RL_HIDDEN_2; i++) g[idx++] = (Math.random() - 0.5) * 2 * st3;
    g[idx++] = 0;
    return g;
}

// ── Strategic feature extraction (face/quark scoring) ──
// 22 features: 16 spatial + 6 temporal for scoring a (face, quarkType) proposal.
function extractStrategicFeatures(xon, face, quarkType, occupied) {
    const f = new Float32Array(RL_STRATEGIC_FEATURES);
    const v = _demoVisits ? (_demoVisits[face] || {}) : {};
    const types = ['pu1', 'pu2', 'pd', 'nd1', 'nd2', 'nu'];
    const counts = types.map(t => v[t] || 0);
    const sum = counts.reduce((a, b) => a + b, 0);
    const mean = sum / 6 || 1;

    // f[0-5]: Per-type ratio deficit for this face (how far each type is from mean)
    for (let i = 0; i < 6; i++) f[i] = Math.max(0, (mean - counts[i]) / mean);

    // f[6]: Is proton face
    f[6] = typeof A_SET !== 'undefined' && A_SET.has(face) ? 1 : 0;

    // f[7]: Face vacancy (no xon executing loop on face)
    let vacant = 1;
    if (typeof _demoXons !== 'undefined') {
        for (const x of _demoXons) {
            if (x.alive && x !== xon && (x._mode === 'tet' || x._mode === 'idle_tet') && x._assignedFace === face) {
                vacant = 0; break;
            }
        }
    }
    f[7] = vacant;

    // f[8]: Target quark type index (normalized 0-1)
    const typeIdx = types.indexOf(quarkType);
    f[8] = typeIdx >= 0 ? typeIdx / 5 : 0;

    // f[9]: Target type deficit specifically (higher = more needed)
    f[9] = typeIdx >= 0 ? Math.max(0, (mean - counts[typeIdx]) / mean) : 0;

    // f[10]: SC activation fraction — how many face SCs are already active
    if (typeof _nucleusTetFaceData !== 'undefined' && _nucleusTetFaceData[face]) {
        const fd = _nucleusTetFaceData[face];
        const scIds = fd.scIds || [];
        let activeCount = 0;
        for (const sid of scIds) {
            if ((typeof activeSet !== 'undefined' && activeSet.has(sid)) ||
                (typeof xonImpliedSet !== 'undefined' && xonImpliedSet.has(sid)) ||
                (typeof impliedSet !== 'undefined' && impliedSet.has(sid))) {
                activeCount++;
            }
        }
        f[10] = scIds.length > 0 ? activeCount / scIds.length : 0;
    }

    // f[11]: Xon on or near face (reachability quality)
    if (typeof _nucleusTetFaceData !== 'undefined' && _nucleusTetFaceData[face]) {
        const fd = _nucleusTetFaceData[face];
        const faceOctNodes = [];
        for (const n of fd.cycle) {
            if (typeof _octNodeSet !== 'undefined' && _octNodeSet.has(n)) faceOctNodes.push(n);
        }
        if (faceOctNodes.includes(xon.node)) f[11] = 1;
        else {
            for (const nb of (baseNeighbors[xon.node] || [])) {
                if (faceOctNodes.includes(nb.node)) { f[11] = 0.5; break; }
            }
        }
    }

    // f[12-15]: Xon's top 4 direction deficits (from _dirBalance)
    const db = xon._dirBalance || new Array(10).fill(0);
    let maxCount = 0;
    for (let i = 0; i < 10; i++) if (db[i] > maxCount) maxCount = db[i];
    if (maxCount > 0) {
        const deficits = [];
        for (let i = 0; i < 10; i++) deficits.push((maxCount - db[i]) / maxCount);
        deficits.sort((a, b) => b - a);
        for (let i = 0; i < 4; i++) f[12 + i] = deficits[i] || 0;
    }

    // ── TEMPORAL FEATURES (f[16]-f[21]) ──
    const ts = typeof _rlTemporalState !== 'undefined' ? _rlTemporalState : null;
    const tick = typeof _demoTick !== 'undefined' ? _demoTick : 0;

    // f[16]: Face recency — ticks since this face was last visited (normalized, capped at 256)
    if (ts && ts.faceLastVisitTick[face] != null) {
        f[16] = Math.min(1, (tick - ts.faceLastVisitTick[face]) / 256);
    } else {
        f[16] = 1; // never visited = max recency signal
    }

    // f[17]: Ratio velocity — is this face's balance improving or deteriorating?
    // Positive = deteriorating (CV increased), negative = improving (CV decreased)
    if (ts && ts.prevFaceCV[face] != null && sum > 0) {
        let variance = 0;
        for (let i = 0; i < 6; i++) variance += (counts[i] - mean) ** 2;
        const currentCV = Math.sqrt(variance / 6) / mean;
        f[17] = Math.max(0, Math.min(1, (currentCV - ts.prevFaceCV[face]) + 0.5));
    } else {
        f[17] = 0.5; // no history — neutral
    }

    // f[18]-f[19]: Cycle phase — sin/cos of position in 64-tick cycle
    const phase = (tick % 64) / 64 * 2 * Math.PI;
    f[18] = (Math.sin(phase) + 1) / 2;  // normalized [0,1]
    f[19] = (Math.cos(phase) + 1) / 2;

    // f[20]: Xon idle duration — how long this xon has been in oct mode (normalized)
    const ms = xon._modeStats || { oct: 0, tet: 0, idle_tet: 0, weak: 0 };
    const totalTicks = ms.oct + ms.tet + ms.idle_tet + ms.weak + 1;
    const octSince = xon._octModeSince || 0;
    f[20] = xon._mode === 'oct' && octSince > 0 ? Math.min(1, (tick - octSince) / 64) : 0;

    // f[21]: Global pressure — fraction of faces below-target on balance
    f[21] = ts ? ts.globalPressure : 0;

    return f;
}

// ── Tactical feature extraction (oct move scoring) ──
function extractRLFeatures(xon, candidate, occupied) {
    const f = new Float32Array(RL_NUM_FEATURES);
    const db = xon._dirBalance || new Array(10).fill(0);
    const ms = xon._modeStats || { oct: 0, tet: 0, idle_tet: 0, weak: 0 };
    const totalTicks = ms.oct + ms.tet + ms.idle_tet + ms.weak + 1;

    let maxCount = 0;
    for (let i = 0; i < 10; i++) if (db[i] > maxCount) maxCount = db[i];
    const dirIdx = candidate.dirIdx != null ? candidate.dirIdx : -1;
    f[0] = dirIdx >= 0 && dirIdx <= 9 ? (maxCount - db[dirIdx]) / (maxCount + 1) : 0;

    let sum = 0;
    for (let i = 0; i < 10; i++) sum += db[i];
    const mean = sum / 10;
    if (mean > 0) {
        let variance = 0;
        for (let i = 0; i < 10; i++) variance += (db[i] - mean) ** 2;
        f[1] = 1 - Math.sqrt(variance / 10) / mean;
    }

    f[2] = candidate._needsMaterialise ? 0 : 1;
    f[3] = _octNodeSet && _octNodeSet.has(candidate.node) ? 1 : 0;
    const nbs = baseNeighbors[candidate.node];
    f[4] = nbs ? Math.min(1, nbs.length / 12) : 0;
    f[5] = candidate.node === xon.prevNode ? 0 : 1;
    f[6] = ms.oct / totalTicks;
    f[7] = ms.tet / totalTicks;
    f[8] = ms.idle_tet / totalTicks;
    for (let d = 4; d < 10; d++) f[5 + d] = maxCount > 0 ? (maxCount - db[d]) / maxCount : 0;
    for (let d = 0; d < 4; d++) f[15 + d] = maxCount > 0 ? (maxCount - db[d]) / maxCount : 0;
    let occCount = 0;
    if (occupied && _octNodeSet) {
        for (const n of _octNodeSet) { if (occupied.has(n)) occCount++; }
        f[19] = occCount / Math.max(1, _octNodeSet.size);
    }
    f[20] = ms.weak / totalTicks;
    f[21] = Math.random() * 0.1;
    return f;
}

// ── IndexedDB persistence for best genome ──
const _RL_IDB_NAME = 'flux_rl_genome';
const _RL_IDB_VERSION = 1;
const _RL_IDB_STORE = 'genomes';
let _rlIDB = null;

async function _rlIDBOpen() {
    return new Promise((resolve) => {
        try {
            const req = indexedDB.open(_RL_IDB_NAME, _RL_IDB_VERSION);
            req.onupgradeneeded = (e) => {
                const db = e.target.result;
                if (!db.objectStoreNames.contains(_RL_IDB_STORE)) db.createObjectStore(_RL_IDB_STORE);
            };
            req.onsuccess = (e) => { _rlIDB = e.target.result; resolve(); };
            req.onerror = () => resolve();
        } catch (e) { resolve(); }
    });
}

async function rlSaveGenome(genome, fitness) {
    if (!_rlIDB) await _rlIDBOpen();
    if (!_rlIDB) return;
    try {
        const tx = _rlIDB.transaction(_RL_IDB_STORE, 'readwrite');
        tx.objectStore(_RL_IDB_STORE).put({
            genome: Array.from(genome),
            fitness,
            timestamp: Date.now(),
            size: genome.length,
        }, 'best');
    } catch (e) { console.warn('[RL] Failed to save genome:', e); }
}

async function rlLoadGenome() {
    if (!_rlIDB) await _rlIDBOpen();
    if (!_rlIDB) return null;
    return new Promise((resolve) => {
        try {
            const tx = _rlIDB.transaction(_RL_IDB_STORE, 'readonly');
            const req = tx.objectStore(_RL_IDB_STORE).get('best');
            req.onsuccess = () => {
                const data = req.result;
                if (data && data.genome) {
                    const genome = new Float32Array(data.genome);
                    console.log(`[RL] Loaded genome from IDB (fitness=${data.fitness?.toFixed(3)}, ${genome.length} params)`);
                    resolve({ genome, fitness: data.fitness || 0 });
                } else resolve(null);
            };
            req.onerror = () => resolve(null);
        } catch (e) { resolve(null); }
    });
}

// ── Initialization ──
async function initRL() {
    if (!_rlCheckAvailable()) {
        console.log('[RL] TF.js not loaded — RL scoring disabled');
        return false;
    }
    _rlModel = createPolicyModel();
    _rlStrategicModel = createStrategicModel();
    if (_rlModel && _rlStrategicModel) {
        console.log(`[RL] Models created: strategic ${RL_STRATEGIC_FEATURES}→${RL_STRATEGIC_HIDDEN}→1 (${RL_STRATEGIC_SIZE}p) + tactical ${RL_NUM_FEATURES}→${RL_HIDDEN_1}→1 (${RL_TACTICAL_SIZE}p) = ${getGenomeSize()} total`);
        // Try loading best genome from IndexedDB
        const saved = await rlLoadGenome();
        if (saved && saved.genome.length === getGenomeSize()) {
            genomeToModel(saved.genome, _rlStrategicModel, _rlModel);
            _rlActiveModel = _rlModel;
            console.log(`[RL] Best genome loaded (fitness=${saved.fitness?.toFixed(3)})`);
        }
        return true;
    }
    return false;
}
