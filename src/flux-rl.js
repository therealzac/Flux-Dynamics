// flux-rl.js — Hybrid GA+NN reinforcement learning for move scoring
// Replaces linear scoreCandidateGA with a 2-layer MLP. GA tournament
// evolves the NN weights (flattened genome) instead of 10 linear coefficients.

// ── Feature count for expanded candidate scoring ──
const RL_NUM_FEATURES = 22;

// ── Model architecture ──
const RL_HIDDEN_1 = 32;
const RL_HIDDEN_2 = 32;

// ── TF.js availability check ──
let _rlAvailable = false;
let _rlModel = null;

function _rlCheckAvailable() {
    if (typeof tf !== 'undefined' && tf.sequential) {
        _rlAvailable = true;
        return true;
    }
    return false;
}

// ── Create the policy MLP ──
// Input: RL_NUM_FEATURES → 32 ReLU → 32 ReLU → 1 linear (score)
function createPolicyModel() {
    if (!_rlCheckAvailable()) {
        console.warn('[RL] TF.js not available — falling back to linear scoring');
        return null;
    }
    const model = tf.sequential();
    model.add(tf.layers.dense({
        inputShape: [RL_NUM_FEATURES],
        units: RL_HIDDEN_1,
        activation: 'relu',
        kernelInitializer: 'glorotNormal',
    }));
    model.add(tf.layers.dense({
        units: RL_HIDDEN_2,
        activation: 'relu',
        kernelInitializer: 'glorotNormal',
    }));
    model.add(tf.layers.dense({
        units: 1,
        activation: 'linear',
        kernelInitializer: 'glorotNormal',
    }));
    // Build the model so weights are initialized
    model.predict(tf.zeros([1, RL_NUM_FEATURES])).dispose();
    return model;
}

// ── Inference: score a single candidate ──
function scoreCandidateRL(features, model) {
    if (!model || !_rlAvailable) return 0;
    const input = tf.tensor2d([features], [1, RL_NUM_FEATURES]);
    const output = model.predict(input);
    const score = output.dataSync()[0];
    input.dispose();
    output.dispose();
    return isFinite(score) ? score : 0;
}

// ── Batch inference: score multiple candidates at once (more efficient) ──
function scoreCandidatesBatchRL(featureArray, model) {
    if (!model || !_rlAvailable || featureArray.length === 0) {
        return new Float32Array(featureArray.length);
    }
    const flat = new Float32Array(featureArray.length * RL_NUM_FEATURES);
    for (let i = 0; i < featureArray.length; i++) {
        flat.set(featureArray[i], i * RL_NUM_FEATURES);
    }
    const input = tf.tensor2d(flat, [featureArray.length, RL_NUM_FEATURES]);
    const output = model.predict(input);
    const scores = output.dataSync();
    input.dispose();
    output.dispose();
    return scores;
}

// ── Genome ↔ Model conversion ──
// Flatten all model weights into a single Float32Array (for GA evolution).
function modelToGenome(model) {
    if (!model) return new Float32Array(0);
    const weights = model.getWeights();
    let totalSize = 0;
    for (const w of weights) totalSize += w.size;
    const genome = new Float32Array(totalSize);
    let offset = 0;
    for (const w of weights) {
        const data = w.dataSync();
        genome.set(data, offset);
        offset += data.length;
    }
    return genome;
}

function genomeToModel(genome, model) {
    if (!model || !genome || genome.length === 0) return;
    const weights = model.getWeights();
    let offset = 0;
    const newWeights = [];
    for (const w of weights) {
        const size = w.size;
        const shape = w.shape;
        const slice = genome.slice(offset, offset + size);
        newWeights.push(tf.tensor(slice, shape));
        offset += size;
    }
    model.setWeights(newWeights);
    // Dispose old tensors
    for (const w of newWeights) w.dispose();
}

// ── Genome size (computed once at init) ──
function getGenomeSize() {
    // 22×32 + 32 + 32×32 + 32 + 32×1 + 1 = 704 + 32 + 1024 + 32 + 32 + 1 = 1825
    return RL_NUM_FEATURES * RL_HIDDEN_1 + RL_HIDDEN_1   // layer 1: weights + bias
         + RL_HIDDEN_1 * RL_HIDDEN_2 + RL_HIDDEN_2       // layer 2: weights + bias
         + RL_HIDDEN_2 * 1 + 1;                          // output: weights + bias
}

// ── GA operations on NN genomes ──
function rlMutateGenome(genome) {
    const m = new Float32Array(genome);
    const nMutations = 1 + Math.floor(Math.random() * Math.max(1, genome.length / 50));
    for (let i = 0; i < nMutations; i++) {
        const idx = Math.floor(Math.random() * m.length);
        m[idx] += (Math.random() - 0.5) * 0.4; // ±0.2 noise
    }
    return m;
}

function rlCrossoverGenome(a, b) {
    // Layer-boundary crossover: swap entire layers with 50% probability
    const child = new Float32Array(a.length);
    // Layer boundaries: [0, L1weights), [L1weights, L1bias), [L1bias, L2weights), etc.
    const l1w = RL_NUM_FEATURES * RL_HIDDEN_1;
    const l1b = l1w + RL_HIDDEN_1;
    const l2w = l1b + RL_HIDDEN_1 * RL_HIDDEN_2;
    const l2b = l2w + RL_HIDDEN_2;
    const ow = l2b + RL_HIDDEN_2;
    const bounds = [0, l1w, l1b, l2w, l2b, ow, a.length];
    for (let i = 0; i < bounds.length - 1; i++) {
        const src = Math.random() < 0.5 ? a : b;
        child.set(src.slice(bounds[i], bounds[i + 1]), bounds[i]);
    }
    return child;
}

function rlRandomGenome() {
    const size = getGenomeSize();
    const g = new Float32Array(size);
    // Xavier-like initialization: scale by sqrt(2/fan_in)
    const scale1 = Math.sqrt(2 / RL_NUM_FEATURES);
    const scale2 = Math.sqrt(2 / RL_HIDDEN_1);
    const scale3 = Math.sqrt(2 / RL_HIDDEN_2);
    let idx = 0;
    // Layer 1 weights
    for (let i = 0; i < RL_NUM_FEATURES * RL_HIDDEN_1; i++) g[idx++] = (Math.random() - 0.5) * 2 * scale1;
    // Layer 1 bias
    for (let i = 0; i < RL_HIDDEN_1; i++) g[idx++] = 0;
    // Layer 2 weights
    for (let i = 0; i < RL_HIDDEN_1 * RL_HIDDEN_2; i++) g[idx++] = (Math.random() - 0.5) * 2 * scale2;
    // Layer 2 bias
    for (let i = 0; i < RL_HIDDEN_2; i++) g[idx++] = 0;
    // Output weights
    for (let i = 0; i < RL_HIDDEN_2; i++) g[idx++] = (Math.random() - 0.5) * 2 * scale3;
    // Output bias
    g[idx++] = 0;
    return g;
}

// ── Expanded feature extraction for oct candidate scoring ──
// Takes an xon and a candidate move, returns RL_NUM_FEATURES-element Float32Array.
function extractRLFeatures(xon, candidate, occupied) {
    const f = new Float32Array(RL_NUM_FEATURES);
    const db = xon._dirBalance || new Array(10).fill(0);
    const ms = xon._modeStats || { oct: 0, tet: 0, idle_tet: 0, weak: 0 };
    const totalTicks = ms.oct + ms.tet + ms.idle_tet + ms.weak + 1;

    // 0: Direction balance score (deficit for this move's direction)
    let maxCount = 0;
    for (let i = 0; i < 10; i++) if (db[i] > maxCount) maxCount = db[i];
    const dirIdx = candidate.dirIdx != null ? candidate.dirIdx : -1;
    f[0] = dirIdx >= 0 && dirIdx <= 9 ? (maxCount - db[dirIdx]) / (maxCount + 1) : 0;

    // 1: Direction balance CV (coefficient of variation) — how balanced overall
    let sum = 0;
    for (let i = 0; i < 10; i++) sum += db[i];
    const mean = sum / 10;
    if (mean > 0) {
        let variance = 0;
        for (let i = 0; i < 10; i++) variance += (db[i] - mean) ** 2;
        f[1] = 1 - Math.sqrt(variance / 10) / mean; // 1 = perfectly balanced
    }

    // 2: SC already materialized (free move)
    f[2] = candidate._needsMaterialise ? 0 : 1;

    // 3: Destination is on oct cage
    f[3] = _octNodeSet && _octNodeSet.has(candidate.node) ? 1 : 0;

    // 4: Destination connectivity (normalized neighbor count)
    const nbs = baseNeighbors[candidate.node];
    f[4] = nbs ? Math.min(1, nbs.length / 12) : 0;

    // 5: Is previous node (would reverse direction)
    f[5] = candidate.node === xon.prevNode ? 0 : 1;

    // 6-8: Mode stats ratios (oct/tet/idle fractions)
    f[6] = ms.oct / totalTicks;
    f[7] = ms.tet / totalTicks;
    f[8] = ms.idle_tet / totalTicks;

    // 9-14: Per-direction deficit for the 6 SC directions (normalized)
    for (let d = 4; d < 10; d++) {
        f[5 + d] = maxCount > 0 ? (maxCount - db[d]) / maxCount : 0;
    }

    // 15-18: Per-direction deficit for the 4 base directions (normalized)
    for (let d = 0; d < 4; d++) {
        f[15 + d] = maxCount > 0 ? (maxCount - db[d]) / maxCount : 0;
    }

    // 19: Occupancy pressure — how many oct nodes are occupied
    let occCount = 0;
    if (occupied && _octNodeSet) {
        for (const n of _octNodeSet) {
            if (occupied.has(n)) occCount++;
        }
        f[19] = occCount / Math.max(1, _octNodeSet.size);
    }

    // 20: Weak mode fraction (how often this xon escapes confinement)
    f[20] = ms.weak / totalTicks;

    // 21: Random exploration noise (small)
    f[21] = Math.random() * 0.1;

    return f;
}

// ── Initialization ──
// Call after page load to set up the RL model (if TF.js is available).
function initRL() {
    if (!_rlCheckAvailable()) {
        console.log('[RL] TF.js not loaded — RL scoring disabled');
        return false;
    }
    _rlModel = createPolicyModel();
    if (_rlModel) {
        const genomeSize = getGenomeSize();
        console.log(`[RL] Model created: ${RL_NUM_FEATURES}→${RL_HIDDEN_1}→${RL_HIDDEN_2}→1 (${genomeSize} params)`);
        return true;
    }
    return false;
}
