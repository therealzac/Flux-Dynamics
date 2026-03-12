// flux-rl.js — PPO-based RL for deuteron choreography
// Strategic layer: face/quark assignment (actor-critic)
// Tactical layer: oct move scoring (actor-critic)
// Trained via Proximal Policy Optimization with per-tick rewards.

'use strict';

// ══════════════════════════════════════════════════════════════════════════════
// §1  ARCHITECTURE CONSTANTS
// ══════════════════════════════════════════════════════════════════════════════

const RL_STRATEGIC_FEATURES = 22;
const RL_STRATEGIC_HIDDEN   = 24;
const RL_STRATEGIC_MAX_ACTIONS = 48; // 8 faces × 6 quark types max
const RL_NUM_FEATURES       = 22;    // tactical
const RL_HIDDEN_1           = 32;
const RL_HIDDEN_2           = 32;
const RL_TACTICAL_MAX_ACTIONS = 12;  // max oct neighbors

// ── PPO Hyperparameters ──
const PPO_CLIP_EPSILON    = 0.2;
const PPO_VALUE_COEFF     = 0.5;
const PPO_ENTROPY_COEFF   = 0.01;
const PPO_GAMMA           = 0.99;
const PPO_LAMBDA          = 0.95;
const PPO_EPOCHS_PER_UPDATE = 4;
const PPO_MINIBATCH_SIZE  = 32;
const PPO_LEARNING_RATE   = 3e-4;
const PPO_ROLLOUT_LENGTH  = 128;
const PPO_MAX_GRAD_NORM   = 0.5;
const PPO_BALANCE_SCALE   = 10.0;   // reward scaling for CV improvement
const PPO_DEFORMATION_BONUS = 0.05; // reward for ticks with SC changes
const PPO_IDLE_TICK_PENALTY = 0.02; // penalty for ticks without SC changes
const PPO_TET_COMPLETION_BONUS = 0.1;

// ══════════════════════════════════════════════════════════════════════════════
// §2  TF.js AVAILABILITY
// ══════════════════════════════════════════════════════════════════════════════

let _rlAvailable = false;
let _rlModel = null;           // tactical actor-critic (for backward compat)
let _rlStrategicModel = null;  // strategic actor-critic
// _rlActiveModel declared in flux-demo-state.js (loaded before this file)

function _rlCheckAvailable() {
    if (typeof tf !== 'undefined' && tf.variable) {
        _rlAvailable = true;
        return true;
    }
    return false;
}

// ══════════════════════════════════════════════════════════════════════════════
// §3  ACTOR-CRITIC MODEL FACTORY
// ══════════════════════════════════════════════════════════════════════════════
//
// Each model is a plain object with tf.variable weights (not tf.sequential),
// enabling gradient computation via tf.variableGrads().
//
// Structure: { actor: {w1,b1,w2,b2,w3,b3}, critic: {w1,b1,w2,b2,w3,b3},
//              forward(featuresTensor, mask) → {logits, value},
//              forwardInference(features) → scores[],
//              dispose() }

function _xavierInit(rows, cols) {
    const scale = Math.sqrt(2.0 / (rows + cols));
    const data = new Float32Array(rows * cols);
    for (let i = 0; i < data.length; i++) {
        // Box-Muller
        const u1 = Math.random() || 1e-10;
        const u2 = Math.random();
        data[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * scale;
    }
    return data;
}

function createActorCritic(inputDim, hiddenDim, maxActions, name) {
    if (!_rlCheckAvailable()) return null;

    const make = (prefix, outDim) => {
        const w1 = tf.variable(tf.tensor2d(_xavierInit(inputDim, hiddenDim), [inputDim, hiddenDim]), true, `${prefix}_w1`);
        const b1 = tf.variable(tf.zeros([hiddenDim]), true, `${prefix}_b1`);
        const w2 = tf.variable(tf.tensor2d(_xavierInit(hiddenDim, hiddenDim), [hiddenDim, hiddenDim]), true, `${prefix}_w2`);
        const b2 = tf.variable(tf.zeros([hiddenDim]), true, `${prefix}_b2`);
        const w3 = tf.variable(tf.tensor2d(_xavierInit(hiddenDim, outDim), [hiddenDim, outDim]), true, `${prefix}_w3`);
        const b3 = tf.variable(tf.zeros([outDim]), true, `${prefix}_b3`);
        return { w1, b1, w2, b2, w3, b3 };
    };

    const actor  = make(`${name}_actor`, maxActions);
    const critic = make(`${name}_critic`, 1);

    function _mlpForward(net, x) {
        // x: [batch, inputDim]
        const h1 = tf.relu(x.matMul(net.w1).add(net.b1));
        const h2 = tf.relu(h1.matMul(net.w2).add(net.b2));
        return h2.matMul(net.w3).add(net.b3);
    }

    const model = {
        actor, critic, name,
        inputDim, hiddenDim, maxActions,

        // Forward pass: returns { logits: [batch, maxActions], value: [batch, 1] }
        forward(featuresTensor, mask) {
            const logits = _mlpForward(actor, featuresTensor);
            const value  = _mlpForward(critic, featuresTensor);
            if (mask) {
                // mask: [batch, maxActions], 1=valid, 0=invalid
                // Set invalid logits to -1e9 so softmax gives ~0 probability
                const maskedLogits = logits.add(mask.sub(1).mul(1e9));
                return { logits: maskedLogits, value };
            }
            return { logits, value };
        },

        // Inference: single observation, returns softmax scores as Float32Array
        forwardInference(features, nValid) {
            return tf.tidy(() => {
                const input = tf.tensor2d([features], [1, inputDim]);
                const logits = _mlpForward(actor, input);
                // Only take first nValid logits
                const sliced = nValid < maxActions
                    ? logits.slice([0, 0], [1, nValid])
                    : logits;
                const probs = tf.softmax(sliced, 1);
                return probs.dataSync();
            });
        },

        // Sample an action from the policy (for training exploration)
        sampleAction(features, nValid) {
            return tf.tidy(() => {
                const input = tf.tensor2d([features], [1, inputDim]);
                const rawLogits = _mlpForward(actor, input);
                const val = _mlpForward(critic, input);
                // Mask invalid actions
                const mask = new Float32Array(maxActions).fill(0);
                for (let i = 0; i < nValid; i++) mask[i] = 1;
                const maskT = tf.tensor2d([mask], [1, maxActions]);
                const maskedLogits = rawLogits.add(maskT.sub(1).mul(1e9));
                // Sample from categorical
                const logProbs = tf.logSoftmax(maskedLogits, 1);
                const probs = tf.softmax(maskedLogits, 1);
                const probsArr = probs.dataSync();
                // Multinomial sample
                let actionIdx = 0;
                const r = Math.random();
                let cumProb = 0;
                for (let i = 0; i < nValid; i++) {
                    cumProb += probsArr[i];
                    if (r < cumProb) { actionIdx = i; break; }
                }
                if (actionIdx >= nValid) actionIdx = nValid - 1;
                const logProb = logProbs.dataSync()[actionIdx];
                const value = val.dataSync()[0];
                return { actionIdx, logProb, value };
            });
        },

        // Get all trainable variables (for optimizer)
        getVars() {
            return [
                actor.w1, actor.b1, actor.w2, actor.b2, actor.w3, actor.b3,
                critic.w1, critic.b1, critic.w2, critic.b2, critic.w3, critic.b3,
            ];
        },

        // Serialize actor weights to Float32Array (critic not needed for inference)
        serializeActor() {
            const parts = [actor.w1, actor.b1, actor.w2, actor.b2, actor.w3, actor.b3];
            let size = 0;
            for (const v of parts) size += v.size;
            const arr = new Float32Array(size);
            let offset = 0;
            for (const v of parts) {
                arr.set(v.dataSync(), offset);
                offset += v.size;
            }
            return arr;
        },

        // Load actor weights from Float32Array
        loadActor(arr) {
            const parts = [actor.w1, actor.b1, actor.w2, actor.b2, actor.w3, actor.b3];
            let offset = 0;
            for (const v of parts) {
                const data = arr.slice(offset, offset + v.size);
                v.assign(tf.tensor(data, v.shape));
                offset += v.size;
            }
        },

        // Actor weight count
        actorParamCount() {
            return [actor.w1, actor.b1, actor.w2, actor.b2, actor.w3, actor.b3]
                .reduce((s, v) => s + v.size, 0);
        },

        dispose() {
            for (const net of [actor, critic]) {
                for (const key of ['w1', 'b1', 'w2', 'b2', 'w3', 'b3']) {
                    if (net[key]) net[key].dispose();
                }
            }
        }
    };

    return model;
}

// ══════════════════════════════════════════════════════════════════════════════
// §4  BACKWARD-COMPATIBLE INFERENCE WRAPPERS
// ══════════════════════════════════════════════════════════════════════════════
// These match the old API so flux-demo-planner.js doesn't need changes
// for inference mode (only training mode adds trajectory collection).

function scoreCandidateRL(features, model) {
    if (!model || !_rlAvailable) return 0;
    try {
        const scores = model.forwardInference(features, 1);
        return isFinite(scores[0]) ? scores[0] : 0;
    } catch (e) { return 0; }
}

function scoreStrategicRL(features, model) {
    if (!model || !_rlAvailable) return 0;
    try {
        const scores = model.forwardInference(features, 1);
        return isFinite(scores[0]) ? scores[0] : 0;
    } catch (e) { return 0; }
}

function scoreCandidatesBatchRL(featureArray, model) {
    if (!model || !_rlAvailable || featureArray.length === 0) {
        return new Float32Array(featureArray.length);
    }
    try {
        return tf.tidy(() => {
            const n = featureArray.length;
            const flat = new Float32Array(n * RL_NUM_FEATURES);
            for (let i = 0; i < n; i++) flat.set(featureArray[i], i * RL_NUM_FEATURES);
            const input = tf.tensor2d(flat, [n, RL_NUM_FEATURES]);
            const logits = model.forward(input).logits;
            // Take first column (score for each candidate)
            return logits.slice([0, 0], [n, 1]).squeeze().dataSync();
        });
    } catch (e) { return new Float32Array(featureArray.length); }
}

// ══════════════════════════════════════════════════════════════════════════════
// §5  TRAJECTORY BUFFER
// ══════════════════════════════════════════════════════════════════════════════

class PPOTrajectoryBuffer {
    constructor() { this.clear(); }

    push(state, actionIdx, logProb, value, nValid) {
        this.states.push(new Float32Array(state));
        this.actions.push(actionIdx);
        this.logProbs.push(logProb);
        this.values.push(value);
        this.nValids.push(nValid);
        this.rewards.push(0); // filled in later by assignReward
    }

    assignReward(reward) {
        // Assign reward to all entries added since last reward assignment
        for (let i = this._lastRewardIdx; i < this.rewards.length; i++) {
            this.rewards[i] = reward;
        }
        this._lastRewardIdx = this.rewards.length;
    }

    computeGAE(lastValue) {
        const T = this.rewards.length;
        if (T === 0) return { advantages: [], returns: [] };
        const advantages = new Float32Array(T);
        const returns = new Float32Array(T);
        let gae = 0;
        for (let t = T - 1; t >= 0; t--) {
            const nextValue = t === T - 1 ? lastValue : this.values[t + 1];
            const delta = this.rewards[t] + PPO_GAMMA * nextValue - this.values[t];
            gae = delta + PPO_GAMMA * PPO_LAMBDA * gae;
            advantages[t] = gae;
            returns[t] = gae + this.values[t];
        }
        // Normalize advantages
        let mean = 0, sq = 0;
        for (let i = 0; i < T; i++) mean += advantages[i];
        mean /= T;
        for (let i = 0; i < T; i++) sq += (advantages[i] - mean) ** 2;
        const std = Math.sqrt(sq / T) + 1e-8;
        for (let i = 0; i < T; i++) advantages[i] = (advantages[i] - mean) / std;
        return { advantages, returns };
    }

    get length() { return this.states.length; }

    clear() {
        this.states = [];
        this.actions = [];
        this.logProbs = [];
        this.values = [];
        this.nValids = [];
        this.rewards = [];
        this._lastRewardIdx = 0;
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// §6  PER-TICK REWARD
// ══════════════════════════════════════════════════════════════════════════════

let _ppoPrevAvgCV = null;
let _ppoTetCompletionsThisTick = 0;
let _ppoGuardFailedThisTick = false;
let _ppoDeformationThisTick = false;

function _ppoComputeAvgCV() {
    if (typeof _demoVisits === 'undefined' || !_demoVisits) return 1;
    const types = ['pu1', 'pu2', 'pd', 'nd1', 'nd2', 'nu'];
    let totalCV = 0, nFaces = 0;
    for (let f = 1; f <= 8; f++) {
        const v = _demoVisits[f];
        if (!v) continue;
        const counts = types.map(t => v[t] || 0);
        const sum = counts.reduce((a, b) => a + b, 0);
        if (sum === 0) continue;
        const mean = sum / 6;
        let variance = 0;
        for (let i = 0; i < 6; i++) variance += (counts[i] - mean) ** 2;
        totalCV += Math.sqrt(variance / 6) / mean;
        nFaces++;
    }
    return nFaces > 0 ? totalCV / nFaces : 1;
}

function computeTickReward() {
    const currentCV = _ppoComputeAvgCV();
    let reward = 0;

    // Balance improvement (positive = CV decreased = good)
    if (_ppoPrevAvgCV !== null) {
        reward += (_ppoPrevAvgCV - currentCV) * PPO_BALANCE_SCALE;
    }

    // Tet completion bonus
    reward += PPO_TET_COMPLETION_BONUS * _ppoTetCompletionsThisTick;

    // Guard failure penalty
    if (_ppoGuardFailedThisTick) reward -= 1.0;

    // Deformation reward: encourage 1:1 tick-to-Planck ratio
    if (_ppoDeformationThisTick) {
        reward += PPO_DEFORMATION_BONUS;
    } else {
        reward -= PPO_IDLE_TICK_PENALTY;
    }

    // Idle oct xon penalty
    if (typeof _demoXons !== 'undefined') {
        let idleCount = 0;
        for (const x of _demoXons) {
            if (x.alive && x._mode === 'oct') idleCount++;
        }
        reward -= 0.01 * idleCount;
    }

    _ppoPrevAvgCV = currentCV;
    _ppoTetCompletionsThisTick = 0;
    _ppoGuardFailedThisTick = false;
    _ppoDeformationThisTick = false;

    return reward;
}

function resetTickRewardState() {
    _ppoPrevAvgCV = null;
    _ppoTetCompletionsThisTick = 0;
    _ppoGuardFailedThisTick = false;
    _ppoDeformationThisTick = false;
}

// ══════════════════════════════════════════════════════════════════════════════
// §7  PPO UPDATE
// ══════════════════════════════════════════════════════════════════════════════

function ppoUpdate(actorCritic, optimizer, buffer, lastValue) {
    const { advantages, returns } = buffer.computeGAE(lastValue);
    const T = buffer.length;
    if (T < 2) return { policyLoss: 0, valueLoss: 0, entropy: 0 };

    const inputDim = actorCritic.inputDim;
    const maxActions = actorCritic.maxActions;

    // Pre-build flat arrays
    const statesFlat  = new Float32Array(T * inputDim);
    const actionsArr  = new Int32Array(T);
    const oldLogProbs = new Float32Array(T);
    const nValidsArr  = new Int32Array(T);
    for (let i = 0; i < T; i++) {
        statesFlat.set(buffer.states[i], i * inputDim);
        actionsArr[i]  = buffer.actions[i];
        oldLogProbs[i] = buffer.logProbs[i];
        nValidsArr[i]  = buffer.nValids[i];
    }

    // Build mask: [T, maxActions]
    const maskFlat = new Float32Array(T * maxActions);
    for (let i = 0; i < T; i++) {
        for (let a = 0; a < nValidsArr[i]; a++) {
            maskFlat[i * maxActions + a] = 1;
        }
    }

    let totalPolicyLoss = 0, totalValueLoss = 0, totalEntropy = 0, totalGradNorm = 0, nUpdates = 0;

    for (let epoch = 0; epoch < PPO_EPOCHS_PER_UPDATE; epoch++) {
        // Shuffle indices
        const indices = Array.from({ length: T }, (_, i) => i);
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }

        // Iterate minibatches
        for (let bStart = 0; bStart < T; bStart += PPO_MINIBATCH_SIZE) {
            const bEnd = Math.min(bStart + PPO_MINIBATCH_SIZE, T);
            const bSize = bEnd - bStart;
            const bIdx = indices.slice(bStart, bEnd);

            const result = tf.tidy(() => {
                // Build batch tensors
                const bStates = new Float32Array(bSize * inputDim);
                const bActions = new Int32Array(bSize);
                const bOldLogProbs = new Float32Array(bSize);
                const bAdvantages = new Float32Array(bSize);
                const bReturns = new Float32Array(bSize);
                const bMask = new Float32Array(bSize * maxActions);

                for (let i = 0; i < bSize; i++) {
                    const idx = bIdx[i];
                    bStates.set(statesFlat.subarray(idx * inputDim, (idx + 1) * inputDim), i * inputDim);
                    bActions[i] = actionsArr[idx];
                    bOldLogProbs[i] = oldLogProbs[idx];
                    bAdvantages[i] = advantages[idx];
                    bReturns[i] = returns[idx];
                    bMask.set(maskFlat.subarray(idx * maxActions, (idx + 1) * maxActions), i * maxActions);
                }

                const statesTensor = tf.tensor2d(bStates, [bSize, inputDim]);
                const maskTensor = tf.tensor2d(bMask, [bSize, maxActions]);
                const advTensor = tf.tensor1d(bAdvantages);
                const retTensor = tf.tensor1d(bReturns);

                // Compute loss inside variableGrads, capturing component values via closure
                let _batchPLoss = 0, _batchVLoss = 0, _batchEntropy = 0;
                const { value: loss, grads } = tf.variableGrads(() => {
                    const { logits, value } = actorCritic.forward(statesTensor, maskTensor);
                    const values = value.squeeze();

                    // Log probabilities for taken actions
                    const allLogProbs = tf.logSoftmax(logits, 1);
                    // Gather log probs for taken actions
                    const actionOneHot = tf.oneHot(tf.tensor1d(bActions, 'int32'), maxActions);
                    const takenLogProbs = allLogProbs.mul(actionOneHot).sum(1);

                    const oldLP = tf.tensor1d(bOldLogProbs);
                    const ratio = tf.exp(takenLogProbs.sub(oldLP));

                    // Clipped surrogate
                    const surr1 = ratio.mul(advTensor);
                    const surr2 = tf.clipByValue(ratio, 1 - PPO_CLIP_EPSILON, 1 + PPO_CLIP_EPSILON).mul(advTensor);
                    const policyLoss = tf.minimum(surr1, surr2).mean().neg();

                    // Value loss
                    const valueLoss = tf.losses.meanSquaredError(retTensor, values).mul(PPO_VALUE_COEFF);

                    // Entropy bonus (safe: clamp probs to avoid 0*-Inf=NaN)
                    const probs = tf.softmax(logits, 1).clipByValue(1e-8, 1);
                    const safeLogProbs = probs.log();
                    const entropy = probs.mul(safeLogProbs).sum(1).mean().neg();
                    const entropyLoss = entropy.mul(-PPO_ENTROPY_COEFF);

                    // Capture component values for dashboard
                    _batchPLoss = policyLoss.dataSync()[0];
                    _batchVLoss = valueLoss.dataSync()[0];
                    _batchEntropy = entropy.dataSync()[0];

                    return policyLoss.add(valueLoss).add(entropyLoss);
                });

                // Gradient clipping
                const gradValues = Object.values(grads);
                let totalNorm = 0;
                for (const g of gradValues) {
                    const gData = g.dataSync();
                    for (let i = 0; i < gData.length; i++) totalNorm += gData[i] * gData[i];
                }
                totalNorm = Math.sqrt(totalNorm);

                if (totalNorm > PPO_MAX_GRAD_NORM) {
                    const scale = PPO_MAX_GRAD_NORM / totalNorm;
                    const clipped = {};
                    for (const [key, g] of Object.entries(grads)) {
                        clipped[key] = g.mul(scale);
                    }
                    optimizer.applyGradients(clipped);
                    for (const g of Object.values(clipped)) g.dispose();
                } else {
                    optimizer.applyGradients(grads);
                }

                // Dispose original grads
                for (const g of Object.values(grads)) {
                    if (!g.isDisposed) g.dispose();
                }

                return { loss: loss.dataSync()[0], pLoss: _batchPLoss, vLoss: _batchVLoss, entropy: _batchEntropy, gradNorm: totalNorm };
            });

            totalPolicyLoss += result.pLoss;
            totalValueLoss += result.vLoss;
            totalEntropy += result.entropy;
            totalGradNorm += result.gradNorm;
            nUpdates++;
        }
    }

    const n = nUpdates || 1;
    return {
        policyLoss: totalPolicyLoss / n,
        valueLoss:  totalValueLoss / n,
        entropy:    totalEntropy / n,
        avgReward:  buffer.rewards && buffer.rewards.length > 0 ? buffer.rewards.reduce((a, b) => a + b, 0) / buffer.rewards.length : 0,
        gradNorm:   totalGradNorm / n,
    };
}

// ══════════════════════════════════════════════════════════════════════════════
// §8  STRATEGIC FEATURE EXTRACTION (22 features)
// ══════════════════════════════════════════════════════════════════════════════

function extractStrategicFeatures(xon, face, quarkType, occupied) {
    const f = new Float32Array(RL_STRATEGIC_FEATURES);
    const v = _demoVisits ? (_demoVisits[face] || {}) : {};
    const types = ['pu1', 'pu2', 'pd', 'nd1', 'nd2', 'nu'];
    const counts = types.map(t => v[t] || 0);
    const sum = counts.reduce((a, b) => a + b, 0);
    const mean = sum / 6 || 1;

    // f[0-5]: Per-type ratio deficit for this face
    for (let i = 0; i < 6; i++) f[i] = Math.max(0, (mean - counts[i]) / mean);

    // f[6]: Is proton face
    f[6] = typeof A_SET !== 'undefined' && A_SET.has(face) ? 1 : 0;

    // f[7]: Face vacancy
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

    // f[9]: Target type deficit specifically
    f[9] = typeIdx >= 0 ? Math.max(0, (mean - counts[typeIdx]) / mean) : 0;

    // f[10]: SC activation fraction
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

    // f[11]: Xon on or near face (reachability)
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

    // f[12-15]: Xon's top 4 direction deficits
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

    // f[16]: Face recency
    if (ts && ts.faceLastVisitTick[face] != null) {
        f[16] = Math.min(1, (tick - ts.faceLastVisitTick[face]) / 256);
    } else {
        f[16] = 1;
    }

    // f[17]: Ratio velocity
    if (ts && ts.prevFaceCV[face] != null && sum > 0) {
        let variance = 0;
        for (let i = 0; i < 6; i++) variance += (counts[i] - mean) ** 2;
        const currentCV = Math.sqrt(variance / 6) / mean;
        f[17] = Math.max(0, Math.min(1, (currentCV - ts.prevFaceCV[face]) + 0.5));
    } else {
        f[17] = 0.5;
    }

    // f[18]-f[19]: Cycle phase
    const phase = (tick % 64) / 64 * 2 * Math.PI;
    f[18] = (Math.sin(phase) + 1) / 2;
    f[19] = (Math.cos(phase) + 1) / 2;

    // f[20]: Xon idle duration
    const ms = xon._modeStats || { oct: 0, tet: 0, idle_tet: 0, weak: 0 };
    const octSince = xon._octModeSince || 0;
    f[20] = xon._mode === 'oct' && octSince > 0 ? Math.min(1, (tick - octSince) / 64) : 0;

    // f[21]: Global pressure
    f[21] = ts ? ts.globalPressure : 0;

    return f;
}

// ══════════════════════════════════════════════════════════════════════════════
// §9  TACTICAL FEATURE EXTRACTION (22 features)
// ══════════════════════════════════════════════════════════════════════════════

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

// ══════════════════════════════════════════════════════════════════════════════
// §10  INDEXEDDB PERSISTENCE
// ══════════════════════════════════════════════════════════════════════════════

const _RL_IDB_NAME = 'flux_rl_genome';
const _RL_IDB_VERSION = 2;  // bumped for PPO format
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

async function rlSaveWeights(strategicAC, tacticalAC, fitness) {
    if (!_rlIDB) await _rlIDBOpen();
    if (!_rlIDB) return;
    try {
        const data = {
            type: 'ppo',
            strategicActor: strategicAC ? Array.from(strategicAC.serializeActor()) : null,
            tacticalActor: tacticalAC ? Array.from(tacticalAC.serializeActor()) : null,
            fitness,
            timestamp: Date.now(),
        };
        const tx = _rlIDB.transaction(_RL_IDB_STORE, 'readwrite');
        tx.objectStore(_RL_IDB_STORE).put(data, 'best');
        console.log(`[RL] Saved PPO weights (fitness=${fitness?.toFixed(3)})`);
    } catch (e) { console.warn('[RL] Failed to save weights:', e); }
}

// Backward compat alias
async function rlSaveGenome(genome, fitness) {
    // Legacy GA save — just store raw genome
    if (!_rlIDB) await _rlIDBOpen();
    if (!_rlIDB) return;
    try {
        const tx = _rlIDB.transaction(_RL_IDB_STORE, 'readwrite');
        tx.objectStore(_RL_IDB_STORE).put({
            type: 'ga', genome: Array.from(genome), fitness, timestamp: Date.now(), size: genome.length,
        }, 'best');
    } catch (e) { console.warn('[RL] Failed to save genome:', e); }
}

async function rlLoadWeights() {
    if (!_rlIDB) await _rlIDBOpen();
    if (!_rlIDB) return null;
    return new Promise((resolve) => {
        try {
            const tx = _rlIDB.transaction(_RL_IDB_STORE, 'readonly');
            const req = tx.objectStore(_RL_IDB_STORE).get('best');
            req.onsuccess = () => {
                const data = req.result;
                if (!data) { resolve(null); return; }
                if (data.type === 'ppo' && data.strategicActor) {
                    console.log(`[RL] Loaded PPO weights from IDB (fitness=${data.fitness?.toFixed(3)})`);
                    resolve({
                        type: 'ppo',
                        strategicActor: new Float32Array(data.strategicActor),
                        tacticalActor: data.tacticalActor ? new Float32Array(data.tacticalActor) : null,
                        fitness: data.fitness || 0,
                    });
                } else if (data.genome) {
                    // Legacy GA format
                    console.log(`[RL] Loaded legacy GA genome from IDB (fitness=${data.fitness?.toFixed(3)})`);
                    resolve({ type: 'ga', genome: new Float32Array(data.genome), fitness: data.fitness || 0 });
                } else {
                    resolve(null);
                }
            };
            req.onerror = () => resolve(null);
        } catch (e) { resolve(null); }
    });
}

// Legacy alias
async function rlLoadGenome() {
    const result = await rlLoadWeights();
    if (!result) return null;
    if (result.type === 'ga') return { genome: result.genome, fitness: result.fitness };
    return null; // PPO format — legacy callers can't use this
}

// ══════════════════════════════════════════════════════════════════════════════
// §11  LEGACY GA OPERATIONS (kept for backward compat, may be removed later)
// ══════════════════════════════════════════════════════════════════════════════

function getGenomeSize() {
    // Legacy: combined strategic + tactical flat genome
    const RL_STRATEGIC_SIZE = RL_STRATEGIC_FEATURES * RL_STRATEGIC_HIDDEN + RL_STRATEGIC_HIDDEN
        + RL_STRATEGIC_HIDDEN * RL_STRATEGIC_HIDDEN + RL_STRATEGIC_HIDDEN
        + RL_STRATEGIC_HIDDEN * 1 + 1;
    const RL_TACTICAL_SIZE = RL_NUM_FEATURES * RL_HIDDEN_1 + RL_HIDDEN_1
        + RL_HIDDEN_1 * RL_HIDDEN_2 + RL_HIDDEN_2
        + RL_HIDDEN_2 * 1 + 1;
    return RL_STRATEGIC_SIZE + RL_TACTICAL_SIZE;
}

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
    const mid = Math.floor(a.length / 2);
    const blocks = [[0, mid], [mid, a.length]];
    for (const [start, end] of blocks) {
        const src = Math.random() < 0.5 ? a : b;
        child.set(src.slice(start, end), start);
    }
    return child;
}

function rlRandomGenome() {
    const size = getGenomeSize();
    const g = new Float32Array(size);
    for (let i = 0; i < size; i++) {
        const u1 = Math.random() || 1e-10;
        const u2 = Math.random();
        g[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * 0.1;
    }
    return g;
}

// ══════════════════════════════════════════════════════════════════════════════
// §12  INITIALIZATION
// ══════════════════════════════════════════════════════════════════════════════

async function initRL() {
    if (!_rlCheckAvailable()) {
        console.log('[RL] TF.js not loaded — RL scoring disabled');
        return false;
    }

    // Create actor-critic models
    _rlStrategicModel = createActorCritic(
        RL_STRATEGIC_FEATURES, RL_STRATEGIC_HIDDEN, RL_STRATEGIC_MAX_ACTIONS, 'strategic'
    );
    _rlModel = createActorCritic(
        RL_NUM_FEATURES, RL_HIDDEN_1, RL_TACTICAL_MAX_ACTIONS, 'tactical'
    );

    if (_rlStrategicModel && _rlModel) {
        console.log(`[RL] Actor-critic models created: strategic (${_rlStrategicModel.actorParamCount()}p) + tactical (${_rlModel.actorParamCount()}p)`);

        // Try loading saved weights
        const saved = await rlLoadWeights();
        if (saved && saved.type === 'ppo') {
            if (saved.strategicActor) _rlStrategicModel.loadActor(saved.strategicActor);
            if (saved.tacticalActor) _rlModel.loadActor(saved.tacticalActor);
            _rlActiveModel = _rlModel;
            console.log(`[RL] PPO weights loaded (fitness=${saved.fitness?.toFixed(3)})`);
        }
        return true;
    }
    return false;
}

// ══════════════════════════════════════════════════════════════════════════════
// §13 TENSOR DASHBOARD — live visualization of PPO internals
// ══════════════════════════════════════════════════════════════════════════════

// Training metrics history (ring buffer)
const _ppoMetricsHistory = [];
const _PPO_METRICS_MAX = 100;

function _ppoRecordMetrics(metrics) {
    const safe = v => (typeof v === 'number' && isFinite(v)) ? v : 0;
    _ppoMetricsHistory.push({
        policyLoss: safe(metrics.policyLoss),
        valueLoss:  safe(metrics.valueLoss),
        entropy:    safe(metrics.entropy),
        avgReward:  safe(metrics.avgReward),
        gradNorm:   safe(metrics.gradNorm)
    });
    if (_ppoMetricsHistory.length > _PPO_METRICS_MAX) _ppoMetricsHistory.shift();
}

// ── Policy Heatmap ──
// Shows action probabilities for strategic (8 faces × 6 types) and tactical
function _drawPolicyHeatmap() {
    const canvas = document.getElementById('rl-policy-canvas');
    if (!canvas || canvas.style.display === 'none') return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    // Background
    ctx.fillStyle = '#0a1018';
    ctx.fillRect(0, 0, W, H);

    // Title
    ctx.fillStyle = '#6ab4ff';
    ctx.font = '9px monospace';
    ctx.fillText('POLICY HEATMAP', 4, 10);

    // Strategic policy: get logits for a dummy state, show as 8×6 grid
    if (_ppoStrategicAC || _rlStrategicModel) {
        const model = _ppoStrategicAC || _rlStrategicModel;
        try {
            const probs = tf.tidy(() => {
                const dummyState = tf.zeros([1, RL_STRATEGIC_FEATURES]);
                const mask = tf.ones([1, RL_STRATEGIC_MAX_ACTIONS]);
                const result = model.forward(dummyState, mask);
                return result.logits.softmax(-1).dataSync();
            });

            // Draw 8 columns (faces) × 6 rows (quark types)
            const FACE_LABELS = ['F1','F2','F3','F4','F5','F6','F7','F8'];
            const QUARK_LABELS = ['pu1','pu2','pd','nd1','nd2','nu'];
            const cellW = 24, cellH = 12;
            const startX = 4, startY = 18;

            // Row labels
            ctx.fillStyle = '#556677';
            ctx.font = '7px monospace';
            for (let q = 0; q < 6; q++) {
                ctx.fillText(QUARK_LABELS[q], startX, startY + q * cellH + 9);
            }

            // Column labels
            for (let f = 0; f < 8; f++) {
                ctx.fillText(FACE_LABELS[f], startX + 26 + f * cellW + 4, startY - 2);
            }

            // Heatmap cells
            const maxProb = Math.max(...probs) || 1;
            for (let f = 0; f < 8; f++) {
                for (let q = 0; q < 6; q++) {
                    const idx = f * 6 + q;
                    const p = idx < probs.length ? probs[idx] : 0;
                    const intensity = Math.min(1, p / maxProb);
                    // Blue → Cyan → White gradient
                    const r = Math.floor(intensity * 180);
                    const g = Math.floor(40 + intensity * 200);
                    const b = Math.floor(80 + intensity * 175);
                    ctx.fillStyle = `rgb(${r},${g},${b})`;
                    ctx.fillRect(startX + 26 + f * cellW, startY + q * cellH, cellW - 1, cellH - 1);

                    // Show probability text for high values
                    if (p > 0.05) {
                        ctx.fillStyle = intensity > 0.6 ? '#000' : '#aaa';
                        ctx.font = '6px monospace';
                        ctx.fillText((p * 100).toFixed(0), startX + 28 + f * cellW, startY + q * cellH + 8);
                    }
                }
            }

            // Strategic label
            ctx.fillStyle = '#4488bb';
            ctx.font = '7px monospace';
            ctx.fillText('strategic', startX + 26 + 8 * cellW + 4, startY + 18);
        } catch(e) { /* model not ready */ }
    }

    // Tactical policy: show as horizontal bar chart for current xon context
    if (_ppoTacticalAC || _rlModel) {
        const model = _ppoTacticalAC || _rlModel;
        try {
            const probs = tf.tidy(() => {
                const dummyState = tf.zeros([1, RL_NUM_FEATURES]);
                const mask = tf.ones([1, RL_TACTICAL_MAX_ACTIONS]);
                const result = model.forward(dummyState, mask);
                return result.logits.softmax(-1).dataSync();
            });

            const startX = 4, startY = 95;
            ctx.fillStyle = '#44bb88';
            ctx.font = '7px monospace';
            ctx.fillText('tactical', startX, startY);

            // Bar chart
            const barW = 16, barH = 14;
            const maxP = Math.max(...probs) || 1;
            for (let i = 0; i < Math.min(12, probs.length); i++) {
                const p = probs[i];
                const intensity = Math.min(1, p / maxP);
                const r = Math.floor(intensity * 100);
                const g = Math.floor(60 + intensity * 195);
                const b = Math.floor(40 + intensity * 140);
                ctx.fillStyle = `rgb(${r},${g},${b})`;
                ctx.fillRect(startX + 42 + i * barW, startY - 10, barW - 1, barH);
                ctx.fillStyle = intensity > 0.5 ? '#000' : '#888';
                ctx.font = '6px monospace';
                ctx.fillText(i.toString(), startX + 44 + i * barW, startY + 1);
            }
        } catch(e) { /* model not ready */ }
    }
}

// ── Weight Distribution Histograms ──
function _drawWeightDistributions() {
    const canvas = document.getElementById('rl-weights-canvas');
    if (!canvas || canvas.style.display === 'none') return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0a1018';
    ctx.fillRect(0, 0, W, H);

    ctx.fillStyle = '#66ffaa';
    ctx.font = '9px monospace';
    ctx.fillText('WEIGHT DISTRIBUTIONS', 4, 10);

    const models = [];
    if (_ppoStrategicAC || _rlStrategicModel) models.push({ name: 'strat', model: _ppoStrategicAC || _rlStrategicModel, color: '#4488cc' });
    if (_ppoTacticalAC || _rlModel) models.push({ name: 'tact', model: _ppoTacticalAC || _rlModel, color: '#44cc88' });

    let offsetY = 16;
    const BINS = 30;
    const histW = 120, histH = 24;

    for (const { name, model, color } of models) {
        try {
            const vars = model.getVars();
            const allWeights = tf.tidy(() => {
                const tensors = Object.values(vars).filter(v => v instanceof tf.Tensor);
                if (tensors.length === 0) return new Float32Array(0);
                const flat = tensors.map(t => t.flatten());
                return tf.concat(flat).dataSync();
            });

            if (allWeights.length === 0) continue;

            // Compute histogram
            let minW = Infinity, maxW = -Infinity;
            for (let i = 0; i < allWeights.length; i++) {
                if (allWeights[i] < minW) minW = allWeights[i];
                if (allWeights[i] > maxW) maxW = allWeights[i];
            }
            const range = maxW - minW || 1;
            const bins = new Float32Array(BINS);
            for (let i = 0; i < allWeights.length; i++) {
                const idx = Math.min(BINS - 1, Math.floor((allWeights[i] - minW) / range * BINS));
                bins[idx]++;
            }
            const maxBin = Math.max(...bins) || 1;

            // Draw label
            ctx.fillStyle = color;
            ctx.font = '7px monospace';
            ctx.fillText(`${name} (${allWeights.length}w)`, 4, offsetY + 8);
            ctx.fillStyle = '#556677';
            ctx.fillText(`[${minW.toFixed(2)}, ${maxW.toFixed(2)}]`, 4, offsetY + 16);

            // Draw histogram bars
            const barW = histW / BINS;
            for (let i = 0; i < BINS; i++) {
                const h = (bins[i] / maxBin) * histH;
                const intensity = bins[i] / maxBin;
                ctx.fillStyle = color + Math.floor(40 + intensity * 200).toString(16).padStart(2, '0');
                ctx.fillRect(130 + i * barW, offsetY + histH - h, barW - 0.5, h);
            }

            // Axis line
            ctx.strokeStyle = '#334455';
            ctx.beginPath();
            ctx.moveTo(130, offsetY + histH);
            ctx.lineTo(130 + histW, offsetY + histH);
            ctx.stroke();

            offsetY += histH + 8;
        } catch(e) { /* model not ready */ }
    }
}

// ── Training Metrics Dashboard ──
function _drawTrainingMetrics() {
    const canvas = document.getElementById('rl-metrics-canvas');
    if (!canvas || canvas.style.display === 'none') return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0a1018';
    ctx.fillRect(0, 0, W, H);

    ctx.fillStyle = '#ffaa66';
    ctx.font = '9px monospace';
    ctx.fillText('TRAINING METRICS', 4, 10);

    const data = _ppoMetricsHistory;
    if (data.length < 2) {
        ctx.fillStyle = '#556677';
        ctx.font = '8px monospace';
        ctx.fillText('Waiting for PPO updates...', 4, 30);
        return;
    }

    // Draw 4 sparklines: policyLoss, valueLoss, entropy, avgReward
    const metrics = [
        { key: 'policyLoss', label: 'policy loss', color: '#ff6666' },
        { key: 'valueLoss',  label: 'value loss',  color: '#66bbff' },
        { key: 'entropy',    label: 'entropy',     color: '#ffcc44' },
        { key: 'avgReward',  label: 'avg reward',  color: '#66ff66' }
    ];

    const sparkW = 150, sparkH = 16;
    const startX = 100, startY = 16;

    for (let m = 0; m < metrics.length; m++) {
        const { key, label, color } = metrics[m];
        const y = startY + m * (sparkH + 6);
        const values = data.map(d => d[key]);
        const minV = Math.min(...values);
        const maxV = Math.max(...values);
        const range = maxV - minV || 1;

        // Label + current value
        ctx.fillStyle = color;
        ctx.font = '7px monospace';
        const latest = values[values.length - 1];
        ctx.fillText(`${label}: ${latest.toFixed(4)}`, 4, y + sparkH / 2 + 3);

        // Sparkline
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        for (let i = 0; i < values.length; i++) {
            const px = startX + (i / (data.length - 1)) * sparkW;
            const py = y + sparkH - ((values[i] - minV) / range) * sparkH;
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
        }
        ctx.stroke();
        ctx.lineWidth = 1;

        // Subtle background
        ctx.fillStyle = color + '10';
        ctx.fillRect(startX, y, sparkW, sparkH);
    }
}

// ── Master dashboard update (called from training loop) ──
function _updateTensorDashboard() {
    _drawPolicyHeatmap();
    _drawWeightDistributions();
    _drawTrainingMetrics();
}
