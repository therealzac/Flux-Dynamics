// ═══════════════════════════════════════════════════════════════════════════════
// flux-solver-gpu.wgsl — Jacobi PBD constraint solver (WebGPU compute shader)
//
// Three passes per iteration:
//   Pass 1: Project distance constraints (1 thread per constraint per instance)
//   Pass 2: Apply accumulated corrections (1 thread per node per instance)
//   Pass 3: Project repulsion constraints (1 thread per repulsion pair per instance)
//
// Batched: B instances solved simultaneously for canMaterialiseQuick pre-filter.
// Each instance = base SC pairs + one candidate SC pair.
//
// Uses atomic i32 for correction accumulation (fixed-point at 1e6 scale).
// ═══════════════════════════════════════════════════════════════════════════════

// ─── Uniforms ───
struct Params {
    N: u32,                    // number of nodes
    numConstraints: u32,       // max constraints per instance
    numRepulsion: u32,         // number of repulsion pairs
    numInstances: u32,         // batch size B
    omega: f32,                // SOR relaxation factor (1.0 default)
    iteration: u32,            // current iteration index
    _pad0: u32,
    _pad1: u32,
}
@group(0) @binding(0) var<uniform> params: Params;

// ─── Storage buffers ───

// Rest positions: N × 4 floats (x, y, z, w=0) — read-only, shared across instances
@group(0) @binding(1) var<storage, read> restPositions: array<vec4<f32>>;

// Per-instance positions: B × N × 4 floats — read-write
@group(0) @binding(2) var<storage, read_write> positions: array<vec4<f32>>;

// Per-instance constraint pairs: B × numConstraints × 2 u32 (nodeA, nodeB)
// Unused slots have nodeA = nodeB = 0xFFFFFFFF (sentinel)
@group(0) @binding(3) var<storage, read> constraints: array<vec2<u32>>;

// Per-instance correction accumulators: B × N × 4 i32 (dx, dy, dz, count) — atomic
// Fixed-point at SCALE factor for atomic accumulation
@group(0) @binding(4) var<storage, read_write> corrections: array<atomic<i32>>;

// Per-instance constraint count: B × 1 u32 — how many constraints this instance has
@group(0) @binding(5) var<storage, read> instanceConstraintCounts: array<u32>;

// Per-instance max error: B × 1 f32 — convergence tracking
@group(0) @binding(6) var<storage, read_write> maxError: array<atomic<u32>>;

// Repulsion pairs: numRepulsion × 2 u32 — read-only, shared across instances
@group(0) @binding(7) var<storage, read> repulsionPairs: array<vec2<u32>>;

// ─── Constants ───
const SCALE: f32 = 1000000.0;   // fixed-point scale for atomic i32
const INV_SCALE: f32 = 0.000001;
const SENTINEL: u32 = 0xFFFFFFFFu;

// ─── Helper: float bits for atomic max ───
fn floatBitsToUint(v: f32) -> u32 {
    return bitcast<u32>(v);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Pass 1: Project distance-1 constraints (Jacobi)
// Each thread handles one constraint for one instance.
// Computes correction and atomically accumulates into per-vertex corrections.
// ═══════════════════════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn projectConstraints(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flatIdx = gid.x;
    let instanceIdx = flatIdx / params.numConstraints;
    let constraintIdx = flatIdx % params.numConstraints;

    if (instanceIdx >= params.numInstances) { return; }

    // Check if this constraint slot is valid
    let numC = instanceConstraintCounts[instanceIdx];
    if (constraintIdx >= numC) { return; }

    // Read constraint pair
    let cOffset = instanceIdx * params.numConstraints + constraintIdx;
    let pair = constraints[cOffset];
    let nodeA = pair.x;
    let nodeB = pair.y;
    if (nodeA == SENTINEL || nodeB == SENTINEL) { return; }

    // Read current positions
    let posOffset = instanceIdx * params.N;
    let pA = positions[posOffset + nodeA];
    let pB = positions[posOffset + nodeB];

    // Compute distance and correction
    let delta = pB.xyz - pA.xyz;
    let dist = length(delta);
    if (dist < 1e-10) { return; } // degenerate — skip

    // Target distance = 1.0 (unit-length constraints)
    let err = dist - 1.0;
    let correction = delta * (err / dist * 0.5 * params.omega);

    // Convert to fixed-point i32
    let cxA = i32(correction.x * SCALE);
    let cyA = i32(correction.y * SCALE);
    let czA = i32(correction.z * SCALE);

    // Accumulate corrections atomically
    // Node A gets +correction, Node B gets -correction
    let corrOffset = instanceIdx * params.N * 4u;
    let offA = corrOffset + nodeA * 4u;
    let offB = corrOffset + nodeB * 4u;

    atomicAdd(&corrections[offA + 0u], cxA);
    atomicAdd(&corrections[offA + 1u], cyA);
    atomicAdd(&corrections[offA + 2u], czA);
    atomicAdd(&corrections[offA + 3u], 1i);  // count

    atomicAdd(&corrections[offB + 0u], -cxA);
    atomicAdd(&corrections[offB + 1u], -cyA);
    atomicAdd(&corrections[offB + 2u], -czA);
    atomicAdd(&corrections[offB + 3u], 1i);  // count
}

// ═══════════════════════════════════════════════════════════════════════════════
// Pass 2: Apply accumulated corrections and clear accumulators
// Each thread handles one node for one instance.
// ═══════════════════════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn applyCorrections(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flatIdx = gid.x;
    let instanceIdx = flatIdx / params.N;
    let nodeIdx = flatIdx % params.N;

    if (instanceIdx >= params.numInstances) { return; }
    if (nodeIdx >= params.N) { return; }

    let corrOffset = instanceIdx * params.N * 4u;
    let off = corrOffset + nodeIdx * 4u;

    // Read and clear atomics
    let cx = atomicExchange(&corrections[off + 0u], 0i);
    let cy = atomicExchange(&corrections[off + 1u], 0i);
    let cz = atomicExchange(&corrections[off + 2u], 0i);
    let count = atomicExchange(&corrections[off + 3u], 0i);

    if (count == 0i) { return; }

    // Average correction
    let invCount = 1.0 / f32(count);
    let dx = f32(cx) * INV_SCALE * invCount;
    let dy = f32(cy) * INV_SCALE * invCount;
    let dz = f32(cz) * INV_SCALE * invCount;

    // Apply
    let posOffset = instanceIdx * params.N + nodeIdx;
    var p = positions[posOffset];
    p.x += dx;
    p.y += dy;
    p.z += dz;
    positions[posOffset] = p;

    // Track max error for convergence
    let err = abs(dx) + abs(dy) + abs(dz);
    let errBits = floatBitsToUint(err);
    atomicMax(&maxError[instanceIdx], errBits);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Pass 3: Project repulsion constraints (Jacobi)
// Same as Pass 1 but only activates when distance < 1.0 (push apart).
// Repulsion pairs are shared across all instances.
// ═══════════════════════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn projectRepulsion(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flatIdx = gid.x;
    let instanceIdx = flatIdx / params.numRepulsion;
    let repIdx = flatIdx % params.numRepulsion;

    if (instanceIdx >= params.numInstances) { return; }
    if (repIdx >= params.numRepulsion) { return; }

    let pair = repulsionPairs[repIdx];
    let nodeA = pair.x;
    let nodeB = pair.y;

    // Read current positions
    let posOffset = instanceIdx * params.N;
    let pA = positions[posOffset + nodeA];
    let pB = positions[posOffset + nodeB];

    let delta = pB.xyz - pA.xyz;
    let dist = length(delta);

    // Only activate if too close (distance < 1.0)
    if (dist >= 1.0 || dist < 1e-10) { return; }

    // Push apart to distance 1.0
    let err = dist - 1.0; // negative since dist < 1.0
    let correction = delta * (err / dist * 0.5 * params.omega);

    let cxA = i32(correction.x * SCALE);
    let cyA = i32(correction.y * SCALE);
    let czA = i32(correction.z * SCALE);

    let corrOffset = instanceIdx * params.N * 4u;
    let offA = corrOffset + nodeA * 4u;
    let offB = corrOffset + nodeB * 4u;

    atomicAdd(&corrections[offA + 0u], cxA);
    atomicAdd(&corrections[offA + 1u], cyA);
    atomicAdd(&corrections[offA + 2u], czA);
    atomicAdd(&corrections[offA + 3u], 1i);

    atomicAdd(&corrections[offB + 0u], -cxA);
    atomicAdd(&corrections[offB + 1u], -cyA);
    atomicAdd(&corrections[offB + 2u], -czA);
    atomicAdd(&corrections[offB + 3u], 1i);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Pass 4: Clear max error for next iteration check
// ═══════════════════════════════════════════════════════════════════════════════
@compute @workgroup_size(64)
fn clearMaxError(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.numInstances) { return; }
    atomicStore(&maxError[idx], 0u);
}
